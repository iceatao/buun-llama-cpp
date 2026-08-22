#include "llama-vbr-artifact-capture.h"
#include "llama-vbr-artifact-stage.h"
#include "llama-vbr-identity-digest.h"
#include "llama-vbr-operation.h"
#include "server-prompt-cache-payload.h"
#include "server-vbr-artifact-store.h"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", \
                    __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

struct synthetic_source {
    std::vector<uint8_t> bytes;
    uint64_t fail_at = UINT64_MAX;
};

static bool read_synthetic(
        const void * opaque, uint64_t offset,
        uint8_t * destination, size_t size) noexcept {
    const auto & source =
        *static_cast<const synthetic_source *>(opaque);
    if (offset >= source.fail_at ||
        offset > source.bytes.size() ||
        size > source.bytes.size() - offset) {
        return false;
    }
    std::memcpy(destination, source.bytes.data() + offset, size);
    return true;
}

struct blocking_capture_source {
    synthetic_source source;
    std::atomic<bool> entered { false };
    std::atomic<bool> release { false };

    static bool read(
            const void * opaque, uint64_t offset,
            uint8_t * destination, size_t size) noexcept {
        auto & self = *const_cast<blocking_capture_source *>(
            static_cast<const blocking_capture_source *>(opaque));
        self.entered.store(true, std::memory_order_release);
        while (!self.release.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        return read_synthetic(&self.source, offset, destination, size);
    }
};

static std::vector<uint8_t> read_chain(
        const artifact_segment_chain & chain) {
    std::vector<uint8_t> out(chain.size());
    CHECK(chain.read(0, out.data(), out.size()));
    return out;
}

static void test_segment_chain_offsets() {
    artifact_segment_chain chain;
    const uint8_t a[] = { 0, 1, 2 };
    const uint8_t b[] = { 3, 4, 5, 6, 7 };
    CHECK(chain.append(a, sizeof(a)));
    CHECK(chain.append(b, sizeof(b)));
    CHECK(chain.size() == 8);
    CHECK(chain.segment_count() == 2);
    CHECK(chain.max_segment_size() == 5);
    uint8_t middle[5] = {};
    CHECK(chain.read(2, middle, sizeof(middle)));
    CHECK(std::vector<uint8_t>(middle, middle + 5) ==
          std::vector<uint8_t>({ 2, 3, 4, 5, 6 }));
    uint8_t one = 0;
    const auto source = chain.source();
    CHECK(source.read(source.context, 7, &one, 1));
    CHECK(one == 7);
    CHECK(!chain.read(8, &one, 1));

    artifact_segment_chain known_size(8);
    CHECK(known_size.append(a, sizeof(a)));
    const auto incomplete_digest =
        vbr_capture_stream_digest(known_size);
    CHECK(!std::any_of(
        incomplete_digest.begin(), incomplete_digest.end(),
        [](uint8_t value) { return value != 0; }));
    CHECK(known_size.append(b, sizeof(b)));
    CHECK(vbr_capture_stream_digest(known_size) ==
          vbr_capture_stream_digest(chain));
    CHECK(!known_size.append(&one, 1));

    auto incomplete = std::make_unique<artifact_segment_chain>(8);
    CHECK(incomplete->append(a, sizeof(a)));
    vbr_capture_sealed_companion refused;
    CHECK(!vbr_capture_seal_companion(
        0, std::move(incomplete), refused));
}

struct range_verify_source {
    std::vector<uint8_t> bytes;
    uint64_t calls = 0;

    static bool read(
            const void * opaque, uint64_t offset,
            uint8_t * destination, size_t size) noexcept {
        auto & self = *const_cast<range_verify_source *>(
            static_cast<const range_verify_source *>(opaque));
        if (offset > self.bytes.size() ||
            size > self.bytes.size() - offset) {
            return false;
        }
        self.calls++;
        std::memcpy(destination, self.bytes.data() + offset, size);
        return true;
    }

    vbr_artifact_byte_source source() const {
        return { bytes.size(), this, read };
    }
};

static void test_authenticated_range_tree() {
    static constexpr size_t CHUNK = VBR_CAPTURE_RANGE_CHUNK_BYTES;
    std::vector<uint8_t> bytes(4*CHUNK + 123);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = uint8_t((i*29 + i/251 + 7) & 0xff);
    }

    artifact_segment_chain chain(VBR_CAPTURE_RANGE_CHUNK_BYTES, 5);
    CHECK(chain.append(bytes.data(), 17));
    CHECK(chain.append(bytes.data() + 17, CHUNK + 91));
    CHECK(chain.append(
        bytes.data() + CHUNK + 108,
        bytes.size() - CHUNK - 108));
    vbr_capture_range_tree tree;
    CHECK(vbr_capture_range_seal(chain, 1024, tree));
    CHECK(tree);
    CHECK(tree.total_bytes() == bytes.size());
    CHECK(tree.chunk_bytes() == CHUNK);
    CHECK(tree.chunk_count() == 5);
    CHECK(tree.metadata_bytes() >= 15*32);
    CHECK(std::any_of(
        tree.root().begin(), tree.root().end(),
        [](uint8_t value) { return value != 0; }));
    CHECK(!chain.append(bytes.data(), 1));

    // Append boundaries do not affect the authenticated root.
    artifact_segment_chain rechunked(VBR_CAPTURE_RANGE_CHUNK_BYTES, 5);
    CHECK(rechunked.append(bytes.data(), bytes.size()));
    vbr_capture_range_tree same;
    CHECK(vbr_capture_range_seal(rechunked, 1024, same));
    CHECK(same.root() == tree.root());

    vbr_capture_range_proof proof;
    const std::vector<vbr_capture_authenticated_range> ranges {
        { CHUNK + 100, 200 },
        { 4*CHUNK + 3, 100 },
    };
    CHECK(vbr_capture_range_prove(tree, ranges, {}, proof));
    CHECK(proof);
    CHECK(proof.root() == tree.root());
    CHECK(proof.ranges().size() == 2);
    CHECK(proof.selected_chunk_count() == 2);
    CHECK(proof.proof_node_count() == 4);
    CHECK(proof.metadata_bytes() <= 1024);

    range_verify_source source { bytes };
    uint64_t bytes_read = 99;
    CHECK(vbr_capture_range_verify(proof, source.source(), &bytes_read));
    CHECK(bytes_read == CHUNK + 123);
    CHECK(source.calls == 2);

    // Selected corruption is detected; omitted chunks are never read.
    source.bytes[CHUNK + 101] ^= 1;
    CHECK(!vbr_capture_range_verify(proof, source.source(), &bytes_read));
    CHECK(bytes_read == 0);
    source.bytes = bytes;
    source.bytes[2*CHUNK + 1] ^= 1;
    source.calls = 0;
    CHECK(vbr_capture_range_verify(proof, source.source(), &bytes_read));
    CHECK(source.calls == 2);

    vbr_capture_range_proof refused = proof;
    auto tight = vbr_capture_range_proof_limits{};
    tight.max_ranges = 1;
    CHECK(!vbr_capture_range_prove(tree, ranges, tight, refused));
    CHECK(!refused);
    tight = {};
    tight.max_selected_chunks = 1;
    CHECK(!vbr_capture_range_prove(tree, ranges, tight, refused));
    tight = {};
    tight.max_proof_nodes = 2;
    CHECK(!vbr_capture_range_prove(tree, ranges, tight, refused));
    tight = {};
    tight.max_metadata_bytes = proof.metadata_bytes() - 1;
    CHECK(!vbr_capture_range_prove(tree, ranges, tight, refused));
    CHECK(!vbr_capture_range_prove(
        tree, { { CHUNK, 0 } }, {}, refused));
    CHECK(!vbr_capture_range_prove(
        tree, { { 2*CHUNK, 4 }, { CHUNK, 4 } }, {}, refused));

    artifact_segment_chain unavailable;
    vbr_capture_range_tree unavailable_tree = tree;
    CHECK(unavailable.append(bytes.data(), 4));
    CHECK(!vbr_capture_range_seal(unavailable, 1024, unavailable_tree));
    CHECK(!unavailable_tree);
    artifact_segment_chain over_cap(VBR_CAPTURE_RANGE_CHUNK_BYTES, 1);
    CHECK(!over_cap.append(bytes.data(), CHUNK + 1));
    CHECK(over_cap.size() == 0);
    artifact_segment_chain metadata_cap(
        VBR_CAPTURE_RANGE_CHUNK_BYTES, 5);
    CHECK(metadata_cap.append(bytes.data(), bytes.size()));
    CHECK(!vbr_capture_range_seal(
        metadata_cap, tree.metadata_bytes() - 1, unavailable_tree));
    CHECK(!unavailable_tree);
}

static void test_registry_quiescence_query() {
    const vbr_controller_instance_id instance { 0x1111, 0x2222 };
    const vbr_controller_instance_id other { 0x3333, 0x4444 };
    CHECK(vbr_operation_registry_quiescent_for(&instance, 1));

    {
        auto binding = vbr_mutation_binding(
            vbr_operation_kind::sequence_edit,
            0, 0, 1,
            vbr_operation_class::explicit_destructive_trim,
            instance, 0);
        vbr_scoped_operation operation(binding);
        CHECK(bool(operation));
        CHECK(!vbr_operation_registry_quiescent_for(&instance, 1));
        CHECK(vbr_operation_registry_quiescent_for(&other, 1));
    }
    CHECK(vbr_operation_registry_quiescent_for(&instance, 1));

    {
        auto binding = vbr_mutation_binding(
            vbr_operation_kind::recovery,
            -1, -1, -1,
            vbr_operation_class::state_api);
        vbr_scoped_operation operation(binding);
        CHECK(bool(operation));
        CHECK(!vbr_operation_registry_quiescent_for(&instance, 1));
        CHECK(!vbr_operation_registry_quiescent_for(&other, 1));
    }
    CHECK(vbr_operation_registry_quiescent_for(&instance, 1));
    CHECK(!vbr_operation_registry_quiescent_for(nullptr, 0));
}

static void test_cpu_ring_boundaries() {
    vbr_capture_stream_status status;
    vbr_capture_ring_create_failure failure;
    auto unavailable = vbr_pinned_chunk_ring::create(
        { {} }, 8, 8, status, nullptr, &failure);
    CHECK(!unavailable);
    CHECK(status == vbr_capture_stream_status::ring_unavailable);
    CHECK(failure ==
          vbr_capture_ring_create_failure::invalid_geometry);

    auto ring = vbr_pinned_chunk_ring::create(
        { {}, {} }, 32, 8, status);
    CHECK(ring);
    CHECK(status == vbr_capture_stream_status::ok);
    CHECK(ring->lane_count() == 2);
    CHECK(ring->capacity_bytes() == 32);

    synthetic_source input;
    input.bytes.resize(41);
    for (size_t i = 0; i < input.bytes.size(); ++i) {
        input.bytes[i] = uint8_t((i*17 + 3) & 0xff);
    }
    vbr_capture_stream_source source;
    source.lane = 1;
    source.size = input.bytes.size();
    source.context = &input;
    source.read = read_synthetic;
    artifact_segment_chain chain;
    vbr_capture_stream_stats stats;
    CHECK(ring->stream(source, chain, stats) ==
          vbr_capture_stream_status::ok);
    CHECK(stats.bytes == input.bytes.size());
    CHECK(stats.chunks == 6);
    CHECK(stats.backpressure_waits > 0);
    CHECK(stats.max_segment_size <= 8);
    CHECK(chain.max_segment_size() <= 8);
    CHECK(chain.size() > ring->capacity_bytes());
    CHECK(read_chain(chain) == input.bytes);

    // Test-local pre-refactor oracle: the historical CPU adapter appended one
    // pageable segment per ring-sized chunk and hashed the resulting byte
    // stream. This pins the D2H facade across the shared-core extraction.
    artifact_segment_chain legacy;
    for (size_t offset = 0; offset < input.bytes.size(); offset += 8) {
        const size_t count = std::min<size_t>(8, input.bytes.size() - offset);
        CHECK(legacy.append(input.bytes.data() + offset, count));
    }
    CHECK(read_chain(legacy) == read_chain(chain));
    CHECK(vbr_capture_stream_digest(legacy) == stats.streaming_digest);
    CHECK(legacy.segment_count() == stats.chunks);

    artifact_segment_chain projected;
    vbr_capture_stream_stats projected_stats;
    const std::vector<vbr_capture_stream_range> ranges {
        { 2, 3 }, { 10, 5 }, { 20, 1 },
    };
    CHECK(ring->stream_ranges(
              source, ranges, projected, projected_stats) ==
          vbr_capture_stream_status::ok);
    std::vector<uint8_t> projected_expected;
    for (const auto & range : ranges) {
        projected_expected.insert(
            projected_expected.end(),
            input.bytes.begin() + range.source_offset,
            input.bytes.begin() + range.source_offset + range.size);
    }
    CHECK(read_chain(projected) == projected_expected);
    CHECK(projected_stats.bytes == projected_expected.size());
    CHECK(projected_stats.chunks == 2);
    CHECK(projected.segment_count() == 2);
    CHECK(projected_stats.streaming_digest ==
          vbr_capture_stream_digest(projected));

    artifact_segment_chain invalid_ranges;
    projected_stats.bytes = 99;
    CHECK(ring->stream_ranges(
              source, { { 4, 3 }, { 6, 2 } },
              invalid_ranges, projected_stats) ==
          vbr_capture_stream_status::invalid_argument);
    CHECK(projected_stats.bytes == 0);
    CHECK(invalid_ranges.size() == 0);

    auto other = vbr_pinned_chunk_ring::create(
        { {} }, 14, 7, status);
    CHECK(other);
    artifact_segment_chain rechunked;
    vbr_capture_stream_stats other_stats;
    source.lane = 0;
    CHECK(other->stream(source, rechunked, other_stats) ==
          vbr_capture_stream_status::ok);
    CHECK(read_chain(rechunked) == input.bytes);
    CHECK(other_stats.streaming_digest == stats.streaming_digest);

    input.fail_at = 16;
    artifact_segment_chain short_chain;
    CHECK(other->stream(source, short_chain, other_stats) ==
          vbr_capture_stream_status::short_read);

    input.fail_at = UINT64_MAX;
    source.fail_completion_at = 1;
    artifact_segment_chain failed_completion;
    CHECK(other->stream(
        source, failed_completion, other_stats) ==
            vbr_capture_stream_status::transfer_failed);
}

struct projected_snapshot_fixture {
    vbr_capture_unit_snapshot snapshot;
    uint32_t acquired = 0;
    uint32_t rechecked = 0;
    uint32_t released = 0;
    bool acquire_ok = true;
    bool recheck_ok = true;

    static bool acquire(
            void * context,
            uint64_t source_namespace,
            uint32_t child_id,
            uint32_t logical_unit_id,
            vbr_capture_unit_snapshot & output) noexcept {
        auto & self = *static_cast<projected_snapshot_fixture *>(context);
        self.acquired++;
        output = self.snapshot;
        return self.acquire_ok &&
               output.source_namespace == source_namespace &&
               output.child_id == child_id &&
               output.logical_unit_id == logical_unit_id;
    }

    static bool recheck(
            void * context,
            const vbr_capture_unit_snapshot & expected) noexcept {
        auto & self = *static_cast<projected_snapshot_fixture *>(context);
        self.rechecked++;
        return self.recheck_ok &&
               expected.generation.repr_gen ==
                   self.snapshot.generation.repr_gen &&
               expected.generation.publish_seq ==
                   self.snapshot.generation.publish_seq &&
               expected.generation.current_type ==
                   self.snapshot.generation.current_type &&
               expected.generation.last_source_type ==
                   self.snapshot.generation.last_source_type &&
               expected.generation.domain ==
                   self.snapshot.generation.domain &&
               expected.generation.promote_hops ==
                   self.snapshot.generation.promote_hops &&
               expected.generation.last_transition ==
                   self.snapshot.generation.last_transition &&
               expected.lineage_uuid == self.snapshot.lineage_uuid &&
               expected.controller_generation ==
                   self.snapshot.controller_generation &&
               expected.mutation_serial == self.snapshot.mutation_serial;
    }

    static void release(
            void * context,
            const vbr_capture_unit_snapshot &) noexcept {
        static_cast<projected_snapshot_fixture *>(context)->released++;
    }

    vbr_capture_unit_snapshot_provider provider() {
        return { this, acquire, recheck, release };
    }
};

static vbr_artifact_stream_placement projected_placement(
        uint64_t manifest,
        llama_seq_id sequence,
        std::initializer_list<uint32_t> cells) {
    GGML_UNUSED(manifest);
    vbr_artifact_stream_placement placement;
    placement.child_id = 0;
    placement.stream_index = 0;
    placement.source_sequence = sequence;
    placement.computation_frontier = 8;
    llama_pos position = 0;
    for (uint32_t cell : cells) {
        placement.cells.push_back({ cell, position++, 0, 0 });
    }
    return placement;
}

static std::vector<uint8_t> projected_rows(
        const std::vector<uint8_t> & source,
        uint64_t row_bytes,
        std::initializer_list<uint32_t> cells) {
    std::vector<uint8_t> output;
    for (uint32_t cell : cells) {
        const size_t offset = size_t(cell*row_bytes);
        output.insert(
            output.end(), source.begin() + offset,
            source.begin() + offset + row_bytes);
    }
    return output;
}

static void test_projected_unit_transfer() {
    vbr_capture_projection_manifest one;
    one.manifest_id = 1;
    one.placements.push_back(projected_placement(1, 1, { 1, 2, 5 }));
    vbr_capture_projection_manifest two;
    two.manifest_id = 2;
    two.placements.push_back(projected_placement(2, 2, { 2, 3 }));
    vbr_capture_projection projection;
    CHECK(vbr_artifact_project_capture_union(
        { 91, { one, two } }, {}, projection));

    synthetic_source first;
    synthetic_source second;
    first.bytes.resize(8*2);
    second.bytes.resize(8*3);
    for (size_t i = 0; i < first.bytes.size(); ++i) {
        first.bytes[i] = uint8_t(10 + i);
    }
    for (size_t i = 0; i < second.bytes.size(); ++i) {
        second.bytes[i] = uint8_t(100 + i);
    }
    vbr_capture_projected_shard_source shard_zero;
    shard_zero.shard_index = 0;
    shard_zero.row_count = 8;
    shard_zero.row_bytes = 2;
    shard_zero.source_identity = 101;
    shard_zero.source.size = first.bytes.size();
    shard_zero.source.context = &first;
    shard_zero.source.read = read_synthetic;
    vbr_capture_projected_shard_source shard_one;
    shard_one.shard_index = 1;
    shard_one.row_count = 8;
    shard_one.row_bytes = 3;
    shard_one.source_identity = 102;
    shard_one.source.size = second.bytes.size();
    shard_one.source.context = &second;
    shard_one.source.read = read_synthetic;
    std::vector<vbr_capture_projected_shard_source> sources {
        shard_one, shard_zero,
    };

    projected_snapshot_fixture snapshot;
    snapshot.snapshot.source_namespace = 91;
    snapshot.snapshot.child_id = 0;
    snapshot.snapshot.logical_unit_id = 7;
    snapshot.snapshot.lineage_uuid = { 17, 19 };
    snapshot.snapshot.controller_generation = 11;
    snapshot.snapshot.mutation_serial = 0;
    snapshot.snapshot.generation.repr_gen = 13;
    snapshot.snapshot.generation.publish_seq = 14;
    snapshot.snapshot.generation.current_type = GGML_TYPE_F16;
    snapshot.snapshot.generation.last_source_type = GGML_TYPE_F16;
    CHECK(vbr_capture_projected_shard_topology(
        sources, snapshot.snapshot.shard_count,
        snapshot.snapshot.shard_topology_digest));

    vbr_capture_stream_status status;
    auto ring = vbr_pinned_chunk_ring::create(
        { {} }, 16, 4, status);
    CHECK(ring);
    CHECK(status == vbr_capture_stream_status::ok);
    vbr_capture_projected_unit captured;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources,
              {},
              snapshot.provider(), *ring, captured) ==
          vbr_capture_stream_status::ok);
    CHECK(snapshot.acquired == 1);
    CHECK(snapshot.rechecked == 1);
    CHECK(snapshot.released == 1);
    CHECK(captured.projection() == projection);
    CHECK(captured.shards().size() == 2);
    CHECK(captured.packed_bytes() == 20);
    CHECK(captured.transfer().bytes == 20);
    CHECK(std::any_of(
        captured.transfer().streaming_digest.begin(),
        captured.transfer().streaming_digest.end(),
        [](uint8_t value) { return value != 0; }));
    auto rebound_sources = sources;
    rebound_sources[0].source.context = &first;
    uint32_t rebound_count = 0;
    std::array<uint8_t, 32> rebound_digest = {};
    CHECK(vbr_capture_projected_shard_topology(
        rebound_sources, rebound_count, rebound_digest));
    CHECK(rebound_digest != captured.snapshot().shard_topology_digest);
    rebound_sources = sources;
    rebound_sources[0].source.tensor_offset = 1;
    CHECK(vbr_capture_projected_shard_topology(
        rebound_sources, rebound_count, rebound_digest));
    CHECK(rebound_digest != captured.snapshot().shard_topology_digest);
    if (captured.shards().size() == 2) {
        CHECK(captured.shards()[0].shard_index == 0);
        CHECK(captured.shards()[1].shard_index == 1);
        CHECK(captured.shards()[0].authenticated_ranges);
        CHECK(captured.shards()[1].authenticated_ranges);
        CHECK(captured.shards()[0].streaming_digest ==
              captured.shards()[0].authenticated_ranges.root());
        CHECK(captured.shards()[1].streaming_digest ==
              captured.shards()[1].authenticated_ranges.root());
        CHECK(captured.shards()[0].authenticated_ranges.total_bytes() == 8);
        CHECK(captured.shards()[1].authenticated_ranges.total_bytes() == 12);
        CHECK(captured.shards()[0].authenticated_ranges.root() !=
              captured.shards()[1].authenticated_ranges.root());
        CHECK(read_chain(*captured.shards()[0].bytes) ==
              projected_rows(first.bytes, 2, { 1, 2, 3, 5 }));
        CHECK(read_chain(*captured.shards()[1].bytes) ==
              projected_rows(second.bytes, 3, { 1, 2, 3, 5 }));
        CHECK(projection->streams[0].segments.size() == 4);
        if (projection->streams[0].segments.size() == 4) {
            CHECK(projection->streams[0].segments[0].packed_first_row == 0);
            CHECK(projection->streams[0].segments[1].packed_first_row == 1);
            CHECK(projection->streams[0].segments[2].packed_first_row == 2);
            CHECK(projection->streams[0].segments[3].packed_first_row == 3);
        }
    }

    // Equal packed bytes under a different manifest dependency geometry must
    // produce a different authenticated unit digest.
    vbr_capture_projection_manifest combined;
    combined.manifest_id = 3;
    combined.placements.push_back(projected_placement(
        3, 3, { 1, 2, 3, 5 }));
    vbr_capture_projection combined_projection;
    CHECK(vbr_artifact_project_capture_union(
        { 91, { combined } }, {}, combined_projection));
    projected_snapshot_fixture combined_snapshot;
    combined_snapshot.snapshot = captured.snapshot();
    vbr_capture_projected_unit combined_capture;
    CHECK(vbr_capture_projected_unit_transfer(
              combined_projection, 0, 0, 7, sources, {},
              combined_snapshot.provider(), *ring, combined_capture) ==
          vbr_capture_stream_status::ok);
    CHECK(combined_capture.packed_bytes() == captured.packed_bytes());
    CHECK(combined_capture.transfer().streaming_digest !=
          captured.transfer().streaming_digest);

    projected_snapshot_fixture other_lineage;
    other_lineage.snapshot = captured.snapshot();
    other_lineage.snapshot.lineage_uuid.lo++;
    vbr_capture_projected_unit lineage_capture;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, {},
              other_lineage.provider(), *ring, lineage_capture) ==
          vbr_capture_stream_status::ok);
    CHECK(lineage_capture.packed_bytes() == captured.packed_bytes());
    CHECK(lineage_capture.transfer().streaming_digest !=
          captured.transfer().streaming_digest);

    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    snapshot.recheck_ok = false;
    vbr_capture_projected_unit changed = captured;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources,
              {},
              snapshot.provider(), *ring, changed) ==
          vbr_capture_stream_status::snapshot_changed);
    CHECK(snapshot.acquired == 1);
    CHECK(snapshot.rechecked == 1);
    CHECK(snapshot.released == 1);
    CHECK(!changed);
    CHECK(changed.shards().empty());

    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    sources[0].source.fail_completion_at = 0;
    vbr_capture_projected_unit failed = captured;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources,
              {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::transfer_failed);
    CHECK(snapshot.acquired == 1);
    CHECK(snapshot.rechecked == 0);
    CHECK(snapshot.released == 1);
    CHECK(!failed);
    CHECK(failed.shards().empty());

    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    snapshot.acquire_ok = false;
    sources[0].source.fail_completion_at = UINT64_MAX;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources,
              {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::snapshot_unavailable);
    CHECK(snapshot.acquired == 1);
    CHECK(snapshot.rechecked == 0);
    CHECK(snapshot.released == 0);

    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    snapshot.snapshot.controller_generation = 0;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources,
              {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::snapshot_unavailable);
    CHECK(snapshot.acquired == 1);
    CHECK(snapshot.rechecked == 0);
    CHECK(snapshot.released == 1);

    // Stable zero serials are valid; odd mutation/publish serials are not.
    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    snapshot.snapshot.mutation_serial = 1;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::snapshot_unavailable);
    CHECK(snapshot.released == 1);
    const auto expect_invalid_snapshot = [&](vbr_capture_unit_snapshot value) {
        projected_snapshot_fixture invalid;
        invalid.snapshot = value;
        CHECK(vbr_capture_projected_unit_transfer(
                  projection, 0, 0, 7, sources, {},
                  invalid.provider(), *ring, failed) ==
              vbr_capture_stream_status::snapshot_unavailable);
        CHECK(invalid.acquired == 1);
        CHECK(invalid.rechecked == 0);
        CHECK(invalid.released == 1);
    };
    auto invalid_snapshot = captured.snapshot();
    invalid_snapshot.generation.last_source_type = -1;
    expect_invalid_snapshot(invalid_snapshot);
    invalid_snapshot = captured.snapshot();
    invalid_snapshot.generation.current_type = GGML_TYPE_COUNT;
    expect_invalid_snapshot(invalid_snapshot);
    invalid_snapshot = captured.snapshot();
    invalid_snapshot.generation.domain = vbr_repr_domain(255);
    expect_invalid_snapshot(invalid_snapshot);
    invalid_snapshot = captured.snapshot();
    invalid_snapshot.generation.last_transition = vbr_repr_transition(255);
    expect_invalid_snapshot(invalid_snapshot);
    invalid_snapshot = captured.snapshot();
    invalid_snapshot.generation.flags = 1;
    expect_invalid_snapshot(invalid_snapshot);
    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    snapshot.snapshot.generation.publish_seq = 15;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::snapshot_unavailable);
    CHECK(snapshot.released == 1);

    // The snapshot authenticates the complete shard set. Reordering is
    // normalized, while omission, sparse/sentinel IDs, and substitution fail.
    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    auto omitted = sources;
    omitted.erase(omitted.begin());
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, omitted, {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::snapshot_unavailable);
    auto substituted = sources;
    substituted[0].source_identity++;
    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, substituted, {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::snapshot_unavailable);
    auto sparse = sources;
    sparse[0].shard_index = 3;
    uint32_t invalid_count = 9;
    std::array<uint8_t, 32> invalid_digest = { 1 };
    CHECK(!vbr_capture_projected_shard_topology(
        sparse, invalid_count, invalid_digest));
    CHECK(invalid_count == 0);
    auto sentinel = sources;
    sentinel[0].shard_index = UINT32_MAX;
    CHECK(!vbr_capture_projected_shard_topology(
        sentinel, invalid_count, invalid_digest));

    vbr_capture_projected_transfer_limits tight;
    tight.max_shards = 1;
    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, tight,
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::projection_invalid);
    CHECK(snapshot.acquired == 0);
    tight = {};
    tight.max_shard_segment_references = 7;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, tight,
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::projection_invalid);
    tight = {};
    tight.max_source_operations = 6;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, tight,
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::projection_invalid);
    CHECK(snapshot.acquired == 0);
    tight.max_source_operations = 7;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, tight,
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::ok);
    CHECK(snapshot.acquired == 1);
    CHECK(snapshot.rechecked == 1);
    CHECK(snapshot.released == 1);
    tight = {};
    tight.max_total_packed_bytes = 19;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, tight,
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::projection_invalid);
    tight = {};
    tight.max_authenticated_chunks = 1;
    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, tight,
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::projection_invalid);
    CHECK(snapshot.acquired == 0);
    tight = {};
    tight.max_authenticated_metadata_bytes = 63;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, tight,
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::projection_invalid);
    CHECK(snapshot.acquired == 0);

    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    sources[0].row_count = 3;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources,
              {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::projection_invalid);
    CHECK(snapshot.acquired == 0);
    CHECK(snapshot.released == 0);
}

struct projected_controller_fixture {
    uint64_t rejected_manifest = 0;
    std::vector<uint64_t> rechecked;

    static bool recheck(
            void * context,
            uint64_t manifest_id,
            const vbr_capture_controller_target * targets,
            size_t target_count) noexcept {
        auto & self = *static_cast<projected_controller_fixture *>(context);
        self.rechecked.push_back(manifest_id);
        if (manifest_id == self.rejected_manifest || !targets ||
            target_count == 0) {
            return false;
        }
        for (size_t i = 0; i < target_count; ++i) {
            if (targets[i].manifest_id != manifest_id ||
                (i != 0 &&
                 targets[i - 1].child_id >= targets[i].child_id)) {
                return false;
            }
        }
        return true;
    }

    vbr_capture_controller_target_provider provider() {
        return { this, recheck };
    }
};

static vbr_unit_generation projected_generation(
        uint64_t repr_gen, ggml_type type = GGML_TYPE_F16) {
    vbr_unit_generation generation;
    generation.repr_gen = repr_gen;
    generation.publish_seq = repr_gen*2;
    generation.current_type = type;
    generation.last_source_type = type;
    return generation;
}

static vbr_capture_controller_target projected_target(
        uint64_t manifest_id,
        uint32_t child_id,
        uint64_t controller_generation,
        const vbr_unit_generation & generation) {
    vbr_capture_controller_target target;
    target.manifest_id = manifest_id;
    target.source_namespace = 707;
    target.child_id = child_id;
    target.lineage_uuid = {
        uint64_t(child_id + 1), controller_generation + 1000,
    };
    target.controller_generation = controller_generation;
    target.units = { generation };
    target.policy.child_id = child_id;
    target.policy.dependency_mode =
        checkpoint_child_dependency_mode::live_guarded;
    target.policy.degrade_order_digest.fill(
        uint8_t(1 + controller_generation + child_id));
    target.policy.policy_digest.fill(
        uint8_t(17 + controller_generation + child_id));
    target.policy.cursor = controller_generation;
    target.policy.floor_type = GGML_TYPE_Q4_0;
    target.policy.pressure_independent_settings = 9;
    target.policy.n_stream = 1;
    target.policy.unified = true;
    target.policy.wm_cells = 8;
    target.policy.current_type_vector_digest =
        vbr_type_vector_digest(std::vector<ggml_type> {
            ggml_type(generation.current_type),
        });
    target.policy.completed_wave = true;
    vbr_artifact_unit_descriptor descriptor;
    descriptor.child_id = child_id;
    descriptor.logical_unit_id = 0;
    descriptor.lineage_uuid = target.lineage_uuid;
    descriptor.repr_gen = generation.repr_gen;
    descriptor.current_type = generation.current_type;
    descriptor.last_source_type = generation.last_source_type;
    descriptor.promote_hops = generation.promote_hops;
    descriptor.last_transition = generation.last_transition;
    descriptor.representation.kind =
        generation.current_type == GGML_TYPE_F16
            ? vbr_artifact_representation_kind::raw
            : vbr_artifact_representation_kind::approximate;
    descriptor.representation.codec_id = 1;
    descriptor.representation.codec_version = 1;
    if (descriptor.representation.kind ==
            vbr_artifact_representation_kind::approximate) {
        descriptor.representation.reference_digest.fill(0x31);
    }
    descriptor.side = vbr_artifact_side::key;
    descriptor.layout = vbr_artifact_layout::row_major;
    descriptor.n_stream = target.policy.n_stream;
    descriptor.unified = target.policy.unified;
    descriptor.wm_cells = target.policy.wm_cells;
    descriptor.rank = 2;
    descriptor.dimensions = { target.policy.wm_cells, 1, 0, 0 };
    descriptor.row_alignment = 1;
    descriptor.row_codec_version = 1;
    vbr_artifact_shard_descriptor shard;
    shard.shard_index = 0;
    shard.topology_index = 0;
    shard.device_ordinal = 0;
    shard.logical_offset = 0;
    shard.row_count = target.policy.wm_cells;
    shard.column_count = 1;
    shard.row_bytes = 1;
    shard.payload_bytes = target.policy.wm_cells;
    descriptor.shards = { shard };
    target.unit_descriptors = { descriptor };
    return target;
}

static void projected_target_streams(
        vbr_capture_controller_target & target,
        uint32_t n_stream) {
    target.policy.n_stream = n_stream;
    target.policy.unified = n_stream == 1;
    for (auto & descriptor : target.unit_descriptors) {
        descriptor.n_stream = n_stream;
        descriptor.unified = n_stream == 1;
    }
}

static void projected_target_row_bytes(
        vbr_capture_controller_target & target,
        uint64_t row_bytes) {
    for (auto & descriptor : target.unit_descriptors) {
        for (auto & shard : descriptor.shards) {
            shard.row_bytes = row_bytes;
            shard.payload_bytes = descriptor.wm_cells*row_bytes;
        }
    }
}

static void bind_projected_manifest_metadata(
        vbr_capture_projection_manifest & manifest,
        const vbr_capture_controller_target & target) {
    manifest.identity_policy_order_digest.fill(
        uint8_t(0x80 + manifest.manifest_id));
    manifest.identity.execution_identity = "projected-test";
    manifest.identity.adapter_config_identity = "adapter";
    manifest.identity.media_content_identity = "media";
    manifest.identity.sequence_epoch = manifest.manifest_id;
    manifest.identity.token_count = 8;
    manifest.identity.next_position = 8;
    manifest.token_block.tokens = {
        1, 2, 3, 4, 5, 6, 7, int32_t(manifest.manifest_id),
    };
    auto & generation = manifest.generation;
    generation.version = 1;
    generation.status = vbr_checkpoint_generation_status::complete;
    generation.identity_policy_order_digest =
        manifest.identity_policy_order_digest;
    vbr_checkpoint_generation_controller controller;
    controller.child_id = target.child_id;
    controller.dependency_mode = target.policy.dependency_mode;
    controller.lineage_uuid = target.lineage_uuid;
    controller.global_generation = target.controller_generation;
    for (const auto & unit : target.units) {
        controller.units.push_back({
            unit.repr_gen, unit.current_type, unit.last_source_type,
            unit.domain, unit.promote_hops, unit.last_transition,
        });
    }
    for (const auto & placement : manifest.placements) {
        vbr_checkpoint_generation_stream stream;
        stream.stream_index = placement.stream_index;
        stream.dependency_seq_id = placement.source_sequence;
        stream.computation_frontier = placement.computation_frontier;
        stream.captured_dependency_count = placement.cells.size();
        for (const auto & cell : placement.cells) {
            const uint32_t page =
                cell.physical_cell/VBR_GENERATION_PAGE_CELLS;
            if (stream.pages.empty() ||
                stream.pages.back().page_index != page) {
                vbr_generation_page_ref ref;
                ref.page_index = page;
                ref.captured_page_gen = 1;
                stream.pages.push_back(ref);
            }
            const uint32_t offset =
                cell.physical_cell%VBR_GENERATION_PAGE_CELLS;
            stream.pages.back().covered_mask[offset/64] |=
                uint64_t(1) << (offset%64);
        }
        controller.streams.push_back(std::move(stream));
    }
    generation.controllers.push_back(std::move(controller));
}

static vbr_capture_projected_unit capture_projected_unit_for_target(
        const vbr_capture_projection & projection,
        const vbr_capture_controller_target & target,
        uint32_t stream_index = 0,
        uint64_t row_bytes = 1) {
    synthetic_source source;
    source.bytes.resize(size_t(8*row_bytes));
    for (uint32_t i = 0; i < source.bytes.size(); ++i) {
        source.bytes[i] = uint8_t(
            i + target.child_id + target.controller_generation);
    }
    vbr_capture_projected_shard_source shard;
    shard.shard_index = 0;
    shard.row_count = 8;
    shard.row_bytes = row_bytes;
    shard.source_identity =
        target.controller_generation*10 + target.child_id + 1;
    shard.source.size = source.bytes.size();
    shard.source.context = &source;
    shard.source.read = read_synthetic;
    std::vector<vbr_capture_projected_shard_source> sources { shard };

    projected_snapshot_fixture snapshot;
    snapshot.snapshot.source_namespace = target.source_namespace;
    snapshot.snapshot.child_id = target.child_id;
    snapshot.snapshot.logical_unit_id = 0;
    snapshot.snapshot.lineage_uuid = target.lineage_uuid;
    snapshot.snapshot.controller_generation = target.controller_generation;
    snapshot.snapshot.mutation_serial = 0;
    snapshot.snapshot.generation = target.units[0];
    CHECK(vbr_capture_projected_shard_topology(
        sources, snapshot.snapshot.shard_count,
        snapshot.snapshot.shard_topology_digest));
    vbr_capture_stream_status status;
    const size_t chunk_bytes = row_bytes == 1 ? 4 :
        VBR_CAPTURE_RANGE_CHUNK_BYTES;
    auto ring = vbr_pinned_chunk_ring::create(
        { {} }, 2*chunk_bytes, chunk_bytes, status);
    CHECK(ring);
    vbr_capture_projected_unit unit;
    if (ring) {
        CHECK(vbr_capture_projected_unit_transfer(
                  projection, target.child_id, stream_index, 0,
                  sources, {}, snapshot.provider(), *ring, unit) ==
              vbr_capture_stream_status::ok);
    }
    return unit;
}

static const vbr_capture_manifest_result * projected_manifest(
        const vbr_capture_manifest_assembly & assembly,
        uint64_t manifest_id) {
    const auto & manifests = assembly.manifests();
    const auto found = std::find_if(
        manifests.begin(), manifests.end(),
        [&](const auto & value) { return value.manifest_id == manifest_id; });
    return found == manifests.end() ? nullptr : &*found;
}

static bool assemble_projected_test_batch(
        const vbr_capture_projection & projection,
        const std::vector<vbr_capture_controller_target> & targets,
        const std::vector<vbr_capture_projected_unit> & units,
        const vbr_capture_controller_target_provider & provider,
        const vbr_capture_manifest_assembly_limits & limits,
        vbr_capture_manifest_assembly & output) {
    auto owned_targets = targets;
    auto owned_units = units;
    return vbr_capture_assemble_manifests(
        projection, std::move(owned_targets), std::move(owned_units),
        provider, limits, output);
}

static void test_preflight_unavailable_projection_rows() {
    auto target = projected_target(
        1, 0, 101, projected_generation(5));
    vbr_capture_projection_manifest available;
    available.manifest_id = 1;
    available.placements.push_back(
        projected_placement(1, 1, { 1, 2 }));
    bind_projected_manifest_metadata(available, target);
    auto unavailable = available;
    unavailable.manifest_id = 2;
    unavailable.dependencies_available = false;
    unavailable.placements.clear();

    vbr_capture_projection mixed;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { unavailable, available } }, {}, mixed));
    CHECK(mixed);
    CHECK(mixed->manifest_count == 2);
    CHECK(mixed->union_cell_count == 2);
    CHECK(mixed->streams.size() == 1);
    auto unit = capture_projected_unit_for_target(mixed, target);
    projected_controller_fixture provider;
    vbr_capture_manifest_assembly assembled;
    CHECK(assemble_projected_test_batch(
        mixed, { target }, { unit }, provider.provider(), {}, assembled));
    CHECK(provider.rechecked == std::vector<uint64_t>({ 1 }));
    CHECK(projected_manifest(assembled, 1)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 2)->state ==
          vbr_capture_manifest_state::dependency_unavailable);

    auto unavailable_three = unavailable;
    unavailable_three.manifest_id = 3;
    vbr_capture_projection all_unavailable;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { unavailable, unavailable_three } }, {}, all_unavailable));
    CHECK(all_unavailable);
    CHECK(all_unavailable->union_cell_count == 0);
    CHECK(all_unavailable->streams.empty());
    provider = {};
    CHECK(assemble_projected_test_batch(
        all_unavailable, {}, {}, provider.provider(), {}, assembled));
    CHECK(provider.rechecked.empty());
    CHECK(assembled.manifests().size() == 2);
    CHECK(std::all_of(
        assembled.manifests().begin(), assembled.manifests().end(),
        [](const auto & manifest) {
            return manifest.state ==
                vbr_capture_manifest_state::dependency_unavailable;
        }));

    auto invalid = unavailable;
    invalid.placements.push_back(projected_placement(2, 2, { 3 }));
    vbr_capture_projection refused = mixed;
    CHECK(!vbr_artifact_project_capture_union(
        { 707, { invalid } }, {}, refused));
    CHECK(!refused);
    invalid = unavailable;
    invalid.dependencies_available = true;
    CHECK(!vbr_artifact_project_capture_union(
        { 707, { invalid } }, {}, refused));
    CHECK(!refused);
}

static vbr_artifact_portable_topology capture_test_topology();

static vbr_projected_manifest_publication projected_publication(
        uint64_t manifest_id,
        const vbr_capture_manifest_assembly & assembly,
        const vbr_artifact_portable_topology & topology) {
    vbr_projected_manifest_publication out;
    out.manifest_id = manifest_id;
    out.topologies = { topology };
    const auto * row = projected_manifest(assembly, manifest_id);
    CHECK(row != nullptr);
    if (!row || row->state != vbr_capture_manifest_state::ready) {
        return out;
    }
    CHECK(row->unit_count == 1);
    if (row->unit_count != 1) {
        return out;
    }
    const uint32_t captured_index = assembly.unit_references()[
        row->first_unit];
    const auto & captured = assembly.projected_units()[captured_index];
    vbr_artifact_portable_accounting_row payload;
    payload.role = vbr_artifact_accounting_role::unit_payload;
    payload.domain = {
        llama_cache_acct_residency::device,
        llama_cache_acct_domain_kind::device_topology,
        0, 0,
    };
    payload.logical_bytes = captured.packed_bytes();
    payload.resident_bytes = captured.packed_bytes();
    out.accounting.push_back(payload);
    const vbr_artifact_portable_domain host {
        llama_cache_acct_residency::pageable_host,
        llama_cache_acct_domain_kind::not_applicable,
        UINT32_MAX, UINT16_MAX,
    };
    out.accounting.push_back({
        vbr_artifact_accounting_role::descriptor_metadata,
        host, 512, 512, llama_cache_acct_attr_kind::artifact,
    });
    out.accounting.push_back({
        vbr_artifact_accounting_role::reference_metadata,
        host, 256, 256, llama_cache_acct_attr_kind::artifact,
    });
    return out;
}

static void test_dependency_scoped_projected_catalog_publication() {
    static_assert(!std::is_copy_constructible_v<
        vbr_capture_sealed_companion>);
    static_assert(std::is_nothrow_move_constructible_v<
        vbr_capture_sealed_companion>);
    std::vector<vbr_capture_projection_manifest> manifests;
    std::vector<vbr_capture_controller_target> targets;
    for (uint64_t i = 1; i <= 8; ++i) {
        vbr_capture_projection_manifest manifest;
        manifest.manifest_id = i;
        manifest.placements.push_back(projected_placement(
            i, llama_seq_id(i), { uint32_t(i - 1) }));
        auto target = projected_target(
            i, 0, 500 + i, projected_generation(20 + i));
        bind_projected_manifest_metadata(manifest, target);
        if (i == 6 || i == 8) {
            vbr_artifact_companion_payload companion;
            companion.kind = vbr_artifact_companion_kind::recurrent;
            companion.format_version = 1;
            companion.build_identity_digest.fill(0x6a);
            companion.domain = {
                llama_cache_acct_residency::pageable_host,
                llama_cache_acct_domain_kind::not_applicable,
                UINT32_MAX, UINT16_MAX,
            };
            companion.payload_bytes = 4;
            manifest.companions.push_back(companion);
        }
        manifests.push_back(std::move(manifest));
        targets.push_back(std::move(target));
    }
    vbr_capture_projection projection;
    CHECK(vbr_artifact_project_capture_union(
        { 707, std::move(manifests) }, {}, projection));
    std::vector<vbr_capture_projected_unit> units;
    for (const auto & target : targets) {
        units.push_back(
            capture_projected_unit_for_target(projection, target));
    }
    projected_controller_fixture controller;
    controller.rejected_manifest = 7;
    vbr_capture_manifest_assembly assembly;
    CHECK(assemble_projected_test_batch(
        projection, targets, units, controller.provider(), {}, assembly));
    CHECK(assembly.manifests().size() == 8);
    CHECK(projected_manifest(assembly, 7)->state ==
          vbr_capture_manifest_state::dependency_unavailable);

    const auto topology = capture_test_topology();
    const uint8_t recurrent_state[] { 9, 8, 7, 6 };
    const auto make_publications = [&]() {
        std::vector<vbr_projected_manifest_publication> values;
        for (uint64_t i = 1; i <= 8; ++i) {
            values.push_back(projected_publication(i, assembly, topology));
        }
        // Manifest 8 declares one required companion but supplies no sealed
        // evidence. It must not poison the six independent publications.
        // Manifest 6 carries real companion evidence. The projected metadata
        // authority hashes and accounts it; malformed bytes cannot hide
        // behind a caller-provided nonzero digest.
        auto & companion_publication = values[5];
        vbr_artifact_companion_payload companion;
        companion.kind = vbr_artifact_companion_kind::recurrent;
        companion.format_version = 1;
        companion.build_identity_digest.fill(0x6a);
        companion.domain = {
            llama_cache_acct_residency::pageable_host,
            llama_cache_acct_domain_kind::not_applicable,
            UINT32_MAX, UINT16_MAX,
        };
        companion.payload_bytes = sizeof(recurrent_state);
        companion_publication.accounting.push_back({
            vbr_artifact_accounting_role::recurrent_payload,
            companion.domain,
            companion.payload_bytes, companion.payload_bytes,
            llama_cache_acct_attr_kind::artifact,
        });
        auto companion_bytes = std::make_unique<artifact_segment_chain>();
        CHECK(companion_bytes->append(
            recurrent_state, sizeof(recurrent_state)));
        vbr_capture_sealed_companion sealed;
        CHECK(vbr_capture_seal_companion(
            0, std::move(companion_bytes), sealed));
        companion_publication.companions.push_back(std::move(sealed));
        return values;
    };

    llama_cache_acct_ledger ledger;
    llama_vbr_artifact_catalog catalog(ledger);
    std::vector<llama_vbr_artifact_domain_binding> bindings;
    CHECK(catalog.bind_topologies({ topology }, bindings));
    CHECK(bindings.size() == 1);

    // Whole-inventory structural refusal is preflight-only. A mismatched late
    // ID must not publish the earlier matching rows or charge the ledger.
    auto malformed_inventory = make_publications();
    malformed_inventory.back().manifest_id = 9;
    std::vector<vbr_projected_manifest_publish_result> malformed_results;
    CHECK(!catalog.publish_projected_batch(
        assembly, std::move(malformed_inventory), {}, malformed_results));
    CHECK(malformed_results.empty());
    CHECK(catalog.snapshot().references == 0);
    CHECK(ledger.snapshot().live_ops == 0);
    auto publications = make_publications();
    llama_cache_acct_resource_domain device;
    CHECK(ledger.make_device_domain(
        topology, llama_cache_acct_device_ordinal { 0 }, device));
    const auto host = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    const llama_cache_acct_completeness_requirement requirements[] {
        { device, llama_cache_acct_producer::live_memory },
        { host, llama_cache_acct_producer::retention_sidecar },
    };
    CHECK(ledger.configure_required_producers(requirements, 2));
    vbr_artifact_package accounting_package;
    accounting_package.topologies = publications.front().topologies;
    accounting_package.manifest.accounting =
        publications.front().accounting;
    CHECK(catalog.configure_accounting(accounting_package));
    for (const auto category : {
            llama_cache_acct_category::live_attention_state,
            llama_cache_acct_category::live_recurrent_state,
            llama_cache_acct_category::recurrent_rollback_planes,
            llama_cache_acct_category::rolling_window_tape }) {
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated,
                llama_cache_acct_measure::reserved }) {
            ledger.gauge_set(category, device, measure, 0);
        }
    }
    for (const auto category : {
            llama_cache_acct_category::full_snapshot_payload,
            llama_cache_acct_category::checkpoint_state_payload,
            llama_cache_acct_category::typed_accelerator_payload }) {
        ledger.gauge_set(
            category, host,
            llama_cache_acct_measure::resident_allocated, 0);
        ledger.gauge_set(
            category, host,
            llama_cache_acct_measure::reserved, 0);
    }
    CHECK(ledger.certify_complete(
        device, llama_cache_acct_producer::live_memory));
    CHECK(ledger.certify_complete(
        host, llama_cache_acct_producer::retention_sidecar));
    llama_cache_budget_config budget;
    llama_cache_budget_device_input input;
    input.backend_device = reinterpret_cast<const void *>(uintptr_t(1));
    input.domain = device;
    input.physical_total = 1ull << 30;
    input.physical_free = 1ull << 29;
    input.phys_state = llama_cache_budget_capacity_state::known;
    input.current_compute_allocated = 0;
    input.configured_compute_reserve = 0;
    input.compute_state = llama_cache_budget_capacity_state::known;
    input.cache_cap_state = llama_cache_budget_capacity_state::unbounded;
    budget.devices.push_back(input);
    budget.host.pageable_state =
        llama_cache_budget_capacity_state::unbounded;
    budget.host.pinned_state =
        llama_cache_budget_capacity_state::unbounded;
    budget.host.total_state =
        llama_cache_budget_capacity_state::unbounded;
    budget.global_cap_state =
        llama_cache_budget_capacity_state::unbounded;
    std::vector<vbr_projected_manifest_publish_result> results;
    vbr_projected_batch_publish_diagnostics diagnostics;
    CHECK(catalog.publish_projected_batch(
        assembly, std::move(publications), budget, results, &diagnostics));
    CHECK(results.size() == 8);
    CHECK(diagnostics.ready_manifests == 7);
    CHECK(diagnostics.published_manifests == 6);
    CHECK(diagnostics.dependency_unavailable == 1);
    CHECK(diagnostics.main_payload_bytes_rehashed == 0);
    CHECK(diagnostics.companion_payload_hash_bytes ==
          3*sizeof(recurrent_state));
    CHECK(results[6].status ==
          vbr_projected_manifest_publish_status::dependency_unavailable);
    CHECK(results[7].status ==
          vbr_projected_manifest_publish_status::companion_unavailable);
    const auto snapshot = catalog.snapshot();
    CHECK(snapshot.references == 6);
    CHECK(snapshot.blobs == 6);

    std::vector<llama_cache_acct_artifact_id> references;
    for (const auto & result : results) {
        if (result.status != vbr_projected_manifest_publish_status::published &&
            result.status != vbr_projected_manifest_publish_status::adopted) {
            continue;
        }
        references.push_back(result.publication.reference_artifact);
        vbr_artifact_package_view view;
        CHECK(catalog.resolve_reference(
            result.publication.reference_artifact, view) ==
              vbr_artifact_resolve_status::ok);
        CHECK(view.validate() == vbr_artifact_status::ok);
        CHECK(view.projected_ranges().size() == 1);
        CHECK(view.manifest().generation.status ==
              vbr_checkpoint_generation_status::complete);
        CHECK(view.manifest().stream_placements.size() == 1);
        if (result.manifest_id == 6) {
            CHECK(view.manifest().companions.size() == 1);
        }
        view.reset();
    }
    for (const auto reference : references) {
        CHECK(catalog.retire(reference) == vbr_artifact_retire_status::retired);
    }
    CHECK(catalog.snapshot().references == 0);
    CHECK(catalog.snapshot().blobs == 0);
    CHECK(ledger.snapshot().live_ops == 0);

    // A projected artifact stores only the cited packed union. Its live
    // controller watermark remains eight rows and must not be conflated with
    // the two-row storage extent.
    auto sparse_target = projected_target(
        90, 0, 909, projected_generation(21));
    vbr_capture_projection_manifest sparse_manifest;
    sparse_manifest.manifest_id = 90;
    sparse_manifest.placements.push_back(
        projected_placement(90, 90, { 2, 5 }));
    bind_projected_manifest_metadata(sparse_manifest, sparse_target);
    vbr_capture_projection sparse_projection;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { sparse_manifest } }, {}, sparse_projection));
    std::vector<vbr_capture_projected_unit> sparse_units {
        capture_projected_unit_for_target(
            sparse_projection, sparse_target),
    };
    vbr_capture_manifest_assembly sparse_assembly;
    projected_controller_fixture sparse_controller;
    CHECK(assemble_projected_test_batch(
        sparse_projection, { sparse_target }, sparse_units,
        sparse_controller.provider(), {}, sparse_assembly));
    auto sparse_publication = projected_publication(
        90, sparse_assembly, topology);
    vbr_artifact_package sparse_accounting_package;
    sparse_accounting_package.topologies = sparse_publication.topologies;
    sparse_accounting_package.manifest.accounting =
        sparse_publication.accounting;
    CHECK(catalog.configure_accounting(sparse_accounting_package));
    results.clear();
    diagnostics = {};
    std::vector<vbr_projected_manifest_publication> sparse_publications;
    sparse_publications.push_back(std::move(sparse_publication));
    CHECK(catalog.publish_projected_batch(
        sparse_assembly, std::move(sparse_publications), budget,
        results, &diagnostics));
    CHECK(results.size() == 1);
    CHECK(results[0].status ==
          vbr_projected_manifest_publish_status::published);
    vbr_artifact_package_view sparse_view;
    CHECK(catalog.resolve_reference(
        results[0].publication.reference_artifact, sparse_view) ==
          vbr_artifact_resolve_status::ok);
    CHECK(sparse_view.validate() == vbr_artifact_status::ok);
    CHECK(sparse_view.units().size() == 1);
    CHECK(sparse_view.units()[0].descriptor.wm_cells == 8);
    CHECK(sparse_view.units()[0].descriptor.shards[0].row_count == 2);
    sparse_view.reset();
    CHECK(catalog.retire(results[0].publication.reference_artifact) ==
          vbr_artifact_retire_status::retired);
    CHECK(ledger.snapshot().live_ops == 0);
}

static void test_manifest_coherent_assembly() {
    vbr_capture_projection_manifest first;
    first.manifest_id = 10;
    first.placements.push_back(projected_placement(10, 10, { 1, 2 }));
    vbr_capture_projection_manifest second;
    second.manifest_id = 20;
    auto second_placement = projected_placement(20, 20, { 3, 4 });
    second_placement.child_id = 1;
    second.placements.push_back(std::move(second_placement));
    vbr_capture_projection_manifest shared;
    shared.manifest_id = 30;
    shared.placements.push_back(projected_placement(30, 30, { 2, 5 }));
    auto shared_second = projected_placement(30, 31, { 4, 6 });
    shared_second.child_id = 1;
    shared.placements.push_back(std::move(shared_second));
    vbr_capture_projection projection;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { shared, second, first } }, {}, projection));

    const auto generation_a = projected_generation(5);
    const auto generation_b = projected_generation(7, GGML_TYPE_Q8_0);
    const auto generation_c = projected_generation(9, GGML_TYPE_Q6_K);
    auto target_first = projected_target(10, 0, 101, generation_a);
    auto target_second = projected_target(20, 1, 202, generation_b);
    auto target_shared_a = projected_target(30, 0, 101, generation_a);
    auto target_shared_b = projected_target(30, 1, 303, generation_c);

    // Representation equality deliberately ignores reference-local identity,
    // but authenticates policy and the complete pointer-free unit schema.
    CHECK(vbr_capture_controller_representation_equal(
        target_first, target_shared_a));
    auto schema_mutant = target_shared_a;
    schema_mutant.policy.cursor++;
    CHECK(!vbr_capture_controller_representation_equal(
        target_first, schema_mutant));
    schema_mutant = target_shared_a;
    schema_mutant.unit_descriptors[0].shards[0].payload_bytes++;
    CHECK(!vbr_capture_controller_representation_equal(
        target_first, schema_mutant));
    schema_mutant = target_shared_a;
    schema_mutant.unit_descriptors[0].shards[0].section_checksum[0] ^= 1;
    CHECK(!vbr_capture_controller_representation_equal(
        target_first, schema_mutant));

    auto stash_target = target_first;
    auto stash_peer = target_shared_a;
    for (auto * value : { &stash_target, &stash_peer }) {
        auto & descriptor = value->unit_descriptors[0];
        descriptor.clean_stash_state =
            vbr_artifact_clean_stash_state::present;
        descriptor.clean_stash.valid_rows = 4;
        descriptor.clean_stash.domain = vbr_repr_domain::full;
        descriptor.clean_stash.layout = vbr_artifact_layout::row_major;
        descriptor.clean_stash.row_count = 4;
        descriptor.clean_stash.column_count = 1;
        descriptor.clean_stash.row_bytes = 2;
        std::array<uint8_t, 32> stash_id = {};
        stash_id.fill(0x5a);
        descriptor.clean_stash.payload_id =
            vbr_stash_payload_id::from_sha256(stash_id);
        auto stash_shard = descriptor.shards[0];
        stash_shard.row_count = 4;
        stash_shard.row_bytes = 2;
        stash_shard.payload_bytes = 8;
        stash_shard.section_checksum.fill(0x6b);
        descriptor.clean_stash.shards = { stash_shard };
    }
    CHECK(vbr_capture_controller_representation_equal(
        stash_target, stash_peer));
    stash_peer.unit_descriptors[0].clean_stash.row_bytes++;
    CHECK(!vbr_capture_controller_representation_equal(
        stash_target, stash_peer));
    stash_peer = stash_target;
    stash_peer.unit_descriptors[0].clean_stash.shards[0]
        .section_checksum[0] ^= 1;
    CHECK(!vbr_capture_controller_representation_equal(
        stash_target, stash_peer));

    std::vector<vbr_capture_controller_target> targets {
        target_shared_b, target_first, target_shared_a, target_second,
    };
    std::vector<vbr_capture_projected_unit> units {
        capture_projected_unit_for_target(projection, target_shared_b),
        capture_projected_unit_for_target(projection, target_second),
        capture_projected_unit_for_target(projection, target_first),
    };

    projected_controller_fixture controller;
    vbr_capture_manifest_assembly assembled;
    CHECK(assemble_projected_test_batch(
        projection, targets, units, controller.provider(), {}, assembled));
    CHECK(bool(assembled));
    CHECK(controller.rechecked == std::vector<uint64_t>({ 10, 20, 30 }));
    CHECK(assembled.controller_targets().size() == 4);
    CHECK(assembled.projected_units().size() == 3);
    CHECK(assembled.range_proofs().size() == 4);
    CHECK(assembled.manifests().size() == 3);
    CHECK(assembled.manifests()[0].manifest_id == 10);
    CHECK(assembled.manifests()[1].manifest_id == 20);
    CHECK(assembled.manifests()[2].manifest_id == 30);
    for (const auto & manifest : assembled.manifests()) {
        CHECK(manifest.state == vbr_capture_manifest_state::ready);
    }
    const auto * manifest_first = projected_manifest(assembled, 10);
    const auto * manifest_shared = projected_manifest(assembled, 30);
    CHECK(manifest_first != nullptr);
    CHECK(manifest_shared != nullptr);
    if (manifest_first && manifest_shared) {
        CHECK(manifest_first->controller_count == 1);
        CHECK(manifest_first->unit_count == 1);
        CHECK(manifest_first->range_proof_count == 1);
        CHECK(manifest_shared->controller_count == 2);
        CHECK(manifest_shared->unit_count == 2);
        CHECK(manifest_shared->range_proof_count == 2);
        const uint32_t first_unit = assembled.unit_references()[
            manifest_first->first_unit];
        const uint32_t shared_unit = assembled.unit_references()[
            manifest_shared->first_unit];
        CHECK(first_unit == shared_unit);
        const auto & first_proof = assembled.range_proofs()[
            manifest_first->first_range_proof].proof;
        const auto & shared_proof = assembled.range_proofs()[
            manifest_shared->first_range_proof].proof;
        CHECK(first_proof.ranges().size() == 1);
        CHECK(shared_proof.ranges().size() == 1);
        if (first_proof.ranges().size() == 1 &&
            shared_proof.ranges().size() == 1) {
            CHECK(first_proof.ranges()[0].offset == 0);
            CHECK(first_proof.ranges()[0].size == 2);
            CHECK(shared_proof.ranges()[0].offset == 1);
            CHECK(shared_proof.ranges()[0].size == 2);
        }
    }
    for (const auto & range : assembled.range_proofs()) {
        CHECK(range.unit_index < assembled.projected_units().size());
        if (range.unit_index >= assembled.projected_units().size()) {
            continue;
        }
        const auto & unit = assembled.projected_units()[range.unit_index];
        CHECK(range.shard_index < unit.shards().size());
        if (range.shard_index < unit.shards().size()) {
            uint64_t bytes_read = 0;
            CHECK(vbr_capture_range_verify(
                range.proof,
                unit.shards()[range.shard_index].bytes->source(),
                &bytes_read));
            CHECK(bytes_read ==
                  unit.shards()[range.shard_index].bytes->size());
        }
    }
    uint64_t all_ready_proof_metadata = 0;
    for (const auto & range : assembled.range_proofs()) {
        all_ready_proof_metadata += range.proof.metadata_bytes() +
            2*sizeof(vbr_capture_manifest_range_proof);
    }
    CHECK(all_ready_proof_metadata > 80);

    // Losing one target generation invalidates only manifests that name it;
    // independently sealed units and manifests remain available.
    auto partial_units = units;
    partial_units.erase(partial_units.begin());
    controller = {};
    CHECK(assemble_projected_test_batch(
        projection, targets, partial_units,
        controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 10)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 20)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 30)->state ==
          vbr_capture_manifest_state::dependency_unavailable);
    CHECK(projected_manifest(assembled, 30)->range_proof_count == 0);
    CHECK(assembled.projected_units().size() == 2);
    CHECK(assembled.range_proofs().size() == 2);

    auto stale_targets = targets;
    stale_targets.front().controller_generation = 404;
    stale_targets.front().policy.cursor = 404;
    controller = {};
    CHECK(assemble_projected_test_batch(
        projection, stale_targets, units,
        controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 10)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 20)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 30)->state ==
          vbr_capture_manifest_state::dependency_unavailable);
    CHECK(assembled.projected_units().size() == 2);
    CHECK(projected_manifest(assembled, 30)->controller_count == 0);
    CHECK(projected_manifest(assembled, 30)->unit_count == 0);

    stale_targets = targets;
    stale_targets.front().units[0].repr_gen++;
    stale_targets.front().unit_descriptors[0].repr_gen++;
    controller = {};
    CHECK(assemble_projected_test_batch(
        projection, stale_targets, units,
        controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 10)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 30)->state ==
          vbr_capture_manifest_state::dependency_unavailable);

    stale_targets = targets;
    stale_targets.front().lineage_uuid.lo++;
    stale_targets.front().unit_descriptors[0].lineage_uuid.lo++;
    controller = {};
    CHECK(assemble_projected_test_batch(
        projection, stale_targets, units,
        controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 10)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 20)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 30)->state ==
          vbr_capture_manifest_state::dependency_unavailable);

    auto missing_targets = targets;
    missing_targets.erase(missing_targets.begin());
    controller = {};
    CHECK(assemble_projected_test_batch(
        projection, missing_targets, units,
        controller.provider(), {}, assembled));
    CHECK(controller.rechecked == std::vector<uint64_t>({ 10, 20 }));
    CHECK(projected_manifest(assembled, 10)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 20)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 30)->state ==
          vbr_capture_manifest_state::dependency_unavailable);

    controller = {};
    CHECK(assemble_projected_test_batch(
        projection, {}, units, controller.provider(), {}, assembled));
    CHECK(controller.rechecked.empty());
    CHECK(assembled.manifests().size() == 3);
    CHECK(std::all_of(
        assembled.manifests().begin(), assembled.manifests().end(),
        [](const auto & manifest) {
            return manifest.state ==
                vbr_capture_manifest_state::dependency_unavailable;
        }));
    CHECK(assembled.projected_units().empty());

    // A manifest cannot relabel an otherwise shared captured unit. Complete
    // descriptor schema is controller-certified across every alias.
    auto mismatched_schema = targets;
    mismatched_schema[2].unit_descriptors[0].side =
        vbr_artifact_side::value;
    CHECK(!assemble_projected_test_batch(
        projection, mismatched_schema, units,
        controller.provider(), {}, assembled));
    CHECK(!assembled);

    controller = {};
    controller.rejected_manifest = 30;
    CHECK(assemble_projected_test_batch(
        projection, targets, units, controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 10)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 20)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 30)->state ==
          vbr_capture_manifest_state::dependency_unavailable);
    CHECK(assembled.projected_units().size() == 2);

    // Exact flat-arena bounds are accepted; one-less limits fail
    // transactionally and clear an earlier successful capability.
    vbr_capture_manifest_assembly_limits exact;
    exact.max_controller_targets = 4;
    exact.max_projected_units = 3;
    exact.max_unit_descriptor_shards = 4;
    exact.max_unit_descriptor_metadata_bytes =
        4*(sizeof(vbr_artifact_unit_descriptor) +
           sizeof(vbr_artifact_shard_descriptor));
    exact.max_manifests = 3;
    exact.max_controller_references = 4;
    exact.max_unit_references = 4;
    exact.max_range_proofs = 4;
    exact.max_range_proof_metadata_bytes = all_ready_proof_metadata;
    controller = {};
    CHECK(assemble_projected_test_batch(
        projection, targets, units, controller.provider(), exact, assembled));
    const auto expect_limit_refusal = [&](auto tighten) {
        auto limited = exact;
        tighten(limited);
        CHECK(!assemble_projected_test_batch(
            projection, targets, units,
            controller.provider(), limited, assembled));
        CHECK(!assembled);
    };
    expect_limit_refusal([](auto & value) {
        value.max_controller_targets = 3;
    });
    expect_limit_refusal([](auto & value) {
        value.max_projected_units = 2;
    });
    expect_limit_refusal([](auto & value) {
        value.max_unit_descriptor_shards = 3;
    });
    expect_limit_refusal([](auto & value) {
        value.max_unit_descriptor_metadata_bytes--;
    });
    expect_limit_refusal([](auto & value) {
        value.max_manifests = 2;
    });
    expect_limit_refusal([](auto & value) {
        value.max_controller_references = 3;
    });
    expect_limit_refusal([](auto & value) {
        value.max_unit_references = 3;
    });
    expect_limit_refusal([](auto & value) {
        value.max_range_proofs = 3;
    });
    expect_limit_refusal([&](auto & value) {
        value.max_range_proof_metadata_bytes =
            all_ready_proof_metadata - 1;
    });

    auto conflicting_targets = targets;
    conflicting_targets[2].units[0].repr_gen++;
    conflicting_targets[2].policy.current_type_vector_digest =
        target_shared_a.policy.current_type_vector_digest;
    CHECK(!assemble_projected_test_batch(
        projection, conflicting_targets, units,
        controller.provider(), {}, assembled));
    CHECK(!assembled);

    conflicting_targets = targets;
    conflicting_targets[2].lineage_uuid.lo++;
    CHECK(!assemble_projected_test_batch(
        projection, conflicting_targets, units,
        controller.provider(), {}, assembled));
    CHECK(!assembled);

    auto duplicate_targets = targets;
    duplicate_targets.push_back(target_first);
    CHECK(!assemble_projected_test_batch(
        projection, duplicate_targets, units,
        controller.provider(), {}, assembled));
    CHECK(!assembled);

    auto malformed_units = units;
    malformed_units.front() = {};
    CHECK(!assemble_projected_test_batch(
        projection, targets, malformed_units,
        controller.provider(), {}, assembled));
    CHECK(!assembled);

    // One controller tuple spans all streams of a child; every projected
    // stream must supply the complete unit vector before the manifest is ready.
    vbr_capture_projection_manifest multi_stream;
    multi_stream.manifest_id = 40;
    multi_stream.placements.push_back(
        projected_placement(40, 40, { 1, 3 }));
    auto stream_one_placement =
        projected_placement(40, 41, { 2, 4 });
    stream_one_placement.stream_index = 1;
    multi_stream.placements.push_back(std::move(stream_one_placement));
    vbr_capture_projection stream_projection;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { multi_stream } }, {}, stream_projection));
    auto stream_target = projected_target(40, 0, 505, generation_a);
    projected_target_streams(stream_target, 2);
    std::vector<vbr_capture_controller_target> stream_targets {
        stream_target,
    };
    std::vector<vbr_capture_projected_unit> stream_units {
        capture_projected_unit_for_target(stream_projection, stream_target, 1),
        capture_projected_unit_for_target(stream_projection, stream_target, 0),
    };
    controller = {};
    CHECK(assemble_projected_test_batch(
        stream_projection, stream_targets, stream_units,
        controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 40)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 40)->controller_count == 1);
    CHECK(projected_manifest(assembled, 40)->unit_count == 2);
    stream_units.pop_back();
    CHECK(assemble_projected_test_batch(
        stream_projection, stream_targets, stream_units,
        controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 40)->state ==
          vbr_capture_manifest_state::dependency_unavailable);

    vbr_capture_projection_manifest selected_stream;
    selected_stream.manifest_id = 50;
    auto selected_placement = projected_placement(50, 50, { 2, 4 });
    selected_placement.stream_index = 1;
    selected_stream.placements.push_back(std::move(selected_placement));
    vbr_capture_projection selected_projection;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { selected_stream } }, {}, selected_projection));
    auto selected_target = projected_target(50, 0, 606, generation_b);
    projected_target_streams(selected_target, 2);
    std::vector<vbr_capture_controller_target> selected_targets {
        selected_target,
    };
    std::vector<vbr_capture_projected_unit> selected_units {
        capture_projected_unit_for_target(
            selected_projection, selected_target, 1),
    };
    controller = {};
    CHECK(assemble_projected_test_batch(
        selected_projection, selected_targets, selected_units,
        controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 50)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 50)->controller_count == 1);
    CHECK(projected_manifest(assembled, 50)->unit_count == 1);

    // Manifest proofs select only their own canonical chunks even when one
    // projected unit physically packs several independent prefixes.
    vbr_capture_projection_manifest chunk_zero;
    chunk_zero.manifest_id = 60;
    chunk_zero.placements.push_back(
        projected_placement(60, 60, { 0 }));
    vbr_capture_projection_manifest chunk_one;
    chunk_one.manifest_id = 70;
    chunk_one.placements.push_back(
        projected_placement(70, 70, { 3 }));
    vbr_capture_projection chunk_projection;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { chunk_zero, chunk_one } }, {}, chunk_projection));
    auto chunk_target_zero = projected_target(60, 0, 707, generation_a);
    auto chunk_target_one = projected_target(70, 0, 707, generation_a);
    projected_target_row_bytes(
        chunk_target_zero, VBR_CAPTURE_RANGE_CHUNK_BYTES);
    projected_target_row_bytes(
        chunk_target_one, VBR_CAPTURE_RANGE_CHUNK_BYTES);
    std::vector<vbr_capture_controller_target> chunk_targets {
        chunk_target_zero, chunk_target_one,
    };
    std::vector<vbr_capture_projected_unit> chunk_units {
        capture_projected_unit_for_target(
            chunk_projection, chunk_target_zero, 0,
            VBR_CAPTURE_RANGE_CHUNK_BYTES),
    };
    controller = {};
    CHECK(assemble_projected_test_batch(
        chunk_projection, chunk_targets, chunk_units,
        controller.provider(), {}, assembled));
    const auto * manifest_zero = projected_manifest(assembled, 60);
    const auto * manifest_one = projected_manifest(assembled, 70);
    CHECK(manifest_zero && manifest_one);
    if (manifest_zero && manifest_one) {
        CHECK(manifest_zero->range_proof_count == 1);
        CHECK(manifest_one->range_proof_count == 1);
        const auto & proof_zero = assembled.range_proofs()[
            manifest_zero->first_range_proof].proof;
        const auto & proof_one = assembled.range_proofs()[
            manifest_one->first_range_proof].proof;
        CHECK(proof_zero.ranges().size() == 1);
        CHECK(proof_one.ranges().size() == 1);
        if (proof_zero.ranges().size() == 1 &&
            proof_one.ranges().size() == 1) {
            CHECK(proof_zero.ranges()[0].offset == 0);
            CHECK(proof_zero.ranges()[0].size ==
                  VBR_CAPTURE_RANGE_CHUNK_BYTES);
            CHECK(proof_one.ranges()[0].offset ==
                  VBR_CAPTURE_RANGE_CHUNK_BYTES);
            CHECK(proof_one.ranges()[0].size ==
                  VBR_CAPTURE_RANGE_CHUNK_BYTES);
        }
        const auto unit_index = assembled.range_proofs()[
            manifest_zero->first_range_proof].unit_index;
        const auto & bytes = assembled.projected_units()[unit_index].
            shards()[0].bytes;
        uint64_t zero_bytes = 0;
        uint64_t one_bytes = 0;
        CHECK(vbr_capture_range_verify(
            proof_zero, bytes->source(), &zero_bytes));
        CHECK(vbr_capture_range_verify(
            proof_one, bytes->source(), &one_bytes));
        CHECK(zero_bytes == VBR_CAPTURE_RANGE_CHUNK_BYTES);
        CHECK(one_bytes == VBR_CAPTURE_RANGE_CHUNK_BYTES);
    }
}

struct generated_h2d_source {
    uint64_t size = 0;

    static uint8_t byte_at(uint64_t offset) noexcept {
        return uint8_t((offset*29 + 17) & 0xff);
    }

    static bool read(
            const void * context, uint64_t offset,
            uint8_t * destination, size_t size) noexcept {
        const auto & source =
            *static_cast<const generated_h2d_source *>(context);
        if (offset > source.size || size > source.size - offset) {
            return false;
        }
        for (size_t i = 0; i < size; ++i) {
            destination[i] = byte_at(offset + i);
        }
        return true;
    }
};

struct fake_h2d_destination {
    struct pending {
        uint64_t offset = 0;
        const uint8_t * data = nullptr;
        size_t size = 0;
        uint64_t digest = 0;
    };
    std::unordered_map<uint64_t, pending> operations;
    uint64_t bytes = 0;
    bool valid = true;

    static uint64_t digest(const uint8_t * data, size_t size) noexcept {
        uint64_t value = 1469598103934665603ull;
        for (size_t i = 0; i < size; ++i) {
            value = (value ^ data[i])*1099511628211ull;
        }
        return value;
    }

    static bool issue(
            void * context, uint64_t ticket, uint64_t offset,
            const uint8_t * data, size_t size,
            bool) noexcept {
        auto & destination =
            *static_cast<fake_h2d_destination *>(context);
        if (!data || size == 0 || destination.operations.count(ticket) != 0) {
            return false;
        }
        for (size_t i = 0; i < size; ++i) {
            if (data[i] != generated_h2d_source::byte_at(offset + i)) {
                destination.valid = false;
                return false;
            }
        }
        destination.operations.emplace(ticket, pending {
            offset, data, size, digest(data, size),
        });
        return true;
    }

    static bool complete(void * context, uint64_t ticket) noexcept {
        auto & destination =
            *static_cast<fake_h2d_destination *>(context);
        const auto found = destination.operations.find(ticket);
        if (found == destination.operations.end()) {
            return false;
        }
        // If the ring reused this pinned chunk before completion, its digest
        // changed and the fake event fails.
        if (digest(found->second.data, found->second.size) !=
                found->second.digest) {
            destination.valid = false;
            return false;
        }
        destination.bytes += found->second.size;
        destination.operations.erase(found);
        return true;
    }
};

static void test_h2d_bounded_streaming() {
    const uint64_t transfer_bytes =
        VBR_PINNED_RING_MAX_BYTES + 1024*1024 + 17;
    generated_h2d_source generated { transfer_bytes };
    vbr_h2d_status status;
    auto ring = vbr_h2d_chunk_ring::create(
        { {} }, 8*1024*1024, 4*1024*1024, status);
    CHECK(ring && status == vbr_h2d_status::ok);
    CHECK(ring->capacity_bytes() < transfer_bytes);

    fake_h2d_destination event_destination;
    vbr_h2d_transfer transfer;
    transfer.source = {
        transfer_bytes, &generated, generated_h2d_source::read,
    };
    transfer.size = transfer_bytes;
    transfer.fake = {
        &event_destination,
        fake_h2d_destination::issue,
        fake_h2d_destination::complete,
        true,
    };
    vbr_h2d_stats stats;
    CHECK(ring->stream(transfer, stats) == vbr_h2d_status::ok);
    CHECK(event_destination.valid);
    CHECK(event_destination.operations.empty());
    CHECK(event_destination.bytes == transfer_bytes);
    CHECK(stats.bytes == transfer_bytes);
    CHECK(stats.backpressure_waits > 0);
    CHECK(stats.event_completions == stats.chunks);
    CHECK(stats.synchronous_fallbacks == 0);
    CHECK(stats.peak_pinned_bytes <= ring->capacity_bytes());

    fake_h2d_destination sync_destination;
    transfer.fake.context = &sync_destination;
    transfer.fake.supports_events = false;
    CHECK(ring->stream(transfer, stats) == vbr_h2d_status::ok);
    CHECK(sync_destination.valid);
    CHECK(sync_destination.bytes == transfer_bytes);
    CHECK(stats.synchronous_fallbacks == stats.chunks);
    CHECK(stats.event_completions == 0);

    fake_h2d_destination failed;
    transfer.fake.context = &failed;
    transfer.fake.supports_events = true;
    transfer.fail_completion_at = 1;
    CHECK(ring->stream(transfer, stats) ==
          vbr_h2d_status::transfer_failed);
    CHECK(failed.operations.empty());

    // H1 production uses one physical direction-neutral ring. The two
    // adapters share its exact capacity and contend at the whole-operation
    // boundary rather than consuming one another's outstanding chunks.
    vbr_pinned_ring_create_failure failure;
    const uint64_t live_before =
        vbr_pinned_ring_live_capacity_bytes();
    auto core = std::shared_ptr<vbr_bounded_pinned_ring_core>(
        vbr_bounded_pinned_ring_core::create(
            { {} }, 16, 8, nullptr, failure));
    CHECK(core);
    CHECK(failure == vbr_pinned_ring_create_failure::none);
    auto capture = vbr_pinned_chunk_ring::attach(core);
    auto adoption = vbr_h2d_chunk_ring::attach(core, { { } });
    CHECK(capture && adoption);
    CHECK(capture && capture->capacity_bytes() == 16);
    CHECK(adoption && adoption->capacity_bytes() == 16);
    CHECK(vbr_pinned_ring_live_capacity_bytes() == live_before + 16);

    blocking_capture_source d2h_input;
    d2h_input.source.bytes.resize(9, 0x5a);
    vbr_capture_stream_source d2h_source;
    d2h_source.size = d2h_input.source.bytes.size();
    d2h_source.context = &d2h_input;
    d2h_source.read = blocking_capture_source::read;
    artifact_segment_chain blocked_chain;
    vbr_capture_stream_stats blocked_stats;
    generated_h2d_source h2d_input { 9 };
    fake_h2d_destination blocked_destination;
    vbr_h2d_transfer blocked_transfer;
    blocked_transfer.source = {
        h2d_input.size, &h2d_input, generated_h2d_source::read,
    };
    blocked_transfer.size = h2d_input.size;
    blocked_transfer.fake = {
        &blocked_destination, fake_h2d_destination::issue,
        fake_h2d_destination::complete, true,
    };
    vbr_capture_stream_status capture_result =
        vbr_capture_stream_status::internal_error;
    std::thread capture_thread([&]() {
        capture_result = capture->stream(
            d2h_source, blocked_chain, blocked_stats);
    });
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(1);
    while (!d2h_input.entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    CHECK(d2h_input.entered.load(std::memory_order_acquire));
    CHECK(adoption->stream(blocked_transfer, stats) ==
          vbr_h2d_status::ring_unavailable);
    CHECK(blocked_chain.size() == 0);
    CHECK(blocked_destination.bytes == 0);
    d2h_input.release.store(true, std::memory_order_release);
    capture_thread.join();
    CHECK(capture_result == vbr_capture_stream_status::ok);
    CHECK(read_chain(blocked_chain) == d2h_input.source.bytes);
    CHECK(adoption->stream(blocked_transfer, stats) == vbr_h2d_status::ok);
    CHECK(blocked_destination.valid);
    CHECK(blocked_destination.bytes == h2d_input.size);

    capture.reset();
    adoption.reset();
    CHECK(vbr_pinned_ring_live_capacity_bytes() == live_before + 16);
    core.reset();
    CHECK(vbr_pinned_ring_live_capacity_bytes() == live_before);
}

static uint64_t ring_resident(
        llama_cache_acct_ledger & ledger,
        const llama_cache_acct_resource_domain & domain) {
    for (const auto & row : ledger.snapshot().cells) {
        if (row.category ==
                llama_cache_acct_category::pinned_preimage_ring &&
            row.domain == domain) {
            return row.cell.measures[size_t(
                llama_cache_acct_measure::
                    resident_allocated)].value;
        }
    }
    return 0;
}

struct ring_charge_fault_context {
    llama_cache_acct_ledger * ledger = nullptr;
};

static void inject_ring_charge_fault(void * opaque) noexcept {
    auto & context = *static_cast<ring_charge_fault_context *>(opaque);
    context.ledger->gauge_set(
        llama_cache_acct_category::pinned_preimage_ring,
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pageable_host),
        llama_cache_acct_measure::resident_allocated, 1);
}

static void test_ring_accounting_once() {
    llama_cache_acct_ledger ledger;
    const auto domain =
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pinned_host);
    const llama_cache_acct_completeness_requirement required {
        domain, llama_cache_acct_producer::retention_sidecar,
    };
    CHECK(ledger.configure_required_producers(&required, 1));
    CHECK(server_vbr_artifact_store_configure_pinned_accounting(
        ledger, domain));
    llama_cache_budget_config budget;
    budget.host.pinned_cap = 1024;
    budget.host.pinned_state =
        llama_cache_budget_capacity_state::known;
    vbr_capture_ring_accounting accounting {
        &ledger, domain, &budget,
    };
    {
        const auto snapshot = ledger.snapshot();
        llama_cache_budget_coordinator coordinator;
        CHECK(coordinator.reset(snapshot, budget));
        llama_cache_budget_plan plan;
        plan.accounting_serial = snapshot.serial;
        plan.entries.push_back({ domain, 16, 0 });
        const auto fit = coordinator.fits(plan);
        CHECK(fit.state == llama_cache_budget_fit_state::fits);
    }
    vbr_capture_stream_status status;
    vbr_capture_ring_create_failure failure;
    llama_cache_budget_config refused_budget = budget;
    refused_budget.host.pinned_cap = 8;
    vbr_capture_ring_accounting refused_accounting {
        &ledger, domain, &refused_budget,
    };
    auto refused = vbr_pinned_chunk_ring::create(
        { {} }, 16, 8, status, &refused_accounting, &failure);
    CHECK(!refused);
    CHECK(status == vbr_capture_stream_status::accounting_refused);
    CHECK(failure ==
          vbr_capture_ring_create_failure::budget_exceeded);

    // M1: a fault raised after physical chunk allocation but during the final
    // C gauge remains accounting_unavailable, exactly as before ring factoring.
    ring_charge_fault_context fault_context { &ledger };
    auto charge_fault_accounting = accounting;
    charge_fault_accounting.charge_fault_context = &fault_context;
    charge_fault_accounting.inject_charge_fault =
        inject_ring_charge_fault;
    auto charge_failed = vbr_pinned_chunk_ring::create(
        { {} }, 16, 8, status, &charge_fault_accounting, &failure);
    CHECK(!charge_failed);
    CHECK(status == vbr_capture_stream_status::accounting_unavailable);
    CHECK(failure ==
          vbr_capture_ring_create_failure::accounting_charge_failed);
    CHECK(ring_resident(ledger, domain) == 0);

    auto ring = vbr_pinned_chunk_ring::create(
        { {} }, 16, 8, status, &accounting, &failure);
    CHECK(ring);
    CHECK(failure == vbr_capture_ring_create_failure::none);
    CHECK(ring_resident(ledger, domain) == 16);
    auto duplicate = vbr_pinned_chunk_ring::create(
        { {} }, 16, 8, status, &accounting);
    CHECK(!duplicate);
    CHECK(ring_resident(ledger, domain) == 16);
    ring.reset();
    CHECK(ring_resident(ledger, domain) == 0);
}

static bool sample_unbounded_host_budget(
        void *,
        llama_cache_budget_config & output) noexcept {
    output = {};
    output.host.pageable_state =
        llama_cache_budget_capacity_state::unbounded;
    output.host.pinned_cap = 1024;
    output.host.pinned_state =
        llama_cache_budget_capacity_state::known;
    output.host.total_state =
        llama_cache_budget_capacity_state::unbounded;
    output.global_cap_state =
        llama_cache_budget_capacity_state::unbounded;
    return true;
}

static vbr_artifact_portable_topology capture_test_topology() {
    llama_cache_acct_shard_topology topology;
    CHECK(llama_cache_acct_build_shard_topology(
        { "synthetic-capture-device" },
        LLAMA_SPLIT_MODE_NONE, 0, nullptr, topology));
    return topology;
}

struct projected_store_budget_context {
    llama_cache_acct_resource_domain device;
    bool available = true;
};

static bool sample_projected_store_budget(
        void * opaque,
        llama_cache_budget_config & output) noexcept {
    const auto * context =
        static_cast<const projected_store_budget_context *>(opaque);
    if (!context || !context->available) {
        return false;
    }
    output = {};
    llama_cache_budget_device_input input;
    input.backend_device = reinterpret_cast<const void *>(uintptr_t(1));
    input.domain = context->device;
    input.physical_total = 1ull << 30;
    input.physical_free = 1ull << 29;
    input.phys_state = llama_cache_budget_capacity_state::known;
    input.current_compute_allocated = 0;
    input.configured_compute_reserve = 0;
    input.compute_state = llama_cache_budget_capacity_state::known;
    input.cache_cap_state =
        llama_cache_budget_capacity_state::unbounded;
    output.devices.push_back(input);
    output.host.pageable_state =
        llama_cache_budget_capacity_state::unbounded;
    output.host.pinned_state =
        llama_cache_budget_capacity_state::unbounded;
    output.host.total_state =
        llama_cache_budget_capacity_state::unbounded;
    output.global_cap_state =
        llama_cache_budget_capacity_state::unbounded;
    return true;
}

static void test_projected_host_batch_store_adapter() {
    const auto topology = capture_test_topology();
    std::vector<vbr_capture_projection_manifest> manifests;
    std::vector<vbr_capture_controller_target> targets;
    for (uint64_t id = 1; id <= 2; ++id) {
        vbr_capture_projection_manifest manifest;
        manifest.manifest_id = id;
        manifest.placements.push_back(projected_placement(
            id, llama_seq_id(id), { uint32_t(id - 1) }));
        auto target = projected_target(
            id, 0, 700 + id, projected_generation(40 + id));
        bind_projected_manifest_metadata(manifest, target);
        manifests.push_back(std::move(manifest));
        targets.push_back(std::move(target));
    }
    vbr_capture_projection projection;
    CHECK(vbr_artifact_project_capture_union(
        { 707, std::move(manifests) }, {}, projection));
    std::vector<vbr_capture_projected_unit> units;
    for (const auto & target : targets) {
        units.push_back(capture_projected_unit_for_target(
            projection, target));
    }
    projected_controller_fixture controller;
    controller.rejected_manifest = 2;
    vbr_capture_manifest_assembly assembly;
    CHECK(assemble_projected_test_batch(
        projection, targets, units, controller.provider(), {}, assembly));

    const auto make_publications = [&](const auto & source_assembly) {
        std::vector<vbr_projected_manifest_publication> publications;
        publications.push_back(projected_publication(
            1, source_assembly, topology));
        publications.push_back(projected_publication(
            2, source_assembly, topology));
        return publications;
    };

    llama_cache_acct_ledger ledger;
    llama_cache_acct_resource_domain device;
    CHECK(ledger.make_device_domain(
        topology, llama_cache_acct_device_ordinal { 0 }, device));
    const auto pageable = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    const auto pinned = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pinned_host);
    const llama_cache_acct_completeness_requirement requirements[] {
        { device, llama_cache_acct_producer::live_memory },
        { pageable, llama_cache_acct_producer::retention_sidecar },
        { pinned, llama_cache_acct_producer::retention_sidecar },
    };
    CHECK(ledger.configure_required_producers(requirements, 3));
    for (const auto category : {
            llama_cache_acct_category::live_attention_state,
            llama_cache_acct_category::live_recurrent_state,
            llama_cache_acct_category::recurrent_rollback_planes,
            llama_cache_acct_category::rolling_window_tape }) {
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated,
                llama_cache_acct_measure::reserved }) {
            ledger.gauge_set(category, device, measure, 0);
        }
    }
    CHECK(server_vbr_artifact_store_observe_empty_accounting(
        ledger, device));
    CHECK(server_vbr_artifact_store_observe_empty_accounting(
        ledger, pageable));
    CHECK(ledger.certify_complete(
        device, llama_cache_acct_producer::live_memory));
    CHECK(ledger.certify_complete(
        pageable, llama_cache_acct_producer::retention_sidecar));
    CHECK(server_vbr_artifact_store_configure_pinned_accounting(
        ledger, pinned));

    projected_store_budget_context budget_context { device, true };
    server_vbr_artifact_store_config config;
    config.ledger = &ledger;
    config.pinned_domain = pinned;
    config.topologies.push_back(topology);
    config.pool_bindings.push_back({
        { 0x3333, 0x4444 }, 0, 0, 0, 0,
    });
    config.lanes.push_back({});
    config.attention_children = 1;
    config.ring_bytes = 16;
    config.chunk_bytes = 8;
    config.budget_context = &budget_context;
    config.sample_budget = sample_projected_store_budget;
    server_vbr_artifact_capture_status create_status;
    auto store = server_vbr_artifact_store::create(
        config, create_status);
    CHECK(store);
    CHECK(create_status == server_vbr_artifact_capture_status::ok);
    if (!store) {
        return;
    }

    const uint64_t baseline_ops = ledger.snapshot().live_ops;
    std::vector<server_vbr_projected_host_publish_result> results;
    server_vbr_projected_host_publish_diagnostics diagnostics;
    CHECK(store->publish_projected_host_batch(
        assembly, make_publications(assembly), results, &diagnostics));
    CHECK(results.size() == 2);
    if (results.size() != 2) {
        return;
    }
    CHECK(results[0].manifest_id == 1);
    CHECK(results[0].status ==
              vbr_projected_manifest_publish_status::published ||
          results[0].status ==
              vbr_projected_manifest_publish_status::adopted);
    CHECK(results[0].payload);
    if (results[0].payload) {
        CHECK(results[0].payload->retirement_owned());
        CHECK(results[0].payload->accounted_by(&ledger));
        CHECK(results[0].payload->reference_artifact().v != 0);
    }
    CHECK(results[1].manifest_id == 2);
    CHECK(results[1].status ==
          vbr_projected_manifest_publish_status::dependency_unavailable);
    CHECK(!results[1].payload);
    CHECK(diagnostics.catalog.ready_manifests == 1);
    CHECK(diagnostics.catalog.published_manifests == 1);
    CHECK(diagnostics.catalog.dependency_unavailable == 1);
    CHECK(diagnostics.host_payloads_retained == 1);
    CHECK(diagnostics.postpublish_retirements == 0);
    CHECK(ledger.snapshot().live_ops > baseline_ops);
    // The typed host door accepts this store-owned capability without a
    // tenant reference and reaches the shared import kernel. The intentionally
    // empty target request then fails at kernel validation, not authorization.
    const auto authenticated_before =
        store->counters().host_imports_authenticated;
    server_vbr_artifact_import_target empty_import;
    const auto host_import = store->import_host_payload(
        std::move(empty_import), results[0].payload);
    CHECK(host_import.status ==
          server_vbr_artifact_import_status::unavailable);
    CHECK(store->counters().host_imports_authenticated ==
          authenticated_before + 1);
    results.clear();
    CHECK(ledger.snapshot().live_ops == baseline_ops);

    auto malformed = make_publications(assembly);
    malformed.back().manifest_id = 3;
    results.resize(1);
    diagnostics.catalog.ready_manifests = 99;
    CHECK(!store->publish_projected_host_batch(
        assembly, std::move(malformed), results, &diagnostics));
    CHECK(results.empty());
    CHECK(diagnostics.catalog.ready_manifests == 0);
    CHECK(ledger.snapshot().live_ops == baseline_ops);

    // A failure after both catalog rows commit must retire only the failed
    // row. Its sibling remains a typed host owner, and releasing that owner
    // returns the complete catalog/ledger state to the pre-batch baseline.
    controller.rejected_manifest = 0;
    vbr_capture_manifest_assembly ready_assembly;
    CHECK(assemble_projected_test_batch(
        projection, targets, units, controller.provider(), {},
        ready_assembly));
    server_vbr_artifact_store_test_door::
        fail_projected_host_adoption_once(*store);
    CHECK(store->publish_projected_host_batch(
        ready_assembly, make_publications(ready_assembly),
        results, &diagnostics));
    CHECK(results.size() == 2);
    if (results.size() == 2) {
        CHECK(results[0].status ==
              vbr_projected_manifest_publish_status::internal_error);
        CHECK(!results[0].payload);
        CHECK(results[1].status ==
                  vbr_projected_manifest_publish_status::published ||
              results[1].status ==
                  vbr_projected_manifest_publish_status::adopted);
        CHECK(results[1].payload);
    }
    CHECK(diagnostics.catalog.published_manifests == 2);
    CHECK(diagnostics.host_payloads_retained == 1);
    CHECK(diagnostics.postpublish_retirements == 1);
    CHECK(ledger.snapshot().live_ops > baseline_ops);
    results.clear();
    CHECK(ledger.snapshot().live_ops == baseline_ops);

    // Re-publishing the same sealed batch adopts the existing physical unit
    // allocations while creating distinct logical reference owners. Dropping
    // either generation of host owners must preserve the other.
    std::vector<server_vbr_projected_host_publish_result> first_owners;
    CHECK(store->publish_projected_host_batch(
        ready_assembly, make_publications(ready_assembly),
        first_owners, &diagnostics));
    CHECK(first_owners.size() == 2);
    std::vector<llama_cache_acct_artifact_id> first_references;
    for (const auto & row : first_owners) {
        CHECK(row.payload);
        CHECK(row.status ==
                  vbr_projected_manifest_publish_status::published ||
              row.status ==
                  vbr_projected_manifest_publish_status::adopted);
        if (row.payload) {
            first_references.push_back(
                row.payload->reference_artifact());
        }
    }
    CHECK(first_references.size() == 2);
    std::vector<server_vbr_projected_host_publish_result> adopted_owners;
    CHECK(store->publish_projected_host_batch(
        ready_assembly, make_publications(ready_assembly),
        adopted_owners, &diagnostics));
    CHECK(adopted_owners.size() == 2);
    for (size_t i = 0; i < adopted_owners.size(); ++i) {
        CHECK(adopted_owners[i].status ==
              vbr_projected_manifest_publish_status::adopted);
        CHECK(adopted_owners[i].payload);
        if (adopted_owners[i].payload && i < first_references.size()) {
            CHECK(adopted_owners[i].payload->reference_artifact() !=
                  first_references[i]);
        }
    }
    CHECK(diagnostics.catalog.published_manifests == 2);
    CHECK(diagnostics.host_payloads_retained == 2);
    const uint64_t adopted_ops = ledger.snapshot().live_ops;
    CHECK(adopted_ops > baseline_ops);
    first_owners.clear();
    CHECK(ledger.snapshot().live_ops > baseline_ops);
    CHECK(ledger.snapshot().live_ops < adopted_ops);
    adopted_owners.clear();
    CHECK(ledger.snapshot().live_ops == baseline_ops);

    budget_context.available = false;
    results.resize(1);
    diagnostics.catalog.ready_manifests = 99;
    CHECK(!store->publish_projected_host_batch(
        assembly, make_publications(assembly), results, &diagnostics));
    CHECK(results.empty());
    CHECK(diagnostics.catalog.ready_manifests == 0);
    CHECK(ledger.snapshot().live_ops == baseline_ops);

    // Two stores may share the global ledger, but a host owner remains bound
    // to the catalog that minted it. Exercise the store gate itself rather
    // than only its catalog predicate.
    budget_context.available = true;
    llama_vbr_artifact_catalog foreign_catalog(ledger);
    std::vector<llama_vbr_artifact_domain_binding> foreign_bindings;
    CHECK(foreign_catalog.bind_topologies(
        { topology }, foreign_bindings));
    llama_cache_budget_config foreign_budget;
    CHECK(sample_projected_store_budget(
        &budget_context, foreign_budget));
    std::vector<vbr_projected_manifest_publish_result> foreign_results;
    CHECK(foreign_catalog.publish_projected_batch(
        ready_assembly, make_publications(ready_assembly), foreign_budget,
        foreign_results));
    std::vector<llama_cache_acct_artifact_id> foreign_references;
    for (const auto & row : foreign_results) {
        if (row.status == vbr_projected_manifest_publish_status::published ||
            row.status == vbr_projected_manifest_publish_status::adopted) {
            foreign_references.push_back(
                row.publication.reference_artifact);
        }
    }
    std::vector<vbr_artifact_package_view> foreign_packages;
    CHECK(foreign_catalog.claim_fresh_host_batch(
        foreign_references, foreign_packages));
    CHECK(foreign_packages.size() == 2);
    if (!foreign_packages.empty()) {
        auto foreign_payload =
            server_prompt_cache_vbr_payload::adopt(
                std::move(foreign_packages.front()));
        CHECK(foreign_payload);
        if (foreign_payload) {
            const auto authenticated =
                store->counters().host_imports_authenticated;
            server_vbr_artifact_import_target foreign_import;
            const auto rejected = store->import_host_payload(
                std::move(foreign_import), foreign_payload);
            CHECK(rejected.status ==
                  server_vbr_artifact_import_status::not_found);
            CHECK(store->counters().host_imports_authenticated ==
                  authenticated);
        }
    }
    foreign_packages.clear();
    CHECK(ledger.snapshot().live_ops == baseline_ops);
}

static void test_capture_reservation_domain_preparation() {
    llama_cache_acct_ledger ledger;
    const auto topology = capture_test_topology();
    llama_cache_acct_resource_domain device;
    CHECK(ledger.make_device_domain(
        topology, llama_cache_acct_device_ordinal { 0 },
        device));
    const llama_cache_acct_completeness_requirement requirement {
        device, llama_cache_acct_producer::live_memory,
    };
    CHECK(ledger.configure_required_producers(&requirement, 1));
    for (const auto category : {
            llama_cache_acct_category::live_attention_state,
            llama_cache_acct_category::live_recurrent_state,
            llama_cache_acct_category::recurrent_rollback_planes,
            llama_cache_acct_category::rolling_window_tape }) {
        ledger.gauge_set(
            category, device,
            llama_cache_acct_measure::resident_allocated, 0);
    }
    CHECK(ledger.certify_complete(
        device, llama_cache_acct_producer::live_memory));
    // A partially activated capture row is not dormant: known resident with
    // unknown reserved evidence makes the coordinator fail closed.
    ledger.gauge_set(
        llama_cache_acct_category::unit_version_payload,
        device, llama_cache_acct_measure::logical_payload, 0);
    ledger.gauge_set(
        llama_cache_acct_category::unit_version_payload,
        device, llama_cache_acct_measure::resident_allocated, 0);

    llama_cache_budget_config budget;
    llama_cache_budget_device_input input;
    input.backend_device =
        reinterpret_cast<const void *>(uintptr_t(1));
    input.domain = device;
    input.physical_total = 1024;
    input.physical_free = 1024;
    input.phys_state =
        llama_cache_budget_capacity_state::known;
    input.current_compute_allocated = 0;
    input.configured_compute_reserve = 0;
    input.compute_state =
        llama_cache_budget_capacity_state::known;
    input.cache_cap_state =
        llama_cache_budget_capacity_state::unbounded;
    budget.devices.push_back(input);

    llama_cache_transaction_leaf leaf;
    leaf.category =
        llama_cache_acct_category::unit_version_payload;
    leaf.domain = device;
    leaf.expected_logical = 16;
    leaf.reserve_resident = 16;
    leaf.stage_resident = 16;
    llama_cache_acct_op_id committed;
    leaf.committed_op = &committed;
    std::vector<llama_cache_transaction_leaf> leaves { leaf };
    {
        auto unavailable =
            llama_cache_prepare_reservation_transaction(
                ledger, budget, leaves);
        CHECK(!unavailable.ready());
        CHECK(unavailable.preparation().status ==
              llama_cache_prepare_status::admission_refused);
    }
    CHECK(server_vbr_artifact_store_observe_empty_accounting(
        ledger, device));
    CHECK(server_vbr_artifact_store_verify_accounting(
        ledger, { device }));
    {
        auto prepared =
            llama_cache_prepare_reservation_transaction(
                ledger, budget, leaves);
        CHECK(prepared.ready());
    }
    CHECK(ledger.snapshot().live_ops == 0);
}

static void test_server_store_construction_and_lifetime() {
    llama_cache_acct_ledger ledger;
    const auto pinned =
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pinned_host);
    const llama_cache_acct_completeness_requirement requirement {
        pinned, llama_cache_acct_producer::retention_sidecar,
    };
    CHECK(ledger.configure_required_producers(&requirement, 1));
    // Reproduce the real-load failure: observing only the ring leaves the
    // host-scoped payload cells in this manifested pinned domain unknown.
    for (const auto measure : {
            llama_cache_acct_measure::logical_payload,
            llama_cache_acct_measure::resident_allocated,
            llama_cache_acct_measure::reserved }) {
        ledger.gauge_set(
            llama_cache_acct_category::pinned_preimage_ring,
            pinned, measure, 0);
    }
    CHECK(ledger.certify_complete(
        pinned, llama_cache_acct_producer::retention_sidecar));

    llama_cache_budget_config budget;
    budget.host.pinned_cap = 1024;
    budget.host.pinned_state =
        llama_cache_budget_capacity_state::known;
    {
        llama_cache_budget_coordinator coordinator;
        const auto snapshot = ledger.snapshot();
        CHECK(coordinator.reset(snapshot, budget));
        llama_cache_budget_plan plan;
        plan.accounting_serial = snapshot.serial;
        plan.entries.push_back({ pinned, 16, 0 });
        CHECK(coordinator.fits(plan).state ==
              llama_cache_budget_fit_state::unavailable);
    }
    CHECK(server_vbr_artifact_store_configure_pinned_accounting(
        ledger, pinned));
    CHECK(!server_vbr_artifact_store_configure_pinned_accounting(
        ledger,
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pageable_host)));

    server_vbr_artifact_store_config config;
    config.ledger = &ledger;
    config.pinned_domain = pinned;
    config.topologies.push_back(capture_test_topology());
    config.pool_bindings.push_back({
        { 0x1111, 0x2222 }, 0, 0, 0, 0,
    });
    config.lanes.push_back({});
    config.attention_children = 1;
    // The physical owner rounds down to complete chunks; both direction
    // adapters must use that constructed capacity rather than the request.
    config.ring_bytes = 17;
    config.chunk_bytes = 8;
    config.sample_budget = sample_unbounded_host_budget;

    const auto baseline = ledger.snapshot();
    CHECK(sample_unbounded_host_budget(nullptr, budget));
    {
        llama_cache_budget_coordinator coordinator;
        const auto snapshot = ledger.snapshot();
        CHECK(coordinator.reset(snapshot, budget));
        llama_cache_budget_plan plan;
        plan.accounting_serial = snapshot.serial;
        plan.entries.push_back({ pinned, 16, 0 });
        CHECK(coordinator.fits(plan).state ==
              llama_cache_budget_fit_state::fits);
    }
    vbr_capture_ring_accounting direct_accounting {
        &ledger, pinned, &budget,
    };
    vbr_capture_stream_status direct_status;
    {
        auto direct = vbr_pinned_chunk_ring::create(
            config.lanes, config.ring_bytes, config.chunk_bytes,
            direct_status, &direct_accounting);
        CHECK(direct);
    }
    CHECK(ring_resident(ledger, pinned) == 0);
    server_vbr_artifact_capture_status status;
    server_vbr_artifact_store_create_diagnostics diagnostics;
    const uint64_t transport_before =
        vbr_pinned_ring_live_capacity_bytes();
    {
        auto store =
            server_vbr_artifact_store::create(
                config, status, &diagnostics);
        CHECK(store);
        CHECK(status == server_vbr_artifact_capture_status::ok);
        CHECK(diagnostics.failure ==
              server_vbr_artifact_store_create_failure::none);
        CHECK(diagnostics.ring_status ==
              vbr_capture_stream_status::ok);
        CHECK(diagnostics.ring_failure ==
              vbr_capture_ring_create_failure::none);
        CHECK(diagnostics.requested_ring_bytes == 17);
        CHECK(diagnostics.constructed_ring_bytes == 16);
        if (!store) {
            return;
        }
        CHECK(store->attention_children() == 1);
        CHECK(store->counters().pinned_bytes == 16);
        CHECK(store->counters().requested == 0);
        CHECK(ring_resident(ledger, pinned) == 16);
        // Capture and import adapters share this one physical allocation.
        CHECK(vbr_pinned_ring_live_capacity_bytes() ==
              transport_before + 16);
        vbr_adopt_stage_policy import_policy;
        CHECK(server_vbr_artifact_store_test_door::
              import_transport_policy(*store, import_policy));
        CHECK(import_policy.ledger == &ledger);
        CHECK(import_policy.persistent_ring);
        CHECK(import_policy.pinned_domain == pinned);
        CHECK(import_policy.pinned_ring_bytes == 16);
        CHECK(import_policy.chunk_bytes == 8);
        CHECK(import_policy.lanes.size() == 1);
        CHECK(vbr_pinned_ring_live_capacity_bytes() ==
              transport_before + 16);
        import_policy.persistent_ring.reset();
    }
    const auto after = ledger.snapshot();
    CHECK(ring_resident(ledger, pinned) == 0);
    CHECK(vbr_pinned_ring_live_capacity_bytes() == transport_before);
    CHECK(after.live_ops == baseline.live_ops);
    CHECK(after.faults_invalid_transition ==
          baseline.faults_invalid_transition);
    CHECK(after.faults_overflow == baseline.faults_overflow);
    CHECK(after.faults_allocation == baseline.faults_allocation);

    config.attention_children = 0;
    auto rejected =
        server_vbr_artifact_store::create(
            config, status, &diagnostics);
    CHECK(!rejected);
    CHECK(status ==
          server_vbr_artifact_capture_status::unavailable);
    CHECK(diagnostics.failure ==
          server_vbr_artifact_store_create_failure::
              attention_child_missing);
}

static void test_server_capture_status_vocabulary() {
    for (size_t i = 0;
         i < size_t(server_vbr_artifact_capture_status::_count);
         ++i) {
        const auto * name =
            server_vbr_artifact_capture_status_name(
                server_vbr_artifact_capture_status(i));
        CHECK(name != nullptr);
        CHECK(std::string(name) != "_count");
    }
    CHECK(std::string(server_vbr_artifact_capture_status_name(
              server_vbr_artifact_capture_status::_count)) ==
          "_count");
    for (size_t i = 0;
         i < size_t(server_vbr_artifact_import_status::_count);
         ++i) {
        const auto * name = server_vbr_artifact_import_status_name(
            server_vbr_artifact_import_status(i));
        CHECK(name != nullptr);
        CHECK(std::string(name) != "_count");
    }
    CHECK(std::string(server_vbr_artifact_import_status_name(
              server_vbr_artifact_import_status::_count)) ==
          "_count");
    for (size_t i = 0;
         i < size_t(vbr_explicit_capture_phase::_count);
         ++i) {
        const auto * name =
            vbr_explicit_capture_phase_name(
                vbr_explicit_capture_phase(i));
        CHECK(name != nullptr);
        CHECK(std::string(name) != "_count");
    }
    for (size_t i = 0;
         i < size_t(vbr_explicit_generation_failure::_count);
         ++i) {
        const auto * name =
            vbr_explicit_generation_failure_name(
            vbr_explicit_generation_failure(i));
        CHECK(name != nullptr);
        CHECK(std::string(name) != "_count");
    }
    for (size_t i = 0;
         i < size_t(vbr_explicit_size_failure::_count);
         ++i) {
        const auto * name =
            vbr_explicit_size_failure_name(
                vbr_explicit_size_failure(i));
        CHECK(name != nullptr);
        CHECK(std::string(name) != "_count");
    }
    for (size_t i = 0;
         i < size_t(vbr_capture_reservation_group::_count);
         ++i) {
        const auto * name =
            vbr_capture_reservation_group_name(
                vbr_capture_reservation_group(i));
        CHECK(name != nullptr);
        CHECK(std::string(name) != "invalid");
    }
}

static void test_server_reference_tenant_authorization() {
    server_vbr_artifact_reference_index index;
    const llama_cache_acct_artifact_id expected { 73 };
    CHECK(index.publish("vbrref_alpha", "tenant-a", expected));
    CHECK(!index.publish("vbrref_alpha", "tenant-a", expected));
    CHECK(!index.publish("malformed", "tenant-a", expected));
    CHECK(!index.publish("vbrref_zero", "tenant-a", {}));

    llama_cache_acct_artifact_id resolved { 999 };
    CHECK(index.authorize("vbrref_alpha", "tenant-a", resolved));
    CHECK(resolved == expected);
    for (const auto & denied : std::vector<std::pair<std::string, std::string>> {
            { "vbrref_alpha", "tenant-b" },
            { "vbrref_missing", "tenant-a" },
            { "malformed", "tenant-a" },
            { "vbrref_alpha", "" },
        }) {
        resolved = { 999 };
        CHECK(!index.authorize(denied.first, denied.second, resolved));
        // Wrong-tenant, nonexistent and malformed tokens expose the same
        // closed miss shape and never return the underlying artifact id.
        CHECK(resolved.v == 0);
    }
}

static void test_server_import_route_classification() {
    using status = server_vbr_artifact_import_status;
    server_vbr_artifact_import_output untouched;
    CHECK(untouched.downward_reserve_status ==
          vbr_downward_reserve_status::not_attempted);
    CHECK(server_vbr_artifact_import_route_precheck(
              false, false, false, false, false) == status::unsupported);
    CHECK(server_vbr_artifact_import_route_precheck(
              true, false, false, true, true) == status::invalid_slot);
    CHECK(server_vbr_artifact_import_route_precheck(
              true, true, true, true, true) == status::slot_processing);
    CHECK(server_vbr_artifact_import_route_precheck(
              true, true, false, false, true) == status::unavailable);
    CHECK(server_vbr_artifact_import_route_precheck(
              true, true, false, true, false) == status::slot_not_empty);
    CHECK(server_vbr_artifact_import_route_precheck(
              true, true, false, true, true) == status::ok);

    for (const auto decision : {
            vbr_import_decision::native_import,
            vbr_import_decision::live_rebased,
            vbr_import_decision::downward_rebase }) {
        CHECK(server_vbr_artifact_import_validation_disposition(
                  vbr_manifest_validation_status::validated,
                  decision) == status::ok);
    }
    for (const auto decision : {
            vbr_import_decision::rebuild,
            vbr_import_decision::cold }) {
        CHECK(server_vbr_artifact_import_validation_disposition(
                  vbr_manifest_validation_status::validated,
                  decision) == status::report_only);
    }
    CHECK(server_vbr_artifact_import_validation_disposition(
              vbr_manifest_validation_status::validated,
              vbr_import_decision::reject) == status::validation_failed);
    CHECK(server_vbr_artifact_import_validation_disposition(
              vbr_manifest_validation_status::unavailable,
              vbr_import_decision::native_import) ==
          status::validation_failed);
}

static void test_fresh_f16_size_generation() {
    // Production ordinary decode maps a padded nonzero watermark even before
    // any retier has created the VBR side stream. That never-degraded state is
    // a complete full-domain F16 extent with zero promote hops.
    vbr_unit_generation fresh;
    fresh.current_type = GGML_TYPE_F16;
    fresh.last_source_type = GGML_TYPE_F16;
    fresh.domain = vbr_repr_domain::full;
    fresh.promote_hops = 0;
    CHECK(vbr_explicit_capture_validate_extent_generation(
              256, GGML_TYPE_F16, 0, fresh) ==
          vbr_explicit_size_failure::none);
    CHECK(vbr_explicit_capture_validate_extent_generation(
              0, GGML_TYPE_F16, 0, fresh) ==
          vbr_explicit_size_failure::wm_cells_zero);

    auto wrong_domain = fresh;
    wrong_domain.domain = vbr_repr_domain::tapped;
    CHECK(vbr_explicit_capture_validate_extent_generation(
              256, GGML_TYPE_F16, 0, wrong_domain) ==
          vbr_explicit_size_failure::domain_mismatch);

    vbr_unit_generation tapped = fresh;
    tapped.current_type = GGML_TYPE_TURBO4_0;
    tapped.last_source_type = GGML_TYPE_F16;
    tapped.domain = vbr_repr_domain::tapped;
    tapped.promote_hops = 1;
    CHECK(vbr_explicit_capture_validate_extent_generation(
              256, GGML_TYPE_TURBO4_0, 1, tapped) ==
          vbr_explicit_size_failure::none);
    CHECK(vbr_explicit_capture_validate_extent_generation(
              256, GGML_TYPE_TURBO4_0, 0, tapped) ==
          vbr_explicit_size_failure::promote_hops_mismatch);
    CHECK(vbr_explicit_capture_validate_extent_generation(
              256, GGML_TYPE_TURBO3_TCQ, 1, tapped) ==
          vbr_explicit_size_failure::extent_type_mismatch);
}

static void test_library_representation_identity() {
    static constexpr char BUILD_A[] = "capture-build-a";
    static constexpr char BUILD_B[] = "capture-build-b";
    const vbr_explicit_representation_policy policy_a {
        BUILD_A, sizeof(BUILD_A) - 1,
    };
    const vbr_explicit_representation_policy policy_b {
        BUILD_B, sizeof(BUILD_B) - 1,
    };
    vbr_explicit_representation_identity a;
    vbr_explicit_representation_identity b;
    CHECK(vbr_explicit_capture_representation_identity(
        &policy_a, GGML_TYPE_F16, false, a));
    CHECK(vbr_explicit_capture_representation_identity(
        &policy_b, GGML_TYPE_F16, false, b));
    CHECK(a.codec_id == uint32_t(GGML_TYPE_F16) + 1);
    CHECK(a.codec_version == 1);
    CHECK(a.codebook_digest != b.codebook_digest);
    CHECK(a.rotation_digest == b.rotation_digest);
    CHECK(a.meansub_digest == b.meansub_digest);
    vbr_explicit_representation_identity missing;
    CHECK(!vbr_explicit_capture_representation_identity(
        nullptr, GGML_TYPE_F16, false, missing));
}

static void test_cuda_ring() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * candidate = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(candidate) ==
                GGML_BACKEND_DEVICE_TYPE_GPU) {
            device = candidate;
            break;
        }
    }
    if (!device) {
        fprintf(stderr, "FAIL: no GPU backend for F3.1 CUDA synthetic gate\n");
        failures++;
        return;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    CHECK(backend != nullptr);
    if (!backend) {
        return;
    }
    const size_t n = 5*1024*1024 + 3;
    std::vector<uint8_t> expected(n);
    for (size_t i = 0; i < n; ++i) {
        expected[i] = uint8_t((i*29 + 11) & 0xff);
    }
    ggml_init_params params = {
        2*ggml_tensor_overhead(), nullptr, true,
    };
    ggml_context * context = ggml_init(params);
    CHECK(context != nullptr);
    ggml_tensor * tensor =
        context ? ggml_new_tensor_1d(
            context, GGML_TYPE_I8, n) : nullptr;
    ggml_backend_buffer_t buffer =
        tensor ? ggml_backend_alloc_ctx_tensors(
            context, backend) : nullptr;
    CHECK(tensor && buffer);
    if (tensor && buffer) {
        ggml_backend_tensor_set(
            tensor, expected.data(), 0, expected.size());
        vbr_capture_stream_status status;
        auto ring = vbr_pinned_chunk_ring::create(
            { { device, backend } },
            2*1024*1024, 1024*1024, status);
        CHECK(ring);
        artifact_segment_chain chain;
        vbr_capture_stream_stats stats;
        vbr_capture_stream_source source;
        source.lane = 0;
        source.size = expected.size();
        source.backend = backend;
        source.device = device;
        source.tensor = tensor;
        CHECK(ring->stream(source, chain, stats) ==
              vbr_capture_stream_status::ok);
        CHECK(chain.max_segment_size() <= 1024*1024);
        CHECK(read_chain(chain) == expected);
        CHECK(stats.event_completions > 0);

        auto sync_ring = vbr_pinned_chunk_ring::create(
            { { device, backend, true } },
            2*1024*1024, 1024*1024, status);
        CHECK(sync_ring);
        artifact_segment_chain sync_chain;
        vbr_capture_stream_stats sync_stats;
        CHECK(sync_ring->stream(source, sync_chain, sync_stats) ==
              vbr_capture_stream_status::ok);
        CHECK(sync_stats.synchronous_fallbacks > 0);
        CHECK(sync_stats.streaming_digest ==
              stats.streaming_digest);
        CHECK(read_chain(sync_chain) == expected);
    }
    if (buffer) {
        ggml_backend_buffer_free(buffer);
    }
    if (context) {
        ggml_free(context);
    }
    ggml_backend_free(backend);
}

static void benchmark_fragmented_range_packing() {
    static constexpr uint32_t RANGE_COUNT = 1048576;
    static constexpr size_t CHUNK_BYTES = 64*1024;
    generated_h2d_source generated;
    generated.size = uint64_t(RANGE_COUNT)*2;
    std::vector<vbr_capture_stream_range> ranges;
    ranges.reserve(RANGE_COUNT);
    for (uint32_t i = 0; i < RANGE_COUNT; ++i) {
        ranges.push_back({ uint64_t(i)*2, 1 });
    }
    vbr_capture_stream_source source;
    source.size = generated.size;
    source.context = &generated;
    source.read = generated_h2d_source::read;
    vbr_capture_stream_status status;
    auto ring = vbr_pinned_chunk_ring::create(
        { {} }, 2*CHUNK_BYTES, CHUNK_BYTES, status);
    CHECK(ring);
    artifact_segment_chain chain(
        VBR_CAPTURE_RANGE_CHUNK_BYTES, 262144);
    vbr_capture_stream_stats stats;
    const auto begin = std::chrono::steady_clock::now();
    CHECK(ring && ring->stream_ranges(
              source, ranges, chain, stats) ==
          vbr_capture_stream_status::ok);
    vbr_capture_range_tree tree;
    CHECK(vbr_capture_range_seal(
        chain, uint64_t(32)*1024*1024, tree));
    vbr_capture_range_proof proof;
    vbr_capture_range_proof_limits proof_limits;
    CHECK(vbr_capture_range_prove(
        tree, { { 0, RANGE_COUNT } }, proof_limits, proof));
    uint64_t proof_bytes_read = 0;
    CHECK(vbr_capture_range_verify(
        proof, chain.source(), &proof_bytes_read));
    const auto elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin).count();
    CHECK(stats.bytes == RANGE_COUNT);
    CHECK(stats.chunks == RANGE_COUNT/CHUNK_BYTES);
    CHECK(chain.segment_count() == stats.chunks);
    CHECK(chain.max_segment_size() == CHUNK_BYTES);
    CHECK(tree.chunk_count() == stats.chunks);
    CHECK(proof.selected_chunk_count() == stats.chunks);
    CHECK(proof_bytes_read == RANGE_COUNT);
    std::vector<uint8_t> packed(RANGE_COUNT);
    CHECK(chain.read(0, packed.data(), packed.size()));
    for (uint32_t i = 0; i < RANGE_COUNT; ++i) {
        if (packed[i] != generated_h2d_source::byte_at(uint64_t(i)*2)) {
            CHECK(false);
            break;
        }
    }
    printf("VBR_CAPTURE_RANGE_PACK_BENCH ranges=%u chunks=%" PRIu64
           " bytes=%" PRIu64 " root_chunks=%u proof_nodes=%u"
           " proof_bytes=%" PRIu64 " elapsed_us=%lld\n",
           RANGE_COUNT, stats.chunks, stats.bytes, tree.chunk_count(),
           proof.proof_node_count(), proof_bytes_read,
           (long long) elapsed);
}

int main(int argc, char ** argv) {
    if (argc == 2 &&
        std::string(argv[1]) == "--range-pack-bench") {
        benchmark_fragmented_range_packing();
        return failures == 0 ? 0 : 1;
    }
    test_segment_chain_offsets();
    test_authenticated_range_tree();
    test_registry_quiescence_query();
    test_cpu_ring_boundaries();
    test_projected_unit_transfer();
    test_preflight_unavailable_projection_rows();
    test_manifest_coherent_assembly();
    test_dependency_scoped_projected_catalog_publication();
    test_h2d_bounded_streaming();
    test_ring_accounting_once();
    test_capture_reservation_domain_preparation();
    test_server_store_construction_and_lifetime();
    test_projected_host_batch_store_adapter();
    test_server_capture_status_vocabulary();
    test_server_reference_tenant_authorization();
    test_server_import_route_classification();
    test_fresh_f16_size_generation();
    test_library_representation_identity();
    if (argc == 2 && std::string(argv[1]) == "--cuda") {
        test_cuda_ring();
    }
    if (failures != 0) {
        fprintf(stderr, "%d F3.1 capture test(s) failed\n", failures);
        return 1;
    }
    printf("VBR artifact capture: PASS\n");
    return 0;
}
