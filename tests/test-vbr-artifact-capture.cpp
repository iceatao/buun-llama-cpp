#include "llama-vbr-artifact-capture.h"
#include "llama-vbr-artifact-stage.h"
#include "llama-vbr-operation.h"
#include "server-vbr-artifact-store.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
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
    CHECK(captured.projection == projection);
    CHECK(captured.shards.size() == 2);
    CHECK(captured.packed_bytes == 20);
    CHECK(captured.transfer.bytes == 20);
    CHECK(std::any_of(
        captured.transfer.streaming_digest.begin(),
        captured.transfer.streaming_digest.end(),
        [](uint8_t value) { return value != 0; }));
    auto rebound_sources = sources;
    rebound_sources[0].source.context = &first;
    uint32_t rebound_count = 0;
    std::array<uint8_t, 32> rebound_digest = {};
    CHECK(vbr_capture_projected_shard_topology(
        rebound_sources, rebound_count, rebound_digest));
    CHECK(rebound_digest != captured.snapshot.shard_topology_digest);
    rebound_sources = sources;
    rebound_sources[0].source.tensor_offset = 1;
    CHECK(vbr_capture_projected_shard_topology(
        rebound_sources, rebound_count, rebound_digest));
    CHECK(rebound_digest != captured.snapshot.shard_topology_digest);
    if (captured.shards.size() == 2) {
        CHECK(captured.shards[0].shard_index == 0);
        CHECK(captured.shards[1].shard_index == 1);
        CHECK(read_chain(*captured.shards[0].bytes) ==
              projected_rows(first.bytes, 2, { 1, 2, 3, 5 }));
        CHECK(read_chain(*captured.shards[1].bytes) ==
              projected_rows(second.bytes, 3, { 1, 2, 3, 5 }));
        CHECK(captured.slices.size() == 4);
        if (captured.slices.size() == 4) {
            CHECK(captured.slices[0].packed_first_row == 0);
            CHECK(captured.slices[1].packed_first_row == 1);
            CHECK(captured.slices[2].packed_first_row == 2);
            CHECK(captured.slices[3].packed_first_row == 3);
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
    combined_snapshot.snapshot = captured.snapshot;
    vbr_capture_projected_unit combined_capture;
    CHECK(vbr_capture_projected_unit_transfer(
              combined_projection, 0, 0, 7, sources, {},
              combined_snapshot.provider(), *ring, combined_capture) ==
          vbr_capture_stream_status::ok);
    CHECK(combined_capture.packed_bytes == captured.packed_bytes);
    CHECK(combined_capture.transfer.streaming_digest !=
          captured.transfer.streaming_digest);

    snapshot = {};
    snapshot.snapshot = captured.snapshot;
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
    CHECK(!changed.projection);
    CHECK(changed.shards.empty());

    snapshot = {};
    snapshot.snapshot = captured.snapshot;
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
    CHECK(!failed.projection);
    CHECK(failed.shards.empty());

    snapshot = {};
    snapshot.snapshot = captured.snapshot;
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
    snapshot.snapshot = captured.snapshot;
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
    snapshot.snapshot = captured.snapshot;
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
    auto invalid_snapshot = captured.snapshot;
    invalid_snapshot.generation.last_source_type = -1;
    expect_invalid_snapshot(invalid_snapshot);
    invalid_snapshot = captured.snapshot;
    invalid_snapshot.generation.current_type = GGML_TYPE_COUNT;
    expect_invalid_snapshot(invalid_snapshot);
    invalid_snapshot = captured.snapshot;
    invalid_snapshot.generation.domain = vbr_repr_domain(255);
    expect_invalid_snapshot(invalid_snapshot);
    invalid_snapshot = captured.snapshot;
    invalid_snapshot.generation.last_transition = vbr_repr_transition(255);
    expect_invalid_snapshot(invalid_snapshot);
    invalid_snapshot = captured.snapshot;
    invalid_snapshot.generation.flags = 1;
    expect_invalid_snapshot(invalid_snapshot);
    snapshot = {};
    snapshot.snapshot = captured.snapshot;
    snapshot.snapshot.generation.publish_seq = 15;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::snapshot_unavailable);
    CHECK(snapshot.released == 1);

    // The snapshot authenticates the complete shard set. Reordering is
    // normalized, while omission, sparse/sentinel IDs, and substitution fail.
    snapshot = {};
    snapshot.snapshot = captured.snapshot;
    auto omitted = sources;
    omitted.erase(omitted.begin());
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, omitted, {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::snapshot_unavailable);
    auto substituted = sources;
    substituted[0].source_identity++;
    snapshot = {};
    snapshot.snapshot = captured.snapshot;
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
    snapshot.snapshot = captured.snapshot;
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

    snapshot = {};
    snapshot.snapshot = captured.snapshot;
    sources[0].row_count = 3;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources,
              {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::projection_invalid);
    CHECK(snapshot.acquired == 0);
    CHECK(snapshot.released == 0);
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
    config.ring_bytes = 16;
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
        CHECK(diagnostics.requested_ring_bytes == 16);
        CHECK(diagnostics.constructed_ring_bytes == 16);
        if (!store) {
            return;
        }
        CHECK(store->attention_children() == 1);
        CHECK(store->counters().pinned_bytes == 16);
        CHECK(store->counters().requested == 0);
        CHECK(ring_resident(ledger, pinned) == 16);
    }
    const auto after = ledger.snapshot();
    CHECK(ring_resident(ledger, pinned) == 0);
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
    artifact_segment_chain chain;
    vbr_capture_stream_stats stats;
    const auto begin = std::chrono::steady_clock::now();
    CHECK(ring && ring->stream_ranges(
              source, ranges, chain, stats) ==
          vbr_capture_stream_status::ok);
    const auto elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin).count();
    CHECK(stats.bytes == RANGE_COUNT);
    CHECK(stats.chunks == RANGE_COUNT/CHUNK_BYTES);
    CHECK(chain.segment_count() == stats.chunks);
    CHECK(chain.max_segment_size() == CHUNK_BYTES);
    std::vector<uint8_t> packed(RANGE_COUNT);
    CHECK(chain.read(0, packed.data(), packed.size()));
    for (uint32_t i = 0; i < RANGE_COUNT; ++i) {
        if (packed[i] != generated_h2d_source::byte_at(uint64_t(i)*2)) {
            CHECK(false);
            break;
        }
    }
    printf("VBR_CAPTURE_RANGE_PACK_BENCH ranges=%u chunks=%" PRIu64
           " bytes=%" PRIu64 " elapsed_us=%lld\n",
           RANGE_COUNT, stats.chunks, stats.bytes, (long long) elapsed);
}

int main(int argc, char ** argv) {
    if (argc == 2 &&
        std::string(argv[1]) == "--range-pack-bench") {
        benchmark_fragmented_range_packing();
        return failures == 0 ? 0 : 1;
    }
    test_segment_chain_offsets();
    test_registry_quiescence_query();
    test_cpu_ring_boundaries();
    test_projected_unit_transfer();
    test_h2d_bounded_streaming();
    test_ring_accounting_once();
    test_capture_reservation_domain_preparation();
    test_server_store_construction_and_lifetime();
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
