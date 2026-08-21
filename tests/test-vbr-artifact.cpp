#include "llama-vbr-artifact.h"
#include "llama-vbr-artifact-catalog.h"
#include "server-cache-lease.h"
#include "server-cache-vbr-proof.h"
#include "server-prompt-cache-payload.h"
#include "llama-vbr-artifact-stage.h"
#include "llama-vbr-artifact-validate.h"
#include "llama-vbr-downward.h"
#include "llama-vbr-identity-digest.h"
#include "llama-sha256.h"

#ifdef VBR_PROMPT_CACHE_PUBLICATION_TEST
#include "server-task.h"
#include "server-retention-sidecar.h"
#endif

#include <algorithm>
#include <atomic>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

static int failures = 0;

static_assert(!std::is_copy_constructible<vbr_artifact_package_view>::value);
static_assert(!std::is_copy_assignable<vbr_artifact_package_view>::value);
static_assert(std::is_nothrow_move_constructible<vbr_artifact_package_view>::value);
static_assert(std::is_nothrow_move_assignable<vbr_artifact_package_view>::value);
static_assert(!std::is_default_constructible<vbr_validated_manifest>::value);
static_assert(!std::is_copy_constructible<vbr_validated_manifest>::value);
static_assert(!std::is_copy_assignable<vbr_validated_manifest>::value);
static_assert(std::is_nothrow_move_constructible<vbr_validated_manifest>::value);
static_assert(std::is_nothrow_move_assignable<vbr_validated_manifest>::value);
static_assert(!std::is_copy_constructible<vbr_staged_payloads>::value);
static_assert(!std::is_copy_assignable<vbr_staged_payloads>::value);
static_assert(std::is_nothrow_move_constructible<vbr_staged_payloads>::value);
static_assert(std::is_nothrow_move_assignable<vbr_staged_payloads>::value);

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

static std::string hex(const std::array<uint8_t, 32> & bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size()*2);
    for (uint8_t byte : bytes) {
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 0x0f]);
    }
    return out;
}

static std::array<uint8_t, 32> digest_of(const std::vector<uint8_t> & bytes) {
    llama_sha256 hash;
    hash.update(bytes.data(), bytes.size());
    return hash.finish();
}

struct memory_source {
    std::vector<uint8_t> bytes;

    static bool read(
            const void * context,
            uint64_t offset,
            uint8_t * destination,
            size_t size) noexcept {
        const auto * self = static_cast<const memory_source *>(context);
        if (offset > self->bytes.size() ||
            size > self->bytes.size() - size_t(offset)) {
            return false;
        }
        memcpy(destination, self->bytes.data() + offset, size);
        return true;
    }

    vbr_artifact_byte_source source() const {
        return { bytes.size(), this, read };
    }
};

struct generated_source {
    uint64_t size = 0;
    uint8_t salt = 0;
    uint64_t calls = 0;
    size_t max_request = 0;

    static bool read(
            const void * context,
            uint64_t offset,
            uint8_t * destination,
            size_t size) noexcept {
        auto * self = const_cast<generated_source *>(
            static_cast<const generated_source *>(context));
        self->calls++;
        self->max_request = std::max(self->max_request, size);
        if (offset > self->size || size > self->size - offset) {
            return false;
        }
        for (size_t i = 0; i < size; ++i) {
            destination[i] =
                uint8_t(self->salt + uint8_t((offset + i)*131u));
        }
        return true;
    }

    vbr_artifact_byte_source source() const {
        return { size, this, read };
    }
};

struct phased_source {
    std::array<uint8_t, 4> first  = { 0x10, 0x11, 0x12, 0x13 };
    std::array<uint8_t, 4> second = { 0xa0, 0xa1, 0xa2, 0xa3 };
    uint32_t calls = 0;
    uint32_t switch_after = 3;

    static bool read(
            const void * context,
            uint64_t offset,
            uint8_t * destination,
            size_t size) noexcept {
        auto * self = const_cast<phased_source *>(
            static_cast<const phased_source *>(context));
        self->calls++;
        const auto & selected =
            self->calls <= self->switch_after ?
                self->first : self->second;
        if (offset > selected.size() ||
            size > selected.size() - size_t(offset)) {
            return false;
        }
        memcpy(destination, selected.data() + offset, size);
        return true;
    }

    vbr_artifact_byte_source source() const {
        return { first.size(), this, read };
    }
};

struct memory_reader {
    const std::vector<uint8_t> * bytes = nullptr;
    size_t position = 0;

    static bool read(
            void * context,
            uint8_t * destination,
            size_t size) noexcept {
        auto * self = static_cast<memory_reader *>(context);
        if (size > self->bytes->size() - self->position) {
            return false;
        }
        memcpy(destination, self->bytes->data() + self->position, size);
        self->position += size;
        return true;
    }
};

struct staged_consumer {
    uint64_t bytes = 0;
    uint32_t finishes = 0;
    bool verified = false;

    static bool consume(
            void * context,
            vbr_artifact_section_kind,
            uint32_t,
            uint32_t,
            bool,
            uint64_t,
            uint64_t,
            const uint8_t *,
            size_t size) noexcept {
        static_cast<staged_consumer *>(context)->bytes += size;
        return true;
    }

    static void finish(void * context, bool success) noexcept {
        auto * self = static_cast<staged_consumer *>(context);
        self->finishes++;
        self->verified = success;
        if (!success) {
            self->bytes = 0;
        }
    }
};

static std::array<uint8_t, 32> marker(uint8_t value) {
    std::array<uint8_t, 32> result;
    result.fill(value);
    return result;
}

struct fixture_storage {
    memory_source payload0 { { 0x10, 0x11, 0x12, 0x13 } };
    memory_source payload1 { { 0x20, 0x21, 0x22, 0x23 } };
    memory_source stash0   { { 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37 } };
    memory_source stash1   { { 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47 } };
    memory_source recurrent { { 0x90, 0x91, 0x92 } };
};

static vbr_artifact_portable_topology make_topology() {
    llama_cache_acct_shard_topology source;
    const std::vector<std::string> devices = {
        "fixture-device-a",
        "fixture-device-b",
    };
    const float weights[] = { 0.6f, 0.4f };
    CHECK(llama_cache_acct_build_shard_topology(
        devices, LLAMA_SPLIT_MODE_TENSOR, 0, weights, source));

    return source;
}

static vbr_artifact_shard_descriptor make_shard(
        uint32_t index,
        const vbr_artifact_byte_source & source) {
    vbr_artifact_shard_descriptor shard;
    shard.shard_index = index;
    shard.topology_index = 0;
    shard.device_ordinal = uint16_t(index);
    shard.logical_offset = index;
    shard.row_count = 1;
    shard.column_count = source.size;
    shard.row_bytes = source.size;
    shard.payload_bytes = source.size;
    shard.payload = source;
    return shard;
}

static vbr_generation_page_ref make_page(uint64_t mask) {
    vbr_generation_page_ref page;
    page.page_index = 0;
    page.captured_page_gen = 7;
    page.covered_mask[0] = mask;
    return page;
}

static vbr_artifact_package make_package(fixture_storage & storage) {
    vbr_artifact_package package;
    package.topologies.push_back(make_topology());

    vbr_artifact_unit_blob blob;
    auto & descriptor = blob.descriptor;
    descriptor.child_id = 0;
    descriptor.logical_unit_id = 0;
    // This UUID is the validated SOURCE controller identity. The golden therefore proves the
    // buun OQ5 native-lineage branch is representable without rebasing it to a target UUID.
    descriptor.lineage_uuid = { 0x0102030405060708ull, 0x1112131415161718ull };
    descriptor.repr_gen = 17;
    descriptor.current_type = GGML_TYPE_TURBO4_0;
    descriptor.last_source_type = GGML_TYPE_TURBO8_0;
    descriptor.promote_hops = 1;
    descriptor.last_transition = vbr_repr_transition::degrade_other;
    descriptor.representation.kind =
        vbr_artifact_representation_kind::approximate;
    descriptor.representation.codec_id = 0x5438;
    descriptor.representation.codec_version = 1;
    descriptor.representation.reference_digest = marker(0x51);
    descriptor.representation.source_loss_history = 1;
    descriptor.representation.checkpoint_codec_hops = 0;
    descriptor.side = vbr_artifact_side::key;
    descriptor.n_stream = 1;
    descriptor.unified = true;
    descriptor.wm_cells = 1;
    descriptor.rank = 2;
    descriptor.dimensions = { 1, 4, 0, 0 };
    descriptor.row_alignment = 4;
    descriptor.row_codec_version = 1;
    descriptor.codebook_digest = marker(0x61);
    descriptor.rotation_digest = marker(0x62);
    descriptor.meansub_digest = marker(0x63);
    descriptor.shards = {
        make_shard(0, storage.payload0.source()),
        make_shard(1, storage.payload1.source()),
    };
    descriptor.clean_stash_state =
        vbr_artifact_clean_stash_state::present;
    descriptor.clean_stash.valid_rows = 1;
    descriptor.clean_stash.domain = vbr_repr_domain::tapped;
    descriptor.clean_stash.row_count = 1;
    descriptor.clean_stash.column_count = 4;
    descriptor.clean_stash.row_bytes = 8;
    descriptor.clean_stash.shards = {
        make_shard(0, storage.stash0.source()),
        make_shard(1, storage.stash1.source()),
    };
    package.unit_blobs.push_back(blob);

    auto & manifest = package.manifest;
    manifest.identity_policy_order_digest = marker(0x71);
    manifest.identity.execution_identity = "exec:qwen";
    manifest.identity.adapter_config_identity = "base:no-lora";
    manifest.identity.media_content_identity = "text-only";
    manifest.identity.sequence_epoch = 3;
    manifest.identity.token_count = 2;
    manifest.identity.next_position = 2;
    manifest.token_block.tokens = { 101, 102 };
    manifest.generation.version = 1;
    manifest.generation.status =
        vbr_checkpoint_generation_status::complete;
    manifest.generation.identity_policy_order_digest =
        manifest.identity_policy_order_digest;

    vbr_checkpoint_generation_controller controller;
    controller.child_id = 0;
    controller.dependency_mode =
        checkpoint_child_dependency_mode::live_guarded;
    controller.lineage_uuid = descriptor.lineage_uuid;
    controller.global_generation = 5;
    controller.units.push_back({
        descriptor.repr_gen,
        descriptor.current_type,
        descriptor.last_source_type,
        vbr_repr_domain::tapped,
        descriptor.promote_hops,
        descriptor.last_transition,
    });
    vbr_checkpoint_generation_stream stream;
    stream.stream_index = 0;
    stream.dependency_seq_id = 0;
    stream.computation_frontier = 2;
    stream.captured_dependency_count = 2;
    stream.pages.push_back(make_page(0x3));
    controller.streams.push_back(stream);
    manifest.generation.controllers.push_back(controller);
    manifest.consistency.kind =
        vbr_artifact_consistency_kind::capture_exact;

    vbr_artifact_controller_policy policy;
    policy.child_id = 0;
    policy.dependency_mode = controller.dependency_mode;
    policy.degrade_order_digest = marker(0x72);
    policy.policy_digest = marker(0x73);
    policy.cursor = 1;
    policy.floor_type = 4;
    policy.pressure_independent_settings = 0x55;
    policy.n_stream = 1;
    policy.unified = true;
    policy.wm_cells = 1;
    policy.current_type_vector_digest = marker(0x74);
    policy.completed_wave = true;
    manifest.controller_policy.push_back(policy);

    vbr_artifact_stream_placement placement;
    placement.child_id = 0;
    placement.stream_index = 0;
    placement.source_sequence = 0;
    placement.computation_frontier = 2;
    placement.cells = {
        { 0, 0, 10, 20 },
        { 1, 1, 11, 21 },
    };
    manifest.stream_placements.push_back(placement);

    vbr_artifact_unit_reference reference;
    reference.lineage_uuid = descriptor.lineage_uuid;
    reference.logical_unit_id = descriptor.logical_unit_id;
    reference.repr_gen = descriptor.repr_gen;
    reference.authorized_stream_refs = { 0 };
    reference.has_stash_reference = true;
    reference.stash_reference.valid_rows = 1;
    reference.stash_reference.domain = vbr_repr_domain::tapped;
    reference.stash_reference.row_count = 1;
    reference.stash_reference.column_count = 4;
    reference.stash_reference.row_bytes = 8;
    reference.stash_reference.captured_sink_count = 2;
    reference.stash_reference.covered_sink_pages = { make_page(0x3) };
    manifest.unit_references.push_back(reference);

    const vbr_artifact_portable_domain device0 {
        llama_cache_acct_residency::device,
        llama_cache_acct_domain_kind::device_topology,
        0,
        0,
    };
    const vbr_artifact_portable_domain device1 {
        llama_cache_acct_residency::device,
        llama_cache_acct_domain_kind::device_topology,
        0,
        1,
    };
    const vbr_artifact_portable_domain host {
        llama_cache_acct_residency::pageable_host,
        llama_cache_acct_domain_kind::not_applicable,
        UINT32_MAX,
        UINT16_MAX,
    };
    manifest.accounting = {
        {
            vbr_artifact_accounting_role::unit_payload,
            device0, 4, 4, llama_cache_acct_attr_kind::artifact,
        },
        {
            vbr_artifact_accounting_role::unit_payload,
            device1, 4, 4, llama_cache_acct_attr_kind::artifact,
        },
        {
            vbr_artifact_accounting_role::clean_stash_payload,
            device0, 8, 8, llama_cache_acct_attr_kind::artifact,
        },
        {
            vbr_artifact_accounting_role::clean_stash_payload,
            device1, 8, 8, llama_cache_acct_attr_kind::artifact,
        },
        {
            vbr_artifact_accounting_role::descriptor_metadata,
            host, 512, 512, llama_cache_acct_attr_kind::artifact,
        },
        {
            vbr_artifact_accounting_role::reference_metadata,
            host, 256, 256, llama_cache_acct_attr_kind::artifact,
        },
    };
    return package;
}

static vbr_artifact_decode_limits limits(uint64_t bytes) {
    vbr_artifact_decode_limits result;
    result.max_total_bytes = bytes;
    return result;
}

static void test_golden_and_native_lineage() {
    fixture_storage storage;
    auto package = make_package(storage);
    std::vector<uint8_t> encoded;
    CHECK(vbr_artifact_encode_vector(
              package, encoded, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(!encoded.empty());
    CHECK(package.topologies[0].digest ==
          llama_cache_acct_compute_topology_digest(package.topologies[0]));
    CHECK(hex(package.unit_blobs[0].unit_version_id.bytes()) ==
          "492279d0d4a8cbbba1cf997c706d077d1209fb720d52e4820788bbc9fccaccce");
    CHECK(hex(package.unit_blobs[0].payload_digest.bytes()) ==
          "8325d422f361b83e19f57dd0e6e566f330961cae4889c2f934bf94619316705f");
    CHECK(hex(package.unit_blobs[0].descriptor.clean_stash.payload_id.bytes()) ==
          "5c20925bd0120766c0a1db7995677754755488a70b3780dcd53e63ab1535cf18");
    CHECK(hex(package.manifest.capture_generation_id.bytes()) ==
          "7227e21ce3e7076d85d625a3e41baca0ebe3a8d078b44f16b97ca69f81a79407");
    CHECK(hex(package.manifest.token_block.digest.bytes()) ==
          "9635811050104e380f761c837ec49756986867248fab5e94877adbb7be90ad68");
    CHECK(hex(package.manifest.manifest_digest.bytes()) ==
          "6e7ac611f37686aa527bae124a115e58ca278a2122c76a336717b4f3b7637d4b");
    CHECK(hex(digest_of(encoded)) ==
          "085b609eb67c847f5086ccd8ae44c9a8cc0243d0ce279bee5c617c63298f03dd");
    CHECK(encoded.size() == 2358);
    CHECK(encoded[0] == 0x56 && encoded[1] == 0x42 &&
          encoded[2] == 0x52 && encoded[3] == 0x32);
    CHECK(encoded[4] == 2 && encoded[5] == 0 &&
          encoded[6] == 0 && encoded[7] == 0);

    vbr_artifact_package decoded;
    CHECK(vbr_artifact_decode_vector(
              encoded, limits(1024*1024), decoded) ==
          vbr_artifact_status::ok);
    CHECK(decoded.version == VBR_UNIT_ARTIFACT_FORMAT_VERSION);
    CHECK(decoded.unit_blobs.size() == 1);
    CHECK(decoded.manifest.generation.controllers.size() == 1);
    CHECK(decoded.manifest.generation.controllers[0].lineage_uuid ==
          package.manifest.generation.controllers[0].lineage_uuid);
    CHECK(decoded.manifest.generation.controllers[0].lineage_uuid ==
          package.unit_blobs[0].descriptor.lineage_uuid);
    CHECK(decoded.manifest.consistency.kind ==
          vbr_artifact_consistency_kind::capture_exact);
    CHECK(decoded.manifest.consistency.source_capture_generation_id ==
          decoded.manifest.capture_generation_id);
    CHECK(decoded.manifest.token_block.tokens ==
          package.manifest.token_block.tokens);
    CHECK(decoded.manifest.token_block.digest ==
          package.manifest.token_block.digest);
    CHECK(decoded.manifest.stream_placements.size() == 1);
    CHECK(decoded.manifest.stream_placements[0].cells[0].ext_x == 10);
    CHECK(decoded.manifest.stream_placements[0].cells[1].ext_y == 21);
}

static void test_v1_decode_and_v2_restore_metadata() {
    fixture_storage storage;
    auto legacy = make_package(storage);
    legacy.version = 1;
    legacy.manifest.version = 1;
    legacy.manifest.token_block = {};
    legacy.manifest.stream_placements.clear();
    std::vector<uint8_t> bytes;
    CHECK(vbr_artifact_encode_vector(legacy, bytes, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(bytes[4] == 1);
    vbr_artifact_package decoded;
    CHECK(vbr_artifact_decode_vector(bytes, limits(1024*1024), decoded) ==
          vbr_artifact_status::ok);
    CHECK(decoded.version == 1);
    CHECK(decoded.manifest.stream_placements.empty());
    CHECK(decoded.manifest.token_block.tokens.empty());
    CHECK(!decoded.manifest.token_block.digest.valid());
}

static void test_identity_and_reference_separation() {
    fixture_storage storage;
    auto first = make_package(storage);
    std::vector<uint8_t> bytes;
    CHECK(vbr_artifact_encode_vector(first, bytes, 1024*1024) ==
          vbr_artifact_status::ok);

    auto ownership_changed = make_package(storage);
    ownership_changed.manifest.generation.controllers[0].streams[0]
        .pages[0].covered_mask[0] = 0x5;
    ownership_changed.manifest.generation.controllers[0].streams[0]
        .captured_dependency_count = 2;
    ownership_changed.manifest.stream_placements[0].cells[1].physical_cell = 2;
    ownership_changed.manifest.unit_references[0].stash_reference
        .covered_sink_pages[0].covered_mask[0] = 0x5;
    CHECK(vbr_artifact_encode_vector(
              ownership_changed, bytes, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(ownership_changed.unit_blobs[0].unit_version_id ==
          first.unit_blobs[0].unit_version_id);
    CHECK(ownership_changed.manifest.manifest_digest !=
          first.manifest.manifest_digest);

    auto token_changed = make_package(storage);
    token_changed.manifest.token_block.tokens[1]++;
    CHECK(vbr_artifact_encode_vector(
              token_changed, bytes, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(token_changed.unit_blobs[0].unit_version_id ==
          first.unit_blobs[0].unit_version_id);
    CHECK(token_changed.manifest.token_block.digest !=
          first.manifest.token_block.digest);

    storage.payload0.bytes[0] ^= 1;
    auto payload_changed = make_package(storage);
    CHECK(vbr_artifact_encode_vector(
              payload_changed, bytes, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(payload_changed.unit_blobs[0].descriptor.lineage_uuid ==
          first.unit_blobs[0].descriptor.lineage_uuid);
    CHECK(payload_changed.unit_blobs[0].descriptor.repr_gen ==
          first.unit_blobs[0].descriptor.repr_gen);
    CHECK(payload_changed.unit_blobs[0].unit_version_id !=
          first.unit_blobs[0].unit_version_id);
    CHECK(payload_changed.unit_blobs[0].descriptor.clean_stash.payload_id ==
          first.unit_blobs[0].descriptor.clean_stash.payload_id);

    storage.payload0.bytes[0] ^= 1;
    auto generation_changed = make_package(storage);
    generation_changed.unit_blobs[0].descriptor.repr_gen++;
    generation_changed.manifest.generation.controllers[0].units[0].repr_gen++;
    generation_changed.manifest.unit_references[0].repr_gen++;
    CHECK(vbr_artifact_encode_vector(
              generation_changed, bytes, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(generation_changed.unit_blobs[0].unit_version_id !=
          first.unit_blobs[0].unit_version_id);
    CHECK(generation_changed.unit_blobs[0].descriptor.repr_gen !=
          first.unit_blobs[0].descriptor.repr_gen);

    auto mismatched_tuple = first;
    CHECK(mismatched_tuple.manifest.unit_references[0].unit_version_id ==
          mismatched_tuple.unit_blobs[0].unit_version_id);
    mismatched_tuple.manifest.unit_references[0].lineage_uuid.lo ^= 1;
    CHECK(vbr_artifact_encode_vector(
              mismatched_tuple, bytes, 1024*1024) ==
          vbr_artifact_status::generation_mismatch);

    // A different logical unit may intern the same immutable clean-stash
    // subobject without sharing the unit address.
    auto shared_stash = make_package(storage);
    shared_stash.unit_blobs[0].descriptor.lineage_uuid.lo++;
    shared_stash.manifest.generation.controllers[0].lineage_uuid.lo++;
    shared_stash.manifest.unit_references[0].lineage_uuid.lo++;
    CHECK(vbr_artifact_encode_vector(
              shared_stash, bytes, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(shared_stash.unit_blobs[0].descriptor.clean_stash.payload_id ==
          first.unit_blobs[0].descriptor.clean_stash.payload_id);
    CHECK(shared_stash.unit_blobs[0].unit_version_id !=
          first.unit_blobs[0].unit_version_id);

    auto absent = make_package(storage);
    absent.unit_blobs[0].descriptor.clean_stash_state =
        vbr_artifact_clean_stash_state::absent_at_source;
    absent.unit_blobs[0].descriptor.clean_stash = {};
    absent.manifest.unit_references[0].has_stash_reference = false;
    absent.manifest.unit_references[0].stash_reference = {};
    CHECK(vbr_artifact_encode_vector(absent, bytes, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(absent.unit_blobs[0].unit_version_id !=
          first.unit_blobs[0].unit_version_id);
    CHECK(absent.manifest.consistency.kind ==
          vbr_artifact_consistency_kind::capture_exact);

    // A source-present stash omission is a different wire state. It cannot claim
    // capture_exact; only a target-side, transition-authorized live_rebased reference may
    // carry it.
    auto omitted = make_package(storage);
    CHECK(vbr_artifact_prepare(omitted) == vbr_artifact_status::ok);
    omitted.unit_blobs[0].descriptor.clean_stash_state =
        vbr_artifact_clean_stash_state::omitted_source_present;
    omitted.unit_blobs[0].descriptor.clean_stash = {};
    omitted.manifest.unit_references[0].has_stash_reference = false;
    omitted.manifest.unit_references[0].stash_reference = {};
    omitted.manifest.accounting.erase(
        std::remove_if(
            omitted.manifest.accounting.begin(),
            omitted.manifest.accounting.end(),
            [](const vbr_artifact_portable_accounting_row & row) {
                return row.role ==
                    vbr_artifact_accounting_role::clean_stash_payload;
            }),
        omitted.manifest.accounting.end());
    CHECK(vbr_artifact_encode_vector(
              omitted, bytes, 1024*1024) != vbr_artifact_status::ok);
    omitted.manifest.consistency.kind =
        vbr_artifact_consistency_kind::live_rebased;
    omitted.manifest.consistency.source_capture_generation_id =
        omitted.manifest.capture_generation_id;
    omitted.manifest.consistency.target_capture_generation_id =
        vbr_capture_generation_id::from_sha256(marker(0x81));
    omitted.manifest.consistency.transition_lineage_id =
        vbr_transition_lineage_id::from_sha256(marker(0x82));
    CHECK(vbr_artifact_encode_vector(
              omitted, bytes, 1024*1024) == vbr_artifact_status::ok);
}

static void test_fail_closed_decode() {
    fixture_storage storage;
    auto package = make_package(storage);
    std::vector<uint8_t> encoded;
    CHECK(vbr_artifact_encode_vector(
              package, encoded, 1024*1024) ==
          vbr_artifact_status::ok);

    vbr_artifact_package decoded = package;
    auto corrupt = encoded;
    corrupt.back() ^= 1;
    CHECK(vbr_artifact_decode_vector(
              corrupt, limits(1024*1024), decoded) !=
          vbr_artifact_status::ok);

    const std::vector<uint8_t> token_wire = {
        1, 0, 0, 0, 2, 0, 0, 0,
        101, 0, 0, 0, 102, 0, 0, 0,
    };
    const auto token_at = std::search(
        encoded.begin(), encoded.end(),
        token_wire.begin(), token_wire.end());
    CHECK(token_at != encoded.end());
    if (token_at != encoded.end()) {
        auto token_corrupt = encoded;
        token_corrupt[size_t(token_at - encoded.begin()) + 8] ^= 1;
        CHECK(vbr_artifact_decode_vector(
                  token_corrupt, limits(1024*1024), decoded) !=
              vbr_artifact_status::ok);
    }

    const std::vector<uint8_t> placement_wire = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 2, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 10, 0, 0, 0,
        20, 0, 0, 0,
    };
    const auto placement_at = std::search(
        encoded.begin(), encoded.end(),
        placement_wire.begin(), placement_wire.end());
    CHECK(placement_at != encoded.end());
    if (placement_at != encoded.end()) {
        auto placement_corrupt = encoded;
        placement_corrupt[size_t(placement_at - encoded.begin()) + 32] ^= 1;
        CHECK(vbr_artifact_decode_vector(
                  placement_corrupt, limits(1024*1024), decoded) !=
              vbr_artifact_status::ok);
    }
    CHECK(decoded.version == 0 && decoded.unit_blobs.empty());

    staged_consumer staged;
    vbr_artifact_payload_consumer consumer {
        &staged, staged_consumer::consume, staged_consumer::finish,
    };
    memory_reader corrupt_reader { &corrupt, 0 };
    vbr_artifact_stream_reader stream {
        &corrupt_reader, memory_reader::read,
    };
    CHECK(vbr_artifact_decode(
              stream, corrupt.size(), limits(1024*1024),
              &consumer, decoded) != vbr_artifact_status::ok);
    CHECK(staged.finishes == 1 && !staged.verified && staged.bytes == 0);

    staged = {};
    memory_reader good_reader { &encoded, 0 };
    stream = { &good_reader, memory_reader::read };
    CHECK(vbr_artifact_decode(
              stream, encoded.size(), limits(1024*1024),
              &consumer, decoded) == vbr_artifact_status::ok);
    CHECK(staged.finishes == 1 && staged.verified && staged.bytes == 24);

    // Integrity is independent of eligibility: corrupt a byte in the first
    // canonical payload while leaving the reference tuple untouched.
    const auto payload_pos = std::search(
        encoded.begin(), encoded.end(),
        storage.payload0.bytes.begin(), storage.payload0.bytes.end());
    CHECK(payload_pos != encoded.end());
    auto corrupt_payload = encoded;
    corrupt_payload[size_t(payload_pos - encoded.begin())] ^= 1;
    const auto tuple_before =
        decoded.manifest.unit_references[0];
    CHECK(std::equal(
        encoded.begin(), encoded.begin() + (payload_pos - encoded.begin()),
        corrupt_payload.begin()));
    CHECK(std::equal(
        encoded.begin() + (payload_pos - encoded.begin()) + 1,
        encoded.end(),
        corrupt_payload.begin() + (payload_pos - encoded.begin()) + 1));
    CHECK(vbr_artifact_decode_vector(
              corrupt_payload, limits(1024*1024), decoded) !=
          vbr_artifact_status::ok);
    CHECK(tuple_before.lineage_uuid ==
          package.manifest.unit_references[0].lineage_uuid);
    CHECK(tuple_before.logical_unit_id ==
          package.manifest.unit_references[0].logical_unit_id);
    CHECK(tuple_before.repr_gen ==
          package.manifest.unit_references[0].repr_gen);

    auto bad_version = encoded;
    bad_version[4] = 3;
    CHECK(vbr_artifact_decode_vector(
              bad_version, limits(1024*1024), decoded) ==
          vbr_artifact_status::unsupported_version);
    auto bad_length = encoded;
    bad_length[16] ^= 1;
    CHECK(vbr_artifact_decode_vector(
              bad_length, limits(1024*1024), decoded) !=
          vbr_artifact_status::ok);
    auto bad_order = encoded;
    bad_order[28] ^= 1;
    CHECK(vbr_artifact_decode_vector(
          bad_order, limits(1024*1024), decoded) !=
          vbr_artifact_status::ok);
    // The topology section begins after the 92-byte package header and
    // 48-byte section header. A count within the configured maximum but
    // impossible for the remaining section must reject before resize().
    auto impossible_count = encoded;
    impossible_count[140] = 16;
    impossible_count[141] = 0;
    impossible_count[142] = 0;
    impossible_count[143] = 0;
    CHECK(vbr_artifact_decode_vector(
              impossible_count, limits(1024*1024), decoded) !=
          vbr_artifact_status::ok);
    auto trailing = encoded;
    trailing.push_back(0);
    CHECK(vbr_artifact_decode_vector(
              trailing, limits(1024*1024), decoded) !=
          vbr_artifact_status::ok);
    encoded.pop_back();
    CHECK(vbr_artifact_decode_vector(
              encoded, limits(1024*1024), decoded) !=
          vbr_artifact_status::ok);

    auto limited = limits(16);
    CHECK(vbr_artifact_decode_vector(
              corrupt, limited, decoded) ==
          vbr_artifact_status::out_of_bounds);
}

static void test_encoder_rejects_source_mutation() {
    fixture_storage storage;
    phased_source changing;
    auto package = make_package(storage);
    package.unit_blobs[0].descriptor.shards[0].payload =
        changing.source();
    std::vector<uint8_t> encoded;
    CHECK(vbr_artifact_encode_vector(
              package, encoded, 1024*1024) ==
          vbr_artifact_status::content_id_mismatch);
    CHECK(encoded.empty());
    CHECK(changing.calls > changing.switch_after);
}

static void test_validation_and_ordering() {
    fixture_storage storage;
    auto package = make_package(storage);
    std::vector<uint8_t> encoded;
    std::vector<uint8_t> canonical;
    CHECK(vbr_artifact_encode_vector(
              package, canonical, 1024*1024) ==
          vbr_artifact_status::ok);

    auto bad_topology = package;
    bad_topology.topologies[0].shard_weights[0]++;
    CHECK(vbr_artifact_encode_vector(
              bad_topology, encoded, 1024*1024) ==
          vbr_artifact_status::topology_mismatch);

    auto bad_accounting = package;
    bad_accounting.manifest.accounting[0].role =
        vbr_artifact_accounting_role::_count;
    CHECK(vbr_artifact_encode_vector(
              bad_accounting, encoded, 1024*1024) !=
          vbr_artifact_status::ok);
    bad_accounting = package;
    bad_accounting.manifest.accounting.push_back(
        bad_accounting.manifest.accounting.front());
    CHECK(vbr_artifact_encode_vector(
              bad_accounting, encoded, 1024*1024) !=
          vbr_artifact_status::ok);

    // Completion order is not wire order: stable shard indices canonicalize an
    // arbitrarily completed vector into identical bytes.
    auto reordered_shards = make_package(storage);
    std::swap(
        reordered_shards.unit_blobs[0].descriptor.shards[0],
        reordered_shards.unit_blobs[0].descriptor.shards[1]);
    std::swap(
        reordered_shards.unit_blobs[0].descriptor.clean_stash.shards[0],
        reordered_shards.unit_blobs[0].descriptor.clean_stash.shards[1]);
    CHECK(vbr_artifact_encode_vector(
              reordered_shards, encoded, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(encoded == canonical);

    auto duplicate_shard = make_package(storage);
    duplicate_shard.unit_blobs[0].descriptor.shards[1].shard_index = 0;
    CHECK(vbr_artifact_encode_vector(
              duplicate_shard, encoded, 1024*1024) !=
          vbr_artifact_status::ok);

    auto bad_enum = package;
    bad_enum.unit_blobs[0].descriptor.layout =
        vbr_artifact_layout::_count;
    CHECK(vbr_artifact_encode_vector(
          bad_enum, encoded, 1024*1024) !=
          vbr_artifact_status::ok);

    auto bad_representation = make_package(storage);
    bad_representation.unit_blobs[0].descriptor.representation.kind =
        vbr_artifact_representation_kind::_count;
    CHECK(vbr_artifact_encode_vector(
              bad_representation, encoded, 1024*1024) !=
          vbr_artifact_status::ok);
    auto bad_recoverability = make_package(storage);
    bad_recoverability.unit_blobs[0].descriptor.recoverability =
        vbr_artifact_recoverability::_count;
    CHECK(vbr_artifact_encode_vector(
              bad_recoverability, encoded, 1024*1024) !=
          vbr_artifact_status::ok);
    auto bad_side = make_package(storage);
    bad_side.unit_blobs[0].descriptor.side = vbr_artifact_side::_count;
    CHECK(vbr_artifact_encode_vector(
              bad_side, encoded, 1024*1024) !=
          vbr_artifact_status::ok);
    auto bad_stash_state = make_package(storage);
    bad_stash_state.unit_blobs[0].descriptor.clean_stash_state =
        vbr_artifact_clean_stash_state::_count;
    CHECK(vbr_artifact_encode_vector(
              bad_stash_state, encoded, 1024*1024) !=
          vbr_artifact_status::ok);
    auto bad_consistency = make_package(storage);
    bad_consistency.manifest.consistency.kind =
        vbr_artifact_consistency_kind::_count;
    CHECK(vbr_artifact_encode_vector(
              bad_consistency, encoded, 1024*1024) !=
          vbr_artifact_status::ok);

    auto bad_placement_set = package;
    bad_placement_set.manifest.stream_placements[0].cells[1].physical_cell = 2;
    CHECK(vbr_artifact_encode_vector(
              bad_placement_set, encoded, 1024*1024) !=
          vbr_artifact_status::ok);
    auto duplicate_position = package;
    duplicate_position.manifest.stream_placements[0].cells[1]
        .logical_position = 0;
    CHECK(vbr_artifact_encode_vector(
              duplicate_position, encoded, 1024*1024) !=
          vbr_artifact_status::ok);

    // Logical positions are unique per source sequence across the complete
    // reference, not merely within one stream. Two streams may cite distinct
    // physical cells for the same sequence, but may not alias a logical token.
    auto cross_stream = package;
    auto & controller = cross_stream.manifest.generation.controllers[0];
    auto second_stream = controller.streams[0];
    second_stream.stream_index = 1;
    second_stream.computation_frontier = 4;
    second_stream.captured_dependency_count = 2;
    second_stream.pages[0].covered_mask[0] = 0x0c;
    controller.streams.push_back(second_stream);
    cross_stream.unit_blobs[0].descriptor.n_stream = 2;
    cross_stream.manifest.controller_policy[0].n_stream = 2;
    cross_stream.manifest.identity.token_count = 4;
    cross_stream.manifest.identity.next_position = 4;
    cross_stream.manifest.token_block.tokens = { 101, 102, 103, 104 };
    vbr_artifact_stream_placement second_placement;
    second_placement.child_id = 0;
    second_placement.stream_index = 1;
    second_placement.source_sequence = 0;
    second_placement.computation_frontier = 4;
    second_placement.cells = {
        { 2, 2, 12, 22 },
        { 3, 3, 13, 23 },
    };
    cross_stream.manifest.stream_placements.push_back(second_placement);
    cross_stream.manifest.unit_references[0].authorized_stream_refs = { 0, 1 };
    CHECK(vbr_artifact_encode_vector(
              cross_stream, encoded, 1024*1024) ==
          vbr_artifact_status::ok);

    auto cross_stream_duplicate_position = cross_stream;
    cross_stream_duplicate_position.manifest.stream_placements[1]
        .cells[0].logical_position = 1;
    CHECK(vbr_artifact_encode_vector(
              cross_stream_duplicate_position, encoded, 1024*1024) !=
          vbr_artifact_status::ok);

    auto beyond_frontier = package;
    beyond_frontier.manifest.stream_placements[0].cells[1]
        .logical_position = 2;
    CHECK(vbr_artifact_encode_vector(
              beyond_frontier, encoded, 1024*1024) !=
          vbr_artifact_status::ok);
    auto token_count_mismatch = package;
    token_count_mismatch.manifest.token_block.tokens.pop_back();
    CHECK(vbr_artifact_encode_vector(
              token_count_mismatch, encoded, 1024*1024) !=
          vbr_artifact_status::ok);

    auto bad_stash_authorization = package;
    bad_stash_authorization.manifest.unit_references[0].stash_reference
        .covered_sink_pages[0].covered_mask[0] = 0x5;
    CHECK(vbr_artifact_encode_vector(
              bad_stash_authorization, encoded, 1024*1024) !=
          vbr_artifact_status::ok);

    auto constrained = limits(canonical.size());
    constrained.max_devices_per_topology = 1;
    vbr_artifact_package decoded;
    CHECK(vbr_artifact_decode_vector(
              canonical, constrained, decoded) !=
          vbr_artifact_status::ok);
    constrained = limits(canonical.size());
    constrained.max_unit_blobs = 0;
    CHECK(vbr_artifact_decode_vector(
              canonical, constrained, decoded) !=
          vbr_artifact_status::ok);
    constrained = limits(canonical.size());
    constrained.max_accounting_rows = 1;
    CHECK(vbr_artifact_decode_vector(
              canonical, constrained, decoded) !=
          vbr_artifact_status::ok);
}

static void test_companion_payload() {
    fixture_storage storage;
    auto package = make_package(storage);
    vbr_artifact_companion_payload companion;
    companion.kind = vbr_artifact_companion_kind::recurrent;
    companion.format_version = 3;
    companion.build_identity_digest = marker(0x91);
    companion.domain = {
        llama_cache_acct_residency::pageable_host,
        llama_cache_acct_domain_kind::not_applicable,
        UINT32_MAX,
        UINT16_MAX,
    };
    companion.payload_bytes = storage.recurrent.bytes.size();
    companion.payload = storage.recurrent.source();
    package.companions.push_back(companion);
    package.manifest.accounting.push_back({
        vbr_artifact_accounting_role::recurrent_payload,
        companion.domain,
        companion.payload_bytes,
        companion.payload_bytes,
        llama_cache_acct_attr_kind::artifact,
    });

    std::vector<uint8_t> encoded;
    CHECK(vbr_artifact_encode_vector(
              package, encoded, 1024*1024) ==
          vbr_artifact_status::ok);
    vbr_artifact_package decoded;
    CHECK(vbr_artifact_decode_vector(
              encoded, limits(1024*1024), decoded) ==
          vbr_artifact_status::ok);
    CHECK(decoded.companions.size() == 1);
    CHECK(decoded.manifest.companions.size() == 1);
    CHECK(decoded.companions[0].domain == companion.domain);
    CHECK(decoded.companions[0].payload_digest ==
          decoded.manifest.companions[0].payload_digest);
}

struct discard_writer {
    uint64_t bytes = 0;

    static bool write(
            void * context,
            const uint8_t *,
            size_t size) noexcept {
        auto * self = static_cast<discard_writer *>(context);
        self->bytes += size;
        return true;
    }
};

static void test_stream_larger_than_capture_ring() {
    constexpr uint64_t ring_bytes = 256ull*1024*1024;
    generated_source generated { ring_bytes + 1, 0x39 };
    fixture_storage storage;
    auto package = make_package(storage);
    auto & descriptor = package.unit_blobs[0].descriptor;
    descriptor.shards.resize(1);
    descriptor.shards[0] = make_shard(0, generated.source());
    descriptor.shards[0].device_ordinal = 0;
    descriptor.wm_cells = 1;
    descriptor.dimensions = { 1, generated.size, 0, 0 };
    descriptor.clean_stash_state =
        vbr_artifact_clean_stash_state::absent_at_source;
    descriptor.clean_stash = {};
    package.manifest.unit_references[0].has_stash_reference = false;
    package.manifest.unit_references[0].stash_reference = {};
    const auto descriptor_row = package.manifest.accounting[
        package.manifest.accounting.size() - 2];
    const auto reference_row = package.manifest.accounting.back();
    package.manifest.accounting = {
        {
            vbr_artifact_accounting_role::unit_payload,
            {
                llama_cache_acct_residency::device,
                llama_cache_acct_domain_kind::device_topology,
                0,
                0,
            },
            generated.size,
            generated.size,
            llama_cache_acct_attr_kind::artifact,
        },
        descriptor_row,
        reference_row,
    };

    discard_writer discarded;
    const vbr_artifact_stream_writer sink {
        &discarded, discard_writer::write,
    };
    uint64_t encoded_size = 0;
    CHECK(vbr_artifact_encode(
              package, sink, ring_bytes + 1024*1024, &encoded_size) ==
          vbr_artifact_status::ok);
    CHECK(encoded_size > ring_bytes);
    CHECK(discarded.bytes == encoded_size);
    CHECK(generated.calls > 1);
    CHECK(generated.max_request <= 1024*1024);
}

static llama_cache_acct_value catalog_cell(
        const llama_cache_acct_snapshot & snapshot,
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_measure measure) {
    const auto row = std::find_if(
        snapshot.cells.begin(), snapshot.cells.end(),
        [&](const llama_cache_acct_cell_row & candidate) {
            return candidate.category == category &&
                   candidate.domain == domain;
        });
    return row == snapshot.cells.end()
        ? llama_cache_acct_value {}
        : row->cell.measures[size_t(measure)];
}

struct catalog_fixture {
    fixture_storage storage;
    vbr_artifact_package package;
    llama_cache_acct_ledger ledger;
    std::unique_ptr<llama_vbr_artifact_catalog> catalog;
    std::vector<llama_vbr_artifact_domain_binding> bindings;
    llama_cache_budget_config budget;
    llama_cache_acct_resource_domain host =
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pageable_host);
    llama_cache_acct_resource_domain pinned =
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pinned_host);

    explicit catalog_fixture(
            bool configure_catalog = true,
            bool include_pinned = false)
        : package(make_package(storage)),
          catalog(new llama_vbr_artifact_catalog(ledger)) {
        CHECK(catalog->bind_topologies(package.topologies, bindings));
        std::vector<llama_cache_acct_completeness_requirement> required;
        required.push_back({
            host, llama_cache_acct_producer::retention_sidecar,
        });
        if (include_pinned) {
            required.push_back({
                pinned, llama_cache_acct_producer::retention_sidecar,
            });
        }
        for (const auto & binding : bindings) {
            required.push_back({
                binding.domain, llama_cache_acct_producer::live_memory,
            });
        }
        CHECK(ledger.configure_required_producers(
            required.data(), required.size()));
        if (configure_catalog) {
            CHECK(catalog->configure_accounting(package));
        }
        const auto initialize = [&](llama_cache_acct_category category,
                                    const llama_cache_acct_resource_domain & domain,
                                    bool transactional) {
            ledger.gauge_set(
                category, domain,
                llama_cache_acct_measure::resident_allocated, 0);
            if (transactional) {
                ledger.gauge_set(
                    category, domain,
                    llama_cache_acct_measure::reserved, 0);
            }
        };
        initialize(
            llama_cache_acct_category::full_snapshot_payload,
            host, true);
        initialize(
            llama_cache_acct_category::checkpoint_state_payload,
            host, true);
        initialize(
            llama_cache_acct_category::typed_accelerator_payload,
            host, true);
        if (include_pinned) {
            for (uint8_t raw = 0;
                 raw < uint8_t(llama_cache_acct_category::_count);
                 ++raw) {
                const auto category = llama_cache_acct_category(raw);
                const auto classification =
                    llama_cache_budget_classify(category);
                if (classification.participation !=
                        llama_cache_budget_capacity_participation::participating ||
                    (classification.scope !=
                         llama_cache_budget_residency_scope::host &&
                     classification.scope !=
                         llama_cache_budget_residency_scope::by_domain)) {
                    continue;
                }
                for (const auto measure : {
                        llama_cache_acct_measure::logical_payload,
                        llama_cache_acct_measure::resident_allocated,
                        llama_cache_acct_measure::reserved }) {
                    ledger.gauge_set(category, pinned, measure, 0);
                }
            }
        }
        for (const auto & binding : bindings) {
            initialize(
                llama_cache_acct_category::live_attention_state,
                binding.domain, false);
            initialize(
                llama_cache_acct_category::live_recurrent_state,
                binding.domain, false);
            initialize(
                llama_cache_acct_category::recurrent_rollback_planes,
                binding.domain, false);
            initialize(
                llama_cache_acct_category::rolling_window_tape,
                binding.domain, false);
        }
        CHECK(ledger.certify_complete(
            host, llama_cache_acct_producer::retention_sidecar));
        if (include_pinned) {
            CHECK(ledger.certify_complete(
                pinned, llama_cache_acct_producer::retention_sidecar));
        }
        for (const auto & binding : bindings) {
            CHECK(ledger.certify_complete(
                binding.domain, llama_cache_acct_producer::live_memory));
            llama_cache_budget_device_input input;
            input.backend_device =
                reinterpret_cast<const void *>(uintptr_t(1));
            input.domain = binding.domain;
            input.physical_total = 1ull << 30;
            // Fake-shard storage is already resident when publication begins,
            // so the injected point-in-time sample must include it in physical
            // used just as a backend sample would.
            input.physical_free = (1ull << 30) - 1024;
            input.phys_state =
                llama_cache_budget_capacity_state::known;
            input.current_compute_allocated = 0;
            input.configured_compute_reserve = 0;
            input.compute_state =
                llama_cache_budget_capacity_state::known;
            input.cache_cap_state =
                llama_cache_budget_capacity_state::unbounded;
            budget.devices.push_back(input);
        }
        budget.host.pageable_state =
            llama_cache_budget_capacity_state::unbounded;
        if (include_pinned) {
            budget.host.pinned_state =
                llama_cache_budget_capacity_state::unbounded;
        }
    }

    std::vector<llama_vbr_artifact_fake_shard_completion>
    completions() const {
        return {
            { 0, 1, true,  true, storage.stash1.bytes },
            { 0, 0, false, true, storage.payload0.bytes },
            { 0, 0, true,  true, storage.stash0.bytes },
            { 0, 1, false, true, storage.payload1.bytes },
        };
    }
};

static vbr_verified_segment verified_segment(
        const llama_vbr_artifact_fake_shard_completion & completion,
        size_t split = SIZE_MAX) {
    auto chain = std::make_shared<artifact_segment_chain>();
    const size_t first = std::min(
        split, completion.bytes.size());
    CHECK(chain->append(completion.bytes.data(), first));
    if (first < completion.bytes.size()) {
        CHECK(chain->append(
            completion.bytes.data() + first,
            completion.bytes.size() - first));
    }
    vbr_verified_segment out;
    out.unit_index = completion.unit_index;
    out.shard_index = completion.shard_index;
    out.clean_stash = completion.clean_stash;
    out.bytes = std::move(chain);
    out.streaming_digest =
        vbr_capture_stream_digest(*out.bytes);
    return out;
}

static void test_catalog_streaming_protocol() {
    catalog_fixture f;
    vbr_capture_stream_status status;
    auto build = f.catalog->begin_capture(
        f.package, f.budget, {}, status);
    CHECK(build);
    CHECK(status == vbr_capture_stream_status::ok);
    if (!build) {
        return;
    }
    const uint64_t claims =
        f.ledger.snapshot().live_ops;
    CHECK(claims ==
          f.package.manifest.accounting.size() + 1);

    auto unit = build->begin_unit(0, status);
    CHECK(unit);
    const auto completions = f.completions();
    const size_t order[] = { 3, 0, 2, 1 };
    for (const size_t index : order) {
        auto segment = verified_segment(
            completions[index], 1);
        CHECK(unit->accept_verified_segment(segment) ==
              vbr_capture_stream_status::ok);
    }
    CHECK(f.ledger.snapshot().live_ops == claims);
    CHECK(unit->seal_unit() ==
          vbr_capture_stream_status::ok);
    auto late = verified_segment(completions[0]);
    CHECK(unit->accept_verified_segment(late) ==
          vbr_capture_stream_status::late_segment);
    const auto streamed = build->publish_reference();
    CHECK(streamed.status ==
          vbr_capture_stream_status::ok);
    CHECK(!streamed.adopted);
    CHECK(streamed.reference_artifact.v != 0);
    CHECK(streamed.unit_content.v != 0);
    unit.reset();
    build.reset();

    auto snapshot = f.ledger.snapshot();
    CHECK(snapshot.live_ops ==
          f.package.manifest.accounting.size());
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::transfer_staging,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 0);

    auto adopted_build = f.catalog->begin_capture(
        f.package, f.budget, {}, status);
    CHECK(adopted_build);
    auto adopted_unit =
        adopted_build->begin_unit(0, status);
    CHECK(adopted_unit);
    for (const auto & completion : f.completions()) {
        auto value = verified_segment(completion, 2);
        CHECK(adopted_unit->accept_verified_segment(value) ==
              vbr_capture_stream_status::ok);
    }
    CHECK(adopted_unit->seal_unit() ==
          vbr_capture_stream_status::ok);
    const auto adopted =
        adopted_build->publish_reference();
    CHECK(adopted.status == vbr_capture_stream_status::ok);
    CHECK(adopted.adopted);
    CHECK(adopted.reference_artifact !=
          streamed.reference_artifact);
    CHECK(adopted.unit_content == streamed.unit_content);
    CHECK(adopted.reference_lineage ==
          streamed.reference_lineage);
    adopted_unit.reset();
    adopted_build.reset();

    snapshot = f.ledger.snapshot();
    CHECK(snapshot.live_ops ==
          2*f.package.manifest.accounting.size());
    for (const auto & binding : f.bindings) {
        CHECK(catalog_cell(
            snapshot,
            llama_cache_acct_category::unit_version_payload,
            binding.domain,
            llama_cache_acct_measure::resident_allocated).value == 4);
        CHECK(catalog_cell(
            snapshot,
            llama_cache_acct_category::clean_stash_payload,
            binding.domain,
            llama_cache_acct_measure::resident_allocated).value == 8);
    }
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_descriptor_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 512);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_reference_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 512);
    const auto adopted_state = f.catalog->snapshot();
    CHECK(adopted_state.blobs == 1);
    CHECK(adopted_state.stashes == 1);
    CHECK(adopted_state.references == 2);
    CHECK(adopted_state.published == 1);
    CHECK(adopted_state.adopted == 1);

    catalog_fixture fake_equivalent;
    const auto fake_first = fake_equivalent.catalog->publish(
        fake_equivalent.package,
        fake_equivalent.completions(),
        fake_equivalent.budget);
    CHECK(fake_first.status ==
          llama_vbr_artifact_publish_status::published);
    CHECK(fake_first.reference_artifact ==
          streamed.reference_artifact);

    catalog_fixture invalid;
    auto bad_build = invalid.catalog->begin_capture(
        invalid.package, invalid.budget, {}, status);
    CHECK(bad_build);
    auto bad_unit = bad_build->begin_unit(0, status);
    CHECK(bad_unit);
    auto segment = verified_segment(
        invalid.completions()[0]);
    CHECK(bad_unit->accept_verified_segment(segment) ==
          vbr_capture_stream_status::ok);
    CHECK(bad_unit->accept_verified_segment(segment) ==
          vbr_capture_stream_status::duplicate_segment);
    CHECK(bad_unit->seal_unit() ==
          vbr_capture_stream_status::duplicate_segment);
    CHECK(bad_build->publish_reference().status ==
          vbr_capture_stream_status::duplicate_segment);
    bad_unit.reset();
    bad_build.reset();
    CHECK(invalid.ledger.snapshot().live_ops == 0);
    CHECK(invalid.catalog->snapshot().references == 0);

    catalog_fixture missing;
    auto missing_build = missing.catalog->begin_capture(
        missing.package, missing.budget, {}, status);
    CHECK(missing_build);
    auto missing_unit =
        missing_build->begin_unit(0, status);
    CHECK(missing_unit);
    auto only = verified_segment(missing.completions()[0]);
    CHECK(missing_unit->accept_verified_segment(only) ==
          vbr_capture_stream_status::ok);
    CHECK(missing_unit->seal_unit() ==
          vbr_capture_stream_status::missing_segment);
    missing_unit.reset();
    missing_build.reset();
    CHECK(missing.ledger.snapshot().live_ops == 0);
    CHECK(missing.catalog->snapshot().references == 0);

    for (const bool commit_fault : { false, true }) {
        catalog_fixture faulted;
        llama_cache_transaction_fault fault;
        if (commit_fault) {
            fault.fail_commit_at = 0;
        } else {
            fault.fail_stage_at = 0;
        }
        auto fault_build = faulted.catalog->begin_capture(
            faulted.package, faulted.budget, fault, status);
        CHECK(fault_build);
        auto fault_unit =
            fault_build->begin_unit(0, status);
        CHECK(fault_unit);
        for (const auto & completion :
             faulted.completions()) {
            auto value = verified_segment(completion);
            CHECK(fault_unit->accept_verified_segment(value) ==
                  vbr_capture_stream_status::ok);
        }
        CHECK(fault_unit->seal_unit() ==
              vbr_capture_stream_status::ok);
        const auto failed = fault_build->publish_reference();
        CHECK(failed.status ==
              (commit_fault
                   ? vbr_capture_stream_status::commit_failed
                   : vbr_capture_stream_status::stage_failed));
        fault_unit.reset();
        fault_build.reset();
        CHECK(faulted.ledger.snapshot().live_ops == 0);
        CHECK(faulted.catalog->snapshot().references == 0);
    }

    catalog_fixture overlap;
    const auto generous_overlap_budget = overlap.budget;
    overlap.budget.host.pageable_cap = 1000;
    overlap.budget.host.pageable_state =
        llama_cache_budget_capacity_state::known;
    vbr_capture_begin_diagnostics refusal_diagnostics;
    auto refused = overlap.catalog->begin_capture(
        overlap.package, overlap.budget, {}, status,
        &refusal_diagnostics);
    CHECK(!refused);
    CHECK(status ==
          vbr_capture_stream_status::accounting_refused);
    CHECK(refusal_diagnostics.reservation_group ==
          vbr_capture_reservation_group::durable_artifact);
    CHECK(refusal_diagnostics.prepare_status ==
          llama_cache_prepare_status::admission_refused);
    CHECK(refusal_diagnostics.admission_status ==
          llama_cache_admission_status::exceeds_budget);
    CHECK(overlap.catalog->snapshot()
              .staging_overlap_refusals == 1);
    CHECK(overlap.ledger.snapshot().live_ops == 0);
    const auto fake_after_refusal = overlap.catalog->publish(
        overlap.package, overlap.completions(),
        generous_overlap_budget);
    CHECK(fake_after_refusal.status ==
          llama_vbr_artifact_publish_status::published);

    catalog_fixture unconfigured(false);
    const auto unavailable = unconfigured.catalog->publish(
        unconfigured.package, unconfigured.completions(),
        unconfigured.budget);
    CHECK(unavailable.status ==
          llama_vbr_artifact_publish_status::
              accounting_unavailable);
    CHECK(unconfigured.ledger.snapshot().live_ops == 0);
}

static void test_catalog_multi_unit_atomic_publish() {
    catalog_fixture f;
    auto & package = f.package;
    auto second = package.unit_blobs.front();
    second.descriptor.logical_unit_id = 1;
    second.descriptor.clean_stash_state =
        vbr_artifact_clean_stash_state::absent_at_source;
    second.descriptor.clean_stash = {};
    second.unit_version_id = {};
    second.payload_digest = {};
    package.unit_blobs.push_back(second);
    package.manifest.generation.controllers[0].units.push_back(
        package.manifest.generation.controllers[0].units.front());
    auto second_reference =
        package.manifest.unit_references.front();
    second_reference.logical_unit_id = 1;
    second_reference.unit_version_id = {};
    second_reference.payload_digest = {};
    second_reference.has_stash_reference = false;
    second_reference.stash_reference = {};
    package.manifest.unit_references.push_back(second_reference);
    for (auto & row : package.manifest.accounting) {
        if (row.role ==
                vbr_artifact_accounting_role::unit_payload) {
            row.logical_bytes *= 2;
            row.resident_bytes *= 2;
        }
    }
    package.manifest.manifest_digest = {};
    package.manifest.capture_generation_id = {};
    package.manifest.consistency = {};

    CHECK(f.catalog->configure_accounting(package));
    vbr_capture_stream_status status;
    {
        auto aborted = f.catalog->begin_capture(
            package, f.budget, {}, status);
        CHECK(aborted);
        auto first_only = aborted
            ? aborted->begin_unit(0, status) : nullptr;
        CHECK(first_only);
        if (first_only) {
            for (const auto & completion : f.completions()) {
                auto segment = verified_segment(completion, 2);
                CHECK(first_only->accept_verified_segment(segment) ==
                      vbr_capture_stream_status::ok);
            }
            CHECK(first_only->seal_unit() ==
                  vbr_capture_stream_status::ok);
        }
        first_only.reset();
        aborted.reset();
        CHECK(f.catalog->snapshot().references == 0);
        CHECK(f.ledger.snapshot().live_ops == 0);
    }
    auto build = f.catalog->begin_capture(
        package, f.budget, {}, status);
    CHECK(build);
    if (!build) {
        return;
    }
    const auto completions = f.completions();
    auto first = build->begin_unit(0, status);
    CHECK(first);
    for (const auto & completion : completions) {
        auto segment = verified_segment(completion, 2);
        CHECK(first->accept_verified_segment(segment) ==
              vbr_capture_stream_status::ok);
    }
    CHECK(first->seal_unit() ==
          vbr_capture_stream_status::ok);
    first.reset();

    auto second_unit = build->begin_unit(1, status);
    CHECK(second_unit);
    for (const auto & completion : completions) {
        if (completion.clean_stash) {
            continue;
        }
        auto copy = completion;
        copy.unit_index = 1;
        auto segment = verified_segment(copy, 3);
        CHECK(second_unit->accept_verified_segment(segment) ==
              vbr_capture_stream_status::ok);
    }
    CHECK(second_unit->seal_unit() ==
          vbr_capture_stream_status::ok);
    second_unit.reset();

    const auto published = build->publish_reference();
    CHECK(published.status == vbr_capture_stream_status::ok);
    CHECK(published.reference_artifact.v != 0);
    const auto snapshot = f.catalog->snapshot();
    CHECK(snapshot.blobs == 2);
    CHECK(snapshot.stashes == 1);
    CHECK(snapshot.references == 1);
    build.reset();
    CHECK(f.catalog->retire(published.reference_artifact) == vbr_artifact_retire_status::retired);
    CHECK(f.catalog->snapshot().blobs == 0);
    CHECK(f.catalog->snapshot().stashes == 0);
    CHECK(f.ledger.snapshot().live_ops == 0);
}

static void test_catalog_streaming_companion_lifetime() {
    catalog_fixture f;
    vbr_artifact_companion_payload companion;
    companion.kind = vbr_artifact_companion_kind::recurrent;
    companion.format_version = 1;
    companion.build_identity_digest = marker(0xa1);
    companion.domain = {
        llama_cache_acct_residency::pageable_host,
        llama_cache_acct_domain_kind::not_applicable,
        UINT32_MAX,
        UINT16_MAX,
    };
    companion.payload_bytes = f.storage.recurrent.bytes.size();
    f.package.companions.push_back(companion);
    f.package.manifest.accounting.push_back({
        vbr_artifact_accounting_role::recurrent_payload,
        companion.domain,
        companion.payload_bytes,
        companion.payload_bytes,
        llama_cache_acct_attr_kind::artifact,
    });
    CHECK(f.catalog->configure_accounting(f.package));

    vbr_capture_stream_status status;
    auto build = f.catalog->begin_capture(
        f.package, f.budget, {}, status);
    CHECK(build);
    if (!build) {
        return;
    }
    auto unit = build->begin_unit(0, status);
    CHECK(unit);
    for (const auto & completion : f.completions()) {
        const auto segment = verified_segment(completion, 2);
        CHECK(unit->accept_verified_segment(segment) ==
              vbr_capture_stream_status::ok);
    }
    CHECK(unit->seal_unit() ==
          vbr_capture_stream_status::ok);
    unit.reset();

    auto companion_bytes =
        std::make_shared<artifact_segment_chain>();
    CHECK(companion_bytes->append(
        f.storage.recurrent.bytes.data(),
        f.storage.recurrent.bytes.size()));
    vbr_verified_companion verified;
    verified.companion_index = 0;
    verified.bytes = companion_bytes;
    verified.streaming_digest =
        vbr_capture_stream_digest(*companion_bytes);
    CHECK(build->accept_verified_companion(verified) ==
          vbr_capture_stream_status::ok);
    const auto published = build->publish_reference();
    CHECK(published.status == vbr_capture_stream_status::ok);
    build.reset();
    companion_bytes.reset();

    const auto snapshot = f.ledger.snapshot();
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::full_snapshot_payload,
        f.host,
        llama_cache_acct_measure::resident_allocated).value ==
        f.storage.recurrent.bytes.size());
    CHECK(f.catalog->retire(published.reference_artifact) == vbr_artifact_retire_status::retired);
    CHECK(f.ledger.snapshot().live_ops == 0);
}

static void test_catalog_charge_once_and_retire() {
    catalog_fixture f;
    const size_t alloc_baseline =
        f.ledger.allocation_registry_size();
    const auto first =
        f.catalog->publish(f.package, f.completions(), f.budget);
    CHECK(first.status ==
          llama_vbr_artifact_publish_status::published);
    CHECK(first.reference_artifact.v != 0);
    CHECK(first.unit_content.v != 0);

    auto snapshot = f.ledger.snapshot();
    for (const auto & binding : f.bindings) {
        CHECK(catalog_cell(
            snapshot,
            llama_cache_acct_category::unit_version_payload,
            binding.domain,
            llama_cache_acct_measure::resident_allocated).value == 4);
        CHECK(catalog_cell(
            snapshot,
            llama_cache_acct_category::clean_stash_payload,
            binding.domain,
            llama_cache_acct_measure::resident_allocated).value == 8);
    }
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_descriptor_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 512);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_reference_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 256);

    const auto second =
        f.catalog->publish(f.package, f.completions(), f.budget);
    CHECK(second.status ==
          llama_vbr_artifact_publish_status::adopted);
    CHECK(second.reference_artifact.v != first.reference_artifact.v);
    CHECK(second.unit_content == first.unit_content);
    CHECK(second.reference_lineage == first.reference_lineage);
    snapshot = f.ledger.snapshot();
    for (const auto & binding : f.bindings) {
        CHECK(catalog_cell(
            snapshot,
            llama_cache_acct_category::unit_version_payload,
            binding.domain,
            llama_cache_acct_measure::resident_allocated).value == 4);
        CHECK(catalog_cell(
            snapshot,
            llama_cache_acct_category::clean_stash_payload,
            binding.domain,
            llama_cache_acct_measure::resident_allocated).value == 8);
    }
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_reference_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 512);
    const auto catalog_state = f.catalog->snapshot();
    CHECK(catalog_state.blobs == 1);
    CHECK(catalog_state.stashes == 1);
    CHECK(catalog_state.references == 2);
    CHECK(f.ledger.allocation_registry_size() > alloc_baseline);

    CHECK(f.catalog->retire(first.reference_artifact) == vbr_artifact_retire_status::retired);
    CHECK(f.ledger.allocation_registry_size() > alloc_baseline);
    snapshot = f.ledger.snapshot();
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::unit_version_payload,
        f.bindings[0].domain,
        llama_cache_acct_measure::resident_allocated).value == 4);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_reference_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 256);
    CHECK(f.catalog->retire(second.reference_artifact) == vbr_artifact_retire_status::retired);
    CHECK(f.ledger.allocation_registry_size() == alloc_baseline);
    snapshot = f.ledger.snapshot();
    CHECK(snapshot.live_ops == 0);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::unit_version_payload,
        f.bindings[0].domain,
        llama_cache_acct_measure::resident_allocated).value == 0);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::clean_stash_payload,
        f.bindings[0].domain,
        llama_cache_acct_measure::resident_allocated).value == 0);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_descriptor_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 0);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_reference_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 0);
    const auto empty = f.catalog->snapshot();
    CHECK(empty.blobs == 0 && empty.stashes == 0 &&
          empty.references == 0);
}

static void test_catalog_all_shard_failures_and_rollback() {
    for (uint32_t mode = 0; mode < 6; ++mode) {
        catalog_fixture f;
        auto completions = f.completions();
        llama_vbr_artifact_publish_fault fault;
        llama_vbr_artifact_publish_status expected;
        switch (mode) {
            case 0:
                completions.pop_back();
                expected =
                    llama_vbr_artifact_publish_status::missing_completion;
                break;
            case 1:
                completions.back() = completions.front();
                expected =
                    llama_vbr_artifact_publish_status::duplicate_completion;
                break;
            case 2:
                completions[1].success = false;
                expected =
                    llama_vbr_artifact_publish_status::shard_failed;
                break;
            case 3:
                fault.fail_stage_at = 1;
                expected =
                    llama_vbr_artifact_publish_status::stage_failed;
                break;
            case 4:
                fault.fail_commit_at = 2;
                expected =
                    llama_vbr_artifact_publish_status::commit_failed;
                break;
            default:
                fault.fail_after_commit = true;
                expected =
                    llama_vbr_artifact_publish_status::publication_failed;
                break;
        }
        const auto result =
            f.catalog->publish(f.package, completions, f.budget, fault);
        CHECK(result.status == expected);
        const auto catalog_state = f.catalog->snapshot();
        CHECK(catalog_state.blobs == 0);
        CHECK(catalog_state.stashes == 0);
        CHECK(catalog_state.references == 0);
        const auto ledger_state = f.ledger.snapshot();
        CHECK(ledger_state.live_ops == 0);
        for (const auto & row : f.package.manifest.accounting) {
            const auto category =
                vbr_artifact_accounting_category(row.role);
            const auto domain =
                row.domain.residency ==
                        llama_cache_acct_residency::device
                    ? f.bindings[row.domain.device_ordinal].domain
                    : f.host;
            for (const auto measure : {
                    llama_cache_acct_measure::logical_payload,
                    llama_cache_acct_measure::resident_allocated,
                    llama_cache_acct_measure::reserved }) {
                const auto value = catalog_cell(
                    ledger_state, category, domain, measure);
                CHECK(value.state == llama_cache_acct_known::known);
                CHECK(value.value == 0);
            }
        }
    }
}

static void test_catalog_destructor_releases_live_references() {
    catalog_fixture f;
    const auto first =
        f.catalog->publish(f.package, f.completions(), f.budget);
    const auto second =
        f.catalog->publish(f.package, f.completions(), f.budget);
    CHECK(first.status ==
          llama_vbr_artifact_publish_status::published);
    CHECK(second.status ==
          llama_vbr_artifact_publish_status::adopted);
    CHECK(f.ledger.snapshot().live_ops > 0);

    f.catalog.reset();
    const auto snapshot = f.ledger.snapshot();
    CHECK(snapshot.live_ops == 0);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::unit_version_payload,
        f.bindings[0].domain,
        llama_cache_acct_measure::resident_allocated).value == 0);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_reference_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 0);
}

static void test_catalog_dedup_race() {
    catalog_fixture f;
    const auto completions = f.completions();
    std::array<llama_vbr_artifact_publish_result, 2> results;
    std::thread a([&] {
        results[0] =
            f.catalog->publish(f.package, completions, f.budget);
    });
    std::thread b([&] {
        results[1] =
            f.catalog->publish(f.package, completions, f.budget);
    });
    a.join();
    b.join();
    const bool first_published =
        results[0].status ==
            llama_vbr_artifact_publish_status::published;
    const bool second_published =
        results[1].status ==
            llama_vbr_artifact_publish_status::published;
    CHECK(first_published != second_published);
    CHECK((results[0].status ==
               llama_vbr_artifact_publish_status::adopted) !=
          (results[1].status ==
               llama_vbr_artifact_publish_status::adopted));
    const auto state = f.catalog->snapshot();
    CHECK(state.blobs == 1 && state.stashes == 1 &&
          state.references == 2);
    CHECK(state.published == 1 && state.adopted == 1);
    CHECK(f.catalog->retire(results[0].reference_artifact) == vbr_artifact_retire_status::retired);
    CHECK(f.catalog->retire(results[1].reference_artifact) == vbr_artifact_retire_status::retired);
}

static void test_catalog_full_id_interning_and_stash_dedup() {
    catalog_fixture f;
    const auto first =
        f.catalog->publish(f.package, f.completions(), f.budget);
    CHECK(first.status ==
          llama_vbr_artifact_publish_status::published);

    fixture_storage changed_storage;
    changed_storage.payload0.bytes[0] ^= 1;
    auto changed = make_package(changed_storage);
    const auto second = f.catalog->publish(changed, {
        { 0, 1, true,  true, changed_storage.stash1.bytes },
        { 0, 0, false, true, changed_storage.payload0.bytes },
        { 0, 0, true,  true, changed_storage.stash0.bytes },
        { 0, 1, false, true, changed_storage.payload1.bytes },
    }, f.budget);
    CHECK(second.status ==
          llama_vbr_artifact_publish_status::published);
    CHECK(second.reference_artifact != first.reference_artifact);
    CHECK(second.unit_content != first.unit_content);
    CHECK(second.reference_lineage != first.reference_lineage);
    const auto state = f.catalog->snapshot();
    CHECK(state.blobs == 2);
    CHECK(state.stashes == 1);
    CHECK(state.references == 2);
    const auto accounting = f.ledger.snapshot();
    for (const auto & binding : f.bindings) {
        CHECK(catalog_cell(
            accounting,
            llama_cache_acct_category::clean_stash_payload,
            binding.domain,
            llama_cache_acct_measure::resident_allocated).value == 8);
    }
    CHECK(f.catalog->retire(first.reference_artifact) == vbr_artifact_retire_status::retired);
    CHECK(f.catalog->retire(second.reference_artifact) == vbr_artifact_retire_status::retired);
}

static void test_catalog_capacity_sequential_and_temporaries() {
    catalog_fixture f;
    for (auto & device : f.budget.devices) {
        device.physical_total = 28;
        device.physical_free = 28;
        device.configured_cache_cap = 28;
        device.cache_cap_state =
            llama_cache_budget_capacity_state::known;
    }
    const auto first =
        f.catalog->publish(f.package, f.completions(), f.budget);
    CHECK(first.status ==
          llama_vbr_artifact_publish_status::published);

    fixture_storage changed_storage;
    changed_storage.payload0.bytes[0] ^= 1;
    auto changed = make_package(changed_storage);
    const auto refused =
        f.catalog->publish(changed, {
            { 0, 0, false, true, changed_storage.payload0.bytes },
            { 0, 1, false, true, changed_storage.payload1.bytes },
            { 0, 0, true, true, changed_storage.stash0.bytes },
            { 0, 1, true, true, changed_storage.stash1.bytes },
        }, f.budget);
    CHECK(refused.status ==
          llama_vbr_artifact_publish_status::admission_refused);
    CHECK(f.catalog->snapshot().references == 1);
    CHECK(f.ledger.snapshot().live_ops > 0);

    llama_cache_budget_config generous = f.budget;
    for (auto & device : generous.devices) {
        device.physical_total = 100;
        device.physical_free = 70;
        device.configured_cache_cap = 100;
    }
    const auto domain = f.bindings[0].domain;
    llama_cache_authority_request staging;
    staging.category =
        llama_cache_acct_category::transfer_staging;
    staging.domain = domain;
    staging.expected_logical = 40;
    staging.expected_resident = 40;
    auto held = llama_cache_admit_reservation(
        f.ledger, generous, staging);
    CHECK(held.status == llama_cache_admission_status::admitted);

    llama_cache_authority_request workspace = staging;
    workspace.category =
        llama_cache_acct_category::codec_workspace;
    workspace.expected_logical = 70;
    workspace.expected_resident = 70;
    auto blocked = llama_cache_admit_reservation(
        f.ledger, generous, workspace);
    CHECK(blocked.status ==
          llama_cache_admission_status::exceeds_budget);
    CHECK(!blocked.claim.has_op());
    CHECK(f.catalog->retire(first.reference_artifact) == vbr_artifact_retire_status::retired);
}

static void test_catalog_package_lease_and_reference_placement() {
    catalog_fixture f;
    const auto first = f.catalog->publish(
        f.package, f.completions(), f.budget);
    CHECK(first.status == llama_vbr_artifact_publish_status::published);

    auto alternate = f.package;
    alternate.manifest.generation.controllers[0].streams[0]
        .pages[0].covered_mask[0] = 0x5;
    alternate.manifest.stream_placements[0].cells[1].physical_cell = 2;
    alternate.manifest.unit_references[0].stash_reference
        .covered_sink_pages[0].covered_mask[0] = 0x5;
    alternate.manifest.manifest_digest = {};
    const auto second = f.catalog->publish(
        alternate, f.completions(), f.budget);
    CHECK(second.status == llama_vbr_artifact_publish_status::adopted);
    CHECK(f.catalog->snapshot().blobs == 1);
    CHECK(f.catalog->snapshot().references == 2);

    vbr_artifact_package_view first_view;
    vbr_artifact_package_view second_view;
    CHECK(f.catalog->resolve_reference(first.reference_artifact, first_view) ==
          vbr_artifact_resolve_status::ok);
    CHECK(f.catalog->resolve_reference(second.reference_artifact, second_view) ==
          vbr_artifact_resolve_status::ok);
    CHECK(first_view.reference_artifact() == first.reference_artifact);
    CHECK(first_view.units().size() == 1);
    CHECK(first_view.units()[0].payload_shards.size() == 2);
    CHECK(first_view.manifest().stream_placements[0].cells[1].physical_cell == 1);
    CHECK(second_view.manifest().stream_placements[0].cells[1].physical_cell == 2);
    CHECK(first_view.units()[0].unit_version_id ==
          second_view.units()[0].unit_version_id);
    auto fallback_proof = server_cache_vbr_fallback_proof_for_test(
        std::move(first_view));
    CHECK(fallback_proof.available());
    CHECK(f.catalog->retire(first.reference_artifact) ==
          vbr_artifact_retire_status::busy);
    fallback_proof = {};
    CHECK(f.catalog->resolve_reference(first.reference_artifact, first_view) ==
          vbr_artifact_resolve_status::ok);
    const auto live_before_busy = f.ledger.snapshot().live_ops;
    CHECK(f.catalog->retire(first.reference_artifact) ==
          vbr_artifact_retire_status::busy);
    CHECK(f.ledger.snapshot().live_ops == live_before_busy);

    vbr_artifact_package_view moved = std::move(first_view);
    CHECK(!first_view);
    CHECK(moved);
    moved.reset();
    CHECK(f.catalog->retire(first.reference_artifact) ==
          vbr_artifact_retire_status::retired);
    second_view.reset();
    CHECK(f.catalog->retire(second.reference_artifact) ==
          vbr_artifact_retire_status::retired);

    catalog_fixture raced;
    const auto published = raced.catalog->publish(
        raced.package, raced.completions(), raced.budget);
    CHECK(published.status == llama_vbr_artifact_publish_status::published);
    std::atomic<bool> go { false };
    std::atomic<bool> release_view { false };
    vbr_artifact_resolve_status resolve_status =
        vbr_artifact_resolve_status::internal_error;
    vbr_artifact_retire_status retire_status =
        vbr_artifact_retire_status::internal_error;
    std::thread resolver([&] {
        while (!go.load(std::memory_order_acquire)) {}
        vbr_artifact_package_view view;
        resolve_status = raced.catalog->resolve_reference(
            published.reference_artifact, view);
        while (view && !release_view.load(std::memory_order_acquire)) {}
    });
    std::thread retiree([&] {
        while (!go.load(std::memory_order_acquire)) {}
        retire_status = raced.catalog->retire(
            published.reference_artifact);
    });
    go.store(true, std::memory_order_release);
    retiree.join();
    release_view.store(true, std::memory_order_release);
    resolver.join();
    CHECK((resolve_status == vbr_artifact_resolve_status::ok &&
           retire_status == vbr_artifact_retire_status::busy) ||
          (resolve_status == vbr_artifact_resolve_status::not_found &&
           retire_status == vbr_artifact_retire_status::retired));
    if (retire_status == vbr_artifact_retire_status::busy) {
        CHECK(raced.catalog->retire(published.reference_artifact) ==
              vbr_artifact_retire_status::retired);
    }
}

static void test_prompt_cache_vbr_payload_fanout_lifetime() {
    catalog_fixture f;
    auto & package = f.package;
    auto second = package.unit_blobs.front();
    second.descriptor.logical_unit_id = 1;
    second.unit_version_id = {};
    second.payload_digest = {};
    package.unit_blobs.push_back(second);
    package.manifest.generation.controllers[0].units.push_back(
        package.manifest.generation.controllers[0].units.front());
    auto second_reference = package.manifest.unit_references.front();
    second_reference.logical_unit_id = 1;
    second_reference.unit_version_id = {};
    second_reference.payload_digest = {};
    package.manifest.unit_references.push_back(second_reference);
    for (auto & row : package.manifest.accounting) {
        if (row.role == vbr_artifact_accounting_role::unit_payload ||
            row.role ==
                vbr_artifact_accounting_role::clean_stash_payload) {
            row.logical_bytes *= 2;
            row.resident_bytes *= 2;
        }
    }
    package.manifest.manifest_digest = {};
    package.manifest.capture_generation_id = {};
    package.manifest.consistency = {};
    CHECK(f.catalog->configure_accounting(package));

    vbr_capture_stream_status status;
    auto build = f.catalog->begin_capture(
        package, f.budget, {}, status);
    CHECK(build);
    if (!build) {
        return;
    }
    const auto completions = f.completions();
    for (size_t unit_index = 0; unit_index < 2; ++unit_index) {
        auto unit = build->begin_unit(unit_index, status);
        CHECK(unit);
        if (!unit) {
            return;
        }
        for (const auto & completion : completions) {
            auto copy = completion;
            copy.unit_index = unit_index;
            CHECK(unit->accept_verified_segment(
                      verified_segment(copy, uint8_t(2 + unit_index))) ==
                  vbr_capture_stream_status::ok);
        }
        CHECK(unit->seal_unit() == vbr_capture_stream_status::ok);
    }
    const auto streamed = build->publish_reference();
    CHECK(streamed.status == vbr_capture_stream_status::ok);
    CHECK(streamed.reference_artifact.v != 0);
    build.reset();
    const auto published_artifact = streamed.reference_artifact;

    vbr_artifact_package_view view;
    CHECK(f.catalog->resolve_reference(
              published_artifact, view) ==
          vbr_artifact_resolve_status::ok);

    std::vector<vbr_artifact_allocation_view> rows;
    rows.insert(rows.end(), view.reference_allocations().begin(),
                view.reference_allocations().end());
    for (const auto & unit : view.units()) {
        rows.insert(rows.end(), unit.payload_allocations.begin(),
                    unit.payload_allocations.end());
        rows.insert(rows.end(), unit.stash_allocations.begin(),
                    unit.stash_allocations.end());
    }
    uint64_t naive_logical = 0;
    uint64_t naive_resident = 0;
    for (const auto & row : rows) {
        naive_logical += row.logical;
        naive_resident += row.resident;
    }
    std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
        return a.allocation.v < b.allocation.v;
    });
    uint64_t expected_logical = 0;
    uint64_t expected_resident = 0;
    size_t expected_allocations = 0;
    uint64_t previous_allocation = 0;
    for (const auto & row : rows) {
        if (!row.allocation || row.allocation.v == previous_allocation) {
            continue;
        }
        expected_logical += row.logical;
        expected_resident += row.resident;
        ++expected_allocations;
        previous_allocation = row.allocation.v;
    }
    CHECK(expected_allocations == rows.size());
    CHECK(expected_logical == naive_logical);
    CHECK(expected_resident == naive_resident);

    // Exercise the same production accounting kernel with a physical alias.
    // Catalog references currently expose unique rows, while future composed
    // variants may repeat one allocation. Naive per-view summation must not
    // double-charge that backing allocation.
    auto aliased_rows = rows;
    const auto duplicate = std::find_if(
        rows.begin(), rows.end(), [](const auto & row) {
            return row.category ==
                llama_cache_acct_category::clean_stash_payload;
        });
    CHECK(duplicate != rows.end());
    if (duplicate != rows.end()) {
        aliased_rows.push_back(*duplicate);
    }
    server_prompt_cache_vbr_accounting_summary aliased_summary;
    CHECK(server_prompt_cache_summarize_vbr_allocations(
        std::move(aliased_rows), aliased_summary));
    CHECK(aliased_summary.allocation_count == expected_allocations);
    CHECK(aliased_summary.logical_bytes == expected_logical);
    CHECK(aliased_summary.resident_bytes == expected_resident);

    if (duplicate != rows.end()) {
        auto conflicting = *duplicate;
        ++conflicting.resident;
        server_prompt_cache_vbr_accounting_summary refused {
            1, 1, 1,
        };
        CHECK(!server_prompt_cache_summarize_vbr_allocations(
            { *duplicate, conflicting }, refused));
        CHECK(refused.logical_bytes == 0);
        CHECK(refused.resident_bytes == 0);
        CHECK(refused.allocation_count == 0);

        auto maximum = *duplicate;
        auto one_more = *duplicate;
        maximum.allocation.v = UINT64_MAX - 1;
        maximum.logical = UINT64_MAX;
        maximum.resident = UINT64_MAX;
        one_more.allocation.v = UINT64_MAX;
        one_more.logical = 1;
        one_more.resident = 1;
        refused = { 1, 1, 1 };
        CHECK(!server_prompt_cache_summarize_vbr_allocations(
            { maximum, one_more }, refused));
        CHECK(refused.logical_bytes == 0);
        CHECK(refused.resident_bytes == 0);
        CHECK(refused.allocation_count == 0);
    }

    auto owner = server_prompt_cache_vbr_payload::adopt(std::move(view));
    CHECK(owner);
    if (!owner) {
        return;
    }
    CHECK(!view);
    CHECK(owner->reference_artifact() == published_artifact);
    CHECK(owner->logical_bytes() == expected_logical);
    CHECK(owner->resident_bytes() == expected_resident);
    CHECK(owner->allocation_count() == expected_allocations);
    CHECK(owner->package().units().size() == 2);
    CHECK(owner->accounted_by(&f.ledger));
    llama_cache_acct_ledger unrelated_ledger;
    CHECK(!owner->accounted_by(&unrelated_ledger));

    auto payload = server_prompt_cache_payload::from_vbr(owner);
    CHECK(payload.kind() ==
          server_prompt_cache_payload_kind::vbr_artifact);
    CHECK(payload.valid());
    CHECK(payload.publishable());
    CHECK(!payload.restorable());
    CHECK(payload.size() == expected_resident);
    CHECK(payload.vbr_artifact() == owner.get());

    std::vector<server_prompt_cache_payload> aliases(8, payload);
    for (const auto & alias : aliases) {
        CHECK(alias.same_storage(payload));
        CHECK(alias.vbr_artifact() == owner.get());
    }

    // Numeric artifact IDs are catalog-local. Independently retained owners
    // must not become exact-redundancy evidence merely because each catalog
    // issued its first reference.
    catalog_fixture identity_a;
    catalog_fixture identity_b;
    const auto identity_a_published = identity_a.catalog->publish(
        identity_a.package, identity_a.completions(), identity_a.budget);
    const auto identity_b_published = identity_b.catalog->publish(
        identity_b.package, identity_b.completions(), identity_b.budget);
    CHECK(identity_a_published.status ==
          llama_vbr_artifact_publish_status::published);
    CHECK(identity_b_published.status ==
          llama_vbr_artifact_publish_status::published);
    CHECK(identity_a_published.reference_artifact ==
          identity_b_published.reference_artifact);
    vbr_artifact_package_view identity_a_view;
    vbr_artifact_package_view identity_b_view;
    CHECK(identity_a.catalog->resolve_reference(
              identity_a_published.reference_artifact,
              identity_a_view) == vbr_artifact_resolve_status::ok);
    CHECK(identity_b.catalog->resolve_reference(
              identity_b_published.reference_artifact,
              identity_b_view) == vbr_artifact_resolve_status::ok);
    auto identity_a_owner = server_prompt_cache_vbr_payload::adopt(
        std::move(identity_a_view));
    auto identity_b_owner = server_prompt_cache_vbr_payload::adopt(
        std::move(identity_b_view));
    CHECK(identity_a_owner && identity_b_owner);
    auto identity_a_payload =
        server_prompt_cache_payload::from_vbr(identity_a_owner);
    auto identity_b_payload =
        server_prompt_cache_payload::from_vbr(identity_b_owner);
    CHECK(!identity_a_payload.same_storage(identity_b_payload));
    identity_a_payload = {};
    identity_b_payload = {};
    identity_a_owner.reset();
    identity_b_owner.reset();
    CHECK(identity_a.catalog->retire(
              identity_a_published.reference_artifact) ==
          vbr_artifact_retire_status::retired);
    CHECK(identity_b.catalog->retire(
              identity_b_published.reference_artifact) ==
          vbr_artifact_retire_status::retired);

    const uint64_t live_ops = f.ledger.snapshot().live_ops;
    owner.reset();
    payload = {};
    CHECK(f.catalog->retire(published_artifact) ==
          vbr_artifact_retire_status::busy);
    while (aliases.size() > 1) {
        aliases.pop_back();
        CHECK(f.catalog->retire(published_artifact) ==
              vbr_artifact_retire_status::busy);
        CHECK(f.ledger.snapshot().live_ops == live_ops);
    }
    aliases.clear();
    CHECK(f.catalog->retire(published_artifact) ==
          vbr_artifact_retire_status::retired);
    CHECK(f.ledger.snapshot().live_ops == 0);

    vbr_artifact_package_view empty;
    CHECK(!server_prompt_cache_vbr_payload::adopt(std::move(empty)));
}

static void test_prompt_cache_vbr_same_frontier_variants() {
    catalog_fixture f;
    const auto compact_reference = f.catalog->publish(
        f.package, f.completions(), f.budget);

    const auto with_quality = [](const vbr_artifact_package & source,
                                 ggml_type type) {
        auto result = source;
        auto & descriptor = result.unit_blobs[0].descriptor;
        descriptor.repr_gen++;
        descriptor.current_type = type;
        descriptor.last_source_type = type == GGML_TYPE_TURBO8_0
            ? GGML_TYPE_F16
            : GGML_TYPE_TURBO4_0;
        descriptor.promote_hops = type == GGML_TYPE_TURBO8_0 ? 0 : 2;
        descriptor.last_transition = type == GGML_TYPE_TURBO8_0
            ? vbr_repr_transition::degrade_f16_to_t8_admitted
            : vbr_repr_transition::degrade_other;
        descriptor.representation.source_loss_history =
            type == GGML_TYPE_TURBO8_0 ? 0 : 2;
        auto & generation =
            result.manifest.generation.controllers[0].units[0];
        generation.repr_gen = descriptor.repr_gen;
        generation.current_type = type;
        generation.last_source_type = descriptor.last_source_type;
        generation.domain = vbr_downward_tier_domain(type);
        generation.promote_hops = descriptor.promote_hops;
        generation.last_transition = descriptor.last_transition;
        auto & reference = result.manifest.unit_references[0];
        reference.repr_gen = descriptor.repr_gen;
        const ggml_type types[] = { type };
        result.manifest.controller_policy[0]
            .current_type_vector_digest =
                vbr_type_vector_digest(types, 1);
        result.manifest.manifest_digest = {};
        result.manifest.capture_generation_id = {};
        result.manifest.consistency = {};
        return result;
    };

    auto anchor_package = with_quality(
        f.package, GGML_TYPE_TURBO8_0);
    CHECK(f.catalog->configure_accounting(anchor_package));
    const auto anchor_reference = f.catalog->publish(
        anchor_package, f.completions(), f.budget);
    CHECK(compact_reference.status ==
          llama_vbr_artifact_publish_status::published);
    CHECK(anchor_reference.status ==
          llama_vbr_artifact_publish_status::published);

    vbr_artifact_package_view compact_view;
    vbr_artifact_package_view anchor_view;
    CHECK(f.catalog->resolve_reference(
              compact_reference.reference_artifact, compact_view) ==
          vbr_artifact_resolve_status::ok);
    CHECK(f.catalog->resolve_reference(
              anchor_reference.reference_artifact, anchor_view) ==
          vbr_artifact_resolve_status::ok);
    auto compact =
        server_prompt_cache_vbr_payload::adopt(std::move(compact_view));
    auto anchor =
        server_prompt_cache_vbr_payload::adopt(std::move(anchor_view));
    CHECK(compact && anchor);
    if (!compact || !anchor) {
        return;
    }

    auto variants = server_prompt_cache_vbr_variant_set::create(
        compact, anchor);
    CHECK(variants);
    if (!variants) {
        return;
    }
    CHECK(variants->compact_current() == compact);
    CHECK(variants->quality_anchor() == anchor);

    // Independently flatten and exact-dedup the two sealed packages. This
    // catches both double charging shared rows and accidentally omitting an
    // anchor-exclusive allocation from the union.
    std::vector<vbr_artifact_allocation_view> expected_rows;
    const auto append_expected = [&](const auto & package) {
        expected_rows.insert(
            expected_rows.end(), package.reference_allocations().begin(),
            package.reference_allocations().end());
        for (const auto & unit : package.units()) {
            expected_rows.insert(
                expected_rows.end(), unit.payload_allocations.begin(),
                unit.payload_allocations.end());
            expected_rows.insert(
                expected_rows.end(), unit.stash_allocations.begin(),
                unit.stash_allocations.end());
        }
    };
    append_expected(compact->package());
    const size_t compact_row_count = expected_rows.size();
    append_expected(anchor->package());
    bool anchor_exclusive = false;
    for (size_t i = compact_row_count; i < expected_rows.size(); ++i) {
        anchor_exclusive |= std::none_of(
            expected_rows.begin(),
            expected_rows.begin() + compact_row_count,
            [&](const auto & compact_row) {
                return compact_row.allocation ==
                    expected_rows[i].allocation;
            });
    }
    std::sort(
        expected_rows.begin(), expected_rows.end(),
        [](const auto & lhs, const auto & rhs) {
            return lhs.allocation.v < rhs.allocation.v;
        });
    uint64_t expected_logical = 0;
    uint64_t expected_resident = 0;
    size_t expected_allocations = 0;
    llama_cache_acct_alloc_id previous;
    for (const auto & row : expected_rows) {
        if (!row.allocation || row.allocation == previous) {
            continue;
        }
        previous = row.allocation;
        expected_logical += row.logical;
        expected_resident += row.resident;
        expected_allocations++;
    }
    CHECK(anchor_exclusive);
    CHECK(expected_allocations < expected_rows.size());
    CHECK(variants->allocation_count() == expected_allocations);
    CHECK(variants->logical_bytes() == expected_logical);
    CHECK(variants->resident_bytes() == expected_resident);

    auto typed = server_prompt_cache_payload::from_vbr_variants(variants);
    CHECK(typed.valid());
    CHECK(typed.publishable());
    CHECK(!typed.restorable());
    CHECK(typed.vbr_variants() == variants.get());
    CHECK(typed.vbr_artifact() == compact.get());
    CHECK(typed.size() == variants->resident_bytes());
    CHECK(typed.accounted_by(&f.ledger));
    auto alias = typed;
    CHECK(alias.same_storage(typed));
    auto independently_wrapped =
        server_prompt_cache_payload::from_vbr(compact);
    auto independently_wrapped_again =
        server_prompt_cache_payload::from_vbr(compact);
    CHECK(independently_wrapped.same_storage(
        independently_wrapped_again));
    auto same_variants = server_prompt_cache_vbr_variant_set::create(
        compact, anchor);
    CHECK(same_variants);
    auto independently_bundled =
        server_prompt_cache_payload::from_vbr_variants(same_variants);
    CHECK(typed.same_storage(independently_bundled));
    CHECK(!typed.same_storage(independently_wrapped));

    // Equal, lower-quality, or reversed roles are not anchors.
    const auto equal_reference = f.catalog->publish(
        f.package, f.completions(), f.budget);
    CHECK(equal_reference.status ==
          llama_vbr_artifact_publish_status::adopted);
    vbr_artifact_package_view equal_view;
    CHECK(f.catalog->resolve_reference(
              equal_reference.reference_artifact, equal_view) ==
          vbr_artifact_resolve_status::ok);
    auto equal_owner =
        server_prompt_cache_vbr_payload::adopt(std::move(equal_view));
    CHECK(equal_owner);
    CHECK(!server_prompt_cache_vbr_variant_set::create(
        compact, equal_owner));
    CHECK(!server_prompt_cache_vbr_variant_set::create(
        anchor, compact));

    auto lower_package = with_quality(
        f.package, GGML_TYPE_TURBO3_TCQ);
    CHECK(f.catalog->configure_accounting(lower_package));
    const auto lower_reference = f.catalog->publish(
        lower_package, f.completions(), f.budget);
    CHECK(lower_reference.status ==
          llama_vbr_artifact_publish_status::published);
    vbr_artifact_package_view lower_view;
    CHECK(f.catalog->resolve_reference(
              lower_reference.reference_artifact, lower_view) ==
          vbr_artifact_resolve_status::ok);
    auto lower_owner =
        server_prompt_cache_vbr_payload::adopt(std::move(lower_view));
    CHECK(lower_owner);
    CHECK(!server_prompt_cache_vbr_variant_set::create(
        compact, lower_owner));

    // A nominally higher tier reconstructed through a worse lossy history is
    // not a quality anchor.
    auto lossy_anchor_package = anchor_package;
    auto & lossy_descriptor =
        lossy_anchor_package.unit_blobs[0].descriptor;
    lossy_descriptor.last_source_type = GGML_TYPE_TURBO4_0;
    lossy_descriptor.promote_hops = 2;
    lossy_descriptor.last_transition = vbr_repr_transition::promote;
    lossy_descriptor.representation.source_loss_history = 2;
    lossy_descriptor.representation.checkpoint_codec_hops = 1;
    auto & lossy_generation = lossy_anchor_package.manifest.generation
        .controllers[0].units[0];
    lossy_generation.last_source_type = lossy_descriptor.last_source_type;
    lossy_generation.promote_hops = lossy_descriptor.promote_hops;
    lossy_generation.last_transition = lossy_descriptor.last_transition;
    lossy_anchor_package.manifest.manifest_digest = {};
    lossy_anchor_package.manifest.capture_generation_id = {};
    lossy_anchor_package.manifest.consistency = {};
    CHECK(f.catalog->configure_accounting(lossy_anchor_package));
    const auto lossy_anchor_reference = f.catalog->publish(
        lossy_anchor_package, f.completions(), f.budget);
    CHECK(lossy_anchor_reference.status ==
          llama_vbr_artifact_publish_status::published);
    vbr_artifact_package_view lossy_anchor_view;
    CHECK(f.catalog->resolve_reference(
              lossy_anchor_reference.reference_artifact,
              lossy_anchor_view) == vbr_artifact_resolve_status::ok);
    auto lossy_anchor = server_prompt_cache_vbr_payload::adopt(
        std::move(lossy_anchor_view));
    CHECK(lossy_anchor);
    CHECK(!server_prompt_cache_vbr_variant_set::create(
        compact, lossy_anchor));

    // A multi-unit candidate may not trade quality between units: improving
    // one while degrading another is incomparable, not an anchor.
    catalog_fixture mixed;
    auto compact_pair_package = mixed.package;
    auto second = compact_pair_package.unit_blobs.front();
    second.descriptor.logical_unit_id = 1;
    second.descriptor.clean_stash_state =
        vbr_artifact_clean_stash_state::absent_at_source;
    second.descriptor.clean_stash = {};
    second.unit_version_id = {};
    second.payload_digest = {};
    compact_pair_package.unit_blobs.push_back(second);
    compact_pair_package.manifest.generation.controllers[0].units.push_back(
        compact_pair_package.manifest.generation.controllers[0].units.front());
    auto second_reference =
        compact_pair_package.manifest.unit_references.front();
    second_reference.logical_unit_id = 1;
    second_reference.unit_version_id = {};
    second_reference.payload_digest = {};
    second_reference.has_stash_reference = false;
    second_reference.stash_reference = {};
    compact_pair_package.manifest.unit_references.push_back(second_reference);
    for (auto & row : compact_pair_package.manifest.accounting) {
        if (row.role == vbr_artifact_accounting_role::unit_payload) {
            row.logical_bytes *= 2;
            row.resident_bytes *= 2;
        }
    }
    compact_pair_package.manifest.manifest_digest = {};
    compact_pair_package.manifest.capture_generation_id = {};
    compact_pair_package.manifest.consistency = {};

    auto high_pair_package = compact_pair_package;
    for (size_t i = 0; i < 2; ++i) {
        auto & descriptor = high_pair_package.unit_blobs[i].descriptor;
        descriptor.repr_gen++;
        descriptor.current_type = GGML_TYPE_TURBO8_0;
        descriptor.last_source_type = GGML_TYPE_F16;
        descriptor.promote_hops = 0;
        descriptor.last_transition =
            vbr_repr_transition::degrade_f16_to_t8_admitted;
        descriptor.representation.source_loss_history = 0;
        auto & generation = high_pair_package.manifest.generation
            .controllers[0].units[i];
        generation.repr_gen = descriptor.repr_gen;
        generation.current_type = descriptor.current_type;
        generation.last_source_type = descriptor.last_source_type;
        generation.domain = vbr_repr_domain::full;
        generation.promote_hops = descriptor.promote_hops;
        generation.last_transition = descriptor.last_transition;
        high_pair_package.manifest.unit_references[i].repr_gen =
            descriptor.repr_gen;
    }
    const ggml_type high_types[] = {
        GGML_TYPE_TURBO8_0,
        GGML_TYPE_TURBO8_0,
    };
    high_pair_package.manifest.controller_policy[0]
        .current_type_vector_digest =
            vbr_type_vector_digest(high_types, 2);
    high_pair_package.manifest.manifest_digest = {};
    high_pair_package.manifest.capture_generation_id = {};
    high_pair_package.manifest.consistency = {};
    // Unit blob order is producer-local; logical tuple matching must make it
    // irrelevant to the variant relation.
    std::reverse(
        high_pair_package.unit_blobs.begin(),
        high_pair_package.unit_blobs.end());

    auto mixed_anchor_package = compact_pair_package;
    const ggml_type mixed_types[] = {
        GGML_TYPE_TURBO8_0,
        GGML_TYPE_TURBO3_TCQ,
    };
    for (size_t i = 0; i < 2; ++i) {
        auto & descriptor = mixed_anchor_package.unit_blobs[i].descriptor;
        descriptor.repr_gen++;
        descriptor.current_type = mixed_types[i];
        descriptor.last_source_type = mixed_types[i] ==
                GGML_TYPE_TURBO8_0
            ? GGML_TYPE_F16
            : GGML_TYPE_TURBO4_0;
        descriptor.promote_hops = mixed_types[i] ==
                GGML_TYPE_TURBO8_0 ? 0 : 2;
        descriptor.last_transition = mixed_types[i] ==
                GGML_TYPE_TURBO8_0
            ? vbr_repr_transition::degrade_f16_to_t8_admitted
            : vbr_repr_transition::degrade_other;
        descriptor.representation.source_loss_history =
            mixed_types[i] == GGML_TYPE_TURBO8_0 ? 0 : 2;
        auto & generation = mixed_anchor_package.manifest.generation
            .controllers[0].units[i];
        generation.repr_gen = descriptor.repr_gen;
        generation.current_type = descriptor.current_type;
        generation.last_source_type = descriptor.last_source_type;
        generation.domain = vbr_downward_tier_domain(mixed_types[i]);
        generation.promote_hops = descriptor.promote_hops;
        generation.last_transition = descriptor.last_transition;
        mixed_anchor_package.manifest.unit_references[i].repr_gen =
            descriptor.repr_gen;
    }
    mixed_anchor_package.manifest.controller_policy[0]
        .current_type_vector_digest =
            vbr_type_vector_digest(mixed_types, 2);
    mixed_anchor_package.manifest.manifest_digest = {};
    mixed_anchor_package.manifest.capture_generation_id = {};
    mixed_anchor_package.manifest.consistency = {};

    const auto publish_pair = [&](vbr_artifact_package & package) {
        CHECK(mixed.catalog->configure_accounting(package));
        vbr_capture_stream_status status;
        auto build = mixed.catalog->begin_capture(
            package, mixed.budget, {}, status);
        CHECK(build && status == vbr_capture_stream_status::ok);
        if (!build) {
            return llama_cache_acct_artifact_id {};
        }
        for (size_t unit_index = 0; unit_index < 2; ++unit_index) {
            auto unit = build->begin_unit(unit_index, status);
            CHECK(unit && status == vbr_capture_stream_status::ok);
            if (!unit) {
                return llama_cache_acct_artifact_id {};
            }
            for (const auto & completion : mixed.completions()) {
                if (completion.clean_stash &&
                    package.unit_blobs[unit_index].descriptor
                        .clean_stash_state !=
                        vbr_artifact_clean_stash_state::present) {
                    continue;
                }
                auto copy = completion;
                copy.unit_index = unit_index;
                CHECK(unit->accept_verified_segment(
                          verified_segment(copy, 2)) ==
                      vbr_capture_stream_status::ok);
            }
            CHECK(unit->seal_unit() == vbr_capture_stream_status::ok);
        }
        const auto published = build->publish_reference();
        CHECK(published.status == vbr_capture_stream_status::ok);
        return published.reference_artifact;
    };
    const auto compact_pair_reference =
        publish_pair(compact_pair_package);
    const auto high_pair_reference =
        publish_pair(high_pair_package);
    const auto mixed_anchor_reference =
        publish_pair(mixed_anchor_package);
    CHECK(compact_pair_reference.v != 0 &&
          high_pair_reference.v != 0 &&
          mixed_anchor_reference.v != 0);
    vbr_artifact_package_view compact_pair_view;
    vbr_artifact_package_view high_pair_view;
    vbr_artifact_package_view mixed_anchor_view;
    CHECK(mixed.catalog->resolve_reference(
              compact_pair_reference, compact_pair_view) ==
          vbr_artifact_resolve_status::ok);
    CHECK(mixed.catalog->resolve_reference(
              high_pair_reference, high_pair_view) ==
          vbr_artifact_resolve_status::ok);
    CHECK(mixed.catalog->resolve_reference(
              mixed_anchor_reference, mixed_anchor_view) ==
          vbr_artifact_resolve_status::ok);
    auto compact_pair = server_prompt_cache_vbr_payload::adopt(
        std::move(compact_pair_view));
    auto high_pair = server_prompt_cache_vbr_payload::adopt(
        std::move(high_pair_view));
    auto mixed_anchor = server_prompt_cache_vbr_payload::adopt(
        std::move(mixed_anchor_view));
    CHECK(compact_pair && high_pair && mixed_anchor);
    CHECK(server_prompt_cache_vbr_variant_set::create(
        compact_pair, high_pair));
    CHECK(!server_prompt_cache_vbr_variant_set::create(
        compact_pair, mixed_anchor));

    // Same semantic frontier is necessary but not sufficient: allocation IDs
    // are catalog-local, so one variant set may not span catalog namespaces.
    catalog_fixture other;
    auto other_anchor_package = with_quality(
        other.package, GGML_TYPE_TURBO8_0);
    CHECK(other.catalog->configure_accounting(other_anchor_package));
    const auto other_reference = other.catalog->publish(
        other_anchor_package, other.completions(), other.budget);
    CHECK(other_reference.status ==
          llama_vbr_artifact_publish_status::published);
    vbr_artifact_package_view other_view;
    CHECK(other.catalog->resolve_reference(
              other_reference.reference_artifact, other_view) ==
          vbr_artifact_resolve_status::ok);
    auto other_owner =
        server_prompt_cache_vbr_payload::adopt(std::move(other_view));
    CHECK(other_owner);
    CHECK(!server_prompt_cache_vbr_variant_set::create(
        compact, other_owner));

    // A reference from the same catalog but a different authenticated token
    // frontier cannot be mislabeled as a quality variant.
    auto different = anchor_package;
    different.manifest.token_block.tokens[0] += 1;
    different.manifest.token_block.digest = {};
    different.manifest.manifest_digest = {};
    different.manifest.capture_generation_id = {};
    different.manifest.consistency = {};
    const auto different_reference = f.catalog->publish(
        different, f.completions(), f.budget);
    CHECK(different_reference.status ==
              llama_vbr_artifact_publish_status::adopted ||
          different_reference.status ==
              llama_vbr_artifact_publish_status::published);
    vbr_artifact_package_view different_view;
    CHECK(f.catalog->resolve_reference(
              different_reference.reference_artifact, different_view) ==
          vbr_artifact_resolve_status::ok);
    auto different_owner =
        server_prompt_cache_vbr_payload::adopt(std::move(different_view));
    CHECK(different_owner);
    CHECK(!server_prompt_cache_vbr_variant_set::create(
        compact, different_owner));

    alias = {};
    typed = {};
    independently_wrapped = {};
    independently_wrapped_again = {};
    independently_bundled = {};
    same_variants.reset();
    variants.reset();
    compact.reset();
    anchor.reset();
    equal_owner.reset();
    lower_owner.reset();
    lossy_anchor.reset();
    different_owner.reset();
    other_owner.reset();
    compact_pair.reset();
    high_pair.reset();
    mixed_anchor.reset();
    CHECK(f.catalog->retire(compact_reference.reference_artifact) ==
          vbr_artifact_retire_status::retired);
    CHECK(f.catalog->retire(anchor_reference.reference_artifact) ==
          vbr_artifact_retire_status::retired);
    CHECK(f.catalog->retire(equal_reference.reference_artifact) ==
          vbr_artifact_retire_status::retired);
    CHECK(f.catalog->retire(lower_reference.reference_artifact) ==
          vbr_artifact_retire_status::retired);
    CHECK(f.catalog->retire(lossy_anchor_reference.reference_artifact) ==
          vbr_artifact_retire_status::retired);
    CHECK(f.catalog->retire(different_reference.reference_artifact) ==
          vbr_artifact_retire_status::retired);
    CHECK(other.catalog->retire(other_reference.reference_artifact) ==
          vbr_artifact_retire_status::retired);
    CHECK(mixed.catalog->retire(compact_pair_reference) ==
          vbr_artifact_retire_status::retired);
    CHECK(mixed.catalog->retire(high_pair_reference) ==
          vbr_artifact_retire_status::retired);
    CHECK(mixed.catalog->retire(mixed_anchor_reference) ==
          vbr_artifact_retire_status::retired);
}

class validator_companion_image final : public vbr_parsed_companion_image {
public:
    validator_companion_image(
            vbr_artifact_companion_kind kind,
            uint32_t format_version) noexcept :
        kind_(kind), format_version_(format_version) {}

    vbr_artifact_companion_kind kind() const noexcept override {
        return kind_;
    }

    uint32_t format_version() const noexcept override {
        return format_version_;
    }

private:
    vbr_artifact_companion_kind kind_;
    uint32_t format_version_;
};

struct validator_serials {
    uint64_t accounting = 0;
    uint64_t policy = 0;
    bool target_stable = true;
    std::array<uint8_t, 32> downward_tree_digest = {};

    static uint64_t read_accounting(const void * context) noexcept {
        return static_cast<const validator_serials *>(context)->accounting;
    }
    static uint64_t read_policy(const void * context) noexcept {
        return static_cast<const validator_serials *>(context)->policy;
    }
    static bool recheck_target(
            const void * context,
            const vbr_target_empty_fingerprint &) noexcept {
        return static_cast<const validator_serials *>(context)->target_stable;
    }
    static bool read_downward_tree(
            const void * context,
            std::array<uint8_t, 32> & output) noexcept {
        output = static_cast<const validator_serials *>(context)
            ->downward_tree_digest;
        return std::any_of(output.begin(), output.end(),
            [](uint8_t value) { return value != 0; });
    }
    static bool parse_companion(
            const void *,
            const vbr_artifact_companion_payload & descriptor,
            const artifact_segment_chain & source,
            const vbr_target_companion_snapshot & target,
            std::unique_ptr<vbr_parsed_companion_image> & output) noexcept {
        if (!target.available || target.target_cookie == nullptr ||
            source.size() != descriptor.payload_bytes) {
            return false;
        }
        output.reset(new (std::nothrow) validator_companion_image(
            descriptor.kind, descriptor.format_version));
        return bool(output);
    }
};

struct validator_fixture {
    catalog_fixture base;
    llama_cache_acct_artifact_id reference_artifact;
    vbr_artifact_package_view view;
    llama_cache_acct_snapshot accounting;
    validator_serials serials;
    vbr_target_validation_snapshot target;
    vbr_adopt_policy policy;

    // stash_case: 0 = exact full prefix, 1 = source-present partial
    // authorization, 2 = tapped/absent at source, 3 = full-domain/absent.
    explicit validator_fixture(
            bool legacy_v1 = false,
            int stash_case = 0,
            bool include_pinned = false)
        : base(true, include_pinned) {
        // Keep the historical artifact golden intact while making this
        // validator fixture's physical authorization fit its one-row unit.
        auto & manifest = base.package.manifest;
        auto & stream = manifest.generation.controllers[0].streams[0];
        auto & placement = manifest.stream_placements[0];
        auto & stash = manifest.unit_references[0].stash_reference;
        if (stash_case == 1) {
            auto & descriptor = base.package.unit_blobs[0].descriptor;
            descriptor.wm_cells = 2;
            for (auto & shard : descriptor.shards) {
                shard.row_count = 2;
                shard.row_bytes = 2;
            }
            manifest.controller_policy[0].wm_cells = 2;
            stash.captured_sink_count = 1;
            stash.covered_sink_pages[0].covered_mask[0] = 2;
        } else {
            manifest.identity.token_count = 1;
            manifest.identity.next_position = 1;
            manifest.token_block.tokens = { 101 };
            stream.computation_frontier = 1;
            stream.captured_dependency_count = 1;
            stream.pages[0].covered_mask[0] = 1;
            placement.computation_frontier = 1;
            placement.cells.resize(1);
            stash.captured_sink_count = 1;
            stash.covered_sink_pages[0].covered_mask[0] = 1;
        }
        if (stash_case >= 2) {
            auto & descriptor = base.package.unit_blobs[0].descriptor;
            descriptor.clean_stash_state =
                vbr_artifact_clean_stash_state::absent_at_source;
            descriptor.clean_stash = {};
            auto & reference = manifest.unit_references[0];
            reference.has_stash_reference = false;
            reference.stash_reference = {};
            manifest.accounting.erase(
                std::remove_if(
                    manifest.accounting.begin(), manifest.accounting.end(),
                    [](const vbr_artifact_portable_accounting_row & row) {
                        return row.role ==
                            vbr_artifact_accounting_role::clean_stash_payload;
                    }),
                manifest.accounting.end());
            if (stash_case == 3) {
                descriptor.current_type = GGML_TYPE_TURBO8_0;
                manifest.generation.controllers[0].units[0].current_type =
                    GGML_TYPE_TURBO8_0;
                manifest.generation.controllers[0].units[0].domain =
                    vbr_repr_domain::full;
            }
        }
        manifest.manifest_digest = {};
        CHECK(base.catalog->configure_accounting(base.package));
        if (legacy_v1) {
            base.package.version = 1;
            manifest.version = 1;
            manifest.stream_placements.clear();
            manifest.token_block = {};
            const auto published = base.catalog->publish(
                base.package, base.completions(), base.budget);
            CHECK(published.status ==
                  llama_vbr_artifact_publish_status::published);
            reference_artifact = published.reference_artifact;
        } else {
            vbr_artifact_companion_payload companion;
            companion.kind = vbr_artifact_companion_kind::recurrent;
            companion.format_version = 1;
            companion.build_identity_digest = marker(0xa1);
            companion.domain = {
                llama_cache_acct_residency::pageable_host,
                llama_cache_acct_domain_kind::not_applicable,
                UINT32_MAX,
                UINT16_MAX,
            };
            companion.payload_bytes = base.storage.recurrent.bytes.size();
            base.package.companions.push_back(companion);
            manifest.accounting.push_back({
                vbr_artifact_accounting_role::recurrent_payload,
                companion.domain,
                companion.payload_bytes,
                companion.payload_bytes,
                llama_cache_acct_attr_kind::artifact,
            });
            CHECK(base.catalog->configure_accounting(base.package));

            vbr_capture_stream_status status;
            auto build = base.catalog->begin_capture(
                base.package, base.budget, {}, status);
            CHECK(build && status == vbr_capture_stream_status::ok);
            auto unit = build->begin_unit(0, status);
            CHECK(unit && status == vbr_capture_stream_status::ok);
            for (const auto & completion : base.completions()) {
                if (completion.clean_stash && stash_case >= 2) {
                    continue;
                }
                const auto segment = verified_segment(completion, 2);
                CHECK(unit->accept_verified_segment(segment) ==
                      vbr_capture_stream_status::ok);
            }
            CHECK(unit->seal_unit() == vbr_capture_stream_status::ok);
            unit.reset();
            auto companion_bytes =
                std::make_shared<artifact_segment_chain>();
            CHECK(companion_bytes->append(
                base.storage.recurrent.bytes.data(),
                base.storage.recurrent.bytes.size()));
            vbr_verified_companion verified;
            verified.companion_index = 0;
            verified.bytes = companion_bytes;
            verified.streaming_digest =
                vbr_capture_stream_digest(*companion_bytes);
            CHECK(build->accept_verified_companion(verified) ==
                  vbr_capture_stream_status::ok);
            const auto published = build->publish_reference();
            CHECK(published.status == vbr_capture_stream_status::ok);
            reference_artifact = published.reference_artifact;
        }
        CHECK(base.catalog->resolve_reference(
                  reference_artifact, view) ==
              vbr_artifact_resolve_status::ok);
        CHECK(view.validate() == vbr_artifact_status::ok);

        accounting = base.ledger.snapshot();
        serials.accounting = accounting.serial;
        serials.policy = 77;

        target.memory_instance_cookie = 0x1001;
        target.target_state_serial = 31;
        target.accounting_serial = accounting.serial;
        target.tree_shape_digest = 0x2002;
        target.policy_epoch = serials.policy;
        target.scheduler_idle = true;
        target.destination_sequence_absent = true;

        vbr_target_child_snapshot child;
        child.child_id = 0;
        child.dependency_mode =
            checkpoint_child_dependency_mode::live_guarded;
        child.memory_cookie = reinterpret_cast<const void *>(uintptr_t(0x3003));
        child.empty = true;
        child.dedicated = true;
        child.armed = true;
        child.lineage_uuid = { 0x55, 0x66 };
        child.instance_id = { 0x77, 0x88 };
        child.state_serial = target.target_state_serial;
        child.policy_epoch = target.policy_epoch;
        child.controller_policy = manifest.controller_policy[0];

        const auto & descriptor = view.units()[0].descriptor;
        vbr_target_unit_snapshot unit;
        unit.child_id = descriptor.child_id;
        unit.logical_unit_id = descriptor.logical_unit_id;
        unit.current_type = descriptor.current_type;
        unit.last_source_type = descriptor.last_source_type;
        unit.promote_hops = descriptor.promote_hops;
        unit.last_transition = descriptor.last_transition;
        unit.representation_kind = descriptor.representation.kind;
        unit.codec_id = descriptor.representation.codec_id;
        unit.codec_version = descriptor.representation.codec_version;
        unit.representation_reference_digest =
            descriptor.representation.reference_digest;
        unit.source_loss_history =
            descriptor.representation.source_loss_history;
        unit.checkpoint_codec_hops =
            descriptor.representation.checkpoint_codec_hops;
        unit.recoverability = descriptor.recoverability;
        unit.side = descriptor.side;
        unit.layout = descriptor.layout;
        unit.row_codec_version = descriptor.row_codec_version;
        unit.current_domain = manifest.generation.controllers[0].units[0].domain;
        unit.codebook_digest = descriptor.codebook_digest;
        unit.rotation_digest = descriptor.rotation_digest;
        unit.meansub_digest = descriptor.meansub_digest;
        unit.n_stream = descriptor.n_stream;
        unit.unified = descriptor.unified;
        unit.wm_cells = descriptor.wm_cells;
        unit.rank = descriptor.rank;
        unit.dimensions = descriptor.dimensions;
        unit.row_alignment = descriptor.row_alignment;
        for (size_t i = 0; i < descriptor.shards.size(); ++i) {
            const auto & source = descriptor.shards[i];
            unit.shards.push_back({
                uint32_t(i),
                reinterpret_cast<const void *>(uintptr_t(0x4000 + i)),
                base.bindings[source.device_ordinal].domain,
                source.topology_index,
                source.device_ordinal,
                base.package.topologies[source.topology_index].digest,
                source.logical_offset,
                source.row_count,
                source.row_bytes,
                source.payload_bytes,
            });
        }
        child.units.push_back(unit);
        target.children.push_back(child);
        for (const auto & companion : view.companions()) {
            target.companions.push_back({
                companion.descriptor.kind,
                companion.descriptor.format_version,
                companion.descriptor.build_identity_digest,
                true,
                reinterpret_cast<const void *>(uintptr_t(0x5005)),
            });
        }

        policy.authorized = true;
        policy.identity.execution_identity =
            manifest.identity.execution_identity;
        policy.identity.adapter_config_identity =
            manifest.identity.adapter_config_identity;
        policy.identity.media_content_identity =
            manifest.identity.media_content_identity;
        policy.identity.sequence_epoch =
            manifest.identity.sequence_epoch;
        policy.identity.requested_frontier =
            manifest.identity.next_position;
        policy.identity.tokens = &manifest.token_block.tokens;
        policy.destination_sequence = 4;
        policy.domain_bindings = base.bindings;
        policy.domain_bindings.push_back({
            UINT32_MAX, UINT16_MAX, base.host,
        });
        policy.accounting_snapshot = &accounting;
        policy.budget_config = &base.budget;
        policy.context = &serials;
        policy.adoption_nonce = 0x100 + uint64_t(stash_case) +
            (legacy_v1 ? 0x10 : 0);
        policy.parse_companion = validator_serials::parse_companion;
        policy.recheck_target_empty = validator_serials::recheck_target;
        policy.read_accounting_serial =
            validator_serials::read_accounting;
        policy.read_policy_epoch = validator_serials::read_policy;
    }
};

static vbr_manifest_validation_result validate(
        validator_fixture & fixture,
        const vbr_target_validation_snapshot & target,
        const vbr_adopt_policy & policy) {
    return vbr_validate_unit_manifest_snapshot(
        target, fixture.view, policy);
}

static void test_manifest_validator_matrix() {
    validator_fixture f;

    vbr_artifact_package_view absent_package;
    const auto absent = vbr_validate_unit_manifest_snapshot(
        f.target, absent_package, f.policy);
    CHECK(absent.status ==
          vbr_manifest_validation_status::unsupported_artifact_version);
    CHECK(absent.decision == vbr_import_decision::reject);
    CHECK(!absent.proof);

    auto native = validate(f, f.target, f.policy);
    CHECK(native.status == vbr_manifest_validation_status::validated);
    CHECK(native.decision == vbr_import_decision::native_import);
    CHECK(native.proof);
    CHECK(native.proof->decision() == vbr_import_decision::native_import);
    CHECK(native.proof->source_artifact() == f.reference_artifact);
    CHECK(native.proof->children().size() == 1);
    CHECK(native.proof->children()[0].placements.size() == 1);
    CHECK(native.proof->children()[0].stash_action ==
          vbr_validated_stash_action::restore_exact);
    CHECK(native.proof->companions().size() == 1);
    CHECK(native.proof->companions()[0].parsed);
    CHECK(native.proof->companions()[0].parsed->kind() ==
          vbr_artifact_companion_kind::recurrent);
    size_t existing_leaves = 0;
    size_t fresh_metadata_leaves = 0;
    for (const auto & leaf : native.proof->accounting_leaves()) {
        if (leaf.existing_allocation) {
            ++existing_leaves;
            CHECK(leaf.reserve_resident == 0);
        } else if (leaf.category ==
                       llama_cache_acct_category::artifact_descriptor_metadata ||
                   leaf.category ==
                       llama_cache_acct_category::artifact_reference_metadata) {
            ++fresh_metadata_leaves;
            CHECK(leaf.reserve_resident != 0);
        }
    }
    CHECK(existing_leaves == 5);
    CHECK(fresh_metadata_leaves == 2);
    CHECK(native.proof->tracker_install().children[0].transition ==
          vbr_tracker_install_transition::native_clone);
    CHECK(native.proof->tracker_install().children[0].lineage_uuid ==
          f.view.manifest().generation.controllers[0].lineage_uuid);
    CHECK(native.proof->tracker_install().children[0].global_generation ==
          f.view.manifest().generation.controllers[0].global_generation);
    const uint64_t first_nonce = native.proof->adoption_nonce();
    CHECK(first_nonce == f.policy.adoption_nonce);

    auto second_policy = f.policy;
    second_policy.adoption_nonce = first_nonce + 1;
    auto second = validate(f, f.target, second_policy);
    CHECK(second.status == vbr_manifest_validation_status::validated);
    CHECK(second.proof && second.proof->adoption_nonce() != first_nonce);
    CHECK(second.proof->adoption_nonce() == second_policy.adoption_nonce);
    CHECK(second.proof->source_artifact() == native.proof->source_artifact());

    // F4.0 clone semantics: source-lineage liveness/collision never blocks a
    // native clone because operation authentication uses the fresh target
    // runtime instance instead.
    auto colliding_lineage = f.target;
    colliding_lineage.children[0].lineage_uuid =
        f.view.manifest().generation.controllers[0].lineage_uuid;
    auto clone_policy = f.policy;
    clone_policy.adoption_nonce = first_nonce + 2;
    auto clone_with_live_source_lineage = validate(
        f, colliding_lineage, clone_policy);
    CHECK(clone_with_live_source_lineage.status ==
          vbr_manifest_validation_status::validated);
    CHECK(clone_with_live_source_lineage.decision ==
          vbr_import_decision::native_import);
    CHECK(clone_with_live_source_lineage.proof);
    CHECK(clone_with_live_source_lineage.proof->tracker_install()
              .children[0].target_instance ==
          colliding_lineage.children[0].instance_id);

    auto observed = f.target;
    observed.children[0].previously_observed = true;
    auto live_policy = f.policy;
    live_policy.adoption_nonce = first_nonce + 3;
    auto live = validate(f, observed, live_policy);
    CHECK(live.status == vbr_manifest_validation_status::validated);
    CHECK(live.decision == vbr_import_decision::live_rebased);
    CHECK(live.proof);
    CHECK(live.proof->tracker_install().children[0].transition ==
          vbr_tracker_install_transition::whole_import);
    CHECK(live.proof->tracker_install().children[0].lineage_uuid ==
          observed.children[0].lineage_uuid);
    CHECK(live.proof->tracker_install().children[0].global_generation == 1);
    CHECK(live.proof->tracker_install().children[0].global_generation !=
          f.view.manifest().generation.controllers[0].global_generation);
    CHECK(live.proof->tracker_install().children[0].units.size() == 1);
    CHECK(live.proof->tracker_install().children[0].units[0].repr_gen == 1);
    CHECK(live.proof->tracker_install().children[0].units[0].repr_gen !=
          f.view.manifest().generation.controllers[0].units[0].repr_gen);
    CHECK(live.proof->tracker_install().children[0].units[0].last_transition ==
          vbr_repr_transition::whole_import);
    // Stash disposition is orthogonal to tracker provenance: an observed
    // target rebases generation but may still restore a complete exact prefix.
    CHECK(live.proof->children()[0].stash_action ==
          vbr_validated_stash_action::restore_exact);

    auto downward_target = f.target;
    auto & downward_unit = downward_target.children[0].units[0];
    downward_unit.current_type = GGML_TYPE_TURBO3_TCQ;
    downward_unit.downward_supported = true;
    downward_unit.downward_movable = true;
    downward_unit.controller_floor_type = GGML_TYPE_TURBO1_TCQ;
    downward_unit.downward_type = downward_unit.current_type;
    downward_unit.downward_domain = vbr_repr_domain::tapped;
    downward_unit.downward_recipe_id = 1;
    downward_unit.downward_recipe_version = 1;
    downward_unit.downward_row_bytes = 2;
    downward_unit.downward_mapped_bytes = 2;
    downward_unit.downward_transfer_bytes = 4;
    downward_unit.downward_codec_workspace_bytes = 4;
    downward_unit.downward_meansub_model_id = 7;
    CHECK(vbr_downward_resolve_recipe(
        static_cast<ggml_type>(f.view.units()[0].descriptor.current_type),
        static_cast<ggml_type>(downward_unit.current_type),
        GGML_TYPE_TURBO1_TCQ, true,
        downward_unit.downward_recipe) ==
        vbr_downward_recipe_status::resolved);
    vbr_downward_policy_child projected_child;
    projected_child.initial_types = {
        static_cast<ggml_type>(f.view.units()[0].descriptor.current_type),
    };
    projected_child.target_types = {
        static_cast<ggml_type>(downward_unit.current_type),
    };
    for (size_t i = 0; i < downward_unit.downward_recipe.n_edges; ++i) {
        const auto & edge = downward_unit.downward_recipe.edges[i];
        projected_child.policy.steps.push_back({
            i, 0, int32_t(edge.source_type), int32_t(edge.target_type), 1,
        });
    }
    projected_child.policy.terminal_progress =
        int64_t(projected_child.policy.steps.size());
    const auto projection =
        vbr_downward_project_policy_prefix({ projected_child });
    CHECK(projection.status == vbr_downward_policy_status::coherent);
    downward_target.children[0].controller_policy.current_type_vector_digest =
        projection.child_type_digests[0];
    downward_target.children[0].controller_policy.cursor +=
        projection.prefix.size();
    downward_unit.downward_build_identity_digest =
        vbr_downward_build_identity(
            downward_unit.downward_recipe,
            downward_unit.downward_meansub_model_id,
            downward_unit.meansub_digest,
            projection.child_type_digests[0], projection.tree_digest);
    llama_cache_budget_plan downward_plan;
    downward_plan.accounting_serial = f.accounting.serial;
    auto downward_policy = f.policy;
    downward_policy.adoption_nonce = first_nonce + 4;
    downward_policy.downward_budget_plan = &downward_plan;
    downward_policy.downward_projection = &projection;
    f.serials.downward_tree_digest = projection.tree_digest;
    downward_policy.read_downward_tree_digest =
        validator_serials::read_downward_tree;
    auto downward = validate(f, downward_target, downward_policy);
    CHECK(downward.status == vbr_manifest_validation_status::validated);
    CHECK(downward.decision == vbr_import_decision::downward_rebase);
    CHECK(downward.proof && downward.proof->children()[0].downward);
    CHECK(downward.proof->tracker_install().children[0].transition ==
          vbr_tracker_install_transition::whole_import);
    CHECK(downward.proof->tracker_install().children[0].units[0].repr_gen == 1);
    CHECK(downward.proof->tracker_install().children[0].units[0].last_transition ==
          vbr_repr_transition::whole_import);

    auto restrictive = f.base.budget;
    for (auto & device : restrictive.devices) {
        device.configured_cache_cap = 0;
        device.cache_cap_state = llama_cache_budget_capacity_state::known;
    }
    auto fallback_policy = f.policy;
    fallback_policy.budget_config = &restrictive;
    auto rebuild = validate(f, f.target, fallback_policy);
    CHECK(rebuild.status == vbr_manifest_validation_status::validated);
    CHECK(rebuild.decision == vbr_import_decision::rebuild);
    CHECK(!rebuild.proof);
    fallback_policy.allow_rebuild = false;
    auto cold = validate(f, f.target, fallback_policy);
    CHECK(cold.status == vbr_manifest_validation_status::validated);
    CHECK(cold.decision == vbr_import_decision::cold);
    fallback_policy.allow_cold = false;
    auto reject = validate(f, f.target, fallback_policy);
    CHECK(reject.status == vbr_manifest_validation_status::validated);
    CHECK(reject.decision == vbr_import_decision::reject);

    auto check_status = [&](vbr_manifest_validation_status expected,
                            vbr_target_validation_snapshot target,
                            vbr_adopt_policy policy) {
        const auto result = validate(f, target, policy);
        CHECK(result.status == expected);
        CHECK(result.decision == vbr_import_decision::reject);
        CHECK(!result.proof);
    };
    auto policy = f.policy;
    policy.authorized = false;
    check_status(vbr_manifest_validation_status::unauthorized, f.target, policy);
    policy = f.policy;
    policy.identity.execution_identity += ":wrong";
    check_status(vbr_manifest_validation_status::identity_mismatch, f.target, policy);
    policy = f.policy;
    auto mismatched_tokens = *policy.identity.tokens;
    mismatched_tokens[0]++;
    policy.identity.tokens = &mismatched_tokens;
    check_status(vbr_manifest_validation_status::token_block_mismatch, f.target, policy);

    auto target = f.target;
    target.children.clear();
    check_status(vbr_manifest_validation_status::memory_tree_mismatch, target, f.policy);
    target = f.target;
    target.scheduler_idle = false;
    check_status(vbr_manifest_validation_status::target_not_idle, target, f.policy);
    target = f.target;
    target.children[0].empty = false;
    check_status(vbr_manifest_validation_status::target_not_empty, target, f.policy);
    target = f.target;
    target.destination_sequence_absent = false;
    check_status(vbr_manifest_validation_status::target_not_empty, target, f.policy);
    target = f.target;
    target.children[0].dedicated = false;
    check_status(vbr_manifest_validation_status::target_not_dedicated, target, f.policy);
    target = f.target;
    target.children[0].armed = false;
    check_status(vbr_manifest_validation_status::target_not_armed, target, f.policy);
    target = f.target;
    target.children[0].units[0].n_stream = 2;
    check_status(vbr_manifest_validation_status::target_not_armed, target, f.policy);
    target = f.target;
    target.children[0].units[0].rank++;
    check_status(vbr_manifest_validation_status::geometry_mismatch, target, f.policy);
    target = f.target;
    target.children[0].units.push_back(target.children[0].units[0]);
    check_status(vbr_manifest_validation_status::geometry_mismatch, target, f.policy);
    target = f.target;
    target.children[0].units[0].shards[0].device_ordinal = 1;
    check_status(vbr_manifest_validation_status::topology_mismatch, target, f.policy);
    target = f.target;
    target.children[0].units[0].codec_id++;
    check_status(vbr_manifest_validation_status::representation_mismatch, target, f.policy);
    target = f.target;
    target.children[0].units[0].codebook_digest[0] ^= 1;
    check_status(vbr_manifest_validation_status::codebook_mismatch, target, f.policy);
    target = f.target;
    target.children[0].controller_policy.policy_digest[0] ^= 1;
    check_status(vbr_manifest_validation_status::policy_mismatch, target, f.policy);
    target = f.target;
    target.children[0].generation_compatible = false;
    check_status(vbr_manifest_validation_status::generation_mismatch, target, f.policy);
    target = f.target;
    target.children[0].ownership_compatible = false;
    check_status(vbr_manifest_validation_status::ownership_mismatch, target, f.policy);
    target = f.target;
    target.children[0].stash_compatible = false;
    check_status(vbr_manifest_validation_status::stash_inconsistent, target, f.policy);
    target = f.target;
    target.companions.clear();
    check_status(
        vbr_manifest_validation_status::required_companion_unavailable,
        target, f.policy);
    policy = f.policy;
    policy.parse_companion = nullptr;
    check_status(
        vbr_manifest_validation_status::required_companion_unavailable,
        f.target, policy);

    policy = f.policy;
    policy.accounting_snapshot = nullptr;
    check_status(vbr_manifest_validation_status::accounting_unavailable, f.target, policy);
    policy = f.policy;
    policy.budget_config = nullptr;
    check_status(vbr_manifest_validation_status::accounting_unavailable, f.target, policy);
    policy = f.policy;
    auto unavailable_budget = f.base.budget;
    unavailable_budget.devices[0].phys_state =
        llama_cache_budget_capacity_state::unavailable;
    policy.budget_config = &unavailable_budget;
    check_status(vbr_manifest_validation_status::budget_unavailable, f.target, policy);
    policy = f.policy;
    policy.native_instance_available = false;
    check_status(vbr_manifest_validation_status::native_lineage_unavailable, f.target, policy);
    policy = f.policy;
    policy.adoption_nonce = 0;
    check_status(vbr_manifest_validation_status::internal_error, f.target, policy);
    policy = f.policy;
    validator_serials drift = f.serials;
    drift.accounting++;
    policy.context = &drift;
    check_status(vbr_manifest_validation_status::unavailable, f.target, policy);
    policy = f.policy;
    drift = f.serials;
    drift.target_stable = false;
    policy.context = &drift;
    check_status(vbr_manifest_validation_status::unavailable, f.target, policy);

    validator_fixture partial(false, 1);
    partial.policy.adoption_nonce = first_nonce + 10;
    const auto partial_result = validate(
        partial, partial.target, partial.policy);
    CHECK(partial_result.status ==
          vbr_manifest_validation_status::validated);
    CHECK(partial_result.decision == vbr_import_decision::live_rebased);
    CHECK(partial_result.proof);
    CHECK(partial_result.proof->children()[0].stash_action ==
          vbr_validated_stash_action::omit_live_rebased);
    CHECK(partial_result.proof->manifest_digest() !=
          native.proof->manifest_digest());
    CHECK(partial_result.proof->adoption_nonce() != first_nonce);

    validator_fixture stashless(false, 2);
    stashless.policy.adoption_nonce = first_nonce + 11;
    const auto stashless_result = validate(
        stashless, stashless.target, stashless.policy);
    CHECK(stashless_result.status ==
          vbr_manifest_validation_status::validated);
    CHECK(stashless_result.decision ==
          vbr_import_decision::live_rebased);
    CHECK(stashless_result.proof);
    CHECK(stashless_result.proof->children()[0].stash_action ==
          vbr_validated_stash_action::none_at_source);

    validator_fixture full_stashless(false, 3);
    full_stashless.policy.adoption_nonce = first_nonce + 12;
    const auto full_stashless_result = validate(
        full_stashless, full_stashless.target, full_stashless.policy);
    CHECK(full_stashless_result.status ==
          vbr_manifest_validation_status::validated);
    CHECK(full_stashless_result.decision ==
          vbr_import_decision::native_import);
    CHECK(full_stashless_result.proof);
    CHECK(full_stashless_result.proof->children()[0].stash_action ==
          vbr_validated_stash_action::none_at_source);

    validator_fixture legacy(true);
    const auto legacy_result = validate(
        legacy, legacy.target, legacy.policy);
    CHECK(legacy_result.status ==
          vbr_manifest_validation_status::restore_metadata_missing);
    CHECK(legacy_result.decision == vbr_import_decision::rebuild);
    CHECK(!legacy_result.proof);
    auto legacy_cold_policy = legacy.policy;
    legacy_cold_policy.allow_rebuild = false;
    const auto legacy_cold = validate(
        legacy, legacy.target, legacy_cold_policy);
    CHECK(legacy_cold.status ==
          vbr_manifest_validation_status::restore_metadata_missing);
    CHECK(legacy_cold.decision == vbr_import_decision::cold);
    CHECK(!legacy_cold.proof);

    for (uint8_t i = 0;
         i < uint8_t(vbr_import_decision::_count); ++i) {
        CHECK(strcmp(vbr_import_decision_name(
                  vbr_import_decision(i)), "invalid") != 0);
    }
    for (uint8_t i = 0;
         i < uint8_t(vbr_manifest_validation_status::_count); ++i) {
        CHECK(strcmp(vbr_manifest_validation_status_name(
                  vbr_manifest_validation_status(i)), "invalid") != 0);
    }
}

static vbr_adopt_stage_policy stage_policy_for(
        validator_fixture & fixture,
        llama_cache_budget_config & budget) {
    budget = fixture.base.budget;
    budget.host.pinned_cap = 1024*1024;
    budget.host.pinned_state =
        llama_cache_budget_capacity_state::known;
    budget.host.total_state =
        llama_cache_budget_capacity_state::unbounded;

    vbr_adopt_stage_policy policy;
    policy.ledger = &fixture.base.ledger;
    policy.budget = &budget;
    policy.pinned_domain = fixture.base.pinned;
    policy.pinned_ring_bytes = 32;
    policy.chunk_bytes = 8;
    for (const auto & binding : fixture.base.bindings) {
        policy.lanes.push_back({ binding.domain, nullptr, nullptr, false });
    }
    return policy;
}

static void test_validated_manifest_staging() {
    validator_fixture fixture(false, 0, true);
    auto validated = validate(fixture, fixture.target, fixture.policy);
    CHECK(validated.status == vbr_manifest_validation_status::validated);
    CHECK(validated.proof);
    const uint64_t nonce = validated.proof->adoption_nonce();
    const auto digest = validated.proof->manifest_digest();
    const uint64_t validation_serial =
        validated.proof->target().accounting_serial;
    const uint64_t baseline_ops = fixture.base.ledger.snapshot().live_ops;

    llama_cache_budget_config budget;
    auto policy = stage_policy_for(fixture, budget);
    auto staged = vbr_stage_validated_manifest(
        std::move(validated.proof), policy);
    CHECK(staged.status == vbr_adopt_stage_status::staged);
    CHECK(staged.manifest);
    CHECK(staged.staged);
    if (staged.staged) {
        CHECK(staged.staged->adoption_nonce() == nonce);
        CHECK(staged.staged->manifest_digest() == digest);
        CHECK(staged.staged->decision() ==
              vbr_import_decision::native_import);
        CHECK(staged.staged->validation_accounting_serial() ==
              validation_serial);
        CHECK(staged.staged->accounting_serial_after_prepare() >
              validation_serial);
        CHECK(staged.staged->accounting_serial_after_prepare() ==
              fixture.base.ledger.snapshot().serial);
        CHECK(staged.staged->claims_ready());
        CHECK(staged.staged->ring_capacity_bytes() == 32);
        CHECK(staged.staged->read_count() >= 5);
        for (const auto & read : staged.staged->reads()) {
            CHECK(read.source);
            CHECK(read.size != 0);
            CHECK(std::any_of(
                read.verified_digest.begin(), read.verified_digest.end(),
                [](uint8_t value) { return value != 0; }));
        }
    }
    CHECK(fixture.base.ledger.snapshot().live_ops > baseline_ops);
    staged.staged.reset();
    CHECK(fixture.base.ledger.snapshot().live_ops == baseline_ops);
    CHECK(staged.manifest && staged.manifest->adoption_nonce() == nonce);

    const auto run_failure = [&](vbr_adopt_stage_policy failed_policy,
                                 vbr_adopt_stage_status expected) {
        fixture.accounting = fixture.base.ledger.snapshot();
        fixture.target.accounting_serial = fixture.accounting.serial;
        fixture.serials.accounting = fixture.accounting.serial;
        fixture.policy.accounting_snapshot = &fixture.accounting;
        auto proof = validate(fixture, fixture.target, fixture.policy);
        CHECK(proof.proof);
        const uint64_t before = fixture.base.ledger.snapshot().live_ops;
        auto result = vbr_stage_validated_manifest(
            std::move(proof.proof), failed_policy);
        CHECK(result.status == expected);
        CHECK(result.manifest);
        CHECK(!result.staged);
        CHECK(fixture.base.ledger.snapshot().live_ops == before);
    };

    policy.fault.fail_source_verify_at = 0;
    run_failure(policy, vbr_adopt_stage_status::source_hash_mismatch);
    policy.fault = {};
    policy.fault.fail_before_prepare = true;
    run_failure(policy, vbr_adopt_stage_status::internal_error);
    policy.fault = {};
    policy.fault.fail_ring_allocation = true;
    run_failure(policy, vbr_adopt_stage_status::ring_unavailable);

    policy.fault = {};
    llama_cache_budget_config refused_budget = budget;
    for (auto & device : refused_budget.devices) {
        device.configured_cache_cap = 0;
        device.cache_cap_state = llama_cache_budget_capacity_state::known;
    }
    policy.budget = &refused_budget;
    run_failure(policy, vbr_adopt_stage_status::admission_refused);

    // Duplicate domain bindings are ambiguous and therefore fail before any
    // claim or ring allocation.
    policy.budget = &budget;
    policy.lanes.push_back(policy.lanes.front());
    run_failure(policy, vbr_adopt_stage_status::source_unavailable);
}

#ifdef VBR_PROMPT_CACHE_PUBLICATION_TEST
static void test_prompt_cache_vbr_atomic_logical_publication() {
    catalog_fixture fixture;

    server_prompt prompt;
    prompt.tokens = server_tokens(llama_tokens { 101, 102 }, false);
    prompt.sequence_epoch = 3;
    std::string media_identity;
    CHECK(prompt.tokens.media_content_identity(
        prompt.n_tokens(), media_identity));
    fixture.package.manifest.identity.media_content_identity =
        media_identity;
    fixture.package.manifest.identity.next_position =
        prompt.tokens.pos_next();

    const auto published = fixture.catalog->publish(
        fixture.package, fixture.completions(), fixture.budget);
    CHECK(published.status ==
          llama_vbr_artifact_publish_status::published);
    vbr_artifact_package_view view;
    CHECK(fixture.catalog->resolve_reference(
              published.reference_artifact, view) ==
          vbr_artifact_resolve_status::ok);
    auto owner = server_prompt_cache_vbr_payload::adopt(
        std::move(view));
    CHECK(owner);
    if (!owner) {
        return;
    }
    auto payload = server_prompt_cache_payload::from_vbr(owner);

    server_retention_sidecar_store retention;
    retention.configure(&fixture.ledger, fixture.host);
    CHECK(retention.enable_prefix_tracking());
    constexpr int32_t source_slot = 4;
    const auto source_key =
        server_retention_instance_key::for_slot(source_slot);
    common_chat_msg_spans spans;
    spans.add(COMMON_CHAT_ROLE_USER, 0, prompt.n_tokens());
    CHECK(retention.publish(
        source_key, common_retention_pool::attention,
        spans, true, prompt.n_tokens(), prompt.n_tokens(), true));
    CHECK(server_prompt_retention_publish_exact_prefix(
        retention, source_key, prompt,
        fixture.package.manifest.identity.adapter_config_identity,
        prompt.n_tokens()));

    server_prompt_cache cache(/* limit_size_mib */ 0,
                              /* limit_tokens */ 0);
    cache.acct = &fixture.ledger;
    cache.retention_obs = &retention;

    auto staged = cache.stage_vbr(
        prompt, payload,
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(staged.size() == 1);
    CHECK(staged.front().payload.publishable());
    CHECK(!staged.front().payload.restorable());
    CHECK(staged.front().payload.accounted_by(&fixture.ledger));
    CHECK(staged.front().payload.size() == owner->resident_bytes());

    if (server_fault("vbr_prompt_cache_prefix_fail")) {
        const auto destination_key =
            server_retention_instance_key::for_host_entry(&staged.front());
        const auto source_artifact = retention.artifact_id(source_key);
        const auto ledger_before = fixture.ledger.snapshot();
        CHECK(!cache.publish(
            std::move(staged), &prompt, source_slot));
        const auto ledger_after = fixture.ledger.snapshot();
        CHECK(cache.states.empty());
        CHECK(retention.artifact_id(destination_key).v == 0);
        CHECK(retention.artifact_id(source_key) == source_artifact);
        CHECK(retention.prefix_tracking_available());
        CHECK(ledger_after.live_ops == ledger_before.live_ops);
        CHECK(ledger_after.serial >= ledger_before.serial);
        return;
    }

    const uint64_t live_ops_before =
        fixture.ledger.snapshot().live_ops;
    server_prompt_cache::iterator logical;
    CHECK(cache.publish(
        std::move(staged), &prompt, source_slot, &logical));
    CHECK(logical != cache.states.end());
    CHECK(cache.states.size() == 1);
    CHECK(!cache.contains(
        prompt.tokens,
        fixture.package.manifest.identity.adapter_config_identity));
    CHECK(logical->payload.vbr_artifact() == owner.get());
    CHECK((logical->release_ops() ==
           std::array<llama_cache_acct_op_id, 3> {}));
    // The catalog already owns every physical VBR allocation. Logical host
    // publication adds only its retention metadata leaf.
    CHECK(fixture.ledger.snapshot().live_ops == live_ops_before + 1);
    auto host_key =
        server_retention_instance_key::for_host_entry(&*logical);
    CHECK(retention.artifact_id(host_key).v != 0);

    // Re-publishing an exact immutable owner is a net-zero physical capacity
    // operation. The preflight must account the allocation union and the
    // later replacement set rather than temporarily charging both aliases.
    cache.limit_size = logical->size();
    cache.limit_tokens = prompt.n_tokens();
    auto refreshed_stage = cache.stage_vbr(
        prompt, payload,
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(refreshed_stage.size() == 1);
    server_prompt_cache::iterator refreshed;
    CHECK(cache.publish(
        std::move(refreshed_stage), &prompt, source_slot, &refreshed));
    CHECK(refreshed != cache.states.end());
    CHECK(cache.states.size() == 1);
    CHECK(fixture.ledger.snapshot().live_ops == live_ops_before + 1);
    CHECK(retention.artifact_id(host_key).v == 0);
    logical = refreshed;
    host_key = server_retention_instance_key::for_host_entry(&*logical);
    CHECK(retention.artifact_id(host_key).v != 0);
    cache.limit_size = 0;
    cache.limit_tokens = 0;

    // H1 does not yet own the catalog-reference release transaction needed
    // to make VBR bytes disappear under cache pressure. A maintenance pass
    // must therefore preserve the logical node instead of erasing it while
    // falsely claiming progress.
    const size_t logical_bytes = logical->size();
    CHECK(logical_bytes > 0);
    cache.limit_size = logical_bytes - 1;
    cache.update();
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().payload.vbr_artifact() == owner.get());

    // A fixed peer remains a lawful pressure victim. Skipping the VBR node
    // must not disable ordinary FIFO progress when releasable storage exists.
    cache.limit_size = 0;
    auto fixed_peer = cache.stage(
        prompt, 64, 0,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(fixed_peer.size() == 1);
    CHECK(cache.publish(std::move(fixed_peer)));
    CHECK(cache.states.size() == 2);
    CHECK(retention.artifact_id(host_key).v != 0);
    cache.limit_size = logical_bytes;
    cache.update();
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().payload.vbr_artifact() == owner.get());
    CHECK(retention.artifact_id(host_key).v != 0);
    cache.limit_size = 0;

    // A fixed incoming publication cannot be mistaken for a successfully
    // reclaimed incumbent merely because all older nodes are protected VBR
    // borrows. The refusal terminal removes it and never returns its dead
    // iterator as a published entry.
    cache.limit_size = logical_bytes;
    auto refused_fixed = cache.stage(prompt, 64, 0, "pressure-incoming");
    CHECK(refused_fixed.size() == 1);
    server_prompt_cache::iterator refused_result = cache.states.begin();
    CHECK(!cache.publish(
        std::move(refused_fixed), nullptr, -1, &refused_result));
    CHECK(refused_result == cache.states.end());
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().payload.vbr_artifact() == owner.get());
    cache.limit_size = 0;

    // Every identity component is checked before allocation/publication.
    server_prompt wrong_epoch = prompt.clone();
    wrong_epoch.sequence_epoch++;
    CHECK(cache.stage_vbr(
        wrong_epoch, payload,
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity).empty());
    CHECK(cache.stage_vbr(
        prompt, payload, "wrong-execution",
        fixture.package.manifest.identity.adapter_config_identity).empty());
    CHECK(cache.stage_vbr(
        prompt, payload,
        fixture.package.manifest.identity.execution_identity,
        "wrong-adapter").empty());

    // Publication binds the logical host lineage to the exact live source
    // frontier; a caller cannot attach sealed bytes to an unrelated/stale
    // source association merely because that slot key exists.
    server_prompt wrong_source = prompt.clone();
    wrong_source.sequence_epoch++;
    auto wrong_source_stage = cache.stage_vbr(
        prompt, payload,
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(wrong_source_stage.size() == 1);
    CHECK(!cache.publish(
        std::move(wrong_source_stage), &wrong_source, source_slot));
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().payload.vbr_artifact() == owner.get());

    // A catalog lease from another accounting owner cannot enter this cache,
    // and refusal leaves both the logical cache and source lineage untouched.
    llama_cache_acct_ledger unrelated;
    server_prompt_cache wrong_ledger(/* limit_size_mib */ 0,
                                     /* limit_tokens */ 0);
    wrong_ledger.acct = &unrelated;
    wrong_ledger.retention_obs = &retention;
    auto wrong_owner = wrong_ledger.stage_vbr(
        prompt, payload,
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(wrong_owner.size() == 1);
    CHECK(!wrong_ledger.publish(
        std::move(wrong_owner), &prompt, source_slot));
    CHECK(wrong_ledger.states.empty());

    // H1 does not synchronously destroy another host entry to publish VBR
    // until the refcount-aware release transaction lands.
    const size_t mib = 1024*1024;
    server_prompt_cache bounded(/* limit_size_mib */ 1,
                                /* limit_tokens */ 0);
    bounded.acct = &fixture.ledger;
    bounded.retention_obs = &retention;
    auto fixed = bounded.stage(prompt, mib, 0, "fixed");
    CHECK(fixed.size() == 1);
    CHECK(bounded.publish(std::move(fixed)));
    auto would_displace = bounded.stage_vbr(
        prompt, payload,
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(would_displace.size() == 1);
    CHECK(!bounded.publish(
        std::move(would_displace), &prompt, source_slot));
    CHECK(bounded.states.size() == 1);
    CHECK(bounded.states.front().payload.restorable());
}

#endif
int main() {
#ifdef VBR_PROMPT_CACHE_PUBLICATION_TEST
    if (server_fault("vbr_prompt_cache_prefix_fail")) {
        test_prompt_cache_vbr_atomic_logical_publication();
        if (failures != 0) {
            fprintf(stderr, "%d VBR prefix rollback test(s) failed\n",
                    failures);
            return 1;
        }
        printf("VBR prompt-cache prefix rollback: PASS\n");
        return 0;
    }
#endif
    test_golden_and_native_lineage();
    test_v1_decode_and_v2_restore_metadata();
    test_identity_and_reference_separation();
    test_fail_closed_decode();
    test_encoder_rejects_source_mutation();
    test_validation_and_ordering();
    test_companion_payload();
    test_stream_larger_than_capture_ring();
    test_catalog_streaming_protocol();
    test_catalog_multi_unit_atomic_publish();
    test_catalog_streaming_companion_lifetime();
    test_catalog_charge_once_and_retire();
    test_catalog_all_shard_failures_and_rollback();
    test_catalog_destructor_releases_live_references();
    test_catalog_dedup_race();
    test_catalog_full_id_interning_and_stash_dedup();
    test_catalog_capacity_sequential_and_temporaries();
    test_catalog_package_lease_and_reference_placement();
    test_prompt_cache_vbr_payload_fanout_lifetime();
    test_prompt_cache_vbr_same_frontier_variants();
    test_manifest_validator_matrix();
    test_validated_manifest_staging();
#ifdef VBR_PROMPT_CACHE_PUBLICATION_TEST
    test_prompt_cache_vbr_atomic_logical_publication();
#endif
    if (failures != 0) {
        fprintf(stderr, "%d VBR artifact test(s) failed\n", failures);
        return 1;
    }
    printf("VBR artifact format: PASS\n");
    return 0;
}
