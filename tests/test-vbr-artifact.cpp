#include "llama-vbr-artifact.h"
#include "llama-vbr-artifact-capture.h"
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
#include "server-cache-authority.h"
#include "server-task.h"
#include "server-retention-sidecar.h"
#endif

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
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
static_assert(std::is_trivially_copyable<
    vbr_capture_projection_segment>::value);

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

    struct shard_completion {
        uint32_t unit_index = UINT32_MAX;
        uint32_t shard_index = UINT32_MAX;
        bool clean_stash = false;
        bool success = true;
        std::vector<uint8_t> bytes;
    };

    std::vector<shard_completion>
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
        const catalog_fixture::shard_completion & completion,
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

static llama_vbr_artifact_publish_result publish_fixture(
        llama_vbr_artifact_catalog & catalog,
        const vbr_artifact_package & package,
        const std::vector<catalog_fixture::shard_completion> & completions,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault = {}) {
    llama_vbr_artifact_publish_result out;
    const uint64_t expected = package.unit_blobs.size() == 1
        ? package.unit_blobs[0].descriptor.shards.size() +
              (package.unit_blobs[0].descriptor.clean_stash_state ==
                       vbr_artifact_clean_stash_state::present
                   ? package.unit_blobs[0].descriptor.clean_stash.shards.size()
                   : 0)
        : 0;
    if (completions.size() != expected) {
        out.status = completions.size() < expected
            ? llama_vbr_artifact_publish_status::missing_completion
            : llama_vbr_artifact_publish_status::duplicate_completion;
        return out;
    }

    vbr_capture_stream_status status;
    auto build = catalog.begin_capture(package, budget, fault, status);
    if (!build) {
        out.status = status ==
                vbr_capture_stream_status::accounting_unavailable
            ? llama_vbr_artifact_publish_status::accounting_unavailable
            : status == vbr_capture_stream_status::accounting_refused
                ? llama_vbr_artifact_publish_status::admission_refused
                : llama_vbr_artifact_publish_status::invalid_argument;
        return out;
    }
    auto unit = build->begin_unit(0, status);
    if (!unit) {
        out.status = llama_vbr_artifact_publish_status::invalid_argument;
        return out;
    }
    for (const auto & completion : completions) {
        if (completion.unit_index != 0) {
            out.status = llama_vbr_artifact_publish_status::invalid_argument;
            return out;
        }
        if (!completion.success) {
            out.status = llama_vbr_artifact_publish_status::shard_failed;
            return out;
        }
        const auto accepted =
            unit->accept_verified_segment(verified_segment(completion));
        if (accepted != vbr_capture_stream_status::ok) {
            out.status = accepted ==
                    vbr_capture_stream_status::duplicate_segment
                ? llama_vbr_artifact_publish_status::duplicate_completion
                : llama_vbr_artifact_publish_status::invalid_argument;
            return out;
        }
    }
    status = unit->seal_unit();
    if (status != vbr_capture_stream_status::ok) {
        out.status = status == vbr_capture_stream_status::missing_segment
            ? llama_vbr_artifact_publish_status::missing_completion
            : llama_vbr_artifact_publish_status::invalid_argument;
        return out;
    }

    const auto streamed = build->publish_reference();
    out.reference_artifact = streamed.reference_artifact;
    out.unit_content = streamed.unit_content;
    out.reference_lineage = streamed.reference_lineage;
    switch (streamed.status) {
        case vbr_capture_stream_status::ok:
            out.status = streamed.adopted
                ? llama_vbr_artifact_publish_status::adopted
                : llama_vbr_artifact_publish_status::published;
            break;
        case vbr_capture_stream_status::format_rejected:
        case vbr_capture_stream_status::hash_mismatch:
            out.status = llama_vbr_artifact_publish_status::format_rejected;
            break;
        case vbr_capture_stream_status::accounting_unavailable:
            out.status =
                llama_vbr_artifact_publish_status::accounting_unavailable;
            break;
        case vbr_capture_stream_status::accounting_refused:
            out.status = llama_vbr_artifact_publish_status::admission_refused;
            break;
        case vbr_capture_stream_status::stage_failed:
            out.status = llama_vbr_artifact_publish_status::stage_failed;
            break;
        case vbr_capture_stream_status::commit_failed:
            out.status = llama_vbr_artifact_publish_status::commit_failed;
            break;
        case vbr_capture_stream_status::publication_failed:
            out.status = llama_vbr_artifact_publish_status::publication_failed;
            break;
        case vbr_capture_stream_status::duplicate_segment:
            out.status = llama_vbr_artifact_publish_status::duplicate_completion;
            break;
        case vbr_capture_stream_status::missing_segment:
            out.status = llama_vbr_artifact_publish_status::missing_completion;
            break;
        case vbr_capture_stream_status::transfer_failed:
        case vbr_capture_stream_status::short_read:
            out.status = llama_vbr_artifact_publish_status::shard_failed;
            break;
        default:
            out.status = llama_vbr_artifact_publish_status::internal_error;
            break;
    }
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
    const auto fake_first = publish_fixture(*fake_equivalent.catalog,
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
    const auto fake_after_refusal = publish_fixture(*overlap.catalog,
        overlap.package, overlap.completions(),
        generous_overlap_budget);
    CHECK(fake_after_refusal.status ==
          llama_vbr_artifact_publish_status::published);

    catalog_fixture unconfigured(false);
    const auto unavailable = publish_fixture(*unconfigured.catalog,
        unconfigured.package, unconfigured.completions(),
        unconfigured.budget);
    CHECK(unavailable.status ==
          llama_vbr_artifact_publish_status::
              admission_refused);
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
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
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
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
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

static void test_catalog_owned_union_retirement() {
    catalog_fixture f;
    const auto first =
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
    const auto second =
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
    CHECK(first.status == llama_vbr_artifact_publish_status::published);
    CHECK(second.status == llama_vbr_artifact_publish_status::adopted);

    vbr_artifact_package_view first_view;
    vbr_artifact_package_view second_view;
    CHECK(f.catalog->resolve_reference(
              first.reference_artifact, first_view) ==
          vbr_artifact_resolve_status::ok);
    CHECK(f.catalog->resolve_reference(
              second.reference_artifact, second_view) ==
          vbr_artifact_resolve_status::ok);
    CHECK(first_view.claim_host_ownership());
    CHECK(second_view.claim_host_ownership());
    CHECK(f.catalog->retire(first.reference_artifact) ==
          vbr_artifact_retire_status::busy);

    const auto before = f.ledger.snapshot();
    uint64_t expected_logical = 0;
    uint64_t expected_resident = 0;
    for (const auto & allocation : before.allocations) {
        CHECK(allocation.logical_bytes <=
              UINT64_MAX - expected_logical);
        CHECK(allocation.resident_bytes <=
              UINT64_MAX - expected_resident);
        expected_logical += allocation.logical_bytes;
        expected_resident += allocation.resident_bytes;
    }

    vbr_artifact_prepared_retire prepared;
    CHECK(first_view.prepare_owned_retire(
        { &first_view, &second_view }, before.serial, prepared));
    CHECK(prepared.ready());
    uint64_t projected_logical = 0;
    uint64_t projected_resident = 0;
    for (const auto & row : prepared.preview().rows) {
        CHECK(row.logical_payload <= UINT64_MAX - projected_logical);
        CHECK(row.resident_allocated <= UINT64_MAX - projected_resident);
        projected_logical += row.logical_payload;
        projected_resident += row.resident_allocated;
    }
    // Two references share the unit/stash allocations. The prepared union
    // releases those bytes once while retaining both reference leaves.
    CHECK(projected_logical == expected_logical);
    CHECK(projected_resident == expected_resident);

    // Abandoning a quote before the logical owners move is a no-op. The same
    // exact reference set can be prepared again at the unchanged serial.
    prepared.reset();
    CHECK(first_view.prepare_owned_retire(
        { &second_view, &first_view }, before.serial, prepared));
    CHECK(prepared.ready());
    CHECK(f.ledger.snapshot().live_ops == before.live_ops);

    first_view.reset();
    second_view.reset();
    CHECK(f.ledger.snapshot().live_ops == before.live_ops);
    CHECK(prepared.commit() ==
          vbr_artifact_prepared_retire_status::retired);
    CHECK(f.ledger.snapshot().live_ops == 0);
    CHECK(f.catalog->snapshot().references == 0);
    CHECK(f.catalog->resolve_reference(
              first.reference_artifact, first_view) ==
          vbr_artifact_resolve_status::not_found);

    // The server wrapper carries the same ownership capability without
    // exposing catalog internals. It is releasable only when the logical
    // payload is the sole immutable owner.
    const auto third =
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
    CHECK(third.status == llama_vbr_artifact_publish_status::published);
    vbr_artifact_package_view third_view;
    CHECK(f.catalog->resolve_reference(
              third.reference_artifact, third_view) ==
          vbr_artifact_resolve_status::ok);
    auto owner = server_prompt_cache_vbr_payload::adopt_owned(
        std::move(third_view));
    CHECK(owner);
    auto payload = server_prompt_cache_payload::from_vbr(owner);
    CHECK(payload.vbr_retirement_owned());
    CHECK(!payload.vbr_retirement_exclusive());
    owner.reset();
    CHECK(payload.vbr_retirement_exclusive());
    vbr_artifact_prepared_retire payload_retire;
    CHECK(payload.prepare_vbr_retire(
        f.ledger.serial(), payload_retire));
    const auto unrelated = f.ledger.reserve(
        llama_cache_acct_category::artifact_reference_metadata,
        f.host, {}, 0, 0);
    CHECK(unrelated);
    CHECK(f.ledger.abort(unrelated));
    payload = {};
    CHECK(payload_retire.commit() ==
          vbr_artifact_prepared_retire_status::retired_projection_stale);
    CHECK(f.ledger.snapshot().live_ops == 0);

    // A host-owned reference that still shares its segment allocations with
    // an independent catalog reference quotes only the metadata it will
    // truly free. Logical payload size is deliberately not victim currency.
    const auto shared_host =
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
    const auto shared_control =
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
    CHECK(shared_host.status ==
          llama_vbr_artifact_publish_status::published);
    CHECK(shared_control.status ==
          llama_vbr_artifact_publish_status::adopted);
    vbr_artifact_package_view shared_view;
    CHECK(f.catalog->resolve_reference(
              shared_host.reference_artifact, shared_view) ==
          vbr_artifact_resolve_status::ok);
    auto shared_owner = server_prompt_cache_vbr_payload::adopt_owned(
        std::move(shared_view));
    CHECK(shared_owner);
    auto shared_payload = server_prompt_cache_payload::from_vbr(
        shared_owner);
    shared_owner.reset();
    vbr_artifact_prepared_retire marginal;
    CHECK(shared_payload.prepare_vbr_retire(
        f.ledger.serial(), marginal));
    uint64_t marginal_resident = 0;
    for (const auto & row : marginal.preview().rows) {
        CHECK(row.resident_allocated <=
              UINT64_MAX - marginal_resident);
        marginal_resident += row.resident_allocated;
    }
    CHECK(marginal_resident > 0);
    CHECK(marginal_resident < shared_payload.size());
    shared_payload = {};
    CHECK(marginal.commit() ==
          vbr_artifact_prepared_retire_status::retired);
    CHECK(f.catalog->snapshot().references == 1);
    CHECK(f.ledger.snapshot().live_ops > 0);
    CHECK(f.catalog->retire(shared_control.reference_artifact) ==
          vbr_artifact_retire_status::retired);
    CHECK(f.ledger.snapshot().live_ops == 0);

    // An early commit is a harmless refusal: it returns the claim token so
    // the later last-owner drop can still execute its cleanup terminal.
    const auto abandoned =
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
    CHECK(abandoned.status ==
          llama_vbr_artifact_publish_status::published);
    vbr_artifact_package_view abandoned_view;
    CHECK(f.catalog->resolve_reference(
              abandoned.reference_artifact, abandoned_view) ==
          vbr_artifact_resolve_status::ok);
    auto abandoned_owner = server_prompt_cache_vbr_payload::adopt_owned(
        std::move(abandoned_view));
    auto abandoned_payload = server_prompt_cache_payload::from_vbr(
        abandoned_owner);
    abandoned_owner.reset();
    vbr_artifact_prepared_retire abandoned_retire;
    CHECK(abandoned_payload.prepare_vbr_retire(
        f.ledger.serial(), abandoned_retire));
    CHECK(abandoned_retire.commit() ==
          vbr_artifact_prepared_retire_status::unavailable);
    CHECK(f.catalog->snapshot().references == 1);
    abandoned_payload = {};
    CHECK(f.catalog->snapshot().references == 0);
    CHECK(f.ledger.snapshot().live_ops == 0);

    const auto unprepared =
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
    CHECK(unprepared.status ==
          llama_vbr_artifact_publish_status::published);
    vbr_artifact_package_view unprepared_view;
    CHECK(f.catalog->resolve_reference(
              unprepared.reference_artifact, unprepared_view) ==
          vbr_artifact_resolve_status::ok);
    auto unprepared_owner = server_prompt_cache_vbr_payload::adopt_owned(
        std::move(unprepared_view));
    CHECK(unprepared_owner);
    unprepared_owner.reset();
    CHECK(f.catalog->snapshot().references == 0);
    CHECK(f.ledger.snapshot().live_ops == 0);

    const auto cancelled =
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
    CHECK(cancelled.status ==
          llama_vbr_artifact_publish_status::published);
    vbr_artifact_package_view cancelled_view;
    CHECK(f.catalog->resolve_reference(
              cancelled.reference_artifact, cancelled_view) ==
          vbr_artifact_resolve_status::ok);
    auto cancelled_owner = server_prompt_cache_vbr_payload::adopt_owned(
        std::move(cancelled_view));
    auto cancelled_payload = server_prompt_cache_payload::from_vbr(
        cancelled_owner);
    cancelled_owner.reset();
    vbr_artifact_prepared_retire cancelled_retire;
    CHECK(cancelled_payload.prepare_vbr_retire(
        f.ledger.serial(), cancelled_retire));
    // Once the owner has gone, capability cancellation becomes the cleanup
    // terminal rather than leaking a claimed reference into teardown.
    cancelled_payload = {};
    cancelled_retire.reset();
    CHECK(f.catalog->snapshot().references == 0);
    CHECK(f.ledger.snapshot().live_ops == 0);

    const auto partial_a =
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
    const auto partial_b =
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
    CHECK(partial_a.status ==
          llama_vbr_artifact_publish_status::published);
    CHECK(partial_b.status ==
          llama_vbr_artifact_publish_status::adopted);
    vbr_artifact_package_view partial_a_view;
    vbr_artifact_package_view partial_b_view;
    CHECK(f.catalog->resolve_reference(
              partial_a.reference_artifact, partial_a_view) ==
          vbr_artifact_resolve_status::ok);
    CHECK(f.catalog->resolve_reference(
              partial_b.reference_artifact, partial_b_view) ==
          vbr_artifact_resolve_status::ok);
    CHECK(partial_a_view.claim_host_ownership());
    CHECK(partial_b_view.claim_host_ownership());
    vbr_artifact_prepared_retire partial_retire;
    CHECK(partial_a_view.prepare_owned_retire(
        { &partial_a_view, &partial_b_view },
        f.ledger.serial(), partial_retire));
    partial_a_view.reset();
    partial_retire.reset();
    CHECK(f.catalog->snapshot().references == 1);
    CHECK(partial_b_view);
    CHECK(partial_b_view.validate() == vbr_artifact_status::ok);
    CHECK(f.ledger.snapshot().live_ops > 0);
    partial_b_view.reset();
    CHECK(f.catalog->snapshot().references == 0);
    CHECK(f.ledger.snapshot().live_ops == 0);
}

static void test_catalog_all_shard_failures_and_rollback() {
    for (uint32_t mode = 0; mode < 6; ++mode) {
        catalog_fixture f;
        auto completions = f.completions();
        llama_cache_transaction_fault fault;
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
            publish_fixture(*f.catalog, f.package, completions, f.budget, fault);
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
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
    const auto second =
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
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
            publish_fixture(*f.catalog, f.package, completions, f.budget);
    });
    std::thread b([&] {
        results[1] =
            publish_fixture(*f.catalog, f.package, completions, f.budget);
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
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
    CHECK(first.status ==
          llama_vbr_artifact_publish_status::published);

    fixture_storage changed_storage;
    changed_storage.payload0.bytes[0] ^= 1;
    auto changed = make_package(changed_storage);
    const auto second = publish_fixture(*f.catalog, changed, {
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
        publish_fixture(*f.catalog, f.package, f.completions(), f.budget);
    CHECK(first.status ==
          llama_vbr_artifact_publish_status::published);

    fixture_storage changed_storage;
    changed_storage.payload0.bytes[0] ^= 1;
    auto changed = make_package(changed_storage);
    const auto refused =
        publish_fixture(*f.catalog, changed, {
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
    const auto first = publish_fixture(*f.catalog,
        f.package, f.completions(), f.budget);
    CHECK(first.status == llama_vbr_artifact_publish_status::published);

    auto alternate = f.package;
    alternate.manifest.generation.controllers[0].streams[0]
        .pages[0].covered_mask[0] = 0x5;
    alternate.manifest.stream_placements[0].cells[1].physical_cell = 2;
    alternate.manifest.unit_references[0].stash_reference
        .covered_sink_pages[0].covered_mask[0] = 0x5;
    alternate.manifest.manifest_digest = {};
    const auto second = publish_fixture(*f.catalog,
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
    const auto published = publish_fixture(*raced.catalog,
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
    const auto identity_a_published = publish_fixture(*identity_a.catalog,
        identity_a.package, identity_a.completions(), identity_a.budget);
    const auto identity_b_published = publish_fixture(*identity_b.catalog,
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
    const auto compact_reference = publish_fixture(*f.catalog,
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
    const auto anchor_reference = publish_fixture(*f.catalog,
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

    // The anchor planner reasons over the global allocation-ID union, not
    // each variant's standalone size. Two logical parents sharing the same
    // anchor release zero bytes individually; after the stable first drop,
    // the survivor becomes the exact physical release candidate.
    const uint64_t shared_anchor_bytes =
        typed.vbr_anchor_resident_bytes();
    CHECK(shared_anchor_bytes > 0);
    std::vector<server_prompt_cache_vbr_anchor_plan_candidate>
        shared_anchor_candidates {
            { &typed, { 101 }, 10, 7, 0, 11, true },
            { &independently_bundled, { 102 }, 10, 7, 0, 12, true },
        };
    std::vector<llama_cache_acct_artifact_id> shared_anchor_plan;
    CHECK(server_prompt_cache_plan_vbr_anchor_releases(
        shared_anchor_candidates, shared_anchor_bytes,
        shared_anchor_bytes - 1, shared_anchor_plan));
    CHECK(shared_anchor_plan.size() == 2);
    CHECK(shared_anchor_plan[0].v == 101);
    CHECK(shared_anchor_plan[1].v == 102);

    // Max logical alias cardinality stays one allocation-union build plus a
    // bounded refcount/ordered-set walk; it must not regress to one global
    // allocation sort per retired parent.
    std::vector<server_prompt_cache_vbr_anchor_plan_candidate>
        max_anchor_candidates;
    max_anchor_candidates.reserve(
        SERVER_PROMPT_CACHE_VBR_ANCHOR_MAX_CANDIDATES);
    for (uint64_t i = 0;
         i < SERVER_PROMPT_CACHE_VBR_ANCHOR_MAX_CANDIDATES; ++i) {
        max_anchor_candidates.push_back({
            &typed, { 1000 + i }, 10, 7, 0, 1000 + i, true,
        });
    }
    CHECK(server_prompt_cache_plan_vbr_anchor_releases(
        max_anchor_candidates, shared_anchor_bytes, 0,
        shared_anchor_plan));
    CHECK(shared_anchor_plan.size() ==
          SERVER_PROMPT_CACHE_VBR_ANCHOR_MAX_CANDIDATES);

    // Equal, lower-quality, or reversed roles are not anchors.
    const auto equal_reference = publish_fixture(*f.catalog,
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
    const auto lower_reference = publish_fixture(*f.catalog,
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
    const auto lossy_anchor_reference = publish_fixture(*f.catalog,
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
    const auto other_reference = publish_fixture(*other.catalog,
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
    const auto different_reference = publish_fixture(*f.catalog,
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
            const auto published = publish_fixture(*base.catalog,
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
    const uint64_t legacy_live_ops =
        fixture.base.ledger.snapshot().live_ops - baseline_ops;
    staged.staged.reset();
    CHECK(fixture.base.ledger.snapshot().live_ops == baseline_ops);
    CHECK(staged.manifest && staged.manifest->adoption_nonce() == nonce);

    // A server-owned ring is already charged and remains physically stable
    // across staging. It therefore removes exactly the per-request pinned
    // transfer claim while retaining the device transfer reservations.
    std::vector<vbr_pinned_ring_lane> persistent_lanes;
    for (const auto & lane : policy.lanes) {
        persistent_lanes.push_back({
            lane.device, lane.backend, lane.force_synchronous,
        });
    }
    vbr_pinned_ring_accounting persistent_accounting {
        &fixture.base.ledger, policy.pinned_domain, &budget,
    };
    vbr_pinned_ring_create_failure persistent_failure;
    auto persistent_core =
        std::shared_ptr<vbr_bounded_pinned_ring_core>(
            vbr_bounded_pinned_ring_core::create(
                persistent_lanes, policy.pinned_ring_bytes,
                policy.chunk_bytes, &persistent_accounting,
                persistent_failure));
    CHECK(persistent_core);
    CHECK(persistent_failure == vbr_pinned_ring_create_failure::none);
    policy.persistent_ring = vbr_h2d_chunk_ring::attach(
        std::move(persistent_core), policy.lanes);
    CHECK(policy.persistent_ring);
    fixture.accounting = fixture.base.ledger.snapshot();
    fixture.target.accounting_serial = fixture.accounting.serial;
    fixture.serials.accounting = fixture.accounting.serial;
    fixture.policy.accounting_snapshot = &fixture.accounting;
    auto persistent_proof = validate(
        fixture, fixture.target, fixture.policy);
    CHECK(persistent_proof.proof);
    const uint64_t persistent_capacity =
        vbr_pinned_ring_live_capacity_bytes();
    const uint64_t persistent_baseline =
        fixture.base.ledger.snapshot().live_ops;
    auto persistent_staged = vbr_stage_validated_manifest(
        std::move(persistent_proof.proof), policy);
    CHECK(persistent_staged.status == vbr_adopt_stage_status::staged);
    CHECK(persistent_staged.staged);
    CHECK(vbr_pinned_ring_live_capacity_bytes() == persistent_capacity);
    CHECK(fixture.base.ledger.snapshot().live_ops >= persistent_baseline);
    const uint64_t persistent_live_ops =
        fixture.base.ledger.snapshot().live_ops - persistent_baseline;
    CHECK(persistent_live_ops + 1 == legacy_live_ops);
    persistent_staged.staged.reset();
    CHECK(fixture.base.ledger.snapshot().live_ops == persistent_baseline);
    policy.persistent_ring.reset();

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

    vbr_pinned_ring_accounting lane_accounting {
        &fixture.base.ledger, policy.pinned_domain, &budget,
    };
    auto lane_core = std::shared_ptr<vbr_bounded_pinned_ring_core>(
        vbr_bounded_pinned_ring_core::create(
            persistent_lanes, policy.pinned_ring_bytes,
            policy.chunk_bytes, &lane_accounting,
            persistent_failure));
    CHECK(lane_core);
    auto lane_mismatch = policy;
    lane_mismatch.persistent_ring = vbr_h2d_chunk_ring::attach(
        std::move(lane_core), policy.lanes);
    CHECK(lane_mismatch.persistent_ring);
    lane_mismatch.lanes.front().domain =
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pageable_host);
    // If compatibility moves below source verification, this becomes the
    // wrong source_hash_mismatch terminal instead of ring_unavailable.
    lane_mismatch.fault.fail_source_verify_at = 0;
    run_failure(lane_mismatch, vbr_adopt_stage_status::ring_unavailable);
    lane_mismatch.persistent_ring.reset();

    // A persistent adapter's cached owner identity is not enough: the live
    // ledger row must still carry the complete physical charge. Refreshing
    // validation after a corrupted zero gauge must remain fail closed.
    auto drift_core = std::shared_ptr<vbr_bounded_pinned_ring_core>(
        vbr_bounded_pinned_ring_core::create(
            persistent_lanes, policy.pinned_ring_bytes,
            policy.chunk_bytes, &lane_accounting,
            persistent_failure));
    CHECK(drift_core);
    auto charged_drift = policy;
    charged_drift.persistent_ring = vbr_h2d_chunk_ring::attach(
        std::move(drift_core), policy.lanes);
    CHECK(charged_drift.persistent_ring);
    fixture.base.ledger.gauge_set(
        llama_cache_acct_category::pinned_preimage_ring,
        policy.pinned_domain,
        llama_cache_acct_measure::logical_payload, 0);
    fixture.base.ledger.gauge_set(
        llama_cache_acct_category::pinned_preimage_ring,
        policy.pinned_domain,
        llama_cache_acct_measure::resident_allocated, 0);
    run_failure(
        charged_drift, vbr_adopt_stage_status::ring_unavailable);
    charged_drift.persistent_ring.reset();

    vbr_h2d_status unaccounted_status;
    policy.persistent_ring = std::shared_ptr<vbr_h2d_chunk_ring>(
        vbr_h2d_chunk_ring::create(
            policy.lanes, policy.pinned_ring_bytes,
            policy.chunk_bytes, unaccounted_status));
    CHECK(policy.persistent_ring);
    CHECK(unaccounted_status == vbr_h2d_status::ok);
    run_failure(policy, vbr_adopt_stage_status::ring_unavailable);
    policy.persistent_ring.reset();

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

    const auto published = publish_fixture(*fixture.catalog,
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
    CHECK(cache.contains_vbr_frontier(
        prompt,
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity));
    server_prompt wrong_vbr_frontier = prompt.clone();
    wrong_vbr_frontier.sequence_epoch++;
    CHECK(!cache.contains_vbr_frontier(
        wrong_vbr_frontier,
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity));
    CHECK(logical->payload.vbr_artifact() == owner.get());
    CHECK(logical->release_ops().empty());
    // The catalog already owns every physical VBR allocation. Logical host
    // publication adds only its retention metadata leaf.
    CHECK(fixture.ledger.snapshot().live_ops == live_ops_before + 1);
    auto host_key =
        server_retention_instance_key::for_host_entry(&*logical);
    CHECK(retention.artifact_id(host_key).v != 0);

    // Live rewind checkpoints are not part of the projected artifact. Their
    // presence must not exclude an ordinary-hybrid source, and the detached
    // VBR node must retain no fixed checkpoint bytes.
    server_prompt checkpoint_source = prompt.clone();
    checkpoint_source.checkpoints.emplace_back();
    checkpoint_source.checkpoints.back().data_tgt.overwrite(
        4, [](uint8_t * data, size_t size) {
            std::fill_n(data, size, uint8_t(9));
        });
    auto checkpoint_stage = cache.stage_vbr(
        checkpoint_source, payload,
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(checkpoint_stage.size() == 1);
    CHECK(checkpoint_stage.front().prompt.checkpoints.empty());

    // Mixed fixed/VBR sizing must preserve the fixed cache's physical-union
    // accounting. The host checkpoint aliases an independently accounted
    // live checkpoint, so the host contributes only its 32-byte snapshot;
    // the shared 100-byte plane is not charged a second time.
    const uint64_t mixed_ops_before = fixture.ledger.snapshot().live_ops;
    common_prompt_checkpoint shared_live;
    shared_live.data_tgt.overwrite(100, [](uint8_t * data, size_t size) {
        std::fill_n(data, size, uint8_t(7));
    });
    const auto shared_allocation = fixture.ledger.new_alloc();
    CHECK(shared_allocation);
    const auto live_checkpoint_op = fixture.ledger.reserve(
        llama_cache_acct_category::checkpoint_state_payload,
        fixture.host, {}, 100, 100);
    CHECK(live_checkpoint_op);
    CHECK(fixture.ledger.stage(
        live_checkpoint_op, shared_allocation, 100));
    CHECK(fixture.ledger.commit(live_checkpoint_op, 100));
    CHECK(shared_live.data_tgt.bind_accounting(
        &fixture.ledger, shared_allocation.v));

    server_prompt_cache_state shared_fixed;
    shared_fixed.payload.fixed_state()->main.assign(32, 3);
    shared_fixed.prompt.checkpoints.push_back(shared_live);
    const auto fixed_snapshot_allocation = fixture.ledger.new_alloc();
    CHECK(fixed_snapshot_allocation);
    const auto fixed_snapshot_op = fixture.ledger.reserve(
        llama_cache_acct_category::full_snapshot_payload,
        fixture.host, {}, 32, 32);
    CHECK(fixed_snapshot_op);
    CHECK(fixture.ledger.stage(
        fixed_snapshot_op, fixed_snapshot_allocation, 32));
    CHECK(fixture.ledger.commit(fixed_snapshot_op, 32));
    const auto host_checkpoint_op = fixture.ledger.reserve(
        llama_cache_acct_category::checkpoint_state_payload,
        fixture.host, {}, 100, 0);
    CHECK(host_checkpoint_op);
    CHECK(fixture.ledger.stage(
        host_checkpoint_op, shared_allocation, 100));
    CHECK(fixture.ledger.commit(host_checkpoint_op, 100));
    CHECK(shared_fixed.prompt.checkpoints.front().data_tgt.bind_accounting(
        &fixture.ledger, shared_allocation.v));
    shared_fixed.acct_ops = { fixed_snapshot_op, host_checkpoint_op };
    shared_fixed.accounting_complete = true;

    server_prompt_cache mixed(/* limit_size_mib */ 0,
                              /* limit_tokens */ 0);
    server_cache_authority mixed_authority;
    mixed.acct = &fixture.ledger;
    mixed.publish_authority = &mixed_authority;
    mixed.retention_obs = &retention;
    mixed.states.push_back(std::move(shared_fixed));
    CHECK(mixed.size() == 32);
    CHECK(fixture.ledger.snapshot().live_ops == mixed_ops_before + 3);
    const uint64_t mixed_exact_cap = uint64_t(owner->resident_bytes()) + 32;
    mixed.limit_size = mixed_exact_cap;
    auto mixed_vbr = mixed.stage_vbr(
        prompt, payload,
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(mixed_vbr.size() == 1);
    CHECK(mixed.publish(
        std::move(mixed_vbr), &prompt, source_slot));
    CHECK(mixed.states.size() == 2);
    CHECK(mixed.size() == mixed_exact_cap);
    CHECK(fixture.ledger.snapshot().live_ops == mixed_ops_before + 4);
    mixed.limit_size = mixed_exact_cap - 1;
    mixed.update();
    CHECK(mixed.states.size() == 1);
    CHECK(mixed.states.front().payload.vbr_artifact() == owner.get());
    CHECK(mixed.size() == owner->resident_bytes());
    CHECK(fixture.ledger.snapshot().live_ops == mixed_ops_before + 2);
    mixed.clear_accounting();
    mixed.states.clear();
    CHECK(fixture.ledger.snapshot().live_ops == mixed_ops_before + 1);
    CHECK(shared_live.data_tgt.unbind_accounting(
        &fixture.ledger, shared_allocation.v));
    CHECK(fixture.ledger.release(live_checkpoint_op));
    CHECK(fixture.ledger.snapshot().live_ops == mixed_ops_before);

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

    // A borrowed/manual VBR view carries no cache-owned retirement authority.
    // Pressure must preserve it rather than erase a logical node while its
    // catalog allocation remains owned elsewhere.
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

    // A borrowed/manual publication cannot displace another host entry; only
    // the exclusive cache-owned capability exercised below may do that.
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

static vbr_artifact_package prompt_cache_quality_anchor_package(
        const vbr_artifact_package & source) {
    auto result = source;
    auto & descriptor = result.unit_blobs[0].descriptor;
    descriptor.repr_gen++;
    descriptor.current_type = GGML_TYPE_TURBO8_0;
    descriptor.last_source_type = GGML_TYPE_F16;
    descriptor.promote_hops = 0;
    descriptor.last_transition =
        vbr_repr_transition::degrade_f16_to_t8_admitted;
    descriptor.representation.source_loss_history = 0;
    auto & generation =
        result.manifest.generation.controllers[0].units[0];
    generation.repr_gen = descriptor.repr_gen;
    generation.current_type = descriptor.current_type;
    generation.last_source_type = descriptor.last_source_type;
    generation.domain = vbr_downward_tier_domain(
        ggml_type(descriptor.current_type));
    generation.promote_hops = descriptor.promote_hops;
    generation.last_transition = descriptor.last_transition;
    result.manifest.unit_references[0].repr_gen = descriptor.repr_gen;
    const ggml_type anchor_types[] = { GGML_TYPE_TURBO8_0 };
    result.manifest.controller_policy[0].current_type_vector_digest =
        vbr_type_vector_digest(anchor_types, 1);
    result.manifest.manifest_digest = {};
    result.manifest.capture_generation_id = {};
    result.manifest.consistency = {};
    return result;
}

static void test_prompt_cache_vbr_pressure_retires_physical_union() {
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

    const auto first = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    const auto second = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    CHECK(first.status == llama_vbr_artifact_publish_status::published);
    CHECK(second.status == llama_vbr_artifact_publish_status::adopted);

    const auto owned_payload = [&](llama_cache_acct_artifact_id reference) {
        vbr_artifact_package_view view;
        CHECK(fixture.catalog->resolve_reference(reference, view) ==
              vbr_artifact_resolve_status::ok);
        auto owner = server_prompt_cache_vbr_payload::adopt_owned(
            std::move(view));
        CHECK(owner);
        return server_prompt_cache_payload::from_vbr(std::move(owner));
    };

    server_cache_authority authority;
    server_retention_sidecar_store retention;
    retention.configure(
        &fixture.ledger, fixture.host, &authority.leases);
    CHECK(retention.enable_prefix_tracking());
    constexpr int32_t source_slot = 9;
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
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;
    cache.retention_obs = &retention;
    cache.lease_obs = &authority.leases;
    cache.lease_execution_identity =
        &fixture.package.manifest.identity.execution_identity;
    cache.retention_df2_authority = true;
    CHECK(cache.enable_retention_shadow());
    const auto publish_owned = [&](llama_cache_acct_artifact_id reference) {
        auto payload = owned_payload(reference);
        auto staged = cache.stage_vbr(
            prompt, std::move(payload),
            fixture.package.manifest.identity.execution_identity,
            fixture.package.manifest.identity.adapter_config_identity);
        CHECK(staged.size() == 1);
        server_prompt_cache::iterator result;
        CHECK(cache.publish(
            std::move(staged), &prompt, source_slot, &result));
        return result;
    };

    auto first_state = publish_owned(first.reference_artifact);
    auto second_state = publish_owned(second.reference_artifact);
    CHECK(cache.states.size() == 2);
    CHECK(first_state->payload.vbr_retirement_exclusive());
    CHECK(second_state->payload.vbr_retirement_exclusive());

    std::vector<const server_prompt_cache_payload *> payloads {
        &first_state->payload,
        &second_state->payload,
    };
    llama_cache_acct_release_set_preview union_preview;
    CHECK(server_prompt_cache_payload::preview_vbr_retire_union(
        payloads, fixture.ledger.serial(), union_preview));
    uint64_t union_bytes = 0;
    for (const auto & row : union_preview.rows) {
        CHECK(row.resident_allocated <= UINT64_MAX - union_bytes);
        union_bytes += row.resident_allocated;
    }
    CHECK(union_bytes > 1);
    CHECK(union_bytes <
          first_state->size() + second_state->size());
    vbr_artifact_prepared_retire first_retire;
    CHECK(first_state->payload.prepare_vbr_retire(
        fixture.ledger.serial(), first_retire));
    uint64_t first_marginal = 0;
    for (const auto & row : first_retire.preview().rows) {
        CHECK(row.resident_allocated <= UINT64_MAX - first_marginal);
        first_marginal += row.resident_allocated;
    }
    CHECK(first_marginal > 0 && first_marginal < union_bytes);
    first_retire.reset();

    // The configured byte bound observes the exact physical union, not the
    // naïve sum of two logical manifests that share sealed segments.
    cache.limit_size = size_t(union_bytes);
    cache.update();
    CHECK(cache.states.size() == 2);

    const auto first_key =
        server_retention_instance_key::for_host_entry(&*first_state);
    const auto second_key =
        server_retention_instance_key::for_host_entry(&*second_state);
    cache.limit_size = size_t(union_bytes - 1);
    cache.update();
    CHECK(cache.states.size() == 1);
    CHECK(&cache.states.front() == &*second_state);
    CHECK(retention.artifact_id(first_key).v == 0);
    CHECK(retention.artifact_id(second_key).v != 0);
    CHECK(fixture.catalog->snapshot().references == 1);
    CHECK(authority.destruction.host_trade_df2_executed == 1);
    CHECK(authority.destruction.host_trade_legacy_fallbacks == 0);
    CHECK(cache.retention_shadow_snapshot().last.proposed_resource ==
          first_marginal);

    vbr_artifact_prepared_retire remaining;
    CHECK(cache.states.front().payload.prepare_vbr_retire(
        fixture.ledger.serial(), remaining));
    uint64_t remaining_bytes = 0;
    for (const auto & row : remaining.preview().rows) {
        CHECK(row.resident_allocated <= UINT64_MAX - remaining_bytes);
        remaining_bytes += row.resident_allocated;
    }
    CHECK(remaining_bytes > 0);
    remaining.reset();
    cache.limit_size = size_t(remaining_bytes - 1);
    cache.update();
    CHECK(cache.states.empty());
    CHECK(retention.artifact_id(second_key).v == 0);
    CHECK(fixture.catalog->snapshot().references == 0);
    CHECK(authority.destruction.host_trade_df2_executed == 2);
    CHECK(authority.destruction.host_trade_legacy_fallbacks == 0);

    // Token pressure uses the logical frontier as its denominator while the
    // same prepared capability still owns the physical retirement terminal.
    const auto token_reference = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    CHECK(token_reference.status ==
          llama_vbr_artifact_publish_status::published);
    cache.limit_size = 0;
    cache.limit_tokens = 0;
    auto token_state = publish_owned(token_reference.reference_artifact);
    CHECK(token_state != cache.states.end());
    cache.limit_tokens = size_t(prompt.n_tokens() - 1);
    cache.update();
    CHECK(cache.states.empty());
    CHECK(fixture.catalog->snapshot().references == 0);
    CHECK(authority.destruction.host_trade_df2_executed == 3);
    CHECK(cache.retention_shadow_snapshot().last.reason ==
          server_cache_destruction_reason::host_token_limit);
    CHECK(cache.retention_shadow_snapshot().last.proposed_resource ==
          size_t(prompt.n_tokens()));

    // A recovery pin may legitimately defer exact-alias dedup. Once the pin
    // closes, byte accounting must still charge the shared sealed owner once,
    // and token pressure may erase one logical alias without retiring the
    // physical catalog reference that the survivor still owns.
    const auto alias_reference = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    CHECK(alias_reference.status ==
          llama_vbr_artifact_publish_status::published);
    auto alias_payload = owned_payload(alias_reference.reference_artifact);
    auto alias_payload_copy = alias_payload;
    cache.limit_size = 0;
    cache.limit_tokens = 0;
    auto alias_stage = cache.stage_vbr(
        prompt, std::move(alias_payload),
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(alias_stage.size() == 1);
    server_prompt_cache::iterator pinned_alias;
    CHECK(cache.publish(
        std::move(alias_stage), &prompt, source_slot, &pinned_alias));
    pinned_alias->recovery_pins = 1;
    auto duplicate_stage = cache.stage_vbr(
        prompt, std::move(alias_payload_copy),
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(duplicate_stage.size() == 1);
    server_prompt_cache::iterator duplicate_alias;
    CHECK(cache.publish(
        std::move(duplicate_stage), &prompt, source_slot, &duplicate_alias));
    CHECK(cache.states.size() == 2);
    pinned_alias->recovery_pins = 0;
    CHECK(pinned_alias->payload.vbr_logical_erase_only());
    CHECK(duplicate_alias->payload.vbr_logical_erase_only());

    const size_t alias_bytes = pinned_alias->size();
    cache.limit_size = alias_bytes;
    cache.update();
    CHECK(cache.states.size() == 2);
    CHECK(fixture.catalog->snapshot().references == 1);

    cache.limit_size = 0;
    cache.limit_tokens = size_t(prompt.n_tokens());
    cache.update();
    CHECK(cache.states.size() == 1);
    CHECK(fixture.catalog->snapshot().references == 1);
    CHECK(cache.states.front().payload.vbr_retirement_exclusive());
    CHECK(authority.destruction.host_trade_df2_executed == 3);

    cache.limit_tokens = 0;
    cache.limit_size = cache.states.front().size() - 1;
    cache.update();
    CHECK(cache.states.empty());
    CHECK(fixture.catalog->snapshot().references == 0);
    CHECK(authority.destruction.host_trade_df2_executed == 4);

    // A zero-byte alias is also lawful capacity cleanup: erasing it makes
    // the survivor exclusive, after which the next bounded iteration can
    // prepare and retire the real physical union. A pin still blocks that
    // second terminal until its durable-use window closes.
    const auto capacity_alias_reference = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    CHECK(capacity_alias_reference.reference_artifact.v != 0);
    auto capacity_alias_payload = owned_payload(
        capacity_alias_reference.reference_artifact);
    auto capacity_alias_copy = capacity_alias_payload;
    cache.limit_size = 0;
    auto capacity_alias_stage = cache.stage_vbr(
        prompt, std::move(capacity_alias_payload),
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(capacity_alias_stage.size() == 1);
    server_prompt_cache::iterator protected_alias;
    CHECK(cache.publish(
        std::move(capacity_alias_stage), &prompt, source_slot,
        &protected_alias));
    protected_alias->recovery_pins = 1;
    auto capacity_duplicate_stage = cache.stage_vbr(
        prompt, std::move(capacity_alias_copy),
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(capacity_duplicate_stage.size() == 1);
    CHECK(cache.publish(
        std::move(capacity_duplicate_stage), &prompt, source_slot));
    const auto mixed_unique_reference = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    CHECK(mixed_unique_reference.reference_artifact.v != 0);
    auto mixed_unique_payload = owned_payload(
        mixed_unique_reference.reference_artifact);
    auto mixed_unique_stage = cache.stage_vbr(
        prompt, std::move(mixed_unique_payload),
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(mixed_unique_stage.size() == 1);
    CHECK(cache.publish(
        std::move(mixed_unique_stage), &prompt, source_slot));
    CHECK(cache.states.size() == 3);
    const size_t mixed_union_bytes = cache.size();
    CHECK(mixed_union_bytes > 1);
    cache.limit_size = mixed_union_bytes - 1;
    cache.update();
    CHECK(cache.states.size() == 1);
    CHECK(&cache.states.front() == &*protected_alias);
    CHECK(fixture.catalog->snapshot().references == 1);
    protected_alias->recovery_pins = 0;
    cache.limit_size = protected_alias->size() - 1;
    cache.update();
    CHECK(cache.states.empty());
    CHECK(fixture.catalog->snapshot().references == 0);
    CHECK(authority.destruction.host_trade_df2_executed == 6);

    // The publication transaction itself—not only later maintenance—may
    // reclaim a lawful incumbent. The incoming node is protected until the
    // exact selected-victim retirement commits.
    const auto publication_incumbent = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    const auto publication_incoming = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    CHECK(publication_incumbent.reference_artifact.v != 0);
    CHECK(publication_incoming.reference_artifact.v != 0);
    cache.limit_size = 0;
    auto incumbent_state = publish_owned(
        publication_incumbent.reference_artifact);
    auto incoming_payload = owned_payload(
        publication_incoming.reference_artifact);
    const size_t incoming_bytes = incoming_payload.size();
    auto incoming_stage = cache.stage_vbr(
        prompt, std::move(incoming_payload),
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(incoming_stage.size() == 1);
    cache.limit_size = incoming_bytes;
    server_prompt_cache::iterator published_under_pressure;
    CHECK(cache.publish(
        std::move(incoming_stage), &prompt, source_slot,
        &published_under_pressure));
    CHECK(cache.states.size() == 1);
    CHECK(published_under_pressure != cache.states.end());
    CHECK(&*published_under_pressure != &*incumbent_state);
    CHECK(fixture.catalog->snapshot().references == 1);

    cache.limit_size = published_under_pressure->size() - 1;
    cache.update();
    CHECK(cache.states.empty());
    CHECK(fixture.catalog->snapshot().references == 0);

    // A limit tightened after staging is revalidated at publish. With the
    // incumbent pinned, refusal retires only the detached incoming owner and
    // cannot drain or rebind the durable incumbent/source association.
    const auto pinned_incumbent_ref = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    const auto refused_incoming_ref = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    CHECK(pinned_incumbent_ref.reference_artifact.v != 0);
    CHECK(refused_incoming_ref.reference_artifact.v != 0);
    cache.limit_size = 0;
    auto protected_state = publish_owned(
        pinned_incumbent_ref.reference_artifact);
    protected_state->recovery_pins = 1;
    const auto protected_key =
        server_retention_instance_key::for_host_entry(&*protected_state);
    const auto protected_artifact = retention.artifact_id(protected_key);
    auto refused_payload = owned_payload(
        refused_incoming_ref.reference_artifact);
    auto refused_stage = cache.stage_vbr(
        prompt, std::move(refused_payload),
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(refused_stage.size() == 1);
    cache.limit_size = protected_state->size();
    server_prompt_cache::iterator refused_publish = cache.states.begin();
    CHECK(!cache.publish(
        std::move(refused_stage), &prompt, source_slot,
        &refused_publish));
    CHECK(refused_publish == cache.states.end());
    CHECK(cache.states.size() == 1);
    CHECK(&cache.states.front() == &*protected_state);
    CHECK(retention.artifact_id(protected_key) == protected_artifact);
    CHECK(fixture.catalog->snapshot().references == 1);
    protected_state->recovery_pins = 0;
    cache.limit_size = protected_state->size() - 1;
    cache.update();
    CHECK(cache.states.empty());
    CHECK(fixture.catalog->snapshot().references == 0);

    // Quality anchors have their own sub-budget and competition. Ordinary
    // compact pressure must not smuggle anchor bytes into a compact victim
    // quote or retire the pair as one ordinary entry.
    const auto compact_reference = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    auto anchor_package =
        prompt_cache_quality_anchor_package(fixture.package);
    CHECK(fixture.catalog->configure_accounting(anchor_package));
    const auto anchor_reference = publish_fixture(*fixture.catalog,
        anchor_package, fixture.completions(), fixture.budget);
    CHECK(compact_reference.reference_artifact.v != 0);
    CHECK(anchor_reference.reference_artifact.v != 0);
    vbr_artifact_package_view compact_view;
    vbr_artifact_package_view anchor_view;
    CHECK(fixture.catalog->resolve_reference(
              compact_reference.reference_artifact, compact_view) ==
          vbr_artifact_resolve_status::ok);
    CHECK(fixture.catalog->resolve_reference(
              anchor_reference.reference_artifact, anchor_view) ==
          vbr_artifact_resolve_status::ok);
    auto compact_owner = server_prompt_cache_vbr_payload::adopt_owned(
        std::move(compact_view));
    auto anchor_owner = server_prompt_cache_vbr_payload::adopt_owned(
        std::move(anchor_view));
    CHECK(compact_owner && anchor_owner);
    auto anchor_variants = server_prompt_cache_vbr_variant_set::create(
        std::move(compact_owner), std::move(anchor_owner));
    CHECK(anchor_variants);
    auto anchor_payload = server_prompt_cache_payload::from_vbr_variants(
        std::move(anchor_variants));
    CHECK(anchor_payload.vbr_has_quality_anchor());
    const size_t anchor_payload_bytes = anchor_payload.size();
    const size_t anchor_marginal_bytes = size_t(
        anchor_payload.vbr_anchor_resident_bytes());
    CHECK(anchor_marginal_bytes > 0);
    CHECK(anchor_marginal_bytes < anchor_payload_bytes);
    const size_t compact_payload_bytes =
        anchor_payload_bytes - anchor_marginal_bytes;

    // With the independent backend gate off, preserve the previous total
    // host-cap behavior: anchors are not free bytes and a borrowed staged
    // pair that exceeds the total cap is refused without cache mutation.
    auto rollback_payload = anchor_payload;
    cache.limit_size = 0;
    auto rollback_stage = cache.stage_vbr(
        prompt, std::move(rollback_payload),
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(rollback_stage.size() == 1);
    cache.limit_size = compact_payload_bytes;
    server_prompt_cache::iterator rollback_result = cache.states.begin();
    CHECK(!cache.publish(
        std::move(rollback_stage), &prompt, source_slot, &rollback_result));
    CHECK(rollback_result == cache.states.end());
    CHECK(cache.states.empty());
    CHECK(fixture.catalog->snapshot().references == 2);

    cache.limit_size = 0;
    cache.quality_anchor_budget_enabled = true;
    cache.limit_anchor_size = anchor_marginal_bytes;
    auto anchor_stage = cache.stage_vbr(
        prompt, std::move(anchor_payload),
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(anchor_stage.size() == 1);
    // The full physical set is larger than this ordinary byte cap, but its
    // compact pool fits exactly. Publication must therefore preserve both
    // variants without invoking ordinary DF2.
    cache.limit_size = compact_payload_bytes;
    server_prompt_cache::iterator anchor_state;
    CHECK(cache.publish(
        std::move(anchor_stage), &prompt, source_slot, &anchor_state));
    CHECK(cache.anchor_size() == anchor_marginal_bytes);
    CHECK(cache.size() == anchor_payload_bytes);
    const auto df2_before_anchor =
        authority.destruction.host_trade_df2_executed;
    cache.update();
    CHECK(cache.states.size() == 1);
    CHECK(&cache.states.front() == &*anchor_state);
    CHECK(anchor_state->payload.vbr_has_quality_anchor());
    CHECK(fixture.catalog->snapshot().references == 2);
    CHECK(authority.destruction.host_trade_df2_executed ==
          df2_before_anchor);

    // A protected over-budget incumbent cannot turn publish(false) into a
    // hidden successful insertion. The compact-only incoming node is retired
    // transactionally while the pinned compact+anchor node remains intact.
    const auto refused_compact_reference = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    CHECK(refused_compact_reference.reference_artifact.v != 0);
    auto refused_compact_payload = owned_payload(
        refused_compact_reference.reference_artifact);
    auto refused_compact_stage = cache.stage_vbr(
        prompt, std::move(refused_compact_payload),
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(refused_compact_stage.size() == 1);
    anchor_state->recovery_pins = 1;
    cache.limit_anchor_size = anchor_marginal_bytes - 1;
    const auto anchor_refusals_before = cache.quality_anchor_refusals;
    server_prompt_cache::iterator refused_compact = cache.states.begin();
    CHECK(!cache.publish(
        std::move(refused_compact_stage), &prompt, source_slot,
        &refused_compact));
    CHECK(refused_compact == cache.states.end());
    CHECK(cache.states.size() == 1);
    CHECK(&cache.states.front() == &*anchor_state);
    CHECK(anchor_state->payload.vbr_has_quality_anchor());
    CHECK(fixture.catalog->snapshot().references == 2);
    CHECK(cache.quality_anchor_refusals ==
          anchor_refusals_before + 1);
    anchor_state->recovery_pins = 0;

    // Tightening only the anchor budget retires the quality variant while
    // preserving the compact node, its stable host association, and the
    // ordinary DF2 counter.
    const auto anchor_host_key =
        server_retention_instance_key::for_host_entry(&*anchor_state);
    const auto anchor_host_artifact = retention.artifact_id(anchor_host_key);
    cache.update();
    CHECK(cache.states.size() == 1);
    CHECK(&cache.states.front() == &*anchor_state);
    CHECK(!anchor_state->payload.vbr_has_quality_anchor());
    CHECK(anchor_state->size() == compact_payload_bytes);
    CHECK(cache.anchor_size() == 0);
    CHECK(cache.quality_anchor_retires == 1);
    CHECK(cache.quality_anchor_refusals ==
          anchor_refusals_before + 1);
    CHECK(retention.artifact_id(anchor_host_key) == anchor_host_artifact);
    CHECK(fixture.catalog->snapshot().references == 1);
    CHECK(authority.destruction.host_trade_df2_executed ==
          df2_before_anchor);
    cache.limit_size = 0;
    cache.quality_anchor_budget_enabled = false;
    cache.clear_accounting();
    cache.states.clear();
    CHECK(fixture.catalog->snapshot().references == 0);

    // Two logical nodes can share one immutable variant owner when a pinned
    // incumbent causes publication dedup to retain an exact alias. Anchor
    // pressure must prepare one physical catalog retirement before changing
    // either node, then replace both logical variants and commit it once.
    const auto shared_compact_reference = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    const auto shared_anchor_reference = publish_fixture(*fixture.catalog,
        anchor_package, fixture.completions(), fixture.budget);
    CHECK(shared_compact_reference.reference_artifact.v != 0);
    CHECK(shared_anchor_reference.reference_artifact.v != 0);
    vbr_artifact_package_view shared_compact_view;
    vbr_artifact_package_view shared_anchor_view;
    CHECK(fixture.catalog->resolve_reference(
              shared_compact_reference.reference_artifact,
              shared_compact_view) == vbr_artifact_resolve_status::ok);
    CHECK(fixture.catalog->resolve_reference(
              shared_anchor_reference.reference_artifact,
              shared_anchor_view) == vbr_artifact_resolve_status::ok);
    auto shared_compact_owner =
        server_prompt_cache_vbr_payload::adopt_owned(
            std::move(shared_compact_view));
    auto shared_anchor_owner =
        server_prompt_cache_vbr_payload::adopt_owned(
            std::move(shared_anchor_view));
    auto shared_variants = server_prompt_cache_vbr_variant_set::create(
        std::move(shared_compact_owner), std::move(shared_anchor_owner));
    CHECK(shared_variants);
    auto shared_first_payload =
        server_prompt_cache_payload::from_vbr_variants(shared_variants);
    auto shared_second_payload =
        server_prompt_cache_payload::from_vbr_variants(shared_variants);
    const size_t shared_anchor_bytes = size_t(
        shared_first_payload.vbr_anchor_resident_bytes());
    CHECK(shared_anchor_bytes > 0);
    shared_variants.reset();

    server_prompt_cache_payload shared_compact_probe_a;
    server_prompt_cache_payload shared_compact_probe_b;
    CHECK(shared_first_payload.prepare_vbr_compact_only(
        shared_compact_probe_a));
    CHECK(shared_second_payload.prepare_vbr_compact_only(
        shared_compact_probe_b));
    std::vector<const server_prompt_cache_payload *> shared_selected {
        &shared_first_payload, &shared_second_payload,
    };
    std::vector<vbr_artifact_prepared_retire> shared_prepared;
    CHECK(!server_prompt_cache_payload::prepare_vbr_anchor_retire_batch(
        shared_selected, fixture.ledger.serial() + 1, shared_prepared));
    CHECK(shared_prepared.empty());
    CHECK(shared_first_payload.vbr_has_quality_anchor());
    CHECK(shared_second_payload.vbr_has_quality_anchor());
    CHECK(fixture.catalog->snapshot().references == 2);
    shared_compact_probe_a = {};
    shared_compact_probe_b = {};

    cache.quality_anchor_budget_enabled = true;
    cache.limit_anchor_size = shared_anchor_bytes;
    cache.limit_size = 0;
    auto shared_first_stage = cache.stage_vbr(
        prompt, std::move(shared_first_payload),
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    server_prompt_cache::iterator shared_first_state;
    CHECK(cache.publish(
        std::move(shared_first_stage), &prompt, source_slot,
        &shared_first_state));
    shared_first_state->recovery_pins = 1;
    auto shared_second_stage = cache.stage_vbr(
        prompt, std::move(shared_second_payload),
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    server_prompt_cache::iterator shared_second_state;
    CHECK(cache.publish(
        std::move(shared_second_stage), &prompt, source_slot,
        &shared_second_state));
    CHECK(cache.states.size() == 2);
    CHECK(shared_first_state->payload.vbr_has_quality_anchor());
    CHECK(shared_second_state->payload.vbr_has_quality_anchor());
    CHECK(fixture.catalog->snapshot().references == 2);

    shared_first_state->recovery_pins = 0;
    cache.limit_anchor_size = 0;
    const auto shared_retires_before = cache.quality_anchor_retires;
    cache.update();
    CHECK(cache.states.size() == 2);
    CHECK(!shared_first_state->payload.vbr_has_quality_anchor());
    CHECK(!shared_second_state->payload.vbr_has_quality_anchor());
    CHECK(cache.anchor_size() == 0);
    CHECK(cache.quality_anchor_retires == shared_retires_before + 2);
    CHECK(fixture.catalog->snapshot().references == 1);
    cache.quality_anchor_budget_enabled = false;
    cache.limit_size = 0;
    cache.clear_accounting();
    cache.states.clear();
    CHECK(fixture.catalog->snapshot().references == 0);

    // Missing anchor-ranking authority is fail-soft for a newly published
    // optional variant: a zero anchor budget strips only that variant and
    // still publishes the compact checkpoint.
    const auto fallback_compact_reference = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    const auto fallback_anchor_reference = publish_fixture(*fixture.catalog,
        anchor_package, fixture.completions(), fixture.budget);
    CHECK(fallback_compact_reference.reference_artifact.v != 0);
    CHECK(fallback_anchor_reference.reference_artifact.v != 0);
    vbr_artifact_package_view fallback_compact_view;
    vbr_artifact_package_view fallback_anchor_view;
    CHECK(fixture.catalog->resolve_reference(
              fallback_compact_reference.reference_artifact,
              fallback_compact_view) == vbr_artifact_resolve_status::ok);
    CHECK(fixture.catalog->resolve_reference(
              fallback_anchor_reference.reference_artifact,
              fallback_anchor_view) == vbr_artifact_resolve_status::ok);
    auto fallback_compact = server_prompt_cache_vbr_payload::adopt_owned(
        std::move(fallback_compact_view));
    auto fallback_anchor = server_prompt_cache_vbr_payload::adopt_owned(
        std::move(fallback_anchor_view));
    auto fallback_variants = server_prompt_cache_vbr_variant_set::create(
        std::move(fallback_compact), std::move(fallback_anchor));
    CHECK(fallback_variants);
    auto fallback_payload =
        server_prompt_cache_payload::from_vbr_variants(
            std::move(fallback_variants));
    CHECK(fallback_payload.vbr_has_quality_anchor());
    auto fallback_stage = cache.stage_vbr(
        prompt, std::move(fallback_payload),
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    CHECK(fallback_stage.size() == 1);
    cache.quality_anchor_budget_enabled = true;
    cache.limit_anchor_size = 0;
    cache.limit_size = compact_payload_bytes;
    cache.lease_obs = nullptr;
    server_prompt_cache::iterator fallback_state;
    CHECK(cache.publish(
        std::move(fallback_stage), &prompt, source_slot, &fallback_state));
    CHECK(fallback_state != cache.states.end());
    CHECK(!fallback_state->payload.vbr_has_quality_anchor());
    CHECK(fallback_state->size() == compact_payload_bytes);
    CHECK(cache.anchor_size() == 0);
    CHECK(fixture.catalog->snapshot().references == 1);
    cache.lease_obs = &authority.leases;
    cache.quality_anchor_budget_enabled = false;
    cache.limit_size = 0;
    cache.clear_accounting();
    cache.states.clear();
    CHECK(fixture.catalog->snapshot().references == 0);

    // Whole-cache destruction is also an ownership terminal: a live owned
    // VBR node must drain its catalog references when the cache dies.
    const auto shutdown_reference = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    CHECK(shutdown_reference.reference_artifact.v != 0);
    {
        server_prompt_cache shutdown_cache(0, 0);
        shutdown_cache.acct = &fixture.ledger;
        shutdown_cache.publish_authority = &authority;
        shutdown_cache.destruction_obs = &authority.destruction;
        shutdown_cache.retention_obs = &retention;
        shutdown_cache.lease_obs = &authority.leases;
        shutdown_cache.lease_execution_identity =
            &fixture.package.manifest.identity.execution_identity;
        auto shutdown_payload = owned_payload(
            shutdown_reference.reference_artifact);
        auto shutdown_stage = shutdown_cache.stage_vbr(
            prompt, std::move(shutdown_payload),
            fixture.package.manifest.identity.execution_identity,
            fixture.package.manifest.identity.adapter_config_identity);
        CHECK(shutdown_stage.size() == 1);
        CHECK(shutdown_cache.publish(
            std::move(shutdown_stage), &prompt, source_slot));
        CHECK(fixture.catalog->snapshot().references == 1);
    }
    CHECK(fixture.catalog->snapshot().references == 0);
}

static void test_prompt_cache_vbr_anchor_prepare_rollback() {
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
    auto anchor_package =
        prompt_cache_quality_anchor_package(fixture.package);
    CHECK(fixture.catalog->configure_accounting(anchor_package));
    const auto compact_reference = publish_fixture(*fixture.catalog,
        fixture.package, fixture.completions(), fixture.budget);
    const auto anchor_reference = publish_fixture(*fixture.catalog,
        anchor_package, fixture.completions(), fixture.budget);
    CHECK(compact_reference.reference_artifact.v != 0);
    CHECK(anchor_reference.reference_artifact.v != 0);

    vbr_artifact_package_view compact_view;
    vbr_artifact_package_view anchor_view;
    CHECK(fixture.catalog->resolve_reference(
              compact_reference.reference_artifact, compact_view) ==
          vbr_artifact_resolve_status::ok);
    CHECK(fixture.catalog->resolve_reference(
              anchor_reference.reference_artifact, anchor_view) ==
          vbr_artifact_resolve_status::ok);
    auto compact_owner = server_prompt_cache_vbr_payload::adopt_owned(
        std::move(compact_view));
    auto anchor_owner = server_prompt_cache_vbr_payload::adopt_owned(
        std::move(anchor_view));
    auto variants = server_prompt_cache_vbr_variant_set::create(
        std::move(compact_owner), std::move(anchor_owner));
    CHECK(variants);
    auto first_payload =
        server_prompt_cache_payload::from_vbr_variants(variants);
    auto second_payload =
        server_prompt_cache_payload::from_vbr_variants(variants);
    const size_t anchor_bytes = size_t(
        first_payload.vbr_anchor_resident_bytes());
    CHECK(anchor_bytes > 0);
    variants.reset();

    server_cache_authority authority;
    server_retention_sidecar_store retention;
    retention.configure(&fixture.ledger, fixture.host, &authority.leases);
    CHECK(retention.enable_prefix_tracking());
    constexpr int32_t source_slot = 9;
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

    server_prompt_cache cache(0, 0);
    cache.acct = &fixture.ledger;
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;
    cache.retention_obs = &retention;
    cache.lease_obs = &authority.leases;
    cache.lease_execution_identity =
        &fixture.package.manifest.identity.execution_identity;
    cache.retention_df2_authority = true;
    CHECK(cache.enable_retention_shadow());
    cache.quality_anchor_budget_enabled = true;
    cache.limit_anchor_size = anchor_bytes;

    auto first_stage = cache.stage_vbr(
        prompt, std::move(first_payload),
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    server_prompt_cache::iterator first_state;
    CHECK(cache.publish(
        std::move(first_stage), &prompt, source_slot, &first_state));
    first_state->recovery_pins = 1;
    auto second_stage = cache.stage_vbr(
        prompt, std::move(second_payload),
        fixture.package.manifest.identity.execution_identity,
        fixture.package.manifest.identity.adapter_config_identity);
    server_prompt_cache::iterator second_state;
    CHECK(cache.publish(
        std::move(second_stage), &prompt, source_slot, &second_state));
    first_state->recovery_pins = 0;
    CHECK(cache.states.size() == 2);
    CHECK(fixture.catalog->snapshot().references == 2);

    cache.limit_anchor_size = 0;
    const auto retires_before = cache.quality_anchor_retires;
    cache.update();
    CHECK(cache.states.size() == 2);
    CHECK(first_state->payload.vbr_has_quality_anchor());
    CHECK(second_state->payload.vbr_has_quality_anchor());
    CHECK(cache.anchor_size() == anchor_bytes);
    CHECK(cache.quality_anchor_retires == retires_before);
    CHECK(fixture.catalog->snapshot().references == 2);

    cache.quality_anchor_budget_enabled = false;
    cache.clear_accounting();
    cache.states.clear();
    CHECK(fixture.catalog->snapshot().references == 0);
}

#endif

static vbr_artifact_stream_placement capture_placement(
        uint32_t child,
        uint32_t stream,
        llama_seq_id sequence,
        llama_pos frontier,
        std::initializer_list<std::pair<uint32_t, llama_pos>> cells) {
    vbr_artifact_stream_placement result;
    result.child_id = child;
    result.stream_index = stream;
    result.source_sequence = sequence;
    result.computation_frontier = frontier;
    for (const auto & cell : cells) {
        result.cells.push_back({ cell.first, cell.second, 0, 0 });
    }
    return result;
}

static void test_sequence_projected_capture_union() {
    vbr_capture_projection_manifest twenty;
    twenty.manifest_id = 20;
    twenty.placements.push_back(capture_placement(
        0, 0, 2, 10, {{0, 0}, {1, 1}, {2, 2}, {4, 4}}));
    twenty.placements.push_back(capture_placement(
        1, 0, 4, 10, {{1, 1}}));
    vbr_capture_projection_manifest ten;
    ten.manifest_id = 10;
    ten.placements.push_back(capture_placement(
        0, 0, 3, 10, {{1, 1}, {2, 2}, {3, 3}}));

    vbr_capture_projection_limits limits;
    vbr_capture_projection plan;
    vbr_capture_projection_batch batch { 77, { twenty, ten } };
    CHECK(vbr_artifact_project_capture_union(
        batch, limits, plan));
    CHECK(plan->source_namespace == 77);
    CHECK(plan->manifest_count == 2);
    CHECK(plan->placement_count == 3);
    CHECK(plan->input_cell_references == 8);
    CHECK(plan->union_cell_count == 6);
    CHECK(plan->dependency_references == 6);
    CHECK(plan->streams.size() == 2);
    if (plan->streams.size() == 2) {
        const auto & first = plan->streams[0];
        CHECK(first.child_id == 0);
        CHECK(first.stream_index == 0);
        CHECK(first.segments.size() == 4);
        if (first.segments.size() == 4) {
            CHECK(first.segments[0].first_physical_cell == 0);
            CHECK(first.segments[0].cell_count == 1);
            CHECK(std::vector<uint64_t>(
                      plan->dependent_manifest_ids.begin() +
                          first.segments[0].first_dependency,
                      plan->dependent_manifest_ids.begin() +
                          first.segments[0].first_dependency +
                          first.segments[0].dependency_count) ==
                  std::vector<uint64_t>({20}));
            CHECK(first.segments[1].first_physical_cell == 1);
            CHECK(first.segments[1].cell_count == 2);
            CHECK(std::vector<uint64_t>(
                      plan->dependent_manifest_ids.begin() +
                          first.segments[1].first_dependency,
                      plan->dependent_manifest_ids.begin() +
                          first.segments[1].first_dependency +
                          first.segments[1].dependency_count) ==
                  std::vector<uint64_t>({10, 20}));
            CHECK(first.segments[2].first_physical_cell == 3);
            CHECK(first.segments[2].cell_count == 1);
            CHECK(std::vector<uint64_t>(
                      plan->dependent_manifest_ids.begin() +
                          first.segments[2].first_dependency,
                      plan->dependent_manifest_ids.begin() +
                          first.segments[2].first_dependency +
                          first.segments[2].dependency_count) ==
                  std::vector<uint64_t>({10}));
            CHECK(first.segments[3].first_physical_cell == 4);
            CHECK(first.segments[3].cell_count == 1);
            CHECK(std::vector<uint64_t>(
                      plan->dependent_manifest_ids.begin() +
                          first.segments[3].first_dependency,
                      plan->dependent_manifest_ids.begin() +
                          first.segments[3].first_dependency +
                          first.segments[3].dependency_count) ==
                  std::vector<uint64_t>({20}));
        }
        const auto & second = plan->streams[1];
        CHECK(second.child_id == 1);
        CHECK(second.stream_index == 0);
        CHECK(second.segments.size() == 1);
        if (second.segments.size() == 1) {
            CHECK(second.segments[0].first_physical_cell == 1);
            CHECK(second.segments[0].cell_count == 1);
            CHECK(std::vector<uint64_t>(
                      plan->dependent_manifest_ids.begin() +
                          second.segments[0].first_dependency,
                      plan->dependent_manifest_ids.begin() +
                          second.segments[0].first_dependency +
                          second.segments[0].dependency_count) ==
                  std::vector<uint64_t>({20}));
        }
    }

    // Caller order cannot affect the transport or dependency plan.
    vbr_capture_projection reversed;
    vbr_capture_projection_batch reversed_batch { 77, { ten, twenty } };
    CHECK(vbr_artifact_project_capture_union(
        reversed_batch, limits, reversed));
    CHECK(reversed->source_namespace == plan->source_namespace);
    CHECK(reversed->manifest_count == plan->manifest_count);
    CHECK(reversed->placement_count == plan->placement_count);
    CHECK(reversed->input_cell_references == plan->input_cell_references);
    CHECK(reversed->union_cell_count == plan->union_cell_count);
    CHECK(reversed->dependency_references == plan->dependency_references);
    CHECK(reversed->streams.size() == plan->streams.size());
    if (reversed->streams.size() == plan->streams.size()) {
        for (size_t i = 0; i < plan->streams.size(); ++i) {
            CHECK(reversed->streams[i].child_id == plan->streams[i].child_id);
            CHECK(reversed->streams[i].stream_index ==
                  plan->streams[i].stream_index);
            CHECK(reversed->streams[i].segments.size() ==
                  plan->streams[i].segments.size());
            const size_t count = std::min(
                reversed->streams[i].segments.size(),
                plan->streams[i].segments.size());
            for (size_t j = 0; j < count; ++j) {
                CHECK(reversed->streams[i].segments[j].first_physical_cell ==
                      plan->streams[i].segments[j].first_physical_cell);
                CHECK(reversed->streams[i].segments[j].cell_count ==
                      plan->streams[i].segments[j].cell_count);
                CHECK(reversed->streams[i].segments[j].first_dependency ==
                      plan->streams[i].segments[j].first_dependency);
                CHECK(reversed->streams[i].segments[j].dependency_count ==
                      plan->streams[i].segments[j].dependency_count);
            }
        }
    }
    CHECK(reversed->dependent_manifest_ids == plan->dependent_manifest_ids);

    // Every refusal is transactional: even a pre-populated output is cleared.
    auto expect_refused = [&](const vbr_capture_projection_batch & refused_batch,
                              const vbr_capture_projection_limits & bound) {
        vbr_capture_projection refused = plan;
        CHECK(!vbr_artifact_project_capture_union(
            refused_batch, bound, refused));
        CHECK(!refused);
    };

    auto exact_limits = limits;
    exact_limits.max_manifests = 2;
    exact_limits.max_placements = 3;
    exact_limits.max_input_cells = 8;
    exact_limits.max_union_cells = 6;
    exact_limits.max_segments = 5;
    exact_limits.max_dependency_references = 6;
    vbr_capture_projection exact;
    CHECK(vbr_artifact_project_capture_union(batch, exact_limits, exact));

    auto too_few_manifests = exact_limits;
    too_few_manifests.max_manifests = 1;
    expect_refused(batch, too_few_manifests);
    auto too_few_placements = exact_limits;
    too_few_placements.max_placements = 2;
    expect_refused(batch, too_few_placements);
    auto too_few_input = exact_limits;
    too_few_input.max_input_cells = 7;
    expect_refused(batch, too_few_input);
    auto too_few_union = limits;
    too_few_union.max_union_cells = 5;
    expect_refused(batch, too_few_union);
    auto too_few_segments = limits;
    too_few_segments.max_segments = 4;
    expect_refused(batch, too_few_segments);
    auto too_few_dependencies = limits;
    too_few_dependencies.max_dependency_references = 5;
    expect_refused(batch, too_few_dependencies);

    auto semantic_batch = batch;
    semantic_batch.manifests[0].token_block.tokens = { 1, 2 };
    semantic_batch.manifests[0].identity.execution_identity = "abc";
    auto semantic_exact = limits;
    semantic_exact.max_token_ids = 2;
    semantic_exact.max_string_bytes = 3;
    CHECK(vbr_artifact_project_capture_union(
        semantic_batch, semantic_exact, exact));
    auto too_many_tokens = semantic_exact;
    too_many_tokens.max_token_ids = 1;
    expect_refused(semantic_batch, too_many_tokens);
    auto too_many_strings = semantic_exact;
    too_many_strings.max_string_bytes = 2;
    expect_refused(semantic_batch, too_many_strings);
    auto too_many_semantic_bytes = semantic_exact;
    too_many_semantic_bytes.max_semantic_metadata_bytes = 10;
    expect_refused(semantic_batch, too_many_semantic_bytes);

    auto no_namespace = batch;
    no_namespace.source_namespace = 0;
    expect_refused(no_namespace, limits);

    auto duplicate_id = ten;
    duplicate_id.manifest_id = twenty.manifest_id;
    expect_refused({ 77, { twenty, duplicate_id } }, limits);

    auto duplicate_stream = ten;
    duplicate_stream.placements.push_back(capture_placement(
        0, 0, 4, 10, {{5, 5}}));
    expect_refused({ 77, { duplicate_stream } }, limits);

    auto unordered = ten;
    std::reverse(unordered.placements[0].cells.begin(),
                 unordered.placements[0].cells.end());
    expect_refused({ 77, { unordered } }, limits);

    auto beyond_frontier = ten;
    beyond_frontier.placements[0].cells.back().logical_position = 10;
    expect_refused({ 77, { beyond_frontier } }, limits);

    auto zero_id = ten;
    zero_id.manifest_id = 0;
    expect_refused({ 77, { zero_id } }, limits);
    auto invalid_physical = ten;
    invalid_physical.placements[0].cells[0].physical_cell = UINT32_MAX;
    expect_refused({ 77, { invalid_physical } }, limits);
    auto invalid_logical = ten;
    invalid_logical.placements[0].cells[0].logical_position = -1;
    expect_refused({ 77, { invalid_logical } }, limits);

    auto duplicate_logical = ten;
    duplicate_logical.placements[0].cells[1].logical_position =
        duplicate_logical.placements[0].cells[0].logical_position;
    expect_refused({ 77, { duplicate_logical } }, limits);

    vbr_capture_projection_manifest cross_placement;
    cross_placement.manifest_id = 30;
    cross_placement.placements.push_back(capture_placement(
        0, 0, 5, 10, {{1, 2}}));
    cross_placement.placements.push_back(capture_placement(
        1, 0, 5, 10, {{2, 2}}));
    expect_refused({ 77, { cross_placement } }, limits);

    // Logical positions are sequence-scoped, so two independent sequences
    // may cite the same position without making the manifest ambiguous.
    cross_placement.placements[1].source_sequence = 6;
    vbr_capture_projection sequence_scoped;
    CHECK(vbr_artifact_project_capture_union(
        { 77, { cross_placement } }, limits, sequence_scoped));
}

static void benchmark_sequence_projected_capture_union() {
    static constexpr uint32_t CELLS = 1048576;
    vbr_capture_projection_manifest even;
    even.manifest_id = 1;
    even.placements.push_back(capture_placement(0, 0, 1, CELLS, {}));
    vbr_capture_projection_manifest odd;
    odd.manifest_id = 2;
    odd.placements.push_back(capture_placement(0, 0, 2, CELLS, {}));
    auto & even_cells = even.placements[0].cells;
    auto & odd_cells = odd.placements[0].cells;
    even_cells.reserve(CELLS/2);
    odd_cells.reserve(CELLS/2);
    for (uint32_t cell = 0; cell < CELLS; ++cell) {
        auto & target = (cell & 1u) ? odd_cells : even_cells;
        target.push_back({ cell, llama_pos(cell/2), 0, 0 });
    }

    vbr_capture_projection_limits limits;
    limits.max_manifests = 2;
    limits.max_placements = 2;
    limits.max_input_cells = CELLS;
    limits.max_union_cells = CELLS;
    limits.max_segments = CELLS;
    limits.max_dependency_references = CELLS;
    vbr_capture_projection plan;
    const auto begin = std::chrono::steady_clock::now();
    CHECK(vbr_artifact_project_capture_union(
        { 1, { std::move(even), std::move(odd) } }, limits, plan));
    const auto elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin).count();
    CHECK(plan->manifest_count == 2);
    CHECK(plan->placement_count == 2);
    CHECK(plan->input_cell_references == CELLS);
    CHECK(plan->union_cell_count == CELLS);
    CHECK(plan->dependency_references == CELLS);
    CHECK(plan->streams.size() == 1);
    if (plan->streams.size() == 1) {
        CHECK(plan->streams[0].segments.size() == CELLS);
    }
    CHECK(plan->dependent_manifest_ids.size() == CELLS);
    printf("VBR_CAPTURE_PROJECTION_BENCH cells=%u segments=%zu "
           "dependencies=%zu elapsed_us=%lld\n",
           CELLS,
           plan->streams.empty() ? 0 : plan->streams[0].segments.size(),
           plan->dependent_manifest_ids.size(),
           (long long) elapsed);
}

int main(int argc, char ** argv) {
    if (argc == 2 &&
        strcmp(argv[1], "--capture-projection-bench") == 0) {
        benchmark_sequence_projected_capture_union();
        return failures == 0 ? 0 : 1;
    }
#ifdef VBR_PROMPT_CACHE_PUBLICATION_TEST
    if (server_fault("vbr_anchor_prepare_fail")) {
        test_prompt_cache_vbr_anchor_prepare_rollback();
        if (failures != 0) {
            fprintf(stderr, "%d VBR anchor rollback test(s) failed\n",
                    failures);
            return 1;
        }
        printf("VBR prompt-cache anchor rollback: PASS\n");
        return 0;
    }
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
    test_catalog_owned_union_retirement();
    test_catalog_all_shard_failures_and_rollback();
    test_catalog_destructor_releases_live_references();
    test_catalog_dedup_race();
    test_catalog_full_id_interning_and_stash_dedup();
    test_catalog_capacity_sequential_and_temporaries();
    test_catalog_package_lease_and_reference_placement();
    test_prompt_cache_vbr_payload_fanout_lifetime();
    test_prompt_cache_vbr_same_frontier_variants();
    test_sequence_projected_capture_union();
    test_manifest_validator_matrix();
    test_validated_manifest_staging();
#ifdef VBR_PROMPT_CACHE_PUBLICATION_TEST
    test_prompt_cache_vbr_atomic_logical_publication();
    test_prompt_cache_vbr_pressure_retires_physical_union();
#endif
    if (failures != 0) {
        fprintf(stderr, "%d VBR artifact test(s) failed\n", failures);
        return 1;
    }
    printf("VBR artifact format: PASS\n");
    return 0;
}
