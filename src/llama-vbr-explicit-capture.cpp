#include "llama-vbr-explicit-capture.h"

#include "llama-vbr-artifact-adopt.h"
#include "llama-vbr-artifact-validate.h"
#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-tree.h"
#include "llama-sha256.h"
#include "llama-vbr-identity-digest.h"
#include "llama-vbr-operation.h"
#include "turbo-rotation-data.h"

#include "ggml-turbo-meansub.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace {

std::array<uint8_t, 32> representation_hash_file_or_marker(
        const char * tag,
        const char * path,
        uint32_t type,
        bool value_side,
        const vbr_explicit_representation_policy & policy,
        bool & ok) {
    llama_sha256_writer writer;
    writer.string(tag, strlen(tag));
    writer.u32(type);
    writer.u32(value_side);
    if (path == nullptr || path[0] == '\0') {
        static constexpr char BUILTIN[] =
            "compiled-in/build-identity";
        writer.string(BUILTIN, sizeof(BUILTIN) - 1);
        if (policy.build_identity == nullptr ||
            policy.build_identity_len == 0) {
            ok = false;
            return {};
        }
        writer.string(
            policy.build_identity, policy.build_identity_len);
        return writer.finish();
    }
    FILE * file = fopen(path, "rb");
    if (file == nullptr) {
        ok = false;
        return {};
    }
    std::array<uint8_t, 64*1024> buffer;
    for (;;) {
        const size_t size =
            fread(buffer.data(), 1, buffer.size(), file);
        if (size != 0) {
            writer.bytes(buffer.data(), size);
        }
        if (size != buffer.size()) {
            if (ferror(file)) {
                ok = false;
            }
            break;
        }
    }
    fclose(file);
    return ok ? writer.finish() : std::array<uint8_t, 32>{};
}

const char * representation_override(
        int32_t type,
        bool value_side) {
    switch (type) {
        case GGML_TYPE_TURBO8_0:
            return std::getenv("TURBO_CB_T8");
        case GGML_TYPE_TURBO4_0:
            return std::getenv("TURBO_CB_T4");
        case GGML_TYPE_TURBO3_TCQ:
        case GGML_TYPE_TURBO2_TCQ: {
            const char * side = std::getenv(value_side
                ? "TURBO_TCQ_CB_V" : "TURBO_TCQ_CB_K");
            return side ? side : std::getenv("TURBO_TCQ_CB");
        }
        case GGML_TYPE_TURBO1_TCQ: {
            const char * side = std::getenv(value_side
                ? "TURBO1_TCQ_CB_V" : "TURBO1_TCQ_CB_K");
            return side ? side : std::getenv("TURBO1_TCQ_CB");
        }
        default:
            return nullptr;
    }
}

std::array<uint8_t, 32> representation_rotation_identity(
        int32_t type,
        bool value_side) {
    llama_sha256_writer writer;
    static constexpr char domain_label[] =
        "buun.vbr.codec-rotation/v1";
    writer.string(domain_label, sizeof(domain_label) - 1);
    writer.u32(uint32_t(type));
    writer.u32(value_side);
    const auto * matrix = value_side
        ? TURBO_ROTATION_RT : TURBO_ROTATION_R;
    writer.bytes(matrix, 128*128*sizeof(matrix[0]));
    return writer.finish();
}

std::array<uint8_t, 32> representation_meansub_identity(
        int32_t type,
        bool value_side,
        const vbr_explicit_representation_policy & policy,
        bool & ok) {
    llama_sha256_writer writer;
    static constexpr char domain_label[] =
        "buun.vbr.codec-meansub/v1";
    writer.string(domain_label, sizeof(domain_label) - 1);
    writer.u32(uint32_t(type));
    writer.u32(value_side);

    const char * disabled = std::getenv("TURBO_MEANSUB_OFF");
    if (disabled != nullptr) {
        static constexpr char OFF[] = "disabled";
        writer.string(OFF, sizeof(OFF) - 1);
        return writer.finish();
    }
    const char * path = std::getenv(
        value_side ? "TURBO_VMEAN_SUB" : "TURBO_KMEAN_SUB");
    if (path != nullptr && path[0] != '\0') {
        return representation_hash_file_or_marker(
            "buun.vbr.codec-meansub-file/v1",
            path, uint32_t(type), value_side, policy, ok);
    }

    int max_layers = 0;
    int max_channels = 0;
    int live_layers = 0;
    const float * active = ggml_turbo_meansub_table(
        policy.turbo_meansub_id, value_side ? 1 : 0,
        &max_layers, &max_channels, &live_layers);
    if (active == nullptr || max_layers <= 0 ||
        max_channels <= 0 || live_layers <= 0) {
        static constexpr char INACTIVE[] = "inactive";
        writer.string(INACTIVE, sizeof(INACTIVE) - 1);
        return writer.finish();
    }
    if (size_t(max_layers) >
            std::numeric_limits<size_t>::max() /
                size_t(max_channels) ||
        size_t(max_layers)*size_t(max_channels) >
            std::numeric_limits<size_t>::max() / sizeof(float)) {
        ok = false;
        return {};
    }
    writer.u32(uint32_t(max_layers));
    writer.u32(uint32_t(max_channels));
    writer.u32(uint32_t(live_layers));
    writer.bytes(
        active,
        size_t(max_layers)*size_t(max_channels)*sizeof(float));
    return writer.finish();
}

bool digest_nonzero(const std::array<uint8_t, 32> & digest) {
    return std::any_of(digest.begin(), digest.end(),
        [](uint8_t value) { return value != 0; });
}

// Shared by the capture stamp and the import target check so the recurrent
// companion identity cannot drift between the two.
constexpr char VBR_RECURRENT_CODEC_DOMAIN[] = "buun.vbr.capture/recurrent-codec";

// Nonzero head of a finished digest (0 is reserved as "absent").
uint64_t digest_head_u64(llama_sha256_writer & writer) {
    const auto digest = writer.finish();
    uint64_t value = 0;
    std::memcpy(&value, digest.data(), sizeof(value));
    return value == 0 ? 1 : value;
}

std::array<uint8_t, 32> tagged_digest(
        const char * tag,
        uint64_t a,
        uint64_t b = 0) {
    llama_sha256_writer writer;
    writer.string(tag, strlen(tag));
    writer.u64(a);
    writer.u64(b);
    return writer.finish();
}

struct capture_transfer_cancelled {};

class chain_io_writer final : public llama_io_write_i {
public:
    static constexpr size_t CHUNK_BYTES = 1024*1024;

    chain_io_writer(
            artifact_segment_chain & chain,
            uint64_t expected_bytes,
            void * continue_context = nullptr,
            vbr_projected_capture_batch_request::continue_transfer_fn
                continue_transfer = nullptr,
            uint64_t * tensor_d2h_bytes = nullptr,
            uint64_t * tensor_d2h_reads = nullptr)
        : chain_(chain), expected_bytes_(expected_bytes),
          scratch_(size_t(std::max<uint64_t>(
              1, std::min<uint64_t>(CHUNK_BYTES, expected_bytes)))),
          continue_context_(continue_context),
          continue_transfer_(continue_transfer),
          tensor_d2h_bytes_(tensor_d2h_bytes),
          tensor_d2h_reads_(tensor_d2h_reads) {
        if (expected_bytes > SIZE_MAX) {
            throw std::length_error("companion payload exceeds size_t");
        }
    }

    void write(const void * source, size_t size) override {
        const auto * cursor = static_cast<const uint8_t *>(source);
        if (cursor == nullptr && size != 0) {
            throw std::invalid_argument("null companion source");
        }
        account(size);
        while (size != 0) {
            ensure_scratch();
            const size_t take = std::min(size, scratch_.size() - used_);
            std::memcpy(scratch_.data() + used_, cursor, take);
            used_ += take;
            cursor += take;
            size -= take;
            if (used_ == scratch_.size()) {
                flush();
            }
        }
    }

    void write_tensor(
            ggml_tensor * tensor,
            size_t offset,
            size_t size) override {
        if (tensor == nullptr) {
            throw std::invalid_argument("null companion tensor");
        }
        account(size);
        while (size != 0) {
            check_continue();
            ensure_scratch();
            const size_t take = std::min(size, scratch_.size() - used_);
            ggml_backend_tensor_get(
                tensor, scratch_.data() + used_, offset, take);
            if (tensor_d2h_bytes_) {
                if (*tensor_d2h_bytes_ > UINT64_MAX - take) {
                    throw std::overflow_error(
                        "companion D2H byte counter overflow");
                }
                *tensor_d2h_bytes_ += take;
            }
            if (tensor_d2h_reads_) {
                if (*tensor_d2h_reads_ == UINT64_MAX) {
                    throw std::overflow_error(
                        "companion D2H read counter overflow");
                }
                ++*tensor_d2h_reads_;
            }
            // Bound cancellation latency to one synchronous tensor quantum
            // even when the queue becomes nonempty immediately after the
            // pre-read probe.
            check_continue();
            used_ += take;
            offset += take;
            size -= take;
            if (used_ == scratch_.size()) {
                flush();
            }
        }
    }

    size_t n_bytes() override {
        return size_t(written_);
    }

    bool finish() {
        flush();
        return written_ == expected_bytes_ &&
               chain_.size() == expected_bytes_;
    }

private:
    void check_continue() const {
        if (continue_transfer_ &&
            !continue_transfer_(continue_context_)) {
            throw capture_transfer_cancelled {};
        }
    }

    void account(size_t size) {
        if (written_ > expected_bytes_ ||
            size > expected_bytes_ - written_ ||
            written_ > SIZE_MAX || size > SIZE_MAX - size_t(written_)) {
            throw std::overflow_error("companion payload size mismatch");
        }
        written_ += size;
    }

    void flush() {
        if (used_ != 0) {
            scratch_.resize(used_);
            if (!chain_.append_owned(std::move(scratch_))) {
                throw std::bad_alloc();
            }
            used_ = 0;
        }
    }

    void ensure_scratch() {
        if (!scratch_.empty()) {
            return;
        }
        const uint64_t remaining = expected_bytes_ - chain_.size();
        if (remaining == 0) {
            throw std::overflow_error("companion payload overrun");
        }
        scratch_.resize(size_t(std::min<uint64_t>(
            CHUNK_BYTES, remaining)));
    }

    artifact_segment_chain & chain_;
    uint64_t expected_bytes_ = 0;
    uint64_t written_ = 0;
    std::vector<uint8_t> scratch_;
    size_t used_ = 0;
    void * continue_context_ = nullptr;
    vbr_projected_capture_batch_request::continue_transfer_fn
        continue_transfer_ = nullptr;
    uint64_t * tensor_d2h_bytes_ = nullptr;
    uint64_t * tensor_d2h_reads_ = nullptr;
};

class counting_io_writer final : public llama_io_write_i {
public:
    void write(const void *, size_t size) override {
        add(size);
    }
    void write_tensor(ggml_tensor *, size_t, size_t size) override {
        add(size);
    }
    size_t n_bytes() override {
        return bytes;
    }
    void add(size_t size) {
        if (size > std::numeric_limits<size_t>::max() - bytes) {
            throw std::bad_alloc();
        }
        bytes += size;
    }
    size_t bytes = 0;
};

struct recurrent_companion_plan {
    llama_memory_recurrent * source = nullptr;
    llama_pos expected_terminal = -1;
    vbr_artifact_companion_payload descriptor;
};

bool recurrent_companion_current(
        const recurrent_companion_plan & plan,
        llama_seq_id sequence) noexcept {
    return plan.source != nullptr && sequence >= 0 &&
        plan.source->seq_pos_min(sequence) == plan.expected_terminal &&
        plan.source->seq_pos_max(sequence) == plan.expected_terminal;
}

bool recurrent_companion_prepare(
        llama_memory_recurrent * source,
        llama_seq_id sequence,
        llama_pos next_position,
        recurrent_companion_plan & output,
        vbr_explicit_capture_status & status) noexcept {
    output = {};
    try {
        if (source == nullptr || sequence < 0 || next_position < 0) {
            status = vbr_explicit_capture_status::
                required_companion_unavailable;
            return false;
        }
        const llama_pos expected_terminal = next_position - 1;
        if (source->seq_pos_min(sequence) != expected_terminal ||
            source->seq_pos_max(sequence) != expected_terminal) {
            status = vbr_explicit_capture_status::source_changed;
            return false;
        }
        counting_io_writer writer;
        source->state_write(writer, sequence, 0);
        if (writer.n_bytes() == 0 ||
            source->seq_pos_min(sequence) != expected_terminal ||
            source->seq_pos_max(sequence) != expected_terminal) {
            status = vbr_explicit_capture_status::
                required_companion_unavailable;
            return false;
        }
        output.source = source;
        output.expected_terminal = expected_terminal;
        output.descriptor.kind = vbr_artifact_companion_kind::recurrent;
        output.descriptor.format_version = 1;
        output.descriptor.build_identity_digest = tagged_digest(
            VBR_RECURRENT_CODEC_DOMAIN, 1);
        output.descriptor.domain = {
            llama_cache_acct_residency::pageable_host,
            llama_cache_acct_domain_kind::not_applicable,
            UINT32_MAX, UINT16_MAX,
        };
        output.descriptor.payload_bytes = writer.n_bytes();
        return true;
    } catch (...) {
        output = {};
        status = vbr_explicit_capture_status::
            required_companion_unavailable;
        return false;
    }
}

bool recurrent_companion_capture(
        const recurrent_companion_plan & plan,
        llama_seq_id sequence,
        std::unique_ptr<artifact_segment_chain> & output,
        std::array<uint8_t, 32> & digest,
        vbr_explicit_capture_status & status,
        void * continue_context = nullptr,
        vbr_projected_capture_batch_request::continue_transfer_fn
            continue_transfer = nullptr,
        uint64_t * d2h_bytes = nullptr,
        uint64_t * d2h_reads = nullptr) noexcept {
    output.reset();
    digest = {};
    if (d2h_bytes) {
        *d2h_bytes = 0;
    }
    if (d2h_reads) {
        *d2h_reads = 0;
    }
    try {
        if (plan.source == nullptr ||
            plan.descriptor.payload_bytes == 0 || sequence < 0) {
            status = vbr_explicit_capture_status::
                required_companion_unavailable;
            return false;
        }
        if (!recurrent_companion_current(plan, sequence)) {
            status = vbr_explicit_capture_status::source_changed;
            return false;
        }
        auto chain = std::make_unique<artifact_segment_chain>(
            plan.descriptor.payload_bytes);
        chain_io_writer writer(
            *chain, plan.descriptor.payload_bytes,
            continue_context, continue_transfer,
            d2h_bytes, d2h_reads);
        plan.source->state_write(writer, sequence, 0);
        if (!writer.finish() ||
            writer.n_bytes() != plan.descriptor.payload_bytes ||
            !recurrent_companion_current(plan, sequence)) {
            status = vbr_explicit_capture_status::source_changed;
            return false;
        }
        digest = vbr_capture_stream_digest(*chain);
        if (!digest_nonzero(digest)) {
            status = vbr_explicit_capture_status::
                required_companion_unavailable;
            return false;
        }
        if (!recurrent_companion_current(plan, sequence)) {
            status = vbr_explicit_capture_status::source_changed;
            return false;
        }
        output = std::move(chain);
        return true;
    } catch (const capture_transfer_cancelled &) {
        output.reset();
        digest = {};
        status = vbr_explicit_capture_status::cancelled;
        return false;
    } catch (...) {
        output.reset();
        digest = {};
        status = vbr_explicit_capture_status::
            required_companion_unavailable;
        return false;
    }
}

void add_accounting(
        std::vector<vbr_artifact_portable_accounting_row> & rows,
        vbr_artifact_accounting_role role,
        const vbr_artifact_portable_domain & domain,
        uint64_t bytes) {
    for (auto & row : rows) {
        if (row.role == role && row.domain == domain) {
            if (bytes > UINT64_MAX - row.logical_bytes ||
                bytes > UINT64_MAX - row.resident_bytes) {
                throw std::overflow_error("artifact accounting overflow");
            }
            row.logical_bytes += bytes;
            row.resident_bytes += bytes;
            return;
        }
    }
    rows.push_back({ role, domain, bytes, bytes,
        llama_cache_acct_attr_kind::artifact });
}

vbr_artifact_portable_domain portable_domain(
        uint32_t topology,
        uint16_t ordinal) {
    return {
        llama_cache_acct_residency::device,
        llama_cache_acct_domain_kind::device_topology,
        topology,
        ordinal,
    };
}

vbr_explicit_capture_status stream_status(
        vbr_capture_stream_status status) {
    switch (status) {
        case vbr_capture_stream_status::ok:
            return vbr_explicit_capture_status::ok;
        case vbr_capture_stream_status::ring_unavailable:
            return vbr_explicit_capture_status::ring_unavailable;
        case vbr_capture_stream_status::cancelled:
            return vbr_explicit_capture_status::cancelled;
        case vbr_capture_stream_status::transfer_failed:
            return vbr_explicit_capture_status::transfer_failed;
        case vbr_capture_stream_status::short_read:
            return vbr_explicit_capture_status::short_read;
        case vbr_capture_stream_status::hash_mismatch:
            return vbr_explicit_capture_status::hash_mismatch;
        case vbr_capture_stream_status::accounting_unavailable:
        case vbr_capture_stream_status::stage_failed:
        case vbr_capture_stream_status::commit_failed:
            return vbr_explicit_capture_status::accounting_failed;
        case vbr_capture_stream_status::accounting_refused:
            return vbr_explicit_capture_status::admission_refused;
        case vbr_capture_stream_status::publication_failed:
            return vbr_explicit_capture_status::publication_failed;
        case vbr_capture_stream_status::snapshot_changed:
            return vbr_explicit_capture_status::source_changed;
        case vbr_capture_stream_status::snapshot_unavailable:
            return vbr_explicit_capture_status::generation_unavailable;
        case vbr_capture_stream_status::format_rejected:
            return vbr_explicit_capture_status::dedup_validation_failed;
        case vbr_capture_stream_status::projection_invalid:
        case vbr_capture_stream_status::invalid_argument:
        case vbr_capture_stream_status::duplicate_segment:
        case vbr_capture_stream_status::missing_segment:
        case vbr_capture_stream_status::late_segment:
        case vbr_capture_stream_status::internal_error:
        case vbr_capture_stream_status::_count:
            return vbr_explicit_capture_status::internal_error;
    }
    return vbr_explicit_capture_status::internal_error;
}

} // namespace

bool vbr_explicit_capture_representation_identity(
        const void * context,
        int32_t current_type,
        bool value_side,
        vbr_explicit_representation_identity & output) noexcept {
    try {
        if (context == nullptr ||
            current_type < 0 || current_type >= GGML_TYPE_COUNT) {
            return false;
        }
        const auto & policy =
            *static_cast<const vbr_explicit_representation_policy *>(
                context);
        output.codec_id = uint32_t(current_type) + 1;
        output.codec_version = 1;
        bool ok = true;
        output.codebook_digest =
            representation_hash_file_or_marker(
                "buun.vbr.codec-codebook/v1",
                representation_override(current_type, value_side),
                uint32_t(current_type), value_side, policy, ok);
        output.rotation_digest =
            representation_rotation_identity(
                current_type, value_side);
        output.meansub_digest =
            representation_meansub_identity(
                current_type, value_side, policy, ok);
        return ok;
    } catch (...) {
        return false;
    }
}

class vbr_live_capture_adapter {
public:
    struct child {
        uint32_t child_id = 0;
        checkpoint_child_dependency_mode dependency_mode =
            checkpoint_child_dependency_mode::absent;
        llama_kv_cache * cache = nullptr;
        std::vector<llama_kv_cache::vbr_capture_unit_plan> units;
        llama_kv_cache::vbr_capture_stability_token stability;
        vbr_checkpoint_generation_controller generation;
        std::vector<vbr_artifact_stream_placement> placements;
    };

    static bool settle(llama_kv_cache & cache) {
        return cache.vbr_capture_settle();
    }

    static bool runtime_pools(
            llama_kv_cache & cache,
            std::vector<vbr_explicit_capture_runtime_pool> & output) {
        if (cache.other != nullptr) {
            return runtime_pools(*cache.other, output);
        }
        const auto instance = cache.vbr_instance_id();
        const auto * tracker =
            cache.vbr_generation_tracker_get();
        if (!cache.vbr_operation_armed() ||
            tracker == nullptr || !tracker->active() ||
            !vbr_controller_instance_id_is_set(instance) ||
            cache.vbr_pools_.empty()) {
            return false;
        }
        for (const auto & pool : cache.vbr_pools_) {
            if (pool.vmm == nullptr || pool.buf == nullptr ||
                pool.device < 0) {
                return false;
            }
            const auto backend_device =
                ggml_backend_buft_get_device(
                    ggml_backend_buffer_get_type(pool.buf));
            if (backend_device == nullptr) {
                return false;
            }
            const auto duplicate = std::find_if(
                output.begin(), output.end(),
                [&](const auto & current) {
                    return current.instance_id == instance &&
                           current.device == pool.device;
                });
            if (duplicate != output.end()) {
                if (duplicate->backend_device != backend_device ||
                    (duplicate->backend != nullptr &&
                     pool.backend != nullptr &&
                     duplicate->backend != pool.backend)) {
                    return false;
                }
                continue;
            }
            output.push_back({
                instance, pool.device, backend_device, pool.backend,
            });
        }
        return true;
    }

    static bool empty(const llama_kv_cache & cache) noexcept {
        return cache.other == nullptr &&
            std::all_of(cache.v_cells.begin(), cache.v_cells.end(),
                [](const llama_kv_cells & cells) {
                    return cells.get_used() == 0;
                }) &&
            std::all_of(cache.vbr_pools_.begin(), cache.vbr_pools_.end(),
                [](const llama_kv_cache::vbr_pool & pool) {
                    return pool.wm_cells == 0;
                });
    }

    static bool append_policy_identity(
            const llama_kv_cache & cache,
            llama_sha256_writer & writer) noexcept {
        llama_kv_cache::vbr_capture_stability_token token;
        if (!cache.vbr_capture_policy_snapshot(token)) {
            return false;
        }
        writer.u64(cache.vbr_representation_epoch());
        writer.bytes(token.degrade_order_digest.data(),
                     token.degrade_order_digest.size());
        writer.bytes(token.policy_digest.data(),
                     token.policy_digest.size());
        writer.u64(token.degrade_cursor);
        writer.u32(uint32_t(token.floor_type));
        writer.u64(token.pressure_independent_settings);
        return true;
    }

    static bool fill_import_child(
            const llama_memory_tree_child & tree_child,
            const vbr_artifact_package_view & package,
            const std::vector<llama_vbr_artifact_domain_binding> & bindings,
            bool previously_observed,
            uint64_t policy_epoch,
            vbr_target_child_snapshot & output,
            bool & type_mismatch) noexcept {
        auto * cache = tree_child.attention;
        if (!cache || !cache->vbr_operation_armed()) {
            return false;
        }
        const auto * tracker = cache->vbr_generation_tracker_get();
        llama_kv_cache::vbr_capture_stability_token live_policy;
        if (!tracker || !tracker->active() || !tracker->stable() ||
            !cache->vbr_capture_policy_snapshot(live_policy) ||
            tree_child.child_id >=
                package.manifest().controller_policy.size()) {
            return false;
        }
        const auto & source_policy =
            package.manifest().controller_policy[tree_child.child_id];
        if (live_policy.degrade_order_digest !=
                source_policy.degrade_order_digest ||
            live_policy.floor_type != source_policy.floor_type ||
            live_policy.pressure_independent_settings !=
                source_policy.pressure_independent_settings ||
            cache->n_stream != source_policy.n_stream ||
            (cache->n_stream == 1) != source_policy.unified) {
            return false;
        }

        output = {};
        output.child_id = tree_child.child_id;
        output.dependency_mode = tree_child.dependency_mode;
        output.memory_cookie = cache;
        output.empty = empty(*cache);
        output.dedicated = cache->other == nullptr;
        output.armed = true;
        output.previously_observed = previously_observed;
        output.lineage_uuid = tracker->lineage_identity();
        output.instance_id = tracker->runtime_instance();
        output.state_serial = cache->vbr_representation_epoch_;
        output.policy_epoch = policy_epoch;
        output.controller_policy = source_policy;

        for (const auto & source_unit : package.units()) {
            const auto & descriptor = source_unit.descriptor;
            if (descriptor.child_id != tree_child.child_id) {
                continue;
            }
            const size_t layer = descriptor.logical_unit_id/2;
            const bool value_side =
                (descriptor.logical_unit_id & 1u) != 0;
            if (layer >= cache->layers.size()) {
                return false;
            }
            const auto & extents =
                cache->vbr_units_of(layer, value_side);
            ggml_tensor * tensor = value_side
                ? cache->layers[layer].v : cache->layers[layer].k;
            if (!tensor ||
                extents.size() != descriptor.shards.size()) {
                return false;
            }
            type_mismatch = type_mismatch ||
                int32_t(tensor->type) != descriptor.current_type;
            const auto live_generation = tracker->unit_generation(
                descriptor.logical_unit_id);
            vbr_target_unit_snapshot target;
            target.child_id = descriptor.child_id;
            target.logical_unit_id = descriptor.logical_unit_id;
            target.current_type = tensor->type;
            target.last_source_type = live_generation.last_source_type;
            target.promote_hops = live_generation.promote_hops;
            target.last_transition = live_generation.last_transition;
            target.representation_kind = descriptor.representation.kind;
            target.codec_id = descriptor.representation.codec_id;
            target.codec_version = descriptor.representation.codec_version;
            target.representation_reference_digest =
                descriptor.representation.reference_digest;
            target.source_loss_history =
                descriptor.representation.source_loss_history;
            target.checkpoint_codec_hops =
                descriptor.representation.checkpoint_codec_hops;
            target.recoverability = descriptor.recoverability;
            target.side = descriptor.side;
            target.layout = descriptor.layout;
            target.row_codec_version = descriptor.row_codec_version;
            target.current_domain = live_generation.domain;
            target.codebook_digest = descriptor.codebook_digest;
            target.rotation_digest = descriptor.rotation_digest;
            target.meansub_digest = descriptor.meansub_digest;
            target.n_stream = descriptor.n_stream;
            target.unified = descriptor.unified;
            target.v_trans = false;
            target.wm_cells = descriptor.wm_cells;
            target.rank = descriptor.rank;
            target.dimensions = descriptor.dimensions;
            target.row_alignment = descriptor.row_alignment;
            for (size_t i = 0; i < extents.size(); ++i) {
                const auto & source = descriptor.shards[i];
                const auto binding = std::find_if(
                    bindings.begin(), bindings.end(),
                    [&](const llama_vbr_artifact_domain_binding & value) {
                        return value.topology_index == source.topology_index &&
                               value.device_ordinal == source.device_ordinal;
                    });
                if (binding == bindings.end() ||
                    source.topology_index >= package.topologies().size() ||
                    !extents[i].first || !extents[i].second ||
                    !extents[i].second->t) {
                    return false;
                }
                target.shards.push_back({
                    uint32_t(i), extents[i].first, binding->domain,
                    source.topology_index, source.device_ordinal,
                    package.topologies()[source.topology_index].digest,
                    source.logical_offset, source.row_count,
                    uint64_t(ggml_row_size(
                        extents[i].second->t->type,
                        extents[i].second->t->ne[0])),
                    uint64_t(ggml_nbytes(extents[i].second->t)),
                });
            }
            output.units.push_back(std::move(target));
        }
        return !output.units.empty();
    }

    static bool recheck_import_child(
            const llama_kv_cache & cache,
            const vbr_child_empty_fingerprint & expected) noexcept {
        const auto * tracker = cache.vbr_generation_tracker_get();
        return expected.memory_cookie == &cache &&
               expected.state_serial == cache.vbr_representation_epoch() &&
               tracker &&
               expected.instance_id == tracker->runtime_instance() &&
               empty(cache);
    }

    static bool reserve_import_downward(
            llama_kv_cache & cache,
            const std::vector<const vbr_validated_child_plan *> & plans,
            llama_cache_acct_ledger & ledger,
            const llama_cache_budget_config & budget,
            vbr_downward_stage_reservation & output) noexcept {
        return cache.vbr_downward_reserve_import(
            plans, ledger, budget, output);
    }

    static bool bind_import_downward(
            const std::vector<llama_memory_tree_child> & tree,
            const vbr_artifact_package_view & package,
            vbr_target_validation_snapshot & output,
            vbr_downward_policy_projection & projection) noexcept {
        struct indexed_child {
            llama_kv_cache * cache = nullptr;
            vbr_target_child_snapshot * target = nullptr;
            std::vector<const vbr_artifact_unit_view *> units;
        };
        const size_t n_controller =
            package.manifest().controller_policy.size();
        std::vector<const llama_memory_tree_child *> tree_index(
            n_controller, nullptr);
        for (const auto & child : tree) {
            if (!child.attention) {
                continue;
            }
            if (child.child_id >= tree_index.size() ||
                tree_index[child.child_id] != nullptr) {
                return false;
            }
            tree_index[child.child_id] = &child;
        }
        std::vector<size_t> child_position(n_controller, SIZE_MAX);
        std::vector<indexed_child> indexed(output.children.size());
        for (size_t i = 0; i < output.children.size(); ++i) {
            auto & target = output.children[i];
            if (target.child_id >= tree_index.size() ||
                child_position[target.child_id] != SIZE_MAX) {
                return false;
            }
            const auto * live = tree_index[target.child_id];
            if (!live || !live->attention ||
                live->attention->vbr_pools_.empty()) {
                return false;
            }
            child_position[target.child_id] = i;
            indexed[i].cache = live->attention;
            indexed[i].target = &target;
            indexed[i].units.resize(
                live->attention->layers.size()*2, nullptr);
        }
        for (const auto & unit : package.units()) {
            const auto child_id = unit.descriptor.child_id;
            const auto unit_id = unit.descriptor.logical_unit_id;
            if (child_id >= child_position.size() ||
                child_position[child_id] == SIZE_MAX) {
                return false;
            }
            auto & units = indexed[child_position[child_id]].units;
            if (unit_id >= units.size() || units[unit_id] != nullptr) {
                return false;
            }
            units[unit_id] = &unit;
        }

        std::vector<vbr_downward_policy_child> policy_children;
        policy_children.reserve(indexed.size());
        for (auto & child : indexed) {
            std::vector<ggml_type> source_types;
            source_types.reserve(child.units.size());
            for (const auto * unit : child.units) {
                if (!unit) {
                    return false;
                }
                source_types.push_back(static_cast<ggml_type>(
                    unit->descriptor.current_type));
            }
            const auto & source_policy = package.manifest().
                controller_policy[child.target->child_id];
            if (source_policy.wm_cells > UINT32_MAX ||
                !child.cache->vbr_downward_policy_input(
                    source_types, source_policy.cursor,
                    uint32_t(source_policy.wm_cells),
                    child.cache->vbr_pools_.front().device,
                    policy_children.emplace_back())) {
                return false;
            }
        }
        projection = vbr_downward_project_policy_prefix(policy_children);
        if (projection.status != vbr_downward_policy_status::coherent) {
            return false;
        }
        for (size_t child_index = 0; child_index < indexed.size();
             ++child_index) {
            auto & child = indexed[child_index];
            for (auto & unit : child.target->units) {
                if (unit.logical_unit_id >= child.units.size() ||
                    !child.units[unit.logical_unit_id]) {
                    return false;
                }
                const auto source_type = static_cast<ggml_type>(
                    child.units[unit.logical_unit_id]->descriptor.current_type);
                // The projected prefix is tree-wide, but only units whose
                // representation changes need a transcode recipe. Unchanged
                // units retain their native geometry and are validated by the
                // ordinary representation path.
                if (source_type != unit.current_type &&
                    !child.cache->vbr_downward_bind_target_unit(
                        source_type, projection,
                        uint32_t(child_index), unit)) {
                    return false;
                }
            }
            child.target->controller_policy.current_type_vector_digest =
                projection.child_type_digests[child_index];
            child.target->controller_policy.cursor =
                projection.final_cursors[child_index];
        }
        return true;
    }

    static bool capture_metadata(
            llama_kv_cache & cache,
            uint32_t child_id,
            checkpoint_child_dependency_mode mode,
            llama_seq_id sequence,
            llama_pos frontier,
            const std::vector<vbr_explicit_capture_pool_binding> & bindings,
            child & output,
            vbr_explicit_generation_failure & failure,
            vbr_explicit_size_failure & size_failure) {
        failure = vbr_explicit_generation_failure::none;
        size_failure = vbr_explicit_size_failure::none;
        llama_kv_cache::vbr_capture_unit_request request;
        request.child_id = child_id;
        request.bindings = &bindings;
        output.child_id = child_id;
        output.dependency_mode = mode;
        output.cache = &cache;
        // Snapshot the byte geometry/generation token first, then capture
        // ownership.  The final stability reread binds the ownership record
        // to that exact token: a mutation between these two calls advances a
        // monotone controller serial or unit publish_seq and fails closed.
        if (!cache.vbr_capture_size_pass(
                request, output.units, output.stability,
                &size_failure)) {
            failure = vbr_explicit_generation_failure::size_pass;
            return false;
        }
        vbr_artifact_stream_placement placement;
        auto * placement_out = mode ==
                checkpoint_child_dependency_mode::live_guarded ?
            &placement : nullptr;
        if (!cache.vbr_capture_generation_record(
                child_id, mode, sequence, frontier,
                output.generation, placement_out, &failure)) {
            if (failure == vbr_explicit_generation_failure::none) {
                failure =
                    vbr_explicit_generation_failure::internal_error;
            }
            return false;
        }
        if (placement_out != nullptr) {
            output.placements.push_back(std::move(placement));
        }
        if (!cache.vbr_capture_stability_matches(output.stability)) {
            failure =
                vbr_explicit_generation_failure::stability_reread_failed;
            return false;
        }
        return true;
    }

    static bool capture_size(
            child & output,
            const std::vector<vbr_explicit_capture_pool_binding> & bindings,
            vbr_explicit_generation_failure & failure,
            vbr_explicit_size_failure & size_failure) {
        failure = vbr_explicit_generation_failure::none;
        size_failure = vbr_explicit_size_failure::none;
        if (output.cache == nullptr) {
            failure = vbr_explicit_generation_failure::internal_error;
            return false;
        }
        llama_kv_cache::vbr_capture_unit_request request;
        request.child_id = output.child_id;
        request.bindings = &bindings;
        if (!output.cache->vbr_capture_size_pass(
                request, output.units, output.stability, &size_failure)) {
            failure = vbr_explicit_generation_failure::size_pass;
            return false;
        }
        return true;
    }

    static bool capture_generation(
            const child & value,
            llama_seq_id sequence,
            llama_pos frontier,
            vbr_checkpoint_generation_controller & generation,
            vbr_artifact_stream_placement & placement,
            vbr_explicit_generation_failure & failure) {
        failure = vbr_explicit_generation_failure::none;
        if (value.cache == nullptr ||
            !value.cache->vbr_capture_generation_record(
                value.child_id, value.dependency_mode,
                sequence, frontier, generation, &placement, &failure)) {
            if (failure == vbr_explicit_generation_failure::none) {
                failure = vbr_explicit_generation_failure::internal_error;
            }
            return false;
        }
        return true;
    }

    static vbr_controller_instance_id instance(const child & value) noexcept {
        return value.cache ? value.cache->vbr_instance_id() :
            vbr_controller_instance_id {};
    }

    static vbr_capture_stream_status transfer_projected_unit(
            const child & value,
            const llama_kv_cache::vbr_capture_unit_plan & plan,
            const vbr_capture_projection & projection,
            uint64_t source_namespace,
            vbr_pinned_chunk_ring & ring,
            vbr_capture_projected_unit & output,
            void * continue_context = nullptr,
            vbr_projected_capture_batch_request::continue_transfer_fn
                continue_transfer = nullptr,
            vbr_capture_stream_stats * attempted = nullptr,
            const vbr_pinned_ring_operation * operation = nullptr) noexcept {
        if (!value.cache) {
            return vbr_capture_stream_status::snapshot_unavailable;
        }
        std::vector<vbr_capture_projected_shard_source> sources;
        if (!value.cache->vbr_capture_projected_sources(plan, sources)) {
            return vbr_capture_stream_status::snapshot_unavailable;
        }
        for (auto & source : sources) {
            source.source.continue_context = continue_context;
            source.source.continue_transfer = continue_transfer;
        }
        llama_kv_cache::vbr_capture_snapshot_session session;
        if (!value.cache->vbr_capture_snapshot_bind(
                plan, sources, source_namespace, session)) {
            return vbr_capture_stream_status::snapshot_unavailable;
        }
        return vbr_capture_projected_unit_transfer(
            projection, value.child_id, 0, plan.logical_unit,
            sources, {}, session.provider(), ring, output, attempted,
            operation);
    }

    struct representation_cache_entry {
        int32_t current_type = -1;
        bool value_side = false;
        vbr_explicit_representation_identity identity;
    };

    static bool capture_schema(
            const child & value,
            const void * representation_context,
            vbr_explicit_capture_request::representation_identity_fn
                representation_identity,
            bool allow_clean_stash,
            std::vector<representation_cache_entry> & representation_cache,
            vbr_artifact_controller_policy & policy,
            std::vector<vbr_artifact_unit_descriptor> & descriptors,
            vbr_explicit_capture_status & status) {
        descriptors.clear();
        if (value.units.empty() || !representation_identity) {
            status = vbr_explicit_capture_status::identity_unavailable;
            return false;
        }
        const auto & first = value.units.front();
        if (first.n_stream == 0 || first.wm_cells == 0) {
            status = vbr_explicit_capture_status::unsupported_layout;
            return false;
        }

        policy = {};
        policy.child_id = value.child_id;
        policy.dependency_mode = value.dependency_mode;
        policy.degrade_order_digest =
            value.stability.degrade_order_digest;
        policy.policy_digest = value.stability.policy_digest;
        policy.cursor = value.stability.degrade_cursor;
        policy.floor_type = value.stability.floor_type;
        policy.pressure_independent_settings =
            value.stability.pressure_independent_settings;
        policy.n_stream = first.n_stream;
        policy.unified = first.unified;
        policy.wm_cells = first.wm_cells;
        policy.completed_wave = value.stability.completed_wave;

        std::vector<ggml_type> current_types;
        current_types.reserve(value.units.size());
        descriptors.reserve(value.units.size());
        for (const auto & plan : value.units) {
            if (plan.child_id != value.child_id ||
                plan.logical_unit != descriptors.size() ||
                plan.n_stream != policy.n_stream ||
                plan.unified != policy.unified ||
                plan.wm_cells != policy.wm_cells ||
                plan.shards.empty()) {
                status = vbr_explicit_capture_status::unsupported_layout;
                return false;
            }
            vbr_artifact_unit_descriptor descriptor;
            descriptor.child_id = value.child_id;
            descriptor.logical_unit_id = plan.logical_unit;
            descriptor.lineage_uuid = value.stability.lineage_uuid;
            descriptor.repr_gen = plan.generation.repr_gen;
            descriptor.current_type = plan.generation.current_type;
            descriptor.last_source_type =
                plan.generation.last_source_type;
            descriptor.promote_hops = plan.generation.promote_hops;
            descriptor.last_transition = plan.generation.last_transition;
            descriptor.representation.kind =
                plan.generation.current_type == GGML_TYPE_F16
                    ? vbr_artifact_representation_kind::raw
                    : vbr_artifact_representation_kind::approximate;
            auto representation = std::find_if(
                representation_cache.begin(), representation_cache.end(),
                [&](const auto & cached) {
                    return cached.current_type ==
                            plan.generation.current_type &&
                        cached.value_side == plan.is_v;
                });
            if (representation == representation_cache.end()) {
                vbr_explicit_representation_identity identity;
                if (!representation_identity(
                        representation_context,
                        plan.generation.current_type,
                        plan.is_v, identity)) {
                    status =
                        vbr_explicit_capture_status::identity_unavailable;
                    return false;
                }
                representation_cache.push_back({
                    plan.generation.current_type, plan.is_v,
                    std::move(identity),
                });
                representation = std::prev(representation_cache.end());
            }
            const auto & identity = representation->identity;
            if (
                identity.codec_id == 0 ||
                identity.codec_version == 0 ||
                !digest_nonzero(identity.codebook_digest) ||
                !digest_nonzero(identity.rotation_digest) ||
                !digest_nonzero(identity.meansub_digest)) {
                status = vbr_explicit_capture_status::identity_unavailable;
                return false;
            }
            descriptor.representation.codec_id = identity.codec_id;
            descriptor.representation.codec_version =
                identity.codec_version;
            llama_sha256_writer representation_hash;
            static constexpr char REPRESENTATION_DOMAIN[] =
                "buun.vbr.capture/representation";
            representation_hash.string(
                REPRESENTATION_DOMAIN,
                sizeof(REPRESENTATION_DOMAIN) - 1);
            representation_hash.u32(
                uint32_t(plan.generation.current_type));
            representation_hash.u32(
                uint32_t(plan.generation.last_source_type));
            representation_hash.u32(identity.codec_id);
            representation_hash.u32(identity.codec_version);
            representation_hash.bytes(
                identity.codebook_digest.data(),
                identity.codebook_digest.size());
            representation_hash.bytes(
                identity.rotation_digest.data(),
                identity.rotation_digest.size());
            representation_hash.bytes(
                identity.meansub_digest.data(),
                identity.meansub_digest.size());
            descriptor.representation.reference_digest =
                representation_hash.finish();
            descriptor.representation.source_loss_history =
                plan.generation.promote_hops;
            descriptor.side = plan.is_v
                ? vbr_artifact_side::value : vbr_artifact_side::key;
            descriptor.n_stream = plan.n_stream;
            descriptor.unified = plan.unified;
            descriptor.wm_cells = plan.wm_cells;
            descriptor.rank = 2;
            uint64_t total_columns = 0;
            uint64_t logical_offset = 0;
            bool has_stash = false;
            uint32_t stash_rows = 0;
            for (const auto & shard : plan.shards) {
                if (shard.columns == 0 ||
                    shard.columns > UINT64_MAX - total_columns) {
                    status = vbr_explicit_capture_status::size_overflow;
                    return false;
                }
                if (shard.stash_bytes != 0 && !allow_clean_stash) {
                    status = vbr_explicit_capture_status::unsupported_layout;
                    return false;
                }
                total_columns += shard.columns;
                vbr_artifact_shard_descriptor wire;
                wire.shard_index = shard.shard_index;
                wire.topology_index = shard.topology_index;
                wire.device_ordinal = shard.device_ordinal;
                wire.logical_offset = logical_offset;
                wire.row_count = plan.wm_cells;
                wire.column_count = shard.columns;
                wire.row_bytes = shard.row_bytes;
                wire.payload_bytes = shard.payload_bytes;
                descriptor.shards.push_back(wire);
                logical_offset += shard.columns;
                if (shard.stash_bytes != 0) {
                    if (shard.columns > UINT64_MAX/sizeof(uint16_t) ||
                        shard.stash_bytes %
                            (shard.columns*sizeof(uint16_t)) != 0 ||
                        shard.stash_bytes /
                            (shard.columns*sizeof(uint16_t)) > UINT32_MAX) {
                        status =
                            vbr_explicit_capture_status::stash_inconsistent;
                        return false;
                    }
                    const uint32_t rows = uint32_t(
                        shard.stash_bytes /
                            (shard.columns*sizeof(uint16_t)));
                    if (has_stash && rows != stash_rows) {
                        status =
                            vbr_explicit_capture_status::stash_inconsistent;
                        return false;
                    }
                    has_stash = true;
                    stash_rows = rows;
                }
            }
            descriptor.dimensions = { plan.wm_cells, total_columns, 0, 0 };
            descriptor.row_alignment = 1;
            descriptor.row_codec_version = 1;
            descriptor.codebook_digest = identity.codebook_digest;
            descriptor.rotation_digest = identity.rotation_digest;
            descriptor.meansub_digest = identity.meansub_digest;
            descriptor.clean_stash_state = has_stash
                ? vbr_artifact_clean_stash_state::present
                : vbr_artifact_clean_stash_state::absent_at_source;
            if (has_stash) {
                if (total_columns > UINT64_MAX/sizeof(uint16_t)) {
                    status = vbr_explicit_capture_status::size_overflow;
                    return false;
                }
                descriptor.clean_stash.valid_rows = stash_rows;
                descriptor.clean_stash.domain = vbr_repr_domain::tapped;
                descriptor.clean_stash.row_count = stash_rows;
                descriptor.clean_stash.column_count = total_columns;
                descriptor.clean_stash.row_bytes =
                    total_columns*sizeof(uint16_t);
                logical_offset = 0;
                for (const auto & shard : plan.shards) {
                    if (shard.stash_bytes == 0) {
                        status =
                            vbr_explicit_capture_status::stash_inconsistent;
                        return false;
                    }
                    vbr_artifact_shard_descriptor wire;
                    wire.shard_index = shard.shard_index;
                    wire.topology_index = shard.topology_index;
                    wire.device_ordinal = shard.device_ordinal;
                    wire.logical_offset = logical_offset;
                    wire.row_count = stash_rows;
                    wire.column_count = shard.columns;
                    wire.row_bytes = shard.columns*sizeof(uint16_t);
                    wire.payload_bytes = shard.stash_bytes;
                    descriptor.clean_stash.shards.push_back(wire);
                    logical_offset += shard.columns;
                }
            }
            current_types.push_back(
                static_cast<ggml_type>(plan.generation.current_type));
            descriptors.push_back(std::move(descriptor));
        }
        policy.current_type_vector_digest =
            vbr_type_vector_digest(current_types);
        status = vbr_explicit_capture_status::ok;
        return true;
    }

    struct projected_target_expectation {
        uint64_t manifest_id = 0;
        llama_seq_id sequence = -1;
        llama_pos frontier = -1;
        const child * source = nullptr;
        vbr_checkpoint_generation_controller generation;
        vbr_artifact_stream_placement placement;
        vbr_capture_controller_target target;
    };

    struct projected_target_recheck_context {
        const std::vector<projected_target_expectation> * expectations =
            nullptr;
    };

    static bool projected_targets_recheck(
            void * opaque,
            uint64_t manifest_id,
            const vbr_capture_controller_target * targets,
            size_t target_count) noexcept {
        try {
            const auto * context =
                static_cast<const projected_target_recheck_context *>(opaque);
            if (!context || !context->expectations || !targets ||
                target_count == 0) {
                return false;
            }
            if (size_t(std::count_if(
                    context->expectations->begin(),
                    context->expectations->end(),
                    [&](const auto & value) {
                        return value.manifest_id == manifest_id;
                    })) != target_count) {
                return false;
            }
            for (size_t i = 0; i < target_count; ++i) {
                const auto & target = targets[i];
                for (size_t prior = 0; prior < i; ++prior) {
                    if (targets[prior].child_id == target.child_id) {
                        return false;
                    }
                }
                const auto expected = std::find_if(
                    context->expectations->begin(),
                    context->expectations->end(),
                    [&](const auto & value) {
                        return value.manifest_id == manifest_id &&
                               value.target.child_id == target.child_id;
                    });
                if (expected == context->expectations->end() ||
                    expected->source == nullptr ||
                    target.manifest_id != expected->target.manifest_id ||
                    target.source_namespace !=
                        expected->target.source_namespace ||
                    !vbr_capture_controller_representation_equal(
                        target, expected->target) ||
                    !stable(*expected->source)) {
                    return false;
                }
                vbr_checkpoint_generation_controller current;
                vbr_artifact_stream_placement placement;
                vbr_explicit_generation_failure failure;
                if (!capture_generation(
                        *expected->source, expected->sequence,
                        expected->frontier, current, placement, failure) ||
                    !(current == expected->generation) ||
                    placement.child_id != expected->placement.child_id ||
                    placement.stream_index !=
                        expected->placement.stream_index ||
                    placement.source_sequence !=
                        expected->placement.source_sequence ||
                    placement.computation_frontier !=
                        expected->placement.computation_frontier ||
                    placement.cells.size() !=
                        expected->placement.cells.size()) {
                    return false;
                }
                for (size_t cell = 0;
                     cell < placement.cells.size(); ++cell) {
                    const auto & lhs = placement.cells[cell];
                    const auto & rhs = expected->placement.cells[cell];
                    if (lhs.physical_cell != rhs.physical_cell ||
                        lhs.logical_position != rhs.logical_position ||
                        lhs.ext_x != rhs.ext_x || lhs.ext_y != rhs.ext_y) {
                        return false;
                    }
                }
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    static bool stable(const child & value) {
        return value.cache != nullptr &&
               value.cache->vbr_capture_stability_matches(value.stability);
    }

    static bool stream(
            const child & value,
            const llama_kv_cache::vbr_capture_unit_plan & unit,
            vbr_unit_build & sink,
            vbr_pinned_chunk_ring & ring,
            vbr_capture_stream_stats & stats) {
        return value.cache != nullptr &&
               value.cache->vbr_capture_stream_unit(
                   unit, sink, ring, stats);
    }
};

namespace {

uint64_t import_tree_digest(
        llama_memory_i & memory,
        const std::vector<llama_memory_tree_child> & tree) {
    llama_sha256_writer writer;
    static constexpr char domain_label[] = "buun.vbr.import-tree/v1";
    writer.string(domain_label, sizeof(domain_label) - 1);
    writer.u64(uint64_t(reinterpret_cast<uintptr_t>(&memory)));
    for (const auto & child : tree) {
        writer.u32(child.child_id);
        writer.u32(uint32_t(child.dependency_mode));
        writer.u32(child.attention != nullptr);
        writer.u32(child.recurrent != nullptr);
        writer.u64(uint64_t(reinterpret_cast<uintptr_t>(
            child.attention ? static_cast<void *>(child.attention) :
            static_cast<void *>(child.recurrent))));
    }
    return digest_head_u64(writer);
}

uint64_t import_policy_epoch(
        const std::vector<llama_memory_tree_child> & tree) {
    llama_sha256_writer writer;
    static constexpr char domain_label[] = "buun.vbr.import-policy/v1";
    writer.string(domain_label, sizeof(domain_label) - 1);
    size_t n_attention = 0;
    for (const auto & child : tree) {
        if (!child.attention) {
            continue;
        }
        writer.u32(child.child_id);
        if (!vbr_live_capture_adapter::append_policy_identity(
                *child.attention, writer)) {
            return 0;
        }
        ++n_attention;
    }
    if (n_attention == 0) {
        return 0;
    }
    return digest_head_u64(writer);
}

} // namespace

namespace {

bool recurrent_target_empty(
        const llama_memory_tree_child & child) noexcept {
    if (!child.recurrent || child.attention) {
        return false;
    }
    const auto provider =
        vbr_recurrent_companion_adoption_provider(*child.recurrent);
    return provider.target_empty &&
           provider.target_empty(provider.context);
}

} // namespace

bool vbr_explicit_capture_runtime_pools(
        llama_memory_i & memory,
        std::vector<vbr_explicit_capture_runtime_pool> & pools,
        uint32_t & attention_children) noexcept {
    pools.clear();
    attention_children = 0;
    try {
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(&memory, tree)) {
            return false;
        }
        for (const auto & node : tree) {
            if (node.attention == nullptr) {
                continue;
            }
            ++attention_children;
            if (!vbr_live_capture_adapter::runtime_pools(
                    *node.attention, pools)) {
                pools.clear();
                attention_children = 0;
                return false;
            }
        }
        return attention_children != 0 && !pools.empty();
    } catch (...) {
        pools.clear();
        attention_children = 0;
        return false;
    }
}

vbr_projected_capture_batch_result vbr_capture_projected_batch(
        llama_memory_i & memory,
        const vbr_projected_capture_batch_request & request) noexcept {
    vbr_projected_capture_batch_result result;
    if (!request.idle_decode_thread || request.manifests.empty() ||
        request.manifests.size() > VBR_PROJECTED_CAPTURE_MAX_MANIFESTS ||
        request.ring == nullptr || request.topologies.empty() ||
        request.pool_bindings.empty() ||
        request.max_packed_bytes == 0 ||
        request.representation_identity == nullptr) {
        result.status = request.idle_decode_thread
            ? vbr_explicit_capture_status::identity_unavailable
            : vbr_explicit_capture_status::slot_not_idle;
        return result;
    }

    try {
        result.phase = vbr_explicit_capture_phase::memory_tree;
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(&memory, tree)) {
            result.status = vbr_explicit_capture_status::unsupported_layout;
            return result;
        }
        std::vector<vbr_live_capture_adapter::child> children;
        std::vector<llama_memory_recurrent *> recurrent_children;
        for (const auto & node : tree) {
            if (node.recurrent != nullptr) {
                recurrent_children.push_back(node.recurrent);
                continue;
            }
            if (node.attention == nullptr ||
                !node.attention->vbr_operation_armed() ||
                node.dependency_mode !=
                    checkpoint_child_dependency_mode::live_guarded) {
                result.status = vbr_explicit_capture_status::unsupported_layout;
                return result;
            }
            vbr_live_capture_adapter::child child;
            child.child_id = node.child_id;
            child.dependency_mode = node.dependency_mode;
            child.cache = node.attention;
            children.push_back(std::move(child));
        }
        if (children.empty()) {
            result.status = vbr_explicit_capture_status::not_armed;
            return result;
        }

        result.phase = vbr_explicit_capture_phase::settlement;
        std::vector<vbr_controller_instance_id> instances;
        instances.reserve(children.size());
        for (auto & child : children) {
            if (!vbr_live_capture_adapter::settle(*child.cache)) {
                result.status =
                    vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
            const auto instance =
                vbr_live_capture_adapter::instance(child);
            if (!vbr_controller_instance_id_is_set(instance) ||
                std::find(instances.begin(), instances.end(), instance) !=
                    instances.end() ||
                vbr_recovery_pending_for(instance)) {
                result.status =
                    vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
            instances.push_back(instance);
        }
        if (!vbr_operation_registry_quiescent_for(
                instances.data(), instances.size())) {
            result.status = vbr_explicit_capture_status::registry_busy;
            return result;
        }

        result.phase = vbr_explicit_capture_phase::metadata_and_manifest;
        struct child_schema {
            vbr_artifact_controller_policy policy;
            std::vector<vbr_artifact_unit_descriptor> descriptors;
        };
        std::vector<child_schema> schemas(children.size());
        std::vector<vbr_live_capture_adapter::representation_cache_entry>
            representation_cache;
        for (size_t i = 0; i < children.size(); ++i) {
            ++result.size_pass_calls;
            if (!vbr_live_capture_adapter::capture_size(
                    children[i], request.pool_bindings,
                    result.generation_failure, result.size_failure)) {
                result.status =
                    vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
            if (children[i].dependency_mode !=
                    checkpoint_child_dependency_mode::live_guarded ||
                children[i].units.empty() ||
                children[i].units.front().n_stream != 1 ||
                !children[i].units.front().unified) {
                result.status =
                    vbr_explicit_capture_status::unsupported_layout;
                return result;
            }
            if (!vbr_live_capture_adapter::capture_schema(
                    children[i], request.representation_context,
                    request.representation_identity,
                    false,
                    representation_cache,
                    schemas[i].policy, schemas[i].descriptors,
                    result.status)) {
                if (result.status ==
                        vbr_explicit_capture_status::internal_error) {
                    result.status =
                        vbr_explicit_capture_status::generation_unavailable;
                }
                return result;
            }
        }

        llama_sha256_writer namespace_hash;
        static constexpr char NAMESPACE_DOMAIN[] =
            "buun.vbr.projected-capture/tree-namespace/v1";
        namespace_hash.string(
            NAMESPACE_DOMAIN, sizeof(NAMESPACE_DOMAIN) - 1);
        namespace_hash.u64(
            uint64_t(reinterpret_cast<uintptr_t>(&memory)));
        static std::atomic<uint64_t> next_capture_namespace { 1 };
        namespace_hash.u64(next_capture_namespace.fetch_add(
            1, std::memory_order_relaxed));
        for (size_t i = 0; i < children.size(); ++i) {
            namespace_hash.u32(children[i].child_id);
            namespace_hash.u64(instances[i].hi);
            namespace_hash.u64(instances[i].lo);
        }
        result.source_namespace = digest_head_u64(namespace_hash);
        if (result.source_namespace == 0) {
            result.status =
                vbr_explicit_capture_status::generation_unavailable;
            return result;
        }

        std::vector<vbr_capture_projection_manifest> projection_manifests;
        std::vector<vbr_capture_controller_target> targets;
        std::vector<vbr_live_capture_adapter::projected_target_expectation>
            expectations;
        projection_manifests.reserve(request.manifests.size());
        targets.reserve(request.manifests.size()*children.size());
        expectations.reserve(request.manifests.size()*children.size());
        std::vector<uint64_t> manifest_ids;
        std::vector<std::vector<recurrent_companion_plan>> recurrent_plans;
        std::vector<bool> manifest_dependency_available;
        manifest_ids.reserve(request.manifests.size());
        recurrent_plans.reserve(request.manifests.size());
        manifest_dependency_available.reserve(request.manifests.size());
        for (const auto & manifest_request : request.manifests) {
            const auto & identity = manifest_request.identity;
            if (manifest_request.manifest_id == 0 ||
                manifest_request.sequence < 0 ||
                identity.execution_identity.empty() ||
                identity.adapter_config_identity.empty() ||
                identity.media_content_identity.empty() ||
                identity.sequence_epoch == 0 || identity.token_count <= 0 ||
                identity.next_position <= 0 ||
                manifest_request.token_block.size() !=
                    size_t(identity.token_count) ||
                std::find(
                    manifest_ids.begin(), manifest_ids.end(),
                    manifest_request.manifest_id) != manifest_ids.end()) {
                result.status =
                    vbr_explicit_capture_status::identity_unavailable;
                return result;
            }
            manifest_ids.push_back(manifest_request.manifest_id);
            vbr_capture_projection_manifest projected;
            projected.manifest_id = manifest_request.manifest_id;
            projected.identity = identity;
            projected.token_block.tokens = manifest_request.token_block;
            projected.generation.version = 1;
            projected.generation.status =
                vbr_checkpoint_generation_status::complete;
            bool dependencies_available = true;
            std::vector<recurrent_companion_plan> manifest_recurrent_plans;
            manifest_recurrent_plans.reserve(recurrent_children.size());
            for (auto * recurrent : recurrent_children) {
                recurrent_companion_plan companion;
                vbr_explicit_capture_status companion_status =
                    vbr_explicit_capture_status::
                        required_companion_unavailable;
                if (!recurrent_companion_prepare(
                        recurrent, manifest_request.sequence,
                        identity.next_position,
                        companion, companion_status)) {
                    dependencies_available = false;
                    projected.companions.clear();
                    manifest_recurrent_plans.clear();
                    break;
                }
                projected.companions.push_back(companion.descriptor);
                manifest_recurrent_plans.push_back(std::move(companion));
            }
            std::vector<vbr_identity_policy_digest_row> identity_policy;
            identity_policy.reserve(children.size());
            for (size_t child_index = 0;
                 child_index < children.size(); ++child_index) {
                auto & child = children[child_index];
                vbr_checkpoint_generation_controller generation;
                vbr_artifact_stream_placement placement;
                if (!vbr_live_capture_adapter::capture_generation(
                        child, manifest_request.sequence,
                        identity.next_position, generation, placement,
                        result.generation_failure)) {
                    result.status =
                        vbr_explicit_capture_status::generation_unavailable;
                    return result;
                }
                const llama_pos terminal_position =
                    identity.next_position - 1;
                const auto terminal_cell = std::max_element(
                    placement.cells.begin(), placement.cells.end(),
                    [](const auto & lhs, const auto & rhs) {
                        return lhs.logical_position < rhs.logical_position;
                    });
                if (terminal_cell == placement.cells.end() ||
                    terminal_cell->logical_position != terminal_position) {
                    dependencies_available = false;
                }
                projected.generation.controllers.push_back(generation);
                projected.placements.push_back(placement);
                identity_policy.push_back({
                    child.child_id, child.dependency_mode,
                    child.stability.lineage_uuid,
                });

                if (!dependencies_available) {
                    continue;
                }
                vbr_capture_controller_target target;
                target.manifest_id = manifest_request.manifest_id;
                target.source_namespace = result.source_namespace;
                target.child_id = child.child_id;
                target.lineage_uuid = child.stability.lineage_uuid;
                target.controller_generation =
                    child.stability.controller_generation;
                target.policy = schemas[child_index].policy;
                target.unit_descriptors = schemas[child_index].descriptors;
                target.units.reserve(child.units.size());
                for (const auto & unit : child.units) {
                    target.units.push_back(unit.generation);
                }
                vbr_live_capture_adapter::projected_target_expectation
                    expectation;
                expectation.manifest_id = manifest_request.manifest_id;
                expectation.sequence = manifest_request.sequence;
                expectation.frontier = identity.next_position;
                expectation.source = &child;
                expectation.generation = std::move(generation);
                expectation.placement = std::move(placement);
                expectation.target = target;
                expectations.push_back(std::move(expectation));
                targets.push_back(std::move(target));
            }
            vbr_checkpoint_frontier_fields frontier;
            frontier.execution_identity = identity.execution_identity.data();
            frontier.execution_identity_len =
                identity.execution_identity.size();
            frontier.adapter_config_identity =
                identity.adapter_config_identity.data();
            frontier.adapter_config_identity_len =
                identity.adapter_config_identity.size();
            frontier.media_content_identity =
                identity.media_content_identity.data();
            frontier.media_content_identity_len =
                identity.media_content_identity.size();
            frontier.sequence_epoch = identity.sequence_epoch;
            frontier.token_count = identity.token_count;
            frontier.next_position = identity.next_position;
            const auto identity_digest =
                vbr_identity_policy_digest(frontier, identity_policy);
            if (digest_nonzero(
                    manifest_request.identity_policy_order_digest) &&
                identity_digest !=
                    manifest_request.identity_policy_order_digest) {
                result.status =
                    vbr_explicit_capture_status::identity_unavailable;
                return result;
            }
            projected.identity_policy_order_digest = identity_digest;
            projected.generation.identity_policy_order_digest =
                identity_digest;
            recurrent_plans.push_back(
                std::move(manifest_recurrent_plans));
            manifest_dependency_available.push_back(
                dependencies_available);
            projection_manifests.push_back(std::move(projected));
        }

        const auto manifest_available = [&](uint64_t manifest_id) {
            const auto manifest = std::find(
                manifest_ids.begin(), manifest_ids.end(), manifest_id);
            return manifest != manifest_ids.end() &&
                manifest_dependency_available[size_t(
                    manifest - manifest_ids.begin())];
        };
        const auto apply_dependency_availability =
                [&](std::vector<vbr_capture_projection_manifest> & manifests) {
            for (auto & projected : manifests) {
                const auto manifest = std::find(
                    manifest_ids.begin(), manifest_ids.end(),
                    projected.manifest_id);
                if (manifest == manifest_ids.end()) {
                    continue;
                }
                const size_t manifest_index = size_t(
                    manifest - manifest_ids.begin());
                if (manifest_dependency_available[manifest_index]) {
                    continue;
                }
                projected.dependencies_available = false;
                projected.placements.clear();
                projected.companions.clear();
                recurrent_plans[manifest_index].clear();
            }
            targets.erase(
                std::remove_if(
                    targets.begin(), targets.end(),
                    [&](const auto & target) {
                        return !manifest_available(target.manifest_id);
                    }),
                targets.end());
            expectations.erase(
                std::remove_if(
                    expectations.begin(), expectations.end(),
                    [&](const auto & expectation) {
                        return !manifest_available(expectation.manifest_id);
                    }),
                expectations.end());
        };
        apply_dependency_availability(projection_manifests);

        vbr_capture_projection projection;
        vbr_capture_projection_limits projection_limits;
        projection_limits.max_manifests =
            VBR_PROJECTED_CAPTURE_MAX_MANIFESTS;
        ++result.projection_calls;
        if (!vbr_artifact_project_capture_union(
                { result.source_namespace,
                  std::move(projection_manifests) },
                projection_limits, projection)) {
            result.status = vbr_explicit_capture_status::generation_unavailable;
            return result;
        }
        result.union_cells = projection->union_cell_count;

        std::vector<vbr_projected_capture_batch_request::
            pretransfer_quote::staging_row> staging_rows;
        std::vector<vbr_artifact_portable_accounting_row>
            attention_accounting;
        std::vector<std::vector<vbr_artifact_portable_accounting_row>>
            attention_reserve_accounting(request.manifests.size());
        uint32_t projected_unit_count = 0;
        const auto preflight_projection_bytes = [&]() {
            result.planned_packed_bytes = 0;
            staging_rows.clear();
            attention_accounting.clear();
            for (auto & rows : attention_reserve_accounting) {
                rows.clear();
            }
            projected_unit_count = 0;
            const auto add_planned_bytes = [&] (
                    const vbr_artifact_portable_domain & domain,
                    uint64_t bytes) {
                if (bytes == 0) {
                    return true;
                }
                if (bytes > UINT64_MAX - result.planned_packed_bytes) {
                    result.status =
                        vbr_explicit_capture_status::size_overflow;
                    return false;
                }
                result.planned_packed_bytes += bytes;
                if (result.planned_packed_bytes > request.max_packed_bytes) {
                    result.status =
                        vbr_explicit_capture_status::accounting_failed;
                    return false;
                }
                const auto found = std::lower_bound(
                    staging_rows.begin(), staging_rows.end(), domain,
                    [&](const auto & row, const auto & key) {
                        return vbr_artifact_portable_domain_less(
                            row.domain, key);
                    });
                if (found != staging_rows.end()) {
                    if (found->domain != domain) {
                        staging_rows.insert(found, { domain, bytes });
                    } else if (bytes > UINT64_MAX - found->bytes) {
                        result.status =
                            vbr_explicit_capture_status::size_overflow;
                        return false;
                    } else {
                        found->bytes += bytes;
                    }
                } else {
                    staging_rows.push_back({ domain, bytes });
                }
                return true;
            };
            for (const auto & child : children) {
                const auto stream = std::find_if(
                    projection->streams.begin(), projection->streams.end(),
                    [&](const auto & value) {
                        return value.child_id == child.child_id &&
                            value.stream_index == 0;
                    });
                if (stream == projection->streams.end()) {
                    continue;
                }
                if (child.units.size() >
                        UINT32_MAX - projected_unit_count) {
                    result.status =
                        vbr_explicit_capture_status::size_overflow;
                    return false;
                }
                projected_unit_count += uint32_t(child.units.size());
                uint64_t projected_rows = 0;
                uint64_t owner_manifest = UINT64_MAX;
                for (const auto & segment : stream->segments) {
                    if (segment.cell_count > UINT64_MAX - projected_rows) {
                        result.status =
                            vbr_explicit_capture_status::size_overflow;
                        return false;
                    }
                    projected_rows += segment.cell_count;
                    if (segment.first_dependency >
                            projection->dependent_manifest_ids.size() ||
                        segment.dependency_count >
                            projection->dependent_manifest_ids.size() -
                                segment.first_dependency) {
                        result.status =
                            vbr_explicit_capture_status::internal_error;
                        return false;
                    }
                    for (uint32_t dependency = 0;
                         dependency < segment.dependency_count;
                         ++dependency) {
                        owner_manifest = std::min(
                            owner_manifest,
                            projection->dependent_manifest_ids[
                                segment.first_dependency + dependency]);
                    }
                }
                const auto owner = std::lower_bound(
                    manifest_ids.begin(), manifest_ids.end(),
                    owner_manifest);
                if (owner_manifest == UINT64_MAX ||
                    owner == manifest_ids.end() ||
                    *owner != owner_manifest) {
                    result.status =
                        vbr_explicit_capture_status::internal_error;
                    return false;
                }
                const size_t owner_index = size_t(owner - manifest_ids.begin());
                for (const auto & unit : child.units) {
                    for (const auto & shard : unit.shards) {
                        if (shard.row_bytes != 0 &&
                            projected_rows > UINT64_MAX/shard.row_bytes) {
                            result.status =
                                vbr_explicit_capture_status::size_overflow;
                            return false;
                        }
                        const vbr_artifact_portable_domain domain {
                            llama_cache_acct_residency::device,
                            llama_cache_acct_domain_kind::device_topology,
                            shard.topology_index,
                            shard.device_ordinal,
                        };
                        if (!add_planned_bytes(
                                domain,
                                projected_rows*shard.row_bytes)) {
                            return false;
                        }
                        add_accounting(
                            attention_accounting,
                            vbr_artifact_accounting_role::unit_payload,
                            domain,
                            projected_rows*shard.row_bytes);
                        add_accounting(
                            attention_reserve_accounting[owner_index],
                            vbr_artifact_accounting_role::unit_payload,
                            domain,
                            projected_rows*shard.row_bytes);
                    }
                }
            }
            for (size_t manifest_index = 0;
                 manifest_index < recurrent_plans.size(); ++manifest_index) {
                if (!manifest_dependency_available[manifest_index]) {
                    continue;
                }
                for (const auto & companion :
                     recurrent_plans[manifest_index]) {
                    if (!add_planned_bytes(
                            companion.descriptor.domain,
                            companion.descriptor.payload_bytes)) {
                        return false;
                    }
                }
            }
            return true;
        };
        if (!preflight_projection_bytes()) {
            return result;
        }

        const auto build_pretransfer_quote = [&] (
                vbr_projected_capture_batch_request::pretransfer_quote &
                    quote) {
            quote = {};
            quote.planned_packed_bytes = result.planned_packed_bytes;
            quote.union_cells = result.union_cells;
            quote.manifests = uint32_t(std::count(
                manifest_dependency_available.begin(),
                manifest_dependency_available.end(), true));
            quote.projected_units = projected_unit_count;
            quote.staging = staging_rows;
            const auto host_domain = vbr_artifact_portable_domain {
                llama_cache_acct_residency::pageable_host,
                llama_cache_acct_domain_kind::not_applicable,
                UINT32_MAX, UINT16_MAX,
            };
            for (size_t manifest_index = 0;
                 manifest_index < manifest_ids.size(); ++manifest_index) {
                if (!manifest_dependency_available[manifest_index]) {
                    continue;
                }
                vbr_projected_capture_batch_request::pretransfer_quote::
                    durable_manifest durable;
                durable.manifest_id = manifest_ids[manifest_index];
                durable.accounting = attention_accounting;
                durable.reserve_accounting =
                    attention_reserve_accounting[manifest_index];
                add_accounting(
                    durable.accounting,
                    vbr_artifact_accounting_role::descriptor_metadata,
                    host_domain,
                    std::max<uint64_t>(
                        1, uint64_t(quote.projected_units)*256));
                add_accounting(
                    durable.accounting,
                    vbr_artifact_accounting_role::reference_metadata,
                    host_domain,
                    std::max<uint64_t>(
                        1, uint64_t(quote.projected_units)*128));
                for (const auto & companion :
                     recurrent_plans[manifest_index]) {
                    add_accounting(
                        durable.accounting,
                        vbr_artifact_accounting_role::recurrent_payload,
                        host_domain,
                        companion.descriptor.payload_bytes);
                }
                quote.durable.push_back(std::move(durable));
            }
            std::sort(
                quote.durable.begin(), quote.durable.end(),
                [](const auto & lhs, const auto & rhs) {
                    return lhs.manifest_id < rhs.manifest_id;
                });
            return true;
        };
        vbr_projected_capture_batch_request::pretransfer_quote current_quote;
        if (!build_pretransfer_quote(current_quote)) {
            return result;
        }
        if (request.pretransfer_admit) {
            result.phase =
                vbr_explicit_capture_phase::reservation_preparation;
            if (!request.pretransfer_admit(
                    request.pretransfer_context, current_quote)) {
                result.status =
                    vbr_explicit_capture_status::admission_refused;
                return result;
            }
        }

        vbr_pinned_ring_operation ring_operation;
        if (result.planned_packed_bytes != 0) {
            ++result.ring_operation_attempts;
            ring_operation = request.ring->try_begin_operation();
            if (!ring_operation) {
                ++result.ring_operation_refusals;
                result.inner_stream_status =
                    vbr_capture_stream_status::ring_unavailable;
                result.status = vbr_explicit_capture_status::ring_unavailable;
                return result;
            }
            ++result.ring_operation_acquires;
        }

        // Payload runway is proven before the first companion D2H byte. A
        // locally stale/failed companion removes only its manifest; rebuild
        // the physical union and re-cap it before attention transfer.
        result.phase = vbr_explicit_capture_phase::companion_capture;
        std::vector<std::vector<vbr_capture_sealed_companion>>
            sealed_companions(request.manifests.size());
        bool rebuild_projection = false;
        for (size_t manifest_index = 0;
             manifest_index < request.manifests.size(); ++manifest_index) {
            auto & sealed = sealed_companions[manifest_index];
            if (!manifest_dependency_available[manifest_index]) {
                continue;
            }
            sealed.reserve(recurrent_plans[manifest_index].size());
            for (size_t companion_index = 0;
                 companion_index < recurrent_plans[manifest_index].size();
                 ++companion_index) {
                std::unique_ptr<artifact_segment_chain> chain;
                std::array<uint8_t, 32> companion_digest;
                vbr_explicit_capture_status companion_status =
                    vbr_explicit_capture_status::
                        required_companion_unavailable;
                vbr_capture_sealed_companion capability;
                uint64_t companion_d2h_bytes = 0;
                uint64_t companion_d2h_reads = 0;
                const bool companion_captured = recurrent_companion_capture(
                        recurrent_plans[manifest_index][companion_index],
                        request.manifests[manifest_index].sequence,
                        chain, companion_digest, companion_status,
                        request.continue_context,
                        request.continue_transfer,
                        &companion_d2h_bytes,
                        &companion_d2h_reads);
                if (companion_d2h_bytes > UINT64_MAX -
                        result.companion_d2h_bytes ||
                    companion_d2h_reads > UINT64_MAX -
                        result.companion_d2h_reads) {
                    result.status =
                        vbr_explicit_capture_status::size_overflow;
                    return result;
                }
                result.companion_d2h_bytes += companion_d2h_bytes;
                result.companion_d2h_reads += companion_d2h_reads;
                if (!companion_captured ||
                    !vbr_capture_seal_companion(
                        uint32_t(companion_index), std::move(chain),
                        capability) ||
                    capability.streaming_digest() != companion_digest ||
                    !recurrent_companion_current(
                        recurrent_plans[manifest_index][companion_index],
                        request.manifests[manifest_index].sequence)) {
                    if (companion_status ==
                            vbr_explicit_capture_status::cancelled) {
                        result.status = companion_status;
                        return result;
                    }
                    manifest_dependency_available[manifest_index] = false;
                    sealed.clear();
                    rebuild_projection = true;
                    break;
                }
                sealed.push_back(std::move(capability));
            }
        }
        if (rebuild_projection) {
            auto revised_manifests = projection->manifests;
            apply_dependency_availability(revised_manifests);
            projection = {};
            ++result.projection_calls;
            if (!vbr_artifact_project_capture_union(
                    { result.source_namespace,
                      std::move(revised_manifests) },
                    projection_limits, projection)) {
                result.status =
                    vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
            result.union_cells = projection->union_cell_count;
            if (!preflight_projection_bytes()) {
                return result;
            }
            if (!build_pretransfer_quote(current_quote)) {
                return result;
            }
            if (request.pretransfer_shrink) {
                result.phase =
                    vbr_explicit_capture_phase::reservation_preparation;
                if (!request.pretransfer_shrink(
                        request.pretransfer_context, current_quote)) {
                    result.status =
                        vbr_explicit_capture_status::admission_refused;
                    return result;
                }
            }
        }

        result.phase = vbr_explicit_capture_phase::pre_transfer_stability;
        for (const auto & child : children) {
            if (!vbr_live_capture_adapter::stable(child)) {
                result.status = vbr_explicit_capture_status::source_changed;
                return result;
            }
        }
        if (!vbr_operation_registry_quiescent_for(
                instances.data(), instances.size())) {
            result.status = vbr_explicit_capture_status::registry_busy;
            return result;
        }

        result.phase = vbr_explicit_capture_phase::unit_transfer;
        std::vector<vbr_capture_projected_unit> projected_units;
        for (const auto & child : children) {
            const bool projected_child = std::any_of(
                projection->streams.begin(), projection->streams.end(),
                [&](const auto & stream) {
                    return stream.child_id == child.child_id &&
                        stream.stream_index == 0;
                });
            if (!projected_child) {
                continue;
            }
            for (const auto & plan : child.units) {
                vbr_capture_projected_unit captured;
                vbr_capture_stream_stats attempted;
                ++result.unit_transfer_calls;
                const auto transferred =
                    vbr_live_capture_adapter::transfer_projected_unit(
                        child, plan, projection, result.source_namespace,
                        *request.ring, captured,
                        request.continue_context,
                        request.continue_transfer, &attempted,
                        ring_operation ? &ring_operation : nullptr);
                if (transferred != vbr_capture_stream_status::ok) {
                    if (attempted.bytes >
                            UINT64_MAX - result.transfer.bytes ||
                        attempted.chunks >
                            UINT64_MAX - result.transfer.chunks ||
                        attempted.submitted_bytes >
                            UINT64_MAX - result.transfer.submitted_bytes ||
                        attempted.submitted_chunks >
                            UINT64_MAX - result.transfer.submitted_chunks ||
                        attempted.backpressure_waits >
                            UINT64_MAX - result.transfer.backpressure_waits ||
                        attempted.event_completions >
                            UINT64_MAX - result.transfer.event_completions ||
                        attempted.synchronous_fallbacks >
                            UINT64_MAX - result.transfer.synchronous_fallbacks) {
                        result.status =
                            vbr_explicit_capture_status::internal_error;
                        return result;
                    }
                    result.transfer.bytes += attempted.bytes;
                    result.transfer.chunks += attempted.chunks;
                    result.transfer.submitted_bytes +=
                        attempted.submitted_bytes;
                    result.transfer.submitted_chunks +=
                        attempted.submitted_chunks;
                    result.transfer.backpressure_waits +=
                        attempted.backpressure_waits;
                    result.transfer.event_completions +=
                        attempted.event_completions;
                    result.transfer.synchronous_fallbacks +=
                        attempted.synchronous_fallbacks;
                    result.transfer.max_segment_size = std::max(
                        result.transfer.max_segment_size,
                        attempted.max_segment_size);
                    if (result.inner_stream_status ==
                            vbr_capture_stream_status::_count) {
                        result.inner_stream_status = transferred;
                    }
                    if (transferred ==
                            vbr_capture_stream_status::cancelled) {
                        result.status =
                            vbr_explicit_capture_status::cancelled;
                        return result;
                    }
                    if (transferred ==
                            vbr_capture_stream_status::snapshot_unavailable ||
                        transferred ==
                            vbr_capture_stream_status::snapshot_changed ||
                        transferred ==
                            vbr_capture_stream_status::transfer_failed ||
                        transferred == vbr_capture_stream_status::short_read ||
                        transferred ==
                            vbr_capture_stream_status::hash_mismatch) {
                        continue;
                    }
                    result.status = stream_status(transferred);
                    return result;
                }
                const auto & stats = captured.transfer();
                if (stats.bytes > UINT64_MAX - result.transfer.bytes ||
                    stats.chunks > UINT64_MAX - result.transfer.chunks ||
                    stats.submitted_bytes >
                        UINT64_MAX - result.transfer.submitted_bytes ||
                    stats.submitted_chunks >
                        UINT64_MAX - result.transfer.submitted_chunks ||
                    stats.backpressure_waits >
                        UINT64_MAX - result.transfer.backpressure_waits ||
                    stats.event_completions >
                        UINT64_MAX - result.transfer.event_completions ||
                    stats.synchronous_fallbacks >
                        UINT64_MAX - result.transfer.synchronous_fallbacks) {
                    result.status = vbr_explicit_capture_status::size_overflow;
                    return result;
                }
                result.transfer.bytes += stats.bytes;
                result.transfer.chunks += stats.chunks;
                result.transfer.submitted_bytes += stats.submitted_bytes;
                result.transfer.submitted_chunks += stats.submitted_chunks;
                result.transfer.backpressure_waits +=
                    stats.backpressure_waits;
                result.transfer.event_completions +=
                    stats.event_completions;
                result.transfer.synchronous_fallbacks +=
                    stats.synchronous_fallbacks;
                result.transfer.max_segment_size = std::max(
                    result.transfer.max_segment_size,
                    stats.max_segment_size);
                ++result.transferred_units;
                projected_units.push_back(std::move(captured));
            }
        }
        // Transport ownership ends with the final D2H chunk. Metadata
        // rechecks, assembly, and catalog publication never hold the ring.
        ring_operation = {};

        // Successful preflight companions remain immutable while attention
        // bytes transfer. A changed recurrent frontier is a violated idle
        // snapshot, so fail the whole batch instead of publishing stale rows.
        for (size_t manifest_index = 0;
             manifest_index < request.manifests.size(); ++manifest_index) {
            if (!manifest_dependency_available[manifest_index]) {
                continue;
            }
            for (const auto & companion : recurrent_plans[manifest_index]) {
                if (!recurrent_companion_current(
                        companion,
                        request.manifests[manifest_index].sequence)) {
                    result.status =
                        vbr_explicit_capture_status::source_changed;
                    return result;
                }
            }
        }

        result.phase = vbr_explicit_capture_phase::post_transfer_stability;
        for (const auto instance : instances) {
            if (vbr_recovery_pending_for(instance)) {
                result.status =
                    vbr_explicit_capture_status::recovery_pending;
                return result;
            }
        }
        if (!vbr_operation_registry_quiescent_for(
                instances.data(), instances.size())) {
            result.status = vbr_explicit_capture_status::registry_busy;
            return result;
        }
        vbr_live_capture_adapter::projected_target_recheck_context
            recheck_context { &expectations };
        vbr_capture_controller_target_provider provider;
        provider.context = &recheck_context;
        provider.recheck =
            vbr_live_capture_adapter::projected_targets_recheck;
        vbr_capture_manifest_assembly_limits assembly_limits;
        assembly_limits.max_manifests =
            VBR_PROJECTED_CAPTURE_MAX_MANIFESTS;
        if (!vbr_capture_assemble_manifests(
                projection, std::move(targets),
                std::move(projected_units), provider,
                assembly_limits, result.assembly)) {
            result.status = vbr_explicit_capture_status::internal_error;
            return result;
        }

        result.publications.reserve(result.assembly.manifests().size());
        for (const auto & manifest : result.assembly.manifests()) {
            vbr_projected_manifest_publication publication;
            publication.manifest_id = manifest.manifest_id;
            publication.topologies = request.topologies;
            const auto requested = std::find(
                manifest_ids.begin(), manifest_ids.end(),
                manifest.manifest_id);
            if (requested == manifest_ids.end()) {
                result.status = vbr_explicit_capture_status::internal_error;
                result.assembly = {};
                result.publications.clear();
                return result;
            }
            const size_t manifest_index = size_t(
                requested - manifest_ids.begin());
            if (manifest.state == vbr_capture_manifest_state::ready) {
                const auto durable = std::lower_bound(
                    current_quote.durable.begin(),
                    current_quote.durable.end(), manifest.manifest_id,
                    [](const auto & lhs, uint64_t id) {
                        return lhs.manifest_id < id;
                    });
                if (durable == current_quote.durable.end() ||
                    durable->manifest_id != manifest.manifest_id ||
                    manifest.unit_count != current_quote.projected_units) {
                    result.status =
                        vbr_explicit_capture_status::internal_error;
                    result.assembly = {};
                    result.publications.clear();
                    return result;
                }
                publication.accounting =
                    std::move(durable->accounting);
                publication.companions = std::move(
                    sealed_companions[manifest_index]);
            }
            result.publications.push_back(std::move(publication));
        }
        result.status = vbr_explicit_capture_status::ok;
        result.phase = vbr_explicit_capture_phase::complete;
        return result;
    } catch (...) {
        result.status = vbr_explicit_capture_status::internal_error;
        result.assembly = {};
        result.publications.clear();
        return result;
    }
}

uint64_t vbr_explicit_import_policy_epoch(
        llama_memory_i & memory) noexcept {
    try {
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(&memory, tree)) {
            return 0;
        }
        return import_policy_epoch(tree);
    } catch (...) {
        return 0;
    }
}

bool vbr_explicit_import_target_snapshot(
        llama_memory_i & memory,
        llama_seq_id destination,
        const vbr_artifact_package_view & package,
        const std::vector<llama_vbr_artifact_domain_binding> & bindings,
        bool previously_observed,
        uint64_t accounting_serial,
        vbr_target_validation_snapshot & output,
        vbr_downward_policy_projection * downward_projection,
        bool * downward_required) noexcept {
    output = {};
    if (downward_projection) {
        *downward_projection = {};
    }
    if (downward_required) {
        *downward_required = false;
    }
    try {
        std::vector<llama_memory_tree_child> tree;
        if (destination < 0 || !package ||
            !llama_memory_tree_collect(&memory, tree)) {
            return false;
        }
        const uint64_t policy_epoch = import_policy_epoch(tree);
        if (policy_epoch == 0) {
            return false;
        }
        output.memory_instance_cookie =
            uint64_t(reinterpret_cast<uintptr_t>(&memory));
        output.accounting_serial = accounting_serial;
        output.policy_epoch = policy_epoch;
        // Idleness is a SCHEDULER-authority fact; the library cannot vouch for
        // it. The route owner asserts it on the snapshot after this returns.
        output.scheduler_idle = false;
        output.destination_sequence_absent =
            memory.seq_pos_min(destination) < 0 &&
            memory.seq_pos_max(destination) < 0;
        output.tree_shape_digest = import_tree_digest(memory, tree);
        output.target_state_serial = 1;
        size_t n_attention = 0;
        size_t n_recurrent = 0;
        bool type_mismatch = false;
        for (const auto & child : tree) {
            if (child.recurrent) {
                if (n_recurrent != 0 || !recurrent_target_empty(child)) {
                    output = {};
                    return false;
                }
                output.companions.push_back({
                    vbr_artifact_companion_kind::recurrent, 1,
                    tagged_digest(
                        VBR_RECURRENT_CODEC_DOMAIN, 1),
                    true, child.recurrent,
                });
                ++n_recurrent;
                continue;
            }
            if (!child.attention) {
                output = {};
                return false;
            }
            vbr_target_child_snapshot snapshot;
            if (!vbr_live_capture_adapter::fill_import_child(
                    child, package, bindings, previously_observed,
                    policy_epoch, snapshot, type_mismatch)) {
                output = {};
                return false;
            }
            if (output.target_state_serial >
                    UINT64_MAX - snapshot.state_serial) {
                output = {};
                return false;
            }
            output.target_state_serial += snapshot.state_serial;
            output.children.push_back(std::move(snapshot));
            ++n_attention;
        }
        if (output.children.empty() ||
            n_attention != output.children.size() ||
            output.children.size() !=
                package.manifest().generation.controllers.size() ||
            n_recurrent != size_t(std::count_if(
                package.companions().begin(), package.companions().end(),
                [](const vbr_artifact_companion_view & companion) {
                    return companion.descriptor.kind ==
                        vbr_artifact_companion_kind::recurrent;
                }))) {
            output = {};
            return false;
        }
        // A projection pointer grants permission to classify a mismatch; it
        // must not make an all-match/native import run the downward binder.
        // The binder rewrites projected policy digests/cursors, so this branch
        // is the semantic separation between native and downward snapshots.
        if (type_mismatch) {
            if (downward_projection == nullptr ||
                !vbr_live_capture_adapter::bind_import_downward(
                    tree, package, output, *downward_projection)) {
                output = {};
                if (downward_projection) {
                    *downward_projection = {};
                }
                return false;
            }
            if (downward_required) {
                *downward_required = true;
            }
        }
        return true;
    } catch (...) {
        output = {};
        if (downward_projection) {
            *downward_projection = {};
        }
        return false;
    }
}

bool vbr_explicit_import_target_recheck(
        llama_memory_i & memory,
        llama_seq_id destination,
        const vbr_target_empty_fingerprint & expected) noexcept {
    try {
        std::vector<llama_memory_tree_child> tree;
        if (destination < 0 ||
            expected.memory_instance_cookie !=
                uint64_t(reinterpret_cast<uintptr_t>(&memory)) ||
            !llama_memory_tree_collect(&memory, tree) ||
            expected.tree_shape_digest != import_tree_digest(memory, tree) ||
            expected.policy_epoch != import_policy_epoch(tree) ||
            memory.seq_pos_min(destination) >= 0 ||
            memory.seq_pos_max(destination) >= 0) {
            return false;
        }
        size_t n_attention = 0;
        size_t n_recurrent = 0;
        for (const auto & child : tree) {
            if (child.recurrent) {
                if (n_recurrent != 0 || !recurrent_target_empty(child)) {
                    return false;
                }
                ++n_recurrent;
                continue;
            }
            if (!child.attention) {
                return false;
            }
            ++n_attention;
            const auto item = std::find_if(
                expected.children.begin(), expected.children.end(),
                [&](const vbr_child_empty_fingerprint & value) {
                    return value.child_id == child.child_id;
                });
            if (item == expected.children.end() ||
                !vbr_live_capture_adapter::recheck_import_child(
                    *child.attention, *item)) {
                return false;
            }
        }
        return n_attention == expected.children.size();
    } catch (...) {
        return false;
    }
}

bool vbr_explicit_import_reserve_downward(
        llama_memory_i & memory,
        const std::vector<vbr_validated_child_plan> & plans,
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget,
        vbr_downward_stage_reservation & output) noexcept {
    output = {};
    try {
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(&memory, tree)) {
            return false;
        }
        output.status = vbr_downward_reserve_status::reserved;
        bool any = false;
        for (const auto & child : tree) {
            if (!child.attention) {
                continue;
            }
            std::vector<const vbr_validated_child_plan *> selected;
            for (const auto & plan : plans) {
                if (plan.child_id == child.child_id && plan.downward) {
                    selected.push_back(&plan);
                }
            }
            if (selected.empty()) {
                continue;
            }
            any = true;
            vbr_downward_stage_reservation one;
            if (!vbr_live_capture_adapter::reserve_import_downward(
                    *child.attention, selected, ledger, budget, one)) {
                output = one;
                return false;
            }
            if (one.status != vbr_downward_reserve_status::reserved &&
                one.status !=
                    vbr_downward_reserve_status::reserved_stashless) {
                output = one;
                return true;
            }
            output.stashless_units.insert(
                output.stashless_units.end(),
                one.stashless_units.begin(),
                one.stashless_units.end());
            if (!one.stashless_units.empty()) {
                output.status =
                    vbr_downward_reserve_status::reserved_stashless;
            }
        }
        if (!any) {
            output.status =
                vbr_downward_reserve_status::projection_unavailable;
            return false;
        }
        return true;
    } catch (...) {
        output = {};
        output.status = vbr_downward_reserve_status::internal_error;
        return false;
    }
}

vbr_explicit_capture_result vbr_capture_explicit_manifest(
        llama_memory_i & memory,
        const vbr_explicit_capture_request & request,
        vbr_unit_version_sink & sink,
        const vbr_explicit_capture_accounting & accounting) noexcept {
    vbr_explicit_capture_result result;
    if (!request.idle_decode_thread) {
        result.status = vbr_explicit_capture_status::slot_not_idle;
        return result;
    }
    if (request.sequence < 0 || request.frontier.next_position < 0 ||
        request.ring == nullptr || accounting.budget == nullptr ||
        request.topologies.empty() || request.pool_bindings.empty() ||
        request.representation_identity == nullptr ||
        request.identity.token_count < 0 ||
        request.token_block.size() !=
            size_t(request.identity.token_count) ||
        request.identity.execution_identity.empty() ||
        request.identity.adapter_config_identity.empty() ||
        request.identity.media_content_identity.empty()) {
        result.status = vbr_explicit_capture_status::identity_unavailable;
        return result;
    }

    try {
        result.phase = vbr_explicit_capture_phase::memory_tree;
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(&memory, tree)) {
            result.status = vbr_explicit_capture_status::unsupported_layout;
            return result;
        }

        std::vector<vbr_live_capture_adapter::child> children;
        std::vector<llama_memory_recurrent *> recurrent;
        for (const auto & node : tree) {
            if (node.attention != nullptr) {
                if (!node.attention->vbr_operation_armed()) {
                    result.status = vbr_explicit_capture_status::not_armed;
                    return result;
                }
                vbr_live_capture_adapter::child child;
                child.child_id = node.child_id;
                child.dependency_mode = node.dependency_mode;
                child.cache = node.attention;
                children.push_back(std::move(child));
            }
            if (node.recurrent != nullptr) {
                recurrent.push_back(node.recurrent);
            }
        }
        if (children.empty()) {
            result.status = vbr_explicit_capture_status::not_armed;
            return result;
        }

        result.phase = vbr_explicit_capture_phase::settlement;
        // Settlement is deliberately before both quiescence proofs. It flushes
        // only already-deferred housekeeping and dirty stash metadata.
        for (auto & child : children) {
            if (!vbr_live_capture_adapter::settle(*child.cache)) {
                result.status = vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
        }

        result.phase =
            vbr_explicit_capture_phase::pre_capture_quiescence;
        std::vector<vbr_controller_instance_id> instances;
        result.phase =
            vbr_explicit_capture_phase::metadata_and_manifest;
        for (auto & child : children) {
            const auto instance = child.cache->vbr_instance_id();
            if (!vbr_controller_instance_id_is_set(instance) ||
                std::any_of(instances.begin(), instances.end(),
                    [&](const auto & current) {
                        return current == instance;
                    })) {
                result.status = vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
            instances.push_back(instance);
            if (vbr_recovery_pending_for(instance)) {
                result.status = vbr_explicit_capture_status::recovery_pending;
                return result;
            }
        }
        if (!vbr_operation_registry_quiescent_for(
                instances.data(), instances.size())) {
            result.status = vbr_explicit_capture_status::registry_busy;
            return result;
        }

        for (auto & child : children) {
            if (!vbr_live_capture_adapter::capture_metadata(
                    *child.cache, child.child_id, child.dependency_mode,
                    request.sequence, request.frontier.next_position,
                    request.pool_bindings, child,
                    result.generation_failure,
                    result.size_failure)) {
                result.status = vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
        }

        std::vector<vbr_identity_policy_digest_row> identity_policy;
        identity_policy.reserve(children.size());
        for (const auto & child : children) {
            identity_policy.push_back({
                child.child_id,
                child.dependency_mode,
                child.stability.lineage_uuid,
            });
        }
        const auto identity_policy_order_digest =
            vbr_identity_policy_digest(
                request.frontier, identity_policy);
        if (digest_nonzero(request.identity_policy_order_digest) &&
            request.identity_policy_order_digest !=
                identity_policy_order_digest) {
            result.status =
                vbr_explicit_capture_status::identity_unavailable;
            return result;
        }

        // Recurrent state uses the existing exact state codec. Accelerator
        // companions use equally typed injected existing codecs.
        vbr_artifact_package package;
        package.topologies = request.topologies;
        package.manifest.identity = request.identity;
        package.manifest.token_block.tokens = request.token_block;
        package.manifest.identity_policy_order_digest =
            identity_policy_order_digest;
        package.manifest.generation.version = 1;
        package.manifest.generation.status =
            vbr_checkpoint_generation_status::complete;
        package.manifest.generation.identity_policy_order_digest =
            identity_policy_order_digest;

        uint32_t global_unit = 0;
        std::vector<vbr_live_capture_adapter::representation_cache_entry>
            representation_cache;
        for (auto & child : children) {
            package.manifest.generation.controllers.push_back(
                child.generation);
            package.manifest.stream_placements.insert(
                package.manifest.stream_placements.end(),
                child.placements.begin(), child.placements.end());
            vbr_artifact_controller_policy policy;
            std::vector<vbr_artifact_unit_descriptor> descriptors;
            if (!vbr_live_capture_adapter::capture_schema(
                    child, request.representation_context,
                    request.representation_identity, true,
                    representation_cache,
                    policy, descriptors, result.status)) {
                return result;
            }
            package.manifest.controller_policy.push_back(std::move(policy));

            for (size_t plan_index = 0;
                 plan_index < child.units.size(); ++plan_index) {
                auto & plan = child.units[plan_index];
                plan.capture_index = global_unit;
                vbr_artifact_unit_blob blob;
                blob.descriptor = std::move(descriptors[plan_index]);
                const bool has_stash = blob.descriptor.clean_stash_state ==
                    vbr_artifact_clean_stash_state::present;
                const uint32_t stash_rows = uint32_t(
                    blob.descriptor.clean_stash.valid_rows);
                const uint64_t total_columns =
                    blob.descriptor.dimensions[1];
                for (const auto & shard : plan.shards) {
                    add_accounting(
                        package.manifest.accounting,
                        vbr_artifact_accounting_role::unit_payload,
                        portable_domain(
                            shard.topology_index,
                            shard.device_ordinal),
                        shard.payload_bytes);
                    if (has_stash) {
                        add_accounting(
                            package.manifest.accounting,
                            vbr_artifact_accounting_role::
                                clean_stash_payload,
                            portable_domain(
                                shard.topology_index,
                                shard.device_ordinal),
                            shard.stash_bytes);
                    }
                }
                package.unit_blobs.push_back(std::move(blob));

                vbr_artifact_unit_reference reference;
                reference.lineage_uuid = child.stability.lineage_uuid;
                reference.logical_unit_id = plan.logical_unit;
                reference.repr_gen = plan.generation.repr_gen;
                reference.authorized_stream_refs = { 0 };
                if (has_stash) {
                    if (child.generation.streams.empty()) {
                        result.status =
                            vbr_explicit_capture_status::stash_inconsistent;
                        return result;
                    }
                    const auto & stream =
                        child.generation.streams.front();
                    reference.has_stash_reference = true;
                    reference.stash_reference.valid_rows = stash_rows;
                    reference.stash_reference.domain =
                        vbr_repr_domain::tapped;
                    reference.stash_reference.row_count = stash_rows;
                    reference.stash_reference.column_count =
                        total_columns;
                    reference.stash_reference.row_bytes =
                        total_columns*sizeof(uint16_t);
                    reference.stash_reference.captured_sink_count =
                        stream.captured_dependency_count;
                    reference.stash_reference.covered_sink_pages =
                        stream.pages;
                }
                package.manifest.unit_references.push_back(
                    std::move(reference));
                ++global_unit;
            }
        }

        struct pending_companion {
            recurrent_companion_plan recurrent;
            const vbr_explicit_companion_provider * provider = nullptr;
            uint64_t bytes = 0;
        };
        std::vector<pending_companion> pending_companions;
        for (auto * memory_recurrent : recurrent) {
            recurrent_companion_plan recurrent_plan;
            if (!recurrent_companion_prepare(
                    memory_recurrent, request.sequence,
                    request.frontier.next_position,
                    recurrent_plan, result.status)) {
                return result;
            }
            pending_companions.push_back({
                recurrent_plan, nullptr,
                recurrent_plan.descriptor.payload_bytes,
            });
            package.companions.push_back(recurrent_plan.descriptor);
        }
        for (const auto & provider : request.companions) {
            if (provider.size == nullptr ||
                provider.capture == nullptr ||
                provider.format_version == 0 ||
                !digest_nonzero(provider.build_identity_digest)) {
                if (provider.required) {
                    result.status =
                        vbr_explicit_capture_status::
                            required_companion_unavailable;
                    return result;
                }
                continue;
            }
            uint64_t companion_size = 0;
            if (!provider.size(
                    provider.context, request.sequence,
                    companion_size) ||
                companion_size == 0 ||
                companion_size >
                    std::numeric_limits<size_t>::max()) {
                if (provider.required) {
                    result.status =
                        vbr_explicit_capture_status::
                            required_companion_unavailable;
                    return result;
                }
                continue;
            }
            pending_companions.push_back({
                {}, &provider, companion_size,
            });
            vbr_artifact_companion_payload companion;
            companion.kind = provider.kind;
            companion.format_version = provider.format_version;
            companion.build_identity_digest =
                provider.build_identity_digest;
            companion.domain = provider.domain;
            companion.payload_bytes = companion_size;
            package.companions.push_back(companion);
        }
        package.manifest.companions = package.companions;
        for (const auto & companion : package.companions) {
            add_accounting(
                package.manifest.accounting,
                companion.kind ==
                    vbr_artifact_companion_kind::recurrent
                    ? vbr_artifact_accounting_role::recurrent_payload
                    : vbr_artifact_accounting_role::
                        typed_accelerator_payload,
                companion.domain, companion.payload_bytes);
            result.companion_bytes += companion.payload_bytes;
        }
        const auto metadata_domain = vbr_artifact_portable_domain {
            llama_cache_acct_residency::pageable_host,
            llama_cache_acct_domain_kind::not_applicable,
            UINT32_MAX, UINT16_MAX,
        };
        add_accounting(
            package.manifest.accounting,
            vbr_artifact_accounting_role::descriptor_metadata,
            metadata_domain,
            std::max<uint64_t>(1, package.unit_blobs.size()*256));
        add_accounting(
            package.manifest.accounting,
            vbr_artifact_accounting_role::reference_metadata,
            metadata_domain,
            std::max<uint64_t>(1,
                package.manifest.unit_references.size()*128));
        package.manifest.consistency.kind =
            vbr_artifact_consistency_kind::capture_exact;

        result.phase =
            vbr_explicit_capture_phase::pre_transfer_stability;
        // Exact equality immediately before the first data byte.
        for (const auto & child : children) {
            if (!vbr_live_capture_adapter::stable(child)) {
                result.status =
                    vbr_explicit_capture_status::source_changed;
                return result;
            }
        }
        if (!vbr_operation_registry_quiescent_for(
                instances.data(), instances.size())) {
            result.status = vbr_explicit_capture_status::registry_busy;
            return result;
        }

        result.phase =
            vbr_explicit_capture_phase::accounting_configuration;
        if (accounting.prepare != nullptr &&
            !accounting.prepare(accounting.context, package)) {
            result.status = vbr_explicit_capture_status::accounting_failed;
            return result;
        }
        result.phase =
            vbr_explicit_capture_phase::reservation_preparation;
        vbr_capture_stream_status begin_status;
        auto build = sink.begin_capture(
            package, *accounting.budget, accounting.fault,
            begin_status, &result.begin_diagnostics);
        if (!build) {
            result.inner_stream_status = begin_status;
            result.status = stream_status(begin_status);
            return result;
        }

        result.phase =
            vbr_explicit_capture_phase::companion_capture;
        // Durable + transfer-staging claims now exist. Only at this point may
        // companion codecs allocate their pageable byte images.
        for (size_t i = 0; i < pending_companions.size(); ++i) {
            std::vector<uint8_t> bytes;
            const auto & pending = pending_companions[i];
            std::unique_ptr<artifact_segment_chain> recurrent_chain;
            std::array<uint8_t, 32> recurrent_digest;
            if (pending.recurrent.source != nullptr) {
                if (!recurrent_companion_capture(
                        pending.recurrent, request.sequence,
                        recurrent_chain, recurrent_digest,
                        result.status)) {
                    return result;
                }
            } else if (pending.provider == nullptr ||
                       !pending.provider->capture(
                           pending.provider->context,
                           request.sequence, bytes)) {
                result.status =
                    vbr_explicit_capture_status::
                        required_companion_unavailable;
                return result;
            }
            // Companion size→data coherence relies on the required idle-slot,
            // no-decode route invariant. F3.3 must enforce that invariant;
            // size equality is intentionally the F3.2 guard, not a second
            // content-hash pass over the existing companion codecs.
            if (pending.recurrent.source == nullptr &&
                bytes.size() != pending.bytes) {
                result.status =
                    vbr_explicit_capture_status::source_changed;
                return result;
            }
            std::shared_ptr<const artifact_segment_chain> chain;
            std::array<uint8_t, 32> digest;
            if (recurrent_chain) {
                chain = std::shared_ptr<const artifact_segment_chain>(
                    std::move(recurrent_chain));
                digest = recurrent_digest;
            } else {
                auto provider_chain =
                    std::make_shared<artifact_segment_chain>();
                static constexpr size_t CHUNK = 1024*1024;
                for (size_t offset = 0; offset < bytes.size();) {
                    const size_t size = std::min(
                        CHUNK, bytes.size() - offset);
                    if (!provider_chain->append(
                            bytes.data() + offset, size)) {
                        result.status =
                            vbr_explicit_capture_status::accounting_failed;
                        return result;
                    }
                    offset += size;
                }
                digest = vbr_capture_stream_digest(*provider_chain);
                chain = std::move(provider_chain);
            }
            vbr_verified_companion verified;
            verified.companion_index = uint32_t(i);
            verified.bytes = chain;
            verified.streaming_digest = digest;
            const auto accepted =
                build->accept_verified_companion(verified);
            if (accepted != vbr_capture_stream_status::ok) {
                result.inner_stream_status = accepted;
                result.status = stream_status(accepted);
                return result;
            }
        }

        result.phase = vbr_explicit_capture_phase::unit_transfer;
        uint32_t unit_index = 0;
        for (const auto & child : children) {
            for (const auto & plan : child.units) {
                vbr_capture_stream_status unit_status;
                auto unit = build->begin_unit(unit_index, unit_status);
                if (!unit) {
                    result.inner_stream_status = unit_status;
                    result.status = stream_status(unit_status);
                    return result;
                }
                vbr_capture_stream_stats stats;
                if (!vbr_live_capture_adapter::stream(
                        child, plan, *unit, *request.ring, stats)) {
                    result.status =
                        vbr_explicit_capture_status::transfer_failed;
                    return result;
                }
                result.chunks += stats.chunks;
                result.backpressure_waits += stats.backpressure_waits;
                result.event_completions += stats.event_completions;
                result.synchronous_fallbacks +=
                    stats.synchronous_fallbacks;
                const auto sealed = unit->seal_unit();
                if (sealed != vbr_capture_stream_status::ok) {
                    result.inner_stream_status = sealed;
                    result.status =
                        vbr_explicit_capture_status::hash_mismatch;
                    return result;
                }
                for (const auto & shard : plan.shards) {
                    if (shard.payload_bytes >
                            UINT64_MAX - result.payload_bytes ||
                        shard.stash_bytes >
                            UINT64_MAX - result.stash_bytes) {
                        result.status =
                            vbr_explicit_capture_status::size_overflow;
                        return result;
                    }
                    result.payload_bytes += shard.payload_bytes;
                    result.stash_bytes += shard.stash_bytes;
                }
                ++unit_index;
            }
        }

        result.phase =
            vbr_explicit_capture_phase::post_transfer_stability;
        // Both levels of stability and quiescence are re-read after all D2H
        // completions, before the catalog's final reference publication.
        for (const auto & child : children) {
            const auto instance = child.cache->vbr_instance_id();
            if (!vbr_live_capture_adapter::stable(child) ||
                vbr_recovery_pending_for(instance)) {
                result.status =
                    vbr_explicit_capture_status::source_changed;
                return result;
            }
        }
        if (!vbr_operation_registry_quiescent_for(
                instances.data(), instances.size())) {
            result.status = vbr_explicit_capture_status::registry_busy;
            return result;
        }

        result.phase = vbr_explicit_capture_phase::publication;
        result.sink = build->publish_reference();
        result.inner_stream_status = result.sink.status;
        result.status = stream_status(result.sink.status);
        result.controllers = children.size();
        result.units = unit_index;
        result.companions = package.companions.size();
        if (result.status == vbr_explicit_capture_status::ok) {
            result.phase = vbr_explicit_capture_phase::complete;
        }
        return result;
    } catch (...) {
        result.status = vbr_explicit_capture_status::internal_error;
        return result;
    }
}

const char * vbr_explicit_capture_phase_name(
        vbr_explicit_capture_phase phase) noexcept {
    switch (phase) {
        case vbr_explicit_capture_phase::validation: return "validation";
        case vbr_explicit_capture_phase::memory_tree: return "memory_tree";
        case vbr_explicit_capture_phase::settlement: return "settlement";
        case vbr_explicit_capture_phase::pre_capture_quiescence: return "pre_capture_quiescence";
        case vbr_explicit_capture_phase::metadata_and_manifest: return "metadata_and_manifest";
        case vbr_explicit_capture_phase::pre_transfer_stability: return "pre_transfer_stability";
        case vbr_explicit_capture_phase::accounting_configuration: return "accounting_configuration";
        case vbr_explicit_capture_phase::reservation_preparation: return "reservation_preparation";
        case vbr_explicit_capture_phase::companion_capture: return "companion_capture";
        case vbr_explicit_capture_phase::unit_transfer: return "unit_transfer";
        case vbr_explicit_capture_phase::post_transfer_stability: return "post_transfer_stability";
        case vbr_explicit_capture_phase::publication: return "publication";
        case vbr_explicit_capture_phase::complete: return "complete";
        case vbr_explicit_capture_phase::_count: return "_count";
    }
    return "invalid";
}

const char * vbr_explicit_generation_failure_name(
        vbr_explicit_generation_failure failure) noexcept {
    switch (failure) {
        case vbr_explicit_generation_failure::none: return "none";
        case vbr_explicit_generation_failure::size_pass: return "size_pass";
        case vbr_explicit_generation_failure::tracker_missing: return "tracker_missing";
        case vbr_explicit_generation_failure::tracker_unstable: return "tracker_unstable";
        case vbr_explicit_generation_failure::tracker_shadow_unavailable: return "tracker_shadow_unavailable";
        case vbr_explicit_generation_failure::invalid_sequence_or_frontier: return "invalid_sequence_or_frontier";
        case vbr_explicit_generation_failure::invalid_stream: return "invalid_stream";
        case vbr_explicit_generation_failure::ownership_index_missing: return "ownership_index_missing";
        case vbr_explicit_generation_failure::ownership_view_missing: return "ownership_view_missing";
        case vbr_explicit_generation_failure::ownership_view_unavailable: return "ownership_view_unavailable";
        case vbr_explicit_generation_failure::ownership_rank_failed: return "ownership_rank_failed";
        case vbr_explicit_generation_failure::ownership_enumeration_failed: return "ownership_enumeration_failed";
        case vbr_explicit_generation_failure::ownership_cardinality_mismatch: return "ownership_cardinality_mismatch";
        case vbr_explicit_generation_failure::stream_capture_failed: return "stream_capture_failed";
        case vbr_explicit_generation_failure::controller_capture_failed: return "controller_capture_failed";
        case vbr_explicit_generation_failure::stability_reread_failed: return "stability_reread_failed";
        case vbr_explicit_generation_failure::internal_error: return "internal_error";
        case vbr_explicit_generation_failure::_count: return "_count";
    }
    return "_count";
}

const char * vbr_explicit_size_failure_name(
        vbr_explicit_size_failure failure) noexcept {
    switch (failure) {
        case vbr_explicit_size_failure::none: return "none";
        case vbr_explicit_size_failure::not_armed: return "not_armed";
        case vbr_explicit_size_failure::tracker_missing: return "tracker_missing";
        case vbr_explicit_size_failure::tracker_unstable: return "tracker_unstable";
        case vbr_explicit_size_failure::bindings_missing: return "bindings_missing";
        case vbr_explicit_size_failure::stream_layout: return "stream_layout";
        case vbr_explicit_size_failure::policy_snapshot: return "policy_snapshot";
        case vbr_explicit_size_failure::unit_index: return "unit_index";
        case vbr_explicit_size_failure::extents_empty: return "extents_empty";
        case vbr_explicit_size_failure::extent_missing: return "extent_missing";
        case vbr_explicit_size_failure::vmm_missing: return "vmm_missing";
        case vbr_explicit_size_failure::backend_unavailable: return "backend_unavailable";
        case vbr_explicit_size_failure::wm_cells_zero: return "wm_cells_zero";
        case vbr_explicit_size_failure::extent_type_mismatch: return "extent_type_mismatch";
        case vbr_explicit_size_failure::promote_hops_mismatch: return "promote_hops_mismatch";
        case vbr_explicit_size_failure::domain_mismatch: return "domain_mismatch";
        case vbr_explicit_size_failure::shard_disagreement: return "shard_disagreement";
        case vbr_explicit_size_failure::binding_missing: return "binding_missing";
        case vbr_explicit_size_failure::topology_order: return "topology_order";
        case vbr_explicit_size_failure::bounds: return "bounds";
        case vbr_explicit_size_failure::stash_bounds: return "stash_bounds";
        case vbr_explicit_size_failure::stability_reread: return "stability_reread";
        case vbr_explicit_size_failure::internal_error: return "internal_error";
        case vbr_explicit_size_failure::_count: return "_count";
    }
    return "_count";
}

vbr_explicit_size_failure vbr_explicit_capture_validate_extent_generation(
        uint32_t wm_cells,
        int32_t extent_type,
        uint8_t extent_promote_hops,
        const vbr_unit_generation & generation) noexcept {
    if (wm_cells == 0) {
        return vbr_explicit_size_failure::wm_cells_zero;
    }
    if (extent_type != generation.current_type) {
        return vbr_explicit_size_failure::extent_type_mismatch;
    }
    if (extent_promote_hops != generation.promote_hops) {
        return vbr_explicit_size_failure::promote_hops_mismatch;
    }
    const auto expected_domain =
        generation.current_type == GGML_TYPE_F16 ||
        generation.current_type == GGML_TYPE_TURBO8_0
            ? vbr_repr_domain::full
            : vbr_repr_domain::tapped;
    if (generation.domain != expected_domain) {
        return vbr_explicit_size_failure::domain_mismatch;
    }
    return vbr_explicit_size_failure::none;
}

const char * vbr_explicit_capture_status_name(
        vbr_explicit_capture_status status) noexcept {
    switch (status) {
        case vbr_explicit_capture_status::ok: return "ok";
        case vbr_explicit_capture_status::not_armed: return "not_armed";
        case vbr_explicit_capture_status::unsupported_layout: return "unsupported_layout";
        case vbr_explicit_capture_status::slot_not_idle: return "slot_not_idle";
        case vbr_explicit_capture_status::identity_unavailable: return "identity_unavailable";
        case vbr_explicit_capture_status::generation_unavailable: return "generation_unavailable";
        case vbr_explicit_capture_status::registry_busy: return "registry_busy";
        case vbr_explicit_capture_status::recovery_pending: return "recovery_pending";
        case vbr_explicit_capture_status::geometry_mismatch: return "geometry_mismatch";
        case vbr_explicit_capture_status::stash_inconsistent: return "stash_inconsistent";
        case vbr_explicit_capture_status::required_companion_unavailable: return "required_companion_unavailable";
        case vbr_explicit_capture_status::size_overflow: return "size_overflow";
        case vbr_explicit_capture_status::ring_unavailable: return "ring_unavailable";
        case vbr_explicit_capture_status::admission_refused: return "admission_refused";
        case vbr_explicit_capture_status::cancelled: return "cancelled";
        case vbr_explicit_capture_status::transfer_failed: return "transfer_failed";
        case vbr_explicit_capture_status::short_read: return "short_read";
        case vbr_explicit_capture_status::event_failed: return "event_failed";
        case vbr_explicit_capture_status::source_changed: return "source_changed";
        case vbr_explicit_capture_status::hash_mismatch: return "hash_mismatch";
        case vbr_explicit_capture_status::dedup_validation_failed: return "dedup_validation_failed";
        case vbr_explicit_capture_status::accounting_failed: return "accounting_failed";
        case vbr_explicit_capture_status::publication_failed: return "publication_failed";
        case vbr_explicit_capture_status::internal_error: return "internal_error";
        case vbr_explicit_capture_status::_count: return "_count";
    }
    return "_count";
}
