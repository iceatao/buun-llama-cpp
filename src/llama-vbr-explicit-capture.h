#pragma once

#include "llama-vbr-artifact-catalog.h"
#include "llama-vbr-checkpoint-types.h"
#include "llama-vbr-generation.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

class llama_memory_i;
struct vbr_target_validation_snapshot;
struct vbr_target_empty_fingerprint;
struct vbr_downward_policy_projection;
struct vbr_downward_stage_reservation;
struct vbr_validated_child_plan;

enum class vbr_explicit_capture_status : uint8_t {
    ok = 0,
    not_armed,
    unsupported_layout,
    slot_not_idle,
    identity_unavailable,
    generation_unavailable,
    registry_busy,
    recovery_pending,
    // Reserved for F3.3's route-level geometry diagnostics. F3.2 maps all
    // private-hook geometry refusals to generation_unavailable.
    geometry_mismatch,
    stash_inconsistent,
    required_companion_unavailable,
    size_overflow,
    ring_unavailable,
    admission_refused,
    cancelled,
    transfer_failed,
    short_read,
    // Reserved for a backend that can report asynchronous event failure.
    // Today's ggml completion API is void, so F3.2 detects it by digest/length.
    event_failed,
    source_changed,
    hash_mismatch,
    dedup_validation_failed,
    accounting_failed,
    publication_failed,
    internal_error,
    _count,
};

const char * vbr_explicit_capture_status_name(
    vbr_explicit_capture_status status) noexcept;

enum class vbr_explicit_capture_phase : uint8_t {
    validation = 0,
    memory_tree,
    settlement,
    pre_capture_quiescence,
    metadata_and_manifest,
    pre_transfer_stability,
    accounting_configuration,
    reservation_preparation,
    companion_capture,
    unit_transfer,
    post_transfer_stability,
    publication,
    complete,
    _count,
};

const char * vbr_explicit_capture_phase_name(
    vbr_explicit_capture_phase phase) noexcept;

// Closed diagnostic for the metadata/generation half of explicit capture.
// This is process-local observability only; it is not part of an artifact or
// cache-plan wire format.
enum class vbr_explicit_generation_failure : uint8_t {
    none = 0,
    size_pass,
    tracker_missing,
    tracker_unstable,
    tracker_shadow_unavailable,
    invalid_sequence_or_frontier,
    invalid_stream,
    ownership_index_missing,
    ownership_view_missing,
    ownership_view_unavailable,
    ownership_rank_failed,
    ownership_enumeration_failed,
    ownership_cardinality_mismatch,
    stream_capture_failed,
    controller_capture_failed,
    stability_reread_failed,
    internal_error,
    _count,
};

const char * vbr_explicit_generation_failure_name(
    vbr_explicit_generation_failure failure) noexcept;

// Closed diagnostic for the byte-geometry half of explicit capture. This
// remains process-local observability; it is never serialized into the
// artifact envelope.
enum class vbr_explicit_size_failure : uint8_t {
    none = 0,
    not_armed,
    tracker_missing,
    tracker_unstable,
    bindings_missing,
    stream_layout,
    policy_snapshot,
    unit_index,
    extents_empty,
    extent_missing,
    vmm_missing,
    backend_unavailable,
    wm_cells_zero,
    extent_type_mismatch,
    promote_hops_mismatch,
    domain_mismatch,
    shard_disagreement,
    binding_missing,
    topology_order,
    bounds,
    stash_bounds,
    stability_reread,
    internal_error,
    _count,
};

const char * vbr_explicit_size_failure_name(
    vbr_explicit_size_failure failure) noexcept;

// Pure production predicate used by size-pass and its CPU regression. A
// never-retiered F16 unit is valid when ordinary decode has established a
// nonzero mapped watermark; the VBR side-stream backend is deliberately not
// part of this generation predicate because capture initializes it lazily.
vbr_explicit_size_failure vbr_explicit_capture_validate_extent_generation(
    uint32_t wm_cells,
    int32_t extent_type,
    uint8_t extent_promote_hops,
    const vbr_unit_generation & generation) noexcept;

// One runtime pool-to-portable-topology binding. Device ordinals are portable
// only within the cited topology; lane identifies the F3.1 D2H ring lane.
struct vbr_explicit_capture_pool_binding {
    vbr_controller_instance_id instance_id;
    int device = -1;
    uint32_t topology_index = UINT32_MAX;
    uint16_t device_ordinal = UINT16_MAX;
    uint32_t lane = UINT32_MAX;
};

// Internal F3.3 discovery result. It exposes only the runtime backend binding
// needed to build the server-owned ring and the portable pool mapping; no KV
// bytes, masks, or ownership state cross this seam.
struct vbr_explicit_capture_runtime_pool {
    vbr_controller_instance_id instance_id;
    int device = -1;
    ggml_backend_dev_t backend_device = nullptr;
    ggml_backend_t backend = nullptr;
};

bool vbr_explicit_capture_runtime_pools(
    llama_memory_i & memory,
    std::vector<vbr_explicit_capture_runtime_pool> & pools,
    uint32_t & attention_children) noexcept;

// F4.3 live-import inspection doors. They share the capture adapter's private
// KV geometry access but are read-only: validation/staging consume the values,
// and only vbr_adopt_empty_manifest may mutate the target.
uint64_t vbr_explicit_import_policy_epoch(
    llama_memory_i & memory) noexcept;
bool vbr_explicit_import_target_snapshot(
    llama_memory_i & memory,
    llama_seq_id destination,
    const vbr_artifact_package_view & package,
    const std::vector<llama_vbr_artifact_domain_binding> & bindings,
    bool previously_observed,
    uint64_t accounting_serial,
    vbr_target_validation_snapshot & output,
    vbr_downward_policy_projection * downward_projection = nullptr,
    bool * downward_required = nullptr) noexcept;
bool vbr_explicit_import_target_recheck(
    llama_memory_i & memory,
    llama_seq_id destination,
    const vbr_target_empty_fingerprint & expected) noexcept;
bool vbr_explicit_import_reserve_downward(
    llama_memory_i & memory,
    const std::vector<vbr_validated_child_plan> & plans,
    llama_cache_acct_ledger & ledger,
    const llama_cache_budget_config & budget,
    vbr_downward_stage_reservation & output) noexcept;

struct vbr_explicit_representation_identity {
    uint32_t codec_id = 0;
    uint32_t codec_version = 0;
    std::array<uint8_t, 32> codebook_digest = {};
    std::array<uint8_t, 32> rotation_digest = {};
    std::array<uint8_t, 32> meansub_digest = {};
};

// Server policy input to the library-owned codec identity recipe. The build
// identity distinguishes compiled-in codebooks; file overrides, rotations,
// and mean-subtraction state are discovered and hashed by the codec layer.
struct vbr_explicit_representation_policy {
    const char * build_identity = nullptr;
    size_t build_identity_len = 0;
    // Baked-mean registry ID of the capturing model (hparams.turbo_meansub_id) since the
    // per-model mean-table isolation; 0 = no baked table (digest records "inactive").
    int turbo_meansub_id = 0;
};

bool vbr_explicit_capture_representation_identity(
    const void * context,
    int32_t current_type,
    bool value_side,
    vbr_explicit_representation_identity & output) noexcept;

struct vbr_explicit_companion_provider {
    using capture_fn = bool (*)(
        const void * context,
        llama_seq_id sequence,
        std::vector<uint8_t> & output) noexcept;
    using size_fn = bool (*)(
        const void * context,
        llama_seq_id sequence,
        uint64_t & output) noexcept;

    vbr_artifact_companion_kind kind =
        vbr_artifact_companion_kind::typed_accelerator;
    uint32_t format_version = 1;
    std::array<uint8_t, 32> build_identity_digest = {};
    vbr_artifact_portable_domain domain;
    bool required = true;
    const void * context = nullptr;
    size_fn size = nullptr;
    capture_fn capture = nullptr;
};

struct vbr_explicit_capture_request {
    using representation_identity_fn = bool (*)(
        const void * context,
        int32_t current_type,
        bool value_side,
        vbr_explicit_representation_identity & output) noexcept;

    llama_seq_id sequence = -1;
    vbr_checkpoint_frontier_fields frontier;
    vbr_artifact_identity_block identity;
    std::vector<llama_token> token_block;
    // Optional expected canonical digest. Zero asks the library to derive it
    // from frontier + ordered memory-tree child policy; a nonzero value must
    // match exactly.
    std::array<uint8_t, 32> identity_policy_order_digest = {};
    bool idle_decode_thread = false;
    vbr_pinned_chunk_ring * ring = nullptr;
    std::vector<vbr_artifact_portable_topology> topologies;
    std::vector<vbr_explicit_capture_pool_binding> pool_bindings;
    std::vector<vbr_explicit_companion_provider> companions;
    const void * representation_context = nullptr;
    representation_identity_fn representation_identity = nullptr;
};

struct vbr_explicit_capture_accounting {
    using prepare_fn = bool (*)(
        void * context,
        const vbr_artifact_package & package) noexcept;

    const llama_cache_budget_config * budget = nullptr;
    llama_cache_transaction_fault fault;
    void * context = nullptr;
    // Called after the exact package accounting manifest exists and before
    // begin_capture. A catalog binding/configuration adapter lives here
    // rather than weakening the generic sink interface.
    prepare_fn prepare = nullptr;
};

struct vbr_explicit_capture_result {
    vbr_explicit_capture_status status =
        vbr_explicit_capture_status::internal_error;
    vbr_explicit_capture_phase phase =
        vbr_explicit_capture_phase::validation;
    // Populated when a sink/ring/catalog boundary supplies a more specific
    // terminal. `_count` means the phase failed before such a boundary.
    vbr_capture_stream_status inner_stream_status =
        vbr_capture_stream_status::_count;
    vbr_explicit_generation_failure generation_failure =
        vbr_explicit_generation_failure::none;
    vbr_explicit_size_failure size_failure =
        vbr_explicit_size_failure::none;
    vbr_capture_begin_diagnostics begin_diagnostics;
    vbr_capture_sink_result sink;
    uint32_t controllers = 0;
    uint32_t units = 0;
    uint32_t companions = 0;
    uint64_t payload_bytes = 0;
    uint64_t stash_bytes = 0;
    uint64_t companion_bytes = 0;
    uint64_t chunks = 0;
    uint64_t backpressure_waits = 0;
    uint64_t event_completions = 0;
    uint64_t synchronous_fallbacks = 0;
};

vbr_explicit_capture_result vbr_capture_explicit_manifest(
    llama_memory_i & memory,
    const vbr_explicit_capture_request & request,
    vbr_unit_version_sink & sink,
    const vbr_explicit_capture_accounting & accounting) noexcept;

// H2's first automatic-capture boundary. One scheduler batch is capped well
// below the generic H1 arenas and is bound to one live memory tree. Semantic
// identity and token storage are owned values; no caller-owned string pointer
// is retained by the sealed projection.
constexpr uint32_t VBR_PROJECTED_CAPTURE_MAX_MANIFESTS = 8;

struct vbr_projected_capture_manifest_request {
    uint64_t manifest_id = 0;
    llama_seq_id sequence = -1;
    vbr_artifact_identity_block identity;
    std::vector<llama_token> token_block;
    // Zero derives the canonical frontier/policy digest. A nonzero value must
    // match, exactly as in explicit capture.
    std::array<uint8_t, 32> identity_policy_order_digest = {};
};

struct vbr_projected_capture_batch_request {
    using representation_identity_fn =
        vbr_explicit_capture_request::representation_identity_fn;

    struct pretransfer_quote {
        struct staging_row {
            vbr_artifact_portable_domain domain;
            uint64_t bytes = 0;
        };
        struct durable_manifest {
            uint64_t manifest_id = 0;
            // Conservative complete catalog rows for this manifest at the
            // currently projected physical union. Content-addressed dedup
            // may repartition these rows downward after D2H, never upward.
            std::vector<vbr_artifact_portable_accounting_row> accounting;
            // Unit-payload subset whose first immutable allocation is owned
            // by this manifest. The complete accounting above remains the
            // conservative final shape; unlisted unit bytes are reference
            // placeholders and reserve no duplicate physical capacity.
            std::vector<vbr_artifact_portable_accounting_row>
                reserve_accounting;
        };

        uint64_t planned_packed_bytes = 0;
        // Conservative compact prompt-cache footprint for the complete
        // projected batch: every manifest-local non-unit row plus the
        // first-owner physical unit union. Existing-catalog content dedup may
        // shrink this after D2H; it can never grow.
        uint64_t projected_host_resident_bytes = 0;
        uint64_t union_cells = 0;
        uint32_t manifests = 0;
        uint32_t projected_units = 0;
        // Exact physical transport payload grouped by accounting domain.
        // The synchronous store converts these rows into one split-phase
        // transfer-staging reservation before companion/attention D2H.
        std::vector<staging_row> staging;
        // Final manifest shapes plus first-allocation ownership for one
        // batch-level durable fence. Dependency-local shrink repartitions
        // this inventory; final assembly partitions independent terminals.
        std::vector<durable_manifest> durable;
    };
    using pretransfer_admit_fn = bool (*)(
        void * context, const pretransfer_quote & quote) noexcept;
    using pretransfer_shrink_fn = bool (*)(
        void * context, const pretransfer_quote & quote) noexcept;
    using continue_transfer_fn = bool (*)(void * context) noexcept;

    bool idle_decode_thread = false;
    // Scheduler-admitted aggregate pageable payload runway for this batch.
    // Required and checked before the first D2H byte.
    uint64_t max_packed_bytes = 0;
    std::vector<vbr_projected_capture_manifest_request> manifests;
    vbr_pinned_chunk_ring * ring = nullptr;
    std::vector<vbr_artifact_portable_topology> topologies;
    std::vector<vbr_explicit_capture_pool_binding> pool_bindings;
    const void * representation_context = nullptr;
    representation_identity_fn representation_identity = nullptr;
    // Invoked exactly once after the initial bounded projection is priced and
    // before companion or attention D2H begins. For nonzero work the
    // batch-long ring operation is already held; the canonical zero-work quote
    // requires none. The synchronous caller may refuse when queued work or a
    // scheduler-owned reservation changed while planning. Dependency-local
    // companion failures may later shrink the admitted union; they never grow
    // it or invoke this callback again. Refusal returns admission_refused with
    // zero unit transfers and releases any ring operation.
    void * pretransfer_context = nullptr;
    pretransfer_admit_fn pretransfer_admit = nullptr;
    // Optional store-owned reservation shrink after a dependency-local
    // companion failure removes rows from the admitted union. It may only
    // replace the original claim with a smaller one and never re-enters the
    // scheduler policy callback.
    pretransfer_shrink_fn pretransfer_shrink = nullptr;
    // Optional cancellation probe used only after pretransfer admission. It
    // is checked between recurrent <=1 MiB writes and attention ring chunks.
    // False aborts the complete batch without publication.
    void * continue_context = nullptr;
    continue_transfer_fn continue_transfer = nullptr;
};

struct vbr_projected_capture_batch_result {
    vbr_explicit_capture_status status =
        vbr_explicit_capture_status::internal_error;
    vbr_explicit_capture_phase phase =
        vbr_explicit_capture_phase::validation;
    vbr_capture_stream_status inner_stream_status =
        vbr_capture_stream_status::_count;
    vbr_explicit_generation_failure generation_failure =
        vbr_explicit_generation_failure::none;
    vbr_explicit_size_failure size_failure =
        vbr_explicit_size_failure::none;
    vbr_capture_manifest_assembly assembly;
    std::vector<vbr_projected_manifest_publication> publications;
    uint64_t source_namespace = 0;
    // First request-order manifest that survived dependency preparation.
    // This is retry-selection evidence only; it authorizes no publication.
    uint64_t first_available_manifest_id = 0;
    uint64_t union_cells = 0;
    uint64_t planned_packed_bytes = 0;
    uint32_t size_pass_calls = 0;
    uint32_t projection_calls = 0;
    uint32_t unit_transfer_calls = 0;
    uint32_t transferred_units = 0;
    uint64_t companion_d2h_bytes = 0;
    uint64_t companion_d2h_reads = 0;
    uint32_t ring_operation_attempts = 0;
    uint32_t ring_operation_acquires = 0;
    uint32_t ring_operation_refusals = 0;
    vbr_capture_stream_stats transfer;
};

// Produces immutable H1 capabilities and the narrow publication envelopes
// consumed by the server-owned catalog adapter. Required recurrent state is
// sealed per manifest; clean stash payloads, payload-complete dependencies,
// and non-unified controllers fail closed in this first slice.
vbr_projected_capture_batch_result vbr_capture_projected_batch(
    llama_memory_i & memory,
    const vbr_projected_capture_batch_request & request) noexcept;
