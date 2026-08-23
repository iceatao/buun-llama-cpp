#pragma once

#include "llama-vbr-artifact-validate.h"
#include "llama-vbr-pinned-ring.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

enum class vbr_h2d_status : uint8_t {
    ok = 0,
    invalid_argument,
    ring_unavailable,
    source_read_failed,
    transfer_failed,
    event_failed,
    internal_error,
    _count,
};

const char * vbr_h2d_status_name(vbr_h2d_status status) noexcept;

struct vbr_h2d_lane_binding {
    llama_cache_acct_resource_domain domain;
    ggml_backend_dev_t device = nullptr;
    ggml_backend_t backend = nullptr;
    bool force_synchronous = false;
};

// The callback form is the CPU/fake-backend seam. `issue` may retain the
// pinned pointer only until its matching complete call. Production uses the
// tensor form and ggml's event/synchronize contract.
struct vbr_h2d_fake_destination {
    using issue_fn = bool (*)(
        void * context, uint64_t ticket, uint64_t offset,
        const uint8_t * data, size_t size, bool asynchronous) noexcept;
    using complete_fn = bool (*)(
        void * context, uint64_t ticket) noexcept;

    void * context = nullptr;
    issue_fn issue = nullptr;
    complete_fn complete = nullptr;
    bool supports_events = false;
};

struct vbr_h2d_transfer {
    uint32_t lane = 0;
    vbr_artifact_byte_source source;
    uint64_t source_offset = 0;
    uint64_t size = 0;

    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    ggml_tensor * destination = nullptr;
    uint64_t destination_offset = 0;
    vbr_h2d_fake_destination fake;

    uint64_t fail_completion_at = UINT64_MAX;
};

struct vbr_h2d_source_range {
    uint64_t source_offset = 0;
    uint64_t size = 0;
};

struct vbr_h2d_packed_transfer {
    uint32_t lane = 0;
    vbr_artifact_byte_source source;
    const vbr_h2d_source_range * ranges = nullptr;
    size_t range_count = 0;
    uint64_t size = 0;

    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    ggml_tensor * destination = nullptr;
    uint64_t destination_offset = 0;
    vbr_h2d_fake_destination fake;

    uint64_t fail_completion_at = UINT64_MAX;
};

struct vbr_h2d_stats {
    uint64_t bytes = 0;
    uint64_t chunks = 0;
    uint64_t backpressure_waits = 0;
    uint64_t event_completions = 0;
    uint64_t synchronous_fallbacks = 0;
    uint64_t peak_pinned_bytes = 0;
};

// Keeps the shared physical ring alive while projection adoption owns its
// direction-neutral operation mutex.
class vbr_h2d_ring_operation {
public:
    vbr_h2d_ring_operation() noexcept = default;
    vbr_h2d_ring_operation(vbr_h2d_ring_operation &&) noexcept;
    vbr_h2d_ring_operation & operator=(
        vbr_h2d_ring_operation &&) noexcept;
    ~vbr_h2d_ring_operation();

    vbr_h2d_ring_operation(const vbr_h2d_ring_operation &) = delete;
    vbr_h2d_ring_operation & operator=(
        const vbr_h2d_ring_operation &) = delete;
    explicit operator bool() const noexcept { return bool(operation_); }

private:
    std::shared_ptr<vbr_bounded_pinned_ring_core> keepalive_;
    vbr_pinned_ring_operation operation_;
    friend class vbr_h2d_chunk_ring;
};

class vbr_h2d_chunk_ring {
public:
    static std::unique_ptr<vbr_h2d_chunk_ring> create(
        const std::vector<vbr_h2d_lane_binding> & lanes,
        uint64_t total_bytes,
        size_t chunk_bytes,
        vbr_h2d_status & status,
        vbr_pinned_ring_create_failure * failure = nullptr) noexcept;

    // Builds the H2D adapter over an already-accounted persistent core.
    static std::shared_ptr<vbr_h2d_chunk_ring> attach(
        std::shared_ptr<vbr_bounded_pinned_ring_core> core,
        const std::vector<vbr_h2d_lane_binding> & lanes) noexcept;

    ~vbr_h2d_chunk_ring();
    vbr_h2d_chunk_ring(const vbr_h2d_chunk_ring &) = delete;
    vbr_h2d_chunk_ring & operator=(const vbr_h2d_chunk_ring &) = delete;

    uint64_t capacity_bytes() const noexcept;
    size_t chunk_bytes() const noexcept;
    size_t lane_count() const noexcept;
    bool compatible_with(
        const llama_cache_acct_ledger * ledger,
        const llama_cache_acct_snapshot & snapshot,
        const llama_cache_acct_resource_domain & domain,
        uint64_t capacity_bytes,
        size_t chunk_bytes,
        const std::vector<vbr_h2d_lane_binding> & lanes) const noexcept;
    vbr_h2d_status stream(
        const vbr_h2d_transfer & transfer,
        vbr_h2d_stats & stats) noexcept;
    // Projection adoption reserves the direction-neutral ring once and packs
    // discontiguous authenticated parent ranges into a dense destination.
    // Exact/full-package adoption continues to use stream().
    vbr_h2d_ring_operation try_begin_operation() noexcept;
    vbr_h2d_status stream_packed_reserved(
        const vbr_h2d_ring_operation & operation,
        const vbr_h2d_packed_transfer & transfer,
        vbr_h2d_stats & stats) noexcept;

private:
    struct impl;
    explicit vbr_h2d_chunk_ring(std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;
};

enum class vbr_staged_read_kind : uint8_t {
    unit_payload = 0,
    clean_stash,
    companion,
    recovery_unit_payload,
    _count,
};

struct vbr_staged_read_descriptor {
    vbr_staged_read_kind kind = vbr_staged_read_kind::unit_payload;
    uint32_t child_id = UINT32_MAX;
    uint32_t logical_unit_id = UINT32_MAX;
    uint32_t shard_index = UINT32_MAX;
    uint32_t lane = UINT32_MAX;
    uint64_t source_offset = 0;
    uint64_t size = 0;
    std::shared_ptr<const artifact_segment_chain> source;
    std::array<uint8_t, 32> verified_digest = {};
    // Populated only for an authenticated prefix projection. Exact reads
    // preserve their historic source_offset-as-destination mapping.
    uint64_t destination_offset = 0;
    std::vector<vbr_h2d_source_range> projection_ranges;
    uint64_t proof_verified_bytes = 0;
};

enum class vbr_adopt_stage_status : uint8_t {
    staged = 0,
    invalid_proof,
    unsupported_decision,
    source_unavailable,
    source_hash_mismatch,
    accounting_unavailable,
    admission_refused,
    ring_unavailable,
    transform_projection_unavailable,
    transform_reserve_failed,
    internal_error,
    _count,
};

const char * vbr_adopt_stage_status_name(
    vbr_adopt_stage_status status) noexcept;

struct vbr_adopt_stage_fault {
    uint32_t fail_source_verify_at = UINT32_MAX;
    bool fail_before_prepare = false;
    bool fail_ring_allocation = false;
};

struct vbr_adopt_stage_policy {
    using reserve_transform_fn = bool (*)(
        void * context,
        const std::vector<vbr_validated_child_plan> & plans,
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget,
        vbr_downward_stage_reservation & output) noexcept;
    llama_cache_acct_ledger * ledger = nullptr;
    const llama_cache_budget_config * budget = nullptr;
    std::vector<vbr_h2d_lane_binding> lanes;
    llama_cache_acct_resource_domain pinned_domain;
    uint64_t pinned_ring_bytes = 0;
    size_t chunk_bytes = 0;
    // Production supplies the store-owned ring. When absent, explicit tests
    // and standalone clients retain the legacy per-stage construction path.
    std::shared_ptr<vbr_h2d_chunk_ring> persistent_ring;
    void * transform_context = nullptr;
    reserve_transform_fn reserve_transform = nullptr;
    vbr_adopt_stage_fault fault;
};

struct vbr_adopt_stage_result;
struct vbr_adopt_result;
struct vbr_composite_publish_hooks;

class vbr_staged_payloads {
public:
    vbr_staged_payloads(vbr_staged_payloads &&) noexcept;
    vbr_staged_payloads & operator=(vbr_staged_payloads &&) noexcept;
    ~vbr_staged_payloads();

    vbr_staged_payloads(const vbr_staged_payloads &) = delete;
    vbr_staged_payloads & operator=(const vbr_staged_payloads &) = delete;

    uint64_t adoption_nonce() const noexcept;
    uint64_t validation_accounting_serial() const noexcept;
    uint64_t accounting_serial_after_prepare() const noexcept;
    const vbr_manifest_digest & manifest_digest() const noexcept;
    const vbr_target_empty_fingerprint & target_fingerprint() const noexcept;
    vbr_import_decision decision() const noexcept;
    size_t read_count() const noexcept;
    const std::vector<vbr_staged_read_descriptor> & reads() const noexcept;
    uint64_t ring_capacity_bytes() const noexcept;
    bool claims_ready() const noexcept;
    const std::vector<uint64_t> & transform_stashless_units() const noexcept;
    bool transform_resources_ready() const noexcept;

private:
    llama_cache_transaction_result adoption_materialize_claims() noexcept;
    vbr_h2d_chunk_ring * adoption_ring() noexcept;
    const std::vector<llama_cache_acct_op_id> &
        adoption_committed_ops() const noexcept;
    struct impl;
    explicit vbr_staged_payloads(std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;

    friend struct vbr_adopt_stage_result;
    friend vbr_adopt_stage_result vbr_stage_validated_manifest(
        std::unique_ptr<vbr_validated_manifest>,
        const vbr_adopt_stage_policy &) noexcept;
    friend vbr_adopt_result vbr_adopt_empty_manifest(
        llama_memory_i &, llama_seq_id,
        vbr_validated_manifest &&, vbr_staged_payloads &&,
        llama_cache_acct_ledger &,
        const vbr_composite_publish_hooks &) noexcept;
};

struct vbr_adopt_stage_result {
    vbr_adopt_stage_status status =
        vbr_adopt_stage_status::internal_error;
    vbr_downward_reserve_status transform_status =
        vbr_downward_reserve_status::not_attempted;
    std::unique_ptr<vbr_validated_manifest> manifest;
    std::unique_ptr<vbr_staged_payloads> staged;
};

vbr_adopt_stage_result vbr_stage_validated_manifest(
    std::unique_ptr<vbr_validated_manifest> proof,
    const vbr_adopt_stage_policy & policy) noexcept;
