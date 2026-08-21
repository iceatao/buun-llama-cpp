#pragma once

#include "llama-cache-authority.h"
#include "llama-vbr-artifact.h"
#include "llama-vbr-pinned-ring.h"

#include "ggml-backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// F3.1 bounded streaming substrate. These types are internal to libllama:
// no live KV/cache or server policy enters this unit.
static constexpr uint64_t VBR_CAPTURE_PINNED_RING_MAX_BYTES =
    VBR_PINNED_RING_MAX_BYTES;

enum class vbr_capture_stream_status : uint8_t {
    ok = 0,
    invalid_argument,
    ring_unavailable,
    transfer_failed,
    short_read,
    duplicate_segment,
    missing_segment,
    late_segment,
    hash_mismatch,
    format_rejected,
    accounting_unavailable,
    accounting_refused,
    stage_failed,
    commit_failed,
    publication_failed,
    internal_error,
    _count,
};

const char * vbr_capture_stream_status_name(
    vbr_capture_stream_status status) noexcept;

using vbr_capture_ring_create_failure =
    vbr_pinned_ring_create_failure;

const char * vbr_capture_ring_create_failure_name(
    vbr_capture_ring_create_failure failure) noexcept;

struct artifact_segment {
    std::shared_ptr<const std::vector<uint8_t>> storage;
    uint64_t offset = 0;
    uint64_t length = 0;
};

// Immutable segmented pageable backing. Each append allocates at most one
// capture chunk; source() supports arbitrary reads across segment boundaries
// and never concatenates the complete artifact.
class artifact_segment_chain {
public:
    artifact_segment_chain();
    ~artifact_segment_chain();

    artifact_segment_chain(const artifact_segment_chain &) = delete;
    artifact_segment_chain & operator=(const artifact_segment_chain &) = delete;
    artifact_segment_chain(artifact_segment_chain &&) noexcept;
    artifact_segment_chain & operator=(artifact_segment_chain &&) noexcept;

    bool append(const uint8_t * data, size_t size) noexcept;
    uint64_t size() const noexcept;
    size_t segment_count() const noexcept;
    size_t max_segment_size() const noexcept;
    bool read(uint64_t offset, uint8_t * destination, size_t size) const noexcept;
    vbr_artifact_byte_source source() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

std::array<uint8_t, 32> vbr_capture_stream_digest(
    const artifact_segment_chain & chain) noexcept;

using vbr_capture_lane = vbr_pinned_ring_lane;

using vbr_capture_ring_accounting =
    vbr_pinned_ring_accounting;

struct vbr_capture_stream_source {
    using read_fn = bool (*)(
        const void * context,
        uint64_t offset,
        uint8_t * destination,
        size_t size) noexcept;

    uint32_t lane = 0;
    uint64_t size = 0;

    // Exactly one source shape is used. tensor != nullptr selects backend D2H;
    // otherwise read supplies deterministic CPU/synthetic bytes.
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    const ggml_tensor * tensor = nullptr;
    uint64_t tensor_offset = 0;
    const void * context = nullptr;
    read_fn read = nullptr;

    // Deterministic synthetic completion-fault seam. Production callers keep
    // UINT64_MAX; tests prove a failed completion drains the ring and exposes
    // no verified segment.
    uint64_t fail_completion_at = UINT64_MAX;
};

struct vbr_capture_stream_stats {
    uint64_t bytes = 0;
    uint64_t chunks = 0;
    uint64_t backpressure_waits = 0;
    uint64_t event_completions = 0;
    uint64_t synchronous_fallbacks = 0;
    size_t max_segment_size = 0;
    std::array<uint8_t, 32> streaming_digest = {};
};

// H1 sequence-projected capture planning. Each logical manifest contributes
// exact live placement evidence; the planner lowers their physical-row union
// into deterministic runs whose dependency sets identify precisely which
// manifests must be cancelled if that run cannot be sealed. This is a
// process-local capture plan, not wire metadata.
struct vbr_capture_projection_manifest {
    uint64_t manifest_id = 0;
    std::vector<vbr_artifact_stream_placement> placements;
};

struct vbr_capture_projection_segment {
    uint32_t first_physical_cell = 0;
    uint32_t cell_count = 0;
    uint32_t first_dependency = 0;
    uint32_t dependency_count = 0;
};

struct vbr_capture_projection_stream {
    uint32_t child_id = UINT32_MAX;
    uint32_t stream_index = UINT32_MAX;
    std::vector<vbr_capture_projection_segment> segments;
};

struct vbr_capture_projection_limits {
    uint32_t max_manifests = 4096;
    uint32_t max_placements = 4096;
    uint32_t max_input_cells = 1048576;
    uint32_t max_union_cells = 1048576;
    uint32_t max_segments = 1048576;
    uint32_t max_dependency_references = 1048576;
};

// One batch is structurally bound to one live memory-tree namespace. Child
// and stream IDs are meaningful only within that immutable source namespace;
// callers must start a separate batch for another live tree.
struct vbr_capture_projection_batch {
    uint64_t source_namespace = 0;
    std::vector<vbr_capture_projection_manifest> manifests;
};

struct vbr_capture_projection_plan {
    uint64_t source_namespace = 0;
    uint32_t manifest_count = 0;
    uint32_t placement_count = 0;
    uint64_t input_cell_references = 0;
    uint64_t union_cell_count = 0;
    uint64_t dependency_references = 0;
    std::vector<vbr_capture_projection_stream> streams;
    std::vector<uint64_t> dependent_manifest_ids;
};

// Allocation failure, malformed placement evidence, duplicate identities, or
// any limit violation clears output and returns false.
bool vbr_artifact_project_capture_union(
    const vbr_capture_projection_batch & batch,
    const vbr_capture_projection_limits & limits,
    vbr_capture_projection_plan & output) noexcept;

// One globally-bounded ring split across per-device lanes. A null device lane
// is the deterministic CPU test path. Real lanes allocate that device's host
// buffer type and use optional backend events; no event means a synchronized
// fallback, never an unbounded allocation.
class vbr_pinned_chunk_ring {
public:
    static std::unique_ptr<vbr_pinned_chunk_ring> create(
        const std::vector<vbr_capture_lane> & lanes,
        uint64_t total_bytes,
        size_t chunk_bytes,
        vbr_capture_stream_status & status,
        const vbr_capture_ring_accounting * accounting =
            nullptr,
        vbr_capture_ring_create_failure * failure =
            nullptr) noexcept;

    ~vbr_pinned_chunk_ring();
    vbr_pinned_chunk_ring(const vbr_pinned_chunk_ring &) = delete;
    vbr_pinned_chunk_ring & operator=(const vbr_pinned_chunk_ring &) = delete;

    uint64_t capacity_bytes() const noexcept;
    size_t chunk_bytes() const noexcept;
    size_t lane_count() const noexcept;

    vbr_capture_stream_status stream(
        const vbr_capture_stream_source & source,
        artifact_segment_chain & destination,
        vbr_capture_stream_stats & stats) noexcept;

private:
    struct impl;
    explicit vbr_pinned_chunk_ring(std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;
};

struct vbr_verified_segment {
    uint32_t unit_index = UINT32_MAX;
    uint32_t shard_index = UINT32_MAX;
    bool clean_stash = false;
    std::shared_ptr<const artifact_segment_chain> bytes;
    std::array<uint8_t, 32> streaming_digest = {};
};

struct vbr_verified_companion {
    uint32_t companion_index = UINT32_MAX;
    std::shared_ptr<const artifact_segment_chain> bytes;
    std::array<uint8_t, 32> streaming_digest = {};
};

struct vbr_capture_sink_result {
    vbr_capture_stream_status status =
        vbr_capture_stream_status::internal_error;
    llama_cache_acct_artifact_id reference_artifact;
    llama_cache_acct_content_digest unit_content;
    llama_cache_acct_lineage_id reference_lineage;
    bool adopted = false;
};

enum class vbr_capture_reservation_group : uint8_t {
    none = 0,
    transfer_staging,
    durable_artifact,
    _count,
};

const char * vbr_capture_reservation_group_name(
    vbr_capture_reservation_group group) noexcept;

struct vbr_capture_begin_diagnostics {
    vbr_capture_reservation_group reservation_group =
        vbr_capture_reservation_group::none;
    llama_cache_prepare_status prepare_status =
        llama_cache_prepare_status::prepared;
    llama_cache_admission_status admission_status =
        llama_cache_admission_status::admitted;
    size_t failed_leaf = SIZE_MAX;
};

class vbr_unit_build {
public:
    virtual ~vbr_unit_build() = default;
    virtual vbr_capture_stream_status accept_verified_segment(
        const vbr_verified_segment & segment) noexcept = 0;
    virtual vbr_capture_stream_status seal_unit() noexcept = 0;
};

class vbr_capture_build {
public:
    virtual ~vbr_capture_build() = default;
    virtual std::unique_ptr<vbr_unit_build> begin_unit(
        uint32_t unit_index,
        vbr_capture_stream_status & status) noexcept = 0;
    virtual vbr_capture_stream_status accept_verified_companion(
        const vbr_verified_companion & companion) noexcept = 0;
    virtual vbr_capture_sink_result publish_reference() noexcept = 0;
};

class vbr_unit_version_sink {
public:
    virtual ~vbr_unit_version_sink() = default;
    virtual std::unique_ptr<vbr_capture_build> begin_capture(
        const vbr_artifact_package & package,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        vbr_capture_stream_status & status,
        vbr_capture_begin_diagnostics * diagnostics =
            nullptr) noexcept = 0;
};
