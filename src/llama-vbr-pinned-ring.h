#pragma once

#include "llama-cache-budget.h"

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// One process-wide ceiling shared by capture (D2H) and adoption (H2D)
// instances. A production store uses one direction-neutral core for both.
static constexpr uint64_t VBR_PINNED_RING_MAX_BYTES = 256ull*1024*1024;

// Process-level pinned transport inventory for diagnostics and admission
// tests. It counts physical ring bytes, not transient transfer claims.
uint64_t vbr_pinned_ring_live_capacity_bytes() noexcept;

enum class vbr_pinned_ring_create_failure : uint8_t {
    none = 0,
    invalid_geometry,
    invalid_accounting_binding,
    existing_ring_charge,
    accounting_update_failed,
    budget_reset_failed,
    budget_unavailable,
    budget_exceeded,
    global_capacity_exceeded,
    invalid_lane_binding,
    duplicate_device_lane,
    host_buffer_type_unavailable,
    host_buffer_allocation_failed,
    host_buffer_too_small,
    host_buffer_base_unavailable,
    lane_underprovisioned,
    accounting_charge_failed,
    internal_error,
    _count,
};

struct vbr_pinned_ring_lane {
    ggml_backend_dev_t device = nullptr;
    ggml_backend_t backend = nullptr;
    bool force_synchronous = false;
};

struct vbr_pinned_ring_accounting {
    llama_cache_acct_ledger * ledger = nullptr;
    llama_cache_acct_resource_domain domain;
    const llama_cache_budget_config * budget = nullptr;
    llama_cache_acct_category category =
        llama_cache_acct_category::pinned_preimage_ring;

    // Deterministic test seam for a ledger fault between physical allocation
    // and the final ring gauge. Production leaves both fields null.
    void * charge_fault_context = nullptr;
    void (*inject_charge_fault)(void * context) noexcept = nullptr;
};

class vbr_bounded_pinned_ring_core;
class vbr_pinned_chunk_ring;
class vbr_h2d_chunk_ring;

// Move-only nonblocking ownership of the complete ring pump. A shared
// direction-neutral core may serve capture or adoption, but never lets the
// two adapters consume each other's outstanding chunks.
class vbr_pinned_ring_operation {
public:
    vbr_pinned_ring_operation() noexcept = default;
    vbr_pinned_ring_operation(vbr_pinned_ring_operation && other) noexcept;
    vbr_pinned_ring_operation & operator=(
        vbr_pinned_ring_operation && other) noexcept;
    ~vbr_pinned_ring_operation();

    vbr_pinned_ring_operation(const vbr_pinned_ring_operation &) = delete;
    vbr_pinned_ring_operation & operator=(
        const vbr_pinned_ring_operation &) = delete;

    explicit operator bool() const noexcept { return owner_ != nullptr; }

private:
    vbr_bounded_pinned_ring_core * owner_ = nullptr;
    std::shared_ptr<vbr_bounded_pinned_ring_core> keepalive_;
    explicit vbr_pinned_ring_operation(
        vbr_bounded_pinned_ring_core * owner) noexcept : owner_(owner) {}
    void reset() noexcept;

    friend class vbr_bounded_pinned_ring_core;
    friend class vbr_pinned_chunk_ring;
};

// Move-only ownership of one ring chunk between acquire and release. The
// adapter that submits a transfer is responsible for waiting before release.
class vbr_pinned_chunk_lease {
public:
    vbr_pinned_chunk_lease() = default;
    vbr_pinned_chunk_lease(vbr_pinned_chunk_lease && other) noexcept;
    vbr_pinned_chunk_lease & operator=(vbr_pinned_chunk_lease && other) noexcept;
    ~vbr_pinned_chunk_lease() = default;

    vbr_pinned_chunk_lease(const vbr_pinned_chunk_lease &) = delete;
    vbr_pinned_chunk_lease & operator=(const vbr_pinned_chunk_lease &) = delete;

    explicit operator bool() const noexcept { return chunk_ != nullptr; }
    uint8_t * data() const noexcept { return data_; }
    size_t capacity() const noexcept { return capacity_; }
    size_t valid() const noexcept { return valid_; }

private:
    vbr_bounded_pinned_ring_core * owner_ = nullptr;
    void * chunk_ = nullptr;
    uint8_t * data_ = nullptr;
    size_t capacity_ = 0;
    size_t valid_ = 0;

    void reset() noexcept;
    friend class vbr_bounded_pinned_ring_core;
};

// Direction-neutral bounded lane/chunk/event/backpressure core. It does not
// know whether bytes flow device->host or host->device and never hashes or
// retains artifact bytes.
class vbr_bounded_pinned_ring_core {
public:
    static std::unique_ptr<vbr_bounded_pinned_ring_core> create(
        const std::vector<vbr_pinned_ring_lane> & lanes,
        uint64_t total_bytes,
        size_t chunk_bytes,
        const vbr_pinned_ring_accounting * accounting,
        vbr_pinned_ring_create_failure & failure) noexcept;

    ~vbr_bounded_pinned_ring_core();
    vbr_bounded_pinned_ring_core(const vbr_bounded_pinned_ring_core &) = delete;
    vbr_bounded_pinned_ring_core & operator=(const vbr_bounded_pinned_ring_core &) = delete;

    uint64_t capacity_bytes() const noexcept;
    size_t chunk_bytes() const noexcept;
    size_t lane_count() const noexcept;

private:
    // Direction adapters provide only byte-oriented fill/consume callbacks.
    // Chunk leases, events, backpressure, and pending ownership stay inside
    // the core so D2H and H2D cannot grow subtly different ring pumps.
    struct pump_step {
        size_t valid = 0;
        ggml_backend_t backend = nullptr;
        uint64_t tag = 0;
        bool adapter_async = false;
        bool adapter_synchronous_fallback = false;
    };

    struct pump_callbacks {
        void * context = nullptr;
        uint32_t ok = 0;
        uint32_t ring_unavailable = 0;
        uint32_t submit_failed = 0;
        uint32_t wait_failed = 0;
        uint32_t internal_error = 0;
        // Cancellation-aware D2H bounds response latency by allowing only
        // one submitted chunk at a time. Legacy/null-cancellation adapters
        // retain the normal pipelined depth.
        bool serialize_submissions = false;
        uint32_t (*fill)(
            void * context, uint8_t * destination,
            size_t capacity, pump_step & step) noexcept = nullptr;
        bool (*more)(void * context) noexcept = nullptr;
        uint32_t (*consume)(
            void * context, const uint8_t * source, size_t size,
            uint64_t tag, bool adapter_async, uint64_t ordinal,
            bool & event_completion) noexcept = nullptr;
        void (*abandon)(
            void * context, uint64_t tag,
            bool adapter_async) noexcept = nullptr;
    };

    struct pump_stats {
        uint64_t bytes = 0;
        uint64_t chunks = 0;
        uint64_t submitted_bytes = 0;
        uint64_t submitted_chunks = 0;
        uint64_t backpressure_waits = 0;
        uint64_t event_completions = 0;
        uint64_t synchronous_fallbacks = 0;
        uint64_t peak_pinned_bytes = 0;
    };

    uint32_t pump(
        uint32_t lane, const pump_callbacks & callbacks,
        pump_stats & stats) noexcept;
    uint32_t pump_reserved(
        const vbr_pinned_ring_operation & operation,
        uint32_t lane, const pump_callbacks & callbacks,
        pump_stats & stats) noexcept;

    // Refuses immediately when the other direction currently owns the ring.
    // Only direction adapters can mint the operation token, so the shared
    // core necessarily outlives it.
    vbr_pinned_ring_operation try_begin_operation() noexcept;
    const vbr_pinned_ring_lane * lane_binding(uint32_t lane) const noexcept;
    bool accounted_to(
        const llama_cache_acct_ledger * ledger,
        const llama_cache_acct_snapshot & snapshot,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_category category) const noexcept;
    vbr_pinned_chunk_lease acquire(
        uint32_t lane, bool & would_block) noexcept;
    bool submit(
        vbr_pinned_chunk_lease & lease,
        size_t valid,
        ggml_backend_t backend,
        bool & synchronous_fallback) noexcept;
    bool wait(
        vbr_pinned_chunk_lease & lease,
        bool & event_completion) noexcept;
    void release(vbr_pinned_chunk_lease & lease) noexcept;
    void end_operation() noexcept;
    struct impl;
    explicit vbr_bounded_pinned_ring_core(std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;
    friend class vbr_pinned_ring_operation;
    friend class vbr_pinned_chunk_ring;
    friend class vbr_h2d_chunk_ring;
};
