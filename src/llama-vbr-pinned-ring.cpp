#include "llama-vbr-pinned-ring.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <mutex>
#include <new>
#include <utility>

namespace {

std::mutex vbr_pinned_ring_capacity_mutex;
uint64_t vbr_pinned_ring_capacity_live = 0;

} // namespace

uint64_t vbr_pinned_ring_live_capacity_bytes() noexcept {
    std::lock_guard<std::mutex> lock(vbr_pinned_ring_capacity_mutex);
    return vbr_pinned_ring_capacity_live;
}

struct vbr_bounded_pinned_ring_core::impl {
    struct chunk {
        ggml_backend_buffer_t buffer = nullptr;
        ggml_backend_event_t event = nullptr;
        std::vector<uint8_t> synthetic;
        uint8_t * data = nullptr;
        size_t valid = 0;
        bool busy = false;

        ~chunk() {
            if (event) {
                ggml_backend_event_free(event);
            }
            if (buffer) {
                ggml_backend_buffer_free(buffer);
            }
        }
    };

    struct lane {
        vbr_pinned_ring_lane binding;
        std::vector<std::unique_ptr<chunk>> chunks;
        size_t next = 0;
    };

    uint64_t capacity = 0;
    size_t chunk_size = 0;
    std::vector<lane> lanes;
    llama_cache_acct_ledger * accounting = nullptr;
    llama_cache_acct_resource_domain accounting_domain;
    llama_cache_acct_category accounting_category =
        llama_cache_acct_category::pinned_preimage_ring;
    void * charge_fault_context = nullptr;
    void (*inject_charge_fault)(void * context) noexcept = nullptr;
    bool ring_charged = false;

    bool global_reserved = false;
    std::mutex operation_mutex;

    ~impl() {
        // Release the process-wide bound only after every pinned allocation
        // in this instance is physically gone.
        lanes.clear();
        if (global_reserved) {
            std::lock_guard<std::mutex> lock(
                vbr_pinned_ring_capacity_mutex);
            if (vbr_pinned_ring_capacity_live >= capacity) {
                vbr_pinned_ring_capacity_live -= capacity;
            } else {
                vbr_pinned_ring_capacity_live = 0;
            }
        }
    }
};

vbr_pinned_ring_operation::vbr_pinned_ring_operation(
        vbr_pinned_ring_operation && other) noexcept
    : owner_(other.owner_) {
    other.owner_ = nullptr;
}

vbr_pinned_ring_operation & vbr_pinned_ring_operation::operator=(
        vbr_pinned_ring_operation && other) noexcept {
    if (this != &other) {
        reset();
        owner_ = other.owner_;
        other.owner_ = nullptr;
    }
    return *this;
}

vbr_pinned_ring_operation::~vbr_pinned_ring_operation() {
    reset();
}

void vbr_pinned_ring_operation::reset() noexcept {
    if (owner_) {
        owner_->end_operation();
        owner_ = nullptr;
    }
}

vbr_pinned_chunk_lease::vbr_pinned_chunk_lease(
        vbr_pinned_chunk_lease && other) noexcept
    : owner_(other.owner_), chunk_(other.chunk_), data_(other.data_),
      capacity_(other.capacity_), valid_(other.valid_) {
    other.reset();
}

vbr_pinned_chunk_lease & vbr_pinned_chunk_lease::operator=(
        vbr_pinned_chunk_lease && other) noexcept {
    if (this != &other) {
        owner_ = other.owner_;
        chunk_ = other.chunk_;
        data_ = other.data_;
        capacity_ = other.capacity_;
        valid_ = other.valid_;
        other.reset();
    }
    return *this;
}

void vbr_pinned_chunk_lease::reset() noexcept {
    owner_ = nullptr;
    chunk_ = nullptr;
    data_ = nullptr;
    capacity_ = 0;
    valid_ = 0;
}

vbr_bounded_pinned_ring_core::vbr_bounded_pinned_ring_core(
        std::unique_ptr<impl> state) noexcept
    : impl_(std::move(state)) {}

vbr_bounded_pinned_ring_core::~vbr_bounded_pinned_ring_core() {
    if (!impl_) {
        return;
    }
    auto * ledger = impl_->accounting;
    const auto domain = impl_->accounting_domain;
    const auto category = impl_->accounting_category;
    const bool charged = impl_->ring_charged;
    impl_->ring_charged = false;
    impl_.reset(); // pinned buffers disappear before their gauge does
    if (ledger && charged) {
        ledger->gauge_set(
            category, domain,
            llama_cache_acct_measure::logical_payload, 0);
        ledger->gauge_set(
            category, domain,
            llama_cache_acct_measure::resident_allocated, 0);
    }
}

std::unique_ptr<vbr_bounded_pinned_ring_core>
vbr_bounded_pinned_ring_core::create(
        const std::vector<vbr_pinned_ring_lane> & lanes,
        uint64_t total_bytes,
        size_t chunk_bytes,
        const vbr_pinned_ring_accounting * accounting,
        vbr_pinned_ring_create_failure & failure) noexcept {
    failure = vbr_pinned_ring_create_failure::none;
    try {
        if (lanes.empty() || chunk_bytes == 0 || total_bytes == 0 ||
            total_bytes > VBR_PINNED_RING_MAX_BYTES ||
            lanes.size() > std::numeric_limits<size_t>::max()/2 ||
            total_bytes / chunk_bytes < lanes.size()*2 ||
            chunk_bytes > std::numeric_limits<uint64_t>::max() /
                (total_bytes / chunk_bytes)) {
            failure = vbr_pinned_ring_create_failure::invalid_geometry;
            return nullptr;
        }
        const size_t n_chunks = size_t(total_bytes / chunk_bytes);
        std::unique_ptr<impl> state(new impl);
        state->chunk_size = chunk_bytes;
        state->capacity = uint64_t(n_chunks)*chunk_bytes;
        {
            std::lock_guard<std::mutex> lock(
                vbr_pinned_ring_capacity_mutex);
            if (state->capacity > VBR_PINNED_RING_MAX_BYTES -
                    vbr_pinned_ring_capacity_live) {
                failure =
                    vbr_pinned_ring_create_failure::global_capacity_exceeded;
                return nullptr;
            }
            vbr_pinned_ring_capacity_live += state->capacity;
            state->global_reserved = true;
        }

        if (accounting) {
            if (!accounting->ledger || !accounting->budget ||
                accounting->domain.residency !=
                    llama_cache_acct_residency::pinned_host) {
                failure = vbr_pinned_ring_create_failure::invalid_accounting_binding;
                return nullptr;
            }
            auto & ledger = *accounting->ledger;
            const auto before = ledger.snapshot();
            const auto existing = std::find_if(
                before.cells.begin(), before.cells.end(),
                [&](const llama_cache_acct_cell_row & row) {
                    return row.category == accounting->category &&
                           row.domain == accounting->domain;
                });
            if (existing == before.cells.end() ||
                existing->certification != llama_cache_acct_known::known) {
                failure = vbr_pinned_ring_create_failure::accounting_update_failed;
                return nullptr;
            }
            for (const auto measure : {
                    llama_cache_acct_measure::logical_payload,
                    llama_cache_acct_measure::resident_allocated,
                    llama_cache_acct_measure::reserved }) {
                const auto value = existing->cell.measures[size_t(measure)];
                if (value.state != llama_cache_acct_known::known) {
                    failure = vbr_pinned_ring_create_failure::accounting_update_failed;
                    return nullptr;
                }
                if (value.value != 0) {
                    failure = vbr_pinned_ring_create_failure::existing_ring_charge;
                    return nullptr;
                }
            }
            ledger.gauge_set(
                accounting->category, accounting->domain,
                llama_cache_acct_measure::logical_payload, 0);
            ledger.gauge_set(
                accounting->category, accounting->domain,
                llama_cache_acct_measure::resident_allocated, 0);
            const auto priced = ledger.snapshot();
            if (priced.faults_overflow != before.faults_overflow ||
                priced.faults_invalid_transition != before.faults_invalid_transition ||
                priced.faults_allocation != before.faults_allocation) {
                failure = vbr_pinned_ring_create_failure::accounting_update_failed;
                return nullptr;
            }
            llama_cache_budget_coordinator coordinator;
            if (!coordinator.reset(priced, *accounting->budget)) {
                failure = vbr_pinned_ring_create_failure::budget_reset_failed;
                return nullptr;
            }
            llama_cache_budget_plan plan;
            plan.accounting_serial = priced.serial;
            plan.entries.push_back({ accounting->domain, state->capacity, 0 });
            const auto fit = coordinator.fits(plan);
            if (fit.state != llama_cache_budget_fit_state::fits) {
                failure = fit.state == llama_cache_budget_fit_state::exceeds
                    ? vbr_pinned_ring_create_failure::budget_exceeded
                    : vbr_pinned_ring_create_failure::budget_unavailable;
                return nullptr;
            }
            state->accounting = &ledger;
            state->accounting_domain = accounting->domain;
            state->accounting_category = accounting->category;
            state->charge_fault_context = accounting->charge_fault_context;
            state->inject_charge_fault = accounting->inject_charge_fault;
        }

        state->lanes.resize(lanes.size());
        for (size_t i = 0; i < lanes.size(); ++i) {
            if ((lanes[i].device == nullptr) != (lanes[i].backend == nullptr) ||
                (lanes[i].backend &&
                 ggml_backend_get_device(lanes[i].backend) != lanes[i].device)) {
                failure = vbr_pinned_ring_create_failure::invalid_lane_binding;
                return nullptr;
            }
            if (lanes[i].device) {
                for (size_t j = 0; j < i; ++j) {
                    if (lanes[j].device == lanes[i].device) {
                        failure = vbr_pinned_ring_create_failure::duplicate_device_lane;
                        return nullptr;
                    }
                }
            }
            state->lanes[i].binding = lanes[i];
        }

        for (size_t i = 0; i < n_chunks; ++i) {
            auto & lane = state->lanes[i % state->lanes.size()];
            std::unique_ptr<impl::chunk> entry(new impl::chunk);
            if (lane.binding.device) {
                auto * host_buft =
                    ggml_backend_dev_host_buffer_type(lane.binding.device);
                if (!host_buft) {
                    failure = vbr_pinned_ring_create_failure::host_buffer_type_unavailable;
                    return nullptr;
                }
                entry->buffer = ggml_backend_buft_alloc_buffer(host_buft, chunk_bytes);
                if (!entry->buffer) {
                    failure = vbr_pinned_ring_create_failure::host_buffer_allocation_failed;
                    return nullptr;
                }
                if (ggml_backend_buffer_get_size(entry->buffer) < chunk_bytes) {
                    failure = vbr_pinned_ring_create_failure::host_buffer_too_small;
                    return nullptr;
                }
                entry->data = static_cast<uint8_t *>(
                    ggml_backend_buffer_get_base(entry->buffer));
                if (!entry->data) {
                    failure = vbr_pinned_ring_create_failure::host_buffer_base_unavailable;
                    return nullptr;
                }
                if (!lane.binding.force_synchronous) {
                    entry->event = ggml_backend_event_new(lane.binding.device);
                }
            } else {
                entry->synthetic.resize(chunk_bytes);
                entry->data = entry->synthetic.data();
            }
            lane.chunks.push_back(std::move(entry));
        }
        for (const auto & lane : state->lanes) {
            if (lane.chunks.size() < 2) {
                failure = vbr_pinned_ring_create_failure::lane_underprovisioned;
                return nullptr;
            }
        }

        if (state->accounting) {
            const auto before = state->accounting->snapshot();
            if (state->inject_charge_fault) {
                state->inject_charge_fault(state->charge_fault_context);
            }
            state->accounting->gauge_set(
                state->accounting_category, state->accounting_domain,
                llama_cache_acct_measure::logical_payload, state->capacity);
            state->accounting->gauge_set(
                state->accounting_category, state->accounting_domain,
                llama_cache_acct_measure::resident_allocated, state->capacity);
            const auto after = state->accounting->snapshot();
            if (after.faults_overflow != before.faults_overflow ||
                after.faults_invalid_transition != before.faults_invalid_transition ||
                after.faults_allocation != before.faults_allocation) {
                state->accounting->gauge_set(
                    state->accounting_category, state->accounting_domain,
                    llama_cache_acct_measure::logical_payload, 0);
                state->accounting->gauge_set(
                    state->accounting_category, state->accounting_domain,
                    llama_cache_acct_measure::resident_allocated, 0);
                failure = vbr_pinned_ring_create_failure::accounting_charge_failed;
                return nullptr;
            }
            state->ring_charged = true;
        }
        failure = vbr_pinned_ring_create_failure::none;
        return std::unique_ptr<vbr_bounded_pinned_ring_core>(
            new vbr_bounded_pinned_ring_core(std::move(state)));
    } catch (...) {
        failure = vbr_pinned_ring_create_failure::internal_error;
        return nullptr;
    }
}

uint64_t vbr_bounded_pinned_ring_core::capacity_bytes() const noexcept {
    return impl_ ? impl_->capacity : 0;
}

size_t vbr_bounded_pinned_ring_core::chunk_bytes() const noexcept {
    return impl_ ? impl_->chunk_size : 0;
}

size_t vbr_bounded_pinned_ring_core::lane_count() const noexcept {
    return impl_ ? impl_->lanes.size() : 0;
}

const vbr_pinned_ring_lane * vbr_bounded_pinned_ring_core::lane_binding(
        uint32_t lane) const noexcept {
    return impl_ && lane < impl_->lanes.size()
        ? &impl_->lanes[lane].binding : nullptr;
}

bool vbr_bounded_pinned_ring_core::accounted_to(
        const llama_cache_acct_ledger * ledger,
        const llama_cache_acct_snapshot & snapshot,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_category category) const noexcept {
    if (!impl_ || !impl_->ring_charged || impl_->accounting != ledger ||
        impl_->accounting_domain != domain ||
        impl_->accounting_category != category ||
        snapshot.completeness_manifest != llama_cache_acct_known::known) {
        return false;
    }
    const auto found = std::find_if(
        snapshot.cells.begin(), snapshot.cells.end(),
        [&](const llama_cache_acct_cell_row & row) {
            return row.category == category && row.domain == domain;
        });
    if (found == snapshot.cells.end() ||
        found->certification != llama_cache_acct_known::known) {
        return false;
    }
    const auto & logical = found->cell.measures[
        size_t(llama_cache_acct_measure::logical_payload)];
    const auto & resident = found->cell.measures[
        size_t(llama_cache_acct_measure::resident_allocated)];
    return logical.state == llama_cache_acct_known::known &&
           resident.state == llama_cache_acct_known::known &&
           logical.value == impl_->capacity &&
           resident.value == impl_->capacity;
}

vbr_pinned_ring_operation
vbr_bounded_pinned_ring_core::try_begin_operation() noexcept {
    if (!impl_) {
        return {};
    }
    try {
        return impl_->operation_mutex.try_lock()
            ? vbr_pinned_ring_operation(this)
            : vbr_pinned_ring_operation{};
    } catch (...) {
        return {};
    }
}

void vbr_bounded_pinned_ring_core::end_operation() noexcept {
    impl_->operation_mutex.unlock();
}

vbr_pinned_chunk_lease vbr_bounded_pinned_ring_core::acquire(
        uint32_t lane_index, bool & would_block) noexcept {
    would_block = false;
    vbr_pinned_chunk_lease out;
    if (!impl_ || lane_index >= impl_->lanes.size()) {
        return out;
    }
    auto & lane = impl_->lanes[lane_index];
    auto * chunk = lane.chunks[lane.next].get();
    if (chunk->busy) {
        would_block = true;
        return out;
    }
    lane.next = (lane.next + 1) % lane.chunks.size();
    out.owner_ = this;
    out.chunk_ = chunk;
    out.data_ = chunk->data;
    out.capacity_ = impl_->chunk_size;
    return out;
}

bool vbr_bounded_pinned_ring_core::submit(
        vbr_pinned_chunk_lease & lease,
        size_t valid,
        ggml_backend_t backend,
        bool & synchronous_fallback) noexcept {
    synchronous_fallback = false;
    if (lease.owner_ != this || !lease.chunk_ || valid == 0 ||
        valid > lease.capacity_) {
        return false;
    }
    auto * chunk = static_cast<impl::chunk *>(lease.chunk_);
    if (chunk->busy) {
        return false;
    }
    chunk->valid = valid;
    chunk->busy = true;
    lease.valid_ = valid;
    if (backend) {
        if (chunk->event) {
            ggml_backend_event_record(chunk->event, backend);
        } else {
            ggml_backend_synchronize(backend);
            synchronous_fallback = true;
        }
    }
    return true;
}

bool vbr_bounded_pinned_ring_core::wait(
        vbr_pinned_chunk_lease & lease,
        bool & event_completion) noexcept {
    event_completion = false;
    if (lease.owner_ != this || !lease.chunk_) {
        return false;
    }
    auto * chunk = static_cast<impl::chunk *>(lease.chunk_);
    if (!chunk->busy || chunk->valid != lease.valid_) {
        return false;
    }
    if (chunk->event) {
        ggml_backend_event_synchronize(chunk->event);
        event_completion = true;
    }
    return true;
}

void vbr_bounded_pinned_ring_core::release(
        vbr_pinned_chunk_lease & lease) noexcept {
    if (lease.owner_ == this && lease.chunk_) {
        auto * chunk = static_cast<impl::chunk *>(lease.chunk_);
        chunk->busy = false;
        chunk->valid = 0;
    }
    lease.reset();
}

uint32_t vbr_bounded_pinned_ring_core::pump(
        uint32_t lane, const pump_callbacks & callbacks,
        pump_stats & stats) noexcept {
    stats = {};
    if (!impl_ || lane >= impl_->lanes.size() || !callbacks.more ||
        !callbacks.fill || !callbacks.consume) {
        return callbacks.internal_error;
    }
    auto operation = try_begin_operation();
    if (!operation) {
        return callbacks.ring_unavailable;
    }

    struct pending_step {
        vbr_pinned_chunk_lease lease;
        uint64_t tag = 0;
        bool adapter_async = false;
    };
    std::deque<pending_step> pending;
    uint64_t live_pinned = 0;

    const auto abandon = [&](pending_step & item) noexcept {
        bool ignored = false;
        (void) wait(item.lease, ignored);
        if (callbacks.abandon) {
            callbacks.abandon(
                callbacks.context, item.tag, item.adapter_async);
        }
        release(item.lease);
    };
    const auto abandon_pending = [&]() noexcept {
        for (auto & item : pending) {
            abandon(item);
        }
        pending.clear();
    };
    const auto drain_front = [&]() noexcept -> uint32_t {
        if (pending.empty()) {
            return callbacks.ok;
        }
        auto item = std::move(pending.front());
        pending.pop_front();
        bool event_completion = false;
        if (!wait(item.lease, event_completion)) {
            release(item.lease);
            return callbacks.wait_failed;
        }
        const uint32_t consumed = callbacks.consume(
            callbacks.context, item.lease.data(), item.lease.valid(),
            item.tag, item.adapter_async, stats.chunks,
            event_completion);
        if (consumed != callbacks.ok) {
            release(item.lease);
            return consumed;
        }
        if (event_completion) {
            stats.event_completions++;
        }
        live_pinned -= item.lease.valid();
        stats.bytes += item.lease.valid();
        stats.chunks++;
        release(item.lease);
        return callbacks.ok;
    };

    try {
        while (callbacks.more(callbacks.context)) {
            bool would_block = false;
            auto lease = acquire(lane, would_block);
            if (!lease && would_block) {
                stats.backpressure_waits++;
                const uint32_t drained = drain_front();
                if (drained != callbacks.ok) {
                    abandon_pending();
                    return drained;
                }
                lease = acquire(lane, would_block);
            }
            if (!lease || would_block) {
                abandon_pending();
                return callbacks.internal_error;
            }

            pump_step step;
            const uint32_t filled = callbacks.fill(
                callbacks.context, lease.data(), lease.capacity(), step);
            if (filled != callbacks.ok) {
                if (step.adapter_async && callbacks.abandon) {
                    callbacks.abandon(
                        callbacks.context, step.tag,
                        step.adapter_async);
                }
                release(lease);
                abandon_pending();
                return filled;
            }
            if (step.valid == 0 || step.valid > lease.capacity()) {
                if (step.adapter_async && callbacks.abandon) {
                    callbacks.abandon(
                        callbacks.context, step.tag,
                        step.adapter_async);
                }
                release(lease);
                abandon_pending();
                return callbacks.internal_error;
            }

            bool synchronous_fallback = false;
            if (!submit(
                    lease, step.valid, step.backend,
                    synchronous_fallback)) {
                if (step.adapter_async && callbacks.abandon) {
                    callbacks.abandon(
                        callbacks.context, step.tag,
                        step.adapter_async);
                }
                release(lease);
                abandon_pending();
                return callbacks.submit_failed;
            }
            stats.synchronous_fallbacks +=
                uint64_t(synchronous_fallback) +
                uint64_t(step.adapter_synchronous_fallback);
            live_pinned += step.valid;
            stats.peak_pinned_bytes =
                std::max(stats.peak_pinned_bytes, live_pinned);
            pending_step current {
                std::move(lease), step.tag, step.adapter_async,
            };
            try {
                pending.push_back(std::move(current));
            } catch (...) {
                abandon(current);
                throw;
            }
        }
        while (!pending.empty()) {
            const uint32_t drained = drain_front();
            if (drained != callbacks.ok) {
                abandon_pending();
                return drained;
            }
        }
        return callbacks.ok;
    } catch (...) {
        abandon_pending();
        stats = {};
        return callbacks.internal_error;
    }
}
