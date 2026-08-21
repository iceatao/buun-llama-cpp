#include "llama-vbr-artifact-capture.h"

#include "llama-sha256.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <limits>
#include <new>
#include <tuple>
#include <utility>

namespace {

struct capture_projection_placement {
    uint64_t manifest_id = 0;
    const vbr_artifact_stream_placement * placement = nullptr;
};

struct capture_projection_cursor {
    const capture_projection_placement * source = nullptr;
    size_t cell = 0;
};

bool capture_checked_add(uint64_t a, uint64_t b, uint64_t & output) {
    if (b > UINT64_MAX - a) {
        return false;
    }
    output = a + b;
    return true;
}

bool capture_cursor_after(
        const capture_projection_cursor & lhs,
        const capture_projection_cursor & rhs) {
    const auto & lhs_cell = lhs.source->placement->cells[lhs.cell];
    const auto & rhs_cell = rhs.source->placement->cells[rhs.cell];
    return std::tie(lhs_cell.physical_cell, lhs.source->manifest_id) >
           std::tie(rhs_cell.physical_cell, rhs.source->manifest_id);
}

} // namespace

bool vbr_artifact_project_capture_union(
        const vbr_capture_projection_batch & batch,
        const vbr_capture_projection_limits & limits,
        vbr_capture_projection_plan & output) noexcept {
    output = {};
    try {
        const auto & manifests = batch.manifests;
        if (batch.source_namespace == 0 || manifests.empty() ||
            manifests.size() > limits.max_manifests ||
            limits.max_manifests == 0 || limits.max_placements == 0 ||
            limits.max_input_cells == 0 || limits.max_union_cells == 0 ||
            limits.max_segments == 0 ||
            limits.max_dependency_references == 0) {
            return false;
        }

        uint64_t placement_count = 0;
        uint64_t input_cells = 0;
        std::vector<uint64_t> manifest_ids;
        manifest_ids.reserve(manifests.size());
        std::vector<capture_projection_placement> placements;
        for (const auto & manifest : manifests) {
            if (manifest.manifest_id == 0 || manifest.placements.empty() ||
                !capture_checked_add(
                    placement_count, manifest.placements.size(),
                    placement_count) ||
                placement_count > limits.max_placements) {
                return false;
            }
            manifest_ids.push_back(manifest.manifest_id);
            std::vector<std::pair<llama_seq_id, llama_pos>> logical_positions;
            for (const auto & placement : manifest.placements) {
                if (placement.child_id == UINT32_MAX ||
                    placement.stream_index == UINT32_MAX ||
                    placement.source_sequence < 0 ||
                    placement.computation_frontier <= 0 ||
                    placement.cells.empty() ||
                    !capture_checked_add(
                        input_cells, placement.cells.size(), input_cells) ||
                    input_cells > limits.max_input_cells) {
                    return false;
                }
                placements.push_back({ manifest.manifest_id, &placement });
                for (size_t i = 0; i < placement.cells.size(); ++i) {
                    const auto & cell = placement.cells[i];
                    if (cell.physical_cell == UINT32_MAX ||
                        cell.logical_position < 0 ||
                        cell.logical_position >=
                            placement.computation_frontier ||
                        (i != 0 &&
                         placement.cells[i - 1].physical_cell >=
                             cell.physical_cell)) {
                        return false;
                    }
                    logical_positions.push_back({
                        placement.source_sequence,
                        cell.logical_position,
                    });
                }
            }
            std::sort(logical_positions.begin(), logical_positions.end());
            if (std::adjacent_find(
                    logical_positions.begin(), logical_positions.end()) !=
                    logical_positions.end()) {
                return false;
            }
        }
        std::sort(manifest_ids.begin(), manifest_ids.end());
        if (std::adjacent_find(manifest_ids.begin(), manifest_ids.end()) !=
                manifest_ids.end()) {
            return false;
        }
        std::sort(placements.begin(), placements.end(),
            [](const auto & lhs, const auto & rhs) {
                return std::tie(lhs.placement->child_id,
                                lhs.placement->stream_index,
                                lhs.manifest_id) <
                       std::tie(rhs.placement->child_id,
                                rhs.placement->stream_index,
                                rhs.manifest_id);
            });
        if (std::adjacent_find(
                placements.begin(), placements.end(),
                [](const auto & lhs, const auto & rhs) {
                    return lhs.manifest_id == rhs.manifest_id &&
                           lhs.placement->child_id ==
                               rhs.placement->child_id &&
                           lhs.placement->stream_index ==
                               rhs.placement->stream_index;
                }) != placements.end()) {
            return false;
        }

        vbr_capture_projection_plan plan;
        plan.source_namespace = batch.source_namespace;
        plan.manifest_count = uint32_t(manifests.size());
        plan.placement_count = uint32_t(placement_count);
        plan.input_cell_references = input_cells;
        uint64_t segment_count = 0;
        std::vector<capture_projection_cursor> heap;
        std::vector<uint64_t> dependencies;
        for (size_t group_begin = 0; group_begin < placements.size();) {
            size_t group_end = group_begin + 1;
            while (group_end < placements.size() &&
                   placements[group_end].placement->child_id ==
                       placements[group_begin].placement->child_id &&
                   placements[group_end].placement->stream_index ==
                       placements[group_begin].placement->stream_index) {
                ++group_end;
            }
            plan.streams.push_back({
                placements[group_begin].placement->child_id,
                placements[group_begin].placement->stream_index,
                {},
            });
            auto & stream = plan.streams.back();
            heap.clear();
            dependencies.clear();
            heap.reserve(group_end - group_begin);
            dependencies.reserve(group_end - group_begin);
            for (size_t i = group_begin; i < group_end; ++i) {
                heap.push_back({ &placements[i], 0 });
            }
            std::make_heap(heap.begin(), heap.end(), capture_cursor_after);

            while (!heap.empty()) {
                const uint32_t cell =
                    heap.front().source->placement->cells[
                        heap.front().cell].physical_cell;
                dependencies.clear();
                while (!heap.empty() &&
                       heap.front().source->placement->cells[
                           heap.front().cell].physical_cell == cell) {
                    std::pop_heap(
                        heap.begin(), heap.end(), capture_cursor_after);
                    auto cursor = heap.back();
                    heap.pop_back();
                    if (!dependencies.empty() &&
                        dependencies.back() == cursor.source->manifest_id) {
                        return false;
                    }
                    dependencies.push_back(cursor.source->manifest_id);
                    ++cursor.cell;
                    if (cursor.cell <
                            cursor.source->placement->cells.size()) {
                        heap.push_back(cursor);
                        std::push_heap(
                            heap.begin(), heap.end(), capture_cursor_after);
                    }
                }

                if (plan.union_cell_count == limits.max_union_cells) {
                    return false;
                }
                const auto dependencies_equal = [&]() {
                    if (stream.segments.empty()) {
                        return false;
                    }
                    const auto & prior = stream.segments.back();
                    return prior.dependency_count == dependencies.size() &&
                        std::equal(
                            dependencies.begin(), dependencies.end(),
                            plan.dependent_manifest_ids.begin() +
                                prior.first_dependency);
                };
                const bool extend = !stream.segments.empty() &&
                    uint64_t(stream.segments.back().first_physical_cell) +
                        stream.segments.back().cell_count == cell &&
                    dependencies_equal();
                if (extend) {
                    if (stream.segments.back().cell_count == UINT32_MAX) {
                        return false;
                    }
                    ++stream.segments.back().cell_count;
                } else {
                    if (segment_count == limits.max_segments ||
                        plan.dependency_references >
                            limits.max_dependency_references ||
                        dependencies.size() >
                            limits.max_dependency_references -
                                plan.dependency_references ||
                        plan.dependent_manifest_ids.size() > UINT32_MAX ||
                        dependencies.size() > UINT32_MAX) {
                        return false;
                    }
                    const uint32_t first_dependency = uint32_t(
                        plan.dependent_manifest_ids.size());
                    plan.dependent_manifest_ids.insert(
                        plan.dependent_manifest_ids.end(),
                        dependencies.begin(), dependencies.end());
                    ++segment_count;
                    plan.dependency_references += dependencies.size();
                    stream.segments.push_back({
                        cell, 1, first_dependency,
                        uint32_t(dependencies.size()),
                    });
                }
                ++plan.union_cell_count;
            }
            group_begin = group_end;
        }
        if (plan.streams.empty()) {
            return false;
        }
        output = std::move(plan);
        return true;
    } catch (...) {
        output = {};
        return false;
    }
}

const char * vbr_capture_stream_status_name(
        vbr_capture_stream_status status) noexcept {
    switch (status) {
        case vbr_capture_stream_status::ok:                  return "ok";
        case vbr_capture_stream_status::invalid_argument:    return "invalid_argument";
        case vbr_capture_stream_status::ring_unavailable:    return "ring_unavailable";
        case vbr_capture_stream_status::transfer_failed:     return "transfer_failed";
        case vbr_capture_stream_status::short_read:          return "short_read";
        case vbr_capture_stream_status::duplicate_segment:   return "duplicate_segment";
        case vbr_capture_stream_status::missing_segment:     return "missing_segment";
        case vbr_capture_stream_status::late_segment:        return "late_segment";
        case vbr_capture_stream_status::hash_mismatch:       return "hash_mismatch";
        case vbr_capture_stream_status::format_rejected:      return "format_rejected";
        case vbr_capture_stream_status::accounting_unavailable: return "accounting_unavailable";
        case vbr_capture_stream_status::accounting_refused:  return "accounting_refused";
        case vbr_capture_stream_status::stage_failed:         return "stage_failed";
        case vbr_capture_stream_status::commit_failed:        return "commit_failed";
        case vbr_capture_stream_status::publication_failed:  return "publication_failed";
        case vbr_capture_stream_status::internal_error:      return "internal_error";
        case vbr_capture_stream_status::_count:              break;
    }
    return "invalid";
}

const char * vbr_capture_reservation_group_name(
        vbr_capture_reservation_group group) noexcept {
    switch (group) {
        case vbr_capture_reservation_group::none: return "none";
        case vbr_capture_reservation_group::transfer_staging: return "transfer_staging";
        case vbr_capture_reservation_group::durable_artifact: return "durable_artifact";
        case vbr_capture_reservation_group::_count: break;
    }
    return "invalid";
}

const char * vbr_capture_ring_create_failure_name(
        vbr_capture_ring_create_failure failure) noexcept {
    switch (failure) {
        case vbr_capture_ring_create_failure::none:
            return "none";
        case vbr_capture_ring_create_failure::invalid_geometry:
            return "invalid_geometry";
        case vbr_capture_ring_create_failure::invalid_accounting_binding:
            return "invalid_accounting_binding";
        case vbr_capture_ring_create_failure::existing_ring_charge:
            return "existing_ring_charge";
        case vbr_capture_ring_create_failure::accounting_update_failed:
            return "accounting_update_failed";
        case vbr_capture_ring_create_failure::budget_reset_failed:
            return "budget_reset_failed";
        case vbr_capture_ring_create_failure::budget_unavailable:
            return "budget_unavailable";
        case vbr_capture_ring_create_failure::budget_exceeded:
            return "budget_exceeded";
        case vbr_capture_ring_create_failure::global_capacity_exceeded:
            return "global_capacity_exceeded";
        case vbr_capture_ring_create_failure::invalid_lane_binding:
            return "invalid_lane_binding";
        case vbr_capture_ring_create_failure::duplicate_device_lane:
            return "duplicate_device_lane";
        case vbr_capture_ring_create_failure::host_buffer_type_unavailable:
            return "host_buffer_type_unavailable";
        case vbr_capture_ring_create_failure::host_buffer_allocation_failed:
            return "host_buffer_allocation_failed";
        case vbr_capture_ring_create_failure::host_buffer_too_small:
            return "host_buffer_too_small";
        case vbr_capture_ring_create_failure::host_buffer_base_unavailable:
            return "host_buffer_base_unavailable";
        case vbr_capture_ring_create_failure::lane_underprovisioned:
            return "lane_underprovisioned";
        case vbr_capture_ring_create_failure::accounting_charge_failed:
            return "accounting_charge_failed";
        case vbr_capture_ring_create_failure::internal_error:
            return "internal_error";
        case vbr_capture_ring_create_failure::_count:
            break;
    }
    return "invalid";
}

namespace {

vbr_capture_stream_status capture_status_for_ring_failure(
        vbr_capture_ring_create_failure failure) noexcept {
    switch (failure) {
        case vbr_capture_ring_create_failure::budget_exceeded:
            return vbr_capture_stream_status::accounting_refused;
        case vbr_capture_ring_create_failure::invalid_accounting_binding:
        case vbr_capture_ring_create_failure::existing_ring_charge:
        case vbr_capture_ring_create_failure::accounting_update_failed:
        case vbr_capture_ring_create_failure::budget_reset_failed:
        case vbr_capture_ring_create_failure::budget_unavailable:
        case vbr_capture_ring_create_failure::accounting_charge_failed:
            return vbr_capture_stream_status::accounting_unavailable;
        case vbr_capture_ring_create_failure::internal_error:
            return vbr_capture_stream_status::internal_error;
        case vbr_capture_ring_create_failure::none:
        case vbr_capture_ring_create_failure::invalid_geometry:
        case vbr_capture_ring_create_failure::global_capacity_exceeded:
        case vbr_capture_ring_create_failure::invalid_lane_binding:
        case vbr_capture_ring_create_failure::duplicate_device_lane:
        case vbr_capture_ring_create_failure::host_buffer_type_unavailable:
        case vbr_capture_ring_create_failure::host_buffer_allocation_failed:
        case vbr_capture_ring_create_failure::host_buffer_too_small:
        case vbr_capture_ring_create_failure::host_buffer_base_unavailable:
        case vbr_capture_ring_create_failure::lane_underprovisioned:
        case vbr_capture_ring_create_failure::_count:
            return vbr_capture_stream_status::ring_unavailable;
    }
    return vbr_capture_stream_status::ring_unavailable;
}

} // namespace

struct artifact_segment_chain::impl {
    std::vector<artifact_segment> segments;
    uint64_t total = 0;
    size_t max_segment = 0;
};

artifact_segment_chain::artifact_segment_chain()
    : impl_(new impl) {}
artifact_segment_chain::~artifact_segment_chain() = default;
artifact_segment_chain::artifact_segment_chain(
        artifact_segment_chain &&) noexcept = default;
artifact_segment_chain & artifact_segment_chain::operator=(
        artifact_segment_chain &&) noexcept = default;

bool artifact_segment_chain::append(
        const uint8_t * data, size_t size) noexcept {
    try {
        if ((!data && size != 0) ||
            size > std::numeric_limits<uint64_t>::max() -
                impl_->total) {
            return false;
        }
        auto bytes =
            std::make_shared<std::vector<uint8_t>>();
        if (size != 0) {
            bytes->assign(data, data + size);
        }
        impl_->segments.push_back({
            std::move(bytes), 0, uint64_t(size),
        });
        impl_->total += size;
        impl_->max_segment =
            std::max(impl_->max_segment, size);
        return true;
    } catch (...) {
        return false;
    }
}

uint64_t artifact_segment_chain::size() const noexcept {
    return impl_->total;
}

size_t artifact_segment_chain::segment_count() const noexcept {
    return impl_->segments.size();
}

size_t artifact_segment_chain::max_segment_size() const noexcept {
    return impl_->max_segment;
}

bool artifact_segment_chain::read(
        uint64_t offset, uint8_t * destination,
        size_t size) const noexcept {
    if ((!destination && size != 0) ||
        offset > impl_->total ||
        size > impl_->total - offset) {
        return false;
    }
    uint64_t cursor = 0;
    size_t remaining = size;
    for (const auto & segment : impl_->segments) {
        const uint64_t end = cursor + segment.length;
        if (offset >= end) {
            cursor = end;
            continue;
        }
        const uint64_t within = offset > cursor
            ? offset - cursor : 0;
        const size_t available =
            size_t(segment.length - within);
        const size_t take = std::min(available, remaining);
        if (take != 0) {
            if (!segment.storage ||
                segment.offset + within >
                    segment.storage->size() ||
                take > segment.storage->size() -
                    size_t(segment.offset + within)) {
                return false;
            }
            std::memcpy(
                destination + (size - remaining),
                segment.storage->data() +
                    size_t(segment.offset + within),
                take);
            remaining -= take;
            offset += take;
            if (remaining == 0) {
                return true;
            }
        }
        cursor = end;
    }
    return remaining == 0;
}

namespace {

bool segment_source_read(
        const void * context, uint64_t offset,
        uint8_t * destination, size_t size) noexcept {
    const auto * chain =
        static_cast<const artifact_segment_chain *>(context);
    return chain &&
           chain->read(offset, destination, size);
}

} // namespace

vbr_artifact_byte_source artifact_segment_chain::source() const noexcept {
    return { size(), this, segment_source_read };
}

std::array<uint8_t, 32> vbr_capture_stream_digest(
        const artifact_segment_chain & chain) noexcept {
    static constexpr char domain_label[] =
        "buun.vbr.capture.segment-stream";
    llama_sha256_writer hash;
    hash.string(domain_label, sizeof(domain_label) - 1);
    hash.u64(chain.size());
    std::array<uint8_t, 64*1024> scratch;
    for (uint64_t offset = 0; offset < chain.size();) {
        const size_t count = size_t(std::min<uint64_t>(
            scratch.size(), chain.size() - offset));
        if (!chain.read(offset, scratch.data(), count)) {
            return {};
        }
        hash.bytes(scratch.data(), count);
        offset += count;
    }
    return hash.finish();
}

struct vbr_pinned_chunk_ring::impl {
    std::unique_ptr<vbr_bounded_pinned_ring_core> core;
};

vbr_pinned_chunk_ring::vbr_pinned_chunk_ring(
        std::unique_ptr<impl> state) noexcept
    : impl_(std::move(state)) {}

vbr_pinned_chunk_ring::~vbr_pinned_chunk_ring() = default;

std::unique_ptr<vbr_pinned_chunk_ring>
vbr_pinned_chunk_ring::create(
        const std::vector<vbr_capture_lane> & lanes,
        uint64_t total_bytes,
        size_t chunk_bytes,
        vbr_capture_stream_status & status,
        const vbr_capture_ring_accounting * accounting,
        vbr_capture_ring_create_failure * failure) noexcept {
    status = vbr_capture_stream_status::ring_unavailable;
    vbr_capture_ring_create_failure reason =
        vbr_capture_ring_create_failure::none;
    try {
        std::unique_ptr<impl> state(new impl);
        state->core = vbr_bounded_pinned_ring_core::create(
            lanes, total_bytes, chunk_bytes, accounting, reason);
        if (!state->core) {
            status = capture_status_for_ring_failure(reason);
            if (failure) {
                *failure = reason;
            }
            return nullptr;
        }
        status = vbr_capture_stream_status::ok;
        if (failure) {
            *failure = reason;
        }
        return std::unique_ptr<vbr_pinned_chunk_ring>(
            new vbr_pinned_chunk_ring(std::move(state)));
    } catch (...) {
        status = vbr_capture_stream_status::internal_error;
        if (failure) {
            *failure = vbr_capture_ring_create_failure::internal_error;
        }
        return nullptr;
    }
}

uint64_t vbr_pinned_chunk_ring::capacity_bytes() const noexcept {
    return impl_ && impl_->core ? impl_->core->capacity_bytes() : 0;
}

size_t vbr_pinned_chunk_ring::chunk_bytes() const noexcept {
    return impl_ && impl_->core ? impl_->core->chunk_bytes() : 0;
}

size_t vbr_pinned_chunk_ring::lane_count() const noexcept {
    return impl_ && impl_->core ? impl_->core->lane_count() : 0;
}

vbr_capture_stream_status vbr_pinned_chunk_ring::stream(
        const vbr_capture_stream_source & source,
        artifact_segment_chain & destination,
        vbr_capture_stream_stats & stats) noexcept {
    stats = {};
    if (source.lane >= impl_->core->lane_count() || source.size == 0) {
        return vbr_capture_stream_status::invalid_argument;
    }
    const auto * lane = impl_->core->lane_binding(source.lane);
    const bool tensor_source = source.tensor != nullptr;
    if (tensor_source) {
        // The store may be constructed before VBR lazily creates its dedicated
        // side-stream backend. Events and pinned buffers are device-scoped, so
        // bind the lane to the physical device rather than one backend handle.
        if (!source.backend || !source.device ||
            !lane || source.device != lane->device ||
            ggml_backend_get_device(source.backend) !=
                lane->device ||
            source.tensor_offset >
                std::numeric_limits<uint64_t>::max() -
                    source.size ||
            source.tensor_offset > ggml_nbytes(source.tensor) ||
            source.size >
                ggml_nbytes(source.tensor) -
                    source.tensor_offset) {
            return vbr_capture_stream_status::invalid_argument;
        }
    } else if (!source.read) {
        return vbr_capture_stream_status::invalid_argument;
    }
    const size_t chunk_size = impl_->core->chunk_bytes();

    std::deque<vbr_pinned_chunk_lease> pending;
    llama_sha256_writer hash;
    static constexpr char domain_label[] =
        "buun.vbr.capture.segment-stream";
    hash.string(domain_label, sizeof(domain_label) - 1);
    hash.u64(source.size);

    const auto synchronize_only = [&]() noexcept {
        for (auto & entry : pending) {
            bool event_completion = false;
            impl_->core->wait(entry, event_completion);
            impl_->core->release(entry);
        }
        pending.clear();
    };
    const auto drain_front = [&]() -> vbr_capture_stream_status {
        if (pending.empty()) {
            return vbr_capture_stream_status::ok;
        }
        auto entry = std::move(pending.front());
        pending.pop_front();
        bool event_completion = false;
        if (!impl_->core->wait(entry, event_completion)) {
            impl_->core->release(entry);
            return vbr_capture_stream_status::internal_error;
        }
        if (event_completion) {
            stats.event_completions++;
        }
        // KNOWN LIMITATION: ggml's asynchronous copy/event APIs return no
        // transfer result. The synthetic seam can report transfer_failed,
        // while a real device error can only surface later as a length or
        // digest mismatch; the F3.2 hardware gate must account for that.
        if (stats.chunks == source.fail_completion_at) {
            impl_->core->release(entry);
            return vbr_capture_stream_status::transfer_failed;
        }
        if (!destination.append(entry.data(), entry.valid())) {
            impl_->core->release(entry);
            return vbr_capture_stream_status::internal_error;
        }
        hash.bytes(entry.data(), entry.valid());
        stats.bytes += entry.valid();
        stats.chunks++;
        impl_->core->release(entry);
        return vbr_capture_stream_status::ok;
    };

    // TODO(F4.2a follow-up): lift shared drive(fill,consume) pump into the core.
    try {
        uint64_t offset = 0;
        while (offset < source.size) {
            bool would_block = false;
            auto entry = impl_->core->acquire(source.lane, would_block);
            if (!entry && would_block) {
                stats.backpressure_waits++;
                const auto drained = drain_front();
                if (drained != vbr_capture_stream_status::ok) {
                    synchronize_only();
                    return drained;
                }
                entry = impl_->core->acquire(source.lane, would_block);
            }
            if (!entry || would_block) {
                synchronize_only();
                return vbr_capture_stream_status::internal_error;
            }
            const size_t count = size_t(std::min<uint64_t>(
                chunk_size, source.size - offset));
            if (tensor_source) {
                ggml_backend_tensor_get_async(
                    source.backend, source.tensor,
                    entry.data(),
                    size_t(source.tensor_offset + offset),
                    count);
            } else if (!source.read(
                           source.context, offset,
                           entry.data(), count)) {
                impl_->core->release(entry);
                synchronize_only();
                return vbr_capture_stream_status::short_read;
            }
            bool synchronous_fallback = false;
            if (!impl_->core->submit(
                    entry, count,
                    tensor_source ? source.backend : nullptr,
                    synchronous_fallback)) {
                impl_->core->release(entry);
                synchronize_only();
                return vbr_capture_stream_status::internal_error;
            }
            if (synchronous_fallback) {
                stats.synchronous_fallbacks++;
            }
            pending.push_back(std::move(entry));
            offset += count;
        }
        while (!pending.empty()) {
            const auto drained = drain_front();
            if (drained != vbr_capture_stream_status::ok) {
                synchronize_only();
                return drained;
            }
        }
        if (stats.bytes != source.size) {
            return vbr_capture_stream_status::short_read;
        }
        stats.max_segment_size =
            destination.max_segment_size();
        stats.streaming_digest = hash.finish();
        return vbr_capture_stream_status::ok;
    } catch (...) {
        synchronize_only();
        stats = {};
        return vbr_capture_stream_status::internal_error;
    }
}
