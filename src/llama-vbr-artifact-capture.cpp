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

vbr_capture_projection::vbr_capture_projection(
        std::shared_ptr<const vbr_capture_projection_plan> plan) noexcept
    : plan_(std::move(plan)) {}

const vbr_capture_projection_plan *
vbr_capture_projection::operator->() const noexcept {
    return plan_.get();
}

const vbr_capture_projection_plan &
vbr_capture_projection::operator*() const noexcept {
    return *plan_;
}

vbr_capture_projection::operator bool() const noexcept {
    return bool(plan_);
}

bool vbr_capture_projection::operator==(
        const vbr_capture_projection & other) const noexcept {
    return plan_ == other.plan_;
}

bool vbr_artifact_project_capture_union(
        const vbr_capture_projection_batch & batch,
        const vbr_capture_projection_limits & limits,
        vbr_capture_projection & output) noexcept {
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
        output = vbr_capture_projection(
            std::make_shared<const vbr_capture_projection_plan>(
                std::move(plan)));
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
        case vbr_capture_stream_status::projection_invalid:   return "projection_invalid";
        case vbr_capture_stream_status::snapshot_unavailable: return "snapshot_unavailable";
        case vbr_capture_stream_status::snapshot_changed:     return "snapshot_changed";
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
    std::vector<uint64_t> segment_ends;
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
        if (impl_->segments.size() == impl_->segments.capacity()) {
            const size_t next = impl_->segments.empty() ? 1 :
                impl_->segments.size() <= SIZE_MAX/2 ?
                    impl_->segments.size()*2 : SIZE_MAX;
            if (next == SIZE_MAX) {
                return false;
            }
            impl_->segments.reserve(next);
        }
        if (impl_->segment_ends.size() == impl_->segment_ends.capacity()) {
            const size_t next = impl_->segment_ends.empty() ? 1 :
                impl_->segment_ends.size() <= SIZE_MAX/2 ?
                    impl_->segment_ends.size()*2 : SIZE_MAX;
            if (next == SIZE_MAX) {
                return false;
            }
            impl_->segment_ends.reserve(next);
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
        impl_->segment_ends.push_back(impl_->total);
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
    if (size == 0) {
        return true;
    }
    const auto first = std::upper_bound(
        impl_->segment_ends.begin(), impl_->segment_ends.end(), offset);
    size_t segment_index = size_t(first - impl_->segment_ends.begin());
    uint64_t cursor = segment_index == 0 ? 0 :
        impl_->segment_ends[segment_index - 1];
    size_t remaining = size;
    for (; segment_index < impl_->segments.size(); ++segment_index) {
        const auto & segment = impl_->segments[segment_index];
        const uint64_t end = impl_->segment_ends[segment_index];
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
    const vbr_capture_stream_range range { 0, source.size };
    return stream_ranges_impl(source, &range, 1, destination, stats);
}

vbr_capture_stream_status vbr_pinned_chunk_ring::stream_ranges(
        const vbr_capture_stream_source & source,
        const std::vector<vbr_capture_stream_range> & ranges,
        artifact_segment_chain & destination,
        vbr_capture_stream_stats & stats) noexcept {
    return stream_ranges_impl(
        source, ranges.data(), ranges.size(), destination, stats);
}

vbr_capture_stream_status vbr_pinned_chunk_ring::stream_ranges_impl(
        const vbr_capture_stream_source & source,
        const vbr_capture_stream_range * ranges,
        size_t range_count,
        artifact_segment_chain & destination,
        vbr_capture_stream_stats & stats) noexcept {
    stats = {};
    if (source.lane >= impl_->core->lane_count() || source.size == 0 ||
        ranges == nullptr || range_count == 0 || destination.size() != 0) {
        return vbr_capture_stream_status::invalid_argument;
    }
    uint64_t transfer_bytes = 0;
    uint64_t prior_end = 0;
    for (size_t i = 0; i < range_count; ++i) {
        const auto & range = ranges[i];
        if (range.size == 0 || range.source_offset < prior_end ||
            range.source_offset > source.size ||
            range.size > source.size - range.source_offset ||
            !capture_checked_add(
                transfer_bytes, range.size, transfer_bytes)) {
            return vbr_capture_stream_status::invalid_argument;
        }
        prior_end = range.source_offset + range.size;
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
    hash.u64(transfer_bytes);

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
        size_t range_index = 0;
        uint64_t range_offset = 0;
        while (range_index < range_count) {
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
            size_t filled = 0;
            while (filled < chunk_size && range_index < range_count) {
                const auto & range = ranges[range_index];
                const size_t count = size_t(std::min<uint64_t>(
                    chunk_size - filled, range.size - range_offset));
                const uint64_t source_offset =
                    range.source_offset + range_offset;
                if (tensor_source) {
                    // Multiple gets are queued on the same backend stream;
                    // submit records one completion event after the complete
                    // packed chunk rather than one event per logical range.
                    ggml_backend_tensor_get_async(
                        source.backend, source.tensor,
                        entry.data() + filled,
                        size_t(source.tensor_offset + source_offset),
                        count);
                } else if (!source.read(
                               source.context, source_offset,
                               entry.data() + filled, count)) {
                    impl_->core->release(entry);
                    synchronize_only();
                    return vbr_capture_stream_status::short_read;
                }
                filled += count;
                range_offset += count;
                if (range_offset == range.size) {
                    ++range_index;
                    range_offset = 0;
                }
            }
            bool synchronous_fallback = false;
            if (!impl_->core->submit(
                    entry, filled,
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
        }
        while (!pending.empty()) {
            const auto drained = drain_front();
            if (drained != vbr_capture_stream_status::ok) {
                synchronize_only();
                return drained;
            }
        }
        if (stats.bytes != transfer_bytes) {
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

bool vbr_capture_projected_shard_topology(
        const std::vector<vbr_capture_projected_shard_source> & sources,
        uint32_t & shard_count,
        std::array<uint8_t, 32> & digest) noexcept {
    shard_count = 0;
    digest = {};
    try {
        if (sources.empty() || sources.size() > UINT32_MAX) {
            return false;
        }
        std::vector<const vbr_capture_projected_shard_source *> ordered;
        ordered.reserve(sources.size());
        for (const auto & source : sources) {
            ordered.push_back(&source);
        }
        std::sort(ordered.begin(), ordered.end(),
            [](const auto * lhs, const auto * rhs) {
                return lhs->shard_index < rhs->shard_index;
            });
        llama_sha256_writer hash;
        static constexpr char DOMAIN[] =
            "buun.vbr.capture/projected-shard-topology";
        hash.string(DOMAIN, sizeof(DOMAIN) - 1);
        hash.u32(uint32_t(ordered.size()));
        for (uint32_t i = 0; i < ordered.size(); ++i) {
            const auto & source = *ordered[i];
            if (source.shard_index != i || source.source_identity == 0 ||
                source.row_count == 0 || source.row_bytes == 0 ||
                source.source.size == 0) {
                return false;
            }
            hash.u32(source.shard_index);
            hash.u32(source.row_count);
            hash.u64(source.row_bytes);
            hash.u64(source.source_identity);
            hash.u64(source.source.size);
            hash.u32(source.source.lane);
            // This digest is deliberately process-local: bind the exact
            // provider-issued byte-source capability as well as its stable
            // identity so an accidental callback/tensor substitution cannot
            // reuse otherwise identical geometry.
            hash.bytes(&source.source.context,
                       sizeof(source.source.context));
            hash.bytes(&source.source.read,
                       sizeof(source.source.read));
            hash.bytes(&source.source.backend,
                       sizeof(source.source.backend));
            hash.bytes(&source.source.device,
                       sizeof(source.source.device));
            hash.bytes(&source.source.tensor,
                       sizeof(source.source.tensor));
            hash.u64(source.source.tensor_offset);
        }
        shard_count = uint32_t(ordered.size());
        digest = hash.finish();
        return std::any_of(
            digest.begin(), digest.end(), [](uint8_t value) {
                return value != 0;
            });
    } catch (...) {
        shard_count = 0;
        digest = {};
        return false;
    }
}

namespace {

bool projected_snapshot_valid(
        const vbr_capture_unit_snapshot & snapshot) noexcept {
    const auto & generation = snapshot.generation;
    return snapshot.source_namespace != 0 &&
           snapshot.child_id != UINT32_MAX &&
           snapshot.logical_unit_id != UINT32_MAX &&
           snapshot.controller_generation != 0 &&
           (snapshot.mutation_serial & 1u) == 0 &&
           generation.repr_gen != 0 &&
           (generation.publish_seq & 1u) == 0 &&
           generation.current_type >= 0 &&
           generation.current_type < GGML_TYPE_COUNT &&
           generation.last_source_type >= 0 &&
           generation.last_source_type < GGML_TYPE_COUNT &&
           generation.domain <= vbr_repr_domain::tapped &&
           generation.last_transition <=
               vbr_repr_transition::recovery_invalidate &&
           generation.flags == 0 && snapshot.shard_count != 0 &&
           std::any_of(
               snapshot.shard_topology_digest.begin(),
               snapshot.shard_topology_digest.end(),
               [](uint8_t value) { return value != 0; });
}

} // namespace

vbr_capture_stream_status vbr_capture_projected_unit_transfer(
        vbr_capture_projection projection,
        uint32_t child_id,
        uint32_t stream_index,
        uint32_t logical_unit_id,
        const std::vector<vbr_capture_projected_shard_source> & sources,
        const vbr_capture_projected_transfer_limits & limits,
        const vbr_capture_unit_snapshot_provider & snapshots,
        vbr_pinned_chunk_ring & ring,
        vbr_capture_projected_unit & output) noexcept {
    output = {};
    try {
        if (!projection || projection->source_namespace == 0 ||
            child_id == UINT32_MAX || stream_index == UINT32_MAX ||
            logical_unit_id == UINT32_MAX || sources.empty() ||
            limits.max_shards == 0 ||
            sources.size() > limits.max_shards ||
            limits.max_shard_segment_references == 0 ||
            limits.max_source_operations == 0 ||
            limits.max_total_packed_bytes == 0 ||
            !snapshots.acquire || !snapshots.recheck ||
            !snapshots.release ||
            projection->dependency_references !=
                projection->dependent_manifest_ids.size()) {
            return vbr_capture_stream_status::projection_invalid;
        }
        const vbr_capture_projection_stream * selected = nullptr;
        for (const auto & stream : projection->streams) {
            if (stream.child_id == child_id &&
                stream.stream_index == stream_index) {
                if (selected != nullptr) {
                    return vbr_capture_stream_status::projection_invalid;
                }
                selected = &stream;
            }
        }
        if (!selected || selected->segments.empty() ||
            selected->segments.size() > UINT32_MAX) {
            return vbr_capture_stream_status::projection_invalid;
        }

        uint64_t prior_end = 0;
        for (const auto & segment : selected->segments) {
            if (segment.cell_count == 0 ||
                segment.first_dependency >
                    projection->dependent_manifest_ids.size() ||
                segment.dependency_count == 0 ||
                segment.dependency_count >
                    projection->dependent_manifest_ids.size() -
                        segment.first_dependency ||
                uint64_t(segment.first_physical_cell) < prior_end) {
                return vbr_capture_stream_status::projection_invalid;
            }
            const uint64_t end =
                uint64_t(segment.first_physical_cell) +
                segment.cell_count;
            if (end > uint64_t(UINT32_MAX) + 1) {
                return vbr_capture_stream_status::projection_invalid;
            }
            const auto dependency_begin =
                projection->dependent_manifest_ids.begin() +
                segment.first_dependency;
            const auto dependency_end =
                dependency_begin + segment.dependency_count;
            if (*dependency_begin == 0 ||
                std::adjacent_find(
                    dependency_begin, dependency_end) != dependency_end ||
                !std::is_sorted(dependency_begin, dependency_end)) {
                return vbr_capture_stream_status::projection_invalid;
            }
            prior_end = end;
        }

        std::vector<const vbr_capture_projected_shard_source *> ordered;
        ordered.reserve(sources.size());
        for (const auto & source : sources) {
            ordered.push_back(&source);
        }
        std::sort(ordered.begin(), ordered.end(),
            [](const auto * lhs, const auto * rhs) {
                return lhs->shard_index < rhs->shard_index;
            });
        uint32_t topology_count = 0;
        std::array<uint8_t, 32> topology_digest = {};
        if (!vbr_capture_projected_shard_topology(
                sources, topology_count, topology_digest) ||
            topology_count != ordered.size() ||
            selected->segments.size() >
                limits.max_shard_segment_references/sources.size()) {
            return vbr_capture_stream_status::projection_invalid;
        }

        std::vector<vbr_capture_projected_slice> slices;
        slices.reserve(selected->segments.size());
        struct cell_range { uint32_t first = 0; uint32_t count = 0; };
        std::vector<cell_range> cell_ranges;
        cell_ranges.reserve(selected->segments.size());
        uint64_t packed_rows = 0;
        for (uint32_t i = 0; i < selected->segments.size(); ++i) {
            const auto & segment = selected->segments[i];
            slices.push_back({ i, packed_rows, segment.cell_count });
            packed_rows += segment.cell_count;
            if (!cell_ranges.empty() &&
                uint64_t(cell_ranges.back().first) +
                        cell_ranges.back().count ==
                    segment.first_physical_cell &&
                segment.cell_count <=
                    UINT32_MAX - cell_ranges.back().count) {
                cell_ranges.back().count += segment.cell_count;
            } else {
                cell_ranges.push_back({
                    segment.first_physical_cell, segment.cell_count,
                });
            }
        }
        std::vector<uint64_t> shard_packed_bytes;
        shard_packed_bytes.reserve(ordered.size());
        uint64_t total_packed_bytes = 0;
        uint64_t source_operations = 0;
        const uint64_t chunk_bytes = ring.chunk_bytes();
        if (chunk_bytes == 0) {
            return vbr_capture_stream_status::projection_invalid;
        }
        for (const auto * source : ordered) {
            if (source->row_count == 0 || source->row_bytes == 0 ||
                source->row_count > UINT64_MAX/source->row_bytes ||
                uint64_t(source->row_count)*source->row_bytes >
                    source->source.size) {
                return vbr_capture_stream_status::projection_invalid;
            }
            for (const auto & segment : selected->segments) {
                const uint64_t end =
                    uint64_t(segment.first_physical_cell) +
                    segment.cell_count;
                if (end > source->row_count ||
                    segment.first_physical_cell >
                        UINT64_MAX/source->row_bytes ||
                    segment.cell_count >
                        UINT64_MAX/source->row_bytes) {
                    return vbr_capture_stream_status::projection_invalid;
                }
                if (uint64_t(segment.first_physical_cell)*
                            source->row_bytes > source->source.size ||
                    uint64_t(segment.cell_count)*source->row_bytes >
                        source->source.size -
                            uint64_t(segment.first_physical_cell)*
                                source->row_bytes) {
                    return vbr_capture_stream_status::projection_invalid;
                }
            }
            if (packed_rows > UINT64_MAX/source->row_bytes) {
                return vbr_capture_stream_status::projection_invalid;
            }
            const uint64_t packed_bytes = packed_rows*source->row_bytes;
            if (packed_bytes == 0 ||
                packed_bytes > limits.max_total_packed_bytes -
                    total_packed_bytes) {
                return vbr_capture_stream_status::projection_invalid;
            }
            total_packed_bytes += packed_bytes;
            shard_packed_bytes.push_back(packed_bytes);

            uint64_t packed_cursor = 0;
            for (const auto & range : cell_ranges) {
                const uint64_t bytes =
                    uint64_t(range.count)*source->row_bytes;
                const uint64_t first_capacity =
                    chunk_bytes - packed_cursor%chunk_bytes;
                uint64_t operations = 1;
                if (bytes > first_capacity) {
                    const uint64_t remaining = bytes - first_capacity;
                    operations += remaining/chunk_bytes;
                    operations += remaining%chunk_bytes != 0;
                }
                if (operations >
                        limits.max_source_operations - source_operations) {
                    return vbr_capture_stream_status::projection_invalid;
                }
                source_operations += operations;
                packed_cursor += bytes;
            }
            if (packed_cursor != packed_bytes) {
                return vbr_capture_stream_status::projection_invalid;
            }
        }

        // One reusable byte-range workspace is allocated before the unit
        // lease. Segment boundaries remain in the shared slice map; adjacent
        // physical runs and small disjoint runs are packed by the ring.
        std::vector<vbr_capture_stream_range> ranges(cell_ranges.size());

        // Hash the immutable projection before acquiring the unit-version
        // lease. Only bounded source reads and snapshot-dependent sealing
        // remain inside the lease interval.
        llama_sha256_writer layout_hash;
        static constexpr char LAYOUT_DOMAIN[] =
            "buun.vbr.capture/projected-layout";
        layout_hash.string(LAYOUT_DOMAIN, sizeof(LAYOUT_DOMAIN) - 1);
        layout_hash.u64(projection->source_namespace);
        layout_hash.u32(child_id);
        layout_hash.u32(stream_index);
        layout_hash.u32(logical_unit_id);
        layout_hash.bytes(topology_digest.data(), topology_digest.size());
        layout_hash.u64(selected->segments.size());
        for (const auto & segment : selected->segments) {
            layout_hash.u32(segment.first_physical_cell);
            layout_hash.u32(segment.cell_count);
            layout_hash.u32(segment.dependency_count);
            for (uint32_t i = 0; i < segment.dependency_count; ++i) {
                layout_hash.u64(projection->dependent_manifest_ids[
                    segment.first_dependency + i]);
            }
        }
        layout_hash.u64(slices.size());
        for (const auto & slice : slices) {
            layout_hash.u32(slice.projection_segment);
            layout_hash.u64(slice.packed_first_row);
            layout_hash.u32(slice.row_count);
        }
        const auto layout_digest = layout_hash.finish();

        vbr_capture_unit_snapshot snapshot;
        if (!snapshots.acquire(
                snapshots.context, projection->source_namespace,
                child_id, logical_unit_id, snapshot)) {
            return vbr_capture_stream_status::snapshot_unavailable;
        }
        struct release_guard {
            const vbr_capture_unit_snapshot_provider * provider = nullptr;
            const vbr_capture_unit_snapshot * snapshot = nullptr;
            bool active = false;
            ~release_guard() {
                if (active) {
                    provider->release(provider->context, *snapshot);
                }
            }
        } release { &snapshots, &snapshot, true };
        if (!projected_snapshot_valid(snapshot) ||
            snapshot.source_namespace != projection->source_namespace ||
            snapshot.child_id != child_id ||
            snapshot.logical_unit_id != logical_unit_id ||
            snapshot.shard_count != topology_count ||
            snapshot.shard_topology_digest != topology_digest) {
            return vbr_capture_stream_status::snapshot_unavailable;
        }

        vbr_capture_projected_unit result;
        result.projection = std::move(projection);
        result.snapshot = snapshot;
        result.child_id = child_id;
        result.stream_index = stream_index;
        result.logical_unit_id = logical_unit_id;
        result.packed_bytes = total_packed_bytes;
        result.slices = std::move(slices);
        result.shards.reserve(ordered.size());
        llama_sha256_writer unit_hash;
        static constexpr char UNIT_DOMAIN[] =
            "buun.vbr.capture/projected-unit";
        unit_hash.string(UNIT_DOMAIN, sizeof(UNIT_DOMAIN) - 1);
        unit_hash.bytes(layout_digest.data(), layout_digest.size());
        unit_hash.u64(snapshot.controller_generation);
        unit_hash.u64(snapshot.mutation_serial);
        unit_hash.u64(snapshot.generation.repr_gen);
        unit_hash.u64(snapshot.generation.publish_seq);
        unit_hash.u32(uint32_t(snapshot.generation.current_type));
        unit_hash.u32(uint32_t(snapshot.generation.last_source_type));
        unit_hash.u32(uint32_t(snapshot.generation.domain));
        unit_hash.u32(snapshot.generation.promote_hops);
        unit_hash.u32(uint32_t(snapshot.generation.last_transition));
        for (size_t shard_index = 0;
             shard_index < ordered.size(); ++shard_index) {
            const auto * shard = ordered[shard_index];
            for (size_t i = 0; i < cell_ranges.size(); ++i) {
                ranges[i] = {
                    uint64_t(cell_ranges[i].first)*shard->row_bytes,
                    uint64_t(cell_ranges[i].count)*shard->row_bytes,
                };
            }
            auto chain = std::make_shared<artifact_segment_chain>();
            vbr_capture_stream_stats stats;
            const auto streamed = ring.stream_ranges(
                shard->source, ranges, *chain, stats);
            if (streamed != vbr_capture_stream_status::ok) {
                return streamed;
            }
            if (stats.bytes != shard_packed_bytes[shard_index] ||
                stats.bytes > UINT64_MAX - result.transfer.bytes ||
                stats.chunks > UINT64_MAX - result.transfer.chunks ||
                stats.backpressure_waits >
                    UINT64_MAX - result.transfer.backpressure_waits ||
                stats.event_completions >
                    UINT64_MAX - result.transfer.event_completions ||
                stats.synchronous_fallbacks >
                    UINT64_MAX - result.transfer.synchronous_fallbacks) {
                return vbr_capture_stream_status::internal_error;
            }
            result.transfer.bytes += stats.bytes;
            result.transfer.chunks += stats.chunks;
            result.transfer.backpressure_waits += stats.backpressure_waits;
            result.transfer.event_completions += stats.event_completions;
            result.transfer.synchronous_fallbacks +=
                stats.synchronous_fallbacks;
            result.transfer.max_segment_size = std::max(
                result.transfer.max_segment_size,
                stats.max_segment_size);
            unit_hash.u32(shard->shard_index);
            unit_hash.u32(shard->row_count);
            unit_hash.u64(shard->row_bytes);
            unit_hash.u64(shard->source_identity);
            unit_hash.bytes(
                stats.streaming_digest.data(),
                stats.streaming_digest.size());
            result.shards.push_back({
                shard->shard_index,
                std::move(chain),
                stats.streaming_digest,
            });
        }
        result.transfer.streaming_digest = unit_hash.finish();
        if (!snapshots.recheck(snapshots.context, snapshot)) {
            return vbr_capture_stream_status::snapshot_changed;
        }
        snapshots.release(snapshots.context, snapshot);
        release.active = false;
        output = std::move(result);
        return vbr_capture_stream_status::ok;
    } catch (...) {
        output = {};
        return vbr_capture_stream_status::internal_error;
    }
}
