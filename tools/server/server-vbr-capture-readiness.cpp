#include "server-vbr-capture-readiness.h"

#include <algorithm>
#include <limits>

namespace {

bool checked_add(uint64_t a, uint64_t b, uint64_t & output) noexcept {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        return false;
    }
    output = a + b;
    return true;
}

bool rate(uint64_t value, uint64_t elapsed_us, uint64_t & output) noexcept {
    if (value == 0 || elapsed_us == 0 ||
        value > std::numeric_limits<uint64_t>::max()/1000000ull) {
        return false;
    }
    output = std::max<uint64_t>(1, value*1000000ull/elapsed_us);
    return true;
}

bool duration_us(uint64_t amount, uint64_t per_second,
                 uint64_t & output) noexcept {
    if (per_second == 0 ||
        amount > (std::numeric_limits<uint64_t>::max()-(per_second-1))/
            1000000ull) {
        return false;
    }
    output = (amount*1000000ull + per_second - 1)/per_second;
    return true;
}

bool capacity_fits(uint64_t capacity, uint64_t committed,
                   uint64_t reserved, uint64_t incoming) noexcept {
    if (capacity == 0) {
        return true;
    }
    uint64_t used = 0;
    return checked_add(committed, reserved, used) &&
        used <= capacity && incoming <= capacity-used;
}

} // namespace

server_vbr_capture_readiness_status server_vbr_capture_readiness_admit(
        const server_vbr_capture_readiness_input & input,
        uint64_t generation,
        server_vbr_capture_readiness_reservation & output) noexcept {
    output = {};
    if (generation == 0 || input.source_slot < 0 ||
        input.candidate_transfer_bytes == 0 ||
        input.candidate_host_bytes == 0 ||
        input.device_capacity_cells == 0 ||
        input.conservative_bandwidth_bytes_per_second == 0) {
        return server_vbr_capture_readiness_status::invalid;
    }
    if (!capacity_fits(
            input.host_capacity_bytes, input.host_committed_bytes,
            input.host_reserved_bytes, input.candidate_host_bytes)) {
        return server_vbr_capture_readiness_status::host_unavailable;
    }
    if (!capacity_fits(
            input.metadata_capacity_bytes, input.metadata_committed_bytes,
            input.metadata_reserved_bytes, input.candidate_metadata_bytes)) {
        return server_vbr_capture_readiness_status::metadata_unavailable;
    }
    if (input.pinned_lane_slots_available < 1) {
        return server_vbr_capture_readiness_status::pinned_unavailable;
    }
    if (input.queue_slots_available < 1) {
        return server_vbr_capture_readiness_status::queue_unavailable;
    }
    uint64_t device_used = 0;
    if (!checked_add(input.device_committed_cells,
                     input.admitted_growth_cells, device_used) ||
        !checked_add(device_used, input.emergency_cells, device_used) ||
        device_used > input.device_capacity_cells) {
        return server_vbr_capture_readiness_status::
            device_runway_unavailable;
    }
    const uint64_t runway_cells = input.device_capacity_cells-device_used;
    uint64_t queued_and_candidate = 0;
    uint64_t transfer_us = 0;
    uint64_t capture_us = 0;
    if (!checked_add(input.queued_capture_bytes,
                     input.candidate_transfer_bytes,
                     queued_and_candidate) ||
        !duration_us(queued_and_candidate,
                     input.conservative_bandwidth_bytes_per_second,
                     transfer_us) ||
        !checked_add(transfer_us, input.publication_margin_us, capture_us)) {
        return server_vbr_capture_readiness_status::invalid;
    }
    uint64_t runway_us = std::numeric_limits<uint64_t>::max();
    if (input.conservative_growth_cells_per_second != 0 &&
        !duration_us(runway_cells,
                     input.conservative_growth_cells_per_second,
                     runway_us)) {
        return server_vbr_capture_readiness_status::invalid;
    }
    if (capture_us > runway_us) {
        return server_vbr_capture_readiness_status::deadline_missed;
    }
    output.generation = generation;
    output.source_slot = input.source_slot;
    output.host_bytes = input.candidate_host_bytes;
    output.pinned_lane_slots = 1;
    output.device_cells = input.emergency_cells;
    output.queue_slots = 1;
    output.metadata_bytes = input.candidate_metadata_bytes;
    output.emergency_cells = input.emergency_cells;
    output.queued_capture_bytes = input.queued_capture_bytes;
    output.conservative_bandwidth_bytes_per_second =
        input.conservative_bandwidth_bytes_per_second;
    output.conservative_growth_cells_per_second =
        input.conservative_growth_cells_per_second;
    output.forecast_capture_us = capture_us;
    output.pressure_runway_us = runway_us;
    return server_vbr_capture_readiness_status::ready;
}

void server_vbr_capture_readiness_estimator::observe_transfer(
        uint64_t bytes, uint64_t elapsed_us) noexcept {
    uint64_t value = 0;
    if (!rate(bytes, elapsed_us, value)) {
        return;
    }
    transfer_[transfer_count_%SAMPLE_COUNT] = value;
    ++transfer_count_;
}

void server_vbr_capture_readiness_estimator::observe_growth(
        uint64_t cells, uint64_t elapsed_us) noexcept {
    uint64_t value = 0;
    if (!rate(cells, elapsed_us, value)) {
        return;
    }
    growth_[growth_count_%SAMPLE_COUNT] = value;
    ++growth_count_;
}

uint64_t server_vbr_capture_readiness_estimator::
conservative_bandwidth_bytes_per_second() const noexcept {
    const size_t count = size_t(std::min<uint64_t>(
        transfer_count_, SAMPLE_COUNT));
    if (count == 0) {
        // Cold-start conservatism matters most for prompt-boundary iSWA: the
        // first transfer gates only that source's next decode. Hybrid model
        // measurements are closer to this floor than PCIe peak bandwidth.
        return 64ull*1024ull*1024ull;
    }
    uint64_t result = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < count; ++i) {
        result = std::min(result, transfer_[i]);
    }
    return result;
}

uint64_t server_vbr_capture_readiness_estimator::
conservative_growth_cells_per_second() const noexcept {
    const size_t count = size_t(std::min<uint64_t>(
        growth_count_, SAMPLE_COUNT));
    if (count == 0) {
        return 1024;
    }
    uint64_t result = 0;
    for (size_t i = 0; i < count; ++i) {
        result = std::max(result, growth_[i]);
    }
    return result;
}
