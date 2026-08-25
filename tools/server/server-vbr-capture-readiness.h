#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

enum class server_vbr_capture_readiness_status : uint8_t {
    ready = 0,
    invalid,
    host_unavailable,
    pinned_unavailable,
    device_runway_unavailable,
    queue_unavailable,
    metadata_unavailable,
    deadline_missed,
    _count,
};

// Scheduler-owned evidence for one admitted background capture. The durable
// host-capacity claim, persistent-ring operation, source capture lease, and
// queue owner remain in their canonical subsystems; this record binds their
// measured quantities and the forecast used to admit them.
struct server_vbr_capture_readiness_reservation {
    uint64_t generation = 0;
    int32_t source_slot = -1;
    uint64_t host_bytes = 0;
    uint64_t pinned_lane_slots = 0;
    uint64_t device_cells = 0;
    uint64_t queue_slots = 0;
    uint64_t metadata_bytes = 0;
    uint64_t emergency_cells = 0;
    uint64_t queued_capture_bytes = 0;
    uint64_t conservative_bandwidth_bytes_per_second = 0;
    uint64_t conservative_growth_cells_per_second = 0;
    uint64_t forecast_capture_us = 0;
    uint64_t pressure_runway_us = 0;

    explicit operator bool() const noexcept { return generation != 0; }
    void reset() noexcept { *this = {}; }
};

struct server_vbr_capture_readiness_input {
    int32_t source_slot = -1;
    uint64_t candidate_transfer_bytes = 0;
    uint64_t candidate_host_bytes = 0;
    uint64_t candidate_metadata_bytes = 0;
    uint64_t queued_capture_bytes = 0;
    uint64_t publication_margin_us = 0;

    uint64_t host_capacity_bytes = 0; // zero means externally/unbounded
    uint64_t host_committed_bytes = 0;
    uint64_t host_reserved_bytes = 0;
    uint64_t metadata_capacity_bytes = 0; // zero means included in host cap
    uint64_t metadata_committed_bytes = 0;
    uint64_t metadata_reserved_bytes = 0;
    uint64_t pinned_lane_slots_available = 0;
    uint64_t queue_slots_available = 0;

    uint64_t device_capacity_cells = 0;
    uint64_t device_committed_cells = 0;
    uint64_t admitted_growth_cells = 0;
    uint64_t emergency_cells = 0;

    uint64_t conservative_bandwidth_bytes_per_second = 0;
    uint64_t conservative_growth_cells_per_second = 0;
};

server_vbr_capture_readiness_status server_vbr_capture_readiness_admit(
    const server_vbr_capture_readiness_input & input,
    uint64_t generation,
    server_vbr_capture_readiness_reservation & output) noexcept;

// Fixed-arena observations: D2H uses the slowest retained sample; live growth
// uses the fastest. This is deliberately conservative and cannot allocate on
// decode, cancellation, or publication paths.
class server_vbr_capture_readiness_estimator {
public:
    static constexpr size_t SAMPLE_COUNT = 16;

    void observe_transfer(uint64_t bytes, uint64_t elapsed_us) noexcept;
    void observe_growth(uint64_t cells, uint64_t elapsed_us) noexcept;

    uint64_t conservative_bandwidth_bytes_per_second() const noexcept;
    uint64_t conservative_growth_cells_per_second() const noexcept;
    uint64_t transfer_samples() const noexcept { return transfer_count_; }
    uint64_t growth_samples() const noexcept { return growth_count_; }

private:
    std::array<uint64_t, SAMPLE_COUNT> transfer_ = {};
    std::array<uint64_t, SAMPLE_COUNT> growth_ = {};
    uint64_t transfer_count_ = 0;
    uint64_t growth_count_ = 0;
};
