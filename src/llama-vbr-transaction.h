#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

namespace llama_vbr_transaction {

struct device_cost {
    uint64_t release          = 0;
    uint64_t kv_growth        = 0;
    uint64_t scratch_growth   = 0;
    uint64_t workspace_growth = 0;
    uint64_t stash_growth     = 0;
    int64_t  capacity_signed  = 0;
};

inline bool add_u64(uint64_t & dst, uint64_t value) {
    if (value > std::numeric_limits<uint64_t>::max() - dst) {
        return false;
    }
    dst += value;
    return true;
}

inline bool finalize(device_cost & cost) {
    uint64_t growth = 0;
    if (!add_u64(growth, cost.kv_growth) ||
        !add_u64(growth, cost.scratch_growth) ||
        !add_u64(growth, cost.workspace_growth) ||
        !add_u64(growth, cost.stash_growth) ||
        cost.release > (uint64_t) std::numeric_limits<int64_t>::max() ||
        growth       > (uint64_t) std::numeric_limits<int64_t>::max()) {
        return false;
    }
    cost.capacity_signed = (int64_t) cost.release - (int64_t) growth;
    return true;
}

inline bool prefix_feasible(
        const std::map<int, device_cost> & costs,
        int demanded_device,
        uint64_t target) {
    const auto demanded = costs.find(demanded_device);
    if (demanded == costs.end() || target > (uint64_t) std::numeric_limits<int64_t>::max() ||
        demanded->second.capacity_signed < (int64_t) target) {
        return false;
    }
    for (const auto & [device, cost] : costs) {
        if (device != demanded_device && cost.capacity_signed < 0) {
            return false;
        }
    }
    return true;
}

struct workspace_request {
    int64_t n_cells   = 0;
    int64_t ne0       = 0;
    int64_t stash_rows = 0;
    bool mean_addback = false;
};

// Price only tuples that will really execute.  In particular, never synthesize a tuple from
// independent maxima of n_cells, ne0 and stash_rows: that can overstate a workspace endpoint
// which no transcode/capture pair can request.
template<typename Project>
bool workspace_endpoint(
        const std::vector<workspace_request> & requests,
        Project project,
        uint64_t & physical_now,
        uint64_t & physical_if_reserved,
        workspace_request * endpoint_request = nullptr) {
    physical_now = 0;
    physical_if_reserved = 0;
    if (endpoint_request) {
        *endpoint_request = {};
    }
    bool have = false;
    bool selected = false;
    for (const auto & request : requests) {
        uint64_t now = 0;
        uint64_t projected = 0;
        if (!project(request, now, projected) || projected < now) {
            return false;
        }
        if (!have) {
            physical_now = now;
            have = true;
        } else if (physical_now != now) {
            return false;
        }
        if (endpoint_request != nullptr &&
            (!selected || physical_if_reserved < projected)) {
            *endpoint_request = request;
            selected = true;
        }
        physical_if_reserved = std::max(physical_if_reserved, projected);
    }
    return true;
}

inline bool grant_threshold(uint64_t bytes_now, uint64_t prior_credit, uint64_t & threshold) {
    if (prior_credit > std::numeric_limits<uint64_t>::max() - bytes_now) {
        return false;
    }
    threshold = bytes_now + prior_credit;
    return true;
}

inline uint64_t grant_row_remaining(uint64_t bytes, uint64_t threshold, uint64_t bytes_now) {
    const uint64_t landed = bytes_now > threshold ? bytes_now - threshold : 0;
    return landed < bytes ? bytes - landed : 0;
}

} // namespace llama_vbr_transaction
