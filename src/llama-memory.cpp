#include "llama-memory.h"

#include <algorithm>

llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

bool llama_memory_status_is_fail(llama_memory_status status) {
    switch (status) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}

llama_memory_vbr_preflight_data llama_memory_vbr_merge_preflight_children(
        const llama_memory_vbr_preflight_data & b,
        const std::vector<llama_memory_vbr_physical_growth> & b_physical,
        const llama_memory_vbr_preflight_data & s,
        const std::vector<llama_memory_vbr_physical_growth> & s_physical,
        std::vector<llama_memory_vbr_physical_growth> * physical,
        uint64_t * domain_comparisons) {
    const auto add_saturated = [](uint64_t lhs, uint64_t rhs) {
        return rhs > UINT64_MAX - lhs ? UINT64_MAX : lhs + rhs;
    };
    llama_memory_vbr_preflight_data r = {};
    r.active          = b.active || s.active;
    r.fits            = b.fits && s.fits;
    r.pools           = s.pools > UINT32_MAX - b.pools
        ? UINT32_MAX : b.pools + s.pools;
    r.watermark_cells = std::max(b.watermark_cells, s.watermark_cells);
    r.bytes_needed    = add_saturated(b.bytes_needed, s.bytes_needed);
    r.bytes_available = add_saturated(b.bytes_available, s.bytes_available);
    std::vector<llama_memory_vbr_physical_growth> local;
    auto & combined = physical ? *physical : local;
    combined.clear();
    combined.reserve(b_physical.size() + s_physical.size());
    const auto less = [&](const auto & lhs, const auto & rhs) {
        if (domain_comparisons) {
            ++*domain_comparisons;
        }
        return lhs < rhs;
    };
    bool device_needed_overflow = false;
    const auto add_device_needed = [&](uint64_t lhs, uint64_t rhs) {
        if (rhs > UINT64_MAX - lhs) {
            device_needed_overflow = true;
            return UINT64_MAX;
        }
        return lhs + rhs;
    };
    const auto combine_row = [&](auto & output, const auto & row) {
        output.kv_needed = add_device_needed(
            output.kv_needed, row.kv_needed);
        output.scratch_k_needed = std::max(
            output.scratch_k_needed, row.scratch_k_needed);
        output.scratch_v_needed = std::max(
            output.scratch_v_needed, row.scratch_v_needed);
        output.scratch_k_current = std::max(
            output.scratch_k_current, row.scratch_k_current);
        output.scratch_v_current = std::max(
            output.scratch_v_current, row.scratch_v_current);
        output.available = std::min(output.available, row.available);
    };
    const auto append_row = [&](const auto & row) {
        if (!combined.empty() && combined.back().same_domain(row)) {
            combine_row(combined.back(), row);
        } else {
            combined.push_back(row);
        }
    };
    const bool b_sorted = std::is_sorted(
        b_physical.begin(), b_physical.end(), less);
    const bool s_sorted = std::is_sorted(
        s_physical.begin(), s_physical.end(), less);
    if (b_sorted && s_sorted) {
        size_t bi = 0;
        size_t si = 0;
        while (bi < b_physical.size() || si < s_physical.size()) {
            if (bi == b_physical.size()) {
                append_row(s_physical[si++]);
            } else if (si == s_physical.size()) {
                append_row(b_physical[bi++]);
            } else if (less(b_physical[bi], s_physical[si])) {
                append_row(b_physical[bi++]);
            } else if (less(s_physical[si], b_physical[bi])) {
                append_row(s_physical[si++]);
            } else {
                append_row(b_physical[bi++]);
                append_row(s_physical[si++]);
            }
        }
    } else {
        combined.insert(
            combined.end(), b_physical.begin(), b_physical.end());
        combined.insert(
            combined.end(), s_physical.begin(), s_physical.end());
        std::sort(combined.begin(), combined.end(), less);
        size_t output = 0;
        for (size_t input = 0; input < combined.size(); ++input) {
            if (output != 0 &&
                combined[output - 1].same_domain(combined[input])) {
                combine_row(combined[output - 1], combined[input]);
            } else {
                if (output != input) {
                    combined[output] = combined[input];
                }
                ++output;
            }
        }
        combined.resize(output);
    }
    r.physical_growth_needed = 0;
    r.physical_growth_available = 0;
    int64_t physical_deficit = 0;
    for (const auto & row : combined) {
        const uint64_t scratch_k_growth =
            row.scratch_k_needed > row.scratch_k_current
                ? row.scratch_k_needed - row.scratch_k_current : 0;
        const uint64_t scratch_v_growth =
            row.scratch_v_needed > row.scratch_v_current
                ? row.scratch_v_needed - row.scratch_v_current : 0;
        uint64_t needed = add_device_needed(
            row.kv_needed, scratch_k_growth);
        needed = add_device_needed(needed, scratch_v_growth);
        r.physical_growth_needed = add_saturated(
            r.physical_growth_needed, needed);
        r.physical_growth_available = add_saturated(
            r.physical_growth_available, row.available);
        const int64_t deficit = needed > row.available
            ? needed - row.available > uint64_t(INT64_MAX)
                ? INT64_MAX : int64_t(needed - row.available)
            : 0;
        physical_deficit = std::max(physical_deficit, deficit);
    }
    if (device_needed_overflow) {
        physical_deficit = INT64_MAX;
    }
    r.max_deficit = std::max({
        b.max_deficit, s.max_deficit, physical_deficit });
    r.fits = r.fits && physical_deficit == 0;
    return r;
}

bool llama_memory_vbr_preflight_tree::reset(size_t leaf_count) noexcept {
    leaf_count_ = 0;
    leaf_base_ = 0;
    ready_ = false;
    merge_count_ = 0;
    domain_comparison_count_ = 0;
    tree_.clear();
    try {
        if (leaf_count == 0) {
            return false;
        }
        size_t base = 1;
        while (base < leaf_count) {
            if (base > SIZE_MAX/2) {
                return false;
            }
            base *= 2;
        }
        tree_.resize(base*2);
        leaf_count_ = leaf_count;
        leaf_base_ = base;
        return true;
    } catch (...) {
        tree_.clear();
        leaf_count_ = 0;
        leaf_base_ = 0;
        return false;
    }
}

bool llama_memory_vbr_preflight_tree::set_leaf(
        size_t index,
        const llama_memory_vbr_preflight_data & preflight,
        const std::vector<llama_memory_vbr_physical_growth> & physical) noexcept {
    if (ready_ || index >= leaf_count_) {
        return false;
    }
    try {
        auto replacement = physical;
        std::sort(replacement.begin(), replacement.end());
        for (size_t i = 1; i < replacement.size(); ++i) {
            if (replacement[i - 1].same_domain(replacement[i])) {
                return false;
            }
        }
        auto & leaf = tree_[leaf_base_ + index];
        leaf.preflight = preflight;
        leaf.physical = std::move(replacement);
        leaf.present = true;
        return true;
    } catch (...) {
        return false;
    }
}

void llama_memory_vbr_preflight_tree::merge_node(size_t index) {
    auto & output = tree_[index];
    const auto & left = tree_[index*2];
    const auto & right = tree_[index*2 + 1];
    if (!left.present && !right.present) {
        output.present = false;
        output.preflight = {};
        output.physical.clear();
    } else if (!left.present || !right.present) {
        const auto & source = left.present ? left : right;
        output.present = true;
        output.preflight = source.preflight;
        output.physical.assign(
            source.physical.begin(), source.physical.end());
    } else {
        output.present = true;
        output.preflight = llama_memory_vbr_merge_preflight_children(
            left.preflight, left.physical,
            right.preflight, right.physical,
            &output.physical, &domain_comparison_count_);
    }
    ++merge_count_;
}

bool llama_memory_vbr_preflight_tree::build() noexcept {
    if (ready_ || leaf_count_ == 0) {
        return false;
    }
    for (size_t i = 0; i < leaf_count_; ++i) {
        if (!tree_[leaf_base_ + i].present) {
            return false;
        }
    }
    try {
        for (size_t i = leaf_base_; i-- > 1;) {
            merge_node(i);
        }
        ready_ = tree_[1].present;
        return ready_;
    } catch (...) {
        ready_ = false;
        return false;
    }
}

bool llama_memory_vbr_preflight_tree::replace_leaf(
        size_t index,
        const llama_memory_vbr_preflight_data & preflight,
        const std::vector<llama_memory_vbr_physical_growth> & physical) noexcept {
    if (!ready_ || index >= leaf_count_ ||
        !std::is_sorted(physical.begin(), physical.end())) {
        return false;
    }
    auto & leaf = tree_[leaf_base_ + index];
    if (leaf.physical.size() != physical.size()) {
        return false;
    }
    for (size_t i = 0; i < physical.size(); ++i) {
        if (!leaf.physical[i].same_domain(physical[i])) {
            return false;
        }
    }
    // Domain shape and every ancestor capacity were fixed by build(), so this
    // update path performs no allocation and cannot leave a partial tree.
    leaf.preflight = preflight;
    std::copy(physical.begin(), physical.end(), leaf.physical.begin());
    size_t node_index = leaf_base_ + index;
    while (node_index > 1) {
        node_index /= 2;
        merge_node(node_index);
    }
    return true;
}

bool llama_memory_vbr_preflight_tree::ready() const noexcept {
    return ready_;
}

const llama_memory_vbr_preflight_data &
llama_memory_vbr_preflight_tree::preflight() const noexcept {
    static const llama_memory_vbr_preflight_data empty = {};
    return ready_ ? tree_[1].preflight : empty;
}

const std::vector<llama_memory_vbr_physical_growth> &
llama_memory_vbr_preflight_tree::physical() const noexcept {
    static const std::vector<llama_memory_vbr_physical_growth> empty;
    return ready_ ? tree_[1].physical : empty;
}

uint64_t llama_memory_vbr_preflight_tree::merge_count() const noexcept {
    return merge_count_;
}

uint64_t llama_memory_vbr_preflight_tree::domain_comparison_count() const noexcept {
    return domain_comparison_count_;
}
