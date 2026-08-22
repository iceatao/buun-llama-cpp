#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

// Pure policy ordering for the VBR tree-shed transaction. This layer knows nothing about VMM
// residency or allocator state: cache adapters provide page-padded LOGICAL progress for each real
// ladder step, and the physical planner prices each yielded prefix independently.
namespace llama_vbr_policy {

struct step {
    size_t  order_index = 0;
    size_t  slot        = 0;
    int32_t type_a      = 0;
    int32_t type_b      = 0;
    int64_t logical_gain = 0;
};

struct child {
    // Logical progress before the first remaining ladder step.  This may be negative when the
    // incoming watermark grows relative to the child's retained mapped watermark.  Selection
    // clamps it to zero, matching the existing proportional policy.
    int64_t initial_progress = 0;
    // Full remaining-consent logical delta on the demanded device.  Non-positive children are
    // ineligible, matching the live selector's terminal_delta gate.
    int64_t terminal_progress = 0;
    std::vector<step> steps;
    // Sealed steps omitted from this projection. A transaction that commits
    // beyond one of these ordinals must defer it for a later boundary rather
    // than losing it behind the monotone cursor.
    std::vector<size_t> blocked_order_indices;
    // Capture-leased units are omitted before prefix selection so one busy
    // unit cannot freeze unrelated tree shedding. The selected units are
    // still guarded again at commit preflight to close the TOCTOU window.
    std::vector<size_t> capture_blocked_order_indices;
};

struct selection {
    size_t child_index = 0;
    size_t child_step_index = 0;
    step   value;
};

enum class result {
    selected,
    exhausted,
    invalid,
    overflow,
};

inline bool checked_add(int64_t a, int64_t b, int64_t & out) {
    if ((b > 0 && a > std::numeric_limits<int64_t>::max() - b) ||
        (b < 0 && a < std::numeric_limits<int64_t>::min() - b)) {
        return false;
    }
    out = a + b;
    return true;
}

inline bool logical_endpoint_bytes(
        uint64_t row_bytes, uint32_t watermark, uint64_t slot_span, uint64_t granularity,
        uint64_t & out) {
    if (granularity == 0 || slot_span == 0) {
        return false;
    }
    if (watermark != 0 && row_bytes > std::numeric_limits<uint64_t>::max() / watermark) {
        return false;
    }
    const uint64_t raw = row_bytes * watermark;
    if (raw >= slot_span) {
        out = slot_span;
        return true;
    }
    if (raw > std::numeric_limits<uint64_t>::max() - (granularity - 1)) {
        return false;
    }
    const uint64_t padded = ((raw + granularity - 1) / granularity) * granularity;
    out = padded < slot_span ? padded : slot_span;
    return true;
}

// Exact a/b < c/d for unsigned 64-bit operands without a cross-product overflow.  Continued-
// fraction comparison reverses the ordering every time it takes reciprocals.  Denominators must
// be nonzero.
inline bool fraction_less(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    bool reverse = false;
    for (;;) {
        const uint64_t qa = a / b;
        const uint64_t qc = c / d;
        if (qa != qc) {
            return reverse ? qa > qc : qa < qc;
        }

        const uint64_t ra = a % b;
        const uint64_t rc = c % d;
        if (ra == 0 || rc == 0) {
            if (ra == 0 && rc == 0) {
                return false;
            }
            // At equal integer parts, the exact integer is smaller than the value with a
            // positive remainder.  Reciprocal levels reverse that relation.
            const bool less = ra == 0;
            return reverse ? !less : less;
        }

        a = b;
        b = ra;
        c = d;
        d = rc;
        reverse = !reverse;
    }
}

// A deterministic stream of real ladder steps.  Child vector order is the stable tie-break;
// callers must insert the ledger root first and its peer second.  When integrated with the
// successor's SWA-root topology, that means an active SWA root wins an exact initial tie.
class shortest_prefix_stream {
public:
    explicit shortest_prefix_stream(
            std::vector<child> children,
            bool retain_history = true) :
        children_(std::move(children)), retain_history_(retain_history) {
        states_.reserve(children_.size());
        for (const child & c : children_) {
            state s;
            s.progress = c.initial_progress;
            states_.push_back(s);
            if (c.terminal_progress > 0) {
                for (const step & st : c.steps) {
                    if (st.logical_gain < 0) {
                        valid_ = false;
                        break;
                    }
                }
            }
        }
        for (size_t i = 0; i < children_.size(); ++i) {
            if (eligible(i)) {
                heap_.push_back(i);
            }
        }
        std::make_heap(
            heap_.begin(), heap_.end(),
            [this](size_t lhs, size_t rhs) { return better(rhs, lhs); });
    }

    result next(selection & out) {
        if (!valid_) {
            return result::invalid;
        }

        if (heap_.empty()) {
            return result::exhausted;
        }
        std::pop_heap(
            heap_.begin(), heap_.end(),
            [this](size_t lhs, size_t rhs) { return better(rhs, lhs); });
        const size_t pick = heap_.back();
        heap_.pop_back();

        state & s = states_[pick];
        const size_t child_step_index = s.next;
        const step & st = children_[pick].steps[child_step_index];
        int64_t progress_next = 0;
        if (!checked_add(s.progress, st.logical_gain, progress_next)) {
            return result::overflow;
        }

        s.progress = progress_next;
        s.next++;
        if (eligible(pick)) {
            heap_.push_back(pick);
            std::push_heap(
                heap_.begin(), heap_.end(),
                [this](size_t lhs, size_t rhs) { return better(rhs, lhs); });
        }
        out = { pick, child_step_index, st };
        if (retain_history_) {
            selected_.push_back(out);
        }
        return result::selected;
    }

    // Physical repricing supplies `accept(prefix)` and is invoked after every real policy step.
    // The first accepted prefix is therefore the shortest prefix of this deterministic stream.
    template<class Accept>
    result shortest_prefix(Accept && accept, std::vector<selection> & out) {
        out.clear();
        selection chosen;
        for (;;) {
            const result r = next(chosen);
            if (r != result::selected) {
                return r;
            }
            out.push_back(chosen);
            if (accept(out)) {
                return result::selected;
            }
        }
    }

    const std::vector<selection> & selected() const {
        return selected_;
    }

    uint64_t comparison_count() const {
        return comparisons_;
    }

private:
    struct state {
        size_t  next = 0;
        int64_t progress = 0;
    };

    bool eligible(size_t index) const {
        return index < children_.size() &&
               children_[index].terminal_progress > 0 &&
               states_[index].next < children_[index].steps.size();
    }

    bool better(size_t lhs, size_t rhs) const {
        comparisons_++;
        const auto & lchild = children_[lhs];
        const auto & rchild = children_[rhs];
        const uint64_t ln = states_[lhs].progress > 0
            ? uint64_t(states_[lhs].progress) : 0;
        const uint64_t rn = states_[rhs].progress > 0
            ? uint64_t(states_[rhs].progress) : 0;
        const uint64_t ld = uint64_t(lchild.terminal_progress);
        const uint64_t rd = uint64_t(rchild.terminal_progress);
        if (fraction_less(ln, ld, rn, rd)) {
            return true;
        }
        if (fraction_less(rn, rd, ln, ld)) {
            return false;
        }
        return lhs < rhs;
    }

    std::vector<child> children_;
    std::vector<state> states_;
    std::vector<selection> selected_;
    std::vector<size_t> heap_;
    mutable uint64_t comparisons_ = 0;
    bool valid_ = true;
    bool retain_history_ = true;
};

} // namespace llama_vbr_policy
