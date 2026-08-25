#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

// One measured degrade-order step. Shared with the read-only hard-seal
// classifier so controller policy continues to have one canonical order.
struct vbr_degrade_step {
    uint8_t il = 0;
    uint8_t is_v = 0;
    uint8_t tier = 0;
};

struct vbr_hard_seal_subject {
    uint8_t il = 0;
    bool is_v = false;
    size_t order_ordinal = 0;

    constexpr bool operator==(const vbr_hard_seal_subject & other) const noexcept {
        return il == other.il && is_v == other.is_v &&
            order_ordinal == other.order_ordinal;
    }
};

struct vbr_hard_seal_classification {
    std::vector<vbr_hard_seal_subject> affected;
};

// Occupied token range affected by a representation transition. The KV core
// knows storage geometry but deliberately knows nothing about server leases;
// the scheduler-owned callback supplies that read-only policy decision.
struct vbr_hard_seal_range {
    int32_t sequence = -1;
    uint32_t first_token = 0;
    uint32_t token_count = 0;

    bool operator==(const vbr_hard_seal_range & other) const noexcept {
        return sequence == other.sequence &&
            first_token == other.first_token &&
            token_count == other.token_count;
    }
};

enum class vbr_hard_seal_guard_result : uint8_t {
    allow,
    hard_lease_blocked,
};

struct vbr_hard_seal_guard {
    // Cheap global gate. When false, the controller must not build cell ranges
    // or live identities. Both callbacks are read-only and scheduler-owned.
    std::function<bool()> any_hard_lease;
    std::function<vbr_hard_seal_guard_result(
        const vbr_hard_seal_subject &,
        const std::vector<vbr_hard_seal_range> &)> inspect;

    explicit operator bool() const noexcept {
        return bool(any_hard_lease) && bool(inspect);
    }
};

// One boundary-local consult. The guard contract is subject-uniform within a
// cache's sealed band: identical occupied ranges must produce one verdict for
// every sealed order subject. This is required because policy planning and
// transaction apply can see the subjects through different order projections.
struct vbr_hard_seal_consult_session {
    vbr_hard_seal_classification classification;
    std::vector<vbr_hard_seal_range> ranges;
    bool any_hard_sampled = false;
    bool any_hard = false;
    bool classified = false;
    bool classification_failed = false;
    bool ranges_built = false;
    bool verdict_sampled = false;
    vbr_hard_seal_guard_result verdict =
        vbr_hard_seal_guard_result::allow;
};

// Frozen  default: crossing from the restorable T8 band into T4.
inline constexpr ggml_type VBR_HARD_SEAL_DEFAULT_FLOOR =
    GGML_TYPE_TURBO4_0;

// Central, read-only classification kernel. It never changes the order,
// cursor, floor, controller serial, or any backend state.
bool vbr_classify_hard_seal(
    const std::vector<vbr_degrade_step> & order,
    uint8_t seal_tier,
    vbr_hard_seal_classification & out) noexcept;

const vbr_hard_seal_subject * vbr_hard_seal_subject_for_step(
    const vbr_hard_seal_classification & classification,
    size_t order_ordinal) noexcept;

// Allocation-free cursor mechanics shared by the live controller and its CPU
// oracle. Deferred sealed steps are considered first on each new boundary,
// but at most once per call so pressure can advance to later units.
bool vbr_hard_seal_next_order_step(
    size_t & cursor,
    size_t limit,
    const std::vector<size_t> & deferred,
    std::vector<uint8_t> & attempted,
    size_t & order_ordinal,
    bool & from_deferred) noexcept;
void vbr_hard_seal_defer_step(
    std::vector<size_t> & deferred,
    size_t order_ordinal,
    std::vector<uint8_t> * attempted = nullptr);
void vbr_hard_seal_retire_step(
    std::vector<size_t> & deferred, size_t order_ordinal) noexcept;
void vbr_hard_seal_defer_jumped_steps(
    std::vector<size_t> & deferred,
    const std::vector<size_t> & blocked,
    size_t final_cursor);

// Consumes every latch sample, including successful decodes. This prevents a
// donor-side refusal from retyping a later unrelated allocation failure.
bool vbr_hard_seal_take_decode_terminal(
    bool decode_failed, bool & blocked) noexcept;
