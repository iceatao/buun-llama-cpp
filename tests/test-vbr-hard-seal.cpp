#include "llama-vbr-hard-seal.h"

#include <cstdio>

int main() {
    const std::vector<vbr_degrade_step> order = {
        { 2, 0, 0 },
        { 7, 1, 1 },
        { 2, 0, 1 },
        { 7, 1, 2 },
    };
    vbr_hard_seal_classification out;
    if (!vbr_classify_hard_seal(order, 1, out) ||
        VBR_HARD_SEAL_DEFAULT_FLOOR != GGML_TYPE_TURBO4_0 ||
        out.affected.size() != 3 ||
        out.affected[0].il != 7 || !out.affected[0].is_v ||
        out.affected[0].order_ordinal != 1 ||
        out.affected[1].il != 2 || out.affected[1].is_v ||
        out.affected[1].order_ordinal != 2 ||
        out.affected[2].il != 7 || !out.affected[2].is_v ||
        out.affected[2].order_ordinal != 3 ||
        vbr_hard_seal_subject_for_step(out, 0) != nullptr ||
        vbr_hard_seal_subject_for_step(out, 1) != &out.affected[0] ||
        vbr_hard_seal_subject_for_step(out, 2) != &out.affected[1] ||
        vbr_hard_seal_subject_for_step(out, 3) != &out.affected[2]) {
        std::fputs("VBR hard-seal classification failed\n", stderr);
        return 1;
    }

    const std::vector<vbr_hard_seal_range> ranges = {
        { 0, 0, 16 },
    };
    size_t consulted = 0;
    const vbr_hard_seal_guard guard = {
        []() { return true; },
        [&](const vbr_hard_seal_subject &,
            const std::vector<vbr_hard_seal_range> & actual) {
            ++consulted;
            return actual == ranges
                ? vbr_hard_seal_guard_result::hard_lease_blocked
                : vbr_hard_seal_guard_result::allow;
        },
    };
    if (!guard.any_hard_lease() ||
        guard.inspect(out.affected[0], ranges) !=
            vbr_hard_seal_guard_result::hard_lease_blocked ||
        guard.inspect(out.affected[1], ranges) !=
            vbr_hard_seal_guard_result::hard_lease_blocked || consulted != 2) {
        std::fputs("VBR hard-seal subject-uniform guard failed\n", stderr);
        return 1;
    }

    size_t cursor = 0;
    std::vector<size_t> deferred;
    std::vector<uint8_t> attempted(order.size(), 0);
    size_t ordinal = 0;
    bool from_deferred = false;
    if (!vbr_hard_seal_next_order_step(
            cursor, order.size(), deferred, attempted,
            ordinal, from_deferred) || ordinal != 0 || from_deferred) {
        std::fputs("VBR hard-seal initial cursor failed\n", stderr);
        return 1;
    }
    vbr_hard_seal_defer_step(deferred, ordinal, &attempted);
    // The same boundary considers the deferred unit once, then advances.
    if (!vbr_hard_seal_next_order_step(
            cursor, order.size(), deferred, attempted,
            ordinal, from_deferred) || ordinal != 1 || from_deferred) {
        std::fputs("VBR hard-seal leased cursor advance failed\n", stderr);
        return 1;
    }
    // A new boundary retries the earlier unit first; release allows it to be
    // retired, after which the monotone cursor resumes at the next unit.
    attempted.assign(order.size(), 0);
    if (!vbr_hard_seal_next_order_step(
            cursor, order.size(), deferred, attempted,
            ordinal, from_deferred) || ordinal != 0 || !from_deferred) {
        std::fputs("VBR hard-seal release retry failed\n", stderr);
        return 1;
    }
    vbr_hard_seal_retire_step(deferred, ordinal);
    attempted.assign(order.size(), 0);
    if (!deferred.empty() || !vbr_hard_seal_next_order_step(
            cursor, order.size(), deferred, attempted,
            ordinal, from_deferred) || ordinal != 2 || from_deferred) {
        std::fputs("VBR hard-seal release retirement failed\n", stderr);
        return 1;
    }
    // All-hard is terminal after each deferred ordinal is considered exactly
    // once; it cannot spin on the first protected unit.
    cursor = order.size();
    deferred = { 1, 2 };
    attempted.assign(order.size(), 0);
    size_t visited = 0;
    while (vbr_hard_seal_next_order_step(
            cursor, order.size(), deferred, attempted,
            ordinal, from_deferred)) {
        if (!from_deferred || (ordinal != 1 && ordinal != 2)) {
            std::fputs("VBR hard-seal all-hard candidate failed\n", stderr);
            return 1;
        }
        ++visited;
    }
    if (visited != 2) {
        std::fputs("VBR hard-seal all-hard terminal failed\n", stderr);
        return 1;
    }

    // A transaction can jump over interleaved sealed steps because only the
    // allowed steps enter its policy prefix. Every sealed ordinal behind the
    // committed cursor must remain deferred for a future boundary.
    deferred.clear();
    vbr_hard_seal_defer_jumped_steps(deferred, { 1, 3, 5 }, 4);
    if (deferred != std::vector<size_t>({ 1, 3 })) {
        std::fputs("VBR hard-seal transaction defer failed\n", stderr);
        return 1;
    }
    std::puts("VBR_TRANSACTION_DEFER interleaved=1,3 cursor=4 PASS");

    // The decode terminal consumes the latch on every outcome. A refusal
    // during a successful donor shed cannot retype a later ordinary failure.
    bool blocked = true;
    if (vbr_hard_seal_take_decode_terminal(false, blocked) || blocked ||
        vbr_hard_seal_take_decode_terminal(true, blocked)) {
        std::fputs("VBR hard-seal successful-decode latch failed\n", stderr);
        return 1;
    }
    blocked = true;
    if (!vbr_hard_seal_take_decode_terminal(true, blocked) || blocked) {
        std::fputs("VBR hard-seal failed-decode terminal failed\n", stderr);
        return 1;
    }
    std::puts("VBR_FAILURE_LATCH successful_shed_then_cell_failure=normal PASS");
    std::puts("VBR hard-seal classification passed");
    return 0;
}
