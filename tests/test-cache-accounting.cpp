// C0 shadow-ledger contract tests: reserve/stage/commit/abort/release state machine,
// minted-allocation identity + immutable citation tuples, charge-once shared allocations,
// reserved-vs-actual byte separation, concurrent-staged transient peak, unknown-vs-zero,
// checked overflow latching, serial coherence (faults included), and attribution round-trip
// through the normalized allocation rows. Every negative case asserts BOTH the failure
// return and the fault counter — the ledger must misbehave loudly and harmlessly.

#include "llama-cache-accounting.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <thread>
#include <vector>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

static const auto CAT  = llama_cache_acct_category::full_snapshot_payload;
static const auto RES  = llama_cache_acct_residency::pageable_host;
static const auto DOM  = llama_cache_acct_resource_domain::non_device(RES);
static const auto META = llama_cache_acct_category::artifact_reference_metadata;

static std::string hex_digest(const std::array<uint8_t, 32> & bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (uint8_t byte : bytes) {
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 0x0f]);
    }
    return out;
}

static llama_cache_acct_value cell(const llama_cache_acct_snapshot & s,
                                   llama_cache_acct_category c,
                                   const llama_cache_acct_resource_domain & domain,
                                   llama_cache_acct_measure m) {
    for (const auto & row : s.cells) {
        if (row.category == c && row.domain == domain) {
            return row.cell.measures[size_t(m)];
        }
    }
    return {};
}

static void configure_domain(
        llama_cache_acct_ledger & ledger,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_producer producer = llama_cache_acct_producer::observer_init) {
    const llama_cache_acct_completeness_requirement requirement = { domain, producer };
    CHECK(ledger.configure_required_producers(&requirement, 1));
    CHECK(ledger.certify_complete(domain, producer));
}

static void configure_default(llama_cache_acct_ledger & ledger) {
    configure_domain(ledger, DOM);
}

// Compile-negative coverage (C/F freeze requirement 3): the full pairwise non-interchange
// matrix over all accounting/topology identities + raw integers lives in the HEADER, so
// every consumer TU enforces it — nothing to repeat here.

// happy path: reserve -> stage -> commit -> release round-trips durable gauges to zero
static void test_lifecycle() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);
    const size_t alloc_baseline = ledger.allocation_registry_size();

    const auto op = ledger.reserve(CAT, DOM, {}, 100, 128);
    CHECK(bool(op));
    auto s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::reserved).value == 128);
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::logical_payload).state ==
          llama_cache_acct_known::unknown); // unknown until a commit, never a fabricated zero

    const auto alloc = ledger.new_alloc();
    CHECK(bool(alloc));
    CHECK(ledger.allocation_registry_size() == alloc_baseline + 1);
    CHECK(ledger.stage(op, alloc, 128));
    CHECK(ledger.commit(op, 100));
    s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::reserved).value == 0);
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::logical_payload).value   == 100);
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::resident_allocated).value == 128);
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::transient_peak).value    == 128);
    CHECK(s.allocations.size() == 1);
    CHECK(s.allocations[0].logical_bytes == 100 && s.allocations[0].resident_bytes == 128);

    CHECK(ledger.release(op));
    s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::logical_payload).value    == 0);
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::resident_allocated).value == 0);
    // a discharged gauge is a MEASURED zero, not unknown
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::logical_payload).state ==
          llama_cache_acct_known::known);
    CHECK(s.allocations.empty());
    CHECK(ledger.allocation_registry_size() == alloc_baseline);
    CHECK(s.faults_invalid_transition == 0 && s.faults_overflow == 0 &&
          s.faults_unknown_id == 0 && s.faults_allocation == 0);
}

// Sol verify-r1 finding 5.1: reserved is charged/unwound by the RESERVED amount even when
// the staged actual differs — reserve 64, stage 32, abort must leave reserved == 0
static void test_reserve_stage_mismatch() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    const auto op = ledger.reserve(CAT, DOM, {}, 64, 64);
    CHECK(ledger.stage(op, ledger.new_alloc(), 32));
    CHECK(ledger.abort(op));
    auto s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::reserved).value == 0);
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::reserved).state ==
          llama_cache_acct_known::known);
    CHECK(s.faults_overflow == 0);

    // and the commit side of the same asymmetry
    const auto op2 = ledger.reserve(CAT, DOM, {}, 64, 64);
    CHECK(ledger.stage(op2, ledger.new_alloc(), 32));
    CHECK(ledger.commit(op2, 10));
    s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::reserved).value == 0);
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::resident_allocated).value == 32);
    CHECK(ledger.release(op2));
}

// Sol verify-r1 finding 5.2: the transient peak is the high-water mark of CONCURRENTLY
// staged bytes — two live stages of 100 and 200 must report 300
static void test_concurrent_peak() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    const auto op1 = ledger.reserve(CAT, DOM, {}, 100, 100);
    const auto op2 = ledger.reserve(CAT, DOM, {}, 200, 200);
    CHECK(ledger.stage(op1, ledger.new_alloc(), 100));
    CHECK(ledger.stage(op2, ledger.new_alloc(), 200));
    auto s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::transient_peak).value == 300);
    CHECK(ledger.abort(op1));
    CHECK(ledger.abort(op2));
    // the peak survives the aborts
    s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::transient_peak).value == 300);
}

// invalid transitions: all fault-counted, none throw; op ids and alloc ids are validated
static void test_invalid_transitions() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    const auto alloc = ledger.new_alloc();
    const auto op = ledger.reserve(CAT, DOM, {}, 10, 10);
    CHECK(!ledger.commit(op, 10));               // commit before stage
    CHECK(!ledger.release(op));                  // release before commit
    CHECK(!ledger.stage(op, llama_cache_acct_alloc_id{}, 10));            // zero alloc id (unknown_id)
    CHECK(!ledger.stage(op, llama_cache_acct_alloc_id{alloc.v + 999}, 10)); // unminted alloc id (unknown_id)
    CHECK(ledger.stage(op, alloc, 10));
    CHECK(!ledger.stage(op, alloc, 10));         // double stage
    CHECK(ledger.commit(op, 10));
    CHECK(!ledger.commit(op, 10));               // double commit
    CHECK(!ledger.abort(op));                    // abort after commit
    CHECK(ledger.release(op));
    CHECK(!ledger.release(op));                  // double release (op erased -> unknown id)

    CHECK(!ledger.stage(llama_cache_acct_op_id{999}, alloc, 1)); // unknown op

    const auto s = ledger.snapshot();
    // early-commit, early-release, double-stage, double-commit, abort-after-commit
    CHECK(s.faults_invalid_transition == 5);
    // zero alloc, unminted alloc, double release (op erased), unknown op
    CHECK(s.faults_unknown_id == 4);
}

// Sol verify-r1 finding 5.4: an allocation's citation tuple is immutable — a second
// transaction citing the same alloc with a different size/category is a fault, never a
// silent merge; a later commit with a different logical size is a fault too
static void test_alloc_tuple_immutable() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    const auto alloc = ledger.new_alloc();
    const auto op1 = ledger.reserve(CAT, DOM, {}, 100, 100);
    const auto op2 = ledger.reserve(META, DOM, {}, 999, 999); // different category
    const auto op3 = ledger.reserve(CAT, DOM, {}, 100, 100);

    CHECK(ledger.stage(op1, alloc, 100));
    CHECK(!ledger.stage(op2, alloc, 999));       // category+size mismatch -> fault
    CHECK(!ledger.stage(op3, alloc, 50));        // size mismatch -> fault
    CHECK(ledger.stage(op3, alloc, 100));        // matching tuple joins

    CHECK(ledger.commit(op1, 80));
    CHECK(!ledger.commit(op3, 81));              // logical mismatch on shared alloc -> fault
    CHECK(ledger.commit(op3, 80));

    auto s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::logical_payload).value == 80); // charged once
    CHECK(s.faults_invalid_transition >= 3);

    CHECK(ledger.release(op1));
    s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::logical_payload).value == 80);
    CHECK(ledger.release(op3));
    s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::logical_payload).value == 0);
    ledger.abort(op2); // cleanup (still reserved)
}

// abort: zero durable delta, reservation unwound, transient peak retained
static void test_abort_retains_peak() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);
    const size_t alloc_baseline = ledger.allocation_registry_size();

    const auto op = ledger.reserve(CAT, DOM, {}, 50, 64);
    CHECK(ledger.stage(op, ledger.new_alloc(), 64));
    CHECK(ledger.allocation_registry_size() == alloc_baseline + 1);
    CHECK(ledger.abort(op));

    const auto s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::reserved).value        == 0);
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::logical_payload).state ==
          llama_cache_acct_known::unknown);
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::transient_peak).value  == 64);
    CHECK(s.allocations.empty()); // the staged-only allocation entry is gone with the abort
    CHECK(ledger.allocation_registry_size() == alloc_baseline);
}

// charge-once: two committed references to one allocation charge the durable bytes once;
// the allocation discharges only when the LAST reference releases
static void test_charge_once() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);
    const size_t alloc_baseline = ledger.allocation_registry_size();

    const auto alloc = ledger.new_alloc();
    CHECK(ledger.allocation_registry_size() == alloc_baseline + 1);
    const auto op1 = ledger.reserve(CAT, DOM, {}, 100, 100);
    const auto op2 = ledger.reserve(CAT, DOM, {}, 100, 100);
    CHECK(ledger.stage(op1, alloc, 100));
    CHECK(ledger.stage(op2, alloc, 100));
    CHECK(ledger.commit(op1, 100));
    CHECK(ledger.commit(op2, 100));

    auto s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::logical_payload).value == 100);
    CHECK(s.allocations.size() == 1);
    CHECK(s.allocations[0].committed_refs == 2);

    // per-reference metadata is a separate leaf, outside the refcount
    ledger.gauge_set(META, DOM,
                     llama_cache_acct_measure::logical_payload, 2 * 16);

    CHECK(ledger.release(op1));
    CHECK(ledger.allocation_registry_size() == alloc_baseline + 1);
    s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::logical_payload).value == 100);

    CHECK(ledger.release(op2));
    CHECK(ledger.allocation_registry_size() == alloc_baseline);
    s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::logical_payload).value == 0);
}

// D-S4: release preview is observational and last-reference-aware. Shared references
// advertise measured zero until the final reference; previewing changes neither serial nor
// fault counters, and the final release applies exactly the advertised delta.
static void test_release_preview() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    const auto alloc = ledger.new_alloc();
    const auto op0 = ledger.reserve(CAT, DOM, {}, 100, 128);
    const auto op1 = ledger.reserve(CAT, DOM, {}, 100, 128);
    CHECK(ledger.stage(op0, alloc, 128));
    CHECK(ledger.stage(op1, alloc, 128));
    CHECK(ledger.commit(op0, 100));
    CHECK(ledger.commit(op1, 100));

    const auto before = ledger.snapshot();
    llama_cache_acct_release_preview preview;
    CHECK(ledger.preview_release(op0, preview));
    CHECK(preview.category == CAT);
    CHECK(preview.domain == DOM);
    CHECK(preview.logical_payload.state == llama_cache_acct_known::known);
    CHECK(preview.logical_payload.value == 0);
    CHECK(preview.resident_allocated.state == llama_cache_acct_known::known);
    CHECK(preview.resident_allocated.value == 0);
    const auto after_preview = ledger.snapshot();
    CHECK(after_preview.serial == before.serial);
    CHECK(after_preview.faults_invalid_transition == before.faults_invalid_transition);
    CHECK(after_preview.faults_unknown_id == before.faults_unknown_id);

    CHECK(ledger.release(op0));
    CHECK(ledger.preview_release(op1, preview));
    CHECK(preview.logical_payload.value == 100);
    CHECK(preview.resident_allocated.value == 128);
    CHECK(ledger.release(op1));

    const auto after = ledger.snapshot();
    CHECK(cell(after, CAT, DOM, llama_cache_acct_measure::logical_payload).value == 0);
    CHECK(cell(after, CAT, DOM, llama_cache_acct_measure::resident_allocated).value == 0);

    const auto faults_before = after.faults_unknown_id;
    CHECK(!ledger.preview_release(op1, preview));
    CHECK(ledger.snapshot().faults_unknown_id == faults_before);
}

// D-S6 SEAM B: a set preview applies last-reference accounting over the entire
// selected union. Neither individual reference frees a shared allocation, while
// selecting both does. The query is serial-bound and remains observational.
static void test_release_set_preview() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    const auto alloc = ledger.new_alloc();
    const auto op0 = ledger.reserve(CAT, DOM, {}, 100, 128);
    const auto op1 = ledger.reserve(CAT, DOM, {}, 100, 128);
    CHECK(ledger.stage(op0, alloc, 128));
    CHECK(ledger.stage(op1, alloc, 128));
    CHECK(ledger.commit(op0, 100));
    CHECK(ledger.commit(op1, 100));

    const auto before = ledger.snapshot();
    llama_cache_acct_release_set_preview preview;
    CHECK(ledger.preview_release_set({ op0 }, before.serial, preview));
    CHECK(preview.accounting_serial == before.serial);
    CHECK(preview.rows.empty());
    CHECK(ledger.preview_release_set({ op1 }, before.serial, preview));
    CHECK(preview.rows.empty());

    CHECK(ledger.preview_release_set({ op0, op1 }, before.serial, preview));
    CHECK(preview.rows.size() == 1);
    CHECK(preview.rows[0].domain == DOM);
    CHECK(preview.rows[0].logical_payload == 100);
    CHECK(preview.rows[0].resident_allocated == 128);
    CHECK(preview.yield_rows.empty());
    CHECK(ledger.preview_release_set(
        { op0, op1 }, before.serial, preview, true));
    CHECK(preview.rows.size() == 1);
    CHECK(preview.yield_rows.size() == 1);
    CHECK(preview.yield_rows[0].category == CAT);
    CHECK(preview.yield_rows[0].domain == DOM);
    CHECK(preview.yield_rows[0].logical_payload == 100);
    CHECK(preview.yield_rows[0].resident_allocated == 128);
    CHECK(ledger.snapshot().serial == before.serial);

    CHECK(!ledger.preview_release_set(
        { op0, op0 }, before.serial, preview));
    CHECK(!ledger.preview_release_set(
        { op0, op1 }, before.serial + 1, preview));
    CHECK(ledger.snapshot().serial == before.serial);

    CHECK(ledger.release_set_if_serial(
        { op0, op1 }, before.serial) ==
          llama_cache_conditional_release_status::released);
    const auto terminal = ledger.snapshot();
    CHECK(terminal.live_ops == 0);
    CHECK(cell(terminal, CAT, DOM,
               llama_cache_acct_measure::logical_payload).value == 0);
    CHECK(cell(terminal, CAT, DOM,
               llama_cache_acct_measure::resident_allocated).value == 0);

    // Owner-gone cleanup cannot depend on an optimistic serial fence. It
    // still validates and applies the complete canonical set under one lock.
    const auto cleanup_alloc = ledger.new_alloc();
    const auto cleanup_op = ledger.reserve(CAT, DOM, {}, 7, 8);
    CHECK(ledger.stage(cleanup_op, cleanup_alloc, 8));
    CHECK(ledger.commit(cleanup_op, 7));
    const uint64_t stale_serial = ledger.serial();
    const auto unrelated = ledger.reserve(CAT, DOM, {}, 0, 0);
    CHECK(unrelated);
    CHECK(ledger.abort(unrelated));
    CHECK(ledger.release_set_if_serial({ cleanup_op }, stale_serial) ==
          llama_cache_conditional_release_status::serial_conflict);
    CHECK(ledger.release_set_current({ cleanup_op }) ==
          llama_cache_conditional_release_status::released);
    CHECK(ledger.snapshot().live_ops == 0);

    const auto alloc_a = ledger.new_alloc();
    const auto alloc_b = ledger.new_alloc();
    const auto op_a = ledger.reserve(CAT, DOM, {}, 1, 1);
    const auto op_b = ledger.reserve(CAT, DOM, {}, 1, 1);
    CHECK(ledger.stage(op_a, alloc_a, 1));
    CHECK(ledger.stage(op_b, alloc_b, 1));
    CHECK(ledger.commit(op_a, 1));
    CHECK(ledger.commit(op_b, 1));
    const auto invalid_before = ledger.snapshot();
    CHECK(ledger.release_set_current({ op_b, op_a }) ==
          llama_cache_conditional_release_status::ledger_fault);
    const auto invalid_after = ledger.snapshot();
    CHECK(invalid_after.live_ops == invalid_before.live_ops);
    CHECK(invalid_after.allocations.size() ==
          invalid_before.allocations.size());
    CHECK(ledger.release_set_current({ op_a, op_b }) ==
          llama_cache_conditional_release_status::released);
    CHECK(ledger.snapshot().live_ops == 0);
}

static void test_release_set_resident_batch_cardinality() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);
    constexpr size_t count = 8192;
    std::vector<llama_cache_acct_op_id> ops;
    ops.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto allocation = ledger.new_alloc();
        const auto op = ledger.reserve(CAT, DOM, {}, 1, 1);
        CHECK(allocation.v != 0);
        CHECK(op.v != 0);
        CHECK(ledger.stage(op, allocation, 1));
        CHECK(ledger.commit(op, 1));
        ops.push_back(op);
    }
    std::vector<llama_cache_acct_release_set_view> sets;
    sets.reserve(ops.size());
    for (const auto & op : ops) {
        sets.push_back({ &op, 1 });
    }
    std::vector<uint64_t> resident;
    const auto begin = std::chrono::steady_clock::now();
    CHECK(ledger.preview_release_set_resident_batch(
        sets, ledger.serial(), resident));
    const auto end = std::chrono::steady_clock::now();
    CHECK(resident.size() == count);
    CHECK(std::all_of(resident.begin(), resident.end(),
        [](uint64_t value) { return value == 1; }));
    std::fprintf(stderr,
        "CACHE_RELEASE_BATCH cardinality=%zu elapsed_us=%" PRIu64 "\n",
        count, uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
            end - begin).count()));
    for (const auto op : ops) {
        CHECK(ledger.release(op));
    }
    CHECK(ledger.snapshot().live_ops == 0);
}

// Sol verify-r2 finding 3: a RETIRED allocation id can never name a new physical allocation.
// Terminal entries are reaped, so a stale id is unknown rather than a retained tombstone; monotone
// allocation ids still prevent reuse. The complete live citation tuple remains immutable.
static void test_alloc_no_resurrection() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    const auto a = ledger.new_alloc();
    const auto op1 = ledger.reserve(CAT, DOM, {}, 1, 1);
    CHECK(ledger.stage(op1, a, 1));
    CHECK(ledger.commit(op1, 1));
    CHECK(ledger.release(op1)); // a is now retired

    const auto op2 = ledger.reserve(CAT, DOM, {}, 2, 2);
    auto s0 = ledger.snapshot();
    CHECK(!ledger.stage(op2, a, 2)); // resurrection under a different size -> fault
    auto s1 = ledger.snapshot();
    CHECK(s1.faults_unknown_id == s0.faults_unknown_id + 1);
    CHECK(ledger.abort(op2));

    // identity-field mismatches on a LIVE shared allocation
    const auto b = ledger.new_alloc();
    const auto op3 = ledger.reserve(CAT, DOM, {}, 5, 5);
    const auto op4 = ledger.reserve(CAT, DOM, {}, 5, 5);
    CHECK(ledger.stage(op3, b, 5, llama_cache_acct_artifact_id{1},
                       llama_cache_acct_content_digest{2}, llama_cache_acct_lineage_id{3}));
    CHECK(!ledger.stage(op4, b, 5, llama_cache_acct_artifact_id{9},
                        llama_cache_acct_content_digest{2}, llama_cache_acct_lineage_id{3}));
    CHECK(!ledger.stage(op4, b, 5, llama_cache_acct_artifact_id{1},
                        llama_cache_acct_content_digest{8}, llama_cache_acct_lineage_id{3}));
    CHECK(!ledger.stage(op4, b, 5, llama_cache_acct_artifact_id{1},
                        llama_cache_acct_content_digest{2}, llama_cache_acct_lineage_id{7}));
    CHECK(ledger.stage(op4, b, 5, llama_cache_acct_artifact_id{1},
                       llama_cache_acct_content_digest{2}, llama_cache_acct_lineage_id{3}));
    CHECK(ledger.abort(op3));
    CHECK(ledger.abort(op4));
}

// Sol verify-r3 blocker: an outstanding STAGED claim defers retirement — the exact
// interleaving commit(op1) → stage(op2) → release(op1) → commit(op2) must accept the join,
// keep valid same-tuple citations working, and retire only after both claim kinds drain
static void test_staged_handoff() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    const auto a = ledger.new_alloc();
    const auto op1 = ledger.reserve(CAT, DOM, {}, 4, 4);
    CHECK(ledger.stage(op1, a, 4));
    CHECK(ledger.commit(op1, 4));

    const auto op2 = ledger.reserve(CAT, DOM, {}, 4, 4);
    CHECK(ledger.stage(op2, a, 4));    // staged while op1 holds the committed claim
    CHECK(ledger.release(op1));        // committed refs hit zero, but op2's claim defers retirement
    CHECK(ledger.commit(op2, 4));      // the handoff join is accepted

    auto s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::logical_payload).value == 4);
    CHECK(s.allocations.size() == 1 && s.allocations[0].committed_refs == 1);

    const auto op3 = ledger.reserve(CAT, DOM, {}, 4, 4);
    CHECK(ledger.stage(op3, a, 4));    // valid same-tuple citation still accepted
    CHECK(ledger.commit(op3, 4));
    CHECK(ledger.release(op2));
    CHECK(ledger.release(op3));        // both claim kinds drained -> retired
    CHECK(ledger.allocation_registry_size() == 0);

    const auto op4 = ledger.reserve(CAT, DOM, {}, 4, 4);
    CHECK(!ledger.stage(op4, a, 4));   // reaped id stays dead
    CHECK(ledger.abort(op4));

    s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::logical_payload).value == 0);
    CHECK(s.live_ops == 0);
}

// live_ops in the snapshot proves nothing leaked: zero after every entry's full lifecycle
static void test_live_ops_zero() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    const auto op = ledger.reserve(CAT, DOM, {}, 8, 8);
    CHECK(ledger.snapshot().live_ops == 1);
    CHECK(ledger.stage(op, ledger.new_alloc(), 8));
    CHECK(ledger.commit(op, 8));
    CHECK(ledger.snapshot().live_ops == 1);
    CHECK(ledger.release(op));
    CHECK(ledger.snapshot().live_ops == 0);

    // an aborted op leaves nothing live either
    const auto op2 = ledger.reserve(CAT, DOM, {}, 8, 8);
    CHECK(ledger.abort(op2));
    CHECK(ledger.snapshot().live_ops == 0);
}

// attribution round-trip: a slot-attributed committed allocation appears as a normalized
// row carrying its attribution (the explicit form D/F consume — no private counters)
static void test_attribution_rows() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    llama_cache_acct_attribution attr;
    attr.kind    = llama_cache_acct_attr_kind::slot;
    attr.slot_id = 3;

    const auto op = ledger.reserve(CAT, DOM, attr, 42, 42);
    CHECK(ledger.stage(op, ledger.new_alloc(), 42,
                       llama_cache_acct_artifact_id{7}, llama_cache_acct_content_digest{8},
                       llama_cache_acct_lineage_id{9}));
    CHECK(ledger.commit(op, 42));

    const auto s = ledger.snapshot();
    CHECK(s.allocations.size() == 1);
    CHECK(s.allocations[0].attribution.kind == llama_cache_acct_attr_kind::slot);
    CHECK(s.allocations[0].attribution.slot_id == 3);
    CHECK(s.allocations[0].artifact_identity.v == 7);
    CHECK(s.allocations[0].content_digest.v    == 8);
    CHECK(s.allocations[0].lineage_identity.v  == 9);
    CHECK(ledger.release(op));
}

// checked overflow latches the cell unavailable (never wraps, never zeros retroactively)
static void test_overflow_latch() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    ledger.gauge_set(CAT, DOM, llama_cache_acct_measure::logical_payload, UINT64_MAX - 1);
    const auto op = ledger.reserve(CAT, DOM, {}, 10, 10);
    CHECK(ledger.stage(op, ledger.new_alloc(), 10));
    CHECK(ledger.commit(op, 10)); // the commit records; the CELL faults

    const auto s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::logical_payload).state ==
          llama_cache_acct_known::unavailable);
    CHECK(s.faults_overflow == 1);
}

// Sol verify-r1 finding 5.3: EVERY observable change bumps the serial — fault counters
// included — so the serial is a usable coherence epoch
static void test_serial_on_fault() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    const auto s0 = ledger.snapshot();
    CHECK(!ledger.release(llama_cache_acct_op_id{424242})); // unknown op -> fault
    const auto s1 = ledger.snapshot();
    CHECK(s1.faults_unknown_id == s0.faults_unknown_id + 1);
    CHECK(s1.serial > s0.serial);

    // and mark_unavailable is observable too
    ledger.mark_unavailable(CAT, DOM, llama_cache_acct_measure::logical_payload);
    const auto s2 = ledger.snapshot();
    CHECK(s2.serial > s1.serial);
}

// snapshot serial: one serial per durable change, snapshots are coherent copies
static void test_snapshot_serial() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    const auto s0 = ledger.snapshot();
    CHECK(ledger.serial() == s0.serial);
    const auto op = ledger.reserve(CAT, DOM, {}, 1, 1);
    const auto s1 = ledger.snapshot();
    CHECK(s1.serial > s0.serial);
    CHECK(ledger.serial() == s1.serial);
    CHECK(ledger.stage(op, ledger.new_alloc(), 1));
    CHECK(ledger.commit(op, 1));
    const auto s2 = ledger.snapshot();
    CHECK(s2.serial > s1.serial);
    CHECK(ledger.serial() == s2.serial);
    // the earlier snapshot is an unchanged copy, not a view
    CHECK(cell(s1, CAT, DOM, llama_cache_acct_measure::logical_payload).state ==
          llama_cache_acct_known::unknown);
    CHECK(cell(s2, CAT, DOM, llama_cache_acct_measure::logical_payload).value == 1);
}

static llama_cache_acct_shard_topology test_topology(
        float first = 0.6f, float second = 0.4f) {
    llama_cache_acct_shard_topology topology;
    const std::vector<std::string> devices = { "test-device-a", "test-device-b" };
    const float weights[] = { first, second };
    CHECK(llama_cache_acct_build_shard_topology(
        devices, 1, 0, weights, topology));
    return topology;
}

// C schema v2: device ordinal + topology are part of both the aggregate key and immutable
// allocation tuple. Same-residency device rows must not merge.
static void test_resource_domains_and_tuple() {
    llama_cache_acct_ledger ledger;
    const auto topology = test_topology();
    // Identity-format golden: both tagged digest domains use llama_sha256_writer. Any
    // serialization change must update this alongside the compile-negative type matrix.
    CHECK(hex_digest(topology.device_identities[0].bytes()) ==
          "2c586154a3d30a96477baa6be95e556271c3301e1444f7dbd6e7fa92ca1aa5e7");
    CHECK(hex_digest(topology.device_identities[1].bytes()) ==
          "1807fa77d0209906177f651aa366307ad7cd14272ee0707b0da596600c5bc340");
    CHECK(hex_digest(topology.digest.bytes()) ==
          "a8c6bb23bba435cedb562510d69619d5992e485be274297bd4e2267e5fc2735e");
    const auto topology_scaled = test_topology(6.0f, 4.0f);
    CHECK(topology_scaled == topology); // one normalizer, not caller-specific rounding
    llama_cache_acct_shard_topology unresolved_auto;
    CHECK(!llama_cache_acct_build_shard_topology(
        std::vector<std::string>{ "test-device-a", "test-device-b" },
        1, 0, nullptr, unresolved_auto)); // resolved auto-split weights are required
    llama_cache_acct_resource_domain d0;
    llama_cache_acct_resource_domain d1;
    CHECK(ledger.make_device_domain(topology, { 0 }, d0));
    CHECK(ledger.make_device_domain(topology, { 1 }, d1));
    CHECK(llama_cache_acct_resource_domain_valid(d0));
    CHECK(llama_cache_acct_resource_domain_valid(d1));
    auto stale_topology = topology;
    stale_topology.shard_weights[0]--;
    stale_topology.shard_weights[1]++;
    llama_cache_acct_resource_domain stale_domain;
    CHECK(!ledger.make_device_domain(stale_topology, { 0 }, stale_domain));
    CHECK(!llama_cache_acct_resource_domain_valid(
        llama_cache_acct_resource_domain::non_device(llama_cache_acct_residency::device)));

    const auto changed_topology = test_topology(0.5f, 0.5f);
    llama_cache_acct_resource_domain d0_changed;
    CHECK(ledger.make_device_domain(changed_topology, { 0 }, d0_changed));
    const llama_cache_acct_completeness_requirement requirements[] = {
        { d0,         llama_cache_acct_producer::observer_init },
        { d1,         llama_cache_acct_producer::observer_init },
        { d0_changed, llama_cache_acct_producer::observer_init },
    };
    CHECK(ledger.configure_required_producers(requirements, 3));
    CHECK(ledger.certify_complete(d0, llama_cache_acct_producer::observer_init));
    CHECK(ledger.certify_complete(d1, llama_cache_acct_producer::observer_init));
    CHECK(ledger.certify_complete(d0_changed, llama_cache_acct_producer::observer_init));
    ledger.gauge_set(CAT, d0, llama_cache_acct_measure::logical_payload, 10);
    ledger.gauge_set(CAT, d1, llama_cache_acct_measure::logical_payload, 20);
    auto s = ledger.snapshot();
    CHECK(cell(s, CAT, d0, llama_cache_acct_measure::logical_payload).value == 10);
    CHECK(cell(s, CAT, d1, llama_cache_acct_measure::logical_payload).value == 20);

    const auto alloc = ledger.new_alloc();
    const auto op0 = ledger.reserve(CAT, d0, {}, 4, 4);
    const auto op1 = ledger.reserve(CAT, d1, {}, 4, 4);
    const auto op2 = ledger.reserve(CAT, d0_changed, {}, 4, 4);
    CHECK(ledger.stage(op0, alloc, 4));
    CHECK(!ledger.stage(op1, alloc, 4)); // ordinal/domain mismatch is a tuple fault
    CHECK(!ledger.stage(op2, alloc, 4)); // topology change is a tuple fault
    CHECK(ledger.abort(op0));
    CHECK(ledger.abort(op1));
    CHECK(ledger.abort(op2));
}

// Completeness is certified per (resource-domain, producer), from a configuration-owned
// manifest. One producer cannot certify another domain, and unavailable is monotone.
static void test_per_domain_completeness() {
    llama_cache_acct_ledger ledger;
    const auto n_a = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::not_applicable);
    const llama_cache_acct_completeness_requirement required[] = {
        { n_a, llama_cache_acct_producer::observer_init },
        { DOM, llama_cache_acct_producer::host_cache },
    };
    CHECK(ledger.configure_required_producers(required, 2));
    auto s = ledger.snapshot();
    CHECK(s.completeness_manifest == llama_cache_acct_known::known);
    CHECK(s.completeness.size() == 2);
    CHECK(s.completeness[0].state == llama_cache_acct_known::unknown);
    CHECK(s.completeness[1].state == llama_cache_acct_known::unknown);

    // A manifested-but-uncertified producer may observe internally, but the snapshot
    // projects its cells unavailable until certification.
    ledger.gauge_set(CAT, DOM, llama_cache_acct_measure::logical_payload, 17);
    s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::logical_payload).state ==
          llama_cache_acct_known::unavailable);

    CHECK(ledger.certify_complete(n_a, llama_cache_acct_producer::observer_init));
    s = ledger.snapshot();
    CHECK(s.completeness[0].state == llama_cache_acct_known::known);
    CHECK(s.completeness[1].state == llama_cache_acct_known::unknown);

    const auto before = s.faults_invalid_transition;
    CHECK(!ledger.certify_complete(DOM, llama_cache_acct_producer::observer_init));
    CHECK(ledger.snapshot().faults_invalid_transition == before + 1);

    // An unmanifested domain is rejected at the producer boundary, not left for a later
    // consumer to discover while summing a partial snapshot.
    const auto unmanifested = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::disk);
    CHECK(!ledger.reserve(CAT, unmanifested, {}, 1, 1));

    ledger.mark_producer_unavailable(DOM, llama_cache_acct_producer::host_cache);
    CHECK(!ledger.certify_complete(DOM, llama_cache_acct_producer::host_cache));
    s = ledger.snapshot();
    CHECK(s.completeness[1].state == llama_cache_acct_known::unavailable);
}

// The only schema-v1 path is an explicit adapter. Non-device rows project exactly; any
// topology-qualified row fails closed instead of flattening device ordinals.
static void test_v1_adapter_fail_closed() {
    llama_cache_acct_ledger host_ledger;
    configure_default(host_ledger);
    host_ledger.gauge_set(CAT, DOM, llama_cache_acct_measure::logical_payload, 7);
    const auto op = host_ledger.reserve(META, DOM, {}, 3, 3);
    CHECK(host_ledger.stage(op, host_ledger.new_alloc(), 3));
    CHECK(host_ledger.commit(op, 3));
    llama_cache_acct_snapshot_v1 v1;
    CHECK(llama_cache_acct_snapshot_to_v1(host_ledger.snapshot(), v1));
    CHECK(v1.cells[size_t(CAT)][size_t(RES)]
              .measures[size_t(llama_cache_acct_measure::logical_payload)].value == 7);
    CHECK(v1.allocations.size() == 1);
    CHECK(v1.allocations[0].residency == RES);
    CHECK(host_ledger.release(op));

    llama_cache_acct_ledger device_ledger;
    llama_cache_acct_shard_topology one_device;
    CHECK(llama_cache_acct_build_shard_topology(
        std::vector<std::string>{ "test-device-a" }, 1, 0, nullptr, one_device));
    llama_cache_acct_resource_domain d0;
    CHECK(device_ledger.make_device_domain(one_device, { 0 }, d0));
    configure_domain(device_ledger, d0);
    device_ledger.gauge_set(CAT, d0, llama_cache_acct_measure::logical_payload, 9);
    CHECK(!llama_cache_acct_snapshot_to_v1(device_ledger.snapshot(), v1));
    CHECK(v1.completeness == llama_cache_acct_known::unavailable);
}

// F0a conditional reserve — serial matches: admits exactly like reserve(), no conflict counted.
static void test_reserve_if_serial_match() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    const uint64_t serial = ledger.snapshot().serial;
    const auto r = ledger.reserve_if_serial(serial, CAT, DOM, {}, 100, 128);
    CHECK(r.status == llama_cache_conditional_reserve_status::admitted);
    CHECK(bool(r.op));
    const auto s = ledger.snapshot();
    CHECK(cell(s, CAT, DOM, llama_cache_acct_measure::reserved).value == 128);
    CHECK(ledger.serial_conflicts() == 0);
    CHECK(s.faults_invalid_transition == 0 && s.faults_overflow == 0);
}

// F0a conditional reserve — serial drifted: refuses with the ledger COMPLETELY untouched (no op,
// no cell change, no serial bump, no fault), only the process-local conflict counter moves.
static void test_reserve_if_serial_drift() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    const uint64_t stale = ledger.snapshot().serial;
    CHECK(bool(ledger.reserve(CAT, DOM, {}, 10, 10)));   // advance the ledger past `stale`
    const auto before = ledger.snapshot();

    const auto r = ledger.reserve_if_serial(stale, CAT, DOM, {}, 100, 128);
    CHECK(r.status == llama_cache_conditional_reserve_status::serial_conflict);
    CHECK(!bool(r.op));

    const auto after = ledger.snapshot();
    CHECK(after.serial == before.serial);              // no bump on drift
    CHECK(cell(after, CAT, DOM, llama_cache_acct_measure::reserved).value ==
          cell(before, CAT, DOM, llama_cache_acct_measure::reserved).value);
    CHECK(after.faults_invalid_transition == before.faults_invalid_transition);
    CHECK(after.faults_overflow == before.faults_overflow);
    CHECK(ledger.serial_conflicts() == 1);             // conflict is telemetry, not a fault
}

// F0a conditional reserve — serial matches but the reservation itself hard-faults (unmanifested
// domain): ledger_fault, distinct from a conflict.
static void test_reserve_if_serial_ledger_fault() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    const auto bad = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::not_applicable);   // never configured on this ledger
    const uint64_t serial = ledger.snapshot().serial;
    const auto r = ledger.reserve_if_serial(serial, CAT, bad, {}, 100, 128);
    CHECK(r.status == llama_cache_conditional_reserve_status::ledger_fault);
    CHECK(!bool(r.op));
    CHECK(ledger.serial_conflicts() == 0);             // a fault is not a conflict
    CHECK(ledger.snapshot().faults_invalid_transition >= 1);
}

// F0a conditional reserve — two admissions priced against one serial: exactly one wins, the other
// sees the drift caused by the first (no fault). The conflict must consume NO op id: a following
// fresh-serial reservation gets the id immediately after the winner's (proves next_op preserved).
static void test_reserve_if_serial_serial_reuse() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);

    const uint64_t serial = ledger.snapshot().serial;
    const auto a = ledger.reserve_if_serial(serial, CAT, DOM, {}, 100, 128);
    const auto b = ledger.reserve_if_serial(serial, CAT, DOM, {}, 100, 128);
    CHECK(a.status == llama_cache_conditional_reserve_status::admitted);
    CHECK(b.status == llama_cache_conditional_reserve_status::serial_conflict);
    CHECK(ledger.serial_conflicts() == 1);
    CHECK(ledger.snapshot().faults_invalid_transition == 0);

    const auto c = ledger.reserve_if_serial(ledger.snapshot().serial, CAT, DOM, {}, 100, 128);
    CHECK(c.status == llama_cache_conditional_reserve_status::admitted);
    CHECK(c.op.v == a.op.v + 1); // b consumed no op id
}

// F0a conditional reserve — a reservation the ledger cannot record must be ledger_fault BEFORE an op
// is minted (Sol review blocker 2): a reserved-aggregate overflow, and an already-unavailable
// reserved cell. Neither may consume an op id or report admitted.
static void test_reserve_if_serial_unrecordable() {
    // (a) reserved overflow: prime reserved near the ceiling, then a big reservation would overflow.
    {
        llama_cache_acct_ledger ledger;
        configure_default(ledger);
        ledger.gauge_set(CAT, DOM, llama_cache_acct_measure::reserved,
                         std::numeric_limits<uint64_t>::max() - 10);
        const auto r = ledger.reserve_if_serial(ledger.snapshot().serial, CAT, DOM, {}, 100, 128);
        CHECK(r.status == llama_cache_conditional_reserve_status::ledger_fault);
        CHECK(!bool(r.op));
        CHECK(ledger.snapshot().faults_overflow >= 1);
        // no op consumed: a valid reservation now gets op id 1
        ledger.gauge_set(CAT, DOM, llama_cache_acct_measure::reserved, 0);
        const auto ok = ledger.reserve_if_serial(ledger.snapshot().serial, CAT, DOM, {}, 1, 1);
        CHECK(ok.status == llama_cache_conditional_reserve_status::admitted);
        CHECK(ok.op.v == 1);
    }
    // (b) already-unavailable reserved cell: reserving into it is a hard fault, not a silent admit.
    {
        llama_cache_acct_ledger ledger;
        configure_default(ledger);
        ledger.mark_unavailable(CAT, DOM, llama_cache_acct_measure::reserved);
        const auto r = ledger.reserve_if_serial(ledger.snapshot().serial, CAT, DOM, {}, 100, 128);
        CHECK(r.status == llama_cache_conditional_reserve_status::ledger_fault);
        CHECK(!bool(r.op));
    }
}

// F0a conditional reserve — under REAL concurrency the serial guard is the admission gate: two
// threads price against one serial and race; exactly one is admitted, the other conflicts, no fault.
static void test_reserve_if_serial_threads() {
    llama_cache_acct_ledger ledger;
    configure_default(ledger);
    const uint64_t serial = ledger.snapshot().serial;

    std::atomic<bool> go{false};
    llama_cache_conditional_reserve_result results[2];
    auto worker = [&](int i) {
        while (!go.load(std::memory_order_acquire)) { /* spin to maximize overlap */ }
        results[i] = ledger.reserve_if_serial(serial, CAT, DOM, {}, 100, 128);
    };
    std::thread t0(worker, 0), t1(worker, 1);
    go.store(true, std::memory_order_release);
    t0.join();
    t1.join();

    int admitted = 0, conflicts = 0;
    for (const auto & r : results) {
        admitted  += r.status == llama_cache_conditional_reserve_status::admitted;
        conflicts += r.status == llama_cache_conditional_reserve_status::serial_conflict;
    }
    CHECK(admitted == 1);
    CHECK(conflicts == 1);
    CHECK(ledger.serial_conflicts() == 1);
    CHECK(ledger.snapshot().faults_invalid_transition == 0 &&
          ledger.snapshot().faults_overflow == 0);
}

int main() {
    test_lifecycle();
    test_reserve_stage_mismatch();
    test_concurrent_peak();
    test_invalid_transitions();
    test_alloc_tuple_immutable();
    test_alloc_no_resurrection();
    test_staged_handoff();
    test_live_ops_zero();
    test_abort_retains_peak();
    test_charge_once();
    test_release_preview();
    test_release_set_preview();
    test_release_set_resident_batch_cardinality();
    test_attribution_rows();
    test_overflow_latch();
    test_serial_on_fault();
    test_snapshot_serial();
    test_resource_domains_and_tuple();
    test_per_domain_completeness();
    test_v1_adapter_fail_closed();
    test_reserve_if_serial_match();
    test_reserve_if_serial_drift();
    test_reserve_if_serial_ledger_fault();
    test_reserve_if_serial_serial_reuse();
    test_reserve_if_serial_unrecordable();
    test_reserve_if_serial_threads();

    if (failures > 0) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    printf("all cache-accounting tests passed\n");
    return EXIT_SUCCESS;
}
