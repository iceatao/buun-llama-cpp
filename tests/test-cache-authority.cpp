// F0a admission-composer + reservation-claim contract tests. The composer is the single place
// that runs snapshot -> fits(reserve-only) -> reserve_if_serial; these tests pin its honest status
// taxonomy (fail-closed on incomplete evidence; no fabricated precision) and the move-only claim's
// auto-abort so an admitted-but-uncommitted reservation can never leak. The reserve_if_serial and
// fits primitives themselves are covered by test-cache-accounting / test-cache-budget.

#include "llama-cache-authority.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

static const auto HOST = llama_cache_acct_resource_domain::non_device(
    llama_cache_acct_residency::pageable_host);
static const auto PAYLOAD = llama_cache_acct_category::full_snapshot_payload;

// Build a ledger with one certified, available durable host cell — mirrors the server's host-cache
// setup order (configure required producers -> create the transactional leaf -> certify). A host
// leaf has an unbounded budget ceiling, so a small reservation against it fits under default caps.
static void configure_fitting_host(llama_cache_acct_ledger & ledger) {
    const llama_cache_acct_completeness_requirement req = {
        HOST, llama_cache_acct_producer::host_cache,
    };
    CHECK(ledger.configure_required_producers(&req, 1));
    // fits() marks a domain unavailable unless every durable+host cell has known resident AND
    // (being transactional) known reserved bytes, so give all three durable+host payload leaves a
    // measured zero on each transactional measure (exactly as the server's host init does).
    for (const auto cat : { llama_cache_acct_category::full_snapshot_payload,
                            llama_cache_acct_category::checkpoint_state_payload,
                            llama_cache_acct_category::typed_accelerator_payload }) {
        for (const auto measure : { llama_cache_acct_measure::logical_payload,
                                    llama_cache_acct_measure::resident_allocated,
                                    llama_cache_acct_measure::reserved }) {
            ledger.gauge_set(cat, HOST, measure, 0);
        }
    }
    CHECK(ledger.certify_complete(HOST, llama_cache_acct_producer::host_cache));
}

static llama_cache_authority_request host_request(uint64_t resident) {
    llama_cache_authority_request req;
    req.category          = PAYLOAD;
    req.domain            = HOST;
    req.expected_logical  = resident;
    req.expected_resident = resident;
    return req;
}

// Configure a fresh host ledger and run one admission against it. The ledger is caller-owned so the
// returned claim's lifetime (and the live_ops assertions) stay in the test's own scope.
static llama_cache_admission_result admit_fresh_host(
        llama_cache_acct_ledger & ledger, uint64_t resident) {
    configure_fitting_host(ledger);
    const llama_cache_budget_config config; // empty devices + default unbounded host caps
    return llama_cache_admit_reservation(ledger, config, host_request(resident));
}

// An unconfigured ledger has a non-known completeness manifest: the composer must refuse before it
// prices anything (fail-closed), never admit on private counters.
static void test_incomplete_evidence() {
    llama_cache_acct_ledger ledger;
    llama_cache_budget_config config;
    const auto res = llama_cache_admit_reservation(ledger, config, host_request(128));
    CHECK(res.status == llama_cache_admission_status::incomplete_evidence);
    CHECK(!res.claim.has_op());
    CHECK(ledger.snapshot().live_ops == 0);
}

// Manifest is known but the priced domain has no budget-visible cell (no gauge): fits() collapses
// to unavailable and the composer reports budget_unavailable — NOT incomplete_evidence.
static void test_budget_unavailable() {
    llama_cache_acct_ledger ledger;
    const llama_cache_acct_completeness_requirement req = {
        HOST, llama_cache_acct_producer::host_cache,
    };
    CHECK(ledger.configure_required_producers(&req, 1));
    CHECK(ledger.certify_complete(HOST, llama_cache_acct_producer::host_cache));
    // No gauge_set: the manifest is known, but HOST owns no durable cell to price against.
    CHECK(ledger.snapshot().completeness_manifest == llama_cache_acct_known::known);

    llama_cache_budget_config config;
    const auto res = llama_cache_admit_reservation(ledger, config, host_request(128));
    CHECK(res.status == llama_cache_admission_status::budget_unavailable);
    CHECK(!res.claim.has_op());
    CHECK(ledger.snapshot().live_ops == 0);
}

static uint64_t host_reserved(llama_cache_acct_ledger & ledger) {
    for (const auto & row : ledger.snapshot().cells) {
        if (row.category == PAYLOAD && row.domain == HOST) {
            return row.cell.measures[size_t(llama_cache_acct_measure::reserved)].value;
        }
    }
    return 0;
}

static uint64_t host_category_measure(
        llama_cache_acct_ledger & ledger,
        llama_cache_acct_category category,
        llama_cache_acct_measure measure) {
    for (const auto & row : ledger.snapshot().cells) {
        if (row.category == category && row.domain == HOST) {
            return row.cell.measures[size_t(measure)].value;
        }
    }
    return 0;
}

static uint64_t host_measure(
        llama_cache_acct_ledger & ledger,
        llama_cache_acct_measure measure) {
    return host_category_measure(ledger, PAYLOAD, measure);
}

// Happy path: the reservation fits and is admitted, handing back a claim that owns the reserved op,
// and the reserved aggregate actually moved by the requested amount (not just live_ops).
static void test_admitted() {
    llama_cache_acct_ledger ledger;
    const auto res = admit_fresh_host(ledger, 128);
    CHECK(res.status == llama_cache_admission_status::admitted);
    CHECK(res.claim.has_op());
    CHECK(ledger.snapshot().live_ops == 1);
    CHECK(host_reserved(ledger) == 128);
}

// Manifest known + host cell present, but a host total cap below the request: fits() reports exceeds
// and the composer maps it to exceeds_budget (distinct from budget_unavailable), reserving nothing.
static void test_exceeds_budget() {
    llama_cache_acct_ledger ledger;
    configure_fitting_host(ledger);
    llama_cache_budget_config config;
    config.host.total_state = llama_cache_budget_capacity_state::known;
    config.host.total_cap   = 100;

    const auto res = llama_cache_admit_reservation(ledger, config, host_request(128));
    CHECK(res.status == llama_cache_admission_status::exceeds_budget);
    CHECK(!res.claim.has_op());
    CHECK(ledger.snapshot().live_ops == 0);
}

// A dropped claim (the only F0a terminal — there is no bare discharge) aborts its reserved op, so an
// admitted-but-abandoned reservation leaves the ledger with zero live ops — the leak guard for F0b.
static void test_claim_auto_abort() {
    llama_cache_acct_ledger ledger;
    {
        const auto res = admit_fresh_host(ledger, 64);
        CHECK(res.status == llama_cache_admission_status::admitted);
        CHECK(ledger.snapshot().live_ops == 1);
    } // res.claim destroyed here -> op aborted
    CHECK(ledger.snapshot().live_ops == 0);
}

// Move-construction transfers ownership; the moved-from claim is inert; exactly one abort fires.
static void test_claim_move_ctor() {
    llama_cache_acct_ledger ledger;
    auto res = admit_fresh_host(ledger, 64);
    CHECK(res.status == llama_cache_admission_status::admitted);
    {
        llama_cache_reservation_claim b(std::move(res.claim));
        CHECK(!res.claim.has_op());
        CHECK(b.has_op());
        CHECK(ledger.snapshot().live_ops == 1);
    }
    CHECK(ledger.snapshot().live_ops == 0); // only b aborted
}

// Move-assignment aborts the destination's own op before taking the source's; self-move is a no-op.
static void test_claim_move_assign() {
    llama_cache_acct_ledger ledger;
    auto dst = admit_fresh_host(ledger, 64);
    llama_cache_budget_config config; // ledger already configured by admit_fresh_host
    auto src = llama_cache_admit_reservation(ledger, config, host_request(64));
    CHECK(dst.status == llama_cache_admission_status::admitted);
    CHECK(src.status == llama_cache_admission_status::admitted);
    CHECK(ledger.snapshot().live_ops == 2);

    dst.claim = std::move(src.claim);
    CHECK(!src.claim.has_op());
    CHECK(dst.claim.has_op());
    CHECK(ledger.snapshot().live_ops == 1); // dst's original op was aborted

    // self-move (through a pointer so the compiler cannot warn): keeps its op.
    llama_cache_reservation_claim * self = &dst.claim;
    *self = std::move(*self);
    CHECK(dst.claim.has_op());
    CHECK(ledger.snapshot().live_ops == 1);
}

// The only non-abort terminal commits through the owning claim. Success disarms it and hands the
// committed id to the durable artifact; destroying the former claim must not abort that transaction.
static void test_claim_commit_terminal() {
    llama_cache_acct_ledger ledger;
    llama_cache_acct_op_id committed;
    {
        auto res = admit_fresh_host(ledger, 64);
        CHECK(res.status == llama_cache_admission_status::admitted);
        const auto alloc = ledger.new_alloc();
        CHECK(bool(alloc));
        CHECK(ledger.stage(res.claim.op(), alloc, 64));
        CHECK(res.claim.commit(64, committed));
        CHECK(bool(committed));
        CHECK(!res.claim.has_op());

        llama_cache_acct_op_id duplicate = { 99 };
        CHECK(!res.claim.commit(64, duplicate));
        CHECK(!bool(duplicate));
    }
    CHECK(ledger.snapshot().live_ops == 1);
    CHECK(host_reserved(ledger) == 0);
    CHECK(host_measure(ledger, llama_cache_acct_measure::logical_payload) == 64);
    CHECK(host_measure(ledger, llama_cache_acct_measure::resident_allocated) == 64);

    // release() is the ledger's existing committed-transaction rollback/retirement primitive.
    CHECK(ledger.release(committed));
    CHECK(ledger.snapshot().live_ops == 0);
    CHECK(host_measure(ledger, llama_cache_acct_measure::logical_payload) == 0);
    CHECK(host_measure(ledger, llama_cache_acct_measure::resident_allocated) == 0);
}

// A failed commit does not disarm the claim. The destructor can still abort its reserved op,
// preventing a double-terminal or failed-commit leak.
static void test_claim_commit_failure_auto_abort() {
    llama_cache_acct_ledger ledger;
    {
        auto res = admit_fresh_host(ledger, 64);
        llama_cache_acct_op_id committed;
        CHECK(!res.claim.commit(64, committed)); // not staged
        CHECK(!bool(committed));
        CHECK(res.claim.has_op());
    }
    CHECK(ledger.snapshot().live_ops == 0);
    CHECK(host_reserved(ledger) == 0);
}

static llama_cache_transaction_leaf transaction_leaf(
        llama_cache_acct_category category,
        uint64_t bytes,
        llama_cache_acct_op_id & committed,
        llama_cache_acct_alloc_id & allocation) {
    llama_cache_transaction_leaf leaf;
    leaf.category = category;
    leaf.domain = HOST;
    leaf.expected_logical = bytes;
    leaf.reserve_resident = bytes;
    leaf.stage_resident = bytes;
    leaf.committed_op = &committed;
    leaf.allocation_out = &allocation;
    return leaf;
}

struct after_admit_probe {
    llama_cache_acct_ledger * ledger = nullptr;
    uint64_t expected_live_ops = 0;
    uint64_t calls = 0;
    bool return_value = true;
};

static bool observe_after_admit(void * opaque) {
    auto & probe = *static_cast<after_admit_probe *>(opaque);
    probe.calls++;
    return probe.ledger &&
        probe.ledger->snapshot().live_ops == probe.expected_live_ops &&
        probe.return_value;
}

// The shared transaction reserves every leaf before the optional preparation hook, then stages and
// commits the group. Outputs are published only after the whole group has reached its terminal.
static void test_transaction_fresh_group() {
    llama_cache_acct_ledger ledger;
    configure_fitting_host(ledger);
    llama_cache_budget_config config;
    llama_cache_acct_op_id committed[2];
    llama_cache_acct_alloc_id allocations[2];
    std::vector<llama_cache_transaction_leaf> leaves {
        transaction_leaf(
            llama_cache_acct_category::full_snapshot_payload,
            40, committed[0], allocations[0]),
        transaction_leaf(
            llama_cache_acct_category::checkpoint_state_payload,
            24, committed[1], allocations[1]),
    };
    after_admit_probe probe { &ledger, 2, 0, true };
    const llama_cache_transaction_after_admit hook {
        &probe, observe_after_admit,
    };

    const auto result = llama_cache_execute_reservation_transaction(
        ledger, config, leaves, {}, hook);
    CHECK(result.status == llama_cache_transaction_status::committed);
    CHECK(result.admission_status == llama_cache_admission_status::admitted);
    CHECK(result.rolled_back == 0);
    CHECK(probe.calls == 1);
    CHECK(bool(committed[0]));
    CHECK(bool(committed[1]));
    CHECK(bool(allocations[0]));
    CHECK(bool(allocations[1]));
    CHECK(allocations[0] != allocations[1]);
    CHECK(ledger.snapshot().live_ops == 2);
    CHECK(host_category_measure(
        ledger, llama_cache_acct_category::full_snapshot_payload,
        llama_cache_acct_measure::resident_allocated) == 40);
    CHECK(host_category_measure(
        ledger, llama_cache_acct_category::checkpoint_state_payload,
        llama_cache_acct_measure::resident_allocated) == 24);

    CHECK(ledger.release(committed[0]));
    CHECK(ledger.release(committed[1]));
    CHECK(ledger.snapshot().live_ops == 0);
}

// An existing-allocation leaf reserves no new resident capacity but stages the immutable allocation's
// full tuple to acquire a second committed reference. First/last-reference charging stays authoritative.
static void test_transaction_existing_allocation() {
    llama_cache_acct_ledger ledger;
    configure_fitting_host(ledger);
    llama_cache_budget_config config;
    llama_cache_acct_op_id first_op;
    llama_cache_acct_alloc_id allocation;
    std::vector<llama_cache_transaction_leaf> fresh {
        transaction_leaf(PAYLOAD, 64, first_op, allocation),
    };
    CHECK(llama_cache_execute_reservation_transaction(
        ledger, config, fresh).status ==
            llama_cache_transaction_status::committed);

    llama_cache_acct_op_id joined_op;
    llama_cache_acct_alloc_id joined_allocation;
    auto joined = transaction_leaf(
        PAYLOAD, 64, joined_op, joined_allocation);
    joined.reserve_resident = 0;
    joined.existing_allocation = allocation;
    std::vector<llama_cache_transaction_leaf> adopted { joined };
    CHECK(llama_cache_execute_reservation_transaction(
        ledger, config, adopted).status ==
            llama_cache_transaction_status::committed);
    CHECK(joined_allocation == allocation);
    CHECK(host_measure(
        ledger, llama_cache_acct_measure::resident_allocated) == 64);

    CHECK(ledger.release(first_op));
    CHECK(host_measure(
        ledger, llama_cache_acct_measure::resident_allocated) == 64);
    CHECK(ledger.release(joined_op));
    CHECK(host_measure(
        ledger, llama_cache_acct_measure::resident_allocated) == 0);
    CHECK(ledger.snapshot().live_ops == 0);
}

// A group fault releases every already-committed leaf, lets the remaining claims abort, and leaves
// success-only output destinations untouched. This is the rollback contract both F0b and F2 consume.
static void test_transaction_fault_rollback() {
    llama_cache_acct_ledger ledger;
    configure_fitting_host(ledger);
    llama_cache_budget_config config;
    llama_cache_acct_op_id committed[2] { { 901 }, { 902 } };
    llama_cache_acct_alloc_id allocations[2] { { 801 }, { 802 } };
    std::vector<llama_cache_transaction_leaf> leaves {
        transaction_leaf(
            llama_cache_acct_category::full_snapshot_payload,
            40, committed[0], allocations[0]),
        transaction_leaf(
            llama_cache_acct_category::checkpoint_state_payload,
            24, committed[1], allocations[1]),
    };
    llama_cache_transaction_fault fault;
    fault.fail_commit_at = 1;

    const auto result = llama_cache_execute_reservation_transaction(
        ledger, config, leaves, fault);
    CHECK(result.status == llama_cache_transaction_status::commit_failed);
    CHECK(result.admission_status ==
          llama_cache_admission_status::internal_fault);
    CHECK(result.failed_leaf == 1);
    CHECK(result.rolled_back == 1);
    CHECK(committed[0].v == 901);
    CHECK(committed[1].v == 902);
    CHECK(allocations[0].v == 801);
    CHECK(allocations[1].v == 802);
    CHECK(ledger.snapshot().live_ops == 0);
    CHECK(host_category_measure(
        ledger, llama_cache_acct_category::full_snapshot_payload,
        llama_cache_acct_measure::resident_allocated) == 0);
    CHECK(host_category_measure(
        ledger, llama_cache_acct_category::checkpoint_state_payload,
        llama_cache_acct_measure::resident_allocated) == 0);

    fault.fail_commit_at = UINT32_MAX;
    fault.fail_after_commit = true;
    const auto post_commit =
        llama_cache_execute_reservation_transaction(
            ledger, config, leaves, fault);
    CHECK(post_commit.status ==
          llama_cache_transaction_status::post_commit_fault);
    CHECK(post_commit.admission_status ==
          llama_cache_admission_status::internal_fault);
    CHECK(post_commit.rolled_back == leaves.size());
    CHECK(ledger.snapshot().live_ops == 0);
}

// A preparation failure occurs only after all claims exist and before any stage. The claims' RAII
// destructors return the ledger to its original state, and no output is exposed.
static void test_transaction_after_admit_failure() {
    llama_cache_acct_ledger ledger;
    configure_fitting_host(ledger);
    llama_cache_budget_config config;
    llama_cache_acct_op_id committed { 901 };
    llama_cache_acct_alloc_id allocation { 801 };
    std::vector<llama_cache_transaction_leaf> leaves {
        transaction_leaf(PAYLOAD, 64, committed, allocation),
    };
    after_admit_probe probe { &ledger, 1, 0, false };
    const llama_cache_transaction_after_admit hook {
        &probe, observe_after_admit,
    };

    const auto result = llama_cache_execute_reservation_transaction(
        ledger, config, leaves, {}, hook);
    CHECK(result.status ==
        llama_cache_transaction_status::after_admit_failed);
    CHECK(probe.calls == 1);
    CHECK(committed.v == 901);
    CHECK(allocation.v == 801);
    CHECK(ledger.snapshot().live_ops == 0);
    CHECK(host_reserved(ledger) == 0);
}

static void test_prepared_transaction_split_phase() {
    llama_cache_acct_ledger ledger;
    configure_fitting_host(ledger);
    llama_cache_budget_config config;
    llama_cache_acct_op_id committed { 901 };
    llama_cache_acct_alloc_id allocation { 801 };
    std::vector<llama_cache_transaction_leaf> leaves {
        transaction_leaf(PAYLOAD, 64, committed, allocation),
    };

    {
        auto dropped =
            llama_cache_prepare_reservation_transaction(
                ledger, config, leaves);
        CHECK(dropped.ready());
        CHECK(dropped.preparation().status ==
              llama_cache_prepare_status::prepared);
        CHECK(ledger.snapshot().live_ops == 1);
        CHECK(host_reserved(ledger) == 64);
    }
    CHECK(ledger.snapshot().live_ops == 0);
    CHECK(host_reserved(ledger) == 0);

    auto prepared =
        llama_cache_prepare_reservation_transaction(
            ledger, config, leaves);
    CHECK(prepared.ready());
    auto changed = leaves;
    changed[0].stage_resident++;
    const auto refused =
        prepared.materialize_and_commit(changed);
    CHECK(refused.status ==
          llama_cache_transaction_status::invalid_argument);
    CHECK(prepared.ready());
    CHECK(ledger.snapshot().live_ops == 1);

    const auto committed_result =
        prepared.materialize_and_commit(leaves);
    CHECK(committed_result.status ==
          llama_cache_transaction_status::committed);
    CHECK(!prepared.ready());
    CHECK(committed.v != 901);
    CHECK(allocation.v != 801);
    CHECK(ledger.snapshot().live_ops == 1);
    CHECK(ledger.release(committed));
    CHECK(ledger.snapshot().live_ops == 0);
}

static void test_prepared_transaction_downward_repartition() {
    llama_cache_acct_ledger ledger;
    configure_fitting_host(ledger);
    llama_cache_budget_config config;
    llama_cache_acct_op_id initial_op;
    llama_cache_acct_alloc_id initial_allocation;
    std::vector<llama_cache_transaction_leaf> initial {
        transaction_leaf(PAYLOAD, 96, initial_op, initial_allocation),
    };
    auto prepared = llama_cache_prepare_reservation_transaction(
        ledger, config, initial);
    CHECK(prepared.ready());
    CHECK(ledger.snapshot().live_ops == 1);
    CHECK(host_reserved(ledger) == 96);

    llama_cache_acct_op_id first_op;
    llama_cache_acct_op_id second_op;
    llama_cache_acct_alloc_id first_allocation;
    llama_cache_acct_alloc_id second_allocation;
    std::vector<llama_cache_transaction_leaf> growth {
        transaction_leaf(PAYLOAD, 64, first_op, first_allocation),
        transaction_leaf(PAYLOAD, 33, second_op, second_allocation),
    };
    CHECK(!prepared.repartition_downward(growth));
    CHECK(prepared.ready());
    CHECK(ledger.snapshot().live_ops == 1);
    CHECK(host_reserved(ledger) == 96);

    llama_cache_acct_op_id foreign_op;
    llama_cache_acct_alloc_id foreign_allocation;
    auto foreign = transaction_leaf(
        llama_cache_acct_category::checkpoint_state_payload,
        0, foreign_op, foreign_allocation);
    foreign.existing_allocation = { 777 };
    std::vector<llama_cache_transaction_leaf> foreign_domain {
        std::move(foreign),
    };
    CHECK(!prepared.repartition_downward(foreign_domain));
    CHECK(prepared.ready());
    CHECK(ledger.snapshot().live_ops == 1);
    CHECK(host_reserved(ledger) == 96);

    std::vector<llama_cache_transaction_leaf> exact {
        transaction_leaf(PAYLOAD, 40, first_op, first_allocation),
        transaction_leaf(PAYLOAD, 20, second_op, second_allocation),
    };
    CHECK(prepared.repartition_downward(exact));
    CHECK(prepared.ready());
    CHECK(ledger.snapshot().live_ops == 2);
    CHECK(host_reserved(ledger) == 60);

    const auto committed = prepared.materialize_and_commit(exact);
    CHECK(committed.status == llama_cache_transaction_status::committed);
    CHECK(!prepared.ready());
    CHECK(first_op && second_op && first_op != second_op);
    CHECK(first_allocation && second_allocation &&
          first_allocation != second_allocation);
    CHECK(ledger.snapshot().live_ops == 2);
    CHECK(host_reserved(ledger) == 0);
    CHECK(ledger.release(first_op));
    CHECK(ledger.release(second_op));
    CHECK(ledger.snapshot().live_ops == 0);

    // A content-addressed final leaf may join an authenticated existing
    // allocation. Its zero resident reservation is a valid downward
    // conversion of the conservative fresh fence, while staging still cites
    // the complete immutable allocation size.
    llama_cache_acct_ledger dedup_ledger;
    configure_fitting_host(dedup_ledger);
    auto owner = llama_cache_admit_reservation(
        dedup_ledger, config, host_request(32));
    CHECK(owner.status == llama_cache_admission_status::admitted);
    const auto shared_allocation = dedup_ledger.new_alloc();
    CHECK(shared_allocation);
    CHECK(dedup_ledger.stage(owner.claim.op(), shared_allocation, 32));
    llama_cache_acct_op_id owner_op;
    CHECK(owner.claim.commit(32, owner_op));

    llama_cache_acct_op_id fence_op;
    llama_cache_acct_alloc_id fence_allocation;
    std::vector<llama_cache_transaction_leaf> fence {
        transaction_leaf(PAYLOAD, 64, fence_op, fence_allocation),
    };
    auto dedup_prepared = llama_cache_prepare_reservation_transaction(
        dedup_ledger, config, fence);
    CHECK(dedup_prepared.ready());
    llama_cache_acct_op_id joined_op;
    llama_cache_acct_alloc_id joined_allocation;
    auto joined = transaction_leaf(
        PAYLOAD, 32, joined_op, joined_allocation);
    joined.reserve_resident = 0;
    joined.existing_allocation = shared_allocation;
    std::vector<llama_cache_transaction_leaf> deduplicated {
        std::move(joined),
    };
    CHECK(dedup_prepared.repartition_downward(deduplicated));
    CHECK(host_reserved(dedup_ledger) == 0);
    const auto joined_result =
        dedup_prepared.materialize_and_commit(deduplicated);
    CHECK(joined_result.status ==
          llama_cache_transaction_status::committed);
    CHECK(joined_allocation == shared_allocation);
    CHECK(dedup_ledger.snapshot().live_ops == 2);
    CHECK(dedup_ledger.release(owner_op));
    CHECK(dedup_ledger.snapshot().live_ops == 1);
    CHECK(dedup_ledger.release(joined_op));
    CHECK(dedup_ledger.snapshot().live_ops == 0);

    llama_cache_acct_ledger dropped_ledger;
    configure_fitting_host(dropped_ledger);
    llama_cache_acct_op_id dropped_initial_op;
    llama_cache_acct_alloc_id dropped_initial_allocation;
    {
        std::vector<llama_cache_transaction_leaf> dropped_initial {
            transaction_leaf(
                PAYLOAD, 32, dropped_initial_op,
                dropped_initial_allocation),
        };
        auto dropped = llama_cache_prepare_reservation_transaction(
            dropped_ledger, config, dropped_initial);
        CHECK(dropped.ready());
        llama_cache_acct_op_id dropped_first_op;
        llama_cache_acct_op_id dropped_second_op;
        llama_cache_acct_alloc_id dropped_first_allocation;
        llama_cache_acct_alloc_id dropped_second_allocation;
        std::vector<llama_cache_transaction_leaf> dropped_final {
            transaction_leaf(
                PAYLOAD, 12, dropped_first_op,
                dropped_first_allocation),
            transaction_leaf(
                PAYLOAD, 8, dropped_second_op,
                dropped_second_allocation),
        };
        CHECK(dropped.repartition_downward(dropped_final));
        CHECK(dropped_ledger.snapshot().live_ops == 2);
        CHECK(host_reserved(dropped_ledger) == 20);
    }
    CHECK(dropped_ledger.snapshot().live_ops == 0);
    CHECK(host_reserved(dropped_ledger) == 0);

    // Keep the projected maximum leaf shape out of quadratic validation.
    // This does not materialize payload; it exercises the exact atomic split
    // and destructor terminal at the 16,384-unit assembly ceiling.
    llama_cache_acct_ledger scale_ledger;
    configure_fitting_host(scale_ledger);
    llama_cache_acct_op_id scale_initial_op;
    llama_cache_acct_alloc_id scale_initial_allocation;
    std::vector<llama_cache_transaction_leaf> scale_initial {
        transaction_leaf(
            PAYLOAD, 16384, scale_initial_op,
            scale_initial_allocation),
    };
    {
        auto scale = llama_cache_prepare_reservation_transaction(
            scale_ledger, config, scale_initial);
        CHECK(scale.ready());
        std::vector<llama_cache_acct_op_id> operations(16384);
        std::vector<llama_cache_acct_alloc_id> allocations(16384);
        std::vector<llama_cache_transaction_leaf> leaves;
        leaves.reserve(16384);
        for (size_t i = 0; i < 16384; ++i) {
            leaves.push_back(transaction_leaf(
                PAYLOAD, 1, operations[i], allocations[i]));
        }
        CHECK(scale.repartition_downward(leaves));
        CHECK(scale_ledger.snapshot().live_ops == 16384);
        CHECK(host_reserved(scale_ledger) == 16384);
    }
    CHECK(scale_ledger.snapshot().live_ops == 0);
    CHECK(host_reserved(scale_ledger) == 0);
}

static void test_prepared_transaction_partition_owned_groups() {
    llama_cache_acct_ledger ledger;
    configure_fitting_host(ledger);
    llama_cache_budget_config config;
    llama_cache_acct_op_id initial_op;
    llama_cache_acct_alloc_id initial_allocation;
    std::vector<llama_cache_transaction_leaf> initial {
        transaction_leaf(PAYLOAD, 96, initial_op, initial_allocation),
    };
    auto prepared = llama_cache_prepare_reservation_transaction(
        ledger, config, initial);
    CHECK(prepared.ready());

    std::vector<llama_cache_acct_op_id> operations(3);
    std::vector<llama_cache_acct_alloc_id> allocations(3);
    std::vector<llama_cache_transaction_leaf> leaves {
        transaction_leaf(PAYLOAD, 40, operations[0], allocations[0]),
        transaction_leaf(PAYLOAD, 20, operations[1], allocations[1]),
        transaction_leaf(PAYLOAD, 10, operations[2], allocations[2]),
    };
    CHECK(prepared.repartition_downward(leaves));
    CHECK(ledger.snapshot().live_ops == 3);
    CHECK(host_reserved(ledger) == 70);

    std::vector<llama_cache_prepared_claim_group> groups;
    CHECK(!prepared.partition({ 2 }, groups));
    CHECK(groups.empty());
    CHECK(prepared.ready());
    CHECK(ledger.snapshot().live_ops == 3);
    CHECK(!prepared.partition({ 0, 3 }, groups));
    CHECK(groups.empty());
    CHECK(prepared.ready());
    CHECK(ledger.snapshot().live_ops == 3);
    CHECK(host_reserved(ledger) == 70);

    llama_cache_acct_op_id seeded_op;
    llama_cache_acct_alloc_id seeded_allocation;
    std::vector<llama_cache_transaction_leaf> seeded_leaf {
        transaction_leaf(PAYLOAD, 24, seeded_op, seeded_allocation),
    };
    auto seeded = llama_cache_prepare_reservation_transaction(
        ledger, config, seeded_leaf);
    CHECK(seeded.ready());
    groups.push_back(std::move(seeded));
    CHECK(!prepared.partition({ 2, 1 }, groups));
    CHECK(groups.size() == 1 && groups[0].ready());
    CHECK(prepared.ready());
    CHECK(ledger.snapshot().live_ops == 4);
    CHECK(host_reserved(ledger) == 94);
    groups.clear();
    CHECK(ledger.snapshot().live_ops == 3);
    CHECK(host_reserved(ledger) == 70);

    CHECK(prepared.partition({ 2, 1 }, groups));
    CHECK(groups.size() == 2);
    CHECK(groups[0].ready());
    CHECK(groups[1].ready());
    CHECK(!prepared.ready());
    const std::vector<llama_cache_transaction_leaf> prefix_leaves {
        leaves[0], leaves[1],
    };
    const auto committed =
        groups[0].materialize_and_commit(prefix_leaves);
    CHECK(committed.status == llama_cache_transaction_status::committed);
    CHECK(!groups[0].ready());
    CHECK(ledger.snapshot().live_ops == 3);
    CHECK(host_reserved(ledger) == 10);

    // Dropping the remainder owns exactly its one reservation. The detached
    // committed prefix remains independently live until ordinary retirement.
    groups[1] = {};
    CHECK(ledger.snapshot().live_ops == 2);
    CHECK(host_reserved(ledger) == 0);
    CHECK(ledger.release(operations[0]));
    CHECK(ledger.release(operations[1]));
    CHECK(ledger.snapshot().live_ops == 0);

    llama_cache_acct_ledger full_ledger;
    configure_fitting_host(full_ledger);
    llama_cache_acct_op_id full_op;
    llama_cache_acct_alloc_id full_allocation;
    std::vector<llama_cache_transaction_leaf> full_leaves {
        transaction_leaf(PAYLOAD, 24, full_op, full_allocation),
    };
    auto full = llama_cache_prepare_reservation_transaction(
        full_ledger, config, full_leaves);
    CHECK(full.ready());
    std::vector<llama_cache_prepared_claim_group> detached;
    CHECK(!full.partition({ 2 }, detached));
    CHECK(detached.empty());
    CHECK(full.ready());
    CHECK(full_ledger.snapshot().live_ops == 1);
    CHECK(host_reserved(full_ledger) == 24);

    CHECK(full.partition({ 1 }, detached));
    CHECK(detached.size() == 1);
    CHECK(detached[0].ready());
    CHECK(!full.ready());
    const auto empty_terminal = full.materialize_and_commit();
    CHECK(empty_terminal.status ==
          llama_cache_transaction_status::internal_fault);
    CHECK(full_ledger.snapshot().live_ops == 1);
    detached.clear();
    CHECK(full_ledger.snapshot().live_ops == 0);
    CHECK(host_reserved(full_ledger) == 0);
}

static void test_atomic_reservation_sets() {
    llama_cache_acct_ledger ledger;
    configure_fitting_host(ledger);
    const auto before = ledger.snapshot();
    llama_cache_acct_resource_domain unknown_device;
    unknown_device.residency = llama_cache_acct_residency::device;
    unknown_device.kind = llama_cache_acct_domain_kind::device_topology;
    unknown_device.device_ordinal = { 0 };
    unknown_device.topology = { 999 };
    const llama_cache_conditional_reserve_request requests[] = {
        { PAYLOAD, HOST, {}, 32, 32 },
        { PAYLOAD, unknown_device, {}, 16, 16 },
    };
    llama_cache_acct_op_id output[] = { { 91 }, { 92 } };
    const auto refused = ledger.reserve_set_if_serial(
        before.serial, requests, 2, output);
    CHECK(refused.status ==
          llama_cache_conditional_reserve_status::ledger_fault);
    CHECK(refused.failed_request == 1);
    CHECK(!output[0] && !output[1]);
    CHECK(ledger.snapshot().live_ops == 0);
    CHECK(host_reserved(ledger) == 0);

    const auto first = ledger.reserve(PAYLOAD, HOST, {}, 32, 32);
    const auto second = ledger.reserve(PAYLOAD, HOST, {}, 16, 16);
    CHECK(first && second && first.v < second.v);
    const llama_cache_acct_op_id invalid_set[] = {
        first, { second.v + 1000 },
    };
    CHECK(!ledger.abort_set(invalid_set, 2));
    CHECK(ledger.snapshot().live_ops == 2);
    CHECK(host_reserved(ledger) == 48);

    const llama_cache_acct_op_id valid_set[] = { first, second };
    const uint64_t invalid_shrink[] = { 24, 17 };
    CHECK(!ledger.shrink_reservation_set(
        valid_set, invalid_shrink, 2));
    CHECK(ledger.snapshot().live_ops == 2);
    CHECK(host_reserved(ledger) == 48);
    const uint64_t valid_shrink[] = { 24, 8 };
    CHECK(ledger.shrink_reservation_set(
        valid_set, valid_shrink, 2));
    CHECK(ledger.snapshot().live_ops == 2);
    CHECK(host_reserved(ledger) == 32);
    CHECK(ledger.abort_set(valid_set, 2));
    CHECK(ledger.snapshot().live_ops == 0);
    CHECK(host_reserved(ledger) == 0);
}

static llama_cache_acct_op_id commit_host_leaf(
        llama_cache_acct_ledger & ledger,
        uint64_t bytes) {
    llama_cache_budget_config config;
    auto admitted = llama_cache_admit_reservation(
        ledger, config, host_request(bytes));
    CHECK(admitted.status == llama_cache_admission_status::admitted);
    const auto alloc = ledger.new_alloc();
    CHECK(bool(alloc));
    CHECK(ledger.stage(admitted.claim.op(), alloc, bytes));
    llama_cache_acct_op_id committed;
    CHECK(admitted.claim.commit(bytes, committed));
    return committed;
}

static void test_prepared_release_set() {
    llama_cache_acct_ledger ledger;
    configure_fitting_host(ledger);
    const auto op = commit_host_leaf(ledger, 64);
    const auto before_invalid = ledger.snapshot();
    CHECK(ledger.release_set_if_serial(
        { op, llama_cache_acct_op_id{ op.v + 1000 } },
        before_invalid.serial) ==
          llama_cache_conditional_release_status::ledger_fault);
    CHECK(ledger.snapshot().live_ops == 1);
    const auto quote = ledger.snapshot();

    auto prepared = llama_cache_prepare_release_set(
        ledger, { op }, quote.serial);
    CHECK(prepared.ready());
    CHECK(prepared.status() ==
          llama_cache_prepare_release_status::prepared);
    CHECK(prepared.preview().rows.size() == 1);
    CHECK(prepared.preview().rows[0].resident_allocated == 64);

    // Any observable ledger write invalidates this preparation without
    // releasing the operation. A fresh prepare absorbs the serial drift.
    ledger.gauge_set(PAYLOAD, HOST,
                     llama_cache_acct_measure::logical_payload, 64);
    CHECK(prepared.commit() ==
          llama_cache_conditional_release_status::serial_conflict);
    CHECK(ledger.snapshot().live_ops == 1);

    const auto fresh = ledger.snapshot();
    auto reprepared = llama_cache_prepare_release_set(
        ledger, { op }, fresh.serial);
    CHECK(reprepared.ready());
    CHECK(reprepared.accounting_serial() == fresh.serial);
    llama_cache_prepared_release_set moved(std::move(reprepared));
    CHECK(!reprepared.ready());
    CHECK(moved.ready());
    CHECK(moved.commit() ==
          llama_cache_conditional_release_status::released);
    CHECK(ledger.snapshot().live_ops == 0);

    auto invalid = llama_cache_prepare_release_set(
        ledger, { op, op }, ledger.snapshot().serial);
    CHECK(!invalid.ready());
    CHECK(invalid.status() ==
          llama_cache_prepare_release_status::invalid_argument);
}

static void test_status_names() {
    CHECK(std::string(llama_cache_admission_status_name(
        llama_cache_admission_status::admitted)) == "admitted");
    CHECK(std::string(llama_cache_admission_status_name(
        llama_cache_admission_status::incomplete_evidence)) == "incomplete_evidence");
    CHECK(std::string(llama_cache_admission_status_name(
        llama_cache_admission_status::internal_fault)) == "internal_fault");
    CHECK(std::string(llama_cache_transaction_status_name(
        llama_cache_transaction_status::committed)) == "committed");
    CHECK(std::string(llama_cache_transaction_status_name(
        llama_cache_transaction_status::post_commit_fault)) ==
        "post_commit_fault");
    CHECK(std::string(llama_cache_prepare_status_name(
        llama_cache_prepare_status::prepared)) == "prepared");
    CHECK(std::string(llama_cache_prepare_release_status_name(
        llama_cache_prepare_release_status::prepared)) == "prepared");
}

int main() {
    test_incomplete_evidence();
    test_budget_unavailable();
    test_admitted();
    test_exceeds_budget();
    test_claim_auto_abort();
    test_claim_move_ctor();
    test_claim_move_assign();
    test_claim_commit_terminal();
    test_claim_commit_failure_auto_abort();
    test_transaction_fresh_group();
    test_transaction_existing_allocation();
    test_transaction_fault_rollback();
    test_transaction_after_admit_failure();
    test_prepared_transaction_split_phase();
    test_prepared_transaction_downward_repartition();
    test_prepared_transaction_partition_owned_groups();
    test_atomic_reservation_sets();
    test_prepared_release_set();
    test_status_names();

    if (failures > 0) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    printf("all cache-authority tests passed\n");
    return EXIT_SUCCESS;
}
