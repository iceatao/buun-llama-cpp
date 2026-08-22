#include "server-cache-lease.h"
#include "common-cache-family.h"
#include "common.h"

#include <cstdio>
#include <limits>
#include <type_traits>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

class fake_clock final : public server_cache_lease_clock {
public:
    uint64_t now = 100;
    uint64_t calls = 0;

    uint64_t now_ns() noexcept override {
        calls++;
        return now;
    }
};

class fake_fallback final : public server_cache_lease_fallback_provider {
public:
    server_cache_lease_fallback_state state =
        server_cache_lease_fallback_state::unavailable;
    uint64_t calls = 0;
    std::shared_ptr<void> owner = std::make_shared<int>(1);

    server_cache_durable_fallback_proof acquire(
            const server_cache_lease_subject &,
            const server_cache_lease_identity &) noexcept override {
        calls++;
        return server_cache_durable_fallback_proof_for_test(state, owner);
    }
};

static server_cache_lease_identity identity(const char * adapter = "base:no-lora") {
    return { "execution-1", adapter, "server-media-prefix-v1:0" };
}

static server_cache_lease_subject subject(
        uint64_t artifact,
        common_retention_artifact_kind kind =
            common_retention_artifact_kind::live_slot,
        int32_t owner_slot = 1) {
    return { llama_cache_acct_artifact_id { artifact }, kind, owner_slot };
}

static server_cache_lease_scope context_scope(uint64_t id = 1) {
    return server_cache_lease_scope::from(server_cache_context_scope_id { id });
}

static server_cache_destruction_request direct_request(
        server_cache_destruction_target_kind kind,
        const server_cache_lease_subject & value) {
    server_cache_destruction_request request;
    request.cls = kind == server_cache_destruction_target_kind::host_artifact
        ? server_cache_destruction_class::host_artifact_drop
        : server_cache_destruction_class::live_range_drop;
    request.reason = server_cache_destruction_reason::idle_reclaim;
    request.add_target(kind, value.owner_slot, value.artifact);
    return request;
}

static void test_closed_scope_types() {
    static_assert(!std::is_copy_constructible<
                      server_cache_durable_fallback_proof>::value, "");
    static_assert(std::is_move_constructible<
                      server_cache_durable_fallback_proof>::value, "");
    static_assert(!std::is_same<server_cache_process_scope_id,
                               server_cache_session_scope_id>::value, "");
    static_assert(!std::is_same<server_cache_process_scope_id,
                               server_cache_context_scope_id>::value, "");
    static_assert(!std::is_same<server_cache_process_scope_id,
                               server_cache_explicit_lease_scope_id>::value, "");
    static_assert(!std::is_same<server_cache_session_scope_id,
                               server_cache_context_scope_id>::value, "");
    static_assert(!std::is_same<server_cache_session_scope_id,
                               server_cache_explicit_lease_scope_id>::value, "");
    static_assert(!std::is_same<server_cache_context_scope_id,
                               server_cache_explicit_lease_scope_id>::value, "");
    static_assert(!std::is_same<server_cache_lease_id,
                               server_cache_process_scope_id>::value, "");
    static_assert(!std::is_same<server_cache_lease_id,
                               server_cache_session_scope_id>::value, "");
    static_assert(!std::is_same<server_cache_lease_id,
                               server_cache_context_scope_id>::value, "");
    static_assert(!std::is_same<server_cache_lease_id,
                               server_cache_explicit_lease_scope_id>::value, "");
    static_assert(!std::is_same<server_cache_lease_id,
                               server_cache_lease_identity_id>::value, "");
    static_assert(std::is_same<
        decltype(common_computation_frontier::execution_identity),
        decltype(server_cache_lease_identity::execution_identity)>::value, "");
    static_assert(std::is_same<
        decltype(common_computation_frontier::adapter_config_identity),
        decltype(server_cache_lease_identity::adapter_config_identity)>::value, "");
    static_assert(std::is_same<
        decltype(common_computation_frontier::media_content_identity),
        decltype(server_cache_lease_identity::media_content_identity)>::value, "");
    CHECK(server_cache_lease_scope::from(
              server_cache_process_scope_id { 1 }).valid());
    CHECK(server_cache_lease_scope::from(
              server_cache_session_scope_id { 1 }).valid());
    CHECK(server_cache_lease_scope::from(
              server_cache_context_scope_id { 1 }).valid());
    CHECK(server_cache_lease_scope::from(
              server_cache_explicit_lease_scope_id { 1 }).valid());
    CHECK(!server_cache_lease_scope::from(
              server_cache_context_scope_id {}).valid());

    fake_clock clock;
    server_cache_lease_table table(&clock);
    const auto id = identity();
    const std::array<server_cache_lease_scope, 4> scopes = {{
        server_cache_lease_scope::from(server_cache_process_scope_id { 1 }),
        server_cache_lease_scope::from(server_cache_session_scope_id { 2 }),
        server_cache_lease_scope::from(server_cache_context_scope_id { 3 }),
        server_cache_lease_scope::from(
            server_cache_explicit_lease_scope_id { 4 }),
    }};
    for (size_t i = 0; i < scopes.size(); ++i) {
        const auto value = subject(90 + i);
        const auto lease = table.grant_soft(value, scopes[i], id, 10);
        CHECK(lease);
        CHECK(table.evaluate(value.artifact, id).cls ==
              server_cache_lease_class::soft);
        CHECK(table.release(lease));
    }

    // Sidecar v1 remains deliberately lease-neutral. D-S6 derives this bit
    // from the table instead of accepting a second authority.
    CHECK(!common_retention_stamp{}.soft_leased);

    common_computation_frontier frontier;
    frontier.execution_identity = "execution";
    frontier.adapter_config_identity = "adapter";
    frontier.media_content_identity = "media";
    const server_cache_lease_identity mirrored = {
        frontier.execution_identity,
        frontier.adapter_config_identity,
        frontier.media_content_identity,
    };
    const server_cache_lease_identity expected = {
        "execution", "adapter", "media",
    };
    CHECK(mirrored.valid());
    CHECK(mirrored == expected);
}

static void test_declared_family_replaces_automatic_weight() {
    const common_cache_family_binding absent;
    CHECK(common_cache_family_main_family(absent, true));
    CHECK(!common_cache_family_main_family(absent, false));
    CHECK(common_cache_family_allows_additional_weight(absent));

    const common_cache_family_binding declared_main {
        { 71 }, common_cache_family_role::main,
    };
    const common_cache_family_binding declared_branch {
        { 71 }, common_cache_family_role::branch,
    };
    CHECK(common_cache_family_main_family(declared_main, false));
    CHECK(!common_cache_family_main_family(declared_branch, true));
    CHECK(!common_cache_family_allows_additional_weight(declared_main));
    CHECK(!common_cache_family_allows_additional_weight(declared_branch));
}

static void test_hard_proof_lifetime_across_clone() {
    fake_clock clock;
    fake_fallback fallback;
    fallback.state = server_cache_lease_fallback_state::available;
    std::weak_ptr<void> lifetime = fallback.owner;
    server_cache_lease_table table(&clock, &fallback);
    const auto source = subject(901);
    const auto destination = subject(
        902, common_retention_artifact_kind::host_entry, -1);
    const auto lease = table.grant_hard(
        source, context_scope(), identity(), 100);
    CHECK(lease);
    fallback.owner.reset();
    CHECK(!lifetime.expired());
    CHECK(table.artifact_cloned(source, destination, identity()));
    table.artifact_retired(source.artifact);
    CHECK(!lifetime.expired());
    table.artifact_retired(destination.artifact);
    CHECK(lifetime.expired());

    auto invalid = server_cache_durable_fallback_proof_for_test(
        server_cache_lease_fallback_state::available, {});
    CHECK(!invalid.available());
}

static void test_soft_renew_expire_release() {
    fake_clock clock;
    server_cache_lease_table table(&clock);
    const auto value = subject(11);
    const auto id = identity();

    const auto lease = table.grant_soft(value, context_scope(), id, 100);
    CHECK(lease);
    auto eval = table.evaluate(value.artifact, id);
    CHECK(eval.state == server_cache_lease_eval_state::known);
    CHECK(eval.cls == server_cache_lease_class::soft);
    CHECK(eval.eligibility == server_cache_lease_eligibility::eligible);

    // A shorter renewal cannot move the absolute deadline backward.
    clock.now = 110;
    CHECK(table.renew(lease, 5));
    clock.now = 199;
    CHECK(table.evaluate(value.artifact, id).cls ==
          server_cache_lease_class::soft);
    clock.now = 200;
    eval = table.evaluate(value.artifact, id);
    CHECK(eval.state == server_cache_lease_eval_state::known);
    CHECK(eval.cls == server_cache_lease_class::none);
    CHECK(!table.renew(lease, 100));
    CHECK(!table.release(lease));

    clock.now = 300;
    const auto second = table.grant_soft(
        value, context_scope(2), id, 100);
    CHECK(second);
    CHECK(table.release(second));
    CHECK(!table.release(second));
    CHECK(table.evaluate(value.artifact, id).cls ==
          server_cache_lease_class::none);
}

static void test_checked_deadline() {
    fake_clock clock;
    clock.now = std::numeric_limits<uint64_t>::max() - 2;
    server_cache_lease_table table(&clock);
    CHECK(!table.grant_soft(subject(12), context_scope(), identity(), 3));
    CHECK(table.unavailable_events() > 0);
    CHECK(!table.event_snapshot().replay_available());
}

static void test_hard_preflight_and_admission() {
    const auto value = subject(
        21, common_retention_artifact_kind::host_entry, -1);
    const auto id = identity();

    fake_clock unavailable_clock;
    fake_fallback unavailable;
    server_cache_lease_table refused(&unavailable_clock, &unavailable);
    CHECK(!refused.grant_hard(value, context_scope(), id, 100));
    CHECK(unavailable.calls == 1);
    CHECK(refused.evaluate(value.artifact, id).cls ==
          server_cache_lease_class::none);
    CHECK(refused.admit(direct_request(
              server_cache_destruction_target_kind::host_artifact, value)) ==
          server_cache_destruction_verdict::admit_unleased);
    const auto refused_events = refused.event_snapshot();
    CHECK(refused_events.size == 1);
    CHECK(refused_events.events[0].kind ==
          server_cache_lease_event_kind::refuse_hard_unavailable);
    CHECK(refused_events.totals[size_t(
              server_cache_lease_event_kind::refuse_hard_unavailable)] == 1);
    CHECK(!refused_events.events[0].lease);

    fake_clock invalid_clock;
    fake_fallback invalid;
    invalid.state = server_cache_lease_fallback_state::invalid;
    server_cache_lease_table invalid_table(&invalid_clock, &invalid);
    CHECK(!invalid_table.grant_hard(value, context_scope(), id, 100));
    CHECK(invalid_table.event_snapshot().events[0].kind ==
          server_cache_lease_event_kind::refuse_hard_invalid);

    fake_clock available_clock;
    fake_fallback available;
    available.state = server_cache_lease_fallback_state::available;
    server_cache_lease_table granted(&available_clock, &available);
    const auto hard = granted.grant_hard(
        value, context_scope(), id, 100);
    CHECK(hard);
    const auto eval = granted.evaluate(value.artifact, id);
    CHECK(eval.cls == server_cache_lease_class::hard);
    CHECK(eval.eligibility == server_cache_lease_eligibility::hard_blocked);
    CHECK(granted.admit(direct_request(
              server_cache_destruction_target_kind::host_artifact, value)) ==
          server_cache_destruction_verdict::would_refuse_hard_leased);
}

static void test_multi_target_precedence() {
    fake_clock clock;
    fake_fallback fallback;
    fallback.state = server_cache_lease_fallback_state::available;
    server_cache_lease_table table(&clock, &fallback);
    const auto id = identity();
    const auto live = subject(31);
    const auto checkpoint = subject(
        32, common_retention_artifact_kind::checkpoint, 7);
    const auto host_checkpoint = subject(
        34, common_retention_artifact_kind::checkpoint, -1);
    CHECK(table.grant_soft(live, context_scope(), id, 100));
    CHECK(table.grant_hard(checkpoint, context_scope(2), id, 100));
    CHECK(table.grant_soft(host_checkpoint, context_scope(3), id, 100));

    CHECK(table.admit(direct_request(
              server_cache_destruction_target_kind::live_target, live)) ==
          server_cache_destruction_verdict::admit_soft_leased);

    server_cache_destruction_request broad;
    broad.cls = server_cache_destruction_class::checkpoint_drop;
    broad.reason = server_cache_destruction_reason::checkpoint_capacity;
    broad.add_target(
        server_cache_destruction_target_kind::checkpoint_ring, 7);
    CHECK(table.admit(broad) ==
          server_cache_destruction_verdict::would_refuse_hard_leased);

    const auto host = subject(
        33, common_retention_artifact_kind::host_entry, -1);
    server_cache_destruction_request host_drop = direct_request(
        server_cache_destruction_target_kind::host_artifact, host);
    CHECK(table.admit(host_drop) ==
          server_cache_destruction_verdict::unavailable);

    server_cache_destruction_request unresolved;
    unresolved.cls = server_cache_destruction_class::live_range_drop;
    unresolved.add_target(
        server_cache_destruction_target_kind::live_target, 1);
    CHECK(table.admit(unresolved) ==
          server_cache_destruction_verdict::unavailable);

    server_cache_destruction_request overflow = unresolved;
    overflow.overflowed = true;
    CHECK(table.admit(overflow) ==
          server_cache_destruction_verdict::unavailable);

    overflow.cls =
        server_cache_destruction_class::mandatory_recovery_reset;
    CHECK(table.admit(overflow) ==
          server_cache_destruction_verdict::admit_mandatory_recovery);
}

static void test_identity_clone_rebind_retire() {
    fake_clock clock;
    fake_fallback fallback;
    fallback.state = server_cache_lease_fallback_state::available;
    server_cache_lease_table table(&clock, &fallback);
    const auto id = identity();
    const auto source = subject(41);
    const auto destination = subject(
        42, common_retention_artifact_kind::host_entry, -1);
    const auto soft = table.grant_soft(source, context_scope(), id, 100);
    const auto hard = table.grant_hard(source, context_scope(), id, 100);
    CHECK(soft);
    CHECK(hard);
    CHECK(table.artifact_cloned(source, destination, id));
    CHECK(table.evaluate(destination.artifact, id).cls ==
          server_cache_lease_class::hard);
    const auto cloned_events = table.event_snapshot();
    size_t clone_count = 0;
    bool cloned_soft = false;
    bool cloned_hard = false;
    for (size_t i = 0; i < cloned_events.size; ++i) {
        const auto & event = cloned_events.events[i];
        if (event.kind != server_cache_lease_event_kind::clone) {
            continue;
        }
        clone_count++;
        cloned_soft |= event.source_lease == soft;
        cloned_hard |= event.source_lease == hard;
    }
    CHECK(clone_count == 2);
    CHECK(cloned_soft);
    CHECK(cloned_hard);

    CHECK(table.artifact_rebound(destination.artifact, id));
    const auto wrong = identity("adapter-2");
    auto eval = table.evaluate(destination.artifact, wrong);
    CHECK(eval.state == server_cache_lease_eval_state::unavailable);
    CHECK(eval.cls == server_cache_lease_class::none);
    CHECK(table.evaluate(destination.artifact, id).cls ==
          server_cache_lease_class::none);
    CHECK(table.admit(direct_request(
              server_cache_destruction_target_kind::host_artifact,
              destination)) ==
          server_cache_destruction_verdict::unavailable);

    table.artifact_retired(source.artifact);
    CHECK(table.evaluate(source.artifact, id).cls ==
          server_cache_lease_class::none);
    CHECK(table.evaluate(source.artifact, {}).state ==
          server_cache_lease_eval_state::unavailable);

    const auto unknown = subject(43);
    table.artifact_identity_unavailable(unknown);
    CHECK(table.evaluate(unknown.artifact, id).state ==
          server_cache_lease_eval_state::unavailable);
    CHECK(table.admit(direct_request(
              server_cache_destruction_target_kind::live_target, unknown)) ==
          server_cache_destruction_verdict::unavailable);
    table.artifact_retired(unknown.artifact);
    CHECK(table.evaluate(unknown.artifact, id).state ==
          server_cache_lease_eval_state::known);

    server_cache_lease_replay_result replay;
    const auto trace = table.event_snapshot();
    CHECK(server_cache_lease_table::replay(trace, replay));
    CHECK(replay.active.empty());
    CHECK(replay.identities.size() == 2);
    CHECK(replay.identity_unavailable.size() == 1);
    CHECK(replay.identity_unavailable[0].artifact == destination.artifact);
}

static void test_replay_and_ring_overflow() {
    fake_clock clock;
    server_cache_lease_table table(&clock);
    const auto id = identity();
    const auto a = subject(51);
    const auto b = subject(
        52, common_retention_artifact_kind::host_entry, -1);
    const auto lease = table.grant_soft(a, context_scope(), id, 100);
    CHECK(lease);
    CHECK(table.renew(lease, 150));
    CHECK(table.artifact_cloned(a, b, id));
    table.artifact_retired(a.artifact);

    const auto snapshot = table.event_snapshot();
    CHECK(snapshot.replay_available());
    server_cache_lease_replay_result replay;
    CHECK(server_cache_lease_table::replay(snapshot, replay));
    CHECK(replay.state == server_cache_lease_eval_state::known);
    CHECK(replay.last_ordinal == snapshot.last_ordinal);
    CHECK(replay.active.size() == 1);
    CHECK(replay.active[0].artifact == b.artifact);
    CHECK(replay.active[0].artifact_kind ==
          common_retention_artifact_kind::host_entry);
    CHECK(replay.active[0].owner_slot == -1);
    CHECK(replay.active[0].identity);
    CHECK(replay.identities.size() == 1);

    fake_clock expiry_clock;
    server_cache_lease_table expired(&expiry_clock);
    CHECK(expired.grant_soft(
        subject(53), context_scope(), id, 10));
    expiry_clock.now = 110;
    CHECK(expired.evaluate(llama_cache_acct_artifact_id { 53 }, id).cls ==
          server_cache_lease_class::none);
    const auto expiry_trace = expired.event_snapshot();
    CHECK(expiry_trace.events[expiry_trace.size - 1].kind ==
          server_cache_lease_event_kind::expire);
    CHECK(server_cache_lease_table::replay(expiry_trace, replay));
    CHECK(replay.active.empty());

    fake_clock overflow_clock;
    server_cache_lease_table overflow(&overflow_clock);
    for (size_t i = 0; i < SERVER_CACHE_LEASE_EVENT_RING + 1; ++i) {
        const auto value = subject(100 + i);
        CHECK(overflow.grant_soft(
            value, context_scope(i + 1), id, 100));
    }
    const auto wrapped = overflow.event_snapshot();
    CHECK(wrapped.overflows > 0);
    CHECK(!wrapped.replay_available());
    CHECK(!server_cache_lease_table::replay(wrapped, replay));
}

static void test_observer_off_zero_work() {
    fake_clock clock;
    server_cache_lease_table table(&clock);
    server_cache_destruction_request request;
    request.cls = server_cache_destruction_class::live_range_drop;
    request.add_target(
        server_cache_destruction_target_kind::live_target, 1);

    const auto admission = server_cache_retention_admit(nullptr, request);
    CHECK(admission.issued);
    CHECK(admission.execution ==
          server_cache_destruction_execution::pass_through);
    CHECK(clock.calls == 0);

    server_cache_destruction_observer observer;
    observer.lease_context = &table;
    observer.lease_evaluator = server_cache_lease_evaluate_request;
    const auto observed = server_cache_retention_admit(&observer, request);
    CHECK(observed.verdict == server_cache_destruction_verdict::unavailable);
    CHECK(observed.execution ==
          server_cache_destruction_execution::pass_through);
    CHECK(clock.calls == 1);
}

static void test_lifecycle_without_debug_consults_lease() {
    fake_clock clock;
    fake_fallback fallback;
    fallback.state = server_cache_lease_fallback_state::available;
    server_cache_lease_table table(&clock, &fallback);
    const auto value = subject(
        88, common_retention_artifact_kind::host_entry, -1);
    CHECK(table.grant_hard(value, context_scope(), identity(), 100));

    // This is the lifecycle-only wiring shape: the policy observer/evaluator
    // exists, while no cache-plan JSON observer is involved. The hard lease is
    // consulted, yet D-A0b still executes pass-through.
    server_cache_destruction_observer lifecycle;
    lifecycle.lease_context = &table;
    lifecycle.lease_evaluator = server_cache_lease_evaluate_request;
    const auto admission = server_cache_retention_admit(
        &lifecycle,
        direct_request(
            server_cache_destruction_target_kind::host_artifact,
            value));
    CHECK(admission.verdict ==
          server_cache_destruction_verdict::would_refuse_hard_leased);
    CHECK(admission.execution ==
          server_cache_destruction_execution::pass_through);
    CHECK(lifecycle.n_events == 1);
}

static void test_batch_inspection_max_cardinality() {
    fake_clock clock;
    server_cache_lease_table table(&clock);
    const auto expected = identity();
    const auto wrong = identity("different-adapter");
    constexpr size_t count = 8192;
    std::vector<server_cache_lease_inspection_request> requests;
    requests.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto value = subject(1000 + i);
        CHECK(table.grant_soft(value, context_scope(), expected, 1000));
        requests.push_back({ value.artifact, &expected });
    }
    requests[count / 2].expected_identity = &wrong;
    table.artifact_identity_unavailable(subject(1000 + count - 1));

    clock.calls = 0;
    std::vector<server_cache_lease_evaluation> inspected;
    CHECK(table.inspect_batch(requests, inspected));
    CHECK(clock.calls == 1);
    CHECK(inspected.size() == count);
    CHECK(inspected.front().state == server_cache_lease_eval_state::known);
    CHECK(inspected.front().cls == server_cache_lease_class::soft);
    CHECK(inspected[count / 2].state ==
          server_cache_lease_eval_state::unavailable);
    CHECK(inspected.back().state ==
          server_cache_lease_eval_state::unavailable);

    requests.back().artifact = requests.front().artifact;
    CHECK(!table.inspect_batch(requests, inspected));
    CHECK(inspected.empty());
}

int main() {
    test_closed_scope_types();
    test_declared_family_replaces_automatic_weight();
    test_soft_renew_expire_release();
    test_checked_deadline();
    test_hard_preflight_and_admission();
    test_multi_target_precedence();
    test_identity_clone_rebind_retire();
    test_replay_and_ring_overflow();
    test_observer_off_zero_work();
    test_lifecycle_without_debug_consults_lease();
    test_hard_proof_lifetime_across_clone();
    test_batch_inspection_max_cardinality();

    if (failures != 0) {
        std::fprintf(stderr, "%d cache-lease test(s) failed\n", failures);
        return 1;
    }
    std::puts("cache lease tests passed");
    return 0;
}
