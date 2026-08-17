#include "server-cache-yield.h"

#include <cstdio>
#include <cstdlib>
#include <unordered_map>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", \
                         __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

static const auto HOST = llama_cache_acct_resource_domain::non_device(
    llama_cache_acct_residency::pageable_host);

static server_cache_yield_candidate candidate(
        uint64_t artifact,
        uint64_t op,
        common_retention_pool pool,
        uint64_t stable,
        uint64_t recency) {
    server_cache_yield_candidate out;
    out.artifact_id = { artifact };
    out.availability = server_retention_candidate_availability::available;
    out.identity_known = true;
    out.lease.state = server_cache_lease_eval_state::known;
    out.lease.cls = server_cache_lease_class::none;
    out.lease.eligibility = server_cache_lease_eligibility::eligible;
    out.record.kind = common_retention_artifact_kind::host_entry;
    auto turns = std::make_shared<common_retention_turn_table>();
    turns->source = common_retention_source_state::known;
    turns->token_count = 10;
    turns->boundaries.push_back({ 0, 0, 1 });
    out.record.turns = std::move(turns);
    out.record.stamp.state = common_retention_score_state::known;
    out.record.stamp.pool = pool;
    out.record.stamp.stable_id = stable;
    out.record.stamp.lineage_id = stable;
    out.record.stamp.recency_ordinal = recency;
    out.record.stamp.coverage_tokens = 10;
    out.lineage.pool = pool;
    out.lineage.lineage_id = stable;
    out.lineage.admission_epoch = 1;
    out.lineage.frequency_epoch = 1;
    out.release_ops = { llama_cache_acct_op_id{op} };
    return out;
}

static server_live_retention_candidate live_candidate(
        int32_t slot,
        uint64_t artifact,
        uint64_t lineage,
        uint64_t coverage,
        uint64_t recency,
        uint64_t cells,
        bool eligible = true) {
    server_live_retention_candidate out;
    out.slot_id = slot;
    out.artifact_id = { artifact };
    out.stamp.state = common_retention_score_state::known;
    out.stamp.pool = common_retention_pool::attention;
    out.stamp.stable_id = artifact;
    out.stamp.lineage_id = lineage;
    out.stamp.recency_ordinal = recency;
    out.stamp.coverage_tokens = coverage;
    out.lineage.pool = common_retention_pool::attention;
    out.lineage.lineage_id = lineage;
    out.lineage.admission_epoch = 1;
    out.lineage.frequency_epoch = 1;
    out.marginal_cells = cells;
    out.eligible = eligible;
    return out;
}

static void test_live_retention_projection() {
    constexpr uint64_t epoch = 10;
    auto hot = live_candidate(0, 10, 10, 100, 1, 100);
    hot.lineage.state = common_retention_frequency_state::promoted;
    hot.lineage.reuse_hits = 4;
    hot.lineage.frequency_q = 4*COMMON_RETENTION_FREQUENCY_ONE;
    hot.lineage.frequency_epoch = epoch;
    hot.lineage.last_credit_epoch = epoch;
    auto cold = live_candidate(1, 20, 20, 100, 2, 100);
    std::vector<server_live_retention_candidate> candidates = { hot, cold };
    CHECK(server_live_retention_prepare(
        candidates.data(), candidates.size(), epoch));
    auto projected = server_live_retention_project_prepared(
        candidates.data(), candidates.size(), epoch);
    CHECK(projected.complete);
    CHECK(projected.candidate_count == 2);
    CHECK(projected.slot_id == 1);
    CHECK(projected.artifact_id.v == 20);
    CHECK(projected.lost_work_tokens == 100);
    CHECK(projected.marginal_cells == 100);

    // Victim removal changes the next choice through a linear re-projection of the prepared
    // inventory; no identity/group re-sort is required between victims in one pressure wave.
    for (auto & candidate : candidates) {
        if (candidate.artifact_id == projected.artifact_id) {
            candidate.present = false;
        }
    }
    projected = server_live_retention_project_prepared(
        candidates.data(), candidates.size(), epoch);
    CHECK(projected.complete);
    CHECK(projected.candidate_count == 1);
    CHECK(projected.slot_id == 0);
    CHECK(projected.artifact_id.v == 10);

    // With identical reuse/work, the entry that releases more physical cells has lower value
    // density and is the lawful pressure victim.
    auto small = live_candidate(14, 110, 110, 100, 13, 10);
    auto large = live_candidate(15, 111, 111, 100, 13, 100);
    candidates = { small, large };
    CHECK(server_live_retention_prepare(
        candidates.data(), candidates.size(), epoch));
    projected = server_live_retention_project_prepared(
        candidates.data(), candidates.size(), epoch);
    CHECK(projected.complete);
    CHECK(projected.slot_id == 15);
    CHECK(projected.artifact_id.v == 111);
    CHECK(projected.marginal_cells == 100);

    // An ineligible shorter alias still retains its lineage's shared prefix. Only the unique
    // 2-token tail disappears with the longer physical owner.
    auto longer = live_candidate(2, 30, 30, 82, 4, 200);
    auto shorter = live_candidate(3, 31, 30, 80, 3, 0, false);
    shorter.lineage = longer.lineage;
    candidates = { longer, shorter };
    CHECK(server_live_retention_prepare(
        candidates.data(), candidates.size(), epoch));
    projected = server_live_retention_project_prepared(
        candidates.data(), candidates.size(), epoch);
    CHECK(projected.complete);
    CHECK(projected.candidate_count == 1);
    CHECK(projected.slot_id == 2);
    CHECK(projected.lost_work_tokens == 2);
    CHECK(projected.marginal_cells == 200);

    auto external = live_candidate(6, 60, 60, 100, 6, 100);
    external.external_shared_coverage_tokens = 80;
    candidates = { external };
    CHECK(server_live_retention_prepare(
        candidates.data(), candidates.size(), epoch));
    projected = server_live_retention_project_prepared(
        candidates.data(), candidates.size(), epoch);
    CHECK(projected.complete);
    CHECK(projected.lost_work_tokens == 20);

    auto same_lineage = live_candidate(7, 61, 60, 70, 5, 0, false);
    same_lineage.lineage = external.lineage;
    candidates = { same_lineage, external };
    CHECK(server_live_retention_prepare(
        candidates.data(), candidates.size(), epoch));
    projected = server_live_retention_project_prepared(
        candidates.data(), candidates.size(), epoch);
    CHECK(projected.complete);
    CHECK(projected.artifact_id.v == 60);
    CHECK(projected.lost_work_tokens == 20);

    auto equal_a = live_candidate(8, 70, 70, 100, 7, 100);
    auto equal_b = live_candidate(9, 71, 70, 100, 8, 0, false);
    equal_b.lineage = equal_a.lineage;
    candidates = { equal_a, equal_b };
    CHECK(server_live_retention_prepare(
        candidates.data(), candidates.size(), epoch));
    projected = server_live_retention_project_prepared(
        candidates.data(), candidates.size(), epoch);
    CHECK(projected.complete);
    CHECK(projected.lost_work_tokens == 0);

    // Physical aliases have zero exclusive release and cannot make a pressure wave progress.
    auto shared = live_candidate(4, 40, 40, 100, 5, 0);
    candidates = { shared };
    CHECK(server_live_retention_prepare(
        candidates.data(), candidates.size(), epoch));
    projected = server_live_retention_project_prepared(
        candidates.data(), candidates.size(), epoch);
    CHECK(!projected.complete);
    CHECK(projected.candidate_count == 0);

    auto duplicate_slot = live_candidate(10, 80, 80, 10, 9, 10);
    auto duplicate_slot_other = live_candidate(10, 81, 81, 10, 10, 10);
    candidates = { duplicate_slot, duplicate_slot_other };
    CHECK(!server_live_retention_prepare(
        candidates.data(), candidates.size(), epoch));

    auto duplicate_artifact = live_candidate(16, 120, 120, 10, 10, 10);
    auto duplicate_artifact_other = live_candidate(17, 120, 121, 10, 11, 10);
    candidates = { duplicate_artifact, duplicate_artifact_other };
    CHECK(!server_live_retention_prepare(
        candidates.data(), candidates.size(), epoch));

    auto tie_high = live_candidate(12, 91, 91, 10, 11, 10);
    auto tie_low  = live_candidate(11, 90, 90, 10, 11, 10);
    candidates = { tie_high, tie_low };
    CHECK(server_live_retention_prepare(
        candidates.data(), candidates.size(), epoch));
    projected = server_live_retention_project_prepared(
        candidates.data(), candidates.size(), epoch);
    CHECK(projected.complete);
    CHECK(projected.artifact_id.v == 90);
    candidates = { tie_low, tie_high };
    CHECK(server_live_retention_prepare(
        candidates.data(), candidates.size(), epoch));
    projected = server_live_retention_project_prepared(
        candidates.data(), candidates.size(), epoch);
    CHECK(projected.complete);
    CHECK(projected.artifact_id.v == 90);

    auto overflow = live_candidate(
        13, 100, 100, UINT64_MAX, 12, 1);
    overflow.lineage.state = common_retention_frequency_state::promoted;
    overflow.lineage.reuse_hits = 2;
    overflow.lineage.frequency_q = 16*COMMON_RETENTION_FREQUENCY_ONE;
    overflow.lineage.frequency_epoch = epoch;
    overflow.lineage.last_credit_epoch = epoch;
    candidates = { overflow };
    CHECK(server_live_retention_prepare(
        candidates.data(), candidates.size(), epoch));
    projected = server_live_retention_project_prepared(
        candidates.data(), candidates.size(), epoch);
    CHECK(!projected.complete);

    // Impossible cross-lineage prefix evidence invalidates the entire projection, including
    // when that provider is retained rather than eligible for removal.
    auto invalid = live_candidate(5, 50, 50, 10, 6, 0, false);
    invalid.external_shared_coverage_tokens = 11;
    candidates = { cold, invalid };
    CHECK(!server_live_retention_prepare(
        candidates.data(), candidates.size(), epoch));
}

// Rebuild yield candidates from a decoded sidecar snapshot (op == stable_id, the
// way the fixtures mint them) so encode->decode->replay reproduction can be checked.
static std::vector<server_cache_yield_candidate> resume_candidates(
        const common_retention_sidecar_snapshot & snapshot) {
    std::vector<server_cache_yield_candidate> out;
    for (const auto & record : snapshot.artifacts) {
        auto value = candidate(
            record.stamp.stable_id,
            record.stamp.stable_id,
            record.stamp.pool,
            record.stamp.stable_id,
            record.stamp.recency_ordinal);
        value.record = record;
        out.push_back(std::move(value));
    }
    return out;
}

struct fixture {
    uint64_t serial = 17;
    uint64_t fit_after_release = 0;
    uint64_t total_before = 100;
    bool stale = false;
    std::unordered_map<uint64_t, uint64_t> bytes;

    server_cache_yield_preview_callback preview() {
        return [this](const auto & ops, uint64_t expected, auto & out) {
            out = {};
            if (stale || expected != serial) {
                return false;
            }
            out.accounting_serial = serial;
            uint64_t released = 0;
            for (const auto op : ops) {
                const auto it = bytes.find(op.v);
                if (it == bytes.end() ||
                    it->second > UINT64_MAX - released) {
                    return false;
                }
                released += it->second;
            }
            if (released > 0) {
                out.rows.push_back({ HOST, released, released });
            }
            return true;
        };
    }

    server_cache_yield_fits_callback fits() {
        return [this](const llama_cache_budget_plan & plan) {
            llama_cache_budget_result out;
            out.accounting_serial = serial;
            if (plan.accounting_serial != serial ||
                plan.entries.size() > 1 ||
                (!plan.entries.empty() &&
                 plan.entries[0].domain != HOST) ||
                (!plan.entries.empty() &&
                 plan.entries[0].release_bytes > total_before)) {
                return out;
            }
            const uint64_t released =
                plan.entries.empty() ? 0 : plan.entries[0].release_bytes;
            out.state = released >= fit_after_release
                ? llama_cache_budget_fit_state::fits
                : llama_cache_budget_fit_state::exceeds;
            llama_cache_budget_row row;
            row.resource.kind =
                llama_cache_budget_resource_kind::accounting_domain;
            row.resource.domain = HOST;
            row.current_resident =
                llama_cache_acct_value::measured(total_before);
            row.before = llama_cache_acct_value::measured(total_before);
            row.released = llama_cache_acct_value::measured(released);
            row.reserved = llama_cache_acct_value::measured(0);
            row.after =
                llama_cache_acct_value::measured(total_before - released);
            row.state = out.state;
            out.domains.push_back(row);
            return out;
        };
    }
};

static void test_atomic_assembler_contract() {
    const auto normalized = candidate(
        7, 9, common_retention_pool::attention, 7, 7);
    server_retention_candidate source;
    source.artifact_id = normalized.artifact_id;
    source.record = normalized.record;
    source.release_ops = normalized.release_ops;
    source.avail = server_retention_candidate_availability::available;

    server_cache_lease_table leases;
    std::vector<server_cache_yield_candidate> out;
    size_t resolve_calls = 0;
    CHECK(server_cache_yield_assemble(
        { source }, leases,
        [&](const auto &, auto &, auto & identity, bool & known) {
            resolve_calls++;
            identity = { "exec", "adapter", "media" };
            known = true;
        },
        out));
    CHECK(resolve_calls == 1);
    CHECK(leases.clock_samples() == 0);
    CHECK(out.size() == 1);
    CHECK(out[0].artifact_id.v == 7);
    CHECK(out[0].release_ops.size() == 1 &&
          out[0].release_ops[0].v == 9);
    CHECK(out[0].lease.state == server_cache_lease_eval_state::known);
}

static void test_status_names() {
    CHECK(std::string(server_cache_yield_status_name(
              server_cache_yield_status::fits)) == "fits");
    CHECK(std::string(server_cache_yield_status_name(
              server_cache_yield_status::insufficient_yield)) ==
          "insufficient_yield");
    CHECK(std::string(server_cache_yield_status_name(
              server_cache_yield_status::unsupported_required)) ==
          "unsupported_required");
    CHECK(std::string(server_cache_yield_status_name(
              server_cache_yield_status::unavailable)) == "unavailable");
    CHECK(std::string(server_cache_yield_status_name(
              server_cache_yield_status::_count)) == "invalid");
    CHECK(std::string(server_cache_yield_status_name(
              static_cast<server_cache_yield_status>(UINT8_MAX))) ==
          "invalid");
}

static void test_lineage_shadow_projection() {
    fixture f;
    f.bytes = { {1, 10}, {2, 20}, {3, 10} };
    auto alias_a = candidate(
        1, 1, common_retention_pool::attention, 1, 1);
    auto alias_b = candidate(
        2, 2, common_retention_pool::attention, 2, 2);
    // Exact aliases share one lineage ledger. Releasing either loses no
    // logical coverage; the projector emits one lineage candidate, not two
    // copies of the same value, and breaks the equal-density tie by recency.
    alias_b.record.stamp.lineage_id = 1;
    alias_b.lineage = alias_a.lineage;

    auto hot = candidate(
        3, 3, common_retention_pool::attention, 3, 3);
    hot.lineage.state = common_retention_frequency_state::promoted;
    hot.lineage.reuse_hits = 2;
    hot.lineage.frequency_q = 2*COMMON_RETENTION_FREQUENCY_ONE;
    hot.lineage.last_credit_epoch = 1;

    const auto projected = server_retention_shadow_project(
        { hot, alias_a, alias_b }, 1, HOST, f.serial, f.preview());
    CHECK(projected.complete);
    CHECK(projected.alternatives.size() == 2);
    CHECK(projected.alternatives[0].lineage_id == 1);
    CHECK(projected.alternatives[0].artifact_ids.front().v == 1);
    CHECK(projected.alternatives[0].lost_work_units == 0);
    CHECK(projected.alternatives[1].lineage_id == 3);
    CHECK(projected.alternatives[1].lost_work_units == 10);

    // An ancestor alias can be discarded at zero logical loss while its
    // longer descendant remains.
    auto alias_b_turns =
        std::make_shared<common_retention_turn_table>(*alias_b.record.turns);
    alias_b_turns->token_count = 20;
    alias_b.record.turns = std::move(alias_b_turns);
    alias_b.record.stamp.coverage_tokens = 20;
    const auto suffix = server_retention_shadow_project(
        { alias_a, alias_b }, 1, HOST, f.serial, f.preview());
    CHECK(suffix.complete && suffix.alternatives.size() == 1);
    CHECK(suffix.alternatives[0].artifact_ids.front().v == 1);
    CHECK(suffix.alternatives[0].lost_work_units == 0);

    // If the ancestor is hard-protected it still contributes retained
    // coverage, but is never proposed. Removing the descendant then loses
    // only the unique suffix rather than charging the common stem twice.
    alias_a.lease.cls = server_cache_lease_class::hard;
    alias_a.lease.eligibility =
        server_cache_lease_eligibility::hard_blocked;
    const auto protected_ancestor = server_retention_shadow_project(
        { alias_a, alias_b }, 1, HOST, f.serial, f.preview());
    CHECK(protected_ancestor.complete &&
          protected_ancestor.alternatives.size() == 1);
    CHECK(protected_ancestor.alternatives[0].artifact_ids.front().v == 2);
    CHECK(protected_ancestor.alternatives[0].lost_work_units == 10);

    // Checkpoints are not prompt-payload substitutes or DF1 victims. A
    // same-frontier checkpoint cannot turn the live/host lost work into zero.
    auto checkpoint = alias_a;
    checkpoint.artifact_id = { 50 };
    checkpoint.record.kind = common_retention_artifact_kind::checkpoint;
    checkpoint.release_ops = { llama_cache_acct_op_id{50} };
    f.bytes.emplace(50, 1000);
    const auto without_checkpoint = server_retention_shadow_project(
        { alias_b, checkpoint }, 1, HOST, f.serial, f.preview());
    CHECK(without_checkpoint.complete);
    CHECK(without_checkpoint.alternatives.size() == 1);
    CHECK(without_checkpoint.alternatives[0].artifact_ids.front().v == 2);
    CHECK(without_checkpoint.alternatives[0].lost_work_units == 20);

    // Ordinary full live/host frontiers remain policy candidates even if the
    // legacy geometry scorer marked the frontier anchor. Mandatory status is
    // artifact-typed and retained only for checkpoints.
    auto full_host = candidate(
        60, 60, common_retention_pool::attention, 60, 60);
    full_host.record.stamp.mandatory_anchor = true;
    f.bytes.emplace(60, 10);
    const auto full_host_projection = server_retention_shadow_project(
        { full_host }, 1, HOST, f.serial, f.preview());
    CHECK(full_host_projection.complete);
    CHECK(full_host_projection.alternatives.size() == 1);

    // Two aliases may jointly release the last references to one allocation
    // even though neither has positive marginal yield by itself.
    auto shared_a = candidate(
        70, 70, common_retention_pool::attention, 70, 70);
    auto shared_b = candidate(
        71, 71, common_retention_pool::attention, 71, 71);
    shared_b.record.stamp.lineage_id = 70;
    shared_b.lineage = shared_a.lineage;
    const server_cache_yield_preview_callback shared_preview =
        [](const auto & ops, uint64_t serial, auto & out) {
            out = {};
            out.accounting_serial = serial;
            if (ops.size() == 2) {
                out.rows.push_back({ HOST, 0, 64 });
            }
            return true;
        };
    const auto shared = server_retention_shadow_project(
        { shared_a, shared_b }, 1, HOST, 91, shared_preview);
    CHECK(shared.complete);
    CHECK(shared.alternatives.size() == 1);
    CHECK(shared.alternatives[0].artifact_ids.size() == 2);
    CHECK(shared.alternatives[0].artifact_ids[0].v == 70);
    CHECK(shared.alternatives[0].artifact_ids[1].v == 71);
    CHECK(shared.alternatives[0].value.marginal_resource == 64);

    // Shared allocations are not restricted to equal-frontier aliases. The
    // exact compound {ancestor, descendant} must be discovered without also
    // evicting an independent longest-prefix member of the same lineage.
    auto shared_ancestor = candidate(
        80, 80, common_retention_pool::attention, 80, 80);
    auto shared_descendant = candidate(
        81, 81, common_retention_pool::attention, 81, 81);
    auto independent_tail = candidate(
        82, 82, common_retention_pool::attention, 82, 82);
    shared_ancestor.record.stamp.coverage_tokens = 10;
    shared_descendant.record.stamp.coverage_tokens = 20;
    independent_tail.record.stamp.coverage_tokens = 30;
    for (auto * value : {
            &shared_ancestor, &shared_descendant, &independent_tail }) {
        auto turns = std::make_shared<common_retention_turn_table>(
            *value->record.turns);
        turns->token_count = 30;
        value->record.turns = std::move(turns);
    }
    shared_descendant.record.stamp.lineage_id = 80;
    independent_tail.record.stamp.lineage_id = 80;
    shared_descendant.lineage = shared_ancestor.lineage;
    independent_tail.lineage = shared_ancestor.lineage;
    const server_cache_yield_preview_callback unequal_preview =
        [](const auto & ops, uint64_t serial, auto & out) {
            out = {};
            out.accounting_serial = serial;
            const auto has = [&](uint64_t id) {
                return std::any_of(ops.begin(), ops.end(),
                    [&](auto op) { return op.v == id; });
            };
            uint64_t bytes = has(82) ? 5 : 0;
            if (has(80) && has(81)) {
                bytes += 64;
            }
            if (bytes != 0) {
                out.rows.push_back({ HOST, 0, bytes });
            }
            return true;
        };
    const auto unequal = server_retention_shadow_project(
        { shared_ancestor, shared_descendant, independent_tail },
        1, HOST, 92, unequal_preview);
    CHECK(unequal.complete);
    CHECK(unequal.alternatives.size() == 1);
    CHECK(unequal.alternatives[0].artifact_ids.size() == 2);
    CHECK(unequal.alternatives[0].artifact_ids[0].v == 80);
    CHECK(unequal.alternatives[0].artifact_ids[1].v == 81);
    CHECK(unequal.alternatives[0].lost_work_units == 0);
    CHECK(unequal.alternatives[0].value.marginal_resource == 64);

    const server_cache_yield_preview_callback removes_max_preview =
        [](const auto & ops, uint64_t serial, auto & out) {
            out = {};
            out.accounting_serial = serial;
            const auto has = [&](uint64_t id) {
                return std::any_of(ops.begin(), ops.end(),
                    [&](auto op) { return op.v == id; });
            };
            if (has(80) && has(82)) {
                out.rows.push_back({ HOST, 0, 64 });
            }
            return true;
        };
    const auto removes_max = server_retention_shadow_project(
        { shared_ancestor, shared_descendant, independent_tail },
        1, HOST, 93, removes_max_preview);
    CHECK(removes_max.complete);
    CHECK(removes_max.alternatives.size() == 1);
    CHECK(removes_max.alternatives[0].artifact_ids.size() == 2);
    CHECK(removes_max.alternatives[0].artifact_ids[0].v == 80);
    CHECK(removes_max.alternatives[0].artifact_ids[1].v == 82);
    CHECK(removes_max.alternatives[0].lost_work_units == 10);

    fixture flood_fixture;
    std::vector<server_cache_yield_candidate> aliases;
    aliases.reserve(100);
    for (uint64_t i = 0; i < 100; ++i) {
        const uint64_t id = 100 + i;
        flood_fixture.bytes.emplace(id, 1);
        auto value = candidate(
            id, id, common_retention_pool::attention, id, id);
        value.record.stamp.lineage_id = 77;
        value.lineage = alias_a.lineage;
        value.lineage.lineage_id = 77;
        aliases.push_back(std::move(value));
    }
    const auto flood = server_retention_shadow_project(
        aliases, 1, HOST, flood_fixture.serial,
        flood_fixture.preview());
    CHECK(flood.complete);
    CHECK(flood.alternatives.size() == 1);
    CHECK(flood.alternatives[0].lineage_id == 77);
    CHECK(flood.alternatives[0].lost_work_units == 0);
}

static void test_planner_scan_preserves_valid_lease() {
    const auto normalized = candidate(
        7, 9, common_retention_pool::attention, 7, 7);
    server_retention_candidate source;
    source.artifact_id = normalized.artifact_id;
    source.record = normalized.record;
    source.release_ops = normalized.release_ops;
    source.avail = server_retention_candidate_availability::available;

    const server_cache_lease_identity valid_identity {
        "exec", "adapter", "media",
    };
    const server_cache_lease_identity wrong_identity {
        "exec", "other-adapter", "media",
    };
    server_cache_lease_table leases;
    const server_cache_lease_subject subject {
        normalized.artifact_id,
        common_retention_artifact_kind::host_entry,
        -1,
    };
    const auto scope = server_cache_lease_scope::from(
        server_cache_context_scope_id { 1 });
    CHECK(leases.grant_soft(
        subject, scope, valid_identity,
        server_cache_lease_table::IMPLICIT_SOFT_TTL_NS));
    const auto events_before = leases.event_snapshot();
    const auto unavailable_before = leases.unavailable_events();
    const auto samples_before = leases.clock_samples();

    std::vector<server_cache_yield_candidate> out;
    CHECK(server_cache_yield_assemble(
        { source }, leases,
        [&](const auto &, auto &, auto & identity, bool & known) {
            identity = wrong_identity;
            known = true;
        },
        out));
    CHECK(out.size() == 1);
    CHECK(out[0].lease.state == server_cache_lease_eval_state::unavailable);

    const auto events_after = leases.event_snapshot();
    CHECK(events_after.size == events_before.size);
    CHECK(events_after.last_ordinal == events_before.last_ordinal);
    CHECK(events_after.totals == events_before.totals);
    CHECK(events_after.identities.size() == events_before.identities.size());
    CHECK(leases.unavailable_events() == unavailable_before);
    CHECK(leases.clock_samples() == samples_before);
    const auto preserved = leases.inspect(
        normalized.artifact_id, valid_identity);
    CHECK(preserved.state == server_cache_lease_eval_state::known);
    CHECK(preserved.cls == server_cache_lease_class::soft);
}

static void test_policy_v1_mixed_pools() {
    fixture f;
    f.fit_after_release = 30;
    f.bytes = { {1, 10}, {2, 10}, {3, 10}, {4, 10} };
    std::vector<server_cache_yield_candidate> candidates = {
        candidate(4, 4, common_retention_pool::recurrent, 4, 4),
        candidate(2, 2, common_retention_pool::attention, 2, 2),
        candidate(3, 3, common_retention_pool::recurrent, 3, 3),
        candidate(1, 1, common_retention_pool::attention, 1, 1),
    };
    const auto result = server_cache_yield_plan(
        candidates, f.serial, f.preview(), f.fits());
    CHECK(result.status == server_cache_yield_status::fits);
    CHECK(result.yield_policy_version == 1);
    CHECK(result.selected[0].size() == 2);
    CHECK(result.selected[0][0].v == 1);
    CHECK(result.selected[0][1].v == 2);
    CHECK(result.selected[1].size() == 1);
    CHECK(result.selected[1][0].v == 3);
    CHECK(result.plan.size() == 1);
    CHECK(result.plan[0].release_bytes == 30);
    CHECK(result.projected_fit.accounting_serial == f.serial);
    CHECK(result.projected_fit.state == llama_cache_budget_fit_state::fits);
    CHECK(result.projected_fit.domains.size() == 1);
    CHECK(result.projected_fit.domains[0].released.value == 30);
}

static void configure_ledger(llama_cache_acct_ledger & ledger) {
    const llama_cache_acct_completeness_requirement req = {
        HOST, llama_cache_acct_producer::host_cache,
    };
    CHECK(ledger.configure_required_producers(&req, 1));
    CHECK(ledger.certify_complete(
        HOST, llama_cache_acct_producer::host_cache));
}

static void test_planner_shared_union() {
    llama_cache_acct_ledger ledger;
    configure_ledger(ledger);
    const auto alloc = ledger.new_alloc();
    const auto op0 = ledger.reserve(
        llama_cache_acct_category::full_snapshot_payload,
        HOST, {}, 100, 128);
    const auto op1 = ledger.reserve(
        llama_cache_acct_category::full_snapshot_payload,
        HOST, {}, 100, 128);
    CHECK(ledger.stage(op0, alloc, 128));
    CHECK(ledger.stage(op1, alloc, 128));
    CHECK(ledger.commit(op0, 100));
    CHECK(ledger.commit(op1, 100));
    const auto serial = ledger.snapshot().serial;

    auto first = candidate(
        1, op0.v, common_retention_pool::attention, 1, 1);
    auto second = candidate(
        2, op1.v, common_retention_pool::attention, 2, 2);
    const auto preview = [&](const auto & ops, uint64_t expected, auto & out) {
        return ledger.preview_release_set(ops, expected, out);
    };
    const auto fits = [serial](const llama_cache_budget_plan & plan) {
        llama_cache_budget_result out;
        out.accounting_serial = serial;
        const uint64_t released =
            plan.entries.empty() ? 0 : plan.entries[0].release_bytes;
        out.state = released >= 128
            ? llama_cache_budget_fit_state::fits
            : llama_cache_budget_fit_state::exceeds;
        return out;
    };
    const auto result = server_cache_yield_plan(
        { first, second }, serial, preview, fits);
    CHECK(result.status == server_cache_yield_status::fits);
    CHECK(result.selected[0].size() == 2);
    CHECK(result.plan.size() == 1);
    CHECK(result.plan[0].release_bytes == 128);
    CHECK(ledger.snapshot().serial == serial);
    CHECK(ledger.release(op0));
    CHECK(ledger.release(op1));
}

static void test_zero_marginal_candidate_skipped() {
    fixture f;
    f.fit_after_release = 20;
    f.bytes = { {1, 10} };
    const auto result = server_cache_yield_plan(
        {
            candidate(1, 1, common_retention_pool::attention, 1, 1),
            candidate(2, 1, common_retention_pool::attention, 2, 2),
        },
        f.serial, f.preview(), f.fits());
    CHECK(result.status == server_cache_yield_status::insufficient_yield);
    CHECK(result.selected[0].size() == 1);
    CHECK(result.selected[0][0].v == 1);
}

static void test_live_soft_order_and_serialized_bit() {
    fixture f;
    f.fit_after_release = 10;
    f.bytes = { {1, 10}, {2, 10} };
    auto first = candidate(
        1, 1, common_retention_pool::attention, 1, 1);
    auto second = candidate(
        2, 2, common_retention_pool::attention, 2, 2);
    first.lease.cls = server_cache_lease_class::soft;
    auto result = server_cache_yield_plan(
        { first, second }, f.serial, f.preview(), f.fits());
    CHECK(result.status == server_cache_yield_status::fits);
    CHECK(result.selected[0].size() == 1);
    CHECK(result.selected[0][0].v == 2);

    second.record.stamp.soft_leased = true;
    result = server_cache_yield_plan(
        { second }, f.serial, f.preview(), f.fits());
    CHECK(result.status == server_cache_yield_status::unavailable);
}

static void test_value_record_order_reproduction() {
    fixture f;
    f.fit_after_release = 20;
    f.bytes = { {1, 10}, {2, 10} };
    auto first = candidate(
        1, 1, common_retention_pool::attention, 1, 1);
    auto second = candidate(
        2, 2, common_retention_pool::attention, 2, 2);
    const auto original = server_cache_yield_plan(
        { second, first }, f.serial, f.preview(), f.fits());
    CHECK(original.status == server_cache_yield_status::fits);

    common_retention_sidecar_snapshot snapshot;
    snapshot.recency_high_water[0] = 2;
    snapshot.stable_high_water[0] = 2;
    snapshot.lineage_high_water[0] = 2;
    snapshot.lineages = {
        { common_retention_pool::attention,
          common_retention_frequency_state::probation, 1, 0, 0, 1, 1, 0, 1000 },
        { common_retention_pool::attention,
          common_retention_frequency_state::probation, 2, 0, 0, 1, 1, 0, 1000 },
    };
    snapshot.artifacts = { second.record, first.record };
    std::vector<uint8_t> bytes;
    CHECK(common_retention_sidecar_encode(snapshot, bytes));
    common_retention_sidecar_snapshot decoded;
    CHECK(common_retention_sidecar_decode(
        bytes.data(), bytes.size(), decoded));

    const auto resumed = resume_candidates(decoded);
    const auto replayed = server_cache_yield_plan(
        resumed, f.serial, f.preview(), f.fits());
    CHECK(replayed.status == server_cache_yield_status::fits);
    CHECK(replayed.selected[0] == original.selected[0]);
}

// D-S GATE — cross-pool determinism. Pins the invariants the shadow yield planner
// must hold before D-A can trust its evidence:
//   (1) attention selection is independent of recurrent presence (attention is
//       walked first from an empty union) and reflects the intra-pool sort, not
//       input order;
//   (2) recurrent selection legitimately DEPENDS on the accumulated attention
//       yield — it is the shortest recurrent prefix that fits after it;
//   (3) a recurrent candidate whose ops are already covered by the attention
//       union is zero-marginal and skipped (shared-op dedup).
static void test_cross_pool_independence() {
    auto make = [](uint64_t fit_after) {
        fixture f;
        f.fit_after_release = fit_after;
        f.bytes = { {1, 10}, {2, 10}, {3, 10}, {4, 10}, {5, 10} };
        return f;
    };

    // (1) attention alone is insufficient at fit=25 (a1+a2 = 20 < 25); its selected
    // prefix + order are identical whether recurrent is absent or present+permuted.
    fixture f = make(25);
    const auto att_only = server_cache_yield_plan(
        {
            candidate(2, 2, common_retention_pool::attention, 2, 2),
            candidate(1, 1, common_retention_pool::attention, 1, 1),
        },
        f.serial, f.preview(), f.fits());
    CHECK(att_only.status == server_cache_yield_status::insufficient_yield);
    CHECK(att_only.selected[0].size() == 2);
    CHECK(att_only.selected[0][0].v == 1);   // sorted by recency, not input order
    CHECK(att_only.selected[0][1].v == 2);
    CHECK(att_only.selected[1].empty());

    const auto with_rec = server_cache_yield_plan(
        {
            candidate(4, 4, common_retention_pool::recurrent, 4, 4),
            candidate(2, 2, common_retention_pool::attention, 2, 2),
            candidate(3, 3, common_retention_pool::recurrent, 3, 3),
            candidate(1, 1, common_retention_pool::attention, 1, 1),
        },
        f.serial, f.preview(), f.fits());
    CHECK(with_rec.status == server_cache_yield_status::fits);
    CHECK(with_rec.selected[0] == att_only.selected[0]);   // byte-identical prefix
    // (2) recurrent = shortest prefix that fits after attention (20): r3 -> 30 >= 25
    CHECK(with_rec.selected[1].size() == 1);
    CHECK(with_rec.selected[1][0].v == 3);

    // (2, cont.) recurrent selection legitimately shifts when the attention yield
    // shrinks: with only a1 (10) the recurrent walk needs r3 AND r4 to reach 25.
    // (Same immutable fit=25 fixture as above; the planner only reads it.)
    const auto thin = server_cache_yield_plan(
        {
            candidate(1, 1, common_retention_pool::attention, 1, 1),
            candidate(4, 4, common_retention_pool::recurrent, 4, 4),
            candidate(3, 3, common_retention_pool::recurrent, 3, 3),
        },
        f.serial, f.preview(), f.fits());
    CHECK(thin.status == server_cache_yield_status::fits);
    CHECK(thin.selected[0].size() == 1);
    CHECK(thin.selected[0][0].v == 1);
    CHECK(thin.selected[1].size() == 2);   // now needs both — depends on attention
    CHECK(thin.selected[1][0].v == 3);
    CHECK(thin.selected[1][1].v == 4);

    // (2, A/B) recurrent INTERNAL order is independent of attention presence: with
    // attention absent, {r4,r3} sorts to the same [r3,r4] the thin arm selected.
    const auto rec_only = server_cache_yield_plan(
        {
            candidate(4, 4, common_retention_pool::recurrent, 4, 4),
            candidate(3, 3, common_retention_pool::recurrent, 3, 3),
        },
        f.serial, f.preview(), f.fits());
    CHECK(rec_only.status == server_cache_yield_status::insufficient_yield);
    CHECK(rec_only.selected[0].empty());
    CHECK(rec_only.selected[1].size() == 2);
    CHECK(rec_only.selected[1][0].v == 3);
    CHECK(rec_only.selected[1][1].v == 4);
    CHECK(rec_only.selected[1] == thin.selected[1]);

    // (3) shared-op dedup: recurrent rs cites op 1, already freed by attention a1
    // -> zero-marginal -> skipped; rn (op 5) is the selected recurrent yield.
    fixture h = make(15);
    const auto dedup = server_cache_yield_plan(
        {
            candidate(1, 1, common_retention_pool::attention, 1, 1),
            candidate(90, 1, common_retention_pool::recurrent, 90, 1),
            candidate(91, 5, common_retention_pool::recurrent, 91, 2),
        },
        h.serial, h.preview(), h.fits());
    CHECK(dedup.status == server_cache_yield_status::fits);
    CHECK(dedup.selected[0].size() == 1);
    CHECK(dedup.selected[0][0].v == 1);
    CHECK(dedup.selected[1].size() == 1);
    CHECK(dedup.selected[1][0].v == 91);   // rs(90) skipped as zero-marginal
}

// D-S GATE — suspend/resume reproduction across BOTH pools (prior coverage was
// attention-only). Encode the retention sidecar, decode it, rebuild candidates in
// decoded order, and require the per-pool selection to reproduce byte-for-byte.
static void test_resume_both_pools() {
    fixture f;
    f.fit_after_release = 25;
    f.bytes = { {1, 10}, {2, 10}, {3, 10}, {4, 10} };
    auto a1 = candidate(1, 1, common_retention_pool::attention, 1, 1);
    auto a2 = candidate(2, 2, common_retention_pool::attention, 2, 2);
    auto r3 = candidate(3, 3, common_retention_pool::recurrent, 3, 3);
    auto r4 = candidate(4, 4, common_retention_pool::recurrent, 4, 4);
    const auto original = server_cache_yield_plan(
        { r4, a2, r3, a1 }, f.serial, f.preview(), f.fits());
    CHECK(original.status == server_cache_yield_status::fits);
    CHECK(original.selected[0].size() == 2);
    CHECK(original.selected[1].size() == 1);

    common_retention_sidecar_snapshot snapshot;
    snapshot.recency_high_water[0] = 2;
    snapshot.recency_high_water[1] = 4;
    snapshot.stable_high_water[0] = 2;
    snapshot.stable_high_water[1] = 4;
    snapshot.lineage_high_water[0] = 2;
    snapshot.lineage_high_water[1] = 4;
    snapshot.lineages = {
        { common_retention_pool::attention,
          common_retention_frequency_state::probation, 1, 0, 0, 1, 1, 0, 1000 },
        { common_retention_pool::attention,
          common_retention_frequency_state::probation, 2, 0, 0, 1, 1, 0, 1000 },
        { common_retention_pool::recurrent,
          common_retention_frequency_state::probation, 3, 0, 0, 1, 1, 0, 1000 },
        { common_retention_pool::recurrent,
          common_retention_frequency_state::probation, 4, 0, 0, 1, 1, 0, 1000 },
    };
    snapshot.artifacts = { r4.record, a2.record, r3.record, a1.record };
    std::vector<uint8_t> bytes;
    CHECK(common_retention_sidecar_encode(snapshot, bytes));
    common_retention_sidecar_snapshot decoded;
    CHECK(common_retention_sidecar_decode(
        bytes.data(), bytes.size(), decoded));

    const auto resumed = resume_candidates(decoded);
    const auto replayed = server_cache_yield_plan(
        resumed, f.serial, f.preview(), f.fits());
    CHECK(replayed.status == server_cache_yield_status::fits);
    CHECK(replayed.selected[0] == original.selected[0]);
    CHECK(replayed.selected[1] == original.selected[1]);
}

static void test_filters_and_terminals() {
    fixture f;
    f.fit_after_release = 10;
    f.bytes = { {1, 10} };
    auto unavailable = candidate(
        1, 1, common_retention_pool::attention, 1, 1);
    unavailable.lease.state = server_cache_lease_eval_state::unavailable;
    unavailable.lease.eligibility =
        server_cache_lease_eligibility::eligible;
    unavailable.has_unsupported_host_spill = true;
    auto result = server_cache_yield_plan(
        { unavailable }, f.serial, f.preview(), f.fits());
    CHECK(result.status == server_cache_yield_status::unavailable);
    CHECK(result.selected[0].empty());
    CHECK(result.unsupported.empty());

    // Synthetic priceable-only catalogs exercise the known P8 terminals.
    // Live catalogs remain unavailable until exact checkpoint/slot op sets land.
    auto protected_candidate = candidate(
        1, 1, common_retention_pool::attention, 1, 1);
    protected_candidate.record.stamp.mandatory_anchor = true;
    result = server_cache_yield_plan(
        { protected_candidate }, f.serial, f.preview(), f.fits());
    CHECK(result.status == server_cache_yield_status::insufficient_yield);

    auto insufficient = candidate(
        1, 1, common_retention_pool::attention, 1, 1);
    f.fit_after_release = 20;
    result = server_cache_yield_plan(
        { insufficient }, f.serial, f.preview(), f.fits());
    CHECK(result.status == server_cache_yield_status::insufficient_yield);

    // Synthetic available+spill input; the live resolver does not produce it.
    insufficient.has_unsupported_host_spill = true;
    result = server_cache_yield_plan(
        { insufficient }, f.serial, f.preview(), f.fits());
    CHECK(result.status == server_cache_yield_status::unsupported_required);
    CHECK(result.unsupported.size() == 1);
    CHECK(result.unsupported[0] == insufficient.artifact_id);
}

static void test_closed_filter_matrix() {
    fixture f;
    f.fit_after_release = 1;
    f.bytes = { {1, 1} };
    const auto run = [&](server_cache_yield_candidate value) {
        return server_cache_yield_plan(
            { std::move(value) }, f.serial, f.preview(), f.fits());
    };

    auto value = candidate(
        1, 1, common_retention_pool::attention, 1, 1);
    value.lease.cls = server_cache_lease_class::hard;
    value.lease.eligibility = server_cache_lease_eligibility::hard_blocked;
    CHECK(run(value).status == server_cache_yield_status::insufficient_yield);

    value = candidate(1, 1, common_retention_pool::attention, 1, 1);
    value.record.stamp.mandatory_anchor = true;
    CHECK(run(value).status == server_cache_yield_status::insufficient_yield);

    value = candidate(1, 1, common_retention_pool::attention, 1, 1);
    value.identity_known = false;
    CHECK(run(value).status == server_cache_yield_status::unavailable);

    value = candidate(1, 1, common_retention_pool::attention, 1, 1);
    value.availability =
        server_retention_candidate_availability::in_flight_mutation;
    CHECK(run(value).status == server_cache_yield_status::unavailable);

    value = candidate(1, 1, common_retention_pool::attention, 1, 1);
    value.record.stamp.state = common_retention_score_state::unavailable;
    value.record.stamp.mapped_turn_ordinal = 0;
    value.record.stamp.anchor_rank = 0;
    CHECK(run(value).status == server_cache_yield_status::unavailable);

    value = candidate(1, 99, common_retention_pool::attention, 1, 1);
    CHECK(run(value).status == server_cache_yield_status::unavailable);
}

static void test_empty_stale_and_capacity() {
    fixture f;
    f.fit_after_release = 0;
    auto result = server_cache_yield_plan(
        {}, f.serial, f.preview(), f.fits());
    CHECK(result.status == server_cache_yield_status::fits);
    CHECK(result.plan.empty());
    CHECK(result.projected_fit.accounting_serial == f.serial);
    CHECK(result.projected_fit.state == llama_cache_budget_fit_state::fits);
    CHECK(result.projected_fit.domains.size() == 1);
    CHECK(result.projected_fit.domains[0].released.value == 0);

    f.fit_after_release = 1;
    f.stale = true;
    result = server_cache_yield_plan(
        {}, f.serial, f.preview(), f.fits());
    // The baseline callback is independently serial-bound by the server. This
    // synthetic fit is non-fitting; the unavailable preview evidence is exposed
    // as soon as a candidate is present.
    CHECK(result.status == server_cache_yield_status::insufficient_yield);
    f.bytes = { {1, 1} };
    result = server_cache_yield_plan(
        { candidate(1, 1, common_retention_pool::attention, 1, 1) },
        f.serial, f.preview(), f.fits());
    CHECK(result.status == server_cache_yield_status::unavailable);

    std::vector<server_cache_yield_candidate> too_many(
        SERVER_CACHE_YIELD_MAX_CANDIDATES + 1);
    result = server_cache_yield_plan(
        too_many, f.serial, f.preview(), f.fits());
    CHECK(result.status == server_cache_yield_status::unavailable);
}

static void test_exception_isolation() {
    fixture f;
    f.fit_after_release = 1;
    f.bytes = { {1, 1} };
    const server_cache_yield_preview_callback throwing_preview =
        [](const auto &, uint64_t, auto &) -> bool {
            throw 1;
        };
    const auto result = server_cache_yield_plan(
        { candidate(1, 1, common_retention_pool::attention, 1, 1) },
        f.serial, throwing_preview, f.fits());
    CHECK(result.status == server_cache_yield_status::unavailable);
    CHECK(result.plan.empty());
    CHECK(result.projected_fit.state ==
          llama_cache_budget_fit_state::unavailable);

    size_t preview_calls = 0;
    const server_cache_yield_preview_callback drifts_at_prefix =
        [&preview_calls](const auto &, uint64_t serial, auto & out) {
            preview_calls++;
            if (preview_calls > 1) {
                return false;
            }
            out = {};
            out.accounting_serial = serial;
            return true;
        };
    const auto drifted = server_cache_yield_plan(
        { candidate(1, 1, common_retention_pool::attention, 1, 1) },
        f.serial, drifts_at_prefix, f.fits());
    CHECK(drifted.status == server_cache_yield_status::unavailable);
    CHECK(drifted.selected[0].empty());
    CHECK(drifted.plan.empty());
}

int main() {
    test_live_retention_projection();
    test_atomic_assembler_contract();
    test_status_names();
    test_lineage_shadow_projection();
    test_planner_scan_preserves_valid_lease();
    test_policy_v1_mixed_pools();
    test_planner_shared_union();
    test_zero_marginal_candidate_skipped();
    test_live_soft_order_and_serialized_bit();
    test_value_record_order_reproduction();
    test_cross_pool_independence();
    test_resume_both_pools();
    test_filters_and_terminals();
    test_closed_filter_matrix();
    test_empty_stale_and_capacity();
    test_exception_isolation();
    if (failures) {
        std::fprintf(stderr, "%d cache-yield checks failed\n", failures);
        return EXIT_FAILURE;
    }
    std::puts("cache-yield checks passed");
    return EXIT_SUCCESS;
}
