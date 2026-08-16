#include "server-cache-yield.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", \
                         __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

const auto HOST = llama_cache_acct_resource_domain::non_device(
    llama_cache_acct_residency::pageable_host);

struct corpus_candidate {
    uint64_t artifact = 0;
    uint64_t lineage = 0;
    uint64_t coverage = 0;
    uint64_t bytes = 0;
    uint64_t recency = 0;
    uint64_t admitted = 1;
    uint64_t frequency_epoch = 1;
    uint64_t last_credit = 0;
    uint64_t hits = 0;
    uint64_t frequency_q = 0;
    uint32_t prior_milli = 1000;
    uint64_t external_shared_coverage = 0;
    bool hard = false;
    bool available = true;
};

server_cache_yield_candidate lower(const corpus_candidate & source) {
    server_cache_yield_candidate out;
    out.artifact_id = { source.artifact };
    out.availability = source.available
        ? server_retention_candidate_availability::available
        : server_retention_candidate_availability::backing_missing_or_stale;
    out.identity_known = source.available;
    out.lease.state = server_cache_lease_eval_state::known;
    out.lease.cls = source.hard
        ? server_cache_lease_class::hard
        : server_cache_lease_class::none;
    out.lease.eligibility = source.hard
        ? server_cache_lease_eligibility::hard_blocked
        : server_cache_lease_eligibility::eligible;
    out.record.kind = common_retention_artifact_kind::host_entry;
    auto turns = std::make_shared<common_retention_turn_table>();
    turns->source = common_retention_source_state::known;
    turns->token_count = source.coverage;
    turns->boundaries.push_back({ 0, 0, source.coverage });
    out.record.turns = std::move(turns);
    out.record.stamp.state = common_retention_score_state::known;
    out.record.stamp.pool = common_retention_pool::attention;
    out.record.stamp.stable_id = source.artifact;
    out.record.stamp.lineage_id = source.lineage;
    out.record.stamp.recency_ordinal = source.recency;
    out.record.stamp.coverage_tokens = source.coverage;
    out.lineage.pool = common_retention_pool::attention;
    out.lineage.lineage_id = source.lineage;
    out.lineage.admission_epoch = source.admitted;
    out.lineage.frequency_epoch = source.frequency_epoch;
    out.lineage.last_credit_epoch = source.last_credit;
    out.lineage.reuse_hits = source.hits;
    out.lineage.frequency_q = source.frequency_q;
    out.lineage.prior_milli = source.prior_milli;
    out.external_shared_coverage_tokens =
        source.external_shared_coverage;
    out.lineage.state = source.hits >= 2
        ? common_retention_frequency_state::promoted
        : common_retention_frequency_state::probation;
    out.release_ops = { llama_cache_acct_op_id { source.artifact } };
    return out;
}

struct corpus_result {
    std::string name;
    bool complete = false;
    size_t selected_artifacts = 0;
    uint64_t victim_lineage = 0;
    uint64_t victim_artifact = 0;
    uint64_t lost_work = 0;
    uint64_t released_bytes = 0;
};

corpus_result project(
        std::string name,
        std::vector<server_cache_yield_candidate> candidates,
        uint64_t epoch,
        const server_cache_yield_preview_callback & preview) {
    const auto projection = server_retention_shadow_project(
        candidates, epoch, HOST, 1, preview);
    corpus_result result;
    result.name = std::move(name);
    result.complete = projection.complete;
    if (projection.complete && !projection.alternatives.empty()) {
        const auto & victim = projection.alternatives.front();
        result.selected_artifacts = victim.artifact_ids.size();
        result.victim_lineage = victim.lineage_id;
        result.victim_artifact = victim.artifact_ids.empty()
            ? 0 : victim.artifact_ids.front().v;
        result.lost_work = victim.lost_work_units;
        result.released_bytes = victim.value.marginal_resource;
    }
    return result;
}

corpus_result run(
        std::string name,
        std::vector<server_cache_yield_candidate> candidates,
        std::unordered_map<uint64_t, uint64_t> bytes,
        uint64_t epoch) {
    const server_cache_yield_preview_callback preview =
        [bytes = std::move(bytes)](
                const auto & ops, uint64_t serial, auto & out) {
            out = {};
            out.accounting_serial = serial;
            uint64_t released = 0;
            for (const auto op : ops) {
                const auto found = bytes.find(op.v);
                if (found == bytes.end() ||
                    found->second > UINT64_MAX - released) {
                    return false;
                }
                released += found->second;
            }
            if (released != 0) {
                out.rows.push_back({ HOST, released, released });
            }
            return true;
        };
    return project(std::move(name), std::move(candidates), epoch, preview);
}

corpus_result run(
        std::string name,
        const std::vector<corpus_candidate> & source,
        uint64_t epoch) {
    std::vector<server_cache_yield_candidate> candidates;
    std::unordered_map<uint64_t, uint64_t> bytes;
    candidates.reserve(source.size());
    bytes.reserve(source.size());
    for (const auto & item : source) {
        candidates.push_back(lower(item));
        bytes.emplace(item.artifact, item.bytes);
    }
    return run(std::move(name), std::move(candidates), std::move(bytes), epoch);
}

common_chat_msg_spans spans() {
    common_chat_msg_spans out;
    out.add(COMMON_CHAT_ROLE_SYSTEM, 0, 4);
    out.add(COMMON_CHAT_ROLE_USER, 4, 4);
    out.add(COMMON_CHAT_ROLE_ASSISTANT, 8, 4);
    return out;
}

corpus_result run_branch_flood() {
    server_retention_sidecar_store store;
    store.configure(nullptr, {}, nullptr);
    const auto main = server_retention_instance_key::for_slot(0);
    CHECK(store.publish(
        main, common_retention_pool::attention, spans(), true,
        80000, 80000, true));
    common_retention_lineage_record main_lineage;
    CHECK(store.lineage_for_instance(main, main_lineage));
    const auto tool_heavy_child = server_retention_instance_key::for_slot(1);

    for (int32_t i = 1; i <= 100; ++i) {
        CHECK(store.credit_reuse(main) !=
              common_retention_credit_result::unavailable);
        const auto child = server_retention_instance_key::for_slot(i);
        CHECK(store.branch(main, child));
        common_retention_lineage_record child_lineage;
        CHECK(store.lineage_for_instance(child, child_lineage));
        CHECK(child_lineage.lineage_id != main_lineage.lineage_id);
        CHECK(child_lineage.reuse_hits == 0);
        CHECK(child_lineage.frequency_q == 0);
        CHECK(child_lineage.state ==
              common_retention_frequency_state::probation);
    }
    CHECK(store.lineage_for_instance(main, main_lineage));
    CHECK(main_lineage.reuse_hits == 100);
    CHECK(main_lineage.state ==
          common_retention_frequency_state::promoted);

    // Repeated tool rounds belong only to the child that actually reuses its
    // prefix. They must not flow backward into the main source lineage.
    const auto main_before_tool_rounds = main_lineage;
    CHECK(store.begin_competition_wave());
    CHECK(store.credit_reuse(tool_heavy_child) ==
          common_retention_credit_result::credited);
    CHECK(store.begin_competition_wave());
    CHECK(store.credit_reuse(tool_heavy_child) ==
          common_retention_credit_result::credited);
    common_retention_lineage_record tool_lineage;
    CHECK(store.lineage_for_instance(tool_heavy_child, tool_lineage));
    CHECK(tool_lineage.reuse_hits == 2);
    CHECK(tool_lineage.state ==
          common_retention_frequency_state::promoted);
    CHECK(store.lineage_for_instance(main, main_lineage));
    CHECK(main_lineage.pool == main_before_tool_rounds.pool);
    CHECK(main_lineage.state == main_before_tool_rounds.state);
    CHECK(main_lineage.lineage_id == main_before_tool_rounds.lineage_id);
    CHECK(main_lineage.reuse_hits == main_before_tool_rounds.reuse_hits);
    CHECK(main_lineage.frequency_q == main_before_tool_rounds.frequency_q);
    CHECK(main_lineage.admission_epoch ==
          main_before_tool_rounds.admission_epoch);
    CHECK(main_lineage.frequency_epoch ==
          main_before_tool_rounds.frequency_epoch);
    CHECK(main_lineage.last_credit_epoch ==
          main_before_tool_rounds.last_credit_epoch);
    CHECK(main_lineage.prior_milli == main_before_tool_rounds.prior_milli);

    const auto catalog = store.candidate_snapshot();
    std::vector<server_cache_yield_candidate> candidates;
    std::unordered_map<uint64_t, uint64_t> bytes;
    candidates.reserve(catalog.size());
    bytes.reserve(catalog.size());
    for (const auto & source : catalog) {
        server_cache_yield_candidate candidate;
        candidate.artifact_id = source.artifact_id;
        candidate.record = source.record;
        candidate.lineage = source.lineage;
        candidate.availability = source.avail;
        candidate.identity_known = true;
        candidate.lease.state = server_cache_lease_eval_state::known;
        candidate.lease.cls = server_cache_lease_class::none;
        candidate.lease.eligibility =
            server_cache_lease_eligibility::eligible;
        candidate.release_ops = {
            llama_cache_acct_op_id { source.artifact_id.v },
        };
        candidates.push_back(std::move(candidate));
        bytes.emplace(source.artifact_id.v, 800);
    }
    auto result = run(
        "main_branch_flood", std::move(candidates), std::move(bytes),
        store.competition_epoch_value());
    CHECK(result.complete);
    CHECK(result.victim_lineage != main_lineage.lineage_id);
    CHECK(result.victim_lineage != tool_lineage.lineage_id);
    return result;
}

corpus_candidate item(
        uint64_t id,
        uint64_t coverage,
        uint64_t bytes,
        uint64_t recency) {
    return { id, id, coverage, bytes, recency };
}

void require_victim(
        const corpus_result & result,
        uint64_t lineage,
        uint64_t artifact = 0) {
    CHECK(result.complete);
    CHECK(result.victim_lineage == lineage);
    if (artifact != 0) {
        CHECK(result.victim_artifact == artifact);
    }
}

std::vector<corpus_result> execute_corpus() {
    std::vector<corpus_result> report;

    // C1: A repeatedly reused main lineage must survive a scan of disposable
    // branches. Tool-call activity on a branch cannot mutate the main ledger.
    report.push_back(run_branch_flood());

    // C2: Phase change. A was saturated but has slept for four half-lives;
    // B has recent separated reuse. A must become the cheaper victim.
    auto phase_a = item(2, 64000, 640, 1);
    phase_a.hits = 16;
    phase_a.frequency_q = 16*COMMON_RETENTION_FREQUENCY_ONE;
    phase_a.last_credit = 1;
    auto phase_b = item(3, 64000, 640, 2);
    phase_b.admitted = 25;
    phase_b.frequency_epoch = 25;
    phase_b.hits = 3;
    phase_b.frequency_q = 3*COMMON_RETENTION_FREQUENCY_ONE;
    phase_b.last_credit = 32;
    report.push_back(run("phase_change", { phase_a, phase_b }, 33));
    require_victim(report.back(), phase_a.lineage);

    // C3: A bounded family prior helps a new main beat an otherwise identical
    // one-shot entry, but is not a lease and decays away.
    auto prior_main = item(4, 32000, 320, 1);
    prior_main.prior_milli = 2000;
    auto ordinary = item(5, 32000, 320, 2);
    report.push_back(run("cold_start_prior", { prior_main, ordinary }, 1));
    require_victim(report.back(), ordinary.lineage);
    report.push_back(run(
        "cold_start_prior_aged", { prior_main, ordinary }, 513));
    require_victim(report.back(), prior_main.lineage);

    // C4: Avoided work matters. An expensive new context may be worth more
    // than a frequently reused trivial context despite probation.
    auto expensive = item(6, 100000, 100, 2);
    auto cheap_hot = item(7, 100, 100, 1);
    cheap_hot.hits = 4;
    cheap_hot.frequency_q = 4*COMMON_RETENTION_FREQUENCY_ONE;
    cheap_hot.last_credit = 1;
    report.push_back(run(
        "expensive_infrequent", { expensive, cheap_hot }, 1));
    require_victim(report.back(), cheap_hot.lineage);

    // C5: Frequency is not a pin. A large cheap hot entry loses to compact
    // valuable coverage when exact value density says it should.
    auto large_hot = item(8, 100, 1000, 1);
    large_hot.hits = 4;
    large_hot.frequency_q = 4*COMMON_RETENTION_FREQUENCY_ONE;
    large_hot.last_credit = 1;
    auto compact = item(9, 100, 50, 2);
    report.push_back(run("cheap_frequent", { large_hot, compact }, 1));
    require_victim(report.back(), large_hot.lineage);

    // C6: With no reuse evidence, probation reduces to deterministic recency
    // and stable-lineage ordering rather than random replacement.
    report.push_back(run("all_one_shot", {
        item(10, 1000, 100, 3),
        item(11, 1000, 100, 1),
        item(12, 1000, 100, 2),
    }, 1));
    require_victim(report.back(), 11);

    // C7: Exact aliases share one lineage value. Adding aliases must not
    // multiply credit; it only exposes a zero-lost-work physical release.
    const std::array<size_t, 5> alias_counts = { 1, 4, 8, 32, 100 };
    for (const size_t count : alias_counts) {
        std::vector<corpus_candidate> aliases;
        aliases.reserve(count + 1);
        auto source = item(20, 20000, 200, 1);
        source.hits = 2;
        source.frequency_q = 2*COMMON_RETENTION_FREQUENCY_ONE;
        source.last_credit = 1;
        aliases.push_back(source);
        for (size_t i = 1; i < count; ++i) {
            auto alias = source;
            alias.artifact = 20 + i;
            alias.recency = 1 + i;
            aliases.push_back(alias);
        }
        aliases.push_back(item(1000 + count, 1000, 100, 200));
        auto result = run(
            "alias_count_" + std::to_string(count), aliases, 1);
        CHECK(result.complete);
        if (count == 1) {
            require_victim(result, 1000 + count);
            CHECK(result.lost_work == 1000);
        } else {
            require_victim(result, source.lineage, source.artifact);
            CHECK(result.selected_artifacts == 1);
            CHECK(result.lost_work == 0);
            CHECK(result.released_bytes == 200);
        }
        report.push_back(std::move(result));
    }

    // C7b: A genuinely shared allocation is released only by the exact
    // last-reference compound. The corpus must not confuse one zero-yield
    // alias with physical progress.
    auto shared_a = lower(item(60, 20000, 0, 1));
    auto shared_b = lower(item(61, 20000, 0, 2));
    shared_b.record.stamp.lineage_id = shared_a.lineage.lineage_id;
    shared_b.lineage = shared_a.lineage;
    const server_cache_yield_preview_callback shared_preview =
        [](const auto & ops, uint64_t serial, auto & out) {
            out = {};
            out.accounting_serial = serial;
            if (ops.size() == 2) {
                out.rows.push_back({ HOST, 200, 200 });
            }
            return true;
        };
    auto shared = project(
        "shared_last_reference", { shared_a, shared_b }, 1,
        shared_preview);
    require_victim(shared, shared_a.lineage.lineage_id);
    CHECK(shared.selected_artifacts == 2);
    CHECK(shared.lost_work == 20000);
    CHECK(shared.released_bytes == 200);
    report.push_back(std::move(shared));

    // C7c: A prefix retained by another lineage reduces only avoided work;
    // it does not merge frequency ledgers or physical release ownership.
    auto shared_stem_child = item(62, 82000, 100, 1);
    shared_stem_child.external_shared_coverage = 80000;
    report.push_back(run("external_shared_child", {
        shared_stem_child, item(63, 10000, 100, 2),
    }, 1));
    require_victim(report.back(), shared_stem_child.lineage);
    CHECK(report.back().lost_work == 2000);

    auto shared_stem_main = item(64, 120000, 100, 1);
    shared_stem_main.external_shared_coverage = 80000;
    report.push_back(run("external_shared_main", {
        shared_stem_main, item(65, 50000, 100, 2),
    }, 1));
    require_victim(report.back(), shared_stem_main.lineage);
    CHECK(report.back().lost_work == 40000);

    auto external_alias_a = lower(item(73, 82000, 0, 1));
    auto external_alias_b = lower(item(74, 82000, 0, 2));
    external_alias_b.record.stamp.lineage_id =
        external_alias_a.lineage.lineage_id;
    external_alias_b.lineage = external_alias_a.lineage;
    external_alias_a.external_shared_coverage_tokens = 80000;
    external_alias_b.external_shared_coverage_tokens = 80000;
    auto external_compound = project(
        "external_shared_compound",
        { external_alias_b, external_alias_a }, 1, shared_preview);
    require_victim(
        external_compound, external_alias_a.lineage.lineage_id);
    CHECK(external_compound.selected_artifacts == 2);
    CHECK(external_compound.lost_work == 2000);
    CHECK(external_compound.released_bytes == 200);
    report.push_back(std::move(external_compound));

    auto invalid_shared = item(66, 82000, 100, 1);
    invalid_shared.external_shared_coverage = 82001;
    report.push_back(run("external_shared_invalid", {
        invalid_shared, item(67, 10000, 100, 2),
    }, 1));
    CHECK(!report.back().complete);
    CHECK(report.back().selected_artifacts == 0);

    invalid_shared.hard = true;
    report.push_back(run("external_shared_invalid_protected", {
        invalid_shared, item(68, 10000, 100, 2),
    }, 1));
    CHECK(!report.back().complete);
    CHECK(report.back().selected_artifacts == 0);

    // C8: Hard protection is an eligibility stratum above frequency.
    auto hard = item(30, 1000, 100, 1);
    hard.hard = true;
    auto lawful = item(31, 1000, 100, 2);
    lawful.hits = 4;
    lawful.frequency_q = 4*COMMON_RETENTION_FREQUENCY_ONE;
    lawful.last_credit = 1;
    report.push_back(run("hard_protection", { hard, lawful }, 1));
    require_victim(report.back(), lawful.lineage);

    // C9: Missing evidence fails the shadow closed. It never fabricates a
    // partial optimum that could later be mistaken for authority.
    auto missing = item(40, 1000, 100, 1);
    missing.available = false;
    report.push_back(run("unavailable_evidence", {
        missing, item(41, 1000, 100, 2),
    }, 1));
    CHECK(!report.back().complete);
    CHECK(report.back().selected_artifacts == 0);
    CHECK(report.back().victim_lineage == 0);

    // C10: Candidate order cannot perturb the deterministic result.
    std::vector<corpus_candidate> permutation = {
        item(50, 1000, 100, 4),
        item(51, 1000, 100, 2),
        item(52, 1000, 100, 3),
        item(53, 1000, 100, 1),
    };
    const auto reference = run("permutation_reference", permutation, 1);
    require_victim(reference, 53);
    report.push_back(reference);
    std::sort(permutation.begin(), permutation.end(),
        [](const auto & a, const auto & b) {
            return a.artifact < b.artifact;
        });
    size_t n_permutations = 0;
    do {
        const auto value = run("permutation", permutation, 1);
        CHECK(value.complete == reference.complete);
        CHECK(value.victim_lineage == reference.victim_lineage);
        CHECK(value.victim_artifact == reference.victim_artifact);
        CHECK(value.lost_work == reference.lost_work);
        CHECK(value.released_bytes == reference.released_bytes);
        n_permutations++;
    } while (std::next_permutation(
        permutation.begin(), permutation.end(),
        [](const auto & a, const auto & b) {
            return a.artifact < b.artifact;
        }));
    CHECK(n_permutations == 24);

    auto tie_a = item(54, 1000, 100, 1);
    auto tie_b = item(55, 1000, 100, 1);
    report.push_back(run("stable_id_tie", { tie_b, tie_a }, 1));
    require_victim(report.back(), tie_a.lineage);

    // C11: Arithmetic overflow and stale/unavailable physical previews all
    // make the counterfactual unusable. A partial alternative is never
    // exposed to the report consumer.
    report.push_back(run("quote_overflow", {
        item(70, UINT64_MAX, 1, 1),
        item(71, 1, 1, 2),
    }, 1));
    CHECK(!report.back().complete);
    CHECK(report.back().selected_artifacts == 0);

    const std::vector<server_cache_yield_candidate> preview_candidates = {
        lower(item(72, 1000, 100, 1)),
    };
    const server_cache_yield_preview_callback preview_failure =
        [](const auto &, uint64_t, auto &) { return false; };
    report.push_back(project(
        "preview_failure", preview_candidates, 1, preview_failure));
    CHECK(!report.back().complete);
    CHECK(report.back().selected_artifacts == 0);
    const server_cache_yield_preview_callback stale_preview =
        [](const auto &, uint64_t serial, auto & out) {
            out = {};
            out.accounting_serial = serial + 1;
            out.rows.push_back({ HOST, 100, 100 });
            return true;
        };
    report.push_back(project(
        "stale_preview", preview_candidates, 1, stale_preview));
    CHECK(!report.back().complete);
    CHECK(report.back().selected_artifacts == 0);

    return report;
}

enum class contention_policy : uint8_t {
    retained_fifo = 0,
    decayed_frequency,
};

struct contention_request {
    const char * key = nullptr;
    uint64_t prompt_tokens = 0;
};

struct contention_entry {
    std::string key;
    server_retention_instance_key instance;
    llama_tokens tokens;
    uint64_t prompt_tokens = 0;
    uint64_t recency = 0;
};

struct contention_result {
    const char * policy = nullptr;
    uint64_t requests = 0;
    uint64_t prompt_tokens = 0;
    uint64_t avoided_prefill_tokens = 0;
    uint64_t fresh_prefill_tokens = 0;
    uint64_t pressure_waves = 0;
    uint64_t evictions = 0;
    uint64_t main_evictions = 0;
    uint64_t tool_evictions = 0;
    uint64_t main_return_reused_tokens = 0;
    uint64_t tool_return_reused_tokens = 0;
    uint64_t evicted_lost_work_tokens = 0;
    uint64_t projected_lost_work_tokens = 0;
    uint64_t final_competition_epoch = 0;
    bool complete = true;
};

common_chat_msg_spans contention_spans(uint64_t prompt_tokens) {
    common_chat_msg_spans out;
    out.add(COMMON_CHAT_ROLE_USER, 0, prompt_tokens);
    return out;
}

llama_tokens contention_tokens(const contention_request & request) {
    llama_tokens out;
    out.reserve(request.prompt_tokens);
    const std::string key = request.key;
    if (key == "main" || key.rfind("agent-", 0) == 0) {
        constexpr uint64_t shared_stem = 80000;
        for (uint64_t i = 0; i < shared_stem; ++i) {
            out.push_back(llama_token(1000 + i));
        }
        uint64_t tail_base = key == "main" ? 200000 : 400000;
        for (const unsigned char ch : key) {
            tail_base = tail_base*33 + ch;
        }
        tail_base %= 100000000;
        for (uint64_t i = shared_stem; i < request.prompt_tokens; ++i) {
            out.push_back(llama_token(tail_base + i - shared_stem));
        }
        return out;
    }

    uint64_t base = 100000000;
    for (const unsigned char ch : key) {
        base = base*33 + ch;
    }
    base %= 1000000000;
    for (uint64_t i = 0; i < request.prompt_tokens; ++i) {
        out.push_back(llama_token(base + i));
    }
    return out;
}

uint64_t contention_lcp(
        const llama_tokens & a,
        const llama_tokens & b) {
    return uint64_t(std::mismatch(
        a.begin(), a.end(), b.begin(), b.end()).first - a.begin());
}

contention_result run_contention_workload(contention_policy policy) {
    static const std::array<contention_request, 18> requests = {{
        { "main",    120000 },
        { "seed-a",   80000 },
        { "seed-b",   80000 },
        { "main",    120000 },
        { "seed-c",   80000 },
        { "main",    120000 },
        { "agent-1",  82000 },
        { "agent-1",  82000 },
        { "agent-2",  82000 },
        { "agent-1",  82000 },
        { "agent-3",  82000 },
        { "agent-4",  82000 },
        { "agent-5",  82000 },
        { "agent-6",  82000 },
        { "agent-7",  82000 },
        { "agent-8",  82000 },
        { "main",    120000 },
        { "agent-1",  82000 },
    }};

    constexpr size_t capacity_entries = 4;
    constexpr uint64_t resident_bytes = 100;
    server_retention_sidecar_store store;
    store.configure(nullptr, {}, nullptr);
    contention_result result;
    result.policy = policy == contention_policy::retained_fifo
        ? "retained_fifo" : "decayed_frequency";
    if (!store.enable_prefix_tracking()) {
        result.complete = false;
        return result;
    }
    std::vector<contention_entry> cache;
    int32_t next_slot = 10000;
    uint64_t next_recency = 1;

    const auto find = [&](const char * key) {
        return std::find_if(cache.begin(), cache.end(),
            [&](const auto & entry) { return entry.key == key; });
    };
    const auto evict = [&](const server_retention_instance_key & incoming) {
        if (!store.begin_competition_wave()) {
            result.complete = false;
            return;
        }
        result.pressure_waves++;
        auto victim = cache.end();
        uint64_t projected_lost_work = 0;
        if (policy == contention_policy::retained_fifo) {
            victim = std::min_element(cache.begin(), cache.end(),
                [&](const auto & a, const auto & b) {
                    if (a.instance == incoming) {
                        return false;
                    }
                    if (b.instance == incoming) {
                        return true;
                    }
                    return a.recency < b.recency;
                });
        } else {
            const auto catalog = store.candidate_snapshot();
            std::unordered_map<uint64_t, uint64_t> external_coverage;
            external_coverage.reserve(catalog.size());
            const auto inventory = store.value_snapshots(
                &external_coverage,
                [](void * context,
                   const server_retention_value_snapshot & value) noexcept {
                    auto & coverage = *static_cast<
                        std::unordered_map<uint64_t, uint64_t> *>(context);
                    try {
                        return coverage.emplace(
                            value.artifact_id.v,
                            value.external_shared_coverage_tokens).second;
                    } catch (...) {
                        return false;
                    }
                });
            if (inventory.status !=
                    server_retention_value_snapshot_status::complete ||
                inventory.size != catalog.size() ||
                external_coverage.size() != catalog.size()) {
                result.complete = false;
                return;
            }
            std::vector<server_cache_yield_candidate> candidates;
            std::unordered_map<uint64_t, uint64_t> bytes;
            candidates.reserve(catalog.size());
            bytes.reserve(catalog.size());
            for (const auto & source : catalog) {
                server_cache_yield_candidate candidate;
                candidate.artifact_id = source.artifact_id;
                candidate.record = source.record;
                candidate.lineage = source.lineage;
                candidate.availability = source.avail;
                candidate.identity_known = true;
                candidate.lease.state =
                    server_cache_lease_eval_state::known;
                candidate.lease.cls = source.instance_key == incoming
                    ? server_cache_lease_class::hard
                    : server_cache_lease_class::none;
                candidate.lease.eligibility =
                    source.instance_key == incoming
                        ? server_cache_lease_eligibility::hard_blocked
                        : server_cache_lease_eligibility::eligible;
                candidate.release_ops = {
                    llama_cache_acct_op_id { source.artifact_id.v },
                };
                const auto coverage = external_coverage.find(
                    source.artifact_id.v);
                if (coverage == external_coverage.end()) {
                    result.complete = false;
                    return;
                }
                candidate.external_shared_coverage_tokens =
                    coverage->second;
                candidates.push_back(std::move(candidate));
                bytes.emplace(source.artifact_id.v, resident_bytes);
            }
            const auto projected = run("contention", std::move(candidates),
                std::move(bytes), store.competition_epoch_value());
            if (!projected.complete || projected.victim_artifact == 0) {
                result.complete = false;
                return;
            }
            victim = std::find_if(cache.begin(), cache.end(),
                [&](const auto & entry) {
                    return store.artifact_id(entry.instance).v ==
                           projected.victim_artifact;
                });
            if (victim != cache.end()) {
                projected_lost_work = projected.lost_work;
            }
        }
        if (victim == cache.end() || victim->instance == incoming) {
            result.complete = false;
            return;
        }
        uint64_t retained_prefix = 0;
        for (auto candidate = cache.begin(); candidate != cache.end();
             ++candidate) {
            if (candidate == victim) {
                continue;
            }
            retained_prefix = std::max(
                retained_prefix,
                contention_lcp(victim->tokens, candidate->tokens));
        }
        const uint64_t actual_lost_work =
            victim->prompt_tokens - retained_prefix;
        if (policy == contention_policy::decayed_frequency) {
            if (projected_lost_work != actual_lost_work) {
                result.complete = false;
                return;
            }
            result.projected_lost_work_tokens += projected_lost_work;
        }
        result.evicted_lost_work_tokens += actual_lost_work;
        result.main_evictions += victim->key == "main";
        result.tool_evictions += victim->key == "agent-1";
        result.evictions++;
        store.retire(victim->instance);
        cache.erase(victim);
    };

    for (size_t request_index = 0;
         request_index < requests.size(); ++request_index) {
        const auto & request = requests[request_index];
        auto request_tokens = contention_tokens(request);
        result.requests++;
        result.prompt_tokens += request.prompt_tokens;
        auto exact = find(request.key);
        auto source = cache.end();
        uint64_t hit_tokens = 0;
        for (auto candidate = cache.begin(); candidate != cache.end();
             ++candidate) {
            const uint64_t lcp = contention_lcp(
                request_tokens, candidate->tokens);
            if (lcp > hit_tokens) {
                source = candidate;
                hit_tokens = lcp;
            }
        }
        if (source != cache.end() &&
            store.credit_reuse(source->instance) ==
                common_retention_credit_result::unavailable) {
            result.complete = false;
            break;
        }
        if (request_index == requests.size() - 2) {
            result.main_return_reused_tokens = hit_tokens;
        }
        if (request_index == requests.size() - 1) {
            result.tool_return_reused_tokens = hit_tokens;
        }
        result.avoided_prefill_tokens += hit_tokens;
        result.fresh_prefill_tokens += request.prompt_tokens - hit_tokens;

        if (exact == cache.end()) {
            const auto destination =
                server_retention_instance_key::for_slot(next_slot++);
            bool admitted = true;
            if (source != cache.end() && hit_tokens != 0) {
                admitted = store.branch(source->instance, destination);
            }
            admitted = admitted && store.publish(destination,
                common_retention_pool::attention,
                contention_spans(request.prompt_tokens), true,
                request.prompt_tokens, request.prompt_tokens, true);
            admitted = admitted && store.publish_prefix(
                destination, "contention-shared-stem-v1", request_tokens);
            if (!admitted) {
                result.complete = false;
                break;
            }
            cache.push_back({ request.key, destination,
                std::move(request_tokens), request.prompt_tokens,
                next_recency++ });
            if (cache.size() > capacity_entries) {
                evict(destination);
            }
        }
        if (!result.complete) {
            break;
        }
    }
    result.final_competition_epoch = store.competition_epoch_value();
    return result;
}

void test_stateful_contention_workload(
        contention_result & fifo,
        contention_result & decayed) {
    fifo = run_contention_workload(contention_policy::retained_fifo);
    decayed = run_contention_workload(
        contention_policy::decayed_frequency);
    const auto main_tokens = contention_tokens({ "main", 120000 });
    const auto agent_1_tokens = contention_tokens({ "agent-1", 82000 });
    const auto agent_2_tokens = contention_tokens({ "agent-2", 82000 });
    const auto seed_tokens = contention_tokens({ "seed-a", 80000 });
    CHECK(contention_lcp(main_tokens, agent_1_tokens) == 80000);
    CHECK(contention_lcp(agent_1_tokens, agent_2_tokens) == 80000);
    CHECK(contention_lcp(main_tokens, seed_tokens) == 0);
    CHECK(fifo.complete);
    CHECK(decayed.complete);
    CHECK(fifo.requests == 18);
    CHECK(fifo.prompt_tokens == 1622000);
    CHECK(fifo.avoided_prefill_tokens == 1204000);
    CHECK(fifo.fresh_prefill_tokens == 418000);
    CHECK(fifo.pressure_waves == 10);
    CHECK(fifo.evictions == 10);
    CHECK(fifo.main_evictions == 1);
    CHECK(fifo.tool_evictions == 1);
    CHECK(fifo.main_return_reused_tokens == 80000);
    CHECK(fifo.tool_return_reused_tokens == 80000);
    CHECK(fifo.evicted_lost_work_tokens == 292000);
    CHECK(fifo.projected_lost_work_tokens == 0);
    CHECK(fifo.final_competition_epoch == 25);

    CHECK(decayed.requests == 18);
    CHECK(decayed.prompt_tokens == 1622000);
    CHECK(decayed.avoided_prefill_tokens == 1242000);
    CHECK(decayed.fresh_prefill_tokens == 380000);
    CHECK(decayed.pressure_waves == 10);
    CHECK(decayed.evictions == 10);
    CHECK(decayed.main_evictions == 0);
    CHECK(decayed.tool_evictions == 2);
    CHECK(decayed.main_return_reused_tokens == 120000);
    CHECK(decayed.tool_return_reused_tokens == 80000);
    CHECK(decayed.evicted_lost_work_tokens == 98000);
    CHECK(decayed.projected_lost_work_tokens == 98000);
    CHECK(decayed.final_competition_epoch == 25);

    // This is a retained-source counterfactual for the planned
    // non-consuming policy, not an efficacy replay of today's consuming
    // fixed-host default. Both arms run identical production-owned
    // admission, divergent branch, reuse-credit, pressure-wave, prefix-index,
    // and retirement transitions; only the pressure victim comparator
    // differs. Both arms account the best retained 80K shared stem, so an
    // evicted main/tool return is a partial hit rather than a false cold miss.
    CHECK(fifo.avoided_prefill_tokens + fifo.fresh_prefill_tokens ==
          fifo.prompt_tokens);
    CHECK(decayed.avoided_prefill_tokens +
          decayed.fresh_prefill_tokens == decayed.prompt_tokens);
}

void emit_json(
        const std::vector<corpus_result> & report,
        const contention_result & fifo,
        const contention_result & decayed) {
    std::printf("{\"version\":2,\"scenarios\":[");
    for (size_t i = 0; i < report.size(); ++i) {
        const auto & value = report[i];
        std::printf(
            "%s{\"name\":\"%s\",\"complete\":%s,"
            "\"selected_artifacts\":%zu,\"victim_lineage\":%llu,"
            "\"victim_artifact\":%llu,"
            "\"lost_work\":%llu,\"released_bytes\":%llu}",
            i == 0 ? "" : ",",
            value.name.c_str(), value.complete ? "true" : "false",
            value.selected_artifacts,
            (unsigned long long) value.victim_lineage,
            (unsigned long long) value.victim_artifact,
            (unsigned long long) value.lost_work,
            (unsigned long long) value.released_bytes);
    }
    const auto emit_workload = [](const contention_result & value) {
        std::printf(
            "{\"policy\":\"%s\",\"complete\":%s,"
            "\"requests\":%llu,\"prompt_tokens\":%llu,"
            "\"avoided_prefill_tokens\":%llu,"
            "\"fresh_prefill_tokens\":%llu,"
            "\"pressure_waves\":%llu,\"evictions\":%llu,"
            "\"main_evictions\":%llu,\"tool_evictions\":%llu,"
            "\"main_return_reused_tokens\":%llu,"
            "\"tool_return_reused_tokens\":%llu,"
            "\"evicted_lost_work_tokens\":%llu,"
            "\"projected_lost_work_tokens\":%llu,"
            "\"final_competition_epoch\":%llu}",
            value.policy, value.complete ? "true" : "false",
            (unsigned long long) value.requests,
            (unsigned long long) value.prompt_tokens,
            (unsigned long long) value.avoided_prefill_tokens,
            (unsigned long long) value.fresh_prefill_tokens,
            (unsigned long long) value.pressure_waves,
            (unsigned long long) value.evictions,
            (unsigned long long) value.main_evictions,
            (unsigned long long) value.tool_evictions,
            (unsigned long long) value.main_return_reused_tokens,
            (unsigned long long) value.tool_return_reused_tokens,
            (unsigned long long) value.evicted_lost_work_tokens,
            (unsigned long long) value.projected_lost_work_tokens,
            (unsigned long long) value.final_competition_epoch);
    };
    std::printf("],\"contention_workload\":{");
    std::printf("\"baseline\":");
    emit_workload(fifo);
    std::printf(",\"candidate\":");
    emit_workload(decayed);
    std::printf("}}\n");
}

} // namespace

int main(int argc, char ** argv) {
    const auto report = execute_corpus();
    contention_result fifo;
    contention_result decayed;
    test_stateful_contention_workload(fifo, decayed);
    const bool json = argc == 2 && std::string(argv[1]) == "--json";
    if (!json && argc != 1) {
        std::fprintf(stderr, "usage: %s [--json]\n", argv[0]);
        return 2;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d retention corpus checks failed\n", failures);
        return 1;
    }
    if (json) {
        emit_json(report, fifo, decayed);
    } else {
        std::printf(
            "retention policy corpus: PASS (%zu scenarios, "
            "shared-stem retained-source delta %llu modeled "
            "fresh-prefill tokens)\n",
            report.size(),
            (unsigned long long) (fifo.fresh_prefill_tokens -
                decayed.fresh_prefill_tokens));
    }
    return 0;
}
