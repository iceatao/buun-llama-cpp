// Shadow-planner estimator/chooser contract tests: profile refusal (no default
// coefficients) + composition rule, calibration validation at the estimator boundary
// (mismatch / version-0 / NaN / negative), per-provider term estimation with version
// stamping, the controlled-disagreement fixture (two valid complete plans whose
// calibrated costs force the independent shadow choice away from the shipped choice,
// exact predicted saving asserted), tie-set membership + planner-owned stable choice,
// unavailability propagation (overflow / dropped derived plan / unresolved candidate /
// missing scalars / unknown n_prompt — each a closed status, all-or-nothing), root
// feasibility (component-only rows never win), and composed-chain estimation.

#include "common-cache-plan-estimate.h"
#include "common.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

using planner_status = common_cache_plan_planner_status;

static const common_cache_plan_calib TEST_CALIB = {
    /*profile*/           "test-model/test-gpu/b512",
    /*estimator_version*/ 1,
    /*replay_us_per_token*/ 100.0,
    /*restore_us_per_byte*/ 0.001,
    /*workspace_setup_us*/  500.0,
};

static common_cache_plan_record make_record(uint64_t n_prompt) {
    common_cache_plan_record rec;
    rec.selection           = common_cache_plan_selection::similarity;
    rec.calibration_profile = TEST_CALIB.profile; // the estimator enforces the match
    rec.n_prompt_tokens     = llama_cache_acct_value::measured(n_prompt);
    return rec;
}

static common_cache_plan_candidate * add_row(common_cache_plan_record & rec,
                                             common_cache_plan_provider prov, int32_t src,
                                             uint8_t phase, uint64_t lcp, uint64_t bytes,
                                             common_cache_plan_disposition disp) {
    auto * c = rec.find_or_add(
        prov, src, phase, -1, common_cache_plan_selection::similarity);
    c->lcp_tokens    = llama_cache_acct_value::measured(lcp);
    c->payload_bytes = llama_cache_acct_value::measured(bytes);
    c->disposition   = disp;
    return c;
}

// An unfitted profile refuses: the checked-in table starts empty.
static void test_profile_refusal() {
    CHECK(common_cache_plan_calib_find("qwen3.5-2b-q4km/rtx-3090/b512") == nullptr);
    CHECK(common_cache_plan_calib_find("") == nullptr);
}

static void test_restore_formula_door() {
    double restore = 0.0;
    double workspace = 0.0;
    CHECK(common_cache_plan_restore_us(
        TEST_CALIB, 1'000'000, restore, workspace));
    CHECK(restore == 1000.0);
    CHECK(workspace == 500.0);

    auto invalid = TEST_CALIB;
    invalid.estimator_version = 0;
    CHECK(!common_cache_plan_restore_us(
        invalid, 1, restore, workspace));
}

// the ONE producer-side profile-composition rule: lowercase, [a-z0-9/.] kept, rest '-'
// (a drifted spelling fails silently — estimators legally refuse forever)
static void test_profile_composition() {
    CHECK(common_cache_plan_calib_profile("Qwen3.5-2B-Q4_K_M", "NVIDIA GeForce RTX 3090", 512,
                                          "kf16-vf16") ==
          "qwen3.5-2b-q4-k-m/nvidia-geforce-rtx-3090/b512/kf16-vf16");
    CHECK(common_cache_plan_calib_profile("m", "cpu", 1, "kf16-vf16") == "m/cpu/b1/kf16-vf16");
    // model-desc + multi-GPU topology shapes survive the squash; KV regime never aliases
    CHECK(common_cache_plan_calib_profile("qwen3 1.7B Q4_K - Medium", "2x NVIDIA RTX 3090 sm1", 2048,
                                          "kvbr-vf16") ==
          "qwen3-1.7b-q4-k---medium/2x-nvidia-rtx-3090-sm1/b2048/kvbr-vf16");
    CHECK(common_cache_plan_calib_profile("m", "cpu", 1, "kf16-vf16") !=
          common_cache_plan_calib_profile("m", "cpu", 1, "kvbr-vf16"));
}

// Every effective VBR dimension must move the key; requested-string
// equality must not alias differing resolved regimes, and an unrepresentable override must
// yield no profile at all
static void test_vbr_regime_key() {
    common_cache_plan_vbr_regime base;
    base.armed = true; base.side_k = base.side_v = true;
    base.budget_mode = "dynamic"; base.family = "dyn"; base.policy = "p1";
    base.schedule = "sched-a"; base.capacity_bits = 4.5; base.selected_bpv = 4.2;
    base.vram_budget_bytes = 0; base.reclaim_floor_bpv = 8.125f; base.reset_keep_frac = 0.25f;
    const std::string ref = common_cache_plan_calib_kv(base, "f16", "f16");
    CHECK(ref.rfind("kvbr-vvbr", 0) == 0);

    // each resolved dimension changes the key
    const auto differs = [&](auto mutate) {
        common_cache_plan_vbr_regime v = base;
        mutate(v);
        return common_cache_plan_calib_kv(v, "f16", "f16") != ref;
    };
    CHECK(differs([](auto & v) { v.budget_mode = "fixed"; }));
    CHECK(differs([](auto & v) { v.family = "static"; }));
    CHECK(differs([](auto & v) { v.policy = "p2"; }));            // same request, resolved differently
    CHECK(differs([](auto & v) { v.schedule = "sched-b"; }));
    CHECK(differs([](auto & v) { v.capacity_bits = 3.0; }));
    CHECK(differs([](auto & v) { v.selected_bpv = 2.25; }));
    CHECK(differs([](auto & v) { v.vram_budget_bytes = 1ull << 30; })); // env budget override
    CHECK(differs([](auto & v) { v.reclaim_floor_bpv = 16.0f; }));
    CHECK(differs([](auto & v) { v.reset_keep_frac = 0.0f; }));
    CHECK(differs([](auto & v) { v.overrides = "VBR_BUDGET_MIB=4096"; }));
    CHECK(differs([](auto & v) { v.side_v = false; }));           // mixed/pinned side

    // identical resolved state is stable
    CHECK(common_cache_plan_calib_kv(base, "f16", "f16") == ref);
    // unrepresentable override -> NO profile (refuse), never an aliased match
    common_cache_plan_vbr_regime unk = base;
    unk.unrepresented_override = true;
    CHECK(common_cache_plan_calib_kv(unk, "f16", "f16").empty());
    // VBR armed via knobs only (no alias) still keys as a vbr regime, not plain f16
    common_cache_plan_vbr_regime knobs = base;
    knobs.side_k = knobs.side_v = false;
    const std::string knob_key = common_cache_plan_calib_kv(knobs, "f16", "f16");
    CHECK(knob_key != "kf16-vf16" && knob_key.find(" vbr ") != std::string::npos);
    // non-armed run keys on types alone
    common_cache_plan_vbr_regime off;
    CHECK(common_cache_plan_calib_kv(off, "f16", "q8_0") == "kf16-vq8_0");

    // End-to-end refusal: an empty segment must yield no profile, not a
    // composed "model/hw/bN/" key that merely fails to match
    CHECK(common_cache_plan_calib_profile("m", "gpu", 512,
              common_cache_plan_calib_kv(unk, "f16", "f16")).empty());
    CHECK(common_cache_plan_calib_profile("m", "", 512, "kf16-vf16").empty());
    CHECK(common_cache_plan_calib_profile("", "gpu", 512, "kf16-vf16").empty());
    CHECK(!common_cache_plan_calib_profile("m", "gpu", 512, "kf16-vf16").empty());

    // Checked-in RTX 3090 27B entry: reconstruct the measured default through the production
    // assembly helper, not a hand-built common_cache_plan_vbr_regime.
    common_params rtx3090_params;
    rtx3090_params.vbr_cache_type_k = rtx3090_params.vbr_cache_type_v = true;
    rtx3090_params.vbr_budget = "dynamic";
    rtx3090_params.vbr_selected_family = "dynamic";
    rtx3090_params.vbr_selected_policy = "runtime-controller";
    rtx3090_params.vbr_capacity_bits = 1.25;
    rtx3090_params.vbr_selected_bpv = 1.25;
    rtx3090_params.vbr_reclaim_floor_bpv = 8.125f;
    rtx3090_params.vbr_reset_keep_frac = 0.25f;
    const auto rtx3090 = common_cache_plan_vbr_regime_from_params(
        rtx3090_params, [](const char *) -> const char * { return nullptr; });
    const std::string rtx3090_profile = common_cache_plan_calib_profile(
        "qwen35-27b-q6-k", "nvidia-geforce-rtx-3090-ngl99", 2048,
        common_cache_plan_calib_kv(rtx3090, "f16", "f16"));
    CHECK(rtx3090_profile ==
          "qwen35-27b-q6-k/nvidia-geforce-rtx-3090-ngl99/b2048/"
          "kvbr-vvbr-vbr-dynamic-dynamic-runtime-controller--1.25-1.25-0-8.125-0.25");
    CHECK(common_cache_plan_calib_find(rtx3090_profile) != nullptr);
}

static void write_test_file(const std::filesystem::path & path, const std::string & bytes) {
    std::ofstream out(path, std::ios::binary);
    CHECK(bool(out));
    out.write(bytes.data(), bytes.size());
    CHECK(bool(out));
}

// File-valued overrides key on SHA-256 content. Moving identical
// content cannot move identity; editing a file in place must; unreadable content refuses.
static void test_vbr_file_override_identity() {
    using grammar = common_cache_plan_vbr_value_grammar;

    const auto unique = std::to_string(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const auto root = std::filesystem::temp_directory_path() /
                      ("llama-cache-plan-vbr-" + unique);
    const auto dir_a = root / "a";
    const auto dir_b = root / "b";
    std::filesystem::create_directories(dir_a);
    std::filesystem::create_directories(dir_b);

    const std::string schedule = "il0k=t8;il0v=f16";
    const auto file_a = dir_a / "schedule.txt";
    const auto file_b = dir_b / "renamed.txt";
    write_test_file(file_a, schedule);
    write_test_file(file_b, schedule);

    std::string a;
    std::string b;
    CHECK(common_cache_plan_vbr_override_identity(
        "VBR_LAYER_SCHEDULE", "@" + file_a.string(), grammar::inline_or_path, a));
    CHECK(common_cache_plan_vbr_override_identity(
        "VBR_LAYER_SCHEDULE", file_b.string(), grammar::inline_or_path, b));
    CHECK(a == b); // same bytes, different path and path grammar

    std::string inline_id;
    CHECK(common_cache_plan_vbr_override_identity(
        "VBR_LAYER_SCHEDULE", schedule, grammar::inline_or_path, inline_id));
    CHECK(inline_id == a); // inline and file forms share the content identity
    std::string spaced_id;
    CHECK(common_cache_plan_vbr_override_identity(
        "VBR_LAYER_SCHEDULE", "  @" + file_b.string() + "  ",
        grammar::inline_or_path, spaced_id));
    CHECK(spaced_id == b); // authoritative runtime trim prevents file-vs-inline aliasing

    write_test_file(file_a, schedule + "il1k=t4\n");
    std::string changed;
    CHECK(common_cache_plan_vbr_override_identity(
        "VBR_LAYER_SCHEDULE", "@" + file_a.string(), grammar::inline_or_path, changed));
    CHECK(changed != a); // same path, changed bytes

    const auto order_a = dir_a / "degrade-order.txt";
    const auto order_b = dir_b / "degrade-order.txt";
    write_test_file(order_a, "il0k:t8\n");
    write_test_file(order_b, "il0k:t8\n");
    std::string order_id_a;
    std::string order_id_b;
    CHECK(common_cache_plan_vbr_override_identity(
        "VBR_DEGRADE_ORDER", order_a.string(), grammar::path, order_id_a));
    CHECK(common_cache_plan_vbr_override_identity(
        "VBR_DEGRADE_ORDER", order_b.string(), grammar::path, order_id_b));
    CHECK(order_id_a == order_id_b);
    write_test_file(order_a, "il0k:t4\n");
    CHECK(common_cache_plan_vbr_override_identity(
        "VBR_DEGRADE_ORDER", order_a.string(), grammar::path, changed));
    CHECK(changed != order_id_a);

    std::string missing = "must-clear";
    CHECK(!common_cache_plan_vbr_override_identity(
        "VBR_DEGRADE_ORDER", (root / "missing.txt").string(), grammar::path, missing));
    CHECK(missing.empty());

    // Policy directories resolve to policy_ladder.json and use the same content rule.
    const auto policy_dir = root / "policy";
    std::filesystem::create_directories(policy_dir);
    write_test_file(policy_dir / "policy_ladder.json", "{\"version\":1}\n");
    std::string policy_dir_id;
    std::string policy_file_id;
    CHECK(common_cache_plan_vbr_override_identity(
        "VBR_POLICY_LADDER", policy_dir.string(), grammar::dir_or_file, policy_dir_id));
    CHECK(common_cache_plan_vbr_override_identity(
        "VBR_POLICY_LADDER", (policy_dir / "policy_ladder.json").string(),
        grammar::dir_or_file,
        policy_file_id));
    CHECK(policy_dir_id == policy_file_id);

    // Pin the primitive and rendering, not only relative equality.
    const auto sha_file = root / "sha-vector.txt";
    write_test_file(sha_file, "abc");
    std::string sha_id;
    CHECK(common_cache_plan_sha256_file_identity(sha_file.string(), sha_id));
    CHECK(sha_id ==
          "sha256-ba7816bf8f01cfea414140de5dae2223"
          "b00361a396177a9cb410ff61f20015ad");

    // Scalar census values remain literal and are not fed through the digest domain.
    std::string scalar;
    CHECK(common_cache_plan_vbr_override_identity(
        "VBR_BUDGET_MIB", "4096", grammar::scalar, scalar));
    CHECK(scalar == "VBR_BUDGET_MIB=4096");

    // Exercise the production assembly helper: unarmed means zero environment work;
    // successful tokens have census order; a later unreadable file latches refusal without
    // erasing either earlier or later successfully represented tokens.
    common_params params;
    // This estimator fixture tests the unarmed/static regime independent of
    // the common CLI's ambient cache default.
    params.vbr_cache_type_k = false;
    params.vbr_cache_type_v = false;
    int getenv_calls = 0;
    const auto no_env = [&](const char *) -> const char * {
        getenv_calls++;
        return nullptr;
    };
    const auto unarmed = common_cache_plan_vbr_regime_from_params(params, no_env);
    CHECK(!unarmed.armed);
    CHECK(getenv_calls == 0);

    params.vbr_cache_type_k = params.vbr_cache_type_v = true;
    std::map<std::string, std::string> env = {
        {"VBR_BUDGET_MIB", "4096"},
        {"VBR_DEGRADE_ORDER", (root / "missing.txt").string()},
        {"VBR_MODE", "1"},
    };
    const auto getenv_map = [&](const char * name) -> const char * {
        const auto it = env.find(name);
        return it == env.end() ? nullptr : it->second.c_str();
    };
    const auto refused = common_cache_plan_vbr_regime_from_params(params, getenv_map);
    CHECK(refused.unrepresented_override);
    CHECK(refused.overrides == "VBR_BUDGET_MIB=4096 VBR_MODE=1");
    CHECK(common_cache_plan_calib_profile(
        "m", "gpu", 1, common_cache_plan_calib_kv(refused, "f16", "f16")).empty());

    // Policy selection sets both params.vbr_selected_schedule and VBR_LAYER_SCHEDULE.
    // Assembly represents the schedule exactly once through the cost-affecting census row;
    // VBR_SELECTED_SCHEDULE remains typed as a path but is non-costing telemetry.
    params.vbr_selected_schedule = file_b.string();
    env = {{"VBR_LAYER_SCHEDULE", "  @" + file_b.string() + "  "}};
    const auto selected = common_cache_plan_vbr_regime_from_params(params, getenv_map);
    CHECK(!selected.unrepresented_override);
    CHECK(selected.schedule.empty());
    CHECK(selected.overrides == b);
    const size_t first_hash = selected.overrides.find("sha256-");
    CHECK(first_hash != std::string::npos);
    CHECK(selected.overrides.find("sha256-", first_hash + 1) == std::string::npos);

    std::filesystem::remove_all(root);
}

// The placement key is pure and positional: reversed heterogeneous device
// orders produce DISTINCT keys, equivalent inputs are stable, empty/ngl==0 maps to cpu
static void test_placement_key() {
    const float ts37[2] = {0.37f, 0.63f};
    const std::vector<std::string> ab = {"RTX 3090", "P100"};
    const std::vector<std::string> ba = {"P100", "RTX 3090"};
    // reversed heterogeneous order differs (mg/ts are positional)
    CHECK(common_cache_plan_calib_hw(ab, 40, 1, 0, ts37) !=
          common_cache_plan_calib_hw(ba, 40, 1, 0, ts37));
    // equivalent inputs are stable
    CHECK(common_cache_plan_calib_hw(ab, 40, 1, 0, ts37) ==
          common_cache_plan_calib_hw(ab, 40, 1, 0, ts37));
    CHECK(common_cache_plan_calib_hw(ab, 40, 1, 0, ts37) ==
          "RTX 3090+P100 ngl40 sm1 mg0 ts37-63");
    // single device: no positional suffixes
    CHECK(common_cache_plan_calib_hw({"RTX 3090"}, 99, 0, 0, nullptr) == "RTX 3090 ngl99");
    // empty placement or ngl==0 -> cpu; null tensor_split -> auto zeros
    CHECK(common_cache_plan_calib_hw({}, 99, 0, 0, nullptr) == "cpu");
    CHECK(common_cache_plan_calib_hw(ab, 0, 1, 0, ts37) == "cpu");
    CHECK(common_cache_plan_calib_hw(ab, 40, 1, 1, nullptr) ==
          "RTX 3090+P100 ngl40 sm1 mg1 ts0-0");
}

// The estimator is a trust boundary: profile mismatch, unreviewed
// version, and non-finite/negative coefficients all refuse as invalid_calibration
static void test_calibration_validation() {
    common_cache_plan_record rec = make_record(100);
    add_row(rec, common_cache_plan_provider::cold_replay, -1, 0, 0, 0,
            common_cache_plan_disposition::accepted);

    common_cache_plan_calib bad = TEST_CALIB;
    bad.profile = "other-model/other-gpu/b64";
    CHECK(common_cache_plan_estimate_and_choose(rec, bad) == planner_status::invalid_calibration);
    CHECK(rec.shadow_choice == -1);

    bad = TEST_CALIB; bad.estimator_version = 0;
    CHECK(common_cache_plan_estimate_and_choose(rec, bad) == planner_status::invalid_calibration);
    bad = TEST_CALIB; bad.replay_us_per_token = -1.0;
    CHECK(common_cache_plan_estimate_and_choose(rec, bad) == planner_status::invalid_calibration);
    bad = TEST_CALIB; bad.restore_us_per_byte = 0.0 / 0.0; // NaN
    CHECK(common_cache_plan_estimate_and_choose(rec, bad) == planner_status::invalid_calibration);

    // record whose profile is EMPTY never matches a fitted entry either
    common_cache_plan_record unprofiled;
    unprofiled.selection = common_cache_plan_selection::similarity;
    unprofiled.n_prompt_tokens = llama_cache_acct_value::measured(10);
    add_row(unprofiled, common_cache_plan_provider::cold_replay, -1, 0, 0, 0,
            common_cache_plan_disposition::accepted);
    CHECK(common_cache_plan_estimate_and_choose(unprofiled, TEST_CALIB) ==
          planner_status::invalid_calibration);

    // and the valid pairing still succeeds after all the refusals above (no residue)
    CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) == planner_status::ok);
}

// term-by-term estimation semantics + version stamping + D-owned terms untouched
static void test_basic_estimation() {
    common_cache_plan_record rec = make_record(1000);

    auto * cold = add_row(rec, common_cache_plan_provider::cold_replay, -1, 0,
                          0, 0, common_cache_plan_disposition::accepted);
    auto * live = add_row(rec, common_cache_plan_provider::live_slot, 0,
                          COMMON_CACHE_PLAN_PHASE_SIMILARITY, 800, 0,
                          common_cache_plan_disposition::valid_not_chosen_cost);
    auto * host = add_row(rec, common_cache_plan_provider::host_cache_entry, 0,
                          COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 900, 1'000'000,
                          common_cache_plan_disposition::accepted);

    CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) == planner_status::ok);

    // cold: 1000 tokens * 100us = 100000us, no restore/workspace
    CHECK(cold->predicted_total_us.value == 100000);
    CHECK(cold->cost_terms[size_t(llama_cache_acct_cost_kind::replay)].raw.value == 1000);
    CHECK(cold->cost_terms[size_t(llama_cache_acct_cost_kind::restore)].estimated_us.value == 0);
    // live: 200 replay tokens * 100 = 20000us
    CHECK(live->predicted_total_us.value == 20000);
    // host: 1MB * 0.001 = 1000us restore + 100 tokens * 100 = 10000us + 500 workspace
    CHECK(host->predicted_total_us.value == 11500);
    CHECK(host->cost_terms[size_t(llama_cache_acct_cost_kind::restore)].raw.value == 1'000'000);
    // version stamped ONLY on estimated terms; D-owned terms stay typed-unavailable
    CHECK(host->cost_terms[size_t(llama_cache_acct_cost_kind::replay)].estimator_version == 1);
    CHECK(host->cost_terms[size_t(llama_cache_acct_cost_kind::transfer)].estimated_us.state ==
          llama_cache_acct_known::unknown);
    CHECK(host->cost_terms[size_t(llama_cache_acct_cost_kind::eviction)].estimated_us.state ==
          llama_cache_acct_known::unknown);
    // host is the clear minimum; live and cold are far outside the floor
    CHECK(rec.shadow_choice >= 0);
    CHECK(&rec.inventory[size_t(rec.shadow_choice)] == host);
    CHECK(rec.n_shadow_ties == 1);
}

// Controlled disagreement: two valid, complete plans; the shipped path
// chose the host entry (deepest prefix), but the calibrated costs make the live slot's
// pure-replay plan cheaper than the host's restore+replay plan. The independent shadow
// choice must disagree with the shipped choice by the exact predicted saving.
static void test_controlled_disagreement() {
    common_cache_plan_record rec = make_record(1000);

    // live slot: lcp 990 -> 10 replay tokens * 100us = 1000us total (no restore)
    auto * live = add_row(rec, common_cache_plan_provider::live_slot, 2,
                          COMMON_CACHE_PLAN_PHASE_SIMILARITY, 990, 0,
                          common_cache_plan_disposition::valid_not_chosen_cost);
    // host entry: lcp 999 (deeper — the shipped heuristic's pick) but a 50MB payload:
    // 50000us restore + 100us replay + 500us workspace = 50600us
    auto * host = add_row(rec, common_cache_plan_provider::host_cache_entry, 0,
                          COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 999, 50'000'000,
                          common_cache_plan_disposition::accepted);
    add_row(rec, common_cache_plan_provider::cold_replay, -1, 0, 0, 0,
            common_cache_plan_disposition::accepted);

    // the SHIPPED choice: host delivered (a simple, non-composed delivery)
    rec.select(common_cache_plan_provider::host_cache_entry, host);
    rec.chosen = common_cache_plan_provider::host_cache_entry;
    rec.shipped_plan_candidate = rec.selected[size_t(rec.chosen)];

    CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) == planner_status::ok);

    // exact disagreement: shadow picked the live slot, shipped picked host
    CHECK(rec.shadow_choice >= 0);
    CHECK(&rec.inventory[size_t(rec.shadow_choice)] == live);
    // shipped plan is NOT in the tie set (floor = max(5% of 1000, 100) = 100us)
    bool shipped_in_ties = false;
    for (uint32_t i = 0; i < rec.n_shadow_ties; i++) {
        shipped_in_ties = shipped_in_ties || rec.shadow_tie_set[i] == rec.shipped_plan_candidate;
    }
    CHECK(!shipped_in_ties);
    // exact predicted saving
    CHECK(live->predicted_total_us.value == 1000);
    CHECK(host->predicted_total_us.value == 50600);
    CHECK(host->predicted_total_us.value - live->predicted_total_us.value == 49600);
}

// tie set: totals within max(5%, 100us) tie; the choice is the smallest schema-v5
// (target_slot_id, provider, source_id, ordinal) key — never a function of the shipped choice
static void test_tie_set() {
    common_cache_plan_record rec = make_record(100);

    // two live slots with identical replay counts -> identical totals
    auto * a = add_row(rec, common_cache_plan_provider::live_slot, 7,
                       COMMON_CACHE_PLAN_PHASE_SIMILARITY, 90, 0,
                       common_cache_plan_disposition::valid_not_chosen_cost);
    auto * b = add_row(rec, common_cache_plan_provider::live_slot, 3,
                       COMMON_CACHE_PLAN_PHASE_SIMILARITY, 90, 0,
                       common_cache_plan_disposition::accepted);
    a->target_slot_id = 9;
    b->target_slot_id = 4;
    (void) a;

    CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) == planner_status::ok);
    CHECK(rec.n_shadow_ties == 2);
    // stable order: target 4 < 9; source/ordinal break ties within one target
    CHECK(&rec.inventory[size_t(rec.shadow_choice)] == b);

    // determinism: rerun on the same record yields the same result
    const int32_t first = rec.shadow_choice;
    CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) == planner_status::ok);
    CHECK(rec.shadow_choice == first && rec.n_shadow_ties == 2);
}

static void test_lru_spec_stratum_and_tie_determinism() {
    common_cache_plan_record rec = make_record(100);
    rec.selection = common_cache_plan_selection::lru;
    rec.id_slot = 9; // legacy LRU winner establishes the hard spec stratum

    const auto add_target = [&](int32_t target, bool spec, uint64_t lcp,
                                common_cache_plan_selection origin) {
        auto * live = rec.find_or_add(
            common_cache_plan_provider::live_slot, target,
            COMMON_CACHE_PLAN_PHASE_LRU, target, origin);
        CHECK(live != nullptr);
        live->lcp_tokens = llama_cache_acct_value::measured(lcp);
        live->payload_bytes = llama_cache_acct_value::measured(0);
        live->disposition =
            common_cache_plan_disposition::valid_not_chosen_cost;
        live->spec_capable = spec;
        live->spec_capable_known = true;
        auto * cold = rec.find_or_add(
            common_cache_plan_provider::cold_replay,
            COMMON_CACHE_PLAN_SOURCE_AGGREGATE, COMMON_CACHE_PLAN_PHASE_LRU,
            target, origin);
        CHECK(cold != nullptr);
        cold->disposition =
            common_cache_plan_disposition::valid_not_chosen_cost;
        return live;
    };

    add_target(9, true, 10, common_cache_plan_selection::lru);
    auto * spec_7 = add_target(
        7, true, 90, common_cache_plan_selection::route_home);
    auto * spec_6 = add_target(
        6, true, 90, common_cache_plan_selection::similarity);
    auto * non_spec = add_target(
        4, false, 100, common_cache_plan_selection::lru);

    CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) ==
          planner_status::ok);
    CHECK(rec.n_shadow_ties == 2);
    CHECK(&rec.inventory[size_t(rec.shadow_choice)] == spec_6);
    CHECK(non_spec->predicted_total_us.value == 0); // cheapest, but wrong stratum
    CHECK(spec_6->predicted_total_us.value ==
          spec_7->predicted_total_us.value);

    // Missing target capability evidence refuses the whole LRU optimum rather
    // than silently dropping one target from the stratum comparison.
    spec_7->spec_capable_known = false;
    CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) ==
          planner_status::incomplete_evidence);
    CHECK(rec.shadow_choice == -1);
}

// unavailability propagation, each a closed status and ENTIRELY all-or-nothing:
// overflow, dropped derived plan, unresolved visited candidate, missing participant
// scalars, unknown n_prompt — never an optimum over a partial set
static void test_unavailability() {
    {
        common_cache_plan_record rec = make_record(100);
        add_row(rec, common_cache_plan_provider::cold_replay, -1, 0, 0, 0,
                common_cache_plan_disposition::accepted);
        rec.inventory_states[size_t(common_cache_plan_provider::live_slot)] =
            common_cache_plan_inventory_state::overflowed;
        CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) ==
              planner_status::incomplete_evidence);
        CHECK(rec.shadow_choice == -1);
    }
    {
        // a dropped derived plan (chain lost at capacity) refuses even when every
        // provider inventory looks intact
        common_cache_plan_record rec = make_record(100);
        add_row(rec, common_cache_plan_provider::cold_replay, -1, 0, 0, 0,
                common_cache_plan_disposition::accepted);
        rec.derived_plans_incomplete = true;
        CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) ==
              planner_status::incomplete_evidence);
    }
    {
        // an unresolved visited candidate (LRU-only slot, no reuse verdict) refuses:
        // honest refusal beats silently minimizing over a partial set
        common_cache_plan_record rec = make_record(100);
        add_row(rec, common_cache_plan_provider::cold_replay, -1, 0, 0, 0,
                common_cache_plan_disposition::accepted);
        auto * lru_only = rec.find_or_add(common_cache_plan_provider::live_slot, 4,
                                          COMMON_CACHE_PLAN_PHASE_LRU);
        lru_only->t_last_used_us = llama_cache_acct_value::measured(123);
        CHECK(lru_only->disposition == common_cache_plan_disposition::unavailable);
        CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) ==
              planner_status::incomplete_evidence);
        // all-or-nothing: the cold row carries NO leftover estimates after the refusal
        CHECK(rec.inventory[0].predicted_total_us.state == llama_cache_acct_known::unknown);
    }
    {
        // a VALID row missing its scalars refuses (never silently skipped)
        common_cache_plan_record rec = make_record(100);
        auto * c = rec.find_or_add(common_cache_plan_provider::live_slot, 0,
                                   COMMON_CACHE_PLAN_PHASE_SIMILARITY);
        c->disposition = common_cache_plan_disposition::accepted; // but no lcp evidence
        CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) ==
              planner_status::incomplete_evidence);
    }
    {
        common_cache_plan_record rec; // n_prompt unknown
        rec.selection = common_cache_plan_selection::similarity;
        rec.calibration_profile = TEST_CALIB.profile;
        add_row(rec, common_cache_plan_provider::cold_replay, -1, 0, 0, 0,
                common_cache_plan_disposition::accepted);
        CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) ==
              planner_status::incomplete_evidence);
    }
}

// Exact-capacity boundary: 96 real rows all fit, then the required
// chain is dropped — derived_plans_incomplete latches and the planner refuses
static void test_capacity_chain_boundary() {
    common_cache_plan_record rec = make_record(100);
    for (size_t i = 0; i < COMMON_CACHE_PLAN_MAX_CANDIDATES; i++) {
        add_row(rec, common_cache_plan_provider::host_cache_entry, (int32_t) i,
                COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 10, 100,
                common_cache_plan_disposition::valid_not_chosen_cost);
    }
    CHECK(rec.n_inventory == COMMON_CACHE_PLAN_MAX_CANDIDATES);
    CHECK(!rec.derived_plans_incomplete);
    CHECK(rec.add_chain(common_cache_plan_provider::host_cache_entry, 0, 1) == nullptr);
    CHECK(rec.derived_plans_incomplete);
    CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) ==
          planner_status::incomplete_evidence);
}

// Production shape of a composed delivery: the chain is the shipped
// plan; the bare checkpoint is component-only and can NEVER win the root optimum even
// when its standalone total is the cheapest number on the record
static void test_chain_composition_and_root_feasibility() {
    common_cache_plan_record rec = make_record(1000);

    auto * host = add_row(rec, common_cache_plan_provider::host_cache_entry, 0,
                          COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 500, 1'000'000,
                          common_cache_plan_disposition::accepted);
    auto * ckpt = add_row(rec, common_cache_plan_provider::live_context_checkpoint, 0,
                          COMMON_CACHE_PLAN_PHASE_CKPT_SCAN, 900, 2'000'000,
                          common_cache_plan_disposition::accepted);
    host->delivered = ckpt->delivered = true;
    ckpt->component_only = true; // its state was only reachable through the host restore

    auto * chain = rec.add_chain(common_cache_plan_provider::host_cache_entry,
                                 (int32_t) (host - rec.inventory.data()),
                                 (int32_t) (ckpt - rec.inventory.data()));
    CHECK(chain != nullptr && chain->is_chain());
    chain->disposition = common_cache_plan_disposition::accepted;
    chain->delivered   = true;
    rec.shipped_plan_candidate = (int32_t) (chain - rec.inventory.data());

    CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) == planner_status::ok);
    // host: 1000 + 50000 + 500 = 51500; ckpt: 2000 + 10000 + 500 = 12500
    CHECK(host->predicted_total_us.value == 51500);
    CHECK(ckpt->predicted_total_us.value == 12500);
    // chain: restores 1000+2000, workspaces 500+500, replay = deepest (ckpt, 100 tok) 10000
    CHECK(chain->predicted_total_us.value == 14000);
    CHECK(chain->cost_terms[size_t(llama_cache_acct_cost_kind::restore)].raw.value == 3'000'000);
    CHECK(chain->cost_terms[size_t(llama_cache_acct_cost_kind::replay)].raw.value == 100);
    // the bare checkpoint is the cheapest total on the record — but it is component-only,
    // so the CHAIN is the shadow choice and the shipped plan agrees with it
    CHECK(&rec.inventory[size_t(rec.shadow_choice)] == chain);
    CHECK(rec.shadow_choice == rec.shipped_plan_candidate);
    bool ckpt_in_ties = false;
    const int32_t ckpt_ord = (int32_t) (ckpt - rec.inventory.data());
    for (uint32_t i = 0; i < rec.n_shadow_ties; i++) {
        ckpt_in_ties = ckpt_in_ties || rec.shadow_tie_set[i] == ckpt_ord;
    }
    CHECK(!ckpt_in_ties);
}

// A non-delivered checkpoint sibling exposed by the host restore is
// also component-only, and its true complete plan (host→sibling cost-loser chain) competes
// in the root optimum — a cheaper sibling chain is a legitimate shadow disagreement
static void test_sibling_chain_alternative() {
    common_cache_plan_record rec = make_record(1000);

    auto * host = add_row(rec, common_cache_plan_provider::host_cache_entry, 0,
                          COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 500, 1'000'000,
                          common_cache_plan_disposition::accepted);
    // the DELIVERED (shipped) checkpoint: shallow, small
    auto * sel  = add_row(rec, common_cache_plan_provider::live_context_checkpoint, 0,
                          COMMON_CACHE_PLAN_PHASE_CKPT_SCAN, 600, 1'000'000,
                          common_cache_plan_disposition::accepted);
    // a deeper valid sibling the shipped scan passed over (short-circuit order)
    auto * sib  = add_row(rec, common_cache_plan_provider::live_context_checkpoint, 1,
                          COMMON_CACHE_PLAN_PHASE_CKPT_SCAN, 950, 1'000'000,
                          common_cache_plan_disposition::valid_not_chosen_cost);
    host->delivered = sel->delivered = true;
    sel->component_only = sib->component_only = true;

    const int32_t host_ord = (int32_t) (host - rec.inventory.data());
    auto * chain_sel = rec.add_chain(common_cache_plan_provider::host_cache_entry,
                                     host_ord, (int32_t) (sel - rec.inventory.data()));
    chain_sel->disposition = common_cache_plan_disposition::accepted;
    chain_sel->delivered   = true;
    rec.shipped_plan_candidate = (int32_t) (chain_sel - rec.inventory.data());
    auto * chain_sib = rec.add_chain(common_cache_plan_provider::host_cache_entry,
                                     host_ord, (int32_t) (sib - rec.inventory.data()));
    chain_sib->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);

    CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) == planner_status::ok);
    // shipped chain: 1000+1000 restore + 400*100 replay + 1000 ws = 43000
    // sibling chain: 2000 restore + 50*100 replay + 1000 ws = 8000 — the shadow winner
    CHECK(chain_sel->predicted_total_us.value == 43000);
    CHECK(chain_sib->predicted_total_us.value == 8000);
    CHECK(&rec.inventory[size_t(rec.shadow_choice)] == chain_sib);
    CHECK(rec.shadow_choice != rec.shipped_plan_candidate); // legitimate disagreement
    // neither bare checkpoint appears in the tie set
    for (uint32_t i = 0; i < rec.n_shadow_ties; i++) {
        CHECK(!rec.inventory[size_t(rec.shadow_tie_set[i])].component_only);
    }
}

static void test_destruction_eviction_term_participates() {
    common_cache_plan_record rec = make_record(100);
    auto * live = add_row(
        rec, common_cache_plan_provider::live_slot, 1,
        COMMON_CACHE_PLAN_PHASE_SIMILARITY, 95, 0,
        common_cache_plan_disposition::accepted);
    auto * cold = add_row(
        rec, common_cache_plan_provider::cold_replay, -1,
        COMMON_CACHE_PLAN_PHASE_SIMILARITY, 0, 0,
        common_cache_plan_disposition::valid_not_chosen_cost);
    auto & eviction = live->cost_terms[size_t(
        llama_cache_acct_cost_kind::eviction)];
    eviction.raw = llama_cache_acct_value::measured(4096);
    eviction.estimated_us = llama_cache_acct_value::measured(20'000);
    eviction.estimator_version = TEST_CALIB.estimator_version;

    CHECK(common_cache_plan_estimate_and_choose(rec, TEST_CALIB) ==
          planner_status::ok);
    CHECK(live->predicted_total_us.value == 20'500);
    CHECK(cold->predicted_total_us.value == 10'000);
    CHECK(&rec.inventory[size_t(rec.shadow_choice)] == cold);

    auto stale = make_record(100);
    auto * stale_live = add_row(
        stale, common_cache_plan_provider::live_slot, 1,
        COMMON_CACHE_PLAN_PHASE_SIMILARITY, 95, 0,
        common_cache_plan_disposition::accepted);
    auto & stale_eviction = stale_live->cost_terms[size_t(
        llama_cache_acct_cost_kind::eviction)];
    stale_eviction.raw = llama_cache_acct_value::measured(4096);
    stale_eviction.estimated_us = llama_cache_acct_value::measured(20'000);
    stale_eviction.estimator_version = TEST_CALIB.estimator_version + 1;
    CHECK(common_cache_plan_estimate_and_choose(stale, TEST_CALIB) ==
          planner_status::incomplete_evidence);
}

int main() {
    test_profile_refusal();
    test_restore_formula_door();
    test_profile_composition();
    test_placement_key();
    test_vbr_regime_key();
    test_vbr_file_override_identity();
    test_calibration_validation();
    test_basic_estimation();
    test_controlled_disagreement();
    test_tie_set();
    test_lru_spec_stratum_and_tie_determinism();
    test_unavailability();
    test_capacity_chain_boundary();
    test_chain_composition_and_root_feasibility();
    test_sibling_chain_alternative();
    test_destruction_eviction_term_participates();

    if (failures > 0) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    printf("all cache-plan-estimate tests passed\n");
    return EXIT_SUCCESS;
}
