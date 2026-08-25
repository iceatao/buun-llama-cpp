#include "common-cache-plan-estimate.h"
#include "arg.h"
#include "common.h"
#include "llama-sha256.h"
#include "llama-vbr-config.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <tuple>

// Fitted calibration table. Filled ONLY by the recorded RTX 3090 microbench sweep + offline fit
// (tools/server/bench/cache-plan-calibrate.py); entries are data reviewed like code.
// Profiles without an entry refuse; there are no default coefficients.

// RTX 3090 campaign on 2026-07-28 (36 cold records over 64-2048 tokens, 6 composed host->ckpt
// restores over 20-50 MiB payloads; replay intercept ~13.3 ms absorbed by the fit)
static const common_cache_plan_calib CALIB_QWEN35_2B_RTX3090_B512 = {
    "qwen35-2b-q4-k---medium/nvidia-geforce-rtx-3090-ngl99/b512/kf16-vf16", 1,
    75.688, 0.000010, 7184.4,
};

// RTX 3090 campaign on 2026-07-28, using the pinned 27B VBR serving configuration with
// -ngl 99 and the
// DEFAULT VBR ladder regime (dynamic budget / auto floor / auto vram / auto policy /
// reclaim 8.125 / reset-keep 0.25 — the key names it; a different ladder config is a
// different regime and must be re-fitted). (the
// auto-fit placement varied per launch — ngl64 measured 979 us/tok, all-GPU measures
// 861.5: the profile key distinguishing placements is load-bearing). 23 cold records,
// 5 checkpoint restores — hybrid payloads constant at 156.9 MiB, restore cost FLAT
// ~158 ms carried by workspace; crossover vs replay ~= 183 tokens.
static const common_cache_plan_calib CALIB_QWEN36_27B_VBR_RTX3090 = {
    "qwen35-27b-q6-k/nvidia-geforce-rtx-3090-ngl99/b2048/kvbr-vvbr-vbr-dynamic-dynamic-runtime-controller--1.25-1.25-0-8.125-0.25", 1,
    861.510, 0.0, 158069.2,
};

static const common_cache_plan_calib * const calib_table[] = {
    &CALIB_QWEN35_2B_RTX3090_B512,
    &CALIB_QWEN36_27B_VBR_RTX3090,
    nullptr, // sentinel so the array is never empty; skipped by the scan below
};

const common_cache_plan_calib * common_cache_plan_calib_find(const std::string & profile) {
    for (const auto * entry : calib_table) {
        if (entry && profile == entry->profile) {
            return entry;
        }
    }
    return nullptr;
}

bool common_cache_plan_restore_us(
        const common_cache_plan_calib & calib,
        uint64_t bytes,
        double & restore_us,
        double & workspace_us) noexcept {
    restore_us = 0.0;
    workspace_us = 0.0;
    if (!calib.profile || calib.estimator_version == 0 ||
        !std::isfinite(calib.restore_us_per_byte) ||
        calib.restore_us_per_byte < 0.0 ||
        !std::isfinite(calib.workspace_setup_us) ||
        calib.workspace_setup_us < 0.0) {
        return false;
    }
    restore_us = (double) bytes * calib.restore_us_per_byte;
    workspace_us = calib.workspace_setup_us;
    if (!std::isfinite(restore_us)) {
        restore_us = 0.0;
        workspace_us = 0.0;
        return false;
    }
    return true;
}

std::string common_cache_plan_sha256_hex_digest(const std::array<uint8_t, 32> & digest) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(digest.size()*2);
    for (uint8_t byte : digest) {
        out.push_back(hex[byte >> 4]);
        out.push_back(hex[byte & 0x0f]);
    }
    return out;
}

static std::string cache_plan_sha256_hex(const std::string & bytes) {
    llama_sha256 hash;
    hash.update(bytes.data(), bytes.size());
    return common_cache_plan_sha256_hex_digest(hash.finish());
}

bool common_cache_plan_sha256_file_identity(const std::string & path, std::string & identity) {
    identity.clear();

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    // chunked: llama_sha256::update is incremental, so a mistyped override naming a
    // multi-GB model file costs O(1) memory instead of materializing the whole file
    llama_sha256 hash;
    char buf[64 * 1024];
    while (file.read(buf, sizeof(buf)) || file.gcount() > 0) {
        hash.update(buf, (size_t) file.gcount());
    }
    if (file.bad()) {
        return false;
    }
    identity = "sha256-" + common_cache_plan_sha256_hex_digest(hash.finish());
    return true;
}

bool common_cache_plan_vbr_override_identity(const std::string & name,
                                             const std::string & value,
                                             common_cache_plan_vbr_value_grammar grammar,
                                             std::string & identity) {
    identity.clear();

    std::string content_identity;
    switch (grammar) {
        case common_cache_plan_vbr_value_grammar::scalar:
            identity = name + "=" + value;
            return true;
        case common_cache_plan_vbr_value_grammar::path:
            if (!common_cache_plan_sha256_file_identity(value, content_identity)) {
                return false;
            }
            break;
        case common_cache_plan_vbr_value_grammar::dir_or_file:
            if (!common_cache_plan_sha256_file_identity(
                    common_vbr_resolve_policy_file(value), content_identity)) {
                return false;
            }
            break;
        case common_cache_plan_vbr_value_grammar::inline_or_path: {
            std::string content;
            std::string source;
            if (!llama_vbr_resolve_layer_schedule(value.c_str(), content, source)) {
                return false;
            }
            content_identity = "sha256-" + cache_plan_sha256_hex(content);
            break;
        }
        default:
            return false;
    }

    identity = name + "=" + content_identity;
    return true;
}

common_cache_plan_vbr_regime common_cache_plan_vbr_regime_from_params(
        const common_params & params,
        const common_cache_plan_getenv_fn & getenv_fn) {
    common_cache_plan_vbr_regime vbr;
    vbr.armed  = params.vbr_enabled();
    vbr.side_k = params.vbr_cache_type_k;
    vbr.side_v = params.vbr_cache_type_v;
    vbr.budget_mode       = params.vbr_budget;
    vbr.family            = params.vbr_selected_family;
    vbr.policy            = params.vbr_selected_policy;
    // Schedule content is represented once by the VBR_LAYER_SCHEDULE census row. Keep
    // this compatibility segment empty so the checked-in default-regime key is stable.
    vbr.capacity_bits     = params.vbr_capacity_bits;
    vbr.selected_bpv      = params.vbr_selected_bpv;
    vbr.vram_budget_bytes = params.vbr_vram_budget_bytes;
    vbr.reclaim_floor_bpv = params.vbr_reclaim_floor_bpv;
    vbr.reset_keep_frac   = params.vbr_reset_keep_frac;

    // Nothing below can affect common_cache_plan_calib_kv's unarmed early return.
    if (!vbr.armed) {
        return vbr;
    }

#define COMMON_CACHE_PLAN_VBR_ENV_FOLD(NAME, AFFECTS, GRAMMAR)                         \
    if (AFFECTS) {                                                                     \
        if (const char * val = getenv_fn(NAME)) {                                      \
            std::string token;                                                         \
            if (!common_cache_plan_vbr_override_identity(                              \
                    NAME, val, common_cache_plan_vbr_value_grammar::GRAMMAR, token)) { \
                vbr.unrepresented_override = true;                                     \
            } else {                                                                   \
                vbr.overrides += (vbr.overrides.empty() ? "" : " ") + token;          \
            }                                                                          \
        }                                                                              \
    }
    COMMON_CACHE_PLAN_VBR_ENV_LIST(COMMON_CACHE_PLAN_VBR_ENV_FOLD)
#undef COMMON_CACHE_PLAN_VBR_ENV_FOLD

    return vbr;
}

std::string common_cache_plan_calib_profile(const std::string & model_stem,
                                            const std::string & hw_desc, int n_batch,
                                            const std::string & kv_desc) {
    // Refusal propagates: an empty segment means its regime could not
    // be established, so there is NO profile — composing "model/hw/bN/" would hand back a
    // nonempty key that merely fails to match, losing the no_profile distinction and
    // risking a collision with a genuinely-empty segment.
    if (model_stem.empty() || hw_desc.empty() || kv_desc.empty()) {
        return "";
    }
    std::string prof = model_stem + "/" + hw_desc + "/b" + std::to_string(n_batch)
                     + "/" + kv_desc;
    for (char & ch : prof) {
        ch = ch >= 'A' && ch <= 'Z' ? char(ch - 'A' + 'a') : ch;
        if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '/' || ch == '.')) {
            ch = '-';
        }
    }
    return prof;
}

std::string common_cache_plan_calib_kv(const common_cache_plan_vbr_regime & vbr,
                                       const std::string & type_k, const std::string & type_v) {
    const std::string k = (vbr.armed && vbr.side_k) ? "vbr" : type_k;
    const std::string v = (vbr.armed && vbr.side_v) ? "vbr" : type_v;
    const std::string kv = "k" + k + "-v" + v;
    if (!vbr.armed) {
        return kv;
    }
    if (vbr.unrepresented_override) {
        return ""; // effective regime unknown -> no profile -> planner refuses
    }
    // every dimension that moves the ladder's cost, in RESOLVED form; separators stay in
    // the profile sanitizer's squash class so the key renders legibly
    // EXACT numeric encoding (r4): %.17g round-trips a double bit-for-bit, so two
    // distinct resolved values can never render to the same token
    char nums[256];
    snprintf(nums, sizeof(nums), "%.17g %.17g %llu %.17g %.17g",
             vbr.capacity_bits, vbr.selected_bpv,
             (unsigned long long) vbr.vram_budget_bytes,
             (double) vbr.reclaim_floor_bpv, (double) vbr.reset_keep_frac);
    std::string out = kv + " vbr " + vbr.budget_mode + " " + vbr.family + " " + vbr.policy +
                      " " + vbr.schedule + " " + nums;
    if (!vbr.overrides.empty()) {
        out += " ovr " + vbr.overrides;
    }
    return out;
}

std::string common_cache_plan_calib_hw(const std::vector<std::string> & gpu_descs,
                                       int n_gpu_layers_eff, int split_mode, int main_gpu,
                                       const float * tensor_split) {
    if (gpu_descs.empty() || n_gpu_layers_eff == 0) {
        return "cpu";
    }
    std::string hw;
    for (const auto & d : gpu_descs) {
        hw += (hw.empty() ? "" : "+") + d;
    }
    hw += " ngl" + std::to_string(n_gpu_layers_eff);
    if (gpu_descs.size() > 1) {
        hw += " sm" + std::to_string(split_mode) + " mg" + std::to_string(main_gpu);
        std::string ts;
        for (size_t i = 0; i < gpu_descs.size(); i++) {
            const int pct = tensor_split ? (int) (tensor_split[i] * 100.0f) : 0;
            ts += (ts.empty() ? "" : "-") + std::to_string(pct);
        }
        hw += " ts" + ts;
    }
    return hw;
}

static const common_cache_plan_candidate * cache_plan_target_live_row(
        const common_cache_plan_record & rec,
        int32_t target_slot_id) noexcept {
    for (uint32_t i = 0; i < rec.n_inventory; ++i) {
        const auto & row = rec.inventory[i];
        if (row.provider == common_cache_plan_provider::live_slot &&
            row.target_slot_id == target_slot_id &&
            row.source_id == target_slot_id && !row.is_chain()) {
            return &row;
        }
    }
    return nullptr;
}

static bool cache_plan_base_row_participates(
        const common_cache_plan_record & rec,
        const common_cache_plan_candidate & row) noexcept {
    return common_cache_plan_origin_in_domain(
               row.origin_tier, rec.selection) &&
           row.viable() && !row.component_only;
}

static bool cache_plan_lru_stratum_complete(
        const common_cache_plan_record & rec) noexcept {
    if (rec.selection != common_cache_plan_selection::lru) {
        return true;
    }
    const auto * legacy = cache_plan_target_live_row(rec, rec.id_slot);
    if (!legacy || !legacy->spec_capable_known) {
        return false;
    }
    for (uint32_t i = 0; i < rec.n_inventory; ++i) {
        const auto & row = rec.inventory[i];
        if (!cache_plan_base_row_participates(rec, row)) {
            continue;
        }
        const auto * target = cache_plan_target_live_row(
            rec, row.target_slot_id);
        if (!target || !target->spec_capable_known) {
            return false;
        }
    }
    return true;
}

// Root-optimum membership: valid AND independently executable from the request's starting
// state — a component-only row (checkpoint exposed by a delivered host entry) is priced as
// a chain component but can never win on its own.
static bool cache_plan_row_participates(
        const common_cache_plan_record & rec,
        const common_cache_plan_candidate & c,
        const common_cache_plan_candidate * lru_legacy) {
    // Authority absorbs tiers cumulatively. A similarity decision may compare only
    // targets that crossed the strict similarity threshold; route-home/LRU
    // rows remain evidence but cannot win until their ratchets are enabled.
    if (!cache_plan_base_row_participates(rec, c)) {
        return false;
    }
    if (rec.selection != common_cache_plan_selection::lru) {
        return true;
    }
    // Speculative capability is a hard eligibility stratum matching legacy.
    // The planner has no calibrated speculation credit;
    // minimize only among targets in the legacy-selected stratum.
    const auto * target = cache_plan_target_live_row(rec, c.target_slot_id);
    return lru_legacy && target && lru_legacy->spec_capable_known &&
           target->spec_capable_known &&
           lru_legacy->spec_capable == target->spec_capable;
}

// the ONE place B terms are written and versions stamped. Optional D-owned
// transfer/eviction terms are included in the same predicted total.
static bool cache_plan_fill_terms(common_cache_plan_candidate & c,
                                  const common_cache_plan_calib & calib,
                                  uint64_t restore_bytes, double restore_us,
                                  uint64_t replay_tokens, double replay_us,
                                  double workspace_us) {
    const auto set_term = [&](llama_cache_acct_cost_kind kind, uint64_t raw, double us) {
        auto & term = c.cost_terms[size_t(kind)];
        term.raw               = llama_cache_acct_value::measured(raw);
        term.estimated_us      = llama_cache_acct_value::measured((uint64_t) std::llround(us));
        term.estimator_version = calib.estimator_version;
    };
    set_term(llama_cache_acct_cost_kind::restore,   restore_bytes, restore_us);
    set_term(llama_cache_acct_cost_kind::replay,    replay_tokens, replay_us);
    set_term(llama_cache_acct_cost_kind::workspace, 0,             workspace_us);
    long double total = restore_us + replay_us + workspace_us;
    // D owns these terms. They remain optional so a B-only record keeps its
    // historical estimate; when a D quote supplies one, the one B optimum
    // includes it without reimplementing the planner or its tie rule.
    for (const auto kind : {
            llama_cache_acct_cost_kind::transfer,
            llama_cache_acct_cost_kind::eviction }) {
        const auto & term = c.cost_terms[size_t(kind)];
        if (term.estimated_us.state == llama_cache_acct_known::known) {
            if (term.raw.state != llama_cache_acct_known::known ||
                term.estimator_version != calib.estimator_version) {
                c.predicted_total_us = {};
                return false;
            }
            total += term.estimated_us.value;
        }
    }
    if (!std::isfinite(total) || total < 0.0L ||
        total > (long double) UINT64_MAX) {
        c.predicted_total_us = {};
        return false;
    }
    c.predicted_total_us = llama_cache_acct_value::measured(
        (uint64_t) std::llround(total));
    return true;
}

// estimate one non-chain row; false = a needed scalar is missing (typed-unknown lcp/bytes)
static bool cache_plan_estimate_row(common_cache_plan_candidate & c, uint64_t n_prompt,
                                    const common_cache_plan_calib & calib) {
    uint64_t restore_bytes = 0;
    uint64_t replay_tokens = 0;
    bool     has_restore   = false;

    switch (c.provider) {
        case common_cache_plan_provider::cold_replay:
            replay_tokens = n_prompt;
            break;
        case common_cache_plan_provider::live_slot:
            // state is already installed: reuse the prefix, replay the rest
            if (c.lcp_tokens.state != llama_cache_acct_known::known) {
                return false;
            }
            replay_tokens = n_prompt > c.lcp_tokens.value ? n_prompt - c.lcp_tokens.value : 0;
            break;
        case common_cache_plan_provider::host_cache_entry:
        case common_cache_plan_provider::live_context_checkpoint:
            if (c.lcp_tokens.state    != llama_cache_acct_known::known ||
                c.payload_bytes.state != llama_cache_acct_known::known) {
                return false;
            }
            restore_bytes = c.payload_bytes.value;
            replay_tokens = n_prompt > c.lcp_tokens.value ? n_prompt - c.lcp_tokens.value : 0;
            has_restore   = true;
            break;
        default:
            return false;
    }

    double restore_us = 0.0;
    double workspace_us = 0.0;
    if (has_restore && !common_cache_plan_restore_us(
            calib, restore_bytes, restore_us, workspace_us)) {
        return false;
    }
    return cache_plan_fill_terms(
        c, calib, restore_bytes, restore_us,
        replay_tokens, (double) replay_tokens * calib.replay_us_per_token,
        workspace_us);
}

static common_cache_plan_planner_status cache_plan_estimate_impl(
        common_cache_plan_record & rec, const common_cache_plan_calib & calib) {
    // At the trust boundary the estimator validates its calibration:
    // exact profile match against the record, a reviewed (nonzero) version, and
    // finite/nonnegative coefficients. A false match here would fabricate economics.
    double restore_check = 0.0;
    double workspace_check = 0.0;
    if (calib.profile == nullptr || rec.calibration_profile != calib.profile ||
        calib.estimator_version == 0 ||
        !std::isfinite(calib.replay_us_per_token) || calib.replay_us_per_token < 0.0 ||
        !common_cache_plan_restore_us(
            calib, 0, restore_check, workspace_check)) {
        return common_cache_plan_planner_status::invalid_calibration;
    }

    // completeness over the DECLARED domain: overflow means the observed inventory lost
    // rows, and a dropped derived plan means the plan SET is incomplete — never an optimum
    // over a partial set. Truncation is fine: the domain is the shipped-visited
    // set by construction.
    for (const auto st : rec.inventory_states) {
        if (st == common_cache_plan_inventory_state::overflowed) {
            return common_cache_plan_planner_status::incomplete_evidence;
        }
    }
    if (rec.derived_plans_incomplete) {
        return common_cache_plan_planner_status::incomplete_evidence;
    }
    if (rec.n_prompt_tokens.state != llama_cache_acct_known::known) {
        return common_cache_plan_planner_status::incomplete_evidence;
    }
    if (!cache_plan_lru_stratum_complete(rec)) {
        return common_cache_plan_planner_status::incomplete_evidence;
    }
    const auto * lru_legacy = rec.selection == common_cache_plan_selection::lru
        ? cache_plan_target_live_row(rec, rec.id_slot) : nullptr;
    const uint64_t n_prompt = rec.n_prompt_tokens.value;

    // First pass: a visited candidate whose shipped phase established
    // neither validity nor invalidity (disposition unavailable — e.g. an LRU-only slot
    // whose reuse was never evaluated) makes the whole optimum unavailable. The
    // observer may not resolve it itself; honest refusal beats silent omission.
    for (uint32_t i = 0; i < rec.n_inventory; i++) {
        if (rec.inventory[i].disposition == common_cache_plan_disposition::unavailable) {
            return common_cache_plan_planner_status::incomplete_evidence;
        }
    }

    // pass 1: estimate every VALID non-chain row (component_only rows included — chains
    // compose from them); any valid row the calibration cannot cover leaves the whole
    // shadow result unavailable (an optimum that silently skipped a valid candidate would
    // be a fabricated verdict)
    for (uint32_t i = 0; i < rec.n_inventory; i++) {
        auto & c = rec.inventory[i];
        if (!c.viable() || c.is_chain()) {
            continue;
        }
        if (!cache_plan_estimate_row(c, n_prompt, calib)) {
            return common_cache_plan_planner_status::incomplete_evidence;
        }
    }

    // pass 2: chain rows compose from their components — restore/workspace add, replay is
    // the DEEPEST component's (the chain replays only past its furthest frontier)
    for (uint32_t i = 0; i < rec.n_inventory; i++) {
        auto & c = rec.inventory[i];
        if (!c.is_chain() || !c.viable()) {
            continue;
        }
        uint64_t restore_bytes = 0, replay_tokens = UINT64_MAX;
        double   restore_us = 0.0, workspace_us = 0.0, replay_us = 0.0;
        bool     ok = false;
        for (const int32_t comp : c.component_ids) {
            if (comp < 0 || uint32_t(comp) >= rec.n_inventory) {
                continue;
            }
            const auto & cc = rec.inventory[size_t(comp)];
            if (cc.predicted_total_us.state != llama_cache_acct_known::known) {
                // a chain over unestimated components has no honest total
                return common_cache_plan_planner_status::incomplete_evidence;
            }
            const auto & rest = cc.cost_terms[size_t(llama_cache_acct_cost_kind::restore)];
            const auto & repl = cc.cost_terms[size_t(llama_cache_acct_cost_kind::replay)];
            const auto & work = cc.cost_terms[size_t(llama_cache_acct_cost_kind::workspace)];
            restore_bytes += rest.raw.value;
            restore_us    += (double) rest.estimated_us.value;
            workspace_us  += (double) work.estimated_us.value;
            if (repl.raw.value < replay_tokens) {
                replay_tokens = repl.raw.value;
                replay_us     = (double) repl.estimated_us.value;
            }
            ok = true;
        }
        if (!ok) {
            return common_cache_plan_planner_status::incomplete_evidence;
        }
        if (!cache_plan_fill_terms(
                c, calib, restore_bytes, restore_us,
                replay_tokens, replay_us, workspace_us)) {
            return common_cache_plan_planner_status::incomplete_evidence;
        }
    }

    // pass 3: minimum + tie set + planner-owned stable choice
    uint64_t min_total  = UINT64_MAX;
    bool     any        = false;
    for (uint32_t i = 0; i < rec.n_inventory; i++) {
        const auto & c = rec.inventory[i];
        if (cache_plan_row_participates(rec, c, lru_legacy) &&
            c.predicted_total_us.state == llama_cache_acct_known::known) {
            min_total = std::min(min_total, c.predicted_total_us.value);
            any = true;
        }
    }
    if (!any) {
        return common_cache_plan_planner_status::incomplete_evidence;
    }
    const double floor_us = std::max((double) min_total * COMMON_CACHE_PLAN_TIE_REL_FLOOR,
                                     COMMON_CACHE_PLAN_TIE_ABS_FLOOR_US);

    rec.n_shadow_ties = 0;
    int32_t choice = -1;
    for (uint32_t i = 0; i < rec.n_inventory; i++) {
        const auto & c = rec.inventory[i];
        if (!cache_plan_row_participates(rec, c, lru_legacy) ||
            c.predicted_total_us.state != llama_cache_acct_known::known) {
            continue;
        }
        if ((double) c.predicted_total_us.value <= (double) min_total + floor_us) {
            rec.shadow_tie_set[rec.n_shadow_ties++] = (int32_t) i;
            // schema-v5 stable planner-owned key: target first, then provider/source/ordinal;
            // never the shipped choice. Target qualification prevents two physical plans
            // from inheriting insertion order as their only distinction.
            const auto & best = choice >= 0 ? rec.inventory[size_t(choice)] : c;
            if (choice < 0 ||
                std::make_tuple(c.target_slot_id, uint8_t(c.provider), c.source_id, (int32_t) i) <
                std::make_tuple(best.target_slot_id, uint8_t(best.provider), best.source_id, choice)) {
                choice = (int32_t) i;
            }
        }
    }
    rec.shadow_choice = choice;
    return choice >= 0 ? common_cache_plan_planner_status::ok
                       : common_cache_plan_planner_status::incomplete_evidence;
}

common_cache_plan_planner_status common_cache_plan_estimate_and_choose(
        common_cache_plan_record & rec, const common_cache_plan_calib & calib) {
    // all-or-nothing is THIS function's postcondition, not a call-site convention: a
    // refusal mid-pass leaves earlier rows carrying committed estimates, which would be
    // exactly the half-estimated evidence the planner forbids: clear before returning non-ok
    const auto status = cache_plan_estimate_impl(rec, calib);
    if (status != common_cache_plan_planner_status::ok) {
        rec.clear_planner_outputs();
    }
    return status;
}

common_cache_plan_planner_status common_cache_plan_run_planner(
        common_cache_plan_record & rec) noexcept {
    try {
        if (rec.calibration_profile.empty()) {
            rec.planner_status = common_cache_plan_planner_status::no_profile;
        } else if (const auto * calib =
                       common_cache_plan_calib_find(rec.calibration_profile)) {
            rec.planner_status = common_cache_plan_estimate_and_choose(rec, *calib);
        } else {
            rec.planner_status = common_cache_plan_planner_status::profile_unfitted;
        }
    } catch (...) {
        rec.clear_planner_outputs();
        rec.planner_status = common_cache_plan_planner_status::internal_fault;
    }
    return rec.planner_status;
}
