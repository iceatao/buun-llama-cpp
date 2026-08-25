#pragma once

#include "common-cache-plan.h"

#include <functional>
#include <string>
#include <vector>

struct common_params;

// One lowercase formatter for every SHA-256 identity emitted by cache-plan surfaces.
std::string common_cache_plan_sha256_hex_digest(
        const std::array<uint8_t, 32> & digest);

// common-cache-plan-estimate.h — B shadow-planner estimators (§7.5), schema v2.
//
// Versioned, policy-free cost estimation over the schema-v2 candidate inventory. SHADOW
// ONLY: fills per-candidate cost terms, predicted totals, and the shadow choice/tie set on
// a finalized-being record; never touches a shipped decision. Runs strictly inside the
// finalize planner boundary; throwing is tolerated there because the boundary clears planner
// outputs), but these functions avoid allocation and do not throw on their own.
//
// Coefficients come only from a fitted calibration profile. No profile means
// no estimates (typed-unavailable), never a default coefficient. Measured actuals are
// separate record fields and are never substituted.

struct common_cache_plan_calib {
    const char * profile;            // stable id: "{model class}/{hardware class}/b{batch}"
    uint32_t     estimator_version;  // bumped on ANY coefficient or formula change
    // fitted coefficients (RTX 3090 microbench sweep; see tools/server/bench/cache-plan-calibrate.py)
    double replay_us_per_token;      // forward replay cost per prompt token
    double restore_us_per_byte;      // pageable-host -> device state install, per byte
    double workspace_setup_us;       // fixed per-restore setup overhead
};

// Checked-in fitted table lookup (data reviewed like code). Returns nullptr when the
// profile has no fitted entry — the caller then leaves planner outputs unavailable.
const common_cache_plan_calib * common_cache_plan_calib_find(const std::string & profile);

// One versioned/trust-checked restore formula for planner estimates and host
// retention pricing. False means the fitted formula cannot be trusted.
bool common_cache_plan_restore_us(
        const common_cache_plan_calib & calib,
        uint64_t bytes,
        double & restore_us,
        double & workspace_us) noexcept;

// THE profile-composition rule (single producer-side spelling, tested): lowercases and
// squashes everything outside [a-z0-9/.] to '-'. The server composes records with this;
// fitted table entries must key on exactly this output — a drifted spelling fails silently
// (estimators legally refuse forever), so no consumer hand-rolls it. kv_desc names the KV
// cache regime ("k<ctk>-v<ctv>"): replay/restore costs differ across KV codecs (VBR,
// turbo encode paths), so an f16 run and a vbr run must never share coefficients.
std::string common_cache_plan_calib_profile(const std::string & model_stem,
                                            const std::string & hw_desc, int n_batch,
                                            const std::string & kv_desc);

// Effective VBR regime identity: requested CLI strings are not the
// regime — `auto` resolves to a selected family/policy/schedule, the aggregate floor
// resolves to a capacity, and documented developer env overrides (VBR_BUDGET_MIB,
// VBR_MIN_BITS, VBR_POLICY_LADDER) can move the controller's budget AFTER CLI resolution.
// Two runs whose requested strings match but whose resolved regimes differ must NOT share
// fitted coefficients. Every field below is an effective/resolved value; `overrides` carries
// canonical override tokens so an overridden run keys distinctly (empty when none are set).
// `unrepresented_override` means the effective state could not be established — the caller
// then has NO profile (refuse) rather than a possibly-aliased one.
enum class common_cache_plan_vbr_value_grammar : uint8_t {
    scalar,
    path,
    dir_or_file,
    inline_or_path,
};

// CLOSED CENSUS of every quoted VBR_* name in src/common/tools/ggml (D-pins r6).
// The CI scan is reader-agnostic: wrapper reads, direct reads, programmatic producers,
// diagnostics, and scripts all have to be classified here. `affects_cost` controls
// whether a set value joins calibration identity. The closed grammar tag determines how
// identity is formed; file forms use SHA-256 content, never paths, and unreadable content
// makes the regime unrepresentable.
// Names that only publish resolved telemetry or label logs are deliberately non-costing.
#define COMMON_CACHE_PLAN_VBR_ENV_LIST(X)                              \
    X("VBR_BUDGET",                    0, scalar)                      \
    X("VBR_BUDGET_MIB",                1, scalar)                      \
    X("VBR_CAPACITY_BITS",             0, scalar)                      \
    X("VBR_DEGRADE_ORDER",             1, path)                        \
    X("VBR_FORCE_GENERIC",             1, scalar)                      \
    X("VBR_FREEZE",                    1, scalar)                      \
    X("VBR_FREEZE_PRESERVE_EMPTY_TIERS", 1, scalar)                   \
    X("VBR_GROWTH_HEADROOM_MIB",       1, scalar)                      \
    X("VBR_LAYER_SCHEDULE",            1, inline_or_path)              \
    X("VBR_LAYER_SCHEDULE_FROM_POLICY",0, scalar)                      \
    X("VBR_LAYER_STRICT",              1, scalar)                      \
    X("VBR_MIN_BITS",                  1, scalar)                      \
    X("VBR_MODE",                      1, scalar)                      \
    X("VBR_POLICY_LADDER",             1, dir_or_file)                 \
    X("VBR_PROMOTE",                   1, scalar)                      \
    X("VBR_RETIER_PREFLIGHT",          0, scalar)                      \
    X("VBR_SCHEDULE_CTX",              1, scalar)                      \
    X("VBR_SELECTED_BPV",              0, scalar)                      \
    X("VBR_SELECTED_FAMILY",           0, scalar)                      \
    X("VBR_SELECTED_KLD",              0, scalar)                      \
    X("VBR_SELECTED_POLICY",           0, scalar)                      \
    X("VBR_SELECTED_SCHEDULE",         0, path)                        \
    X("VBR_STASH_CAPTURE_ONLY",        1, scalar)                      \
    X("VBR_STASH_ROWS",                1, scalar)                      \
    X("VBR_TRACE",                     0, scalar)                      \
    X("VBR_TRANSCODE_FIDELITY",        1, scalar)                      \
    X("VBR_TRANSCODE_NOTILE",          1, scalar)                      \
    X("VBR_TRANSCODE_TEST",            1, scalar)                      \
    X("VBR_TRANSCODE_TEST_N",          1, scalar)                      \
    X("VBR_VMM",                       1, scalar)                      \
    X("VBR_VRAM_BUDGET",               0, scalar)                      \
    X("VBR_VRAM_HEADROOM_MIB",         1, scalar)

struct common_cache_plan_vbr_regime {
    bool        armed = false;
    bool        side_k = false;      // K took the vbr alias
    bool        side_v = false;
    std::string budget_mode;         // resolved budget mode (dynamic/fixed tier)
    std::string family;              // vbr_selected_family
    std::string policy;              // vbr_selected_policy
    std::string schedule;            // compatibility key segment; schedule content is
                                     // represented once by the VBR_LAYER_SCHEDULE token
    double      capacity_bits = 0.0; // resolved aggregate floor (bits/value)
    double      selected_bpv  = 0.0; // measured BPV of the selected rung
    uint64_t    vram_budget_bytes = 0; // resolved explicit budget, 0 == auto
    float       reclaim_floor_bpv = 0.0f;
    float       reset_keep_frac   = 0.0f;
    std::string overrides;           // canonical env-override tokens, empty when none
    bool        unrepresented_override = false;
};

// THE KV-regime segment, pure and tested. Non-VBR runs key on the ggml type names only.
// Returns an EMPTY string when the regime is armed but unrepresentable — the profile is
// then empty and the planner refuses (no_profile), never a possibly-aliased match.
std::string common_cache_plan_calib_kv(const common_cache_plan_vbr_regime & vbr,
                                       const std::string & type_k, const std::string & type_v);

// Calibration-identity-only content hashing. For a cost-affecting census value, returns
// NAME=value for scalar values and NAME=sha256-<lower-hex> for file-valued values. The
// VBR_LAYER_SCHEDULE grammar accepts inline schedules, @file, or a bare file path;
// VBR_POLICY_LADDER accepts a JSON file or a directory containing policy_ladder.json.
// false means a required file could not be read and the calibration profile must refuse.
bool common_cache_plan_vbr_override_identity(const std::string & name,
                                             const std::string & value,
                                             common_cache_plan_vbr_value_grammar grammar,
                                             std::string & identity);

// Calibration-identity file digest. This does not change cache-plan record identity digests or
// sampled-prefix telemetry.
bool common_cache_plan_sha256_file_identity(const std::string & path, std::string & identity);

using common_cache_plan_getenv_fn = std::function<const char *(const char *)>;

// THE production regime assembly path, separated from server control flow so override
// ordering, refusal accumulation, and fail-closed profile composition are tested directly.
common_cache_plan_vbr_regime common_cache_plan_vbr_regime_from_params(
        const common_params & params,
        const common_cache_plan_getenv_fn & getenv_fn);

// THE placement-key (hardware class) construction, pure and tested: POSITIONAL device
// order (main_gpu / tensor_split index into it — reversed heterogeneous orders must
// produce distinct keys), effective ngl (0 or no devices → "cpu"), and for multi-GPU the
// split mode / main gpu / per-device tensor-split percents. `tensor_split` may be null
// (treated as all-zero / auto).
std::string common_cache_plan_calib_hw(const std::vector<std::string> & gpu_descs,
                                       int n_gpu_layers_eff, int split_mode, int main_gpu,
                                       const float * tensor_split);

// Tie resolution floor (planner-owned, deterministic): candidates whose predicted totals
// are within max(5% of the minimum, 100us) of the minimum form the tie set. The recorded
// shadow choice is the tie-set member with the smallest (provider, source_id, ordinal) key
// and never a function of the shipped choice; agreement is computed offline
// as shipped-in-tie-set.
constexpr double COMMON_CACHE_PLAN_TIE_REL_FLOOR = 0.05;
constexpr double COMMON_CACHE_PLAN_TIE_ABS_FLOOR_US = 100.0;

// Estimate every valid candidate (disposition accepted / valid_not_chosen_cost; chain rows
// composed from their components; component_only rows are estimated as components but
// EXCLUDED from the root optimum) and fill shadow_choice + shadow_tie_set. Returns a
// closed status; anything but `ok` leaves ALL planner outputs typed-unavailable (the
// all-or-nothing clear is this function's postcondition):
//   invalid_calibration — profile mismatch vs rec.calibration_profile, version 0, or
//       non-finite/negative coefficients (validated HERE, at the trust boundary);
//   incomplete_evidence — provider overflow, a dropped derived plan, an unresolved visited
//       candidate (disposition unavailable), a valid row missing the scalars estimation
//       needs, unknown n_prompt, or an empty participant set — never a partial optimum.
// Covered terms are restore/replay/workspace. Callers may prefill transfer or
// eviction; the same total/optimum consumes those optional terms.
common_cache_plan_planner_status common_cache_plan_estimate_and_choose(
        common_cache_plan_record & rec, const common_cache_plan_calib & calib);

// The single planner attempt boundary shared by pre-mutation planner staging and
// legacy finalize fallback. It owns profile lookup and exception isolation.
common_cache_plan_planner_status common_cache_plan_run_planner(
        common_cache_plan_record & rec) noexcept;
