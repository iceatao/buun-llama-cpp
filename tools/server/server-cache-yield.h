#pragma once

#include "server-cache-lease.h"
#include "server-retention-sidecar.h"
#include "../../src/llama-cache-budget.h"

#include <array>
#include <functional>
#include <vector>

struct common_cache_plan_yield_domain;

constexpr uint32_t SERVER_CACHE_YIELD_POLICY_VERSION = 1;
constexpr size_t SERVER_CACHE_YIELD_MAX_CANDIDATES =
    SERVER_RETENTION_MAX_CANDIDATES;
// Exhaustive compound enumeration is intentionally small and fail-closed.
// Larger zero-marginal alias groups remain shadow-unavailable until the
// accounting ledger can expose their allocation dependency graph directly.
constexpr size_t SERVER_RETENTION_SHADOW_MAX_COMPOUND = 12;
constexpr size_t SERVER_RETENTION_SHADOW_MAX_COMPOUND_EVALUATIONS = 8192;

enum class server_cache_yield_status : uint8_t {
    fits = 0,
    insufficient_yield,
    unsupported_required,
    unavailable,
    _count,
};

const char * server_cache_yield_status_name(
    server_cache_yield_status status) noexcept;

struct server_cache_yield_candidate {
    llama_cache_acct_artifact_id artifact_id;
    common_retention_artifact_record record;
    common_retention_lineage_record lineage;
    server_retention_candidate_availability availability =
        server_retention_candidate_availability::backing_missing_or_stale;
    server_cache_lease_evaluation lease;
    bool identity_known = false;
    // Exact prefix coverage retained by a different lineage if this
    // candidate's lineage is removed. The prefix-index owner computes this
    // from authenticated content identity; the pure projector only consumes
    // the bounded scalar. Zero preserves the pre-prefix-index behavior.
    uint64_t external_shared_coverage_tokens = 0;
    std::vector<llama_cache_acct_op_id> release_ops;
    bool has_unsupported_host_spill = false;
};

// DF1 counterfactual only. One result is projected per lineage, regardless of
// how many live/host aliases carry that lineage. `lost_work_units` is the
// unique prefix coverage that disappears after the proposed release, not a
// copied value for every alias.
struct server_retention_shadow_alternative {
    common_retention_pool pool = common_retention_pool::attention;
    uint64_t lineage_id = 0;
    std::vector<llama_cache_acct_artifact_id> artifact_ids;
    uint64_t lost_work_units = 0;
    common_retention_shadow_value value;
};

struct server_retention_shadow_projection {
    bool complete = false;
    std::vector<server_retention_shadow_alternative> alternatives;
};

// Allocation-free lowering used by live VBR fallback/reclaim. The caller owns a bounded
// candidate array and supplies exact physical cells that would disappear with each sequence.
// Preparation sorts that array once per pressure wave; every subsequent victim projection is
// a linear scan over the immutable grouping. Eligibility is already the intersection of the
// caller's hard strata (processing/incoming/pins/leases/speculative capability); ineligible
// entries still contribute retained prefix coverage.
struct server_live_retention_candidate {
    int32_t slot_id = -1;
    llama_cache_acct_artifact_id artifact_id;
    common_retention_stamp stamp;
    common_retention_lineage_record lineage;
    uint64_t external_shared_coverage_tokens = 0;
    uint64_t marginal_cells = 0;
    bool present = true;
    bool eligible = false;
};

struct server_live_retention_projection {
    bool complete = false;
    uint64_t candidate_count = 0;
    int32_t slot_id = -1;
    llama_cache_acct_artifact_id artifact_id;
    common_retention_pool pool = common_retention_pool::attention;
    uint64_t lineage_id = 0;
    uint64_t lost_work_tokens = 0;
    uint64_t marginal_cells = 0;
};

struct server_retention_singleton_quote {
    uint64_t lost_work_units = 0;
    common_retention_shadow_value value;
};

// Shared fixed-host/live-VBR policy kernel. Inventory owners separately derive the maximum and
// second surviving frontiers, but this is the only owner of singleton lost-work and value math.
bool server_retention_quote_singleton(
    const common_retention_stamp & stamp,
    const common_retention_lineage_record & lineage,
    uint64_t maximum_coverage,
    uint64_t second_coverage,
    uint32_t maximum_count,
    uint64_t external_shared_coverage,
    uint64_t marginal_resource,
    uint64_t competition_epoch,
    const common_retention_frequency_config & config,
    server_retention_singleton_quote & out) noexcept;

// Prepare immutable identity/group order once per reclaim wave. Between projections the caller
// may only refresh present/eligible/external coverage/marginal cells. A removed slot becomes
// present=false; it no longer contributes retained coverage.
bool server_live_retention_prepare(
    server_live_retention_candidate * candidates,
    size_t size,
    uint64_t competition_epoch) noexcept;

server_live_retention_projection server_live_retention_project_prepared(
    const server_live_retention_candidate * candidates,
    size_t size,
    uint64_t competition_epoch,
    const common_retention_frequency_config & config = {}) noexcept;

struct server_cache_yield_result {
    server_cache_yield_status status =
        server_cache_yield_status::unavailable;
    uint32_t yield_policy_version = SERVER_CACHE_YIELD_POLICY_VERSION;
    uint64_t accounting_serial = 0;
    std::array<std::vector<llama_cache_acct_artifact_id>,
               size_t(common_retention_pool::_count)> selected;
    std::vector<llama_cache_budget_plan_entry> plan;
    std::vector<llama_cache_acct_artifact_id> unsupported;
    // Winning D-S2 fit projection. Its domain rows are the canonical source of
    // resident/before/released/reserved/after at accounting_serial; D-S7 lowers
    // them to accounting-only wire types.
    llama_cache_budget_result projected_fit;
};

using server_cache_yield_preview_callback = std::function<bool(
    const std::vector<llama_cache_acct_op_id> &,
    uint64_t,
    llama_cache_acct_release_set_preview &)>;
using server_cache_yield_fits_callback = std::function<llama_cache_budget_result(
    const llama_cache_budget_plan &)>;
using server_cache_yield_candidate_resolver = std::function<void(
    const server_retention_candidate &,
    server_cache_yield_candidate &,
    server_cache_lease_identity &,
    bool & identity_known)>;

// Shared lowering doors for D-S6 and D-A shadow projections. Release bytes
// always come from the exact batch preview; coverage is never a byte estimate.
bool server_cache_yield_release_plan(
    const llama_cache_acct_release_set_preview & release,
    uint64_t accounting_serial,
    llama_cache_budget_plan & plan) noexcept;

bool server_cache_yield_lower_domain(
    const llama_cache_budget_row & row,
    common_cache_plan_yield_domain & out) noexcept;

// Impure server-side half: joins the catalog value with live backing/identity
// through one injected resolver, then performs exactly one lease evaluation.
bool server_cache_yield_assemble(
    const std::vector<server_retention_candidate> & catalog,
    server_cache_lease_table & leases,
    const server_cache_yield_candidate_resolver & resolver,
    std::vector<server_cache_yield_candidate> & out) noexcept;

// Pure yield-ladder planner. Candidates already carry one evaluated lease result
// and validated backing operation ids; all impure ledger/budget access is injected.
server_cache_yield_result server_cache_yield_plan(
    const std::vector<server_cache_yield_candidate> & candidates,
    uint64_t accounting_serial,
    const server_cache_yield_preview_callback & preview,
    const server_cache_yield_fits_callback & fits,
    uint32_t policy_version = SERVER_CACHE_YIELD_POLICY_VERSION) noexcept;

// Pure shadow selector. It performs no mutation and cannot authorize release.
// Marginal resource comes only from the serial-bound accounting preview.
server_retention_shadow_projection server_retention_shadow_project(
    const std::vector<server_cache_yield_candidate> & candidates,
    uint64_t competition_epoch,
    const llama_cache_acct_resource_domain & pressured_domain,
    uint64_t accounting_serial,
    const server_cache_yield_preview_callback & preview,
    const common_retention_frequency_config & config = {}) noexcept;
