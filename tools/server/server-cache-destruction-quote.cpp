#include "server-cache-destruction-quote.h"

#include "ggml.h"
#include "../../common/common-cache-plan-estimate.h"

#include "../../src/llama-sha256.h"

#include <algorithm>
#include <map>
#include <nlohmann/json.hpp>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace {

bool artifact_ptr_less(
        const server_cache_destruction_artifact * a,
        const server_cache_destruction_artifact * b) noexcept {
    return a->candidate.artifact_id < b->candidate.artifact_id;
}

common_cache_plan_destruction_manifest_digest manifest_digest(
        const std::vector<const server_cache_destruction_artifact *> & manifest) {
    llama_sha256_writer hash;
    static constexpr char tag[] = "cache-destruction-manifest-v1";
    hash.string(tag, sizeof(tag) - 1);
    hash.u64(manifest.size());
    for (const auto * artifact : manifest) {
        hash.u64(artifact->candidate.artifact_id.v);
        hash.u32(uint32_t(artifact->kind));
        hash.u32(uint32_t(artifact->pool));
    }
    return common_cache_plan_destruction_manifest_digest::from_sha256(
        hash.finish());
}

std::string digest_key(
        const common_cache_plan_destruction_manifest_digest & digest) {
    return std::string(reinterpret_cast<const char *>(digest.bytes().data()),
                       digest.bytes().size());
}

common_cache_plan_destruction_reason artifact_refusal_base(
        const server_cache_destruction_artifact & artifact,
        common_cache_plan_destruction_lease_verdict & lease) noexcept {
    lease = common_cache_plan_destruction_lease_verdict::unleased;
    if (!artifact.candidate.identity_known) {
        return common_cache_plan_destruction_reason::identity_unavailable;
    }
    if (artifact.candidate.availability !=
            server_retention_candidate_availability::available ||
        artifact.candidate.record.stamp.state !=
            common_retention_score_state::known) {
        return common_cache_plan_destruction_reason::manifest_incomplete;
    }
    if (artifact.mandatory_anchor ||
        artifact.candidate.record.stamp.mandatory_anchor) {
        lease = common_cache_plan_destruction_lease_verdict::mandatory_recovery;
        return common_cache_plan_destruction_reason::mandatory_anchor;
    }
    if (artifact.candidate.lease.state != server_cache_lease_eval_state::known ||
        artifact.candidate.lease.cls == server_cache_lease_class::_count ||
        artifact.candidate.lease.eligibility ==
            server_cache_lease_eligibility::_count) {
        lease = common_cache_plan_destruction_lease_verdict::unavailable;
        return common_cache_plan_destruction_reason::lease_unavailable;
    }
    if (server_cache_lease_is_hard(artifact.candidate.lease)) {
        lease = common_cache_plan_destruction_lease_verdict::hard_leased;
        return common_cache_plan_destruction_reason::hard_lease_blocked;
    }
    if (artifact.candidate.lease.cls == server_cache_lease_class::soft) {
        lease = common_cache_plan_destruction_lease_verdict::soft_leased;
    }
    if ((!artifact.fixed_pool_logical_ownership &&
         artifact.candidate.release_ops.empty()) ||
        std::any_of(
            artifact.candidate.release_ops.begin(),
            artifact.candidate.release_ops.end(),
            [](auto op) { return !op; })) {
        return common_cache_plan_destruction_reason::release_evidence_unavailable;
    }
    return common_cache_plan_destruction_reason::none;
}

common_cache_plan_destruction_reason server_cache_destruction_artifact_refusal(
        const server_cache_destruction_artifact & artifact,
        bool mutation_boundary,
        common_cache_plan_destruction_lease_verdict & lease) noexcept {
    const auto base = artifact_refusal_base(artifact, lease);
    if (base != common_cache_plan_destruction_reason::none) {
        return base;
    }
    // Quote assembly constructs these fields from the same catalog record.
    // Mutation-boundary callers are externally supplied and must re-prove the
    // structural join in addition to the shared eligibility ladder.
    if (mutation_boundary &&
        (artifact.kind != artifact.candidate.record.kind ||
         artifact.pool != artifact.candidate.record.stamp.pool ||
         artifact.pool >= common_retention_pool::_count)) {
        return common_cache_plan_destruction_reason::manifest_incomplete;
    }
    return common_cache_plan_destruction_reason::none;
}

void refuse(common_cache_plan_destruction_receipt & out,
            common_cache_plan_destruction_reason reason) noexcept {
    out.state = common_cache_plan_destruction_state::refused;
    out.reason = reason;
    out.selected_attention.clear();
    out.selected_recurrent.clear();
}

void bind_recovery(
        common_cache_plan_destruction_receipt & quote,
        common_cache_plan_destruction_effect_set effects,
        common_cache_plan_recovery_citation citation) noexcept {
    const auto displacement =
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::cross_target_displacement) |
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::destructive_similarity_retarget) |
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::same_target_cold_replacement);
    const bool prospective_cross_target =
        citation == common_cache_plan_recovery_citation::prospective &&
        effects != 0 && (effects & ~displacement) == 0;
    if (citation == common_cache_plan_recovery_citation::resolved ||
        prospective_cross_target) {
        quote.state = common_cache_plan_destruction_state::quoted;
        quote.reason = common_cache_plan_destruction_reason::none;
        quote.displaced_fate = common_cache_plan_displaced_fate::retained_host;
        quote.recovery_citation = citation;
    } else {
        quote.displaced_fate = common_cache_plan_displaced_fate::unavailable;
        quote.recovery_citation = common_cache_plan_recovery_citation::unavailable;
        refuse(quote, common_cache_plan_destruction_reason::recovery_unavailable);
    }
}

bool domain_row_equal(const common_cache_plan_yield_domain & a,
                      const llama_cache_budget_row & b) noexcept {
    const auto value_equal = [](const llama_cache_acct_value & lhs,
                                const llama_cache_acct_value & rhs) {
        return lhs.state == rhs.state && lhs.value == rhs.value;
    };
    return b.resource.kind ==
               llama_cache_budget_resource_kind::accounting_domain &&
           a.domain == b.resource.domain &&
           value_equal(a.current_resident_bytes, b.current_resident) &&
           value_equal(a.fit_before_bytes, b.before) &&
           value_equal(a.projected_release_bytes, b.released) &&
           value_equal(a.projected_reserve_bytes, b.reserved) &&
           value_equal(a.projected_after_bytes, b.after);
}

bool value_equal(
        const llama_cache_acct_value & a,
        const llama_cache_acct_value & b) noexcept {
    return a.state == b.state && a.value == b.value;
}

bool effect_rows_equal(
        const common_cache_plan_yield_domain & quoted,
        const common_cache_plan_yield_domain & current,
        bool full_row) noexcept {
    if (quoted.domain != current.domain ||
        !value_equal(
            quoted.projected_release_bytes,
            current.projected_release_bytes)) {
        return false;
    }
    return !full_row ||
           (value_equal(
                quoted.current_resident_bytes,
                current.current_resident_bytes) &&
            value_equal(quoted.fit_before_bytes, current.fit_before_bytes) &&
            value_equal(
                quoted.projected_reserve_bytes,
                current.projected_reserve_bytes) &&
            value_equal(
                quoted.projected_after_bytes,
                current.projected_after_bytes));
}

bool effect_equivalent(
        const common_cache_plan_destruction_receipt & quote,
        const common_cache_plan_destruction_effect_digest & current_effect,
        const std::vector<common_cache_plan_yield_domain> & quoted_domains,
        const std::vector<common_cache_plan_yield_domain> & current_domains,
        bool full_rows) noexcept {
    if (!quote.union_effect_digest.valid() || !current_effect.valid() ||
        quote.union_effect_digest != current_effect ||
        quoted_domains.size() != current_domains.size()) {
        return false;
    }
    return std::all_of(
        quoted_domains.begin(), quoted_domains.end(),
        [&](const auto & row) {
            return std::any_of(
                current_domains.begin(), current_domains.end(),
                [&](const auto & current) {
                    return effect_rows_equal(row, current, full_rows);
                });
        });
}

} // namespace

nlohmann::ordered_json server_cache_destruction_receipt_json(
        const common_cache_plan_destruction_receipt & receipt,
        uint64_t projected_bytes,
        const char * action_class) {
    using json = nlohmann::ordered_json;
    json effects = json::array();
    for (uint8_t raw =
             uint8_t(common_cache_plan_destruction_effect::none) + 1;
         raw < uint8_t(common_cache_plan_destruction_effect::_count);
         ++raw) {
        const auto effect = common_cache_plan_destruction_effect(raw);
        if (common_cache_plan_destruction_effect_has(
                receipt.effects, effect)) {
            effects.push_back(common_cache_plan_destruction_effect_name(effect));
        }
    }
    json victims = json::array();
    for (const auto artifact : receipt.selected_attention) {
        victims.push_back(artifact.v);
    }
    for (const auto artifact : receipt.selected_recurrent) {
        victims.push_back(artifact.v);
    }
    json recovery_source = common_cache_acct_known_name(
        llama_cache_acct_known::unavailable);
    if (receipt.recovery_source_artifact_id.v != 0 &&
        receipt.recovery_source_manifest_digest.valid()) {
        recovery_source = json {
            { "artifact_id", receipt.recovery_source_artifact_id.v },
            { "manifest_digest", common_cache_plan_sha256_hex_digest(
                  receipt.recovery_source_manifest_digest.bytes()) },
        };
    }
    json out = {
        { "state", common_cache_plan_destruction_state_name(receipt.state) },
        { "reason", common_cache_plan_destruction_reason_name(receipt.reason) },
        { "payload_kind", receipt.payload_kind ==
                common_cache_plan_payload_kind::unavailable
            ? json(nullptr)
            : json(common_cache_plan_payload_kind_name(receipt.payload_kind)) },
    };
    if (action_class) {
        out["action_class"] = action_class;
    }
    out["effects"] = std::move(effects);
    out["victim_ids"] = std::move(victims);
    out["recovery_source"] = std::move(recovery_source);
    out["projected_bytes"] = projected_bytes;
    return out;
}

common_cache_plan_destruction_effect_digest
server_cache_destruction_union_effect_digest(
        const std::vector<llama_cache_acct_op_id> & ops,
        const llama_cache_acct_release_set_preview & release) {
    llama_sha256_writer hash;
    static constexpr char tag[] = "cache-destruction-union-effect-v1";
    hash.string(tag, sizeof(tag) - 1);
    hash.u64(ops.size());
    for (const auto op : ops) {
        hash.u64(op.v);
    }
    hash.u64(release.rows.size());
    for (const auto & row : release.rows) {
        hash.u32(uint32_t(row.domain.residency));
        hash.u32(uint32_t(row.domain.kind));
        hash.u32(row.domain.topology.v);
        hash.u32(row.domain.device_ordinal.v);
        hash.u64(row.logical_payload);
        hash.u64(row.resident_allocated);
    }
    return common_cache_plan_destruction_effect_digest::from_sha256(
        hash.finish());
}

common_cache_plan_destruction_recovery_digest
server_cache_destruction_recovery_source_digest(
        llama_cache_acct_artifact_id artifact,
        const std::vector<llama_cache_acct_op_id> & ops) {
    if (artifact.v == 0 || ops.empty() ||
        std::any_of(ops.begin(), ops.end(), [](const auto op) {
            return !op;
        })) {
        return {};
    }
    auto canonical = ops;
    std::sort(canonical.begin(), canonical.end());
    canonical.erase(
        std::unique(canonical.begin(), canonical.end()), canonical.end());
    llama_sha256_writer hash;
    static constexpr char tag[] = "cache-destruction-recovery-source-v1";
    hash.string(tag, sizeof(tag) - 1);
    hash.u64(artifact.v);
    hash.u64(canonical.size());
    for (const auto op : canonical) {
        hash.u64(op.v);
    }
    return common_cache_plan_destruction_recovery_digest::from_sha256(
        hash.finish());
}

void server_cache_destruction_certify_receipt(
        common_cache_plan_destruction_receipt & receipt,
        common_cache_plan_displaced_fate fate,
        llama_cache_acct_artifact_id recovery_artifact,
        const std::vector<llama_cache_acct_op_id> & recovery_ops) noexcept {
    receipt.state = common_cache_plan_destruction_state::certified;
    receipt.reason = common_cache_plan_destruction_reason::none;
    receipt.displaced_fate = fate;
    receipt.recovery_citation = common_cache_plan_recovery_citation::resolved;
    receipt.recovery_source_artifact_id = recovery_artifact;
    receipt.recovery_source_manifest_digest =
        server_cache_destruction_recovery_source_digest(
            recovery_artifact, recovery_ops);
}

server_cache_recovery_pin::~server_cache_recovery_pin() {
    reset();
}

server_cache_recovery_pin::server_cache_recovery_pin(
        server_cache_recovery_pin && other) noexcept
    : context_(other.context_),
      release_(other.release_),
      artifacts_(std::move(other.artifacts_)),
      ops_(std::move(other.ops_)) {
    other.context_ = nullptr;
    other.release_ = nullptr;
}

server_cache_recovery_pin & server_cache_recovery_pin::operator=(
        server_cache_recovery_pin && other) noexcept {
    if (this != &other) {
        reset();
        context_ = other.context_;
        release_ = other.release_;
        artifacts_ = std::move(other.artifacts_);
        ops_ = std::move(other.ops_);
        other.context_ = nullptr;
        other.release_ = nullptr;
    }
    return *this;
}

server_cache_recovery_pin server_cache_recovery_pin::acquire(
        void * context,
        release_fn release,
        std::vector<llama_cache_acct_artifact_id> artifacts,
        std::vector<llama_cache_acct_op_id> ops) noexcept {
    server_cache_recovery_pin out;
    try {
        if (!context || !release || (artifacts.empty() && ops.empty())) {
            return out;
        }
        std::sort(artifacts.begin(), artifacts.end());
        artifacts.erase(std::unique(artifacts.begin(), artifacts.end()),
                        artifacts.end());
        std::sort(ops.begin(), ops.end());
        ops.erase(std::unique(ops.begin(), ops.end()), ops.end());
        if ((!artifacts.empty() && artifacts.front().v == 0) ||
            (!ops.empty() && !ops.front())) {
            return out;
        }
        out.context_ = context;
        out.release_ = release;
        out.artifacts_ = std::move(artifacts);
        out.ops_ = std::move(ops);
    } catch (...) {
        out.reset();
    }
    return out;
}

bool server_cache_recovery_pin::disjoint(
        const std::vector<llama_cache_acct_artifact_id> & artifacts,
        const std::vector<llama_cache_acct_op_id> & ops) const noexcept {
    if (!valid()) {
        return false;
    }
    for (const auto artifact : artifacts) {
        if (std::binary_search(
                artifacts_.begin(), artifacts_.end(), artifact)) {
            return false;
        }
    }
    for (const auto op : ops) {
        if (std::binary_search(
                ops_.begin(), ops_.end(), op)) {
            return false;
        }
    }
    return true;
}

bool server_cache_recovery_pin::binds_exact(
        llama_cache_acct_artifact_id artifact,
        const std::vector<llama_cache_acct_op_id> & ops) const noexcept {
    if (!valid() || artifact.v == 0 || artifacts_.size() != 1 ||
        artifacts_.front() != artifact) {
        return false;
    }
    try {
        auto expected = ops;
        std::sort(expected.begin(), expected.end());
        expected.erase(std::unique(expected.begin(), expected.end()),
                       expected.end());
        return !expected.empty() &&
               std::none_of(expected.begin(), expected.end(),
                            [](auto op) { return !op; }) &&
               expected == ops_;
    } catch (...) {
        return false;
    }
}

void server_cache_recovery_pin::reset() noexcept {
    if (context_ && release_) {
        release_(context_);
    }
    context_ = nullptr;
    release_ = nullptr;
    artifacts_.clear();
    ops_.clear();
}

common_cache_plan_destruction_reason
server_cache_prepared_release_capability::commit(
        server_cache_recovery_pin & retained_pin) noexcept {
    const bool same_thread =
        scheduler_owner_ == std::this_thread::get_id();
    GGML_ASSERT(same_thread);
    if (!same_thread || !ready() || retained_pin.valid()) {
        return common_cache_plan_destruction_reason::internal_fault;
    }
    switch (release_.commit()) {
        case llama_cache_conditional_release_status::released:
            retained_pin = std::move(pin_);
            return common_cache_plan_destruction_reason::none;
        case llama_cache_conditional_release_status::serial_conflict:
            return common_cache_plan_destruction_reason::effect_drift;
        case llama_cache_conditional_release_status::ledger_fault:
        case llama_cache_conditional_release_status::_count:
            return common_cache_plan_destruction_reason::accounting_unavailable;
    }
    return common_cache_plan_destruction_reason::internal_fault;
}

server_cache_prepare_release_result server_cache_prepare_release_set(
        const common_cache_plan_destruction_quote & quote,
        const std::vector<server_cache_destruction_artifact> & current_artifacts,
        llama_cache_acct_ledger & ledger,
        uint64_t fresh_accounting_serial,
        const server_cache_destruction_projection_callback & project,
        server_cache_recovery_pin && recovery_pin) noexcept {
    server_cache_prepare_release_result out;
    try {
        if (quote.receipt.state !=
                common_cache_plan_destruction_state::quoted ||
            // Production receipts are minted from 1. Sequence zero therefore
            // means the receipt was never minted, irrespective of which
            // read-only caller produced it, and is invalid at this capability
            // boundary.
            quote.receipt.admission_sequence == 0 ||
            !quote.receipt.union_effect_digest.valid() ||
            !project) {
            return out;
        }
        std::vector<llama_cache_acct_artifact_id> victims =
            quote.receipt.selected_attention;
        victims.insert(victims.end(),
                       quote.receipt.selected_recurrent.begin(),
                       quote.receipt.selected_recurrent.end());
        std::vector<const server_cache_destruction_artifact *> manifest;
        std::vector<llama_cache_acct_op_id> current_ops;
        manifest.reserve(victims.size());
        std::unordered_map<uint64_t,
                           const server_cache_destruction_artifact *> current_by_id;
        current_by_id.reserve(current_artifacts.size());
        for (const auto & artifact : current_artifacts) {
            if (artifact.candidate.artifact_id.v == 0 ||
                !current_by_id.emplace(
                    artifact.candidate.artifact_id.v, &artifact).second) {
                out.reason =
                    common_cache_plan_destruction_reason::manifest_incomplete;
                return out;
            }
        }
        for (const auto victim : victims) {
            const auto found = current_by_id.find(victim.v);
            if (found == current_by_id.end()) {
                out.reason = common_cache_plan_destruction_reason::manifest_incomplete;
                return out;
            }
            const auto & current = *found->second;
            common_cache_plan_destruction_lease_verdict lease;
            out.reason = server_cache_destruction_artifact_refusal(
                current, true, lease);
            if (out.reason != common_cache_plan_destruction_reason::none) {
                return out;
            }
            manifest.push_back(&current);
            current_ops.insert(current_ops.end(),
                               current.candidate.release_ops.begin(),
                               current.candidate.release_ops.end());
        }
        std::sort(manifest.begin(), manifest.end(), artifact_ptr_less);
        std::sort(current_ops.begin(), current_ops.end());
        current_ops.erase(
            std::unique(current_ops.begin(), current_ops.end()),
            current_ops.end());
        if (manifest_digest(manifest) != quote.receipt.manifest_digest ||
            !quote.receipt.manifest_digest.valid()) {
            out.status = server_cache_prepare_release_status::effect_drift;
            out.reason = common_cache_plan_destruction_reason::effect_drift;
            return out;
        }
        if (!recovery_pin.valid() ||
            !recovery_pin.disjoint(victims, current_ops)) {
            out.status = server_cache_prepare_release_status::recovery_unavailable;
            out.reason = common_cache_plan_destruction_reason::recovery_unavailable;
            return out;
        }

        auto prepared = llama_cache_prepare_release_set(
            ledger, current_ops, fresh_accounting_serial);
        switch (prepared.status()) {
            case llama_cache_prepare_release_status::prepared:
                break;
            case llama_cache_prepare_release_status::serial_conflict:
                out.status = server_cache_prepare_release_status::serial_conflict;
                out.reason = common_cache_plan_destruction_reason::accounting_unavailable;
                return out;
            case llama_cache_prepare_release_status::invalid_argument:
                out.status = server_cache_prepare_release_status::invalid_quote;
                out.reason = common_cache_plan_destruction_reason::manifest_incomplete;
                return out;
            case llama_cache_prepare_release_status::ledger_fault:
                out.status = server_cache_prepare_release_status::accounting_unavailable;
                out.reason = common_cache_plan_destruction_reason::accounting_unavailable;
                return out;
            case llama_cache_prepare_release_status::internal_fault:
            case llama_cache_prepare_release_status::_count:
                out.status = server_cache_prepare_release_status::internal_fault;
                out.reason = common_cache_plan_destruction_reason::internal_fault;
                return out;
        }
        std::vector<common_cache_plan_yield_domain> current_domains;
        if (!prepared.ops().empty() &&
            !project(prepared.preview(), current_domains)) {
            out.status = server_cache_prepare_release_status::accounting_unavailable;
            out.reason = common_cache_plan_destruction_reason::capacity_refused;
            return out;
        }
        const auto effect = server_cache_destruction_union_effect_digest(
            prepared.ops(), prepared.preview());
        out.reason = server_cache_destruction_effect_recheck(
            quote.receipt, effect, quote.projected_domains, current_domains);
        if (out.reason != common_cache_plan_destruction_reason::none) {
            out.status = server_cache_prepare_release_status::effect_drift;
            return out;
        }

        out.capability.release_ = std::move(prepared);
        out.capability.pin_ = std::move(recovery_pin);
        out.capability.scheduler_owner_ = std::this_thread::get_id();
        out.status = server_cache_prepare_release_status::prepared;
        out.reason = common_cache_plan_destruction_reason::none;
        return out;
    } catch (...) {
        out = {};
        out.status = server_cache_prepare_release_status::internal_fault;
        out.reason = common_cache_plan_destruction_reason::internal_fault;
        return out;
    }
}

bool server_cache_destruction_has_effect(
        const common_cache_plan_record & rec,
        int32_t legacy_candidate,
        common_cache_plan_destruction_effect_set permitted_effects) noexcept {
    for (uint32_t i = 0; i < rec.n_inventory; ++i) {
        const auto & candidate = rec.inventory[i];
        if (candidate.viable() && !candidate.component_only &&
            (!candidate.is_chain() || candidate.component_ids[0] >= 0) &&
            common_cache_plan_origin_in_domain(
                candidate.origin_tier, rec.selection) &&
            server_cache_destruction_effects_for(
                rec, int32_t(i), legacy_candidate,
                permitted_effects) != 0) {
            return true;
        }
    }
    return false;
}

bool server_cache_destruction_quote_all(
        common_cache_plan_record & rec,
        int32_t legacy_plan_candidate,
        const std::vector<server_cache_destruction_artifact> & artifacts,
        uint64_t accounting_serial,
        const server_cache_destruction_preview_callback & preview,
        const server_cache_destruction_projection_callback & project,
        const server_cache_destruction_quote_options & options,
        common_cache_plan_destruction_counters & counters) noexcept {
    rec.destruction_quotes.clear();
    rec.destruction = {};
    rec.destruction_legacy_plan_candidate = legacy_plan_candidate;
    rec.destruction.admission_sequence = options.admission_sequence;
    const auto fail_whole_pass = [&](
            common_cache_plan_destruction_state state,
            common_cache_plan_destruction_reason reason) {
        rec.destruction.state = state;
        rec.destruction.reason = reason;
        counters.observe(rec.selection, rec.destruction);
        return false;
    };
    if (!options.lifecycle_available) {
        return fail_whole_pass(
            common_cache_plan_destruction_state::refused,
            common_cache_plan_destruction_reason::lifecycle_disabled);
    }
    if (legacy_plan_candidate < 0 ||
        uint32_t(legacy_plan_candidate) >= rec.n_inventory ||
        rec.inventory_saturated() ||
        artifacts.size() > SERVER_CACHE_YIELD_MAX_CANDIDATES) {
        return fail_whole_pass(
            common_cache_plan_destruction_state::refused,
            common_cache_plan_destruction_reason::manifest_incomplete);
    }
    if ((options.admission_sequence == 0 && !options.preview_unminted) ||
        !preview || !project) {
        return fail_whole_pass(
            common_cache_plan_destruction_state::failed,
            common_cache_plan_destruction_reason::internal_fault);
    }
    try {
        std::map<int32_t, std::vector<const server_cache_destruction_artifact *>> by_slot;
        std::map<int32_t, std::vector<const server_cache_destruction_artifact *>> by_host;
        std::array<std::vector<const server_cache_destruction_artifact *>,
                   size_t(common_retention_pool::_count)> pools;
        for (const auto & artifact : artifacts) {
            if (artifact.pool >= common_retention_pool::_count) {
                return fail_whole_pass(
                    common_cache_plan_destruction_state::refused,
                    common_cache_plan_destruction_reason::manifest_incomplete);
            }
            pools[size_t(artifact.pool)].push_back(&artifact);
        }
        for (auto & pool : pools) {
            std::sort(pool.begin(), pool.end(), artifact_ptr_less);
        }
        for (const auto & pool : pools) {
            for (const auto * artifact : pool) {
                if (artifact->kind == common_retention_artifact_kind::host_entry &&
                    artifact->host_source_id >= 0) {
                    by_host[artifact->host_source_id].push_back(artifact);
                } else if (artifact->owner_slot >= 0) {
                    by_slot[artifact->owner_slot].push_back(artifact);
                }
            }
        }
        std::unordered_map<std::string, common_cache_plan_destruction_quote> memo;
        memo.reserve(rec.n_inventory);
        rec.destruction_quotes.reserve(rec.n_inventory);
        for (uint32_t i = 0; i < rec.n_inventory; ++i) {
            const auto & candidate = rec.inventory[i];
            if (!candidate.viable() || candidate.component_only ||
                !common_cache_plan_origin_in_domain(
                    candidate.origin_tier, rec.selection) ||
                (candidate.is_chain() && candidate.component_ids[0] < 0)) {
                continue;
            }
            const auto effects = server_cache_destruction_effects_for(
                rec, int32_t(i), legacy_plan_candidate,
                options.permitted_effects);
            if (effects == 0) {
                continue;
            }

            std::vector<const server_cache_destruction_artifact *> manifest;
            std::unordered_set<uint64_t> manifest_ids;
            const auto add_manifest = [&](const auto & rows) {
                for (const auto * artifact : rows) {
                    if (manifest_ids.insert(
                            artifact->candidate.artifact_id.v).second) {
                        manifest.push_back(artifact);
                    }
                }
            };
            if ((effects & SERVER_CACHE_LIVE_DISPLACEMENT_EFFECTS) != 0) {
                const auto it = by_slot.find(candidate.target_slot_id);
                if (it != by_slot.end()) {
                    add_manifest(it->second);
                }
            }
            if (common_cache_plan_destruction_effect_has(
                    effects,
                    common_cache_plan_destruction_effect::
                        different_host_source_consumption)) {
                const int32_t source =
                    server_cache_plan_host_source(rec, int32_t(i));
                const auto it = by_host.find(source);
                if (source < 0 || it == by_host.end() || it->second.empty()) {
                    // Consumption is an independent destructive effect. A
                    // displacement manifest must never mask missing host
                    // coverage and become a silently partial exact union.
                    manifest.clear();
                    manifest_ids.clear();
                } else {
                    add_manifest(it->second);
                }
            }
            std::sort(manifest.begin(), manifest.end(), artifact_ptr_less);

            common_cache_plan_destruction_quote staged;
            auto & quote = staged.receipt;
            quote.plan_candidate = int32_t(i);
            quote.payload_kind = candidate.payload_kind;
            quote.admission_sequence = options.admission_sequence;
            quote.effects = effects;
            quote.quote_accounting_serial = accounting_serial;
            if (manifest.empty()) {
                refuse(quote, common_cache_plan_destruction_reason::manifest_incomplete);
                counters.observe(rec.selection, quote);
                rec.destruction_quotes.push_back(std::move(staged));
                continue;
            }
            quote.manifest_digest = manifest_digest(manifest);
            const auto key = digest_key(quote.manifest_digest);
            const auto cached = memo.find(key);
            if (cached != memo.end()) {
                const int32_t plan_candidate = quote.plan_candidate;
                const auto candidate_effects = quote.effects;
                const auto candidate_payload_kind = quote.payload_kind;
                staged = cached->second;
                auto & cached_receipt = staged.receipt;
                cached_receipt.plan_candidate = plan_candidate;
                cached_receipt.effects = candidate_effects;
                cached_receipt.payload_kind = candidate_payload_kind;
                if (cached_receipt.reason ==
                        common_cache_plan_destruction_reason::none ||
                    cached_receipt.reason ==
                        common_cache_plan_destruction_reason::recovery_unavailable) {
                    bind_recovery(
                        cached_receipt, candidate_effects,
                        options.recovery_citation);
                }
                counters.quote_memo_hits++;
                counters.observe(rec.selection, cached_receipt);
                rec.destruction_quotes.push_back(std::move(staged));
                continue;
            }
            counters.quote_memo_misses++;

            std::vector<llama_cache_acct_op_id> ops;
            std::unordered_set<uint64_t> op_ids;
            bool unavailable = false;
            quote.lease_verdict = common_cache_plan_destruction_lease_verdict::unleased;
            for (const auto * artifact : manifest) {
                common_cache_plan_destruction_lease_verdict verdict;
                const auto reason = server_cache_destruction_artifact_refusal(
                    *artifact, false, verdict);
                if (reason != common_cache_plan_destruction_reason::none) {
                    quote.lease_verdict = verdict;
                    refuse(quote, reason);
                    unavailable = true;
                    break;
                }
                if (verdict == common_cache_plan_destruction_lease_verdict::soft_leased) {
                    quote.lease_verdict = verdict;
                }
                (artifact->pool == common_retention_pool::attention
                    ? quote.selected_attention : quote.selected_recurrent)
                    .push_back(artifact->candidate.artifact_id);
                for (const auto op : artifact->candidate.release_ops) {
                    if (!op_ids.insert(op.v).second) {
                        continue;
                    }
                    ops.push_back(op);
                }
            }
            const bool known_zero_live_union = !unavailable && ops.empty() &&
                !manifest.empty() && std::all_of(
                    manifest.begin(), manifest.end(), [](const auto * artifact) {
                        return artifact->fixed_pool_logical_ownership &&
                               artifact->candidate.release_ops.empty();
                    });
            if (!unavailable && ops.empty() && !known_zero_live_union) {
                refuse(quote, common_cache_plan_destruction_reason::release_evidence_unavailable);
                unavailable = true;
            }
            std::sort(ops.begin(), ops.end());
            llama_cache_acct_release_set_preview released;
            if (!unavailable && !preview(ops, accounting_serial, released)) {
                refuse(quote, common_cache_plan_destruction_reason::accounting_unavailable);
                unavailable = true;
            }
            if (!unavailable) {
                quote.union_effect_digest =
                    server_cache_destruction_union_effect_digest(ops, released);
                if (!ops.empty() &&
                    !project(released, staged.projected_domains)) {
                    refuse(quote, common_cache_plan_destruction_reason::capacity_refused);
                    unavailable = true;
                }
            }
            if (!unavailable) {
                bind_recovery(quote, effects, options.recovery_citation);
            }
            memo.emplace(key, staged);
            counters.observe(rec.selection, quote);
            rec.destruction_quotes.push_back(std::move(staged));
        }
        return true;
    } catch (...) {
        rec.destruction_quotes.clear();
        return fail_whole_pass(
            common_cache_plan_destruction_state::failed,
            common_cache_plan_destruction_reason::internal_fault);
    }
}

common_cache_plan_destruction_quote
server_cache_destruction_quote_single_artifact(
        const server_cache_destruction_artifact & victim,
        common_cache_plan_destruction_effect_set effects,
        uint64_t accounting_serial,
        uint64_t admission_sequence,
        const server_cache_destruction_preview_callback & preview,
        const server_cache_destruction_projection_callback & project) noexcept {
    common_cache_plan_destruction_quote out;
    auto & receipt = out.receipt;
    receipt.effects = effects;
    receipt.plan_candidate = -1;
    receipt.admission_sequence = admission_sequence;
    receipt.quote_accounting_serial = accounting_serial;
    try {
        common_cache_plan_destruction_lease_verdict lease;
        const auto reason = server_cache_destruction_artifact_refusal(
            victim, false, lease);
        receipt.lease_verdict = lease;
        if (reason != common_cache_plan_destruction_reason::none) {
            refuse(receipt, reason);
            return out;
        }

        std::vector<const server_cache_destruction_artifact *> manifest = {
            &victim,
        };
        receipt.manifest_digest = manifest_digest(manifest);
        auto ops = victim.candidate.release_ops;
        std::sort(ops.begin(), ops.end());
        ops.erase(std::unique(ops.begin(), ops.end()), ops.end());
        llama_cache_acct_release_set_preview release;
        if (!preview || !preview(ops, accounting_serial, release)) {
            refuse(
                receipt,
                common_cache_plan_destruction_reason::accounting_unavailable);
            return out;
        }
        receipt.union_effect_digest =
            server_cache_destruction_union_effect_digest(ops, release);
        if (!project || !project(release, out.projected_domains)) {
            refuse(
                receipt,
                common_cache_plan_destruction_reason::capacity_refused);
            return out;
        }
        (victim.pool == common_retention_pool::attention
             ? receipt.selected_attention
             : receipt.selected_recurrent)
            .push_back(victim.candidate.artifact_id);
        receipt.state = common_cache_plan_destruction_state::quoted;
        receipt.reason = common_cache_plan_destruction_reason::none;
        return out;
    } catch (...) {
        out = {};
        out.receipt.effects = effects;
        out.receipt.state = common_cache_plan_destruction_state::failed;
        out.receipt.reason =
            common_cache_plan_destruction_reason::internal_fault;
        out.receipt.admission_sequence = admission_sequence;
        return out;
    }
}

common_cache_plan_destruction_quote
server_cache_destruction_quote_redundant_host(
        const server_cache_destruction_artifact & victim,
        uint64_t accounting_serial,
        uint64_t admission_sequence,
        const server_cache_destruction_preview_callback & preview,
        const server_cache_destruction_projection_callback & project) noexcept {
    return server_cache_destruction_quote_single_artifact(
        victim,
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::
                different_host_source_consumption),
        accounting_serial, admission_sequence, preview, project);
}

void server_cache_destruction_select_quote(
        common_cache_plan_record & rec,
        common_cache_plan_destruction_counters & counters,
        common_cache_plan_destruction_effect_set permitted_effects) noexcept {
    if (rec.destruction_quotes.empty()) {
        return;
    }
    if (!server_cache_plan_shadow_choice_valid(rec)) {
        rec.destruction.state = common_cache_plan_destruction_state::failed;
        rec.destruction.reason =
            common_cache_plan_destruction_reason::internal_fault;
        rec.destruction.selected_attention.clear();
        rec.destruction.selected_recurrent.clear();
        counters.observe(rec.selection, rec.destruction);
        return;
    }
    const auto it = std::find_if(
        rec.destruction_quotes.begin(), rec.destruction_quotes.end(),
        [&](const auto & quote) {
            return quote.receipt.plan_candidate == rec.shadow_choice;
        });
    if (it != rec.destruction_quotes.end()) {
        const uint64_t duration = rec.destruction.quote_duration_us;
        rec.destruction = it->receipt;
        rec.destruction.quote_duration_us = duration;
        return;
    }
    if (server_cache_destruction_effects_for(
            rec, rec.shadow_choice,
            rec.destruction_legacy_plan_candidate,
            permitted_effects) == 0) {
        const uint64_t duration = rec.destruction.quote_duration_us;
        const uint64_t sequence = rec.destruction.admission_sequence;
        rec.destruction = {};
        rec.destruction.quote_duration_us = duration;
        rec.destruction.admission_sequence = sequence;
        return;
    }
    rec.destruction.state = common_cache_plan_destruction_state::failed;
    rec.destruction.reason = common_cache_plan_destruction_reason::internal_fault;
    rec.destruction.selected_attention.clear();
    rec.destruction.selected_recurrent.clear();
    counters.observe(rec.selection, rec.destruction);
}

void server_cache_destruction_select_preview(
        common_cache_plan_record & rec,
        common_cache_plan_destruction_counters & counters,
        int32_t legacy_plan_candidate,
        bool lifecycle_available,
        common_cache_plan_destruction_effect_set permitted_effects) noexcept {
    if (lifecycle_available) {
        server_cache_destruction_select_quote(
            rec, counters, permitted_effects);
        return;
    }
    if (!server_cache_plan_shadow_choice_valid(rec)) {
        return;
    }
    rec.destruction_legacy_plan_candidate = legacy_plan_candidate;
    rec.destruction.plan_candidate = rec.shadow_choice;
    rec.destruction.effects = server_cache_destruction_effects_for(
        rec, rec.shadow_choice, legacy_plan_candidate, permitted_effects);
    if (rec.destruction.effects != 0) {
        rec.destruction.state =
            common_cache_plan_destruction_state::refused;
        rec.destruction.reason =
            common_cache_plan_destruction_reason::lifecycle_disabled;
        return;
    }
    const uint64_t duration = rec.destruction.quote_duration_us;
    rec.destruction = {};
    rec.destruction.quote_duration_us = duration;
}

void server_cache_destruction_finalize_projection(
        common_cache_plan_record & rec,
        const server_cache_yield_result & yield) noexcept {
    auto & receipt = rec.destruction;
    const auto quoted = std::find_if(
        rec.destruction_quotes.begin(), rec.destruction_quotes.end(),
        [&](const auto & item) {
            return item.receipt.plan_candidate == receipt.plan_candidate;
        });
    if (receipt.state != common_cache_plan_destruction_state::quoted ||
        !receipt.union_effect_digest.valid() ||
        quoted == rec.destruction_quotes.end() ||
        quoted->projected_domains.empty()) {
        return;
    }
    if (yield.status == server_cache_yield_status::fits &&
        yield.projected_fit.state == llama_cache_budget_fit_state::fits) {
        std::vector<llama_cache_acct_artifact_id> quote_ids = receipt.selected_attention;
        quote_ids.insert(quote_ids.end(), receipt.selected_recurrent.begin(), receipt.selected_recurrent.end());
        std::vector<llama_cache_acct_artifact_id> yield_ids = yield.selected[size_t(common_retention_pool::attention)];
        yield_ids.insert(yield_ids.end(), yield.selected[size_t(common_retention_pool::recurrent)].begin(),
                         yield.selected[size_t(common_retention_pool::recurrent)].end());
        std::sort(quote_ids.begin(), quote_ids.end());
        std::sort(yield_ids.begin(), yield_ids.end());
        bool same = quote_ids == yield_ids &&
                    quoted->projected_domains.size() == yield.projected_fit.domains.size();
        for (const auto & row : quoted->projected_domains) {
            const auto it = std::find_if(
                yield.projected_fit.domains.begin(), yield.projected_fit.domains.end(),
                [&](const auto & candidate) { return domain_row_equal(row, candidate); });
            same = same && it != yield.projected_fit.domains.end();
        }
        receipt.post_finalize_comparison = same
            ? common_cache_plan_destruction_comparison::matched
            : common_cache_plan_destruction_comparison::differed;
    } else if (yield.status ==
                   server_cache_yield_status::insufficient_yield) {
        receipt.post_finalize_comparison = common_cache_plan_destruction_comparison::
            ds6_insufficient_yield;
    } else if (yield.status ==
                   server_cache_yield_status::unsupported_required) {
        receipt.post_finalize_comparison = common_cache_plan_destruction_comparison::
            ds6_unsupported_required;
    } else {
        receipt.post_finalize_comparison =
            common_cache_plan_destruction_comparison::ds6_unavailable;
    }

    // Schema v6 defines the selected destruction quote as the sole projected-byte
    // source once its exact union is available. Yield accounting remains an independent
    // comparator: its complete result becomes matched/differed above and any
    // incomplete verdict becomes comparison=unavailable. We intentionally do
    // not serialize a second status/byte table. Actual remains explicitly
    // not_observed because shadow quoting never mutates.
    rec.yield.status = common_cache_plan_yield_status::fits;
    rec.yield.plan_state = common_cache_plan_yield_plan_state::planned;
    rec.yield.actual_state = common_cache_plan_yield_actual_state::not_observed;
    rec.yield.yield_policy_version = yield.yield_policy_version;
    rec.yield.accounting_serial = rec.acct.serial;
    rec.yield.selected_attention = receipt.selected_attention;
    rec.yield.selected_recurrent = receipt.selected_recurrent;
    rec.yield.projected_domains = quoted->projected_domains;
    rec.yield.actual_domains.clear();
}

bool server_cache_destruction_effect_matches(
        const common_cache_plan_destruction_receipt & quote,
        const common_cache_plan_destruction_effect_digest & current_effect,
        const std::vector<common_cache_plan_yield_domain> & quoted_domains,
        const std::vector<common_cache_plan_yield_domain> & current_domains) noexcept {
    return effect_equivalent(
        quote, current_effect, quoted_domains, current_domains, true);
}

common_cache_plan_destruction_reason server_cache_destruction_effect_recheck(
        const common_cache_plan_destruction_receipt & quote,
        const common_cache_plan_destruction_effect_digest & current_effect,
        const std::vector<common_cache_plan_yield_domain> & quoted_domains,
        const std::vector<common_cache_plan_yield_domain> & current_domains) noexcept {
    // Mutation-boundary equivalence is deliberately narrower than the projected-yield
    // full-row oracle above. Unrelated gauge/reservation traffic may change
    // before/after while leaving the selected union effect intact; only the
    // domain and exact projected release are capability inputs.
    return effect_equivalent(
               quote, current_effect, quoted_domains, current_domains, false)
        ? common_cache_plan_destruction_reason::none
        : common_cache_plan_destruction_reason::effect_drift;
}
