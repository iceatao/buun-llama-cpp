#pragma once

#include "../../src/llama-cache-accounting.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

class vbr_artifact_package_view;
class vbr_artifact_prepared_retire;
struct vbr_artifact_allocation_view;

struct server_prompt_cache_vbr_accounting_summary {
    uint64_t logical_bytes = 0;
    uint64_t resident_bytes = 0;
    size_t allocation_count = 0;
};

struct server_prompt_cache_vbr_budget_summary {
    uint64_t compact_resident_bytes = 0;
    uint64_t anchor_resident_bytes = 0;
    size_t compact_allocations = 0;
    size_t anchor_allocations = 0;
};

// Internal H1 accounting kernel. The catalog is free to expose the same
// physical allocation through multiple immutable views; charge each exact
// allocation ID once and reject inconsistent aliases.
bool server_prompt_cache_summarize_vbr_allocations(
    std::vector<vbr_artifact_allocation_view> allocations,
    server_prompt_cache_vbr_accounting_summary & summary) noexcept;

struct server_prompt_data {
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;

    size_t size() const noexcept {
        return main.size() + drft.size();
    }
};

// Immutable lease over one sealed catalog package. The catalog remains the
// payload owner; copies of the shared pointer fan out without copying bytes or
// taking another catalog borrow. Releasing the final shared pointer drops the
// borrow and makes the reference eligible for its ordinary catalog retire
// transaction.
class server_prompt_cache_vbr_payload {
public:
    static std::shared_ptr<const server_prompt_cache_vbr_payload> adopt(
        vbr_artifact_package_view && package) noexcept;
    // Transfer the catalog reference into cache-owned retirement authority.
    // Unlike adopt(), dropping the last immutable owner retires the reference.
    static std::shared_ptr<const server_prompt_cache_vbr_payload> adopt_owned(
        vbr_artifact_package_view && package) noexcept;

    ~server_prompt_cache_vbr_payload();
    server_prompt_cache_vbr_payload(
        const server_prompt_cache_vbr_payload &) = delete;
    server_prompt_cache_vbr_payload & operator=(
        const server_prompt_cache_vbr_payload &) = delete;

    llama_cache_acct_artifact_id reference_artifact() const noexcept;
    uint64_t logical_bytes() const noexcept;
    uint64_t resident_bytes() const noexcept;
    size_t allocation_count() const noexcept;
    const vbr_artifact_package_view & package() const noexcept;
    bool accounted_by(const llama_cache_acct_ledger * ledger) const noexcept;
    bool retirement_owned() const noexcept;

private:
    struct impl;
    explicit server_prompt_cache_vbr_payload(
        std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;
};

using server_prompt_cache_vbr_owner =
    std::shared_ptr<const server_prompt_cache_vbr_payload>;

class server_prompt_cache_payload;

// Immutable same-frontier VBR variants for one logical prefix node. Compact
// current is required; a higher-quality anchor is optional and never replaces
// it. Shared catalog allocations are charged once across the set.
class server_prompt_cache_vbr_variant_set {
public:
    static std::shared_ptr<const server_prompt_cache_vbr_variant_set> create(
        server_prompt_cache_vbr_owner compact_current,
        server_prompt_cache_vbr_owner quality_anchor = {}) noexcept;

    ~server_prompt_cache_vbr_variant_set();
    server_prompt_cache_vbr_variant_set(
        const server_prompt_cache_vbr_variant_set &) = delete;
    server_prompt_cache_vbr_variant_set & operator=(
        const server_prompt_cache_vbr_variant_set &) = delete;

    const server_prompt_cache_vbr_owner & compact_current() const noexcept;
    const server_prompt_cache_vbr_owner & quality_anchor() const noexcept;
    uint64_t logical_bytes() const noexcept;
    uint64_t resident_bytes() const noexcept;
    uint64_t anchor_resident_bytes() const noexcept;
    size_t allocation_count() const noexcept;
    bool accounted_by(const llama_cache_acct_ledger * ledger) const noexcept;
    bool retirement_owned() const noexcept;
    bool retirement_exclusive() const noexcept;
    bool logical_erase_preserves_storage() const noexcept;
    bool has_quality_anchor() const noexcept;
    std::shared_ptr<const server_prompt_cache_vbr_variant_set>
        compact_only() const noexcept;
    bool preview_retire(
        uint64_t expected_serial,
        llama_cache_acct_release_set_preview & out) const noexcept;
    bool prepare_retire(
        uint64_t expected_serial,
        vbr_artifact_prepared_retire & out) const noexcept;
    static bool preview_retire_union(
        const std::vector<const server_prompt_cache_vbr_variant_set *> & variants,
        uint64_t expected_serial,
        llama_cache_acct_release_set_preview & out) noexcept;

private:
    bool prepare_anchor_retire_owned(
        uint64_t expected_serial,
        vbr_artifact_prepared_retire & out) const noexcept;
    struct impl;
    explicit server_prompt_cache_vbr_variant_set(
        std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;
    friend class server_prompt_cache_payload;
};

enum class server_prompt_cache_payload_kind : uint8_t {
    fixed_state = 0,
    vbr_artifact,
    _count,
};

// One logical host entry may carry either the legacy fixed state image or an
// immutable VBR catalog lease. H1 may publish either payload, but only fixed
// state is restorable until H2 wires VBR import/adoption.
class server_prompt_cache_payload {
public:
    using vbr_owner = server_prompt_cache_vbr_owner;
    using vbr_variant_owner =
        std::shared_ptr<const server_prompt_cache_vbr_variant_set>;

    server_prompt_cache_payload() = default;

    static server_prompt_cache_payload from_vbr(vbr_owner owner) noexcept;
    static server_prompt_cache_payload from_vbr_variants(
        vbr_variant_owner variants) noexcept;

    server_prompt_cache_payload_kind kind() const noexcept;
    server_prompt_data * fixed_state() noexcept;
    const server_prompt_data * fixed_state() const noexcept;
    const server_prompt_cache_vbr_payload * vbr_artifact() const noexcept;
    const server_prompt_cache_vbr_variant_set *
        vbr_variants() const noexcept;

    bool valid() const noexcept;
    bool publishable() const noexcept;
    bool restorable() const noexcept;
    bool accounted_by(const llama_cache_acct_ledger * ledger) const noexcept;
    bool vbr_retirement_owned() const noexcept;
    bool vbr_retirement_exclusive() const noexcept;
    bool vbr_logical_erase_only() const noexcept;
    bool vbr_has_quality_anchor() const noexcept;
    uint64_t vbr_anchor_resident_bytes() const noexcept;
    bool prepare_vbr_compact_only(
        server_prompt_cache_payload & out) const noexcept;
    // Prepare every physical anchor retirement that will be caused by
    // replacing this complete selected logical batch. Shared variant/anchor
    // owners are counted structurally: partial groups remain logical-only,
    // while every last-owner group receives one catalog capability before
    // the caller mutates its first payload.
    static bool prepare_vbr_anchor_retire_batch(
        const std::vector<const server_prompt_cache_payload *> & selected,
        uint64_t expected_serial,
        std::vector<vbr_artifact_prepared_retire> & out) noexcept;
    bool preview_vbr_retire(
        uint64_t expected_serial,
        llama_cache_acct_release_set_preview & out) const noexcept;
    bool prepare_vbr_retire(
        uint64_t expected_serial,
        vbr_artifact_prepared_retire & out) const noexcept;
    static bool preview_vbr_retire_union(
        const std::vector<const server_prompt_cache_payload *> & payloads,
        uint64_t expected_serial,
        llama_cache_acct_release_set_preview & out) noexcept;
    static bool summarize_vbr_budgets(
        const std::vector<const server_prompt_cache_payload *> & payloads,
        server_prompt_cache_vbr_budget_summary & out) noexcept;
    size_t size() const noexcept;
    bool same_storage(const server_prompt_cache_payload & other) const noexcept;

private:
    std::variant<server_prompt_data, vbr_variant_owner> storage_;
};

// One anchor-budget planning row. Parent value and recency are already
// normalized by the shared retention kernel; this layer owns only exact
// physical allocation sharing and the anchor-byte/stable tie-breaks.
struct server_prompt_cache_vbr_anchor_plan_candidate {
    const server_prompt_cache_payload * payload = nullptr;
    llama_cache_acct_artifact_id artifact_id;
    uint64_t parent_value_q = 0;
    uint64_t recency_ordinal = 0;
    uint8_t pool = 0;
    uint64_t lineage_id = 0;
    bool eligible = false;
};

constexpr size_t SERVER_PROMPT_CACHE_VBR_ANCHOR_MAX_CANDIDATES = 8192;

// Simulate one complete anchor pressure wave against the exact global
// allocation-ID union. Ineligible anchors retain allocation references.
// Selected artifacts are ordered by parent value, then the current marginal
// anchor bytes, then stable identity. No catalog or ledger state is mutated.
bool server_prompt_cache_plan_vbr_anchor_releases(
    const std::vector<server_prompt_cache_vbr_anchor_plan_candidate> & candidates,
    uint64_t current_anchor_bytes,
    uint64_t limit_anchor_bytes,
    std::vector<llama_cache_acct_artifact_id> & selected) noexcept;
