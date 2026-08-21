#pragma once

#include "../../src/llama-cache-accounting.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

class vbr_artifact_package_view;
struct vbr_artifact_allocation_view;

struct server_prompt_cache_vbr_accounting_summary {
    uint64_t logical_bytes = 0;
    uint64_t resident_bytes = 0;
    size_t allocation_count = 0;
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

private:
    struct impl;
    explicit server_prompt_cache_vbr_payload(
        std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;
};

using server_prompt_cache_vbr_owner =
    std::shared_ptr<const server_prompt_cache_vbr_payload>;

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
    size_t allocation_count() const noexcept;
    bool accounted_by(const llama_cache_acct_ledger * ledger) const noexcept;

private:
    struct impl;
    explicit server_prompt_cache_vbr_variant_set(
        std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;
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
    size_t size() const noexcept;
    bool same_storage(const server_prompt_cache_payload & other) const noexcept;

private:
    std::variant<server_prompt_data, vbr_variant_owner> storage_;
};
