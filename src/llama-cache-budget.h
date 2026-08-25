#pragma once

#include "llama-cache-accounting.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Shadow budget arithmetic. These types are deliberately process-local:
// they are neither accounting schema nor cache-plan wire-record fields.
enum class llama_cache_budget_capacity_state : uint8_t {
    known = 0,
    unbounded,
    unavailable,
    _count,
};

enum class llama_cache_budget_reserve_provenance : uint8_t {
    configured = 0,
    measured,
    _count,
};

enum class llama_cache_budget_fit_state : uint8_t {
    fits = 0,
    exceeds,
    unavailable,
    _count,
};

// The classification vocabulary is closed. The CI census requires exactly one
// entry for every llama_cache_acct_category. `excluded` means the budget has no
// certified capacity-participating producer for that leaf yet; it is not a claim that
// the underlying subsystem consumes no memory. Retention metadata remains
// non-participating while the budget is observational.
enum class llama_cache_budget_capacity_participation : uint8_t {
    excluded = 0,
    participating,
    _count,
};

enum class llama_cache_budget_accounting_mode : uint8_t {
    direct_gauge = 0,
    transactional,
    _count,
};

enum class llama_cache_budget_residency_scope : uint8_t {
    none = 0,
    device,
    host,
    by_domain,
    _count,
};

#define LLAMA_CACHE_BUDGET_CATEGORY_TABLE(X)                                                   \
    X(live_attention_state,                  participating, direct_gauge,  device)              \
    X(live_recurrent_state,                  participating, direct_gauge,  device)              \
    X(recurrent_rollback_planes,             participating, direct_gauge,  device)              \
    X(full_snapshot_payload,                 participating, transactional, host)                \
    X(checkpoint_state_payload,              participating, transactional, host)                \
    X(typed_accelerator_payload,             participating, transactional, host)                \
    X(checkpoint_generation_page_metadata,   excluded, direct_gauge,  none)                     \
    X(checkpoint_generation_unit_metadata,   excluded, direct_gauge,  none)                     \
    X(live_generation_metadata,              excluded, direct_gauge,  none)                     \
    X(ownership_index_metadata,              excluded, direct_gauge,  none)                     \
    X(unit_version_payload,                  participating, transactional, by_domain)           \
    X(clean_stash_payload,                   participating, transactional, by_domain)           \
    X(artifact_descriptor_metadata,          participating, transactional, by_domain)           \
    X(artifact_reference_metadata,           participating, transactional, by_domain)           \
    X(transfer_staging,                      participating, transactional, by_domain)           \
    X(codec_workspace,                       participating, transactional, by_domain)           \
    X(pinned_preimage_ring,                  participating, direct_gauge,  by_domain)           \
    X(rolling_window_tape,                   participating, direct_gauge,  device)              \
    X(container_overhead,                    excluded, direct_gauge,  none)

#define LLAMA_CACHE_BUDGET_COUNT_CATEGORY(name, participation, mode, scope) + 1
constexpr size_t LLAMA_CACHE_BUDGET_CATEGORY_COUNT =
    0 LLAMA_CACHE_BUDGET_CATEGORY_TABLE(LLAMA_CACHE_BUDGET_COUNT_CATEGORY);
#undef LLAMA_CACHE_BUDGET_COUNT_CATEGORY
static_assert(LLAMA_CACHE_BUDGET_CATEGORY_COUNT ==
              size_t(llama_cache_acct_category::_count),
              "every accounting category needs one budget classification");

struct llama_cache_budget_category_classification {
    llama_cache_budget_capacity_participation participation =
        llama_cache_budget_capacity_participation::excluded;
    llama_cache_budget_accounting_mode mode       = llama_cache_budget_accounting_mode::direct_gauge;
    llama_cache_budget_residency_scope scope      = llama_cache_budget_residency_scope::none;
};

llama_cache_budget_category_classification llama_cache_budget_classify(
        llama_cache_acct_category category) noexcept;

struct llama_cache_budget_device_input {
    // Opaque ggml_backend_dev_t identity. The library never dereferences it.
    const void * backend_device = nullptr;
    llama_cache_acct_resource_domain domain;

    uint64_t physical_total = 0;
    uint64_t physical_free  = 0;
    llama_cache_budget_capacity_state phys_state =
        llama_cache_budget_capacity_state::unavailable;

    uint64_t current_compute_allocated  = 0;
    uint64_t configured_compute_reserve = 0;
    llama_cache_budget_reserve_provenance reserve_provenance =
        llama_cache_budget_reserve_provenance::configured;
    llama_cache_budget_capacity_state compute_state =
        llama_cache_budget_capacity_state::unavailable;

    uint64_t configured_cache_cap = 0;
    llama_cache_budget_capacity_state cache_cap_state =
        llama_cache_budget_capacity_state::unbounded;
};

struct llama_cache_budget_host_input {
    uint64_t pageable_cap = 0;
    llama_cache_budget_capacity_state pageable_state =
        llama_cache_budget_capacity_state::unbounded;
    uint64_t pinned_cap = 0;
    llama_cache_budget_capacity_state pinned_state =
        llama_cache_budget_capacity_state::known;

    // Optional shared host ceiling. Unbounded means no combined constraint.
    uint64_t total_cap = 0;
    llama_cache_budget_capacity_state total_state =
        llama_cache_budget_capacity_state::unbounded;
};

struct llama_cache_budget_config {
    std::vector<llama_cache_budget_device_input> devices;
    llama_cache_budget_host_input host;

    uint64_t administrative_global_cap = 0;
    llama_cache_budget_capacity_state global_cap_state =
        llama_cache_budget_capacity_state::unbounded;
};

struct llama_cache_budget_plan_entry {
    llama_cache_acct_resource_domain domain;
    uint64_t reserve_bytes = 0;
    uint64_t release_bytes = 0;
};

struct llama_cache_budget_plan {
    uint64_t accounting_serial = 0;
    std::vector<llama_cache_budget_plan_entry> entries;
};

enum class llama_cache_budget_resource_kind : uint8_t {
    accounting_domain = 0,
    physical_device,
    host_residency,
    host_total,
    administrative_global,
    _count,
};

struct llama_cache_budget_resource_key {
    llama_cache_budget_resource_kind kind =
        llama_cache_budget_resource_kind::accounting_domain;
    llama_cache_acct_resource_domain domain;
    const void * backend_device = nullptr;
    llama_cache_acct_residency residency =
        llama_cache_acct_residency::not_applicable;
};

struct llama_cache_budget_row {
    llama_cache_budget_resource_key resource;

    llama_cache_budget_capacity_state ceiling_state =
        llama_cache_budget_capacity_state::unavailable;
    llama_cache_acct_value ceiling;
    llama_cache_acct_value current_resident;
    llama_cache_acct_value before;
    llama_cache_acct_value released;
    llama_cache_acct_value reserved;
    llama_cache_acct_value after;
    llama_cache_budget_capacity_state headroom_state =
        llama_cache_budget_capacity_state::unavailable;
    llama_cache_acct_value headroom_after;
    llama_cache_budget_fit_state state = llama_cache_budget_fit_state::unavailable;
};

struct llama_cache_budget_result {
    uint64_t accounting_serial = 0;
    std::vector<llama_cache_budget_row> domains;
    std::vector<llama_cache_budget_row> groups;
    llama_cache_budget_fit_state state = llama_cache_budget_fit_state::unavailable;
};

class llama_cache_budget_coordinator {
public:
    bool reset(const llama_cache_acct_snapshot & snapshot,
               const llama_cache_budget_config & config) noexcept;
    bool reset(llama_cache_acct_snapshot && snapshot,
               const llama_cache_budget_config & config) noexcept;

    llama_cache_budget_result fits(const llama_cache_budget_plan & plan) const noexcept;

private:
    llama_cache_acct_snapshot snapshot_;
    llama_cache_budget_config config_;
    bool configured_ = false;
};

const char * llama_cache_budget_capacity_state_name(
        llama_cache_budget_capacity_state state) noexcept;
const char * llama_cache_budget_reserve_provenance_name(
        llama_cache_budget_reserve_provenance provenance) noexcept;
const char * llama_cache_budget_fit_state_name(
        llama_cache_budget_fit_state state) noexcept;
