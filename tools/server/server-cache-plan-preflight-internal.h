#pragma once

#include "server-cache-plan-preflight.h"

#include <cstdint>
#include <unordered_map>

// Internal host-identity transport. Keys are the same stable list-node
// instance keys used by the retention catalog. This object never writes host
// entries and dies with one preflight pass; it stays out of server-task.h's
// transitive public-view include path.
class server_cache_plan_local_source_registry {
public:
    bool get_or_assign(uintptr_t instance, int32_t & source_id);
    bool find(uintptr_t instance, int32_t & source_id) const noexcept;
    size_t size() const noexcept { return source_ids_.size(); }

private:
    std::unordered_map<uintptr_t, int32_t> source_ids_;
    int32_t next_source_id_ = 0;
};

struct server_cache_plan_preflight_semantics {
    bool completion_semantics = false;
    bool host_lookup_enabled = false;
    common_cache_plan_recovery_citation recovery_citation =
        common_cache_plan_recovery_citation::unavailable;
};

// The preflight task enum is deliberately not a completion. This pure door
// supplies the as-if-completion predicates without overloading task identity.
server_cache_plan_preflight_semantics server_cache_plan_preflight_semantics_for(
    bool is_preflight,
    bool native_completion,
    bool update_cache,
    bool prompt_cache_available,
    bool adapter_matches) noexcept;
