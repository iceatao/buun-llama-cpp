#include "server-cache-lease.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

namespace {

class steady_lease_clock final : public server_cache_lease_clock {
public:
    uint64_t now_ns() noexcept override {
        using namespace std::chrono;
        const auto now = steady_clock::now().time_since_epoch();
        const auto ns = duration_cast<nanoseconds>(now).count();
        return ns > 0 ? uint64_t(ns) : 0;
    }
};

class unavailable_fallback_provider final :
        public server_cache_lease_fallback_provider {
public:
    server_cache_durable_fallback_proof acquire(
            const server_cache_lease_subject &,
            const server_cache_lease_identity &) noexcept override {
        return {};
    }
};

steady_lease_clock default_clock;
unavailable_fallback_provider default_fallback;

bool checked_increment(uint64_t & value) noexcept {
    if (value == std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    ++value;
    return true;
}

bool target_is_lease_applicable(server_cache_destruction_target_kind kind) noexcept {
    return kind == server_cache_destruction_target_kind::live_target ||
           kind == server_cache_destruction_target_kind::checkpoint_ring ||
           kind == server_cache_destruction_target_kind::host_artifact;
}

} // namespace

server_cache_durable_fallback_proof
server_cache_durable_fallback_proof_for_test(
        server_cache_lease_fallback_state state,
        std::shared_ptr<void> owner) noexcept {
    if (state == server_cache_lease_fallback_state::available && !owner) {
        state = server_cache_lease_fallback_state::unavailable;
    } else if (state != server_cache_lease_fallback_state::available) {
        owner.reset();
    }
    return { state, std::move(owner) };
}

server_cache_lease_scope server_cache_lease_scope::from(
        server_cache_process_scope_id value) noexcept {
    return { server_cache_lease_scope_kind::process, value.v };
}

server_cache_lease_scope server_cache_lease_scope::from(
        server_cache_session_scope_id value) noexcept {
    return { server_cache_lease_scope_kind::session, value.v };
}

server_cache_lease_scope server_cache_lease_scope::from(
        server_cache_context_scope_id value) noexcept {
    return { server_cache_lease_scope_kind::context, value.v };
}

server_cache_lease_scope server_cache_lease_scope::from(
        server_cache_explicit_lease_scope_id value) noexcept {
    return { server_cache_lease_scope_kind::lease, value.v };
}

bool server_cache_lease_scope::valid() const noexcept {
    return kind < server_cache_lease_scope_kind::_count && id != 0;
}

bool server_cache_lease_identity::valid() const noexcept {
    return !execution_identity.empty() &&
           !adapter_config_identity.empty() &&
           !media_content_identity.empty();
}

bool operator==(
        const server_cache_lease_identity & a,
        const server_cache_lease_identity & b) noexcept {
    return a.execution_identity == b.execution_identity &&
           a.adapter_config_identity == b.adapter_config_identity &&
           a.media_content_identity == b.media_content_identity;
}

bool server_cache_lease_subject::valid() const noexcept {
    return artifact.v != 0 &&
           kind < common_retention_artifact_kind::_count;
}

server_cache_lease_table::server_cache_lease_table(
        server_cache_lease_clock * clock_in,
        server_cache_lease_fallback_provider * fallback_in) noexcept :
    clock(clock_in ? clock_in : &default_clock),
    fallback(fallback_in ? fallback_in : &default_fallback) {
}

void server_cache_lease_table::bind_fallback_provider(
        server_cache_lease_fallback_provider * provider) noexcept {
    fallback = provider ? provider : &default_fallback;
}

server_cache_context_scope_id
server_cache_lease_table::new_context_scope() noexcept {
    if (!available || next_context_scope_id == 0) {
        mark_table_unavailable();
        return {};
    }
    const server_cache_context_scope_id result { next_context_scope_id };
    if (!checked_increment(next_context_scope_id)) {
        mark_table_unavailable();
    }
    return result;
}

uint64_t server_cache_lease_table::sample_now() noexcept {
    n_clock_samples++;
    return clock->now_ns();
}

void server_cache_lease_table::mark_table_unavailable() noexcept {
    available = false;
    n_unavailable++;
}

bool server_cache_lease_table::checked_deadline(
        uint64_t now, uint64_t ttl, uint64_t & out) noexcept {
    if (ttl == 0 || ttl > std::numeric_limits<uint64_t>::max() - now) {
        mark_table_unavailable();
        return false;
    }
    out = now + ttl;
    return true;
}

server_cache_lease_id server_cache_lease_table::issue_lease_id() noexcept {
    if (!available || next_lease_id == 0) {
        mark_table_unavailable();
        return {};
    }
    const server_cache_lease_id result { next_lease_id };
    if (!checked_increment(next_lease_id)) {
        mark_table_unavailable();
    }
    return result;
}

server_cache_lease_identity_id server_cache_lease_table::intern_identity(
        const server_cache_lease_identity & value) noexcept {
    const auto existing = std::find_if(
        identities.begin(), identities.end(),
        [&](const auto & item) { return item.value == value; });
    if (existing != identities.end()) {
        return existing->id;
    }
    if (!available || next_identity_id == 0) {
        mark_table_unavailable();
        return {};
    }
    const server_cache_lease_identity_id result { next_identity_id };
    if (!checked_increment(next_identity_id)) {
        mark_table_unavailable();
        return {};
    }
    try {
        identities.push_back({ result, value });
        return result;
    } catch (...) {
        mark_table_unavailable();
        return {};
    }
}

void server_cache_lease_table::record(
        server_cache_lease_event_kind kind,
        const entry * value,
        server_cache_lease_id source,
        server_cache_lease_fallback_state fallback_state) noexcept {
    if (next_event_ordinal == 0) {
        mark_table_unavailable();
        return;
    }
    server_cache_lease_event event;
    event.kind = kind;
    event.source_lease = source;
    event.fallback = fallback_state;
    event.ordinal = next_event_ordinal;
    if (value) {
        event.lease = value->lease;
        event.scope = value->scope;
        event.artifact = value->subject.artifact;
        event.artifact_kind = value->subject.kind;
        event.owner_slot = value->subject.owner_slot;
        event.identity = value->identity_id;
        event.cls = value->cls;
        event.ttl_ns = value->ttl_ns;
    }
    events[size_t(n_events % events.size())] = event;
    event_totals[size_t(kind)]++;
    n_events++;
    if (n_events > events.size()) {
        n_event_overflows++;
    }
    if (!checked_increment(next_event_ordinal)) {
        mark_table_unavailable();
    }
}

server_cache_lease_table::entry * server_cache_lease_table::emit_grant(
        const server_cache_lease_subject & subject,
        const server_cache_lease_scope & scope,
        server_cache_lease_identity_id identity_id,
        server_cache_lease_class cls,
        uint64_t now,
        uint64_t deadline,
        uint64_t ttl_ns,
        server_cache_lease_event_kind kind,
        server_cache_durable_fallback_proof proof) noexcept {
    entry value;
    value.lease = issue_lease_id();
    value.subject = subject;
    value.scope = scope;
    value.identity_id = identity_id;
    value.cls = cls;
    value.granted_at_ns = now;
    value.expires_at_ns = deadline;
    value.ttl_ns = ttl_ns;
    value.fallback_proof = std::move(proof);
    if (!value.lease || !identity_id || !add_entry(std::move(value), kind)) {
        return nullptr;
    }
    return &leases.back();
}

bool server_cache_lease_table::add_entry(
        entry && value,
        server_cache_lease_event_kind kind,
        server_cache_lease_id source) noexcept {
    try {
        leases.push_back(std::move(value));
        record(kind, &leases.back(), source,
               server_cache_lease_fallback_state::available);
        leases.back().last_event_ordinal = next_event_ordinal - 1;
        return true;
    } catch (...) {
        mark_table_unavailable();
        return false;
    }
}

server_cache_lease_table::entry server_cache_lease_table::clone_core(
        const entry & source) noexcept {
    entry clone;
    clone.scope = source.scope;
    clone.identity_id = source.identity_id;
    clone.cls = source.cls;
    clone.expires_at_ns = source.expires_at_ns;
    clone.ttl_ns = source.ttl_ns;
    clone.fallback_proof = source.fallback_proof.retain();
    clone.owner = source.owner;
    clone.proven_frontier = source.proven_frontier;
    clone.explicit_hard = source.explicit_hard;
    clone.orphaned = source.orphaned;
    clone.subject_lost = source.subject_lost;
    return clone;
}

bool server_cache_lease_table::owned_scope_match(
        const entry & value,
        server_cache_explicit_lease_scope_id scope,
        server_cache_lease_owner_id owner) noexcept {
    return value.owner == owner && value.explicit_hard &&
           value.scope.kind == server_cache_lease_scope_kind::lease &&
           value.scope.id == scope.v;
}

bool server_cache_lease_table::orphan_entry(entry & value) noexcept {
    if (!value.explicit_hard || value.orphaned) {
        return false;
    }
    value.orphaned = true;
    record(server_cache_lease_event_kind::orphan_hard, &value, {},
           server_cache_lease_fallback_state::available);
    value.last_event_ordinal = next_event_ordinal - 1;
    return true;
}

void server_cache_lease_table::mark_subject_lost(size_t index) noexcept {
    const auto scope = server_cache_explicit_lease_scope_id {
        leases[index].scope.id,
    };
    const auto owner = leases[index].owner;
    for (auto & value : leases) {
        if (&value != &leases[index] &&
            !owned_scope_match(value, scope, owner)) {
            continue;
        }
        if (!value.subject_lost) {
            record(server_cache_lease_event_kind::invalidate_identity,
                   &value, {}, server_cache_lease_fallback_state::available);
            value.last_event_ordinal = next_event_ordinal - 1;
            value.subject_lost = true;
        }
        value.fallback_proof = {};
    }
}

void server_cache_lease_table::invalidate_entry(
        size_t index, server_cache_lease_event_kind kind) noexcept {
    record(kind, &leases[index], {}, server_cache_lease_fallback_state::available);
    leases.erase(leases.begin() + index);
}

void server_cache_lease_table::mark_identity_unavailable(
        const server_cache_lease_subject & subject) noexcept {
    const auto existing = std::find_if(
        identity_unavailable.begin(), identity_unavailable.end(),
        [&](const auto & value) {
            return value.artifact == subject.artifact;
        });
    if (existing != identity_unavailable.end()) {
        return;
    }
    try {
        identity_unavailable.push_back(subject);
        entry marker;
        marker.subject = subject;
        record(
            server_cache_lease_event_kind::mark_identity_unavailable,
            &marker, {}, server_cache_lease_fallback_state::invalid);
    } catch (...) {
        mark_table_unavailable();
    }
}

void server_cache_lease_table::clear_identity_unavailable(
        llama_cache_acct_artifact_id artifact) noexcept {
    const auto existing = std::find_if(
        identity_unavailable.begin(), identity_unavailable.end(),
        [&](const auto & value) {
            return value.artifact == artifact;
        });
    if (existing == identity_unavailable.end()) {
        return;
    }
    entry marker;
    marker.subject = *existing;
    record(
        server_cache_lease_event_kind::clear_identity_unavailable,
        &marker, {}, server_cache_lease_fallback_state::available);
    identity_unavailable.erase(existing);
}

void server_cache_lease_table::expire_due(uint64_t now) noexcept {
    for (size_t i = 0; i < leases.size();) {
        if (!leases[i].explicit_hard && leases[i].expires_at_ns <= now) {
            invalidate_entry(i, server_cache_lease_event_kind::expire);
        } else {
            ++i;
        }
    }
}

server_cache_lease_id server_cache_lease_table::grant_soft(
        const server_cache_lease_subject & subject,
        const server_cache_lease_scope & scope,
        const server_cache_lease_identity & identity,
        uint64_t ttl_ns) noexcept {
    const uint64_t now = sample_now();
    expire_due(now);
    uint64_t deadline = 0;
    if (!available || !subject.valid() || !scope.valid() || !identity.valid() ||
        !checked_deadline(now, ttl_ns, deadline)) {
        n_unavailable++;
        return {};
    }
    const auto identity_id = intern_identity(identity);
    if (!identity_id) {
        return {};
    }
    clear_identity_unavailable(subject.artifact);

    // The implicit context lease is renewed in place when the same artifact and
    // scope are observed again; a changed identity invalidates before replacement.
    for (size_t i = 0; i < leases.size(); ++i) {
        auto & existing = leases[i];
        if (existing.subject.artifact == subject.artifact &&
            existing.scope.kind == scope.kind &&
            existing.scope.id == scope.id) {
            if (existing.identity_id != identity_id) {
                invalidate_entry(i, server_cache_lease_event_kind::invalidate_identity);
                break;
            }
            if (deadline > existing.expires_at_ns) {
                existing.expires_at_ns = deadline;
                existing.ttl_ns = ttl_ns;
            }
            record(server_cache_lease_event_kind::renew, &existing, {},
                   server_cache_lease_fallback_state::available);
            existing.last_event_ordinal = next_event_ordinal - 1;
            return existing.lease;
        }
    }

    const auto * granted = emit_grant(
        subject, scope, identity_id, server_cache_lease_class::soft,
        now, deadline, ttl_ns, server_cache_lease_event_kind::grant_soft);
    return granted ? granted->lease : server_cache_lease_id{};
}

server_cache_lease_id server_cache_lease_table::grant_hard(
        const server_cache_lease_subject & subject,
        const server_cache_lease_scope & scope,
        const server_cache_lease_identity & identity,
        uint64_t ttl_ns) noexcept {
    const auto * granted = grant_hard_entry(subject, scope, identity, ttl_ns);
    return granted ? granted->lease : server_cache_lease_id{};
}

server_cache_lease_table::entry * server_cache_lease_table::grant_hard_entry(
        const server_cache_lease_subject & subject,
        const server_cache_lease_scope & scope,
        const server_cache_lease_identity & identity,
        uint64_t ttl_ns) noexcept {
    const uint64_t now = sample_now();
    expire_due(now);
    uint64_t deadline = 0;
    if (!available || !subject.valid() || !scope.valid() || !identity.valid() ||
        !checked_deadline(now, ttl_ns, deadline)) {
        n_unavailable++;
        return nullptr;
    }
    auto proof = fallback->acquire(subject, identity);
    const auto identity_id = intern_identity(identity);
    if (!identity_id) {
        return nullptr;
    }
    const auto proof_state = proof.state();
    if (!proof.available()) {
        entry refusal;
        refusal.subject = subject;
        refusal.scope = scope;
        refusal.identity_id = identity_id;
        refusal.cls = server_cache_lease_class::hard;
        refusal.ttl_ns = ttl_ns;
        record(
            proof_state == server_cache_lease_fallback_state::invalid
                ? server_cache_lease_event_kind::refuse_hard_invalid
                : server_cache_lease_event_kind::refuse_hard_unavailable,
            &refusal, {}, proof_state);
        return nullptr;
    }
    clear_identity_unavailable(subject.artifact);
    return emit_grant(
        subject, scope, identity_id, server_cache_lease_class::hard,
        now, deadline, ttl_ns, server_cache_lease_event_kind::grant_hard,
        std::move(proof));
}

server_cache_lease_id server_cache_lease_table::grant_hard_owned(
        const server_cache_lease_subject & subject,
        const server_cache_lease_scope & scope,
        const server_cache_lease_identity & identity,
        server_cache_lease_owner_id owner,
        const server_cache_lease_frontier & proven_frontier,
        uint64_t ttl_ns) noexcept {
    if (!owner || !proven_frontier.valid()) {
        n_unavailable++;
        return {};
    }
    auto * granted = grant_hard_entry(subject, scope, identity, ttl_ns);
    if (!granted) {
        return {};
    }
    granted->owner = owner;
    granted->proven_frontier = proven_frontier;
    granted->explicit_hard = true;
    granted->orphaned = false;
    return granted->lease;
}

bool server_cache_lease_table::renew_owned(
        server_cache_lease_id lease,
        server_cache_lease_owner_id owner,
        const server_cache_lease_frontier & proven_frontier,
        uint64_t ttl_ns) noexcept {
    const uint64_t now = sample_now();
    expire_due(now);
    auto it = std::find_if(leases.begin(), leases.end(),
        [lease, owner](const entry & value) {
            return value.lease == lease && value.owner == owner &&
                   value.explicit_hard;
        });
    uint64_t deadline = 0;
    if (it == leases.end() || !proven_frontier.valid() ||
        !checked_deadline(now, ttl_ns, deadline)) {
        return false;
    }
    const auto identity = std::find_if(
        identities.begin(), identities.end(), [&](const auto & value) {
            return value.id == it->identity_id;
        });
    if (identity == identities.end()) {
        mark_table_unavailable();
        return false;
    }
    auto proof = fallback->acquire(it->subject, identity->value);
    if (!proof.available()) {
        return false;
    }
    it->fallback_proof = std::move(proof);
    it->proven_frontier = proven_frontier;
    it->expires_at_ns = deadline;
    it->ttl_ns = ttl_ns;
    it->orphaned = false;
    record(server_cache_lease_event_kind::renew, &*it, {},
           server_cache_lease_fallback_state::available);
    it->last_event_ordinal = next_event_ordinal - 1;
    return true;
}

bool server_cache_lease_table::orphan_owner(
        server_cache_lease_owner_id owner) noexcept {
    bool changed = false;
    for (auto & value : leases) {
        if (value.owner == owner) {
            changed |= orphan_entry(value);
        }
    }
    return changed;
}

bool server_cache_lease_table::orphan_owned_scope(
        server_cache_explicit_lease_scope_id scope,
        server_cache_lease_owner_id owner) noexcept {
    bool changed = false;
    for (auto & value : leases) {
        if (owned_scope_match(value, scope, owner)) {
            changed |= orphan_entry(value);
        }
    }
    return changed;
}

bool server_cache_lease_table::release_owned_scope(
        server_cache_explicit_lease_scope_id scope,
        server_cache_lease_owner_id owner) noexcept {
    if (!scope.v || !owner) {
        return false;
    }
    bool released = false;
    for (size_t i = 0; i < leases.size();) {
        if (owned_scope_match(leases[i], scope, owner)) {
            invalidate_entry(i, server_cache_lease_event_kind::release);
            released = true;
        } else {
            ++i;
        }
    }
    return released;
}

bool server_cache_lease_table::lease_active(
        server_cache_lease_id lease) const noexcept {
    return std::any_of(leases.begin(), leases.end(),
        [lease](const entry & value) { return value.lease == lease; });
}

bool server_cache_lease_table::lease_subject_lost(
        server_cache_lease_id lease) const noexcept {
    const auto found = std::find_if(leases.begin(), leases.end(),
        [lease](const entry & value) { return value.lease == lease; });
    return found != leases.end() && found->subject_lost;
}

bool server_cache_lease_table::owned_scope_active(
        server_cache_explicit_lease_scope_id scope,
        server_cache_lease_owner_id owner) const noexcept {
    return std::any_of(leases.begin(), leases.end(), [&](const entry & value) {
        return owned_scope_match(value, scope, owner);
    });
}

bool server_cache_lease_table::renew(
        server_cache_lease_id lease, uint64_t ttl_ns) noexcept {
    const uint64_t now = sample_now();
    expire_due(now);
    const auto it = std::find_if(leases.begin(), leases.end(),
        [lease](const entry & value) { return value.lease == lease; });
    if (it == leases.end()) {
        return false;
    }
    uint64_t deadline = 0;
    if (!checked_deadline(now, ttl_ns, deadline)) {
        return false;
    }
    if (deadline > it->expires_at_ns) {
        it->expires_at_ns = deadline;
        it->ttl_ns = ttl_ns;
    }
    record(server_cache_lease_event_kind::renew, &*it, {},
           server_cache_lease_fallback_state::available);
    it->last_event_ordinal = next_event_ordinal - 1;
    return true;
}

bool server_cache_lease_table::release(server_cache_lease_id lease) noexcept {
    const uint64_t now = sample_now();
    expire_due(now);
    for (size_t i = 0; i < leases.size(); ++i) {
        if (leases[i].lease == lease) {
            invalidate_entry(i, server_cache_lease_event_kind::release);
            return true;
        }
    }
    return false;
}

void server_cache_lease_table::lifecycle_point() noexcept {
    const uint64_t now = sample_now();
    expire_due(now);
}

void server_cache_lease_table::artifact_identity_unavailable(
        const server_cache_lease_subject & subject) noexcept {
    expire_due(sample_now());
    if (!subject.valid()) {
        n_unavailable++;
        return;
    }
    mark_identity_unavailable(subject);
    n_unavailable++;
}

void server_cache_lease_table::artifact_retired(
        llama_cache_acct_artifact_id artifact) noexcept {
    const uint64_t now = sample_now();
    expire_due(now);
    for (size_t i = 0; i < leases.size();) {
        if (leases[i].subject.artifact == artifact) {
            if (leases[i].explicit_hard) {
                mark_subject_lost(i);
                ++i;
            } else {
                invalidate_entry(i, server_cache_lease_event_kind::release);
            }
        } else {
            ++i;
        }
    }
    clear_identity_unavailable(artifact);
}

bool server_cache_lease_table::artifact_cloned(
        const server_cache_lease_subject & source,
        const server_cache_lease_subject & destination,
        const server_cache_lease_identity & destination_identity) noexcept {
    const uint64_t now = sample_now();
    expire_due(now);
    if (!source.valid() || !destination.valid() ||
        !destination_identity.valid()) {
        if (destination.valid()) {
            artifact_identity_unavailable(destination);
        }
        n_unavailable++;
        return false;
    }
    const auto destination_identity_id = intern_identity(destination_identity);
    if (!destination_identity_id) {
        return false;
    }
    struct pending_clone {
        entry value;
        server_cache_lease_id parent;
    };
    std::vector<pending_clone> clones;
    try {
        for (const auto & existing : leases) {
            if (existing.subject.artifact != source.artifact) {
                continue;
            }
            if (existing.identity_id != destination_identity_id) {
                mark_identity_unavailable(destination);
                n_unavailable++;
                return false;
            }
            pending_clone clone;
            clone.value = clone_core(existing);
            clone.parent = existing.lease;
            clone.value.lease = issue_lease_id();
            clone.value.subject = destination;
            clone.value.granted_at_ns = now;
            clone.value.ttl_ns = existing.expires_at_ns - now;
            if (!clone.value.lease || clone.value.ttl_ns == 0) {
                return false;
            }
            clones.push_back(std::move(clone));
        }
        for (auto & clone : clones) {
            if (!add_entry(
                    std::move(clone.value),
                    server_cache_lease_event_kind::clone,
                    clone.parent)) {
                return false;
            }
        }
        clear_identity_unavailable(destination.artifact);
        return true;
    } catch (...) {
        mark_table_unavailable();
        return false;
    }
}

bool server_cache_lease_table::artifact_replaced(
        const server_cache_lease_subject & source,
        const server_cache_lease_subject & destination,
        const server_cache_lease_identity & destination_identity,
        const server_cache_lease_frontier & current_frontier) noexcept {
    const uint64_t now = sample_now();
    expire_due(now);
    if (!source.valid() || !destination.valid() ||
        source.artifact == destination.artifact ||
        !destination_identity.valid() || !current_frontier.valid()) {
        return false;
    }

    // Validate the complete replacement set before mutating one entry. Token
    // content is represented by the stable sequence epoch and the proven
    // frontier; it is deliberately not folded into the execution/adapter/
    // media identity. A trim or rebind therefore fails this containment test.
    for (const auto & lease : leases) {
        if (lease.subject.artifact != source.artifact || lease.subject_lost) {
            continue;
        }
        const auto identity = std::find_if(
            identities.begin(), identities.end(), [&](const auto & value) {
                return value.id == lease.identity_id;
            });
        if (identity == identities.end() ||
            identity->value != destination_identity ||
            (lease.proven_frontier.valid() &&
             (lease.proven_frontier.sequence_epoch !=
                  current_frontier.sequence_epoch ||
              lease.proven_frontier.token_count >
                  current_frontier.token_count ||
              lease.proven_frontier.next_position >
                  current_frontier.next_position))) {
            return false;
        }
    }
    for (auto & lease : leases) {
        if (lease.subject.artifact == source.artifact &&
            !lease.subject_lost) {
            lease.subject = destination;
        }
    }
    // The source association is gone after the caller commits this atomic
    // replacement. Do not retain a diagnostic marker that can only name the
    // destroyed observer artifact.
    clear_identity_unavailable(source.artifact);
    clear_identity_unavailable(destination.artifact);
    return true;
}

bool server_cache_lease_table::artifact_rebound(
        llama_cache_acct_artifact_id artifact,
        const server_cache_lease_identity & expected_identity) noexcept {
    const uint64_t now = sample_now();
    expire_due(now);
    if (!expected_identity.valid()) {
        const auto it = std::find_if(
            leases.begin(), leases.end(),
            [&](const auto & value) {
                return value.subject.artifact == artifact;
            });
        if (it != leases.end()) {
            artifact_identity_unavailable(it->subject);
        }
        n_unavailable++;
        return false;
    }
    const auto expected_identity_id = intern_identity(expected_identity);
    if (!expected_identity_id) {
        return false;
    }
    bool matched = false;
    bool mismatched = false;
    server_cache_lease_subject rebound_subject;
    for (size_t i = 0; i < leases.size();) {
        if (leases[i].subject.artifact != artifact) {
            ++i;
            continue;
        }
        matched = true;
        if (leases[i].identity_id != expected_identity_id) {
            rebound_subject = leases[i].subject;
            mismatched = true;
            if (leases[i].explicit_hard) {
                mark_subject_lost(i);
                ++i;
            } else {
                invalidate_entry(
                    i, server_cache_lease_event_kind::invalidate_identity);
            }
        } else {
            ++i;
        }
    }
    if (mismatched) {
        mark_identity_unavailable(rebound_subject);
    } else if (matched) {
        clear_identity_unavailable(artifact);
    }
    return matched;
}

server_cache_lease_evaluation server_cache_lease_table::evaluate(
        llama_cache_acct_artifact_id artifact,
        const server_cache_lease_identity & expected_identity) noexcept {
    server_cache_lease_evaluation result;
    const uint64_t now = sample_now();
    expire_due(now);
    if (!available || artifact.v == 0 || !expected_identity.valid()) {
        return result;
    }
    const auto expected_identity_id = intern_identity(expected_identity);
    if (!expected_identity_id) {
        return result;
    }
    if (std::any_of(
            identity_unavailable.begin(), identity_unavailable.end(),
            [&](const auto & value) {
                return value.artifact == artifact;
            })) {
        return result;
    }
    result.state = server_cache_lease_eval_state::known;
    bool mismatched = false;
    server_cache_lease_subject mismatch_subject;
    for (size_t i = 0; i < leases.size();) {
        if (leases[i].subject.artifact != artifact) {
            ++i;
            continue;
        }
        if (leases[i].identity_id != expected_identity_id) {
            mismatch_subject = leases[i].subject;
            mismatched = true;
            if (leases[i].explicit_hard) {
                mark_subject_lost(i);
                ++i;
            } else {
                invalidate_entry(
                    i, server_cache_lease_event_kind::invalidate_identity);
            }
            result.state = server_cache_lease_eval_state::unavailable;
            result.cls = server_cache_lease_class::none;
            result.eligibility = server_cache_lease_eligibility::eligible;
            continue;
        }
        if (leases[i].subject_lost) {
            result.state = server_cache_lease_eval_state::unavailable;
            result.cls = server_cache_lease_class::none;
            result.eligibility = server_cache_lease_eligibility::eligible;
            ++i;
            continue;
        }
        if (leases[i].cls > result.cls) {
            result.cls = leases[i].cls;
        }
        ++i;
    }
    if (mismatched) {
        mark_identity_unavailable(mismatch_subject);
    }
    if (result.cls == server_cache_lease_class::hard) {
        result.eligibility = server_cache_lease_eligibility::hard_blocked;
    }
    return result;
}

server_cache_lease_evaluation server_cache_lease_table::inspect(
        llama_cache_acct_artifact_id artifact,
        const server_cache_lease_identity & expected_identity) const noexcept {
    server_cache_lease_evaluation result;
    if (!available || artifact.v == 0 || !expected_identity.valid()) {
        return result;
    }
    if (std::any_of(
            identity_unavailable.begin(), identity_unavailable.end(),
            [&](const auto & value) {
                return value.artifact == artifact;
            })) {
        return result;
    }

    // Observe expiry without driving the lifecycle state machine. A later
    // lifecycle point records and removes the same expired entries.
    const uint64_t now = clock->now_ns();
    result.state = server_cache_lease_eval_state::known;
    for (const auto & lease : leases) {
        if (lease.subject.artifact != artifact ||
            lease.subject_lost ||
            (!lease.explicit_hard && lease.expires_at_ns <= now)) {
            continue;
        }
        const auto identity = std::find_if(
            identities.begin(), identities.end(),
            [&](const auto & value) {
                return value.id == lease.identity_id;
            });
        if (identity == identities.end() ||
            identity->value != expected_identity) {
            result.state = server_cache_lease_eval_state::unavailable;
            result.cls = server_cache_lease_class::none;
            result.eligibility = server_cache_lease_eligibility::eligible;
            return result;
        }
        if (lease.cls > result.cls) {
            result.cls = lease.cls;
        }
    }
    if (result.cls == server_cache_lease_class::hard) {
        result.eligibility = server_cache_lease_eligibility::hard_blocked;
    }
    return result;
}

bool server_cache_lease_table::inspect_batch(
        const std::vector<server_cache_lease_inspection_request> & requests,
        std::vector<server_cache_lease_evaluation> & out) const noexcept {
    out.clear();
    try {
        out.resize(requests.size());
        if (!available) {
            return true;
        }

        std::vector<size_t> request_order(requests.size());
        for (size_t i = 0; i < requests.size(); ++i) {
            request_order[i] = i;
        }
        std::sort(request_order.begin(), request_order.end(),
            [&](size_t a, size_t b) {
                return requests[a].artifact.v < requests[b].artifact.v;
            });
        for (size_t i = 0; i < request_order.size(); ++i) {
            const auto & request = requests[request_order[i]];
            if (request.artifact.v == 0 || !request.expected_identity ||
                !request.expected_identity->valid() ||
                (i != 0 && requests[request_order[i - 1]].artifact.v ==
                    request.artifact.v)) {
                out.clear();
                return false;
            }
            out[request_order[i]].state =
                server_cache_lease_eval_state::known;
        }

        std::vector<size_t> identity_order(identities.size());
        for (size_t i = 0; i < identities.size(); ++i) {
            identity_order[i] = i;
        }
        std::sort(identity_order.begin(), identity_order.end(),
            [&](size_t a, size_t b) {
                return identities[a].id.v < identities[b].id.v;
            });

        const auto find_request = [&](uint64_t artifact) {
            return std::lower_bound(
                request_order.begin(), request_order.end(), artifact,
                [&](size_t index, uint64_t value) {
                    return requests[index].artifact.v < value;
                });
        };
        for (const auto & unavailable : identity_unavailable) {
            const auto it = find_request(unavailable.artifact.v);
            if (it != request_order.end() &&
                requests[*it].artifact == unavailable.artifact) {
                out[*it] = {};
            }
        }

        const uint64_t now = clock->now_ns();
        for (const auto & lease : leases) {
            if (lease.subject_lost ||
                (!lease.explicit_hard && lease.expires_at_ns <= now)) {
                continue;
            }
            const auto request_it = find_request(lease.subject.artifact.v);
            if (request_it == request_order.end() ||
                requests[*request_it].artifact != lease.subject.artifact) {
                continue;
            }
            auto & result = out[*request_it];
            if (result.state != server_cache_lease_eval_state::known) {
                continue;
            }
            const auto identity_it = std::lower_bound(
                identity_order.begin(), identity_order.end(),
                lease.identity_id.v,
                [&](size_t index, uint64_t value) {
                    return identities[index].id.v < value;
                });
            if (identity_it == identity_order.end() ||
                identities[*identity_it].id != lease.identity_id ||
                identities[*identity_it].value !=
                    *requests[*request_it].expected_identity) {
                result = {};
                continue;
            }
            if (lease.cls > result.cls) {
                result.cls = lease.cls;
            }
        }
        for (auto & result : out) {
            if (result.state == server_cache_lease_eval_state::known &&
                result.cls == server_cache_lease_class::hard) {
                result.eligibility =
                    server_cache_lease_eligibility::hard_blocked;
            }
        }
        return true;
    } catch (...) {
        out.clear();
        return false;
    }
}

server_cache_lease_evaluation server_cache_lease_table::inspect_range(
        llama_cache_acct_artifact_id artifact,
        const server_cache_lease_identity & expected_identity,
        uint64_t sequence_epoch,
        uint64_t first_token,
        uint64_t token_count) const noexcept {
    auto result = inspect(artifact, expected_identity);
    if (result.state != server_cache_lease_eval_state::known ||
        result.cls != server_cache_lease_class::hard || token_count == 0 ||
        first_token > std::numeric_limits<uint64_t>::max() - token_count) {
        return result;
    }
    const uint64_t end = first_token + token_count;
    bool saw_unscoped_hard = false;
    bool overlaps_scoped_hard = false;
    bool saw_soft = false;
    const uint64_t now = clock->now_ns();
    for (const auto & lease : leases) {
        if (lease.subject.artifact != artifact ||
            lease.subject_lost ||
            (!lease.explicit_hard && lease.expires_at_ns <= now)) {
            continue;
        }
        if (lease.cls == server_cache_lease_class::soft) {
            saw_soft = true;
            continue;
        }
        if (!lease.explicit_hard || !lease.proven_frontier.valid()) {
            saw_unscoped_hard = true;
            continue;
        }
        if (lease.proven_frontier.sequence_epoch != sequence_epoch) {
            result.state = server_cache_lease_eval_state::unavailable;
            return result;
        }
        overlaps_scoped_hard |= first_token < lease.proven_frontier.token_count &&
                                end > 0;
    }
    if (!saw_unscoped_hard && !overlaps_scoped_hard) {
        result.cls = saw_soft ? server_cache_lease_class::soft
                              : server_cache_lease_class::none;
        result.eligibility = server_cache_lease_eligibility::eligible;
    }
    return result;
}

bool server_cache_lease_table::has_hard_lease() const noexcept {
    if (!available) {
        return false;
    }
    const uint64_t now = clock->now_ns();
    return std::any_of(leases.begin(), leases.end(), [&](const entry & lease) {
        return lease.cls == server_cache_lease_class::hard &&
            !lease.subject_lost &&
            (lease.explicit_hard || lease.expires_at_ns > now);
    });
}

bool server_cache_hard_lease_blocks_range(
        const server_cache_lease_table * leases,
        llama_cache_acct_artifact_id artifact,
        const server_cache_lease_identity & identity,
        uint64_t sequence_epoch,
        uint64_t first_token,
        uint64_t token_count) noexcept {
    return leases != nullptr && server_cache_lease_is_hard(
        leases->inspect_range(
            artifact, identity, sequence_epoch, first_token, token_count));
}

bool server_cache_has_hard_lease(
        const server_cache_lease_table * leases) noexcept {
    return leases != nullptr && leases->has_hard_lease();
}

bool server_cache_lease_table::admit_checkpoint_ring(
        const server_cache_destruction_target & target,
        bool & saw_soft,
        bool & saw_hard) const noexcept {
    if (target.slot_id < 0) {
        return false;
    }
    for (const auto & value : leases) {
        if (value.subject.kind ==
                common_retention_artifact_kind::checkpoint &&
            value.subject.owner_slot == target.slot_id) {
            saw_hard |= value.cls == server_cache_lease_class::hard;
            saw_soft |= value.cls == server_cache_lease_class::soft;
        }
    }
    return std::none_of(
        identity_unavailable.begin(), identity_unavailable.end(),
        [&](const auto & value) {
            return value.kind == common_retention_artifact_kind::checkpoint &&
                   value.owner_slot == target.slot_id;
        });
}

bool server_cache_lease_table::admit_scalar_artifact(
        const server_cache_destruction_target & target,
        bool & saw_soft,
        bool & saw_hard) const noexcept {
    if (!target.artifact_known || target.artifact.v == 0 ||
        std::any_of(
            identity_unavailable.begin(), identity_unavailable.end(),
            [&](const auto & value) {
                return value.artifact == target.artifact;
            })) {
        return false;
    }
    if (target.kind == server_cache_destruction_target_kind::host_artifact) {
        // A host-cache entry owns copied checkpoint records, but the retention catalog does not
        // yet persist their parent relation. Fail closed if any host-side
        // checkpoint evidence cannot be assigned to this scalar target.
        const auto unresolved_host_checkpoint = [&](const auto & value) {
            return value.kind == common_retention_artifact_kind::checkpoint &&
                   value.owner_slot < 0;
        };
        if (std::any_of(
                leases.begin(), leases.end(),
                [&](const auto & value) {
                    return unresolved_host_checkpoint(value.subject);
                }) ||
            std::any_of(
                identity_unavailable.begin(), identity_unavailable.end(),
                unresolved_host_checkpoint)) {
            return false;
        }
    }
    for (const auto & value : leases) {
        if (value.subject.artifact == target.artifact) {
            saw_hard |= value.cls == server_cache_lease_class::hard;
            saw_soft |= value.cls == server_cache_lease_class::soft;
        }
    }
    return true;
}

server_cache_destruction_verdict server_cache_lease_table::admit(
        const server_cache_destruction_request & request) noexcept {
    if (request.cls == server_cache_destruction_class::mandatory_recovery_reset) {
        lifecycle_point();
        return server_cache_destruction_verdict::admit_mandatory_recovery;
    }
    const uint64_t now = sample_now();
    expire_due(now);
    if (!available || request.overflowed) {
        return server_cache_destruction_verdict::unavailable;
    }

    bool saw_applicable = false;
    bool saw_soft = false;
    bool saw_hard = false;
    for (size_t i = 0; i < request.n_targets; ++i) {
        const auto & target = request.targets[i];
        if (target.kind >= server_cache_destruction_target_kind::_count) {
            return server_cache_destruction_verdict::unavailable;
        }
        if (!target_is_lease_applicable(target.kind)) {
            continue;
        }
        saw_applicable = true;

        if (target.kind == server_cache_destruction_target_kind::checkpoint_ring) {
            if (!admit_checkpoint_ring(target, saw_soft, saw_hard)) {
                return server_cache_destruction_verdict::unavailable;
            }
        } else if (!admit_scalar_artifact(target, saw_soft, saw_hard)) {
            return server_cache_destruction_verdict::unavailable;
        }
    }
    if (!saw_applicable) {
        return server_cache_destruction_verdict::admit_unleased;
    }
    if (saw_hard) {
        return server_cache_destruction_verdict::would_refuse_hard_leased;
    }
    if (saw_soft) {
        return server_cache_destruction_verdict::admit_soft_leased;
    }
    return server_cache_destruction_verdict::admit_unleased;
}

server_cache_lease_event_snapshot
server_cache_lease_table::event_snapshot() const noexcept {
    server_cache_lease_event_snapshot out;
    out.totals = event_totals;
    out.size = size_t(std::min<uint64_t>(n_events, events.size()));
    out.overflows = n_event_overflows;
    out.unavailable = !available;
    try {
        out.identities = identities;
    } catch (...) {
        out.unavailable = true;
        out.overflows++;
        return out;
    }
    if (out.size == 0) {
        return out;
    }
    const uint64_t first = n_events - out.size;
    for (size_t i = 0; i < out.size; ++i) {
        out.events[i] = events[size_t((first + i) % events.size())];
    }
    out.first_ordinal = out.events[0].ordinal;
    out.last_ordinal = out.events[out.size - 1].ordinal;
    return out;
}

server_cache_lease_retry_witness
server_cache_lease_table::retry_witness(
        bool include_next_expiry) const noexcept {
    server_cache_lease_retry_witness out;
    if (!available || clock == nullptr) {
        return out;
    }
    out.available = true;
    out.event_ordinal = next_event_ordinal > 0
        ? next_event_ordinal - 1 : 0;
    out.now_ns = clock->now_ns();
    if (!include_next_expiry) {
        return out;
    }
    for (const auto & lease : leases) {
        if (lease.explicit_hard || lease.subject_lost || lease.orphaned ||
            lease.expires_at_ns <= out.now_ns) {
            continue;
        }
        if (out.next_expiry_ns == 0 ||
            lease.expires_at_ns < out.next_expiry_ns) {
            out.next_expiry_ns = lease.expires_at_ns;
        }
    }
    return out;
}

bool server_cache_lease_table::replay(
        const server_cache_lease_event_snapshot & snapshot,
        server_cache_lease_replay_result & out) noexcept {
    out = {};
    if (!snapshot.replay_available()) {
        return false;
    }
    try {
        out.identities = snapshot.identities;
        for (const auto & identity : out.identities) {
            if (!identity.id || !identity.value.valid() ||
                std::count_if(
                    out.identities.begin(), out.identities.end(),
                    [&](const auto & item) {
                        return item.id == identity.id;
                    }) != 1) {
                return false;
            }
        }
        for (size_t i = 0; i < snapshot.size; ++i) {
            const auto & event = snapshot.events[i];
            if (event.ordinal != i + 1 ||
                event.kind >= server_cache_lease_event_kind::_count ||
                event.cls >= server_cache_lease_class::_count ||
                event.artifact_kind >=
                    common_retention_artifact_kind::_count ||
                ((event.kind !=
                      server_cache_lease_event_kind::mark_identity_unavailable &&
                  event.kind !=
                      server_cache_lease_event_kind::clear_identity_unavailable) &&
                 (!event.identity ||
                  std::none_of(
                      out.identities.begin(), out.identities.end(),
                      [&](const auto & item) {
                          return item.id == event.identity;
                      })))) {
                return false;
            }
            const auto active = std::find_if(
                out.active.begin(), out.active.end(),
                [&](const auto & value) {
                    return value.lease == event.lease;
                });
            switch (event.kind) {
                case server_cache_lease_event_kind::grant_soft:
                case server_cache_lease_event_kind::grant_hard:
                case server_cache_lease_event_kind::clone:
                    if (!event.lease || active != out.active.end()) {
                        return false;
                    }
                    out.active.push_back(event);
                    break;
                case server_cache_lease_event_kind::renew:
                case server_cache_lease_event_kind::orphan_hard:
                    if (active == out.active.end()) {
                        return false;
                    }
                    *active = event;
                    break;
                case server_cache_lease_event_kind::expire:
                case server_cache_lease_event_kind::release:
                case server_cache_lease_event_kind::invalidate_identity:
                    if (active == out.active.end()) {
                        return false;
                    }
                    out.active.erase(active);
                    break;
                case server_cache_lease_event_kind::mark_identity_unavailable:
                    if (event.lease || event.artifact.v == 0 ||
                        std::any_of(
                            out.identity_unavailable.begin(),
                            out.identity_unavailable.end(),
                            [&](const auto & value) {
                                return value.artifact == event.artifact;
                            })) {
                        return false;
                    }
                    out.identity_unavailable.push_back({
                        event.artifact, event.artifact_kind, event.owner_slot,
                    });
                    break;
                case server_cache_lease_event_kind::clear_identity_unavailable:
                    if (event.lease) {
                        return false;
                    }
                    {
                        const auto marker = std::find_if(
                            out.identity_unavailable.begin(),
                            out.identity_unavailable.end(),
                            [&](const auto & value) {
                                return value.artifact == event.artifact;
                            });
                        if (marker == out.identity_unavailable.end()) {
                            return false;
                        }
                        out.identity_unavailable.erase(marker);
                    }
                    break;
                case server_cache_lease_event_kind::refuse_hard_unavailable:
                case server_cache_lease_event_kind::refuse_hard_invalid:
                    if (event.lease) {
                        return false;
                    }
                    break;
                case server_cache_lease_event_kind::_count:
                    return false;
            }
            out.last_ordinal = event.ordinal;
        }
        out.state = server_cache_lease_eval_state::known;
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

server_cache_destruction_verdict server_cache_lease_evaluate_request(
        void * context,
        const server_cache_destruction_request & request) noexcept {
    auto * table = static_cast<server_cache_lease_table *>(context);
    return table
        ? table->admit(request)
        : server_cache_destruction_verdict::unavailable;
}
