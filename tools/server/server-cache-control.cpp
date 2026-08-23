#include "server-cache-control.h"

#include "server-cache-vbr-proof.h"
#include "server-vbr-artifact-store.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <random>
#include <thread>

namespace {

class steady_control_clock final : public server_cache_lease_clock {
public:
    uint64_t now_ns() noexcept override {
        using namespace std::chrono;
        const auto value = duration_cast<nanoseconds>(
            steady_clock::now().time_since_epoch()).count();
        return value > 0 ? uint64_t(value) : 0;
    }
};

class random_control_tokens final : public server_cache_control_token_source {
public:
    bool next(server_cache_control_token & out) noexcept override {
        out = {};
        try {
            std::random_device random;
            uint64_t high = (uint64_t(random()) << 32) ^ random();
            uint64_t low = (uint64_t(random()) << 32) ^ random();
            high ^= uint64_t(reinterpret_cast<uintptr_t>(this));
            low ^= uint64_t(std::chrono::steady_clock::now()
                                .time_since_epoch().count());
            if (high == 0 || low == 0) {
                return false;
            }
            out = { high, low };
            return true;
        } catch (...) {
            return false;
        }
    }
};

steady_control_clock default_clock;
random_control_tokens default_tokens;

server_cache_control_status resolve_vbr_from_store(
        void * context,
        const server_cache_control_selector & selector,
        server_cache_lease_subject & subject,
        server_cache_lease_identity & identity,
        server_cache_lease_frontier & frontier,
        server_cache_durable_fallback_proof & pin) noexcept {
    auto * artifacts = static_cast<server_vbr_artifact_store *>(context);
    if (!artifacts) {
        return server_cache_control_status::fallback_unavailable;
    }
    vbr_artifact_package_view package;
    if (!artifacts->resolve_control_reference(
            selector.reference, selector.tenant_key, package)) {
        return server_cache_control_status::not_found;
    }
    const auto & source = package.manifest().identity;
    if (source.token_count < 0) {
        return server_cache_control_status::identity_unavailable;
    }
    subject = {
        package.reference_artifact(),
        common_retention_artifact_kind::host_entry,
        -1,
    };
    identity = {
        source.execution_identity,
        source.adapter_config_identity,
        source.media_content_identity,
    };
    frontier = {
        source.sequence_epoch,
        uint64_t(source.token_count),
        source.next_position,
    };
    pin = server_cache_vbr_fallback_proof(std::move(package));
    return subject.valid() && identity.valid() && frontier.valid() &&
            pin.available()
        ? server_cache_control_status::ok
        : server_cache_control_status::identity_unavailable;
}

uint64_t token_digest(
        server_cache_control_token token, uint64_t secret) noexcept {
    uint64_t value = token.high ^ secret ^
        (token.low + 0x9e3779b97f4a7c15ULL +
         (token.high << 6) + (token.high >> 2));
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}

bool frontier_extends(
        const server_cache_lease_frontier & current,
        const server_cache_lease_frontier & proven) noexcept {
    return current.sequence_epoch == proven.sequence_epoch &&
           (current.token_count > proven.token_count ||
            current.next_position > proven.next_position);
}

} // namespace

server_cache_control_status server_cache_control_task_precheck(
        bool request_present,
        bool lifecycle_available,
        bool substrate_available) noexcept {
    if (!request_present) {
        return server_cache_control_status::invalid_request;
    }
    if (!lifecycle_available || !substrate_available) {
        return server_cache_control_status::not_supported;
    }
    return server_cache_control_status::ok;
}

struct server_cache_control_authority::impl {
    enum class holder_state : uint8_t { active, orphaned_hard, closed };

    struct lease_record {
        server_cache_control_token handle;
        uint64_t handle_digest = 0;
        server_cache_lease_id table_lease;
        server_cache_lease_class cls = server_cache_lease_class::none;
        server_cache_control_selector subject;
        server_cache_control_selector fallback;
        common_cache_family_binding cache_family;
        server_cache_lease_frontier lease_frontier;
        server_cache_lease_frontier proven_frontier;
        uint64_t expires_at_ns = 0;
        bool protected_bytes_known = false;
        bool fallback_pinned_bytes_known = false;
        uint64_t protected_bytes = 0;
        uint64_t fallback_pinned_bytes = 0;
        bool shared_fallback = false;
        bool orphaned = false;
        bool subject_lost = false;
        bool released = false;
        server_cache_durable_fallback_proof subject_pin;
    };

    struct family_record {
        server_cache_control_token handle;
        uint64_t handle_digest = 0;
        common_cache_family_id id;
        std::string label;
    };

    struct family_binding_record {
        uint64_t handle_digest = 0;
        common_cache_family_binding binding;
    };

    struct holder_record {
        server_cache_lease_owner_id id;
        uint64_t session_digest = 0;
        uint64_t recovery_digest = 0;
        uint64_t expires_at_ns = 0;
        holder_state state = holder_state::active;
        std::vector<family_record> families;
        std::vector<family_binding_record> family_bindings;
        std::vector<lease_record> leases;
        uint64_t next_event_ordinal = 1;
        uint64_t last_dropped_event_ordinal = 0;
    };

    struct event_record {
        server_cache_control_event_view view;
        server_cache_lease_owner_id holder;
    };
    struct replay_record {
        server_cache_control_operation operation =
            server_cache_control_operation::holder_create;
        uint64_t key = 0;
        uint64_t holder_digest = 0;
        server_cache_control_result result;
    };

    server_cache_lease_table * table = nullptr;
    server_retention_sidecar_store * retention = nullptr;
    server_cache_lease_clock * clock = nullptr;
    server_cache_control_token_source * tokens = nullptr;
    void * refresh_context = nullptr;
    server_cache_control_config::refresh_subject_fn refresh_subject = nullptr;
    void * resolve_vbr_context = nullptr;
    server_cache_control_config::resolve_vbr_fn resolve_vbr = nullptr;
    void * host_proof_context = nullptr;
    server_cache_control_config::acquire_host_proof_fn acquire_host_proof = nullptr;
    void * selector_evidence_context = nullptr;
    server_cache_control_config::selector_evidence_fn selector_evidence = nullptr;
    size_t max_holders = 0;
    size_t max_leases = 0;
    size_t max_families = 0;
    size_t max_family_bindings = 0;
    size_t test_fail_note_after = std::numeric_limits<size_t>::max();
    size_t note_attempts = 0;
    bool test_fail_remember = false;
    uint64_t next_holder_id = 1;
    uint64_t next_family_id = 1;
    std::vector<holder_record> holders;
    std::vector<event_record> events;
    std::vector<replay_record> replays;

    server_cache_durable_fallback_proof pending_proof;
    server_cache_lease_subject pending_subject;
    server_cache_lease_identity pending_identity;
    bool pending = false;
    bool core_available = false;
    bool grant_path_available = true;
    std::thread::id owner_thread;
    uint64_t token_secret = 0;

    void assert_owner() noexcept {
        const auto current = std::this_thread::get_id();
        if (owner_thread == std::thread::id{}) {
            owner_thread = current;
        }
        GGML_ASSERT(owner_thread == current &&
                    "cache control authority must run on scheduler thread");
    }

    uint64_t now() noexcept { return clock->now_ns(); }
    uint64_t digest(server_cache_control_token token) const noexcept {
        return token_digest(token, token_secret);
    }

    bool deadline(uint64_t ttl, uint64_t & out) noexcept {
        const uint64_t current = now();
        if (current == 0 || ttl == 0 ||
            ttl > std::numeric_limits<uint64_t>::max() - current) {
            return false;
        }
        out = current + ttl;
        return true;
    }

    void note(server_cache_control_operation operation,
              server_cache_control_status status,
              server_cache_lease_class cls,
              server_cache_lease_owner_id holder,
              const server_cache_control_request & request,
              server_cache_control_token lease) noexcept {
        if (note_attempts++ >= test_fail_note_after) {
            grant_path_available = false;
            return;
        }
        try {
            if (events.size() == SERVER_CACHE_LEASE_EVENT_RING) {
                const auto dropped = events.front();
                const auto owner = std::find_if(
                    holders.begin(), holders.end(), [&](const holder_record & value) {
                        return value.id == dropped.holder;
                    });
                if (owner != holders.end()) {
                    owner->last_dropped_event_ordinal = dropped.view.ordinal;
                }
                events.erase(events.begin());
            }
            auto owner = std::find_if(
                holders.begin(), holders.end(), [&](const holder_record & value) {
                    return value.id == holder;
                });
            if (owner == holders.end()) {
                return;
            }
            server_cache_control_event_kind kind =
                server_cache_control_event_kind::refuse;
            if (status == server_cache_control_status::ok ||
                status == server_cache_control_status::already_released) {
                if (operation == server_cache_control_operation::lease_renew) {
                    kind = server_cache_control_event_kind::renew;
                } else if (operation == server_cache_control_operation::lease_release ||
                           operation == server_cache_control_operation::holder_close) {
                    kind = server_cache_control_event_kind::release;
                } else {
                    kind = server_cache_control_event_kind::grant;
                }
            }
            const bool lease_operation =
                operation == server_cache_control_operation::lease_acquire ||
                operation == server_cache_control_operation::lease_renew ||
                operation == server_cache_control_operation::lease_release;
            auto subject_kind = lease_operation
                ? request.subject.kind
                : server_cache_control_subject_kind::_count;
            auto family_role = request.family_role;
            if (lease_operation && lease) {
                const uint64_t lease_digest = digest(lease);
                const auto record = std::find_if(
                    owner->leases.begin(), owner->leases.end(),
                    [lease_digest](const lease_record & value) {
                        return value.handle_digest == lease_digest;
                    });
                if (record != owner->leases.end()) {
                    subject_kind = record->subject.kind;
                    family_role = record->cache_family.declared()
                        ? record->cache_family.role
                        : common_cache_family_role::_count;
                }
            }
            events.push_back({ {
                owner->next_event_ordinal++, now() / 1000000ULL, kind,
                status, cls, subject_kind, family_role, lease,
            }, holder });
        } catch (...) {
            grant_path_available = false;
        }
    }

    void note_expire(holder_record & holder, const lease_record & lease) noexcept {
        server_cache_control_request request;
        request.subject = lease.subject;
        request.family_role = lease.cache_family.declared()
            ? lease.cache_family.role : common_cache_family_role::_count;
        note(server_cache_control_operation::lease_release,
             server_cache_control_status::lease_expired, lease.cls,
             holder.id, request, lease.handle);
        if (!events.empty() && events.back().holder == holder.id) {
            events.back().view.kind = server_cache_control_event_kind::expire;
        }
    }

    void scrub_replays(const holder_record & holder) noexcept {
        replays.erase(std::remove_if(
            replays.begin(), replays.end(), [&](const replay_record & replay) {
                return replay.holder_digest == holder.id.v ||
                    (replay.result.holder &&
                     digest(replay.result.holder) == holder.session_digest) ||
                    (replay.result.holder_recovery &&
                     digest(replay.result.holder_recovery) == holder.recovery_digest);
            }), replays.end());
    }

    size_t lease_count() const noexcept {
        size_t count = 0;
        for (const auto & holder : holders) {
            count += size_t(std::count_if(
                holder.leases.begin(), holder.leases.end(),
                [](const lease_record & lease) { return !lease.released; }));
        }
        return count;
    }

    size_t family_count() const noexcept {
        size_t count = 0;
        for (const auto & holder : holders) {
            count += holder.families.size();
        }
        return count;
    }

    size_t family_binding_count() const noexcept {
        size_t count = 0;
        for (const auto & holder : holders) {
            count += holder.family_bindings.size();
        }
        return count;
    }

    bool issue_token(server_cache_control_token & out) noexcept {
        for (int attempt = 0; attempt < 4; ++attempt) {
            if (!tokens->next(out) || !out) {
                return false;
            }
            const uint64_t digest = this->digest(out);
            const bool exists = std::any_of(
                holders.begin(), holders.end(), [&](const holder_record & holder) {
                    if (holder.session_digest == digest ||
                        holder.recovery_digest == digest) {
                        return true;
                    }
                    if (std::any_of(
                            holder.families.begin(), holder.families.end(),
                            [digest](const family_record & family) {
                                return family.handle_digest == digest;
                            }) ||
                        std::any_of(
                            holder.family_bindings.begin(),
                            holder.family_bindings.end(),
                            [digest](const family_binding_record & binding) {
                                return binding.handle_digest == digest;
                            })) {
                        return true;
                    }
                    return std::any_of(
                        holder.leases.begin(), holder.leases.end(),
                        [digest](const lease_record & lease) {
                            return lease.handle_digest == digest;
                        });
                });
            if (!exists) {
                return true;
            }
        }
        out = {};
        return false;
    }

    const server_cache_control_result * replay(
            server_cache_control_operation operation,
            uint64_t key,
            uint64_t holder_digest) const noexcept {
        if (key == 0) {
            return nullptr;
        }
        const auto found = std::find_if(
            replays.begin(), replays.end(), [&](const replay_record & value) {
                return value.operation == operation && value.key == key &&
                       value.holder_digest == holder_digest;
            });
        return found == replays.end() ? nullptr : &found->result;
    }

    bool remember(
            server_cache_control_operation operation,
            uint64_t key,
            uint64_t holder_digest,
            const server_cache_control_result & result) noexcept {
        if (key == 0) {
            return true;
        }
        if (test_fail_remember) {
            return false;
        }
        try {
            if (replays.size() == SERVER_CACHE_LEASE_EVENT_RING) {
                replays.erase(replays.begin());
            }
            replays.push_back({ operation, key, holder_digest, result });
            return true;
        } catch (...) {
            return false;
        }
    }

    holder_record * authenticate(server_cache_control_token token) noexcept {
        if (!token) {
            return nullptr;
        }
        const uint64_t digest = this->digest(token);
        const auto found = std::find_if(holders.begin(), holders.end(),
            [digest](const holder_record & holder) {
                return holder.state != holder_state::closed &&
                       holder.session_digest == digest;
            });
        return found == holders.end() ? nullptr : &*found;
    }

    lease_record * find_lease(
            holder_record & holder,
            server_cache_control_token token) noexcept {
        if (!token) {
            return nullptr;
        }
        const uint64_t digest = this->digest(token);
        const auto found = std::find_if(holder.leases.begin(), holder.leases.end(),
            [digest](const lease_record & lease) {
                return lease.handle_digest == digest;
            });
        return found == holder.leases.end() ? nullptr : &*found;
    }

    family_record * find_family(
            holder_record & holder,
            server_cache_control_token token) noexcept {
        if (!token) {
            return nullptr;
        }
        const uint64_t digest = this->digest(token);
        const auto found = std::find_if(
            holder.families.begin(), holder.families.end(),
            [digest](const family_record & family) {
                return family.handle_digest == digest;
            });
        return found == holder.families.end() ? nullptr : &*found;
    }

    const std::string * family_label(
            const holder_record & holder,
            common_cache_family_id id) const noexcept {
        const auto found = std::find_if(
            holder.families.begin(), holder.families.end(),
            [id](const family_record & family) { return family.id == id; });
        return found == holder.families.end() ? nullptr : &found->label;
    }

    class staged_proof {
    public:
        staged_proof(
                impl & owner,
                const server_cache_lease_subject & subject,
                const server_cache_lease_identity & identity,
                server_cache_durable_fallback_proof && proof) :
            owner_(owner) {
            owner_.pending_subject = subject;
            owner_.pending_identity = identity;
            owner_.pending_proof = std::move(proof);
            owner_.pending = true;
        }
        ~staged_proof() {
            owner_.pending = false;
            owner_.pending_proof = {};
        }
        staged_proof(const staged_proof &) = delete;
        staged_proof & operator=(const staged_proof &) = delete;

    private:
        impl & owner_;
    };

    bool release_record(
            holder_record & holder, lease_record & lease) noexcept {
        if (lease.released) {
            return false;
        }
        bool released = true;
        if (lease.cls == server_cache_lease_class::hard) {
            if (table->owned_scope_active(
                    { lease.handle_digest }, holder.id)) {
                released = table->release_owned_scope(
                    { lease.handle_digest }, holder.id);
            }
        } else if (table->lease_active(lease.table_lease)) {
            released = table->release(lease.table_lease);
        }
        if (!released) {
            return false;
        }
        lease.released = true;
        lease.subject_lost = false;
        lease.subject_pin = {};
        return true;
    }

    bool refresh_subject_lost(lease_record & lease) noexcept {
        if (lease.released || lease.cls != server_cache_lease_class::hard) {
            return false;
        }
        if (lease.subject_lost || table->lease_subject_lost(lease.table_lease) ||
            !table->lease_active(lease.table_lease)) {
            lease.subject_lost = true;
            lease.subject_pin = {};
            return true;
        }
        return false;
    }

    server_cache_control_status resolve_subject(
            const server_cache_control_selector & selector,
            server_cache_lease_subject & subject,
            server_cache_lease_identity & identity,
            server_cache_lease_frontier & frontier,
            server_cache_durable_fallback_proof * subject_pin = nullptr) noexcept {
        subject = {};
        identity = {};
        frontier = {};
        if (selector.kind == server_cache_control_subject_kind::live_checkpoint) {
            return server_cache_control_status::not_supported;
        }
        if (selector.kind == server_cache_control_subject_kind::vbr_reference) {
            if (!resolve_vbr) {
                return server_cache_control_status::fallback_unavailable;
            }
            server_cache_durable_fallback_proof pin;
            const auto status = resolve_vbr(
                resolve_vbr_context, selector, subject, identity,
                frontier, pin);
            if (status == server_cache_control_status::ok && subject_pin) {
                *subject_pin = std::move(pin);
                if (!subject_pin->available()) {
                    return server_cache_control_status::identity_unavailable;
                }
            }
            return status;
        }
        server_cache_lease_identity refreshed_identity = selector.identity;
        server_cache_lease_frontier refreshed_frontier = selector.frontier;
        if (refresh_subject && !refresh_subject(
                refresh_context, selector, refreshed_identity,
                refreshed_frontier)) {
            return server_cache_control_status::stale_capability;
        }
        if (!retention || !refreshed_identity.valid() ||
            !refreshed_frontier.valid()) {
            return server_cache_control_status::identity_unavailable;
        }
        if (selector.kind == server_cache_control_subject_kind::live_prefix &&
            selector.retention_key.kind !=
                common_retention_artifact_kind::live_slot) {
            return server_cache_control_status::not_found;
        }
        if (selector.kind == server_cache_control_subject_kind::host_snapshot &&
            selector.retention_key.kind !=
                common_retention_artifact_kind::host_entry) {
            return server_cache_control_status::not_found;
        }
        server_retention_candidate candidate;
        if (!retention->candidate_for_instance(selector.retention_key, candidate)) {
            return server_cache_control_status::not_found;
        }
        if (candidate.avail != server_retention_candidate_availability::available) {
            return server_cache_control_status::subject_busy;
        }
        subject = { candidate.artifact_id, candidate.record.kind,
                    selector.retention_key.owner_slot };
        identity = std::move(refreshed_identity);
        frontier = refreshed_frontier;
        return server_cache_control_status::ok;
    }

    server_cache_control_status resolve_fallback(
            const server_cache_control_selector & selector,
            llama_cache_acct_artifact_id subject_artifact,
            const server_cache_lease_identity & subject_identity,
            const server_cache_lease_frontier & subject_frontier,
            server_cache_durable_fallback_proof & proof) noexcept {
        proof = {};
        server_cache_lease_subject fallback_subject;
        server_cache_lease_identity fallback_identity;
        server_cache_lease_frontier fallback_frontier;
        if (selector.kind == server_cache_control_subject_kind::vbr_reference) {
            if (!resolve_vbr) {
                return server_cache_control_status::fallback_unavailable;
            }
            const auto status = resolve_vbr(
                resolve_vbr_context, selector, fallback_subject,
                fallback_identity, fallback_frontier, proof);
            if (status != server_cache_control_status::ok) {
                return status;
            }
        } else if (selector.kind ==
                server_cache_control_subject_kind::host_snapshot) {
            const auto status = resolve_subject(
                selector, fallback_subject, fallback_identity,
                fallback_frontier);
            if (status != server_cache_control_status::ok) {
                return status == server_cache_control_status::identity_unavailable
                    ? status
                    : server_cache_control_status::fallback_unavailable;
            }
            if (!acquire_host_proof) {
                return server_cache_control_status::fallback_unavailable;
            }
            proof = acquire_host_proof(host_proof_context, selector);
        } else if (selector.kind ==
                server_cache_control_subject_kind::live_checkpoint) {
            return server_cache_control_status::not_supported;
        } else {
            return server_cache_control_status::fallback_invalid;
        }
        if (fallback_subject.artifact == subject_artifact ||
            fallback_identity != subject_identity ||
            !(fallback_frontier == subject_frontier)) {
            proof = {};
            return server_cache_control_status::fallback_invalid;
        }
        if (proof.state() == server_cache_lease_fallback_state::invalid) {
            return server_cache_control_status::fallback_invalid;
        }
        return proof.available() ? server_cache_control_status::ok
                                 : server_cache_control_status::fallback_unavailable;
    }
};

server_cache_control_authority::server_cache_control_authority(
        const server_cache_control_config & config) noexcept {
    try {
        state_ = std::make_unique<impl>();
        state_->table = config.leases;
        state_->retention = config.retention;
        state_->clock = config.clock ? config.clock : &default_clock;
        state_->tokens = config.tokens ? config.tokens : &default_tokens;
        state_->refresh_context = config.refresh_context;
        state_->refresh_subject = config.refresh_subject;
        state_->resolve_vbr_context = config.resolve_vbr
            ? config.resolve_vbr_context
            : config.artifacts;
        state_->resolve_vbr = config.resolve_vbr
            ? config.resolve_vbr
            : resolve_vbr_from_store;
        state_->host_proof_context = config.host_proof_context;
        state_->acquire_host_proof = config.acquire_host_proof;
        state_->selector_evidence_context = config.selector_evidence_context;
        state_->selector_evidence = config.selector_evidence;
        state_->max_holders = config.max_holders;
        state_->max_leases = config.max_leases;
        state_->max_families = config.max_families;
        state_->max_family_bindings = config.max_family_bindings;
        state_->test_fail_note_after = config.test_fail_note_after;
        state_->test_fail_remember = config.test_fail_remember;
        server_cache_control_token secret;
        if (state_->tokens->next(secret)) {
            state_->token_secret = secret.high ^
                ((secret.low << 17) | (secret.low >> 47));
        }
        state_->core_available = state_->table && state_->retention &&
                     state_->max_holders > 0 && state_->max_leases > 0 &&
                     state_->max_families > 0 &&
                     state_->max_family_bindings > 0 &&
                     state_->token_secret != 0;
        if (state_->core_available) {
            state_->table->bind_fallback_provider(this);
        }
    } catch (...) {
        state_.reset();
    }
}

server_cache_control_authority::~server_cache_control_authority() {
    if (state_ && state_->table) {
        for (auto & holder : state_->holders) {
            for (auto & lease : holder.leases) {
                if (!lease.released) {
                    if (lease.cls == server_cache_lease_class::hard) {
                        (void) state_->table->release_owned_scope(
                            { lease.handle_digest }, holder.id);
                    } else {
                        (void) state_->table->release(lease.table_lease);
                    }
                }
            }
        }
        state_->table->bind_fallback_provider(nullptr);
    }
}

bool server_cache_control_authority::available() const noexcept {
    return state_ && state_->core_available;
}

server_cache_control_status
server_cache_control_authority::resolve_family_binding(
        server_cache_control_token token,
        common_cache_family_binding & out) noexcept {
    out = {};
    if (!available() || !token) {
        return server_cache_control_status::not_found;
    }
    state_->assert_owner();
    lifecycle_point();
    const uint64_t digest = state_->digest(token);
    for (const auto & holder : state_->holders) {
        if (holder.state != impl::holder_state::active) {
            continue;
        }
        const auto found = std::find_if(
            holder.family_bindings.begin(), holder.family_bindings.end(),
            [digest](const impl::family_binding_record & binding) {
                return binding.handle_digest == digest;
            });
        if (found != holder.family_bindings.end()) {
            out = found->binding;
            return server_cache_control_status::ok;
        }
    }
    return server_cache_control_status::not_found;
}

server_cache_durable_fallback_proof
server_cache_control_authority::acquire(
        const server_cache_lease_subject & subject,
        const server_cache_lease_identity & identity) noexcept {
    if (!state_ || !state_->pending ||
        state_->pending_subject.artifact != subject.artifact ||
        state_->pending_identity != identity) {
        return {};
    }
    state_->pending = false;
    return std::move(state_->pending_proof);
}

void server_cache_control_authority::lifecycle_point() noexcept {
    if (!available()) {
        return;
    }
    state_->assert_owner();
    const uint64_t now = state_->now();
    for (auto & holder : state_->holders) {
        for (auto & lease : holder.leases) {
            if (!lease.released && !lease.orphaned &&
                lease.cls == server_cache_lease_class::hard &&
                lease.expires_at_ns <= now) {
                lease.orphaned = true;
                (void) state_->table->orphan_owned_scope(
                    { lease.handle_digest }, holder.id);
                state_->note_expire(holder, lease);
            }
        }
        if (holder.state != impl::holder_state::active ||
            holder.expires_at_ns > now) {
            continue;
        }
        bool hard = false;
        for (auto & lease : holder.leases) {
            if (lease.released) {
                continue;
            }
            if (lease.cls == server_cache_lease_class::hard) {
                lease.orphaned = true;
                hard = true;
                state_->note_expire(holder, lease);
            } else {
                if (state_->release_record(holder, lease)) {
                    state_->note_expire(holder, lease);
                }
            }
        }
        if (hard) {
            (void) state_->table->orphan_owner(holder.id);
            holder.state = impl::holder_state::orphaned_hard;
        } else {
            holder.state = impl::holder_state::closed;
        }
        state_->scrub_replays(holder);
        holder.session_digest = 0;
    }
    state_->table->lifecycle_point();
}

server_cache_control_result server_cache_control_authority::execute(
        server_cache_control_operation operation,
        const server_cache_control_request & request) noexcept {
    server_cache_control_result out;
    if (!available() || operation >= server_cache_control_operation::_count) {
        out.status = server_cache_control_status::not_supported;
        return out;
    }
    state_->assert_owner();
    lifecycle_point();
    server_cache_lease_owner_id event_owner;
    const auto finish = [&](server_cache_control_status status,
                            server_cache_lease_class cls =
                                server_cache_lease_class::none) {
        out.status = status;
        if (operation != server_cache_control_operation::lease_inspect &&
            operation != server_cache_control_operation::events &&
            event_owner) {
            const auto lease = out.lease ? out.lease : request.lease;
            state_->note(operation, status, cls, event_owner, request, lease);
        }
        return out;
    };
    try {
        if (operation == server_cache_control_operation::holder_create) {
            if (!state_->grant_path_available) {
                return finish(server_cache_control_status::internal_fault);
            }
            if (const auto * replay = state_->replay(
                    operation, request.idempotency_key, 0)) {
                return *replay;
            }
            state_->holders.erase(
                std::remove_if(
                    state_->holders.begin(), state_->holders.end(),
                    [](const impl::holder_record & holder) {
                        return holder.state == impl::holder_state::closed;
                    }),
                state_->holders.end());
            if (state_->holders.size() >= state_->max_holders) {
                return finish(server_cache_control_status::capacity_refused);
            }
            uint64_t deadline = 0;
            server_cache_control_token session;
            server_cache_control_token recovery;
            if (!state_->deadline(request.ttl_ns, deadline) ||
                !state_->issue_token(session) ||
                !state_->issue_token(recovery) ||
                session == recovery ||
                state_->next_holder_id == 0) {
                return finish(server_cache_control_status::internal_fault);
            }
            impl::holder_record holder;
            holder.id = { state_->next_holder_id++ };
            event_owner = holder.id;
            holder.session_digest = state_->digest(session);
            holder.recovery_digest = state_->digest(recovery);
            holder.expires_at_ns = deadline;
            state_->holders.push_back(std::move(holder));
            out.holder = session;
            out.holder_recovery = recovery;
            out.expires_at_ns = deadline;
            out.max_leases = uint32_t(std::min(
                state_->max_leases,
                size_t(std::numeric_limits<uint32_t>::max())));
            out.status = server_cache_control_status::ok;
            if (!state_->remember(
                    operation, request.idempotency_key, 0, out)) {
                state_->grant_path_available = false;
            }
            return finish(server_cache_control_status::ok);
        }
        if (operation == server_cache_control_operation::holder_reattach) {
            if (!request.recovery) {
                return finish(server_cache_control_status::not_found);
            }
            const uint64_t digest = state_->digest(request.recovery);
            auto holder = std::find_if(state_->holders.begin(), state_->holders.end(),
                [digest](const auto & item) {
                    return item.state == impl::holder_state::orphaned_hard &&
                           item.recovery_digest == digest;
                });
            uint64_t deadline = 0;
            server_cache_control_token session;
            if (holder == state_->holders.end() ||
                !state_->deadline(request.ttl_ns, deadline) ||
                !state_->issue_token(session)) {
                return finish(server_cache_control_status::not_found);
            }
            holder->session_digest = state_->digest(session);
            holder->expires_at_ns = deadline;
            holder->state = impl::holder_state::active;
            event_owner = holder->id;
            out.holder = session;
            out.expires_at_ns = deadline;
            for (const auto & lease : holder->leases) {
                if (!lease.released && lease.orphaned && lease.handle) {
                    out.orphaned_leases.push_back({
                        lease.handle, lease.subject.kind, lease.proven_frontier,
                    });
                }
            }
            for (const auto & family : holder->families) {
                out.families.push_back({ family.handle, family.label });
            }
            return finish(server_cache_control_status::ok);
        }

        auto * holder = state_->authenticate(request.holder);
        if (!holder) {
            return finish(server_cache_control_status::not_found);
        }
        event_owner = holder->id;
        if (operation == server_cache_control_operation::family_register ||
            operation == server_cache_control_operation::family_bind ||
            operation == server_cache_control_operation::lease_acquire) {
            if (const auto * replay = state_->replay(
                    operation, request.idempotency_key,
                    holder->id.v)) {
                return *replay;
            }
        }
        if (operation == server_cache_control_operation::holder_close) {
            for (auto & lease : holder->leases) {
                if (lease.released) {
                    continue;
                }
                if (!state_->release_record(*holder, lease)) {
                    return finish(server_cache_control_status::internal_fault);
                }
            }
            holder->state = impl::holder_state::closed;
            state_->scrub_replays(*holder);
            holder->session_digest = 0;
            return finish(server_cache_control_status::ok);
        }
        if (operation == server_cache_control_operation::events) {
            out.events_overflowed = holder->last_dropped_event_ordinal != 0 &&
                request.after_ordinal <= holder->last_dropped_event_ordinal;
            for (const auto & event : state_->events) {
                if (event.holder == holder->id &&
                    event.view.ordinal > request.after_ordinal &&
                    out.events.size() < request.event_limit) {
                    out.events.push_back(event.view);
                }
            }
            return finish(server_cache_control_status::ok);
        }

        if (operation == server_cache_control_operation::family_register) {
            if (!state_->grant_path_available) {
                return finish(server_cache_control_status::internal_fault);
            }
            if (state_->family_count() >= state_->max_families) {
                return finish(server_cache_control_status::capacity_refused);
            }
            server_cache_control_token handle;
            if (state_->next_family_id == 0 ||
                !state_->issue_token(handle)) {
                return finish(server_cache_control_status::internal_fault);
            }
            impl::family_record family;
            family.handle = handle;
            family.handle_digest = state_->digest(handle);
            family.id = { state_->next_family_id++ };
            family.label = request.family_label;
            holder->families.push_back(family);
            out.family = handle;
            out.families.push_back({ handle, request.family_label });
            out.status = server_cache_control_status::ok;
            if (!state_->remember(
                    operation, request.idempotency_key,
                    holder->id.v, out)) {
                state_->grant_path_available = false;
            }
            return finish(server_cache_control_status::ok);
        }

        if (operation == server_cache_control_operation::family_bind) {
            auto * family = state_->find_family(*holder, request.family);
            if (!family) {
                return finish(server_cache_control_status::not_found);
            }
            if (request.family_role >= common_cache_family_role::_count) {
                return finish(server_cache_control_status::invalid_request);
            }
            if (!state_->grant_path_available) {
                return finish(server_cache_control_status::internal_fault);
            }
            if (state_->family_binding_count() >=
                    state_->max_family_bindings) {
                return finish(server_cache_control_status::capacity_refused);
            }
            server_cache_control_token handle;
            if (!state_->issue_token(handle)) {
                return finish(server_cache_control_status::internal_fault);
            }
            impl::family_binding_record binding;
            binding.handle_digest = state_->digest(handle);
            binding.binding = { family->id, request.family_role };
            holder->family_bindings.push_back(binding);
            out.family_binding = handle;
            out.cache_family = binding.binding;
            out.status = server_cache_control_status::ok;
            if (!state_->remember(
                    operation, request.idempotency_key,
                    holder->id.v, out)) {
                state_->grant_path_available = false;
            }
            return finish(server_cache_control_status::ok);
        }

        if (operation == server_cache_control_operation::lease_acquire) {
            if (!state_->grant_path_available) {
                return finish(server_cache_control_status::internal_fault,
                              request.requested_class);
            }
            if (state_->lease_count() >= state_->max_leases ||
                (request.requested_class != server_cache_lease_class::soft &&
                 request.requested_class != server_cache_lease_class::hard)) {
                return finish(server_cache_control_status::capacity_refused,
                              request.requested_class);
            }
            common_cache_family_binding cache_family;
            if (request.family_binding) {
                const uint64_t digest = state_->digest(request.family_binding);
                const auto found = std::find_if(
                    holder->family_bindings.begin(), holder->family_bindings.end(),
                    [digest](const impl::family_binding_record & value) {
                        return value.handle_digest == digest;
                    });
                if (found == holder->family_bindings.end()) {
                    return finish(server_cache_control_status::not_found,
                                  request.requested_class);
                }
                cache_family = found->binding;
            }
            server_cache_lease_subject subject;
            server_cache_lease_identity identity;
            server_cache_lease_frontier frontier;
            server_cache_durable_fallback_proof subject_pin;
            auto status = state_->resolve_subject(
                request.subject, subject, identity, frontier,
                request.requested_class == server_cache_lease_class::hard
                    ? &subject_pin
                    : nullptr);
            if (status != server_cache_control_status::ok) {
                return finish(status, request.requested_class);
            }
            uint64_t deadline = 0;
            server_cache_control_token handle;
            if (!state_->deadline(request.ttl_ns, deadline) ||
                !state_->issue_token(handle)) {
                return finish(server_cache_control_status::internal_fault,
                              request.requested_class);
            }
            server_cache_lease_class granted_class = request.requested_class;
            server_cache_lease_id table_lease;
            if (granted_class == server_cache_lease_class::hard) {
                server_cache_durable_fallback_proof proof;
                status = state_->resolve_fallback(
                    request.fallback, subject.artifact, identity, frontier,
                    proof);
                if (status != server_cache_control_status::ok) {
                    if (!request.allow_soft_fallback) {
                        return finish(status, request.requested_class);
                    }
                    subject_pin = {};
                    granted_class = server_cache_lease_class::soft;
                }
                if (granted_class == server_cache_lease_class::hard) {
                    impl::staged_proof staged(
                        *state_, subject, identity, std::move(proof));
                    table_lease = state_->table->grant_hard_owned(
                        subject,
                        server_cache_lease_scope::from(
                            server_cache_explicit_lease_scope_id {
                                state_->digest(handle) }),
                        identity, holder->id, frontier, request.ttl_ns);
                    if (!table_lease) {
                        return finish(server_cache_control_status::hard_lease_blocked,
                                      request.requested_class);
                    }
                }
            }
            if (granted_class == server_cache_lease_class::soft) {
                table_lease = state_->table->grant_soft(
                    subject,
                    server_cache_lease_scope::from(
                        server_cache_explicit_lease_scope_id {
                            state_->digest(handle) }),
                    identity, request.ttl_ns);
                if (!table_lease) {
                    return finish(server_cache_control_status::capacity_refused,
                                  granted_class);
                }
            }
            impl::lease_record record;
            record.handle = handle;
            record.handle_digest = state_->digest(handle);
            record.table_lease = table_lease;
            record.cls = granted_class;
            record.subject = request.subject;
            record.fallback = granted_class == server_cache_lease_class::hard
                ? request.fallback : server_cache_control_selector{};
            if (granted_class != server_cache_lease_class::hard) {
                record.fallback.kind = server_cache_control_subject_kind::_count;
            }
            record.cache_family = cache_family;
            record.lease_frontier = frontier;
            record.proven_frontier = frontier;
            record.expires_at_ns = deadline;
            record.subject_pin = std::move(subject_pin);
            if (state_->selector_evidence) {
                record.protected_bytes_known = state_->selector_evidence(
                    state_->selector_evidence_context, request.subject,
                    record.protected_bytes, record.shared_fallback);
                bool fallback_shared = false;
                if (granted_class == server_cache_lease_class::hard) {
                    record.fallback_pinned_bytes_known =
                        state_->selector_evidence(
                            state_->selector_evidence_context,
                            request.fallback,
                            record.fallback_pinned_bytes, fallback_shared);
                    record.shared_fallback = fallback_shared;
                }
            }
            out.protected_bytes_known = record.protected_bytes_known;
            out.fallback_pinned_bytes_known =
                record.fallback_pinned_bytes_known;
            out.protected_bytes = record.protected_bytes;
            out.fallback_pinned_bytes = record.fallback_pinned_bytes;
            out.shared_fallback = record.shared_fallback;
            try {
                holder->leases.push_back(std::move(record));
            } catch (...) {
                if (granted_class == server_cache_lease_class::hard) {
                    (void) state_->table->release_owned_scope(
                        { state_->digest(handle) }, holder->id);
                } else {
                    (void) state_->table->release(table_lease);
                }
                return finish(server_cache_control_status::internal_fault,
                              granted_class);
            }
            out.lease = handle;
            out.granted_class = granted_class;
            out.subject_kind = request.subject.kind;
            out.cache_family = cache_family;
            if (cache_family.declared()) {
                if (const auto * label = state_->family_label(
                        *holder, cache_family.family)) {
                    out.family_label = *label;
                }
            }
            out.protection = server_cache_control_protection_state::current;
            out.lease_frontier = frontier;
            out.proven_frontier = frontier;
            out.expires_at_ns = deadline;
            out.fallback_kind = granted_class == server_cache_lease_class::hard
                ? request.fallback.kind
                : server_cache_control_subject_kind::_count;
            out.status = server_cache_control_status::ok;
            if (!state_->remember(
                    operation, request.idempotency_key,
                    holder->id.v, out)) {
                // The lease is already committed; retain it and return the
                // opaque handle rather than make response loss duplicate it.
                state_->grant_path_available = false;
            }
            return finish(server_cache_control_status::ok,
                          granted_class);
        }

        auto * lease = state_->find_lease(*holder, request.lease);
        if (!lease) {
            return finish(server_cache_control_status::not_found);
        }
        if (operation == server_cache_control_operation::lease_release) {
            if (lease->released) {
                return finish(server_cache_control_status::already_released,
                              lease->cls);
            }
            if (!state_->release_record(*holder, *lease)) {
                return finish(server_cache_control_status::internal_fault,
                              lease->cls);
            }
            return finish(server_cache_control_status::ok, lease->cls);
        }

        out.lease = request.lease;
        out.subject_kind = lease->subject.kind;
        out.fallback_kind = lease->fallback.kind;
        out.cache_family = lease->cache_family;
        if (lease->cache_family.declared()) {
            if (const auto * label = state_->family_label(
                    *holder, lease->cache_family.family)) {
                out.family_label = *label;
            }
        }
        out.protected_bytes_known = lease->protected_bytes_known;
        out.fallback_pinned_bytes_known = lease->fallback_pinned_bytes_known;
        out.protected_bytes = lease->protected_bytes;
        out.fallback_pinned_bytes = lease->fallback_pinned_bytes;
        out.shared_fallback = lease->shared_fallback;

        server_cache_lease_subject current_subject;
        server_cache_lease_identity current_identity;
        server_cache_lease_frontier current_frontier;
        server_cache_durable_fallback_proof refreshed_subject_pin;
        if (!state_->table->lease_active(lease->table_lease)) {
            if (lease->cls == server_cache_lease_class::soft) {
                lease->released = true;
                lease->subject_pin = {};
                return finish(server_cache_control_status::lease_expired,
                              lease->cls);
            }
            lease->subject_lost = true;
            lease->subject_pin = {};
        } else if (state_->refresh_subject_lost(*lease)) {
            lease->subject_pin = {};
        }
        if (lease->subject_lost) {
            out.granted_class = lease->cls;
            out.lease_frontier = lease->lease_frontier;
            out.proven_frontier = lease->proven_frontier;
            out.expires_at_ns = lease->expires_at_ns;
            out.protection = server_cache_control_protection_state::subject_lost;
            return finish(server_cache_control_status::subject_lost,
                          lease->cls);
        }
        if (lease->orphaned &&
            operation == server_cache_control_operation::lease_inspect) {
            out.granted_class = lease->cls;
            out.lease_frontier = lease->lease_frontier;
            out.proven_frontier = lease->proven_frontier;
            out.expires_at_ns = lease->expires_at_ns;
            out.protection = server_cache_control_protection_state::orphaned;
            return finish(server_cache_control_status::orphaned, lease->cls);
        }
        const auto current = state_->resolve_subject(
            lease->subject, current_subject, current_identity,
            current_frontier,
            lease->cls == server_cache_lease_class::hard
                ? &refreshed_subject_pin
                : nullptr);
        if (current != server_cache_control_status::ok) {
            return finish(server_cache_control_status::stale_capability,
                          lease->cls);
        }
        out.granted_class = lease->cls;
        out.lease_frontier = current_frontier;
        out.proven_frontier = lease->proven_frontier;
        out.expires_at_ns = lease->expires_at_ns;
        if (lease->orphaned) {
            out.protection = server_cache_control_protection_state::orphaned;
            if (operation == server_cache_control_operation::lease_inspect) {
                return finish(server_cache_control_status::orphaned, lease->cls);
            }
        }
        const bool partial = frontier_extends(
            current_frontier, lease->proven_frontier);
        if (operation == server_cache_control_operation::lease_inspect) {
            out.protection = partial
                ? server_cache_control_protection_state::partially_stale
                : server_cache_control_protection_state::current;
            return finish(partial ? server_cache_control_status::partially_stale
                                  : server_cache_control_status::ok,
                          lease->cls);
        }
        if (operation != server_cache_control_operation::lease_renew) {
            return finish(server_cache_control_status::invalid_request,
                          lease->cls);
        }
        if (!state_->grant_path_available) {
            return finish(server_cache_control_status::internal_fault,
                          lease->cls);
        }
        uint64_t deadline = 0;
        if (!state_->deadline(request.ttl_ns, deadline)) {
            return finish(server_cache_control_status::invalid_request,
                          lease->cls);
        }
        bool renewed = false;
        if (lease->cls == server_cache_lease_class::hard) {
            server_cache_durable_fallback_proof proof;
            const auto fallback = state_->resolve_fallback(
                request.fallback, current_subject.artifact, current_identity,
                current_frontier, proof);
            if (fallback != server_cache_control_status::ok) {
                return finish(
                    fallback == server_cache_control_status::not_found
                        ? server_cache_control_status::fallback_unavailable
                        : fallback,
                    lease->cls);
            }
            impl::staged_proof staged(
                *state_, current_subject, current_identity,
                std::move(proof));
            renewed = state_->table->renew_owned(
                lease->table_lease, holder->id, current_frontier,
                request.ttl_ns);
        } else {
            renewed = state_->table->renew(
                lease->table_lease, request.ttl_ns);
        }
        if (!renewed) {
            return finish(server_cache_control_status::stale_capability,
                          lease->cls);
        }
        lease->lease_frontier = current_frontier;
        lease->proven_frontier = current_frontier;
        lease->expires_at_ns = deadline;
        lease->orphaned = false;
        if (lease->cls == server_cache_lease_class::hard) {
            lease->fallback = request.fallback;
            lease->fallback_pinned_bytes_known = false;
            lease->fallback_pinned_bytes = 0;
            lease->shared_fallback = false;
            if (state_->selector_evidence) {
                lease->fallback_pinned_bytes_known =
                    state_->selector_evidence(
                        state_->selector_evidence_context,
                        request.fallback, lease->fallback_pinned_bytes,
                        lease->shared_fallback);
            }
        }
        if (refreshed_subject_pin.available()) {
            lease->subject_pin = std::move(refreshed_subject_pin);
        }
        out.lease_frontier = current_frontier;
        out.proven_frontier = current_frontier;
        out.expires_at_ns = deadline;
        out.protection = server_cache_control_protection_state::current;
        out.fallback_kind = lease->fallback.kind;
        out.fallback_pinned_bytes_known = lease->fallback_pinned_bytes_known;
        out.fallback_pinned_bytes = lease->fallback_pinned_bytes;
        out.shared_fallback = lease->shared_fallback;
        return finish(server_cache_control_status::ok, lease->cls);
    } catch (...) {
        state_->pending = false;
        state_->pending_proof = {};
        return finish(server_cache_control_status::internal_fault);
    }
}

const char * server_cache_control_status_name(
        server_cache_control_status status) noexcept {
    switch (status) {
        case server_cache_control_status::ok: return "ok";
        case server_cache_control_status::invalid_request: return "invalid_request";
        case server_cache_control_status::not_supported: return "not_supported";
        case server_cache_control_status::not_found: return "not_found";
        case server_cache_control_status::identity_unavailable: return "identity_unavailable";
        case server_cache_control_status::subject_busy: return "subject_busy";
        case server_cache_control_status::fallback_unavailable: return "fallback_unavailable";
        case server_cache_control_status::fallback_invalid: return "fallback_invalid";
        case server_cache_control_status::hard_lease_blocked: return "hard_lease_blocked";
        case server_cache_control_status::lease_conflict: return "lease_conflict";
        case server_cache_control_status::lease_expired: return "lease_expired";
        case server_cache_control_status::partially_stale: return "partially_stale";
        case server_cache_control_status::subject_lost: return "subject_lost";
        case server_cache_control_status::orphaned: return "orphaned";
        case server_cache_control_status::already_released: return "already_released";
        case server_cache_control_status::profile_unfitted: return "profile_unfitted";
        case server_cache_control_status::capacity_refused: return "capacity_refused";
        case server_cache_control_status::stale_capability: return "stale_capability";
        case server_cache_control_status::internal_fault: return "internal_fault";
        case server_cache_control_status::_count: break;
    }
    return "internal_fault";
}
