#pragma once

#include "../../src/llama-cache-accounting.h"

#include <array>
#include <cstddef>
#include <cstdint>

// One closed lifecycle inventory. The second argument is the physical choke point allowed to
// contain the corresponding raw primitive; the third is its logical admission owner; the
// fourth names the accounting release owner. Two full-slot classes deliberately share one
// manifest builder/admission owner. Only host artifacts own committed C operations today:
// their raw eraser is physical-only and the legacy wrapper owns the lifecycle-off terminal.
// CI extracts the mapping, so adding a class or raw primitive without extending it fails.
#define SERVER_CACHE_DESTRUCTION_INVENTORY(X) \
    X(slot_drop,                server_cache_slot_drop_impl,                observe_full_slot,                   legacy_wrapper_or_capability) \
    X(live_range_drop,          server_cache_live_range_drop_impl,          observe_live_range_drop,             none) \
    X(host_artifact_drop,       server_prompt_cache_destroy_entry_impl,     server_prompt_cache_observe_drop,     legacy_wrapper_or_capability) \
    X(checkpoint_drop,          server_cache_checkpoint_drop_impl,          observe_checkpoint_drop,             legacy_wrapper_or_capability) \
    X(token_ledger_truncate,    server_cache_token_ledger_truncate_impl,    token_ledger_truncate,               none) \
    X(mandatory_recovery_reset, server_cache_mandatory_recovery_reset_impl, observe_full_slot,                   none)

enum class server_cache_destruction_class : uint8_t {
#define SERVER_CACHE_DESTRUCTION_CLASS(name, symbol, admission, release_owner) name,
    SERVER_CACHE_DESTRUCTION_INVENTORY(SERVER_CACHE_DESTRUCTION_CLASS)
#undef SERVER_CACHE_DESTRUCTION_CLASS
    _count,
};

enum class server_cache_destruction_reason : uint8_t {
    slot_rebind = 0,
    child_release,
    idle_reclaim,
    live_prefix_replace,
    context_shift,
    checkpoint_invalidated,
    checkpoint_thin,
    checkpoint_capacity,
    host_dedup,
    host_capacity,
    host_token_limit,
    host_consumed_restore,
    host_shutdown,
    low_lcp_reset,
    restore_failure,
    trim_rejection,
    transient_speculative,
    _count,
};

enum class server_cache_destruction_target_kind : uint8_t {
    live_target = 0,
    live_draft,
    token_ledger,
    checkpoint_ring,
    rolling_window,
    typed_accelerator,
    host_artifact,
    _count,
};

enum class server_cache_destruction_verdict : uint8_t {
    admit_unleased = 0,
    admit_soft_leased,
    would_refuse_hard_leased,
    admit_mandatory_recovery,
    unavailable,
    _count,
};

enum class server_cache_destruction_execution : uint8_t {
    pass_through = 0,
    // The legacy-selected host victim is still chosen by the existing
    // FIFO/dedup policy, but its exact C release is committed through the
    // prepared capability after the physical erase. Lease verdicts remain
    // pricing evidence here; this terminal does not select the victim.
    prepared_release,
    // A host victim was erased only after all three payload planes were
    // proved identical to a disjoint survivor and the survivor was pinned
    // through the exact prepared-release commit.
    redundant_host_eviction,
    // A cache-plan-priced host capacity/token victim, certified against a
    // disjoint retained or pre-authorized durable recovery source.
    priced_host_eviction,
    // One independently accounted live checkpoint member was removed
    // under a pinned replay-source certificate. Host-entry checkpoint rings
    // remain aggregate-owned and can never reach this terminal.
    priced_checkpoint_thinning,
    // The cache plan selected an occupied live target only after a durable recovery
    // source and the complete fixed-pool/checkpoint union were certified.
    priced_live_displacement,
    _count,
};

struct server_cache_destruction_target {
    server_cache_destruction_target_kind kind =
        server_cache_destruction_target_kind::live_target;
    int32_t slot_id = -1;
    llama_cache_acct_artifact_id artifact;
    bool artifact_known = false;
};

struct server_cache_destruction_yield {
    llama_cache_acct_category category =
        llama_cache_acct_category::container_overhead;
    llama_cache_acct_resource_domain domain;
    bool domain_known = false;
    llama_cache_acct_measure measure =
        llama_cache_acct_measure::logical_payload;
    llama_cache_acct_value value = {
        0, llama_cache_acct_known::unavailable,
    };
};

constexpr size_t SERVER_CACHE_DESTRUCTION_MAX_TARGETS = 8;
constexpr size_t SERVER_CACHE_DESTRUCTION_MAX_YIELDS  = 16;
constexpr size_t SERVER_CACHE_DESTRUCTION_EVENT_RING  = 64;

struct server_cache_destruction_request {
    server_cache_destruction_class cls =
        server_cache_destruction_class::slot_drop;
    server_cache_destruction_reason reason =
        server_cache_destruction_reason::slot_rebind;
    std::array<server_cache_destruction_target,
               SERVER_CACHE_DESTRUCTION_MAX_TARGETS> targets = {};
    std::array<server_cache_destruction_yield,
               SERVER_CACHE_DESTRUCTION_MAX_YIELDS> yields = {};
    uint8_t n_targets = 0;
    uint8_t n_yields  = 0;
    bool overflowed   = false;

    void add_target(server_cache_destruction_target_kind kind, int32_t slot_id) noexcept {
        if (n_targets >= targets.size()) {
            overflowed = true;
            return;
        }
        auto & target = targets[n_targets++];
        target.kind    = kind;
        target.slot_id = slot_id;
    }

    void add_target(
            server_cache_destruction_target_kind kind,
            int32_t slot_id,
            llama_cache_acct_artifact_id artifact) noexcept {
        add_target(kind, slot_id);
        if (n_targets > 0 && !overflowed) {
            auto & target = targets[n_targets - 1];
            target.artifact = artifact;
            target.artifact_known = artifact.v != 0;
        }
    }

    void add_yield(const server_cache_destruction_yield & value) noexcept {
        if (n_yields >= yields.size()) {
            overflowed = true;
            return;
        }
        yields[n_yields++] = value;
    }

    void add_yield(llama_cache_acct_category category) noexcept {
        server_cache_destruction_yield value;
        value.category = category;
        add_yield(value);
    }
};

using server_cache_lease_evaluator = server_cache_destruction_verdict (*)(
        void * context,
        const server_cache_destruction_request & request) noexcept;

// A logical operation keeps this small token across split physical phases. Admission and retirement
// execute pass-through; execution authority can change later without recutting joined
// operations such as low-LCP reset.
struct server_cache_destruction_admission {
    server_cache_destruction_class cls =
        server_cache_destruction_class::slot_drop;
    server_cache_destruction_reason reason =
        server_cache_destruction_reason::slot_rebind;
    server_cache_destruction_verdict verdict =
        server_cache_destruction_verdict::admit_unleased;
    server_cache_destruction_execution execution =
        server_cache_destruction_execution::pass_through;
    uint64_t sequence = 0;
    bool issued = false;
    bool observer_recorded = false;

    bool covers(
            server_cache_destruction_class expected_class,
            server_cache_destruction_reason expected_reason) const noexcept {
        return issued && cls == expected_class && reason == expected_reason;
    }
};

struct server_cache_destruction_event {
    server_cache_destruction_request request;
    server_cache_destruction_verdict verdict =
        server_cache_destruction_verdict::admit_unleased;
    server_cache_destruction_execution execution =
        server_cache_destruction_execution::pass_through;
    uint64_t sequence = 0;
};

// Transient, process-local lifecycle observer. It is deliberately absent from cache-plan JSON
// until final projection. Recording is fixed-capacity and noexcept; overwriting the oldest detail never
// loses the monotone per-class totals.
struct server_cache_destruction_observer {
    std::array<server_cache_destruction_event,
               SERVER_CACHE_DESTRUCTION_EVENT_RING> events = {};
    std::array<uint64_t, size_t(server_cache_destruction_class::_count)> totals = {};
    uint64_t n_events   = 0;
    uint64_t overflows  = 0;
    uint64_t host_restores_retained = 0;
    uint64_t host_restores_consumed = 0;
    uint64_t prepared_release_commits = 0;
    uint64_t prepared_release_fallbacks = 0;
    uint64_t redundant_host_certified = 0;
    uint64_t redundant_host_executed = 0;
    uint64_t redundant_host_refused = 0;
    uint64_t redundant_host_release_bytes = 0;
    uint64_t host_trade_attempted = 0;
    uint64_t host_trade_certified = 0;
    uint64_t host_trade_executed = 0;
    uint64_t host_trade_refused = 0;
    uint64_t host_trade_unpriced = 0;
    uint64_t host_trade_legacy_fallbacks = 0;
    uint64_t host_trade_retention_capacity_executed = 0;
    uint64_t host_trade_hard_lease_vetoes = 0;
    uint64_t host_trade_publication_skips = 0;
    uint64_t host_trade_substrate_unavailable = 0;
    uint64_t host_trade_main_family_evictions = 0;
    uint64_t host_trade_soft_lease_evictions = 0;
    uint64_t host_trade_zero_destruction_ties = 0;
    uint64_t host_trade_release_bytes = 0;
    uint64_t checkpoint_thin_attempted = 0;
    uint64_t checkpoint_thin_certified = 0;
    uint64_t checkpoint_thin_executed = 0;
    uint64_t checkpoint_thin_refused = 0;
    uint64_t checkpoint_thin_heuristic_refusals = 0;
    uint64_t checkpoint_thin_mandatory_refusals = 0;
    uint64_t checkpoint_thin_hard_lease_refusals = 0;
    uint64_t checkpoint_publication_skips = 0;
    uint64_t checkpoint_thin_release_bytes = 0;
    uint64_t live_displacement_certified = 0;
    uint64_t live_displacement_executed = 0;
    uint64_t live_displacement_refused = 0;
    void * lease_context = nullptr;
    server_cache_lease_evaluator lease_evaluator = nullptr;

    server_cache_destruction_event & event_slot_for_sequence(
            uint64_t sequence) noexcept {
        return events[size_t((sequence - 1) % events.size())];
    }

    server_cache_destruction_event * event_for_sequence(
            uint64_t sequence) noexcept {
        if (sequence == 0) {
            return nullptr;
        }
        auto & event = event_slot_for_sequence(sequence);
        return event.sequence == sequence ? &event : nullptr;
    }

    uint64_t observe(
            const server_cache_destruction_request & request,
            server_cache_destruction_verdict verdict) noexcept {
        const size_t cls = size_t(request.cls);
        if (cls >= totals.size()) {
            overflows++;
            return 0;
        }
        const uint64_t sequence = n_events + 1;
        server_cache_destruction_event & event =
            event_slot_for_sequence(sequence);
        event.request   = request;
        event.verdict   = verdict;
        event.execution = server_cache_destruction_execution::pass_through;
        event.sequence  = sequence;
        totals[cls]++;
        n_events++;
        if (request.overflowed) {
            overflows++;
        }
        return event.sequence;
    }

    void note_host_restore(bool retained) noexcept {
        if (retained) {
            host_restores_retained++;
        } else {
            host_restores_consumed++;
        }
    }

    void note_prepared_release(uint64_t sequence, bool committed) noexcept {
        if (committed) {
            prepared_release_commits++;
        } else {
            prepared_release_fallbacks++;
        }
        if (!committed || sequence == 0) {
            return;
        }
        if (auto * event = event_for_sequence(sequence)) {
            event->execution =
                server_cache_destruction_execution::prepared_release;
        }
    }

    void note_prepared_release_batch(
            const uint64_t * sequences,
            size_t count,
            bool committed) noexcept {
        if (committed) {
            prepared_release_commits++;
        } else {
            prepared_release_fallbacks++;
        }
        if (!committed || !sequences) {
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            if (auto * event = event_for_sequence(sequences[i])) {
                event->execution =
                    server_cache_destruction_execution::prepared_release;
            }
        }
    }

    void note_live_displacement_certified() noexcept {
        live_displacement_certified++;
    }

    void note_live_displacement_refused() noexcept {
        live_displacement_refused++;
    }

    void note_live_displacement_executed(uint64_t sequence) noexcept {
        live_displacement_executed++;
        if (auto * event = event_for_sequence(sequence)) {
            event->execution =
                server_cache_destruction_execution::priced_live_displacement;
        }
    }

    void note_redundant_host_refused(uint64_t) noexcept {
        redundant_host_refused++;
    }

    void note_redundant_host_executed(
            uint64_t sequence,
            uint64_t released_bytes) noexcept {
        redundant_host_certified++;
        redundant_host_executed++;
        redundant_host_release_bytes += released_bytes;
        if (auto * event = event_for_sequence(sequence)) {
            event->execution =
                server_cache_destruction_execution::redundant_host_eviction;
        }
    }

    void note_host_trade_refused() noexcept {
        host_trade_attempted++;
        host_trade_refused++;
    }

    void note_host_trade_unpriced() noexcept {
        host_trade_attempted++;
        host_trade_unpriced++;
    }

    void note_host_trade_veto() noexcept {
        host_trade_attempted++;
        host_trade_hard_lease_vetoes++;
    }

    void note_host_trade_publication_skip() noexcept {
        host_trade_publication_skips++;
    }

    void note_host_trade_substrate_fault() noexcept {
        host_trade_substrate_unavailable++;
    }

    void note_host_trade_executed(
            uint64_t sequence,
            uint64_t released_bytes,
            bool main_family,
            bool soft_leased,
            bool zero_destruction_tie) noexcept {
        host_trade_certified++;
        host_trade_executed++;
        host_trade_release_bytes += released_bytes;
        host_trade_main_family_evictions += main_family ? 1 : 0;
        host_trade_soft_lease_evictions += soft_leased ? 1 : 0;
        host_trade_zero_destruction_ties += zero_destruction_tie ? 1 : 0;
        if (auto * event = event_for_sequence(sequence)) {
            event->execution =
                server_cache_destruction_execution::priced_host_eviction;
        }
    }

    void note_checkpoint_thin_refused() noexcept {
        checkpoint_thin_attempted++;
        checkpoint_thin_refused++;
    }

    void note_checkpoint_thin_heuristic_refused() noexcept {
        checkpoint_thin_heuristic_refusals++;
    }

    void note_checkpoint_thin_mandatory_refused() noexcept {
        checkpoint_thin_mandatory_refusals++;
    }

    void note_checkpoint_thin_hard_lease_refused() noexcept {
        checkpoint_thin_hard_lease_refusals++;
    }

    void note_checkpoint_publication_skip() noexcept {
        checkpoint_publication_skips++;
    }

    void note_checkpoint_thin_executed(
            uint64_t sequence,
            uint64_t released_bytes) noexcept {
        checkpoint_thin_attempted++;
        checkpoint_thin_certified++;
        checkpoint_thin_executed++;
        checkpoint_thin_release_bytes += released_bytes;
        if (auto * event = event_for_sequence(sequence)) {
            event->execution =
                server_cache_destruction_execution::
                    priced_checkpoint_thinning;
        }
    }
};

// The one retention-admission API. The authority layer supplies the type-erased evaluator
// under cache-debug OR cache-lifecycle; only record/log emission remains
// for debug and production use. Execution remains pass-through until lifecycle policy enables it.
inline server_cache_destruction_admission server_cache_retention_admit(
        server_cache_destruction_observer * observer,
        const server_cache_destruction_request & request) noexcept {
    server_cache_destruction_admission admission;
    admission.cls    = request.cls;
    admission.reason = request.reason;
    admission.issued = true;
    if (observer) {
        if (observer->lease_evaluator) {
            admission.verdict = observer->lease_evaluator(
                observer->lease_context, request);
        }
        admission.sequence = observer->observe(request, admission.verdict);
        admission.observer_recorded = admission.sequence != 0;
    }
    return admission;
}
