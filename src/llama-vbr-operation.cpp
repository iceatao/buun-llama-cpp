#include "llama-vbr-operation.h"

#include "llama-cparams.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <mutex>

namespace {

constexpr size_t VBR_OPERATION_REGISTRY_CAPACITY = 4096;

// VBR_OPERATION_ALLOCATOR_DEFINITION
std::atomic<uint64_t> g_vbr_next_operation_id { 1 };
std::atomic<bool> g_vbr_operation_id_exhausted { false };
std::array<std::atomic<uint64_t>, VBR_OPERATION_REGISTRY_CAPACITY> g_vbr_live_operations {};
// Authenticated mutation binding retained per live slot.
// One mutex serializes every binding write, read, reuse, and the
// whole recovery state machine — the registries run per-operation, never per-token, so a lock
// is the honest concurrency model. The atomic ID
// slots remain ONLY as the lock-free is_live fast path.
std::mutex g_vbr_registry_mutex;
std::array<vbr_operation_binding, VBR_OPERATION_REGISTRY_CAPACITY> g_vbr_live_bindings {};

vbr_operation_id vbr_operation_allocate() {
    if (g_vbr_operation_id_exhausted.load(std::memory_order_acquire)) {
        return {};
    }

    uint64_t expected = g_vbr_next_operation_id.load(std::memory_order_relaxed);
    for (;;) {
        if (expected == 0) {
            g_vbr_operation_id_exhausted.store(true, std::memory_order_release);
            return {};
        }

        // UINT64_MAX is reserved as the slot-claim sentinel (): exhaust one id early rather
        // than ever handing out a value that could alias a mid-claim slot.
        if (expected == std::numeric_limits<uint64_t>::max()) {
            g_vbr_operation_id_exhausted.store(true, std::memory_order_release);
            return {};
        }
        const uint64_t next = expected + 1;
        if (g_vbr_next_operation_id.compare_exchange_weak(
                    expected, next, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return { expected };
        }
    }
}

size_t vbr_operation_slot(vbr_operation_id operation_id) {
    return static_cast<size_t>(operation_id.value % VBR_OPERATION_REGISTRY_CAPACITY);
}

} // namespace

vbr_operation_id vbr_operation_registry_begin(vbr_operation_binding & binding) {
    // VBR_OPERATION_MINT_SITE
    if (binding.operation_id) {
        return {};
    }
    // The manifest itself is validated at mint: closed enums per target, a
    // sane target count, and at least one target for mutate-phase operations.
    if (static_cast<uint8_t>(binding.kind) >=
            static_cast<uint8_t>(vbr_operation_kind::count) ||
        static_cast<uint8_t>(binding.child_phase) >=
            static_cast<uint8_t>(vbr_operation_phase::count) ||
        binding.n_targets > vbr_operation_binding::MAX_TARGETS ||
        (binding.child_phase == vbr_operation_phase::mutate && binding.n_targets == 0)) {
        return {};
    }
    for (uint8_t t = 0; t < binding.n_targets; ++t) {
        const auto & target = binding.targets[t];
        // Closed validation domain per target. The registrant mask is a nonzero
        // subset of the kind's canonical set (equality valid): mask != 0 && no out-of-kind
        // bit. Targets carry the mutate phase (the stated convention). Mutation targets bind
        // exact nonzero instances — recovery alone is capability-subset-restricted instead.
        // Ranges follow the closed per-kind enumeration; seq stays in its declared domain
        // (-1 = declared wildcard).
        if (static_cast<uint8_t>(target.operation_class) >=
                    static_cast<uint8_t>(vbr_operation_class::count) ||
            target.child_phase != vbr_operation_phase::mutate ||
            target.registrant_mask == 0 ||
            (target.registrant_mask & ~vbr_operation_kind_registrants(binding.kind)) != 0 ||
            (binding.kind != vbr_operation_kind::recovery &&
             !vbr_controller_instance_id_is_set(target.instance_id)) ||
            // The closed sequence domain is a declared wildcard or [0, LLAMA_MAX_SEQ).
            target.seq_id < -1 || target.seq_id >= LLAMA_MAX_SEQ ||
            !vbr_target_range_valid(binding.kind, target.range)) {
            return {};
        }
    }

    const vbr_operation_id operation_id = vbr_operation_allocate();
    if (!operation_id) {
        return {};
    }

    constexpr uint64_t VBR_SLOT_CLAIM_SENTINEL = std::numeric_limits<uint64_t>::max();
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    const size_t first = vbr_operation_slot(operation_id);
    for (size_t i = 0; i < VBR_OPERATION_REGISTRY_CAPACITY; ++i) {
        const size_t slot_index = (first + i) % VBR_OPERATION_REGISTRY_CAPACITY;
        auto & slot = g_vbr_live_operations[slot_index];
        uint64_t empty = 0;
        // Claim exclusively first (0 -> sentinel); only the claimant writes the binding,
        // then publishes the real id with release ordering. Readers ignore the sentinel.
        if (!slot.compare_exchange_strong(
                    empty, VBR_SLOT_CLAIM_SENTINEL, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            continue;
        }
        vbr_operation_binding staged = binding;
        staged.operation_id          = operation_id;
        g_vbr_live_bindings[slot_index] = staged;
        slot.store(operation_id.value, std::memory_order_release);
        binding.operation_id = operation_id;
        return operation_id;
    }

    // The ID is intentionally burned: allocation never reuses an identity even when the bounded
    // live-operation registry is temporarily full.
    return {};
}

bool vbr_operation_registry_end(vbr_operation_id operation_id) {
    if (!operation_id) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    const size_t first = vbr_operation_slot(operation_id);
    for (size_t i = 0; i < VBR_OPERATION_REGISTRY_CAPACITY; ++i) {
        auto & slot = g_vbr_live_operations[(first + i) % VBR_OPERATION_REGISTRY_CAPACITY];
        uint64_t expected = operation_id.value;
        if (slot.compare_exchange_strong(
                    expected, 0, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

bool vbr_operation_registry_is_live(vbr_operation_id operation_id) {
    if (!operation_id) {
        return false;
    }

    const size_t first = vbr_operation_slot(operation_id);
    for (size_t i = 0; i < VBR_OPERATION_REGISTRY_CAPACITY; ++i) {
        if (g_vbr_live_operations[(first + i) % VBR_OPERATION_REGISTRY_CAPACITY].load(
                    std::memory_order_acquire) == operation_id.value) {
            return true;
        }
    }
    return false;
}

bool vbr_operation_registry_has_capacity() {
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    for (const auto & slot : g_vbr_live_operations) {
        if (slot.load(std::memory_order_acquire) == 0) {
            return true;
        }
    }
    return false;
}

bool vbr_operation_registry_binding(vbr_operation_id operation_id, vbr_operation_binding & out) {
    if (!operation_id) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    const size_t first = vbr_operation_slot(operation_id);
    for (size_t i = 0; i < VBR_OPERATION_REGISTRY_CAPACITY; ++i) {
        const size_t slot = (first + i) % VBR_OPERATION_REGISTRY_CAPACITY;
        if (g_vbr_live_operations[slot].load(std::memory_order_acquire) != operation_id.value) {
            continue;
        }
        // Under the registry mutex the copy cannot race an end or reuse.
        out = g_vbr_live_bindings[slot];
        return true;
    }
    return false;
}

bool vbr_operation_registry_quiescent_for(
        const vbr_controller_instance_id * instances,
        size_t n_instances) noexcept {
    return vbr_operation_registry_quiescent_for_except(
        instances, n_instances, {});
}

bool vbr_operation_registry_quiescent_for_except(
        const vbr_controller_instance_id * instances,
        size_t n_instances,
        vbr_operation_id allowed) noexcept {
    if (instances == nullptr || n_instances == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    for (size_t slot = 0; slot < VBR_OPERATION_REGISTRY_CAPACITY;
         ++slot) {
        const uint64_t live =
            g_vbr_live_operations[slot].load(std::memory_order_acquire);
        if (live == 0 ||
            live == std::numeric_limits<uint64_t>::max()) {
            continue;
        }
        const auto & binding = g_vbr_live_bindings[slot];
        GGML_ASSERT(binding.operation_id.value == live);
        if (allowed && binding.operation_id == allowed) {
            continue;
        }
        for (uint8_t i = 0; i < binding.n_targets; ++i) {
            const auto & target = binding.targets[i];
            for (size_t p = 0; p < n_instances; ++p) {
                if (!vbr_controller_instance_id_is_set(target.instance_id) ||
                    target.instance_id == instances[p]) {
                    return false;
                }
            }
        }
    }
    return true;
}

void vbr_recovery_autorecord_on_close(vbr_operation_id operation_id);

bool vbr_operation_registry_close(vbr_operation_id operation_id, vbr_operation_outcome outcome) {
    if (outcome != vbr_operation_outcome::committed) {
        // A reserved-but-unrecorded recovery record for this operation transitions to
        // `recorded` automatically so a non-committed close can never orphan the reservation
        // No odd serial is permitted without an authenticated resolution path.
        vbr_recovery_autorecord_on_close(operation_id);
    }
    return vbr_operation_registry_end(operation_id);
}

// ---------------------------------------------------------------------------
// Authenticated operation-recovery ring.
// ---------------------------------------------------------------------------

namespace {

constexpr int32_t VBR_RECOVERY_RING_CAPACITY = 64;

vbr_failed_operation_record g_vbr_recovery_ring[VBR_RECOVERY_RING_CAPACITY];
// Per-slot acknowledgement nonces prevent a stale token from acknowledging a reused slot.
uint64_t g_vbr_quarantine_nonce[VBR_RECOVERY_RING_CAPACITY] = {};
uint64_t g_vbr_quarantine_nonce_next = 1;
// Lock-free fast path for the per-decode-boundary drain: only nonzero when awaiting_ack
// records exist; the empty ring is the overwhelmingly common state.
std::atomic<uint32_t> g_vbr_quarantine_pending { 0 };

vbr_failed_operation_record * recovery_slot(int32_t record_index) {
    if (record_index < 0 || record_index >= VBR_RECOVERY_RING_CAPACITY) {
        return nullptr;
    }
    return &g_vbr_recovery_ring[record_index];
}


vbr_failed_operation_record * reserved_record(int32_t record_index, vbr_operation_id operation_id) {
    auto * record = recovery_slot(record_index);
    return record != nullptr && record->state == vbr_recovery_state::reserved &&
                   record->binding.operation_id == operation_id
               ? record
               : nullptr;
}

vbr_failed_operation_record * minted_record(int32_t record_index) {
    auto * record = recovery_slot(record_index);
    return record != nullptr && record->state == vbr_recovery_state::capability_minted ? record : nullptr;
}

}  // namespace

void vbr_recovery_autorecord_on_close(vbr_operation_id operation_id) {
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    for (auto & record : g_vbr_recovery_ring) {
        if (record.state == vbr_recovery_state::reserved &&
            record.binding.operation_id == operation_id) {
            record.state         = vbr_recovery_state::recorded;
            record.failure_site  = vbr_recovery_failure_site::exception_unwind;
            record.phase_reached = record.binding.child_phase;
        }
    }
}

static int32_t vbr_recovery_reserve_locked(const vbr_operation_binding & binding,
                                           vbr_controller_instance_id owner_instance) {
    if (!binding.operation_id) {
        return -1;
    }
    for (int32_t i = 0; i < VBR_RECOVERY_RING_CAPACITY; ++i) {
        auto & record = g_vbr_recovery_ring[i];
        if (record.state == vbr_recovery_state::free_slot) {
            record               = {};
            record.binding       = binding;
            record.owner_instance = owner_instance;
            record.state         = vbr_recovery_state::reserved;
            return i;
        }
    }
    return -1;  // ring exhausted: caller takes the shadow-unavailable path
}

int32_t vbr_recovery_reserve(const vbr_operation_binding & binding,
                             vbr_controller_instance_id owner_instance) {
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    return vbr_recovery_reserve_locked(binding, owner_instance);
}

int32_t vbr_recovery_reserve(vbr_operation_id operation_id,
                             vbr_controller_instance_id owner_instance) {
    if (!operation_id) {
        return -1;
    }
    // One lock acquisition for lookup and reserve.
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    const size_t first = vbr_operation_slot(operation_id);
    for (size_t i = 0; i < VBR_OPERATION_REGISTRY_CAPACITY; ++i) {
        const size_t slot = (first + i) % VBR_OPERATION_REGISTRY_CAPACITY;
        if (g_vbr_live_operations[slot].load(std::memory_order_acquire) != operation_id.value) {
            continue;
        }
        return vbr_recovery_reserve_locked(g_vbr_live_bindings[slot], owner_instance);
    }
    return -1;
}

bool vbr_recovery_release_unused(int32_t record_index, vbr_operation_id operation_id) {
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    auto * record = reserved_record(record_index, operation_id);
    if (record == nullptr) {
        return false;
    }
    *record = {};
    return true;
}

bool vbr_recovery_record_failure(int32_t                   record_index,
                                 vbr_operation_id          operation_id,
                                 vbr_operation_phase       phase_reached,
                                 vbr_recovery_failure_site failure_site,
                                 bool                      dest_bytes_observable) {
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    auto * record = reserved_record(record_index, operation_id);
    if (record == nullptr) {
        return false;
    }
    record->state                 = vbr_recovery_state::recorded;
    record->phase_reached         = phase_reached;
    record->failure_site          = failure_site;
    record->dest_bytes_observable = dest_bytes_observable;
    return true;
}

// Reserved for the cross-stream pending-owner fence; currently
// unreachable: armed VBR forces n_stream == 1 and the seq_cp fence asserts.
bool vbr_recovery_set_source_token(int32_t                       record_index,
                                   vbr_operation_id              operation_id,
                                   uint16_t                      src_stream,
                                   uint16_t                      dst_stream,
                                   vbr_operation_range           src_range,
                                   const std::vector<uint32_t> & src_page_gens) {
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    auto * record = reserved_record(record_index, operation_id);
    if (record == nullptr) {
        return false;
    }
    record->src_stream    = src_stream;
    record->dst_stream    = dst_stream;
    record->src_range     = src_range;
    record->src_page_gens = src_page_gens;
    return true;
}

bool vbr_recovery_get_record(int32_t record_index, vbr_failed_operation_record & out) {
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    auto * record = recovery_slot(record_index);
    if (record == nullptr || record->state == vbr_recovery_state::free_slot) {
        return false;
    }
    out = *record;
    return true;
}

vbr_recovery_capability vbr_recovery_mint(int32_t record_index) {
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    vbr_recovery_capability capability;
    auto * record = recovery_slot(record_index);
    if (record == nullptr || record->state != vbr_recovery_state::recorded) {
        return capability;  // empty: mint right requires a recorded failure (§1.7)
    }
    record->state             = vbr_recovery_state::capability_minted;
    capability.record_index_  = record_index;
    return capability;
}

vbr_recovery_capability::vbr_recovery_capability(vbr_recovery_capability && other) noexcept :
    record_index_(other.record_index_) {
    other.record_index_ = -1;
}

vbr_recovery_capability::~vbr_recovery_capability() {
    if (record_index_ < 0) {
        return;
    }
    // Fail-closed: destruction without explicit resolution quarantines the record and latches
    // the pending flag the owner must consume with a global invalidation.
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    auto * record = minted_record(record_index_);
    if (record != nullptr) {
        record->state                         = vbr_recovery_state::awaiting_ack;
        g_vbr_quarantine_nonce[record_index_] = g_vbr_quarantine_nonce_next++;
        g_vbr_quarantine_pending.fetch_add(1, std::memory_order_release);
    }
    record_index_ = -1;
}

bool vbr_recovery_capability::target_allowed(uint16_t stream, llama_seq_id seq_id,
                                             llama_pos p0, llama_pos p1) const {
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    const auto * record = minted_record(record_index_);
    if (record == nullptr) {
        return false;
    }
    // Validate against the manifest targets: a recovery mutation must fall inside one
    // declared target (wildcards only if the binding declared them). The stream is
    // target-exact through the ONE stream predicate; the src/dst source-token fields never
    // widen authorization (their default 0 authorized stream 0 against any record). A future
    // cross-stream recovery must declare explicit source/destination TARGETS instead.
    for (uint8_t t = 0; t < record->binding.n_targets; ++t) {
        const auto & target = record->binding.targets[t];
        const bool stream_ok = target.stream_matches(stream);
        const bool seq_ok    = target.seq_id < 0 || seq_id == target.seq_id;
        const bool range_ok  = target.range.p0 < 0 ||
                               (p0 >= target.range.p0 && p1 <= target.range.p1);
        if (stream_ok && seq_ok && range_ok) {
            return true;
        }
    }
    return false;
}

bool vbr_recovery_capability::resolve_completed() {
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    auto * record = minted_record(record_index_);
    if (record == nullptr) {
        return false;
    }
    // Resolution reclaims the ring slot; records are recovery evidence, not history.
    *record       = {};
    record_index_ = -1;
    return true;
}

bool vbr_recovery_capability::resolve_quarantined() {
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    auto * record = minted_record(record_index_);
    if (record == nullptr) {
        return false;
    }
    // Retain the record and manifest targets until the owning tracker acknowledges invalidation.
    record->state                            = vbr_recovery_state::awaiting_ack;
    g_vbr_quarantine_nonce[record_index_]    = g_vbr_quarantine_nonce_next++;
    g_vbr_quarantine_pending.fetch_add(1, std::memory_order_release);
    record_index_                            = -1;
    return true;
}

vbr_quarantine_work vbr_recovery_take_quarantine(vbr_controller_instance_id instance) {
    if (g_vbr_quarantine_pending.load(std::memory_order_acquire) == 0) {
        return {};
    }
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    for (int32_t i = 0; i < VBR_RECOVERY_RING_CAPACITY; ++i) {
        auto & record = g_vbr_recovery_ring[i];
        if (record.state != vbr_recovery_state::awaiting_ack || record.taken) {
            continue;
        }
        // Match only the owner instance, never the composite manifest.
        const bool matches = !vbr_controller_instance_id_is_set(record.owner_instance) ||
                             record.owner_instance == instance;
        if (!matches) {
            continue;
        }
        record.taken          = true;
        record.taker_instance = instance;
        vbr_quarantine_work work;
        work.token   = { i, g_vbr_quarantine_nonce[i] };
        work.binding = record.binding;
        return work;
    }
    return {};
}

bool vbr_recovery_untake_quarantine(vbr_quarantine_token token,
                                    vbr_controller_instance_id instance) {
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    if (!token || token.record_index >= VBR_RECOVERY_RING_CAPACITY) {
        return false;
    }
    auto & record = g_vbr_recovery_ring[token.record_index];
    if (record.state != vbr_recovery_state::awaiting_ack || !record.taken ||
        record.taker_instance != instance ||
        g_vbr_quarantine_nonce[token.record_index] != token.nonce) {
        return false;
    }
    record.taken          = false;
    record.taker_instance = {};
    return true;
}

static bool ring_has_nonfree(
        vbr_controller_instance_id instance,
        bool count_wildcard_owner,
        vbr_operation_id allowed_reserved_operation = {}) {
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    for (const auto & record : g_vbr_recovery_ring) {
        if (allowed_reserved_operation &&
            record.state == vbr_recovery_state::reserved &&
            record.binding.operation_id == allowed_reserved_operation) {
            continue;
        }
        if (record.state != vbr_recovery_state::free_slot &&
            (record.owner_instance == instance ||
             (count_wildcard_owner &&
              !vbr_controller_instance_id_is_set(record.owner_instance)))) {
            return true;
        }
    }
    return false;
}

bool vbr_recovery_pending_for(vbr_controller_instance_id instance) {
    // Full cause set: reserved (in-flight operation) and
    // capability_minted (recovery mid-execution) block re-arm exactly like recorded and
    // awaiting_ack; every non-free state is unresolved recovery work serviceable here.
    return ring_has_nonfree(instance, /*count_wildcard_owner=*/true);
}

bool vbr_recovery_pending_for_except(
        vbr_controller_instance_id instance,
        vbr_operation_id allowed_reserved_operation) {
    if (!allowed_reserved_operation) {
        return true;
    }
    return ring_has_nonfree(
        instance, /*count_wildcard_owner=*/true,
        allowed_reserved_operation);
}

bool vbr_recovery_owned_by(vbr_controller_instance_id instance) {
    if (!vbr_controller_instance_id_is_set(instance)) {
        return false;
    }
    return ring_has_nonfree(instance, /*count_wildcard_owner=*/false);
}

int32_t vbr_recovery_advance_recorded(vbr_controller_instance_id instance) {
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    int32_t advanced = 0;
    for (int32_t i = 0; i < VBR_RECOVERY_RING_CAPACITY; ++i) {
        auto & record = g_vbr_recovery_ring[i];
        if (record.state != vbr_recovery_state::recorded) {
            continue;
        }
        const bool matches = !vbr_controller_instance_id_is_set(record.owner_instance) ||
                             record.owner_instance == instance;
        if (!matches) {
            continue;
        }
        record.state             = vbr_recovery_state::awaiting_ack;
        g_vbr_quarantine_nonce[i] = g_vbr_quarantine_nonce_next++;
        g_vbr_quarantine_pending.fetch_add(1, std::memory_order_release);
        ++advanced;
    }
    return advanced;
}

bool vbr_recovery_ack_quarantine(vbr_quarantine_token token,
                                 vbr_controller_instance_id instance) {
    std::lock_guard<std::mutex> lock(g_vbr_registry_mutex);
    if (!token || token.record_index >= VBR_RECOVERY_RING_CAPACITY ||
        g_vbr_recovery_ring[token.record_index].state != vbr_recovery_state::awaiting_ack ||
        !g_vbr_recovery_ring[token.record_index].taken ||
        g_vbr_recovery_ring[token.record_index].taker_instance != instance ||
        g_vbr_quarantine_nonce[token.record_index] != token.nonce) {
        return false;
    }
    g_vbr_recovery_ring[token.record_index]        = {};
    g_vbr_quarantine_nonce[token.record_index]     = 0;
    g_vbr_quarantine_pending.fetch_sub(1, std::memory_order_release);
    return true;
}

vbr_scoped_operation::vbr_scoped_operation(vbr_operation_binding binding) : binding_(binding) {
    binding_.operation_id = {};
    vbr_operation_registry_begin(binding_);
}

vbr_scoped_operation::vbr_scoped_operation(vbr_scoped_operation && other) noexcept :
    binding_(other.binding_) {
    other.binding_.operation_id = {};
}

vbr_scoped_operation::~vbr_scoped_operation() {
    // Destruction without an
    // explicit close is a failure — forgotten paths and unwind can never commit.
    if (binding_.operation_id) {
        vbr_operation_registry_close(binding_.operation_id, vbr_operation_outcome::failed);
    }
}

bool vbr_scoped_operation::close(vbr_operation_outcome outcome) {
    if (!binding_.operation_id) {
        return false;
    }
    const bool ok         = vbr_operation_registry_close(binding_.operation_id, outcome);
    binding_.operation_id = {};
    return ok;
}

vbr_operation_id vbr_scoped_operation::release() {
    const vbr_operation_id id = binding_.operation_id;
    binding_.operation_id     = {};
    return id;
}
