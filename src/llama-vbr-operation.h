#pragma once

#include "llama.h"
#include "llama-vbr-controller-id.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// Process-local coordination identity. It is deliberately a strong internal type so it cannot
// accidentally enter a checkpoint/state envelope. The public C freeze API exposes only its
// opaque uint64_t value.
struct vbr_operation_id {
    uint64_t value = 0;

    explicit operator bool() const {
        return value != 0;
    }
};

inline bool operator==(vbr_operation_id lhs, vbr_operation_id rhs) {
    return lhs.value == rhs.value;
}

inline bool operator!=(vbr_operation_id lhs, vbr_operation_id rhs) {
    return !(lhs == rhs);
}

// Rev-9 closed provenance vocabulary. New values require updating the registry inventory below;
// its compile-time coverage checks intentionally make an unregistered extension fail the build.
enum class vbr_mutation_family : uint8_t {
    append,
    occupied_reuse,
    trim,
    seq_share,
    seq_keep,
    shift,
    import,
    restore,
    clear,
    reset,
    degrade,
    promote,
    shed,
    recovery,
    count,
};

enum class vbr_operation_class : uint8_t {
    ordinary_decode,
    checkpoint_restore,
    restore_one_behind_trim,
    swa_wrap,
    explicit_destructive_trim,
    dependency_seq_remove,
    speculative_backup,
    prompt_share,
    sibling_owner_remove,
    host_import,
    state_api,
    controller,
    count,
};

enum class vbr_operation_kind : uint8_t {
    retier_freeze,
    decode,
    sequence_edit,
    checkpoint_restore,
    state_import,
    state_export,
    controller_retier,
    recovery,
    count,
};

// VBR_OPERATION_KIND_EXHAUSTIVE
constexpr std::array<const char *, static_cast<size_t>(vbr_operation_kind::count)>
        VBR_OPERATION_KIND_NAMES = {{
    "retier_freeze",
    "decode",
    "sequence_edit",
    "checkpoint_restore",
    "state_import",
    "state_export",
    "controller_retier",
    "recovery",
}};
static_assert(VBR_OPERATION_KIND_NAMES.size() ==
        static_cast<size_t>(vbr_operation_kind::count),
        "every VBR operation kind must have closed registry vocabulary");

enum class vbr_operation_phase : uint8_t {
    root,
    prepare,
    mutate,
    publish,
    cleanup,
    recovery,
    count,
};

struct vbr_operation_range {
    llama_pos p0 = -1;
    llama_pos p1 = -1;
};

// The complete closed authentication tuple, per target. A
// multi-sequence ubatch carries one target per touched sequence; sequence ops carry one.
// empty instance = target valid for any controller (single-cache ops); seq -1 = wildcard sequence
// (whole-cache edits); range {-1,-1} = whole range. Wildcards are themselves authenticated —
// an event can only use one if its binding declared it.
// "Any stream" wildcard — a protocol fact of target authentication, not a per-site literal.
constexpr uint16_t VBR_STREAM_ANY = 0xFFFF;

struct vbr_operation_target {
    vbr_controller_instance_id instance_id = {};
    // v3.2 pin 1: the COMPLETE closed tuple per target — class/registrants/phase are
    // target-local, not binding-global (the iSWA composite authorizes ordinary_decode on the
    // base instance and swa_wrap on the SWA instance under ONE operation).
    vbr_operation_class operation_class = vbr_operation_class::state_api;
    uint32_t            registrant_mask = 0;
    vbr_operation_phase child_phase     = vbr_operation_phase::mutate;
    uint16_t            stream  = 0;
    llama_seq_id        seq_id  = -1;
    vbr_operation_range range   = {};

    // The one spelling of the load-bearing instance-wildcard predicate.
    bool instance_matches(vbr_controller_instance_id instance) const {
        return !vbr_controller_instance_id_is_set(instance_id) || instance_id == instance;
    }
    bool stream_matches(uint16_t s) const {
        return stream == VBR_STREAM_ANY || stream == s;
    }
    // The event-constant half of the authentication tuple — the ONE spelling shared by the
    // begin-time and per-stamp covering searches (they must never diverge).
    bool matches(vbr_controller_instance_id instance, uint16_t s,
                 vbr_operation_class cls, uint32_t registrant_bit) const {
        return instance_matches(instance) && stream_matches(s) && operation_class == cls &&
               (registrant_mask & registrant_bit) != 0 &&
               child_phase == vbr_operation_phase::mutate;
    }
    // Per-stamp coverage predicates. Wildcards match only where the manifest
    // DECLARED them (seq -1 / range {-1,-1}); an unknown position (-1) is covered only by a
    // whole-range target — whole-cache edits stamp cells whose position is not consulted.
    bool seq_covers(llama_seq_id seq) const {
        return seq_id == -1 || (seq >= 0 && seq == seq_id);
    }
    bool pos_covers(llama_pos pos) const {
        if (range.p0 < 0) {
            return true;  // kind-declared wildcard range
        }
        if (pos < 0) {
            return range.p0 == 0 && range.p1 == std::numeric_limits<llama_pos>::max();
        }
        return pos >= range.p0 && pos < range.p1;
    }
};

struct vbr_operation_binding {
    static constexpr uint8_t MAX_TARGETS = 16;

    vbr_operation_id    operation_id = {};
    vbr_operation_kind  kind         = vbr_operation_kind::retier_freeze;
    vbr_operation_phase child_phase  = vbr_operation_phase::root;  // registry-level sanity only
    uint8_t             n_targets    = 0;
    std::array<vbr_operation_target, MAX_TARGETS> targets = {};

    // Transitional accessors: single-target ops keep the old shape readable.
    llama_seq_id        seq_id() const { return n_targets > 0 ? targets[0].seq_id : -1; }
    vbr_operation_range range()  const { return n_targets > 0 ? targets[0].range : vbr_operation_range{}; }

    // Covering-target search — used by event authentication and quarantine routing so the
    // matching semantics exist exactly once. Class + registrant + phase are target-local.
    const vbr_operation_target * find_covering_target(vbr_controller_instance_id instance,
                                                      uint16_t stream,
                                                      vbr_operation_class operation_class,
                                                      uint32_t registrant_bit) const {
        for (uint8_t t = 0; t < n_targets; ++t) {
            const auto & target = targets[t];
            if (target.matches(instance, stream, operation_class, registrant_bit)) {
                return &target;
            }
        }
        return nullptr;
    }

    // Per-stamp covering-target selection: the full authenticated tuple including
    // the stamped (seq, pre-mutation position). The target index is returned so per-target
    // evidence (lazy extents) binds to the SELECTED record, never a scope-global one.
    const vbr_operation_target * find_covering_target_at(vbr_controller_instance_id instance,
                                                         uint16_t stream,
                                                         vbr_operation_class operation_class,
                                                         uint32_t registrant_bit,
                                                         llama_seq_id seq, llama_pos pos,
                                                         uint8_t * index_out = nullptr) const {
        for (uint8_t t = 0; t < n_targets; ++t) {
            const auto & target = targets[t];
            if (target.matches(instance, stream, operation_class, registrant_bit) &&
                target.seq_covers(seq) && target.pos_covers(pos)) {
                if (index_out != nullptr) {
                    *index_out = t;
                }
                return &target;
            }
        }
        return nullptr;
    }
};

enum class vbr_mutation_registrant : uint8_t {
    apply_ubatch_append,
    apply_ubatch_occupied_reuse,
    seq_rm,
    seq_cp,
    seq_keep,
    seq_add,
    seq_div,
    state_read_meta,
    state_read_data,
    state_read_install,
    state_read_cleanup,
    whole_import,
    explicit_restore_adopt,
    clear,
    full_reset,
    degrade_next,
    promote_next,
    execute_shed,
    authenticated_recovery,
    count,
};

constexpr uint16_t vbr_operation_class_bit(vbr_operation_class operation_class) {
    return uint16_t(1u) << static_cast<uint8_t>(operation_class);
}
static_assert(static_cast<size_t>(vbr_operation_class::count) <= 16,
        "VBR registry allowed-class mask must be widened with the closed class enum");

struct vbr_mutation_registration {
    vbr_mutation_registrant registrant;
    vbr_mutation_family     family;
    vbr_operation_phase     phase;
    uint16_t                allowed_classes;
};

// Closed mutation-registration inventory. apply_ubatch owns assignment registration; find_slot
// remains a read-only planner and is intentionally absent. This table is inert in operation registry; subsequent
// WS-A commits attach dispatch at these audited sites without reopening the vocabulary.
constexpr std::array<vbr_mutation_registration,
        static_cast<size_t>(vbr_mutation_registrant::count)> VBR_MUTATION_REGISTRY = {{
    { vbr_mutation_registrant::apply_ubatch_append,         vbr_mutation_family::append,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::ordinary_decode) |
      vbr_operation_class_bit(vbr_operation_class::swa_wrap) },
    { vbr_mutation_registrant::apply_ubatch_occupied_reuse, vbr_mutation_family::occupied_reuse,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::ordinary_decode) |
      vbr_operation_class_bit(vbr_operation_class::swa_wrap) },
    { vbr_mutation_registrant::seq_rm,                      vbr_mutation_family::trim,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::checkpoint_restore) |
      vbr_operation_class_bit(vbr_operation_class::restore_one_behind_trim) |
      vbr_operation_class_bit(vbr_operation_class::swa_wrap) |
      vbr_operation_class_bit(vbr_operation_class::explicit_destructive_trim) |
      vbr_operation_class_bit(vbr_operation_class::dependency_seq_remove) |
      vbr_operation_class_bit(vbr_operation_class::speculative_backup) |
      vbr_operation_class_bit(vbr_operation_class::sibling_owner_remove) |
      vbr_operation_class_bit(vbr_operation_class::state_api) },
    { vbr_mutation_registrant::seq_cp,                      vbr_mutation_family::seq_share,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::speculative_backup) |
      vbr_operation_class_bit(vbr_operation_class::prompt_share) |
      vbr_operation_class_bit(vbr_operation_class::state_api) },
    { vbr_mutation_registrant::seq_keep,                    vbr_mutation_family::seq_keep,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::state_api) |
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::seq_add,                     vbr_mutation_family::shift,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::state_api) |
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::seq_div,                     vbr_mutation_family::shift,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::state_api) |
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::state_read_meta,             vbr_mutation_family::import,
      vbr_operation_phase::prepare,
      vbr_operation_class_bit(vbr_operation_class::checkpoint_restore) |
      vbr_operation_class_bit(vbr_operation_class::host_import) |
      vbr_operation_class_bit(vbr_operation_class::state_api) },
    { vbr_mutation_registrant::state_read_data,             vbr_mutation_family::import,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::checkpoint_restore) |
      vbr_operation_class_bit(vbr_operation_class::host_import) |
      vbr_operation_class_bit(vbr_operation_class::state_api) },
    { vbr_mutation_registrant::state_read_install,          vbr_mutation_family::import,
      vbr_operation_phase::publish,
      vbr_operation_class_bit(vbr_operation_class::checkpoint_restore) |
      vbr_operation_class_bit(vbr_operation_class::host_import) |
      vbr_operation_class_bit(vbr_operation_class::state_api) },
    { vbr_mutation_registrant::state_read_cleanup,          vbr_mutation_family::import,
      vbr_operation_phase::cleanup,
      vbr_operation_class_bit(vbr_operation_class::checkpoint_restore) |
      vbr_operation_class_bit(vbr_operation_class::host_import) |
      vbr_operation_class_bit(vbr_operation_class::state_api) },
    { vbr_mutation_registrant::whole_import,                vbr_mutation_family::import,
      vbr_operation_phase::publish,
      vbr_operation_class_bit(vbr_operation_class::host_import) |
      vbr_operation_class_bit(vbr_operation_class::state_api) },
    { vbr_mutation_registrant::explicit_restore_adopt,      vbr_mutation_family::restore,
      vbr_operation_phase::publish,
      vbr_operation_class_bit(vbr_operation_class::checkpoint_restore) },
    { vbr_mutation_registrant::clear,                       vbr_mutation_family::clear,
      vbr_operation_phase::cleanup,
      vbr_operation_class_bit(vbr_operation_class::state_api) |
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::full_reset,                  vbr_mutation_family::reset,
      vbr_operation_phase::cleanup,
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::degrade_next,                vbr_mutation_family::degrade,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::promote_next,                vbr_mutation_family::promote,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::execute_shed,                vbr_mutation_family::shed,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::authenticated_recovery,      vbr_mutation_family::recovery,
      vbr_operation_phase::recovery,
      vbr_operation_class_bit(vbr_operation_class::controller) },
}};

constexpr bool vbr_mutation_registry_is_exhaustive() {
    std::array<bool, static_cast<size_t>(vbr_mutation_registrant::count)> registrants = {};
    std::array<bool, static_cast<size_t>(vbr_mutation_family::count)> families = {};
    std::array<bool, static_cast<size_t>(vbr_operation_class::count)> classes = {};
    std::array<bool, static_cast<size_t>(vbr_operation_phase::count)> phases = {};

    for (const auto & registration : VBR_MUTATION_REGISTRY) {
        registrants[static_cast<size_t>(registration.registrant)] = true;
        families[static_cast<size_t>(registration.family)] = true;
        phases[static_cast<size_t>(registration.phase)] = true;
        for (size_t i = 0; i < classes.size(); ++i) {
            if ((registration.allowed_classes & (uint16_t(1u) << i)) != 0) {
                classes[i] = true;
            }
        }
    }
    for (bool present : registrants) {
        if (!present) {
            return false;
        }
    }
    for (bool present : families) {
        if (!present) {
            return false;
        }
    }
    for (bool present : classes) {
        if (!present) {
            return false;
        }
    }
    // Root is the operation owner rather than a mutation site; all mutation phases must occur.
    for (size_t i = 1; i < phases.size(); ++i) {
        if (!phases[i]) {
            return false;
        }
    }
    return true;
}

// VBR_MUTATION_INVENTORY_EXHAUSTIVE
static_assert(vbr_mutation_registry_is_exhaustive(),
        "VBR mutation registrants, families, classes, and phases must remain closed and exhaustive");

enum class vbr_stable_read_registrant : uint8_t {
    checkpoint_capture,
    state_export,
    oracle_read,
    count,
};

// Stable readers are registry participants but do not stamp a mutation family.
// VBR_STABLE_READ_INVENTORY_EXHAUSTIVE
constexpr std::array<vbr_stable_read_registrant,
        static_cast<size_t>(vbr_stable_read_registrant::count)> VBR_STABLE_READ_REGISTRY = {{
    vbr_stable_read_registrant::checkpoint_capture,
    vbr_stable_read_registrant::state_export,
    vbr_stable_read_registrant::oracle_read,
}};
constexpr bool vbr_stable_read_registry_is_exhaustive() {
    std::array<bool, static_cast<size_t>(vbr_stable_read_registrant::count)> present = {};
    for (vbr_stable_read_registrant registrant : VBR_STABLE_READ_REGISTRY) {
        present[static_cast<size_t>(registrant)] = true;
    }
    for (bool registered : present) {
        if (!registered) {
            return false;
        }
    }
    return true;
}
static_assert(vbr_stable_read_registry_is_exhaustive(),
        "capture, export, and oracle stable-read guards must stay exhaustive");

// Extent-entry family for an operation, derived from its authenticated kind/class rather
// than caller-supplied strings. swa_wrap is the one class whose provenance family is
// occupied_reuse regardless of kind (§5.5 row 2).
constexpr vbr_mutation_family vbr_operation_kind_family(vbr_operation_kind  kind,
                                                        vbr_operation_class operation_class =
                                                                vbr_operation_class::ordinary_decode) {
    return operation_class == vbr_operation_class::swa_wrap ? vbr_mutation_family::occupied_reuse
         : kind == vbr_operation_kind::decode              ? vbr_mutation_family::append
         : kind == vbr_operation_kind::sequence_edit       ? vbr_mutation_family::trim
         : kind == vbr_operation_kind::checkpoint_restore  ? vbr_mutation_family::restore
         : kind == vbr_operation_kind::state_import        ? vbr_mutation_family::import
         : kind == vbr_operation_kind::controller_retier   ? vbr_mutation_family::degrade
         : kind == vbr_operation_kind::recovery            ? vbr_mutation_family::recovery
                                                           : vbr_mutation_family::clear;
}

// Closed registrant masks per operation kind: the manifest declares which registrants an
// operation may authorize; begin_event checks membership. Derived once here, never per-site.
constexpr uint32_t vbr_registrant_bit(vbr_mutation_registrant registrant) {
    return uint32_t(1u) << static_cast<uint8_t>(registrant);
}
constexpr uint32_t vbr_operation_kind_registrants(vbr_operation_kind kind) {
    return kind == vbr_operation_kind::decode
               ? vbr_registrant_bit(vbr_mutation_registrant::apply_ubatch_append) |
                 vbr_registrant_bit(vbr_mutation_registrant::apply_ubatch_occupied_reuse) |
                 vbr_registrant_bit(vbr_mutation_registrant::seq_rm)  // composite purge (§7.3)
         : kind == vbr_operation_kind::sequence_edit
               ? vbr_registrant_bit(vbr_mutation_registrant::seq_rm) |
                 vbr_registrant_bit(vbr_mutation_registrant::seq_cp) |
                 vbr_registrant_bit(vbr_mutation_registrant::seq_keep) |
                 vbr_registrant_bit(vbr_mutation_registrant::seq_add) |
                 vbr_registrant_bit(vbr_mutation_registrant::seq_div) |
                 vbr_registrant_bit(vbr_mutation_registrant::clear) |
                 vbr_registrant_bit(vbr_mutation_registrant::full_reset)
         : kind == vbr_operation_kind::checkpoint_restore
               ? vbr_registrant_bit(vbr_mutation_registrant::seq_rm) |
                 vbr_registrant_bit(vbr_mutation_registrant::state_read_meta) |
                 vbr_registrant_bit(vbr_mutation_registrant::state_read_data) |
                 vbr_registrant_bit(vbr_mutation_registrant::state_read_install) |
                 vbr_registrant_bit(vbr_mutation_registrant::state_read_cleanup) |
                 vbr_registrant_bit(vbr_mutation_registrant::explicit_restore_adopt)
         : kind == vbr_operation_kind::state_import
               // seq_rm belongs here for the same reason it leads checkpoint_restore's mask:
               // the per-seq branch of state_read_meta clears the destination sequence as its
               // first act. Missing bit = GGML_ASSERT(event) on any per-seq import with the
               // generation tracker armed (pre-existing on master since the checkpoint merge;
               // exposed by test-state-restore-fragmented in the 2026-08-09 sync gate).
               ? vbr_registrant_bit(vbr_mutation_registrant::seq_rm) |
                 vbr_registrant_bit(vbr_mutation_registrant::state_read_meta) |
                 vbr_registrant_bit(vbr_mutation_registrant::state_read_data) |
                 vbr_registrant_bit(vbr_mutation_registrant::state_read_install) |
                 vbr_registrant_bit(vbr_mutation_registrant::state_read_cleanup) |
                 vbr_registrant_bit(vbr_mutation_registrant::whole_import) |
                 vbr_registrant_bit(vbr_mutation_registrant::clear) |
                 vbr_registrant_bit(vbr_mutation_registrant::full_reset)
         : kind == vbr_operation_kind::controller_retier
               ? vbr_registrant_bit(vbr_mutation_registrant::degrade_next) |
                 vbr_registrant_bit(vbr_mutation_registrant::promote_next) |
                 vbr_registrant_bit(vbr_mutation_registrant::execute_shed)
         : kind == vbr_operation_kind::recovery
               ? ~uint32_t(0)  // target-subset-restricted by the capability instead
               : 0;
}

// Raw public-API range sentinels (-1) normalize to canonical [0, max) before any
// manifest is built, so the closed mint range rules below see canonical values — the ONE
// spelling of that clamp (recovery/controller_retier legitimately keep {-1,-1} and never
// route through it).
inline void vbr_normalize_edit_range(llama_pos & p0, llama_pos & p1) {
    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }
}

// Closed per-kind range enumeration for mutate-phase targets. Range-bearing kinds
// require a nonempty canonical p0 < p1. The two special forms are ENUMERATED, never generic:
//   {-1,-1} wildcard -> recovery (capability-subset-restricted afterwards) and
//                       controller_retier (unit-level representation ops carry no cell range)
//   p0 == p1 empty   -> sequence_edit only (the public seq API passes empty no-op ranges
//                       through; refusing them would latch unavailable on a harmless call)
constexpr bool vbr_target_range_valid(vbr_operation_kind kind, vbr_operation_range range) {
    if (range.p0 < 0 || range.p1 < 0) {
        return range.p0 == -1 && range.p1 == -1 &&
               (kind == vbr_operation_kind::recovery ||
                kind == vbr_operation_kind::controller_retier);
    }
    if (range.p0 == range.p1) {
        return kind == vbr_operation_kind::sequence_edit;
    }
    return range.p0 < range.p1;
}

// One construction rule for mutation-shaped bindings (child_phase = mutate is the stated
// convention, not a per-site choice). Used by the cache scope, wrapper adoption, and tests.
// The class and registrant mask are authenticated into the manifest here.
constexpr vbr_operation_target vbr_make_target(vbr_operation_kind  kind,
                                               vbr_operation_class operation_class,
                                               vbr_controller_instance_id instance,
                                               uint16_t stream, llama_seq_id seq_id,
                                               llama_pos p0, llama_pos p1) {
    vbr_operation_target target;
    target.instance_id     = instance;
    target.operation_class = operation_class;
    target.registrant_mask = vbr_operation_kind_registrants(kind);
    target.child_phase     = vbr_operation_phase::mutate;
    target.stream          = stream;
    target.seq_id          = seq_id;
    target.range           = { p0, p1 };
    return target;
}

// Appends one mutate target for an ARMED controller; an empty instance appends nothing and a
// full manifest refuses. The ONE spelling of "one exact-instance target per armed child" used by
// the composite wrappers and the cross-cache shed root.
inline bool vbr_binding_add_instance_target(vbr_operation_binding & binding,
                                            vbr_operation_kind  kind,
                                            vbr_operation_class operation_class,
                                            vbr_controller_instance_id instance,
                                            uint16_t stream, llama_seq_id seq_id,
                                            llama_pos p0, llama_pos p1) {
    if (!vbr_controller_instance_id_is_set(instance) ||
        binding.n_targets >= vbr_operation_binding::MAX_TARGETS) {
        return false;
    }
    binding.targets[binding.n_targets++] =
            vbr_make_target(kind, operation_class, instance, stream, seq_id, p0, p1);
    return true;
}

constexpr vbr_operation_binding vbr_mutation_binding(vbr_operation_kind  kind,
                                                     llama_seq_id        seq_id,
                                                     llama_pos           p0,
                                                     llama_pos           p1,
                                                     vbr_operation_class operation_class =
                                                             vbr_operation_class::state_api,
                                                     vbr_controller_instance_id instance = {},
                                                     uint16_t stream  = 0) {
    vbr_operation_binding binding;
    binding.kind        = kind;
    binding.child_phase = vbr_operation_phase::mutate;
    binding.n_targets   = 1;
    binding.targets[0]  = vbr_make_target(kind, operation_class, instance,
                                          stream, seq_id, p0, p1);
    return binding;
}

// The sole process-global minting entry point. Composite memories must only forward its result.
vbr_operation_id vbr_operation_registry_begin(vbr_operation_binding & binding);
bool vbr_operation_registry_end(vbr_operation_id operation_id);
bool vbr_operation_registry_is_live(vbr_operation_id operation_id);
// Re-arm capacity probe: true when the bounded live-operation registry has at
// least one free slot, checked under the registry mutex (boundary-rate only).
bool vbr_operation_registry_has_capacity();

// The registry retains the immutable binding while the operation is live, so mutation
// events and extent entries can be validated against the authenticated (registrant-checked)
// kind/seq/range instead of caller-supplied values. Returns false once the operation ended.
bool vbr_operation_registry_binding(vbr_operation_id operation_id, vbr_operation_binding & out);

// Read-only stable-reader probe. It scans the fixed registry under the
// existing mutex and never allocates or mints an operation id.
bool vbr_operation_registry_quiescent_for(
    const vbr_controller_instance_id * instances,
    size_t n_instances) noexcept;
bool vbr_operation_registry_quiescent_for_except(
    const vbr_controller_instance_id * instances,
    size_t n_instances,
    vbr_operation_id allowed) noexcept;

// Explicit operation close semantics. `vbr_operation_registry_end` remains the
// committed-close alias for the non-mutating legacy callers (freeze scopes).
enum class vbr_operation_outcome : uint8_t {
    committed,
    aborted,
    failed,
};
bool vbr_operation_registry_close(vbr_operation_id operation_id, vbr_operation_outcome outcome);

// ---------------------------------------------------------------------------
// Authenticated operation recovery: registry-owned failed-operation records with a
// monotone state machine and single-use, target-restricted capabilities. Records are RESERVED
// at operation begin for every fence-spanning/destructive operation — before any potentially
// observable mutation — so an odd controller serial can never exist without an authenticated
// resolution path. Ring exhaustion at reserve time makes generation tracking for that
// operation shadow-unavailable; the legacy mutation proceeds untouched.
// ---------------------------------------------------------------------------

enum class vbr_recovery_state : uint8_t {
    free_slot,
    reserved,             // reserved at op begin; released unused at clean commit
    recorded,             // operation terminated without commit
    capability_minted,    // single mint consumed the record's mint right
    awaiting_ack,         // Quarantined: retains targets until the owning tracker acknowledges.
};

enum class vbr_recovery_failure_site : uint8_t {
    none,
    metadata_mutation,
    deferred_byte_copy,
    publication,
    exception_unwind,
};

struct vbr_failed_operation_record {
    vbr_operation_binding     binding;
    vbr_operation_phase       phase_reached         = vbr_operation_phase::root;
    vbr_recovery_failure_site failure_site          = vbr_recovery_failure_site::none;
    bool                      dest_bytes_observable = false;
    vbr_recovery_state        state                 = vbr_recovery_state::free_slot;
    // The reservation's immutable owner instance: takes and advances match this,
    // never the (possibly composite) manifest, so the base child can never service the SWA
    // child's failure.
    vbr_controller_instance_id owner_instance       = {};
    // In-service marker: a taken record cannot be retaken, and only the
    // taking instance's ack (validated below) reclaims it.
    bool                      taken                 = false;
    vbr_controller_instance_id taker_instance       = {};
    uint16_t                  src_stream            = 0;
    uint16_t                  dst_stream            = 0;
    vbr_operation_range       src_range             = {};
    // Source-stability token: the source stream's generation tracker page generations
    // over the copied source range, captured at reserve. Bounded (<= pages * 4 B).
    std::vector<uint32_t>     src_page_gens;
};

// Reserve/release/record. reserve returns a negative index when the ring is exhausted.
int32_t vbr_recovery_reserve(vbr_operation_id operation_id,
                             vbr_controller_instance_id owner_instance = {});
int32_t vbr_recovery_reserve(const vbr_operation_binding & binding,
                             vbr_controller_instance_id owner_instance = {});
bool    vbr_recovery_release_unused(int32_t record_index, vbr_operation_id operation_id);
bool    vbr_recovery_record_failure(int32_t                   record_index,
                                    vbr_operation_id          operation_id,
                                    vbr_operation_phase       phase_reached,
                                    vbr_recovery_failure_site failure_site,
                                    bool                      dest_bytes_observable);
bool    vbr_recovery_set_source_token(int32_t                       record_index,
                                      vbr_operation_id              operation_id,
                                      uint16_t                      src_stream,
                                      uint16_t                      dst_stream,
                                      vbr_operation_range           src_range,
                                      const std::vector<uint32_t> & src_page_gens);
bool    vbr_recovery_get_record(int32_t record_index, vbr_failed_operation_record & out);

// Single-use capability. Only mintable from a `recorded` entry (callers cannot construct one:
// §1.7 "cannot self-declare"). Destruction without an explicit resolve fail-closes the record
// to resolved_quarantined and latches a pending-quarantine flag the owner MUST consume by
// performing the tracker global invalidation (asserted by tests + CI).
class vbr_recovery_capability {
  public:
    ~vbr_recovery_capability();

    vbr_recovery_capability(const vbr_recovery_capability &)             = delete;
    vbr_recovery_capability & operator=(const vbr_recovery_capability &) = delete;
    vbr_recovery_capability(vbr_recovery_capability && other) noexcept;
    vbr_recovery_capability & operator=(vbr_recovery_capability &&) = delete;

    explicit operator bool() const { return record_index_ >= 0; }

    // Target-subset validation: every recovery mutation must fall inside the recorded
    // (stream, seq, range). Out-of-subset is a hard reject; the record stays unresolved.
    bool target_allowed(uint16_t stream, llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;

    bool resolve_completed();
    bool resolve_quarantined();

  private:
    friend vbr_recovery_capability vbr_recovery_mint(int32_t record_index);
    vbr_recovery_capability() = default;

    int32_t record_index_ = -1;
};

vbr_recovery_capability vbr_recovery_mint(int32_t record_index);

// Tokenized invalidate-then-ack quarantine on the fixed ring. A
// quarantined record (explicit resolve_quarantined or fail-closed capability destruction)
// transitions to awaiting_ack RETAINING its manifest targets. The owning tracker's cache takes
// the pending quarantine for ITS instance at the next decode boundary, performs the target/global
// invalidation, and acks with the token — only the ack reclaims the slot. Multi-controller
// failures occupy distinct ring slots; nothing collapses to a process-global bit.
struct vbr_quarantine_token {
    int32_t  record_index = -1;
    uint64_t nonce        = 0;

    explicit operator bool() const { return record_index >= 0 && nonce != 0; }
};

struct vbr_quarantine_work {
    vbr_quarantine_token  token;
    // Retained manifest targets. The current consumer performs a GLOBAL invalidation (a safe
    // over-approximation of the per-target requirement); target-scoped invalidation can
    // consume these when a finer path exists.
    vbr_operation_binding binding;
};

// Take one pending quarantine owned by `instance` (or a legacy wildcard-owner record).
// Returns an empty optional-like work item (token false) when none is serviceable.
vbr_quarantine_work vbr_recovery_take_quarantine(vbr_controller_instance_id instance);
// Production advancement of `recorded` failures transitions this instance's
// recorded slots to awaiting_ack (quarantine) so the take/ack drain resolves them. Returns
// the number advanced.
int32_t vbr_recovery_advance_recorded(vbr_controller_instance_id instance);
// Is any unresolved recovery work (recorded/awaiting/taken) serviceable here?
bool vbr_recovery_pending_for(vbr_controller_instance_id instance);
// Import opens its authenticated recovery reservation before the final empty
// target recheck. Ignore only that operation's still-reserved record; every
// other non-free record, including wildcard recovery work, remains a conflict.
bool vbr_recovery_pending_for_except(
    vbr_controller_instance_id instance,
    vbr_operation_id allowed_reserved_operation);
// Retirement is narrower than service routing: a legacy wildcard record is serviceable
// by any controller, but it is not owned by every controller.
bool vbr_recovery_owned_by(vbr_controller_instance_id instance);
// Acknowledge with the token AFTER performing the invalidation; only the instance that took the
// record may acknowledge it; this reclaims the slot.
bool vbr_recovery_ack_quarantine(vbr_quarantine_token token,
                                 vbr_controller_instance_id instance);
// Release an un-serviced take (invalidation could not run): the record becomes takeable again
// at a later boundary instead of being stuck in-service.
bool vbr_recovery_untake_quarantine(vbr_quarantine_token token,
                                    vbr_controller_instance_id instance);

// The only generic mint-owning RAII. All operation minting outside the
// legacy freeze wrapper flows through this type, keeping registry_begin call sites confined to
// this translation unit. Close is outcome-coded; destruction without close commits.
class vbr_scoped_operation {
  public:
    explicit vbr_scoped_operation(vbr_operation_binding binding);
    ~vbr_scoped_operation();

    vbr_scoped_operation(const vbr_scoped_operation &)             = delete;
    vbr_scoped_operation & operator=(const vbr_scoped_operation &) = delete;
    vbr_scoped_operation(vbr_scoped_operation && other) noexcept;
    vbr_scoped_operation & operator=(vbr_scoped_operation &&) = delete;

    explicit operator bool() const { return static_cast<bool>(binding_.operation_id); }
    vbr_operation_id              id() const { return binding_.operation_id; }
    const vbr_operation_binding & binding() const { return binding_; }

    bool close(vbr_operation_outcome outcome);
    // Transfer ownership of the open operation for deferred decode lifetime: the
    // receiver becomes responsible for the outcome-coded close.
    vbr_operation_id release();

  private:
    vbr_operation_binding binding_ = {};
};

// Internal owner for operations whose call shape permits ordinary C++ lifetime management.
// The legacy public freeze begin/end ABI is manually paired at its C boundary, while its sole
// server caller is already protected by server_vbr_retier_freeze_scope.
class vbr_operation_registry_guard {
public:
    explicit vbr_operation_registry_guard(vbr_operation_binding binding) : op_(binding) {}

    vbr_operation_registry_guard(const vbr_operation_registry_guard &) = delete;
    vbr_operation_registry_guard & operator=(const vbr_operation_registry_guard &) = delete;
    vbr_operation_registry_guard(vbr_operation_registry_guard &&) = delete;
    vbr_operation_registry_guard & operator=(vbr_operation_registry_guard &&) = delete;

    bool active() const { return static_cast<bool>(op_); }

    const vbr_operation_binding & binding() const { return op_.binding(); }

    bool finish() { return op_.close(vbr_operation_outcome::committed); }

private:
    vbr_scoped_operation op_;
};
