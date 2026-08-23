#pragma once

#include "llama-vbr-extent.h"
#include "llama-vbr-generation-types.h"
#include "llama-vbr-operation.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <memory>
#include <vector>

struct vbr_artifact_stream_placement;
struct vbr_tracker_install_child;
struct vbr_generation_stream_state;
class vbr_occupied_replacement_guard;

class vbr_tracker_import_image {
  public:
    vbr_tracker_import_image();
    ~vbr_tracker_import_image();
    vbr_tracker_import_image(vbr_tracker_import_image &&) noexcept;
    vbr_tracker_import_image & operator=(vbr_tracker_import_image &&) noexcept;
    vbr_tracker_import_image(const vbr_tracker_import_image &) = delete;
    vbr_tracker_import_image & operator=(const vbr_tracker_import_image &) = delete;

    bool ready() const noexcept;
    bool stable() const noexcept;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
    friend class vbr_generation_tracker;
};

struct vbr_unit_generation {
    uint64_t            repr_gen         = 0;
    uint64_t            publish_seq      = 0;
    int32_t             current_type     = -1;
    int32_t             last_source_type = -1;
    vbr_repr_domain     domain           = vbr_repr_domain::full;
    uint8_t             promote_hops     = 0;
    vbr_repr_transition last_transition  = vbr_repr_transition::initial;
    uint8_t             flags            = 0;
};

enum class vbr_generation_stamp_kind : uint8_t {
    dependency,
    membership,
};

class vbr_generation_tracker;

enum class vbr_generation_teardown_state : uint8_t {
    clean,
    active_event,
    operation_live,
    recovery_owned,
    instance_owner_mismatch,
    _count,
};

const char * vbr_generation_teardown_state_name(
    vbr_generation_teardown_state state) noexcept;

// P1v2 (v6): per-target lazy extent supplier. The tracker calls back into the owning mutation
// scope at each destructive stamp with the SELECTED manifest-target index; the scope reserves
// that target's extent on first use and returns it (empty on reservation failure — the scope
// then latches unavailable). Damage evidence therefore binds to the selected target's own
// extent, never a scope-global one chosen before the per-cell target was known.
using vbr_event_extent_fn = vbr_extent_handle (*)(void * ctx, uint8_t target_index);

struct vbr_generation_event {
    vbr_generation_event() = default;
    ~vbr_generation_event();

    vbr_generation_event(const vbr_generation_event &)             = delete;
    vbr_generation_event & operator=(const vbr_generation_event &) = delete;
    vbr_generation_event(vbr_generation_event && other) noexcept;
    vbr_generation_event & operator=(vbr_generation_event &&) = delete;

    bool finish();

    explicit operator bool() const { return serial != 0; }

  private:
    friend class vbr_generation_tracker;
    uint64_t                  serial          = 0;
    uint32_t                  stream          = 0;
    vbr_generation_stamp_kind stamp_kind      = vbr_generation_stamp_kind::dependency;
    vbr_mutation_family       family          = vbr_mutation_family::append;
    vbr_operation_class       operation_class = vbr_operation_class::ordinary_decode;
    bool                      destructive     = false;
    bool                      imported        = false;
    vbr_operation_id          operation_id    = {};
    // P1v2 (v6): the full authenticated manifest copy (bounded; events are boundary-rate).
    // Covering-target selection happens PER STAMP against it, keyed by the stamped
    // (seq, pre-mutation position) — the single-cached-target model is gone.
    vbr_operation_binding     manifest        = {};
    uint32_t                  registrant_bit  = 0;
    // P1v2: a stamp with no covering target poisons the event — the tracker latches
    // shadow-unavailable IMMEDIATELY and every further stamp from this event is inert.
    bool                      poisoned        = false;
    vbr_event_extent_fn       extent_fn       = nullptr;
    void *                    extent_ctx      = nullptr;
    vbr_generation_tracker *  owner_          = nullptr;
};

// Armed-only dual-write store. The raw arrays stay private behind value-returning accessors.
// Mutation sites use the closed A0 registrant table
// through begin_event(); no caller supplies an open-ended family string.
class vbr_generation_tracker {
  public:
    vbr_generation_tracker(uint32_t n_stream, uint32_t n_cells, uint32_t n_units,
                           vbr_lineage_uuid lineage = {});
    ~vbr_generation_tracker();

    vbr_generation_tracker(const vbr_generation_tracker &)             = delete;
    vbr_generation_tracker & operator=(const vbr_generation_tracker &) = delete;
    vbr_generation_tracker(vbr_generation_tracker &&)                  = delete;
    vbr_generation_tracker & operator=(vbr_generation_tracker &&)      = delete;

    bool     active() const;
    uint32_t stream_count() const;

    // C4/F2 (v3 design): persistent shadow-unavailable state. While set, events are inert,
    // capture is unavailable, and qualification cannot advance.
    bool shadow_unavailable() const { return shadow_unavailable_; }
    // P4v2 (v6): the latch records the controller generation at latch time so clearing can
    // demand a MONOTONE proof — a sanctioned global transition that provably happened
    // strictly AFTER the latch (and therefore after the untracked legacy interval following
    // it). Re-latching keeps the newest token.
    void set_shadow_unavailable() {
        shadow_unavailable_  = true;
        generation_at_latch_ = generation_at_latch_ > global_generation_
                                       ? generation_at_latch_ : global_generation_;
    }
    // P4v2 (v6): the WHOLE availability-creating transition in one place — while latched (and
    // only then) it proves an empty recovery ring for this instance across every non-free state
    // and registry capacity, performs the sanctioned global invalidation (the named exemption
    // that runs outside any operation), and clears the monotone latch that invalidation
    // provably post-dates. Boundaries just call this; the ordering cannot be re-derived
    // wrongly at a second call site. Returns true when armed (already or newly).
    bool try_rearm();

    // Clears ONLY when the causes are provably gone: the extent slab is healthy AND the
    // controller generation advanced strictly past the latch token (a clean global
    // transition followed the untracked interval). try_rearm() is the caller-facing wrapper
    // that additionally proves registry capacity and an empty recovery ring first.
    bool try_clear_shadow_unavailable() {
        if (!shadow_unavailable_) {
            return true;
        }
        if (extents_.exhausted_latched() || global_generation_ <= generation_at_latch_) {
            return false;
        }
        shadow_unavailable_ = false;
        return true;
    }
    uint32_t cell_count() const;
    uint32_t unit_count() const;
    bool     stable() const;
    // The destructor remains the enforcing boundary.  This read-only view lets
    // transaction terminals and tests prove the same predicate before a
    // successfully imported controller is handed back to decode.
    vbr_generation_teardown_state teardown_state() const noexcept;

    vbr_lineage_uuid            lineage_identity() const;
    vbr_controller_instance_id  runtime_instance() const;
    uint64_t                    controller_generation() const;
    // F9: value accessor for the evaluator's post-read stability recheck. Even = stable.
    uint64_t      mutation_serial() const { return mutation_serial_; }

    // A2: every mutation event cites the live operation it belongs to; the tracker validates
    // the citation against the registry-retained binding and copies the authenticated
    // manifest into the event. For provenance-bearing (destructive/import) families the
    // event carries the owning scope's per-target extent supplier — stamps then bind durable
    // ABA-safe references to the SELECTED target's extent; a null supplier is legal for
    // ordinary families.
    vbr_generation_event begin_event(vbr_mutation_registrant   registrant,
                                     vbr_operation_class       operation_class,
                                     uint32_t                  stream,
                                     vbr_generation_stamp_kind stamp_kind,
                                     vbr_operation_id          operation_id,
                                     vbr_event_extent_fn       extent_fn   = nullptr,
                                     void *                    extent_ctx  = nullptr,
                                     bool                      destructive = false,
                                     bool                      imported    = false);
    bool stamp_cell(vbr_generation_event & event, uint32_t cell,
                    llama_seq_id membership_seq = -1, llama_pos pre_mutation_pos = -1);
    // P1v2 (v6): shared-cell form — a covering target must exist for EVERY member of the
    // cell's sequence set (target-set proof); absent that, the stamp poisons the event.
    bool stamp_cell(vbr_generation_event & event, uint32_t cell,
                    const llama_seq_id * seqs, int32_t n_seqs, llama_pos pre_mutation_pos);

    // A2 extent store owned by the tracker so reference lifecycle stays co-located with the
    // stamps that cite it. Exposed only to the kv-cache operation scope and the evaluator.
    vbr_extent_store &       extent_store()       { return extents_; }
    const vbr_extent_store & extent_store() const { return extents_; }

    // Extent references for the evaluator (raw comparisons stay in the evaluator TU; these are
    // value accessors like the generation getters above).
    vbr_extent_ref dependency_extent(uint32_t stream, uint32_t cell) const;
    vbr_extent_ref membership_extent(uint32_t stream, uint32_t cell) const;
    bool           dependency_in_range(uint32_t stream, uint32_t cell) const;
    bool           membership_in_range(uint32_t stream, uint32_t cell) const;

    // A2 exhaustion/no-wrap recovery: global invalidation + extent slab reset as one action
    // (design Rev 4 items 2-3). Registrant-validated like global_transition.
    bool global_invalidate_and_reset_extents(vbr_mutation_registrant registrant,
                                             vbr_operation_class     operation_class,
                                             vbr_operation_id        operation_id = {});

    bool global_transition(vbr_mutation_registrant registrant, vbr_operation_class operation_class,
                           vbr_operation_id operation_id = {});
    bool initialize_unit(uint32_t unit, int32_t type, vbr_repr_domain domain);
    // P5v2 (v6): publication cites the EXACT registrant driving it — never an OR of
    // plausible ones; a cited operation's manifest must cover that single bit.
    bool publish_unit(uint32_t                unit,
                      int32_t                 source_type,
                      int32_t                 target_type,
                      vbr_repr_domain         domain,
                      uint8_t                 promote_hops,
                      vbr_repr_transition     transition,
                      vbr_mutation_registrant registrant,
                      vbr_operation_id        operation_id = {});

    // F4 import is built completely off-side. Preparation may allocate; the
    // final swap is allocation-free and preserves this tracker's enrolled
    // process-local runtime instance.
    bool prepare_import_image(
        const vbr_tracker_install_child & plan,
        const vbr_checkpoint_generation_controller & source,
        llama_seq_id destination,
        const std::vector<vbr_artifact_stream_placement> & placements,
        vbr_tracker_import_image & output) noexcept;
    // Occupied replacement already owns a canonical, strictly ordered cell
    // map.  Consume it in place instead of cloning a million-cell placement
    // and rebuilding uniqueness through a node-allocating set.
    bool prepare_relocated_import_image(
        const vbr_tracker_install_child & plan,
        const vbr_checkpoint_generation_controller & source,
        llama_seq_id destination,
        const vbr_occupied_replacement_guard & replacement,
        vbr_tracker_import_image & output) noexcept;
    bool import_image_installable(
        const vbr_tracker_import_image & image,
        vbr_operation_id operation_id) const noexcept;
    void install_import_image_swap(
        vbr_tracker_import_image & image) noexcept;

    // Read-only accessors are intentionally value-returning. Callers may capture them, but
    // mutation still goes exclusively through authenticated tracker events and imports.
    uint32_t            page_generation(uint32_t stream, uint32_t page) const;
    uint32_t            page_destructive_generation(uint32_t stream, uint32_t page) const;
    uint32_t            page_import_generation(uint32_t stream, uint32_t page) const;
    uint32_t            dependency_generation(uint32_t stream, uint32_t cell) const;
    uint32_t            membership_generation(uint32_t stream, uint32_t cell) const;
    uint16_t            dependency_provenance(uint32_t stream, uint32_t cell) const;
    uint16_t            membership_provenance(uint32_t stream, uint32_t cell) const;
    llama_seq_id        last_membership_seq(uint32_t stream, uint32_t cell) const;
    vbr_unit_generation unit_generation(uint32_t unit) const;

  private:
    bool prepare_import_image_impl(
        const vbr_tracker_install_child & plan,
        const vbr_checkpoint_generation_controller & source,
        llama_seq_id destination,
        const std::vector<vbr_artifact_stream_placement> * placements,
        const vbr_occupied_replacement_guard * replacement,
        vbr_tracker_import_image & output) noexcept;
    friend struct vbr_generation_event;
    bool reset_page_generations_before_wrap();
    bool reset_unit_generations_before_wrap();
    bool finish_event(uint64_t serial);

    vbr_lineage_uuid                      lineage_uuid_       = {};
    vbr_controller_instance_id            instance_id_        = {};
    bool                                  instance_enrolled_  = false;
    bool                                  shadow_unavailable_ = false;
    // P4v2 (v6): monotone latch token — controller generation at the newest latch.
    uint64_t                              generation_at_latch_ = 0;
    // F9 (v3.2 pin 2): unit tuples publish/read under this lock; the publish_seq parity/
    // equality checks remain as cheap in-lock assertions. Boundary-rate only, never per-token.
    mutable std::mutex                    units_mutex_;
    vbr_extent_store                      extents_;
    uint64_t                              global_generation_  = 1;
    uint64_t                              mutation_serial_    = 0;
    uint64_t                              event_serial_       = 0;
    uint32_t                              active_event_depth_ = 0;
    static constexpr uint32_t             MAX_EVENT_DEPTH     = 64;
    std::array<uint64_t, MAX_EVENT_DEPTH> active_event_stack_ = {};
    uint32_t                              n_cells_            = 0;
    std::vector<vbr_generation_stream_state> streams_;
    std::vector<vbr_unit_generation>      units_;
};

// Production capture consumes a controller-owned exact dependency index and passes its canonical
// physical cells here. This helper never discovers dependencies by scanning the cache; the
// independent oracle has a separate implementation and trust boundary.
bool vbr_generation_capture_stream(const vbr_generation_tracker &     tracker,
                                   uint32_t                           stream,
                                   llama_seq_id                       dependency_seq_id,
                                   llama_pos                          computation_frontier,
                                   const std::vector<uint32_t> &      canonical_dependency_cells,
                                   vbr_checkpoint_generation_stream & output);

bool vbr_generation_capture_controller(const vbr_generation_tracker &                        tracker,
                                       uint32_t                                              child_id,
                                       checkpoint_child_dependency_mode                      dependency_mode,
                                       const std::vector<vbr_checkpoint_generation_stream> & streams,
                                       vbr_checkpoint_generation_controller &                output);
