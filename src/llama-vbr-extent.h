#pragma once

#include "llama-vbr-generation-types.h"
#include "llama-vbr-operation.h"

#include <array>
#include <cstdint>
#include <vector>

// Committed-extent store and dual-view ownership index.
//
// The extent store is the durable admission evidence that replaces the forbidden debug ring:
// every destructive/import mutation records its committed [p0,p1) extent here, and expected-
// tombstone classification consumes ONLY entries whose state reached `committed` at the
// operation family's success boundary. Entries are two-phase because stamps flow while the
// controller serial is odd (during mutation), before bytes are provably landed:
//
//   prepared  -> cited by stamps, never admission evidence
//   submitted -> async graph submitted (append/reuse families only), still not evidence
//   committed -> family success boundary crossed; sole evidence state
//   failed    -> aborted/failed operations; never evidence, reclaimable
//
// ABA safety: a stamp reference carries (index, expected_gen); a reference is valid only while
// the slot's generation matches. Slots are reused only at refcount zero with a generation
// bump; generations never wrap — approaching the limit performs a global-invalidation slab
// reset instead (all outstanding references become obsolete by construction).

enum class vbr_extent_state : uint8_t {
    free_slot,
    prepared,
    submitted,
    committed,
    failed,
};

struct vbr_extent_entry {
    vbr_mutation_family family          = vbr_mutation_family::append;
    vbr_operation_class operation_class = vbr_operation_class::ordinary_decode;
    vbr_extent_state    state           = vbr_extent_state::free_slot;
    uint8_t             pad0            = 0;
    uint16_t            stream          = 0;
    uint16_t            pad1            = 0;
    llama_seq_id        seq_id          = -1;
    llama_pos           p0              = -1;
    llama_pos           p1              = -1;
    uint32_t            slot_gen        = 0;
};
static_assert(sizeof(vbr_extent_entry) == 24, "extent entries are 24 B naturally aligned");

struct vbr_extent_ref {
    uint32_t index        = 0;
    uint32_t expected_gen = 0;

    explicit operator bool() const { return expected_gen != 0; }
};

// Opaque handle held by the mutation event between reserve and resolution.
struct vbr_extent_handle {
    uint32_t index        = 0;
    uint32_t expected_gen = 0;

    explicit operator bool() const { return expected_gen != 0; }
};

class vbr_extent_store {
  public:
    static constexpr uint32_t CAPACITY = 8192;

    vbr_extent_store();

    vbr_extent_store(const vbr_extent_store &)             = delete;
    vbr_extent_store & operator=(const vbr_extent_store &) = delete;

    // Allocation writes the full entry as `prepared`. Returns an empty handle on exhaustion —
    // the caller must then take the invalidate-before-mutate path (availability transition +
    // qualification reset) and MAY call reset_all() once every outstanding reference is
    // obsolete by that same global invalidation.
    vbr_extent_handle reserve(vbr_mutation_family family,
                              vbr_operation_class operation_class,
                              uint16_t            stream,
                              llama_seq_id        seq_id,
                              llama_pos           p0,
                              llama_pos           p1);

    // Family-boundary transitions. submit() is only legal from prepared (async append/reuse);
    // commit() from prepared or submitted; fail() from any non-free state.
    bool submit(vbr_extent_handle handle);
    bool commit(vbr_extent_handle handle);
    bool fail(vbr_extent_handle handle);

    // Reference accounting for cell stamps. add_ref validates the handle before counting.
    vbr_extent_ref add_ref(vbr_extent_handle handle);
    void           release_ref(vbr_extent_ref ref);

    // Admission lookup: returns the entry ONLY when the reference generation matches and the
    // entry reached `committed`. Everything else classifies as unknown at the evaluator.
    const vbr_extent_entry * lookup_committed(vbr_extent_ref ref) const;

    // Global-invalidation slab reset (exhaustion recovery / pre-wrap). The caller must have
    // performed the tracker global invalidation FIRST so every stored reference is already
    // obsolete; this then reclaims all slots and bumps generations past every outstanding ref.
    void reset_all();

    bool     exhausted_latched() const { return exhausted_; }
    uint32_t live_entries() const;

  private:
    bool handle_valid(vbr_extent_handle handle) const;
    bool slot_matches(uint32_t index, uint32_t expected_gen) const;
    void maybe_reclaim(uint32_t index);

    std::vector<vbr_extent_entry> entries_;
    std::vector<uint32_t>         refcounts_;
    std::vector<uint32_t>         free_list_;
    bool                          exhausted_ = false;
};

// Dual-view ownership index: physical per-(stream,seq) page masks for
// canonical dependency-set enumeration + a lazily allocated per-active-seq logical-position
// Fenwick tree for exact rank-below-frontier. Positions outside [0, n_cells) mark the
// (stream,seq) view unavailable (fail-closed shadow-unavailable; legacy never consults this).
class vbr_ownership_index {
  public:
    // One page geometry for the whole generation subsystem: capture assumes the
    // index's pages match the tracker's, so the constants must be the same symbols.
    static constexpr uint32_t MASK_WORDS_PER_PAGE = VBR_GENERATION_MASK_WORDS;

    vbr_ownership_index(uint32_t n_stream, uint32_t n_seq_max, uint32_t n_cells);
    ~vbr_ownership_index();  // out-of-line: views_ holds an incomplete type here

    vbr_ownership_index(const vbr_ownership_index &)             = delete;
    vbr_ownership_index & operator=(const vbr_ownership_index &) = delete;

    // Transactional updates — called from the SAME registrant transactions that stamp the
    // generation tracker. `pos` is the cell's logical position at the time of the change.
    bool add_cell(uint32_t stream, llama_seq_id seq_id, uint32_t cell, llama_pos pos);
    bool remove_cell(uint32_t stream, llama_seq_id seq_id, uint32_t cell, llama_pos pos);
    // Position change (seq_add/seq_div): remove at old_pos + add at new_pos as one call so a
    // shift that leaves the domain marks unavailable exactly once.
    bool move_cell(uint32_t stream, llama_seq_id seq_id, uint32_t cell, llama_pos old_pos, llama_pos new_pos);
    void clear_seq(uint32_t stream, llama_seq_id seq_id);
    void clear_all();

    // Exact number of owned cells with position < frontier. Returns false (rank invalid) when
    // the view is unavailable — capture must then mark generation shadow-unavailable.
    bool rank_below(uint32_t stream, llama_seq_id seq_id, llama_pos frontier, uint32_t & rank) const;

    // Canonical enumeration for capture: appends ALL owned cells in physical page order; the
    // caller applies its own position-<-frontier filter (positions live in llama_kv_cells).
    // False when the view is unavailable.
    bool enumerate_owned(uint32_t stream, llama_seq_id seq_id, std::vector<uint32_t> & cells) const;

    bool available(uint32_t stream, llama_seq_id seq_id) const;
    // Diagnostics distinguish a view that was never populated from one that
    // was populated and later failed the bounded logical-position contract.
    bool initialized(uint32_t stream, llama_seq_id seq_id) const;

    // Memory note: the Fenwick domain is n_cells positions * 4 bytes,
    // ~800 KB per ACTIVE seq at 200k cells, lazily allocated on first add; capacity is
    // RETAINED at clear_seq for seq-id reuse. Masks cost pages * 32 B per active seq.

  private:
    struct seq_view;
    seq_view *       find_view(uint32_t stream, llama_seq_id seq_id);
    const seq_view * find_view(uint32_t stream, llama_seq_id seq_id) const;
    seq_view &       obtain_view(uint32_t stream, llama_seq_id seq_id);

    uint32_t n_stream_  = 0;
    uint32_t n_seq_max_ = 0;
    uint32_t n_cells_   = 0;
    uint32_t n_pages_   = 0;

    // Flat O(1) map: slot = stream * n_seq_max + seq_id. Views are lazily
    // initialized and keep their vector capacity across clear_seq for seq-id reuse.
    std::vector<seq_view> views_;
};
