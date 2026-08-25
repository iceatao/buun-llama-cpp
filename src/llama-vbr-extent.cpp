#include "llama-vbr-extent.h"
#include "llama-bit-ops.h"

#include <algorithm>
#include <limits>
#include <new>

// ---------------------------------------------------------------------------
// vbr_extent_store
// ---------------------------------------------------------------------------

vbr_extent_store::vbr_extent_store() {
    entries_.resize(CAPACITY);
    refcounts_.resize(CAPACITY, 0);
    free_list_.reserve(CAPACITY);
    // reset_all() from gen 0 yields gen 2 everywhere (even = resolved): a default-constructed
    // ref/handle (expected_gen 0) can never validate. One slab-init spelling.
    reset_all();
}

bool vbr_extent_store::handle_valid(vbr_extent_handle handle) const {
    return slot_matches(handle.index, handle.expected_gen) &&
           entries_[handle.index].state != vbr_extent_state::free_slot;
}

bool vbr_extent_store::slot_matches(uint32_t index, uint32_t expected_gen) const {
    return expected_gen != 0 && index < CAPACITY && entries_[index].slot_gen == expected_gen;
}

// Reclaim a resolved (committed/failed) slot at refcount zero. The odd->even generation-parity
// rule lives here and nowhere else; live prepared/submitted entries stay until resolution.
void vbr_extent_store::maybe_reclaim(uint32_t index) {
    auto & entry = entries_[index];
    if (refcounts_[index] != 0 ||
        (entry.state != vbr_extent_state::committed && entry.state != vbr_extent_state::failed)) {
        return;
    }
    entry.state = vbr_extent_state::free_slot;
    entry.slot_gen += 1;  // odd -> even: resolved
    free_list_.push_back(index);
}

vbr_extent_handle vbr_extent_store::reserve(vbr_mutation_family family,
                                            vbr_operation_class operation_class,
                                            uint16_t            stream,
                                            llama_seq_id        seq_id,
                                            llama_pos           p0,
                                            llama_pos           p1) {
    if (free_list_.empty()) {
        // Exhaustion latches; the caller must invalidate-before-mutate and reset_all().
        exhausted_ = true;
        return {};
    }
    const uint32_t index = free_list_.back();
    free_list_.pop_back();

    auto & entry = entries_[index];
    // Odd generation = live slot. The bump is what invalidates any stale ref left over from a
    // prior occupant of this slot.
    if (entry.slot_gen >= std::numeric_limits<uint32_t>::max() - 2) {
        // No-wrap rule: refuse and latch; the owner performs the global
        // invalidation + reset_all() which rebases every generation.
        free_list_.push_back(index);
        exhausted_ = true;
        return {};
    }
    entry.slot_gen += 1;  // even -> odd: live
    entry.family          = family;
    entry.operation_class = operation_class;
    entry.state           = vbr_extent_state::prepared;
    entry.stream          = stream;
    entry.seq_id          = seq_id;
    entry.p0              = p0;
    entry.p1              = p1;
    refcounts_[index]     = 0;

    return { index, entry.slot_gen };
}

bool vbr_extent_store::submit(vbr_extent_handle handle) {
    if (!handle_valid(handle) || entries_[handle.index].state != vbr_extent_state::prepared) {
        return false;
    }
    entries_[handle.index].state = vbr_extent_state::submitted;
    return true;
}

bool vbr_extent_store::commit(vbr_extent_handle handle) {
    if (!handle_valid(handle)) {
        return false;
    }
    auto & entry = entries_[handle.index];
    if (entry.state != vbr_extent_state::prepared && entry.state != vbr_extent_state::submitted) {
        return false;
    }
    entry.state = vbr_extent_state::committed;
    maybe_reclaim(handle.index);  // committed but never cited: immediately reclaimable
    return true;
}

bool vbr_extent_store::fail(vbr_extent_handle handle) {
    if (!handle_valid(handle)) {
        return false;
    }
    entries_[handle.index].state = vbr_extent_state::failed;
    maybe_reclaim(handle.index);
    return true;
}

vbr_extent_ref vbr_extent_store::add_ref(vbr_extent_handle handle) {
    if (!handle_valid(handle)) {
        return {};
    }
    ++refcounts_[handle.index];
    return { handle.index, handle.expected_gen };
}

void vbr_extent_store::release_ref(vbr_extent_ref ref) {
    if (!slot_matches(ref.index, ref.expected_gen)) {
        // Obsolete reference (slot already reset/reused) — nothing to release.
        return;
    }
    if (refcounts_[ref.index] == 0) {
        return;
    }
    if (--refcounts_[ref.index] == 0) {
        maybe_reclaim(ref.index);
    }
}

const vbr_extent_entry * vbr_extent_store::lookup_committed(vbr_extent_ref ref) const {
    if (!slot_matches(ref.index, ref.expected_gen) ||
        entries_[ref.index].state != vbr_extent_state::committed) {
        return nullptr;
    }
    return &entries_[ref.index];
}

void vbr_extent_store::reset_all() {
    free_list_.clear();
    for (uint32_t i = 0; i < CAPACITY; ++i) {
        auto & entry = entries_[i];
        // Rebase past every outstanding generation so any surviving stale ref mismatches.
        // (+2 keeps parity even = resolved; +2 also covers a slot that was live.)
        entry.slot_gen = (entry.slot_gen | 1u) + 1;
        entry.state    = vbr_extent_state::free_slot;
        refcounts_[i]  = 0;
        free_list_.push_back(CAPACITY - 1 - i);
    }
    exhausted_ = false;
}

uint32_t vbr_extent_store::live_entries() const {
    return CAPACITY - static_cast<uint32_t>(free_list_.size());
}

// ---------------------------------------------------------------------------
// vbr_ownership_index
// ---------------------------------------------------------------------------

struct vbr_ownership_index::seq_view {
    bool         in_use = false;
    bool         unavailable = false;
    uint32_t     owned  = 0;

    // Physical page masks: pages * 4 words, cell-indexed.
    std::vector<uint64_t> page_masks;
    // Logical-position Fenwick over [0, n_cells): fenwick[i] counts positions. 1-based.
    std::vector<uint32_t> fenwick;
};

namespace {

void fenwick_update(std::vector<uint32_t> & tree, uint32_t pos, int32_t delta) {
    for (uint32_t i = pos + 1; i < tree.size(); i += i & (~i + 1)) {
        tree[i] = static_cast<uint32_t>(static_cast<int64_t>(tree[i]) + delta);
    }
}

uint32_t fenwick_prefix(const std::vector<uint32_t> & tree, uint32_t count) {
    // sum of positions < count
    uint32_t sum = 0;
    for (uint32_t i = count; i > 0; i -= i & (~i + 1)) {
        sum += tree[i];
    }
    return sum;
}


struct mask_pos {
    uint32_t word;
    uint64_t bit;
};

mask_pos mask_locate(uint32_t cell) {
    return { (cell / VBR_GENERATION_PAGE_CELLS) * vbr_ownership_index::MASK_WORDS_PER_PAGE +
                     (cell % VBR_GENERATION_PAGE_CELLS) / 64,
             uint64_t(1) << (cell % 64) };
}

}  // namespace

vbr_ownership_index::vbr_ownership_index(uint32_t n_stream, uint32_t n_seq_max, uint32_t n_cells) :
    n_stream_(n_stream),
    n_seq_max_(n_seq_max),
    n_cells_(n_cells),
    n_pages_((n_cells + VBR_GENERATION_PAGE_CELLS - 1) / VBR_GENERATION_PAGE_CELLS) {
    // Flat O(1) slot map: views are lazy but their slots are preallocated.
    views_.resize(size_t(n_stream_) * n_seq_max_);
}

vbr_ownership_index::seq_view * vbr_ownership_index::find_view(uint32_t stream, llama_seq_id seq_id) {
    if (stream >= n_stream_ || seq_id < 0 || uint32_t(seq_id) >= n_seq_max_) {
        return nullptr;
    }
    auto & view = views_[size_t(stream) * n_seq_max_ + uint32_t(seq_id)];
    return view.in_use ? &view : nullptr;
}

const vbr_ownership_index::seq_view * vbr_ownership_index::find_view(uint32_t stream, llama_seq_id seq_id) const {
    return const_cast<vbr_ownership_index *>(this)->find_view(stream, seq_id);
}

vbr_ownership_index::seq_view & vbr_ownership_index::obtain_view(uint32_t stream, llama_seq_id seq_id) {
    auto & view = views_.at(size_t(stream) * n_seq_max_ + uint32_t(seq_id));
    if (!view.in_use) {
        view.in_use      = true;
        view.unavailable = false;
        view.owned       = 0;
        // assign() reuses retained capacity on seq-id reuse (no shrink churn).
        view.page_masks.assign(size_t(n_pages_) * MASK_WORDS_PER_PAGE, 0);
        view.fenwick.assign(size_t(n_cells_) + 1, 0);
    }
    return view;
}

bool vbr_ownership_index::add_cell(uint32_t stream, llama_seq_id seq_id, uint32_t cell, llama_pos pos) {
    if (stream >= n_stream_ || cell >= n_cells_) {
        return false;
    }
    // Lazy view allocation may throw mid-mutation; fail closed to an
    // unavailable view instead of unwinding through a registrant transaction.
    seq_view * view_ptr = nullptr;
    try {
        view_ptr = &obtain_view(stream, seq_id);
    } catch (const std::bad_alloc &) {
        if (auto * existing = find_view(stream, seq_id)) {
            existing->unavailable = true;
        }
        return false;
    }
    auto & view = *view_ptr;
    if (pos < 0 || static_cast<uint32_t>(pos) >= n_cells_) {
        // Fail-closed domain restriction.
        view.unavailable = true;
        return false;
    }
    const auto loc = mask_locate(cell);
    if ((view.page_masks[loc.word] & loc.bit) != 0) {
        return true;  // already owned; idempotent
    }
    view.page_masks[loc.word] |= loc.bit;
    fenwick_update(view.fenwick, static_cast<uint32_t>(pos), +1);
    ++view.owned;
    return true;
}

bool vbr_ownership_index::remove_cell(uint32_t stream, llama_seq_id seq_id, uint32_t cell, llama_pos pos) {
    auto * view = find_view(stream, seq_id);
    if (view == nullptr || cell >= n_cells_) {
        return view != nullptr;
    }
    const auto loc = mask_locate(cell);
    if ((view->page_masks[loc.word] & loc.bit) == 0) {
        return true;
    }
    view->page_masks[loc.word] &= ~loc.bit;
    if (pos >= 0 && static_cast<uint32_t>(pos) < n_cells_) {
        fenwick_update(view->fenwick, static_cast<uint32_t>(pos), -1);
    } else {
        // Removing a cell whose recorded position was out of domain: the view was already
        // unavailable when it went out of domain; keep it unavailable.
        view->unavailable = true;
    }
    if (view->owned > 0) {
        --view->owned;
    }
    return true;
}

bool vbr_ownership_index::move_cell(uint32_t stream, llama_seq_id seq_id, uint32_t cell,
                                    llama_pos old_pos, llama_pos new_pos) {
    auto * view = find_view(stream, seq_id);
    if (view == nullptr || cell >= n_cells_) {
        return view != nullptr;
    }
    const auto loc = mask_locate(cell);
    if ((view->page_masks[loc.word] & loc.bit) == 0) {
        return true;  // not owned by this view; nothing to move
    }
    if (old_pos >= 0 && static_cast<uint32_t>(old_pos) < n_cells_) {
        fenwick_update(view->fenwick, static_cast<uint32_t>(old_pos), -1);
    }
    if (new_pos >= 0 && static_cast<uint32_t>(new_pos) < n_cells_) {
        fenwick_update(view->fenwick, static_cast<uint32_t>(new_pos), +1);
    } else {
        view->unavailable = true;
        return false;
    }
    return true;
}

void vbr_ownership_index::clear_seq(uint32_t stream, llama_seq_id seq_id) {
    if (auto * view = find_view(stream, seq_id)) {
        // Capacity is deliberately RETAINED: server slots cycle seq ids, and re-zeroing via
        // assign() is cheaper than an allocator round-trip per request.
        view->in_use      = false;
        view->unavailable = false;
        view->owned       = 0;
    }
}

void vbr_ownership_index::clear_all() {
    // The flat slot map is fixed: reset views in place; the lookup domain
    // (n_stream * n_seq_max slots) never changes after construction.
    for (auto & view : views_) {
        view.in_use      = false;
        view.unavailable = false;
        view.owned       = 0;
    }
}

bool vbr_ownership_index::available(uint32_t stream, llama_seq_id seq_id) const {
    const auto * view = find_view(stream, seq_id);
    return view != nullptr && !view->unavailable;
}

bool vbr_ownership_index::initialized(uint32_t stream, llama_seq_id seq_id) const {
    return find_view(stream, seq_id) != nullptr;
}

bool vbr_ownership_index::rank_below(uint32_t stream, llama_seq_id seq_id, llama_pos frontier,
                                     uint32_t & rank) const {
    const auto * view = find_view(stream, seq_id);
    if (view == nullptr || view->unavailable || frontier < 0) {
        return false;
    }
    const uint32_t bound = std::min<uint32_t>(static_cast<uint32_t>(frontier), n_cells_);
    rank = fenwick_prefix(view->fenwick, bound);
    return true;
}

bool vbr_ownership_index::enumerate_owned(uint32_t stream, llama_seq_id seq_id,
                                          std::vector<uint32_t> & cells) const {
    const auto * view = find_view(stream, seq_id);
    if (view == nullptr || view->unavailable) {
        return false;
    }
    for (uint32_t page = 0; page < n_pages_; ++page) {
        for (uint32_t w = 0; w < MASK_WORDS_PER_PAGE; ++w) {
            uint64_t word = view->page_masks[size_t(page) * MASK_WORDS_PER_PAGE + w];
            while (word != 0) {
                const uint32_t bit  = llama_countr_zero_u64(word);
                const uint32_t cell = page * VBR_GENERATION_PAGE_CELLS + w * 64 + bit;
                cells.push_back(cell);
                word &= word - 1;
            }
        }
    }
    return true;
}



vbr_ownership_index::~vbr_ownership_index() = default;
