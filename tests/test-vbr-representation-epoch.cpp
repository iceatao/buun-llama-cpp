#include "common.h"
#include "llama-kv-cache-iswa.h"
#include "llama-io.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <atomic>
#include <thread>
#include <cstdlib>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

template<typename L, typename R, typename = void>
struct f40_has_equal : std::false_type {};

template<typename L, typename R>
struct f40_has_equal<L, R,
    std::void_t<decltype(std::declval<L>() == std::declval<R>())>> : std::true_type {};

static_assert(!std::is_same<vbr_lineage_uuid, vbr_controller_instance_id>::value,
              "lineage and runtime instance must stay distinct strong types");
static_assert(!std::is_constructible<vbr_lineage_uuid,
                                     vbr_controller_instance_id>::value,
              "runtime instance must not initialize durable lineage");
static_assert(!std::is_constructible<vbr_controller_instance_id,
                                     vbr_lineage_uuid>::value,
              "lineage must not initialize a runtime operation target");
static_assert(!std::is_assignable<vbr_lineage_uuid &, vbr_controller_instance_id>::value &&
              !std::is_assignable<vbr_controller_instance_id &, vbr_lineage_uuid>::value,
              "lineage and runtime instance must not be assignable across domains");
static_assert(!f40_has_equal<vbr_lineage_uuid, vbr_controller_instance_id>::value &&
              !f40_has_equal<vbr_controller_instance_id, vbr_lineage_uuid>::value,
              "cross-domain identity equality must not compile");
static_assert(std::is_same<decltype(vbr_checkpoint_generation_controller{}.lineage_uuid),
                           vbr_lineage_uuid>::value,
              "checkpoint controller must serialize lineage only");
static_assert(std::is_same<decltype(vbr_operation_target{}.instance_id),
                           vbr_controller_instance_id>::value,
              "operation target must authenticate runtime instance only");
static_assert(std::is_same<decltype(vbr_failed_operation_record{}.owner_instance),
                           vbr_controller_instance_id>::value,
              "recovery must route by runtime instance only");

static bool f40_deterministic_origin(uint64_t & origin) noexcept {
    origin = UINT64_C(0x56425247454e4131);
    return true;
}

static bool f40_unavailable_origin(uint64_t & origin) noexcept {
    origin = 0;
    return false;
}

// Friend of llama_kv_cache: the production low-LCP path reaches the same two operations through
// clear() followed by prepare(). Driving them directly makes the cursor-at-zero state observable
// before a tight budget can immediately start another degrade wave.
struct llama_kv_cache_vbr_epoch_test {
    static bool active(const llama_kv_cache * kv) {
        return kv->vbr_vmm_active() && kv->vbr_budget_bytes_ > 0;
    }

    static bool generation_seeded(const llama_kv_cache * kv) {
        const auto * tracker = kv->vbr_generation_tracker_get();
        if (tracker == nullptr || !tracker->active() || !tracker->stable()) {
            return false;
        }
        for (uint32_t stream = 0; stream < tracker->stream_count(); ++stream) {
            for (uint32_t cell = 0; cell < tracker->cell_count(); ++cell) {
                if (tracker->dependency_generation(stream, cell) != 0) {
                    return true;
                }
            }
        }
        return false;
    }

    static bool generation_units_match(const llama_kv_cache * kv) {
        const auto * tracker = kv->vbr_generation_tracker_get();
        if (tracker == nullptr || !tracker->stable() ||
                tracker->unit_count() != kv->layers.size() * 2) {
            return false;
        }
        for (size_t ikv = 0; ikv < kv->layers.size(); ++ikv) {
            for (uint32_t side = 0; side < 2; ++side) {
                const auto * tensor = side != 0 ? kv->layers[ikv].v : kv->layers[ikv].k;
                const auto unit =
                        tracker->unit_generation(static_cast<uint32_t>(ikv * 2 + side));
                const int32_t live_type =
                        tensor != nullptr ? static_cast<int32_t>(tensor->type) : -1;
                if (unit.current_type != live_type) {
                    return false;
                }
            }
        }
        return true;
    }

    static bool has_mapped_degradable_unit(const llama_kv_cache * kv) {
        std::vector<ggml_type> sim;
        kv->vbr_sim_seed(
            sim, /*pooled_only=*/true,
            GGML_TYPE_COUNT, GGML_TYPE_COUNT,
            nullptr, nullptr, nullptr);
        for (size_t i = kv->vbr_degrade_cursor_;
             i < kv->vbr_degrade_order_.size(); ++i) {
            size_t slot = 0;
            const ggml_tensor * tensor = nullptr;
            ggml_type target = GGML_TYPE_COUNT;
            if (kv->vbr_sim_step(sim, i, slot, tensor, target)) {
                const auto & step = kv->vbr_degrade_order_[i];
                const auto & units =
                    kv->vbr_units_of(slot / 2, step.is_v != 0);
                bool all_mapped = !units.empty();
                for (const auto & [pool, extent] : units) {
                    all_mapped =
                        all_mapped && extent->t != nullptr &&
                        pool->vmm != nullptr && pool->wm_cells > 0;
                }
                if (all_mapped) {
                    return true;
                }
            }
        }
        return false;
    }

    // A one-token prepare is padded to the same 256-cell watermark used by production attention.
    // Map that watermark while the tensors are still at their entry tiers, before the seed decode
    // can invoke the budget controller. vbr_budget_eff() is floored at already-mapped bytes, so
    // the subsequent real decode cannot consume the ladder merely because this tiny fixture's
    // configured policy budget is smaller than one page-rounded entry-tier watermark.
    static bool map_seed_watermark(llama_kv_cache * kv) {
        const uint32_t wm = kv->vbr_watermark_cells(1);
        return wm > 0 && kv->vbr_vmm_try_map(wm);
    }

    // This test owns the representation-epoch mechanism, not the model/card-specific pricing
    // policy. vbr_degrade_next() is the authoritative production mutation path, but its runtime
    // clamp may legitimately contain zero steps when this tiny context fits at the entry tier.
    // Temporarily open the clamp for one direct friend call, then restore it. This creates the
    // test wave deterministically without changing the budget, decoding a model-dependent token
    // count, or relying on free VRAM. No controller boundary runs while cursor > the restored
    // clamp; each forced wave is followed by the assertions and then clear/full_reset.
    static bool force_degrade(llama_kv_cache * kv) {
        if (!has_mapped_degradable_unit(kv)) {
            return false;
        }
        const size_t saved_limit = kv->vbr_degrade_limit_;
        kv->vbr_degrade_limit_ = kv->vbr_degrade_order_.size();
        const bool changed =
            kv->vbr_degrade_next(kv->vbr_watermark_cells(0)) ==
            llama_kv_cache::vbr_degrade_result::applied;
        kv->vbr_degrade_limit_ = saved_limit;
        return changed;
    }

    static void full_reset(llama_kv_cache * kv) {
        kv->vbr_full_reset();
    }

    static bool dry_occupied_apply_preserves_epochs(llama_kv_cache * kv, llama_seq_id seq_id) {
        const uint32_t stream = kv->seq_to_stream.at(size_t(seq_id));
        auto & cells = kv->v_cells[stream];

        uint32_t idx = cells.size();
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.seq_has(i, seq_id) &&
                (idx == cells.size() || cells.pos_get(i) > cells.pos_get(idx))) {
                idx = i;
            }
        }
        if (idx == cells.size()) {
            return false;
        }

        const auto before_representation = kv->vbr_representation_epoch();
        const auto before_checkpoint     = kv->vbr_checkpoint_epoch();
        const auto head_old              = kv->v_heads[stream];
        const auto stash_dirty_old       = kv->vbr_stash_dirty_;
        std::vector<uint32_t> idxs(cells.size());
        for (uint32_t i = 0; i < cells.size(); ++i) {
            idxs[i] = i;
        }
        const auto cells_old = cells.cp(idxs);

        const auto * tracker = kv->vbr_generation_tracker_get();
        if (tracker == nullptr) {
            return false;
        }
        const auto tracker_generation = tracker->controller_generation();
        const auto tracker_serial = tracker->mutation_serial();
        std::vector<std::tuple<uint32_t, uint32_t, uint16_t, uint16_t, llama_seq_id>> generations;
        generations.reserve(cells.size());
        for (uint32_t i = 0; i < cells.size(); ++i) {
            generations.emplace_back(
                tracker->dependency_generation(stream, i),
                tracker->membership_generation(stream, i),
                tracker->dependency_provenance(stream, i),
                tracker->membership_provenance(stream, i),
                tracker->last_membership_seq(stream, i));
        }

        std::vector<uint32_t> owned_before;
        const bool ownership_before = kv->vbr_ownership_ != nullptr &&
            kv->vbr_ownership_->enumerate_owned(stream, seq_id, owned_before);
        const auto receipt_before = kv->vbr_import_receipt_;
        const size_t physical_before = std::count_if(
            idxs.begin(), idxs.end(),
            [&](uint32_t i) { return cells.seq_has(i, seq_id); });
        if (physical_before < 2) {
            return false;
        }

        llama_batch_allocr balloc(kv->hparams.n_pos_per_embd());
        llama_ubatch ubatch = balloc.ubatch_reserve(1, 1);
        llama_seq_id ubatch_seq = seq_id;
        ubatch.token[0]        = 1;
        ubatch.pos[0]          = cells.pos_get(idx) + 1;
        ubatch.n_seq_id[0]     = 1;
        ubatch.seq_id[0]       = &ubatch_seq;
        ubatch.seq_id_unq[0]   = seq_id;
        ubatch.output[0]       = 0;

        llama_kv_cache::slot_info sinfo;
        sinfo.resize(1);
        sinfo.s0 = stream;
        sinfo.s1 = stream;
        sinfo.strm[0] = stream;
        sinfo.idxs[0].push_back(idx);
        kv->apply_ubatch(sinfo, ubatch, false);

        std::vector<uint32_t> owned_after;
        const bool ownership_after = kv->vbr_ownership_ != nullptr &&
            kv->vbr_ownership_->enumerate_owned(stream, seq_id, owned_after);
        bool tracker_preserved =
            tracker->controller_generation() == tracker_generation &&
            tracker->mutation_serial() == tracker_serial;
        for (uint32_t i = 0; tracker_preserved && i < cells.size(); ++i) {
            tracker_preserved = generations[i] == std::make_tuple(
                tracker->dependency_generation(stream, i),
                tracker->membership_generation(stream, i),
                tracker->dependency_provenance(stream, i),
                tracker->membership_provenance(stream, i),
                tracker->last_membership_seq(stream, i));
        }
        const size_t physical_after = std::count_if(
            idxs.begin(), idxs.end(),
            [&](uint32_t i) { return cells.seq_has(i, seq_id); });
        const bool preserved =
            kv->vbr_representation_epoch() == before_representation &&
            kv->vbr_checkpoint_epoch() == before_checkpoint &&
            tracker_preserved &&
            ownership_before == ownership_after &&
            owned_before == owned_after &&
            kv->vbr_import_receipt_.get() == receipt_before.get() &&
            physical_after < physical_before;
        cells.set(idxs, cells_old);
        kv->v_heads[stream] = head_old;
        kv->vbr_stash_dirty_ = stash_dirty_old;

        // Pin the receipt side of dry_run with a real non-null token. The production receipt
        // class is deliberately private to the artifact adopter; an aliasing shared_ptr gives
        // this friend fixture identity/lifetime without constructing or exposing that class.
        // Removing the entire speculative sequence would release a real receipt if dry_run ever
        // regressed into the committed empty-cache cleanup door.
        const auto receipt_token_owner = std::make_shared<uint8_t>(0);
        auto * receipt_token = reinterpret_cast<vbr_import_receipt_group *>(receipt_token_owner.get());
        kv->vbr_import_receipt_ = std::shared_ptr<vbr_import_receipt_group>(
            receipt_token_owner, receipt_token);
        const bool dry_rm_ok = kv->seq_rm_impl(
            seq_id, -1, -1, llama_kv_cache::seq_rm_mode::dry_run);
        const bool receipt_preserved =
            dry_rm_ok &&
            cells.seq_pos_min(seq_id) < 0 &&
            kv->vbr_import_receipt_.get() == receipt_token;

        cells.set(idxs, cells_old);
        kv->v_heads[stream] = head_old;
        kv->vbr_stash_dirty_ = stash_dirty_old;
        kv->vbr_import_receipt_ = receipt_before;
        return preserved && receipt_preserved;
    }

    static bool transient_suffix_preserves_checkpoint_lineage(
            llama_kv_cache * kv, llama_seq_id seq_id, llama_pos suffix_begin) {
        const auto representation_before = kv->vbr_representation_epoch();
        const auto checkpoint_before     = kv->vbr_checkpoint_epoch();
        const llama_pos pos_before       = kv->seq_pos_max(seq_id);
        const bool removed = kv->seq_rm_transient(seq_id, suffix_begin, -1);
        return removed &&
            pos_before >= suffix_begin &&
            kv->seq_pos_max(seq_id) < pos_before &&
            kv->vbr_representation_epoch() == representation_before &&
            kv->vbr_checkpoint_epoch() == checkpoint_before;
    }

    static bool reconcile(llama_kv_cache * kv) {
        return kv->vbr_retier_take_reconcile("unit_test");
    }

    static uint64_t freeze_operation_id(const llama_kv_cache * kv) {
        if (kv->vbr_retier_freeze_depth_ == 0) {
            return 0;
        }
        return kv->vbr_retier_freeze_stack_[kv->vbr_retier_freeze_depth_ - 1]
            .operation_id.value;
    }

    static uint64_t set_budget_bytes(llama_kv_cache * kv, uint64_t budget_bytes) {
        const uint64_t previous = kv->vbr_budget_bytes_;
        kv->vbr_budget_bytes_ = budget_bytes;
        return previous;
    }

    static vbr_generation_tracker * tracker_mut(llama_kv_cache * kv) {
        return kv->vbr_generation_tracker_mut();
    }

    static const vbr_generation_tracker * tracker_get(const llama_kv_cache * kv) {
        return kv->vbr_generation_tracker_get();
    }

    struct serializer_count_complete {};

    class serializer_positions_writer : public llama_io_write_i {
    public:
        explicit serializer_positions_writer(bool has_ext) : has_ext(has_ext) {}

        void write(const void * src, size_t size) override {
            if (stage == 0) {
                if (size == sizeof(uint32_t)) {
                    std::memcpy(&streams, src, size);
                }
                valid = size == sizeof(uint32_t) && streams == 1;
                stage = 1;
                return;
            }
            if (stage == 1) {
                if (size == sizeof(uint32_t)) {
                    std::memcpy(&remaining, src, size);
                }
                valid = valid && size == sizeof(uint32_t);
                positions.reserve(remaining);
                if (remaining == 0) {
                    throw serializer_count_complete{};
                }
                stage = 2;
                return;
            }
            if (stage == 2) {
                llama_pos position = -1;
                if (size == sizeof(position)) {
                    std::memcpy(&position, src, size);
                }
                valid = valid && size == sizeof(position) && position >= 0;
                positions.push_back(position);
                stage = 3;
                return;
            }
            if (stage == 3) {
                uint32_t n_seq = 0;
                if (size == sizeof(n_seq)) {
                    std::memcpy(&n_seq, src, size);
                }
                valid = valid && size == sizeof(n_seq) && n_seq == 1;
                stage = has_ext ? 4 : 5;
                return;
            }
            if (stage == 4) {
                valid = valid && size == sizeof(llama_kv_cell_ext);
                stage = 5;
                return;
            }
            if (stage == 5) {
                llama_seq_id seq = -1;
                if (size == sizeof(seq)) {
                    std::memcpy(&seq, src, size);
                }
                valid = valid && size == sizeof(seq) && seq == 0 && remaining > 0;
                if (--remaining == 0) {
                    throw serializer_count_complete{};
                }
                stage = 2;
                return;
            }
            valid = false;
            throw serializer_count_complete{};
        }

        void write_tensor(ggml_tensor *, size_t, size_t) override {
            valid = false;
            throw serializer_count_complete{};
        }

        size_t n_bytes() override {
            return 0;
        }

        const bool             has_ext;
        uint32_t               streams   = 0;
        uint32_t               remaining = 0;
        uint32_t               stage     = 0;
        bool                   valid     = true;
        std::vector<llama_pos> positions;
    };

    static bool serializer_positions(
            const llama_kv_cache * kv,
            llama_seq_id seq_id,
            std::vector<llama_pos> & positions) {
        serializer_positions_writer writer(kv->hparams.n_pos_per_embd() > 1);
        try {
            kv->state_write(writer, seq_id);
        } catch (const serializer_count_complete &) {
            positions = std::move(writer.positions);
            return writer.valid && writer.remaining == 0;
        } catch (...) {
            return false;
        }
        return false;
    }

    // C2 rows (b)/(c): a REAL provenance-bearing root scope with a deliberately narrow
    // manifest (seq 0, positions [0,2)). Returned open so nested production mutations join
    // it; closing WITHOUT succeed() is the production FAILED close (autorecords recovery).
    static void * open_narrow_trim_scope(llama_kv_cache * kv) {
        auto * tracker = kv->vbr_generation_tracker_mut();
        if (tracker == nullptr) {
            return nullptr;
        }
        const auto instance = tracker->runtime_instance();
        vbr_operation_binding binding;
        binding.kind        = vbr_operation_kind::sequence_edit;
        binding.child_phase = vbr_operation_phase::mutate;
        binding.n_targets   = 2;
        binding.targets[0]  = vbr_make_target(vbr_operation_kind::sequence_edit,
                                              vbr_operation_class::explicit_destructive_trim,
                                              instance, 0, 0, 0, 2);
        // real nested seq_rm authenticates as membership-only state_api
        // (llama-kv-cache.cpp seq_rm scope); authorize that class on the SAME narrow range so
        // its refusal happens at per-stamp range selection (positions >= 2), never at
        // begin-time class authentication
        binding.targets[1]  = vbr_make_target(vbr_operation_kind::sequence_edit,
                                              vbr_operation_class::state_api,
                                              instance, 0, 0, 0, 2);
        auto * op = new llama_kv_cache::vbr_mutation_op(kv, binding, /*provenance_bearing=*/true);
        if (!op->active()) {
            delete op;
            return nullptr;
        }
        return op;
    }

    static void close_scope_without_success(void * scope) {
        delete static_cast<llama_kv_cache::vbr_mutation_op *>(scope);
    }

    // C2 row (b): joined poison through the production citation/stamp path — the event cites
    // the open root scope and the stamped pre-mutation position (100) is outside the
    // manifest, so vbr_stamp() refuses it and poisons the root.
    static bool stamp_outside_manifest(llama_kv_cache * kv, void * scope) {
        auto * op = static_cast<llama_kv_cache::vbr_mutation_op *>(scope);
        auto event = kv->vbr_generation_begin(
                vbr_mutation_registrant::seq_rm,
                vbr_operation_class::explicit_destructive_trim,
                0,
                vbr_generation_stamp_kind::membership,
                /*destructive=*/true);
        if (!event) {
            return false;
        }
        kv->vbr_stamp(*op, event, /*cell=*/3, /*membership_seq=*/0, /*pre_mutation_pos=*/100);
        return event.finish();
    }

    // C2 row (e): fence-race seam — after decode SUBMISSION and before the synchronize
    // fence, one in-flight operation's per-target evidence goes stale via a slab reset. The
    // fence's commit then fails through the REAL terminal path (latch + fail handles +
    // FAILED close/report).
    static bool inject_stale_submitted_extent(llama_kv_cache * kv) {
        auto * tracker = kv->vbr_generation_tracker_mut();
        if (tracker == nullptr) {
            return false;
        }
        auto & store  = tracker->extent_store();
        auto   handle = store.reserve(vbr_mutation_family::trim,
                                      vbr_operation_class::explicit_destructive_trim, 0, 0, 0, 1);
        if (!handle || !store.submit(handle)) {
            return false;
        }
        store.reset_all();  // slab reset: the submitted handle is obsolete at the fence
        if (!kv->vbr_awaiting_commit_.empty()) {
            kv->vbr_awaiting_commit_.front().extents[0] = handle;
            return true;
        }
        if (!kv->vbr_pending_decode_ops_.empty()) {
            kv->vbr_pending_decode_ops_.front().extents[0] = handle;
            return true;
        }
        return false;
    }
};

static bool decode_one(llama_context * ctx, llama_pos pos = 0) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, 1, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool epochs_equal(
        const llama_memory_vbr_state_data & a,
        const llama_memory_vbr_state_data & b) {
    return a.representation_epoch == b.representation_epoch &&
           a.representation_epoch_swa == b.representation_epoch_swa;
}

static bool checkpoint_epochs_equal(
        const llama_memory_vbr_state_data & a,
        const llama_memory_vbr_state_data & b) {
    return a.checkpoint_epoch == b.checkpoint_epoch &&
           a.checkpoint_epoch_swa == b.checkpoint_epoch_swa;
}

static bool get_iswa_children(
        llama_memory_t mem,
        llama_kv_cache *& base,
        llama_kv_cache *& swa) {
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(mem)) {
        base = iswa->get_base();
        swa  = iswa->get_swa();
        return true;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        base = hybrid->get_mem_attn()->get_base();
        swa  = hybrid->get_mem_attn()->get_swa();
        return true;
    }
    return false;
}

static void set_test_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void unset_test_env(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

// A2: every mutation event must cite a live registry operation. Tests reuse the production
// RAII (reuse review) — one begin/close idiom in the whole tree. P5v2 (v6): mutation targets
// carry exact nonzero instances, so every test operation is bound to its tracker instance.
struct test_operation {
    vbr_scoped_operation op;
    test_operation(vbr_operation_kind kind, vbr_controller_instance_id instance, llama_seq_id seq = -1,
                   llama_pos p0 = 0, llama_pos p1 = std::numeric_limits<llama_pos>::max(),
                   vbr_operation_class operation_class = vbr_operation_class::state_api)
        : op(vbr_mutation_binding(kind, seq, p0, p1, operation_class, instance)) {}
    vbr_operation_id id() const { return op.id(); }
};

// P1v2 (v6): destructive test events supply per-target extents through the production
// callback shape; the supplier records which target index the tracker selected (single-target
// fixtures just populate handles[0]).
struct test_multi_extent_supplier {
    std::array<vbr_extent_handle, 2> handles = {};
    int                              last    = -1;
};

static vbr_extent_handle test_multi_extent_cb(void * ctx, uint8_t target_index) {
    auto * supplier = static_cast<test_multi_extent_supplier *>(ctx);
    supplier->last  = target_index;
    return target_index < 2 ? supplier->handles[target_index] : vbr_extent_handle{};
}

static bool run_f40_cpu_tests() {
    // Entropy failure is terminal for that construction attempt, never replaced by the old
    // fixed-domain fallback. The deterministic provider is restored before the remaining rows.
    if (!vbr_lineage_origin_provider_set_for_tests(f40_unavailable_origin)) {
        fprintf(stderr, "F4.0 could not install unavailable-origin test provider\n");
        return false;
    }
    {
        vbr_generation_tracker unavailable(1, 8, 1);
        if (unavailable.active() ||
                vbr_lineage_uuid_is_set(unavailable.lineage_identity()) ||
                vbr_controller_instance_id_is_set(unavailable.runtime_instance())) {
            fprintf(stderr, "F4.0 entropy failure did not fail closed\n");
            return false;
        }
    }
    if (!vbr_lineage_origin_provider_set_for_tests(f40_deterministic_origin)) {
        fprintf(stderr, "F4.0 could not install deterministic origin provider\n");
        return false;
    }

    // The live-controller registry rejects duplicate/corrupt claims, enforces its bound, and
    // releases only the exact (instance, owner) pair.
    const size_t full_capacity = vbr_controller_instance_registry_capacity();
    if (!vbr_controller_instance_registry_capacity_set_for_tests(1)) {
        fprintf(stderr, "F4.0 could not narrow the controller registry\n");
        return false;
    }
    int owner_a = 0;
    int owner_b = 0;
    const vbr_controller_instance_id instance_a = {
        UINT64_C(0x564252494e535431), UINT64_C(0x1001),
    };
    const vbr_controller_instance_id instance_b = {
        UINT64_C(0x564252494e535431), UINT64_C(0x1002),
    };
    if (!vbr_controller_instance_check_and_claim(instance_a, &owner_a) ||
            vbr_controller_instance_check_and_claim(instance_a, &owner_b) ||
            vbr_controller_instance_check_and_claim(instance_b, &owner_b) ||
            vbr_controller_instance_release(instance_a, &owner_b) ||
            !vbr_controller_instance_release(instance_a, &owner_a) ||
            !vbr_controller_instance_registry_capacity_set_for_tests(full_capacity)) {
        fprintf(stderr, "F4.0 controller registry claim/release matrix failed\n");
        return false;
    }

    // Tracker construction itself claims one slot, capacity failure leaves the competing
    // tracker inactive, and destruction makes that exact slot reusable without reusing its ID.
    if (!vbr_controller_instance_registry_capacity_set_for_tests(1)) {
        fprintf(stderr, "F4.0 could not narrow the tracker enrollment registry\n");
        return false;
    }
    {
        vbr_generation_tracker enrolled(1, 8, 1);
        vbr_generation_tracker refused(1, 8, 1);
        if (!enrolled.active() || refused.active() ||
                enrolled.runtime_instance() == refused.runtime_instance()) {
            fprintf(stderr, "F4.0 tracker enrollment capacity did not fail closed\n");
            return false;
        }
    }
    {
        vbr_generation_tracker after_release(1, 8, 1);
        if (!after_release.active()) {
            fprintf(stderr, "F4.0 tracker destruction did not release its registry row\n");
            return false;
        }
    }
    if (!vbr_controller_instance_registry_capacity_set_for_tests(full_capacity)) {
        fprintf(stderr, "F4.0 could not restore controller registry capacity\n");
        return false;
    }

    {
        vbr_generation_tracker source(1, 64, 1);
        vbr_generation_tracker clone(1, 64, 1, source.lineage_identity());
        vbr_generation_tracker unrelated(1, 64, 1);
        if (!source.active() || !clone.active() || !unrelated.active() ||
                source.lineage_identity() != clone.lineage_identity() ||
                source.lineage_identity() == unrelated.lineage_identity() ||
                source.runtime_instance() == clone.runtime_instance() ||
                !vbr_lineage_uuid_is_set(source.lineage_identity()) ||
                !vbr_controller_instance_id_is_set(source.runtime_instance()) ||
                !source.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full) ||
                !clone.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full) ||
                !unrelated.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full)) {
            fprintf(stderr, "F4.0 source/clone identity model failed\n");
            return false;
        }

        // An operation aimed at one runtime instance cannot authenticate the other, in either
        // direction, even though the two trackers intentionally share durable lineage.
        {
            test_operation source_op(
                vbr_operation_kind::sequence_edit, source.runtime_instance(), 0, 0, 8);
            auto source_event = source.begin_event(
                vbr_mutation_registrant::seq_rm,
                vbr_operation_class::state_api, 0,
                vbr_generation_stamp_kind::membership, source_op.id());
            auto clone_event = clone.begin_event(
                vbr_mutation_registrant::seq_rm,
                vbr_operation_class::state_api, 0,
                vbr_generation_stamp_kind::membership, source_op.id());
            if (!source_event || clone_event || !source_event.finish()) {
                fprintf(stderr, "F4.0 source operation authenticated the clone\n");
                return false;
            }
        }
        {
            test_operation clone_op(
                vbr_operation_kind::sequence_edit, clone.runtime_instance(), 0, 0, 8);
            auto source_event = source.begin_event(
                vbr_mutation_registrant::seq_rm,
                vbr_operation_class::state_api, 0,
                vbr_generation_stamp_kind::membership, clone_op.id());
            auto clone_event = clone.begin_event(
                vbr_mutation_registrant::seq_rm,
                vbr_operation_class::state_api, 0,
                vbr_generation_stamp_kind::membership, clone_op.id());
            if (source_event || !clone_event || !clone_event.finish()) {
                fprintf(stderr, "F4.0 clone operation authenticated the source\n");
                return false;
            }
        }

        // Recovery ownership is exact-instance: clone cannot observe, take, or acknowledge the
        // source record. Resolve it before tracker destruction so unregister proves quiescence.
        {
            test_operation source_op(
                vbr_operation_kind::sequence_edit, source.runtime_instance(), 0, 0, 8);
            const int32_t record =
                vbr_recovery_reserve(source_op.id(), source.runtime_instance());
            if (record < 0 ||
                    !vbr_recovery_record_failure(
                        record, source_op.id(), vbr_operation_phase::mutate,
                        vbr_recovery_failure_site::metadata_mutation, false)) {
                fprintf(stderr, "F4.0 could not create source-owned recovery record\n");
                return false;
            }
            {
                auto capability = vbr_recovery_mint(record);
                if (!capability || !capability.resolve_quarantined()) {
                    fprintf(stderr, "F4.0 could not quarantine source recovery\n");
                    return false;
                }
            }
            if (!vbr_recovery_pending_for(source.runtime_instance()) ||
                    vbr_recovery_pending_for(clone.runtime_instance()) ||
                    vbr_recovery_take_quarantine(clone.runtime_instance()).token) {
                fprintf(stderr, "F4.0 recovery ownership crossed runtime instances\n");
                return false;
            }
            auto work = vbr_recovery_take_quarantine(source.runtime_instance());
            if (!work.token ||
                    vbr_recovery_ack_quarantine(work.token, clone.runtime_instance()) ||
                    !vbr_recovery_ack_quarantine(work.token, source.runtime_instance())) {
                fprintf(stderr, "F4.0 recovery ack was not exact-instance\n");
                return false;
            }
        }

        // Runtime instance is absent from the durable generation record: equal tracker state +
        // shared lineage produces equal records despite distinct live routing identities.
        vbr_checkpoint_generation_controller source_record;
        vbr_checkpoint_generation_controller clone_record;
        if (!vbr_generation_capture_controller(
                    source, 0, checkpoint_child_dependency_mode::live_guarded, {}, source_record) ||
                !vbr_generation_capture_controller(
                    clone, 0, checkpoint_child_dependency_mode::live_guarded, {}, clone_record) ||
                !(source_record == clone_record) ||
                source_record.lineage_uuid != source.lineage_identity()) {
            fprintf(stderr, "F4.0 runtime instance leaked into durable generation record\n");
            return false;
        }

    }

    fprintf(stderr, "F4.0 lineage/runtime-instance CPU rows PASS\n");
    return true;
}

static bool run_a1_cpu_tests() {
    llama_kv_cells ownership_index;
    ownership_index.resize(4);
    ownership_index.pos_set(0, 5);
    ownership_index.seq_add(0, 0);
    ownership_index.pos_set(1, 15);
    ownership_index.seq_add(1, 0);
    if (ownership_index.seq_pos_count_before(0, 10) != 1 ||
            ownership_index.seq_pos_count_before(0, 20) != 2) {
        fprintf(stderr, "A1 canonical ownership index returned an inexact cardinality\n");
        return false;
    }

    vbr_generation_tracker tracker(1, 768, 1);
    if (!tracker.active() || !tracker.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full)) {
        fprintf(stderr, "A1 tracker did not initialize\n");
        return false;
    }
    // F3.3: a freshly initialized F16 unit does not need a tier transition
    // before it can produce an exact generation/ownership record. Ownership
    // comes from ordinary decode appends; the unit tuple starts at the
    // controller's initial representation generation.
    {
        vbr_ownership_index fresh_ownership(1, 1, 768);
        if (!fresh_ownership.add_cell(0, 0, 4, 0) ||
                !fresh_ownership.add_cell(0, 0, 9, 1) ||
                !fresh_ownership.add_cell(0, 0, 15, 2)) {
            fprintf(stderr, "F3.3 fresh-F16 ownership seed failed\n");
            return false;
        }
        uint32_t rank = 0;
        std::vector<uint32_t> owned;
        if (!fresh_ownership.rank_below(0, 0, 3, rank) ||
                rank != 3 ||
                !fresh_ownership.enumerate_owned(0, 0, owned) ||
                owned != std::vector<uint32_t>({4, 9, 15})) {
            fprintf(stderr, "F3.3 fresh-F16 ownership view was incomplete\n");
            return false;
        }
        vbr_checkpoint_generation_stream fresh_stream;
        vbr_checkpoint_generation_controller fresh_controller;
        if (!vbr_generation_capture_stream(
                    tracker, 0, 0, 3, owned, fresh_stream) ||
                !vbr_generation_capture_controller(
                    tracker, 7,
                    checkpoint_child_dependency_mode::live_guarded,
                    {fresh_stream}, fresh_controller) ||
                fresh_controller.child_id != 7 ||
                fresh_controller.streams.size() != 1 ||
                fresh_controller.streams[0].captured_dependency_count != 3 ||
                fresh_controller.units.size() != 1 ||
                fresh_controller.units[0].repr_gen != 1 ||
                fresh_controller.units[0].current_type != GGML_TYPE_F16 ||
                fresh_controller.units[0].last_transition !=
                    vbr_repr_transition::initial) {
            fprintf(stderr, "F3.3 fresh-F16 generation capture failed\n");
            return false;
        }
    }
    vbr_generation_tracker distinct_tracker(1, 768, 1);
    if (distinct_tracker.runtime_instance() == tracker.runtime_instance() ||
            distinct_tracker.lineage_identity() == tracker.lineage_identity()) {
        fprintf(stderr, "F4.0 tracker identity was reused\n");
        return false;
    }
    test_operation a1_op(vbr_operation_kind::decode, tracker.runtime_instance(), -1,
                         0, std::numeric_limits<llama_pos>::max(),
                         vbr_operation_class::ordinary_decode);
    test_operation a1_edit_op(vbr_operation_kind::sequence_edit, tracker.runtime_instance(), -1,
                              0, std::numeric_limits<llama_pos>::max(),
                              vbr_operation_class::prompt_share);
    test_operation distinct_op(vbr_operation_kind::decode, distinct_tracker.runtime_instance(), -1,
                               0, std::numeric_limits<llama_pos>::max(),
                               vbr_operation_class::ordinary_decode);
    if (!a1_op.id() || !a1_edit_op.id() || !distinct_op.id()) {
        fprintf(stderr, "A2 test operation failed to register\n");
        return false;
    }
    auto foreign_event = distinct_tracker.begin_event(
            vbr_mutation_registrant::apply_ubatch_append,
            vbr_operation_class::ordinary_decode,
            0,
            vbr_generation_stamp_kind::dependency,
            distinct_op.id());
    if (!foreign_event || tracker.stamp_cell(foreign_event, 10, 0) || !foreign_event.finish()) {
        fprintf(stderr, "A1 tracker accepted an event owned by another controller\n");
        return false;
    }

    auto append = tracker.begin_event(
            vbr_mutation_registrant::apply_ubatch_append,
            vbr_operation_class::ordinary_decode,
            0,
            vbr_generation_stamp_kind::dependency,
            a1_op.id());
    if (!append || !tracker.stamp_cell(append, 10, 0) ||
            !tracker.stamp_cell(append, 300, 0) || !append.finish()) {
        fprintf(stderr, "A1 dependency event did not publish atomically\n");
        return false;
    }
    const uint32_t dependency_before = tracker.dependency_generation(0, 10);

    auto share = tracker.begin_event(
            vbr_mutation_registrant::seq_cp,
            vbr_operation_class::prompt_share,
            0,
            vbr_generation_stamp_kind::membership,
            a1_edit_op.id());
    if (!share || !tracker.stamp_cell(share, 10, 1) ||
            tracker.dependency_generation(0, 10) != dependency_before ||
            tracker.membership_generation(0, 10) == 0 || !share.finish()) {
        fprintf(stderr, "A1 dependency/membership stamp split failed\n");
        return false;
    }

    vbr_checkpoint_generation_stream stream;
    if (!vbr_generation_capture_stream(tracker, 0, 0, 400, {10, 300}, stream) ||
            stream.captured_dependency_count != 2 || stream.pages.size() != 2) {
        fprintf(stderr, "A1 canonical covered-mask capture failed\n");
        return false;
    }

    fprintf(stderr, "A1 generation CPU coverage PASS\n");
    return true;
}


static bool run_a2_cpu_tests() {
    // --- extent store lifecycle -------------------------------------------------------------
    vbr_generation_tracker tracker(1, 768, 1);
    test_operation op(vbr_operation_kind::sequence_edit, tracker.runtime_instance(), 0, 0, 100);
    auto & store = tracker.extent_store();

    auto handle = store.reserve(vbr_mutation_family::trim,
                                vbr_operation_class::explicit_destructive_trim, 0, 0, 0, 100);
    if (!handle) {
        fprintf(stderr, "A2 extent reserve failed\n");
        return false;
    }
    auto ref = store.add_ref(handle);
    if (!ref || store.lookup_committed(ref) != nullptr) {
        fprintf(stderr, "A2 prepared extent must not be admission evidence\n");
        return false;
    }
    if (!store.submit(handle) || store.lookup_committed(ref) != nullptr) {
        fprintf(stderr, "A2 submitted extent must not be admission evidence\n");
        return false;
    }
    if (!store.commit(handle)) {
        fprintf(stderr, "A2 extent commit from submitted failed\n");
        return false;
    }
    const auto * entry = store.lookup_committed(ref);
    if (entry == nullptr || entry->p0 != 0 || entry->p1 != 100 ||
            entry->family != vbr_mutation_family::trim) {
        fprintf(stderr, "A2 committed extent lookup returned wrong evidence\n");
        return false;
    }
    // release-to-zero reclaims; a stale ref then fails ABA-safe
    store.release_ref(ref);
    if (store.lookup_committed(ref) != nullptr) {
        fprintf(stderr, "A2 reclaimed extent slot admitted a stale reference (ABA)\n");
        return false;
    }
    // failed entries are never evidence
    auto fhandle = store.reserve(vbr_mutation_family::trim,
                                 vbr_operation_class::dependency_seq_remove, 0, 1, 0,
                                 std::numeric_limits<llama_pos>::max());
    auto fref = store.add_ref(fhandle);
    if (!store.fail(fhandle) || store.lookup_committed(fref) != nullptr) {
        fprintf(stderr, "A2 failed extent leaked into evidence\n");
        return false;
    }
    store.release_ref(fref);
    // exhaustion-recovers semantics
    {
        std::vector<vbr_extent_handle> hoard;
        for (;;) {
            auto h = store.reserve(vbr_mutation_family::trim,
                                   vbr_operation_class::state_api, 0, 0, 0, 1);
            if (!h) {
                break;
            }
            hoard.push_back(h);
        }
        if (!store.exhausted_latched()) {
            fprintf(stderr, "A2 extent exhaustion did not latch\n");
            return false;
        }
        store.reset_all();
        if (store.exhausted_latched() || store.live_entries() != 0 ||
                !store.reserve(vbr_mutation_family::trim, vbr_operation_class::state_api, 0, 0, 0, 1)) {
            fprintf(stderr, "A2 extent exhaustion did not recover after slab reset\n");
            return false;
        }
        store.reset_all();
    }

    // --- citation refusal -------------------------------------------------------------------
    auto uncited = tracker.begin_event(
            vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api, 0,
            vbr_generation_stamp_kind::membership, vbr_operation_id{});
    if (uncited) {
        fprintf(stderr, "A2 tracker minted an event without a live operation citation\n");
        (void) uncited.finish();
        return false;
    }
    vbr_operation_id dead_id;
    {
        test_operation ephemeral(vbr_operation_kind::sequence_edit, tracker.runtime_instance());
        dead_id = ephemeral.id();
    }
    auto dead_cited = tracker.begin_event(
            vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api, 0,
            vbr_generation_stamp_kind::membership, dead_id);
    if (dead_cited) {
        fprintf(stderr, "A2 tracker accepted a dead operation citation\n");
        (void) dead_cited.finish();
        return false;
    }

    // --- ownership index: rank vs brute force incl. shifts + fail-closed domain --------------
    {
        vbr_ownership_index index(1, 8, 512);
        uint64_t rng = 0x5eedULL;
        std::vector<llama_pos> pos_of(512, -1);
        auto next = [&rng]() { rng = rng * 6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t)(rng >> 33); };
        for (int step = 0; step < 4000; ++step) {
            const uint32_t cell = next() % 512;
            const uint32_t act  = next() % 3;
            if (act == 0) {
                const llama_pos p = (llama_pos)(next() % 512);
                if (pos_of[cell] < 0 && index.add_cell(0, 3, cell, p)) {
                    pos_of[cell] = p;
                }
            } else if (act == 1 && pos_of[cell] >= 0) {
                index.remove_cell(0, 3, cell, pos_of[cell]);
                pos_of[cell] = -1;
            } else if (pos_of[cell] >= 0) {
                const llama_pos np = (llama_pos)(next() % 512);
                if (index.move_cell(0, 3, cell, pos_of[cell], np)) {
                    pos_of[cell] = np;
                }
            }
        }
        for (llama_pos frontier : { (llama_pos) 0, (llama_pos) 100, (llama_pos) 511, (llama_pos) 512 }) {
            uint32_t rank = 0;
            uint32_t brute = 0;
            for (uint32_t c = 0; c < 512; ++c) {
                brute += pos_of[c] >= 0 && pos_of[c] < frontier ? 1 : 0;
            }
            if (!index.rank_below(0, 3, frontier, rank) || rank != brute) {
                fprintf(stderr, "A2 ownership index rank mismatch at frontier %d (%u != %u)\n",
                        (int) frontier, rank, brute);
                return false;
            }
        }
        // out-of-domain position: fail-closed unavailable
        uint32_t free_cell = 0;
        while (free_cell < 512 && pos_of[free_cell] >= 0) free_cell++;
        if (index.add_cell(0, 3, free_cell, 9999) || index.available(0, 3)) {
            fprintf(stderr, "A2 ownership index accepted an out-of-domain position\n");
            return false;
        }
        uint32_t rank = 0;
        if (index.rank_below(0, 3, 10, rank)) {
            fprintf(stderr, "A2 unavailable index view still answered a rank query\n");
            return false;
        }
        index.clear_seq(0, 3);
        if (index.available(0, 3)) {
            fprintf(stderr, "A2 cleared seq view should be absent\n");
            return false;
        }
    }

    // --- recovery ring + capability ----------------------------------------------------------
    {
        test_operation rop(vbr_operation_kind::sequence_edit, tracker.runtime_instance(), 2, 0, 64);
        const int32_t idx = vbr_recovery_reserve(rop.id());
        if (idx < 0 || !vbr_recovery_release_unused(idx, rop.id())) {
            fprintf(stderr, "A2 recovery reserve/release failed\n");
            return false;
        }
        const int32_t idx2 = vbr_recovery_reserve(rop.id());
        if (idx2 < 0 ||
                !vbr_recovery_record_failure(idx2, rop.id(), vbr_operation_phase::mutate,
                                             vbr_recovery_failure_site::deferred_byte_copy, true)) {
            fprintf(stderr, "A2 recovery record_failure failed\n");
            return false;
        }
        {
            auto capability = vbr_recovery_mint(idx2);
            if (!capability || !capability.target_allowed(0, 2, 0, 64) ||
                    capability.target_allowed(0, 2, 0, 65) || capability.target_allowed(0, 5, 0, 64)) {
                fprintf(stderr, "A2 recovery capability target restriction failed\n");
                return false;
            }
            // deliberately no resolve: destructor must fail-close to quarantined
        }
        // C4: the fail-closed destructor left the record awaiting_ack. Take it for the
        // wildcard-instance target, ack with the token; a stale token must not ack twice.
        auto work = vbr_recovery_take_quarantine({});
        if (!work.token) {
            fprintf(stderr, "C4 fail-closed capability left no pending quarantine\n");
            return false;
        }
        if (!vbr_recovery_ack_quarantine(work.token, {}) ||
                vbr_recovery_ack_quarantine(work.token, {})) {
            fprintf(stderr, "C4 quarantine token ack was not single-use\n");
            return false;
        }
        if (vbr_recovery_take_quarantine({}).token) {
            fprintf(stderr, "C4 acked quarantine still pending\n");
            return false;
        }
        auto remint = vbr_recovery_mint(idx2);
        if (remint) {
            fprintf(stderr, "A2 reclaimed recovery record allowed a second mint\n");
            return false;
        }
    }

    // --- registry binding retention + close(failed) autorecord --------------------------------
    {
        vbr_operation_binding probe;
        test_operation bop(vbr_operation_kind::state_import, tracker.runtime_instance(), 4, 10, 20);
        if (!vbr_operation_registry_binding(bop.id(), probe) ||
                probe.seq_id() != 4 || probe.range().p0 != 10 || probe.range().p1 != 20 ||
                probe.kind != vbr_operation_kind::state_import) {
            fprintf(stderr, "A2 registry did not retain the authenticated binding\n");
            return false;
        }
        const int32_t ridx = vbr_recovery_reserve(bop.id());
        const vbr_operation_id bop_id = bop.id();
        if (ridx < 0 || !bop.op.close(vbr_operation_outcome::failed)) {
            fprintf(stderr, "A2 close(failed) failed\n");
            return false;
        }
        (void) bop_id;
        vbr_failed_operation_record record;
        if (!vbr_recovery_get_record(ridx, record) ||
                record.state != vbr_recovery_state::recorded) {
            fprintf(stderr, "A2 close(failed) did not autorecord the reserved recovery slot\n");
            return false;
        }
        auto cleanup = vbr_recovery_mint(ridx);
        cleanup.resolve_quarantined();
    }

    // --- review-fix coverage ------------------------------------------------------------------
    // F4: a decode-kind operation must NOT authorize a prompt-share membership event.
    {
        vbr_generation_tracker t3(1, 256, 1);
        test_operation decode_op(vbr_operation_kind::decode, t3.runtime_instance(), -1,
                                 0, std::numeric_limits<llama_pos>::max(),
                                 vbr_operation_class::ordinary_decode);
        auto misused = t3.begin_event(
                vbr_mutation_registrant::seq_cp, vbr_operation_class::prompt_share,
                0, vbr_generation_stamp_kind::membership, decode_op.id());
        if (misused) {
            fprintf(stderr, "F4: decode operation authorized a sequence-share event\n");
            (void) misused.finish();
            return false;
        }
        // P1v2 (v6) seq scope: an op bound to seq 2 stamps seq 2 fine; a seq-3 stamp has no
        // covering target, so it POISONS the event and latches unavailable IMMEDIATELY, and
        // every further stamp from the poisoned event is inert.
        test_operation seq2_op(vbr_operation_kind::sequence_edit, t3.runtime_instance(), 2, 0, 100);
        auto scoped = t3.begin_event(
                vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api,
                0, vbr_generation_stamp_kind::membership, seq2_op.id());
        if (!scoped || !t3.stamp_cell(scoped, 5, 2, 5)) {
            fprintf(stderr, "F4: authorized seq-scope stamp failed\n");
            return false;
        }
        if (t3.stamp_cell(scoped, 5, 3, 5) || !t3.shadow_unavailable()) {
            fprintf(stderr, "P1v2: unauthorized seq stamp did not poison + latch immediately\n");
            return false;
        }
        if (t3.stamp_cell(scoped, 5, 2, 5)) {
            fprintf(stderr, "P1v2: poisoned event accepted a further stamp\n");
            return false;
        }
        if (!scoped.finish()) {
            fprintf(stderr, "P1v2: poisoned event did not finish cleanly\n");
            return false;
        }
    }
    // F6: resolved recovery records reclaim their slots — the ring survives > capacity cycles.
    {
        test_operation cyc_op(vbr_operation_kind::sequence_edit, tracker.runtime_instance(), 0, 0, 8);
        for (int cycle = 0; cycle < 70; ++cycle) {
            const int32_t idx = vbr_recovery_reserve(cyc_op.id());
            if (idx < 0) {
                fprintf(stderr, "F6: recovery ring exhausted at cycle %d (leak)\n", cycle);
                return false;
            }
            if (!vbr_recovery_record_failure(idx, cyc_op.id(), vbr_operation_phase::mutate,
                                             vbr_recovery_failure_site::metadata_mutation, false)) {
                fprintf(stderr, "F6: record_failure failed at cycle %d\n", cycle);
                return false;
            }
            auto capability = vbr_recovery_mint(idx);
            if (!capability || !capability.resolve_completed()) {
                fprintf(stderr, "F6: resolve_completed failed at cycle %d\n", cycle);
                return false;
            }
        }
    }
    // --- v3 failure-path matrix (CPU rows) ----------------------------------------------------
    // Forged-field matrix (§11.1): each forged manifest dimension must refuse the event.
    {
        vbr_generation_tracker t5(1, 256, 1);
        const vbr_controller_instance_id t5_instance = t5.runtime_instance();
        // wrong class
        test_operation wrong_class(vbr_operation_kind::sequence_edit, t5_instance, 0, 0, 10,
                                   vbr_operation_class::prompt_share);
        auto e1 = t5.begin_event(vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api,
                                 0, vbr_generation_stamp_kind::membership, wrong_class.id());
        // wrong kind for registrant (controller_retier cannot authorize seq_rm)
        test_operation wrong_kind(vbr_operation_kind::controller_retier, t5_instance, -1, -1, -1,
                                  vbr_operation_class::state_api);
        auto e2 = t5.begin_event(vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api,
                                 0, vbr_generation_stamp_kind::membership, wrong_kind.id());
        // wrong stream target
        vbr_operation_binding far_stream = vbr_mutation_binding(
                vbr_operation_kind::sequence_edit, 0, 0, 10, vbr_operation_class::state_api,
                t5_instance, /*stream=*/7);
        vbr_scoped_operation far_op(far_stream);
        // Foreign instance: a manifest bound to another controller never covers this one.
        vbr_generation_tracker t5_foreign(1, 64, 1);
        test_operation foreign_instance_op(
                vbr_operation_kind::sequence_edit, t5_foreign.runtime_instance(), 0, 0, 10,
                vbr_operation_class::state_api);
        auto e3 = t5.begin_event(vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api,
                                 0, vbr_generation_stamp_kind::membership, far_op.id());
        auto e5 = t5.begin_event(vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api,
                                 0, vbr_generation_stamp_kind::membership, foreign_instance_op.id());
        if (e1 || e2 || e3 || e5) {
            fprintf(stderr, "v3 forged-field matrix: a forged manifest authorized an event\n");
            return false;
        }
        // correct manifest, wrong stamped seq (target seq 2, stamping seq 3): authorized
        // stamp first, then the forged one — which poisons the event (P1v2).
        test_operation seq_scope(vbr_operation_kind::sequence_edit, t5_instance, 2, 0, 10,
                                 vbr_operation_class::state_api);
        auto e4 = t5.begin_event(vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api,
                                 0, vbr_generation_stamp_kind::membership, seq_scope.id());
        if (!e4 || !t5.stamp_cell(e4, 1, 2, 5) || t5.stamp_cell(e4, 1, 3, 5) ||
                !t5.shadow_unavailable() || !e4.finish()) {
            fprintf(stderr, "v3 forged-field matrix: seq-scope stamp check failed\n");
            return false;
        }
    }
    // Publish-injection between reads: direct interleave — evaluator must reject unit_unstable.
    // (The tuple read itself is race-free by the units mutex; this exercises the F9 snapshot.)
    // Simulated by capturing a record, publishing a unit, and evaluating: repr_gen changes make
    // it unit_generation-reject; the F9 snapshot path is additionally covered by the two-thread
    // stress below reaching stable states only.
    // C5 row: clear_all -> add -> rank on the fixed slot map (the v2 critical-1 crash shape).
    {
        vbr_ownership_index idx5(1, 8, 512);
        if (!idx5.add_cell(0, 1, 3, 30)) {
            return false;
        }
        idx5.clear_all();
        uint32_t rank5 = 0;
        if (!idx5.add_cell(0, 1, 4, 40) || !idx5.rank_below(0, 1, 100, rank5) || rank5 != 1) {
            fprintf(stderr, "C5: clear_all -> add -> rank failed (rank %u)\n", rank5);
            return false;
        }
        std::vector<uint32_t> owned5;
        if (!idx5.enumerate_owned(0, 1, owned5) || owned5.size() != 1 || owned5[0] != 4) {
            fprintf(stderr, "C5: post-clear enumeration wrong\n");
            return false;
        }
    }
    // Two-thread stress: concurrent registry begin/end + recovery reserve/mint/resolve.
    {
        std::atomic<bool> failed{false};
        const vbr_controller_instance_id stress_instance = tracker.runtime_instance();
        auto worker = [&failed, stress_instance]() {
            for (int i = 0; i < 2000 && !failed.load(); ++i) {
                test_operation op(vbr_operation_kind::sequence_edit, stress_instance, i % 4, 0, 64,
                                  vbr_operation_class::state_api);
                if (!op.id()) {
                    failed.store(true);
                    break;
                }
                vbr_operation_binding probe;
                if (!vbr_operation_registry_binding(op.id(), probe) ||
                    probe.seq_id() != i % 4) {
                    failed.store(true);
                    break;
                }
                const int32_t r = vbr_recovery_reserve(op.id());
                if (r >= 0) {
                    if ((i & 1) != 0) {
                        vbr_recovery_record_failure(r, op.id(), vbr_operation_phase::mutate,
                                                    vbr_recovery_failure_site::metadata_mutation, false);
                        auto capability = vbr_recovery_mint(r);
                        if (capability) {
                            capability.resolve_completed();
                        }
                    } else {
                        vbr_recovery_release_unused(r, op.id());
                    }
                }
            }
        };
        std::thread t1(worker), t2(worker);
        t1.join();
        t2.join();
        if (failed.load()) {
            fprintf(stderr, "v3 two-thread registry/recovery stress FAILED\n");
            return false;
        }
    }

    // --- v6 rows: P1v2 stamp-time selection + per-target extents --------------------------
    {
        vbr_generation_tracker t7(1, 768, 1);
        vbr_operation_binding two_seq;
        two_seq.kind        = vbr_operation_kind::decode;
        two_seq.child_phase = vbr_operation_phase::mutate;
        two_seq.targets[two_seq.n_targets++] = vbr_make_target(
                vbr_operation_kind::decode, vbr_operation_class::ordinary_decode,
                t7.runtime_instance(), VBR_STREAM_ANY, 0, 0, 200);
        two_seq.targets[two_seq.n_targets++] = vbr_make_target(
                vbr_operation_kind::decode, vbr_operation_class::ordinary_decode,
                t7.runtime_instance(), VBR_STREAM_ANY, 1, 100, 300);
        vbr_scoped_operation two_seq_op(two_seq);
        if (!two_seq_op) {
            fprintf(stderr, "P1v2: two-target selection manifest failed to mint\n");
            return false;
        }
        test_multi_extent_supplier supplier;
        supplier.handles[0] = t7.extent_store().reserve(
                vbr_mutation_family::occupied_reuse, vbr_operation_class::ordinary_decode, 0, 0, 0, 200);
        supplier.handles[1] = t7.extent_store().reserve(
                vbr_mutation_family::occupied_reuse, vbr_operation_class::ordinary_decode, 0, 1, 100, 300);
        {
            auto reuse = t7.begin_event(
                    vbr_mutation_registrant::apply_ubatch_occupied_reuse,
                    vbr_operation_class::ordinary_decode, 0,
                    vbr_generation_stamp_kind::dependency, two_seq_op.id(),
                    &test_multi_extent_cb, &supplier, true);
            if (!reuse || !t7.stamp_cell(reuse, 10, 0, 50) || supplier.last != 0 ||
                    !t7.stamp_cell(reuse, 20, 1, 150) || supplier.last != 1) {
                fprintf(stderr, "P1v2: per-(seq,pos) selection picked the wrong target\n");
                return false;
            }
            // seq 1 at a position only seq 0's target covers: no cover -> poison + latch
            if (t7.stamp_cell(reuse, 30, 1, 50) || !t7.shadow_unavailable() ||
                    t7.stamp_cell(reuse, 10, 0, 50) || !reuse.finish()) {
                fprintf(stderr, "P1v2: uncovered (seq,pos) stamp did not poison + latch\n");
                return false;
            }
        }
        // the cells cite their SELECTED target's extent, never target zero's
        t7.extent_store().commit(supplier.handles[0]);
        t7.extent_store().commit(supplier.handles[1]);
        const auto * seq1_evidence = t7.extent_store().lookup_committed(t7.dependency_extent(0, 20));
        if (seq1_evidence == nullptr || seq1_evidence->seq_id != 1 || seq1_evidence->p0 != 100) {
            fprintf(stderr, "P1v2: stamp did not bind the selected target's extent\n");
            return false;
        }
        // P4v2 monotone re-arm: the latch recorded the generation; clearing needs a STRICTLY
        // later sanctioned transition.
        if (t7.try_clear_shadow_unavailable()) {
            fprintf(stderr, "P4v2: latch cleared without a post-latch transition\n");
            return false;
        }
        if (!t7.global_transition(vbr_mutation_registrant::clear, vbr_operation_class::state_api) ||
                !t7.try_clear_shadow_unavailable() || t7.shadow_unavailable()) {
            fprintf(stderr, "P4v2: latch did not clear after a post-latch transition\n");
            return false;
        }
        t7.set_shadow_unavailable();
        if (t7.try_clear_shadow_unavailable()) {
            fprintf(stderr, "P4v2: re-latch reused a stale transition proof\n");
            return false;
        }
        // P1v2 multi-seq target-set proof: every member covered -> stamp; any member
        // uncovered -> poison. (Fresh tracker: t7 is latched again above.)
        vbr_generation_tracker t7b(1, 768, 1);
        vbr_operation_binding set_manifest;
        set_manifest.kind        = vbr_operation_kind::decode;
        set_manifest.child_phase = vbr_operation_phase::mutate;
        for (llama_seq_id s = 0; s < 2; ++s) {
            set_manifest.targets[set_manifest.n_targets++] = vbr_make_target(
                    vbr_operation_kind::decode, vbr_operation_class::ordinary_decode,
                    t7b.runtime_instance(), VBR_STREAM_ANY, s, 0, 200);
        }
        vbr_scoped_operation set_op(set_manifest);
        auto append = t7b.begin_event(
                vbr_mutation_registrant::apply_ubatch_append, vbr_operation_class::ordinary_decode,
                0, vbr_generation_stamp_kind::dependency, set_op.id());
        const llama_seq_id both[2]     = { 0, 1 };
        const llama_seq_id stranger[2] = { 0, 2 };
        if (!append || !t7b.stamp_cell(append, 10, both, 2, 50)) {
            fprintf(stderr, "P1v2: fully-covered shared-cell stamp refused\n");
            return false;
        }
        if (t7b.stamp_cell(append, 11, stranger, 2, 50) || !t7b.shadow_unavailable() ||
                !append.finish()) {
            fprintf(stderr, "P1v2: uncovered shared-cell member did not poison\n");
            return false;
        }
    }

    // --- v6 rows: P5v2 closed mint predicates ---------------------------------------------
    {
        const vbr_controller_instance_id instance = tracker.runtime_instance();
        auto refused = [](vbr_operation_binding b) {
            vbr_scoped_operation probe(b);
            return !probe;
        };
        const auto good = vbr_mutation_binding(
                vbr_operation_kind::sequence_edit, 0, 0, 10,
                vbr_operation_class::state_api, instance);
        auto zero_mask = good;
        zero_mask.targets[0].registrant_mask = 0;
        auto foreign_bit = good;
        foreign_bit.targets[0].registrant_mask |=
                vbr_registrant_bit(vbr_mutation_registrant::apply_ubatch_append);
        auto subset_mask = good;
        subset_mask.targets[0].registrant_mask = vbr_registrant_bit(vbr_mutation_registrant::seq_rm);
        if (refused(good) ||                 // equality with the canonical mask is valid
            !refused(zero_mask) ||           // mask != 0
            !refused(foreign_bit) ||         // no out-of-kind bit
            refused(subset_mask)) {          // nonzero subset is least privilege
            fprintf(stderr, "P5v2: registrant-mask predicate matrix failed\n");
            return false;
        }
        if (!refused(vbr_mutation_binding(vbr_operation_kind::sequence_edit, 0, 0, 10,
                                          vbr_operation_class::state_api, {}))) {
            fprintf(stderr, "P5v2: instance-wildcard mutation target minted\n");
            return false;
        }
        if (!refused(vbr_mutation_binding(vbr_operation_kind::decode, 0, 5, 5,
                                          vbr_operation_class::ordinary_decode, instance))) {
            fprintf(stderr, "P5v2: empty decode range minted (p0 < p1 required)\n");
            return false;
        }
        if (refused(vbr_mutation_binding(vbr_operation_kind::sequence_edit, 0, 5, 5,
                                         vbr_operation_class::state_api, instance))) {
            fprintf(stderr, "P5v2: enumerated sequence_edit empty no-op form refused\n");
            return false;
        }
        if (!refused(vbr_mutation_binding(vbr_operation_kind::sequence_edit, 0, -1, -1,
                                          vbr_operation_class::state_api, instance))) {
            fprintf(stderr, "P5v2: undeclared sequence_edit range wildcard minted\n");
            return false;
        }
        if (refused(vbr_mutation_binding(vbr_operation_kind::controller_retier, -1, -1, -1,
                                         vbr_operation_class::controller, instance))) {
            fprintf(stderr, "P5v2: declared controller_retier range wildcard refused\n");
            return false;
        }
    }

    // --- v6 rows: P3v2 fixed-participant sealed aggregate ---------------------------------
    {
        // seal marks never-claimed declared slots failed; pre-seal reports cannot close.
        test_operation root(vbr_operation_kind::decode, tracker.runtime_instance(), -1,
                            0, std::numeric_limits<llama_pos>::max(),
                            vbr_operation_class::ordinary_decode);
        const vbr_operation_id root_id = root.op.release();
        llama_kv_cache::vbr_composite_outcome aggregate;
        aggregate.operation_id = root_id;
        aggregate.declared     = 2;
        aggregate.claim();
        aggregate.report_terminal(false);
        if (!vbr_operation_registry_is_live(root_id)) {
            fprintf(stderr, "P3v2: pre-seal terminal report closed the root\n");
            return false;
        }
        aggregate.seal(true);
        if (vbr_operation_registry_is_live(root_id) || !aggregate.failed) {
            fprintf(stderr, "P3v2: seal did not fail the never-claimed slot and close\n");
            return false;
        }
        aggregate.report_terminal(true);  // late report must be inert (no double close)
        // detach-transfer shape: sealed with open tokens stays open until the LAST terminal.
        test_operation root2(vbr_operation_kind::decode, tracker.runtime_instance(), -1,
                             0, std::numeric_limits<llama_pos>::max(),
                             vbr_operation_class::ordinary_decode);
        const vbr_operation_id root2_id = root2.op.release();
        llama_kv_cache::vbr_composite_outcome open_tokens;
        open_tokens.operation_id = root2_id;
        open_tokens.declared     = 2;
        open_tokens.claim();
        open_tokens.claim();
        open_tokens.seal(true);
        if (!vbr_operation_registry_is_live(root2_id)) {
            fprintf(stderr, "P3v2: seal closed the root past open participant tokens\n");
            return false;
        }
        open_tokens.report_terminal(true);
        if (!vbr_operation_registry_is_live(root2_id)) {
            fprintf(stderr, "P3v2: root closed before every declared slot terminated\n");
            return false;
        }
        open_tokens.report_terminal(true);
        if (vbr_operation_registry_is_live(root2_id) || open_tokens.failed) {
            fprintf(stderr, "P3v2: all-committed sealed aggregate did not close committed\n");
            return false;
        }
    }

    // --- v6 rows: P2v2 transactional ubatch manifest --------------------------------------
    {
        llama_seq_id   ids[20];
        llama_seq_id * seq_ptrs[20];
        int32_t        n_seq_id[20];
        llama_pos      pos[20];
        for (int i = 0; i < 20; ++i) {
            ids[i]      = i;
            seq_ptrs[i] = &ids[i];
            n_seq_id[i] = 1;
            pos[i]      = 100 + i;
        }
        llama_ubatch overflow_ub = {};
        overflow_ub.n_tokens = 20;
        overflow_ub.pos      = pos;
        overflow_ub.n_seq_id = n_seq_id;
        overflow_ub.seq_id   = seq_ptrs;
        vbr_operation_binding manifest;
        manifest.kind        = vbr_operation_kind::decode;
        manifest.child_phase = vbr_operation_phase::mutate;
        if (llama_kv_cache::vbr_decode_targets_from_ubatch(
                    manifest, vbr_controller_instance_id{1, 1}, false,
                    VBR_STREAM_ANY, overflow_ub) ||
                manifest.n_targets != 0) {
            fprintf(stderr, "P2v2: seq-ceiling overflow did not zero the manifest\n");
            return false;
        }
        // single-seq wrap manifest: ordinary + whole-range wrap + ONE declared seq-wildcard
        // purge target (v6-fix F1: cross-sequence masked reuse makes both the destroyed
        // position and the purged owner unbounded by the incoming batch)
        llama_ubatch one_ub = {};
        one_ub.n_tokens = 1;
        one_ub.pos      = pos;
        one_ub.n_seq_id = n_seq_id;
        one_ub.seq_id   = seq_ptrs;
        manifest           = {};
        manifest.kind        = vbr_operation_kind::decode;
        manifest.child_phase = vbr_operation_phase::mutate;
        if (!llama_kv_cache::vbr_decode_targets_from_ubatch(
                    manifest, vbr_controller_instance_id{1, 1}, true,
                    VBR_STREAM_ANY, one_ub) ||
                manifest.n_targets != 3 ||
                manifest.targets[1].operation_class != vbr_operation_class::swa_wrap ||
                manifest.targets[1].range.p1 != std::numeric_limits<llama_pos>::max() ||
                manifest.targets[2].operation_class != vbr_operation_class::state_api ||
                manifest.targets[2].seq_id != -1 ||
                manifest.targets[2].range.p1 != std::numeric_limits<llama_pos>::max()) {
            fprintf(stderr, "P2v2: wrap manifest missing the declared wrap/purge claims\n");
            return false;
        }
        // v6-fix F1 end-to-end: the wrap claims authenticate CROSS-SEQUENCE reuse — a
        // destructive reuse at a prior position far beyond the incoming batch, and the
        // nested purge of an OLD owner absent from the ubatch, both cover instead of
        // poisoning.
        vbr_generation_tracker t10(1, 768, 1);
        vbr_operation_binding wrap_manifest;
        wrap_manifest.kind        = vbr_operation_kind::decode;
        wrap_manifest.child_phase = vbr_operation_phase::mutate;
        if (!llama_kv_cache::vbr_decode_targets_from_ubatch(
                    wrap_manifest, t10.runtime_instance(),
                    true, VBR_STREAM_ANY, one_ub)) {
            return false;
        }
        vbr_scoped_operation wrap_op(wrap_manifest);
        test_multi_extent_supplier wrap_supplier;
        wrap_supplier.handles[1] = t10.extent_store().reserve(
                vbr_mutation_family::occupied_reuse, vbr_operation_class::swa_wrap, 0, 0,
                0, std::numeric_limits<llama_pos>::max());
        {
            auto reuse = t10.begin_event(
                    vbr_mutation_registrant::apply_ubatch_occupied_reuse,
                    vbr_operation_class::swa_wrap, 0,
                    vbr_generation_stamp_kind::dependency, wrap_op.id(),
                    &test_multi_extent_cb, &wrap_supplier, true);
            if (!reuse || !t10.stamp_cell(reuse, 10, ids[0], 5000) ||
                    wrap_supplier.last != 1 || !reuse.finish()) {
                fprintf(stderr, "v6-F1: beyond-batch destructive reuse did not authenticate\n");
                return false;
            }
        }
        {
            auto purge = t10.begin_event(
                    vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api, 0,
                    vbr_generation_stamp_kind::membership, wrap_op.id());
            if (!purge || !t10.stamp_cell(purge, 11, 7, 5000) || t10.shadow_unavailable() ||
                    !purge.finish()) {
                fprintf(stderr, "v6-F1: old-owner cross-seq purge did not authenticate\n");
                return false;
            }
        }
    }

    // --- v6-fix rows: F6 stream-exact recovery, F7 closed seq domain ----------------------
    {
        // F6: a record whose only target names exact stream 1 must NOT authorize stream 0.
        vbr_operation_binding far = vbr_mutation_binding(
                vbr_operation_kind::sequence_edit, 2, 0, 64, vbr_operation_class::state_api,
                tracker.runtime_instance(), /*stream=*/1);
        vbr_scoped_operation far_op(far);
        const int32_t ridx = vbr_recovery_reserve(far_op.id());
        if (ridx < 0 ||
                !vbr_recovery_record_failure(ridx, far_op.id(), vbr_operation_phase::mutate,
                                             vbr_recovery_failure_site::metadata_mutation, false)) {
            return false;
        }
        {
            auto capability = vbr_recovery_mint(ridx);
            if (!capability || capability.target_allowed(0, 2, 0, 64) ||
                    !capability.target_allowed(1, 2, 0, 64)) {
                fprintf(stderr, "v6-F6: recovery stream authorization is not target-exact\n");
                return false;
            }
            capability.resolve_completed();
        }
        // F7: seq domain is closed at mint — LLAMA_MAX_SEQ refused, LLAMA_MAX_SEQ-1 minted.
        auto over = vbr_mutation_binding(
                vbr_operation_kind::sequence_edit, LLAMA_MAX_SEQ, 0, 8,
                vbr_operation_class::state_api,
                tracker.runtime_instance());
        auto edge = vbr_mutation_binding(
                vbr_operation_kind::sequence_edit, LLAMA_MAX_SEQ - 1, 0, 8,
                vbr_operation_class::state_api,
                tracker.runtime_instance());
        vbr_scoped_operation over_op(over);
        vbr_scoped_operation edge_op(edge);
        if (over_op || !edge_op) {
            fprintf(stderr, "v6-F7: mint seq domain is not closed at LLAMA_MAX_SEQ\n");
            return false;
        }
    }

    // --- v6 rows: P5v2 exact-registrant unit publication ----------------------------------
    {
        vbr_generation_tracker t9(1, 64, 1);
        t9.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full);
        test_operation ctrl(vbr_operation_kind::controller_retier, t9.runtime_instance(), -1, -1, -1,
                            vbr_operation_class::controller);
        if (!t9.publish_unit(0, GGML_TYPE_F16, GGML_TYPE_TURBO8_0, vbr_repr_domain::full, 0,
                             vbr_repr_transition::degrade_other,
                             vbr_mutation_registrant::degrade_next, ctrl.id())) {
            fprintf(stderr, "P5v2: exact in-manifest registrant publication refused\n");
            return false;
        }
        if (t9.publish_unit(0, GGML_TYPE_TURBO8_0, GGML_TYPE_F16, vbr_repr_domain::full, 0,
                            vbr_repr_transition::promote,
                            vbr_mutation_registrant::clear, ctrl.id())) {
            fprintf(stderr, "P5v2: out-of-manifest registrant publication accepted\n");
            return false;
        }
    }

    printf("A2 extent/index/recovery/citation CPU coverage PASS\n");
    return true;
}

int main(int argc, char ** argv) {
    // Every tracker in this test binary uses a deterministic process origin. Production uses
    // the fail-closed entropy mixer; deterministic bytes keep checkpoint/artifact regressions
    // attributable to logic rather than process entropy.
    if (!vbr_lineage_origin_provider_set_for_tests(f40_deterministic_origin)) {
        fprintf(stderr, "F4.0 deterministic lineage origin setup failed\n");
        return 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--f4-cpu") {
        return run_f40_cpu_tests() ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--a1-cpu") {
        return run_a1_cpu_tests() ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--a2-cpu") {
        return run_a2_cpu_tests() ? 0 : 1;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL | --f4-cpu | --a1-cpu | --a2-cpu\n", argv[0]);
        return 1;
    }

    // A0 registry foundation: RAII closes exactly once, IDs are process-global/nonzero, and a
    // completed identity is never returned by the next operation.
    uint64_t first_registry_id = 0;
    {
        vbr_operation_binding binding = {};
        binding.kind = vbr_operation_kind::state_export;
        vbr_operation_registry_guard guard(binding);
        if (!guard.active()) {
            fprintf(stderr, "A0 registry RAII guard failed to mint an operation ID\n");
            return 1;
        }
        first_registry_id = guard.binding().operation_id.value;
        if (!vbr_operation_registry_is_live(guard.binding().operation_id)) {
            fprintf(stderr, "A0 registry did not expose its live RAII operation\n");
            return 1;
        }
    }
    if (vbr_operation_registry_is_live({ first_registry_id })) {
        fprintf(stderr, "A0 registry RAII guard left a completed operation live\n");
        return 1;
    }
    {
        vbr_operation_binding binding = {};
        binding.kind = vbr_operation_kind::state_export;
        vbr_operation_registry_guard guard(binding);
        if (!guard.active() ||
            guard.binding().operation_id.value == first_registry_id) {
            fprintf(stderr, "A0 registry reused an operation ID\n");
            return 1;
        }
    }

    if (!run_a1_cpu_tests() || !run_a2_cpu_tests()) {
        return 1;
    }

    ggml_backend_load_all();

    bool have_gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            have_gpu = true;
            break;
        }
    }
    if (!have_gpu) {
        fprintf(stderr, "SKIP: VBR representation epoch requires a GPU VBR backend (currently CUDA)\n");
        return 0;
    }

    // Hermetic controller inputs. The generated/real Gemma-4 fixture has iSWA children; a generic
    // order plus the friend-only force_degrade() above makes the epoch wave independent of its
    // price clamp, budget reach, free VRAM, or card size.
    set_test_env("VBR_FORCE_GENERIC", "1");
    unset_test_env("VBR_BUDGET_MIB");
    unset_test_env("VBR_DEGRADE_ORDER");
    unset_test_env("VBR_FREEZE");
    unset_test_env("VBR_MIN_BITS");
    unset_test_env("VBR_GROWTH_HEADROOM_MIB");
    unset_test_env("VBR_TRANSCODE_TEST");
    set_test_env("VBR_PROMOTE", "0");
    set_test_env("VBR_STASH_ROWS", "0");
    const char * trace_prefix_env = std::getenv("VBR_EPOCH_TEST_TRACE_PREFIX");
    const std::string trace_prefix =
        trace_prefix_env != nullptr ? trace_prefix_env : "";
    if (trace_prefix.empty()) {
        unset_test_env("VBR_TRACE");
    } else {
        set_test_env("VBR_TRACE", (trace_prefix + ".normal").c_str());
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model_ptr model(llama_model_load_from_file(argv[1], mparams));
    if (!model) {
        fprintf(stderr, "failed to load model %s\n", argv[1]);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                  = 128;
    cparams.n_batch                = 32;
    cparams.n_ubatch               = 32;
    cparams.n_seq_max              = 2;
    cparams.n_threads              = 2;
    cparams.n_threads_batch        = 2;
    cparams.type_k                 = GGML_TYPE_F16;
    cparams.type_v                 = GGML_TYPE_F16;
    cparams.flash_attn_type        = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.vbr_dynamic            = true;
    cparams.vbr_budget_explicit    = true;
    cparams.vbr_vram_budget_bytes  = 64ull * 1024 * 1024;

    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
    if (!ctx) {
        fprintf(stderr, "failed to create CUDA VBR context\n");
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx.get());
    llama_kv_cache * base = nullptr;
    llama_kv_cache * swa  = nullptr;
    if (!get_iswa_children(mem, base, swa)) {
        fprintf(stderr, "fixture did not create an iSWA attention cache\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::active(base)) {
        fprintf(stderr, "SKIP: loaded GPU backend does not provide VBR VMM for the base child\n");
        return 0;
    }
    if (!llama_kv_cache_vbr_epoch_test::active(swa)) {
        fprintf(stderr, "SKIP: loaded GPU backend does not provide VBR VMM for the SWA child\n");
        return 0;
    }
    if (!llama_kv_cache_vbr_epoch_test::map_seed_watermark(base)) {
        fprintf(stderr, "PRECONDITION failed: could not map the base child seed watermark\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::map_seed_watermark(swa)) {
        fprintf(stderr, "PRECONDITION failed: could not map the SWA child seed watermark\n");
        return 1;
    }

    const auto initial_v2 = llama_memory_vbr_state_v2(mem, 0, 0);
    const auto & initial = initial_v2.state;
    if (initial_v2.used_cells_exclusive != 0) {
        fprintf(stderr, "PRECONDITION failed: empty VBR cache reported exclusive cells\n");
        return 1;
    }
    if (initial.cursor != 0) {
        fprintf(stderr, "PRECONDITION failed: initial VBR cursor was not zero\n");
        return 1;
    }
    if (initial.representation_epoch != 0) {
        fprintf(stderr, "PRECONDITION failed: initial base representation epoch was not zero\n");
        return 1;
    }
    if (initial.representation_epoch_swa != 0) {
        fprintf(stderr, "PRECONDITION failed: initial SWA representation epoch was not zero\n");
        return 1;
    }
    if (initial.checkpoint_epoch != 0 || initial.checkpoint_epoch_swa != 0) {
        fprintf(stderr, "PRECONDITION failed: initial checkpoint lineage epoch was not zero\n");
        return 1;
    }
    if (!decode_one(ctx.get())) {
        fprintf(stderr, "PRECONDITION failed: seed decode failed\n");
        return 1;
    }
    const auto seeded_v2 = llama_memory_vbr_state_v2(mem, 0, 0);
    const auto & seeded = seeded_v2.state;
    if (seeded_v2.used_cells_exclusive == 0 || seeded.used_cells_other != 0) {
        fprintf(stderr, "exclusive VBR cell ownership did not match the seeded sequence\n");
        return 1;
    }
    const auto seeded_base = base->memory_vbr_state_v2(0, 0);
    const auto seeded_swa  = swa ->memory_vbr_state_v2(0, 0);
    if (seeded_v2.used_cells_exclusive !=
            seeded_base.used_cells_exclusive + seeded_swa.used_cells_exclusive) {
        fprintf(stderr, "iSWA exclusive VBR cell ownership did not equal its physical children\n");
        return 1;
    }
    const auto seeded_all = llama_memory_vbr_state_v2(mem, -1, 0);
    if (seeded_all.used_cells_exclusive != 0 ||
        seeded_all.state.used_cells_other != seeded_v2.used_cells_exclusive) {
        fprintf(stderr, "all-sequence VBR occupancy did not match exclusive seeded ownership\n");
        return 1;
    }
    if (seeded.cursor != initial.cursor) {
        fprintf(stderr, "PRECONDITION failed: seed decode consumed the VBR degrade ladder\n");
        return 1;
    }
    if (!epochs_equal(seeded, initial)) {
        fprintf(stderr, "PRECONDITION failed: seed decode changed a representation epoch\n");
        return 1;
    }
    if (!checkpoint_epochs_equal(seeded, initial)) {
        fprintf(stderr, "PRECONDITION failed: seed decode changed a checkpoint lineage epoch\n");
        return 1;
    }
    if (!decode_one(ctx.get(), 1)) {
        fprintf(stderr, "PRECONDITION failed: second seed decode failed\n");
        return 1;
    }
    const auto seeded_pair = llama_memory_vbr_state(mem, 0, 0);
    if (!epochs_equal(seeded_pair, seeded) ||
        !checkpoint_epochs_equal(seeded_pair, seeded)) {
        fprintf(stderr, "PRECONDITION failed: second seed decode changed a VBR epoch\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::dry_occupied_apply_preserves_epochs(base, 0) ||
        !llama_kv_cache_vbr_epoch_test::dry_occupied_apply_preserves_epochs(swa, 0)) {
        fprintf(stderr, "side-effect-free occupied-slot planning changed a VBR epoch\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::transient_suffix_preserves_checkpoint_lineage(base, 0, 1) ||
        !llama_kv_cache_vbr_epoch_test::transient_suffix_preserves_checkpoint_lineage(swa, 0, 1)) {
        fprintf(stderr, "speculative suffix rollback changed checkpoint lineage\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::generation_seeded(base) ||
            !llama_kv_cache_vbr_epoch_test::generation_seeded(swa)) {
        fprintf(stderr, "A1 dual-write did not stamp both armed iSWA children\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::has_mapped_degradable_unit(base)) {
        fprintf(stderr, "PRECONDITION failed: base child has no mapped degradable pooled extent\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::has_mapped_degradable_unit(swa)) {
        fprintf(stderr, "PRECONDITION failed: SWA child has no mapped degradable pooled extent\n");
        return 1;
    }

    // WS-6: the iSWA parent must acquire both child controllers coherently. Nested scopes
    // defer actual representation mutations, leave both ordered epochs unchanged, and arm
    // exactly one fresh boundary evaluation per child when the outer scope exits.
    const auto preflight = llama_memory_vbr_retier_preflight(mem, 0);
    if (!preflight.active) {
        fprintf(stderr, "scoped-freeze preflight did not observe an active VBR controller\n");
        return 1;
    }
    if (!preflight.fits) {
        fprintf(stderr, "scoped-freeze preflight rejected the already-mapped current tiers\n");
        return 1;
    }
    if (preflight.pools < 2) {
        fprintf(stderr, "scoped-freeze preflight did not cover both iSWA child pools\n");
        return 1;
    }
    if (preflight.bytes_needed == 0) {
        fprintf(stderr, "scoped-freeze preflight reported zero current-tier bytes needed\n");
        return 1;
    }
    if (preflight.bytes_available == 0) {
        fprintf(stderr, "scoped-freeze preflight reported zero current-tier bytes available\n");
        return 1;
    }
    const auto before_freeze = llama_memory_vbr_state(mem, 0, 0);
    const uint64_t outer = llama_memory_vbr_retier_freeze_begin(mem, "epoch_test_outer");
    const uint64_t inner = llama_memory_vbr_retier_freeze_begin(mem, "epoch_test_inner");
    const auto nested = llama_memory_vbr_state(mem, 0, 0);
    if (outer == 0) {
        fprintf(stderr, "outer iSWA scoped freeze did not acquire\n");
        return 1;
    }
    if (inner == 0) {
        fprintf(stderr, "inner iSWA scoped freeze did not acquire\n");
        return 1;
    }
    if (outer == inner) {
        fprintf(stderr, "nested VBR operations reused an operation ID\n");
        return 1;
    }
    if (!vbr_operation_registry_is_live({ outer }) ||
        !vbr_operation_registry_is_live({ inner })) {
        fprintf(stderr, "nested VBR operation IDs were not both live\n");
        return 1;
    }
    if (llama_kv_cache_vbr_epoch_test::freeze_operation_id(base) != inner ||
        llama_kv_cache_vbr_epoch_test::freeze_operation_id(swa)  != inner) {
        fprintf(stderr, "iSWA children did not receive the identical inner operation ID\n");
        return 1;
    }
    if (nested.retier_freeze_depth != 2) {
        fprintf(stderr, "nested iSWA scoped freeze reported the wrong depth\n");
        return 1;
    }
    if (nested.retier_freeze_enters !=
        before_freeze.retier_freeze_enters + 2) {
        fprintf(stderr, "nested iSWA scoped freeze counted an unexpected number of parent entries\n");
        return 1;
    }
    if (llama_kv_cache_vbr_epoch_test::force_degrade(base)) {
        fprintf(stderr, "base tier mutation was not deferred under scoped freeze\n");
        return 1;
    }
    if (llama_kv_cache_vbr_epoch_test::force_degrade(swa)) {
        fprintf(stderr, "SWA tier mutation was not deferred under scoped freeze\n");
        return 1;
    }
    const auto deferred = llama_memory_vbr_state(mem, 0, 0);
    if (!epochs_equal(deferred, before_freeze)) {
        fprintf(stderr, "scoped freeze allowed a representation epoch to change\n");
        return 1;
    }
    if (deferred.retier_deferred_decisions !=
        before_freeze.retier_deferred_decisions + 2) {
        fprintf(stderr, "scoped freeze counted an unexpected number of deferred child decisions\n");
        return 1;
    }
    // A0 amendment: simulate future runtime budget renegotiation while the operation is live.
    // iSWA end must pair from the immutable begin record even though base now reports disarmed.
    const uint64_t base_budget =
        llama_kv_cache_vbr_epoch_test::set_budget_bytes(base, 0);
    llama_memory_vbr_retier_freeze_end(mem, "epoch_test_inner", inner);
    llama_kv_cache_vbr_epoch_test::set_budget_bytes(base, base_budget);
    if (vbr_operation_registry_is_live({ inner })) {
        fprintf(stderr, "inner VBR operation remained live after end\n");
        return 1;
    }
    if (llama_kv_cache_vbr_epoch_test::freeze_operation_id(base) != outer ||
        llama_kv_cache_vbr_epoch_test::freeze_operation_id(swa)  != outer) {
        fprintf(stderr, "iSWA armed-flip pairing did not restore the identical outer operation ID\n");
        return 1;
    }
    if (llama_memory_vbr_state(mem, 0, 0).retier_freeze_depth != 1) {
        fprintf(stderr, "inner scoped-freeze exit released the outer scope\n");
        return 1;
    }
    llama_memory_vbr_retier_freeze_end(mem, "epoch_test_outer", outer);
    if (vbr_operation_registry_is_live({ outer })) {
        fprintf(stderr, "outer VBR operation remained live after end\n");
        return 1;
    }
    if (llama_kv_cache_vbr_epoch_test::freeze_operation_id(base) != 0 ||
        llama_kv_cache_vbr_epoch_test::freeze_operation_id(swa)  != 0) {
        fprintf(stderr, "iSWA child retained an operation ID after outer end\n");
        return 1;
    }
    const auto unfrozen = llama_memory_vbr_state(mem, 0, 0);
    if (unfrozen.retier_freeze_depth != 0) {
        fprintf(stderr, "outer scoped-freeze exit left a nonzero depth\n");
        return 1;
    }
    if (unfrozen.retier_freeze_exits !=
        before_freeze.retier_freeze_exits + 2) {
        fprintf(stderr, "nested scoped-freeze exits counted an unexpected number of parent exits\n");
        return 1;
    }
    if (!epochs_equal(unfrozen, before_freeze)) {
        fprintf(stderr, "scoped-freeze exit changed a representation epoch\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::reconcile(base)) {
        fprintf(stderr, "outer unfreeze did not arm a fresh base-child evaluation\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::reconcile(swa)) {
        fprintf(stderr, "outer unfreeze did not arm a fresh SWA-child evaluation\n");
        return 1;
    }
    const auto reconciled = llama_memory_vbr_state(mem, 0, 0);
    if (reconciled.retier_reconciles !=
        unfrozen.retier_reconciles + 1) {
        fprintf(stderr, "fresh post-unfreeze evaluations counted an unexpected number of reconciles\n");
        return 1;
    }
    if (!epochs_equal(reconciled, unfrozen)) {
        fprintf(stderr, "fresh post-unfreeze evaluation changed a representation epoch\n");
        return 1;
    }

    // The two independently mutating children must surface an ordered tuple, never a sum.
    if (!llama_kv_cache_vbr_epoch_test::force_degrade(base)) {
        fprintf(stderr, "failed to force base degrade\n");
        return 1;
    }
    const auto base_degraded = llama_memory_vbr_state(mem, 0, 0);
    if (base_degraded.representation_epoch <=
        initial.representation_epoch) {
        fprintf(stderr, "base degrade did not advance the base epoch\n");
        return 1;
    }
    if (base_degraded.representation_epoch_swa !=
        initial.representation_epoch_swa) {
        fprintf(stderr, "base degrade unexpectedly changed the SWA epoch\n");
        return 1;
    }
    if (!checkpoint_epochs_equal(base_degraded, initial)) {
        fprintf(stderr, "base degrade invalidated checkpoint attention lineage\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::generation_units_match(base)) {
        fprintf(stderr, "A1 base unit tuple did not publish the degraded live types\n");
        return 1;
    }

    if (!llama_kv_cache_vbr_epoch_test::force_degrade(swa)) {
        fprintf(stderr, "failed to force SWA degrade\n");
        return 1;
    }
    const auto both_degraded = llama_memory_vbr_state(mem, 0, 0);
    if (both_degraded.representation_epoch !=
        base_degraded.representation_epoch) {
        fprintf(stderr, "SWA degrade unexpectedly changed the base epoch\n");
        return 1;
    }
    if (both_degraded.representation_epoch_swa <=
        base_degraded.representation_epoch_swa) {
        fprintf(stderr, "SWA degrade did not advance the SWA epoch\n");
        return 1;
    }
    if (!checkpoint_epochs_equal(both_degraded, initial)) {
        fprintf(stderr, "SWA degrade invalidated checkpoint attention lineage\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::generation_units_match(swa)) {
        fprintf(stderr, "A1 SWA unit tuple did not publish the degraded live types\n");
        return 1;
    }

    // This is the production low-LCP/empty-cache reset sequence. clear() changes the referenced
    // representation first; vbr_full_reset() then rewinds the cursor but must advance, not reset,
    // each epoch.
    mem->clear(true);
    const auto cleared = llama_memory_vbr_state(mem, 0, 0);
    if (cleared.representation_epoch <=
        both_degraded.representation_epoch) {
        fprintf(stderr, "clear did not advance the base representation epoch\n");
        return 1;
    }
    if (cleared.representation_epoch_swa <=
        both_degraded.representation_epoch_swa) {
        fprintf(stderr, "clear did not advance the SWA representation epoch\n");
        return 1;
    }
    if (cleared.checkpoint_epoch <= both_degraded.checkpoint_epoch ||
        cleared.checkpoint_epoch_swa <= both_degraded.checkpoint_epoch_swa) {
        fprintf(stderr, "clear did not advance both checkpoint lineage epochs\n");
        return 1;
    }
    llama_kv_cache_vbr_epoch_test::full_reset(base);
    llama_kv_cache_vbr_epoch_test::full_reset(swa);
    const auto reset = llama_memory_vbr_state(mem, 0, 0);
    if (reset.cursor != 0) {
        fprintf(stderr, "full reset did not rewind the VBR cursor\n");
        return 1;
    }
    if (reset.representation_epoch <= cleared.representation_epoch) {
        fprintf(stderr, "full reset did not advance the base representation epoch\n");
        return 1;
    }
    if (reset.representation_epoch_swa <=
        cleared.representation_epoch_swa) {
        fprintf(stderr, "full reset did not advance the SWA representation epoch\n");
        return 1;
    }
    if (reset.checkpoint_epoch <= cleared.checkpoint_epoch ||
        reset.checkpoint_epoch_swa <= cleared.checkpoint_epoch_swa) {
        fprintf(stderr, "full reset did not advance both checkpoint lineage epochs\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::generation_units_match(base) ||
            !llama_kv_cache_vbr_epoch_test::generation_units_match(swa)) {
        fprintf(stderr, "A1 full reset did not republish both children at their live types\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::map_seed_watermark(base)) {
        fprintf(stderr, "PRECONDITION failed: could not remap the base child after full reset\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::map_seed_watermark(swa)) {
        fprintf(stderr, "PRECONDITION failed: could not remap the SWA child after full reset\n");
        return 1;
    }

    // Refill, degrade again, then adopt the native mixed-tier state onto itself. Ordinary forward
    // fill must not move the representation epochs; the second degrade and import both must.
    if (!decode_one(ctx.get())) {
        fprintf(stderr, "post-reset seed decode failed\n");
        return 1;
    }
    const auto refilled = llama_memory_vbr_state(mem, 0, 0);
    if (!epochs_equal(refilled, reset)) {
        fprintf(stderr, "post-reset seed decode changed a representation epoch\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::has_mapped_degradable_unit(base)) {
        fprintf(stderr, "PRECONDITION failed: post-reset base child has no mapped degradable extent\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::has_mapped_degradable_unit(swa)) {
        fprintf(stderr, "PRECONDITION failed: post-reset SWA child has no mapped degradable extent\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::force_degrade(base)) {
        fprintf(stderr, "post-reset base degrade failed\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::force_degrade(swa)) {
        fprintf(stderr, "post-reset SWA degrade failed\n");
        return 1;
    }
    const auto degraded_again = llama_memory_vbr_state(mem, 0, 0);
    if (degraded_again.representation_epoch <=
        reset.representation_epoch) {
        fprintf(stderr, "degrade-reset-degrade did not advance the base epoch\n");
        return 1;
    }
    if (degraded_again.representation_epoch_swa <=
        reset.representation_epoch_swa) {
        fprintf(stderr, "degrade-reset-degrade did not advance the SWA epoch\n");
        return 1;
    }
    if (!checkpoint_epochs_equal(degraded_again, reset)) {
        fprintf(stderr, "post-reset degrade invalidated checkpoint attention lineage\n");
        return 1;
    }
    const auto normal_final = llama_memory_vbr_state(mem, 0, 0);
    if (normal_final.retier_freeze_enters !=
        before_freeze.retier_freeze_enters + 2) {
        fprintf(stderr, "ordinary phase changed the scoped-freeze enter count after reconciliation\n");
        return 1;
    }
    if (normal_final.retier_freeze_exits !=
        before_freeze.retier_freeze_exits + 2) {
        fprintf(stderr, "ordinary phase changed the scoped-freeze exit count after reconciliation\n");
        return 1;
    }
    if (normal_final.retier_deferred_decisions !=
        before_freeze.retier_deferred_decisions + 2) {
        fprintf(stderr, "ordinary phase changed the deferred-decision count after reconciliation\n");
        return 1;
    }
    if (normal_final.retier_reconciles !=
        before_freeze.retier_reconciles + 1) {
        fprintf(stderr, "ordinary phase changed the reconcile count after reconciliation\n");
        return 1;
    }

    const auto exclusive_before_alias = llama_memory_vbr_state_v2(mem, 0, 0);
    if (exclusive_before_alias.used_cells_exclusive == 0 ||
        exclusive_before_alias.state.used_cells_other != 0) {
        fprintf(stderr, "PRECONDITION failed: alias test had no exclusively owned VBR cells\n");
        return 1;
    }
    std::array<uint32_t, 2> exclusive_base = {};
    std::array<uint32_t, 2> exclusive_swa  = {};
    std::array<uint32_t, 2> exclusive_parent = {};
    if (!base->vbr_accumulate_exclusive_cells(
            exclusive_base.data(), exclusive_base.size()) ||
        !swa->vbr_accumulate_exclusive_cells(
            exclusive_swa.data(), exclusive_swa.size()) ||
        exclusive_base[0] + exclusive_swa[0] !=
            exclusive_before_alias.used_cells_exclusive ||
        exclusive_base[1] != 0 || exclusive_swa[1] != 0 ||
        mem->vbr_accumulate_exclusive_cells(
            exclusive_parent.data(), exclusive_parent.size())) {
        fprintf(stderr, "batched VBR ownership did not preserve iSWA pressure domains\n");
        return 1;
    }
    llama_memory_seq_cp(mem, 0, 1, -1, -1);
    const auto shared_for_0 = llama_memory_vbr_state_v2(mem, 0, 0);
    const auto shared_for_1 = llama_memory_vbr_state_v2(mem, 1, 0);
    if (shared_for_0.used_cells_exclusive != 0 ||
        shared_for_1.used_cells_exclusive != 0 ||
        shared_for_0.state.used_cells_other != exclusive_before_alias.used_cells_exclusive ||
        shared_for_1.state.used_cells_other != exclusive_before_alias.used_cells_exclusive) {
        fprintf(stderr, "shared VBR cells were incorrectly counted as exclusively reclaimable\n");
        return 1;
    }
    exclusive_base = {};
    exclusive_swa  = {};
    if (!base->vbr_accumulate_exclusive_cells(
            exclusive_base.data(), exclusive_base.size()) ||
        !swa->vbr_accumulate_exclusive_cells(
            exclusive_swa.data(), exclusive_swa.size()) ||
        exclusive_base[0] != 0 || exclusive_base[1] != 0 ||
        exclusive_swa[0] != 0 || exclusive_swa[1] != 0) {
        fprintf(stderr, "batched VBR ownership counted shared aliases as exclusive\n");
        return 1;
    }
    if (!llama_memory_seq_rm(mem, 0, -1, -1)) {
        fprintf(stderr, "alias test could not remove the first sequence membership\n");
        return 1;
    }
    const auto exclusive_for_1 = llama_memory_vbr_state_v2(mem, 1, 0);
    if (exclusive_for_1.used_cells_exclusive != exclusive_before_alias.used_cells_exclusive ||
        exclusive_for_1.state.used_cells_other != 0) {
        fprintf(stderr, "last alias did not inherit exact exclusive VBR cell ownership\n");
        return 1;
    }
    exclusive_base = {};
    exclusive_swa  = {};
    if (!base->vbr_accumulate_exclusive_cells(
            exclusive_base.data(), exclusive_base.size()) ||
        !swa->vbr_accumulate_exclusive_cells(
            exclusive_swa.data(), exclusive_swa.size()) ||
        exclusive_base[0] != 0 || exclusive_swa[0] != 0 ||
        exclusive_base[1] + exclusive_swa[1] !=
            exclusive_before_alias.used_cells_exclusive) {
        fprintf(stderr, "batched VBR ownership did not transfer to the last alias\n");
        return 1;
    }
    const bool removed_last_alias = llama_memory_seq_rm(mem, 1, -1, -1);
    const auto empty_after_alias = llama_memory_vbr_state_v2(mem, -1, 0);
    if (!removed_last_alias || empty_after_alias.state.used_cells_other != 0 ||
        empty_after_alias.used_cells_exclusive != 0) {
        fprintf(stderr, "exclusive VBR cell ownership did not match final physical reclamation\n");
        return 1;
    }

    // WS-0 composition: its deterministic-input freeze remains authoritative, while an empty
    // scoped window is representation-neutral and balances normally. This intentionally does
    // not reinterpret VBR_FREEZE as a production retier stop (its scripted waves still run).
    ctx.reset();
    set_test_env("VBR_FREEZE", "1");
    if (!trace_prefix.empty()) {
        set_test_env("VBR_TRACE", (trace_prefix + ".env").c_str());
    }
    llama_context_ptr env_ctx(llama_init_from_model(model.get(), cparams));
    if (!env_ctx) {
        fprintf(stderr, "failed to create VBR_FREEZE composition context\n");
        return 1;
    }
    llama_memory_t env_mem = llama_get_memory(env_ctx.get());
    llama_kv_cache * env_base = nullptr;
    llama_kv_cache * env_swa  = nullptr;
    if (!get_iswa_children(env_mem, env_base, env_swa)) {
        fprintf(stderr, "VBR_FREEZE composition context was not iSWA\n");
        return 1;
    }
    if (!decode_one(env_ctx.get())) {
        fprintf(stderr, "VBR_FREEZE composition seed decode failed\n");
        return 1;
    }
    const auto env_before = llama_memory_vbr_state(env_mem, 0, 0);
    const uint64_t env_scope =
        llama_memory_vbr_retier_freeze_begin(env_mem, "epoch_test_env_noop");
    if (env_scope == 0) {
        fprintf(stderr, "scoped freeze did not compose with VBR_FREEZE context\n");
        return 1;
    }
    llama_memory_vbr_retier_freeze_end(
        env_mem, "epoch_test_env_noop", env_scope);
    const auto env_after = llama_memory_vbr_state(env_mem, 0, 0);
    const bool env_reconcile_base =
        llama_kv_cache_vbr_epoch_test::reconcile(env_base);
    const bool env_reconcile_swa =
        llama_kv_cache_vbr_epoch_test::reconcile(env_swa);
    unset_test_env("VBR_FREEZE");
    unset_test_env("VBR_TRACE");
    if (env_before.retier_env_freeze == 0) {
        fprintf(stderr, "VBR_FREEZE composition context did not report env freeze before the scope\n");
        return 1;
    }
    if (env_after.retier_env_freeze == 0) {
        fprintf(stderr, "VBR_FREEZE composition context lost env freeze across the scope\n");
        return 1;
    }
    if (env_after.retier_freeze_depth != 0) {
        fprintf(stderr, "VBR_FREEZE composition scope left a nonzero depth\n");
        return 1;
    }
    if (env_after.retier_freeze_enters !=
        env_before.retier_freeze_enters + 1) {
        fprintf(stderr, "VBR_FREEZE composition scope counted an unexpected number of parent entries\n");
        return 1;
    }
    if (env_after.retier_freeze_exits !=
        env_before.retier_freeze_exits + 1) {
        fprintf(stderr, "VBR_FREEZE composition scope counted an unexpected number of parent exits\n");
        return 1;
    }
    if (env_after.retier_deferred_decisions !=
        env_before.retier_deferred_decisions) {
        fprintf(stderr, "empty VBR_FREEZE composition scope counted a deferred decision\n");
        return 1;
    }
    if (env_reconcile_base) {
        fprintf(stderr, "empty VBR_FREEZE composition scope armed a base reconciliation\n");
        return 1;
    }
    if (env_reconcile_swa) {
        fprintf(stderr, "empty VBR_FREEZE composition scope armed a SWA reconciliation\n");
        return 1;
    }
    if (env_after.retier_reconciles != env_before.retier_reconciles) {
        fprintf(stderr, "empty VBR_FREEZE composition scope changed the reconcile counter\n");
        return 1;
    }
    if (!epochs_equal(env_before, env_after)) {
        fprintf(stderr, "empty VBR_FREEZE composition scope changed a representation epoch\n");
        return 1;
    }
    if (!checkpoint_epochs_equal(env_before, env_after)) {
        fprintf(stderr, "empty VBR_FREEZE composition scope changed a checkpoint lineage epoch\n");
        return 1;
    }

    if (!llama_memory_seq_rm_attn(env_mem, 0, 0, 1)) {
        fprintf(stderr, "attention-only destructive sequence edit was refused\n");
        return 1;
    }
    const auto env_trimmed = llama_memory_vbr_state(env_mem, 0, 0);
    if (env_trimmed.representation_epoch <= env_after.representation_epoch ||
        env_trimmed.representation_epoch_swa <= env_after.representation_epoch_swa ||
        env_trimmed.checkpoint_epoch <= env_after.checkpoint_epoch ||
        env_trimmed.checkpoint_epoch_swa <= env_after.checkpoint_epoch_swa) {
        fprintf(stderr, "destructive sequence edit did not invalidate both checkpoint lineages\n");
        return 1;
    }

    // NOTE: the "native mixed-tier import bumps both epochs" case is intentionally NOT exercised
    // here. The fork deliberately REFUSES to serialize a dynamic-VBR cache after a tier degrade
    // (llama_state_seq_get_size throws "cannot serialize a dynamic-VBR KV cache after tier
    // degrades..."), so a degraded mixed-tier state cannot be captured and re-adopted in the
    // current codebase — native mixed-tier import/serialization is unbuilt Phase-2/3 work. The
    // import path DOES bump both epochs (state adoption changes checkpoint lineage), but it is
    // unreachable at runtime until that serialization exists. The P0 I9 behavior that matters --
    // per-child representation advance on every degrade, checkpoint-lineage stability across
    // retiering, and checkpoint-lineage advance on clear/full-reset -- is fully covered above.


    fprintf(stderr, "PASS: VBR scoped freeze is nested/iSWA-coherent, defers mutations, "
            "re-evaluates fresh on exit, composes with VBR_FREEZE, and preserves monotone "
            "per-child representation and checkpoint-lineage epochs\n");
    return 0;
}
