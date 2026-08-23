#pragma once

#include "llama-cache-authority.h"
#include "llama-memory-tree.h"
#include "llama-vbr-artifact-catalog.h"
#include "llama-vbr-operation.h"
#include "llama-vbr-downward.h"
#include "llama-vbr-occupied-replacement.h"
#include "llama-vbr-upward.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct vbr_explicit_representation_identity;

// F4.1b is a pure evidence classifier. A decision describes the honest next
// action; the validation status separately says whether the evidence was
// complete enough to trust that decision.
enum class vbr_import_decision : uint8_t {
    native_import = 0,
    live_rebased,
    downward_rebase,
    upward_reconstruct,
    rebuild,
    cold,
    reject,
    _count,
};

enum class vbr_manifest_validation_status : uint8_t {
    validated = 0,
    unauthorized,
    unsupported_artifact_version,
    restore_metadata_missing,
    malformed,
    checksum_or_digest_mismatch,
    identity_mismatch,
    token_block_mismatch,
    memory_tree_mismatch,
    target_not_idle,
    target_not_empty,
    target_not_dedicated,
    target_not_armed,
    geometry_mismatch,
    topology_mismatch,
    representation_mismatch,
    codebook_mismatch,
    policy_mismatch,
    generation_mismatch,
    ownership_mismatch,
    stash_inconsistent,
    required_companion_unavailable,
    accounting_unavailable,
    budget_unavailable,
    native_lineage_unavailable,
    unavailable,
    internal_error,
    _count,
};

static_assert(uint8_t(vbr_import_decision::_count) == 7);
static_assert(uint8_t(vbr_manifest_validation_status::_count) == 27);

const char * vbr_import_decision_name(vbr_import_decision decision) noexcept;
const char * vbr_manifest_validation_status_name(
    vbr_manifest_validation_status status) noexcept;

struct vbr_target_validation_snapshot;

// Immutable classification of the artifact schedule against one canonical
// target snapshot.  This deliberately precedes any transcode/materialization:
// unsupported upward cases remain observable instead of being collapsed into
// a failed downward bind.
enum class vbr_import_schedule_status : uint8_t {
    unavailable = 0,
    exact,
    downward,
    upward_same_domain,
    upward_cross_domain,
    mixed_direction_unsupported,
    _count,
};
static_assert(uint8_t(vbr_import_schedule_status::_count) == 6);

enum class vbr_import_transform_kind : uint8_t {
    none = 0,
    downward,
    upward_same_domain,
    upward_cross_domain,
    _count,
};
static_assert(uint8_t(vbr_import_transform_kind::_count) == 4);

const char * vbr_import_schedule_status_name(
    vbr_import_schedule_status status) noexcept;

struct vbr_import_schedule_unit {
    uint32_t child_id = UINT32_MAX;
    uint32_t logical_unit_id = UINT32_MAX;
    int32_t source_type = -1;
    int32_t target_type = -1;
    vbr_repr_domain source_domain = vbr_repr_domain::full;
    vbr_repr_domain target_domain = vbr_repr_domain::full;

    bool operator==(const vbr_import_schedule_unit & other) const noexcept {
        return child_id == other.child_id &&
               logical_unit_id == other.logical_unit_id &&
               source_type == other.source_type &&
               target_type == other.target_type &&
               source_domain == other.source_domain &&
               target_domain == other.target_domain;
    }
};

// Canonical direction reducer used by the quote builder and model-free
// mutation tests. Rows must already carry canonical tier domains.
vbr_import_schedule_status vbr_classify_import_schedule_units(
    const std::vector<vbr_import_schedule_unit> & units) noexcept;

class vbr_import_schedule_quote {
public:
    vbr_import_schedule_status status() const noexcept { return status_; }
    const vbr_manifest_digest & manifest_digest() const noexcept {
        return manifest_digest_;
    }
    uint64_t source_payload_bytes() const noexcept {
        return source_payload_bytes_;
    }
    uint64_t target_mapped_bytes() const noexcept {
        return target_mapped_bytes_;
    }
    uint64_t accounting_serial() const noexcept {
        return accounting_serial_;
    }
    const std::vector<vbr_import_schedule_unit> & units() const noexcept {
        return units_;
    }
    const vbr_import_destination_projection & destination() const noexcept {
        return destination_;
    }

private:
    vbr_import_schedule_status status_ =
        vbr_import_schedule_status::unavailable;
    vbr_manifest_digest manifest_digest_;
    uint64_t memory_instance_cookie_ = 0;
    uint64_t target_state_serial_ = 0;
    uint64_t accounting_serial_ = 0;
    uint64_t tree_shape_digest_ = 0;
    uint64_t policy_epoch_ = 0;
    uint64_t source_payload_bytes_ = 0;
    uint64_t target_mapped_bytes_ = 0;
    std::vector<vbr_import_schedule_unit> units_;
    vbr_import_destination_projection destination_;

    friend bool vbr_quote_import_schedule(
        const vbr_target_validation_snapshot &,
        const vbr_artifact_package_view &,
        vbr_import_schedule_quote &) noexcept;
    friend bool vbr_import_schedule_quote_matches(
        const vbr_import_schedule_quote &,
        const vbr_target_validation_snapshot &,
        const vbr_artifact_package_view &) noexcept;
    friend bool vbr_rebind_import_schedule_quote(
        const vbr_target_validation_snapshot &,
        const vbr_artifact_package_view &,
        const vbr_import_destination_projection &,
        vbr_import_schedule_quote &) noexcept;
    friend class vbr_live_capture_adapter;
};

struct vbr_import_identity {
    std::string execution_identity;
    std::string adapter_config_identity;
    std::string media_content_identity;
    uint64_t sequence_epoch = 0;
    llama_pos requested_frontier = -1;
    // Borrowed from the package lease held for the whole validation call.
    // The validated proof retains that lease, so this remains valid while the
    // authenticated identity is observable from the proof.
    const std::vector<llama_token> * tokens = nullptr;
};

struct vbr_target_shard_snapshot {
    uint32_t shard_index = UINT32_MAX;
    const void * pool_cookie = nullptr;
    llama_cache_acct_resource_domain domain;
    uint32_t topology_index = UINT32_MAX;
    uint16_t device_ordinal = UINT16_MAX;
    llama_cache_acct_topology_digest topology_digest;
    uint64_t logical_offset = 0;
    uint64_t row_count = 0;
    uint64_t row_bytes = 0;
    uint64_t mapped_bytes = 0;
};

struct vbr_target_unit_snapshot {
    uint32_t child_id = UINT32_MAX;
    uint32_t logical_unit_id = UINT32_MAX;
    int32_t current_type = -1;
    int32_t last_source_type = -1;
    uint8_t promote_hops = 0;
    vbr_repr_transition last_transition = vbr_repr_transition::initial;
    vbr_artifact_representation_kind representation_kind =
        vbr_artifact_representation_kind::raw;
    uint32_t codec_id = 0;
    uint32_t codec_version = 0;
    std::array<uint8_t, 32> representation_reference_digest = {};
    uint32_t source_loss_history = 0;
    uint32_t checkpoint_codec_hops = 0;
    vbr_artifact_recoverability recoverability =
        vbr_artifact_recoverability::sealed_payload;
    vbr_artifact_side side = vbr_artifact_side::key;
    vbr_artifact_layout layout = vbr_artifact_layout::row_major;
    uint32_t row_codec_version = 0;
    vbr_repr_domain current_domain = vbr_repr_domain::full;
    std::array<uint8_t, 32> codebook_digest = {};
    std::array<uint8_t, 32> rotation_digest = {};
    std::array<uint8_t, 32> meansub_digest = {};
    int32_t meansub_model_id = -1;
    int32_t meansub_layer = -1;
    bool meansub_baked = false;
    uint32_t n_stream = 0;
    bool unified = false;
    bool v_trans = false;
    uint64_t wm_cells = 0;
    uint32_t rank = 0;
    std::array<uint64_t, 4> dimensions = {};
    uint64_t row_alignment = 0;
    std::vector<vbr_target_shard_snapshot> shards;
    bool downward_supported = false;
    bool downward_movable = false;
    int32_t controller_floor_type = -1;
    int32_t downward_type = -1;
    vbr_repr_domain downward_domain = vbr_repr_domain::full;
    uint32_t downward_recipe_id = 0;
    uint32_t downward_recipe_version = 0;
    std::array<uint8_t, 32> downward_build_identity_digest = {};
    uint64_t downward_row_bytes = 0;
    uint64_t downward_mapped_bytes = 0;
    uint64_t downward_transfer_bytes = 0;
    uint64_t downward_codec_workspace_bytes = 0;
    // F4.2 proof inputs. Each transform direction carries its one canonical
    // recipe and build identity; both bind the controller-owned selected tree.
    vbr_downward_recipe downward_recipe;
    int32_t downward_meansub_model_id = -1;
    bool upward_supported = false;
    int32_t upward_type = -1;
    vbr_repr_domain upward_domain = vbr_repr_domain::full;
    uint32_t upward_recipe_id = 0;
    uint32_t upward_recipe_version = 0;
    std::array<uint8_t, 32> upward_build_identity_digest = {};
    uint64_t upward_row_bytes = 0;
    uint64_t upward_mapped_bytes = 0;
    uint64_t upward_transfer_bytes = 0;
    uint64_t upward_codec_workspace_bytes = 0;
    vbr_upward_recipe upward_recipe;
    int32_t upward_meansub_model_id = -1;
    vbr_upward_representation_identity upward_source_identity;
    vbr_upward_representation_identity upward_target_identity;
};

struct vbr_target_child_snapshot {
    uint32_t child_id = UINT32_MAX;
    checkpoint_child_dependency_mode dependency_mode =
        checkpoint_child_dependency_mode::absent;
    const void * memory_cookie = nullptr;
    bool empty = false;
    bool dedicated = false;
    bool armed = false;
    bool previously_observed = false;
    bool generation_compatible = true;
    bool ownership_compatible = true;
    bool stash_compatible = true;
    vbr_lineage_uuid lineage_uuid;
    vbr_controller_instance_id instance_id;
    uint64_t state_serial = 0;
    uint64_t policy_epoch = 0;
    vbr_artifact_controller_policy controller_policy;
    std::vector<vbr_target_unit_snapshot> units;
};

struct vbr_target_companion_snapshot {
    vbr_artifact_companion_kind kind =
        vbr_artifact_companion_kind::_count;
    uint32_t format_version = 0;
    std::array<uint8_t, 32> build_identity_digest = {};
    bool available = false;
    const void * target_cookie = nullptr;
};

// Provider-specific, allocation-only parse result. Validators may build this
// bounded CPU image, but it exposes no installation method; F4.2 must consume
// it under the adoption journal rather than invoking legacy state_read().
class vbr_parsed_companion_image {
public:
    virtual ~vbr_parsed_companion_image();
    virtual vbr_artifact_companion_kind kind() const noexcept = 0;
    virtual uint32_t format_version() const noexcept = 0;

    vbr_parsed_companion_image(const vbr_parsed_companion_image &) = delete;
    vbr_parsed_companion_image & operator=(
        const vbr_parsed_companion_image &) = delete;

protected:
    vbr_parsed_companion_image() = default;
};

// Immutable result of one canonical target-tree inspection. Production fills
// it from the one llama_memory_tree_collect() walk; CPU tests may feed this
// value directly to the pure core.
struct vbr_target_validation_snapshot {
    uint64_t memory_instance_cookie = 0;
    uint64_t target_state_serial = 0;
    uint64_t accounting_serial = 0;
    uint64_t tree_shape_digest = 0;
    uint64_t policy_epoch = 0;
    bool scheduler_idle = false;
    bool destination_sequence_absent = false;
    std::vector<vbr_target_child_snapshot> children;
    std::vector<vbr_target_companion_snapshot> companions;
};

// Quotes an authenticated target image. Destination selection is controller-
// owned; the rebind overload attaches its already-normalized projected image
// without revalidating or rematerializing the package.
bool vbr_quote_import_schedule(
    const vbr_target_validation_snapshot & target,
    const vbr_artifact_package_view & package,
    vbr_import_schedule_quote & output) noexcept;
bool vbr_rebind_import_schedule_quote(
    const vbr_target_validation_snapshot & target,
    const vbr_artifact_package_view & package,
    const vbr_import_destination_projection & destination,
    vbr_import_schedule_quote & output) noexcept;
bool vbr_import_schedule_quote_matches(
    const vbr_import_schedule_quote & quote,
    const vbr_target_validation_snapshot & target,
    const vbr_artifact_package_view & package) noexcept;

struct vbr_target_empty_fingerprint;
class vbr_staged_payloads;
struct vbr_adopt_result;
struct vbr_composite_publish_hooks;

struct vbr_adopt_policy {
    using inspect_target_fn = bool (*)(
        const void * context,
        llama_memory_i & target,
        const std::vector<llama_memory_tree_child> & canonical_tree,
        vbr_target_validation_snapshot & output) noexcept;
    using serial_fn = uint64_t (*)(const void * context) noexcept;
    using transform_digest_fn = bool (*)(
        const void * context, std::array<uint8_t, 32> & output) noexcept;
    using target_recheck_fn = bool (*)(
        const void * context,
        const vbr_target_empty_fingerprint & expected) noexcept;
    using representation_identity_fn = bool (*)(
        const void * context,
        int32_t current_type,
        bool value_side,
        int32_t meansub_model_id,
        vbr_explicit_representation_identity & output) noexcept;
    using parse_companion_fn = bool (*)(
        const void * context,
        const vbr_artifact_companion_payload & descriptor,
        const artifact_segment_chain & source,
        const vbr_target_companion_snapshot & target,
        std::unique_ptr<vbr_parsed_companion_image> & output) noexcept;

    bool authorized = false;
    vbr_import_identity identity;
    llama_seq_id destination_sequence = -1;
    bool allow_native = true;
    bool allow_live_rebased = true;
    bool allow_downward = true;
    bool allow_upward = true;
    bool allow_rebuild = true;
    bool allow_cold = true;
    // Compatibility spelling: false means a fresh target runtime instance
    // could not be minted/enrolled. Source-lineage liveness is irrelevant.
    bool native_instance_available = true;
    // Caller-issued, nonzero, non-reusing capability token. Keeping issuance
    // outside the validator makes the evidence core a pure function of its
    // inputs while the manifest digest and target fingerprint prevent splice.
    uint64_t adoption_nonce = 0;
    std::vector<llama_vbr_artifact_domain_binding> domain_bindings;
    const llama_cache_acct_snapshot * accounting_snapshot = nullptr;
    const llama_cache_budget_config * budget_config = nullptr;
    const llama_cache_budget_plan * transform_budget_plan = nullptr;
    const vbr_downward_policy_projection * downward_projection = nullptr;
    const vbr_import_schedule_quote * schedule_quote = nullptr;
    const void * context = nullptr;
    inspect_target_fn inspect_target = nullptr;
    parse_companion_fn parse_companion = nullptr;
    target_recheck_fn recheck_target_empty = nullptr;
    serial_fn read_accounting_serial = nullptr;
    serial_fn read_policy_epoch = nullptr;
    transform_digest_fn read_transform_tree_digest = nullptr;
    // Optional one-shot authority for the narrow occupied-target importer.
    // Validation consumes it only after proving the exact singleton/full-
    // artifact envelope; ordinary construction-empty imports leave it null.
    vbr_occupied_replacement_guard * occupied_replacement = nullptr;
    const void * occupied_representation_context = nullptr;
    representation_identity_fn occupied_representation_identity = nullptr;
};

struct vbr_child_empty_fingerprint {
    uint32_t child_id = UINT32_MAX;
    const void * memory_cookie = nullptr;
    uint64_t state_serial = 0;
    vbr_controller_instance_id instance_id;
};

struct vbr_target_empty_fingerprint {
    uint64_t memory_instance_cookie = 0;
    uint64_t target_state_serial = 0;
    uint64_t accounting_serial = 0;
    uint64_t tree_shape_digest = 0;
    uint64_t policy_epoch = 0;
    bool previously_observed = false;
    std::vector<vbr_child_empty_fingerprint> children;
};

enum class vbr_validated_stash_action : uint8_t {
    restore_exact = 0,
    consume_exact_then_drop,
    omit_live_rebased,
    none_at_source,
    _count,
};

struct vbr_authorized_cell_run {
    uint32_t first_physical_cell = 0;
    uint32_t cell_count = 0;
};

// Prefix projections deliberately separate packed host rows from freshly
// allocated destination cells.  Ordinary whole-artifact plans leave this
// vector empty and retain their historical same-offset behavior.
struct vbr_validated_projection_run {
    uint64_t source_offset = 0;
    uint64_t destination_offset = 0;
    uint64_t size = 0;
};

struct vbr_validated_shard_plan {
    uint32_t shard_index = UINT32_MAX;
    llama_cache_acct_resource_domain domain;
    const void * target_pool_cookie = nullptr;
    uint64_t logical_offset = 0;
    uint64_t row_count = 0;
    uint64_t row_bytes = 0;
    uint64_t target_row_bytes = 0;
    uint64_t target_mapped_bytes = 0;
    uint64_t payload_bytes = 0;
    std::shared_ptr<const artifact_segment_chain> source;
    std::vector<vbr_validated_projection_run> projection_runs;
    vbr_capture_range_proof projection_proof;
};

struct vbr_validated_child_plan {
    uint32_t child_id = UINT32_MAX;
    checkpoint_child_dependency_mode dependency_mode =
        checkpoint_child_dependency_mode::absent;
    uint32_t logical_unit_id = UINT32_MAX;
    const void * target_pool_cookie = nullptr;
    vbr_artifact_unit_descriptor descriptor;
    std::vector<vbr_validated_shard_plan> shards;
    std::vector<vbr_authorized_cell_run> authorized_runs;
    std::vector<vbr_artifact_stream_placement> placements;
    int32_t selected_target_type = -1;
    vbr_repr_domain source_domain = vbr_repr_domain::full;
    vbr_repr_domain selected_target_domain = vbr_repr_domain::full;
    uint32_t transcode_recipe_id = 0;
    uint32_t transcode_recipe_version = 0;
    std::array<uint8_t, 32> transcode_build_identity_digest = {};
    vbr_downward_recipe transcode_recipe;
    vbr_upward_recipe upward_recipe;
    std::array<uint8_t, 32> transcode_policy_digest = {};
    std::array<uint8_t, 32> transcode_tree_digest = {};
    int32_t transcode_meansub_model_id = -1;
    vbr_upward_representation_identity transcode_source_identity;
    vbr_upward_representation_identity transcode_target_identity;
    uint64_t target_controller_cursor = 0;
    vbr_import_transform_kind transform_kind =
        vbr_import_transform_kind::none;
    // Controller generation and live extent metadata published by adoption.
    // Tapped upward reconstruction is one additional lossy promotion hop;
    // full-domain T8->F16 retains the established whole-import reset.
    int32_t target_last_source_type = -1;
    uint8_t target_promote_hops = 0;
    vbr_validated_stash_action stash_action =
        vbr_validated_stash_action::none_at_source;
    uint64_t target_row_bytes = 0;
    uint64_t target_mapped_bytes = 0;
    uint64_t transfer_bytes = 0;
    uint64_t codec_workspace_bytes = 0;
    vbr_artifact_unit_reference unit_reference;
    vbr_artifact_controller_policy controller_policy;
    vbr_operation_target operation_target;
};

struct vbr_validated_companion_plan {
    vbr_artifact_companion_payload descriptor;
    const void * target_cookie = nullptr;
    std::shared_ptr<const artifact_segment_chain> source;
    std::unique_ptr<vbr_parsed_companion_image> parsed;
};

enum class vbr_tracker_install_transition : uint8_t {
    native_clone = 0,
    whole_import,
    _count,
};
static_assert(uint8_t(vbr_tracker_install_transition::_count) == 2);

struct vbr_tracker_install_child {
    uint32_t child_id = UINT32_MAX;
    vbr_tracker_install_transition transition =
        vbr_tracker_install_transition::_count;
    vbr_lineage_uuid lineage_uuid;
    vbr_controller_instance_id target_instance;
    uint64_t global_generation = 0;
    std::vector<vbr_checkpoint_unit_generation> units;
};

struct vbr_tracker_install_plan {
    std::vector<vbr_tracker_install_child> children;
};

struct vbr_manifest_validation_result;

class vbr_validated_manifest {
public:
    vbr_validated_manifest(vbr_validated_manifest &&) noexcept;
    vbr_validated_manifest & operator=(vbr_validated_manifest &&) noexcept;
    ~vbr_validated_manifest();

    vbr_validated_manifest(const vbr_validated_manifest &) = delete;
    vbr_validated_manifest & operator=(const vbr_validated_manifest &) = delete;

    vbr_import_decision decision() const noexcept { return decision_; }
    const vbr_target_empty_fingerprint & target() const noexcept { return target_; }
    const vbr_manifest_digest & manifest_digest() const noexcept { return manifest_digest_; }
    const vbr_capture_generation_id & capture_generation_id() const noexcept {
        return capture_generation_id_;
    }
    const vbr_import_identity & authenticated_identity() const noexcept {
        return authenticated_identity_;
    }
    const vbr_artifact_token_block & token_block() const noexcept { return token_block_; }
    const std::vector<vbr_validated_child_plan> & children() const noexcept { return children_; }
    const std::vector<vbr_validated_companion_plan> & companions() const noexcept {
        return companions_;
    }
    const std::vector<llama_cache_transaction_leaf> & accounting_leaves() const noexcept {
        return accounting_leaves_;
    }
    const vbr_tracker_install_plan & tracker_install() const noexcept {
        return tracker_install_;
    }
    const vbr_checkpoint_generation_controller * source_controller(
            uint32_t child_id) const noexcept {
        const auto found = std::find_if(
            source_controllers_.begin(), source_controllers_.end(),
            [&](const auto & value) { return value.child_id == child_id; });
        return found == source_controllers_.end() ? nullptr : &*found;
    }
    uint64_t adoption_nonce() const noexcept { return adoption_nonce_; }
    llama_cache_acct_artifact_id source_artifact() const noexcept {
        return source_projection_
            ? source_projection_.parent_artifact()
            : source_lease_.reference_artifact();
    }
    // F4.2 staging retains this lease through the proof and uses the same
    // canonical package door for its pre-transfer repeatability check.
    const vbr_artifact_package_view & source_package() const noexcept {
        return source_lease_;
    }
    bool is_prefix_projection() const noexcept {
        return bool(source_projection_);
    }
    const vbr_artifact_attention_prefix_projection &
    source_projection() const noexcept {
        return source_projection_;
    }
    // Allocation-free catalog/borrow recheck used immediately before the
    // proof-aware H2D operation begins.
    bool projection_transfer_ready() const noexcept {
        return source_projection_ && source_projection_.transfer_ready();
    }
    bool is_occupied_replacement() const noexcept {
        return occupied_replacement_ != nullptr;
    }
    const vbr_occupied_replacement_guard * occupied_replacement() const noexcept {
        return occupied_replacement_.get();
    }
    const std::vector<vbr_occupied_replacement_relocation_run> &
    relocation_runs() const noexcept {
        static const std::vector<vbr_occupied_replacement_relocation_run> empty;
        return occupied_replacement_
            ? occupied_replacement_->relocation_runs() : empty;
    }

private:
    vbr_validated_manifest() = default;

    vbr_artifact_package_view source_lease_;
    vbr_artifact_attention_prefix_projection source_projection_;
    std::unique_ptr<vbr_occupied_replacement_guard> occupied_replacement_;
    vbr_import_decision decision_ = vbr_import_decision::reject;
    vbr_target_empty_fingerprint target_;
    vbr_manifest_digest manifest_digest_;
    vbr_capture_generation_id capture_generation_id_;
    vbr_import_identity authenticated_identity_;
    vbr_artifact_token_block token_block_;
    std::vector<vbr_validated_child_plan> children_;
    std::vector<vbr_validated_companion_plan> companions_;
    std::vector<llama_cache_transaction_leaf> accounting_leaves_;
    vbr_tracker_install_plan tracker_install_;
    std::vector<vbr_checkpoint_generation_controller> source_controllers_;
    uint64_t adoption_nonce_ = 0;
    const void * recheck_context_ = nullptr;
    vbr_adopt_policy::target_recheck_fn recheck_target_empty_ = nullptr;
    vbr_adopt_policy::serial_fn read_accounting_serial_ = nullptr;
    vbr_adopt_policy::serial_fn read_policy_epoch_ = nullptr;
    vbr_adopt_policy::transform_digest_fn
        read_transform_tree_digest_ = nullptr;
    const void * occupied_representation_context_ = nullptr;
    vbr_adopt_policy::representation_identity_fn
        occupied_representation_identity_ = nullptr;

    friend struct vbr_manifest_validation_result;
    friend vbr_manifest_validation_result vbr_validate_unit_manifest(
        llama_memory_i &, const vbr_artifact_package_view &,
        const vbr_adopt_policy &) noexcept;
    friend vbr_manifest_validation_result
    vbr_validate_unit_manifest_snapshot(
        const vbr_target_validation_snapshot &,
        const vbr_artifact_package_view &,
        const vbr_adopt_policy &) noexcept;
    friend vbr_manifest_validation_result
    vbr_validate_attention_prefix_projection(
        const vbr_target_validation_snapshot &,
        vbr_artifact_attention_prefix_projection &&,
        const vbr_adopt_policy &) noexcept;
    friend vbr_adopt_result vbr_adopt_empty_manifest(
        llama_memory_i &, llama_seq_id,
        vbr_validated_manifest &&, vbr_staged_payloads &&,
        llama_cache_acct_ledger &,
        const vbr_composite_publish_hooks &) noexcept;
};

struct vbr_manifest_validation_result {
    vbr_manifest_validation_status status =
        vbr_manifest_validation_status::internal_error;
    vbr_import_decision decision = vbr_import_decision::reject;
    std::unique_ptr<vbr_validated_manifest> proof;
};

vbr_manifest_validation_result vbr_validate_unit_manifest(
    llama_memory_i & target,
    const vbr_artifact_package_view & package,
    const vbr_adopt_policy & policy) noexcept;

// Pure value-core used after the canonical target-tree snapshot and by the
// CPU matrix. It has the same proof and drift semantics as the live wrapper.
vbr_manifest_validation_result vbr_validate_unit_manifest_snapshot(
    const vbr_target_validation_snapshot & target,
    const vbr_artifact_package_view & package,
    const vbr_adopt_policy & policy) noexcept;

// Canonical construction-empty, text-only attention-prefix validation.  The
// capability is consumed on entry; every failure releases its parent borrow.
// This first slice accepts exact representation only, installs a fresh dense
// destination as live_rebased/whole_import, and never validates or reads the
// unselected parent suffix.
vbr_manifest_validation_result vbr_validate_attention_prefix_projection(
    const vbr_target_validation_snapshot & target,
    vbr_artifact_attention_prefix_projection && projection,
    const vbr_adopt_policy & policy) noexcept;
