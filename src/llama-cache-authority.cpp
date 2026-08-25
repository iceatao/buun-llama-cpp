#include "llama-cache-authority.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

const char * llama_cache_admission_status_name(llama_cache_admission_status status) noexcept {
    switch (status) {
        case llama_cache_admission_status::admitted:            return "admitted";
        case llama_cache_admission_status::incomplete_evidence: return "incomplete_evidence";
        case llama_cache_admission_status::budget_unavailable:  return "budget_unavailable";
        case llama_cache_admission_status::exceeds_budget:      return "exceeds_budget";
        case llama_cache_admission_status::serial_conflict:     return "serial_conflict";
        case llama_cache_admission_status::ledger_fault:        return "ledger_fault";
        case llama_cache_admission_status::internal_fault:      return "internal_fault";
        case llama_cache_admission_status::_count:              break;
    }
    return "unknown";
}

const char * llama_cache_transaction_status_name(
        llama_cache_transaction_status status) noexcept {
    switch (status) {
        case llama_cache_transaction_status::committed:          return "committed";
        case llama_cache_transaction_status::invalid_argument:   return "invalid_argument";
        case llama_cache_transaction_status::admission_refused:  return "admission_refused";
        case llama_cache_transaction_status::after_admit_failed: return "after_admit_failed";
        case llama_cache_transaction_status::stage_failed:       return "stage_failed";
        case llama_cache_transaction_status::commit_failed:      return "commit_failed";
        case llama_cache_transaction_status::post_commit_fault:  return "post_commit_fault";
        case llama_cache_transaction_status::internal_fault:     return "internal_fault";
        case llama_cache_transaction_status::_count:             break;
    }
    return "unknown";
}

const char * llama_cache_prepare_status_name(
        llama_cache_prepare_status status) noexcept {
    switch (status) {
        case llama_cache_prepare_status::prepared:          return "prepared";
        case llama_cache_prepare_status::invalid_argument:  return "invalid_argument";
        case llama_cache_prepare_status::admission_refused: return "admission_refused";
        case llama_cache_prepare_status::internal_fault:    return "internal_fault";
        case llama_cache_prepare_status::_count:            break;
    }
    return "unknown";
}

const char * llama_cache_prepare_release_status_name(
        llama_cache_prepare_release_status status) noexcept {
    switch (status) {
        case llama_cache_prepare_release_status::prepared:         return "prepared";
        case llama_cache_prepare_release_status::invalid_argument: return "invalid_argument";
        case llama_cache_prepare_release_status::serial_conflict:  return "serial_conflict";
        case llama_cache_prepare_release_status::ledger_fault:     return "ledger_fault";
        case llama_cache_prepare_release_status::internal_fault:   return "internal_fault";
        case llama_cache_prepare_release_status::_count:           break;
    }
    return "unknown";
}

llama_cache_prepared_release_set::llama_cache_prepared_release_set(
        llama_cache_prepared_release_set && other) noexcept
    : ledger_(other.ledger_),
      ops_(std::move(other.ops_)),
      preview_(std::move(other.preview_)),
      status_(other.status_) {
    other.ledger_ = nullptr;
}

llama_cache_prepared_release_set &
llama_cache_prepared_release_set::operator=(
        llama_cache_prepared_release_set && other) noexcept {
    if (this != &other) {
        ledger_ = other.ledger_;
        ops_ = std::move(other.ops_);
        preview_ = std::move(other.preview_);
        status_ = other.status_;
        other.ledger_ = nullptr;
    }
    return *this;
}

llama_cache_conditional_release_status
llama_cache_prepared_release_set::commit() noexcept {
    if (!ready()) {
        return llama_cache_conditional_release_status::ledger_fault;
    }
    auto * ledger = ledger_;
    ledger_ = nullptr;
    return ledger->release_set_if_serial(ops_, preview_.accounting_serial);
}

llama_cache_prepared_release_set llama_cache_prepare_release_set(
        llama_cache_acct_ledger & ledger,
        const std::vector<llama_cache_acct_op_id> & selected,
        uint64_t expected_serial) noexcept {
    return llama_cache_prepare_release_set(
        ledger, selected, expected_serial, false);
}

llama_cache_prepared_release_set llama_cache_prepare_release_set(
        llama_cache_acct_ledger & ledger,
        const std::vector<llama_cache_acct_op_id> & selected,
        uint64_t expected_serial,
        bool include_category_yields) noexcept {
    llama_cache_prepared_release_set out;
    try {
        if (expected_serial == 0) {
            return out;
        }
        out.ops_ = selected;
        std::sort(out.ops_.begin(), out.ops_.end());
        if ((!out.ops_.empty() && !out.ops_.front()) ||
            std::adjacent_find(out.ops_.begin(), out.ops_.end()) !=
                out.ops_.end()) {
            return out;
        }
        if (!ledger.preview_release_set(
                out.ops_, expected_serial, out.preview_,
                include_category_yields)) {
            // A fresh snapshot distinguishes benign serial drift from a hard
            // invalid operation set without turning drift into a ledger fault.
            out.status_ = ledger.snapshot().serial != expected_serial
                ? llama_cache_prepare_release_status::serial_conflict
                : llama_cache_prepare_release_status::ledger_fault;
            return out;
        }
        out.ledger_ = &ledger;
        out.status_ = llama_cache_prepare_release_status::prepared;
        return out;
    } catch (...) {
        out = {};
        out.status_ = llama_cache_prepare_release_status::internal_fault;
        return out;
    }
}

llama_cache_reservation_claim::llama_cache_reservation_claim(
        llama_cache_acct_ledger * ledger, llama_cache_acct_op_id op) noexcept
    : ledger_(ledger), op_(op) {}

void llama_cache_reservation_claim::abort_if_live() noexcept {
    if (has_op()) {
        ledger_->abort(op_);
    }
    release();
}

llama_cache_reservation_claim::~llama_cache_reservation_claim() {
    abort_if_live();
}

llama_cache_reservation_claim::llama_cache_reservation_claim(
        llama_cache_reservation_claim && other) noexcept
    : ledger_(other.ledger_), op_(other.op_) {
    other.release();
}

llama_cache_reservation_claim & llama_cache_reservation_claim::operator=(
        llama_cache_reservation_claim && other) noexcept {
    if (this != &other) {
        abort_if_live();
        ledger_ = other.ledger_;
        op_     = other.op_;
        other.release();
    }
    return *this;
}

bool llama_cache_reservation_claim::commit(
        uint64_t logical_bytes,
        llama_cache_acct_op_id & committed_op) noexcept {
    committed_op = {};
    if (!has_op() || !ledger_->commit(op_, logical_bytes)) {
        return false;
    }
    committed_op = op_;
    release();
    return true;
}

llama_cache_admission_result llama_cache_admit_reservation(
        llama_cache_acct_ledger          & ledger,
        const llama_cache_budget_config  & budget_config,
        const llama_cache_authority_request & request) noexcept try {
    // 1. Coherent snapshot under one serial.
    llama_cache_acct_snapshot snap = ledger.snapshot();

    // 2. Fail-closed on incomplete evidence: an explicitly non-known manifest means the ledger
    //    cannot vouch for completeness, so refuse before pricing (never admit on private counters).
    if (snap.completeness_manifest != llama_cache_acct_known::known) {
        return { llama_cache_admission_status::incomplete_evidence, {} };
    }

    // 3. Local coordinator (reset() mutates it; a shared instance is unsafe across admissions).
    //    Move the potentially-large accounting snapshot into the one-shot coordinator now that
    //    this composer is on the publication authority path.
    llama_cache_budget_coordinator coordinator;
    const uint64_t accounting_serial = snap.serial;
    if (!coordinator.reset(std::move(snap), budget_config)) {
        return { llama_cache_admission_status::budget_unavailable, {} };
    }

    // 4. Reserve-only plan priced at the snapshot's serial: one domain, no release credits.
    llama_cache_budget_plan plan;
    plan.accounting_serial = accounting_serial;
    plan.entries.push_back({ request.domain, request.expected_resident, /* release_bytes */ 0 });

    // 5. Price it.
    const llama_cache_budget_result fit = coordinator.fits(plan);
    switch (fit.state) {
        case llama_cache_budget_fit_state::fits:
            break;
        case llama_cache_budget_fit_state::exceeds:
            return { llama_cache_admission_status::exceeds_budget, {} };
        case llama_cache_budget_fit_state::unavailable:
        case llama_cache_budget_fit_state::_count:
        default:
            return { llama_cache_admission_status::budget_unavailable, {} };
    }

    // 6. Conditional reserve at the priced serial (single-shot: drift refuses and the caller re-drives).
    const llama_cache_conditional_reserve_result cr = ledger.reserve_if_serial(
        accounting_serial, request.category, request.domain, request.attribution,
        request.expected_logical, request.expected_resident);
    switch (cr.status) {
        case llama_cache_conditional_reserve_status::admitted:
            return { llama_cache_admission_status::admitted,
                     llama_cache_reservation_claim(&ledger, cr.op) };
        case llama_cache_conditional_reserve_status::serial_conflict:
            return { llama_cache_admission_status::serial_conflict, {} };
        case llama_cache_conditional_reserve_status::ledger_fault:
        case llama_cache_conditional_reserve_status::_count:
        default:
            return { llama_cache_admission_status::ledger_fault, {} };
    }
} catch (...) {
    // The only throwing step is plan.entries.push_back (the ledger/coordinator calls are noexcept);
    // a function-try-block turns any allocation failure into a typed fail-closed verdict, so no
    // exception ever crosses the admission authority boundary.
    return { llama_cache_admission_status::internal_fault, {} };
}

struct llama_cache_prepared_claim_group::impl {
    llama_cache_acct_ledger * ledger = nullptr;
    std::vector<llama_cache_transaction_leaf> leaves;
    std::vector<llama_cache_admission_result> admissions;
    std::vector<llama_cache_acct_op_id> reserved_ops;
    llama_cache_prepare_result preparation;
    bool consumed = false;
};

llama_cache_prepared_claim_group::llama_cache_prepared_claim_group() = default;
llama_cache_prepared_claim_group::~llama_cache_prepared_claim_group() {
    abort_if_live();
}

llama_cache_prepared_claim_group::llama_cache_prepared_claim_group(
        std::unique_ptr<impl> state) noexcept
    : impl_(std::move(state)) {}

llama_cache_prepared_claim_group::llama_cache_prepared_claim_group(
        llama_cache_prepared_claim_group &&) noexcept = default;

llama_cache_prepared_claim_group &
llama_cache_prepared_claim_group::operator=(
        llama_cache_prepared_claim_group && other) noexcept {
    if (this != &other) {
        abort_if_live();
        impl_ = std::move(other.impl_);
    }
    return *this;
}

void llama_cache_prepared_claim_group::abort_if_live() noexcept {
    if (!impl_ || impl_->consumed || impl_->reserved_ops.empty() ||
        !impl_->ledger) {
        return;
    }
    if (!impl_->ledger->abort_set(
            impl_->reserved_ops.data(), impl_->reserved_ops.size())) {
        return;
    }
    for (auto & admission : impl_->admissions) {
        admission.claim.release();
    }
    impl_->reserved_ops.clear();
}

bool llama_cache_prepared_claim_group::ready() const noexcept {
    return impl_ &&
           !impl_->consumed &&
           impl_->preparation.status ==
               llama_cache_prepare_status::prepared;
}

const llama_cache_prepare_result &
llama_cache_prepared_claim_group::preparation() const noexcept {
    static const llama_cache_prepare_result invalid;
    return impl_ ? impl_->preparation : invalid;
}

bool llama_cache_prepared_claim_group::shrink_equal_reservations(
        const std::vector<uint64_t> & resident_bytes) noexcept {
    if (!ready() || resident_bytes.size() != impl_->leaves.size() ||
        impl_->reserved_ops.size() != impl_->leaves.size()) {
        return false;
    }
    for (size_t i = 0; i < resident_bytes.size(); ++i) {
        const auto & leaf = impl_->leaves[i];
        if (leaf.expected_logical != leaf.reserve_resident ||
            leaf.reserve_resident != leaf.stage_resident ||
            resident_bytes[i] > leaf.reserve_resident) {
            return false;
        }
    }
    if (!impl_->ledger->shrink_reservation_set(
            impl_->reserved_ops.data(), resident_bytes.data(),
            resident_bytes.size())) {
        return false;
    }
    for (size_t i = 0; i < resident_bytes.size(); ++i) {
        impl_->leaves[i].expected_logical = resident_bytes[i];
        impl_->leaves[i].reserve_resident = resident_bytes[i];
        impl_->leaves[i].stage_resident = resident_bytes[i];
    }
    return true;
}

bool llama_cache_prepared_claim_group::repartition_downward(
        const std::vector<llama_cache_transaction_leaf> & leaves) noexcept {
    if (!ready() || leaves.empty()) {
        return false;
    }
    try {
        // Complete every allocation before the ledger's irreversible atomic
        // repartition. After that terminal only claim disarms and vector moves
        // remain, so an exception cannot leave a falsely-ready group whose
        // recorded operations were already replaced.
        std::vector<llama_cache_transaction_leaf> replacement_leaves = leaves;
        std::vector<llama_cache_conditional_reserve_request> requests(
            leaves.size());
        std::vector<llama_cache_acct_op_id> operations(leaves.size());
        std::vector<llama_cache_admission_result> admissions(leaves.size());
        std::unordered_set<const void *> operation_outputs;
        std::unordered_set<const void *> allocation_outputs;
        operation_outputs.reserve(leaves.size());
        allocation_outputs.reserve(leaves.size());
        for (size_t i = 0; i < leaves.size(); ++i) {
            const auto & leaf = leaves[i];
            if (!leaf.committed_op ||
                (leaf.existing_allocation && leaf.reserve_resident != 0) ||
                (!leaf.existing_allocation &&
                 leaf.stage_resident != leaf.reserve_resident) ||
                !operation_outputs.insert(leaf.committed_op).second ||
                (leaf.allocation_out &&
                 !allocation_outputs.insert(leaf.allocation_out).second)) {
                return false;
            }
            requests[i] = {
                leaf.category, leaf.domain, leaf.attribution,
                leaf.expected_logical, leaf.reserve_resident,
            };
        }
        if (!impl_->ledger->repartition_reservation_set_downward(
                impl_->reserved_ops.data(), impl_->reserved_ops.size(),
                requests.data(), requests.size(), operations.data())) {
            return false;
        }
        for (size_t i = 0; i < operations.size(); ++i) {
            admissions[i].status = llama_cache_admission_status::admitted;
            admissions[i].claim = llama_cache_reservation_claim(
                impl_->ledger, operations[i]);
        }
        for (auto & admission : impl_->admissions) {
            admission.claim.release();
        }
        impl_->leaves = std::move(replacement_leaves);
        impl_->reserved_ops = std::move(operations);
        impl_->admissions = std::move(admissions);
        return true;
    } catch (...) {
        return false;
    }
}

bool llama_cache_prepared_claim_group::partition(
        const std::vector<size_t> & counts,
        std::vector<llama_cache_prepared_claim_group> & output) noexcept {
    if (!ready() || counts.empty() || !output.empty() ||
        impl_->admissions.size() != impl_->leaves.size() ||
        impl_->reserved_ops.size() != impl_->leaves.size()) {
        return false;
    }
    try {
        size_t total = 0;
        std::vector<std::unique_ptr<impl>> states;
        std::vector<llama_cache_prepared_claim_group> groups;
        states.reserve(counts.size());
        groups.reserve(counts.size());
        for (const auto count : counts) {
            if (count == 0 || count > impl_->leaves.size() - total) {
                return false;
            }
            total += count;
            std::unique_ptr<impl> child(new impl);
            child->ledger = impl_->ledger;
            child->preparation = impl_->preparation;
            child->leaves.reserve(count);
            child->admissions.reserve(count);
            child->reserved_ops.reserve(count);
            states.push_back(std::move(child));
        }
        if (total != impl_->leaves.size()) {
            return false;
        }

        size_t source = 0;
        for (size_t group = 0; group < counts.size(); ++group) {
            auto & child = *states[group];
            for (size_t i = 0; i < counts[group]; ++i, ++source) {
                child.leaves.push_back(impl_->leaves[source]);
                child.admissions.push_back(
                    std::move(impl_->admissions[source]));
                child.reserved_ops.push_back(impl_->reserved_ops[source]);
            }
            groups.push_back(llama_cache_prepared_claim_group(
                std::move(states[group])));
        }
        impl_->consumed = true;
        impl_->leaves.clear();
        impl_->admissions.clear();
        impl_->reserved_ops.clear();
        output = std::move(groups);
        return true;
    } catch (...) {
        return false;
    }
}

llama_cache_prepared_claim_group
llama_cache_prepare_reservation_transaction(
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget_config,
        const std::vector<llama_cache_transaction_leaf> & leaves) noexcept {
    std::unique_ptr<llama_cache_prepared_claim_group::impl> state;
    try {
        state.reset(new llama_cache_prepared_claim_group::impl);
        state->ledger = &ledger;
        state->leaves = leaves;
        if (leaves.empty()) {
            state->preparation.status =
                llama_cache_prepare_status::invalid_argument;
            return llama_cache_prepared_claim_group(
                std::move(state));
        }
        std::unordered_set<const void *> operation_outputs;
        std::unordered_set<const void *> allocation_outputs;
        operation_outputs.reserve(leaves.size());
        allocation_outputs.reserve(leaves.size());
        for (size_t i = 0; i < leaves.size(); ++i) {
            const auto & leaf = leaves[i];
            if (!leaf.committed_op ||
                (leaf.existing_allocation &&
                 leaf.reserve_resident != 0) ||
                (!leaf.existing_allocation &&
                 leaf.stage_resident !=
                     leaf.reserve_resident) ||
                !operation_outputs.insert(leaf.committed_op).second ||
                (leaf.allocation_out &&
                 !allocation_outputs.insert(leaf.allocation_out).second)) {
                state->preparation.status =
                    llama_cache_prepare_status::invalid_argument;
                state->preparation.failed_leaf = i;
                return llama_cache_prepared_claim_group(
                    std::move(state));
            }
        }

        state->admissions.resize(leaves.size());
        std::vector<llama_cache_conditional_reserve_request> requests(
            leaves.size());
        std::vector<llama_cache_acct_op_id> operations(leaves.size());
        llama_cache_budget_plan plan;
        plan.entries.reserve(leaves.size());
        for (size_t i = 0; i < leaves.size(); ++i) {
            const auto & leaf = leaves[i];
            requests[i] = {
                leaf.category, leaf.domain, leaf.attribution,
                leaf.expected_logical, leaf.reserve_resident,
            };
            const auto position = std::lower_bound(
                plan.entries.begin(), plan.entries.end(), leaf.domain,
                [&](const auto & entry, const auto & domain) {
                    return llama_cache_acct_resource_domain_less(
                        entry.domain, domain);
                });
            if (position != plan.entries.end() &&
                position->domain == leaf.domain) {
                if (leaf.reserve_resident >
                        UINT64_MAX - position->reserve_bytes) {
                    state->preparation.status =
                        llama_cache_prepare_status::invalid_argument;
                    state->preparation.failed_leaf = i;
                    return llama_cache_prepared_claim_group(
                        std::move(state));
                }
                position->reserve_bytes += leaf.reserve_resident;
            } else {
                plan.entries.insert(position, {
                    leaf.domain, leaf.reserve_resident, 0,
                });
            }
        }
        static constexpr uint32_t MAX_ATTEMPTS = 3;
        for (uint32_t attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
            auto snapshot = ledger.snapshot();
            state->preparation.attempts = attempt + 1;
            if (snapshot.completeness_manifest !=
                    llama_cache_acct_known::known) {
                state->preparation.status =
                    llama_cache_prepare_status::admission_refused;
                state->preparation.admission_status =
                    llama_cache_admission_status::incomplete_evidence;
                return llama_cache_prepared_claim_group(std::move(state));
            }
            const uint64_t serial = snapshot.serial;
            llama_cache_budget_coordinator coordinator;
            if (!coordinator.reset(std::move(snapshot), budget_config)) {
                state->preparation.status =
                    llama_cache_prepare_status::admission_refused;
                state->preparation.admission_status =
                    llama_cache_admission_status::budget_unavailable;
                return llama_cache_prepared_claim_group(std::move(state));
            }
            plan.accounting_serial = serial;
            const auto fit = coordinator.fits(plan);
            if (fit.state != llama_cache_budget_fit_state::fits) {
                state->preparation.status =
                    llama_cache_prepare_status::admission_refused;
                state->preparation.admission_status =
                    fit.state == llama_cache_budget_fit_state::exceeds
                        ? llama_cache_admission_status::exceeds_budget
                        : llama_cache_admission_status::budget_unavailable;
                return llama_cache_prepared_claim_group(std::move(state));
            }
            const auto reserved = ledger.reserve_set_if_serial(
                serial, requests.data(), requests.size(), operations.data());
            if (reserved.status ==
                    llama_cache_conditional_reserve_status::serial_conflict) {
                state->preparation.serial_retries++;
                continue;
            }
            if (reserved.status !=
                    llama_cache_conditional_reserve_status::admitted) {
                state->preparation.status =
                    llama_cache_prepare_status::admission_refused;
                state->preparation.admission_status =
                    llama_cache_admission_status::ledger_fault;
                state->preparation.failed_leaf = reserved.failed_request;
                return llama_cache_prepared_claim_group(std::move(state));
            }
            for (size_t i = 0; i < operations.size(); ++i) {
                state->admissions[i].status =
                    llama_cache_admission_status::admitted;
                state->admissions[i].claim =
                    llama_cache_reservation_claim(&ledger, operations[i]);
            }
            state->reserved_ops = operations;
            state->preparation.status =
                llama_cache_prepare_status::prepared;
            state->preparation.admission_status =
                llama_cache_admission_status::admitted;
            return llama_cache_prepared_claim_group(std::move(state));
        }
        state->preparation.status =
            llama_cache_prepare_status::admission_refused;
        state->preparation.admission_status =
            llama_cache_admission_status::serial_conflict;
        return llama_cache_prepared_claim_group(std::move(state));
    } catch (...) {
        if (!state) {
            try {
                state.reset(
                    new llama_cache_prepared_claim_group::impl);
            } catch (...) {
                return {};
            }
        }
        state->preparation.status =
            llama_cache_prepare_status::internal_fault;
        return llama_cache_prepared_claim_group(
            std::move(state));
    }
}

llama_cache_transaction_result
llama_cache_prepared_claim_group::materialize_and_commit(
        const llama_cache_transaction_fault & fault,
        const llama_cache_transaction_after_admit &
            after_admit) noexcept {
    if (!impl_) {
        return {};
    }
    return materialize_and_commit(
        impl_->leaves, fault, after_admit);
}

llama_cache_transaction_result
llama_cache_prepared_claim_group::materialize_and_commit(
        const std::vector<llama_cache_transaction_leaf> &
            finalized_leaves,
        const llama_cache_transaction_fault & fault,
        const llama_cache_transaction_after_admit &
            after_admit) noexcept {
    llama_cache_transaction_result out;
    try {
        if ((after_admit.context != nullptr) !=
                (after_admit.run != nullptr)) {
            out.status =
                llama_cache_transaction_status::invalid_argument;
            return out;
        }
        if (!ready()) {
            if (impl_ &&
                impl_->preparation.status ==
                    llama_cache_prepare_status::
                        invalid_argument) {
                out.status =
                    llama_cache_transaction_status::
                        invalid_argument;
            } else if (impl_ &&
                       impl_->preparation.status ==
                           llama_cache_prepare_status::
                               admission_refused) {
                out.status =
                    llama_cache_transaction_status::
                        admission_refused;
            } else {
                out.status =
                    llama_cache_transaction_status::
                        internal_fault;
            }
            if (impl_) {
                out.admission_status =
                    impl_->preparation.admission_status;
                out.failed_leaf =
                    impl_->preparation.failed_leaf;
                out.attempts =
                    impl_->preparation.attempts;
                out.serial_retries =
                    impl_->preparation.serial_retries;
            }
            return out;
        }
        if (finalized_leaves.size() !=
                impl_->leaves.size()) {
            out.status =
                llama_cache_transaction_status::
                    invalid_argument;
            return out;
        }
        std::unordered_set<const void *> operation_outputs;
        std::unordered_set<const void *> allocation_outputs;
        operation_outputs.reserve(finalized_leaves.size());
        allocation_outputs.reserve(finalized_leaves.size());
        for (size_t i = 0; i < finalized_leaves.size(); ++i) {
            const auto & prepared = impl_->leaves[i];
            const auto & finalized = finalized_leaves[i];
            if (prepared.category != finalized.category ||
                prepared.domain != finalized.domain ||
                prepared.attribution.kind !=
                    finalized.attribution.kind ||
                prepared.attribution.slot_id !=
                    finalized.attribution.slot_id ||
                prepared.attribution.artifact !=
                    finalized.attribution.artifact ||
                prepared.expected_logical !=
                    finalized.expected_logical ||
                prepared.reserve_resident !=
                    finalized.reserve_resident ||
                prepared.stage_resident !=
                    finalized.stage_resident ||
                prepared.existing_allocation !=
                    finalized.existing_allocation ||
                !finalized.committed_op ||
                !operation_outputs.insert(
                    finalized.committed_op).second ||
                (finalized.allocation_out &&
                 !allocation_outputs.insert(
                    finalized.allocation_out).second)) {
                out.status =
                    llama_cache_transaction_status::
                        invalid_argument;
                out.failed_leaf = i;
                return out;
            }
        }
        impl_->consumed = true;
        out.serial_retries =
            impl_->preparation.serial_retries;

        if (after_admit.run &&
            !after_admit.run(after_admit.context)) {
            out.status =
                llama_cache_transaction_status::
                    after_admit_failed;
            return out;
        }

        auto & ledger = *impl_->ledger;
        const auto & leaves = finalized_leaves;
        auto & admissions = impl_->admissions;
        std::vector<llama_cache_acct_alloc_id> allocations(
            leaves.size());
        for (size_t i = 0; i < leaves.size(); ++i) {
            if (fault.fail_stage_at == i) {
                out.status =
                    llama_cache_transaction_status::stage_failed;
                out.failed_leaf = i;
                return out;
            }
            allocations[i] = leaves[i].existing_allocation
                ? leaves[i].existing_allocation
                : ledger.new_alloc();
            // The ledger has no join-without-stage primitive.
            // Re-staging an existing immutable allocation therefore
            // raises transient_peak briefly even though no payload
            // bytes move and durable charge remains governed by
            // first/last reference.
            if (!allocations[i] ||
                !ledger.stage(
                    admissions[i].claim.op(),
                    allocations[i],
                    leaves[i].stage_resident,
                    leaves[i].artifact,
                    leaves[i].content,
                    leaves[i].lineage)) {
                out.status =
                    llama_cache_transaction_status::stage_failed;
                out.failed_leaf = i;
                return out;
            }
        }

        std::vector<llama_cache_acct_op_id> committed;
        committed.reserve(leaves.size());
        struct rollback_guard {
            llama_cache_acct_ledger * ledger = nullptr;
            std::vector<llama_cache_acct_op_id> *
                operations = nullptr;
            uint64_t * rolled_back = nullptr;
            bool keep = false;

            void rollback() noexcept {
                if (keep || !ledger || !operations ||
                    !rolled_back) {
                    return;
                }
                for (const auto op : *operations) {
                    if (ledger->release(op)) {
                        (*rolled_back)++;
                    }
                }
                keep = true;
            }

            ~rollback_guard() {
                rollback();
            }
        } rollback {
            &ledger, &committed, &out.rolled_back, false,
        };

        for (size_t i = 0; i < leaves.size(); ++i) {
            if (fault.fail_commit_at == i) {
                out.status =
                    llama_cache_transaction_status::commit_failed;
                out.failed_leaf = i;
                rollback.rollback();
                return out;
            }
            llama_cache_acct_op_id operation;
            if (!admissions[i].claim.commit(
                    leaves[i].expected_logical, operation)) {
                out.status =
                    llama_cache_transaction_status::commit_failed;
                out.failed_leaf = i;
                rollback.rollback();
                return out;
            }
            committed.push_back(operation);
        }

        if (fault.fail_after_commit) {
            out.status =
                llama_cache_transaction_status::
                    post_commit_fault;
            rollback.rollback();
            return out;
        }

        for (size_t i = 0; i < leaves.size(); ++i) {
            *leaves[i].committed_op = committed[i];
            if (leaves[i].allocation_out) {
                *leaves[i].allocation_out = allocations[i];
            }
        }
        rollback.keep = true;
        out.admission_status =
            llama_cache_admission_status::admitted;
        out.status =
            llama_cache_transaction_status::committed;
        return out;
    } catch (...) {
        out.status =
            llama_cache_transaction_status::internal_fault;
        return out;
    }
}

llama_cache_transaction_result
llama_cache_execute_reservation_transaction(
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget_config,
        const std::vector<llama_cache_transaction_leaf> & leaves,
        const llama_cache_transaction_fault & fault,
        const llama_cache_transaction_after_admit &
            after_admit) noexcept {
    if ((after_admit.context != nullptr) !=
            (after_admit.run != nullptr)) {
        llama_cache_transaction_result out;
        out.status =
            llama_cache_transaction_status::invalid_argument;
        return out;
    }
    auto prepared =
        llama_cache_prepare_reservation_transaction(
            ledger, budget_config, leaves);
    return prepared.materialize_and_commit(
        fault, after_admit);
}
