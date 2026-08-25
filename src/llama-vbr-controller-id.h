#pragma once

#include <cstddef>
#include <cstdint>

// Durable trajectory identity. The two words are an opaque 128-bit value: consumers may only
// test equality/nonzero or feed the words, in order, to a canonical serializer/hash. The
// allocator currently uses {process origin, monotone counter}, but that split is not a contract.
struct vbr_lineage_uuid {
    uint64_t hi = 0;
    uint64_t lo = 0;
};

constexpr bool operator==(vbr_lineage_uuid lhs, vbr_lineage_uuid rhs) noexcept {
    return lhs.hi == rhs.hi && lhs.lo == rhs.lo;
}

constexpr bool operator!=(vbr_lineage_uuid lhs, vbr_lineage_uuid rhs) noexcept {
    return !(lhs == rhs);
}

constexpr bool vbr_lineage_uuid_is_set(vbr_lineage_uuid uuid) noexcept {
    return uuid.hi != 0 && uuid.lo != 0;
}

// Process-local live-controller routing identity. It is never serialized or hashed into durable
// state. Operation authentication, recovery routing and registry quiescence use this type only.
struct vbr_controller_instance_id {
    uint64_t hi = 0;
    uint64_t lo = 0;
};

constexpr bool operator==(vbr_controller_instance_id lhs,
                          vbr_controller_instance_id rhs) noexcept {
    return lhs.hi == rhs.hi && lhs.lo == rhs.lo;
}

constexpr bool operator!=(vbr_controller_instance_id lhs,
                          vbr_controller_instance_id rhs) noexcept {
    return !(lhs == rhs);
}

constexpr bool vbr_controller_instance_id_is_set(vbr_controller_instance_id id) noexcept {
    return id.hi != 0 && id.lo != 0;
}

using vbr_lineage_origin_provider = bool (*)(uint64_t & origin) noexcept;

// The sole allocator doors. Both latch on exhaustion and never reuse an issued value.
vbr_lineage_uuid            vbr_lineage_uuid_allocate() noexcept;
vbr_controller_instance_id vbr_controller_instance_id_allocate() noexcept;

// One process-global live-controller registry. Duplicate lineages are intentionally irrelevant;
// only a runtime instance may be claimed, and release must cite the exact owner cookie.
bool vbr_controller_instance_check_and_claim(
    vbr_controller_instance_id instance,
    const void * owner) noexcept;
bool vbr_controller_instance_release(
    vbr_controller_instance_id instance,
    const void * owner) noexcept;
// Allocation-free exact-owner probe used by the import barrier. It never
// claims or refreshes an instance and therefore cannot turn stale evidence
// into a live routing identity.
bool vbr_controller_instance_owned_by(
    vbr_controller_instance_id instance,
    const void * owner) noexcept;

// Internal test seams. Reconfiguration is accepted only while the live-controller registry is
// empty, so a test cannot rewrite identity under a live tracker.
bool vbr_lineage_origin_provider_set_for_tests(
    vbr_lineage_origin_provider provider) noexcept;
bool vbr_controller_instance_registry_capacity_set_for_tests(
    size_t capacity) noexcept;
size_t vbr_controller_instance_registry_capacity() noexcept;
