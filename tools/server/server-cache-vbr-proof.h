#pragma once

#include "server-cache-lease.h"

class vbr_artifact_package_view;

// Production adapter. Only the scheduler-owned tenant-bound resolver
// may call it; contract scans ban other production call sites.
server_cache_durable_fallback_proof
server_cache_vbr_fallback_proof(
    vbr_artifact_package_view && package) noexcept;

// Private compatibility/test door.
server_cache_durable_fallback_proof
server_cache_vbr_fallback_proof_for_test(
    vbr_artifact_package_view && package) noexcept;
