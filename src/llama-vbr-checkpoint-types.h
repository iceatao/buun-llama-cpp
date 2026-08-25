#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>

// Shared leaf vocabulary for explicit VBR artifact generation records.

enum class vbr_checkpoint_generation_status : uint8_t {
    complete,
    generation_unknown,
};

// Server-layer checkpoint identity/frontier fields bound into the record digest.
// next_position is the EXCLUSIVE computation frontier (dependencies are pos < next_position),
// matching the capture-side filter — never pos_max.
struct vbr_checkpoint_frontier_fields {
    const char * execution_identity          = nullptr;
    size_t       execution_identity_len      = 0;
    const char * adapter_config_identity     = nullptr;
    size_t       adapter_config_identity_len = 0;
    const char * media_content_identity      = nullptr;
    size_t       media_content_identity_len  = 0;
    uint64_t     sequence_epoch = 0;
    int64_t      token_count    = 0;
    llama_pos    next_position  = -1;
};
