#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif
#include "common.cuh"
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>

static bool dflash_cuda_try(cudaError_t result, const char * operation) {
    if (result == cudaSuccess) {
        return true;
    }
    // These allocation/lifetime paths are intentionally fallible rather than
    // CUDA_CHECK-fatal.  Still report the originating operation directly so a
    // later checked call cannot inherit an anonymous sticky error.
    int device = -1;
    (void) cudaGetDevice(&device);
    fprintf(stderr, "dflash CUDA error in %s on device %d: %s (%d)\n",
            operation, device, cudaGetErrorString(result), (int) result);
    fflush(stderr);
    return false;
}

// GPU cross-attention ring buffer for DFlash speculative decoding.
// Keeps per-layer ring buffers on GPU and interleaves them into the layout
// expected by the drafter's target_hidden tensor, avoiding the CPU round-trip.

struct dflash_cross_ring_gpu {
    int device;               // CUDA device where ring buffers are allocated
    int n_layers;
    int n_embd;
    int ring_size;

    float ** d_layer_rings;   // device: array of n_layers device pointers
    float *  d_staging;       // device: interleaved output [ring_size * n_layers * n_embd]
    float ** h_layer_ptrs;    // host: copy of per-layer device pointers
};

// Interleave kernel: reads per-layer circular ring, writes interleaved output.
// Grid: (cross_len, n_layers), Block: 256
// Each thread block copies one (token, layer) slice of n_embd floats.
__global__ static void k_cross_ring_interleave(
        const float * const * __restrict__ d_rings,
        float * __restrict__ d_out,
        const int ring_size,
        const int read_start,
        const int cross_len,
        const int n_layers,
        const int n_embd) {
    const int t = blockIdx.x; // token index [0, cross_len)
    const int l = blockIdx.y; // layer index [0, n_layers)

    if (t >= cross_len || l >= n_layers) return;

    const int slot = (read_start + t) % ring_size;
    const float * src = d_rings[l] + (size_t)slot * n_embd;
    float * dst = d_out + (size_t)t * n_layers * n_embd + (size_t)l * n_embd;

    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        dst[i] = src[i];
    }
}

extern "C" void * dflash_cross_ring_gpu_alloc(int n_layers, int n_embd, int ring_size) {
    // env var override
    const char * env = getenv("GGML_DFLASH_GPU_RING");
    if (env && atoi(env) == 0) {
        return nullptr;
    }

    auto * ring = new dflash_cross_ring_gpu();
    if (!dflash_cuda_try(cudaGetDevice(&ring->device), "cross-ring cudaGetDevice")) {
        delete ring;
        return nullptr;
    }
    ring->n_layers  = n_layers;
    ring->n_embd    = n_embd;
    ring->ring_size = ring_size;
    // per-layer ring buffers on device
    ring->h_layer_ptrs = new float*[n_layers];
    for (int l = 0; l < n_layers; l++) {
        cudaError_t err = cudaMalloc(&ring->h_layer_ptrs[l], (size_t)ring_size * n_embd * sizeof(float));
        if (!dflash_cuda_try(err, "cross-ring layer cudaMalloc")) {
            for (int j = 0; j < l; j++) {
                (void) dflash_cuda_try(cudaFree(ring->h_layer_ptrs[j]), "cross-ring rollback cudaFree");
            }
            delete[] ring->h_layer_ptrs;
            delete ring;
            return nullptr;
        }
        if (!dflash_cuda_try(
                cudaMemset(ring->h_layer_ptrs[l], 0,
                    (size_t)ring_size * n_embd * sizeof(float)),
                "cross-ring layer cudaMemset")) {
            for (int j = 0; j <= l; j++) {
                (void) dflash_cuda_try(cudaFree(ring->h_layer_ptrs[j]), "cross-ring rollback cudaFree");
            }
            delete[] ring->h_layer_ptrs;
            delete ring;
            return nullptr;
        }
    }

    // device array of layer pointers
    cudaError_t err = cudaMalloc(&ring->d_layer_rings, n_layers * sizeof(float *));
    if (!dflash_cuda_try(err, "cross-ring pointer-table cudaMalloc")) {
        for (int l = 0; l < n_layers; l++) {
            (void) dflash_cuda_try(cudaFree(ring->h_layer_ptrs[l]), "cross-ring rollback cudaFree");
        }
        delete[] ring->h_layer_ptrs;
        delete ring;
        return nullptr;
    }
    if (!dflash_cuda_try(
            cudaMemcpy(ring->d_layer_rings, ring->h_layer_ptrs,
                n_layers * sizeof(float *), cudaMemcpyHostToDevice),
            "cross-ring pointer-table cudaMemcpy")) {
        (void) dflash_cuda_try(cudaFree(ring->d_layer_rings), "cross-ring rollback cudaFree");
        for (int l = 0; l < n_layers; l++) {
            (void) dflash_cuda_try(cudaFree(ring->h_layer_ptrs[l]), "cross-ring rollback cudaFree");
        }
        delete[] ring->h_layer_ptrs;
        delete ring;
        return nullptr;
    }

    // staging buffer for interleaved output
    err = cudaMalloc(&ring->d_staging, (size_t)ring_size * n_layers * n_embd * sizeof(float));
    if (!dflash_cuda_try(err, "cross-ring staging cudaMalloc")) {
        (void) dflash_cuda_try(cudaFree(ring->d_layer_rings), "cross-ring rollback cudaFree");
        for (int l = 0; l < n_layers; l++) {
            (void) dflash_cuda_try(cudaFree(ring->h_layer_ptrs[l]), "cross-ring rollback cudaFree");
        }
        delete[] ring->h_layer_ptrs;
        delete ring;
        return nullptr;
    }

    size_t total_mb = ((size_t)ring_size * n_embd * sizeof(float) * n_layers +
                       (size_t)ring_size * n_layers * n_embd * sizeof(float)) / (1024 * 1024);
    fprintf(stderr, "dflash gpu ring: allocated %d layers x %d slots x %d embd + staging (~%zu MB)\n",
            n_layers, ring_size, n_embd, total_mb);

    return ring;
}

extern "C" void dflash_cross_ring_gpu_free(void * handle) {
    if (!handle) return;
    auto * ring = (dflash_cross_ring_gpu *)handle;

    (void) dflash_cuda_try(cudaFree(ring->d_staging), "cross-ring staging cudaFree");
    (void) dflash_cuda_try(cudaFree(ring->d_layer_rings), "cross-ring pointer-table cudaFree");
    for (int l = 0; l < ring->n_layers; l++) {
        (void) dflash_cuda_try(cudaFree(ring->h_layer_ptrs[l]), "cross-ring layer cudaFree");
    }
    delete[] ring->h_layer_ptrs;
    delete ring;
}

// Split a [ring_pos, ring_pos + n_tokens) span into at most two contiguous segments
// (wrap-around) and invoke copy(ring_tok, other_tok, seg_tokens) for each — the one
// place that owns the ring wrap math for write/write_d2d/read below.
template <typename F>
static void ring_span_for_each(int ring_size, int ring_pos, int n_tokens, F && copy) {
    const int pos   = ((ring_pos % ring_size) + ring_size) % ring_size;
    const int first = ring_size - pos;
    if (first >= n_tokens) {
        copy(pos, 0, n_tokens);
    } else {
        copy(pos, 0, first);
        copy(0, first, n_tokens - first);
    }
}

// Upload host data to a specific position in the GPU ring for one layer.
// Handles wrap-around: if ring_pos + n_tokens > ring_size, splits into two copies.
extern "C" void dflash_cross_ring_gpu_write(
        void * handle, int layer, int ring_pos,
        const float * host_data, int n_tokens, int n_embd) {
    if (!handle) return;
    auto * ring = (dflash_cross_ring_gpu *)handle;

    if (layer < 0 || layer >= ring->n_layers) return;
    if (n_tokens <= 0) return;

    // The GPU ring holds only ring_size tokens (the cross-attention window). A prefill
    // decode chunk can hand us far more (e.g. a >512-token -b batch), and only the last
    // ring_size of them can survive in the ring. Writing the whole run overflows the
    // destination allocation — cudaMemcpyAsync then fails with "invalid argument", and
    // because that error is latched it surfaces later at an unrelated CUDA_CHECK. Keep
    // only the most recent ring_size tokens, advancing ring_pos past the dropped ones.
    if (n_tokens > ring->ring_size) {
        const int skip = n_tokens - ring->ring_size;
        host_data += (size_t)skip * n_embd;
        ring_pos  += skip;
        n_tokens   = ring->ring_size;
    }

    // Ensure cudaStreamPerThread belongs to the ring's device regardless of
    // which GPU the caller (target model decode) last set as current.
    CUDA_CHECK(cudaSetDevice(ring->device));

    float * dst = ring->h_layer_ptrs[layer];
    const size_t stride = (size_t)n_embd * sizeof(float);

    ring_span_for_each(ring->ring_size, ring_pos, n_tokens, [&](int ring_tok, int src_tok, int n) {
        CUDA_CHECK(cudaMemcpyAsync(
                dst + (size_t)ring_tok * n_embd,
                host_data + (size_t)src_tok * n_embd,
                (size_t)n * stride, cudaMemcpyHostToDevice, cudaStreamPerThread));
    });
}

extern "C" void dflash_cross_ring_gpu_set_tensor(void * d_dst, const void * d_src, size_t offset, size_t size);

// Device-to-device variant of the ring write: source is a device pointer (e.g. the
// target's graph-embedded capture staging), possibly on another GPU. Same clamp and
// wrap-around handling as the host write; peer copies resolved from pointer attributes.
extern "C" void dflash_cross_ring_gpu_write_d2d(
        void * handle, int layer, int ring_pos,
        const void * dev_src, int n_tokens, int n_embd) {
    if (!handle || !dev_src) return;
    auto * ring = (dflash_cross_ring_gpu *)handle;

    if (layer < 0 || layer >= ring->n_layers) return;
    if (n_tokens <= 0) return;

    const float * src = (const float *)dev_src;
    if (n_tokens > ring->ring_size) {
        const int skip = n_tokens - ring->ring_size;
        src      += (size_t)skip * n_embd;
        ring_pos += skip;
        n_tokens  = ring->ring_size;
    }

    CUDA_CHECK(cudaSetDevice(ring->device));

    float * dst = ring->h_layer_ptrs[layer];
    const size_t stride = (size_t)n_embd * sizeof(float);

    ring_span_for_each(ring->ring_size, ring_pos, n_tokens, [&](int ring_tok, int src_tok, int n) {
        dflash_cross_ring_gpu_set_tensor(dst, src + (size_t)src_tok * n_embd,
                                         (size_t)ring_tok * stride, (size_t)n * stride);
    });
}

// Read a token range out of the ring into host memory (checkpoint persistence when the
// CPU ring is not being maintained). Handles wrap-around; synchronous.
extern "C" void dflash_cross_ring_gpu_read(
        void * handle, int layer, int ring_pos,
        float * host_dst, int n_tokens, int n_embd) {
    if (!handle || !host_dst) return;
    auto * ring = (dflash_cross_ring_gpu *)handle;

    if (layer < 0 || layer >= ring->n_layers) return;
    if (n_tokens <= 0 || n_tokens > ring->ring_size) return;

    CUDA_CHECK(cudaSetDevice(ring->device));

    const float * src = ring->h_layer_ptrs[layer];
    const size_t stride = (size_t)n_embd * sizeof(float);

    ring_span_for_each(ring->ring_size, ring_pos, n_tokens, [&](int ring_tok, int dst_tok, int n) {
        CUDA_CHECK(cudaMemcpy(
                host_dst + (size_t)dst_tok * n_embd,
                src + (size_t)ring_tok * n_embd,
                (size_t)n * stride, cudaMemcpyDeviceToHost));
    });
}

// Launch interleave kernel. Returns device pointer to interleaved staging buffer.
extern "C" const float * dflash_cross_ring_gpu_interleave(
        void * handle, int write_pos, int filled, int ctx_window) {
    if (!handle) return nullptr;
    auto * ring = (dflash_cross_ring_gpu *)handle;

    int cross_len = filled < ctx_window ? filled : ctx_window;
    if (cross_len <= 0) return nullptr;

    CUDA_CHECK(cudaSetDevice(ring->device));

    int read_start = ((write_pos - cross_len) % ring->ring_size + ring->ring_size) % ring->ring_size;

    dim3 grid(cross_len, ring->n_layers);
    dim3 block(256);

    k_cross_ring_interleave<<<grid, block, 0, cudaStreamPerThread>>>(
        (const float * const *)ring->d_layer_rings,
        ring->d_staging,
        ring->ring_size,
        read_start,
        cross_len,
        ring->n_layers,
        ring->n_embd);
    CUDA_CHECK(cudaGetLastError());

    // sync so staging is ready before drafter decode reads it
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));

    return ring->d_staging;
}

// ---------------------------------------------------------------------------
// Projected cross-KV cache: per-(drafter layer, ring slot) K/V projections.
// Slots map 1:1 to the cross ring above (slot = committed pos % ring_size), so
// a ring overwrite is invalidated simply by re-projecting the overwritten
// slots — which the update path does for every newly written token.
// K rows are pre-RoPE (positions are window-relative and slide every draft
// call, so RoPE must stay in the drafter graph); V rows are final.
// ---------------------------------------------------------------------------

struct dflash_crosskv_cache {
    int device;
    int n_layers;
    int ring_size;
    int quant;            // 0 = f32 rows, 1 = q8_0 rows (32-elem blocks, half scale)
    int64_t k_row;        // floats per token (K)
    int64_t v_row;        // floats per token (V)
    int64_t k_row_bytes;  // stored bytes per token
    int64_t v_row_bytes;
    uint8_t ** k_rings;   // host array of per-layer device pointers
    uint8_t ** v_rings;
};

// q8_0 block: half scale + 32 int8 (matches ggml block_q8_0 semantics: d = amax/127)
#define CKV_Q8_BLOCK   32
#define CKV_Q8_BYTES   (2 + CKV_Q8_BLOCK)

__global__ static void k_crosskv_quant_q8_0(
        const float * __restrict__ src, uint8_t * __restrict__ dst,
        const int groups_per_row, const int64_t row, const int n_tokens) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_tokens * groups_per_row) return;
    const int tok = g / groups_per_row;
    const int grp = g % groups_per_row;

    const float * s = src + (size_t)tok * row + (size_t)grp * CKV_Q8_BLOCK;
    uint8_t * d = dst + (size_t)tok * groups_per_row * CKV_Q8_BYTES + (size_t)grp * CKV_Q8_BYTES;

    float amax = 0.0f;
    for (int i = 0; i < CKV_Q8_BLOCK; i++) amax = fmaxf(amax, fabsf(s[i]));
    const float scale = amax / 127.0f;
    const float id    = scale != 0.0f ? 1.0f/scale : 0.0f;

    *(half *)d = __float2half(scale);
    int8_t * q = (int8_t *)(d + 2);
    for (int i = 0; i < CKV_Q8_BLOCK; i++) q[i] = (int8_t)roundf(s[i] * id);
}

__global__ static void k_crosskv_dequant_q8_0(
        const uint8_t * __restrict__ src, float * __restrict__ dst,
        const int groups_per_row, const int64_t row, const int n_tokens) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_tokens * groups_per_row) return;
    const int tok = g / groups_per_row;
    const int grp = g % groups_per_row;

    const uint8_t * s = src + (size_t)tok * groups_per_row * CKV_Q8_BYTES + (size_t)grp * CKV_Q8_BYTES;
    float * d = dst + (size_t)tok * row + (size_t)grp * CKV_Q8_BLOCK;

    const float scale = __half2float(*(const half *)s);
    const int8_t * q = (const int8_t *)(s + 2);
    for (int i = 0; i < CKV_Q8_BLOCK; i++) d[i] = q[i] * scale;
}

extern "C" void * dflash_crosskv_alloc(int n_layers, int64_t k_row, int64_t v_row, int ring_size, int quant) {
    if (quant != 0 && (k_row % CKV_Q8_BLOCK != 0 || v_row % CKV_Q8_BLOCK != 0)) {
        fprintf(stderr, "dflash crosskv: rows not divisible by %d — falling back to f32 storage\n", CKV_Q8_BLOCK);
        quant = 0;
    }

    auto * c = new dflash_crosskv_cache();
    if (!dflash_cuda_try(cudaGetDevice(&c->device), "cross-KV cudaGetDevice")) {
        delete c;
        return nullptr;
    }
    c->n_layers  = n_layers;
    c->ring_size = ring_size;
    c->quant     = quant;
    c->k_row     = k_row;
    c->v_row     = v_row;
    c->k_row_bytes = quant ? (k_row/CKV_Q8_BLOCK)*CKV_Q8_BYTES : k_row*(int64_t)sizeof(float);
    c->v_row_bytes = quant ? (v_row/CKV_Q8_BLOCK)*CKV_Q8_BYTES : v_row*(int64_t)sizeof(float);
    c->k_rings   = new uint8_t*[n_layers]();
    c->v_rings   = new uint8_t*[n_layers]();

    auto fail = [&]() {
        for (int l = 0; l < n_layers; l++) {
            if (c->k_rings[l]) {
                (void) dflash_cuda_try(cudaFree(c->k_rings[l]), "cross-KV rollback K cudaFree");
            }
            if (c->v_rings[l]) {
                (void) dflash_cuda_try(cudaFree(c->v_rings[l]), "cross-KV rollback V cudaFree");
            }
        }
        delete[] c->k_rings;
        delete[] c->v_rings;
        delete c;
        return (void *) nullptr;
    };

    for (int l = 0; l < n_layers; l++) {
        if (!dflash_cuda_try(
                cudaMalloc(&c->k_rings[l], (size_t)ring_size * c->k_row_bytes),
                "cross-KV K cudaMalloc")) {
            return fail();
        }
        if (!dflash_cuda_try(
                cudaMalloc(&c->v_rings[l], (size_t)ring_size * c->v_row_bytes),
                "cross-KV V cudaMalloc")) {
            return fail();
        }
        // zero-init: cold slots must stay finite (they can be gathered as masked pad;
        // an all-zero q8_0 block dequantizes to zeros)
        if (!dflash_cuda_try(
                cudaMemset(c->k_rings[l], 0, (size_t)ring_size * c->k_row_bytes),
                "cross-KV K cudaMemset") ||
            !dflash_cuda_try(
                cudaMemset(c->v_rings[l], 0, (size_t)ring_size * c->v_row_bytes),
                "cross-KV V cudaMemset")) {
            return fail();
        }
    }

    size_t total_kb = (size_t)n_layers * ring_size * (c->k_row_bytes + c->v_row_bytes) / 1024;
    fprintf(stderr, "dflash crosskv: allocated %d layers x %d slots (K %lld + V %lld floats/tok, %s, ~%zu KB)\n",
            n_layers, ring_size, (long long)k_row, (long long)v_row, quant ? "q8_0" : "f32", total_kb);
    return c;
}

extern "C" void dflash_crosskv_free(void * handle) {
    if (!handle) return;
    auto * c = (dflash_crosskv_cache *)handle;
    for (int l = 0; l < c->n_layers; l++) {
        (void) dflash_cuda_try(cudaFree(c->k_rings[l]), "cross-KV K cudaFree");
        (void) dflash_cuda_try(cudaFree(c->v_rings[l]), "cross-KV V cudaFree");
    }
    delete[] c->k_rings;
    delete[] c->v_rings;
    delete c;
}

// Write n_tokens projected rows (device src, contiguous f32 [n_tokens, row]) into
// the K (which==0) or V (which==1) ring at [ring_pos, ring_pos+n) with wrap-around.
// f32 mode: plain D2D copies. q8_0 mode: quantize kernel per span.
extern "C" void dflash_crosskv_write(
        void * handle, int layer, int which, int ring_pos,
        const void * dev_src, int n_tokens) {
    if (!handle || !dev_src || n_tokens <= 0) return;
    auto * c = (dflash_crosskv_cache *)handle;
    if (layer < 0 || layer >= c->n_layers || n_tokens > c->ring_size) return;

    CUDA_CHECK(cudaSetDevice(c->device));

    uint8_t * dst = which == 0 ? c->k_rings[layer] : c->v_rings[layer];
    const int64_t row       = which == 0 ? c->k_row       : c->v_row;
    const int64_t row_bytes = which == 0 ? c->k_row_bytes : c->v_row_bytes;
    const float * src = (const float *)dev_src;

    ring_span_for_each(c->ring_size, ring_pos, n_tokens, [&](int ring_tok, int src_tok, int n) {
        if (c->quant) {
            const int gpr = (int)(row / CKV_Q8_BLOCK);
            const int total = n * gpr;
            k_crosskv_quant_q8_0<<<(total + 255)/256, 256, 0, cudaStreamPerThread>>>(
                src + (size_t)src_tok * row, dst + (size_t)ring_tok * row_bytes, gpr, row, n);
            CUDA_CHECK(cudaGetLastError());
        } else {
            CUDA_CHECK(cudaMemcpyAsync(
                    dst + (size_t)ring_tok * row_bytes,
                    src + (size_t)src_tok * row,
                    (size_t)n * row_bytes, cudaMemcpyDeviceToDevice,
                    cudaStreamPerThread));
        }
    });
}

// Gather the window [start, start+n_tokens) of the K/V ring into a device f32
// destination (drafter graph input tensor) at byte offset dst_off.
extern "C" void dflash_crosskv_read_window(
        void * handle, int layer, int which, int start, int n_tokens,
        void * dev_dst, size_t dst_off) {
    if (!handle || !dev_dst || n_tokens <= 0) return;
    auto * c = (dflash_crosskv_cache *)handle;
    if (layer < 0 || layer >= c->n_layers || n_tokens > c->ring_size) return;

    CUDA_CHECK(cudaSetDevice(c->device));

    const uint8_t * src = which == 0 ? c->k_rings[layer] : c->v_rings[layer];
    const int64_t row       = which == 0 ? c->k_row       : c->v_row;
    const int64_t row_bytes = which == 0 ? c->k_row_bytes : c->v_row_bytes;
    char * dst = (char *)dev_dst + dst_off;

    ring_span_for_each(c->ring_size, start, n_tokens, [&](int ring_tok, int dst_tok, int n) {
        if (c->quant) {
            const int gpr = (int)(row / CKV_Q8_BLOCK);
            const int total = n * gpr;
            k_crosskv_dequant_q8_0<<<(total + 255)/256, 256, 0, cudaStreamPerThread>>>(
                src + (size_t)ring_tok * row_bytes, (float *)(dst + (size_t)dst_tok * row * sizeof(float)),
                gpr, row, n);
            CUDA_CHECK(cudaGetLastError());
        } else {
            CUDA_CHECK(cudaMemcpyAsync(
                    dst + (size_t)dst_tok * row_bytes,
                    src + (size_t)ring_tok * row_bytes,
                    (size_t)n * row_bytes, cudaMemcpyDeviceToDevice,
                    cudaStreamPerThread));
        }
    });
}

// Synchronize the stream all ring/cache copies run on. Needed before handing
// data to a compute path that runs on a different (backend) stream.
extern "C" void dflash_crosskv_sync(void) {
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

// D2D copy: from device source to device destination (raw pointers).
// Uses peer copy when source and destination are on different devices.
extern "C" void dflash_cross_ring_gpu_set_tensor(
        void * d_dst, const void * d_src, size_t offset, size_t size) {
    if (!d_dst || !d_src || size == 0) return;

    cudaPointerAttributes dst_attr, src_attr;
    CUDA_CHECK(cudaPointerGetAttributes(&dst_attr, (const char *)d_dst + offset));
    CUDA_CHECK(cudaPointerGetAttributes(&src_attr, d_src));

    if (dst_attr.type == cudaMemoryTypeDevice && src_attr.type == cudaMemoryTypeDevice
            && dst_attr.device != src_attr.device) {
        CUDA_CHECK(cudaMemcpyPeerAsync(
                (char *)d_dst + offset, dst_attr.device,
                d_src, src_attr.device, size, cudaStreamPerThread));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(
                (char *)d_dst + offset, d_src, size,
                cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    }
}
