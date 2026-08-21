#include "dflash2-conv.cuh"

template <typename BaseT>
static __device__ __forceinline__ float dflash2_base_to_f32(BaseT value) {
    return (float) value;
}

template <typename BaseT>
static __global__ void dflash2_conv_f32(
        const float * __restrict__ hidden,
        const float * __restrict__ projected,
        const BaseT * __restrict__ base,
        float * __restrict__ dst,
        int64_t n_embd,
        int64_t n_tokens,
        int64_t n_groups,
        int32_t side,
        int32_t group_size,
        int32_t block_size) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_embd*n_tokens) {
        return;
    }
    const int64_t token = i / n_embd;
    const int64_t ch    = i - token*n_embd;
    const int64_t group = ch / group_size;
    const int64_t dbase = token*(4*n_groups) + side*2*n_groups + group;
    const int64_t bbase = side*2*n_embd + ch;
    // Match the generic graph's ADD -> MUL -> ADD tensor boundaries exactly.
    // Explicit RN operations prevent compiler contraction from changing selector
    // decisions on proposals whose logits are nearly tied.
    const float c0 = __fadd_rn(dflash2_base_to_f32(base[bbase]), projected[dbase]);
    const float c1 = __fadd_rn(dflash2_base_to_f32(base[bbase + n_embd]), projected[dbase + n_groups]);
    const float prev = token % block_size ? hidden[i - n_embd] : 0.0f;
    dst[i] = __fadd_rn(__fmul_rn(c0, hidden[i]), __fmul_rn(c1, prev));
}

void ggml_cuda_op_dflash2_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * hidden    = dst->src[0];
    const ggml_tensor * projected = dst->src[1];
    const ggml_tensor * base      = dst->src[2];
    const int32_t side       = ggml_get_op_params_i32(dst, 0);
    const int32_t group_size = ggml_get_op_params_i32(dst, 1);
    const int32_t block_size = ggml_get_op_params_i32(dst, 2);
    const int64_t n_embd     = hidden->ne[0];
    const int64_t n_tokens   = hidden->ne[1];
    const int64_t n_groups   = n_embd/group_size;
    const int64_t n          = n_embd*n_tokens;
    const int threads = 256;
    const int blocks  = (int) ((n + threads - 1)/threads);

    const ggml_cuda_kernel_launch_params launch =
        ggml_cuda_kernel_launch_params(blocks, threads, 0, ctx.stream());
    if (base->type == GGML_TYPE_F16) {
        ggml_cuda_kernel_launch(dflash2_conv_f32<half>, launch,
                (const float *) hidden->data, (const float *) projected->data,
                (const half *) base->data, (float *) dst->data,
                n_embd, n_tokens, n_groups, side, group_size, block_size);
    } else {
        GGML_ASSERT(base->type == GGML_TYPE_F32);
        ggml_cuda_kernel_launch(dflash2_conv_f32<float>, launch,
                (const float *) hidden->data, (const float *) projected->data,
                (const float *) base->data, (float *) dst->data,
                n_embd, n_tokens, n_groups, side, group_size, block_size);
    }
}
