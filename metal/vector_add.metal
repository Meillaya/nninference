#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        out[gid] = a[gid] + b[gid];
    }
}


kernel void logits_matmul(
    device const float* hidden [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    const uint rows = dims.x;
    const uint cols = dims.y;
    if (row >= rows) return;

    float sum = 0.0f;
    const uint base = row * cols;
    for (uint col = 0; col < cols; col += 1) {
        sum += weights[base + col] * hidden[col];
    }
    logits[row] = sum;
}

kernel void logits_matmul_tg(
    device const float* hidden [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    const uint rows = dims.x;
    const uint cols = dims.y;
    if (row >= rows) return;

    threadgroup float partial[256];
    float sum = 0.0f;
    const uint base = row * cols;
    for (uint col = tid; col < cols; col += threads_per_group) {
        sum += weights[base + col] * hidden[col];
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        logits[row] = partial[0];
    }
}
