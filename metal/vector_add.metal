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
