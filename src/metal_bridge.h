#ifndef NNINFERENCE_METAL_BRIDGE_H
#define NNINFERENCE_METAL_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NnMetalProbe {
    char device_name[256];
    uint64_t recommended_max_working_set_size;
    uint64_t max_threads_per_threadgroup;
    uint8_t supports_non_uniform_threadgroups;
} NnMetalProbe;

typedef struct NnMetalSmokeResult {
    uint32_t n;
    uint32_t mismatches;
    float max_abs_error;
    double elapsed_ms;
} NnMetalSmokeResult;

int nn_metal_probe(NnMetalProbe *out_probe, char *err, size_t err_len);

int nn_metal_run_vector_add(
    const char *metallib_path,
    const float *a,
    const float *b,
    float *out,
    uint32_t n,
    NnMetalProbe *out_probe,
    NnMetalSmokeResult *out_result,
    char *err,
    size_t err_len
);


int nn_metal_run_logits_matmul(
    const char *metallib_path,
    const float *hidden,
    const float *weights_row_major,
    float *logits_out,
    uint32_t rows,
    uint32_t cols,
    NnMetalProbe *out_probe,
    NnMetalSmokeResult *out_result,
    char *err,
    size_t err_len
);

#ifdef __cplusplus
}
#endif

#endif
