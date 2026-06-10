#import "metal_bridge.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static void nn_set_error(char *err, size_t err_len, const char *fmt, ...) {
    if (err == NULL || err_len == 0) return;
    va_list args;
    va_start(args, fmt);
    vsnprintf(err, err_len, fmt, args);
    va_end(args);
    err[err_len - 1] = '\0';
}

static void nn_clear_error(char *err, size_t err_len) {
    if (err != NULL && err_len > 0) err[0] = '\0';
}

static void nn_fill_probe(id<MTLDevice> device, id<MTLComputePipelineState> pipeline, NnMetalProbe *probe) {
    if (probe == NULL) return;
    memset(probe, 0, sizeof(*probe));
    const char *name = [[device name] UTF8String];
    if (name != NULL) {
        strncpy(probe->device_name, name, sizeof(probe->device_name) - 1);
    }
    if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
        probe->recommended_max_working_set_size = (uint64_t)[device recommendedMaxWorkingSetSize];
    }
    if (pipeline != nil) {
        probe->max_threads_per_threadgroup = (uint64_t)[pipeline maxTotalThreadsPerThreadgroup];
    }
    if ([device respondsToSelector:@selector(supportsFamily:)]) {
        probe->supports_non_uniform_threadgroups = 1;
    }
}

int nn_metal_probe(NnMetalProbe *out_probe, char *err, size_t err_len) {
    @autoreleasepool {
        nn_clear_error(err, err_len);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            nn_set_error(err, err_len, "MTLCreateSystemDefaultDevice returned nil");
            return 1;
        }
        nn_fill_probe(device, nil, out_probe);
        return 0;
    }
}

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
) {
    @autoreleasepool {
        nn_clear_error(err, err_len);
        if (metallib_path == NULL || a == NULL || b == NULL || out == NULL || n == 0) {
            nn_set_error(err, err_len, "invalid vector_add arguments");
            return 2;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            nn_set_error(err, err_len, "MTLCreateSystemDefaultDevice returned nil");
            return 3;
        }

        NSError *ns_error = nil;
        NSString *library_path = [NSString stringWithUTF8String:metallib_path];
        id<MTLLibrary> library = [device newLibraryWithFile:library_path error:&ns_error];
        if (library == nil) {
            nn_set_error(err, err_len, "newLibraryWithFile failed: %s", [[ns_error localizedDescription] UTF8String]);
            return 4;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"vector_add"];
        if (function == nil) {
            nn_set_error(err, err_len, "Metal function vector_add not found");
            return 5;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&ns_error];
        if (pipeline == nil) {
            nn_set_error(err, err_len, "newComputePipelineStateWithFunction failed: %s", [[ns_error localizedDescription] UTF8String]);
            return 6;
        }
        nn_fill_probe(device, pipeline, out_probe);

        const NSUInteger byte_count = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> buffer_a = [device newBufferWithLength:byte_count options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_b = [device newBufferWithLength:byte_count options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_out = [device newBufferWithLength:byte_count options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_n = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        if (buffer_a == nil || buffer_b == nil || buffer_out == nil || buffer_n == nil) {
            nn_set_error(err, err_len, "failed to allocate shared Metal buffers");
            return 7;
        }
        memcpy([buffer_a contents], a, byte_count);
        memcpy([buffer_b contents], b, byte_count);
        memset([buffer_out contents], 0, byte_count);
        memcpy([buffer_n contents], &n, sizeof(uint32_t));

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (queue == nil || command_buffer == nil || encoder == nil) {
            nn_set_error(err, err_len, "failed to create Metal command encoder");
            return 8;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_a offset:0 atIndex:0];
        [encoder setBuffer:buffer_b offset:0 atIndex:1];
        [encoder setBuffer:buffer_out offset:0 atIndex:2];
        [encoder setBuffer:buffer_n offset:0 atIndex:3];

        NSUInteger threads_per_group = [pipeline maxTotalThreadsPerThreadgroup];
        if (threads_per_group > 256) threads_per_group = 256;
        if (threads_per_group == 0) threads_per_group = 1;
        if (threads_per_group > n) threads_per_group = n;
        MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
        MTLSize grid_size = MTLSizeMake(n, 1, 1);

        CFAbsoluteTime start_time = CFAbsoluteTimeGetCurrent();
        if ([encoder respondsToSelector:@selector(dispatchThreads:threadsPerThreadgroup:)]) {
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        } else {
            const NSUInteger group_count = ((NSUInteger)n + threads_per_group - 1) / threads_per_group;
            [encoder dispatchThreadgroups:MTLSizeMake(group_count, 1, 1) threadsPerThreadgroup:threadgroup_size];
        }
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        CFAbsoluteTime end_time = CFAbsoluteTimeGetCurrent();

        if ([command_buffer error] != nil) {
            nn_set_error(err, err_len, "Metal command buffer failed: %s", [[command_buffer.error localizedDescription] UTF8String]);
            return 9;
        }

        memcpy(out, [buffer_out contents], byte_count);

        if (out_result != NULL) {
            memset(out_result, 0, sizeof(*out_result));
            out_result->n = n;
            out_result->elapsed_ms = (end_time - start_time) * 1000.0;
            for (uint32_t i = 0; i < n; i += 1) {
                const float expected = a[i] + b[i];
                const float abs_error = fabsf(out[i] - expected);
                if (abs_error > out_result->max_abs_error) out_result->max_abs_error = abs_error;
                if (abs_error > 1.0e-6f) out_result->mismatches += 1;
            }
        }
        return 0;
    }
}

int nn_metal_run_logits_matmul(
    const char *metallib_path,
    const float *hidden,
    const float *weights_row_major,
    float *logits_out,
    uint32_t rows,
    uint32_t cols,
    uint32_t repeat_count,
    NnMetalProbe *out_probe,
    NnMetalSmokeResult *out_result,
    char *err,
    size_t err_len
) {
    @autoreleasepool {
        nn_clear_error(err, err_len);
        if (metallib_path == NULL || hidden == NULL || weights_row_major == NULL || logits_out == NULL || rows == 0 || cols == 0) {
            nn_set_error(err, err_len, "invalid logits_matmul arguments");
            return 20;
        }
        if (repeat_count == 0) repeat_count = 1;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            nn_set_error(err, err_len, "MTLCreateSystemDefaultDevice returned nil");
            return 21;
        }

        NSError *ns_error = nil;
        NSString *library_path = [NSString stringWithUTF8String:metallib_path];
        id<MTLLibrary> library = [device newLibraryWithFile:library_path error:&ns_error];
        if (library == nil) {
            nn_set_error(err, err_len, "newLibraryWithFile failed: %s", [[ns_error localizedDescription] UTF8String]);
            return 22;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"logits_matmul"];
        if (function == nil) {
            nn_set_error(err, err_len, "Metal function logits_matmul not found");
            return 23;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&ns_error];
        if (pipeline == nil) {
            nn_set_error(err, err_len, "newComputePipelineStateWithFunction failed: %s", [[ns_error localizedDescription] UTF8String]);
            return 24;
        }
        nn_fill_probe(device, pipeline, out_probe);

        const NSUInteger hidden_bytes = (NSUInteger)cols * sizeof(float);
        const NSUInteger weight_bytes = (NSUInteger)rows * (NSUInteger)cols * sizeof(float);
        const NSUInteger logits_bytes = (NSUInteger)rows * sizeof(float);
        id<MTLBuffer> hidden_buffer = [device newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> weights_buffer = [device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> logits_buffer = [device newBufferWithLength:logits_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> dims_buffer = [device newBufferWithLength:sizeof(uint32_t) * 2 options:MTLResourceStorageModeShared];
        if (hidden_buffer == nil || weights_buffer == nil || logits_buffer == nil || dims_buffer == nil) {
            nn_set_error(err, err_len, "failed to allocate shared Metal buffers");
            return 25;
        }
        memcpy([hidden_buffer contents], hidden, hidden_bytes);
        memcpy([weights_buffer contents], weights_row_major, weight_bytes);
        memset([logits_buffer contents], 0, logits_bytes);
        uint32_t dims[2] = { rows, cols };
        memcpy([dims_buffer contents], dims, sizeof(dims));

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (queue == nil || command_buffer == nil || encoder == nil) {
            nn_set_error(err, err_len, "failed to create Metal command encoder");
            return 26;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:hidden_buffer offset:0 atIndex:0];
        [encoder setBuffer:weights_buffer offset:0 atIndex:1];
        [encoder setBuffer:logits_buffer offset:0 atIndex:2];
        [encoder setBuffer:dims_buffer offset:0 atIndex:3];

        NSUInteger threads_per_group = [pipeline maxTotalThreadsPerThreadgroup];
        if (threads_per_group > 256) threads_per_group = 256;
        if (threads_per_group == 0) threads_per_group = 1;
        if (threads_per_group > rows) threads_per_group = rows;
        MTLSize threadgroup_size = MTLSizeMake(threads_per_group, 1, 1);
        MTLSize grid_size = MTLSizeMake(rows, 1, 1);

        CFAbsoluteTime start_time = CFAbsoluteTimeGetCurrent();
        for (uint32_t repeat = 0; repeat < repeat_count; repeat += 1) {
            if ([encoder respondsToSelector:@selector(dispatchThreads:threadsPerThreadgroup:)]) {
                [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
            } else {
                const NSUInteger group_count = ((NSUInteger)rows + threads_per_group - 1) / threads_per_group;
                [encoder dispatchThreadgroups:MTLSizeMake(group_count, 1, 1) threadsPerThreadgroup:threadgroup_size];
            }
        }
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        CFAbsoluteTime end_time = CFAbsoluteTimeGetCurrent();

        if ([command_buffer error] != nil) {
            nn_set_error(err, err_len, "Metal command buffer failed: %s", [[command_buffer.error localizedDescription] UTF8String]);
            return 27;
        }

        memcpy(logits_out, [logits_buffer contents], logits_bytes);
        if (out_result != NULL) {
            memset(out_result, 0, sizeof(*out_result));
            out_result->n = rows;
            out_result->elapsed_ms = (end_time - start_time) * 1000.0;
        }
        return 0;
    }
}
