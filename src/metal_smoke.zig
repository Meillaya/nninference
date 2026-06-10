const std = @import("std");
const Io = std.Io;
const c = @cImport({
    @cInclude("metal_bridge.h");
});

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    const args = try init.minimal.args.toSlice(arena);
    const metallib_path = if (args.len > 1) args[1] else "zig-out/metal/kernels.metallib";
    const metallib_path_z = try arena.dupeZ(u8, metallib_path);

    const n: usize = 1024;
    var lhs: [n]f32 = undefined;
    var rhs: [n]f32 = undefined;
    var out: [n]f32 = undefined;
    for (0..n) |i| {
        lhs[i] = @as(f32, @floatFromInt(i)) * 0.25 - 13.0;
        rhs[i] = @as(f32, @floatFromInt(i % 17)) * -0.5 + 7.0;
        out[i] = std.math.nan(f32);
    }

    var probe: c.NnMetalProbe = std.mem.zeroes(c.NnMetalProbe);
    var result: c.NnMetalSmokeResult = std.mem.zeroes(c.NnMetalSmokeResult);
    var err: [2048]u8 = std.mem.zeroes([2048]u8);

    const rc = c.nn_metal_run_vector_add(
        metallib_path_z.ptr,
        lhs[0..].ptr,
        rhs[0..].ptr,
        out[0..].ptr,
        @intCast(n),
        &probe,
        &result,
        err[0..].ptr,
        err.len,
    );
    if (rc != 0) {
        try stdout.print("metal smoke failed: rc={d} err={s}\n", .{ rc, cString(&err) });
        return error.MetalSmokeFailed;
    }
    if (result.mismatches != 0 or result.max_abs_error > 1.0e-6) {
        try stdout.print(
            "metal smoke mismatch: mismatches={d} max_abs_error={d}\n",
            .{ result.mismatches, result.max_abs_error },
        );
        return error.MetalSmokeMismatch;
    }

    try stdout.print(
        "{{\"device\":\"{s}\",\"recommended_max_working_set_size\":{d},\"max_threads_per_threadgroup\":{d},\"supports_non_uniform_threadgroups\":{},\"n\":{d},\"mismatches\":{d},\"max_abs_error\":{d},\"elapsed_ms\":{d}}}\n",
        .{
            cString(&probe.device_name),
            probe.recommended_max_working_set_size,
            probe.max_threads_per_threadgroup,
            probe.supports_non_uniform_threadgroups != 0,
            result.n,
            result.mismatches,
            result.max_abs_error,
            result.elapsed_ms,
        },
    );
}

fn cString(ptr: anytype) []const u8 {
    const bytes = std.mem.sliceAsBytes(ptr);
    return std.mem.sliceTo(bytes, 0);
}
