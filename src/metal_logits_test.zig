const std = @import("std");
const Io = std.Io;
const c = @cImport({
    @cInclude("metal_bridge.h");
});

const fixture_magic = "NNLGFIX1";

const Tolerance = struct {
    max_abs: f32,
    max_rel: f32,
};

const Diff = struct {
    max_abs: f32 = 0.0,
    max_rel: f32 = 0.0,
    mismatches: u32 = 0,
};

const Fixture = struct {
    rows: usize,
    cols: usize,
    tol: Tolerance,
    hidden: []f32,
    weights: []f32,
    expected: []f32,

    fn deinit(self: Fixture, allocator: std.mem.Allocator) void {
        allocator.free(self.hidden);
        allocator.free(self.weights);
        allocator.free(self.expected);
    }
};

const Kernel = enum {
    scalar,
    threadgroup,
    auto,

    fn metalName(self: Kernel) [:0]const u8 {
        return switch (self) {
            .scalar => "logits_matmul",
            .threadgroup => "logits_matmul_tg",
            .auto => "auto",
        };
    }

    fn label(self: Kernel) []const u8 {
        return switch (self) {
            .scalar => "scalar",
            .threadgroup => "threadgroup",
            .auto => "auto",
        };
    }
};

const BufferMode = enum {
    copy,
    nocopy,

    fn label(self: BufferMode) []const u8 {
        return switch (self) {
            .copy => "copy",
            .nocopy => "nocopy",
        };
    }
};

const BenchmarkCommandMode = enum {
    per_iter,
    batched,

    fn label(self: BenchmarkCommandMode) []const u8 {
        return switch (self) {
            .per_iter => "per_iter",
            .batched => "batched",
        };
    }
};

const CompareMode = enum {
    full,
    topk,

    fn label(self: CompareMode) []const u8 {
        return switch (self) {
            .full => "full",
            .topk => "topk",
        };
    }
};

const Options = struct {
    metallib_path: []const u8,
    fixture_path: ?[]const u8 = null,
    expect_topk: bool = false,
    kernel_repeats: u32 = 1,
    benchmark_iters: u32 = 0,
    benchmark_command_mode: BenchmarkCommandMode = .per_iter,
    compare_mode: CompareMode = .full,
    kernel: Kernel = .scalar,
    matrix_kernels: ?[]const u8 = null,
    buffer_mode: BufferMode = .copy,
};

fn usage() []const u8 {
    return
    \\metal_logits_v1 <metallib> [--fixture artifacts/.../fixture.bin] [--expect-topk] [--kernel scalar|threadgroup|auto] [--matrix-kernels scalar,threadgroup] [--buffer-mode copy|nocopy] [--kernel-repeats N] [--benchmark-iters N] [--benchmark-command-mode per_iter|batched] [--compare-mode full|topk]
    \\
    \\Prototype-only CLI for the sidecar Metal LM-head logits projection.
    \\It does not run the transformer or replace infer_cpu_v1's default HF bridge.
    \\--kernel auto is opt-in: the Metal bridge chooses threadgroup only when
    \\the pipeline supports its fixed 256-thread layout, otherwise scalar.
    \\--benchmark-iters keeps one fixture load and one Metal buffer setup alive
    \\for N measured command-buffer iterations.
    \\--compare-mode topk is opt-in benchmark instrumentation: it requires
    \\--expect-topk and skips the full per-logit tolerance scan while still
    \\checking fixture top-1/top-20. The default remains full.
    \\--matrix-kernels is benchmark-only: it loads the fixture once, emits a
    \\shared-load header, then emits one JSON result per requested kernel.
    \\
    ;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const arena = init.arena.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    const args = try init.minimal.args.toSlice(arena);
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try stdout.writeAll(usage());
            return;
        }
    }
    const opt = try parseOptions(args);
    const metallib_path_z = try arena.dupeZ(u8, opt.metallib_path);

    if (opt.fixture_path) |path| {
        const fixture_load_start = Io.Clock.awake.now(init.io);
        const fixture = try loadFixture(init.io, allocator, path);
        const fixture_load_ms = elapsedMsSince(init.io, fixture_load_start);
        defer fixture.deinit(allocator);
        if (opt.matrix_kernels) |matrix_kernels| {
            try runMatrixCases(init.io, stdout, allocator, metallib_path_z, "checkpoint_f32_logits_matmul", fixture, opt.expect_topk, opt.kernel_repeats, opt.benchmark_iters, opt.benchmark_command_mode, opt.compare_mode, matrix_kernels, opt.buffer_mode, fixture_load_ms);
        } else {
            try runCase(init.io, stdout, allocator, metallib_path_z, "checkpoint_f32_logits_matmul", fixture, opt.expect_topk, opt.kernel_repeats, opt.benchmark_iters, opt.benchmark_command_mode, opt.compare_mode, opt.kernel, opt.buffer_mode, fixture_load_ms);
        }
    } else {
        const fixture = try makeTinyFixture(allocator);
        defer fixture.deinit(allocator);
        if (opt.matrix_kernels) |matrix_kernels| {
            try runMatrixCases(init.io, stdout, allocator, metallib_path_z, "tiny_f32_logits_matmul", fixture, opt.expect_topk, opt.kernel_repeats, opt.benchmark_iters, opt.benchmark_command_mode, opt.compare_mode, matrix_kernels, opt.buffer_mode, 0.0);
        } else {
            try runCase(init.io, stdout, allocator, metallib_path_z, "tiny_f32_logits_matmul", fixture, opt.expect_topk, opt.kernel_repeats, opt.benchmark_iters, opt.benchmark_command_mode, opt.compare_mode, opt.kernel, opt.buffer_mode, 0.0);
        }
    }
}

fn parseOptions(args: []const []const u8) !Options {
    var opt = Options{ .metallib_path = "kernels.metallib" };
    var i: usize = 1;
    if (i < args.len and !std.mem.startsWith(u8, args[i], "--")) {
        opt.metallib_path = args[i];
        i += 1;
    }
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--fixture")) {
            i += 1;
            if (i >= args.len) return error.MissingFixturePath;
            opt.fixture_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--expect-topk")) {
            opt.expect_topk = true;
        } else if (std.mem.eql(u8, args[i], "--kernel")) {
            i += 1;
            if (i >= args.len) return error.MissingKernel;
            opt.kernel = try parseKernel(args[i]);
        } else if (std.mem.eql(u8, args[i], "--matrix-kernels")) {
            i += 1;
            if (i >= args.len) return error.MissingMatrixKernels;
            opt.matrix_kernels = args[i];
        } else if (std.mem.eql(u8, args[i], "--buffer-mode")) {
            i += 1;
            if (i >= args.len) return error.MissingBufferMode;
            if (std.mem.eql(u8, args[i], "copy")) {
                opt.buffer_mode = .copy;
            } else if (std.mem.eql(u8, args[i], "nocopy")) {
                opt.buffer_mode = .nocopy;
            } else {
                return error.InvalidBufferMode;
            }
        } else if (std.mem.eql(u8, args[i], "--kernel-repeats")) {
            i += 1;
            if (i >= args.len) return error.MissingKernelRepeats;
            opt.kernel_repeats = try std.fmt.parseInt(u32, args[i], 10);
            if (opt.kernel_repeats == 0) return error.InvalidKernelRepeats;
        } else if (std.mem.eql(u8, args[i], "--benchmark-iters")) {
            i += 1;
            if (i >= args.len) return error.MissingBenchmarkIters;
            opt.benchmark_iters = try std.fmt.parseInt(u32, args[i], 10);
            if (opt.benchmark_iters == 0) return error.InvalidBenchmarkIters;
        } else if (std.mem.eql(u8, args[i], "--benchmark-command-mode")) {
            i += 1;
            if (i >= args.len) return error.MissingBenchmarkCommandMode;
            opt.benchmark_command_mode = try parseBenchmarkCommandMode(args[i]);
        } else if (std.mem.eql(u8, args[i], "--compare-mode") or std.mem.eql(u8, args[i], "--comparison-mode")) {
            i += 1;
            if (i >= args.len) return error.MissingCompareMode;
            opt.compare_mode = try parseCompareMode(args[i]);
        } else {
            std.debug.print("unknown argument: {s}\n", .{args[i]});
            return error.UnknownArgument;
        }
    }
    if (opt.compare_mode == .topk and !opt.expect_topk) return error.TopKCompareModeRequiresExpectTopK;
    return opt;
}

fn parseKernel(label: []const u8) !Kernel {
    if (std.mem.eql(u8, label, "scalar")) return .scalar;
    if (std.mem.eql(u8, label, "threadgroup")) return .threadgroup;
    if (std.mem.eql(u8, label, "auto")) return .auto;
    return error.InvalidKernel;
}

fn parseBenchmarkCommandMode(label: []const u8) !BenchmarkCommandMode {
    if (std.mem.eql(u8, label, "per_iter")) return .per_iter;
    if (std.mem.eql(u8, label, "batched")) return .batched;
    return error.InvalidBenchmarkCommandMode;
}

fn parseCompareMode(label: []const u8) !CompareMode {
    if (std.mem.eql(u8, label, "full")) return .full;
    if (std.mem.eql(u8, label, "topk")) return .topk;
    return error.InvalidCompareMode;
}

fn makeTinyFixture(allocator: std.mem.Allocator) !Fixture {
    const rows: usize = 17;
    const cols: usize = 19;
    const hidden = try allocator.alloc(f32, cols);
    errdefer allocator.free(hidden);
    const weights = try allocator.alloc(f32, rows * cols);
    errdefer allocator.free(weights);
    const expected = try allocator.alloc(f32, rows);
    errdefer allocator.free(expected);

    for (0..cols) |col| {
        const signed: f32 = @floatFromInt(@as(isize, @intCast(col)) - 9);
        hidden[col] = signed * 0.125;
    }
    for (0..rows) |row| {
        for (0..cols) |col| {
            const raw: i32 = @intCast((row * 7 + col * 3) % 23);
            weights[row * cols + col] = @as(f32, @floatFromInt(raw - 11)) * 0.03125;
        }
    }
    cpuMatmulF64(hidden, weights, expected, rows, cols);
    return .{
        .rows = rows,
        .cols = cols,
        .tol = .{ .max_abs = 1.0e-4, .max_rel = 1.0e-4 },
        .hidden = hidden,
        .weights = weights,
        .expected = expected,
    };
}

fn loadFixture(io: Io, allocator: std.mem.Allocator, path: []const u8) !Fixture {
    var file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    var header: [24]u8 = undefined;
    if (try file.readPositionalAll(io, &header, 0) != header.len) return error.BadFixtureSize;
    if (!std.mem.eql(u8, header[0..8], fixture_magic)) return error.BadFixtureMagic;
    const rows = std.mem.readInt(u32, header[8..12], .little);
    const cols = std.mem.readInt(u32, header[12..16], .little);
    const tol = Tolerance{
        .max_abs = readF32(header[16..20]),
        .max_rel = readF32(header[20..24]),
    };
    const rows_usize: usize = rows;
    const cols_usize: usize = cols;
    const hidden_len = cols_usize;
    const weights_len = rows_usize * cols_usize;
    const expected_len = rows_usize;
    const expected_bytes = 24 + 4 * (hidden_len + weights_len + expected_len);
    if (stat.size != expected_bytes) return error.BadFixtureSize;

    var offset: u64 = 24;
    const hidden = try allocator.alloc(f32, hidden_len);
    errdefer allocator.free(hidden);
    try readF32SliceDirect(file, io, hidden, &offset);
    const weights = try allocator.alloc(f32, weights_len);
    errdefer allocator.free(weights);
    try readF32SliceDirect(file, io, weights, &offset);
    const expected = try allocator.alloc(f32, expected_len);
    errdefer allocator.free(expected);
    try readF32SliceDirect(file, io, expected, &offset);

    return .{
        .rows = rows_usize,
        .cols = cols_usize,
        .tol = tol,
        .hidden = hidden,
        .weights = weights,
        .expected = expected,
    };
}

fn readF32(bytes: *const [4]u8) f32 {
    return @bitCast(std.mem.readInt(u32, bytes, .little));
}

fn readF32SliceDirect(file: Io.File, io: Io, out: []f32, offset: *u64) !void {
    const bytes = std.mem.sliceAsBytes(out);
    if (try file.readPositionalAll(io, bytes, offset.*) != bytes.len) return error.BadFixtureSize;
    offset.* += bytes.len;
}

fn runCase(io: Io, stdout: *Io.Writer, allocator: std.mem.Allocator, metallib_path_z: [:0]const u8, fixture_name: []const u8, fixture: Fixture, expect_topk: bool, kernel_repeats: u32, benchmark_iters: u32, benchmark_command_mode: BenchmarkCommandMode, compare_mode: CompareMode, kernel: Kernel, buffer_mode: BufferMode, fixture_load_ms: f64) !void {
    const total_start = Io.Clock.awake.now(io);
    const actual = try allocator.alloc(f32, fixture.rows);
    defer allocator.free(actual);
    @memset(actual, std.math.nan(f32));

    var probe: c.NnMetalProbe = std.mem.zeroes(c.NnMetalProbe);
    var result: c.NnMetalSmokeResult = std.mem.zeroes(c.NnMetalSmokeResult);
    var benchmark: c.NnMetalBenchmarkResult = std.mem.zeroes(c.NnMetalBenchmarkResult);
    var err: [2048]u8 = std.mem.zeroes([2048]u8);

    const bridge_start = Io.Clock.awake.now(io);
    const rc = if (benchmark_iters == 0)
        c.nn_metal_run_logits_matmul(
            metallib_path_z.ptr,
            kernel.metalName().ptr,
            fixture.hidden.ptr,
            fixture.weights.ptr,
            actual.ptr,
            @intCast(fixture.rows),
            @intCast(fixture.cols),
            kernel_repeats,
            if (buffer_mode == .nocopy) 1 else 0,
            &probe,
            &result,
            err[0..].ptr,
            err.len,
        )
    else if (benchmark_command_mode == .batched)
        c.nn_metal_benchmark_logits_matmul_persistent_batched(
            metallib_path_z.ptr,
            kernel.metalName().ptr,
            fixture.hidden.ptr,
            fixture.weights.ptr,
            actual.ptr,
            @intCast(fixture.rows),
            @intCast(fixture.cols),
            benchmark_iters,
            kernel_repeats,
            if (buffer_mode == .nocopy) 1 else 0,
            &probe,
            &result,
            &benchmark,
            err[0..].ptr,
            err.len,
        )
    else
        c.nn_metal_benchmark_logits_matmul_persistent(
            metallib_path_z.ptr,
            kernel.metalName().ptr,
            fixture.hidden.ptr,
            fixture.weights.ptr,
            actual.ptr,
            @intCast(fixture.rows),
            @intCast(fixture.cols),
            benchmark_iters,
            kernel_repeats,
            if (buffer_mode == .nocopy) 1 else 0,
            &probe,
            &result,
            &benchmark,
            err[0..].ptr,
            err.len,
        );
    const metal_bridge_wall_ms = elapsedMsSince(io, bridge_start);
    if (rc != 0) {
        try stdout.print("metal logits test failed: rc={d} err={s}\n", .{ rc, cString(&err) });
        return error.MetalLogitsFailed;
    }

    const compare_start = Io.Clock.awake.now(io);
    const full_compare_ran = compare_mode == .full;
    const diff = if (full_compare_ran) compare(fixture.expected, actual, fixture.tol) else Diff{};
    if (full_compare_ran and diff.mismatches != 0) {
        try stdout.print(
            "metal logits mismatch: fixture={s} mismatches={d} max_abs={d} max_rel={d}\n",
            .{ fixture_name, diff.mismatches, diff.max_abs, diff.max_rel },
        );
        return error.MetalLogitsMismatch;
    }

    const expected_top1 = argmax(fixture.expected);
    const actual_top1 = argmax(actual);
    var expected_top: [20]usize = undefined;
    var actual_top: [20]usize = undefined;
    const top1_match = expected_top1 == actual_top1;
    var top20_set_match = true;
    if (expect_topk) {
        fillTopK(fixture.expected, &expected_top);
        fillTopK(actual, &actual_top);
        top20_set_match = sameSet(&expected_top, &actual_top);
        if (!top1_match or !top20_set_match) {
            try stdout.print(
                "metal logits top-k mismatch: expected_top1={d} actual_top1={d} top20_set_match={}\n",
                .{ expected_top1, actual_top1, top20_set_match },
            );
            return error.MetalLogitsTopKMismatch;
        }
    }
    const host_compare_ms = elapsedMsSince(io, compare_start);
    const total_cli_measured_ms = fixture_load_ms + elapsedMsSince(io, total_start);

    const elapsed_ms_per_repeat = if (benchmark_iters == 0)
        result.elapsed_ms / @as(f64, @floatFromInt(kernel_repeats))
    else
        benchmark.elapsed_ms_per_kernel_repeat;

    const actual_buffer_mode = if (result.used_no_copy_buffers != 0) "nocopy" else "copy";
    const actual_kernel_label = cString(&result.actual_kernel_name);
    try stdout.print(
        "{{\"fixture\":\"{s}\",\"device\":\"{s}\",\"kernel\":\"{s}\",\"actual_kernel\":\"{s}\",\"buffer_mode\":\"{s}\",\"actual_buffer_mode\":\"{s}\",\"used_no_copy_buffers\":{},\"rows\":{d},\"cols\":{d},\"compare_mode\":\"{s}\",\"full_compare_ran\":{}",
        .{
            fixture_name,
            cString(&probe.device_name),
            kernel.label(),
            actual_kernel_label,
            buffer_mode.label(),
            actual_buffer_mode,
            result.used_no_copy_buffers != 0,
            fixture.rows,
            fixture.cols,
            compare_mode.label(),
            full_compare_ran,
        },
    );
    if (full_compare_ran) {
        try stdout.print(
            ",\"max_abs_diff\":{d},\"max_rel_diff\":{d},\"mismatches\":{d}",
            .{ diff.max_abs, diff.max_rel, diff.mismatches },
        );
    } else {
        try stdout.writeAll(",\"max_abs_diff\":null,\"max_rel_diff\":null,\"mismatches\":null");
    }
    try stdout.print(
        ",\"tolerance_max_abs\":{d},\"tolerance_max_rel\":{d},\"expected_top1\":{d},\"actual_top1\":{d},\"top1_match\":{},\"top20_set_match\":{},\"kernel_repeats\":{d},\"elapsed_ms\":{d},\"elapsed_ms_per_repeat\":{d},\"fixture_load_ms\":{d},\"metal_bridge_wall_ms\":{d},\"host_compare_ms\":{d},\"total_cli_measured_ms\":{d}",
        .{
            fixture.tol.max_abs,
            fixture.tol.max_rel,
            expected_top1,
            actual_top1,
            top1_match,
            top20_set_match,
            kernel_repeats,
            result.elapsed_ms,
            elapsed_ms_per_repeat,
            fixture_load_ms,
            metal_bridge_wall_ms,
            host_compare_ms,
            total_cli_measured_ms,
        },
    );
    if (benchmark_iters != 0) {
        try stdout.print(
            ",\"benchmark_iters\":{d},\"persistent_command_mode\":\"{s}\",\"persistent_setup_ms\":{d},\"persistent_elapsed_ms\":{d},\"persistent_ms_per_iter\":{d},\"persistent_ms_per_kernel_repeat\":{d}",
            .{
                benchmark.iterations,
                benchmark_command_mode.label(),
                benchmark.setup_ms,
                benchmark.elapsed_ms,
                benchmark.elapsed_ms_per_iteration,
                benchmark.elapsed_ms_per_kernel_repeat,
            },
        );
    }
    try stdout.writeAll("}\n");
}

fn runMatrixCases(io: Io, stdout: *Io.Writer, allocator: std.mem.Allocator, metallib_path_z: [:0]const u8, fixture_name: []const u8, fixture: Fixture, expect_topk: bool, kernel_repeats: u32, benchmark_iters: u32, benchmark_command_mode: BenchmarkCommandMode, compare_mode: CompareMode, matrix_kernels: []const u8, buffer_mode: BufferMode, shared_fixture_load_ms: f64) !void {
    try stdout.print(
        "{{\"matrix\":\"begin\",\"fixture\":\"{s}\",\"shared_fixture_load_ms\":{d},\"rows\":{d},\"cols\":{d},\"buffer_mode\":\"{s}\",\"kernel_repeats\":{d},\"benchmark_iters\":{d},\"benchmark_command_mode\":\"{s}\",\"compare_mode\":\"{s}\"}}\n",
        .{ fixture_name, shared_fixture_load_ms, fixture.rows, fixture.cols, buffer_mode.label(), kernel_repeats, benchmark_iters, benchmark_command_mode.label(), compare_mode.label() },
    );
    var count: usize = 0;
    var split = std.mem.splitScalar(u8, matrix_kernels, ',');
    while (split.next()) |raw_kernel| {
        const trimmed = std.mem.trim(u8, raw_kernel, " \t\r\n");
        if (trimmed.len == 0) continue;
        const kernel = try parseKernel(trimmed);
        try runCase(io, stdout, allocator, metallib_path_z, fixture_name, fixture, expect_topk, kernel_repeats, benchmark_iters, benchmark_command_mode, compare_mode, kernel, buffer_mode, 0.0);
        count += 1;
    }
    if (count == 0) return error.EmptyMatrixKernels;
    try stdout.print("{{\"matrix\":\"end\",\"kernel_count\":{d}}}\n", .{count});
}

fn elapsedMsSince(io: Io, start: Io.Timestamp) f64 {
    const elapsed = start.durationTo(Io.Clock.awake.now(io));
    return @as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_ms;
}

fn argmax(values: []const f32) usize {
    var best_index: usize = 0;
    var best_value = values[0];
    for (values[1..], 1..) |value, index| {
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }
    return best_index;
}

fn fillTopK(values: []const f32, out: *[20]usize) void {
    var scores: [20]f32 = @splat(-std.math.inf(f32));
    out.* = @splat(std.math.maxInt(usize));
    for (values, 0..) |value, index| {
        var pos: usize = 0;
        while (pos < scores.len) : (pos += 1) {
            if (value > scores[pos]) {
                var move: usize = scores.len - 1;
                while (move > pos) : (move -= 1) {
                    scores[move] = scores[move - 1];
                    out[move] = out[move - 1];
                }
                scores[pos] = value;
                out[pos] = index;
                break;
            }
        }
    }
}

fn sameSet(a: *const [20]usize, b: *const [20]usize) bool {
    for (a) |item_a| {
        var found = false;
        for (b) |item_b| {
            if (item_a == item_b) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

fn cpuMatmulF64(hidden: []const f32, weights: []const f32, out: []f32, rows: usize, cols: usize) void {
    for (0..rows) |row| {
        var sum: f64 = 0.0;
        for (0..cols) |col| {
            sum += @as(f64, @floatCast(weights[row * cols + col])) * @as(f64, @floatCast(hidden[col]));
        }
        out[row] = @floatCast(sum);
    }
}

fn compare(expected: []const f32, actual: []const f32, tol: Tolerance) Diff {
    var diff = Diff{};
    for (expected, actual) |e, a| {
        const abs = @abs(e - a);
        const denom = @max(@abs(e), 1.0e-12);
        const rel = abs / denom;
        diff.max_abs = @max(diff.max_abs, abs);
        diff.max_rel = @max(diff.max_rel, rel);
        if (!std.math.isFinite(a) or (abs > tol.max_abs and rel > tol.max_rel)) diff.mismatches += 1;
    }
    return diff;
}

fn cString(ptr: anytype) []const u8 {
    const bytes = std.mem.sliceAsBytes(ptr);
    return std.mem.sliceTo(bytes, 0);
}
