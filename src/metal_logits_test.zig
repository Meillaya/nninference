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

const Options = struct {
    metallib_path: []const u8,
    fixture_path: ?[]const u8 = null,
    expect_topk: bool = false,
};

fn usage() []const u8 {
    return
    \\metal_logits_v1 <metallib> [--fixture artifacts/.../fixture.bin] [--expect-topk]
    \\
    \\Prototype-only CLI for the sidecar Metal LM-head logits projection.
    \\It does not run the transformer or replace infer_cpu_v1's default HF bridge.
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
        const fixture = try loadFixture(init.io, allocator, path);
        defer fixture.deinit(allocator);
        try runCase(stdout, allocator, metallib_path_z, "checkpoint_f32_logits_matmul", fixture, opt.expect_topk);
    } else {
        const fixture = try makeTinyFixture(allocator);
        defer fixture.deinit(allocator);
        try runCase(stdout, allocator, metallib_path_z, "tiny_f32_logits_matmul", fixture, opt.expect_topk);
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
        } else {
            std.debug.print("unknown argument: {s}\n", .{args[i]});
            return error.UnknownArgument;
        }
    }
    return opt;
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
    const bytes = try Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(2 * 1024 * 1024 * 1024));
    defer allocator.free(bytes);
    if (bytes.len < 24 or !std.mem.eql(u8, bytes[0..8], fixture_magic)) return error.BadFixtureMagic;
    const rows = std.mem.readInt(u32, bytes[8..12], .little);
    const cols = std.mem.readInt(u32, bytes[12..16], .little);
    const tol = Tolerance{
        .max_abs = readF32(bytes[16..20]),
        .max_rel = readF32(bytes[20..24]),
    };
    const rows_usize: usize = rows;
    const cols_usize: usize = cols;
    const hidden_len = cols_usize;
    const weights_len = rows_usize * cols_usize;
    const expected_len = rows_usize;
    const expected_bytes = 24 + 4 * (hidden_len + weights_len + expected_len);
    if (bytes.len != expected_bytes) return error.BadFixtureSize;

    var offset: usize = 24;
    const hidden = try readF32Slice(allocator, bytes, &offset, hidden_len);
    errdefer allocator.free(hidden);
    const weights = try readF32Slice(allocator, bytes, &offset, weights_len);
    errdefer allocator.free(weights);
    const expected = try readF32Slice(allocator, bytes, &offset, expected_len);
    errdefer allocator.free(expected);

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

fn readF32Slice(allocator: std.mem.Allocator, bytes: []const u8, offset: *usize, len: usize) ![]f32 {
    const out = try allocator.alloc(f32, len);
    errdefer allocator.free(out);
    for (out) |*item| {
        item.* = readF32(bytes[offset.*..][0..4]);
        offset.* += 4;
    }
    return out;
}

fn runCase(stdout: *Io.Writer, allocator: std.mem.Allocator, metallib_path_z: [:0]const u8, fixture_name: []const u8, fixture: Fixture, expect_topk: bool) !void {
    const actual = try allocator.alloc(f32, fixture.rows);
    defer allocator.free(actual);
    @memset(actual, std.math.nan(f32));

    var probe: c.NnMetalProbe = std.mem.zeroes(c.NnMetalProbe);
    var result: c.NnMetalSmokeResult = std.mem.zeroes(c.NnMetalSmokeResult);
    var err: [2048]u8 = std.mem.zeroes([2048]u8);

    const rc = c.nn_metal_run_logits_matmul(
        metallib_path_z.ptr,
        fixture.hidden.ptr,
        fixture.weights.ptr,
        actual.ptr,
        @intCast(fixture.rows),
        @intCast(fixture.cols),
        &probe,
        &result,
        err[0..].ptr,
        err.len,
    );
    if (rc != 0) {
        try stdout.print("metal logits test failed: rc={d} err={s}\n", .{ rc, cString(&err) });
        return error.MetalLogitsFailed;
    }

    const diff = compare(fixture.expected, actual, fixture.tol);
    if (diff.mismatches != 0) {
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

    try stdout.print(
        "{{\"fixture\":\"{s}\",\"device\":\"{s}\",\"rows\":{d},\"cols\":{d},\"max_abs_diff\":{d},\"max_rel_diff\":{d},\"mismatches\":{d},\"tolerance_max_abs\":{d},\"tolerance_max_rel\":{d},\"expected_top1\":{d},\"actual_top1\":{d},\"top1_match\":{},\"top20_set_match\":{},\"elapsed_ms\":{d}}}\n",
        .{
            fixture_name,
            cString(&probe.device_name),
            fixture.rows,
            fixture.cols,
            diff.max_abs,
            diff.max_rel,
            diff.mismatches,
            fixture.tol.max_abs,
            fixture.tol.max_rel,
            expected_top1,
            actual_top1,
            top1_match,
            top20_set_match,
            result.elapsed_ms,
        },
    );
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
