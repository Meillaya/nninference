const std = @import("std");
const Io = std.Io;

const Candidate = struct {
    token_id: usize,
    logit: f64,
    prob: f64 = 0.0,
};

const Options = struct {
    model_dir: []const u8 = "Qwen3.5-0.8B",
    prompt: ?[]const u8 = null,
    temperature: f64 = 0.6,
    top_p: f64 = 0.95,
    top_k: usize = 20,
    greedy: bool = false,
    json: bool = false,
    seed: u64 = 0x1234_5678_9abc_def0,
    logits_out: ?[]const u8 = null,
};

const BridgeMeta = struct {
    vocab_size: usize,
    hf_forward_argmax_token_id: usize,
    hf_generate_token_id: usize,
};

fn usage() []const u8 {
    return
    \\infer_cpu_v1 --prompt <text> [--model-dir Qwen3.5-0.8B]
    \\
    \\Options:
    \\  --temperature <float>  Sampling temperature (default 0.6); <=0 implies greedy
    \\  --top-p <float>        Nucleus sampling cutoff (default 0.95)
    \\  --top-k <int>          Top-k candidate cap (default 20)
    \\  --greedy               Deterministic argmax selection
    \\  --seed <int>           Sampling seed
    \\  --json                 Emit machine-readable JSON
    \\  --logits-out <path>    Also write full HF prefill logits as little-endian f32
    \\  --help                 Show this help
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
    var opt = Options{};

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try stdout.writeAll(usage());
            return;
        } else if (std.mem.eql(u8, arg, "--model-dir")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            opt.model_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            opt.prompt = args[i];
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            opt.temperature = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            opt.top_p = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            opt.top_k = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            opt.seed = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--logits-out")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            opt.logits_out = args[i];
        } else if (std.mem.eql(u8, arg, "--greedy")) {
            opt.greedy = true;
        } else if (std.mem.eql(u8, arg, "--json")) {
            opt.json = true;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            return error.UnknownArgument;
        }
    }

    const prompt = opt.prompt orelse {
        try stdout.writeAll(usage());
        return error.MissingPrompt;
    };

    if (opt.top_p <= 0.0 or opt.top_p > 1.0) return error.InvalidTopP;
    if (opt.temperature < 0.0) return error.InvalidTemperature;

    const bridge = try runBridge(allocator, init.io, opt, prompt);
    defer allocator.free(bridge.stdout);
    defer allocator.free(bridge.stderr);

    if (bridge.term != .exited or bridge.term.exited != 0) {
        std.debug.print("hf bridge failed:\n{s}\n", .{bridge.stderr});
        return error.BridgeFailed;
    }

    var candidates = std.ArrayList(Candidate).empty;
    defer candidates.deinit(allocator);
    const meta = try parseBridgeOutput(allocator, bridge.stdout, &candidates);
    if (candidates.items.len == 0) return error.NoCandidates;

    const selected = try selectToken(candidates.items, opt);

    if (opt.json) {
        try stdout.print(
            "{{\"selected_token_id\":{d},\"hf_forward_argmax_token_id\":{d},\"hf_generate_token_id\":{d},\"vocab_size\":{d},\"candidate_count\":{d},\"greedy\":{},\"temperature\":{d},\"top_p\":{d},\"top_k\":{d}}}\n",
            .{ selected.token_id, meta.hf_forward_argmax_token_id, meta.hf_generate_token_id, meta.vocab_size, candidates.items.len, opt.greedy or opt.temperature == 0.0, opt.temperature, opt.top_p, opt.top_k },
        );
    } else {
        try stdout.print("selected token id: {d}\n", .{selected.token_id});
        try stdout.print("hf forward argmax token id: {d}\n", .{meta.hf_forward_argmax_token_id});
        try stdout.print("hf generate token id: {d}\n", .{meta.hf_generate_token_id});
    }
}

fn runBridge(allocator: std.mem.Allocator, io: Io, opt: Options, prompt: []const u8) !std.process.RunResult {
    var count_buf: [32]u8 = undefined;
    const candidate_count = if (opt.top_k == 0) @as(usize, 1024) else @max(opt.top_k, 1);
    const count_arg = try std.fmt.bufPrint(&count_buf, "{d}", .{candidate_count});

    var argv = std.ArrayList([]const u8).empty;
    defer argv.deinit(allocator);
    try argv.append(allocator, "uv");
    try argv.append(allocator, "run");
    try argv.append(allocator, "python");
    try argv.append(allocator, "scripts/hf_prefill_bridge.py");
    try argv.append(allocator, "--model-dir");
    try argv.append(allocator, opt.model_dir);
    try argv.append(allocator, "--prompt");
    try argv.append(allocator, prompt);
    try argv.append(allocator, "--candidate-count");
    try argv.append(allocator, count_arg);
    if (opt.logits_out) |path| {
        try argv.append(allocator, "--logits-out");
        try argv.append(allocator, path);
    }

    return try std.process.run(allocator, io, .{
        .argv = argv.items,
        .stdout_limit = .limited(64 * 1024 * 1024),
        .stderr_limit = .limited(16 * 1024 * 1024),
    });
}

fn parseBridgeOutput(allocator: std.mem.Allocator, text: []const u8, candidates: *std.ArrayList(Candidate)) !BridgeMeta {
    var lines = std.mem.splitScalar(u8, text, '\n');

    var meta_line: ?[]const u8 = null;
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \r\t");
        if (std.mem.startsWith(u8, line, "{") and std.mem.indexOf(u8, line, "\"vocab_size\"") != null) {
            meta_line = line;
            break;
        }
    }
    const meta = try parseMeta(meta_line orelse return error.BadBridgeOutput);

    var found_marker = false;
    while (lines.next()) |line_raw| {
        if (std.mem.eql(u8, std.mem.trim(u8, line_raw, " \r\t"), "CANDIDATES")) {
            found_marker = true;
            break;
        }
    }
    if (!found_marker) return error.BadBridgeOutput;

    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \r\t");
        if (line.len == 0) continue;
        var parts = std.mem.splitScalar(u8, line, '\t');
        const id_s = parts.next() orelse return error.BadCandidate;
        const logit_s = parts.next() orelse return error.BadCandidate;
        try candidates.append(allocator, .{
            .token_id = try std.fmt.parseInt(usize, id_s, 10),
            .logit = try std.fmt.parseFloat(f64, logit_s),
        });
    }
    return meta;
}

fn parseMeta(line: []const u8) !BridgeMeta {
    return .{
        .vocab_size = try parseJsonInt(line, "\"vocab_size\":"),
        .hf_forward_argmax_token_id = try parseJsonInt(line, "\"hf_forward_argmax_token_id\":"),
        .hf_generate_token_id = try parseJsonInt(line, "\"hf_generate_token_id\":"),
    };
}

fn parseJsonInt(text: []const u8, key: []const u8) !usize {
    const start = std.mem.indexOf(u8, text, key) orelse return error.MissingJsonKey;
    var i: usize = start + key.len;
    const begin = i;
    while (i < text.len and text[i] >= '0' and text[i] <= '9') : (i += 1) {}
    if (i == begin) return error.BadJsonNumber;
    return try std.fmt.parseInt(usize, text[begin..i], 10);
}

fn selectToken(candidates: []Candidate, opt: Options) !Candidate {
    var items = candidates;
    std.mem.sort(Candidate, items, {}, candidateGreater);
    if (opt.greedy or opt.temperature == 0.0) return items[0];

    const use_len = if (opt.top_k == 0) items.len else @min(opt.top_k, items.len);
    var max_logit = items[0].logit;
    for (items[0..use_len]) |item| max_logit = @max(max_logit, item.logit);

    var total: f64 = 0.0;
    for (items[0..use_len]) |*item| {
        item.prob = std.math.exp((item.logit - max_logit) / opt.temperature);
        total += item.prob;
    }
    if (total <= 0.0 or !std.math.isFinite(total)) return error.BadProbability;

    for (items[0..use_len]) |*item| item.prob /= total;
    std.mem.sort(Candidate, items[0..use_len], {}, candidateProbGreater);

    var nucleus_len: usize = 0;
    var cumulative: f64 = 0.0;
    while (nucleus_len < use_len) : (nucleus_len += 1) {
        cumulative += items[nucleus_len].prob;
        if (cumulative >= opt.top_p) {
            nucleus_len += 1;
            break;
        }
    }
    nucleus_len = @max(nucleus_len, 1);

    var nucleus_total: f64 = 0.0;
    for (items[0..nucleus_len]) |item| nucleus_total += item.prob;

    var prng = std.Random.DefaultPrng.init(opt.seed);
    const draw = prng.random().float(f64) * nucleus_total;
    var seen: f64 = 0.0;
    for (items[0..nucleus_len]) |item| {
        seen += item.prob;
        if (draw <= seen) return item;
    }
    return items[nucleus_len - 1];
}

fn candidateGreater(_: void, a: Candidate, b: Candidate) bool {
    return a.logit > b.logit;
}

fn candidateProbGreater(_: void, a: Candidate, b: Candidate) bool {
    return a.prob > b.prob;
}
