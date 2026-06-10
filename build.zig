const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "infer_cpu_v1",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run infer_cpu_v1");
    run_step.dependOn(&run_cmd.step);

    const metal_mod = b.createModule(.{
        .root_source_file = b.path("src/metal_smoke.zig"),
        .target = target,
        .optimize = optimize,
    });
    metal_mod.addIncludePath(b.path("src"));
    metal_mod.addCSourceFile(.{
        .file = b.path("src/metal_bridge.m"),
        .flags = &.{"-fobjc-arc"},
        .language = .objective_c,
    });
    metal_mod.linkFramework("Foundation", .{});
    metal_mod.linkFramework("Metal", .{});

    const metal_smoke = b.addExecutable(.{
        .name = "metal_smoke",
        .root_module = metal_mod,
    });

    const compile_metal = b.addSystemCommand(&.{
        "xcrun", "-sdk", "macosx", "metal", "-c",
    });
    compile_metal.addFileArg(b.path("metal/vector_add.metal"));
    compile_metal.addArg("-o");
    const air_file = compile_metal.addOutputFileArg("vector_add.air");

    const link_metallib = b.addSystemCommand(&.{
        "xcrun", "-sdk", "macosx", "metallib",
    });
    link_metallib.addFileArg(air_file);
    link_metallib.addArg("-o");
    const metallib_file = link_metallib.addOutputFileArg("kernels.metallib");

    const metal_lib_step = b.step("metal-lib", "Compile Metal kernels into a build-cache metallib");
    metal_lib_step.dependOn(&link_metallib.step);

    const metal_smoke_run = b.addRunArtifact(metal_smoke);
    metal_smoke_run.addFileArg(metallib_file);
    const metal_smoke_step = b.step("metal-smoke", "Run the Metal vector-add capability smoke test");
    metal_smoke_step.dependOn(&metal_smoke_run.step);
}
