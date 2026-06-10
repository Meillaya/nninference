#!/usr/bin/env python3
"""Consolidate local nninference benchmark findings into report artifacts.

The output is intentionally dependency-free: it writes JSON, Markdown, and SVG
charts using only the Python standard library so the repo keeps uv dependency
hygiene unchanged.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "findings" / "kimi_style_local"
KIMI_URL = "https://www.kimi.com/blog/kimi-k2-6"
CHART_FILENAMES = [
    "cli_time_by_milestone.svg",
    "cli_speedup_vs_g054.svg",
    "persistent_kernel_timing.svg",
    "persistent_speedup_vs_g054.svg",
    "fixture_load_ms.svg",
]


@dataclass(frozen=True)
class Bar:
    label: str
    value: float
    unit: str
    source: str
    note: str = ""
    lower_is_better: bool = True


def read_json(path: str) -> dict[str, Any]:
    full = ROOT / path
    if not full.exists():
        raise FileNotFoundError(full)
    return json.loads(full.read_text())


def get(data: dict[str, Any], *path: Any) -> Any:
    cur: Any = data
    for key in path:
        if isinstance(cur, dict):
            cur = cur[key]
        elif isinstance(cur, list):
            cur = cur[key]
        else:
            raise KeyError(path)
    return cur


def mean_ms(path: str, key: str) -> float:
    return float(get(read_json(path), key, "mean_ms"))


def improvement_pct(old: float, new: float) -> float:
    return ((old - new) / old) * 100.0


def fmt(x: float, digits: int = 2) -> str:
    if math.isnan(x):
        return "n/a"
    return f"{x:.{digits}f}"


def svg_bar_chart(title: str, subtitle: str, bars: list[Bar], out: Path, *, width: int = 1100) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    margin_left = 300
    margin_right = 170
    row_h = 58
    top = 112
    bottom = 44
    height = top + row_h * len(bars) + bottom
    max_v = max(bar.value for bar in bars) if bars else 1.0
    plot_w = width - margin_left - margin_right
    colors = ["#111827", "#2563eb", "#059669", "#7c3aed", "#ea580c", "#dc2626", "#0891b2"]

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    parts.append(f'<text x="32" y="44" font-family="Inter, ui-sans-serif, system-ui, -apple-system" font-size="28" font-weight="800" fill="#111827">{html.escape(title)}</text>')
    parts.append(f'<text x="32" y="74" font-family="Inter, ui-sans-serif, system-ui, -apple-system" font-size="15" fill="#4b5563">{html.escape(subtitle)}</text>')
    for i, bar in enumerate(bars):
        y = top + i * row_h
        w = max(2.0, (bar.value / max_v) * plot_w)
        color = colors[i % len(colors)]
        parts.append(f'<text x="32" y="{y + 24}" font-family="Inter, ui-sans-serif, system-ui, -apple-system" font-size="15" font-weight="650" fill="#111827">{html.escape(bar.label)}</text>')
        parts.append(f'<rect x="{margin_left}" y="{y}" width="{w:.1f}" height="30" rx="8" fill="{color}" opacity="0.90"/>')
        parts.append(f'<text x="{margin_left + w + 12:.1f}" y="{y + 21}" font-family="Inter, ui-sans-serif, system-ui, -apple-system" font-size="14" font-weight="700" fill="#111827">{fmt(bar.value)} {html.escape(bar.unit)}</text>')
        if bar.note:
            parts.append(f'<text x="{margin_left}" y="{y + 48}" font-family="Inter, ui-sans-serif, system-ui, -apple-system" font-size="12" fill="#6b7280">{html.escape(bar.note)}</text>')
    parts.append(f'<text x="32" y="{height - 18}" font-family="Inter, ui-sans-serif, system-ui, -apple-system" font-size="12" fill="#6b7280">Generated from local artifact JSON. Lower is better unless stated otherwise.</text>')
    parts.append('</svg>\n')
    out.write_text("\n".join(parts))


def svg_speedup_chart(title: str, subtitle: str, bars: list[Bar], baseline: float, out: Path, *, width: int = 1100) -> None:
    speed_bars = [
        Bar(
            label=bar.label,
            value=baseline / bar.value if bar.value else 0.0,
            unit="×",
            source=bar.source,
            note=f"{fmt(bar.value)} {bar.unit}; source: {bar.source}",
            lower_is_better=False,
        )
        for bar in bars
    ]
    svg_bar_chart(title, subtitle, speed_bars, out, width=width)


def table_md(headers: list[str], rows: list[list[str]]) -> str:
    sep = ["---" for _ in headers]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def build() -> None:
    out = DEFAULT_OUT
    charts = out / "charts"
    out.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    for stale_svg in charts.glob("*.svg"):
        stale_svg.unlink()

    cli_bars = [
        Bar("G054: pre-loader scalar/copy", mean_ms("artifacts/benchmarks/g054_instrumented_breakdown.json", "metal_cli_measured_total_ms"), "ms", "g054_instrumented_breakdown.json", "Before direct fixture loading; full-vocab LM-head sidecar."),
        Bar("G056: direct fixture loader", mean_ms("artifacts/benchmarks/g056_direct_loader.json", "metal_cli_measured_total_ms"), "ms", "g056_direct_loader.json", "Same scalar/copy metric after direct loading."),
        Bar("G057: strict no-copy", mean_ms("artifacts/benchmarks/g057_nocopy.json", "metal_cli_measured_total_ms"), "ms", "g057_nocopy.json", "Opt-in no-copy path; defaults stayed copy."),
        Bar("G060: best matrix row", float(get(read_json("artifacts/benchmarks/g060_matrix_r7_p10.json"), "ranked_by_measured_total_median_ms", 0, "measured_total_median_ms")), "ms", "g060_matrix_r7_p10.json", "Medium-confidence threadgroup/nocopy median."),
    ]

    kernel_bars = [
        Bar("G054: scalar/copy persistent", float(get(read_json("artifacts/benchmarks/g054_instrumented_breakdown.json"), "persistent_metal", "record", "persistent_ms_per_kernel_repeat")), "ms/repeat", "g054_instrumented_breakdown.json", "Early persistent scalar baseline."),
        Bar("G057: scalar/nocopy", float(get(read_json("artifacts/benchmarks/g057_nocopy.json"), "persistent_metal", "record", "persistent_ms_per_kernel_repeat")), "ms/repeat", "g057_nocopy.json", "No-copy reduced bridge overhead."),
        Bar("G060: threadgroup/nocopy", float(get(read_json("artifacts/benchmarks/g060_matrix_r7_p10.json"), "ranked_by_persistent_ms_per_kernel_repeat", 0, "persistent_ms_per_kernel_repeat")), "ms/repeat", "g060_matrix_r7_p10.json", "Best medium-confidence matrix row."),
        Bar("G080: per_iter best", float(get(read_json("artifacts/benchmarks/g084_regression_summary/summary.json"), "best_by_mode", "per_iter", "median_ms_per_kernel_repeat")), "ms/repeat", "g084_regression_summary/summary.json", "Best retained per-iter command-mode winner."),
        Bar("G080: batched best", float(get(read_json("artifacts/benchmarks/g084_regression_summary/summary.json"), "best_by_mode", "batched", "median_ms_per_kernel_repeat")), "ms/repeat", "g084_regression_summary/summary.json", "Best retained batched command-mode winner."),
        Bar("G086: session/copy probe", float(get(read_json("artifacts/benchmarks/g086_session/reuse_session_copy_samples2_final.json"), "ranked_by_persistent_ms_per_kernel_repeat", 0, "persistent_ms_per_kernel_repeat")), "ms/repeat", "g086_session/reuse_session_copy_samples2_final.json", "Low-sample copy-backed retained session; not promoted."),
    ]

    fixture_bars = [
        Bar("G054 fixture load", mean_ms("artifacts/benchmarks/g054_instrumented_breakdown.json", "metal_cli_fixture_load_ms"), "ms", "g054_instrumented_breakdown.json", "Read+copy dominated early one-shot runs."),
        Bar("G056 direct loader", mean_ms("artifacts/benchmarks/g056_direct_loader.json", "metal_cli_fixture_load_ms"), "ms", "g056_direct_loader.json", "Direct loading cut fixture load time."),
        Bar("G057 no-copy path", mean_ms("artifacts/benchmarks/g057_nocopy.json", "metal_cli_fixture_load_ms"), "ms", "g057_nocopy.json", "Similar load time; bridge transfer improved."),
    ]

    svg_bar_chart("Full-vocab LM-head CLI time by milestone", "Local sidecar benchmark, not full token generation throughput", cli_bars, charts / "cli_time_by_milestone.svg")
    svg_speedup_chart("Speedup vs G054 CLI baseline", "Derived from local measured_total_ms artifacts", cli_bars, cli_bars[0].value, charts / "cli_speedup_vs_g054.svg")
    svg_bar_chart("Persistent LM-head kernel timing", "Median/best per-kernel-repeat from local benchmark artifacts", kernel_bars, charts / "persistent_kernel_timing.svg")
    svg_speedup_chart("Persistent kernel speedup vs G054", "Lower ms/repeat converted to speedup; metrics are sidecar-only", kernel_bars, kernel_bars[0].value, charts / "persistent_speedup_vs_g054.svg")
    svg_bar_chart("Fixture-load bottleneck collapse", "Direct loading removed the largest early one-shot bottleneck", fixture_bars, charts / "fixture_load_ms.svg")

    kimi_claim = {
        "source_url": KIMI_URL,
        "paraphrase": "The Kimi post describes a long-horizon Mac inference case improving Qwen3.5-0.8B throughput from about 15 to 193 tokens/s over 12+ hours and 14 iterations.",
        "critical_comparison": "This repo did not reproduce that end-to-end token/s metric. Local artifacts measure HF bridge alignment plus a Metal LM-head logits sidecar, so comparisons below are engineering-pattern comparisons, not model/product benchmark equivalence.",
        "claimed_speedup_x": 193.0 / 15.0,
    }

    cli_speedup = cli_bars[0].value / min(b.value for b in cli_bars)
    persistent_speedup = kernel_bars[0].value / min(b.value for b in kernel_bars)
    findings = {
        "generated_from": "local artifact JSON under artifacts/benchmarks",
        "inspiration_source": kimi_claim,
        "local_scope": "Qwen3.5-0.8B HF bridge plus Zig/Metal LM-head logits projection sidecar; not full native transformer inference.",
        "headline_metrics": {
            "cli_measured_total_best_speedup_vs_g054_x": cli_speedup,
            "cli_measured_total_best_ms": min(b.value for b in cli_bars),
            "persistent_kernel_best_speedup_vs_g054_x": persistent_speedup,
            "persistent_kernel_best_ms_per_repeat": min(b.value for b in kernel_bars),
            "fixture_load_improvement_pct_g054_to_g056": improvement_pct(fixture_bars[0].value, fixture_bars[1].value),
            "kimi_claimed_tokens_per_second_speedup_x": kimi_claim["claimed_speedup_x"],
        },
        "cli_time_by_milestone": [asdict(b) for b in cli_bars],
        "persistent_kernel_timing": [asdict(b) for b in kernel_bars],
        "fixture_load_timing": [asdict(b) for b in fixture_bars],
        "charts": [f"charts/{name}" for name in CHART_FILENAMES],
        "critical_findings": [
            "The strongest local one-shot improvement came from data movement and fixture loading, not from replacing the full model runtime.",
            "The best retained command-mode evidence stayed around 10.59 ms per LM-head kernel repeat for threadgroup/nocopy in G080/G084.",
            "G086 retained sessions are useful as an opt-in boundary probe, but low-sample copy-backed session timings did not justify promotion over no-copy per_iter/batched baselines.",
            "All reported local wins preserve the HF bridge and alignment gates; they should not be described as full Qwen native token/s throughput.",
        ],
    }
    (out / "consolidated_findings.json").write_text(json.dumps(findings, indent=2) + "\n")

    rows = [
        ["Kimi blog case", "End-to-end Qwen3.5-0.8B generation throughput", "~15 → ~193 tokens/s", "External claim; inspiration only"],
        ["Local CLI sidecar", "Full-vocab LM-head artifact measured_total_ms", f"{fmt(cli_bars[0].value)} → {fmt(min(b.value for b in cli_bars))} ms", f"{fmt(cli_speedup)}× lower local sidecar latency"],
        ["Local persistent kernel", "LM-head ms/kernel-repeat", f"{fmt(kernel_bars[0].value)} → {fmt(min(b.value for b in kernel_bars))} ms", f"{fmt(persistent_speedup)}× lower sidecar kernel repeat"],
        ["Local retained session", "Copy-backed retained row-major boundary", f"{fmt(kernel_bars[-1].value)} ms/repeat", "Correct but not promoted; no-copy explicitly rejected"],
    ]

    report = f"""# Kimi-style local benchmark consolidation\n\nSource inspiration: <{KIMI_URL}>. The Kimi post describes a long-running Mac inference optimization case and reports about **15 → 193 tokens/s** for Qwen3.5-0.8B. This report uses that style of critical benchmark storytelling, but all local numbers below come from this repository's artifacts and are **not** the same end-to-end token/s metric.\n\n## Executive takeaways\n\n- Best local one-shot sidecar latency improved from **{fmt(cli_bars[0].value)} ms** to **{fmt(min(b.value for b in cli_bars))} ms** (**{fmt(cli_speedup)}× lower**) across the measured milestones included here.\n- Best persistent LM-head timing improved from **{fmt(kernel_bars[0].value)} ms/repeat** to **{fmt(min(b.value for b in kernel_bars))} ms/repeat** (**{fmt(persistent_speedup)}× lower**).\n- The biggest early win was not a kernel trick: fixture loading fell by **{fmt(findings['headline_metrics']['fixture_load_improvement_pct_g054_to_g056'])}%** from G054 to G056.\n- G086 retained sessions are correctly isolated and copy-backed, but current evidence says **probe, not promote**.\n\n## Critical comparison matrix\n\n{table_md(['Case', 'Metric', 'Reported range', 'Interpretation'], rows)}\n\n## Graphs\n\n![Full-vocab LM-head CLI time by milestone](charts/cli_time_by_milestone.svg)\n\n![Speedup vs G054 CLI baseline](charts/cli_speedup_vs_g054.svg)\n\n![Persistent LM-head kernel timing](charts/persistent_kernel_timing.svg)\n\n![Persistent kernel speedup vs G054](charts/persistent_speedup_vs_g054.svg)\n\n![Fixture-load bottleneck collapse](charts/fixture_load_ms.svg)\n\n## Critical findings\n\n1. **Apples-to-oranges guardrail:** Kimi reports end-to-end generation throughput; this repo currently has a correct HF bridge plus a Metal LM-head logits sidecar. Local charts must be read as sidecar engineering progress.\n2. **Bottleneck moved:** G054 showed fixture loading dominated; G056 direct loading collapsed that cost. Subsequent work improved benchmark fidelity and kernel/command-mode comparisons.\n3. **No-copy remained the strongest measured data-movement path:** strict no-copy and threadgroup/nocopy variants produced the best retained local command-mode evidence while preserving correctness gates.\n4. **Later kernel experiments plateaued:** the report anchors on the G084 regression summary instead of treating every exploratory trial as a promoted benchmark series; current evidence supports regression tracking and integration-boundary work over new speed claims.\n5. **Retained sessions are a boundary enabler, not a speed claim:** G086 proves a safe opt-in retained row-major session API while rejecting retained no-copy, but medium-confidence promotion remains deferred.\n\n## Artifact sources\n\n- `artifacts/benchmarks/g054_instrumented_breakdown.json`\n- `artifacts/benchmarks/g056_direct_loader.json`\n- `artifacts/benchmarks/g057_nocopy.json`\n- `artifacts/benchmarks/g060_matrix_r7_p10.json`\n- `artifacts/benchmarks/g084_regression_summary/summary.json`\n- `artifacts/benchmarks/g086_session/reuse_session_copy_samples2_final.json`\n\nMachine-readable summary: `consolidated_findings.json`.\n"""
    (out / "report.md").write_text(report)
    print(json.dumps({"out": str(out), "charts": findings["charts"], "headline_metrics": findings["headline_metrics"]}, indent=2))


if __name__ == "__main__":
    build()
