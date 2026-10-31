"""Command-line interface for the GPU Render Regression Test Harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import HarnessConfig, PerformanceConfig, ToleranceConfig
from .runner import RegressionHarnessRunner
from .scenes import ALL_SCENES


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=["auto", "moderngl", "pyopengl"],
        default="auto",
        help="Off-screen OpenGL backend to use (default: auto).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Off-screen framebuffer width in pixels (default: 512).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Off-screen framebuffer height in pixels (default: 512).",
    )
    parser.add_argument(
        "--references-dir",
        type=Path,
        default=Path("references"),
        help="Directory containing stored golden reference PNGs (default: references).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where rendered frames, diffs, logs, and HTML reports are saved (default: reports).",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional list of scene IDs (or substrings) to run.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-harness",
        description=(
            "GPU Render Regression Test Harness (ModernGL / PyOpenGL). "
            "Renders off-screen OpenGL test scenes, compares frames against golden "
            "reference images using pixel tolerance and SSIM, profiles frame-time "
            "stability across repeated runs, and emits structured JSONL logs and an HTML report."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    # 1. `run` command
    run_p = subparsers.add_parser(
        "run", help="Run the off-screen OpenGL regression test suite."
    )
    _add_common_arguments(run_p)
    run_p.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warm-up iterations per scene (default: 3).",
    )
    run_p.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of timed iterations per scene for stability profiling (default: 20).",
    )
    run_p.add_argument(
        "--pixel-tolerance",
        type=float,
        default=0.01,
        help="Per-channel normalized pixel difference tolerance in [0, 1] (default: 0.01).",
    )
    run_p.add_argument(
        "--max-diff-ratio",
        type=float,
        default=0.001,
        help="Maximum allowed fraction of pixels exceeding pixel-tolerance (default: 0.001).",
    )
    run_p.add_argument(
        "--min-ssim",
        type=float,
        default=0.990,
        help="Minimum acceptable SSIM similarity score in [-1, 1] (default: 0.990).",
    )
    run_p.add_argument(
        "--frame-budget-ms",
        type=float,
        default=33.33,
        help="Target maximum mean frame time budget in ms (default: 33.33).",
    )
    run_p.add_argument(
        "--inject-regression",
        nargs="*",
        default=[],
        help="Intentionally inject shader/pipeline regressions into specified scene IDs (or 'all').",
    )
    run_p.add_argument(
        "--update-references",
        action="store_true",
        help="Overwrite stored golden reference images with newly rendered frames.",
    )
    run_p.add_argument(
        "--no-auto-reference",
        action="store_true",
        help="Fail instead of automatically generating missing reference images.",
    )
    run_p.add_argument(
        "--save-all-diffs",
        action="store_true",
        help="Save diff heatmaps and 4-panel strips even for passing scenes.",
    )

    # 2. `generate-references` command
    gen_p = subparsers.add_parser(
        "generate-references",
        help="Render all test scenes cleanly and store golden reference images.",
    )
    _add_common_arguments(gen_p)

    # 3. `list-scenes` command
    subparsers.add_parser(
        "list-scenes",
        help="List all registered off-screen OpenGL test scenes and their features.",
    )

    return parser


def _print_run_summary(result) -> None:
    dev = result.device_info
    print("=" * 88)
    print(
        f"GPU RENDER REGRESSION TEST HARNESS | Backend: {dev.get('backend')} | "
        f"GPU: {dev.get('gl_renderer')} ({dev.get('gl_version')})"
    )
    print("=" * 88)
    header = (
        f"{'Scene ID':<32} {'Status':<8} {'SSIM':<9} {'DiffPx%':<9} "
        f"{'Mean(ms)':<9} {'P95(ms)':<9} {'CV':<7} {'Stability'}"
    )
    print(header)
    print("-" * 88)
    for s in result.scenes:
        status_str = "PASS" if s.passed else "FAIL"
        diff_pct = s.comparison.diff_pixel_ratio * 100.0
        print(
            f"{s.scene_id:<32} {status_str:<8} "
            f"{s.comparison.ssim_score:<9.5f} {diff_pct:<8.3f}% "
            f"{s.performance.mean_ms:<9.3f} {s.performance.p95_ms:<9.3f} "
            f"{s.performance.cv:<7.3f} {s.performance.stability_status}"
        )
    print("-" * 88)
    verdict = "PASSED" if result.overall_passed else "FAILED"
    print(
        f"OVERALL: {verdict} ({result.passed_scenes}/{result.total_scenes} scenes passed) | "
        f"Mean SSIM: {result.mean_ssim:.5f} | "
        f"Mean Frame: {result.overall_mean_frame_ms:.3f} ms | "
        f"Duration: {result.total_duration_s:.2f} s"
    )
    print(f"HTML Report : {result.artifacts['html_report']}")
    print(f"JSON Report : {result.artifacts['json_report']}")
    print(f"JUnit XML   : {result.artifacts['junit_xml']}")
    print(f"JSONL Logs  : {result.artifacts['jsonl_log']}")
    print("=" * 88)


def main(argv: Optional[List[str]] = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or args_list[0].startswith("-"):
        args_list = ["run", *args_list]

    parser = build_parser()
    args = parser.parse_args(args_list)

    if args.command == "list-scenes":
        print(f"Registered OpenGL Test Scenes ({len(ALL_SCENES)}):")
        for scene in ALL_SCENES:
            m = scene.metadata
            print(f"  - {m.scene_id:<32} [{m.category}] : {m.title}")
            print(f"      Features: {', '.join(m.features)}")
        return 0

    if args.command == "generate-references":
        cfg = HarnessConfig(
            width=args.width,
            height=args.height,
            backend=args.backend,
            references_dir=args.references_dir,
            output_dir=args.output_dir,
            scene_filter=args.scenes,
        )
        runner = RegressionHarnessRunner(cfg)
        written = runner.generate_references()
        print(f"Generated {len(written)} golden reference image(s) in '{cfg.references_dir}':")
        for p in written:
            print(f"  - {p}")
        return 0

    if args.command == "run":
        tol = ToleranceConfig(
            pixel_tolerance=args.pixel_tolerance,
            max_diff_pixel_ratio=args.max_diff_ratio,
            min_ssim=args.min_ssim,
        )
        perf = PerformanceConfig(
            warmup_iterations=args.warmup,
            timed_iterations=args.iterations,
            frame_budget_ms=args.frame_budget_ms,
        )
        cfg = HarnessConfig(
            width=args.width,
            height=args.height,
            backend=args.backend,
            references_dir=args.references_dir,
            output_dir=args.output_dir,
            scene_filter=args.scenes,
            inject_regressions=args.inject_regression or [],
            update_references=args.update_references,
            generate_missing_references=not args.no_auto_reference,
            save_all_diffs=args.save_all_diffs,
            tolerance=tol,
            performance=perf,
        )
        runner = RegressionHarnessRunner(cfg)
        result = runner.run()
        _print_run_summary(result)
        return 0 if result.overall_passed else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
