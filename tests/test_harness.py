"""Unit and integration tests for the GPU Render Regression Test Harness."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from gpu_harness.backends import ModernGLBackend, PyOpenGLBackend, create_backend
from gpu_harness.comparator import (
    compare_frames,
    compute_ssim,
    load_rgba_image,
    save_rgba_image,
)
from gpu_harness.config import HarnessConfig, PerformanceConfig, ToleranceConfig
from gpu_harness.cli import main as cli_main
from gpu_harness.profiler import analyze_frame_times
from gpu_harness.runner import RegressionHarnessRunner
from gpu_harness.scenes import ALL_SCENES, get_scenes


def test_scene_registry_has_8_distinct_scenes() -> None:
    """Verify 8 OpenGL scenes are registered covering triangles, textures, blending, depth, instancing, lighting."""
    assert len(ALL_SCENES) == 8
    ids = [s.metadata.scene_id for s in ALL_SCENES]
    assert len(set(ids)) == 8
    assert "01_rgb_triangle" in ids
    assert "03_checker_texture_2d" in ids
    assert "04_alpha_blending_glass" in ids
    assert "05_depth_test_intersection" in ids
    assert "08_phong_shaded_sphere" in ids

    filtered = get_scenes(["depth"])
    assert len(filtered) == 2


def test_moderngl_and_pyopengl_cross_backend_parity() -> None:
    """Verify ModernGL and PyOpenGL backends render bit-identical / SSIM>=0.999 frames."""
    frames_mgl = {}
    mgl = ModernGLBackend(128, 128)
    try:
        assert mgl.backend_name == "moderngl"
        assert "gl_renderer" in mgl.get_device_info()
        for scene in ALL_SCENES:
            scene.render(mgl, inject_regression=False)
            mgl.finish()
            frames_mgl[scene.metadata.scene_id] = mgl.read_pixels()
    finally:
        mgl.release()

    frames_pgl = {}
    pgl = PyOpenGLBackend(128, 128)
    try:
        assert pgl.backend_name == "pyopengl"
        assert "gl_renderer" in pgl.get_device_info()
        for scene in ALL_SCENES:
            scene.render(pgl, inject_regression=False)
            pgl.finish()
            frames_pgl[scene.metadata.scene_id] = pgl.read_pixels()
    finally:
        pgl.release()

    for scene in ALL_SCENES:
        sid = scene.metadata.scene_id
        frame_mgl = frames_mgl[sid]
        frame_pgl = frames_pgl[sid]
        assert frame_mgl.shape == (128, 128, 4)
        assert frame_pgl.shape == (128, 128, 4)
        assert float(np.std(frame_mgl[..., :3])) > 10.0

        ssim_val, _ = compute_ssim(frame_mgl, frame_pgl)
        assert ssim_val >= 0.999, f"Cross-backend mismatch on {sid}: SSIM={ssim_val}"


def test_comparator_tolerance_ssim_and_diff_saving(tmp_path: Path) -> None:
    """Verify pixel tolerance, SSIM calculation, and diff image creation when comparison fails."""
    rng = np.random.default_rng(42)
    ref = rng.integers(30, 220, size=(64, 64, 4), dtype=np.uint8)
    ref[..., 3] = 255

    # 1. Identical frame passes with SSIM=1.0 and does not save diff by default
    tol = ToleranceConfig(pixel_tolerance=0.02, max_diff_pixel_ratio=0.005, min_ssim=0.99)
    diff_file = tmp_path / "identical_diff.png"
    res_ident = compare_frames(ref, ref.copy(), tol, diff_output_path=diff_file)
    assert res_ident.passed is True
    assert res_ident.ssim_score == pytest.approx(1.0, abs=1e-5)
    assert res_ident.diff_pixel_count == 0
    assert not diff_file.exists()

    # 2. Minor 1-unit (1/255 ~= 0.0039) perturbation within pixel_tolerance=0.02 passes
    slight = ref.copy()
    slight[:10, :10, 0] = np.clip(slight[:10, :10, 0].astype(int) + 1, 0, 255).astype(np.uint8)
    res_slight = compare_frames(ref, slight, tol)
    assert res_slight.passed is True
    assert res_slight.tolerance_passed is True
    assert res_slight.ssim_passed is True

    # 3. Large structural patch corruption fails both tolerance and SSIM and saves diff + panel PNGs
    corrupted = ref.copy()
    corrupted[16:48, 16:48, :3] = 255 - corrupted[16:48, 16:48, :3]
    fail_diff = tmp_path / "corrupted_diff.png"
    fail_panel = tmp_path / "corrupted_panel.png"
    res_fail = compare_frames(
        ref,
        corrupted,
        tol,
        diff_output_path=fail_diff,
        panel_output_path=fail_panel,
    )
    assert res_fail.passed is False
    assert res_fail.tolerance_passed is False
    assert res_fail.ssim_passed is False
    assert res_fail.ssim_score < 0.85
    assert fail_diff.is_file()
    assert fail_panel.is_file()
    loaded_diff = load_rgba_image(fail_diff)
    assert loaded_diff.shape == (64, 64, 4)


def test_frame_time_profiler_stability_classifications() -> None:
    """Verify frame-time statistics and stability status classifications."""
    perf_cfg = PerformanceConfig(
        warmup_iterations=2,
        timed_iterations=10,
        frame_budget_ms=16.67,
        max_cv_threshold=0.25,
    )

    # Stable run
    stable = analyze_frame_times(
        warmup_times_ms=[4.2, 1.1],
        frame_times_ms=[1.0, 1.05, 0.98, 1.02, 1.01, 0.99, 1.03, 1.00, 0.97, 1.04],
        perf_cfg=perf_cfg,
        max_interframe_diff_u8=0,
    )
    assert stable.stability_status == "STABLE"
    assert stable.within_budget is True
    assert stable.cv_stable is True
    assert stable.deterministic_frames is True
    assert stable.mean_fps > 900.0

    # High jitter run (CV > 0.25)
    jitter = analyze_frame_times(
        warmup_times_ms=[2.0],
        frame_times_ms=[0.5, 3.5, 0.6, 4.0, 0.5, 3.8],
        perf_cfg=perf_cfg,
        max_interframe_diff_u8=0,
    )
    assert jitter.stability_status == "JITTER_WARNING"
    assert jitter.cv_stable is False

    # Budget exceeded run
    over_budget = analyze_frame_times(
        warmup_times_ms=[25.0],
        frame_times_ms=[22.0, 23.0, 21.5, 22.5],
        perf_cfg=perf_cfg,
        max_interframe_diff_u8=0,
    )
    assert over_budget.stability_status == "BUDGET_EXCEEDED"
    assert over_budget.within_budget is False


def test_end_to_end_harness_pass_and_regression_detection(tmp_path: Path) -> None:
    """Run full harness pipeline: golden pass followed by injected regression failure."""
    ref_dir = tmp_path / "references"
    out_pass = tmp_path / "reports_pass"
    out_fail = tmp_path / "reports_fail"

    # 1. Passing run (auto-generates missing references, runs 3 timed iterations)
    rc_pass = cli_main(
        [
            "run",
            "--width",
            "128",
            "--height",
            "128",
            "--warmup",
            "1",
            "--iterations",
            "3",
            "--references-dir",
            str(ref_dir),
            "--output-dir",
            str(out_pass),
        ]
    )
    assert rc_pass == 0
    assert (out_pass / "index.html").is_file()
    assert (out_pass / "results.json").is_file()
    assert (out_pass / "junit.xml").is_file()
    assert (out_pass / "logs" / "test_events.jsonl").is_file()

    summary_pass = json.loads((out_pass / "results.json").read_text(encoding="utf-8"))
    assert summary_pass["overall_passed"] is True
    assert summary_pass["passed_scenes"] == 8
    assert summary_pass["mean_ssim"] == pytest.approx(1.0, abs=1e-5)

    # Verify JSONL structured log lines
    log_lines = (out_pass / "logs" / "test_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_lines) >= 20
    first_evt = json.loads(log_lines[0])
    assert first_evt["event_type"] == "harness_start"

    # 2. Failing run with injected regression on 2 scenes
    rc_fail = cli_main(
        [
            "run",
            "--width",
            "128",
            "--height",
            "128",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--references-dir",
            str(ref_dir),
            "--output-dir",
            str(out_fail),
            "--inject-regression",
            "01_rgb_triangle",
            "05_depth_test_intersection",
        ]
    )
    assert rc_fail == 1
    summary_fail = json.loads((out_fail / "results.json").read_text(encoding="utf-8"))
    assert summary_fail["overall_passed"] is False
    assert summary_fail["failed_scenes"] == 2
    assert (out_fail / "diffs" / "01_rgb_triangle_diff.png").is_file()
    assert (out_fail / "diffs" / "05_depth_test_intersection_diff.png").is_file()

    # Verify JUnit XML contains 2 <failure> elements
    tree = ET.parse(out_fail / "junit.xml")
    root = tree.getroot()
    assert root.attrib["tests"] == "8"
    assert root.attrib["failures"] == "2"
    failures = root.findall(".//failure")
    assert len(failures) == 2

    # 3. Verify CLI `list-scenes` and `generate-references` subcommands
    assert cli_main(["list-scenes"]) == 0
    gen_ref_dir = tmp_path / "gen_refs"
    assert (
        cli_main(
            [
                "generate-references",
                "--width",
                "64",
                "--height",
                "64",
                "--scenes",
                "01_rgb_triangle",
                "--references-dir",
                str(gen_ref_dir),
            ]
        )
        == 0
    )
    assert (gen_ref_dir / "01_rgb_triangle.png").is_file()
