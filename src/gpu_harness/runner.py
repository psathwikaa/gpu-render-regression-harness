"""Core test harness runner coordinating off-screen rendering, profiling, comparison, and reporting."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .backends import create_backend
from .comparator import (
    ComparisonResult,
    compare_frames,
    load_rgba_image,
    save_rgba_image,
)
from .config import HarnessConfig
from .logger import StructuredEventLogger, write_json_report, write_junit_xml
from .profiler import FrameStabilityStats, analyze_frame_times
from .reporter import generate_html_report
from .scenes import TestScene, get_scenes


@dataclass
class SceneRunResult:
    """Complete execution result for a single OpenGL test scene."""

    scene_id: str
    title: str
    category: str
    description: str
    features: List[str]
    regression_description: str
    regression_injected: bool
    passed: bool
    reference_image_path: str
    rendered_image_path: str
    comparison: ComparisonResult
    performance: FrameStabilityStats

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HarnessRunResult:
    """Aggregate result of running the GPU Render Regression Test Harness."""

    overall_passed: bool
    total_scenes: int
    passed_scenes: int
    failed_scenes: int
    pass_rate: float
    mean_ssim: float
    overall_mean_frame_ms: float
    total_duration_s: float
    started_at: str
    completed_at: str
    device_info: Dict[str, Any]
    config: Dict[str, Any]
    scenes: List[SceneRunResult]
    artifacts: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_passed": self.overall_passed,
            "total_scenes": self.total_scenes,
            "passed_scenes": self.passed_scenes,
            "failed_scenes": self.failed_scenes,
            "pass_rate": round(self.pass_rate, 4),
            "mean_ssim": round(self.mean_ssim, 6),
            "overall_mean_frame_ms": round(self.overall_mean_frame_ms, 4),
            "total_duration_s": round(self.total_duration_s, 4),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "device_info": self.device_info,
            "config": self.config,
            "scenes": [s.to_dict() for s in self.scenes],
            "artifacts": self.artifacts,
        }


class RegressionHarnessRunner:
    """Executes the off-screen OpenGL regression test suite."""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config

    def _should_inject_regression(self, scene_id: str) -> bool:
        targets = [t.strip().lower() for t in self.config.inject_regressions if t.strip()]
        if not targets:
            return False
        if "all" in targets:
            return True
        sid = scene_id.lower()
        return any(t == sid or t in sid for t in targets)

    def generate_references(self) -> List[Path]:
        """Render all selected scenes cleanly and store golden reference PNGs."""
        self.config.references_dir.mkdir(parents=True, exist_ok=True)
        scenes = get_scenes(self.config.scene_filter)
        backend = create_backend(
            self.config.backend, self.config.width, self.config.height
        )
        written: List[Path] = []
        try:
            for scene in scenes:
                scene.render(backend, inject_regression=False)
                backend.finish()
                frame = backend.read_pixels()
                ref_path = self.config.references_dir / f"{scene.metadata.scene_id}.png"
                save_rgba_image(frame, ref_path)
                written.append(ref_path)
        finally:
            backend.release()
        return written

    def run(self) -> HarnessRunResult:
        """Execute all configured scenes, compare against references, profile frame times, and emit reports."""
        suite_start_ns = time.perf_counter_ns()
        started_at = datetime.now(timezone.utc).isoformat()

        self.config.references_dir.mkdir(parents=True, exist_ok=True)
        self.config.rendered_dir.mkdir(parents=True, exist_ok=True)
        self.config.diffs_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

        event_logger = StructuredEventLogger(self.config.jsonl_log_path)
        event_logger.log_event(
            "harness_start",
            level="INFO",
            message="Starting GPU Render Regression Test Harness",
            config=self.config.to_dict(),
        )

        scenes: List[TestScene] = get_scenes(self.config.scene_filter)
        backend = create_backend(
            self.config.backend, self.config.width, self.config.height
        )

        try:
            device_info = backend.get_device_info()
            event_logger.log_event(
                "gpu_context_init",
                level="INFO",
                message=f"Initialized off-screen OpenGL context via {device_info.get('backend')}",
                **device_info,
            )

            scene_results: List[SceneRunResult] = []

            for scene in scenes:
                meta = scene.metadata
                sid = meta.scene_id
                inject_reg = self._should_inject_regression(sid)

                ref_path = self.config.references_dir / f"{sid}.png"
                ren_path = self.config.rendered_dir / f"{sid}.png"
                diff_path = self.config.diffs_dir / f"{sid}_diff.png"
                panel_path = self.config.diffs_dir / f"{sid}_panel.png"

                # Ensure golden reference exists if auto-generation or update is enabled
                if self.config.update_references or (
                    self.config.generate_missing_references and not ref_path.is_file()
                ):
                    scene.render(backend, inject_regression=False)
                    backend.finish()
                    golden_frame = backend.read_pixels()
                    save_rgba_image(golden_frame, ref_path)
                    event_logger.log_event(
                        "reference_generated",
                        level="INFO",
                        scene_id=sid,
                        message=f"Saved golden reference image to {ref_path}",
                        reference_path=str(ref_path),
                    )

                event_logger.log_event(
                    "scene_start",
                    level="INFO",
                    scene_id=sid,
                    message=f"Rendering scene '{meta.title}'",
                    category=meta.category,
                    inject_regression=inject_reg,
                    warmup_iterations=self.config.performance.warmup_iterations,
                    timed_iterations=self.config.performance.timed_iterations,
                )

                # 1. Warm-up iterations
                warmup_times_ms: List[float] = []
                for _ in range(self.config.performance.warmup_iterations):
                    t0 = time.perf_counter_ns()
                    scene.render(backend, inject_regression=inject_reg)
                    backend.finish()
                    t1 = time.perf_counter_ns()
                    warmup_times_ms.append((t1 - t0) / 1_000_000.0)

                # 2. Timed iterations for performance & stability analysis
                frame_times_ms: List[float] = []
                first_timed_frame: np.ndarray | None = None
                last_timed_frame: np.ndarray | None = None

                n_timed = max(1, self.config.performance.timed_iterations)
                for idx in range(n_timed):
                    t0 = time.perf_counter_ns()
                    scene.render(backend, inject_regression=inject_reg)
                    backend.finish()
                    t1 = time.perf_counter_ns()
                    elapsed_ms = (t1 - t0) / 1_000_000.0
                    frame_times_ms.append(elapsed_ms)

                    if idx == 0 or idx == n_timed - 1:
                        snap = backend.read_pixels()
                        if idx == 0:
                            first_timed_frame = snap
                        if idx == n_timed - 1:
                            last_timed_frame = snap

                assert last_timed_frame is not None
                assert first_timed_frame is not None

                max_interframe_diff = int(
                    np.max(
                        np.abs(
                            last_timed_frame.astype(np.int32)
                            - first_timed_frame.astype(np.int32)
                        )
                    )
                )

                perf_stats = analyze_frame_times(
                    warmup_times_ms=warmup_times_ms,
                    frame_times_ms=frame_times_ms,
                    perf_cfg=self.config.performance,
                    max_interframe_diff_u8=max_interframe_diff,
                )

                event_logger.log_event(
                    "scene_performance",
                    level="INFO" if perf_stats.stability_status == "STABLE" else "WARN",
                    scene_id=sid,
                    message=f"Scene '{sid}' mean={perf_stats.mean_ms:.3f}ms, P95={perf_stats.p95_ms:.3f}ms, CV={perf_stats.cv:.3f} ({perf_stats.stability_status})",
                    mean_ms=perf_stats.mean_ms,
                    median_ms=perf_stats.median_ms,
                    p95_ms=perf_stats.p95_ms,
                    std_ms=perf_stats.std_ms,
                    cv=perf_stats.cv,
                    stability_status=perf_stats.stability_status,
                    deterministic_frames=perf_stats.deterministic_frames,
                )

                # Save rendered image
                save_rgba_image(last_timed_frame, ren_path)

                # 3. Compare with stored reference image
                if not ref_path.is_file():
                    raise FileNotFoundError(
                        f"Missing reference image for scene '{sid}' at {ref_path}. "
                        f"Run with --generate-references or enable generate_missing_references."
                    )

                ref_frame = load_rgba_image(ref_path)
                comp_result = compare_frames(
                    img_ref=ref_frame,
                    img_ren=last_timed_frame,
                    tolerance=self.config.tolerance,
                    diff_output_path=diff_path,
                    panel_output_path=panel_path,
                    save_diff_on_pass=self.config.save_all_diffs,
                )

                scene_passed = comp_result.passed
                if (
                    self.config.fail_on_perf_regression
                    and not perf_stats.within_budget
                ):
                    scene_passed = False
                    comp_result.failure_reasons.append(
                        f"Mean frame time {perf_stats.mean_ms:.3f}ms exceeded budget {perf_stats.frame_budget_ms:.3f}ms."
                    )

                event_logger.log_event(
                    "scene_comparison",
                    level="INFO" if scene_passed else "ERROR",
                    scene_id=sid,
                    message=(
                        f"Scene '{sid}' {'PASSED' if scene_passed else 'FAILED'} "
                        f"(SSIM={comp_result.ssim_score:.5f}, diff_px={comp_result.diff_pixel_ratio * 100:.3f}%)"
                    ),
                    passed=scene_passed,
                    ssim_score=comp_result.ssim_score,
                    psnr_db=comp_result.psnr_db,
                    diff_pixel_count=comp_result.diff_pixel_count,
                    diff_pixel_ratio=comp_result.diff_pixel_ratio,
                    max_channel_error_u8=comp_result.max_channel_error_u8,
                    diff_image_path=comp_result.diff_image_path,
                    failure_reasons=comp_result.failure_reasons,
                )

                scene_results.append(
                    SceneRunResult(
                        scene_id=sid,
                        title=meta.title,
                        category=meta.category,
                        description=meta.description,
                        features=list(meta.features),
                        regression_description=meta.regression_description,
                        regression_injected=inject_reg,
                        passed=scene_passed,
                        reference_image_path=str(ref_path),
                        rendered_image_path=str(ren_path),
                        comparison=comp_result,
                        performance=perf_stats,
                    )
                )
        finally:
            backend.release()

        suite_end_ns = time.perf_counter_ns()
        completed_at = datetime.now(timezone.utc).isoformat()
        total_duration_s = (suite_end_ns - suite_start_ns) / 1_000_000_000.0

        total_scenes = len(scene_results)
        passed_scenes = sum(1 for s in scene_results if s.passed)
        failed_scenes = total_scenes - passed_scenes
        overall_passed = failed_scenes == 0 and total_scenes > 0
        pass_rate = (passed_scenes / float(total_scenes)) if total_scenes > 0 else 0.0
        mean_ssim = (
            float(np.mean([s.comparison.ssim_score for s in scene_results]))
            if scene_results
            else 0.0
        )
        overall_mean_frame_ms = (
            float(np.mean([s.performance.mean_ms for s in scene_results]))
            if scene_results
            else 0.0
        )

        artifacts = {
            "html_report": str(self.config.html_report_path),
            "json_report": str(self.config.json_report_path),
            "junit_xml": str(self.config.junit_xml_path),
            "jsonl_log": str(self.config.jsonl_log_path),
            "rendered_dir": str(self.config.rendered_dir),
            "diffs_dir": str(self.config.diffs_dir),
        }

        run_result = HarnessRunResult(
            overall_passed=overall_passed,
            total_scenes=total_scenes,
            passed_scenes=passed_scenes,
            failed_scenes=failed_scenes,
            pass_rate=pass_rate,
            mean_ssim=mean_ssim,
            overall_mean_frame_ms=overall_mean_frame_ms,
            total_duration_s=total_duration_s,
            started_at=started_at,
            completed_at=completed_at,
            device_info=device_info,
            config=self.config.to_dict(),
            scenes=scene_results,
            artifacts=artifacts,
        )

        event_logger.log_event(
            "harness_complete",
            level="INFO" if overall_passed else "ERROR",
            message=(
                f"Harness complete: {passed_scenes}/{total_scenes} passed "
                f"(mean SSIM={mean_ssim:.5f}, duration={total_duration_s:.2f}s)"
            ),
            overall_passed=overall_passed,
            passed_scenes=passed_scenes,
            failed_scenes=failed_scenes,
            mean_ssim=round(mean_ssim, 6),
            overall_mean_frame_ms=round(overall_mean_frame_ms, 4),
            total_duration_s=round(total_duration_s, 4),
            artifacts=artifacts,
        )

        report_dict = run_result.to_dict()
        write_json_report(report_dict, self.config.json_report_path)
        write_junit_xml(report_dict, self.config.junit_xml_path)
        generate_html_report(
            report_dict,
            self.config.html_report_path,
            jsonl_log_path=self.config.jsonl_log_path,
        )

        return run_result
