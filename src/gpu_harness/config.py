"""Configuration dataclasses for the GPU Render Regression Test Harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ToleranceConfig:
    """Image comparison thresholds for pixel-level tolerance and SSIM similarity."""

    # Per-channel absolute pixel tolerance in [0.0, 1.0] (e.g., 2/255 ~= 0.0078)
    pixel_tolerance: float = 0.01
    # Maximum fraction of pixels allowed to exceed pixel_tolerance (e.g., 0.001 = 0.1%)
    max_diff_pixel_ratio: float = 0.001
    # Minimum acceptable Structural Similarity Index Measure (SSIM) in [-1.0, 1.0]
    min_ssim: float = 0.990
    # Amplification factor applied when rendering visual diff heatmaps
    diff_amplification: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceConfig:
    """Frame-time recording and stability evaluation settings."""

    # Number of warm-up renders per scene (excluded from steady-state statistics)
    warmup_iterations: int = 3
    # Number of timed render iterations per scene
    timed_iterations: int = 20
    # Maximum expected mean frame time in milliseconds before flagging a budget warning
    frame_budget_ms: float = 33.33
    # Maximum Coefficient of Variation (std / mean) before flagging frame-time jitter warning
    max_cv_threshold: float = 0.50
    # Verify that repeated iterations of the same scene produce identical pixels
    check_frame_determinism: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HarnessConfig:
    """Top-level configuration for a regression test harness run."""

    width: int = 512
    height: int = 512
    backend: str = "auto"  # "auto", "moderngl", "pyopengl"
    references_dir: Path = field(default_factory=lambda: Path("references"))
    output_dir: Path = field(default_factory=lambda: Path("reports"))
    scene_filter: Optional[List[str]] = None
    inject_regressions: List[str] = field(default_factory=list)
    update_references: bool = False
    generate_missing_references: bool = True
    save_all_diffs: bool = False
    fail_on_perf_regression: bool = False
    tolerance: ToleranceConfig = field(default_factory=ToleranceConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    def __post_init__(self) -> None:
        self.references_dir = Path(self.references_dir)
        self.output_dir = Path(self.output_dir)

    @property
    def rendered_dir(self) -> Path:
        return self.output_dir / "rendered"

    @property
    def diffs_dir(self) -> Path:
        return self.output_dir / "diffs"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def jsonl_log_path(self) -> Path:
        return self.logs_dir / "test_events.jsonl"

    @property
    def json_report_path(self) -> Path:
        return self.output_dir / "results.json"

    @property
    def junit_xml_path(self) -> Path:
        return self.output_dir / "junit.xml"

    @property
    def html_report_path(self) -> Path:
        return self.output_dir / "index.html"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "backend": self.backend,
            "references_dir": str(self.references_dir),
            "output_dir": str(self.output_dir),
            "scene_filter": self.scene_filter,
            "inject_regressions": self.inject_regressions,
            "update_references": self.update_references,
            "generate_missing_references": self.generate_missing_references,
            "save_all_diffs": self.save_all_diffs,
            "fail_on_perf_regression": self.fail_on_perf_regression,
            "tolerance": self.tolerance.to_dict(),
            "performance": self.performance.to_dict(),
        }
