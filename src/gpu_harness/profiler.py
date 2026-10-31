"""Frame-time performance profiler and multi-run stability analyzer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np

from .config import PerformanceConfig


@dataclass
class FrameStabilityStats:
    """Statistical summary of frame times and stability across repeated runs of a scene."""

    warmup_iterations: int
    timed_iterations: int
    warmup_times_ms: List[float]
    frame_times_ms: List[float]
    mean_ms: float
    median_ms: float
    std_ms: float
    cv: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    iqr_ms: float
    mean_fps: float
    frame_budget_ms: float
    max_cv_threshold: float
    within_budget: bool
    cv_stable: bool
    deterministic_frames: bool
    max_interframe_diff_u8: int
    stability_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_frame_times(
    warmup_times_ms: List[float],
    frame_times_ms: List[float],
    perf_cfg: PerformanceConfig,
    max_interframe_diff_u8: int = 0,
) -> FrameStabilityStats:
    """Compute statistical performance and stability metrics from repeated frame timings."""
    if not frame_times_ms:
        raise ValueError("frame_times_ms must contain at least one timed iteration.")

    arr = np.array(frame_times_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    median_ms = float(np.median(arr))
    std_ms = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    cv = float(std_ms / mean_ms) if mean_ms > 1e-9 else 0.0
    min_ms = float(np.min(arr))
    max_ms = float(np.max(arr))
    p95_ms = float(np.percentile(arr, 95))
    p99_ms = float(np.percentile(arr, 99))
    p75, p25 = np.percentile(arr, [75, 25])
    iqr_ms = float(p75 - p25)
    mean_fps = float(1000.0 / mean_ms) if mean_ms > 1e-9 else 0.0

    within_budget = mean_ms <= perf_cfg.frame_budget_ms
    robust_cv = (iqr_ms / median_ms) if median_ms > 1e-9 else 0.0
    cv_stable = (cv <= perf_cfg.max_cv_threshold) or (
        robust_cv <= perf_cfg.max_cv_threshold and iqr_ms <= 0.5
    )
    deterministic_frames = max_interframe_diff_u8 == 0

    if not within_budget:
        stability_status = "BUDGET_EXCEEDED"
    elif not deterministic_frames:
        stability_status = "NON_DETERMINISTIC"
    elif not cv_stable:
        stability_status = "JITTER_WARNING"
    else:
        stability_status = "STABLE"

    return FrameStabilityStats(
        warmup_iterations=len(warmup_times_ms),
        timed_iterations=len(frame_times_ms),
        warmup_times_ms=[round(t, 4) for t in warmup_times_ms],
        frame_times_ms=[round(t, 4) for t in frame_times_ms],
        mean_ms=round(mean_ms, 4),
        median_ms=round(median_ms, 4),
        std_ms=round(std_ms, 4),
        cv=round(cv, 4),
        min_ms=round(min_ms, 4),
        max_ms=round(max_ms, 4),
        p95_ms=round(p95_ms, 4),
        p99_ms=round(p99_ms, 4),
        iqr_ms=round(iqr_ms, 4),
        mean_fps=round(mean_fps, 2),
        frame_budget_ms=perf_cfg.frame_budget_ms,
        max_cv_threshold=perf_cfg.max_cv_threshold,
        within_budget=within_budget,
        cv_stable=cv_stable,
        deterministic_frames=deterministic_frames,
        max_interframe_diff_u8=int(max_interframe_diff_u8),
        stability_status=stability_status,
    )
