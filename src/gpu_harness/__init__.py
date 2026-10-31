"""GPU Render Regression Test Harness.

Off-screen OpenGL rendering test suite with reference image comparison (tolerance + SSIM),
frame-time performance and stability profiling, structured logging, and HTML reporting.
"""

__version__ = "1.0.0"

from .config import HarnessConfig, PerformanceConfig, ToleranceConfig
from .runner import HarnessRunResult, RegressionHarnessRunner, SceneRunResult

__all__ = [
    "__version__",
    "HarnessConfig",
    "ToleranceConfig",
    "PerformanceConfig",
    "RegressionHarnessRunner",
    "HarnessRunResult",
    "SceneRunResult",
]
