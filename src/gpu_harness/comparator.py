"""Image comparison engine with pixel-level tolerance, SSIM similarity, and diff visualization."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from .config import ToleranceConfig


@dataclass
class ComparisonResult:
    """Detailed metrics from comparing a rendered frame against a stored reference image."""

    passed: bool
    tolerance_passed: bool
    ssim_passed: bool
    ssim_score: float
    psnr_db: float
    mae: float
    rmse: float
    max_abs_error: float
    max_channel_error_u8: int
    diff_pixel_count: int
    total_pixels: int
    diff_pixel_ratio: float
    pixel_tolerance: float
    max_diff_pixel_ratio_threshold: float
    min_ssim_threshold: float
    failure_reasons: list[str]
    diff_image_path: Optional[str] = None
    panel_image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_rgba_image(path: Path) -> np.ndarray:
    """Load an image file from disk as an (H, W, 4) uint8 RGBA array."""
    with Image.open(path) as img:
        rgba = img.convert("RGBA")
        return np.array(rgba, dtype=np.uint8)


def save_rgba_image(arr: np.ndarray, path: Path) -> None:
    """Save an (H, W, 4) or (H, W, 3) uint8 array to disk as a PNG image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8))
    img.save(path, format="PNG")


def compute_ssim(
    img_ref: np.ndarray,
    img_ren: np.ndarray,
    sigma: float = 1.5,
) -> Tuple[float, np.ndarray]:
    """Compute Structural Similarity Index Measure (SSIM) between two RGBA/RGB uint8 images.

    Implements the Gaussian-windowed SSIM formulation of Wang et al. (2004) across
    RGB channels in [0, 1] floating point space.

    Returns:
        Tuple of (mean_ssim_score, ssim_map_2d) where ssim_map_2d has shape (H, W).
    """
    if img_ref.shape != img_ren.shape:
        raise ValueError(
            f"Image dimension mismatch: reference={img_ref.shape} vs rendered={img_ren.shape}"
        )

    # Use RGB channels normalized to [0.0, 1.0]
    x = img_ref[..., :3].astype(np.float64) / 255.0
    y = img_ren[..., :3].astype(np.float64) / 255.0

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    channel_maps = []
    for ch in range(3):
        xc = x[..., ch]
        yc = y[..., ch]

        mu_x = gaussian_filter(xc, sigma=sigma, mode="reflect")
        mu_y = gaussian_filter(yc, sigma=sigma, mode="reflect")

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = gaussian_filter(xc * xc, sigma=sigma, mode="reflect") - mu_x_sq
        sigma_y_sq = gaussian_filter(yc * yc, sigma=sigma, mode="reflect") - mu_y_sq
        sigma_xy = gaussian_filter(xc * yc, sigma=sigma, mode="reflect") - mu_xy

        # Guard against small negative values from floating-point rounding
        sigma_x_sq = np.maximum(sigma_x_sq, 0.0)
        sigma_y_sq = np.maximum(sigma_y_sq, 0.0)

        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_ch = numerator / denominator
        channel_maps.append(ssim_ch)

    ssim_map = np.mean(np.stack(channel_maps, axis=-1), axis=-1)
    mean_ssim = float(np.clip(np.mean(ssim_map), -1.0, 1.0))
    return mean_ssim, ssim_map.astype(np.float32)


def _colorize_heatmap(intensity_01: np.ndarray) -> np.ndarray:
    """Map a 2D [0, 1] error intensity array to an Inferno/Magma-like RGBA uint8 heatmap."""
    val = np.clip(intensity_01, 0.0, 1.0).astype(np.float32)
    # Piecewise control stops:
    # 0.00 -> dark slate (12, 14, 24)
    # 0.25 -> deep violet (95, 25, 135)
    # 0.55 -> crimson orange (225, 55, 45)
    # 0.85 -> bright amber (252, 185, 35)
    # 1.00 -> hot white-yellow (255, 252, 215)
    stops_x = np.array([0.0, 0.25, 0.55, 0.85, 1.0], dtype=np.float32)
    stops_r = np.array([12.0, 95.0, 225.0, 252.0, 255.0], dtype=np.float32)
    stops_g = np.array([14.0, 25.0, 55.0, 185.0, 252.0], dtype=np.float32)
    stops_b = np.array([24.0, 135.0, 45.0, 35.0, 215.0], dtype=np.float32)

    r = np.interp(val, stops_x, stops_r)
    g = np.interp(val, stops_x, stops_g)
    b = np.interp(val, stops_x, stops_b)

    out = np.zeros((*val.shape, 4), dtype=np.uint8)
    out[..., 0] = np.clip(r, 0, 255).astype(np.uint8)
    out[..., 1] = np.clip(g, 0, 255).astype(np.uint8)
    out[..., 2] = np.clip(b, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def build_diff_heatmap(
    img_ref: np.ndarray,
    img_ren: np.ndarray,
    amplification: float = 10.0,
) -> np.ndarray:
    """Create an amplified pixel-difference heatmap blended with subtle reference luminance context."""
    ref_f = img_ref[..., :3].astype(np.float32) / 255.0
    ren_f = img_ren[..., :3].astype(np.float32) / 255.0

    # Per-pixel max channel error in [0, 1]
    err_max = np.max(np.abs(ren_f - ref_f), axis=-1)
    amplified = np.clip(err_max * amplification, 0.0, 1.0)
    heatmap = _colorize_heatmap(amplified)

    # Blend subtle grayscale reference luminance where error is near zero so geometry is visible
    luma = (
        0.299 * ref_f[..., 0] + 0.587 * ref_f[..., 1] + 0.114 * ref_f[..., 2]
    ) * 255.0
    ghost_rgb = np.stack([luma * 0.25 + 10.0] * 3, axis=-1)
    err_mask = np.clip(err_max * (amplification * 2.0), 0.0, 1.0)[..., np.newaxis]

    blended_rgb = heatmap[..., :3].astype(np.float32) * err_mask + ghost_rgb * (
        1.0 - err_mask
    )
    out = np.zeros_like(heatmap)
    out[..., :3] = np.clip(blended_rgb, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def build_comparison_panel(
    img_ref: np.ndarray,
    img_ren: np.ndarray,
    diff_heatmap: np.ndarray,
    ssim_map: np.ndarray,
) -> np.ndarray:
    """Create a 4-panel horizontal diagnostic composite:
    [Reference | Rendered | Amplified Pixel Diff | Structural Error (1 - SSIM)].
    """
    h, w, _ = img_ref.shape
    header_h = 28
    pad = 6
    total_w = w * 4 + pad * 5
    total_h = h + header_h + pad * 2

    canvas = Image.new("RGBA", (total_w, total_h), (15, 18, 26, 255))
    draw = ImageDraw.Draw(canvas)

    # Convert (1 - ssim_map) into structural error heatmap
    struct_err = np.clip((1.0 - ssim_map) * 5.0, 0.0, 1.0)
    ssim_heat = _colorize_heatmap(struct_err)

    panels = [
        ("REFERENCE (GOLDEN)", img_ref),
        ("RENDERED (ACTUAL)", img_ren),
        ("PIXEL DIFF (AMPLIFIED)", diff_heatmap),
        ("STRUCTURAL ERROR (1-SSIM)", ssim_heat),
    ]

    for idx, (label, arr) in enumerate(panels):
        x0 = pad + idx * (w + pad)
        y0 = header_h + pad
        draw.text((x0 + 6, 7), label, fill=(215, 222, 238, 255))
        sub_img = Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8))
        canvas.paste(sub_img, (x0, y0))
        draw.rectangle(
            [x0 - 1, y0 - 1, x0 + w, y0 + h],
            outline=(58, 68, 92, 255),
            width=1,
        )

    return np.array(canvas, dtype=np.uint8)


def compare_frames(
    img_ref: np.ndarray,
    img_ren: np.ndarray,
    tolerance: ToleranceConfig,
    diff_output_path: Optional[Path] = None,
    panel_output_path: Optional[Path] = None,
    save_diff_on_pass: bool = False,
) -> ComparisonResult:
    """Compare a rendered RGBA frame with a reference RGBA image using tolerance and SSIM.

    Saves a diff heatmap image (and optional 4-panel composite) whenever the comparison
    fails or when `save_diff_on_pass=True`.
    """
    if img_ref.shape != img_ren.shape:
        raise ValueError(
            f"Frame dimensions {img_ren.shape} do not match reference {img_ref.shape}"
        )

    ref_f = img_ref.astype(np.float64) / 255.0
    ren_f = img_ren.astype(np.float64) / 255.0

    abs_diff = np.abs(ren_f - ref_f)
    mae = float(np.mean(abs_diff))
    mse = float(np.mean(abs_diff * abs_diff))
    rmse = math.sqrt(mse)
    psnr_db = 100.0 if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))

    max_abs_error = float(np.max(abs_diff))
    max_channel_error_u8 = int(
        np.max(np.abs(img_ren.astype(np.int32) - img_ref.astype(np.int32)))
    )

    # A pixel fails tolerance if ANY of its RGBA channels exceeds tolerance.pixel_tolerance
    pixel_max_channel_err = np.max(abs_diff, axis=-1)
    exceeding_mask = pixel_max_channel_err > tolerance.pixel_tolerance
    diff_pixel_count = int(np.sum(exceeding_mask))
    total_pixels = int(img_ref.shape[0] * img_ref.shape[1])
    diff_pixel_ratio = float(diff_pixel_count) / float(total_pixels)

    ssim_score, ssim_map = compute_ssim(img_ref, img_ren)

    tolerance_passed = diff_pixel_ratio <= tolerance.max_diff_pixel_ratio
    ssim_passed = ssim_score >= tolerance.min_ssim
    passed = tolerance_passed and ssim_passed

    failure_reasons: list[str] = []
    if not tolerance_passed:
        failure_reasons.append(
            f"Outlier pixel ratio {diff_pixel_ratio * 100:.3f}% ({diff_pixel_count}/{total_pixels} px "
            f"with channel diff > {tolerance.pixel_tolerance:.4f}) exceeds threshold "
            f"{tolerance.max_diff_pixel_ratio * 100:.3f}%."
        )
    if not ssim_passed:
        failure_reasons.append(
            f"SSIM similarity {ssim_score:.5f} is below required minimum {tolerance.min_ssim:.5f}."
        )

    saved_diff_str: Optional[str] = None
    saved_panel_str: Optional[str] = None

    if (not passed or save_diff_on_pass) and diff_output_path is not None:
        diff_img = build_diff_heatmap(
            img_ref, img_ren, amplification=tolerance.diff_amplification
        )
        save_rgba_image(diff_img, diff_output_path)
        saved_diff_str = str(diff_output_path)

        if panel_output_path is not None:
            panel_img = build_comparison_panel(img_ref, img_ren, diff_img, ssim_map)
            save_rgba_image(panel_img, panel_output_path)
            saved_panel_str = str(panel_output_path)

    return ComparisonResult(
        passed=passed,
        tolerance_passed=tolerance_passed,
        ssim_passed=ssim_passed,
        ssim_score=round(ssim_score, 6),
        psnr_db=round(psnr_db, 3),
        mae=round(mae, 6),
        rmse=round(rmse, 6),
        max_abs_error=round(max_abs_error, 6),
        max_channel_error_u8=max_channel_error_u8,
        diff_pixel_count=diff_pixel_count,
        total_pixels=total_pixels,
        diff_pixel_ratio=round(diff_pixel_ratio, 6),
        pixel_tolerance=tolerance.pixel_tolerance,
        max_diff_pixel_ratio_threshold=tolerance.max_diff_pixel_ratio,
        min_ssim_threshold=tolerance.min_ssim,
        failure_reasons=failure_reasons,
        diff_image_path=saved_diff_str,
        panel_image_path=saved_panel_str,
    )
