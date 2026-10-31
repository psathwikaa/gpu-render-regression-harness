"""Self-contained HTML report generator with embedded images and SVG frame-time charts."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, BaseLoader


def _encode_image_base64(path_str: Optional[str]) -> Optional[str]:
    """Encode a PNG file on disk as a data:image/png;base64 URI for self-contained HTML reports."""
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_file():
        return None
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _build_frame_time_svg(
    warmup_ms: List[float],
    timed_ms: List[float],
    mean_ms: float,
    p95_ms: float,
) -> str:
    """Generate an inline SVG chart plotting warmup + timed iteration frame times."""
    all_vals = list(warmup_ms) + list(timed_ms)
    if not all_vals:
        return ""

    w = 540
    h = 150
    pad_l, pad_r, pad_t, pad_b = 46, 18, 18, 28
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    max_y = max(max(all_vals) * 1.18, p95_ms * 1.15, 0.1)
    min_y = 0.0
    n = len(all_vals)

    def x_pos(idx: int) -> float:
        if n <= 1:
            return pad_l + plot_w * 0.5
        return pad_l + (idx / float(n - 1)) * plot_w

    def y_pos(val: float) -> float:
        norm = (val - min_y) / (max_y - min_y) if max_y > min_y else 0.5
        return pad_t + plot_h * (1.0 - max(0.0, min(1.0, norm)))

    # Polyline points for all iterations
    pts = " ".join(f"{x_pos(i):.1f},{y_pos(v):.1f}" for i, v in enumerate(all_vals))
    area_pts = (
        f"{x_pos(0):.1f},{pad_t + plot_h:.1f} "
        + pts
        + f" {x_pos(n - 1):.1f},{pad_t + plot_h:.1f}"
    )

    y_mean = y_pos(mean_ms)
    y_p95 = y_pos(p95_ms)
    warmup_count = len(warmup_ms)
    x_Boundary = (
        (x_pos(warmup_count - 1) + x_pos(warmup_count)) * 0.5
        if 0 < warmup_count < n
        else None
    )

    circles: List[str] = []
    for i, v in enumerate(all_vals):
        cx = x_pos(i)
        cy = y_pos(v)
        is_warmup = i < warmup_count
        fill = "#f59e0b" if is_warmup else "#38bdf8"
        label = f"Warmup #{i + 1}: {v:.3f} ms" if is_warmup else f"Run #{i - warmup_count + 1}: {v:.3f} ms"
        circles.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.3" fill="{fill}" stroke="#0f172a" stroke-width="1.2">'
            f"<title>{label}</title></circle>"
        )

    warmup_sep = ""
    if x_Boundary is not None:
        warmup_sep = (
            f'<line x1="{x_Boundary:.1f}" y1="{pad_t}" x2="{x_Boundary:.1f}" y2="{pad_t + plot_h}" '
            f'stroke="#f59e0b" stroke-dasharray="3,3" stroke-width="1" opacity="0.7"/>'
            f'<text x="{x_Boundary - 4:.1f}" y="{pad_t + 10}" fill="#f59e0b" font-size="9" text-anchor="end">Warmup</text>'
        )

    return f"""
    <svg viewBox="0 0 {w} {h}" width="100%" height="{h}" xmlns="http://www.w3.org/2000/svg" class="ft-svg">
      <defs>
        <linearGradient id="ftGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.32"/>
          <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.02"/>
        </linearGradient>
      </defs>
      <!-- Grid lines -->
      <line x1="{pad_l}" y1="{pad_t}" x2="{w - pad_r}" y2="{pad_t}" stroke="#1e293b" stroke-width="1"/>
      <line x1="{pad_l}" y1="{pad_t + plot_h * 0.5:.1f}" x2="{w - pad_r}" y2="{pad_t + plot_h * 0.5:.1f}" stroke="#1e293b" stroke-width="1"/>
      <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{w - pad_r}" y2="{pad_t + plot_h}" stroke="#334155" stroke-width="1"/>
      <!-- Y-axis labels -->
      <text x="{pad_l - 6}" y="{pad_t + 4}" fill="#94a3b8" font-size="10" text-anchor="end">{max_y:.2f}ms</text>
      <text x="{pad_l - 6}" y="{pad_t + plot_h * 0.5 + 3:.1f}" fill="#94a3b8" font-size="10" text-anchor="end">{max_y * 0.5:.2f}ms</text>
      <text x="{pad_l - 6}" y="{pad_t + plot_h + 3}" fill="#94a3b8" font-size="10" text-anchor="end">0.00ms</text>
      <!-- P95 and Mean threshold lines -->
      <line x1="{pad_l}" y1="{y_p95:.1f}" x2="{w - pad_r}" y2="{y_p95:.1f}" stroke="#a855f7" stroke-dasharray="4,3" stroke-width="1.1"/>
      <text x="{w - pad_r - 2}" y="{max(pad_t + 9, y_p95 - 3):.1f}" fill="#c084fc" font-size="9" text-anchor="end">P95 ({p95_ms:.2f}ms)</text>
      <line x1="{pad_l}" y1="{y_mean:.1f}" x2="{w - pad_r}" y2="{y_mean:.1f}" stroke="#10b981" stroke-dasharray="5,3" stroke-width="1.2"/>
      <text x="{pad_l + 4}" y="{max(pad_t + 9, y_mean - 3):.1f}" fill="#34d399" font-size="9">Mean ({mean_ms:.2f}ms)</text>
      {warmup_sep}
      <!-- Area & line -->
      <polygon points="{area_pts}" fill="url(#ftGrad)"/>
      <polyline fill="none" stroke="#38bdf8" stroke-width="2" points="{pts}"/>
      {"".join(circles)}
      <!-- X-axis caption -->
      <text x="{pad_l + plot_w * 0.5:.1f}" y="{h - 6}" fill="#94a3b8" font-size="10" text-anchor="middle">
        Frame Sequence ({warmup_count} Warmup + {len(timed_ms)} Timed Iterations)
      </text>
    </svg>
    """


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>GPU Render Regression Test Report</title>
  <style>
    :root {
      --bg-primary: #0b0f19;
      --bg-card: #111827;
      --bg-elevated: #1e293b;
      --border: #283548;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --pass: #10b981;
      --pass-bg: rgba(16, 185, 129, 0.14);
      --fail: #ef4444;
      --fail-bg: rgba(239, 68, 68, 0.15);
      --warn: #f59e0b;
      --warn-bg: rgba(245, 158, 11, 0.15);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
      background-color: var(--bg-primary);
      color: var(--text-main);
      line-height: 1.55;
      padding: 28px 32px;
    }
    .container { max-width: 1440px; margin: 0 auto; }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding-bottom: 22px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }
    .title-group h1 {
      font-size: 1.65rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #f8fafc;
    }
    .title-group p {
      color: var(--text-muted);
      font-size: 0.9rem;
      margin-top: 4px;
    }
    .verdict-banner {
      padding: 10px 20px;
      border-radius: 8px;
      font-weight: 700;
      font-size: 0.95rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      border: 1px solid;
    }
    .verdict-pass {
      background: var(--pass-bg);
      color: #34d399;
      border-color: rgba(16, 185, 129, 0.4);
    }
    .verdict-fail {
      background: var(--fail-bg);
      color: #f87171;
      border-color: rgba(239, 68, 68, 0.45);
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .kpi-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px 18px;
    }
    .kpi-label {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .kpi-value {
      font-size: 1.55rem;
      font-weight: 700;
      color: #f8fafc;
    }
    .kpi-sub {
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 4px;
    }
    .section-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px 22px;
      margin-bottom: 24px;
    }
    .section-title {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 14px;
      color: #e2e8f0;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }
    th, td {
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    th {
      color: var(--text-muted);
      font-weight: 600;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: rgba(30, 41, 59, 0.5);
    }
    tr:hover td { background: rgba(30, 41, 59, 0.35); }
    a.scene-link { color: var(--accent); text-decoration: none; font-weight: 600; }
    a.scene-link:hover { text-decoration: underline; }
    .badge {
      display: inline-block;
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 0.74rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .badge-pass { background: var(--pass-bg); color: #34d399; border: 1px solid rgba(16,185,129,0.35); }
    .badge-fail { background: var(--fail-bg); color: #f87171; border: 1px solid rgba(239,68,68,0.35); }
    .badge-warn { background: var(--warn-bg); color: #fbbf24; border: 1px solid rgba(245,158,11,0.35); }
    .badge-tag {
      background: var(--bg-elevated);
      color: #cbd5e1;
      border: 1px solid #334155;
      font-weight: 500;
      text-transform: none;
      margin-right: 5px;
      margin-bottom: 4px;
    }
    .scene-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 22px;
      margin-bottom: 24px;
    }
    .scene-card.failed-card {
      border-color: rgba(239, 68, 68, 0.6);
      box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.2);
    }
    .scene-header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 16px;
    }
    .scene-title { font-size: 1.2rem; font-weight: 700; color: #f8fafc; }
    .scene-desc { font-size: 0.88rem; color: var(--text-muted); margin: 6px 0 10px 0; max-width: 900px; }
    .failure-box {
      background: var(--fail-bg);
      border: 1px solid rgba(239, 68, 68, 0.45);
      color: #fca5a5;
      padding: 12px 16px;
      border-radius: 8px;
      margin-bottom: 16px;
      font-size: 0.88rem;
    }
    .failure-box ul { margin-left: 18px; margin-top: 4px; }
    .visual-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }
    .img-panel {
      background: #090d16;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      text-align: center;
    }
    .img-panel-header {
      font-size: 0.76rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 8px;
    }
    .img-panel img {
      width: 100%;
      max-width: 320px;
      height: auto;
      border-radius: 4px;
      border: 1px solid #1e293b;
      image-rendering: auto;
    }
    .metrics-perf-grid {
      display: grid;
      grid-template-columns: 1fr 1.2fr;
      gap: 18px;
      align-items: start;
    }
    @media (max-width: 960px) {
      .metrics-perf-grid { grid-template-columns: 1fr; }
    }
    .mini-table td { padding: 6px 10px; font-size: 0.83rem; }
    .mini-table td:first-child { color: var(--text-muted); }
    .mini-table td:last-child { font-family: "JetBrains Mono", Consolas, monospace; text-align: right; font-weight: 600; }
    .chart-box {
      background: #090d16;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
    }
    .chart-box-title {
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-muted);
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
    }
    .log-viewer {
      background: #070a12;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      max-height: 320px;
      overflow-y: auto;
      font-family: "JetBrains Mono", Consolas, monospace;
      font-size: 0.78rem;
      color: #cbd5e1;
      white-space: pre-wrap;
    }
    footer {
      text-align: center;
      color: var(--text-muted);
      font-size: 0.8rem;
      margin-top: 32px;
      padding-top: 16px;
      border-top: 1px solid var(--border);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="title-group">
        <h1>GPU Render Regression Test Harness</h1>
        <p>
          Generated: {{ report.completed_at }} &bull;
          Backend: <strong>{{ report.device_info.backend }}</strong> &bull;
          GPU: <strong>{{ report.device_info.gl_renderer }} ({{ report.device_info.gl_version }})</strong> &bull;
          Resolution: <strong>{{ report.device_info.resolution }}</strong>
        </p>
      </div>
      <div class="verdict-banner {{ 'verdict-pass' if report.overall_passed else 'verdict-fail' }}">
        {{ 'PASS — All Scenes Verified' if report.overall_passed else 'FAIL — Render Regressions Detected' }}
      </div>
    </header>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Scenes Passed</div>
        <div class="kpi-value">{{ report.passed_scenes }} / {{ report.total_scenes }}</div>
        <div class="kpi-sub">Pass Rate: {{ "%.1f"|format(report.pass_rate * 100.0) }}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Mean SSIM Similarity</div>
        <div class="kpi-value">{{ "%.5f"|format(report.mean_ssim) }}</div>
        <div class="kpi-sub">Threshold: &ge; {{ "%.4f"|format(report.config.tolerance.min_ssim) }}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Pixel Tolerance</div>
        <div class="kpi-value">&le; {{ "%.2f"|format(report.config.tolerance.pixel_tolerance * 100.0) }}%</div>
        <div class="kpi-sub">Max Outlier Pixels: {{ "%.2f"|format(report.config.tolerance.max_diff_pixel_ratio * 100.0) }}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Mean Render Frame Time</div>
        <div class="kpi-value">{{ "%.3f"|format(report.overall_mean_frame_ms) }} ms</div>
        <div class="kpi-sub">{{ "%.1f"|format(1000.0 / report.overall_mean_frame_ms if report.overall_mean_frame_ms > 0 else 0) }} FPS avg ({{ report.config.performance.timed_iterations }} runs/scene)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Total Suite Duration</div>
        <div class="kpi-value">{{ "%.2f"|format(report.total_duration_s) }} s</div>
        <div class="kpi-sub">Vendor: {{ report.device_info.gl_vendor }}</div>
      </div>
    </div>

    <div class="section-card">
      <div class="section-title">
        <span>Scene Regression &amp; Frame-Time Stability Summary</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Scene ID</th>
            <th>Category</th>
            <th>Status</th>
            <th>SSIM</th>
            <th>PSNR (dB)</th>
            <th>Outlier Pixels</th>
            <th>Max Diff (u8)</th>
            <th>Mean (ms)</th>
            <th>P95 (ms)</th>
            <th>StdDev / CV</th>
            <th>Stability</th>
          </tr>
        </thead>
        <tbody>
          {% for s in scenes %}
          <tr>
            <td><a class="scene-link" href="#scene-{{ s.scene_id }}">{{ s.scene_id }}</a></td>
            <td>{{ s.category }}</td>
            <td>
              <span class="badge {{ 'badge-pass' if s.passed else 'badge-fail' }}">
                {{ 'PASS' if s.passed else 'FAIL' }}
              </span>
            </td>
            <td>{{ "%.5f"|format(s.comparison.ssim_score) }}</td>
            <td>{{ "%.2f"|format(s.comparison.psnr_db) }}</td>
            <td>{{ "%.3f"|format(s.comparison.diff_pixel_ratio * 100.0) }}% ({{ s.comparison.diff_pixel_count }})</td>
            <td>{{ s.comparison.max_channel_error_u8 }} / 255</td>
            <td>{{ "%.3f"|format(s.performance.mean_ms) }}</td>
            <td>{{ "%.3f"|format(s.performance.p95_ms) }}</td>
            <td>{{ "%.3f"|format(s.performance.std_ms) }}ms ({{ "%.2f"|format(s.performance.cv) }})</td>
            <td>
              <span class="badge {{ 'badge-pass' if s.performance.stability_status == 'STABLE' else 'badge-warn' }}">
                {{ s.performance.stability_status }}
              </span>
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    {% for s in scenes %}
    <div id="scene-{{ s.scene_id }}" class="scene-card {{ '' if s.passed else 'failed-card' }}">
      <div class="scene-header">
        <div>
          <div class="scene-title">{{ s.scene_id }} &mdash; {{ s.title }}</div>
          <div class="scene-desc">{{ s.description }}</div>
          <div>
            <span class="badge badge-tag">{{ s.category }}</span>
            {% for feat in s.features %}
            <span class="badge badge-tag">{{ feat }}</span>
            {% endfor %}
            {% if s.regression_injected %}
            <span class="badge badge-fail">INJECTED REGRESSION: {{ s.regression_description }}</span>
            {% endif %}
          </div>
        </div>
        <div>
          <span class="badge {{ 'badge-pass' if s.passed else 'badge-fail' }}" style="font-size: 0.9rem; padding: 6px 14px;">
            {{ 'PASSED' if s.passed else 'FAILED' }}
          </span>
        </div>
      </div>

      {% if not s.passed and s.comparison.failure_reasons %}
      <div class="failure-box">
        <strong>Regression Failure Diagnostics:</strong>
        <ul>
          {% for reason in s.comparison.failure_reasons %}
          <li>{{ reason }}</li>
          {% endfor %}
        </ul>
      </div>
      {% endif %}

      <div class="visual-grid">
        <div class="img-panel">
          <div class="img-panel-header">Stored Reference (Golden)</div>
          {% if s.ref_b64 %}
          <img src="{{ s.ref_b64 }}" alt="Reference {{ s.scene_id }}" />
          {% endif %}
        </div>
        <div class="img-panel">
          <div class="img-panel-header">Rendered Off-Screen Frame</div>
          {% if s.ren_b64 %}
          <img src="{{ s.ren_b64 }}" alt="Rendered {{ s.scene_id }}" />
          {% endif %}
        </div>
        {% if s.diff_b64 %}
        <div class="img-panel">
          <div class="img-panel-header">Amplified Diff Heatmap ({{ "%.0f"|format(report.config.tolerance.diff_amplification) }}x)</div>
          <img src="{{ s.diff_b64 }}" alt="Diff {{ s.scene_id }}" />
        </div>
        {% endif %}
      </div>

      {% if s.panel_b64 %}
      <div class="img-panel" style="margin-bottom: 18px;">
        <div class="img-panel-header">4-Panel Diagnostic Strip (Reference | Rendered | Pixel Diff | 1 - SSIM Structural Error)</div>
        <img src="{{ s.panel_b64 }}" alt="Comparison Panel {{ s.scene_id }}" style="max-width: 100%;" />
      </div>
      {% endif %}

      <div class="metrics-perf-grid">
        <div>
          <table class="mini-table">
            <thead>
              <tr><th colspan="2">Image Accuracy &amp; Frame Stability Metrics</th></tr>
            </thead>
            <tbody>
              <tr>
                <td>SSIM Similarity (Min &ge; {{ "%.4f"|format(s.comparison.min_ssim_threshold) }})</td>
                <td>{{ "%.6f"|format(s.comparison.ssim_score) }}</td>
              </tr>
              <tr>
                <td>Peak Signal-to-Noise Ratio (PSNR)</td>
                <td>{{ "%.2f"|format(s.comparison.psnr_db) }} dB</td>
              </tr>
              <tr>
                <td>Mean Absolute Error (MAE) / RMSE</td>
                <td>{{ "%.6f"|format(s.comparison.mae) }} / {{ "%.6f"|format(s.comparison.rmse) }}</td>
              </tr>
              <tr>
                <td>Max Channel Error (0..255)</td>
                <td>{{ s.comparison.max_channel_error_u8 }} ({{ "%.4f"|format(s.comparison.max_abs_error) }})</td>
              </tr>
              <tr>
                <td>Outlier Pixels (&gt; {{ "%.4f"|format(s.comparison.pixel_tolerance) }} tol)</td>
                <td>{{ s.comparison.diff_pixel_count }} / {{ s.comparison.total_pixels }} ({{ "%.4f"|format(s.comparison.diff_pixel_ratio * 100.0) }}%)</td>
              </tr>
              <tr>
                <td>Frame Time Mean / Median (P50)</td>
                <td>{{ "%.4f"|format(s.performance.mean_ms) }} ms / {{ "%.4f"|format(s.performance.median_ms) }} ms</td>
              </tr>
              <tr>
                <td>Frame Time P95 / P99 / Max</td>
                <td>{{ "%.4f"|format(s.performance.p95_ms) }} ms / {{ "%.4f"|format(s.performance.p99_ms) }} ms / {{ "%.4f"|format(s.performance.max_ms) }} ms</td>
              </tr>
              <tr>
                <td>StdDev / Coefficient of Variation (CV)</td>
                <td>{{ "%.4f"|format(s.performance.std_ms) }} ms / {{ "%.4f"|format(s.performance.cv) }}</td>
              </tr>
              <tr>
                <td>Inter-Frame Pixel Determinism</td>
                <td>{{ 'IDENTICAL (0 diff across runs)' if s.performance.deterministic_frames else 'DRIFT (' ~ s.performance.max_interframe_diff_u8 ~ ' u8)' }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="chart-box">
          <div class="chart-box-title">
            <span>Frame Time Stability Across Repeated Runs</span>
            <span>Mean FPS: {{ "%.1f"|format(s.performance.mean_fps) }}</span>
          </div>
          {{ s.svg_chart | safe }}
        </div>
      </div>
    </div>
    {% endfor %}

    <div class="section-card">
      <div class="section-title">
        <span>Structured Log Excerpt (test_events.jsonl)</span>
        <span style="font-size: 0.8rem; color: var(--text-muted);">{{ log_lines|length }} structured events recorded</span>
      </div>
      <div class="log-viewer">{% for line in log_lines %}{{ line }}
{% endfor %}</div>
    </div>

    <footer>
      GPU Render Regression Test Harness v1.0.0 &bull; Off-Screen OpenGL ({{ report.device_info.backend }}) &bull; GitLab CI Ready
    </footer>
  </div>
</body>
</html>
"""


def generate_html_report(
    report_data: Dict[str, Any],
    output_path: Path,
    jsonl_log_path: Optional[Path] = None,
) -> None:
    """Render the self-contained HTML report with embedded Base64 images and SVG frame charts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    enriched_scenes: List[Dict[str, Any]] = []
    for scene in report_data.get("scenes", []):
        s_copy = dict(scene)
        comp = s_copy.get("comparison", {})
        perf = s_copy.get("performance", {})

        s_copy["ref_b64"] = _encode_image_base64(s_copy.get("reference_image_path"))
        s_copy["ren_b64"] = _encode_image_base64(s_copy.get("rendered_image_path"))
        s_copy["diff_b64"] = _encode_image_base64(comp.get("diff_image_path"))
        s_copy["panel_b64"] = _encode_image_base64(comp.get("panel_image_path"))

        s_copy["svg_chart"] = _build_frame_time_svg(
            warmup_ms=perf.get("warmup_times_ms", []),
            timed_ms=perf.get("frame_times_ms", []),
            mean_ms=float(perf.get("mean_ms", 0.0)),
            p95_ms=float(perf.get("p95_ms", 0.0)),
        )
        enriched_scenes.append(s_copy)

    log_lines: List[str] = []
    if jsonl_log_path and jsonl_log_path.is_file():
        log_lines = jsonl_log_path.read_text(encoding="utf-8").splitlines()

    env = Environment(loader=BaseLoader(), autoescape=True)
    template = env.from_string(HTML_TEMPLATE)
    html_str = template.render(
        report=report_data,
        scenes=enriched_scenes,
        log_lines=log_lines,
    )
    output_path.write_text(html_str, encoding="utf-8")
