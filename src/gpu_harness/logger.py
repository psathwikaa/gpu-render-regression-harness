"""Structured JSONL event logger, JSON summary writer, and GitLab JUnit XML exporter."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class StructuredEventLogger:
    """Writes structured JSON-Lines (JSONL) events for every stage of the harness run."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate existing log file at the start of a run
        self.log_path.write_text("", encoding="utf-8")
        self._events: List[Dict[str, Any]] = []

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_event(
        self,
        event_type: str,
        level: str = "INFO",
        scene_id: str | None = None,
        message: str = "",
        **payload: Any,
    ) -> Dict[str, Any]:
        """Record a structured JSON event to disk and in-memory buffer."""
        record: Dict[str, Any] = {
            "timestamp": self._utc_now_iso(),
            "level": level.upper(),
            "event_type": event_type,
        }
        if scene_id is not None:
            record["scene_id"] = scene_id
        if message:
            record["message"] = message
        if payload:
            record["data"] = payload

        self._events.append(record)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=False) + "\n")
        return record

    @property
    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)


def write_json_report(report_data: Dict[str, Any], output_path: Path) -> None:
    """Write the complete structured JSON summary report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)


def write_junit_xml(report_data: Dict[str, Any], output_path: Path) -> None:
    """Write a JUnit XML report compatible with GitLab CI pipeline test reports."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenes = report_data.get("scenes", [])
    total_tests = len(scenes)
    failures = sum(1 for s in scenes if not s.get("passed", False))
    total_time_s = float(report_data.get("total_duration_s", 0.0))

    testsuite = ET.Element(
        "testsuite",
        attrib={
            "name": "GPU Render Regression Suite",
            "tests": str(total_tests),
            "failures": str(failures),
            "errors": "0",
            "skipped": "0",
            "time": f"{total_time_s:.4f}",
            "timestamp": report_data.get("started_at", ""),
        },
    )

    props = ET.SubElement(testsuite, "properties")
    dev = report_data.get("device_info", {})
    for k, v in dev.items():
        ET.SubElement(props, "property", attrib={"name": str(k), "value": str(v)})

    for scene in scenes:
        perf = scene.get("performance", {})
        comp = scene.get("comparison", {})
        scene_time_s = (
            float(perf.get("mean_ms", 0.0))
            * float(perf.get("timed_iterations", 1))
            / 1000.0
        )
        tc = ET.SubElement(
            testsuite,
            "testcase",
            attrib={
                "classname": f"gpu_harness.scenes.{scene.get('category', 'OpenGL')}",
                "name": f"{scene.get('scene_id')} ({scene.get('title')})",
                "time": f"{scene_time_s:.4f}",
            },
        )
        if not scene.get("passed", False):
            reasons = comp.get("failure_reasons", ["Render regression detected"])
            msg = "; ".join(reasons)
            fail_el = ET.SubElement(
                tc,
                "failure",
                attrib={
                    "message": msg,
                    "type": "RenderRegressionError",
                },
            )
            fail_el.text = (
                f"Scene: {scene.get('scene_id')}\n"
                f"SSIM: {comp.get('ssim_score')} (min: {comp.get('min_ssim_threshold')})\n"
                f"Diff Pixel Ratio: {comp.get('diff_pixel_ratio')} "
                f"(max: {comp.get('max_diff_pixel_ratio_threshold')})\n"
                f"Max Channel Diff (u8): {comp.get('max_channel_error_u8')}\n"
                f"Diff Image: {comp.get('diff_image_path')}\n"
                f"Reasons: {msg}"
            )

        sys_out = ET.SubElement(tc, "system-out")
        sys_out.text = (
            f"SSIM={comp.get('ssim_score')} | PSNR={comp.get('psnr_db')} dB | "
            f"MeanFrame={perf.get('mean_ms')} ms | P95={perf.get('p95_ms')} ms | "
            f"CV={perf.get('cv')} | Stability={perf.get('stability_status')}"
        )

    tree = ET.ElementTree(testsuite)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
