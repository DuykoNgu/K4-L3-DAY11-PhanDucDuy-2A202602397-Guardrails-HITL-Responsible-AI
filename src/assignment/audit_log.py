"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._inputs: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Ghi câu hỏi + mốc thời gian bắt đầu, khoá theo request_id (fallback user_id)."""
        key = request_id or user_id
        self._open[key] = time.time()
        self._inputs[key] = {
            "request_id": request_id,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Ghi câu trả lời + lớp nào chặn + latency; append vào self.logs."""
        key = request_id or user_id
        started = self._open.pop(key, None)
        entry = self._inputs.pop(key, {})
        self.logs.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "input": entry.get("input", ""),
                "started_at": entry.get("started_at"),
                "finished_at": utc_now_iso(),
                "latency_ms": round((time.time() - started) * 1000, 2) if started else None,
                "blocked": blocked,
                "layer": layer,
                "output": text,
            }
        )

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
