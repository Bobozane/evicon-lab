"""Offline-only JSONL persistence and reading for private probe results."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .models import ProbeResult


class ProbeResultStoreError(ValueError):
    """Raised when isolated ProbeResult files cannot be written or read safely."""


def write_probe_results(
    output_dir: str | Path,
    run_id: str,
    results: list[ProbeResult],
) -> Path:
    """Persist private results under a new `probes` directory without overwriting."""
    if not run_id.strip():
        raise ProbeResultStoreError("run_id must be non-blank")
    if not results:
        raise ProbeResultStoreError("results must contain at least one ProbeResult")
    probes_directory = Path(output_dir) / run_id / "probes"
    try:
        probes_directory.mkdir(parents=True, exist_ok=False)
        output_path = probes_directory / "probe_results.jsonl"
        with output_path.open("x", encoding="utf-8") as handle:
            for result in results:
                handle.write(result.model_dump_json() + "\n")
    except OSError as exc:
        raise ProbeResultStoreError(
            f"refusing to overwrite or create probe results at {probes_directory}: {exc}"
        ) from exc
    return output_path


def read_probe_results(path: str | Path) -> list[ProbeResult]:
    """Read isolated results for later offline metrics; no online caller uses this."""
    result_path = Path(path)
    try:
        lines = result_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ProbeResultStoreError(f"cannot read probe results {result_path}: {exc}") from exc
    if not lines:
        raise ProbeResultStoreError(f"probe results {result_path} must not be empty")

    results: list[ProbeResult] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ProbeResultStoreError(f"{result_path}:{line_number}: blank JSONL lines are not allowed")
        try:
            results.append(ProbeResult.model_validate_json(line))
        except (ValidationError, json.JSONDecodeError) as exc:
            raise ProbeResultStoreError(
                f"{result_path}:{line_number}: invalid ProbeResult JSON"
            ) from exc
    return results
