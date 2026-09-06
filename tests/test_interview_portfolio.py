from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from evicon.interview_portfolio import (
    InterviewPortfolioError,
    main,
    run_portfolio,
    write_portfolio,
)


def test_run_portfolio_is_complete_and_offline() -> None:
    result = run_portfolio()
    report = result.report

    assert report.status == "completed"
    assert all(item.status == "passed" for item in report.checks)
    assert report.summary.factorization_requests == 192
    assert report.summary.stateful_update_requests == 288
    assert report.summary.cascade_run_count == 48
    assert report.summary.replay_pass_count == 48
    assert report.summary.application_replay_pass_count == 48
    assert report.summary.controller_fixture_count == 16
    assert report.summary.controller_valid_proposal_count == 16
    assert len(report.controller_examples) == 4
    assert len(report.technology_stack) >= 8
    assert any("Python 3.11" in item for item in report.technology_stack)
    assert any("Pydantic 2" in item for item in report.technology_stack)
    assert any("Agent 运行时" in item for item in report.technology_stack)
    assert len(result.trace_rows) == 64
    assert len(result.component_sha256) == 15
    assert "src/evicon/interview_portfolio.py" in result.component_sha256
    assert {row["stage"] for row in result.trace_rows} == {"controller", "cascade"}
    assert report.safety.network_enabled is False
    assert report.safety.api_key_read is False
    assert report.safety.real_provider_constructed is False
    assert report.safety.evaluator_private_truth_exposed is False


def test_dry_run_does_not_use_network_or_write_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline portfolio attempted a network connection")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVICON_LLM_API_KEY", "sentinel-api-key")
    monkeypatch.setattr(socket, "create_connection", fail_network)

    assert main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert payload["status"] == "completed"
    assert payload["safety"]["network_enabled"] is False
    assert payload["safety"]["api_key_read"] is False
    assert "sentinel-api-key" not in output
    assert list(tmp_path.iterdir()) == []


def test_written_artifacts_are_redacted_and_hash_bound(tmp_path: Path) -> None:
    output_root = tmp_path / "portfolio"
    receipt = write_portfolio(output_root)

    expected_files = {
        "portfolio_report.json",
        "portfolio_report.md",
        "portfolio_trace.jsonl",
        "portfolio_receipt.json",
    }
    assert {path.name for path in output_root.iterdir()} == expected_files
    assert set(receipt.output_files) == expected_files

    for filename, expected_digest in receipt.output_sha256.items():
        actual_digest = hashlib.sha256((output_root / filename).read_bytes()).hexdigest()
        assert actual_digest == expected_digest

    report = json.loads((output_root / "portfolio_report.json").read_text())
    trace_rows = [
        json.loads(line)
        for line in (output_root / "portfolio_trace.jsonl").read_text().splitlines()
    ]
    assert len(trace_rows) == 64
    assert report["safety"]["raw_model_responses_saved"] is False
    assert report["safety"]["prompts_saved"] is False
    assert len(report["technology_stack"]) >= 8
    assert "已实现的 Agent 技术栈" in (output_root / "portfolio_report.md").read_text()
    assert all("prompt" not in json.dumps(row).lower() for row in trace_rows)
    assert all("api_key" not in json.dumps(row).lower() for row in trace_rows)


def test_output_directory_requires_explicit_overwrite(tmp_path: Path) -> None:
    output_root = tmp_path / "portfolio"
    write_portfolio(output_root)

    with pytest.raises(InterviewPortfolioError) as exc_info:
        write_portfolio(output_root)
    assert exc_info.value.code == "output_exists_use_overwrite"

    replacement = write_portfolio(output_root, overwrite=True)
    assert replacement.status == "completed"
