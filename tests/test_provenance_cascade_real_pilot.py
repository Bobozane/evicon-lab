from __future__ import annotations

import json
from pathlib import Path

from evicon.cascade_agent_integration_smoke import _FakeProvider
from evicon.llm_contract import LLMProviderError, ProviderErrorCode
from evicon.provenance_cascade_real_pilot import execute_hd_real_pilot

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd.v2.toml"
APPROVAL = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd_approval_2026-08-21.toml"
AUTHORIZATION = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd_execution_authorization_2026-08-21.toml"
ENVIRONMENT = {
    "EVICON_LLM_MODEL": "fake-cascade-agent",
    "EVICON_LLM_BASE_URL": "https://offline.invalid/v1",
    "EVICON_LLM_API_KEY": "unit-test-placeholder",
}


def _execute(tmp_path: Path, factory, *, resume: bool = False, **updates):
    arguments = dict(
        config_path=CONFIG, approval_path=APPROVAL, authorization_path=AUTHORIZATION,
        allow_network=True, confirm_run=True, confirm_request_cap=864,
        confirm_completion_reservation_cap=221184, resume=resume,
        environment=ENVIRONMENT, provider_factory=factory,
        output_root_for_testing=tmp_path / "pilot",
    )
    arguments.update(updates)
    return execute_hd_real_pilot(**arguments)


def test_default_gate_is_network_disabled_and_creates_nothing(tmp_path: Path) -> None:
    summary = execute_hd_real_pilot(
        config_path=CONFIG, approval_path=APPROVAL, authorization_path=AUTHORIZATION,
        output_root_for_testing=tmp_path / "pilot",
    )
    assert summary.status == "blocked"
    assert summary.network == "disabled"
    assert summary.provider_constructed is False
    assert not (tmp_path / "pilot").exists()


def test_fake_provider_completes_fixed_48_run_plan(tmp_path: Path) -> None:
    providers = []

    def factory(_config):
        provider = _FakeProvider()
        providers.append(provider)
        return provider

    summary = _execute(tmp_path, factory)
    assert summary.status == "completed"
    assert summary.completed_run_count == 48
    assert summary.logical_request_count == 864
    assert summary.completion_reserved_token_count == 221184
    assert summary.replay_status == "passed"
    assert sum(provider.calls for provider in providers) == 864
    root = tmp_path / "pilot"
    assert (root / "pilot_batch_receipt.json").exists()
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.json*"))
    for forbidden in ("system_prompt", "user_prompt", "api_key", "provider_metadata", "ground_truth_label", "source_independence_label"):
        assert forbidden not in text.lower()


def test_failed_run_resumes_without_replaying_completed_coordinates(tmp_path: Path) -> None:
    class FailOnce:
        calls = 0
        failed = False

        def complete(self, request):
            self.calls += 1
            if not type(self).failed and self.calls == 5:
                type(self).failed = True
                raise LLMProviderError(ProviderErrorCode.TIMEOUT, "redacted")
            return _FakeProvider().complete(request)

    first_providers = []
    first = _execute(tmp_path, lambda _config: first_providers.append(FailOnce()) or first_providers[-1])
    assert first.status == "failed"
    assert first.completed_run_count == 0
    assert first.logical_request_count == 5

    resumed_providers = []
    resumed = _execute(tmp_path, lambda _config: resumed_providers.append(_FakeProvider()) or resumed_providers[-1], resume=True)
    assert resumed.status == "completed"
    assert resumed.logical_request_count == 864
    assert sum(provider.calls for provider in resumed_providers) == 860
    first_ledger = tmp_path / "pilot" / "hd-cascade-false-majority-20260901-no_intervention" / "request_ledger.jsonl"
    assert len(first_ledger.read_text(encoding="utf-8").splitlines()) == 38


def test_wrong_confirmation_and_existing_output_are_zero_call_blocks(tmp_path: Path) -> None:
    calls = 0

    def factory(_config):
        nonlocal calls
        calls += 1
        return _FakeProvider()

    wrong = _execute(tmp_path, factory, confirm_request_cap=863)
    assert wrong.status == "blocked" and calls == 0
    (tmp_path / "pilot").mkdir()
    existing = _execute(tmp_path, factory)
    assert existing.status == "blocked" and existing.error_code == "output_root_exists" and calls == 0
