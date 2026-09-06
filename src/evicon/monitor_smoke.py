"""Local-only smoke cases for the pure ConformityMonitor."""

from __future__ import annotations

import json

from .conformity_monitor import ConformityMonitor, MonitorConfig, MonitorInput


def smoke_config() -> MonitorConfig:
    """Return demonstration-only parameters; they are not experimental calibration."""
    return MonitorConfig(
        collapse_weight=0.2,
        minority_weight=0.3,
        evidence_weight=0.4,
        evidence_quality_weight=0.1,
        harm_weight=0.5,
        collapse_trigger=0.5,
        minority_trigger=0.5,
        evidence_sufficient_threshold=0.7,
        high_risk_threshold=0.5,
        medium_risk_threshold=0.25,
        task_quality_sufficient_threshold=0.6,
        version="smoke-v1",
    )


def smoke_cases() -> list[tuple[str, MonitorInput]]:
    """Return three deterministic local summaries without loading any run artifact."""
    common = {
        "run_id": "monitor-smoke",
        "scenario_id": "monitor-smoke-scenario",
        "round_id": 1,
        "remaining_budget": 0.0,
        "observable_peer_count": 2,
        "metadata": {"purpose": "local smoke"},
    }
    return [
        (
            "stable",
            MonitorInput(
                **common,
                current_diversity=0.6,
                previous_diversity=0.6,
                current_coverage=0.8,
                previous_coverage=0.8,
                minority_loss=0.0,
                evidence_gain=0.0,
                evidence_quality=0.0,
                harm_risk=0.0,
                task_quality=0.8,
            ),
        ),
        (
            "social-collapse-without-evidence",
            MonitorInput(
                **common,
                current_diversity=0.1,
                previous_diversity=0.9,
                current_coverage=0.2,
                previous_coverage=0.9,
                minority_loss=0.8,
                evidence_gain=0.0,
                evidence_quality=0.0,
                harm_risk=0.4,
                task_quality=0.8,
            ),
        ),
        (
            "evidence-supported-update",
            MonitorInput(
                **common,
                current_diversity=0.1,
                previous_diversity=0.9,
                current_coverage=0.7,
                previous_coverage=0.8,
                minority_loss=0.1,
                evidence_gain=1.0,
                evidence_quality=1.0,
                harm_risk=0.0,
                task_quality=0.8,
            ),
        ),
    ]


def main() -> int:
    config = smoke_config()
    for label, monitor_input in smoke_cases():
        result = ConformityMonitor.evaluate(monitor_input, config)
        print(
            json.dumps(
                {
                    "case": label,
                    "risk_score": result.risk_score,
                    "risk_level": result.risk_level.value,
                    "should_intervene": result.should_intervene,
                    "reasons": [reason.value for reason in result.reasons],
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
