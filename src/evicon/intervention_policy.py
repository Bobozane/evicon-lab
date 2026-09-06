"""Public policy contracts with minority-retention precedence.

The implementation module retains the small deterministic helpers.  This
public boundary makes the action-precedence rule explicit without changing
the existing online runner contracts.
"""

from __future__ import annotations

from . import intervention_policy_core as _core
from .conformity_monitor import MonitorReason
from .models.protocols import InterventionAction

InterventionDecision = _core.InterventionDecision
InterventionHistoryEntry = _core.InterventionHistoryEntry
PolicyConfig = _core.PolicyConfig
PolicyInput = _core.PolicyInput
PolicyReason = _core.PolicyReason
TargetCandidate = _core.TargetCandidate
TargetSelection = _core.TargetSelection
TargetSelectionReason = _core.TargetSelectionReason
select_targets = _core.select_targets


class InterventionPolicy:
    """Side-effect-free eligibility policy with deterministic action selection."""

    @staticmethod
    def decide(policy_input: PolicyInput, config: PolicyConfig) -> InterventionDecision:
        """Emit a decision without invoking a model or applying an intervention."""
        decision = _core.InterventionPolicy.decide(policy_input, config)
        monitor = policy_input.monitor_result
        selection = select_targets(policy_input.target_candidates, config, config.max_targets)

        # A safe, eligible minority representative should receive dissent
        # solicitation before the generic insufficient-evidence fallback.
        if (
            decision.action is InterventionAction.REQUEST_EVIDENCE
            and monitor.signals.minority_loss > 0.0
            and MonitorReason.HARM_RISK not in monitor.reasons
            and selection.target_agent_ids
        ):
            return _core._with_budget(
                policy_input,
                config,
                action=InterventionAction.SOLICIT_DISSENT,
                target_agent_ids=selection.target_agent_ids,
                reason_codes=[
                    PolicyReason.MINORITY_LOSS,
                    PolicyReason.TARGETS_SELECTED,
                ],
            )
        return decision


__all__ = [
    "InterventionDecision",
    "InterventionHistoryEntry",
    "InterventionPolicy",
    "PolicyConfig",
    "PolicyInput",
    "PolicyReason",
    "TargetCandidate",
    "TargetSelection",
    "TargetSelectionReason",
    "select_targets",
]
