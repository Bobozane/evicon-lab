"""Offline smoke for the provenance-cascade outcome/evaluator boundary."""
from __future__ import annotations
import json
from pathlib import Path
from .cascade_protocol import CascadeProtocolRunner, CascadeScenarioLoader
from .cascade_outcomes import CascadeOutcomeLedger, ClaimStance, OutcomeSource, PublicClaimOutcome
from .cascade_evaluation import CascadeOfflineEvaluator
from .provenance_cascade import EvaluatorTruthLoader
from .provenance_cascade_preregistration import ProvenanceCascadePreregistrationLoader

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs" / "provenance_cascade"
PREREG = ROOT / "configs" / "studies" / "provenance_cascade_pilot_preregistration.toml"
SCENARIOS = ("false_majority", "true_minority_correction", "independent_true_consensus", "unresolved_disagreement")

def _outcomes(run, scenario_type: str) -> CascadeOutcomeLedger:
    final_round = run.round_count - 1
    by_agent_claim: dict[tuple[str, str], list] = {}
    for event in run.ledger.events:
        by_agent_claim.setdefault((event.target_agent_id, event.claim_id), []).append(event)
    claims = {event.claim_id for event in run.ledger.events}
    outcomes=[]
    for agent in run.agent_ids:
        for claim in sorted(claims):
            events=by_agent_claim.get((agent,claim), [])
            if scenario_type == "unresolved_disagreement": stance=ClaimStance.UNCERTAIN
            elif events: stance=ClaimStance.ENDORSES
            else: stance=ClaimStance.NO_POSITION
            outcomes.append(PublicClaimOutcome(scenario_id=run.scenario_id, agent_id=agent, claim_id=claim, round_id=final_round, stance=stance, content_ids=tuple(e.content_id for e in events), evidence_card_ids=tuple(x for e in events for x in e.evidence_card_ids), source=OutcomeSource.DETERMINISTIC_FIXTURE))
    return CascadeOutcomeLedger(scenario_id=run.scenario_id, agent_ids=run.agent_ids, outcomes=tuple(outcomes))

def main() -> None:
    prereg=ProvenanceCascadePreregistrationLoader.load(PREREG)
    rows=[]
    for index, name in enumerate(SCENARIOS):
        scenario=CascadeScenarioLoader.load(BASE / "scenarios" / f"{name}.toml")
        run=CascadeProtocolRunner().run_scenario(scenario, prereg.seeds[0])
        outcome=_outcomes(run, name)
        truth=EvaluatorTruthLoader.load(BASE / "fixtures" / f"{name}.evaluator_private.json")
        report=CascadeOfflineEvaluator.evaluate(run=run, graph=scenario.graph, outcomes=outcome, truth=truth, preregistration=prereg)
        rows.append({"scenario_id":name,"replay":report.replay_status,"metrics":[{"name":m.metric_name,"status":m.status.value,"value":m.value,"denominator":m.denominator} for m in report.metrics],"outcome_sha256":report.outcome_sha256[:12],"development_only":True,"not_paper_result":True,"no_causal_conclusion":True})
    print(json.dumps({"scenarios":rows,"private_truth_exposed":False,"network":"disabled","results_written":False}, ensure_ascii=True, sort_keys=True))
if __name__ == "__main__": main()
