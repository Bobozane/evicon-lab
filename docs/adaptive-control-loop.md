# Adaptive Control Loop

`AdaptiveProtocolRunner` is an opt-in FakeLLM-only controller. After a round
has completed, it observes the public `DialogueState`, evaluates
`ConformityMonitor`, builds public `TargetCandidate` values, calls the existing
`InterventionPolicy`, and compiles a plan with `InterventionExecutor`.

The timing is deliberately split:

1. Finish every Agent turn from one round-start snapshot.
2. Emit `online_state_observed` and `monitor_evaluated`.
3. Emit `target_candidates_built` and `policy_decided`.
4. Create and reserve a non-`no_op` plan for the next round.
5. At the next round boundary, apply it through the existing prospective plan
   application mechanism.
6. Only after actual application move the reservation to spent budget and
   start cooldown.

The current round is never changed by a plan created after it. No controller
step invokes another model, reads ProbeResult, reads hidden profiles or offline
metrics, modifies `DialogueState`, or generates a mediator prompt.

`FakeOnlineStateProvider` and `FakeTargetCandidateProvider` use deterministic
scripted public signals and Agent IDs. They are engineering fixtures, not
claims about real experiments. Online inputs are validated as `MonitorInput`
and private/prompt-like fields are rejected.

Adaptive audit events include online observation, monitor output, candidate
construction, policy decisions, plan creation/scheduling, and budget/cooldown
transitions. They contain numeric summaries and identifiers only. A controller
history records planned, applied, rejected, or cancelled states; it does not
invent intervention success or failure feedback.

Run the local demonstration with:

```bash
uv run python -m evicon.adaptive_smoke
```

The smoke output demonstrates control-loop sequencing and replayability only;
it is not an effectiveness or causal result.
