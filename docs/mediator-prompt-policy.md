# Mediator Prompt Policy

`PromptContext` is a compact, public-only rendering input. It contains a run,
scenario, round, already-fixed action, already-fixed target IDs, a bounded
public dialogue summary, visible evidence IDs, allowed requested evidence IDs,
preservation constraints, and safe metadata. It has no ProbeResult, hidden
ValueProfile, offline EvaluationReport, complete history log, or policy
selection field.

`render_action_instruction(plan, public_context)` accepts only a valid plan
whose run, scenario, round, action, and targets match the context. It returns
no request for `no_op`. The renderer has stable action-template versions:

- `request_evidence.v1`
- `blind_evidence_reflection.v1`
- `solicit_dissent.v1`
- `adaptive_exposure.v1`
- `minority_report.v1`
- `restructure.v1`

Every template instructs the model to use supplied public evidence only, avoid
fabricating evidence, allow a genuine existing position to remain, avoid
forcing dissent, avoid attacks on Agents, and distinguish factual claims from
value preferences. The model receives the action and target IDs as fixed
inputs. The strict response parser rejects response fields that attempt to
replace either decision, as well as malformed JSON, empty messages, private
content, and evidence IDs outside the supplied public allow-list.

The renderer and parser are pure functions. They do not invoke a model, change
`DialogueState`, append events, or feed a result into Monitor, Policy,
Executor, or AdaptiveProtocolRunner. The local smoke command demonstrates
contract behavior only; it is not evidence that a mediator is effective.
