# Provider Smoke

Run the default protected smoke command with:

```bash
uv run python -m evicon.provider_smoke
```

It prints `network_disabled` and does not construct a transport request, write
an event log, write a results directory, run a multi-Agent protocol, or invoke
the adaptive controller.

A real request requires the explicit command:

```bash
uv run python -m evicon.provider_smoke --allow-network
```

Before doing so, set `EVICON_LLM_BASE_URL`, `EVICON_LLM_API_KEY`, and
`EVICON_LLM_MODEL` in the process environment. The command creates exactly
one fixed `request_evidence` mediator request with a small token limit. It
uses the existing renderer and strict parser, so a successful provider call
can still produce an invalid parser result. Parser failure never triggers a
provider retry.

The normal output contains only provider/model/request identifiers, status,
finish reason, token usage, latency, and parser status. It never prints the
prompt or full response. `--print-response` is an explicit debugging option
and limits its response preview to 400 characters. This smoke is a connection
and contract check, not an experiment or effectiveness result.
