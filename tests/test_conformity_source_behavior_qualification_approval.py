from evicon.conformity_source_behavior_qualification_approval import DEFAULT_APPROVAL, load_approval, safe_preflight

def test_default_approval_is_accepted_and_compatibility_blocked():
    approval = load_approval(DEFAULT_APPROVAL)
    report = safe_preflight(DEFAULT_APPROVAL)
    assert approval.acceptance_status == "accepted"
    assert report["status"] == "source_behavior_approval_gate_ready"
    assert report["blocking_reasons"] == ["provider_compatibility_not_requested"]
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False

def test_approval_rejects_unknown_fields(tmp_path):
    source = open(DEFAULT_APPROVAL, encoding="utf-8").read()
    path = tmp_path / "approval.toml"
    path.write_text(source + "\nunknown_field = true\n", encoding="utf-8")
    report = safe_preflight(path)
    assert report["status"] == "blocked"
    assert report["blocking_reasons"] == ["source_behavior_approval_invalid"]
