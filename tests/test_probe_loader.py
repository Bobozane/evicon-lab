"""Strict ProbeSet configuration loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon.models import ProbeSet
from evicon.probe_loader import ProbeLoadError, ProbeSetLoader


def probe_text(
    *,
    duplicate_id: bool = False,
    omit_transparency: bool = False,
    invalid_scale: bool = False,
    mismatch_dimension: bool = False,
) -> str:
    second_dimension = "autonomy" if mismatch_dimension else "transparency"
    fairness_holdout_id = "fairness-standard" if duplicate_id else "fairness-holdout"
    scale = '["1"]' if invalid_scale else '["1", "2", "3", "4", "5", "6", "7"]'
    transparency_items = "" if omit_transparency else f'''
[[items]]
probe_id = "transparency-standard"
text = "Explain reasons."
dimension = "{second_dimension}"
response_scale = {scale}
is_holdout = false

[[items]]
probe_id = "transparency-holdout"
text = "Keep reasons opaque."
dimension = "{second_dimension}"
response_scale = {scale}
is_holdout = true
reverse_scored = true
'''
    return f'''probe_set_id = "fixture-probes"
dimensions = ["fairness", "transparency"]
version = "1.0"

[[items]]
probe_id = "fairness-standard"
text = "Treat cases consistently."
dimension = "fairness"
response_scale = {scale}
is_holdout = false

[[items]]
probe_id = "{fairness_holdout_id}"
text = "Avoid unjustified unequal treatment."
dimension = "fairness"
response_scale = {scale}
is_holdout = true
{transparency_items}'''


def test_checked_in_probe_set_loads_and_marks_holdouts() -> None:
    project_root = Path(__file__).resolve().parents[1]
    probe_set = ProbeSetLoader.load(project_root / "configs/probes/smoke_probe.toml")

    assert probe_set.dimensions == ["fairness", "transparency"]
    assert len(probe_set.items) == 4
    assert [item.probe_id for item in probe_set.items_for_holdout(True)] == [
        "fairness-holdout",
        "transparency-holdout",
    ]
    assert type(probe_set).model_validate_json(probe_set.model_dump_json()) == probe_set


def test_probe_set_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ProbeSet.model_validate(
            {
                "probe_set_id": "strict-probes",
                "dimensions": ["fairness"],
                "items": [
                    {
                        "probe_id": "fairness-one",
                        "text": "A strict fixture item.",
                        "dimension": "fairness",
                        "response_scale": ["1", "2"],
                        "is_holdout": False,
                    }
                ],
                "version": "1.0",
                "unexpected": True,
            }
        )


@pytest.mark.parametrize(
    ("name", "contents", "field"),
    [
        ("duplicate.toml", probe_text(duplicate_id=True), "probe_id"),
        ("missing_dimension.toml", probe_text(omit_transparency=True), "dimension"),
        ("bad_scale.toml", probe_text(invalid_scale=True), "response_scale"),
        ("mismatch.toml", probe_text(mismatch_dimension=True), "dimensions"),
    ],
)
def test_invalid_probe_sets_are_rejected(
    tmp_path: Path,
    name: str,
    contents: str,
    field: str,
) -> None:
    path = tmp_path / name
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ProbeLoadError) as error:
        ProbeSetLoader.load(path)

    assert str(path) in str(error.value)
    assert field in str(error.value)
