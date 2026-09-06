"""Shared validation helpers for EviCon-Lab data contracts."""

from __future__ import annotations

from pydantic import JsonValue

PUBLIC_AUDIENCE = "*"
Metadata = dict[str, JsonValue]


def normalized_text(value: str, field_name: str) -> str:
    """Return stripped text or reject empty contract fields."""
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def identifier_list(
    values: list[str],
    field_name: str,
    *,
    allow_public_audience: bool = False,
) -> list[str]:
    """Normalize an ordered identifier list and reject ambiguous membership."""
    normalized = [normalized_text(value, field_name) for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    if PUBLIC_AUDIENCE in normalized:
        if not allow_public_audience or normalized != [PUBLIC_AUDIENCE]:
            raise ValueError(
                f"{field_name} may contain {PUBLIC_AUDIENCE!r} only as its sole value"
            )
    return normalized


def can_view(visible_to: list[str], agent_id: str) -> bool:
    """Return whether an explicit audience list grants an agent visibility."""
    return PUBLIC_AUDIENCE in visible_to or agent_id in visible_to
