"""Conservative Provider-facing schema for source-relation comprehension.

The canonical response model and parser live in
``conformity_source_relation_comprehension_calibration_v1`` and are not
changed by this module.  This file only describes the wire schema sent to an
OpenAI-compatible endpoint after the original v1 schema was rejected with a
``response_format_unsupported`` error.

The intermediary's accepted schema subset is intentionally treated as
unknown.  The wire contract therefore avoids tuple validation (``prefixItems``),
singleton constraints (``const``), and definitions/references (``$defs`` and
``$ref``).  The local canonical parser remains responsible for exact ordering,
cardinality, identifiers, and cross-field consistency.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final

from .conformity_source_relation_comprehension_calibration_v1 import (
    RESPONSE_SCHEMA_SHA256 as CANONICAL_RESPONSE_SCHEMA_SHA256,
    SCHEMA_NAME as CANONICAL_SCHEMA_NAME,
)


# The provider schema is a new wire identity.  It deliberately keeps the
# canonical response's semantic version in its name while identifying the
# subset projection separately.
PROVIDER_SCHEMA_NAME: Final[str] = (
    "conformity_source_relation_comprehension_response_v1_provider_subset_v2"
)
PROVIDER_SCHEMA_VERSION: Final[str] = (
    "conformity_source_relation_comprehension_response_provider_subset.v2"
)

# The protocol/template remain the v1 public task.  Only the Provider-facing
# JSON Schema is projected here.
CANONICAL_SCHEMA_VERSION: Final[str] = (
    "conformity_source_relation_comprehension_response.v1"
)

REMOVED_PROVIDER_KEYWORDS: Final[tuple[str, ...]] = (
    "prefixItems",
    "const",
    "$defs",
    "$ref",
)


def _root_assignment_schema() -> dict[str, Any]:
    """Return an inlined object schema for one public root assignment."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "content_id": {
                "type": "string",
                "enum": ["message-a", "message-b"],
            },
            # A JSON Schema type union is supported by the conservative
            # provider subset used by the other amended protocols.
            "source_root_id": {"type": ["string", "null"]},
        },
        "required": ["content_id", "source_root_id"],
    }


# This is intentionally hand-authored rather than generated from Pydantic.
# Generation would re-introduce the constructs that caused the v1 request to
# be rejected by the intermediary.  The canonical parser still validates the
# exact same logical response after transport.
PROVIDER_RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "visible_social_message_count": {
            "type": "integer",
            "enum": [2],
        },
        "source_relation": {
            "type": "string",
            "enum": [
                "not_shown",
                "same_shown_root",
                "distinct_shown_roots",
            ],
        },
        "visible_social_root_count": {
            "type": "integer",
            "enum": [0, 1, 2],
        },
        "social_root_assignments": {
            "type": "array",
            "items": _root_assignment_schema(),
            "minItems": 2,
            "maxItems": 2,
        },
        "direct_evidence_card_ids": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["evidence-card-01"],
            },
            "maxItems": 1,
        },
        "support_kind": {
            "type": "string",
            "enum": ["social_reports_only", "direct_evidence_present"],
        },
        "source_structure_alone_decides_truth": {
            "type": "boolean",
            "enum": [False],
        },
    },
    "required": [
        "visible_social_message_count",
        "source_relation",
        "visible_social_root_count",
        "social_root_assignments",
        "direct_evidence_card_ids",
        "support_kind",
        "source_structure_alone_decides_truth",
    ],
}


def _walk_schema(value: object) -> tuple[str, ...]:
    """Collect forbidden keyword names without serializing public material."""

    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in REMOVED_PROVIDER_KEYWORDS:
                found.append(key)
            found.extend(_walk_schema(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_schema(child))
    return tuple(found)


def provider_schema_forbidden_keywords() -> tuple[str, ...]:
    """Return any disallowed JSON Schema keywords found in the wire schema."""

    return _walk_schema(PROVIDER_RESPONSE_SCHEMA)


if provider_schema_forbidden_keywords():  # pragma: no cover - import invariant
    raise RuntimeError("provider_subset_schema_contains_removed_keyword")


PROVIDER_RESPONSE_SCHEMA_SHA256: Final[str] = hashlib.sha256(
    json.dumps(
        PROVIDER_RESPONSE_SCHEMA,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

# Conventional aliases make the module usable by compatibility runners while
# keeping the provider/canonical distinction explicit above.
RESPONSE_SCHEMA: Final[dict[str, Any]] = PROVIDER_RESPONSE_SCHEMA
RESPONSE_SCHEMA_SHA256: Final[str] = PROVIDER_RESPONSE_SCHEMA_SHA256
SCHEMA_NAME: Final[str] = PROVIDER_SCHEMA_NAME
SCHEMA_VERSION: Final[str] = PROVIDER_SCHEMA_VERSION


__all__ = [
    "CANONICAL_RESPONSE_SCHEMA_SHA256",
    "CANONICAL_SCHEMA_NAME",
    "CANONICAL_SCHEMA_VERSION",
    "PROVIDER_RESPONSE_SCHEMA",
    "PROVIDER_RESPONSE_SCHEMA_SHA256",
    "PROVIDER_SCHEMA_NAME",
    "PROVIDER_SCHEMA_VERSION",
    "REMOVED_PROVIDER_KEYWORDS",
    "RESPONSE_SCHEMA",
    "RESPONSE_SCHEMA_SHA256",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "provider_schema_forbidden_keywords",
]
