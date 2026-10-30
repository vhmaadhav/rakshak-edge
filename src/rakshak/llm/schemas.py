"""Result schemas for cloud labelling sessions, and a small dependency-free validator.

A result file is JSON Lines. Each line is one object: the schema fields plus a top-level ``id``
that must match an id in the matching task chunk. The validator implements only the JSON Schema
subset used here (type, enum, required, properties, additionalProperties, items, min/max length
and items).
"""

from __future__ import annotations

import copy
from typing import Any

POLICY_LABELS = ["benign_transactional", "benign_personal", "promotion", "scam", "unclear"]
SCAM_TYPES = [
    "kyc_update",
    "account_block",
    "upi_collect",
    "otp_theft",
    "prize_lottery",
    "job_task",
    "investment",
    "loan",
    "electricity",
    "parcel_customs",
    "digital_arrest",
    "remote_access",
    "impersonation_other",
    "none",
]
CONFIDENCE = ["low", "medium", "high"]
ZERO_SHOT_LABELS = ["scam", "not_scam"]

_BOOLEAN_FIELDS = [
    "urgency",
    "otp_pin_request",
    "payment_request",
    "remote_access_app",
    "link_present",
]


def _annotate_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "policy_label": {"type": "string", "enum": POLICY_LABELS},
        "scam_type": {"type": "string", "enum": SCAM_TYPES},
    }
    for name in _BOOLEAN_FIELDS:
        properties[name] = {"type": "boolean"}
    properties["impersonated_brand"] = {"type": ["string", "null"], "maxLength": 80}
    properties["evidence_spans"] = {
        "type": "array",
        "maxItems": 4,
        "items": {"type": "string", "minLength": 1, "maxLength": 120},
    }
    properties["confidence"] = {"type": "string", "enum": CONFIDENCE}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


ANNOTATE_SCHEMA: dict[str, Any] = _annotate_schema()

ZERO_SHOT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ZERO_SHOT_LABELS},
        "confidence": {"type": "string", "enum": CONFIDENCE},
    },
    "required": ["label", "confidence"],
    "additionalProperties": False,
}

# The audit job uses the same annotation schema as the relabel job.
SCHEMAS: dict[str, dict[str, Any]] = {
    "annotate": ANNOTATE_SCHEMA,
    "audit": ANNOTATE_SCHEMA,
    "zero_shot": ZERO_SHOT_SCHEMA,
}

# Schemas whose evidence_spans must be verbatim substrings of the message.
SPAN_SCHEMAS = frozenset({"annotate", "audit"})


def with_id(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of an object schema with the required top-level ``id`` field added."""
    out = copy.deepcopy(schema)
    out["properties"] = {
        "id": {"type": "string", "minLength": 1, "maxLength": 64},
        **out["properties"],
    }
    out["required"] = ["id", *out["required"]]
    return out


_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
}


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Return a list of human-readable problems; an empty list means the value is valid."""
    types = schema.get("type")
    if types is not None:
        names = types if isinstance(types, list) else [types]
        if not any(_TYPE_CHECKS[name](value) for name in names):
            got = type(value).__name__
            return [f"{path}: expected {'|'.join(names)}, got {got}"]

    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not an allowed value")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(validate(item, schema["items"], f"{path}[{index}]"))
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing '{key}'")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: unexpected key '{key}'")
        for key, sub_schema in properties.items():
            if key in value:
                errors.extend(validate(value[key], sub_schema, f"{path}.{key}"))
    return errors
