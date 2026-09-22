"""Optional JSON Schema validation for ``path_b_evidence`` payloads."""
from __future__ import annotations

import json
from typing import Any

import jsonschema

from cross.config.paths import CROSS_ROOT

_SCHEMA_CACHE: dict[str, Any] | None = None


def path_b_evidence_schema() -> dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        schema_path = CROSS_ROOT / "schemas" / "path_b_evidence.schema.json"
        with open(schema_path, encoding="utf-8") as f:
            _SCHEMA_CACHE = json.load(f)
    return _SCHEMA_CACHE


def validate_path_b_evidence(evidence: dict[str, Any]) -> list[str]:
    """Validate full ``path_b_evidence`` object against schemas/path_b_evidence.schema.json."""
    if not isinstance(evidence, dict):
        return ["evidence root must be an object"]
    schema = path_b_evidence_schema()
    errs: list[str] = []
    v = jsonschema.Draft7Validator(schema)
    for e in sorted(v.iter_errors(evidence), key=lambda x: x.path):
        path_s = "/".join(str(p) for p in e.absolute_path)
        errs.append(f"{path_s}: {e.message}" if path_s else e.message)
    return errs


_FLOW_CORR_CACHE: dict[str, Any] | None = None
_FLOW_SEG_CACHE: dict[str, Any] | None = None


def flow_correspondence_schema() -> dict[str, Any]:
    global _FLOW_CORR_CACHE
    if _FLOW_CORR_CACHE is None:
        schema_path = CROSS_ROOT / "schemas" / "flow_correspondence.schema.json"
        with open(schema_path, encoding="utf-8") as f:
            _FLOW_CORR_CACHE = json.load(f)
    return _FLOW_CORR_CACHE


def flow_segment_list_schema() -> dict[str, Any]:
    global _FLOW_SEG_CACHE
    if _FLOW_SEG_CACHE is None:
        schema_path = CROSS_ROOT / "schemas" / "flow_segment.schema.json"
        with open(schema_path, encoding="utf-8") as f:
            _FLOW_SEG_CACHE = json.load(f)
    return _FLOW_SEG_CACHE


def validate_flow_correspondence(decoded: list[Any]) -> list[str]:
    if not isinstance(decoded, list):
        return ["decoded_correspondences must be a list"]
    schema = flow_correspondence_schema()
    errs: list[str] = []
    v = jsonschema.Draft7Validator(schema)
    for e in sorted(v.iter_errors(decoded), key=lambda x: x.path):
        path_s = "/".join(str(p) for p in e.absolute_path)
        errs.append(f"{path_s}: {e.message}" if path_s else e.message)
    return errs


def validate_flow_segment_list(segments: list[Any]) -> list[str]:
    if not isinstance(segments, list):
        return ["flow segments must be a list"]
    schema = flow_segment_list_schema()
    errs: list[str] = []
    v = jsonschema.Draft7Validator(schema)
    for e in sorted(v.iter_errors(segments), key=lambda x: x.path):
        path_s = "/".join(str(p) for p in e.absolute_path)
        errs.append(f"{path_s}: {e.message}" if path_s else e.message)
    return errs


__all__ = [
    "path_b_evidence_schema",
    "validate_path_b_evidence",
    "flow_correspondence_schema",
    "flow_segment_list_schema",
    "validate_flow_correspondence",
    "validate_flow_segment_list",
]
