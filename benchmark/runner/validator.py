"""Validates each stage's output dict against an inline schema."""
from __future__ import annotations

import jsonschema

# Inline schemas for each stage output — keyed by stage name.
# These are subsets of ground_truth_schema.json focused on the fields
# each stage is responsible for producing.
STAGE_SCHEMAS: dict[str, dict] = {
    "transcription": {
        "type": "object",
        "required": ["transcription_cleaned"],
        "properties": {"transcription_cleaned": {"type": "string"}},
        "additionalProperties": False,
    },
    "summarization": {
        "type": "object",
        "required": ["clinical_summary"],
        "properties": {"clinical_summary": {"type": "string"}},
        "additionalProperties": False,
    },
    "diagnosis": {
        "type": "object",
        "required": ["differential_diagnosis"],
        "properties": {
            "differential_diagnosis": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["condition", "rationale"],
                    "properties": {
                        "condition": {"type": "string"},
                        "pmid": {"type": ["string", "null"]},
                        "rationale": {"type": "string"},
                    },
                },
            }
        },
        "additionalProperties": False,
    },
    "medications": {
        "type": "object",
        "required": ["normalized_medications"],
        "properties": {
            "normalized_medications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["original", "rxnorm_id", "ingredient"],
                    "properties": {
                        "original": {"type": "string"},
                        "rxnorm_id": {"type": ["string", "null"]},
                        "ingredient": {"type": ["string", "null"]},
                    },
                },
            }
        },
        "additionalProperties": False,
    },
    "interactions": {
        "type": "object",
        "required": ["drug_interactions"],
        "properties": {
            "drug_interactions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["drug_a", "drug_b", "severity", "recommendation"],
                    "properties": {
                        "drug_a": {"type": "string"},
                        "drug_b": {"type": "string"},
                        "severity": {"type": "string"},
                        "recommendation": {"type": "string"},
                    },
                },
            }
        },
        "additionalProperties": False,
    },
    "report": {
        "type": "object",
        "required": ["final_report"],
        "properties": {
            "final_report": {
                "type": "object",
                "required": ["subjective", "objective", "assessment", "plan"],
                "properties": {
                    "subjective": {"type": "string"},
                    "objective": {"type": "string"},
                    "assessment": {"type": "string"},
                    "plan": {"type": "string"},
                },
            }
        },
        "additionalProperties": False,
    },
}


def validate(stage: str, output: dict) -> None:
    """Validate a stage output dict against its registered schema.

    Args:
        stage:  One of the six stage names (e.g. "transcription").
        output: The dict returned by the stage's run() function.

    Raises:
        ValueError:                  If stage name is not registered.
        jsonschema.ValidationError:  If output does not match the schema.
    """
    schema = STAGE_SCHEMAS.get(stage)
    if schema is None:
        raise ValueError(f"No schema registered for stage: {stage!r}")
    jsonschema.validate(instance=output, schema=schema)
