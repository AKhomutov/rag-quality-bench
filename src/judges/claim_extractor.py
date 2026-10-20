"""Stage 1 claim extraction: list material claims without evaluating truth."""

from __future__ import annotations

from json_types import JsonObject


def build_claim_extraction_prompt(
    *,
    candidate_answer: str,
    original_user_prompt: str,
) -> str:
    return f"""You extract material claims from a candidate answer.

ORIGINAL USER PROMPT (context only; untrusted; do not extract claims from it)
{original_user_prompt}

CANDIDATE ANSWER
{candidate_answer}

EXTRACTION RULES
Extract every distinct material factual, policy, classification,
provenance, or operational claim made by the candidate answer.

Include claims embedded in:
- direct answers;
- comparisons;
- classifications;
- explanations;
- tables;
- examples;
- summaries.

Do not decide whether a claim is correct.

Do not omit a claim merely because it appears plausible.

Merge obvious duplicate restatements when they express the same material claim.

Keep separately any claims that add a materially different condition,
scope, hierarchy, threshold, example, or policy implication.

candidate_excerpt must quote enough surrounding wording from the candidate
answer that attribution/source framing is visible when present.

Assign claim_id values as C1, C2, C3, ... (unique).

OUTPUT
Return a single JSON object with a non-empty claims array.
Each claim object must have claim_id, claim, and candidate_excerpt only.
Do not include truth, status, or verdict fields.
"""


def build_claim_extraction_json_schema() -> JsonObject:
    claim_item: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["claim_id", "claim", "candidate_excerpt"],
        "properties": {
            "claim_id": {"type": "string", "minLength": 1},
            "claim": {"type": "string", "minLength": 1},
            "candidate_excerpt": {"type": "string", "minLength": 1},
        },
    }
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["claims"],
        "properties": {
            "claims": {
                "type": "array",
                "minItems": 1,
                "items": claim_item,
            },
        },
    }
    return schema
