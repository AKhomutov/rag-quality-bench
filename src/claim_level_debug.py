from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from claim_level_results import (
    ClaimAuditItem,
    ClaimLevelJudgeResult,
    QuestionCoverageItem,
    claim_level_passed,
    parse_claim_level_judge_result,
)
from clients.ollama_client import OllamaGenerationResult


@dataclass(frozen=True)
class ClaimLevelDebugTrace:
    case_id: str
    trial_number: int | None
    prompt_fixture_path: str
    prompt_sha256: str
    sut_prompt: str
    candidate_answer: str
    candidate_sha256: str
    authoritative_source: str
    reference_facts: list[str]
    judge_prompt: str
    raw_response: str
    result: ClaimLevelJudgeResult
    passed: bool
    model: str
    model_digest: str | None
    temperature: float
    keep_alive: int | str | None
    claim_level_schema_version: int | None
    num_ctx: int | None = None
    generation: OllamaGenerationResult | None = None


@dataclass(frozen=True)
class ClaimLevelDebugErrorTrace:
    case_id: str
    trial_number: int | None
    prompt_fixture_path: str
    prompt_sha256: str
    sut_prompt: str
    candidate_answer: str
    candidate_sha256: str
    authoritative_source: str
    reference_facts: list[str]
    judge_prompt: str
    raw_response: str
    error: str
    model: str
    model_digest: str | None
    temperature: float
    keep_alive: int | str | None
    claim_level_schema_version: int | None
    num_ctx: int | None = None
    generation: OllamaGenerationResult | None = None


def _claim_to_object(item: ClaimAuditItem) -> dict[str, object]:
    return {
        "claim": item.claim,
        "candidate_excerpt": item.candidate_excerpt,
        "support_status": item.support_status,
        "reference_fact_ids": list(item.reference_fact_ids),
        "provenance_issue": item.provenance_issue,
        "explanation": item.explanation,
    }


def _coverage_to_object(item: QuestionCoverageItem) -> dict[str, object]:
    return {
        "question_number": item.question_number,
        "status": item.status,
        "explanation": item.explanation,
    }


def _result_to_object(result: ClaimLevelJudgeResult) -> dict[str, object]:
    return {
        "claims": [_claim_to_object(item) for item in result.claims],
        "question_coverage": [_coverage_to_object(item) for item in result.question_coverage],
        "final_table_present": result.final_table_present,
        "final_summary_present": result.final_summary_present,
        "rationale": result.rationale,
    }


def _generation_to_object(
    generation: OllamaGenerationResult | None,
    raw_response: str,
) -> dict[str, object] | None:
    if generation is None:
        return None
    return {
        "done": generation.done,
        "done_reason": generation.done_reason,
        "prompt_eval_count": generation.prompt_eval_count,
        "eval_count": generation.eval_count,
        "total_duration": generation.total_duration,
        "load_duration": generation.load_duration,
        "prompt_eval_duration": generation.prompt_eval_duration,
        "eval_duration": generation.eval_duration,
        "raw_response_chars": len(raw_response),
    }


def claim_level_debug_trace_to_object(trace: ClaimLevelDebugTrace) -> dict[str, object]:
    return {
        "case_id": trace.case_id,
        "trial_number": trace.trial_number,
        "prompt_fixture_path": trace.prompt_fixture_path,
        "prompt_sha256": trace.prompt_sha256,
        "sut_prompt": trace.sut_prompt,
        "candidate_answer": trace.candidate_answer,
        "candidate_sha256": trace.candidate_sha256,
        "authoritative_source": trace.authoritative_source,
        "reference_facts": list(trace.reference_facts),
        "judge_prompt": trace.judge_prompt,
        "raw_response": trace.raw_response,
        "result": _result_to_object(trace.result),
        "passed": trace.passed,
        "claim_level_schema_version": trace.claim_level_schema_version,
        "judge": {
            "model": trace.model,
            "model_digest": trace.model_digest,
            "temperature": trace.temperature,
            "keep_alive": trace.keep_alive,
            "num_ctx": trace.num_ctx,
        },
        "generation": _generation_to_object(trace.generation, trace.raw_response),
    }


def claim_level_debug_error_trace_to_object(
    trace: ClaimLevelDebugErrorTrace,
) -> dict[str, object]:
    return {
        "case_id": trace.case_id,
        "trial_number": trace.trial_number,
        "prompt_fixture_path": trace.prompt_fixture_path,
        "prompt_sha256": trace.prompt_sha256,
        "sut_prompt": trace.sut_prompt,
        "candidate_answer": trace.candidate_answer,
        "candidate_sha256": trace.candidate_sha256,
        "authoritative_source": trace.authoritative_source,
        "reference_facts": list(trace.reference_facts),
        "judge_prompt": trace.judge_prompt,
        "raw_response": trace.raw_response,
        "error": trace.error,
        "claim_level_schema_version": trace.claim_level_schema_version,
        "judge": {
            "model": trace.model,
            "model_digest": trace.model_digest,
            "temperature": trace.temperature,
            "keep_alive": trace.keep_alive,
            "num_ctx": trace.num_ctx,
        },
        "generation": _generation_to_object(trace.generation, trace.raw_response),
    }


def append_claim_level_debug_trace(
    path: Path,
    trace: ClaimLevelDebugTrace | ClaimLevelDebugErrorTrace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(trace, ClaimLevelDebugTrace):
        payload = claim_level_debug_trace_to_object(trace)
    else:
        payload = claim_level_debug_error_trace_to_object(trace)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as file:
        file.write(line)
        file.write("\n")


def parse_and_trace_claim_level_judge_response(
    *,
    path: Path,
    case_id: str,
    trial_number: int | None,
    prompt_fixture_path: str,
    prompt_sha256: str,
    sut_prompt: str,
    candidate_answer: str,
    candidate_sha256: str,
    authoritative_source: str,
    reference_facts: list[str],
    judge_prompt: str,
    raw_response: str,
    model: str,
    model_digest: str | None,
    temperature: float,
    keep_alive: int | str | None,
    claim_level_schema_version: int | None,
    num_ctx: int | None = None,
    generation: OllamaGenerationResult | None = None,
) -> ClaimLevelJudgeResult:
    try:
        result = parse_claim_level_judge_result(raw_response)
    except (TypeError, ValueError) as exc:
        append_claim_level_debug_trace(
            path,
            ClaimLevelDebugErrorTrace(
                case_id=case_id,
                trial_number=trial_number,
                prompt_fixture_path=prompt_fixture_path,
                prompt_sha256=prompt_sha256,
                sut_prompt=sut_prompt,
                candidate_answer=candidate_answer,
                candidate_sha256=candidate_sha256,
                authoritative_source=authoritative_source,
                reference_facts=list(reference_facts),
                judge_prompt=judge_prompt,
                raw_response=raw_response,
                error=str(exc),
                model=model,
                model_digest=model_digest,
                temperature=temperature,
                keep_alive=keep_alive,
                claim_level_schema_version=claim_level_schema_version,
                num_ctx=num_ctx,
                generation=generation,
            ),
        )
        msg = (
            f"claim-level judge execution error for {case_id!r}"
            f" trial={trial_number!r}: {exc}\n"
            f"raw response:\n{raw_response}"
        )
        raise RuntimeError(msg) from exc

    append_claim_level_debug_trace(
        path,
        ClaimLevelDebugTrace(
            case_id=case_id,
            trial_number=trial_number,
            prompt_fixture_path=prompt_fixture_path,
            prompt_sha256=prompt_sha256,
            sut_prompt=sut_prompt,
            candidate_answer=candidate_answer,
            candidate_sha256=candidate_sha256,
            authoritative_source=authoritative_source,
            reference_facts=list(reference_facts),
            judge_prompt=judge_prompt,
            raw_response=raw_response,
            result=result,
            passed=claim_level_passed(result),
            model=model,
            model_digest=model_digest,
            temperature=temperature,
            keep_alive=keep_alive,
            claim_level_schema_version=claim_level_schema_version,
            num_ctx=num_ctx,
            generation=generation,
        ),
    )
    return result
