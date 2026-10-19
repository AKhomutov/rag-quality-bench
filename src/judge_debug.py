from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from judge_results import (
    JudgeDimension,
    JudgeScore,
    judge_passed,
    parse_judge_score,
    total_score,
)


@dataclass(frozen=True)
class JudgeDebugTrace:
    case_id: str
    trial_number: int | None
    candidate_answer: str
    reference_facts: list[str]
    case_instruction: str | None
    prompt: str
    raw_response: str
    score: JudgeScore
    total_score: int
    threshold: int
    passed: bool
    model: str
    temperature: float


@dataclass(frozen=True)
class JudgeDebugErrorTrace:
    case_id: str
    trial_number: int | None
    candidate_answer: str
    reference_facts: list[str]
    case_instruction: str | None
    prompt: str
    raw_response: str
    error: str
    model: str
    temperature: float


def create_judge_debug_run_dir(reports_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = reports_root / "judge_debug" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def case_trace_path(run_dir: Path, case_id: str) -> Path:
    return run_dir / f"{case_id}.jsonl"


def _dimension_to_object(dimension: JudgeDimension) -> dict[str, object]:
    return {
        "score": dimension.score,
        "evidence": dimension.evidence,
    }


def _score_to_object(score: JudgeScore) -> dict[str, object]:
    fact_checks: list[dict[str, object]] = []
    for check in score.fact_checks:
        fact_checks.append(
            {
                "fact": check.fact,
                "candidate_position": check.candidate_position,
                "assessment": check.assessment,
            }
        )
    return {
        "groundedness": _dimension_to_object(score.groundedness),
        "source_authority": _dimension_to_object(score.source_authority),
        "unsupported_reasoning": _dimension_to_object(score.unsupported_reasoning),
        "answer_correctness": _dimension_to_object(score.answer_correctness),
        "fact_checks": fact_checks,
        "rationale": score.rationale,
    }


def judge_debug_trace_to_object(trace: JudgeDebugTrace) -> dict[str, object]:
    return {
        "case_id": trace.case_id,
        "trial_number": trace.trial_number,
        "candidate_answer": trace.candidate_answer,
        "reference_facts": list(trace.reference_facts),
        "case_instruction": trace.case_instruction,
        "prompt": trace.prompt,
        "raw_response": trace.raw_response,
        "score": _score_to_object(trace.score),
        "total_score": trace.total_score,
        "threshold": trace.threshold,
        "passed": trace.passed,
        "judge": {
            "model": trace.model,
            "temperature": trace.temperature,
        },
    }


def judge_debug_error_trace_to_object(trace: JudgeDebugErrorTrace) -> dict[str, object]:
    return {
        "case_id": trace.case_id,
        "trial_number": trace.trial_number,
        "candidate_answer": trace.candidate_answer,
        "reference_facts": list(trace.reference_facts),
        "case_instruction": trace.case_instruction,
        "prompt": trace.prompt,
        "raw_response": trace.raw_response,
        "error": trace.error,
        "judge": {
            "model": trace.model,
            "temperature": trace.temperature,
        },
    }


def append_judge_debug_trace(
    path: Path,
    trace: JudgeDebugTrace | JudgeDebugErrorTrace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(trace, JudgeDebugTrace):
        payload = judge_debug_trace_to_object(trace)
    else:
        payload = judge_debug_error_trace_to_object(trace)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as file:
        file.write(line)
        file.write("\n")


def parse_and_trace_judge_response(
    *,
    path: Path,
    case_id: str,
    trial_number: int | None,
    candidate_answer: str,
    reference_facts: list[str],
    case_instruction: str | None,
    prompt: str,
    raw_response: str,
    threshold: int,
    model: str,
    temperature: float,
) -> JudgeScore:
    try:
        score = parse_judge_score(raw_response)
    except (TypeError, ValueError) as exc:
        append_judge_debug_trace(
            path,
            JudgeDebugErrorTrace(
                case_id=case_id,
                trial_number=trial_number,
                candidate_answer=candidate_answer,
                reference_facts=list(reference_facts),
                case_instruction=case_instruction,
                prompt=prompt,
                raw_response=raw_response,
                error=str(exc),
                model=model,
                temperature=temperature,
            ),
        )
        msg = (
            f"judge execution error for {case_id!r}"
            f" trial={trial_number!r}: {exc}\n"
            f"raw response:\n{raw_response}"
        )
        raise RuntimeError(msg) from exc

    append_judge_debug_trace(
        path,
        JudgeDebugTrace(
            case_id=case_id,
            trial_number=trial_number,
            candidate_answer=candidate_answer,
            reference_facts=list(reference_facts),
            case_instruction=case_instruction,
            prompt=prompt,
            raw_response=raw_response,
            score=score,
            total_score=total_score(score),
            threshold=threshold,
            passed=judge_passed(score, threshold),
            model=model,
            temperature=temperature,
        ),
    )
    return score
