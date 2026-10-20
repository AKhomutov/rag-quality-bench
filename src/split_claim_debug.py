"""Debug artifact writers for the split claim pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import cast

from clients.ollama_client import OllamaGenerationResult
from split_claim_results import (
    BaseClaimJudgement,
    DependencyJudgement,
    ExtractedClaim,
    FinalClaimJudgement,
    SplitClaimMetrics,
)
from workspace_setup import PROJECT_ROOT

REPORTS_ROOT = PROJECT_ROOT / "reports"


def create_split_claim_debug_run_dir(reports_root: Path | None = None) -> Path:
    root = REPORTS_ROOT if reports_root is None else reports_root
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = root / "judge_debug" / f"{timestamp}_split_claim"
    (run_dir / "base_batches").mkdir(parents=True, exist_ok=True)
    (run_dir / "dependency_batches").mkdir(parents=True, exist_ok=True)
    return run_dir


def _generation_to_object(generation: OllamaGenerationResult) -> dict[str, object]:
    return {
        "done": generation.done,
        "done_reason": generation.done_reason,
        "prompt_eval_count": generation.prompt_eval_count,
        "eval_count": generation.eval_count,
        "total_duration": generation.total_duration,
        "load_duration": generation.load_duration,
        "prompt_eval_duration": generation.prompt_eval_duration,
        "eval_duration": generation.eval_duration,
        "raw_response_chars": len(generation.response),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_extraction_artifact(
    run_dir: Path,
    *,
    prompt: str,
    generation: OllamaGenerationResult,
    claims: list[ExtractedClaim],
    model: str,
    model_digest: str,
    temperature: int,
    num_ctx: int,
    keep_alive: int | str | None,
    wall_time_seconds: float,
) -> None:
    payload: dict[str, object] = {
        "stage": "extraction",
        "model": model,
        "model_digest": model_digest,
        "temperature": temperature,
        "num_ctx": num_ctx,
        "keep_alive": keep_alive,
        "prompt": prompt,
        "raw_response": generation.response,
        "completion_metadata": _generation_to_object(generation),
        "parsed_claim_count": len(claims),
        "parsed_claims": [cast(dict[str, object], asdict(item)) for item in claims],
        "wall_time_seconds": wall_time_seconds,
    }
    _write_json(run_dir / "extraction.json", payload)


def write_base_batch_artifact(
    run_dir: Path,
    *,
    batch_index: int,
    claims: list[ExtractedClaim],
    prompt: str,
    generation: OllamaGenerationResult,
    judgements: list[BaseClaimJudgement],
    model: str,
    model_digest: str,
    temperature: int,
    num_ctx: int,
    keep_alive: int | str | None,
    wall_time_seconds: float,
) -> None:
    payload: dict[str, object] = {
        "stage": "base_claim_batch",
        "batch_index": batch_index,
        "claim_ids": [claim.claim_id for claim in claims],
        "extracted_claims": [cast(dict[str, object], asdict(item)) for item in claims],
        "model": model,
        "model_digest": model_digest,
        "temperature": temperature,
        "num_ctx": num_ctx,
        "keep_alive": keep_alive,
        "prompt": prompt,
        "raw_response": generation.response,
        "completion_metadata": _generation_to_object(generation),
        "parsed_judgements": [cast(dict[str, object], asdict(item)) for item in judgements],
        "wall_time_seconds": wall_time_seconds,
    }
    _write_json(run_dir / "base_batches" / f"batch_{batch_index:03d}.json", payload)


def write_dependency_batch_artifact(
    run_dir: Path,
    *,
    batch_index: int,
    claims: list[ExtractedClaim],
    prompt: str,
    generation: OllamaGenerationResult,
    judgements: list[DependencyJudgement],
    model: str,
    model_digest: str,
    temperature: int,
    num_ctx: int,
    keep_alive: int | str | None,
    wall_time_seconds: float,
) -> None:
    payload: dict[str, object] = {
        "stage": "dependency_batch",
        "batch_index": batch_index,
        "claim_ids": [claim.claim_id for claim in claims],
        "extracted_claims": [cast(dict[str, object], asdict(item)) for item in claims],
        "model": model,
        "model_digest": model_digest,
        "temperature": temperature,
        "num_ctx": num_ctx,
        "keep_alive": keep_alive,
        "prompt": prompt,
        "raw_response": generation.response,
        "completion_metadata": _generation_to_object(generation),
        "parsed_dependencies": [cast(dict[str, object], asdict(item)) for item in judgements],
        "wall_time_seconds": wall_time_seconds,
    }
    _write_json(run_dir / "dependency_batches" / f"batch_{batch_index:03d}.json", payload)


def write_summary_artifact(
    run_dir: Path,
    *,
    candidate_sha256: str,
    candidate_length: int,
    metrics: SplitClaimMetrics,
    passed: bool,
    finals: list[FinalClaimJudgement],
    extraction_wall_time: float,
    base_judge_wall_time: float,
    dependency_wall_time: float,
    overall_wall_time: float,
    base_batch_count: int,
    dependency_batch_count: int,
    dependency_eligible_count: int,
    average_base_batch_duration: float,
    average_claim_equivalent_base_time: float,
    average_dependency_batch_duration: float,
    execution_errors: list[str],
) -> None:
    payload: dict[str, object] = {
        "candidate_sha256": candidate_sha256,
        "candidate_length": candidate_length,
        "verdict": "PASS" if passed else "FAIL",
        "metrics": cast(dict[str, object], asdict(metrics)),
        "final_claims": [cast(dict[str, object], asdict(item)) for item in finals],
        "extraction_wall_time_seconds": extraction_wall_time,
        "base_judge_wall_time_seconds": base_judge_wall_time,
        "dependency_wall_time_seconds": dependency_wall_time,
        "overall_wall_time_seconds": overall_wall_time,
        "base_batch_count": base_batch_count,
        "dependency_batch_count": dependency_batch_count,
        "dependency_eligible_count": dependency_eligible_count,
        "average_base_batch_duration_seconds": average_base_batch_duration,
        "average_claim_equivalent_base_time_seconds": average_claim_equivalent_base_time,
        "average_dependency_batch_duration_seconds": average_dependency_batch_duration,
        "execution_errors": execution_errors,
    }
    _write_json(run_dir / "summary.json", payload)
