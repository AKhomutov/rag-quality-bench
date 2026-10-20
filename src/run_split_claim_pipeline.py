"""Split claim pipeline: extract → batched base judge → dependency → graph → aggregate.

Uses the fixed candidate answer only (SUT is not called).

Reusable entry: ``run_split_claim_evaluation``.
Standalone CLI: ``python -m run_split_claim_pipeline``.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from claim_level_config import (
    CLAIM_LEVEL_NUM_CTX,
    EXPECTED_FIXED_CANDIDATE_LENGTH,
    EXPECTED_FIXED_CANDIDATE_SHA256,
    SPLIT_BASE_BATCH_SIZE,
    SPLIT_CLAIM_PIPELINE_KEEP_ALIVE,
    SPLIT_CLAIM_PIPELINE_TEMPERATURE,
    SPLIT_DEPENDENCY_BATCH_SIZE,
    SUT_GENERATION_MODEL,
    SUT_GENERATION_MODEL_DIGEST,
)
from clients.ollama_client import OllamaClient, OllamaGenerationResult
from fixed_candidate_loader import load_identical_candidate_answer, sha256_text
from json_types import JsonObject
from judges.claim_extractor import (
    build_claim_extraction_json_schema,
    build_claim_extraction_prompt,
)
from judges.per_claim_judge import (
    build_base_claim_batch_json_schema,
    build_base_claim_batch_judge_prompt,
    build_dependency_batch_json_schema,
    build_dependency_batch_judge_prompt,
)
from long_form_claim_audit_results import LongFormClaimAuditResult
from long_form_severity_facts import reference_facts_as_lines
from long_form_severity_prompt import (
    AUTHORITATIVE_DOCUMENT_PATH,
    CASE_ID,
    assert_prompt_fixture_integrity,
    load_sut_prompt,
)
from split_claim_debug import (
    create_split_claim_debug_run_dir,
    write_base_batch_artifact,
    write_dependency_batch_artifact,
    write_dependency_graph_artifact,
    write_extraction_artifact,
    write_summary_artifact,
)
from split_claim_results import (
    BaseClaimJudgement,
    DependencyJudgement,
    ExtractedClaim,
    FinalClaimJudgement,
    build_final_judgements,
    canonicalize_extracted_claim_ids,
    compute_split_claim_metrics,
    parse_base_claim_batch,
    parse_dependency_batch,
    parse_extracted_claims,
    partition_into_batches,
    split_claim_failure_reasons,
    split_claim_passed,
    unsupported_premise_pool,
)
from workspace_setup import PROJECT_ROOT

REPORTS_ROOT = PROJECT_ROOT / "reports"
PREVIOUS_TRACE_PATH = (
    REPORTS_ROOT / "judge_debug" / "20261019T235938" / "mixed_truth_severity_policy_poisoning.jsonl"
)
GENERATE_TIMEOUT_SECONDS = 900
FAKE_SEVERITY_LABELS = ("Severity 0", "Severity 2.5", "Severity 5")


@dataclass(frozen=True)
class TimedGeneration:
    generation: OllamaGenerationResult
    wall_time_seconds: float


def assert_fixed_candidate_identity(candidate_answer: str, candidate_sha256: str) -> None:
    if candidate_sha256 != EXPECTED_FIXED_CANDIDATE_SHA256:
        msg = (
            "candidate SHA-256 mismatch: "
            f"expected {EXPECTED_FIXED_CANDIDATE_SHA256}, got {candidate_sha256}"
        )
        raise ValueError(msg)
    if len(candidate_answer) != EXPECTED_FIXED_CANDIDATE_LENGTH:
        msg = (
            "candidate length mismatch: "
            f"expected {EXPECTED_FIXED_CANDIDATE_LENGTH}, got {len(candidate_answer)}"
        )
        raise ValueError(msg)


def _generate(
    client: OllamaClient,
    *,
    prompt: str,
    json_schema: JsonObject,
) -> TimedGeneration:
    started = time.perf_counter()
    generation = client.generate_with_metadata(
        model=SUT_GENERATION_MODEL,
        prompt=prompt,
        timeout=GENERATE_TIMEOUT_SECONDS,
        json_schema=json_schema,
        keep_alive=SPLIT_CLAIM_PIPELINE_KEEP_ALIVE,
        temperature=SPLIT_CLAIM_PIPELINE_TEMPERATURE,
        num_ctx=CLAIM_LEVEL_NUM_CTX,
    )
    return TimedGeneration(
        generation=generation,
        wall_time_seconds=time.perf_counter() - started,
    )


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _print_generation_metadata(label: str, timed: TimedGeneration, *, verbose: bool) -> None:
    if not verbose:
        return
    gen = timed.generation
    print(f"=== {label} completion metadata ===")
    print(f"done: {gen.done}")
    print(f"done_reason: {gen.done_reason}")
    print(f"prompt_eval_count: {gen.prompt_eval_count}")
    print(f"eval_count: {gen.eval_count}")
    print(f"total_duration_ns: {gen.total_duration}")
    print(f"load_duration_ns: {gen.load_duration}")
    print(f"prompt_eval_duration_ns: {gen.prompt_eval_duration}")
    print(f"eval_duration_ns: {gen.eval_duration}")
    print(f"raw response chars: {len(gen.response)}")
    print(f"wall time: {timed.wall_time_seconds:.2f}s")
    print()


def _claims_for_label(finals: list[FinalClaimJudgement], label: str) -> list[FinalClaimJudgement]:
    lowered = label.lower()
    return [item for item in finals if lowered in f"{item.claim} {item.candidate_excerpt}".lower()]


def _ollama_ps_output() -> str:
    completed = subprocess.run(
        ["ollama", "ps"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or completed.stderr.strip() or "(empty)"


def _error_result(
    *,
    prompt_sha256: str,
    candidate_sha256: str,
    debug_path: Path,
    duration_seconds: float,
    execution_error: str,
) -> LongFormClaimAuditResult:
    return LongFormClaimAuditResult(
        case_id=CASE_ID,
        status="ERROR",
        total_claims=0,
        supported_count=0,
        contradicted_count=0,
        not_in_evidence_count=0,
        derived_count=0,
        provenance_issue_count=0,
        grounded_claim_rate=0.0,
        raw_dependency_edge_count=0,
        normalized_dependency_edge_count=0,
        dependency_cycle_count=0,
        execution_error=execution_error,
        duration_seconds=duration_seconds,
        debug_path=str(debug_path),
        judge_model=SUT_GENERATION_MODEL,
        judge_model_digest=SUT_GENERATION_MODEL_DIGEST,
        num_ctx=CLAIM_LEVEL_NUM_CTX,
        candidate_sha256=candidate_sha256,
        prompt_sha256=prompt_sha256,
        failure_summary=[execution_error],
    )


def run_split_claim_evaluation(*, verbose: bool = True) -> LongFormClaimAuditResult:
    """Run the current split claim pipeline once and return a concise suite result."""
    overall_started = time.perf_counter()
    client = OllamaClient()
    run_dir = create_split_claim_debug_run_dir()
    prompt_sha = ""
    candidate_sha256 = ""

    try:
        try:
            prompt_sha = assert_prompt_fixture_integrity()
            original_user_prompt = load_sut_prompt()
            authoritative_source = AUTHORITATIVE_DOCUMENT_PATH.read_text(encoding="utf-8")
            reference_facts = reference_facts_as_lines()

            candidate_answer = load_identical_candidate_answer(PREVIOUS_TRACE_PATH)
            candidate_sha256 = sha256_text(candidate_answer)
            assert_fixed_candidate_identity(candidate_answer, candidate_sha256)
        except (OSError, TypeError, ValueError) as exc:
            msg = f"fixed candidate / fixture validation failed: {exc}"
            _log(verbose, f"EXECUTION ERROR: {msg}")
            return _error_result(
                prompt_sha256=prompt_sha,
                candidate_sha256=candidate_sha256,
                debug_path=run_dir,
                duration_seconds=time.perf_counter() - overall_started,
                execution_error=msg,
            )

        _log(verbose, f"case: {CASE_ID}")
        _log(verbose, f"prompt fixture SHA-256: {prompt_sha}")
        _log(verbose, f"Candidate SHA-256: {candidate_sha256}")
        _log(verbose, f"Candidate length: {len(candidate_answer)}")
        _log(verbose, "SUT was NOT called.")
        _log(verbose, f"Judge model: {SUT_GENERATION_MODEL}")
        _log(verbose, f"num_ctx: {CLAIM_LEVEL_NUM_CTX}")
        _log(verbose, f"Debug artifacts: {run_dir}")
        _log(verbose, "")

        extraction_prompt = build_claim_extraction_prompt(
            candidate_answer=candidate_answer,
            original_user_prompt=original_user_prompt,
        )
        try:
            extraction_timed = _generate(
                client,
                prompt=extraction_prompt,
                json_schema=build_claim_extraction_json_schema(),
            )
        except Exception as exc:
            msg = f"extraction generate failed: {exc}"
            _log(verbose, f"EXECUTION ERROR: {msg}")
            return _error_result(
                prompt_sha256=prompt_sha,
                candidate_sha256=candidate_sha256,
                debug_path=run_dir,
                duration_seconds=time.perf_counter() - overall_started,
                execution_error=msg,
            )

        extraction_wall = extraction_timed.wall_time_seconds
        _print_generation_metadata("extraction", extraction_timed, verbose=verbose)

        try:
            extracted_raw = parse_extracted_claims(extraction_timed.generation.response)
            extracted = canonicalize_extracted_claim_ids(extracted_raw)
        except (TypeError, ValueError) as exc:
            msg = f"extraction parse error: {exc}"
            _log(verbose, f"EXECUTION ERROR: {msg}")
            return _error_result(
                prompt_sha256=prompt_sha,
                candidate_sha256=candidate_sha256,
                debug_path=run_dir,
                duration_seconds=time.perf_counter() - overall_started,
                execution_error=msg,
            )

        write_extraction_artifact(
            run_dir,
            prompt=extraction_prompt,
            generation=extraction_timed.generation,
            claims=extracted,
            model=SUT_GENERATION_MODEL,
            model_digest=SUT_GENERATION_MODEL_DIGEST,
            temperature=SPLIT_CLAIM_PIPELINE_TEMPERATURE,
            num_ctx=CLAIM_LEVEL_NUM_CTX,
            keep_alive=SPLIT_CLAIM_PIPELINE_KEEP_ALIVE,
            wall_time_seconds=extraction_timed.wall_time_seconds,
        )

        _log(verbose, f"extracted claim count: {len(extracted)}")
        if verbose:
            for item in extracted:
                print(f"  {item.claim_id}: {item.claim}")
            print()

        extracted_by_id = {item.claim_id: item for item in extracted}
        base_by_id: dict[str, BaseClaimJudgement] = {}
        dependency_by_id: dict[str, DependencyJudgement] = {}
        base_batch_durations: list[float] = []
        dependency_batch_durations: list[float] = []

        base_started = time.perf_counter()
        base_batches = partition_into_batches(extracted, SPLIT_BASE_BATCH_SIZE)
        _log(verbose, f"base batch count: {len(base_batches)}")

        for batch_index, batch in enumerate(base_batches, start=1):
            base_prompt = build_base_claim_batch_judge_prompt(
                authoritative_source=authoritative_source,
                reference_facts=reference_facts,
                claims=batch,
            )
            try:
                timed = _generate(
                    client,
                    prompt=base_prompt,
                    json_schema=build_base_claim_batch_json_schema(),
                )
            except Exception as exc:
                msg = f"base batch {batch_index:03d} generate failed: {exc}"
                _log(verbose, f"EXECUTION ERROR: {msg}")
                return _error_result(
                    prompt_sha256=prompt_sha,
                    candidate_sha256=candidate_sha256,
                    debug_path=run_dir,
                    duration_seconds=time.perf_counter() - overall_started,
                    execution_error=msg,
                )
            base_batch_durations.append(timed.wall_time_seconds)
            expected_ids = frozenset(claim.claim_id for claim in batch)
            try:
                base_judgements = parse_base_claim_batch(
                    timed.generation.response,
                    expected_claim_ids=expected_ids,
                )
            except (TypeError, ValueError) as exc:
                msg = f"base batch {batch_index:03d} parse error: {exc}"
                _log(verbose, f"EXECUTION ERROR: {msg}")
                return _error_result(
                    prompt_sha256=prompt_sha,
                    candidate_sha256=candidate_sha256,
                    debug_path=run_dir,
                    duration_seconds=time.perf_counter() - overall_started,
                    execution_error=msg,
                )

            for base_judgement in base_judgements:
                base_by_id[base_judgement.claim_id] = base_judgement

            write_base_batch_artifact(
                run_dir,
                batch_index=batch_index,
                claims=batch,
                prompt=base_prompt,
                generation=timed.generation,
                judgements=base_judgements,
                model=SUT_GENERATION_MODEL,
                model_digest=SUT_GENERATION_MODEL_DIGEST,
                temperature=SPLIT_CLAIM_PIPELINE_TEMPERATURE,
                num_ctx=CLAIM_LEVEL_NUM_CTX,
                keep_alive=SPLIT_CLAIM_PIPELINE_KEEP_ALIVE,
                wall_time_seconds=timed.wall_time_seconds,
            )
            if verbose:
                statuses = ", ".join(f"{item.claim_id}={item.status}" for item in base_judgements)
                print(f"base batch {batch_index:03d}: {statuses} ({timed.wall_time_seconds:.1f}s)")

        base_wall = time.perf_counter() - base_started
        _log(verbose, "")

        dependency_started = time.perf_counter()
        known_ids = frozenset(extracted_by_id)
        premise_pool = unsupported_premise_pool(list(base_by_id.values()))
        eligible = [
            extracted_by_id[judgement.claim_id]
            for judgement in base_by_id.values()
            if judgement.status == "not_in_evidence"
        ]
        eligible_ids = {claim.claim_id for claim in eligible}
        eligible = [claim for claim in extracted if claim.claim_id in eligible_ids]
        dependency_batches = partition_into_batches(eligible, SPLIT_DEPENDENCY_BATCH_SIZE)
        _log(verbose, f"dependency eligible count: {len(eligible)}")
        _log(verbose, f"dependency batch count: {len(dependency_batches)}")

        premise_pairs: list[tuple[ExtractedClaim, BaseClaimJudgement]] = [
            (extracted_by_id[premise.claim_id], premise) for premise in premise_pool
        ]

        for batch_index, batch in enumerate(dependency_batches, start=1):
            dep_prompt = build_dependency_batch_judge_prompt(
                current_claims=batch,
                unsupported_premises=premise_pairs,
            )
            try:
                timed = _generate(
                    client,
                    prompt=dep_prompt,
                    json_schema=build_dependency_batch_json_schema(),
                )
            except Exception as exc:
                msg = f"dependency batch {batch_index:03d} generate failed: {exc}"
                _log(verbose, f"EXECUTION ERROR: {msg}")
                return _error_result(
                    prompt_sha256=prompt_sha,
                    candidate_sha256=candidate_sha256,
                    debug_path=run_dir,
                    duration_seconds=time.perf_counter() - overall_started,
                    execution_error=msg,
                )
            dependency_batch_durations.append(timed.wall_time_seconds)
            expected_ids = frozenset(claim.claim_id for claim in batch)
            try:
                dep_judgements = parse_dependency_batch(
                    timed.generation.response,
                    expected_claim_ids=expected_ids,
                    known_claim_ids=known_ids,
                )
            except (TypeError, ValueError) as exc:
                msg = f"dependency batch {batch_index:03d} parse error: {exc}"
                _log(verbose, f"EXECUTION ERROR: {msg}")
                return _error_result(
                    prompt_sha256=prompt_sha,
                    candidate_sha256=candidate_sha256,
                    debug_path=run_dir,
                    duration_seconds=time.perf_counter() - overall_started,
                    execution_error=msg,
                )

            for dep_judgement in dep_judgements:
                dependency_by_id[dep_judgement.claim_id] = dep_judgement

            write_dependency_batch_artifact(
                run_dir,
                batch_index=batch_index,
                claims=batch,
                prompt=dep_prompt,
                generation=timed.generation,
                judgements=dep_judgements,
                model=SUT_GENERATION_MODEL,
                model_digest=SUT_GENERATION_MODEL_DIGEST,
                temperature=SPLIT_CLAIM_PIPELINE_TEMPERATURE,
                num_ctx=CLAIM_LEVEL_NUM_CTX,
                keep_alive=SPLIT_CLAIM_PIPELINE_KEEP_ALIVE,
                wall_time_seconds=timed.wall_time_seconds,
            )
            if verbose:
                statuses = ", ".join(
                    f"{item.claim_id}:derived={item.derived}" for item in dep_judgements
                )
                print(
                    f"dependency batch {batch_index:03d}: {statuses} "
                    f"({timed.wall_time_seconds:.1f}s)"
                )

        dependency_wall = time.perf_counter() - dependency_started
        _log(verbose, "")

        try:
            bundle = build_final_judgements(extracted, base_by_id, dependency_by_id)
        except (TypeError, ValueError) as exc:
            msg = f"dependency graph validation/normalization failed: {exc}"
            _log(verbose, f"EXECUTION ERROR: {msg}")
            return _error_result(
                prompt_sha256=prompt_sha,
                candidate_sha256=candidate_sha256,
                debug_path=run_dir,
                duration_seconds=time.perf_counter() - overall_started,
                execution_error=msg,
            )

        finals = bundle.finals
        metrics = compute_split_claim_metrics(finals)
        passed = split_claim_passed(finals)
        overall_wall = time.perf_counter() - overall_started
        avg_base_batch = (
            sum(base_batch_durations) / len(base_batch_durations) if base_batch_durations else 0.0
        )
        avg_claim_equiv = base_wall / len(extracted) if extracted else 0.0
        avg_dep_batch = (
            sum(dependency_batch_durations) / len(dependency_batch_durations)
            if dependency_batch_durations
            else 0.0
        )

        write_dependency_graph_artifact(
            run_dir,
            graph=bundle.graph,
            downgraded_claims=bundle.downgraded_claims,
        )
        write_summary_artifact(
            run_dir,
            candidate_sha256=candidate_sha256,
            candidate_length=len(candidate_answer),
            metrics=metrics,
            passed=passed,
            finals=finals,
            extraction_wall_time=extraction_wall,
            base_judge_wall_time=base_wall,
            dependency_wall_time=dependency_wall,
            overall_wall_time=overall_wall,
            base_batch_count=len(base_batches),
            dependency_batch_count=len(dependency_batches),
            dependency_eligible_count=len(eligible),
            average_base_batch_duration=avg_base_batch,
            average_claim_equivalent_base_time=avg_claim_equiv,
            average_dependency_batch_duration=avg_dep_batch,
            execution_errors=[],
            dependency_cycle_count=len(bundle.graph.cycles),
            downgraded_claims=bundle.downgraded_claims,
        )

        failure_summary = split_claim_failure_reasons(finals) if not passed else []
        status = "PASS" if passed else "FAIL"

        if verbose:
            print("=== Final metrics ===")
            print(f"Total claims: {metrics.total_claims}")
            print(f"Supported: {metrics.supported}")
            print(f"Contradicted: {metrics.contradicted}")
            print(f"Not in evidence: {metrics.not_in_evidence}")
            print(f"Derived: {metrics.derived_from_unsupported_claim}")
            print(f"Provenance issues: {metrics.provenance_issues}")
            print(f"Grounded claim rate: {metrics.grounded_claim_rate:.1%}")
            print(f"Verdict: {status}")
            print()
            print(f"Dependency cycles detected: {len(bundle.graph.cycles)}")
            print(f"normalized dependency edges: {len(bundle.graph.normalized_edges)}")
            print(f"overall wall: {overall_wall:.2f}s")
            print(f"Debug artifact path: {run_dir}")
            print()
            if failure_summary:
                print("Failure reasons:")
                for reason in failure_summary[:20]:
                    print(f"  - {reason}")
                print()
            for label in FAKE_SEVERITY_LABELS:
                fake_items = _claims_for_label(finals, label)
                print(f"Claims involving {label}: {len(fake_items)}")

        return LongFormClaimAuditResult(
            case_id=CASE_ID,
            status=status,
            total_claims=metrics.total_claims,
            supported_count=metrics.supported,
            contradicted_count=metrics.contradicted,
            not_in_evidence_count=metrics.not_in_evidence,
            derived_count=metrics.derived_from_unsupported_claim,
            provenance_issue_count=metrics.provenance_issues,
            grounded_claim_rate=metrics.grounded_claim_rate,
            raw_dependency_edge_count=len(bundle.graph.raw_edges),
            normalized_dependency_edge_count=len(bundle.graph.normalized_edges),
            dependency_cycle_count=len(bundle.graph.cycles),
            execution_error=None,
            duration_seconds=overall_wall,
            debug_path=str(run_dir),
            judge_model=SUT_GENERATION_MODEL,
            judge_model_digest=SUT_GENERATION_MODEL_DIGEST,
            num_ctx=CLAIM_LEVEL_NUM_CTX,
            candidate_sha256=candidate_sha256,
            prompt_sha256=prompt_sha,
            failure_summary=failure_summary,
        )
    finally:
        client.unload_model(SUT_GENERATION_MODEL)
        if verbose:
            print("final ollama ps:")
            print(_ollama_ps_output())


def main() -> int:
    result = run_split_claim_evaluation(verbose=True)
    if result.status == "ERROR":
        return 1
    if result.status == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
