"""Split claim pipeline: extract → batched base judge → batched dependency → aggregate.

Uses the fixed candidate answer only (SUT is not called).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

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
PREVIOUS_OVERALL_SECONDS = 549.0
PREVIOUS_BASE_CALLS = 54
PREVIOUS_DEPENDENCY_CALLS = 15
PREVIOUS_METRICS = {
    "claims": 54,
    "supported": 26,
    "contradicted": 13,
    "not_in_evidence": 5,
    "derived": 10,
    "provenance": 20,
    "grounded": 0.481,
    "verdict": "FAIL",
}


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


def _print_generation_metadata(label: str, timed: TimedGeneration) -> None:
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


def main() -> None:
    overall_started = time.perf_counter()
    execution_errors: list[str] = []
    client = OllamaClient()
    extraction_wall = 0.0
    base_wall = 0.0
    dependency_wall = 0.0
    extracted: list[ExtractedClaim] = []
    base_by_id: dict[str, BaseClaimJudgement] = {}
    dependency_by_id: dict[str, DependencyJudgement] = {}
    base_batch_durations: list[float] = []
    dependency_batch_durations: list[float] = []
    base_batches: list[list[ExtractedClaim]] = []
    dependency_batches: list[list[ExtractedClaim]] = []
    eligible: list[ExtractedClaim] = []
    run_dir = create_split_claim_debug_run_dir()
    candidate_sha256 = ""
    candidate_length = 0
    extraction_timed: TimedGeneration | None = None

    try:
        prompt_sha = assert_prompt_fixture_integrity()
        original_user_prompt = load_sut_prompt()
        authoritative_source = AUTHORITATIVE_DOCUMENT_PATH.read_text(encoding="utf-8")
        reference_facts = reference_facts_as_lines()

        candidate_answer = load_identical_candidate_answer(PREVIOUS_TRACE_PATH)
        candidate_sha256 = sha256_text(candidate_answer)
        candidate_length = len(candidate_answer)
        assert_fixed_candidate_identity(candidate_answer, candidate_sha256)

        print(f"case: {CASE_ID}")
        print(f"prompt fixture SHA-256: {prompt_sha}")
        print()
        print("Candidate source:")
        print(PREVIOUS_TRACE_PATH.relative_to(PROJECT_ROOT))
        print()
        print("Candidate SHA-256:")
        print(candidate_sha256)
        print()
        print("Candidate length:")
        print(candidate_length)
        print()
        print("SUT was NOT called.")
        print(f"Judge model: {SUT_GENERATION_MODEL}")
        print(f"Judge digest: {SUT_GENERATION_MODEL_DIGEST}")
        print(f"num_ctx: {CLAIM_LEVEL_NUM_CTX}")
        print(f"temperature: {SPLIT_CLAIM_PIPELINE_TEMPERATURE}")
        print(f"keep_alive (during run): {SPLIT_CLAIM_PIPELINE_KEEP_ALIVE}")
        print(f"SPLIT_BASE_BATCH_SIZE: {SPLIT_BASE_BATCH_SIZE}")
        print(f"SPLIT_DEPENDENCY_BATCH_SIZE: {SPLIT_DEPENDENCY_BATCH_SIZE}")
        print()
        print(f"Debug artifacts: {run_dir}")
        print()

        # --- Stage 1: extraction ---
        extraction_prompt = build_claim_extraction_prompt(
            candidate_answer=candidate_answer,
            original_user_prompt=original_user_prompt,
        )
        extraction_timed = _generate(
            client,
            prompt=extraction_prompt,
            json_schema=build_claim_extraction_json_schema(),
        )
        extraction_wall = extraction_timed.wall_time_seconds
        _print_generation_metadata("extraction", extraction_timed)

        try:
            extracted_raw = parse_extracted_claims(extraction_timed.generation.response)
            extracted = canonicalize_extracted_claim_ids(extracted_raw)
        except (TypeError, ValueError) as exc:
            execution_errors.append(f"extraction parse error: {exc}")
            print(f"EXECUTION ERROR (extraction): {exc}")
            raise SystemExit(1) from exc

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

        print(f"extracted claim count: {len(extracted)}")
        print("extracted claims:")
        for item in extracted:
            print(f"  {item.claim_id}: {item.claim}")
        print()

        extracted_by_id = {item.claim_id: item for item in extracted}

        # --- Stage 2: batched base judge ---
        base_started = time.perf_counter()
        base_batches = partition_into_batches(extracted, SPLIT_BASE_BATCH_SIZE)
        print(f"base batch count: {len(base_batches)}")

        for batch_index, batch in enumerate(base_batches, start=1):
            base_prompt = build_base_claim_batch_judge_prompt(
                authoritative_source=authoritative_source,
                reference_facts=reference_facts,
                claims=batch,
            )
            timed = _generate(
                client,
                prompt=base_prompt,
                json_schema=build_base_claim_batch_json_schema(),
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
                execution_errors.append(msg)
                print(f"EXECUTION ERROR: {msg}")
                raise SystemExit(1) from exc

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
            statuses = ", ".join(f"{item.claim_id}={item.status}" for item in base_judgements)
            print(f"base batch {batch_index:03d}: {statuses} ({timed.wall_time_seconds:.1f}s)")

        base_wall = time.perf_counter() - base_started
        print()
        print("base classification counts:")
        for status in ("supported", "contradicted", "not_in_evidence"):
            count = sum(1 for judgement in base_by_id.values() if judgement.status == status)
            print(f"  {status}: {count}")
        print()

        # --- Stage 2B: batched dependency ---
        dependency_started = time.perf_counter()
        known_ids = frozenset(extracted_by_id)
        premise_pool = unsupported_premise_pool(list(base_by_id.values()))
        eligible = [
            extracted_by_id[judgement.claim_id]
            for judgement in base_by_id.values()
            if judgement.status == "not_in_evidence"
        ]
        # Preserve extracted order for eligible claims.
        eligible_ids = {claim.claim_id for claim in eligible}
        eligible = [claim for claim in extracted if claim.claim_id in eligible_ids]
        dependency_batches = partition_into_batches(eligible, SPLIT_DEPENDENCY_BATCH_SIZE)
        print(f"dependency eligible count: {len(eligible)}")
        print(f"dependency batch count: {len(dependency_batches)}")

        premise_pairs: list[tuple[ExtractedClaim, BaseClaimJudgement]] = [
            (extracted_by_id[premise.claim_id], premise) for premise in premise_pool
        ]

        for batch_index, batch in enumerate(dependency_batches, start=1):
            dep_prompt = build_dependency_batch_judge_prompt(
                current_claims=batch,
                unsupported_premises=premise_pairs,
            )
            timed = _generate(
                client,
                prompt=dep_prompt,
                json_schema=build_dependency_batch_json_schema(),
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
                execution_errors.append(msg)
                print(f"EXECUTION ERROR: {msg}")
                raise SystemExit(1) from exc

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
            statuses = ", ".join(
                f"{item.claim_id}:derived={item.derived}" for item in dep_judgements
            )
            print(
                f"dependency batch {batch_index:03d}: {statuses} ({timed.wall_time_seconds:.1f}s)"
            )

        dependency_wall = time.perf_counter() - dependency_started
        print()

        finals = build_final_judgements(extracted, base_by_id, dependency_by_id)
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

        write_summary_artifact(
            run_dir,
            candidate_sha256=candidate_sha256,
            candidate_length=candidate_length,
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
            execution_errors=execution_errors,
        )

        print("=== Final metrics ===")
        print(f"Total claims: {metrics.total_claims}")
        print(f"Supported: {metrics.supported}")
        print(f"Contradicted: {metrics.contradicted}")
        print(f"Not in evidence: {metrics.not_in_evidence}")
        print(f"Derived: {metrics.derived_from_unsupported_claim}")
        print(f"Provenance issues: {metrics.provenance_issues}")
        print(f"Grounded claim rate: {metrics.grounded_claim_rate:.1%}")
        print(f"Verdict: {'PASS' if passed else 'FAIL'}")
        print()

        print("=== Comparison vs previous unbatched run ===")
        print(
            f"previous: claims={PREVIOUS_METRICS['claims']} "
            f"supported={PREVIOUS_METRICS['supported']} "
            f"contradicted={PREVIOUS_METRICS['contradicted']} "
            f"not_in_evidence={PREVIOUS_METRICS['not_in_evidence']} "
            f"derived={PREVIOUS_METRICS['derived']} "
            f"provenance={PREVIOUS_METRICS['provenance']} "
            f"grounded={PREVIOUS_METRICS['grounded']:.1%} "
            f"verdict={PREVIOUS_METRICS['verdict']}"
        )
        print(
            f"current:  claims={metrics.total_claims} "
            f"supported={metrics.supported} "
            f"contradicted={metrics.contradicted} "
            f"not_in_evidence={metrics.not_in_evidence} "
            f"derived={metrics.derived_from_unsupported_claim} "
            f"provenance={metrics.provenance_issues} "
            f"grounded={metrics.grounded_claim_rate:.1%} "
            f"verdict={'PASS' if passed else 'FAIL'}"
        )
        print()

        if not passed:
            print("Failure reasons:")
            for reason in split_claim_failure_reasons(finals):
                print(f"  - {reason}")
            print()

        for label in FAKE_SEVERITY_LABELS:
            fake_items = _claims_for_label(finals, label)
            print(f"Claims involving {label}: {len(fake_items)}")
            for fake_item in fake_items:
                print(f"  claim_id: {fake_item.claim_id}")
                print(f"  claim: {fake_item.claim}")
                print(f"  status: {fake_item.status}")
                print(f"  facts: {fake_item.reference_fact_ids}")
                print(f"  provenance: {fake_item.provenance_issue}")
                print(f"  depends_on: {fake_item.depends_on}")
                print(f"  explanation: {fake_item.explanation}")
            print()

        derived_items = [
            final for final in finals if final.status == "derived_from_unsupported_claim"
        ]
        print(f"derived_from_unsupported_claim findings: {len(derived_items)}")
        for derived in derived_items:
            print(f"  - {derived.claim_id}: {derived.claim}")
            print(f"    depends_on: {derived.depends_on}")
            print(f"    explanation: {derived.explanation}")
        print()

        print("Dependency edges:")
        edges = [(final.claim_id, dep) for final in finals for dep in final.depends_on]
        if not edges:
            print("  (none)")
        else:
            for src, dst in edges:
                print(f"  {src} -> {dst}")
        print()

        print("Judge execution errors:")
        if not execution_errors:
            print("  (none)")
        else:
            for err in execution_errors:
                print(f"  - {err}")
        print()

        ollama_calls_before = 1 + PREVIOUS_BASE_CALLS + PREVIOUS_DEPENDENCY_CALLS
        ollama_calls_after = 1 + len(base_batches) + len(dependency_batches)
        speedup = PREVIOUS_OVERALL_SECONDS / overall_wall if overall_wall > 0 else 0.0

        print("=== Runtime ===")
        print(f"extracted claims: {len(extracted)}")
        print(f"base batch count: {len(base_batches)}")
        print(f"dependency eligible count: {len(eligible)}")
        print(f"dependency batch count: {len(dependency_batches)}")
        print(f"Ollama calls before vs after: {ollama_calls_before} -> {ollama_calls_after}")
        print(f"extraction wall: {extraction_wall:.2f}s")
        print(f"base total wall: {base_wall:.2f}s")
        print(f"dependency total wall: {dependency_wall:.2f}s")
        print(f"overall wall: {overall_wall:.2f}s")
        print(f"average base batch wall: {avg_base_batch:.2f}s")
        print(f"average claim-equivalent base time: {avg_claim_equiv:.2f}s")
        print(f"average dependency batch wall: {avg_dep_batch:.2f}s")
        print(f"previous overall: {PREVIOUS_OVERALL_SECONDS:.0f}s")
        print(f"speedup vs 549s baseline: {speedup:.2f}x")
        print()
        print(f"Debug artifact path: {run_dir}")
        print()

    finally:
        client.unload_model(SUT_GENERATION_MODEL)
        print("final ollama ps:")
        print(_ollama_ps_output())


if __name__ == "__main__":
    main()
