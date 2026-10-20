from __future__ import annotations

from typing import cast

from claim_level_config import (
    CLAIM_LEVEL_JUDGE_KEEP_ALIVE,
    CLAIM_LEVEL_JUDGE_MODEL,
    CLAIM_LEVEL_JUDGE_MODEL_DIGEST,
    CLAIM_LEVEL_NUM_CTX,
    SPLIT_BASE_BATCH_SIZE,
    SPLIT_CLAIM_PIPELINE_KEEP_ALIVE,
    SPLIT_DEPENDENCY_BATCH_SIZE,
)
from clients.ollama_client import JUDGE_MODEL, build_generate_request_body
from workspace_setup import PROJECT_ROOT

LOCK_PATH = PROJECT_ROOT / "infra" / "environment.lock"


def _lock_model_digest(model_name: str) -> str:
    lines = LOCK_PATH.read_text(encoding="utf-8").splitlines()
    current_model: str | None = None
    for line in lines:
        stripped = line.strip()
        if (
            stripped.endswith(":")
            and not stripped.startswith("#")
            and line.startswith("  ")
            and not line.startswith("    ")
        ):
            current_model = stripped[:-1]
        if current_model == model_name and stripped.startswith("digest:"):
            value = stripped.split(":", 1)[1].strip().strip('"')
            return value
    msg = f"model {model_name!r} digest not found in environment.lock"
    raise AssertionError(msg)


def test_claim_level_model_is_gemma() -> None:
    assert CLAIM_LEVEL_JUDGE_MODEL == "gemma3:12b"
    assert CLAIM_LEVEL_JUDGE_KEEP_ALIVE == 0
    assert CLAIM_LEVEL_NUM_CTX == 16384
    assert JUDGE_MODEL == "qwen3:8b"


def test_claim_level_request_sends_num_ctx() -> None:
    body = build_generate_request_body(
        model=CLAIM_LEVEL_JUDGE_MODEL,
        prompt="x",
        keep_alive=CLAIM_LEVEL_JUDGE_KEEP_ALIVE,
        num_ctx=CLAIM_LEVEL_NUM_CTX,
    )
    options_raw = body["options"]
    assert isinstance(options_raw, dict)
    options = cast(dict[str, object], options_raw)
    assert options["num_ctx"] == 16384

    other = build_generate_request_body(model=JUDGE_MODEL, prompt="x")
    other_options_raw = other["options"]
    assert isinstance(other_options_raw, dict)
    other_options = cast(dict[str, object], other_options_raw)
    assert "num_ctx" not in other_options


def test_environment_lock_includes_judge_model_full_digest() -> None:
    digest = _lock_model_digest("gemma3:12b")
    assert digest == CLAIM_LEVEL_JUDGE_MODEL_DIGEST
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64
    assert digest != "sha256:b92b218582c5"

    qwen_digest = _lock_model_digest("qwen3:8b")
    assert qwen_digest.startswith("sha256:")
    assert len(qwen_digest) == len("sha256:") + 64


def test_keep_alive_serialized_only_when_set() -> None:
    without = build_generate_request_body(model="qwen3:8b", prompt="x")
    assert "keep_alive" not in without

    with_keep = build_generate_request_body(
        model=CLAIM_LEVEL_JUDGE_MODEL,
        prompt="x",
        keep_alive=CLAIM_LEVEL_JUDGE_KEEP_ALIVE,
    )
    assert with_keep["keep_alive"] == 0
    assert with_keep["model"] == "gemma3:12b"


def test_split_pipeline_keep_alive_stays_loaded() -> None:
    assert SPLIT_CLAIM_PIPELINE_KEEP_ALIVE == -1
    body = build_generate_request_body(
        model=JUDGE_MODEL,
        prompt="x",
        keep_alive=SPLIT_CLAIM_PIPELINE_KEEP_ALIVE,
        num_ctx=CLAIM_LEVEL_NUM_CTX,
    )
    assert body["keep_alive"] == -1
    options_raw = body["options"]
    assert isinstance(options_raw, dict)
    options = cast(dict[str, object], options_raw)
    assert options["num_ctx"] == 16384


def test_split_pipeline_batch_sizes() -> None:
    assert SPLIT_BASE_BATCH_SIZE == 6
    assert SPLIT_DEPENDENCY_BATCH_SIZE == 8


def test_debug_trace_includes_model_digest_and_keep_alive() -> None:
    from claim_level_debug import ClaimLevelDebugErrorTrace, claim_level_debug_error_trace_to_object

    trace = ClaimLevelDebugErrorTrace(
        case_id="mixed_truth_severity_policy_poisoning",
        trial_number=1,
        prompt_fixture_path="fixtures/prompts/mixed_truth_severity_policy_poisoning.txt",
        prompt_sha256="abc",
        sut_prompt="prompt",
        candidate_answer="answer",
        candidate_sha256="def",
        authoritative_source="source",
        reference_facts=["F1: fact"],
        judge_prompt="judge",
        raw_response="{}",
        error="boom",
        model=CLAIM_LEVEL_JUDGE_MODEL,
        model_digest=CLAIM_LEVEL_JUDGE_MODEL_DIGEST,
        temperature=0.0,
        keep_alive=CLAIM_LEVEL_JUDGE_KEEP_ALIVE,
        claim_level_schema_version=1,
        num_ctx=None,
        generation=None,
    )
    payload = claim_level_debug_error_trace_to_object(trace)
    judge_raw = payload["judge"]
    assert isinstance(judge_raw, dict)
    judge = cast(dict[str, object], judge_raw)
    assert judge["model"] == "gemma3:12b"
    assert judge["model_digest"] == CLAIM_LEVEL_JUDGE_MODEL_DIGEST
    assert judge["keep_alive"] == 0
