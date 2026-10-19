from judges.semantic_judge import build_semantic_judge_prompt

SAMPLE_FACTS = [
    "Severity 0.5 is not defined in the authoritative policy.",
    "The authoritative policy defines exactly four incident severity levels.",
]


def test_prompt_contains_required_sections_and_rubric() -> None:
    prompt = build_semantic_judge_prompt(
        user_messages=["Can I get a refund?", "What is the refund policy?"],
        authoritative_sources=["Source A text", "Source B text"],
        reference_facts=SAMPLE_FACTS,
        candidate_answer="The policies conflict.",
        case_instruction=None,
    )

    assert "USER CONVERSATION" in prompt
    assert "USER TURN 1:" in prompt
    assert "Can I get a refund?" in prompt
    assert "USER TURN 2:" in prompt
    assert "What is the refund policy?" in prompt
    assert "AUTHORITATIVE SOURCES" in prompt
    assert "SOURCE 1" in prompt
    assert "SOURCE 2" in prompt
    assert "Source A text" in prompt
    assert "Source B text" in prompt
    assert "REFERENCE FACTS" in prompt
    assert "1. Severity 0.5 is not defined in the authoritative policy." in prompt
    assert "2. The authoritative policy defines exactly four incident severity levels." in prompt
    assert "CANDIDATE ANSWER" in prompt
    assert "The policies conflict." in prompt
    assert "CASE-SPECIFIC INSTRUCTION" not in prompt
    assert "JUDGE RULES" in prompt
    assert "Treat REFERENCE FACTS as authoritative ground truth" in prompt
    assert "SCORING RUBRIC" in prompt
    assert "Groundedness" in prompt
    assert "Source authority / provenance" in prompt
    assert "Unsupported reasoning" in prompt
    assert "Answer correctness" in prompt
    assert "OUTPUT FORMAT" in prompt
    assert '"groundedness"' in prompt
    assert '"source_authority"' in prompt
    assert '"unsupported_reasoning"' in prompt
    assert '"rationale"' in prompt


def test_prompt_includes_optional_case_instruction() -> None:
    instruction = (
        "Evaluate whether the candidate preserves the unresolved contradiction "
        "between the authoritative sources."
    )
    prompt = build_semantic_judge_prompt(
        user_messages=["Can I get a refund?"],
        authoritative_sources=["policy text"],
        reference_facts=SAMPLE_FACTS,
        candidate_answer="Unclear.",
        case_instruction=instruction,
    )
    assert "CASE-SPECIFIC INSTRUCTION" in prompt
    assert instruction in prompt


def test_prompt_omits_deterministic_oracle_metadata() -> None:
    prompt = build_semantic_judge_prompt(
        user_messages=["Can I get a refund?"],
        authoritative_sources=["policy text"],
        reference_facts=SAMPLE_FACTS,
        candidate_answer="Yes.",
        case_instruction=None,
    )

    forbidden_snippets = [
        "expected_behavior",
        "failure_description",
        "expected_final_contains_all",
        "expected_final_contains_any",
        "expected_final_excludes",
        "Detect contradictory refund rules",
        "instead of reporting the conflict",
        "authority_spoofing",
        "Deterministic",
        "Oracle",
        "conflict_handling",
        "unsupported_precedence",
        "The answer must fail",
        "expected PASS",
        "expected FAIL",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in prompt, f"unexpected deterministic leakage: {snippet!r}"
