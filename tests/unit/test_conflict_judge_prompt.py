from judges.conflict_judge import build_conflict_judge_prompt


def test_prompt_contains_required_sections_and_rubric() -> None:
    prompt = build_conflict_judge_prompt(
        user_messages=["Can I get a refund?", "What is the refund policy?"],
        authoritative_sources=["Source A text", "Source B text"],
        candidate_answer="The policies conflict.",
    )

    assert "USER CONVERSATION" in prompt
    assert "USER MESSAGE 1:" in prompt
    assert "Can I get a refund?" in prompt
    assert "USER MESSAGE 2:" in prompt
    assert "What is the refund policy?" in prompt
    assert "AUTHORITATIVE SOURCES" in prompt
    assert "SOURCE 1" in prompt
    assert "SOURCE 2" in prompt
    assert "Source A text" in prompt
    assert "Source B text" in prompt
    assert "CANDIDATE ANSWER" in prompt
    assert "The policies conflict." in prompt
    assert "SCORING RUBRIC" in prompt
    assert "Groundedness" in prompt
    assert "Conflict handling" in prompt
    assert "Unsupported precedence" in prompt
    assert "Answer correctness" in prompt
    assert "OUTPUT FORMAT" in prompt
    assert '"groundedness"' in prompt
    assert '"rationale"' in prompt


def test_prompt_omits_deterministic_oracle_metadata() -> None:
    prompt = build_conflict_judge_prompt(
        user_messages=["Can I get a refund?"],
        authoritative_sources=["policy text"],
        candidate_answer="Yes.",
    )

    forbidden_snippets = [
        "expected_behavior",
        "failure_description",
        "expected_final_contains_all",
        "expected_final_contains_any",
        "expected_final_excludes",
        "Detect contradictory refund rules",
        "instead of reporting the conflict",
        "7 calendar days",
        "non-refundable",
        "takes precedence",
        "specific policy overrides",
        "cannot determine",
        "Deterministic",
        "Oracle",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in prompt, f"unexpected deterministic leakage: {snippet!r}"
