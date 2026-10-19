from long_form_severity_prompt import (
    PROMPT_FIXTURE_PATH,
    PROMPT_FIXTURE_RELATIVE_PATH,
    PROMPT_FIXTURE_SHA256,
    assert_prompt_fixture_integrity,
    compute_prompt_sha256,
    load_sut_prompt,
)


def test_prompt_fixture_sha256_matches_pinned_digest() -> None:
    digest = compute_prompt_sha256()
    assert digest == PROMPT_FIXTURE_SHA256
    assert assert_prompt_fixture_integrity() == PROMPT_FIXTURE_SHA256


def test_prompt_loaded_without_normalization() -> None:
    loaded = load_sut_prompt()
    raw_bytes = PROMPT_FIXTURE_PATH.read_bytes()
    assert loaded.encode("utf-8") == raw_bytes
    assert loaded.endswith("\n")
    assert loaded == raw_bytes.decode("utf-8")
    # No strip/format: leading/trailing whitespace beyond the single trailing newline
    # must remain identical to the fixture bytes.
    assert not loaded.startswith(" ")
    assert loaded.endswith("paragraph.\n")
    assert "Severity 0 — Emergency" in loaded
    assert "Severity 2.5 — Elevated" in loaded
    assert "Severity 5 — Informational" in loaded
    assert PROMPT_FIXTURE_RELATIVE_PATH.endswith("mixed_truth_severity_policy_poisoning.txt")
