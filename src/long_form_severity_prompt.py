"""Immutable long-form SUT prompt for mixed-truth severity policy poisoning.

Trailing-newline convention:
  The fixture is a UTF-8 text file that ends with exactly one trailing newline
  (POSIX text-file convention). Loaders must read the file as UTF-8 and return
  that exact string — no strip(), no format(), no other normalization — before
  sending it as the SUT user message.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from workspace_setup import PROJECT_ROOT

CASE_ID = "mixed_truth_severity_policy_poisoning"
CASE_CATEGORY = "long_form_policy_poisoning"

PROMPT_FIXTURE_RELATIVE_PATH = "fixtures/prompts/mixed_truth_severity_policy_poisoning.txt"
PROMPT_FIXTURE_PATH = PROJECT_ROOT / PROMPT_FIXTURE_RELATIVE_PATH

# SHA-256 of the exact UTF-8 fixture bytes (including the trailing newline).
PROMPT_FIXTURE_SHA256 = "5d7b243dd661091694f443e4771a7481c93af3d101d96f22e84901ee5e36487c"

AUTHORITATIVE_DOCUMENT_PATH = PROJECT_ROOT / "fixtures" / "documents" / "acme_support_policy.md"


def load_sut_prompt() -> str:
    """Load the immutable SUT prompt without normalization."""
    return PROMPT_FIXTURE_PATH.read_text(encoding="utf-8")


def compute_prompt_sha256(path: Path | None = None) -> str:
    target = PROMPT_FIXTURE_PATH if path is None else path
    return hashlib.sha256(target.read_bytes()).hexdigest()


def assert_prompt_fixture_integrity() -> str:
    digest = compute_prompt_sha256()
    if digest != PROMPT_FIXTURE_SHA256:
        msg = (
            f"prompt fixture SHA-256 mismatch for {PROMPT_FIXTURE_RELATIVE_PATH}: "
            f"expected {PROMPT_FIXTURE_SHA256}, got {digest}"
        )
        raise ValueError(msg)
    return digest
