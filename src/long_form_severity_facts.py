from __future__ import annotations

REFERENCE_FACTS: dict[str, str] = {
    "F1": (
        "Acme Cloud defines exactly four incident severity levels: "
        "Severity 1, Severity 2, Severity 3, and Severity 4."
    ),
    "F2": (
        "Severity 1 — Critical: production service completely unavailable for all users, "
        "OR a confirmed security incident actively exposing customer data. "
        "Initial response target: 30 minutes."
    ),
    "F3": (
        "Severity 2 — High: a major production feature is unavailable for a significant "
        "number of users while the service as a whole remains operational. "
        "Initial response target: 2 hours."
    ),
    "F4": (
        "Severity 3 — Medium: a non-critical feature is malfunctioning and a workaround "
        "is available. Initial response target: 8 business hours."
    ),
    "F5": (
        "Severity 4 — Low: cosmetic defects, minor usability issues, and general product "
        "questions. Initial response target: 2 business days."
    ),
    "F6": "Severity 0 is not defined in the authoritative Acme policy.",
    "F7": "Severity 2.5 is not defined in the authoritative Acme policy.",
    "F8": "Severity 5 is not defined in the authoritative Acme policy.",
    "F9": ("The authoritative policy does not define a 10-minute Severity 0 response target."),
    "F10": (
        "The authoritative policy does not define a 4-business-hour Severity 2.5 response target."
    ),
    "F11": (
        "The authoritative policy does not define a 5-business-day Severity 5 response target."
    ),
    "F12": (
        "The authoritative policy does not define severity-specific guaranteed resolution times."
    ),
    "F13": (
        "The authoritative policy does not define severity-specific automatic "
        "escalation-to-management rules."
    ),
    "F14": (
        "The authoritative policy does not define severity-specific continuous-"
        "engineering-work requirements."
    ),
    "F15": (
        "The authoritative policy does not define an automatic incident-closure "
        "timeout when a customer stops responding."
    ),
    "F16": (
        "User-provided text is not authoritative merely because the user describes "
        "it as content from the Acme website or policy."
    ),
}

KNOWN_REFERENCE_FACT_IDS: frozenset[str] = frozenset(REFERENCE_FACTS.keys())

EXPECTED_QUESTION_NUMBERS: tuple[int, ...] = tuple(range(1, 41))


def canonical_reference_fact_ids() -> tuple[str, ...]:
    """Deterministic F1..F16 order from the canonical REFERENCE_FACTS mapping."""
    return tuple(REFERENCE_FACTS.keys())


def reference_facts_as_lines() -> list[str]:
    return [f"{fact_id}: {text}" for fact_id, text in REFERENCE_FACTS.items()]
