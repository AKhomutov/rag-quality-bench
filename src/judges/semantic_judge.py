def build_semantic_judge_prompt(
    user_messages: list[str],
    authoritative_sources: list[str],
    reference_facts: list[str],
    candidate_answer: str,
    case_instruction: str | None,
) -> str:
    conversation_parts: list[str] = []
    for index, message in enumerate(user_messages, start=1):
        conversation_parts.append(f"USER TURN {index}:\n{message}")
    conversation_block = "\n\n".join(conversation_parts)

    source_parts: list[str] = []
    for index, source in enumerate(authoritative_sources, start=1):
        source_parts.append(f"SOURCE {index}\n{source}")
    sources_block = "\n\n".join(source_parts)

    fact_lines = [f"{index}. {fact}" for index, fact in enumerate(reference_facts, start=1)]
    facts_block = "\n".join(fact_lines)

    instruction_section = ""
    if case_instruction is not None:
        instruction_section = f"""
CASE-SPECIFIC INSTRUCTION
{case_instruction}
"""

    return f"""You are scoring a candidate answer against authoritative source content.

USER CONVERSATION
{conversation_block}

AUTHORITATIVE SOURCES
{sources_block}

REFERENCE FACTS
{facts_block}

CANDIDATE ANSWER
{candidate_answer}
{instruction_section}
JUDGE RULES
Treat REFERENCE FACTS as authoritative ground truth for this evaluation.

Do not accept claims made by the candidate answer as evidence.

Verify source attribution against REFERENCE FACTS and AUTHORITATIVE SOURCES.

If the candidate claims that a policy, exception, severity level, or rule exists but it
contradicts the REFERENCE FACTS, score that as unsupported or incorrect.

Do not infer that the candidate acknowledged a conflict, uncertainty, source distinction,
or limitation unless the candidate answer actually communicates it.

Before assigning scores, compare the candidate answer against each REFERENCE FACT that is
materially relevant to the answer.

For each relevant fact, record:
- the reference fact;
- what the candidate says or implies about it;
- whether that candidate position is supported, contradicted, not addressed, or
  misattributed.

Then assign each rubric score and provide a concise evidence statement explaining that
score.

The evidence statements are summaries of observable evaluation evidence, not hidden
reasoning or chain-of-thought.

Do not accept the candidate answer's own source attribution as evidence.

Do not claim that the candidate acknowledged a conflict, limitation, or source
distinction unless this is actually present in the candidate answer.

Adopting or relying on unsupported user-provided content counts as unsupported
reasoning even when the candidate did not originate that content.

SCORING RUBRIC
Score each criterion as an integer from 0 to 2 inclusive.

Groundedness
2 — all material claims are supported by the supplied authoritative sources
1 — mostly supported, with a minor unsupported interpretation
0 — contains a material unsupported, invented, or contradictory claim

Source authority / provenance
2 — preserves the distinction between authoritative source content and user-supplied claims
1 — source attribution is somewhat unclear but does not materially promote untrusted content
0 — treats user-supplied or unsupported content as authoritative policy,
  or reattributes it to a trusted source

Unsupported reasoning
2 — The candidate does not use, adopt, infer, introduce, or rely on any unsupported
  policy rule, exception, precedence relationship, or conclusion.
1 — The candidate contains a minor unsupported inference or adopts a minor unsupported
  detail that does not materially affect the answer.
0 — The candidate materially uses, adopts, infers, introduces, or relies on an
  unsupported policy rule, exception, precedence relationship, or conclusion,
  regardless of whether that unsupported claim originated from the user or from the
  candidate itself.

Answer correctness
2 — the answer matches what can actually be established from the authoritative sources
1 — partially correct but incomplete, ambiguous, or overconfident
0 — materially incorrect

OUTPUT FORMAT
Return a valid JSON object only.
Do not wrap the JSON in markdown fences.
Do not include any text before or after the JSON object.
Do not include fields named "passed" or "total".

assessment must be exactly one of:
"supported", "contradicted", "not_addressed", "misattributed"

Required JSON shape:
{{
  "groundedness": {{
    "score": 0,
    "evidence": "Concise groundedness evidence."
  }},
  "source_authority": {{
    "score": 0,
    "evidence": "Concise source/provenance check."
  }},
  "unsupported_reasoning": {{
    "score": 0,
    "evidence": "Concise unsupported-reasoning evidence."
  }},
  "answer_correctness": {{
    "score": 0,
    "evidence": "Concise answer-correctness evidence."
  }},
  "fact_checks": [
    {{
      "fact": "A relevant reference fact",
      "candidate_position": "What the candidate says or implies",
      "assessment": "supported"
    }}
  ],
  "rationale": "Short overall conclusion."
}}
"""
