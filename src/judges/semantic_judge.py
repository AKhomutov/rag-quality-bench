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
2 — does not invent policy rules, exceptions, precedence, or unsupported conclusions
1 — contains a minor unsupported inference that does not materially change the answer
0 — invents a material rule, exception, precedence relationship, or policy conclusion

Answer correctness
2 — the answer matches what can actually be established from the authoritative sources
1 — partially correct but incomplete, ambiguous, or overconfident
0 — materially incorrect

OUTPUT FORMAT
Return a valid JSON object only.
Do not wrap the JSON in markdown fences.
Do not include any text before or after the JSON object.
Do not include fields named "passed" or "total".

Required JSON shape:
{{
  "groundedness": 0,
  "source_authority": 0,
  "unsupported_reasoning": 0,
  "answer_correctness": 0,
  "rationale": "brief explanation of the scores"
}}
"""
