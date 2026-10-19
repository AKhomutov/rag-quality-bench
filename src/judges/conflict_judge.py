def build_conflict_judge_prompt(
    user_messages: list[str],
    authoritative_sources: list[str],
    candidate_answer: str,
) -> str:
    conversation_parts: list[str] = []
    for index, message in enumerate(user_messages, start=1):
        conversation_parts.append(f"USER MESSAGE {index}:\n{message}")
    conversation_block = "\n\n".join(conversation_parts)

    source_parts: list[str] = []
    for index, source in enumerate(authoritative_sources, start=1):
        source_parts.append(f"SOURCE {index}\n{source}")
    sources_block = "\n\n".join(source_parts)

    return f"""You are scoring a candidate answer about conflicting authoritative sources.

USER CONVERSATION
{conversation_block}

AUTHORITATIVE SOURCES
{sources_block}

CANDIDATE ANSWER
{candidate_answer}

SCORING RUBRIC
Score each criterion as an integer from 0 to 2 inclusive.

Groundedness
2 — all material claims are supported by the supplied authoritative sources
1 — mostly supported, with a minor unsupported interpretation
0 — contains a material unsupported or invented claim

Conflict handling
2 — explicitly recognizes that the authoritative sources contain incompatible refund rules
1 — expresses uncertainty or inconsistency but does not clearly identify the source conflict
0 — silently reconciles, ignores, or resolves the contradiction

Unsupported precedence
2 — does not invent any source-priority or precedence rule
1 — vaguely implies that one source may take priority without a clear invented rule
0 — invents a precedence rule such as "specific overrides general",
  "this is the exception", "later policy wins", etc.

Answer correctness
2 — conclusion matches what can actually be established from the supplied sources
1 — partially correct but overconfident or incomplete
0 — materially incorrect

OUTPUT FORMAT
Return a valid JSON object only.
Do not wrap the JSON in markdown fences.
Do not include any text before or after the JSON object.
Do not include fields named "passed" or "total".

Required JSON shape:
{{
  "groundedness": 0,
  "conflict_handling": 0,
  "unsupported_precedence": 0,
  "answer_correctness": 0,
  "rationale": "brief explanation of the scores"
}}
"""
