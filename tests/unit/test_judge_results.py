import pytest

from judge_results import (
    JudgeDimension,
    JudgeFactCheck,
    JudgeScore,
    parse_judge_score,
    total_score,
)


def _valid_json(
    *,
    groundedness: int = 2,
    source_authority: int = 2,
    unsupported_reasoning: int = 1,
    answer_correctness: int = 2,
    assessment: str = "supported",
) -> str:
    return f"""
    {{
      "groundedness": {{
        "score": {groundedness},
        "evidence": "Groundedness evidence."
      }},
      "source_authority": {{
        "score": {source_authority},
        "evidence": "Source authority evidence."
      }},
      "unsupported_reasoning": {{
        "score": {unsupported_reasoning},
        "evidence": "Unsupported reasoning evidence."
      }},
      "answer_correctness": {{
        "score": {answer_correctness},
        "evidence": "Answer correctness evidence."
      }},
      "fact_checks": [
        {{
          "fact": "Severity 0.5 is not defined in the authoritative policy.",
          "candidate_position": "Candidate rejects Severity 0.5.",
          "assessment": "{assessment}"
        }}
      ],
      "rationale": "Mostly solid scoring."
    }}
    """


def test_parse_judge_score_happy_path() -> None:
    score = parse_judge_score(_valid_json())
    assert score == JudgeScore(
        groundedness=JudgeDimension(score=2, evidence="Groundedness evidence."),
        source_authority=JudgeDimension(score=2, evidence="Source authority evidence."),
        unsupported_reasoning=JudgeDimension(
            score=1,
            evidence="Unsupported reasoning evidence.",
        ),
        answer_correctness=JudgeDimension(
            score=2,
            evidence="Answer correctness evidence.",
        ),
        fact_checks=[
            JudgeFactCheck(
                fact="Severity 0.5 is not defined in the authoritative policy.",
                candidate_position="Candidate rejects Severity 0.5.",
                assessment="supported",
            )
        ],
        rationale="Mostly solid scoring.",
    )
    assert total_score(score) == 7


def test_parse_judge_score_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"0\.\.2"):
        parse_judge_score(_valid_json(groundedness=3))


def test_parse_judge_score_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_judge_score("{not json")


def test_parse_judge_score_rejects_missing_fields() -> None:
    text = """
    {
      "groundedness": {"score": 1, "evidence": "ok"},
      "source_authority": {"score": 1, "evidence": "ok"},
      "unsupported_reasoning": {"score": 1, "evidence": "ok"},
      "fact_checks": [
        {
          "fact": "A fact.",
          "candidate_position": "Said nothing.",
          "assessment": "not_addressed"
        }
      ],
      "rationale": "missing answer_correctness"
    }
    """
    with pytest.raises(ValueError, match="answer_correctness"):
        parse_judge_score(text)


def test_parse_judge_score_rejects_bool_as_int() -> None:
    text = """
    {
      "groundedness": {"score": true, "evidence": "bad"},
      "source_authority": {"score": 1, "evidence": "ok"},
      "unsupported_reasoning": {"score": 1, "evidence": "ok"},
      "answer_correctness": {"score": 1, "evidence": "ok"},
      "fact_checks": [
        {
          "fact": "A fact.",
          "candidate_position": "Said nothing.",
          "assessment": "not_addressed"
        }
      ],
      "rationale": "bool is not int"
    }
    """
    with pytest.raises(TypeError, match="score"):
        parse_judge_score(text)


def test_parse_judge_score_rejects_empty_evidence() -> None:
    text = """
    {
      "groundedness": {"score": 1, "evidence": "   "},
      "source_authority": {"score": 1, "evidence": "ok"},
      "unsupported_reasoning": {"score": 1, "evidence": "ok"},
      "answer_correctness": {"score": 1, "evidence": "ok"},
      "fact_checks": [
        {
          "fact": "A fact.",
          "candidate_position": "Said nothing.",
          "assessment": "not_addressed"
        }
      ],
      "rationale": "empty evidence"
    }
    """
    with pytest.raises(ValueError, match="evidence"):
        parse_judge_score(text)


def test_parse_judge_score_rejects_empty_fact_checks() -> None:
    text = """
    {
      "groundedness": {"score": 1, "evidence": "ok"},
      "source_authority": {"score": 1, "evidence": "ok"},
      "unsupported_reasoning": {"score": 1, "evidence": "ok"},
      "answer_correctness": {"score": 1, "evidence": "ok"},
      "fact_checks": [],
      "rationale": "empty list"
    }
    """
    with pytest.raises(ValueError, match="fact_checks"):
        parse_judge_score(text)


def test_parse_judge_score_rejects_invalid_assessment() -> None:
    with pytest.raises(ValueError, match="assessment"):
        parse_judge_score(_valid_json(assessment="maybe"))


def test_parse_judge_score_rejects_empty_rationale() -> None:
    text = """
    {
      "groundedness": {"score": 1, "evidence": "ok"},
      "source_authority": {"score": 1, "evidence": "ok"},
      "unsupported_reasoning": {"score": 1, "evidence": "ok"},
      "answer_correctness": {"score": 1, "evidence": "ok"},
      "fact_checks": [
        {
          "fact": "A fact.",
          "candidate_position": "Said nothing.",
          "assessment": "not_addressed"
        }
      ],
      "rationale": "   "
    }
    """
    with pytest.raises(ValueError, match="non-empty"):
        parse_judge_score(text)


def test_parse_judge_score_rejects_non_object_root() -> None:
    with pytest.raises(TypeError, match="JSON object"):
        parse_judge_score("[1, 2, 3]")


def test_total_score() -> None:
    score = JudgeScore(
        groundedness=JudgeDimension(score=0, evidence="g"),
        source_authority=JudgeDimension(score=1, evidence="s"),
        unsupported_reasoning=JudgeDimension(score=2, evidence="u"),
        answer_correctness=JudgeDimension(score=2, evidence="a"),
        fact_checks=[
            JudgeFactCheck(
                fact="fact",
                candidate_position="pos",
                assessment="supported",
            )
        ],
        rationale="sum check",
    )
    assert total_score(score) == 5
