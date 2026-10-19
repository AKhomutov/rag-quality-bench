import pytest

from judge_results import JudgeScore, parse_judge_score, total_score


def test_parse_judge_score_happy_path() -> None:
    text = """
    {
      "groundedness": 2,
      "conflict_handling": 2,
      "unsupported_precedence": 1,
      "answer_correctness": 2,
      "rationale": "Mostly solid conflict handling."
    }
    """
    score = parse_judge_score(text)
    assert score == JudgeScore(
        groundedness=2,
        conflict_handling=2,
        unsupported_precedence=1,
        answer_correctness=2,
        rationale="Mostly solid conflict handling.",
    )
    assert total_score(score) == 7


def test_parse_judge_score_rejects_out_of_range() -> None:
    text = """
    {
      "groundedness": 3,
      "conflict_handling": 2,
      "unsupported_precedence": 1,
      "answer_correctness": 2,
      "rationale": "bad range"
    }
    """
    with pytest.raises(ValueError, match=r"0\.\.2"):
        parse_judge_score(text)


def test_parse_judge_score_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_judge_score("{not json")


def test_parse_judge_score_rejects_missing_fields() -> None:
    text = """
    {
      "groundedness": 1,
      "conflict_handling": 1,
      "unsupported_precedence": 1,
      "rationale": "missing answer_correctness"
    }
    """
    with pytest.raises(ValueError, match="answer_correctness"):
        parse_judge_score(text)


def test_parse_judge_score_rejects_bool_as_int() -> None:
    text = """
    {
      "groundedness": true,
      "conflict_handling": 1,
      "unsupported_precedence": 1,
      "answer_correctness": 1,
      "rationale": "bool is not int"
    }
    """
    with pytest.raises(TypeError, match="groundedness"):
        parse_judge_score(text)


def test_parse_judge_score_rejects_string_number() -> None:
    text = """
    {
      "groundedness": "2",
      "conflict_handling": 1,
      "unsupported_precedence": 1,
      "answer_correctness": 1,
      "rationale": "string number"
    }
    """
    with pytest.raises(TypeError, match="groundedness"):
        parse_judge_score(text)


def test_parse_judge_score_rejects_empty_rationale() -> None:
    text = """
    {
      "groundedness": 1,
      "conflict_handling": 1,
      "unsupported_precedence": 1,
      "answer_correctness": 1,
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
        groundedness=0,
        conflict_handling=1,
        unsupported_precedence=2,
        answer_correctness=2,
        rationale="sum check",
    )
    assert total_score(score) == 5
