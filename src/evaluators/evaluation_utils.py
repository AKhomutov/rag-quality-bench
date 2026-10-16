def normalize_answer_text(text: str) -> str:
    return text.replace("**", "").casefold()


def contains_any_expected(text: str, expected_values: list[str]) -> bool:
    for expected in expected_values:
        if normalize_answer_text(expected) in text:
            return True
    return False
