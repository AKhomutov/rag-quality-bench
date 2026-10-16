from typing import cast


def validate_yaml_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"expected YAML object, got {type(value).__name__}"
        raise TypeError(msg)

    result: dict[str, object] = {}
    raw_items = cast(list[tuple[object, object]], list(value.items()))

    for key, item in raw_items:
        if not isinstance(key, str):
            msg = f"expected string key, got {type(key).__name__}"
            raise TypeError(msg)

        result[key] = item

    return result


def get_required_string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)

    if not isinstance(value, str):
        msg = f"expected string field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)

    return value


def get_required_list(mapping: dict[str, object], key: str) -> list[object]:
    value = mapping.get(key)

    if not isinstance(value, list):
        msg = f"expected list field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)

    return cast(list[object], value)


def get_required_string_list(mapping: dict[str, object], key: str) -> list[str]:
    value = get_required_list(mapping, key)
    result: list[str] = []

    for item in value:
        if not isinstance(item, str):
            msg = f"expected string items in {key!r}, got {type(item).__name__}"
            raise TypeError(msg)
        result.append(item)

    return result
