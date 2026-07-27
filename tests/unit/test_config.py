from drugmatch.config import deep_merge


def test_deep_merge_preserves_nested_values() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
