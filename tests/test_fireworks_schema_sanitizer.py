# tests/test_fireworks_schema_sanitizer.py
"""Tests for _sanitize_schema_for_fireworks in llm_client."""
import copy


def test_oneOf_converted_to_anyOf():
    """oneOf is not supported by Fireworks; must become anyOf."""
    from services.llm_client import _sanitize_schema_for_fireworks

    schema = {
        "type": "object",
        "properties": {
            "patches": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {"type": "object", "properties": {"op": {"type": "string"}}},
                        {"type": "object", "properties": {"id": {"type": "string"}}},
                    ]
                },
            }
        },
    }
    result = _sanitize_schema_for_fireworks(copy.deepcopy(schema))
    items = result["properties"]["patches"]["items"]
    assert "oneOf" not in items
    assert "anyOf" in items
    assert len(items["anyOf"]) == 2


def test_type_array_with_null_converted_to_anyOf():
    """'type': ['string', 'null'] must become anyOf with separate type entries."""
    from services.llm_client import _sanitize_schema_for_fireworks

    schema = {
        "type": "object",
        "properties": {
            "resolution": {
                "type": ["string", "null"],
                "description": "What resolved it.",
            }
        },
    }
    result = _sanitize_schema_for_fireworks(copy.deepcopy(schema))
    prop = result["properties"]["resolution"]
    assert "type" not in prop
    assert "anyOf" in prop
    assert {"type": "string"} in prop["anyOf"]
    assert {"type": "null"} in prop["anyOf"]
    assert prop["description"] == "What resolved it."


def test_enum_with_null_split_into_anyOf():
    """enum containing null must be split: string enum (no null) + null type."""
    from services.llm_client import _sanitize_schema_for_fireworks

    schema = {
        "type": "object",
        "properties": {
            "disc_secondary": {
                "type": ["string", "null"],
                "enum": ["D", "I", "S", "C", None],
                "description": "Secondary DISC type.",
            }
        },
    }
    result = _sanitize_schema_for_fireworks(copy.deepcopy(schema))
    prop = result["properties"]["disc_secondary"]
    assert "type" not in prop
    assert "enum" not in prop
    assert "anyOf" in prop
    # One branch: string enum without null
    string_branch = [b for b in prop["anyOf"] if b.get("type") == "string"]
    assert len(string_branch) == 1
    assert string_branch[0]["enum"] == ["D", "I", "S", "C"]
    # One branch: null type
    null_branch = [b for b in prop["anyOf"] if b.get("type") == "null"]
    assert len(null_branch) == 1
    assert prop["description"] == "Secondary DISC type."


def test_plain_schema_passes_through_unchanged():
    """A schema with no unsupported constructs should come back identical."""
    from services.llm_client import _sanitize_schema_for_fireworks

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }
    original = copy.deepcopy(schema)
    result = _sanitize_schema_for_fireworks(copy.deepcopy(schema))
    assert result == original


def test_deeply_nested_oneOf_is_converted():
    """oneOf inside nested objects should also be converted."""
    from services.llm_client import _sanitize_schema_for_fireworks

    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {
                    "inner": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "integer"},
                        ]
                    }
                },
            }
        },
    }
    result = _sanitize_schema_for_fireworks(copy.deepcopy(schema))
    inner = result["properties"]["outer"]["properties"]["inner"]
    assert "oneOf" not in inner
    assert "anyOf" in inner


def test_original_schema_is_not_mutated():
    """The function must not mutate the input schema."""
    from services.llm_client import _sanitize_schema_for_fireworks

    schema = {
        "type": "object",
        "properties": {
            "field": {
                "type": ["string", "null"],
                "enum": ["a", "b", None],
            }
        },
    }
    original = copy.deepcopy(schema)
    _sanitize_schema_for_fireworks(schema)
    assert schema == original
