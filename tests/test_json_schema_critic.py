"""Tests for sextant.reflexion.json_schema_critic."""

from __future__ import annotations

from sextant.reflexion import json_schema_critic


def test_passes_on_valid_json():
    critic = json_schema_critic({
        "type": "object",
        "required": ["name"],
        "properties": {"name": {"type": "string"}},
    })
    passed, msg = critic('{"name": "alice"}')
    assert passed is True
    assert msg == "PASS"


def test_fails_on_bad_json():
    critic = json_schema_critic({"type": "object"})
    passed, msg = critic("this is not json")
    assert passed is False
    assert "not valid JSON" in msg


def test_fails_when_required_key_missing():
    critic = json_schema_critic({
        "type": "object",
        "required": ["age"],
    })
    passed, msg = critic('{"name": "alice"}')
    assert passed is False
    # Either jsonschema's message or our minimal validator's message.
    assert "age" in msg or "required" in msg.lower()


def test_fails_when_type_wrong():
    critic = json_schema_critic({"type": "array"})
    passed, msg = critic('{"k": "v"}')
    assert passed is False
    assert "array" in msg or "type" in msg.lower()


def test_strips_markdown_fences():
    critic = json_schema_critic({"type": "object"})
    passed, msg = critic('```json\n{"x": 1}\n```')
    assert passed is True


def test_strips_unlabeled_fences():
    critic = json_schema_critic({"type": "object"})
    passed, _ = critic('```\n{"x": 1}\n```')
    assert passed is True


def test_nested_required_keys_enforced():
    schema = {
        "type": "object",
        "required": ["user"],
        "properties": {
            "user": {
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "integer"}},
            },
        },
    }
    critic = json_schema_critic(schema)
    ok, _ = critic('{"user": {"id": 42}}')
    assert ok is True
    fail, msg = critic('{"user": {}}')
    assert fail is False
    assert "id" in msg.lower() or "required" in msg.lower()


def test_array_items_validated():
    schema = {"type": "array", "items": {"type": "integer"}}
    critic = json_schema_critic(schema)
    ok, _ = critic("[1, 2, 3]")
    assert ok is True
    fail, msg = critic('[1, "two", 3]')
    assert fail is False
    assert "integer" in msg or "type" in msg.lower()
