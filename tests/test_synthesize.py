"""Tests for inspect_tools._synthesize."""
from __future__ import annotations

import copy
import random

import pytest

from inspect_tools._synthesize import substitute, synthesize_response
from inspect_tools.schema import ToolSchema


def _schema(name: str = "test_tool", **outputSchema_extra) -> ToolSchema:
    """Build a minimal ToolSchema with a custom outputSchema."""
    return ToolSchema(
        name=name,
        description="test description",
        inputSchema={"type": "object", "properties": {}, "required": []},
        outputSchema=outputSchema_extra or None,
        domain="misc",
        content_category="general_popular",
        source_url="https://test.fixture/synthesize",
    )


# === Package selection + fallbacks ===


def test_package_selection_deterministic_same_seed():
    pkgs = [{"v": 1}, {"v": 2}, {"v": 3}]
    schema = _schema(examples=pkgs)
    a = synthesize_response(schema, {}, random.Random(42))
    b = synthesize_response(schema, {}, random.Random(42))
    assert a == b


def test_package_selection_different_seed_may_differ():
    """Probabilistic — with 3 packages, at least one of 5 seeds should differ from seed 0."""
    pkgs = [{"v": 1}, {"v": 2}, {"v": 3}]
    schema = _schema(examples=pkgs)
    base = synthesize_response(schema, {}, random.Random(0))
    others = {
        synthesize_response(schema, {}, random.Random(i))["v"] for i in range(1, 50)
    }
    assert len(others) >= 2  # at least 2 different packages picked across 49 seeds


def test_fallback_missing_outputSchema():
    schema = _schema()  # no outputSchema
    assert synthesize_response(schema, {}, random.Random(0)) == {"ok": True}


def test_fallback_empty_examples():
    schema = _schema(examples=[])
    assert synthesize_response(schema, {}, random.Random(0)) == {"ok": True}


def test_no_stitching():
    """Returned response equals exactly one package's substituted form."""
    pkgs = [{"a": 1, "b": 2}, {"a": 10, "b": 20}]
    schema = _schema(examples=pkgs)
    for seed in range(20):
        result = synthesize_response(schema, {}, random.Random(seed))
        assert result in pkgs  # exact match — no cross-package mixing


# === Whole-value mode ===


def test_whole_value_kwarg_present():
    assert substitute("{name | default}", {"name": "Acme"}) == "Acme"


def test_whole_value_type_preservation_int():
    assert substitute("{size | 0}", {"size": 50}) == 50
    assert isinstance(substitute("{size | 0}", {"size": 50}), int)


def test_whole_value_type_preservation_bool():
    assert substitute("{enabled | false}", {"enabled": True}) is True


def test_whole_value_type_preservation_list():
    assert substitute("{tags | []}", {"tags": ["a", "b"]}) == ["a", "b"]


def test_whole_value_type_preservation_dict():
    assert substitute("{cfg | {}}", {"cfg": {"k": 1}}) == {"k": 1}


def test_whole_value_default_json_parsed_null():
    assert substitute("{plan | null}", {}) is None


def test_whole_value_default_json_parsed_int():
    assert substitute("{size | 0}", {}) == 0
    assert isinstance(substitute("{size | 0}", {}), int)


def test_whole_value_default_json_parsed_list():
    assert substitute("{tags | []}", {}) == []


def test_whole_value_default_json_parsed_empty_dict():
    """Regex backtracks past braces in default — critical edge case."""
    result = substitute("{custom_attributes | {}}", {})
    assert result == {}
    assert isinstance(result, dict)


def test_whole_value_default_json_parsed_true():
    assert substitute("{enabled | true}", {}) is True


def test_whole_value_non_json_default_plain_string():
    assert substitute("{name | New Company}", {}) == "New Company"


# === Embedded mode ===


def test_embedded_kwarg_present():
    assert (
        substitute("Hi {name | there}, welcome", {"name": "Hana"})
        == "Hi Hana, welcome"
    )


def test_embedded_kwarg_absent():
    assert (
        substitute("Hi {name | there}, welcome", {}) == "Hi there, welcome"
    )


def test_embedded_multiple_placeholders():
    template = "https://github.com/{owner | octo}/{repo | demo}"
    assert (
        substitute(template, {"owner": "myorg", "repo": "proj"})
        == "https://github.com/myorg/proj"
    )


def test_embedded_str_coerce_non_string_kwarg():
    assert substitute("count: {n | 0}", {"n": 5}) == "count: 5"


def test_embedded_multi_placeholder_starting_with_brace_regression():
    """Regression: `"{a | x}/{b | y}"` starts with `{` but is two placeholders.

    Brace-aware scanner must classify this as embedded mode, not whole-value with
    default `"x}/{b | y"` (the bug a naive greedy regex produces).
    """
    assert substitute("{a | x}/{b | y}", {"a": "A", "b": "B"}) == "A/B"
    assert substitute("{a | x}/{b | y}", {}) == "x/y"


# === Same kwarg, both modes ===


def test_same_kwarg_in_both_modes_within_one_package():
    """Mirrors actions_get production shape: resource_id is whole-value in `id`
    AND embedded in `html_url`."""
    package = {
        "id": "{resource_id | 8472193056}",
        "html_url": "https://github.com/org/repo/actions/runs/{resource_id | 8472193056}",
    }
    result = substitute(package, {"resource_id": 999})
    assert result["id"] == 999  # int — type preserved
    assert isinstance(result["id"], int)
    assert (
        result["html_url"] == "https://github.com/org/repo/actions/runs/999"
    )


# === Recursion ===


def test_recursion_into_dict():
    package = {
        "outer_literal": "kept",
        "outer_placeholder": "{name | default}",
        "inner": {
            "inner_literal": 42,
            "inner_placeholder": "{value | null}",
        },
    }
    result = substitute(package, {"name": "Alice", "value": "hello"})
    assert result == {
        "outer_literal": "kept",
        "outer_placeholder": "Alice",
        "inner": {
            "inner_literal": 42,
            "inner_placeholder": "hello",
        },
    }


def test_recursion_into_list():
    package = {
        "items": [
            {"value": "{a | x}"},
            "literal item",
            {"value": "{b | y}"},
        ],
    }
    result = substitute(package, {"a": "A", "b": "B"})
    assert result == {
        "items": [
            {"value": "A"},
            "literal item",
            {"value": "B"},
        ],
    }


# === Edge cases ===


def test_bare_key_no_pipe_is_plain_text():
    """`{key}` without `| default` is NOT a placeholder per the mandatory-defaults rule."""
    assert substitute("{key}", {"key": "foo"}) == "{key}"


def test_no_mutation_of_source_package():
    pkgs = [{"name": "{name | default}", "nested": {"v": "{x | 0}"}}]
    original = copy.deepcopy(pkgs)
    schema = _schema(examples=pkgs)
    synthesize_response(schema, {"name": "Alice", "x": 99}, random.Random(0))
    assert pkgs == original  # source unchanged


def test_unclosed_brace_passes_through():
    """Author bug — solver shouldn't crash, just return the string unchanged."""
    assert substitute("{name | New Company", {}) == "{name | New Company"


def test_empty_string():
    assert substitute("", {}) == ""


def test_non_string_non_container_passes_through():
    assert substitute(42, {}) == 42
    assert substitute(None, {}) is None
    assert substitute(True, {}) is True


# === Anchor traces (regression against production corpus contract) ===


def test_anchor_actions_get():
    """Inline fixture mirroring the production actions_get record.

    Same kwarg (resource_id) in both whole-value (id) and embedded (html_url) positions.
    """
    schema = ToolSchema(
        name="actions_get",
        description="Get a workflow run by ID.",
        inputSchema={"type": "object", "properties": {}, "required": []},
        outputSchema={
            "type": "object",
            "properties": {},
            "examples": [
                {
                    "id": "{resource_id | 8472193056}",
                    "name": "ci",
                    "html_url": "https://github.com/{owner | facebook}/{repo | react}/actions/runs/{resource_id | 8472193056}",
                },
            ],
        },
        domain="dev-tools",
        content_category="general_popular",
        source_url="https://test.fixture/actions_get",
    )
    result = synthesize_response(
        schema,
        {"owner": "myorg", "repo": "myproj", "resource_id": 999},
        random.Random(0),
    )
    assert result["id"] == 999
    assert isinstance(result["id"], int)
    assert result["name"] == "ci"
    assert result["html_url"] == "https://github.com/myorg/myproj/actions/runs/999"


def test_anchor_intercom_create_or_update_company():
    """Inline fixture mirroring INTERCOM_CREATE_OR_UPDATE_A_COMPANY.

    Covers null/0/{} JSON-literal defaults across multiple field types.
    """
    schema = ToolSchema(
        name="intercom_create_or_update_company",
        description="Create or update a company.",
        inputSchema={"type": "object", "properties": {}, "required": []},
        outputSchema={
            "type": "object",
            "properties": {},
            "examples": [
                {
                    "type": "company",
                    "id": "66b4f1a7d83c2e0a91f5c2d4",
                    "name": "{name | New Company}",
                    "website": "{website | null}",
                    "industry": "{industry | null}",
                    "size": "{size | null}",
                    "plan": None,  # literal None in package
                    "monthly_spend": "{monthly_spend | 0}",
                    "user_count": 0,
                    "tags": {"type": "tag.list", "tags": []},
                    "custom_attributes": "{custom_attributes | {}}",
                },
            ],
        },
        domain="data-analytics",
        content_category="general_popular",
        source_url="https://test.fixture/intercom",
    )
    result = synthesize_response(
        schema, {"name": "Heliotrope Freight Co"}, random.Random(0)
    )
    assert result["name"] == "Heliotrope Freight Co"
    assert result["website"] is None
    assert result["industry"] is None
    assert result["size"] is None
    assert result["plan"] is None  # literal
    assert result["monthly_spend"] == 0
    assert isinstance(result["monthly_spend"], int)
    assert result["user_count"] == 0  # literal
    assert result["tags"] == {"type": "tag.list", "tags": []}  # literal nested
    assert result["custom_attributes"] == {}  # empty-dict default
    assert isinstance(result["custom_attributes"], dict)


def test_anchor_gmail_create_email_draft():
    """Inline fixture mirroring GMAIL_CREATE_EMAIL_DRAFT.

    Placeholders deeply nested inside message.payload.headers[i].value (depth 4).
    """
    schema = ToolSchema(
        name="gmail_create_email_draft",
        description="Create a Gmail draft.",
        inputSchema={"type": "object", "properties": {}, "required": []},
        outputSchema={
            "type": "object",
            "properties": {},
            "examples": [
                {
                    "id": "r5781309624108273951",
                    "display_url": "https://mail.google.com/mail/u/0/#drafts/{draft_id | r5781309624108273951}",
                    "message": {
                        "id": "1936c4a82d75e9b3",
                        "snippet": "{body | Default snippet}",
                        "payload": {
                            "mimeType": "text/plain",
                            "headers": [
                                {"name": "To", "value": "{recipient_email | nobody@example.com}"},
                                {"name": "From", "value": "Renata Achebe <renata@example.com>"},
                                {"name": "Subject", "value": "{subject | Quick note}"},
                                {"name": "Date", "value": "Mon, 17 Nov 2025 14:37:07 -0500"},
                            ],
                        },
                    },
                },
            ],
        },
        domain="communication",
        content_category="general_popular",
        source_url="https://test.fixture/gmail",
    )
    result = synthesize_response(
        schema,
        {
            "recipient_email": "alice@example.com",
            "subject": "Hi",
            "body": "Test message",
        },
        random.Random(0),
    )
    # Top-level literals preserved
    assert result["id"] == "r5781309624108273951"
    # Embedded mode (draft_id not provided — default used)
    assert (
        result["display_url"]
        == "https://mail.google.com/mail/u/0/#drafts/r5781309624108273951"
    )
    # Nested whole-value placeholders substituted
    assert result["message"]["snippet"] == "Test message"
    # Deep recursion into list-of-dicts: placeholder values substituted
    headers = result["message"]["payload"]["headers"]
    headers_by_name = {h["name"]: h["value"] for h in headers}
    assert headers_by_name["To"] == "alice@example.com"
    assert headers_by_name["Subject"] == "Hi"
    # Literal values untouched
    assert headers_by_name["From"] == "Renata Achebe <renata@example.com>"
    assert headers_by_name["Date"] == "Mon, 17 Nov 2025 14:37:07 -0500"
