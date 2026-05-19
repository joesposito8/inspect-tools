"""Tests for inspect_tools._inject."""
from __future__ import annotations

import random

import pytest

from inspect_tools._inject import schema_to_tool_def, schema_to_tool_info
from inspect_tools._library import load_corpus
from inspect_tools._seed import derive_seed
from inspect_tools.schema import ToolSchema


def _basic_schema(name: str = "test_tool") -> ToolSchema:
    return ToolSchema(
        name=name,
        description="A test tool.",
        inputSchema={
            "type": "object",
            "properties": {
                "x": {"type": "string", "description": "An x param."},
                "y": {"type": "string"},  # missing description — exercises auto-fill
            },
            "required": ["x"],
        },
        outputSchema={
            "type": "object",
            "properties": {},
            "examples": [
                {"id": "pkg_1", "echo": "{x | default_x}"},
                {"id": "pkg_2", "echo": "{x | default_x}"},
            ],
        },
        domain="misc",
        content_category="general_popular",
        source_url="https://test.fixture/inject",
    )


# === schema_to_tool_info ===


def test_schema_to_tool_info_basic():
    info = schema_to_tool_info(_basic_schema())
    assert info.name == "test_tool"
    assert info.description == "A test tool."
    assert "x" in info.parameters.properties
    assert "y" in info.parameters.properties


def test_schema_to_tool_info_auto_fills_missing_description():
    info = schema_to_tool_info(_basic_schema())
    # `y` was missing a description in inputSchema; should be auto-filled to param name
    assert info.parameters.properties["y"].description == "y"
    # `x` had its own description preserved
    assert info.parameters.properties["x"].description == "An x param."


# === schema_to_tool_def ===


def test_schema_to_tool_def_basic(make_state):
    state = make_state()
    td = schema_to_tool_def(
        _basic_schema(),
        state=state,
        solver_namespace="context_exhaustion",
        trial_seed=42,
    )
    assert td.name == "test_tool"
    assert td.description == "A test tool."
    assert "x" in td.parameters.properties


def test_schema_to_tool_def_auto_fills_missing_description(make_state):
    state = make_state()
    td = schema_to_tool_def(
        _basic_schema(),
        state=state,
        solver_namespace="context_exhaustion",
        trial_seed=42,
    )
    assert td.parameters.properties["y"].description == "y"


async def test_invocation_increments_default_namespace(make_state):
    state = make_state()
    td = schema_to_tool_def(
        _basic_schema(),
        state=state,
        solver_namespace="context_exhaustion",
        trial_seed=42,
    )
    tool = td.as_tool()
    await tool(x="hello")
    assert state.metadata["inspect_tools"]["context_exhaustion"]["filler_invocations"] == 1
    await tool(x="world")
    assert state.metadata["inspect_tools"]["context_exhaustion"]["filler_invocations"] == 2


async def test_invocation_custom_namespace(make_state):
    """Verifies v1.x sibling-solver reuse with uniform counter field name."""
    state = make_state()
    td = schema_to_tool_def(
        _basic_schema(),
        state=state,
        solver_namespace="test_sibling",
        trial_seed=42,
    )
    tool = td.as_tool()
    await tool(x="hello")
    assert state.metadata["inspect_tools"]["test_sibling"]["filler_invocations"] == 1


async def test_default_response_fn_is_synthesize_response(make_state):
    """When response_fn is omitted, synthesize_response is used. Verify by checking output shape."""
    state = make_state()
    td = schema_to_tool_def(
        _basic_schema(),
        state=state,
        solver_namespace="context_exhaustion",
        trial_seed=42,
    )
    tool = td.as_tool()
    result = await tool(x="hello")
    # synthesize_response picks a package and substitutes; echo field should be the kwarg
    assert isinstance(result, dict)
    assert result.get("echo") == "hello"


async def test_custom_response_fn_invoked(make_state):
    state = make_state()
    received_args = {}

    def my_response_fn(schema, kwargs, rng):
        received_args["schema_name"] = schema.name
        received_args["kwargs"] = kwargs
        received_args["rng_class"] = type(rng).__name__
        return {"custom": True}

    td = schema_to_tool_def(
        _basic_schema(),
        state=state,
        solver_namespace="context_exhaustion",
        trial_seed=42,
        response_fn=my_response_fn,
    )
    tool = td.as_tool()
    result = await tool(x="hello", y="world")
    assert result == {"custom": True}
    assert received_args["schema_name"] == "test_tool"
    assert received_args["kwargs"] == {"x": "hello", "y": "world"}
    assert received_args["rng_class"] == "Random"


async def test_same_trial_determinism(make_state):
    """Same trial_seed + same schema → same call_rng seed → same response package."""
    state1 = make_state(sample_id="s1")
    state2 = make_state(sample_id="s2")  # different state, same trial_seed
    schema = _basic_schema()
    td1 = schema_to_tool_def(
        schema, state=state1, solver_namespace="context_exhaustion", trial_seed=42
    )
    td2 = schema_to_tool_def(
        schema, state=state2, solver_namespace="context_exhaustion", trial_seed=42
    )
    r1 = await td1.as_tool()(x="hello")
    r2 = await td2.as_tool()(x="hello")
    assert r1 == r2  # same trial_seed + same tool name → same call_rng → same package


async def test_cross_trial_variance(make_state):
    """Different trial_seed → different call_rng → potentially different package (≥2 packages)."""
    schema = _basic_schema()
    results = []
    for seed in range(1, 20):
        state = make_state()
        td = schema_to_tool_def(
            schema, state=state, solver_namespace="context_exhaustion", trial_seed=seed
        )
        results.append((await td.as_tool()(x="hello"))["id"])
    # With 2 packages and 19 different seeds, we should see both ids at least once
    assert len(set(results)) >= 2


async def test_fresh_closure_per_schema(make_state):
    """Two ToolDef builds for different schemas produce distinct callables."""
    state = make_state()
    s1 = _basic_schema("tool_a")
    s2 = _basic_schema("tool_b")
    td1 = schema_to_tool_def(
        s1, state=state, solver_namespace="context_exhaustion", trial_seed=42
    )
    td2 = schema_to_tool_def(
        s2, state=state, solver_namespace="context_exhaustion", trial_seed=42
    )
    assert td1.as_tool() is not td2.as_tool()


# === Real-corpus smoke ===


def test_real_corpus_smoke_schema_to_tool_info():
    """schema_to_tool_info on first 20 records — exercises production-shape JSON-Schema."""
    corpus = load_corpus()
    for schema in corpus[:20]:
        info = schema_to_tool_info(schema)
        assert info.name == schema.name
        assert info.description == schema.description


def test_real_corpus_smoke_schema_to_tool_def(make_state):
    """schema_to_tool_def on first 20 records — verifies adapter handles real shapes."""
    state = make_state()
    corpus = load_corpus()
    for schema in corpus[:20]:
        td = schema_to_tool_def(
            schema,
            state=state,
            solver_namespace="context_exhaustion",
            trial_seed=42,
        )
        assert td.name == schema.name


def test_execute_signature_is_introspectable(make_state):
    """Inspect's tool-call dispatch (_call_tools.py:719) does
    `typing.get_type_hints(execute)` and raises ValueError if any
    `signature.parameters` entry is missing a hint. The `**kwargs` form needs
    `**kwargs: Any` — silently masked when the model never invokes an
    injected tool. Caught during ICP audit Phase C.5."""
    import inspect
    import typing

    td = schema_to_tool_def(
        _basic_schema(),
        state=make_state(),
        solver_namespace="context_exhaustion",
        trial_seed=42,
    )
    sig = inspect.signature(td.tool)
    hints = typing.get_type_hints(td.tool)
    for param_name in sig.parameters:
        assert param_name in hints, (
            f"_inject.execute signature parameter {param_name!r} has no type "
            f"annotation; Inspect's tool dispatch will fail when the model "
            f"actually invokes an injected tool. Annotate as `: Any` (mirrors "
            f"inspect_ai/tool/_mcp/_local.py)."
        )
