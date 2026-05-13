"""Tokenization, ToolDef adaptation, and additive injection for the Solver.

The token counts produced here are a *counting proxy* against the OpenAI
function-calling JSON shape. They are not the wire format any specific provider
will send. cl100k_base is approximate for non-OpenAI providers (Anthropic uses
a different BPE; expected drift ~5-15%). The Solver param ``tokenizer`` is
overridable so ICP-4/ICP-7 can swap if needed.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

import tiktoken
from inspect_ai.solver import TaskState
from inspect_ai.tool import Tool, ToolDef
from inspect_ai.tool._tool_params import ToolParams

from inspect_context_pressure._types import ToolSchema

# Per-schema overhead approximating OpenAI function-call accounting (delimiters,
# `"type": "function"` wrapper, etc.). Empirically ~4 tokens.
_PER_SCHEMA_OVERHEAD_TOKENS = 4

FILLER_INVOCATION_KEY = "filler_invocations"
NoOpFn = Callable[..., Awaitable[str]]


def _serialize_tooldef_for_counting(td: ToolDef) -> str:
    """Canonical OpenAI function-shape JSON used only for token counting."""
    payload = {
        "type": "function",
        "function": {
            "name": td.name,
            "description": td.description,
            "parameters": td.parameters.model_dump(exclude_none=True),
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def count_tools_tokens(tools: list[Tool], encoding: tiktoken.Encoding) -> int:
    """Sum tokens of the OpenAI-shape serialization of each tool, plus overhead.

    Counting proxy — not the wire format. ``encoding`` is a tiktoken Encoding
    (typically ``tiktoken.get_encoding("cl100k_base")``).
    """
    total = 0
    for tool in tools:
        td = ToolDef(tool)
        total += len(encoding.encode(_serialize_tooldef_for_counting(td)))
        total += _PER_SCHEMA_OVERHEAD_TOKENS
    return total


def count_schema_tokens(schema: ToolSchema, encoding: tiktoken.Encoding) -> int:
    """Token cost of a raw schema dict, matching the same OpenAI function shape."""
    payload = {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["parameters"],
        },
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return len(encoding.encode(body)) + _PER_SCHEMA_OVERHEAD_TOKENS


def _make_filler_noop(state: TaskState) -> NoOpFn:
    """Build a per-state async no-op that records its own invocation.

    Each invocation increments ``state.metadata['filler_invocations']``. ICP-6
    picks this up for the manifest. Returning a benign string (rather than
    raising) avoids polluting eval scorers that count tool errors.
    """

    async def filler_noop(**kwargs: Any) -> str:
        state.metadata[FILLER_INVOCATION_KEY] = state.metadata.get(FILLER_INVOCATION_KEY, 0) + 1
        return "This is a no-op tool. Recorded as filler invocation."

    return filler_noop


def _ensure_param_descriptions(params: dict) -> dict:
    """Walk a JSON-Schema parameters dict and fill missing field descriptions.

    Inspect's ``ToolDef`` validation rejects any parameter whose schema lacks a
    ``description``. Real-world MCP schemas often omit them. We fill with a
    benign placeholder so injection never fails on under-documented schemas.
    """
    if not isinstance(params, dict):
        return params
    out = dict(params)
    properties = out.get("properties")
    if isinstance(properties, dict):
        new_props = {}
        for key, value in properties.items():
            if isinstance(value, dict) and "description" not in value:
                value = {**value, "description": f"Filler parameter '{key}'."}
            new_props[key] = value
        out["properties"] = new_props
    return out


def to_inspect_tool_def(schema: ToolSchema, on_invoke: NoOpFn | None = None) -> ToolDef:
    """Adapt a ``ToolSchema`` to an Inspect ``ToolDef``.

    If ``on_invoke`` is None, attaches a stateless async no-op that returns the
    standard filler string but does not increment any counter (use the
    state-bound variant from ``_make_filler_noop`` for telemetry).

    Explicit ``description=`` is required because all filler tools share the
    same callable body, and ``ToolDef``'s docstring inference would collide.
    """
    if on_invoke is None:

        async def _stateless_noop(**kwargs: Any) -> str:
            return "This is a no-op tool. Recorded as filler invocation."

        on_invoke = _stateless_noop

    sanitized = _ensure_param_descriptions(schema["parameters"])
    return ToolDef(
        tool=on_invoke,
        name=schema["name"],
        description=schema["description"],
        parameters=ToolParams(**sanitized),
    )


def inject_filler_tools(state: TaskState, schemas: list[ToolSchema]) -> int:
    """Append filler tools to ``state.tools`` (additive). Returns count added.

    Each schema gets a *fresh* ``on_invoke`` closure. Inspect stores tool
    metadata on the callable object itself, so reusing a single callable
    across schemas would have them overwrite each other's registry info.
    """
    state.metadata.setdefault(FILLER_INVOCATION_KEY, 0)
    new_tools = [
        to_inspect_tool_def(schema, on_invoke=_make_filler_noop(state)).as_tool()
        for schema in schemas
    ]
    state.tools = list(state.tools) + new_tools
    return len(new_tools)
