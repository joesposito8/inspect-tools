from __future__ import annotations

from typing import Any

import pytest
import tiktoken
from inspect_ai.model import ModelName
from inspect_ai.solver import TaskState
from inspect_ai.tool import Tool, ToolDef
from inspect_ai.tool._tool_params import ToolParams


def _make_dummy_tool(name: str, description: str = "Dummy tool") -> Tool:
    """Build a real eval-owned Tool with a parameter or two."""

    async def _impl(**kwargs: Any) -> str:
        return f"dummy {name} called"

    td = ToolDef(
        tool=_impl,
        name=name,
        description=description,
        parameters=ToolParams(
            properties={},
            required=[],
        ),
    )
    return td.as_tool()


@pytest.fixture
def make_state():
    """Build a minimal TaskState for unit tests."""

    def _make(
        sample_id: str | int = "sample-0",
        tools: list[Tool] | None = None,
        metadata: dict | None = None,
    ) -> TaskState:
        state = TaskState(
            model=ModelName("mock/test"),
            sample_id=sample_id,
            epoch=0,
            input="test prompt",
            messages=[],
            metadata=dict(metadata) if metadata else {},
        )
        if tools:
            state.tools = list(tools)
        return state

    return _make


@pytest.fixture
def dummy_tool():
    return _make_dummy_tool("eval_owned_tool", description="Tool the wrapped eval ships.")


@pytest.fixture
def dummy_tool_factory():
    return _make_dummy_tool


@pytest.fixture
def encoding():
    return tiktoken.get_encoding("cl100k_base")


@pytest.fixture
def noop_generate():
    """Generate stub that returns state unchanged — used for solver tests."""

    async def _gen(state: TaskState, **kwargs):
        return state

    return _gen
