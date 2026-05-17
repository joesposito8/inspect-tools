"""Shared pytest fixtures."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import ModelName
from inspect_ai.solver import TaskState
from inspect_ai.tool import Tool, ToolDef
from inspect_ai.tool._tool_params import ToolParams


def _make_dummy_tool(name: str, description: str = "Dummy tool") -> Tool:
    async def _impl(**kwargs: Any) -> str:
        return f"dummy {name} called"

    td = ToolDef(
        tool=_impl,
        name=name,
        description=description,
        parameters=ToolParams(properties={}, required=[]),
    )
    return td.as_tool()


@pytest.fixture
def make_state():
    def _make(
        sample_id: str | int = "sample-0",
        epoch: int = 1,
        tools: list[Tool] | None = None,
        metadata: dict | None = None,
    ) -> TaskState:
        state = TaskState(
            model=ModelName("mock/test"),
            sample_id=sample_id,
            epoch=epoch,
            input="test prompt",
            messages=[],
            metadata=dict(metadata) if metadata else {},
        )
        if tools:
            state.tools = list(tools)
        return state

    return _make


@pytest.fixture
def dummy_tool() -> Tool:
    return _make_dummy_tool("eval_owned_tool", description="Tool the wrapped eval ships.")


@pytest.fixture
def dummy_tool_factory():
    return _make_dummy_tool


@pytest.fixture
def noop_generate():
    async def _gen(state: TaskState, **kwargs):
        return state

    return _gen


@pytest.fixture
def mock_model():
    """Async stub. count_tool_tokens returns 100 * len(tools) by default; tests can
    override .count_tool_tokens.side_effect or .return_value as needed."""
    model = AsyncMock()
    model.count_tool_tokens = AsyncMock(side_effect=lambda tools: 100 * len(tools))
    return model
