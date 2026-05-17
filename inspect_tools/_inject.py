"""ToolSchema → Inspect adapters.

schema_to_tool_info: counting/metadata-only ToolInfo (no state, no closure).
schema_to_tool_def:  state-bound ToolDef with telemetry + response generation.

Mirrors inspect_ai/tool/_mcp/_local.py:201-250.
"""
from __future__ import annotations

import random
from typing import Any, Callable

from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolDef
from inspect_ai.tool._tool_info import ToolInfo
from inspect_ai.tool._tool_params import ToolParams

from inspect_tools._seed import derive_seed
from inspect_tools._synthesize import synthesize_response
from inspect_tools.schema import ToolSchema

ResponseFn = Callable[[ToolSchema, dict, random.Random], "dict | str"]


def _build_parameters(schema: ToolSchema) -> ToolParams:
    """MCP-shape inputSchema dict → Inspect-shape ToolParams; auto-fill missing param descs."""
    parameters = ToolParams.model_validate(schema.inputSchema)
    for name, param in parameters.properties.items():
        param.description = param.description or name
    return parameters


def schema_to_tool_info(schema: ToolSchema) -> ToolInfo:
    """Counting/metadata-only ToolInfo. Used at @solver-construction time."""
    return ToolInfo(
        name=schema.name,
        description=schema.description,
        parameters=_build_parameters(schema),
    )


def schema_to_tool_def(
    schema: ToolSchema,
    *,
    state: TaskState,
    solver_namespace: str,
    trial_seed: int,
    response_fn: ResponseFn = synthesize_response,
) -> ToolDef:
    """Executable ToolDef. Writes invocation telemetry under
    state.metadata['inspect_tools'][solver_namespace]['invocations'].

    Per-(trial, tool) RNG seeding ensures same-trial reproducibility of response packages.
    """

    async def execute(**kwargs: Any) -> dict | str:
        sub = state.metadata.setdefault("inspect_tools", {}).setdefault(
            solver_namespace, {}
        )
        sub["invocations"] = sub.get("invocations", 0) + 1
        call_rng = random.Random(derive_seed(trial_seed, schema.name))
        return response_fn(schema, kwargs, call_rng)

    return ToolDef(
        execute,
        name=schema.name,
        description=schema.description,
        parameters=_build_parameters(schema),
    )
