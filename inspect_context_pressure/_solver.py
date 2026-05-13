"""``context_pressure()`` Solver — additive tool-schema injection at depth.

Precedence rule:
- ``depth_schedule`` not None and ``state.metadata['target_tokens']`` is set:
  N injected schemas is depth-derived. The Solver counts the eval's own
  tools-parameter token baseline and greedily samples additional schemas
  until total tokens reach ``target_tokens``. ``pool_size`` is ignored.
- Otherwise: N = ``pool_size`` (literal count). Mirrors AgentHarm's
  ``n_irrelevant_tools`` semantics.
"""

from __future__ import annotations

import random

import tiktoken
from inspect_ai.solver import Generate, Solver, TaskState, solver

from inspect_context_pressure._inject import (
    FILLER_INVOCATION_KEY,
    count_schema_tokens,
    count_tools_tokens,
    inject_filler_tools,
)
from inspect_context_pressure._library import filter_pool, load_fixture_library
from inspect_context_pressure._seed import derive_seed
from inspect_context_pressure._types import ToolSchema

DEFAULT_DEPTH_SCHEDULE: list[int] = [4_000, 16_000, 64_000, 256_000]

CATEGORY_A = "A_general_popular"
CATEGORY_B = "B_vacuous_controls"


def _greedy_n_for_target(
    pool_costs: list[int],
    remaining_tokens: int,
    rng: random.Random,
) -> int:
    """Pick schema count whose cumulative tokens reach ``remaining_tokens``.

    Walks an ``rng``-shuffled view of the precomputed cost list, summing until
    the budget is met or the pool is exhausted.
    """
    if remaining_tokens <= 0:
        return 0
    order = list(range(len(pool_costs)))
    rng.shuffle(order)
    cumulative = 0
    for i, idx in enumerate(order, start=1):
        cumulative += pool_costs[idx]
        if cumulative >= remaining_tokens:
            return i
    return len(pool_costs)


@solver
def context_pressure(
    composition_spec: dict | None = None,
    pool_size: int = 5,
    depth_schedule: list[int] | None = DEFAULT_DEPTH_SCHEDULE,
    tokenizer: str = "cl100k_base",
    domain_filter: list[str] | None = None,
    content_category: str = CATEGORY_A,
    exclude_names: list[str] | None = None,
    extend_with: list[ToolSchema] | None = None,
) -> Solver:
    """Inject sampled MCP tool schemas into the wrapped task's ``tools`` parameter.

    Args:
        composition_spec: Optional dict with ``tool_categories`` and/or
            ``exclude_keywords``. ICP-5 will populate this from a classifier;
            for ICP-3 callers may pass a literal dict or ``None``.
        pool_size: Literal schema count when ``depth_schedule`` is None;
            ignored when ``depth_schedule`` is provided.
        depth_schedule: List of depth targets (token counts). When set,
            schema count scales to ``state.metadata['target_tokens']``. Pass
            ``None`` for count-based sampling via ``pool_size``.
        tokenizer: tiktoken encoding name. cl100k_base default; this is a
            counting proxy and approximates non-OpenAI provider tokenizers.
        domain_filter: Restrict pool to these domains.
        content_category: ``A_general_popular`` (default) or
            ``B_vacuous_controls`` for ICP-7's Gamage-vs-Levy isolation arm.
        exclude_names: Block schemas by name (avoids collision with the
            wrapped eval's own tool names).
        extend_with: User-supplied schemas appended to the shipped pool.

    The Solver does not invoke ``generate``; chain it before the wrapped
    task's own solver list so the model run sees the inflated ``tools``
    parameter. The wrapped task's scorer is untouched.
    """
    encoding = tiktoken.get_encoding(tokenizer)
    pool = filter_pool(
        load_fixture_library(),
        domain_filter=domain_filter,
        content_category=content_category,
        exclude_names=exclude_names,
        extend_with=extend_with,
        composition_spec=composition_spec,
    )
    pool_costs = [count_schema_tokens(s, encoding) for s in pool]

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.metadata is None:
            state.metadata = {}
        state.metadata.setdefault(FILLER_INVOCATION_KEY, 0)

        if not pool:
            return state

        target = state.metadata.get("target_tokens")
        shape_seed = state.metadata.get("shape_seed", 0)
        seed = derive_seed(state.sample_id, target or 0, shape_seed)
        rng = random.Random(seed)

        if depth_schedule is not None and target is not None:
            baseline = count_tools_tokens(state.tools, encoding)
            n = _greedy_n_for_target(pool_costs, target - baseline, rng)
        else:
            n = min(pool_size, len(pool))

        if n == 0:
            return state

        sampled = rng.sample(pool, n)
        inject_filler_tools(state, sampled)
        return state

    return solve
