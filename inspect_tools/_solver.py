"""@solver context_exhaustion(...) + _fill_prefix bisection."""
from __future__ import annotations

import random

from inspect_ai.model import get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool._tool_info import ToolInfo
from inspect_ai.tool._tool_util import tool_to_tool_info

from inspect_tools._inject import schema_to_tool_def, schema_to_tool_info
from inspect_tools._library import filter_pool, load_corpus
from inspect_tools._seed import derive_seed
from inspect_tools.schema import ToolSchema

_CONTENT_CATEGORY = ["general_popular"]


async def _fill_prefix(
    model,
    base_infos: list[ToolInfo],
    pool_infos: list[ToolInfo],
    target: int,
) -> int:
    """Smallest n such that count(base + pool_infos[:n]) >= target,
    or len(pool_infos) if unreachable.

    Galloping search (double until cross), then binary search inside the bracket.
    ~5-8 model.count_tool_tokens calls per trial; free for Anthropic.
    """
    n = 1
    while n <= len(pool_infos):
        if await model.count_tool_tokens(base_infos + pool_infos[:n]) >= target:
            break
        n *= 2
    lo, hi = n // 2, min(n, len(pool_infos))
    while lo < hi:
        mid = (lo + hi) // 2
        if await model.count_tool_tokens(base_infos + pool_infos[:mid]) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@solver
def context_exhaustion(
    target_tokens: int | None = None,
    n_filler: int = 5,
    domain_filter: list[str] | None = None,
    exclude_names: list[str] | None = None,
    extend_with: list[ToolSchema] | None = None,
) -> Solver:
    """Inject MCP tool schemas into the eval's tools list.

    - target_tokens set:  bisect to that token total (model-aware via count_tool_tokens)
    - target_tokens None: inject min(n_filler, len(pool)) schemas (no model calls)

    Multi-depth sweeps: define `@task def my_eval(target_tokens: int)` and run
    `eval_set([my_eval(d) for d in depths])`. Variance studies: pass `epochs=N`.
    """
    # Hoist at @solver construction time
    filtered_pool = filter_pool(
        load_corpus(),
        content_category=_CONTENT_CATEGORY,
        domain_filter=domain_filter,
        exclude_names=exclude_names,
        extend_with=extend_with,
    )
    pool_tool_infos = [schema_to_tool_info(s) for s in filtered_pool]
    pool_filter_dict = {
        "target_tokens": target_tokens,
        "n_filler": n_filler,
        "content_category": _CONTENT_CATEGORY,
        "domain_filter": domain_filter,
        "exclude_names": exclude_names,
        "extend_with_names": [s.name for s in extend_with] if extend_with else [],
    }

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        trial_seed = derive_seed(state.sample_id, state.epoch, target_tokens or 0)
        rng = random.Random(trial_seed)
        indices = list(range(len(filtered_pool)))
        rng.shuffle(indices)

        if target_tokens is not None and filtered_pool:
            model = get_model()
            base_infos = [tool_to_tool_info(t) for t in state.tools]
            shuffled_infos = [pool_tool_infos[i] for i in indices]
            n = await _fill_prefix(model, base_infos, shuffled_infos, target_tokens)
            chosen = [filtered_pool[i] for i in indices[:n]]
        else:
            n = min(n_filler, len(filtered_pool))
            chosen = [filtered_pool[i] for i in indices[:n]]

        for schema in chosen:
            state.tools.append(
                schema_to_tool_def(
                    schema,
                    state=state,
                    solver_namespace="context_exhaustion",
                    trial_seed=trial_seed,
                ).as_tool()
            )

        manifest = state.metadata.setdefault("inspect_tools", {}).setdefault(
            "context_exhaustion", {}
        )
        manifest["injected_tool_names"] = [s.name for s in chosen]
        manifest["pool_filter"] = pool_filter_dict
        manifest["library_seed_per_sample"] = trial_seed
        manifest["target_tokens"] = target_tokens
        manifest.setdefault("invocations", 0)  # execute closures increment from here

        return state

    return solve
