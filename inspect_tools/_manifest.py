"""Run-level manifest writer.

Writes static-per-task fields to EvalSpec.metadata at task start so analyzers
can read package_version, wrapped_task, etc. from EvalLog.eval.metadata.

Persistence path: inspect_ai/log/_recorders/eval.py reuses the same EvalSpec
reference between log_start (start.json) and log_finish (header.json).
read_eval_log returns header.json. So mutations to data.spec.metadata in
on_task_start land in the canonical read path even though start.json is stale.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from inspect_ai.hooks import Hooks, TaskStart, hooks

from inspect_tools import _solver
from inspect_tools._library import corpus_sha


def _package_version() -> str:
    try:
        return version("inspect-tools")
    except PackageNotFoundError:
        return "0.0.0+unknown"


@hooks(
    name="inspect_tools_run_manifest",
    description="Write run-level inspect_tools manifest fields to EvalSpec.metadata.",
)
class RunManifestHook(Hooks):
    async def on_task_start(self, data: TaskStart) -> None:
        if not _solver._CONTEXT_EXHAUSTION_ACTIVE:
            return
        spec = data.spec
        md = spec.metadata if spec.metadata is not None else {}
        sub = md.setdefault("inspect_tools", {}).setdefault("context_exhaustion", {})
        sub["package_version"] = _package_version()
        sub["wrapped_task"] = spec.task
        sub["wrapped_task_version"] = spec.task_version
        sub["tool_corpus_version"] = corpus_sha()
        sub["tokenizer_id"] = f"{spec.model}:count_tool_tokens"
        sub["models_evaluated"] = [spec.model]
        target = (spec.task_args or {}).get("target_tokens")
        sub["depth_schedule"] = [target] if target is not None else []
        spec.metadata = md
