"""End-to-end smoke: wrap a stub Task at 2 depths, confirm both layers of the
manifest are written, and metrics aggregate across depths."""
from __future__ import annotations

import json

from inspect_ai import Task, eval, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState, generate

from inspect_tools import context_exhaustion, score_at_depth


@scorer(metrics=[score_at_depth()])
def constant_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=1.0)
    return score


@task
def stub_task(target_tokens: int) -> Task:
    return Task(
        dataset=MemoryDataset(
            [
                Sample(input="hello", id=f"s{i}", target="ok")
                for i in range(3)
            ]
        ),
        solver=[context_exhaustion(target_tokens=target_tokens), generate()],
        scorer=constant_scorer(),
    )


_PER_SAMPLE_KEYS = {
    "injected_tool_names",
    "pool_filter",
    "library_seed_per_sample",
    "sampling_seed",
    "target_tokens",
    "actual_tokens",
    "tool_corpus_version",
    "filler_invocations",
}

_RUN_LEVEL_KEYS = {
    "package_version",
    "wrapped_task",
    "wrapped_task_version",
    "tool_corpus_version",
    "tokenizer_id",
    "depth_schedule",
    "models_evaluated",
}


def test_e2e_two_depths_manifest_and_metrics(tmp_path, capsys):
    depths = [2_000, 8_000]
    logs = eval(
        [stub_task(target_tokens=d) for d in depths],
        model="mockllm/model",
        log_dir=str(tmp_path),
        display="none",
    )
    assert len(logs) == 2

    for log, depth in zip(logs, depths):
        assert log.status == "success", f"depth {depth} status={log.status}"

        # Run-level manifest in EvalLog.eval.metadata
        run_md = (log.eval.metadata or {}).get("inspect_tools", {}).get(
            "context_exhaustion", {}
        )
        missing = _RUN_LEVEL_KEYS - set(run_md.keys())
        assert not missing, f"depth {depth}: run-level missing {missing}; got {run_md}"
        assert run_md["wrapped_task"] == "stub_task"
        assert run_md["models_evaluated"] == ["mockllm/model"]
        assert run_md["depth_schedule"] == [depth]
        assert run_md["tokenizer_id"] == "mockllm/model:count_tool_tokens"

        # Per-sample manifest in each EvalSample.metadata
        assert log.samples, f"depth {depth}: no samples"
        for s in log.samples:
            per = (s.metadata or {}).get("inspect_tools", {}).get(
                "context_exhaustion", {}
            )
            missing = _PER_SAMPLE_KEYS - set(per.keys())
            assert not missing, f"depth {depth} sample {s.id}: missing {missing}"
            assert per["target_tokens"] == depth
            assert per["sampling_seed"] == per["library_seed_per_sample"]

    # Metrics: aggregate scores from both logs by depth and confirm
    # score_at_depth yields the 2-cell dict.
    from inspect_ai.scorer import SampleScore
    sample_scores: list[SampleScore] = []
    for log in logs:
        for s in log.samples or []:
            sc = next(iter(s.scores.values()))
            sample_scores.append(
                SampleScore(
                    score=sc,
                    sample_id=s.id,
                    sample_metadata=s.metadata,
                )
            )
    out = score_at_depth()(sample_scores)
    for d in depths:
        assert str(d) in out
        assert out[f"{d}_n"] == 3.0

    # Print a populated manifest snippet for the completion report.
    snippet = {
        "run_level": (logs[0].eval.metadata or {})["inspect_tools"]["context_exhaustion"],
        "per_sample_first": (logs[0].samples[0].metadata or {})["inspect_tools"][
            "context_exhaustion"
        ],
    }
    # Trim injected_tool_names for the printed snippet
    ps = dict(snippet["per_sample_first"])
    ps["injected_tool_names"] = (
        ps["injected_tool_names"][:3] + ["..."] if ps.get("injected_tool_names") else []
    )
    snippet["per_sample_first"] = ps
    print("\n=== e2e manifest snippet ===")
    print(json.dumps(snippet, indent=2, default=str))
