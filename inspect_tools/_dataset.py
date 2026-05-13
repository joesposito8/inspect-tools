"""Dataset helpers for the context_exhaustion Solver."""

from __future__ import annotations

import copy
from typing import Iterable

from inspect_ai.dataset import Dataset, MemoryDataset, Sample


def replicate_across_depths(
    dataset: Dataset | Iterable[Sample],
    depth_schedule: list[int],
) -> MemoryDataset:
    """Expand each sample into one replica per target depth.

    Each replica is deep-copied and tagged with
    ``metadata['target_tokens']`` set to the depth value. This is the metric
    grouping key downstream depth-aware metrics (ICP-6) use.

    Note: this helper does not vary ``shape_seed`` — pool replication for
    diversity is a future helper. The Solver derives a deterministic per-trial
    seed from ``(sample_id, target_tokens, shape_seed)``, so all replicates
    produced here share ``shape_seed=0`` and differ only by depth.
    """
    out: list[Sample] = []
    name: str | None = None
    location: str | None = None
    if isinstance(dataset, Dataset):
        name = dataset.name
        location = dataset.location

    for sample in dataset:
        for depth in depth_schedule:
            replica = copy.deepcopy(sample)
            metadata = dict(replica.metadata) if replica.metadata else {}
            metadata["target_tokens"] = depth
            replica.metadata = metadata
            out.append(replica)

    return MemoryDataset(samples=out, name=name, location=location)
