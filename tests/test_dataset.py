from inspect_ai.dataset import MemoryDataset, Sample

from inspect_tools import replicate_across_depths


def _samples():
    return [
        Sample(input="prompt-a", target="target-a", id="s1", metadata={"category": "x"}),
        Sample(input="prompt-b", target="target-b", id="s2", metadata={"category": "y"}),
    ]


def test_replicate_expands_correctly():
    ds = MemoryDataset(_samples())
    out = replicate_across_depths(ds, [4_000, 16_000, 64_000])
    assert len(out) == 2 * 3
    targets = [s.metadata["target_tokens"] for s in out]
    assert targets == [4_000, 16_000, 64_000, 4_000, 16_000, 64_000]


def test_replicate_preserves_metadata():
    ds = MemoryDataset(_samples())
    out = replicate_across_depths(ds, [4_000])
    for replica in out:
        assert replica.metadata["category"] in {"x", "y"}
        assert replica.metadata["target_tokens"] == 4_000


def test_replicate_preserves_id_and_input():
    ds = MemoryDataset(_samples())
    out = replicate_across_depths(ds, [4_000, 16_000])
    inputs = [s.input for s in out]
    ids = [s.id for s in out]
    assert inputs == ["prompt-a", "prompt-a", "prompt-b", "prompt-b"]
    assert ids == ["s1", "s1", "s2", "s2"]


def test_replicate_handles_empty_metadata():
    ds = MemoryDataset([Sample(input="p", target="t", id="s")])
    out = replicate_across_depths(ds, [4_000])
    assert out[0].metadata["target_tokens"] == 4_000


def test_replicate_does_not_mutate_input_dataset():
    """Deep-copy guarantees: tweaking a replica's metadata must not affect originals."""
    ds = MemoryDataset(_samples())
    out = replicate_across_depths(ds, [4_000])
    out[0].metadata["category"] = "mutated"
    assert ds.samples[0].metadata["category"] == "x"
