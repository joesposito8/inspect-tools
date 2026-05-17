"""Tests for inspect_tools._seed.derive_seed."""
from inspect_tools._seed import derive_seed


def test_same_inputs_same_seed():
    assert derive_seed("s1", 1, 4000) == derive_seed("s1", 1, 4000)


def test_different_sample_id_differs():
    assert derive_seed("s1", 1, 4000) != derive_seed("s2", 1, 4000)


def test_different_epoch_differs():
    assert derive_seed("s1", 1, 4000) != derive_seed("s1", 2, 4000)


def test_different_depth_differs():
    assert derive_seed("s1", 1, 4000) != derive_seed("s1", 1, 16000)


def test_variadic_different_tool_name_differs():
    """Per-call response seeding: (trial_seed, tool_a) vs (trial_seed, tool_b)."""
    assert derive_seed("s1", 1, 4000, "tool_a") != derive_seed(
        "s1", 1, 4000, "tool_b"
    )


def test_variadic_composition_stable():
    """derive_seed(trial_seed, schema.name) gives a stable result for stable inputs."""
    trial_seed = derive_seed("s1", 1, 4000)
    a = derive_seed(trial_seed, "tool_a")
    b = derive_seed(trial_seed, "tool_a")
    assert a == b


def test_cross_process_stability():
    """Hardcoded expected output for fixed inputs.

    SHA-256("s1|1|4000")[:8] interpreted big-endian.
    """
    import hashlib

    expected = int.from_bytes(
        hashlib.sha256(b"s1|1|4000").digest()[:8], "big"
    )
    assert derive_seed("s1", 1, 4000) == expected


def test_returns_int():
    assert isinstance(derive_seed("s1", 1, 4000), int)


def test_fits_in_64_bits():
    seed = derive_seed("s1", 1, 4000)
    assert 0 <= seed < 2**64
