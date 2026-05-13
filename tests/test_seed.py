from inspect_context_pressure._seed import derive_seed


def test_derive_seed_deterministic():
    a = derive_seed("sample-1", 4_000, 0)
    b = derive_seed("sample-1", 4_000, 0)
    assert a == b


def test_derive_seed_varies_on_depth():
    a = derive_seed("sample-1", 4_000, 0)
    b = derive_seed("sample-1", 64_000, 0)
    assert a != b


def test_derive_seed_varies_on_sample_id():
    a = derive_seed("sample-1", 4_000, 0)
    b = derive_seed("sample-2", 4_000, 0)
    assert a != b


def test_derive_seed_varies_on_shape_seed():
    a = derive_seed("sample-1", 4_000, 0)
    b = derive_seed("sample-1", 4_000, 1)
    assert a != b


def test_derive_seed_int_sample_id():
    """sample_id may be int per Inspect contract."""
    a = derive_seed(42, 4_000, 0)
    b = derive_seed(42, 4_000, 0)
    assert a == b
    assert isinstance(a, int)
