import hashlib


def derive_seed(sample_id: str | int, depth: int, shape_seed: int) -> int:
    """Derive a stable 64-bit seed for per-trial schema sampling.

    Uses SHA-256 over a delimited string instead of Python's built-in ``hash``
    because the latter is randomized per process when ``PYTHONHASHSEED`` is
    unset. Reproducibility across runs is a hard requirement.
    """
    payload = f"{sample_id}|{depth}|{shape_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")
