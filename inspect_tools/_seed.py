"""Per-trial seed derivation."""
import hashlib


def derive_seed(*parts: object) -> int:
    """SHA-256 over `|`-joined parts, first 8 bytes as a 64-bit int.

    Variadic so callers compose any tuple — the trial seed comes from
    (sample_id, epoch, target_tokens); per-tool-call seeds come from
    (trial_seed, schema.name).
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
