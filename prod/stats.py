"""Accuracy intervals. Wilson is the default for every reported accuracy."""
from math import sqrt


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval. Returns (lo, hi) on the [0, 1] scale, or None if n=0."""
    if n <= 0:
        return None
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def accuracy_with_ci(successes: int, n: int) -> dict:
    lo_hi = wilson_interval(successes, n)
    return {
        "n": n,
        "successes": successes,
        "accuracy": (successes / n) if n else None,
        "wilson_95": {"lo": lo_hi[0], "hi": lo_hi[1]} if lo_hi else None,
    }
