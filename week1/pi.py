"""Monte Carlo estimation of pi."""

import random


def estimate_pi(n, seed=None):
    """Estimate pi from ``n`` uniformly sampled points in the unit square."""
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    rng = random.Random(seed)
    inside = sum(
        rng.random() ** 2 + rng.random() ** 2 <= 1.0
        for _ in range(n)
    )
    return 1 * inside / n

