"""Percentile intervals for ratios under whole-cluster resampling."""
from __future__ import annotations

import numpy as np


def ratio_interval(values: np.ndarray, iterations: int, seed: int) -> dict:
    """Resample cluster numerator/denominator sums with equal cluster probability.

    The estimator is the pooled ratio, not an unweighted average of cluster
    ratios. Zero-denominator replicates are excluded and counted explicitly.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        raise ValueError('Expected a nonempty cluster-by-2 array')
    if iterations <= 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid bootstrap iterations or cluster sums')
    rng = np.random.default_rng(seed)
    draws = []
    # Bound temporary allocations independently of the number of iterations.
    batch = max(1, min(256, 1000000 // len(values)))
    for start in range(0, iterations, batch):
        indices = rng.integers(0, len(values), size=(min(batch, iterations-start), len(values)))
        sums = values[indices].sum(axis=1)
        draws.extend((sums[sums[:, 1] > 0, 0] / sums[sums[:, 1] > 0, 1]).tolist())
    total = values.sum(axis=0)
    bounds = np.quantile(draws, [.025, .975]).tolist() if draws else [None, None]
    return dict(estimate=float(total[0]/total[1]) if total[1] else None,
                lower=bounds[0], upper=bounds[1], clusters=len(values),
                valid_replicates=len(draws), undefined_replicates=iterations-len(draws),
                degenerate_cluster_count=len(values) < 2)
