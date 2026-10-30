"""Deterministic Python reproduction of deepTAD's Wilcoxon boundary filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu


@dataclass(frozen=True)
class BoundaryFilterResult:
    candidate_indices: np.ndarray
    candidate_pvalues: np.ndarray
    filtered_indices: np.ndarray
    gap_indices: np.ndarray


def _scale_upper_diagonals(contacts: np.ndarray, maximum_offset: int) -> np.ndarray:
    scaled = np.asarray(contacts, dtype=np.float64).copy()
    n = len(scaled)
    for offset in range(1, min(maximum_offset, n - 1) + 1):
        rows = np.arange(n - offset)
        values = contacts[rows, rows + offset]
        standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        if standard_deviation > 0 and np.isfinite(standard_deviation):
            scaled[rows, rows + offset] = (values - float(np.mean(values))) / standard_deviation
        else:
            scaled[rows, rows + offset] = np.nan
    return scaled


def _boundary_samples(
    matrix: np.ndarray, boundary: int, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return crossing-diamond and within-side samples for a zero-based boundary."""
    n = len(matrix)
    left = np.arange(max(0, boundary - window_size + 1), boundary + 1)
    right = np.arange(boundary + 1, min(n, boundary + window_size + 1))
    crossing = matrix[np.ix_(left, right)].reshape(-1)

    upstream_start = max(0, boundary - window_size)
    upstream = matrix[upstream_start : boundary + 1, upstream_start : boundary + 1]
    downstream_end = min(n, boundary + window_size + 1)
    downstream = matrix[boundary + 1 : downstream_end, boundary + 1 : downstream_end]
    within = np.concatenate(
        (
            upstream[np.triu_indices(len(upstream), k=1)],
            downstream[np.triu_indices(len(downstream), k=1)],
        )
    )
    return crossing[np.isfinite(crossing)], within[np.isfinite(within)]


def _boundary_samples_from_diagonal_stats(
    contacts: np.ndarray,
    boundary: int,
    window_size: int,
    means: np.ndarray,
    deviations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the public upper-diagonal z-scores without a dense scaled copy."""
    n = len(contacts)

    def standardized(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        offsets = columns - rows
        values = np.asarray(contacts[rows, columns], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = (values - means[offsets]) / deviations[offsets]
        return scores[np.isfinite(scores)]

    left = np.arange(max(0, boundary - window_size + 1), boundary + 1)
    right = np.arange(boundary + 1, min(n, boundary + window_size + 1))
    crossing = standardized(np.broadcast_to(left[:, None], (len(left), len(right))),
                            np.broadcast_to(right[None, :], (len(left), len(right))))
    upstream = np.arange(max(0, boundary - window_size), boundary + 1)
    downstream = np.arange(boundary + 1, min(n, boundary + window_size + 1))
    up_i, up_j = np.triu_indices(len(upstream), k=1)
    down_i, down_j = np.triu_indices(len(downstream), k=1)
    within = np.concatenate((
        standardized(upstream[up_i], upstream[up_j]),
        standardized(downstream[down_i], downstream[down_j]),
    ))
    return crossing, within


def _processing_mask(contacts: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(contacts)
    gaps = np.zeros(n, dtype=bool)
    for index in range(n):
        left = max(0, index - window_size)
        right = min(n, index + window_size + 1)
        gaps[index] = float(np.sum(contacts[index, left:right])) == 0.0

    process = np.zeros(n, dtype=bool)
    valid = np.flatnonzero(~gaps)
    if len(valid):
        run_start = 0
        for position in range(1, len(valid) + 1):
            if position == len(valid) or valid[position] != valid[position - 1] + 1:
                run = valid[run_start:position]
                if len(run) >= 4:
                    process[run] = True
                run_start = position
    return process, np.flatnonzero(gaps)


def filter_deeptad_boundaries(
    contacts: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    window_size: int,
    pvalue_cutoff: float = 0.05,
) -> BoundaryFilterResult:
    """Apply the public deepTAD statistical filter to zero-based candidates."""
    matrix = np.asarray(contacts)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("contacts must be a square matrix")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("contacts must be finite")
    if candidates.ndim != 1 or np.any(np.diff(candidates) <= 0):
        raise ValueError("candidate_indices must be a strictly increasing vector")
    if len(candidates) and (
        int(candidates[0]) < 0 or int(candidates[-1]) >= len(matrix)
    ):
        raise ValueError("candidate index lies outside the contact matrix")
    if window_size < 1 or not 0 < pvalue_cutoff < 1:
        raise ValueError("invalid filter window or p-value cutoff")

    process, gaps = _processing_mask(matrix, window_size)
    means = np.zeros(2 * window_size + 1, dtype=np.float64)
    deviations = np.zeros(2 * window_size + 1, dtype=np.float64)
    for offset in range(1, min(2 * window_size, len(matrix) - 1) + 1):
        diagonal = np.asarray(np.diagonal(matrix, offset=offset), dtype=np.float64)
        means[offset] = float(np.mean(diagonal))
        deviations[offset] = float(np.std(diagonal, ddof=1)) if len(diagonal) > 1 else 0.0
    pvalues = np.ones(len(candidates), dtype=np.float64)
    for position, boundary in enumerate(candidates):
        if not process[boundary] or boundary == len(matrix) - 1:
            continue
        crossing, within = _boundary_samples_from_diagonal_stats(
            matrix, int(boundary), window_size, means, deviations,
        )
        if len(crossing) == 0 or len(within) == 0:
            continue
        pvalues[position] = float(
            mannwhitneyu(
                crossing,
                within,
                alternative="less",
                method="asymptotic",
                use_continuity=True,
            ).pvalue
        )

    retained = list(candidates[pvalues < pvalue_cutoff])
    retained_pvalues = {
        int(index): float(pvalue)
        for index, pvalue in zip(candidates, pvalues)
    }
    adjacent_left = [
        int(left)
        for left, right in zip(retained[:-1], retained[1:])
        if right == left + 1
    ]
    for left in adjacent_left:
        right = left + 1
        remove = max(
            (left, right),
            key=lambda index: (retained_pvalues[index], index),
        )
        if remove in retained:
            retained.remove(remove)

    return BoundaryFilterResult(
        candidate_indices=candidates,
        candidate_pvalues=pvalues,
        filtered_indices=np.asarray(retained, dtype=np.int64),
        gap_indices=gaps.astype(np.int64),
    )
