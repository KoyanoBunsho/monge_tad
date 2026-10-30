"""Evaluation utilities for fixed-leaf deepTAD hierarchy experiments."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


def dual_boundary_peak_support(
    domains: Sequence[tuple[int, int]],
    peak_intervals: np.ndarray,
    *,
    resolution: int,
    tolerance_bp: int,
) -> tuple[int, int]:
    """Count domains with a peak near both genomic endpoints."""
    peaks = np.asarray(peak_intervals, dtype=np.int64)
    if peaks.ndim != 2 or peaks.shape[1:] != (2,):
        raise ValueError("peak_intervals must have shape [peaks, 2]")
    if resolution < 1 or tolerance_bp < 0:
        raise ValueError("resolution must be positive and tolerance nonnegative")

    def supported(coordinate: int) -> bool:
        low = coordinate - tolerance_bp
        high = coordinate + tolerance_bp
        return bool(np.any((peaks[:, 0] <= high) & (peaks[:, 1] >= low)))

    total = 0
    dual = 0
    for left, right in domains:
        if right <= left:
            raise ValueError("domain must be nonempty")
        total += 1
        dual += supported(left * resolution) and supported(right * resolution)
    return int(dual), total


@dataclass(frozen=True)
class TADAdjustedR2:
    distances: np.ndarray
    scores: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores)) if len(self.scores) else float("nan")


@dataclass(frozen=True)
class DistanceNormalizedContacts:
    """Prefix sums of observed/expected contacts along genomic diagonals."""

    size: int
    maximum_distance: int
    prefixes: tuple[np.ndarray, ...]

    @classmethod
    def from_contacts(
        cls, contacts: np.ndarray, *, maximum_distance: int,
        validate_finite: bool = True,
    ) -> "DistanceNormalizedContacts":
        # Keep a memory-mapped chromosome matrix in its source dtype. Only the
        # short diagonals used below need float64 precision; converting the
        # full matrix would allocate many gigabytes at 10 kb resolution.
        matrix = np.asarray(contacts)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("contacts must be a square matrix")
        if maximum_distance < 1 or (validate_finite and np.any(~np.isfinite(matrix))):
            raise ValueError("contacts must be finite and maximum_distance positive")
        prefixes = []
        for distance in range(min(maximum_distance, len(matrix) - 1) + 1):
            diagonal = np.asarray(np.diagonal(matrix, offset=distance), dtype=np.float64)
            expected = float(np.mean(diagonal))
            normalized = (
                diagonal / expected if expected > 0 else np.zeros_like(diagonal)
            )
            prefixes.append(np.concatenate(([0.0], np.cumsum(normalized))))
        return cls(len(matrix), len(prefixes) - 1, tuple(prefixes))

    def interval_mean(self, left: int, right: int, *, ignore_diagonals: int = 2) -> float:
        if left < 0 or right > self.size or right <= left:
            raise ValueError("invalid interval")
        total = 0.0
        count = 0
        upper = min(self.maximum_distance, right - left - 1)
        for distance in range(ignore_diagonals + 1, upper + 1):
            total += float(
                self.prefixes[distance][right - distance]
                - self.prefixes[distance][left]
            )
            count += right - left - distance
        return total / count if count else float("nan")

    def structural_contrast(
        self, left: int, right: int, *, ignore_diagonals: int = 2
    ) -> float:
        """Return inside O/E minus same-sized upstream/downstream O/E."""
        inside = self.interval_mean(left, right, ignore_diagonals=ignore_diagonals)
        length = right - left
        flanks = []
        if left - length >= 0:
            flanks.append(
                self.interval_mean(
                    left - length, left, ignore_diagonals=ignore_diagonals
                )
            )
        if right + length <= self.size:
            flanks.append(
                self.interval_mean(right, right + length, ignore_diagonals=ignore_diagonals)
            )
        finite = [value for value in flanks if np.isfinite(value)]
        if not np.isfinite(inside) or not finite:
            return float("-inf")
        return inside - float(np.mean(finite))


def tad_adjusted_r_squared(
    contacts: np.ndarray,
    domains: Sequence[tuple[int, int]],
    *,
    maximum_distance: int,
) -> TADAdjustedR2:
    """Compute the distance-stratified TAD-adjR2 defined by An et al.

    Coordinates are zero-based and half-open.  At each genomic distance, a
    contact is assigned to its smallest containing domain; contacts covered by
    nested children are thereby excluded from their parents.  Remaining
    contacts form the gap group.  Each group is represented by its mean contact
    frequency and the residual variance receives the published ``n-p-1``
    degrees-of-freedom correction, where ``p`` is the number of eligible TADs.
    """
    matrix = np.asarray(contacts)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("contacts must be a square matrix")
    for start in range(0, len(matrix), 256):
        if np.any(~np.isfinite(matrix[start:start + 256])):
            raise ValueError("contacts must be finite")
    if maximum_distance < 1:
        raise ValueError("maximum_distance must be positive")

    unique = sorted(
        {(int(left), int(right)) for left, right in domains},
        key=lambda interval: (
            interval[1] - interval[0],
            interval[0],
            interval[1],
        ),
        reverse=True,
    )
    if any(left < 0 or right <= left or right > len(matrix) for left, right in unique):
        raise ValueError("domain lies outside the contact matrix")

    distances: list[int] = []
    scores: list[float] = []
    for distance in range(1, min(maximum_distance, len(matrix) - 1) + 1):
        observed = np.asarray(np.diagonal(matrix, offset=distance), dtype=np.float64)
        labels = np.full(len(observed), -1, dtype=np.int64)
        eligible = 0
        # Large domains are written first, then overwritten by their smallest
        # containing descendant.
        for label, (left, right) in enumerate(unique):
            if right - left <= distance:
                continue
            labels[left : right - distance] = label
            eligible += 1

        predicted = np.empty_like(observed)
        for label in np.unique(labels):
            member = labels == label
            predicted[member] = float(np.mean(observed[member]))

        residual_df = len(observed) - eligible - 1
        total = float(np.sum((observed - float(np.mean(observed))) ** 2))
        if residual_df <= 0 or total <= 0:
            continue
        residual = float(np.sum((observed - predicted) ** 2))
        score = 1.0 - (residual / residual_df) / (total / (len(observed) - 1))
        distances.append(distance)
        scores.append(score)

    return TADAdjustedR2(
        distances=np.asarray(distances, dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float64),
    )
