"""Full cubic CKY, exact Knuth, and empirical window decoders."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DecodeResult:
    score: float
    spans: frozenset[tuple[int, int]]
    chart: np.ndarray
    splits: np.ndarray
    candidates: int
    fallbacks: int = 0


def _backtrace(splits: np.ndarray, length: int) -> frozenset[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()

    def visit(i: int, j: int) -> None:
        if j - i < 2:
            return
        spans.add((i, j))
        k = int(splits[i, j])
        if not i < k < j:
            raise ValueError(f"invalid split {k} for span ({i}, {j})")
        visit(i, k)
        visit(k, j)

    if length >= 2:
        visit(0, length)
    return frozenset(spans)


def full_cky(span_scores: np.ndarray) -> DecodeResult:
    """Decode a maximum-scoring binary tree in cubic time.

    ``span_scores[i, j]`` is the local score for half-open span [i,j).
    The smallest split wins exact ties.
    """
    n = span_scores.shape[0] - 1
    chart = np.zeros((n + 1, n + 1), dtype=np.float64)
    splits = np.full((n + 1, n + 1), -1, dtype=np.int32)
    candidates = 0
    for width in range(2, n + 1):
        for i in range(0, n - width + 1):
            j = i + width
            best_score = -math.inf
            best_split = -1
            for k in range(i + 1, j):
                candidate = chart[i, k] + chart[k, j]
                candidates += 1
                if candidate > best_score:
                    best_score = candidate
                    best_split = k
            chart[i, j] = span_scores[i, j] + best_score
            splits[i, j] = best_split
    spans = _backtrace(splits, n)
    return DecodeResult(float(chart[0, n]), spans, chart, splits, candidates)


def full_cky_min(span_costs: np.ndarray) -> DecodeResult:
    """Decode a minimum-cost binary tree in cubic time.

    The local cost of a half-open span ``[i, j)`` is counted only when the
    span has width at least two. The smallest split wins exact ties.
    """
    maximum = full_cky(-span_costs)
    return DecodeResult(
        -maximum.score,
        maximum.spans,
        -maximum.chart,
        maximum.splits,
        maximum.candidates,
        maximum.fallbacks,
    )


def knuth_cky_min(span_costs: np.ndarray) -> DecodeResult:
    """Decode with self-referential Knuth intervals in quadratic time.

    Optimality requires the quadrangle inequality and interval-inclusion
    monotonicity for ``span_costs``. The function deliberately does not infer
    those preconditions; callers can use :mod:`faster_cky.structure` to audit
    or repair an interval-cost matrix before decoding. On non-QI costs this
    is the direct, unmodified Knuth heuristic: no repair, anchor, widening,
    oracle splits, or full-range fallback is used. Finite costs still give
    nonempty windows and O(n^2) candidates, but may give a suboptimal tree.
    """
    n = span_costs.shape[0] - 1
    chart = np.zeros((n + 1, n + 1), dtype=np.float64)
    splits = np.full((n + 1, n + 1), -1, dtype=np.int32)
    candidates = 0
    for width in range(2, n + 1):
        for i in range(0, n - width + 1):
            j = i + width
            if width == 2:
                lower = upper = i + 1
            else:
                lower = max(i + 1, int(splits[i, j - 1]))
                upper = min(j - 1, int(splits[i + 1, j]))
                if lower > upper:
                    raise ValueError(
                        "self-referential Knuth window invariant failed "
                        f"at span ({i}, {j}); check for nonfinite costs"
                    )

            best_cost = math.inf
            best_split = -1
            for k in range(lower, upper + 1):
                candidate = chart[i, k] + chart[k, j]
                candidates += 1
                if candidate < best_cost:
                    best_cost = candidate
                    best_split = k
            chart[i, j] = span_costs[i, j] + best_cost
            splits[i, j] = best_split

    spans = _backtrace(splits, n)
    return DecodeResult(float(chart[0, n]), spans, chart, splits, candidates)


def anchored_window_cky_min(
    span_costs: np.ndarray, anchor_splits: np.ndarray, radius: int
) -> DecodeResult:
    """Minimize original costs inside windows from a structured anchor chart.

    The anchor splits typically come from an exact-QI surrogate. For span
    ``[i,j)``, the method searches from ``K0(i,j-1)-radius`` through
    ``K0(i+1,j)+radius``. No exactness claim is implicit unless the resulting
    error certificate is zero.
    """
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    n = span_costs.shape[0] - 1
    if anchor_splits.shape != span_costs.shape:
        raise ValueError("anchor split and span-cost shapes must match")
    chart = np.zeros((n + 1, n + 1), dtype=np.float64)
    splits = np.full((n + 1, n + 1), -1, dtype=np.int32)
    candidates = 0
    for width in range(2, n + 1):
        for i in range(0, n - width + 1):
            j = i + width
            if width == 2:
                lower = upper = i + 1
            else:
                lower = max(i + 1, int(anchor_splits[i, j - 1]) - radius)
                upper = min(j - 1, int(anchor_splits[i + 1, j]) + radius)
                if lower > upper:
                    raise ValueError(
                        f"anchor Knuth bounds crossed at span ({i}, {j})"
                    )
            best_cost = math.inf
            best_split = -1
            for k in range(lower, upper + 1):
                candidate = chart[i, k] + chart[k, j]
                candidates += 1
                if candidate < best_cost:
                    best_cost = candidate
                    best_split = k
            chart[i, j] = span_costs[i, j] + best_cost
            splits[i, j] = best_split
    spans = _backtrace(splits, n)
    return DecodeResult(float(chart[0, n]), spans, chart, splits, candidates)


def tree_cost(span_costs: np.ndarray, spans: Iterable[tuple[int, int]]) -> float:
    """Return the additive local cost of a binary tree's internal spans."""
    return float(sum(span_costs[i, j] for i, j in spans))


def greedy_top_down_min(span_costs: np.ndarray) -> DecodeResult:
    """Choose splits by immediate child costs, then recurse; smallest ties win.

    This simple approximate baseline uses the same additive tree objective.
    Singletons cost zero, and at most O(n^2) split candidates are visited.
    """
    n = span_costs.shape[0] - 1
    splits = np.full((n + 1, n + 1), -1, dtype=np.int32)
    chart = np.zeros((n + 1, n + 1), dtype=np.float64)
    stack = [(0, n)]
    spans = []
    candidates = 0
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        best, split = math.inf, -1
        for k in range(i + 1, j):
            value = (span_costs[i, k] if k - i > 1 else 0.0) + (
                span_costs[k, j] if j - k > 1 else 0.0
            )
            candidates += 1
            if value < best:
                best, split = value, k
        if split < 0:
            raise ValueError("greedy decoding requires finite local costs")
        splits[i, j] = split
        spans.append((i, j))
        stack.extend([(split, j), (i, split)])
    for i, j in reversed(spans):
        k = int(splits[i, j])
        chart[i, j] = span_costs[i, j] + chart[i, k] + chart[k, j]
    return DecodeResult(float(chart[0, n]), frozenset(spans), chart, splits, candidates)


def greedy_adjacent_merge_min(span_costs: np.ndarray) -> DecodeResult:
    """Build a tree by repeatedly merging the cheapest adjacent clusters.

    At each step, the local span cost of the newly created parent is the
    merge priority.  This is the agglomerative baseline used to isolate the
    effect of global CKY optimization while keeping exactly the same local
    objective.  The leftmost merge wins exact ties.
    """
    n = span_costs.shape[0] - 1
    chart = np.zeros((n + 1, n + 1), dtype=np.float64)
    splits = np.full((n + 1, n + 1), -1, dtype=np.int32)
    clusters = [(i, i + 1) for i in range(n)]
    spans: set[tuple[int, int]] = set()
    candidates = 0
    while len(clusters) > 1:
        best_cost = math.inf
        best_index = -1
        for index in range(len(clusters) - 1):
            left, middle = clusters[index]
            next_left, right = clusters[index + 1]
            if middle != next_left:
                raise ValueError("adjacent-merge clusters must form a partition")
            candidates += 1
            cost = float(span_costs[left, right])
            if cost < best_cost:
                best_cost = cost
                best_index = index
        if best_index < 0:
            raise ValueError("greedy decoding requires finite local costs")
        left, middle = clusters[best_index]
        _, right = clusters[best_index + 1]
        splits[left, right] = middle
        spans.add((left, right))
        clusters[best_index : best_index + 2] = [(left, right)]

    def accumulate(i: int, j: int) -> float:
        if j - i < 2:
            return 0.0
        k = int(splits[i, j])
        if not i < k < j:
            raise ValueError(f"invalid greedy split {k} for span ({i}, {j})")
        value = float(span_costs[i, j]) + accumulate(i, k) + accumulate(k, j)
        chart[i, j] = value
        return value

    score = accumulate(0, n) if n >= 2 else 0.0
    return DecodeResult(score, frozenset(spans), chart, splits, candidates)


def window_cky(span_scores: np.ndarray, radius: int) -> DecodeResult:
    """Decode inside an expanded empirical Knuth window.

    The search interval is [K(i,j-1)-r, K(i+1,j)+r]. If the interval is
    empty, the implementation evaluates the full split range. A nonempty
    interval can still miss the full-CKY optimum; agreement is measured by
    the analysis rather than assumed.
    """
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    n = span_scores.shape[0] - 1
    chart = np.zeros((n + 1, n + 1), dtype=np.float64)
    splits = np.full((n + 1, n + 1), -1, dtype=np.int32)
    candidates = 0
    fallbacks = 0
    for width in range(2, n + 1):
        for i in range(0, n - width + 1):
            j = i + width
            if width == 2:
                lower = upper = i + 1
            else:
                lower = max(i + 1, int(splits[i, j - 1]) - radius)
                upper = min(j - 1, int(splits[i + 1, j]) + radius)
                if lower > upper:
                    lower, upper = i + 1, j - 1
                    fallbacks += 1
            best_score = -math.inf
            best_split = -1
            for k in range(lower, upper + 1):
                candidate = chart[i, k] + chart[k, j]
                candidates += 1
                if candidate > best_score:
                    best_score = candidate
                    best_split = k
            chart[i, j] = span_scores[i, j] + best_score
            splits[i, j] = best_split
    spans = _backtrace(splits, n)
    return DecodeResult(
        float(chart[0, n]), spans, chart, splits, candidates, fallbacks
    )


def split_diagnostics(result: DecodeResult) -> tuple[list[int], list[float]]:
    """Return Knuth-bound violations and distance-normalized split margins."""
    n = result.chart.shape[0] - 1
    violations: list[int] = []
    margins: list[float] = []
    for width in range(3, n + 1):
        for i in range(0, n - width + 1):
            j = i + width
            left = int(result.splits[i, j - 1])
            current = int(result.splits[i, j])
            right = int(result.splits[i + 1, j])
            violations.append(max(left - current, current - right, 0))

            best_value = result.chart[i, current] + result.chart[current, j]
            slopes = []
            for k in range(i + 1, j):
                if k == current:
                    continue
                value = result.chart[i, k] + result.chart[k, j]
                slopes.append((best_value - value) / abs(k - current))
            if slopes:
                margins.append(float(min(slopes)))
    return violations, margins


def oracle_window_coverage(
    splits: np.ndarray, radii: Iterable[int]
) -> dict[int, tuple[int, int]]:
    """Count full-CKY split containment in windows formed by full neighbors."""
    n = splits.shape[0] - 1
    counts = {radius: [0, 0] for radius in radii}
    for width in range(3, n + 1):
        for i in range(0, n - width + 1):
            j = i + width
            left = int(splits[i, j - 1])
            current = int(splits[i, j])
            right = int(splits[i + 1, j])
            for radius, count in counts.items():
                count[1] += 1
                if left - radius <= current <= right + radius:
                    count[0] += 1
    return {radius: (count[0], count[1]) for radius, count in counts.items()}
