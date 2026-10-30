"""Exact-QI pair costs, defect audits, and a provable convex repair."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class KnuthDefects:
    """Maximum adjacent defects for the two Knuth--Yao conditions."""

    quadrangle: float
    inclusion: float
    quadrangle_count: int
    inclusion_count: int


@dataclass(frozen=True)
class ConvexRepair:
    """A length-only correction that makes interval costs Knuth-compatible."""

    costs: np.ndarray
    correction_by_width: np.ndarray
    quadratic_coefficient: float
    linear_coefficient: float
    original_defects: KnuthDefects
    repaired_defects: KnuthDefects
    penalty_minimum: float
    penalty_maximum: float
    regret_bound: float


@dataclass(frozen=True)
class MobiusRepair:
    """Positive-part projection of pairwise M\"obius coefficients."""

    costs: np.ndarray
    coefficients: np.ndarray
    removed_negative_coefficients: np.ndarray
    residual_costs: np.ndarray
    original_defects: KnuthDefects
    repaired_defects: KnuthDefects
    pair_count_bound: float
    span_order_bound: float
    regret_bound: float


@dataclass(frozen=True)
class RegretCertificate:
    """Output-dependent regret certificate relative to original costs."""

    repaired_lower_bound: float
    residual_upper_bound: float
    candidate_original_cost: float
    regret_bound: float


def _square_size(matrix: np.ndarray) -> int:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("expected a square matrix")
    if matrix.shape[0] < 1:
        raise ValueError("expected a nonempty matrix")
    return matrix.shape[0]


def pair_span_costs(
    pair_distances: np.ndarray, *, normalize: bool
) -> np.ndarray:
    """Construct pair-sum or average-pair costs for every token span.

    ``pair_distances`` is an ``n`` by ``n`` nonnegative distance matrix.
    Singleton span costs are zero. The implementation forms every span cost in
    quadratic time after a quadratic prefix-sum construction.
    """
    n = _square_size(pair_distances)
    if np.any(~np.isfinite(pair_distances)):
        raise ValueError("pair distances must be finite")
    if np.any(pair_distances < -1e-12):
        raise ValueError("pair distances must be nonnegative")

    distances = np.maximum(pair_distances.astype(np.float64, copy=False), 0.0)
    column_prefix = np.zeros((n + 1, n), dtype=np.float64)
    column_prefix[1:] = np.cumsum(distances, axis=0)

    costs = np.full((n + 1, n + 1), np.nan, dtype=np.float64)
    np.fill_diagonal(costs, 0.0)
    for i in range(n):
        costs[i, i + 1] = 0.0
        pair_sum = 0.0
        for j in range(i + 2, n + 1):
            new_token = j - 1
            pair_sum += column_prefix[new_token, new_token] - column_prefix[i, new_token]
            width = j - i
            denominator = math.comb(width, 2) if normalize else 1
            costs[i, j] = pair_sum / denominator
    return costs


def mobius_coefficients(costs: np.ndarray) -> np.ndarray:
    r"""Invert an interval-cost matrix into signed pair interactions.

    The inversion assumes zero diagonal and singleton costs. For ``p < q``,

    ``z[p,q] = w[p,q+1] + w[p+1,q] - w[p,q] - w[p+1,q+1]``.

    The reconstruction identity is
    ``w[i,j] = sum(z[p,q] for i <= p < q < j)``. Nonnegative coefficients are
    equivalent to all adjacent quadrangle inequalities.
    """
    n = _square_size(costs) - 1
    if n < 1:
        raise ValueError("expected costs for at least one token")
    if not np.allclose(np.diag(costs), 0.0, atol=1e-10):
        raise ValueError("diagonal costs must be zero")
    if any(abs(float(costs[i, i + 1])) > 1e-10 for i in range(n)):
        raise ValueError("singleton costs must be zero")

    coefficients = np.zeros((n, n), dtype=np.float64)
    for p in range(n):
        for q in range(p + 1, n):
            coefficients[p, q] = (
                costs[p, q + 1]
                + costs[p + 1, q]
                - costs[p, q]
                - costs[p + 1, q + 1]
            )
    return coefficients


def costs_from_pair_coefficients(coefficients: np.ndarray) -> np.ndarray:
    """Reconstruct all interval costs from signed upper-triangular coefficients."""
    n = _square_size(coefficients)
    column_prefix = np.zeros((n + 1, n), dtype=np.float64)
    column_prefix[1:] = np.cumsum(coefficients, axis=0)
    costs = np.full((n + 1, n + 1), np.nan, dtype=np.float64)
    np.fill_diagonal(costs, 0.0)
    for i in range(n):
        costs[i, i + 1] = 0.0
        total = 0.0
        for j in range(i + 2, n + 1):
            q = j - 1
            total += column_prefix[q, q] - column_prefix[i, q]
            costs[i, j] = total
    return costs


def positive_mobius_repair(costs: np.ndarray) -> MobiusRepair:
    r"""Clip negative pair interactions and return a certified exact-QI cost.

    If ``e[p,q] = max(-z[p,q], 0)``, the repaired interval cost equals the
    original cost plus the sum of ``e[p,q]`` inside the interval. Every tree
    contains the root interval, whose residual cancels when two trees are
    compared. A pair's remaining co-occurrence-count range is at most
    ``n - (q-p) - 1``. An independent bound sums the ``n-2`` largest residual
    costs among non-root intervals. The smaller bound certifies the
    original-objective regret.
    """
    n = _square_size(costs) - 1
    coefficients = mobius_coefficients(costs)
    removed = np.maximum(-coefficients, 0.0)
    positive = np.maximum(coefficients, 0.0)
    repaired = costs_from_pair_coefficients(positive)
    residual = costs_from_pair_coefficients(removed)
    pair_count_bound = 0.0
    for p in range(n):
        for q in range(p + 1, n):
            pair_count_bound += (n - (q - p) - 1) * removed[p, q]

    nonroot_residuals = [
        float(residual[i, j])
        for width in range(2, n + 1)
        for i in range(n - width + 1)
        for j in [i + width]
        if not (i == 0 and j == n)
    ]
    count = max(0, n - 2)
    values = np.asarray(nonroot_residuals)
    span_order_bound = (
        float(-np.partition(-values, count - 1)[:count].sum()) if count else 0.0
    )
    regret_bound = min(pair_count_bound, span_order_bound)
    return MobiusRepair(
        repaired,
        coefficients,
        removed,
        residual,
        knuth_defects(costs),
        knuth_defects(repaired, tolerance=1e-10),
        float(pair_count_bound),
        float(span_order_bound),
        float(regret_bound),
    )


def mobius_regret_certificate(
    original_costs: np.ndarray,
    repair: MobiusRepair,
    repaired_optimum_score: float,
    candidate_spans: frozenset[tuple[int, int]],
) -> RegretCertificate:
    r"""Certify a candidate under an arbitrary signed-interaction objective.

    Write ``w = w_plus - R``, where ``w_plus`` is the repaired exact-QI cost
    and ``R`` is a nonnegative pair-sum residual. If ``U_R`` upper-bounds the
    residual cost of every tree, then

    ``OPT_w >= OPT_w_plus - U_R``.

    Two quadratic-time relaxations provide ``U_R``. The pair relaxation uses
    the largest possible ancestor count for each pair. The span-order
    relaxation selects the largest ``n-2`` non-root residual intervals and
    adds the mandatory root residual.
    """
    n = _square_size(original_costs) - 1
    root_residual = float(repair.residual_costs[0, n]) if n >= 2 else 0.0
    pair_upper = root_residual + repair.pair_count_bound
    span_upper = root_residual + repair.span_order_bound
    residual_upper = min(pair_upper, span_upper)
    candidate_cost = float(
        sum(original_costs[i, j] for i, j in candidate_spans)
    )
    lower_bound = float(repaired_optimum_score - residual_upper)
    regret_bound = max(0.0, candidate_cost - lower_bound)
    return RegretCertificate(
        lower_bound,
        float(residual_upper),
        candidate_cost,
        float(regret_bound),
    )


def interval_order_lower_bound(costs: np.ndarray) -> float:
    """Quadratic lower bound: root plus the n-2 cheapest non-root spans.

    Every full binary tree has exactly n-1 internal spans, including the root.
    Dropping compatibility of the remaining spans gives a lower bound even
    for signed costs. Singleton costs are ignored, as in the decoders.
    """
    n = _square_size(costs) - 1
    if n < 2:
        return 0.0
    count = n - 2
    values = np.asarray([
        costs[i, i + width]
        for width in range(2, n)
        for i in range(n - width + 1)
    ])
    extra = float(np.partition(values, count - 1)[:count].sum()) if count else 0.0
    return float(costs[0, n]) + extra


def knuth_defects(costs: np.ndarray, tolerance: float = 0.0) -> KnuthDefects:
    """Return maximal adjacent QI and one-step inclusion defects.

    Adjacent inequalities characterize the full quadrangle inequality by
    telescoping. One-step containment inequalities characterize monotonicity
    under arbitrary interval inclusion.
    """
    n = _square_size(costs) - 1
    if n < 1:
        raise ValueError("expected costs for at least one token")
    quadrangle = 0.0
    quadrangle_count = 0
    for width in range(1, n):
        for i in range(0, n - width):
            j = i + width
            defect = (
                costs[i, j]
                + costs[i + 1, j + 1]
                - costs[i, j + 1]
                - costs[i + 1, j]
            )
            quadrangle = max(quadrangle, float(defect) - tolerance)
            quadrangle_count += 1

    inclusion = 0.0
    inclusion_count = 0
    for width in range(2, n + 1):
        for i in range(0, n - width + 1):
            j = i + width
            inclusion = max(
                inclusion,
                float(costs[i + 1, j] - costs[i, j]) - tolerance,
                float(costs[i, j - 1] - costs[i, j]) - tolerance,
            )
            inclusion_count += 2

    return KnuthDefects(
        max(0.0, quadrangle),
        max(0.0, inclusion),
        quadrangle_count,
        inclusion_count,
    )


def _penalty_extrema(correction: np.ndarray) -> tuple[float, float]:
    """Return the minimum and maximum tree sums of a width-only correction."""
    n = len(correction) - 1
    minimum = np.zeros(n + 1, dtype=np.float64)
    maximum = np.zeros(n + 1, dtype=np.float64)
    for width in range(2, n + 1):
        minimum[width] = correction[width] + min(
            minimum[k] + minimum[width - k] for k in range(1, width)
        )
        maximum[width] = correction[width] + max(
            maximum[k] + maximum[width - k] for k in range(1, width)
        )
    return float(minimum[n]), float(maximum[n])


def convex_knuth_repair(costs: np.ndarray) -> ConvexRepair:
    r"""Add the smallest two-parameter convex correction used by our bound.

    Let ``epsilon`` be the largest adjacent quadrangle defect. Adding
    ``epsilon / 2 * (length**2 - 1)`` removes every such defect. A nonnegative
    linear term then removes any remaining one-step inclusion defect. The
    correction is zero for singleton spans and preserves exact tree rankings
    up to a length-dependent additive perturbation.
    """
    n = _square_size(costs) - 1
    if n < 1:
        raise ValueError("expected costs for at least one token")
    original = knuth_defects(costs)
    quadratic = original.quadrangle / 2.0

    linear = 0.0
    for width in range(2, n + 1):
        quadratic_increment = quadratic * (2 * width - 1)
        for i in range(0, n - width + 1):
            j = i + width
            linear = max(
                linear,
                float(costs[i + 1, j] - costs[i, j]) - quadratic_increment,
                float(costs[i, j - 1] - costs[i, j]) - quadratic_increment,
            )
    linear = max(0.0, linear)

    correction = np.zeros(n + 1, dtype=np.float64)
    for width in range(2, n + 1):
        correction[width] = (
            quadratic * (width * width - 1) + linear * (width - 1)
        )

    repaired = costs.copy().astype(np.float64)
    for width in range(1, n + 1):
        for i in range(0, n - width + 1):
            repaired[i, i + width] += correction[width]
    np.fill_diagonal(repaired, 0.0)

    repaired_defects = knuth_defects(repaired, tolerance=1e-10)
    penalty_minimum, penalty_maximum = _penalty_extrema(correction)
    return ConvexRepair(
        repaired,
        correction,
        quadratic,
        linear,
        original,
        repaired_defects,
        penalty_minimum,
        penalty_maximum,
        penalty_maximum - penalty_minimum,
    )
