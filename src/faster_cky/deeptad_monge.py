"""Monge hierarchy objectives over fixed atomic TADs.

The neural boundary detector is deliberately outside this module.  Given its
ordered atomic TADs, boundary probabilities, and/or the adjacent cosine
similarities used by deepTAD, this module constructs a Knuth--Yao-compatible
ordered binary hierarchy objective.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .structure import pair_span_costs


def deeptad_boundary_patches(contacts: np.ndarray) -> np.ndarray:
    """Extract the public deepTAD 10-by-10 diagonal patch for every bin."""
    matrix = np.asarray(contacts, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("contacts must be a square matrix")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("contacts must be finite")
    # Read only the hundred diagonally adjacent pixels for each bin. Padding
    # the entire chromosome matrix would duplicate gigabytes at 10 kb.
    size = len(matrix)
    patches = np.zeros((size, 10, 10), dtype=np.float32)
    bins = np.arange(size)
    for patch_row in range(10):
        rows = bins + patch_row - 4
        valid_rows = (rows >= 0) & (rows < size)
        for patch_column in range(10):
            columns = bins + patch_column - 4
            valid = valid_rows & (columns >= 0) & (columns < size)
            patches[valid, patch_row, patch_column] = matrix[rows[valid], columns[valid]]
    return patches


def normalize_deeptad_patches(patches: np.ndarray) -> np.ndarray:
    """Apply deepTAD's row-wise standardization without changing patch order."""
    values = np.asarray(patches, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (10, 10):
        raise ValueError("patches must have shape [bins, 10, 10]")
    rows = values.reshape(-1, 10)
    means = rows.mean(axis=1, keepdims=True)
    standard_deviations = np.sqrt(rows.var(axis=1, keepdims=True)) + 1e-10
    normalized = (rows - means) / standard_deviations
    return normalized.reshape(values.shape).astype(np.float32, copy=False)


def validate_atomic_tads(
    atomic_tads: Sequence[tuple[int, int]], matrix_size: int | None = None
) -> np.ndarray:
    """Return ordered, nonoverlapping half-open atomic TAD coordinates.

    Public deepTAD excludes each one-bin boundary from both neighboring leaf
    TADs.  Consequently adjacent leaves normally have a one-bin gap.
    """
    coordinates = np.asarray(atomic_tads, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
        raise ValueError("atomic_tads must be a nonempty sequence of (left, right) pairs")
    if len(coordinates) < 2:
        raise ValueError("at least two atomic TADs are required")
    if np.any(coordinates[:, 0] < 0) or np.any(
        coordinates[:, 1] <= coordinates[:, 0]
    ):
        raise ValueError("atomic TADs must be nonempty and nonnegative")
    if np.any(coordinates[1:, 0] < coordinates[:-1, 1]):
        raise ValueError("atomic TADs must be ordered and nonoverlapping")
    if matrix_size is not None and int(coordinates[-1, 1]) > matrix_size:
        raise ValueError("atomic TAD coordinates exceed the contact matrix")
    return coordinates


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0 or second_norm == 0:
        return 1.0 if first_norm == second_norm else 0.0
    return float(np.dot(first, second) / (first_norm * second_norm))


def adjacent_deeptad_cosines(
    contacts: np.ndarray, atomic_tads: Sequence[tuple[int, int]]
) -> np.ndarray:
    """Reproduce deepTAD's fixed adjacent-leaf cosine features.

    Each atomic TAD is represented by its mean contact to every other atomic
    TAD, divided by its within-TAD mean.  For a neighboring pair, the two
    entries referring to the pair itself are omitted, matching the public
    ``mergeTAD.ipynb`` implementation.  Diagonal pixels are excluded from the
    within-TAD means by zeroing the contact-matrix diagonal first.
    """
    matrix = np.asarray(contacts, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("contacts must be a square matrix")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("contacts must be finite")
    coordinates = validate_atomic_tads(atomic_tads, len(matrix))
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0.0)

    count = len(coordinates)
    normalized = np.zeros((count, count), dtype=np.float64)
    for index, (left, right) in enumerate(coordinates):
        within = float(np.mean(matrix[left:right, left:right]))
        if within <= 0:
            continue
        for feature, (other_left, other_right) in enumerate(coordinates):
            normalized[index, feature] = (
                float(np.mean(matrix[left:right, other_left:other_right])) / within
            )

    similarities = np.empty(count - 1, dtype=np.float64)
    for boundary in range(count - 1):
        keep = np.ones(count, dtype=bool)
        keep[boundary : boundary + 2] = False
        similarities[boundary] = _cosine(
            normalized[boundary, keep], normalized[boundary + 1, keep]
        )
    return np.clip(similarities, -1.0, 1.0)


def atomic_profile_dissimilarities(
    contacts: np.ndarray, atomic_tads: Sequence[tuple[int, int]]
) -> np.ndarray:
    """Return all-pairs dissimilarities between deepTAD contact profiles.

    The representation matches :func:`adjacent_deeptad_cosines`: every atomic
    TAD is represented by mean contact to all atomic TADs divided by its own
    within-TAD mean. For a pair ``p,q``, their two direct entries are omitted
    before cosine similarity is computed. The resulting ``1-cosine`` matrix is
    nonnegative and can therefore be added as a pair potential without losing
    exact Monge structure.
    """
    matrix = np.asarray(contacts, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("contacts must be a square matrix")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("contacts must be finite")
    coordinates = validate_atomic_tads(atomic_tads, len(matrix))
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0.0)
    count = len(coordinates)
    profiles = np.zeros((count, count), dtype=np.float64)
    for index, (left, right) in enumerate(coordinates):
        within = float(np.mean(matrix[left:right, left:right]))
        if within <= 0:
            continue
        for feature, (other_left, other_right) in enumerate(coordinates):
            profiles[index, feature] = (
                float(np.mean(matrix[left:right, other_left:other_right])) / within
            )
    output = np.zeros((count, count), dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            keep = np.ones(count, dtype=bool)
            keep[[first, second]] = False
            value = 1.0 - _cosine(profiles[first, keep], profiles[second, keep])
            output[first, second] = output[second, first] = value
    return np.clip(output, 0.0, 2.0)


def atomic_tad_regions(
    number_of_bins: int,
    boundary_indices: Sequence[int],
    gap_indices: Sequence[int],
    *,
    minimum_region_bins: int = 4,
) -> list[list[tuple[int, int]]]:
    """Reproduce deepTAD leaf TADs without crossing gap bins.

    Boundary bins and gap bins are excluded from the leaves, matching
    ``Convert.Bin.To.Domain.TMP`` in the pinned public R implementation.
    Coordinates returned here are zero-based and half-open.
    """
    if number_of_bins < 1 or minimum_region_bins < 1:
        raise ValueError("bin and minimum-region counts must be positive")
    boundaries = np.asarray(boundary_indices, dtype=np.int64)
    gaps = np.asarray(gap_indices, dtype=np.int64)
    if boundaries.ndim != 1 or gaps.ndim != 1:
        raise ValueError("boundary and gap indices must be vectors")
    if np.any((boundaries < 0) | (boundaries >= number_of_bins)) or np.any(
        (gaps < 0) | (gaps >= number_of_bins)
    ):
        raise ValueError("boundary or gap index lies outside the chromosome")

    valid = np.ones(number_of_bins, dtype=bool)
    valid[gaps] = False
    regions: list[list[tuple[int, int]]] = []
    start = 0
    while start < number_of_bins:
        while start < number_of_bins and not valid[start]:
            start += 1
        if start == number_of_bins:
            break
        end = start
        while end < number_of_bins and valid[end]:
            end += 1
        if end - start >= minimum_region_bins:
            excluded = {
                int(index)
                for index in boundaries
                if start <= index < end and valid[index]
            }
            leaves: list[tuple[int, int]] = []
            domain_start = start
            for index in sorted(excluded):
                if domain_start < index:
                    leaves.append((domain_start, index))
                domain_start = index + 1
            if domain_start < end:
                leaves.append((domain_start, end))
            if leaves:
                regions.append(leaves)
        start = end
    return regions


def public_deeptad_merge(
    contacts: np.ndarray,
    atomic_tads: Sequence[tuple[int, int]],
    *,
    cosine_threshold: float = 0.8,
) -> frozenset[tuple[int, int]]:
    """Reproduce the fixed-link run merge in public ``mergeTAD.ipynb``."""
    similarities = adjacent_deeptad_cosines(contacts, atomic_tads)
    return public_deeptad_merge_from_similarities(
        contacts,
        atomic_tads,
        similarities,
        cosine_threshold=cosine_threshold,
    )


def public_deeptad_merge_from_similarities(
    contacts: np.ndarray,
    atomic_tads: Sequence[tuple[int, int]],
    similarities: Sequence[float],
    *,
    cosine_threshold: float = 0.8,
    average_contact: float | None = None,
) -> frozenset[tuple[int, int]]:
    """Run public deepTAD's linear merge from cached cosine features."""
    matrix = np.asarray(contacts, dtype=np.float64)
    coordinates = validate_atomic_tads(atomic_tads, len(matrix))
    if not -1 <= cosine_threshold <= 1:
        raise ValueError("cosine threshold must lie in [-1, 1]")
    cosine_values = np.asarray(similarities, dtype=np.float64)
    if cosine_values.shape != (len(coordinates) - 1,) or np.any(
        ~np.isfinite(cosine_values)
    ):
        raise ValueError("similarities must contain one finite value per separator")
    if average_contact is None:
        matrix = matrix.copy()
        np.fill_diagonal(matrix, 0.0)
        nonzero = matrix[matrix != 0]
        average_contact = float(np.mean(nonzero)) if len(nonzero) else 0.0
    elif not np.isfinite(average_contact):
        raise ValueError("average_contact must be finite")
    merge_links = np.zeros(len(coordinates) - 1, dtype=bool)
    for index, similarity in enumerate(cosine_values):
        left_start, left_end = coordinates[index]
        right_start, right_end = coordinates[index + 1]
        cross_mean = float(
            np.mean(matrix[right_start:right_end, left_start:left_end])
        )
        merge_links[index] = (
            similarity >= cosine_threshold and cross_mean >= float(average_contact)
        )

    output = {tuple(map(int, interval)) for interval in coordinates}
    index = 0
    while index < len(coordinates):
        end = index
        while end < len(merge_links) and merge_links[end]:
            end += 1
        if end > index:
            output.add((int(coordinates[index, 0]), int(coordinates[end, 1])))
        index = end + 1
    return frozenset(output)


def separator_weights(
    *,
    boundary_probabilities: Sequence[float] | None = None,
    cosine_similarities: Sequence[float] | None = None,
    probability_weight: float = 1.0,
    cosine_weight: float = 1.0,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Combine boundary probabilities and cosine dissimilarities into ``q``."""
    if probability_weight < 0 or cosine_weight < 0:
        raise ValueError("separator feature weights must be nonnegative")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must lie strictly between zero and one")
    if boundary_probabilities is None and cosine_similarities is None:
        raise ValueError("at least one separator feature is required")

    probabilities = (
        None
        if boundary_probabilities is None
        else np.asarray(boundary_probabilities, dtype=np.float64)
    )
    similarities = (
        None
        if cosine_similarities is None
        else np.asarray(cosine_similarities, dtype=np.float64)
    )
    arrays = [values for values in (probabilities, similarities) if values is not None]
    if any(values.ndim != 1 or len(values) == 0 for values in arrays):
        raise ValueError("separator features must be nonempty vectors")
    if len({len(values) for values in arrays}) != 1:
        raise ValueError("separator feature vectors must have equal lengths")
    if any(np.any(~np.isfinite(values)) for values in arrays):
        raise ValueError("separator features must be finite")

    result = np.zeros(len(arrays[0]), dtype=np.float64)
    if probabilities is not None:
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("boundary probabilities must lie in [0, 1]")
        result += probability_weight * -np.log(
            np.clip(1.0 - probabilities, epsilon, 1.0)
        )
    if similarities is not None:
        if np.any((similarities < -1) | (similarities > 1)):
            raise ValueError("cosine similarities must lie in [-1, 1]")
        result += cosine_weight * (1.0 - similarities)
    return result


def quadratic_duration_costs(
    maximum_length: int, *, quadratic: float, linear: float = 0.0
) -> np.ndarray:
    """Return a nonnegative, nondecreasing discrete-convex duration table."""
    if maximum_length < 1 or quadratic < 0 or linear < 0:
        raise ValueError("length and duration coefficients must be nonnegative")
    lengths = np.arange(maximum_length + 1, dtype=np.float64)
    return quadratic * lengths**2 + linear * lengths


def distance_expected_profile(contacts: np.ndarray) -> np.ndarray:
    """Return the chromosome-wide mean contact at each genomic separation."""
    matrix = np.asarray(contacts)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("contacts must be a square matrix")
    for start in range(0, len(matrix), 256):
        block = matrix[start:start + 256]
        if np.any(~np.isfinite(block)) or np.any(block < 0):
            raise ValueError("contacts must be finite and nonnegative")
    if matrix.dtype == np.float32:
        # Sequential reads avoid revisiting every mmap page for every lag on
        # chromosome matrices larger than the available filesystem cache.
        sums = np.zeros(len(matrix), dtype=np.float64)
        for row in range(len(matrix)):
            sums[:len(matrix)-row] += matrix[row, row:].astype(np.float64)
        return sums / np.arange(len(matrix), 0, -1, dtype=np.float64)
    return np.asarray(
        [float(np.mean(np.asarray(np.diagonal(matrix, offset=lag), dtype=np.float64))) for lag in range(len(matrix))],
        dtype=np.float64,
    )


def atomic_pair_mean_oe(
    contacts: np.ndarray,
    atomic_tads: Sequence[tuple[int, int]],
    *,
    expected_profile: Sequence[float] | None = None,
    validate_contacts: bool = True,
) -> np.ndarray:
    r"""Return mean distance-normalized contact for every atomic-TAD pair.

    Each entry aggregates all bin pairs between two atomic TADs. Expected
    contacts are estimated chromosome-wide by genomic separation unless a
    fixed profile is supplied. The diagonal is set to one as a neutral O/E
    value; downstream pair-potential constructors reset self costs to zero.
    """
    matrix = np.asarray(contacts)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("contacts must be a square matrix")
    if validate_contacts:
        for start in range(0, len(matrix), 256):
            block = matrix[start:start + 256]
            if np.any(~np.isfinite(block)) or np.any(block < 0):
                raise ValueError("contacts must be finite and nonnegative")
    coordinates = validate_atomic_tads(atomic_tads, len(matrix))
    expected = (
        distance_expected_profile(matrix)
        if expected_profile is None
        else np.asarray(expected_profile, dtype=np.float64)
    )
    if expected.ndim != 1 or len(expected) < len(matrix):
        raise ValueError("expected_profile must cover every genomic separation")
    if np.any(~np.isfinite(expected)) or np.any(expected < 0):
        raise ValueError("expected_profile must be finite and nonnegative")

    count = len(coordinates)
    mean_oe = np.ones((count, count), dtype=np.float64)
    for first in range(count):
        first_bins = np.arange(coordinates[first, 0], coordinates[first, 1])
        for second in range(first + 1, count):
            second_bins = np.arange(coordinates[second, 0], coordinates[second, 1])
            separations = second_bins[None, :] - first_bins[:, None]
            denominators = expected[separations]
            valid = denominators > 0
            if np.any(valid):
                observed = np.asarray(matrix[np.ix_(first_bins, second_bins)], dtype=np.float64)
                value = float(np.mean(observed[valid] / denominators[valid]))
            else:
                value = 1.0
            mean_oe[first, second] = mean_oe[second, first] = value
    return mean_oe


def pair_potentials_from_mean_oe(
    mean_oe: np.ndarray,
    *,
    transform: str = "hinge",
    threshold: float = 1.0,
) -> np.ndarray:
    """Map atomic O/E values to nonnegative incompatibility costs.

    ``hinge`` is the proposed ``[threshold - O/E]_+`` construction.
    ``reciprocal`` and ``exponential`` retain ordering above O/E=1 while
    preserving the nonnegativity needed for exact Monge structure.
    """
    values = np.asarray(mean_oe, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("mean_oe must be square")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("mean_oe must be finite and nonnegative")
    if threshold <= 0 or not math.isfinite(threshold):
        raise ValueError("threshold must be finite and positive")
    if transform == "hinge":
        result = np.maximum(threshold - values, 0.0)
    elif transform == "reciprocal":
        result = threshold / (threshold + values)
    elif transform == "exponential":
        result = np.exp(-values / threshold)
    else:
        raise ValueError(f"unknown pair-potential transform: {transform}")
    result = (result + result.T) / 2
    np.fill_diagonal(result, 0.0)
    return result


def separator_path_pair_potentials(
    separators: Sequence[float], *, aggregation: str = "maximum"
) -> np.ndarray:
    """Lift separator evidence to all atomic-TAD pairs across each boundary.

    For atomic TADs ``p < q``, the potential is either the maximum or the sum
    of separator weights on the path ``p, ..., q``. This makes a strong
    separator influence every pair that crosses it while retaining a
    nonnegative pair matrix and hence exact Monge structure.
    """
    values = np.asarray(separators, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("separators must be a nonempty vector")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("separator weights must be finite and nonnegative")
    count = len(values) + 1
    output = np.zeros((count, count), dtype=np.float64)
    for first in range(count):
        running = 0.0
        for second in range(first + 1, count):
            if aggregation == "maximum":
                running = max(running, float(values[second - 1]))
            elif aggregation == "sum":
                running += float(values[second - 1])
            else:
                raise ValueError(f"unknown separator aggregation: {aggregation}")
            output[first, second] = output[second, first] = running
    return output


def atomic_pair_oe_deficits(
    contacts: np.ndarray,
    atomic_tads: Sequence[tuple[int, int]],
    *,
    expected_profile: Sequence[float] | None = None,
) -> np.ndarray:
    r"""Build the nonnegative pair potential ``[1 - mean(O/E)]_+``."""
    return pair_potentials_from_mean_oe(
        atomic_pair_mean_oe(
            contacts, atomic_tads, expected_profile=expected_profile
        ),
        transform="hinge",
        threshold=1.0,
    )


def add_pair_potentials(
    interval_costs: np.ndarray,
    pair_potentials: np.ndarray,
    *,
    weight: float = 1.0,
) -> np.ndarray:
    """Add an exact-Monge nonnegative pair sum to local interval costs."""
    costs = np.asarray(interval_costs, dtype=np.float64)
    pairs = np.asarray(pair_potentials, dtype=np.float64)
    if weight < 0 or not math.isfinite(weight):
        raise ValueError("pair weight must be finite and nonnegative")
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError("interval_costs must be square")
    if pairs.shape != (len(costs) - 1, len(costs) - 1):
        raise ValueError("pair_potentials have incompatible shape")
    if np.any(~np.isfinite(pairs)) or np.any(pairs < 0):
        raise ValueError("pair potentials must be finite and nonnegative")
    return costs + weight * pair_span_costs(pairs, normalize=False)


def validate_duration_costs(
    duration_costs: Sequence[float], maximum_length: int, tolerance: float = 1e-10
) -> np.ndarray:
    """Validate nonnegative, nondecreasing discrete convexity through the data range."""
    values = np.asarray(duration_costs, dtype=np.float64)
    if values.ndim != 1 or len(values) <= maximum_length:
        raise ValueError("duration_costs do not cover all observed genomic lengths")
    values = values[: maximum_length + 1]
    if np.any(~np.isfinite(values)):
        raise ValueError("duration costs must be finite")
    if abs(float(values[0])) > tolerance or np.any(values < -tolerance):
        raise ValueError("duration costs must start at zero and be nonnegative")
    first = np.diff(values)
    if np.any(first < -tolerance):
        raise ValueError("duration costs must be nondecreasing")
    if np.any(np.diff(first) < -tolerance):
        raise ValueError("duration costs must be discrete convex")
    return values


def monge_hierarchy_costs(
    atomic_tads: Sequence[tuple[int, int]],
    separators: Sequence[float],
    duration_costs: Sequence[float],
) -> np.ndarray:
    """Build local interval costs for an exact Knuth--Yao hierarchy DP.

    The returned matrix uses half-open indices over atomic TADs.  Singleton
    entries contain ``g(leaf_length)`` so the complete weight matrix satisfies
    the quadrangle inequality.  The project decoders deliberately ignore
    singleton local costs, which is equivalent to subtracting their sum from
    every complete tree and therefore leaves every split and optimum unchanged.
    Every parent interval pays its duration cost and every separator contained
    in that interval.
    """
    coordinates = validate_atomic_tads(atomic_tads)
    separator_array = np.asarray(separators, dtype=np.float64)
    if separator_array.shape != (len(coordinates) - 1,):
        raise ValueError("one separator weight is required between each atomic TAD")
    if np.any(~np.isfinite(separator_array)) or np.any(separator_array < 0):
        raise ValueError("separator weights must be finite and nonnegative")

    maximum_length = int(coordinates[-1, 1] - coordinates[0, 0])
    durations = validate_duration_costs(duration_costs, maximum_length)
    prefix = np.concatenate(([0.0], np.cumsum(separator_array)))
    count = len(coordinates)
    costs = np.full((count + 1, count + 1), np.nan, dtype=np.float64)
    np.fill_diagonal(costs, 0.0)
    for left in range(count):
        costs[left, left + 1] = durations[
            int(coordinates[left, 1] - coordinates[left, 0])
        ]
        for right in range(left + 2, count + 1):
            length = int(coordinates[right - 1, 1] - coordinates[left, 0])
            separator_sum = prefix[right - 1] - prefix[left]
            costs[left, right] = durations[length] + separator_sum
    return costs


def duration_costs_from_raw_curvatures(
    raw_curvatures: Sequence[float], *, raw_initial_slope: float = 0.0
) -> np.ndarray:
    """Map unconstrained parameters to a discrete-convex duration table.

    This NumPy reference parameterization mirrors a learnable implementation:
    the initial slope and every slope increment pass through ``softplus``.
    """
    raw = np.asarray(raw_curvatures, dtype=np.float64)
    if raw.ndim != 1 or np.any(~np.isfinite(raw)) or not math.isfinite(raw_initial_slope):
        raise ValueError("raw duration parameters must be finite")

    def softplus(value: np.ndarray | float) -> np.ndarray | float:
        return np.logaddexp(0.0, value)

    slopes = np.empty(len(raw) + 1, dtype=np.float64)
    slopes[0] = float(softplus(raw_initial_slope))
    if len(raw):
        slopes[1:] = slopes[0] + np.cumsum(softplus(raw))
    return np.concatenate(([0.0], np.cumsum(slopes)))
