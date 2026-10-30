import numpy as np
import pytest

from faster_cky.decoder import full_cky_min, knuth_cky_min
from faster_cky.deeptad_evaluation import DistanceNormalizedContacts
from faster_cky.deeptad_pair_ablation import (
    Objective,
    ParentSelector,
    _region_costs,
    _select_tree_parents,
)
from faster_cky.deeptad_validation_experiment import Chromosome, Region
from faster_cky.deeptad_monge import (
    add_pair_potentials,
    adjacent_deeptad_cosines,
    atomic_profile_dissimilarities,
    atomic_pair_oe_deficits,
    atomic_pair_mean_oe,
    atomic_tad_regions,
    deeptad_boundary_patches,
    duration_costs_from_raw_curvatures,
    monge_hierarchy_costs,
    normalize_deeptad_patches,
    pair_potentials_from_mean_oe,
    public_deeptad_merge,
    public_deeptad_merge_from_similarities,
    quadratic_duration_costs,
    separator_path_pair_potentials,
    separator_weights,
)
from faster_cky.structure import knuth_defects


def test_monge_hierarchy_matches_full_cky_for_irregular_atomic_tads() -> None:
    atomic_tads = [(0, 3), (3, 8), (8, 10), (10, 17), (17, 21)]
    separators = np.asarray([0.2, 3.0, 0.1, 0.7])
    durations = quadratic_duration_costs(21, quadratic=0.03, linear=0.1)
    costs = monge_hierarchy_costs(atomic_tads, separators, durations)

    defects = knuth_defects(costs, tolerance=1e-10)
    assert defects.quadrangle == 0
    assert defects.inclusion == 0
    full = full_cky_min(costs)
    knuth = knuth_cky_min(costs)
    assert full.spans == knuth.spans
    assert full.score == pytest.approx(knuth.score)
    assert knuth.candidates < full.candidates


def test_monge_hierarchy_allows_public_one_bin_boundary_gaps() -> None:
    atomic_tads = [(0, 3), (4, 8), (9, 12), (13, 18)]
    costs = monge_hierarchy_costs(
        atomic_tads,
        separators=[0.2, 2.0, 0.4],
        duration_costs=quadratic_duration_costs(18, quadratic=0.02),
    )
    assert knuth_defects(costs, tolerance=1e-10).quadrangle == 0
    assert knuth_defects(costs, tolerance=1e-10).inclusion == 0
    assert full_cky_min(costs).spans == knuth_cky_min(costs).spans


def test_deeptad_patch_extraction_preserves_bin_order_and_normalizes_rows() -> None:
    contacts = np.arange(36, dtype=np.float32).reshape(6, 6)
    patches = deeptad_boundary_patches(contacts)
    assert patches.shape == (6, 10, 10)
    assert patches[0, 4, 4] == contacts[0, 0]
    assert patches[5, 4, 4] == contacts[5, 5]
    assert np.count_nonzero(patches[0, :4]) == 0
    normalized = normalize_deeptad_patches(patches)
    row_variances = normalized.reshape(-1, 10).var(axis=1)
    nonconstant = patches.reshape(-1, 10).var(axis=1) > 0
    assert np.allclose(normalized.reshape(-1, 10).mean(axis=1), 0, atol=1e-6)
    assert np.allclose(row_variances[nonconstant], 1, atol=1e-5)


def test_random_convex_hierarchies_preserve_the_exact_tree() -> None:
    rng = np.random.default_rng(19)
    for count in range(3, 15):
        widths = rng.integers(1, 8, size=count)
        boundaries = np.concatenate(([0], np.cumsum(widths)))
        atomic_tads = list(zip(boundaries[:-1], boundaries[1:]))
        separators = rng.exponential(size=count - 1)
        durations = duration_costs_from_raw_curvatures(
            rng.normal(size=int(boundaries[-1]) - 1),
            raw_initial_slope=float(rng.normal()),
        )
        costs = monge_hierarchy_costs(atomic_tads, separators, durations)
        defects = knuth_defects(costs, tolerance=1e-9)
        full = full_cky_min(costs)
        knuth = knuth_cky_min(costs)
        assert defects.quadrangle == 0
        assert defects.inclusion == 0
        assert full.spans == knuth.spans
        assert full.score == pytest.approx(knuth.score)


def test_strong_separator_becomes_the_root_split() -> None:
    atomic_tads = [(0, 2), (2, 4), (4, 6), (6, 8)]
    costs = monge_hierarchy_costs(
        atomic_tads,
        separators=[0.0, 10.0, 0.0],
        duration_costs=quadratic_duration_costs(8, quadratic=0.01),
    )
    result = knuth_cky_min(costs)
    assert result.splits[0, 4] == 2


def test_deeptad_features_and_learnable_duration_preserve_constraints() -> None:
    contacts = np.asarray(
        [
            [5, 4, 1, 1, 0, 0],
            [4, 5, 1, 1, 0, 0],
            [1, 1, 5, 4, 2, 2],
            [1, 1, 4, 5, 2, 2],
            [0, 0, 2, 2, 5, 4],
            [0, 0, 2, 2, 4, 5],
        ],
        dtype=float,
    )
    atomic_tads = [(0, 2), (2, 4), (4, 6)]
    cosines = adjacent_deeptad_cosines(contacts, atomic_tads)
    separators = separator_weights(
        boundary_probabilities=[0.2, 0.9],
        cosine_similarities=cosines,
        probability_weight=0.5,
        cosine_weight=2.0,
    )
    durations = duration_costs_from_raw_curvatures(
        np.zeros(5), raw_initial_slope=-2
    )
    costs = monge_hierarchy_costs(atomic_tads, separators, durations)
    assert np.all(separators >= 0)
    assert np.all(np.diff(durations) >= 0)
    assert np.all(np.diff(durations, n=2) >= -1e-12)
    assert knuth_defects(costs, tolerance=1e-10).quadrangle == 0


def test_gap_regions_and_public_run_merge_do_not_cross_gaps() -> None:
    regions = atomic_tad_regions(
        14,
        boundary_indices=[2, 5, 9, 11],
        gap_indices=[6, 7],
        minimum_region_bins=2,
    )
    assert regions == [[(0, 2), (3, 5)], [(8, 9), (10, 11), (12, 14)]]

    contacts = np.ones((14, 14), dtype=float)
    np.fill_diagonal(contacts, 0)
    merged = public_deeptad_merge(contacts, regions[1], cosine_threshold=0.8)
    cached = public_deeptad_merge_from_similarities(
        contacts,
        regions[1],
        adjacent_deeptad_cosines(contacts, regions[1]),
        cosine_threshold=0.8,
    )
    assert cached == merged
    assert (8, 11) in merged
    assert all(not (left < 7 < right) for left, right in merged)


def test_nonconvex_duration_is_rejected() -> None:
    with pytest.raises(ValueError, match="discrete convex"):
        monge_hierarchy_costs(
            [(0, 1), (1, 2), (2, 3)],
            separators=[0.0, 0.0],
            duration_costs=[0.0, 1.0, 3.0, 4.0],
        )


def test_nonnegative_atomic_pair_potentials_are_exact_monge() -> None:
    atomic_tads = [(0, 2), (2, 4), (4, 6), (6, 8)]
    base = monge_hierarchy_costs(
        atomic_tads,
        separators=[0.1, 0.2, 0.1],
        duration_costs=quadratic_duration_costs(8, quadratic=0.01),
    )
    pairs = np.asarray(
        [
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0],
            [2.0, 2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 0.0],
        ]
    )
    costs = add_pair_potentials(base, pairs)
    defects = knuth_defects(costs, tolerance=1e-10)
    full = full_cky_min(costs)
    knuth = knuth_cky_min(costs)
    assert defects.quadrangle == 0
    assert defects.inclusion == 0
    assert full.spans == knuth.spans
    assert full.score == pytest.approx(knuth.score)
    assert knuth.splits[0, 4] == 2


def test_atomic_pair_oe_deficit_distinguishes_weak_contacts() -> None:
    contacts = np.ones((8, 8), dtype=float)
    contacts = np.triu(contacts) + np.triu(contacts, 1).T
    # Keep the two within-pair blocks stronger than chromosome-wide expectation
    # and make contacts across the central separator absent.
    contacts[0:4, 0:4] = 4.0
    contacts[4:8, 4:8] = 4.0
    contacts[0:4, 4:8] = 0.0
    contacts[4:8, 0:4] = 0.0
    pairs = atomic_pair_oe_deficits(
        contacts,
        [(0, 2), (2, 4), (4, 6), (6, 8)],
        expected_profile=np.ones(8),
    )
    assert pairs[0, 1] == 0.0
    assert pairs[2, 3] == 0.0
    assert pairs[1, 2] == pytest.approx(1.0)
    assert np.all(pairs >= 0)
    assert np.allclose(pairs, pairs.T)


def test_non_saturating_pair_transforms_preserve_contact_order() -> None:
    mean_oe = np.asarray(
        [[1.0, 0.5, 2.0], [0.5, 1.0, 4.0], [2.0, 4.0, 1.0]]
    )
    hinge = pair_potentials_from_mean_oe(mean_oe, transform="hinge")
    reciprocal = pair_potentials_from_mean_oe(mean_oe, transform="reciprocal")
    exponential = pair_potentials_from_mean_oe(mean_oe, transform="exponential")
    assert hinge[0, 2] == hinge[1, 2] == 0.0
    assert reciprocal[0, 2] > reciprocal[1, 2] > 0.0
    assert exponential[0, 2] > exponential[1, 2] > 0.0
    assert np.all(np.diag(reciprocal) == 0.0)


def test_atomic_pair_mean_oe_uses_distance_specific_expectation() -> None:
    contacts = np.ones((6, 6), dtype=float)
    contacts = np.triu(contacts) + np.triu(contacts, 1).T
    mean_oe = atomic_pair_mean_oe(
        contacts,
        [(0, 2), (2, 4), (4, 6)],
        expected_profile=np.full(6, 2.0),
    )
    assert mean_oe[0, 1] == pytest.approx(0.5)


def test_atomic_profile_dissimilarities_are_nonnegative_and_symmetric() -> None:
    contacts = np.asarray(
        [
            [5, 4, 3, 3, 1, 1],
            [4, 5, 3, 3, 1, 1],
            [3, 3, 5, 4, 1, 1],
            [3, 3, 4, 5, 1, 1],
            [1, 1, 1, 1, 5, 4],
            [1, 1, 1, 1, 4, 5],
        ],
        dtype=float,
    )
    values = atomic_profile_dissimilarities(
        contacts, [(0, 2), (2, 4), (4, 6)]
    )
    assert values.shape == (3, 3)
    assert np.all(values >= 0)
    assert np.allclose(values, values.T)
    assert np.all(np.diag(values) == 0)


def test_separator_path_pair_potential_propagates_strong_boundaries() -> None:
    maximum = separator_path_pair_potentials([0.2, 3.0, 0.5])
    summed = separator_path_pair_potentials([0.2, 3.0, 0.5], aggregation="sum")
    assert maximum[0, 1] == pytest.approx(0.2)
    assert maximum[0, 3] == pytest.approx(3.0)
    assert maximum[1, 3] == pytest.approx(3.0)
    assert summed[0, 3] == pytest.approx(3.7)
    assert np.all(maximum >= 0)
    assert np.allclose(maximum, maximum.T)


def test_direct_and_path_separator_terms_can_be_ablated_independently() -> None:
    region = Region(
        leaves=((0, 2), (2, 4), (4, 6)),
        probabilities=np.asarray([0.2, 0.7]),
        cosines=np.asarray([0.5, 0.2]),
        public_nodes=frozenset(),
    )
    chromosome = Chromosome(
        name="chr1",
        contacts=np.ones((6, 6), dtype=float),
        regions=(region,),
        peak_intervals=np.empty((0, 2), dtype=np.int64),
    )
    full = Objective("full", 0.0, 0.0, 1.0, None, separator_pair_weight=1.0)
    no_direct = Objective(
        "no_direct",
        0.0,
        0.0,
        1.0,
        None,
        separator_pair_weight=1.0,
        direct_separator_weight=0.0,
    )
    no_path = Objective("no_path", 0.0, 0.0, 1.0, None)
    full_costs = _region_costs(chromosome, region, full, expected_profile=None)
    direct_ablated = _region_costs(
        chromosome, region, no_direct, expected_profile=None
    )
    path_ablated = _region_costs(chromosome, region, no_path, expected_profile=None)
    assert not np.allclose(full_costs, direct_ablated)
    assert not np.allclose(full_costs, path_ablated)
    assert knuth_defects(direct_ablated, tolerance=1e-10).quadrangle == 0
    assert knuth_defects(path_ablated, tolerance=1e-10).quadrangle == 0


def test_parent_selection_can_use_a_fixed_ratio_and_endpoint_evidence() -> None:
    contacts = np.ones((8, 8), dtype=float)
    region = Region(
        leaves=((0, 2), (2, 4), (4, 6), (6, 8)),
        probabilities=np.asarray([0.1, 0.1, 0.1]),
        cosines=np.asarray([0.1, 0.9, 0.1]),
        public_nodes=frozenset(),
    )
    selector = ParentSelector(
        boundary_feature="cosine_dissimilarity",
        boundary_reduction="minimum",
        boundary_weight=10.0,
        missing_endpoint_value=0.0,
        cardinality_mode="fixed_ratio_region",
        parent_ratio=1 / 3,
    )
    selected = _select_tree_parents(
        region,
        frozenset({(0, 1), (1, 2), (2, 3), (3, 4), (1, 3), (0, 3), (0, 4)}),
        DistanceNormalizedContacts.from_contacts(contacts, maximum_distance=4),
        selector,
    )
    assert selected == {(2, 6)}
