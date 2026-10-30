import numpy as np
import pytest

from faster_cky.deeptad_evaluation import (
    DistanceNormalizedContacts,
    dual_boundary_peak_support,
    tad_adjusted_r_squared,
)


def test_tad_adjusted_r_squared_rewards_matching_domains() -> None:
    contacts = np.ones((12, 12), dtype=float)
    contacts[:6, :6] = 8.0
    contacts[6:, 6:] = 5.0
    contacts = np.triu(contacts) + np.triu(contacts, 1).T

    gap_only = tad_adjusted_r_squared(contacts, [], maximum_distance=4)
    matching = tad_adjusted_r_squared(
        contacts, [(0, 6), (6, 12)], maximum_distance=4
    )
    assert gap_only.mean == pytest.approx(0.0)
    assert matching.mean > 0.8


def test_nested_children_override_parent_groups() -> None:
    contacts = np.arange(100, dtype=float).reshape(10, 10)
    contacts = contacts + contacts.T
    result = tad_adjusted_r_squared(
        contacts, [(0, 10), (0, 5), (5, 10)], maximum_distance=3
    )
    assert np.array_equal(result.distances, [1, 2, 3])
    assert np.all(np.isfinite(result.scores))


def test_tad_adjusted_r_squared_validates_inputs() -> None:
    with pytest.raises(ValueError, match="outside"):
        tad_adjusted_r_squared(
            np.ones((5, 5)), [(0, 6)], maximum_distance=2
        )


def test_distance_normalized_structural_contrast_prefers_enriched_interval() -> None:
    contacts = np.ones((30, 30), dtype=float)
    contacts[10:20, 10:20] = 5.0
    contacts = np.triu(contacts) + np.triu(contacts, 1).T
    normalized = DistanceNormalizedContacts.from_contacts(
        contacts, maximum_distance=8
    )
    assert normalized.structural_contrast(10, 20) > 0
    assert normalized.structural_contrast(0, 10) < 0


def test_dual_boundary_peak_support_requires_both_ends() -> None:
    supported, total = dual_boundary_peak_support(
        [(1, 5), (1, 8)],
        np.asarray([[900, 1100], [4900, 5100]]),
        resolution=1000,
        tolerance_bp=100,
    )
    assert (supported, total) == (1, 2)
