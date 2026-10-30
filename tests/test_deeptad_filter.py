import numpy as np

from faster_cky.deeptad_filter import filter_deeptad_boundaries


def test_filter_retains_a_low_contact_boundary_and_removes_adjacent_duplicate() -> None:
    matrix = np.full((30, 30), 5.0)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0.0)
    matrix[:15, 15:] = 0.01
    matrix[15:, :15] = 0.01
    result = filter_deeptad_boundaries(
        matrix,
        np.asarray([14, 15, 22]),
        window_size=5,
        pvalue_cutoff=0.05,
    )
    assert 14 in result.filtered_indices or 15 in result.filtered_indices
    assert not ({14, 15} <= set(result.filtered_indices))
    assert 22 not in result.filtered_indices


def test_filter_marks_zero_contact_bins_as_gaps() -> None:
    matrix = np.ones((12, 12))
    matrix[6] = 0
    matrix[:, 6] = 0
    result = filter_deeptad_boundaries(
        matrix, np.asarray([5, 6]), window_size=3
    )
    assert 6 in result.gap_indices
    assert 6 not in result.filtered_indices
