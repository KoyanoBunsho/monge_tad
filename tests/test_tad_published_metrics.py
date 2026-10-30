import numpy as np
import pytest
from faster_cky.tad_published_metrics import clustertad_quality, long_ctcf_support, outermost_intervals


def test_clustertad_uses_published_triangle_not_full_rectangle():
    c = np.array([[0, 5, 1, 100], [5, 0, 2, 3], [1, 2, 0, 9], [100, 3, 9, 0]], dtype=float)
    rows = clustertad_quality(c, [(0, 2), (2, 4)])
    assert [r["intra"] for r in rows] == [5, 9]
    assert [r["inter"] for r in rows] == [2, 2]
    assert [r["quality"] for r in rows] == [3, 7]
    assert [r["inter_pixels"] for r in rows] == [3, 3]


def test_clustertad_matches_literal_one_based_author_loops():
    c = np.random.default_rng(42).integers(0, 30, size=(14, 14)).astype(float)
    domains = [(0, 3), (4, 8), (8, 14)]
    rows = clustertad_quality(c, domains)
    for i, ((a, b), result) in enumerate(zip(domains, rows)):
        intra = [c[u-1, v-1] for u in range(a+1, b+1) for v in range(u+1, b+1)]
        inter = []
        for j in [i-1, i+1]:
            if 0 <= j < len(domains):
                (x, y), (z, w) = sorted([domains[i], domains[j]])
                for u in range(x+1, z+1):
                    increment = u-x
                    for k, v in enumerate(range(y+1, w+1), start=1):
                        inter.append(c[u-1, v-1])
                        if k == increment:
                            break
        assert result["quality"] == pytest.approx(np.mean(intra)-np.mean(inter))
        assert result["inter_pixels"] == len(inter)


def test_hierarchy_layer_and_missing_neighbor():
    assert outermost_intervals([(0, 8), (0, 3), (1, 7), (8, 12), (8, 12)]) == [(0, 8), (8, 12)]
    assert clustertad_quality(np.ones((8, 8)), [(0, 8)])[0]["quality"] is None


def test_long_uses_both_endpoints_and_ten_kb_in_base_pairs():
    peaks = np.array([[10_000, 10_100], [109_000, 109_100]])
    assert long_ctcf_support([(0, 100_000)], peaks)["rate"] == 1
    assert long_ctcf_support([(0, 120_000)], peaks)["rate"] == 0
    assert long_ctcf_support([(0, 100_000)], np.empty((0, 2), dtype=int))["rate"] == 0
