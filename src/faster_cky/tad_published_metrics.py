"""Published TAD validation measures, with explicit hierarchy preprocessing.

ClusterTAD: Oluwadare and Cheng (2017), doi:10.1186/s12859-017-1931-2.
The intra/inter calculation follows intra.m, inter_v2.m, and QualityCheck.m at
commit 3d1ffa5b91743ee2bd55a5559561b4240d2f6cb1, including the triangular inter
region and pixel-count weighting of the two neighboring regions.

CTCF: Long et al. (2022), doi:10.1186/s13072-022-00436-9: a ChIP-seq peak
within 10 kb of each boundary. No motif orientation is inferred by this measure.
"""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np

Span = tuple[int, int]


def outermost_intervals(domains: Sequence[Span]) -> list[Span]:
    """Remove contained intervals, retaining the outer layer and crossing pairs."""
    result: list[Span] = []
    greatest_end = -1
    for a, b in sorted(set(domains), key=lambda span: (span[0], -span[1])):
        if a < 0 or b <= a:
            raise ValueError(f"Invalid half-open interval: {(a, b)}")
        if b > greatest_end:
            result.append((a, b))
            greatest_end = b
    return result


def _inter_sum_count(matrix: np.ndarray, left: Span, right: Span) -> tuple[float, int]:
    # Translate the author's inclusive, one-based loops to half-open zero-based
    # slices. The published code does not use the entire inter-domain rectangle.
    a, b = left
    c, d = right
    total, count = 0.0, 0
    for u in range(a, c):
        stop = min(d, b + u - a + 1)
        if stop > b:
            total += float(np.sum(matrix[u, b:stop], dtype=np.float64))
            count += stop - b
    return total, count


def clustertad_quality(matrix: np.ndarray, domains: Sequence[Span]) -> list[dict]:
    """Return the author's per-domain intra-minus-inter quality on supplied order.

    Call outermost_intervals explicitly before scoring a hierarchy. A singleton
    domain set has no adjacent domain and receives an unavailable score, rather
    than treating the missing between-domain mean as zero.
    """
    intervals = sorted(set(domains))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square contact matrix")
    if any(a < 0 or b <= a or b > len(matrix) for a, b in intervals):
        raise ValueError("Domain coordinates are outside the contact matrix")
    adjacent = [_inter_sum_count(matrix, a, b) for a, b in zip(intervals, intervals[1:])]
    rows = []
    for i, (a, b) in enumerate(intervals):
        intra_sum = sum(float(np.sum(matrix[u, u+1:b], dtype=np.float64)) for u in range(a, b-1))
        intra_count = (b-a)*(b-a-1)//2
        intra = intra_sum / intra_count if intra_count else 0.0
        neighbors = adjacent[max(0, i-1):min(len(adjacent), i+1)]
        inter_sum = sum(s for s, _ in neighbors)
        inter_count = sum(n for _, n in neighbors)
        inter = inter_sum / inter_count if inter_count else None
        if not np.isfinite(intra) or (inter is not None and not np.isfinite(inter)):
            raise ValueError("Nonfinite contact values in evaluated pixels")
        rows.append(dict(start_bin=a, end_bin=b, intra=intra, inter=inter,
                         intra_pixels=intra_count, inter_pixels=inter_count,
                         quality=intra-inter if inter is not None else None))
    return rows


def long_ctcf_support(domains_bp: Sequence[Span], peaks_bp: np.ndarray) -> dict:
    """Fraction with a ChIP-seq peak within +/-10,000 bp of both endpoints."""
    peaks = np.asarray(peaks_bp, dtype=np.int64).reshape(-1, 2)
    if len(peaks):
        peaks = peaks[np.argsort(peaks[:, 0], kind="stable")]
        starts, ends = peaks[:, 0], np.maximum.accumulate(peaks[:, 1])
    else:
        starts = ends = np.array([], dtype=np.int64)

    def near(x):
        idx = np.searchsorted(starts, x+10_000, side="right")-1
        return bool(idx >= 0 and ends[idx] >= x-10_000)

    domains = sorted(set(domains_bp))
    if any(a < 0 or b <= a for a, b in domains):
        raise ValueError("Expected nonempty genomic intervals")
    supported = sum(near(a) and near(b) for a, b in domains)
    return dict(supported=supported, evaluated=len(domains),
                rate=supported/len(domains) if domains else None, tolerance_bp=10_000)
