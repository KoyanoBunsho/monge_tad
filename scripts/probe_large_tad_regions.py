"""Run the frozen deepTAD boundary stages from a narrow band of an mcool file.

The 10x10 deepTAD patches require offsets at most nine; the Wilcoxon filter
requires offsets at most sixteen. Reading these diagonals avoids a dense
whole-chromosome matrix while preserving the existing preprocessing rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cooler
import h5py
import numpy as np

from faster_cky.deeptad_filter import BoundaryFilterResult
from faster_cky.deeptad_monge import atomic_tad_regions, normalize_deeptad_patches
from scipy.stats import mannwhitneyu


def read_band(path: Path, chromosome: str, resolution: int, maximum_lag: int = 16, duplicate_policy: str = "last") -> tuple[np.ndarray, dict]:
    """Return upper diagonals; `last` matches the existing dense-matrix adapter."""
    if duplicate_policy not in {"last", "sum"}:
        raise ValueError("duplicate_policy must be last or sum")
    contact_map = cooler.Cooler(f"{path}::/resolutions/{resolution}")
    start, end = contact_map.extent(chromosome)
    weights = contact_map.bins().fetch(chromosome)["weight"].to_numpy()
    n = int(end - start)
    band = np.zeros((maximum_lag + 1, n), dtype=np.float32)
    duplicate_pixels = 0
    selected_pixels = 0
    with h5py.File(path) as handle:
        group = handle[f"resolutions/{resolution}"]
        offsets = group["indexes/bin1_offset"]
        pixel_start, pixel_end = int(offsets[start]), int(offsets[end])
        pixels = group["pixels"]
        for offset in range(pixel_start, pixel_end, 2_000_000):
            stop = min(offset + 2_000_000, pixel_end)
            row = pixels["bin1_id"][offset:stop] - start
            col = pixels["bin2_id"][offset:stop] - start
            count = pixels["count"][offset:stop]
            lag = col - row
            keep = (row >= 0) & (col < n) & (lag >= 0) & (lag <= maximum_lag) & (count > 0)
            if not np.any(keep):
                continue
            row, col, lag, count = row[keep], col[keep], lag[keep], count[keep]
            value = count * weights[row] * weights[col]
            valid = np.isfinite(value) & (value > 0)
            row, lag, value = row[valid], lag[valid], value[valid]
            if len(row):
                linear = lag * n + row
                duplicate_pixels += len(linear) - len(np.unique(linear))
                if duplicate_policy == "sum":
                    np.add.at(band, (lag, row), value.astype(np.float32))
                else:
                    band[lag, row] = value.astype(np.float32)
                selected_pixels += len(row)
    return band, {
        "chromosome": chromosome, "bins": n, "valid_weight_bins": int(np.count_nonzero(np.isfinite(weights) & (weights > 0))),
        "selected_band_pixels": selected_pixels, "within_chunk_duplicate_pixels": duplicate_pixels,
        "duplicate_policy": duplicate_policy,
    }


def band_patches(band: np.ndarray) -> np.ndarray:
    n = band.shape[1]
    patches = np.zeros((n, 10, 10), dtype=np.float32)
    centers = np.arange(n)
    for a in range(10):
        rows = centers + a - 4
        for b in range(10):
            cols = centers + b - 4
            lag = abs(b - a)
            valid = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
            patches[valid, a, b] = band[lag, np.minimum(rows[valid], cols[valid])]
    return normalize_deeptad_patches(patches)


def band_gaps(band: np.ndarray, window_size: int = 8) -> np.ndarray:
    n = band.shape[1]
    active = band[0] != 0
    for lag in range(1, window_size + 1):
        present = band[lag, : n - lag] != 0
        active[: n - lag] |= present
        active[lag:] |= present
    return np.flatnonzero(~active)


def filter_band(band: np.ndarray, probabilities: np.ndarray, window_size: int = 8, pvalue_cutoff: float = 0.05) -> BoundaryFilterResult:
    n = band.shape[1]
    candidates = np.flatnonzero(probabilities > 0.5)
    gaps = band_gaps(band, window_size)
    active = np.ones(n, dtype=bool)
    active[gaps] = False
    process = np.zeros(n, dtype=bool)
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end - start >= 4:
            process[start:end] = True
    scaled = np.full_like(band, np.nan, dtype=np.float64)
    for lag in range(1, 2 * window_size + 1):
        values = band[lag, : n - lag].astype(np.float64)
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        if std > 0 and np.isfinite(std):
            scaled[lag, : n - lag] = (values - float(np.mean(values))) / std

    def sample(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if not len(left) or not len(right):
            return np.empty(0, dtype=np.float64)
        a, b = np.broadcast_arrays(left[:, None], right[None, :])
        values = scaled[b - a, a]
        return values[np.isfinite(values)]

    pvalues = np.ones(len(candidates), dtype=np.float64)
    for position, boundary in enumerate(candidates):
        if not process[boundary] or boundary == n - 1:
            continue
        left = np.arange(max(0, boundary - window_size + 1), boundary + 1)
        right = np.arange(boundary + 1, min(n, boundary + window_size + 1))
        crossing = sample(left, right)
        upstream = np.arange(max(0, boundary - window_size), boundary + 1)
        downstream = np.arange(boundary + 1, min(n, boundary + window_size + 1))
        ua, ub = np.triu_indices(len(upstream), k=1)
        da, db = np.triu_indices(len(downstream), k=1)
        within = np.concatenate((scaled[upstream[ub] - upstream[ua], upstream[ua]], scaled[downstream[db] - downstream[da], downstream[da]]))
        within = within[np.isfinite(within)]
        if len(crossing) and len(within):
            pvalues[position] = float(mannwhitneyu(crossing, within, alternative="less", method="asymptotic", use_continuity=True).pvalue)
    retained = list(candidates[pvalues < pvalue_cutoff])
    pvalue_by_index = dict(zip(candidates.tolist(), pvalues.tolist()))
    for left, right in zip(retained[:-1], retained[1:]):
        if right == left + 1:
            remove = max((left, right), key=lambda index: (pvalue_by_index[index], index))
            if remove in retained:
                retained.remove(remove)
    return BoundaryFilterResult(candidates, pvalues, np.asarray(retained, dtype=np.int64), gaps)


def summarize(n: int, boundaries: np.ndarray, gaps: np.ndarray, chromosome: str, resolution: int) -> list[dict]:
    rows = []
    for leaves in atomic_tad_regions(n, boundaries, gaps):
        left, right = leaves[0][0], leaves[-1][1]
        lengths = np.asarray([b - a for a, b in leaves])
        rows.append({"chromosome": chromosome, "start_bp": int(left * resolution), "end_bp": int(right * resolution),
                     "span_mb": float((right - left) * resolution / 1e6), "m": int(len(leaves)),
                     "median_leaf_kb": float(np.median(lengths)) * resolution / 1000})
    return sorted(rows, key=lambda row: row["m"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mcool", type=Path)
    source.add_argument("--band-npy", type=Path, help="Precomputed 0--16 contact diagonals")
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--resolution", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, help="Existing or model-generated boundary probability NPZ")
    parser.add_argument("--duplicate-policy", choices=("last", "sum"), default="last")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.band_npy:
        band = np.load(args.band_npy)
        if band.ndim != 2 or band.shape[0] < 17:
            raise ValueError("band must have at least 17 diagonals")
        metadata = {"chromosome": args.chromosome, "bins": int(band.shape[1]), "source_band": str(args.band_npy)}
    else:
        band, metadata = read_band(args.mcool, args.chromosome, args.resolution, duplicate_policy=args.duplicate_policy)
    metadata["band_seconds"] = time.perf_counter() - started
    label = args.chromosome
    np.save(args.output_dir / f"{label}_patches.npy", band_patches(band))
    metadata["gap_bins"] = int(len(band_gaps(band)))
    if args.prediction:
        with np.load(args.prediction) as archive:
            probabilities = archive["boundary_probability"]
        result = filter_band(band, probabilities)
        np.savez_compressed(args.output_dir / f"{label}_filtered.npz", candidate_index=result.candidate_indices,
                            candidate_probability=probabilities[result.candidate_indices], candidate_pvalue=result.candidate_pvalues,
                            filtered_index=result.filtered_indices, filtered_probability=probabilities[result.filtered_indices], gap_index=result.gap_indices)
        rows = summarize(band.shape[1], result.filtered_indices, result.gap_indices, label, args.resolution)
        (args.output_dir / f"{label}_regions.json").write_text(json.dumps(rows, indent=2) + "\n")
        metadata.update({"candidate_boundaries": int(len(result.candidate_indices)), "filtered_boundaries": int(len(result.filtered_indices)),
                         "regions": int(len(rows)), "maximum_m": int(max((row["m"] for row in rows), default=0))})
    (args.output_dir / f"{label}_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
