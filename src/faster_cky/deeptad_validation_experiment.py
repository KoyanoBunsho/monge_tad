"""Select a fixed Monge hierarchy objective on deepTAD validation chromosomes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .decoder import full_cky_min, knuth_cky_min
from .deeptad_evaluation import (
    DistanceNormalizedContacts,
    dual_boundary_peak_support,
    tad_adjusted_r_squared,
)
from .deeptad_monge import (
    adjacent_deeptad_cosines,
    atomic_tad_regions,
    monge_hierarchy_costs,
    public_deeptad_merge_from_similarities,
    quadratic_duration_costs,
    separator_weights,
)
from .tad_matrix_io import load_dense_matrix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Region:
    leaves: tuple[tuple[int, int], ...]
    probabilities: np.ndarray
    cosines: np.ndarray
    public_nodes: frozenset[tuple[int, int]]
    cosine_seconds: float = 0.0
    public_merge_seconds: float = 0.0


@dataclass(frozen=True)
class Chromosome:
    name: str
    contacts: np.ndarray
    regions: tuple[Region, ...]
    peak_intervals: np.ndarray


def _load_peaks(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    opener = gzip.open if path.suffix == ".gz" else open
    by_chromosome: dict[str, list[tuple[int, int]]] = {}
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            chromosome = fields[0]
            if not chromosome.startswith("chr"):
                chromosome = f"chr{chromosome}"
            by_chromosome.setdefault(chromosome, []).append(
                (int(fields[1]), int(fields[2]))
            )
    return {
        chromosome: np.asarray(intervals, dtype=np.int64)
        for chromosome, intervals in by_chromosome.items()
    }


def _load_split(
    chromosomes: list[str],
    matrix_dir: Path,
    prediction_dir: Path,
    filtered_dir: Path,
    minimum_region_bins: int,
    peaks: dict[str, np.ndarray],
    resolution: int = 25_000,
    sample_prefix: str = "HIC002",
) -> list[Chromosome]:
    loaded = []
    display_resolution = resolution // 1000
    for chromosome in chromosomes:
        contacts, _ = load_dense_matrix(
            matrix_dir / f"{sample_prefix}_{display_resolution}k_KR.chr{chromosome}",
            dtype=np.float64,
            mmap=True,
        )
        with np.load(
            prediction_dir / f"{sample_prefix}_{display_resolution}k_KR.chr{chromosome}.npz"
        ) as archive:
            probabilities = archive["boundary_probability"]
        with np.load(
            filtered_dir / f"{sample_prefix}_{display_resolution}k_KR.chr{chromosome}.npz"
        ) as archive:
            boundaries = archive["filtered_index"]
            gaps = archive["gap_index"]
        leaf_regions = atomic_tad_regions(
            len(contacts),
            boundaries,
            gaps,
            minimum_region_bins=minimum_region_bins,
        )
        diagonal = np.diagonal(contacts)
        nonzero_count = int(np.count_nonzero(contacts)) - int(np.count_nonzero(diagonal))
        global_average_contact = (
            (float(np.sum(contacts)) - float(np.sum(diagonal))) / nonzero_count
            if nonzero_count
            else 0.0
        )
        regions = []
        for leaves in leaf_regions:
            if len(leaves) < 2:
                regions.append(
                    Region(tuple(leaves), np.empty(0), np.empty(0), frozenset(leaves))
                )
                continue
            offset = leaves[0][0]
            right_edge = leaves[-1][1]
            local_contacts = contacts[offset:right_edge, offset:right_edge]
            local_leaves = [
                (left - offset, right - offset) for left, right in leaves
            ]
            started = time.perf_counter()
            cosines = adjacent_deeptad_cosines(local_contacts, local_leaves)
            cosine_seconds = time.perf_counter() - started
            started = time.perf_counter()
            local_public_nodes = public_deeptad_merge_from_similarities(
                local_contacts,
                local_leaves,
                cosines,
                average_contact=global_average_contact,
            )
            public_merge_seconds = time.perf_counter() - started
            public_nodes = frozenset(
                (left + offset, right + offset)
                for left, right in local_public_nodes
            )
            separator_bins = np.asarray(
                [right for _, right in leaves[:-1]], dtype=np.int64
            )
            regions.append(
                Region(
                    tuple(leaves),
                    probabilities[separator_bins].astype(np.float64),
                    cosines,
                    public_nodes,
                    cosine_seconds,
                    public_merge_seconds,
                )
            )
        name = f"chr{chromosome}"
        loaded.append(
            Chromosome(
                name,
                contacts,
                tuple(regions),
                peaks.get(name, np.empty((0, 2), dtype=np.int64)),
            )
        )
    return loaded


def _decode(
    chromosomes: list[Chromosome],
    *,
    alpha: float,
    beta: float,
    duration_scale: float,
    maximum_distance: int,
    mode: str = "features",
    seed: int = 123,
    verify_full: bool = False,
    parent_ratio: float | None = None,
    resolution: int = 25_000,
    peak_tolerance_bp: int = 25_000,
) -> dict[str, object]:
    chromosome_rows = []
    all_exact = True
    for chromosome_number, chromosome in enumerate(chromosomes):
        nodes: set[tuple[int, int]] = set()
        leaves: set[tuple[int, int]] = set()
        parent_candidates: list[tuple[float, int, int, tuple[int, int]]] = []
        normalized = (
            DistanceNormalizedContacts.from_contacts(
                chromosome.contacts, maximum_distance=maximum_distance
            )
            if parent_ratio is not None
            else None
        )
        full_candidates = 0
        knuth_candidates = 0
        full_seconds = 0.0
        knuth_seconds = 0.0
        for region_number, region in enumerate(chromosome.regions):
            leaves.update(region.leaves)
            nodes.update(region.leaves)
            if len(region.leaves) < 2:
                continue
            started = time.perf_counter()
            separators = separator_weights(
                boundary_probabilities=region.probabilities,
                cosine_similarities=region.cosines,
                probability_weight=alpha,
                cosine_weight=beta,
            )
            if mode == "constant":
                separators = np.ones_like(separators)
            elif mode == "shuffled":
                rng = np.random.default_rng(
                    seed + chromosome_number * 10_000 + region_number
                )
                separators = rng.permutation(separators)
            elif mode != "features":
                raise ValueError(f"unknown mode: {mode}")
            offset = region.leaves[0][0]
            local = [(left - offset, right - offset) for left, right in region.leaves]
            duration = quadratic_duration_costs(
                region.leaves[-1][1] - offset,
                quadratic=1.0 / duration_scale**2,
            )
            costs = monge_hierarchy_costs(local, separators, duration)
            knuth = knuth_cky_min(costs)
            knuth_seconds += time.perf_counter() - started
            knuth_candidates += knuth.candidates
            if verify_full:
                started = time.perf_counter()
                full = full_cky_min(costs)
                full_seconds += time.perf_counter() - started
                full_candidates += full.candidates
                all_exact &= full.spans == knuth.spans and bool(
                    np.isclose(full.score, knuth.score)
                )
            for left, right in knuth.spans:
                node = (region.leaves[left][0], region.leaves[right - 1][1])
                if normalized is None:
                    nodes.add(node)
                else:
                    parent_candidates.append(
                        (
                            normalized.structural_contrast(*node),
                            -(node[1] - node[0]),
                            -node[0],
                            node,
                        )
                    )
        if parent_ratio is not None:
            keep = min(
                len(parent_candidates), int(round(parent_ratio * len(leaves)))
            )
            parent_candidates.sort(reverse=True)
            nodes.update(row[-1] for row in parent_candidates[:keep])
        metric = tad_adjusted_r_squared(
            chromosome.contacts, sorted(nodes), maximum_distance=maximum_distance
        )
        parents = nodes - leaves
        supported, evaluated = dual_boundary_peak_support(
            sorted(parents),
            chromosome.peak_intervals,
            resolution=resolution,
            tolerance_bp=peak_tolerance_bp,
        )
        chromosome_rows.append(
            {
                "chromosome": chromosome.name,
                "leaf_nodes": len(leaves),
                "output_nodes": len(nodes),
                "parent_nodes": len(nodes - leaves),
                "tad_adjusted_r2": metric.mean,
                "dual_end_ctcf_supported": supported,
                "dual_end_ctcf_evaluated": evaluated,
                "dual_end_ctcf_rate": supported / evaluated if evaluated else None,
                "knuth_candidates": knuth_candidates,
                "knuth_seconds": knuth_seconds,
                "full_candidates": full_candidates if verify_full else None,
                "full_seconds": full_seconds if verify_full else None,
            }
        )
    return {
        "alpha": alpha,
        "beta": beta,
        "duration_scale_bins": duration_scale,
        "mode": mode,
        "parent_ratio": parent_ratio,
        "mean_tad_adjusted_r2": float(
            np.mean([row["tad_adjusted_r2"] for row in chromosome_rows])
        ),
        "all_full_knuth_exact": all_exact if verify_full else None,
        "chromosomes": chromosome_rows,
    }


def _public_baseline(
    chromosomes: list[Chromosome],
    maximum_distance: int,
    *,
    resolution: int,
    peak_tolerance_bp: int,
) -> dict[str, object]:
    rows = []
    for chromosome in chromosomes:
        nodes = {
            node for region in chromosome.regions for node in region.public_nodes
        }
        leaves = {node for region in chromosome.regions for node in region.leaves}
        metric = tad_adjusted_r_squared(
            chromosome.contacts, sorted(nodes), maximum_distance=maximum_distance
        )
        parents = nodes - leaves
        supported, evaluated = dual_boundary_peak_support(
            sorted(parents),
            chromosome.peak_intervals,
            resolution=resolution,
            tolerance_bp=peak_tolerance_bp,
        )
        rows.append(
            {
                "chromosome": chromosome.name,
                "leaf_nodes": len(leaves),
                "output_nodes": len(nodes),
                "parent_nodes": len(nodes - leaves),
                "tad_adjusted_r2": metric.mean,
                "dual_end_ctcf_supported": supported,
                "dual_end_ctcf_evaluated": evaluated,
                "dual_end_ctcf_rate": supported / evaluated if evaluated else None,
                "cosine_seconds": sum(
                    region.cosine_seconds for region in chromosome.regions
                ),
                "merge_seconds": sum(
                    region.public_merge_seconds for region in chromosome.regions
                ),
                "merge_candidates": sum(
                    max(0, len(region.leaves) - 1) for region in chromosome.regions
                ),
            }
        )
    return {
        "name": "public_deeptad_merge",
        "mean_tad_adjusted_r2": float(
            np.mean([row["tad_adjusted_r2"] for row in rows])
        ),
        "chromosomes": rows,
    }


def _markdown(result: dict[str, object]) -> str:
    dev = result["development"]
    test = result["test"]
    lines = [
        "# Fixed-leaf deepTAD Monge validation",
        "",
        "The objective coefficients were selected on chr13--19 and frozen before chr20--22 evaluation.",
        "The contrast-collapsed row retains a development-fixed parent/leaf ratio and ranks nodes by distance-normalized contact contrast.",
        "",
        "## Result",
        "",
        "| Split | Method | mean TAD-adjR2 | parent nodes | dual-end CTCF | method candidates | method seconds | shared cosine seconds | oracle exact |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split_name, section in (("chr13--19", dev), ("chr20--22", test)):
        baseline = section["public"]
        lines.append(
            f"| {split_name} | public deepTAD merge | {baseline['mean_tad_adjusted_r2']:.4f} | "
            f"{sum(row['parent_nodes'] for row in baseline['chromosomes'])} | "
            f"{sum(row['dual_end_ctcf_supported'] for row in baseline['chromosomes'])}/"
            f"{sum(row['dual_end_ctcf_evaluated'] for row in baseline['chromosomes'])} | "
            f"{sum(row['merge_candidates'] for row in baseline['chromosomes'])} | "
            f"{sum(row['merge_seconds'] for row in baseline['chromosomes']):.4f} | "
            f"{sum(row['cosine_seconds'] for row in baseline['chromosomes']):.4f} | - |"
        )
        for row in section["comparisons"]:
            lines.append(
                f"| {split_name} | {row['name']} | {row['mean_tad_adjusted_r2']:.4f} | "
                f"{sum(item['parent_nodes'] for item in row['chromosomes'])} | "
                f"{sum(item['dual_end_ctcf_supported'] for item in row['chromosomes'])}/"
                f"{sum(item['dual_end_ctcf_evaluated'] for item in row['chromosomes'])} | "
                f"{sum(item['knuth_candidates'] for item in row['chromosomes'])} | "
                f"{sum(item['knuth_seconds'] for item in row['chromosomes']):.4f} | "
                f"{sum(item['cosine_seconds'] for item in baseline['chromosomes']):.4f} | "
                f"{row['all_full_knuth_exact']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation limit",
            "",
            "The local TAD-adjR2 implementation follows the published distance-stratified adjusted-R2 formula,",
            "but it does not reproduce deepTAD's published absolute chr20--22 values from the public artifacts.",
            "Only within-run relative comparisons are currently supported. The full-dendrogram score is exploratory",
            "because that output contains many more parent nodes than the public merge; the contrast-collapsed row controls this density.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    peaks = _load_peaks(args.ctcf_peaks)
    dev = _load_split(
        args.dev_chromosomes,
        args.matrix_dir,
        args.dev_prediction_dir,
        args.dev_filtered_dir,
        args.minimum_region_bins,
        peaks,
        args.resolution,
        args.sample_prefix,
    )
    test = _load_split(
        args.test_chromosomes,
        args.matrix_dir,
        args.test_prediction_dir,
        args.test_filtered_dir,
        args.minimum_region_bins,
        peaks,
        args.resolution,
        args.sample_prefix,
    )
    grid = []
    for alpha in args.alpha_grid:
        for duration_scale in args.duration_scale_grid:
            grid.append(
                _decode(
                    dev,
                    alpha=alpha,
                    beta=1.0,
                    duration_scale=duration_scale,
                    maximum_distance=args.maximum_distance,
                )
            )
    selected = max(
        grid,
        key=lambda row: (
            row["mean_tad_adjusted_r2"],
            -row["alpha"],
            -row["duration_scale_bins"],
        ),
    )
    alpha = float(selected["alpha"])
    scale = float(selected["duration_scale_bins"])

    dev_public = _public_baseline(
        dev,
        args.maximum_distance,
        resolution=args.resolution,
        peak_tolerance_bp=args.peak_tolerance_bp,
    )
    test_public = _public_baseline(
        test,
        args.maximum_distance,
        resolution=args.resolution,
        peak_tolerance_bp=args.peak_tolerance_bp,
    )
    dev_parent_count = sum(row["parent_nodes"] for row in dev_public["chromosomes"])
    dev_leaf_count = sum(row["leaf_nodes"] for row in dev_public["chromosomes"])
    parent_ratio = dev_parent_count / dev_leaf_count
    comparison_specs = [
        ("selected_full_tree", alpha, 1.0, scale, "features", None),
        ("selected_contrast_collapsed", alpha, 1.0, scale, "features", parent_ratio),
        ("probability_only_full_tree", 1.0, 0.0, scale, "features", None),
        ("constant_q_full_tree", 0.0, 1.0, scale, "constant", None),
        ("shuffled_q_full_tree", alpha, 1.0, scale, "shuffled", None),
    ]

    def comparisons(chromosomes: list[Chromosome]) -> list[dict[str, object]]:
        rows = []
        for name, a, b, duration, mode, collapse_ratio in comparison_specs:
            row = _decode(
                chromosomes,
                alpha=a,
                beta=b,
                duration_scale=duration,
                maximum_distance=args.maximum_distance,
                mode=mode,
                seed=args.seed,
                verify_full=not args.skip_full_verification,
                parent_ratio=collapse_ratio,
                resolution=args.resolution,
                peak_tolerance_bp=args.peak_tolerance_bp,
            )
            row["name"] = name
            rows.append(row)
        return rows

    result = {
        "schema_version": 1,
        "dataset": {
            "sample_prefix": args.sample_prefix,
            "resolution": args.resolution,
        },
        "metric": {
            "name": "distance-stratified TAD-adjusted R2",
            "maximum_distance_bins": args.maximum_distance,
            "absolute_value_calibrated_to_deepTAD_paper": False,
            "ctcf_peaks": str(args.ctcf_peaks) if args.ctcf_peaks else None,
            "ctcf_peaks_sha256": _sha256(args.ctcf_peaks) if args.ctcf_peaks else None,
            "ctcf_peaks_source": (
                "GEO GSM749704, GM12878 CTCF ChIP-seq, hg19 narrowPeak"
                if args.ctcf_peaks
                else None
            ),
            "peak_tolerance_bp": args.peak_tolerance_bp,
        },
        "selection": {
            "chromosomes": args.dev_chromosomes,
            "alpha": alpha,
            "beta": 1.0,
            "duration_scale_bins": scale,
            "collapsed_parent_per_leaf_ratio": parent_ratio,
        },
        "development": {
            "public": dev_public,
            "comparisons": comparisons(dev),
        },
        "test": {
            "public": test_public,
            "comparisons": comparisons(test),
        },
        "development_grid": grid,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "RESULTS.md").write_text(_markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--dev-prediction-dir", type=Path, required=True)
    parser.add_argument("--dev-filtered-dir", type=Path, required=True)
    parser.add_argument("--test-prediction-dir", type=Path, required=True)
    parser.add_argument("--test-filtered-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ctcf-peaks", type=Path)
    parser.add_argument("--resolution", type=int, default=25_000)
    parser.add_argument("--sample-prefix", default="HIC002")
    parser.add_argument("--peak-tolerance-bp", type=int, default=25_000)
    parser.add_argument(
        "--dev-chromosomes",
        nargs="+",
        default=["13", "14", "15", "16", "17", "18", "19"],
    )
    parser.add_argument("--test-chromosomes", nargs="+", default=["20", "21", "22"])
    parser.add_argument(
        "--alpha-grid", nargs="+", type=float, default=[0.0, 0.01, 0.03, 0.1, 0.3]
    )
    parser.add_argument(
        "--duration-scale-grid", nargs="+", type=float, default=[10, 20, 40, 80, 160]
    )
    parser.add_argument("--maximum-distance", type=int, default=60)
    parser.add_argument("--minimum-region-bins", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-full-verification", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result["selection"], sort_keys=True))


if __name__ == "__main__":
    main()
