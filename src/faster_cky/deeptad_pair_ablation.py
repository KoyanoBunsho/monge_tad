"""Select a nonnegative pair-potential hierarchy objective on development data.

The experiment keeps deepTAD's boundary detector, filtered atomic TADs, and
per-region output cardinality fixed.  It changes only the hierarchy tree used
to supply parent-TAD candidates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
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
    add_pair_potentials,
    atomic_profile_dissimilarities,
    atomic_pair_mean_oe,
    distance_expected_profile,
    monge_hierarchy_costs,
    pair_potentials_from_mean_oe,
    quadratic_duration_costs,
    separator_weights,
    separator_path_pair_potentials,
)
from .deeptad_validation_experiment import Chromosome, Region, _load_peaks, _load_split
from .structure import knuth_defects


Span = tuple[int, int]


@dataclass(frozen=True)
class Objective:
    name: str
    pair_weight: float
    probability_weight: float
    cosine_weight: float
    duration_scale: float | None
    pair_transform: str = "hinge"
    profile_weight: float = 0.0
    separator_pair_weight: float = 0.0
    separator_pair_aggregation: str = "maximum"
    direct_separator_weight: float = 1.0


@dataclass(frozen=True)
class ParentSelector:
    """Rank biological parent TADs within an exact full binary tree."""

    boundary_feature: str = "none"
    boundary_reduction: str = "minimum"
    boundary_weight: float = 0.0
    missing_endpoint_value: float = 0.0
    cardinality_mode: str = "match_public_region"
    parent_ratio: float | None = None
    maximum_parent_bins: int | None = None


def _zero_duration(maximum_length: int) -> np.ndarray:
    return np.zeros(maximum_length + 1, dtype=np.float64)


def _region_costs(
    chromosome: Chromosome,
    region: Region,
    objective: Objective,
    *,
    expected_profile: np.ndarray | None,
    pair_mean_oe: np.ndarray | None = None,
    profile_dissimilarities: np.ndarray | None = None,
) -> np.ndarray:
    offset = region.leaves[0][0]
    local = [(left - offset, right - offset) for left, right in region.leaves]
    maximum_length = region.leaves[-1][1] - offset
    if objective.probability_weight or objective.cosine_weight:
        separators = separator_weights(
            boundary_probabilities=region.probabilities,
            cosine_similarities=region.cosines,
            probability_weight=objective.probability_weight,
            cosine_weight=objective.cosine_weight,
        )
    else:
        separators = np.zeros(len(region.leaves) - 1, dtype=np.float64)
    durations = (
        _zero_duration(maximum_length)
        if objective.duration_scale is None
        else quadratic_duration_costs(
            maximum_length, quadratic=1.0 / objective.duration_scale**2
        )
    )
    costs = monge_hierarchy_costs(
        local, objective.direct_separator_weight * separators, durations
    )
    if objective.pair_weight or objective.profile_weight or objective.separator_pair_weight:
        pairs = np.zeros((len(region.leaves), len(region.leaves)), dtype=np.float64)
        if objective.pair_weight:
            mean_oe = (
                atomic_pair_mean_oe(
                    chromosome.contacts,
                    region.leaves,
                    expected_profile=expected_profile,
                )
                if pair_mean_oe is None
                else pair_mean_oe
            )
            pairs += objective.pair_weight * pair_potentials_from_mean_oe(
                mean_oe,
                transform=objective.pair_transform,
            )
        if objective.profile_weight:
            profile = (
                atomic_profile_dissimilarities(chromosome.contacts, region.leaves)
                if profile_dissimilarities is None
                else profile_dissimilarities
            )
            pairs += objective.profile_weight * profile
        if objective.separator_pair_weight:
            pairs += objective.separator_pair_weight * separator_path_pair_potentials(
                separators, aggregation=objective.separator_pair_aggregation
            )
        costs = add_pair_potentials(costs, pairs)
    return costs


def _standardize(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    standardized = np.full(len(array), -5.0, dtype=np.float64)
    if np.any(finite):
        mean = float(np.mean(array[finite]))
        standard_deviation = float(np.std(array[finite]))
        standardized[finite] = (array[finite] - mean) / (
            standard_deviation if standard_deviation > 1e-12 else 1.0
        )
    return standardized


def _endpoint_boundary_score(
    region: Region,
    left: int,
    right: int,
    selector: ParentSelector,
) -> float:
    endpoint_values = []
    for separator_index in (left - 1, right - 1):
        if separator_index < 0 or separator_index >= len(region.probabilities):
            endpoint_values.append(selector.missing_endpoint_value)
            continue
        probability = float(region.probabilities[separator_index])
        cosine_dissimilarity = 1.0 - float(region.cosines[separator_index])
        if selector.boundary_feature == "probability":
            endpoint_values.append(probability)
        elif selector.boundary_feature == "cosine_dissimilarity":
            endpoint_values.append(cosine_dissimilarity)
        elif selector.boundary_feature == "combined_separator":
            endpoint_values.append(
                -np.log(max(1e-8, 1.0 - probability)) + cosine_dissimilarity
            )
        else:
            raise ValueError(f"unknown parent boundary feature: {selector.boundary_feature}")
    if selector.boundary_reduction == "minimum":
        return min(endpoint_values)
    if selector.boundary_reduction == "mean":
        return float(np.mean(endpoint_values))
    raise ValueError(f"unknown endpoint reduction: {selector.boundary_reduction}")


def _select_tree_parents(
    region: Region,
    spans: frozenset[tuple[int, int]],
    normalized: DistanceNormalizedContacts,
    selector: ParentSelector,
) -> set[Span]:
    indexed_candidates = [
        (left, right, (region.leaves[left][0], region.leaves[right - 1][1]))
        for left, right in spans
        if right - left > 1
    ]
    contrast = _standardize(
        [normalized.structural_contrast(*node) for _, _, node in indexed_candidates]
    )
    if selector.boundary_feature == "none" or selector.boundary_weight == 0:
        boundary = np.zeros(len(indexed_candidates), dtype=np.float64)
    else:
        boundary = _standardize(
            [
                _endpoint_boundary_score(region, left, right, selector)
                for left, right, _ in indexed_candidates
            ]
        )
    ranked = sorted(
        zip(indexed_candidates, contrast + selector.boundary_weight * boundary),
        key=lambda item: (
            item[1],
            -(item[0][2][1] - item[0][2][0]),
            -item[0][2][0],
        ),
        reverse=True,
    )
    if selector.cardinality_mode == "match_public_region":
        parent_count = len(set(region.public_nodes) - set(region.leaves))
    elif selector.cardinality_mode == "fixed_ratio_region":
        if selector.parent_ratio is None or not 0 <= selector.parent_ratio <= 1:
            raise ValueError("fixed-ratio parent selection requires a ratio in [0, 1]")
        parent_count = int(round(selector.parent_ratio * len(indexed_candidates)))
    else:
        raise ValueError(f"unknown parent cardinality mode: {selector.cardinality_mode}")
    if selector.maximum_parent_bins is not None:
        if selector.maximum_parent_bins < 1:
            raise ValueError("maximum_parent_bins must be positive")
        ranked = [
            item
            for item in ranked
            if item[0][2][1] - item[0][2][0] <= selector.maximum_parent_bins
        ]
    return {
        node
        for ((_, _, node), _) in ranked[:parent_count]
    }


def _region_metric(
    chromosome: Chromosome,
    region: Region,
    nodes: set[Span],
    *,
    maximum_distance: int,
    resolution: int,
    peak_tolerance_bp: int,
) -> dict[str, object]:
    left = region.leaves[0][0]
    right = region.leaves[-1][1]
    local_nodes = [(a - left, b - left) for a, b in nodes]
    adjusted = tad_adjusted_r_squared(
        chromosome.contacts[left:right, left:right],
        local_nodes,
        maximum_distance=min(maximum_distance, right - left - 1),
    ).mean
    parents = nodes - set(region.leaves)
    supported, evaluated = (
        dual_boundary_peak_support(
            sorted(parents),
            chromosome.peak_intervals,
            resolution=resolution,
            tolerance_bp=peak_tolerance_bp,
        )
        if len(chromosome.peak_intervals)
        else (0, 0)
    )
    return {
        "atomic_tads": len(region.leaves),
        "bins": right - left,
        "parent_nodes": len(parents),
        "tad_adjusted_r2": adjusted,
        "dual_end_ctcf_supported": supported,
        "dual_end_ctcf_evaluated": evaluated,
    }


def evaluate_objective(
    chromosomes: list[Chromosome],
    objective: Objective,
    *,
    maximum_distance: int,
    resolution: int,
    peak_tolerance_bp: int,
    verify_full_max_leaves: int = 0,
    pair_cache: dict[tuple[str, int], np.ndarray] | None = None,
    profile_cache: dict[tuple[str, int], np.ndarray] | None = None,
    parent_selector: ParentSelector = ParentSelector(),
) -> dict[str, object]:
    chromosome_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    all_verified_exact = True
    verified_regions = 0
    total_seconds = 0.0
    total_candidates = 0
    for chromosome in chromosomes:
        expected = (
            distance_expected_profile(chromosome.contacts)
            if pair_cache is None and objective.pair_weight
            else None
        )
        normalized = DistanceNormalizedContacts.from_contacts(
            chromosome.contacts, maximum_distance=maximum_distance
        )
        nodes: set[Span] = set()
        leaves: set[Span] = set()
        chromosome_seconds = 0.0
        chromosome_candidates = 0
        for region_index, region in enumerate(chromosome.regions):
            region_leaves = set(region.leaves)
            leaves.update(region_leaves)
            region_nodes = set(region_leaves)
            if len(region.leaves) >= 2:
                started = time.perf_counter()
                costs = _region_costs(
                    chromosome,
                    region,
                    objective,
                    expected_profile=expected,
                    pair_mean_oe=(
                        None
                        if pair_cache is None
                        else pair_cache[(chromosome.name, region_index)]
                    ),
                    profile_dissimilarities=(
                        None
                        if profile_cache is None
                        else profile_cache[(chromosome.name, region_index)]
                    ),
                )
                decoded = knuth_cky_min(costs)
                elapsed = time.perf_counter() - started
                chromosome_seconds += elapsed
                chromosome_candidates += decoded.candidates
                defects = knuth_defects(costs, tolerance=1e-9)
                if defects.quadrangle or defects.inclusion:
                    raise AssertionError(
                        f"non-Monge objective {objective.name}: {defects}"
                    )
                if verify_full_max_leaves and len(region.leaves) <= verify_full_max_leaves:
                    full = full_cky_min(costs)
                    verified_regions += 1
                    all_verified_exact &= (
                        full.spans == decoded.spans
                        and bool(np.isclose(full.score, decoded.score))
                    )
                region_nodes.update(
                    _select_tree_parents(
                        region, decoded.spans, normalized, parent_selector
                    )
                )
            nodes.update(region_nodes)
            metric = _region_metric(
                chromosome,
                region,
                region_nodes,
                maximum_distance=maximum_distance,
                resolution=resolution,
                peak_tolerance_bp=peak_tolerance_bp,
            )
            metric.update({"chromosome": chromosome.name, "region": region_index})
            region_rows.append(metric)

        adjusted = tad_adjusted_r_squared(
            chromosome.contacts, sorted(nodes), maximum_distance=maximum_distance
        ).mean
        parents = nodes - leaves
        supported, evaluated = (
            dual_boundary_peak_support(
                sorted(parents),
                chromosome.peak_intervals,
                resolution=resolution,
                tolerance_bp=peak_tolerance_bp,
            )
            if len(chromosome.peak_intervals)
            else (0, 0)
        )
        chromosome_rows.append(
            {
                "chromosome": chromosome.name,
                "leaf_nodes": len(leaves),
                "parent_nodes": len(parents),
                "tad_adjusted_r2": adjusted,
                "dual_end_ctcf_supported": supported,
                "dual_end_ctcf_evaluated": evaluated,
                "knuth_seconds": chromosome_seconds,
                "knuth_candidates": chromosome_candidates,
            }
        )
        total_seconds += chromosome_seconds
        total_candidates += chromosome_candidates

    supported = sum(int(row["dual_end_ctcf_supported"]) for row in chromosome_rows)
    evaluated = sum(int(row["dual_end_ctcf_evaluated"]) for row in chromosome_rows)
    return {
        "objective": asdict(objective),
        "parent_selector": asdict(parent_selector),
        "mean_tad_adjusted_r2": float(
            np.mean([float(row["tad_adjusted_r2"]) for row in chromosome_rows])
        ),
        "dual_end_ctcf_supported": supported,
        "dual_end_ctcf_evaluated": evaluated,
        "dual_end_ctcf_rate": supported / evaluated if evaluated else None,
        "knuth_seconds": total_seconds,
        "knuth_candidates": total_candidates,
        "verified_regions": verified_regions,
        "all_verified_full_cky_exact": all_verified_exact if verified_regions else None,
        "chromosomes": chromosome_rows,
        "regions": region_rows,
    }


def evaluate_public(
    chromosomes: list[Chromosome],
    *,
    maximum_distance: int,
    resolution: int,
    peak_tolerance_bp: int,
) -> dict[str, object]:
    chromosome_rows = []
    region_rows = []
    for chromosome in chromosomes:
        nodes: set[Span] = set()
        leaves: set[Span] = set()
        for region_index, region in enumerate(chromosome.regions):
            region_nodes = set(region.public_nodes)
            nodes.update(region_nodes)
            leaves.update(region.leaves)
            metric = _region_metric(
                chromosome,
                region,
                region_nodes,
                maximum_distance=maximum_distance,
                resolution=resolution,
                peak_tolerance_bp=peak_tolerance_bp,
            )
            metric.update({"chromosome": chromosome.name, "region": region_index})
            region_rows.append(metric)
        adjusted = tad_adjusted_r_squared(
            chromosome.contacts, sorted(nodes), maximum_distance=maximum_distance
        ).mean
        parents = nodes - leaves
        supported, evaluated = (
            dual_boundary_peak_support(
                sorted(parents),
                chromosome.peak_intervals,
                resolution=resolution,
                tolerance_bp=peak_tolerance_bp,
            )
            if len(chromosome.peak_intervals)
            else (0, 0)
        )
        chromosome_rows.append(
            {
                "chromosome": chromosome.name,
                "leaf_nodes": len(leaves),
                "parent_nodes": len(parents),
                "tad_adjusted_r2": adjusted,
                "dual_end_ctcf_supported": supported,
                "dual_end_ctcf_evaluated": evaluated,
                "merge_seconds": sum(r.public_merge_seconds for r in chromosome.regions),
                "merge_candidates": sum(max(0, len(r.leaves) - 1) for r in chromosome.regions),
            }
        )
    supported = sum(int(row["dual_end_ctcf_supported"]) for row in chromosome_rows)
    evaluated = sum(int(row["dual_end_ctcf_evaluated"]) for row in chromosome_rows)
    return {
        "objective": {"name": "A_public_deepTAD"},
        "mean_tad_adjusted_r2": float(
            np.mean([float(row["tad_adjusted_r2"]) for row in chromosome_rows])
        ),
        "dual_end_ctcf_supported": supported,
        "dual_end_ctcf_evaluated": evaluated,
        "dual_end_ctcf_rate": supported / evaluated if evaluated else None,
        "chromosomes": chromosome_rows,
        "regions": region_rows,
    }


def _selection_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    objective = row["objective"]
    assert isinstance(objective, dict)
    return (
        float(row["mean_tad_adjusted_r2"]),
        float(row["dual_end_ctcf_rate"] or -1.0),
        -float(objective.get("pair_weight", 0.0)),
        -float(objective.get("probability_weight", 0.0)),
    )


def _select_candidate(
    rows: list[dict[str, object]], public: dict[str, object]
) -> tuple[dict[str, object], bool]:
    public_rate = public["dual_end_ctcf_rate"]
    if public_rate is None:
        return max(rows, key=_selection_key), True
    eligible = [
        row
        for row in rows
        if row["dual_end_ctcf_rate"] is not None
        and float(row["dual_end_ctcf_rate"]) >= float(public_rate)
    ]
    if eligible:
        return max(eligible, key=_selection_key), True
    return max(
        rows,
        key=lambda row: (
            float(row["dual_end_ctcf_rate"] or -1.0),
            float(row["mean_tad_adjusted_r2"]),
        ),
    ), False


def _long_strata(row: dict[str, object], thresholds: list[int]) -> list[dict[str, object]]:
    output = []
    regions = row["regions"]
    assert isinstance(regions, list)
    for threshold in thresholds:
        selected = [r for r in regions if int(r["atomic_tads"]) >= threshold]
        finite = [float(r["tad_adjusted_r2"]) for r in selected if np.isfinite(r["tad_adjusted_r2"])]
        supported = sum(int(r["dual_end_ctcf_supported"]) for r in selected)
        evaluated = sum(int(r["dual_end_ctcf_evaluated"]) for r in selected)
        output.append(
            {
                "minimum_atomic_tads": threshold,
                "regions": len(selected),
                "mean_region_tad_adjusted_r2": float(np.mean(finite)) if finite else None,
                "dual_end_ctcf_supported": supported,
                "dual_end_ctcf_evaluated": evaluated,
                "dual_end_ctcf_rate": supported / evaluated if evaluated else None,
            }
        )
    return output


def _markdown(result: dict[str, object]) -> str:
    selector = result["parent_selector"]
    evaluation_role = result["evaluation_role"]
    lines = [
        f"# DeepTAD fixed-leaf pair-potential {evaluation_role} evaluation",
        "",
        "All rows use the same filtered deepTAD boundaries and atomic TADs.",
        f"Parent cardinality rule: {result['cardinality_control']}.",
        (
            "Only the hierarchy objective changes. Hyperparameters are selected on these development chromosomes."
            if evaluation_role == "development"
            else "The hierarchy and parent-selection hyperparameters were frozen on separate development chromosomes."
        ),
        f"Tree parents are ranked by standardized contact contrast plus {selector['boundary_weight']:g} times "
        f"standardized {selector['boundary_reduction']} endpoint {selector['boundary_feature']} evidence "
        f"(missing endpoint value {selector['missing_endpoint_value']:g}).",
        "",
        "| Variant | pair transform | O/E weight | profile weight | path weight | probability weight | cosine weight | duration scale | TAD-adjR2 | dual-end CTCF | seconds | full=KY |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["selected"]:
        obj = row["objective"]
        duration = obj.get("duration_scale")
        ctcf = (
            "-"
            if row["dual_end_ctcf_rate"] is None
            else f"{row['dual_end_ctcf_supported']}/{row['dual_end_ctcf_evaluated']} "
            f"({100 * row['dual_end_ctcf_rate']:.1f}%)"
        )
        lines.append(
            f"| {obj['name']} | {obj.get('pair_transform', '-')} | {obj.get('pair_weight', 0):g} | {obj.get('profile_weight', 0):g} | {obj.get('separator_pair_weight', 0):g} | "
            f"{obj.get('probability_weight', 0):g} | {obj.get('cosine_weight', 0):g} | "
            f"{'-' if duration is None else f'{duration:g}'} | "
            f"{row['mean_tad_adjusted_r2']:.4f} | "
            f"{ctcf} | "
            f"{row.get('knuth_seconds', sum(c.get('merge_seconds', 0) for c in row['chromosomes'])):.4f} | "
            f"{row.get('all_verified_full_cky_exact', '-')} |"
        )
    lines.extend(["", "## Long-region strata", ""])
    for row in result["selected"]:
        lines.extend([
            f"### {row['objective']['name']}", "",
            "| minimum atomic TADs | regions | mean region TAD-adjR2 | dual-end CTCF |",
            "|---:|---:|---:|---:|",
        ])
        for item in row["long_strata"]:
            adj = "-" if item["mean_region_tad_adjusted_r2"] is None else f"{item['mean_region_tad_adjusted_r2']:.4f}"
            rate = "-" if item["dual_end_ctcf_rate"] is None else f"{100 * item['dual_end_ctcf_rate']:.1f}%"
            lines.append(
                f"| {item['minimum_atomic_tads']} | {item['regions']} | {adj} | "
                f"{item['dual_end_ctcf_supported']}/{item['dual_end_ctcf_evaluated']} ({rate}) |"
            )
        lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    peaks = _load_peaks(args.ctcf_peaks)
    chromosomes = _load_split(
        args.chromosomes,
        args.matrix_dir,
        args.prediction_dir,
        args.filtered_dir,
        args.minimum_region_bins,
        peaks,
        args.resolution,
        args.sample_prefix,
    )
    common = {
        "maximum_distance": args.maximum_distance,
        "resolution": args.resolution,
        "peak_tolerance_bp": args.peak_tolerance_bp,
    }
    parent_selector = ParentSelector(
        boundary_feature=args.parent_boundary_feature,
        boundary_reduction=args.parent_boundary_reduction,
        boundary_weight=args.parent_boundary_weight,
        missing_endpoint_value=args.parent_missing_endpoint_value,
        cardinality_mode=args.parent_cardinality_mode,
        parent_ratio=args.parent_ratio,
        maximum_parent_bins=args.maximum_parent_bins,
    )
    public = evaluate_public(chromosomes, **common)
    pair_cache: dict[tuple[str, int], np.ndarray] = {}
    profile_cache: dict[tuple[str, int], np.ndarray] | None = (
        {} if any(weight != 0 for weight in args.profile_weight_grid) else None
    )
    pair_feature_seconds = 0.0
    profile_feature_seconds = 0.0
    for chromosome in chromosomes:
        expected = distance_expected_profile(chromosome.contacts)
        for region_index, region in enumerate(chromosome.regions):
            if len(region.leaves) < 2:
                continue
            started = time.perf_counter()
            pair_cache[(chromosome.name, region_index)] = atomic_pair_mean_oe(
                chromosome.contacts,
                region.leaves,
                expected_profile=expected,
            )
            pair_feature_seconds += time.perf_counter() - started
            if profile_cache is not None:
                started = time.perf_counter()
                profile_cache[(chromosome.name, region_index)] = (
                    atomic_profile_dissimilarities(
                        chromosome.contacts, region.leaves
                    )
                )
                profile_feature_seconds += time.perf_counter() - started
    grids: dict[str, list[dict[str, object]]] = {}

    objectives: dict[str, list[Objective]] = {
        "B_separator_duration": [
            Objective("B_separator_duration", 0.0, alpha, 1.0, scale)
            for alpha in args.probability_weight_grid
            for scale in args.duration_scale_grid
        ],
        "C_pair": [
            Objective("C_pair", 1.0, 0.0, 0.0, None, transform)
            for transform in args.pair_transforms
        ],
        "D_pair_separator": [
            Objective("D_pair_separator", pair, alpha, 1.0, None, transform, profile, path, args.separator_pair_aggregation)
            for transform in args.pair_transforms
            for pair in args.pair_weight_grid
            for profile in args.profile_weight_grid
            for path in args.separator_pair_weight_grid
            for alpha in args.probability_weight_grid
        ],
        "E_pair_separator_duration": [
            Objective("E_pair_separator_duration", pair, alpha, 1.0, scale, transform, profile, path, args.separator_pair_aggregation)
            for transform in args.pair_transforms
            for pair in args.pair_weight_grid
            for profile in args.profile_weight_grid
            for path in args.separator_pair_weight_grid
            for alpha in args.probability_weight_grid
            for scale in args.duration_scale_grid
        ],
    }
    if args.evaluation_role == "heldout" and any(
        len(candidates) != 1 for candidates in objectives.values()
    ):
        raise ValueError(
            "heldout evaluation requires exactly one frozen configuration per family"
        )
    selected = [public]
    for family, candidates in objectives.items():
        rows = [
            evaluate_objective(
                chromosomes,
                objective,
                pair_cache=pair_cache,
                profile_cache=profile_cache,
                parent_selector=parent_selector,
                **common,
            )
            for objective in candidates
        ]
        grids[family] = rows
        best, constraint_met = _select_candidate(rows, public)
        objective = Objective(**best["objective"])
        selected_row = evaluate_objective(
            chromosomes,
            objective,
            verify_full_max_leaves=args.verify_full_max_leaves,
            pair_cache=pair_cache,
            profile_cache=profile_cache,
            parent_selector=parent_selector,
            **common,
        )
        selected_row["ctcf_noninferiority_constraint_met"] = constraint_met
        selected.append(selected_row)
    for row in selected:
        row["long_strata"] = _long_strata(row, args.long_thresholds)

    result = {
        "schema_version": 1,
        "evaluation_role": args.evaluation_role,
        "evaluation_chromosomes": [f"chr{x}" for x in args.chromosomes],
        "held_out_data_used": args.evaluation_role == "heldout",
        "sample_prefix": args.sample_prefix,
        "resolution": args.resolution,
        "cardinality_control": (
            "match public deepTAD parent count within every gap-free region"
            if parent_selector.cardinality_mode == "match_public_region"
            else f"select round({parent_selector.parent_ratio:g} * internal candidates) within every gap-free region"
        ),
        "selection_rule": (
            "one frozen configuration per family; no held-out selection"
            if args.evaluation_role == "heldout"
            else "maximize mean chromosome TAD-adjR2 subject to dual-end CTCF rate being at least public deepTAD; "
            "if infeasible, maximize dual-end CTCF then TAD-adjR2"
        ),
        "pair_potential": "nonnegative transform of chromosome-distance-normalized mean contact",
        "parent_selector": asdict(parent_selector),
        "pair_feature_seconds": pair_feature_seconds,
        "profile_feature_seconds": profile_feature_seconds,
        "selected": selected,
        "development_grid": grids,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "RESULTS.md").write_text(_markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--filtered-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ctcf-peaks", type=Path)
    parser.add_argument("--sample-prefix", default="HIC002")
    parser.add_argument(
        "--evaluation-role", choices=["development", "heldout"], default="development"
    )
    parser.add_argument("--resolution", type=int, default=25_000)
    parser.add_argument("--chromosomes", nargs="+", default=["13", "14", "15", "16", "17", "18", "19"])
    parser.add_argument("--minimum-region-bins", type=int, default=4)
    parser.add_argument("--maximum-distance", type=int, default=60)
    parser.add_argument("--peak-tolerance-bp", type=int, default=25_000)
    parser.add_argument("--pair-weight-grid", nargs="+", type=float, default=[0.1, 0.3, 1.0, 3.0, 10.0])
    parser.add_argument("--pair-transforms", nargs="+", choices=["hinge", "reciprocal", "exponential"], default=["hinge", "reciprocal", "exponential"])
    parser.add_argument("--profile-weight-grid", nargs="+", type=float, default=[0.0, 0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--separator-pair-weight-grid", nargs="+", type=float, default=[0.0, 0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--separator-pair-aggregation", choices=["maximum", "sum"], default="maximum")
    parser.add_argument(
        "--parent-boundary-feature",
        choices=["none", "probability", "cosine_dissimilarity", "combined_separator"],
        default="none",
    )
    parser.add_argument(
        "--parent-boundary-reduction", choices=["minimum", "mean"], default="minimum"
    )
    parser.add_argument("--parent-boundary-weight", type=float, default=0.0)
    parser.add_argument("--parent-missing-endpoint-value", type=float, default=0.0)
    parser.add_argument(
        "--parent-cardinality-mode",
        choices=["match_public_region", "fixed_ratio_region"],
        default="match_public_region",
    )
    parser.add_argument("--parent-ratio", type=float)
    parser.add_argument("--maximum-parent-bins", type=int)
    parser.add_argument("--probability-weight-grid", nargs="+", type=float, default=[0.0, 0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--duration-scale-grid", nargs="+", type=float, default=[10.0, 20.0, 40.0, 80.0, 160.0])
    parser.add_argument("--long-thresholds", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--verify-full-max-leaves", type=int, default=128)
    args = parser.parse_args()
    if args.parent_boundary_weight < 0:
        parser.error("--parent-boundary-weight must be nonnegative")
    if args.parent_cardinality_mode == "fixed_ratio_region" and (
        args.parent_ratio is None or not 0 <= args.parent_ratio <= 1
    ):
        parser.error("--parent-ratio in [0, 1] is required for fixed_ratio_region")
    result = run(args)
    print(json.dumps([row["objective"] for row in result["selected"]], sort_keys=True))


if __name__ == "__main__":
    main()
