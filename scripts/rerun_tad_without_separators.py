"""Re-evaluate the separator-free objective without overwriting previous runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

import numpy as np
from recompute_published_tad_metrics import CONDITIONS, PEAKS
from faster_cky.decoder import full_cky_min, greedy_adjacent_merge_min, knuth_cky_min
from faster_cky.deeptad_evaluation import DistanceNormalizedContacts, tad_adjusted_r_squared
from faster_cky.deeptad_monge import atomic_pair_mean_oe, distance_expected_profile
from faster_cky.deeptad_pair_ablation import _region_costs, _select_tree_parents
from faster_cky.deeptad_validation_experiment import _load_peaks, _load_split
from faster_cky.tad_configuration import OBJECTIVE as PREVIOUS_OBJECTIVE, SELECTOR
from faster_cky.tad_published_metrics import clustertad_quality, long_ctcf_support, outermost_intervals

BASE = ROOT / "artifacts/tad_all_chromosomes_20261022"
OUT = ROOT / "artifacts/tad_no_separator_20261029"
ORDER = ("gm12878_25kb", "gm12878_10kb", "k562_25kb", "imr90_25kb",
         "gm12878_5kb", "k562_5kb", "imr90_5kb")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def objective(resolution):
    return replace(PREVIOUS_OBJECTIVE, name="no_separator", duration_scale=4000000 / resolution,
                   probability_weight=0.0, cosine_weight=0.0,
                   direct_separator_weight=0.0, separator_pair_weight=0.0)


def adjusted_r2(matrix, domains, maximum_distance):
    """The existing metric, aggregating group means with bincount.

    Domain assignment, degrees of freedom, distances and averaging are identical
    to tad_adjusted_r_squared. The benchmark checks old and new objective outputs
    against the previously computed chromosome-level reference scores.
    """
    unique = sorted(set(domains), key=lambda ab: (ab[1] - ab[0], ab[0], ab[1]), reverse=True)
    if any(not 0 <= a < b <= len(matrix) for a, b in unique):
        raise ValueError("Invalid domain coordinates")
    scores = []
    for distance in range(1, min(maximum_distance, len(matrix) - 1) + 1):
        observed = np.asarray(np.diagonal(matrix, offset=distance), dtype=np.float64)
        if not np.all(np.isfinite(observed)):
            raise ValueError("Nonfinite observed contacts")
        labels = np.zeros(len(observed), dtype=np.int64)
        eligible = 0
        for label, (a, b) in enumerate(unique, 1):
            if b - a > distance:
                labels[a:b-distance] = label
                eligible += 1
        counts = np.bincount(labels)
        sums = np.bincount(labels, weights=observed, minlength=len(counts))
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        total = float(np.sum((observed - np.mean(observed)) ** 2))
        df = len(observed) - eligible - 1
        if df > 0 and total > 0:
            residual = float(np.sum((observed - means[labels]) ** 2))
            scores.append(1 - (residual / df) / (total / (len(observed) - 1)))
    if not scores:
        raise ValueError("No eligible distances for adjusted R2")
    return statistics.mean(scores)


def evaluate(matrix, domains, leaves, peaks, resolution, audit_reference=False):
    parents = domains - leaves
    r2 = adjusted_r2(matrix, domains, 1500000 // resolution)
    if audit_reference:
        reference = tad_adjusted_r_squared(matrix, sorted(domains), maximum_distance=1500000 // resolution).mean
        if not math.isclose(r2, reference, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Metric implementation mismatch: {r2} vs {reference}")
    quality_rows = clustertad_quality(matrix, outermost_intervals(sorted(domains)))
    quality = [r["quality"] for r in quality_rows if r["quality"] is not None]
    support = long_ctcf_support([(a*resolution, b*resolution) for a, b in parents], peaks)
    return dict(domains=len(domains), parents=len(parents), r2=r2,
                quality=statistics.mean(quality) if quality else None,
                ctcf_percent=100*support["rate"] if support["rate"] is not None else None,
                ctcf=support, outermost_domains=len(quality_rows), quality_scored_domains=len(quality))


def benchmark(condition, chromosome, audit_reference=False):
    started = time.perf_counter()
    prefix, resolution = CONDITIONS[condition]
    old = BASE / condition
    peaks = _load_peaks(ROOT / PEAKS[condition])
    sample = _load_split([str(chromosome)], old / "matrices", old / "predictions", old / "filtered",
                         4, peaks, resolution=resolution, sample_prefix=prefix)[0]
    expected = distance_expected_profile(sample.contacts)
    normalized = DistanceNormalizedContacts.from_contacts(sample.contacts, maximum_distance=1500000 // resolution)
    proposed = objective(resolution)
    variants = {
        "Ours": proposed,
        "No length": replace(proposed, name="no_separator_no_length", duration_scale=None),
        "Length only": replace(proposed, name="length_only", pair_weight=0.0),
        "With separators": replace(PREVIOUS_OBJECTIVE, duration_scale=4000000 / resolution),
    }
    selector = replace(SELECTOR, maximum_parent_bins=3000000 // resolution)
    matched = replace(selector, cardinality_mode="match_public_region")
    leaves = set().union(*(set(region.leaves) for region in sample.regions))
    nodes = {name: set(leaves) for name in [*variants, "Greedy", "Matched count"]}
    trees = []
    for index, region in enumerate(sample.regions):
        if len(region.leaves) < 2:
            continue
        pairs = atomic_pair_mean_oe(sample.contacts, region.leaves, expected_profile=expected, validate_contacts=False)
        costs = _region_costs(sample, region, proposed, expected_profile=expected, pair_mean_oe=pairs)
        exact = knuth_cky_min(costs)
        standard = full_cky_min(costs)
        greedy = greedy_adjacent_merge_min(costs)
        score_equal = bool(np.isclose(exact.score, standard.score, rtol=1e-10, atol=1e-8))
        tree_equal = exact.spans == standard.spans
        if not score_equal or not tree_equal or exact.score > greedy.score + 1e-8:
            raise ValueError(f"Optimization audit failed: {condition} chr{chromosome} region {index}")
        trees.append(dict(region=index, leaves=len(region.leaves), exact_score=exact.score,
                          greedy_score=greedy.score, standard_score=standard.score,
                          score_equal=score_equal, tree_equal=tree_equal,
                          spans=sorted(exact.spans)))
        nodes["Ours"].update(_select_tree_parents(region, exact.spans, normalized, selector))
        nodes["Matched count"].update(_select_tree_parents(region, exact.spans, normalized, matched))
        nodes["Greedy"].update(_select_tree_parents(region, greedy.spans, normalized, selector))
        for name, variant in variants.items():
            if name == "Ours":
                continue
            changed = _region_costs(sample, region, variant, expected_profile=expected, pair_mean_oe=pairs)
            tree = knuth_cky_min(changed)
            nodes[name].update(_select_tree_parents(region, tree.spans, normalized, selector))
    construction_seconds = time.perf_counter() - started
    metrics = {name: evaluate(sample.contacts, spans, leaves, sample.peak_intervals, resolution, audit_reference)
               for name, spans in nodes.items()}
    previous = read(old / "evaluation" / f"chr{chromosome}" / "results.json")
    checks = {}
    for name, old_name in [("Ours", "no_boundary"), ("With separators", "full"), ("Length only", "duration_only")]:
        prior = previous["score_ablation"][old_name]["mean_tad_adjusted_r2"]
        checks[name] = dict(previous_r2=prior, recalculated_r2=metrics[name]["r2"],
                            equal=math.isclose(prior, metrics[name]["r2"], rel_tol=0, abs_tol=1e-12))
        if not checks[name]["equal"]:
            raise ValueError(f"Frozen-ablation reproduction failed: {condition} chr{chromosome} {name}: {checks[name]}")
    result = dict(complete=True, condition=condition, chromosome=f"chr{chromosome}",
                  objective=asdict(proposed), selector=asdict(selector),
                  matrix_shape=list(sample.contacts.shape), leaves=sorted(leaves),
                  nodes={name: sorted(spans) for name, spans in nodes.items()},
                  scores=metrics, regions=trees, validation=checks,
                  construction_and_audit_seconds=construction_seconds,
                  total_seconds=time.perf_counter()-started,
                  metric_reference_audit=audit_reference)
    write(OUT / "benchmark" / condition / f"chr{chromosome}.json", result)
    print(condition, f"chr{chromosome}", metrics["Ours"], f'{result["total_seconds"]:.1f}s', flush=True)


def campaign(conditions, chromosomes):
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for condition in conditions:
            for chromosome in chromosomes:
                target = OUT / "benchmark" / condition / f"chr{chromosome}.json"
                if read(target).get("complete"):
                    continue
                job = f"{condition}.chr{chromosome}"
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "benchmark",
                           "--conditions", condition, "--chromosomes", str(chromosome),
                           "--output-dir", str(OUT)]
                record = dict(job=job, status="running", started=time.time(), command=command,
                              timeout_seconds=1800, pid=os.getpid())
                write(OUT / "status.json", record)
                log = OUT / "logs" / f"{job}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w") as stream:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    try:
                        code = process.wait(timeout=1800)
                        state = "completed" if code == 0 and read(target).get("complete") else "runtime_error"
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        code, state = None, "timeout"
                record.update(status=state, returncode=code, finished=time.time())
                write(OUT / "jobs" / f"{job}.json", record)
                print(job, state, flush=True)
        write(OUT / "status.json", dict(status="benchmark_queue_finished", finished=time.time()))


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["benchmark", "campaign"])
    parser.add_argument("--conditions", nargs="+", choices=ORDER, default=list(ORDER))
    parser.add_argument("--chromosomes", nargs="+", type=int, default=list(range(1,23)))
    parser.add_argument("--audit-reference", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if any(not 1 <= n <= 22 for n in args.chromosomes):
        parser.error("Expected autosomes 1 through 22")
    if args.phase == "campaign":
        campaign(args.conditions, args.chromosomes)
    else:
        for condition in args.conditions:
            for chromosome in args.chromosomes:
                benchmark(condition, chromosome, args.audit_reference)


if __name__ == "__main__":
    main()
