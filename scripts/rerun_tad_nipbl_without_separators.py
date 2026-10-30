"""Repeat both fixed-coordinate NIPBL folds for the separator-free objective."""
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import replace
import json
import statistics
import time

from rerun_tad_without_separators import (
    ROOT, OUT, PREVIOUS_OBJECTIVE, SELECTOR, np, objective, read, write,
    _load_split, _region_costs, _select_tree_parents, knuth_cky_min,
    DistanceNormalizedContacts, distance_expected_profile,
    atomic_pair_mean_oe, clustertad_quality, outermost_intervals,
)
from faster_cky.tad_matrix_io import load_dense_matrix


def run(chromosome, fold):
    target = OUT / "nipbl" / f"chr{chromosome}_fold{fold}.json"
    if read(target).get("complete"):
        return
    started = time.perf_counter()
    samples = dict(train="GSM7325028" if fold else "GSM7325027",
                   control="GSM7325027" if fold else "GSM7325028",
                   nipbl="GSM7325029" if fold else "GSM7325030")
    matrices = ROOT / "data/processed/501mel_nipbl_10kb"
    original = ROOT / "artifacts/tad_501mel_nipbl"
    sample = _load_split([str(chromosome)], matrices,
                         original / f'boundary_predictions_{samples["train"]}',
                         original / f'filtered_boundaries_{samples["train"]}',
                         4, {}, resolution=10000, sample_prefix=samples["train"])[0]
    # Preserve the original four-map common-bin convention, including a possible
    # single trailing-bin difference between contact maps.
    sizes = []
    for accession in ("GSM7325027", "GSM7325028", "GSM7325029", "GSM7325030"):
        matrix, _ = load_dense_matrix(matrices / f"{accession}_10k_KR.chr{chromosome}")
        sizes.append(len(matrix))
        del matrix
    sample = replace(sample, contacts=sample.contacts[:min(sizes), :min(sizes)])
    normalized = DistanceNormalizedContacts.from_contacts(sample.contacts, maximum_distance=60)
    expected = distance_expected_profile(sample.contacts)
    selector = replace(SELECTOR, maximum_parent_bins=None)
    proposed, previous = objective(10000), replace(PREVIOUS_OBJECTIVE, duration_scale=400.0)
    parents, previous_parents = set(), set()
    for region in sample.regions:
        if len(region.leaves) < 2:
            continue
        pairs = atomic_pair_mean_oe(sample.contacts, region.leaves,
                                   expected_profile=expected, validate_contacts=False)
        for setting, destination in ((proposed, parents), (previous, previous_parents)):
            costs = _region_costs(sample, region, setting, expected_profile=expected, pair_mean_oe=pairs)
            tree = knuth_cky_min(costs)
            destination.update(_select_tree_parents(region, tree.spans, normalized, selector))
    suffix = "_swapped" if fold else ""
    frozen = read(original / f"structural_pilot_chr{chromosome}{suffix}.json")
    reference = {(p["start_bin"], p["end_bin"]) for r in frozen["regions"]
                 for p in r["methods_by_source"]["control_train"]["KY-W"]["parent_rows"]}
    if reference != previous_parents:
        raise ValueError(f"Original NIPBL parents could not be reproduced: chr{chromosome}, fold {fold}")
    outer = outermost_intervals(sorted(parents))
    scores = {}
    for label in ("control", "nipbl"):
        matrix, _ = load_dense_matrix(matrices / f'{samples[label]}_10k_KR.chr{chromosome}')
        scores[label] = clustertad_quality(matrix, outer)
        del matrix
    paired = []
    for control, nipbl in zip(scores["control"], scores["nipbl"]):
        if (control["start_bin"], control["end_bin"]) != (nipbl["start_bin"], nipbl["end_bin"]):
            raise ValueError("NIPBL evaluation coordinates differ")
        a, b = control["quality"], nipbl["quality"]
        paired.append(dict(start_bin=control["start_bin"], end_bin=control["end_bin"],
                           control=a, nipbl=b, delta=b-a if a is not None and b is not None else None))
    write(target, dict(complete=True, chromosome=f"chr{chromosome}", fold=fold, samples=samples,
                       parent_coordinates=sorted(parents), all_selected_parents=len(parents),
                       outermost_parents=len(outer), previous_parents_reproduced=True,
                       paired=paired, seconds=time.perf_counter()-started))
    print(f"chr{chromosome} fold{fold}: {len(parents)} parents, {len(outer)} outermost, "
          f"{time.perf_counter()-started:.1f}s", flush=True)


def summarize():
    rows, missing = [], []
    for chromosome in range(1,23):
        for fold in (0,1):
            data = read(OUT / "nipbl" / f"chr{chromosome}_fold{fold}.json")
            if not data.get("complete"):
                missing.append([chromosome,fold])
                continue
            rows.extend(dict(chromosome=f"chr{chromosome}", fold=fold, **r) for r in data["paired"])
    def mean_rows(selected):
        valid = [r for r in selected if r["delta"] is not None]
        return dict(parents=len(selected), valid_pairs=len(valid),
                    **{k: statistics.mean(r[k] for r in valid) if valid else None
                       for k in ("control", "nipbl", "delta")})
    result = dict(complete=not missing, missing=missing, rows=rows, **mean_rows(rows),
                  folds=[dict(fold=f, **mean_rows([r for r in rows if r["fold"]==f])) for f in (0,1)])
    write(OUT / "nipbl_summary.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}), flush=True)


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromosomes", nargs="+", type=int, default=list(range(1,23)))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args=parser.parse_args()
    OUT = args.output_dir.resolve()
    if any(not 1 <= n <= 22 for n in args.chromosomes):
        parser.error("Expected autosomes 1 through 22")
    for chromosome in args.chromosomes:
        for fold in (0,1):
            run(chromosome, fold)
    summarize()


if __name__=="__main__":
    main()
