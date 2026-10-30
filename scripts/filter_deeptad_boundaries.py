"""Filter pinned deepTAD boundary predictions with the public statistical rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from faster_cky.deeptad_filter import filter_deeptad_boundaries
from faster_cky.tad_matrix_io import load_dense_matrix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chromosomes", nargs="+", default=["20", "21", "22"])
    parser.add_argument("--resolution", type=int, default=25_000)
    parser.add_argument("--sample-prefix", default="HIC002")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--pvalue-cutoff", type=float, default=0.05)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    display_resolution = args.resolution // 1000
    for chromosome in args.chromosomes:
        chromosome_started = time.perf_counter()
        matrix_stem = args.matrix_dir / f"{args.sample_prefix}_{display_resolution}k_KR.chr{chromosome}"
        prediction_path = (
            args.prediction_dir / f"{args.sample_prefix}_{display_resolution}k_KR.chr{chromosome}.npz"
        )
        started = time.perf_counter()
        contacts, matrix_path = load_dense_matrix(
            matrix_stem, dtype=np.float64, mmap=True
        )
        matrix_load_seconds = time.perf_counter() - started
        with np.load(prediction_path) as archive:
            probabilities = archive["boundary_probability"]
        candidates = np.flatnonzero(probabilities > 0.5)
        started = time.perf_counter()
        result = filter_deeptad_boundaries(
            contacts,
            candidates,
            window_size=args.window_size,
            pvalue_cutoff=args.pvalue_cutoff,
        )
        filter_seconds = time.perf_counter() - started
        output_path = args.output_dir / f"{args.sample_prefix}_{display_resolution}k_KR.chr{chromosome}.npz"
        np.savez_compressed(
            output_path,
            candidate_index=result.candidate_indices,
            candidate_probability=probabilities[result.candidate_indices],
            candidate_pvalue=result.candidate_pvalues,
            filtered_index=result.filtered_indices,
            filtered_probability=probabilities[result.filtered_indices],
            gap_index=result.gap_indices,
        )
        tsv_path = args.output_dir / f"{args.sample_prefix}_{display_resolution}k_KR.chr{chromosome}.tsv"
        lines = ["bin_index_0based\tbin_index_1based\tprobability\tpvalue"]
        pvalue_by_index = dict(
            zip(result.candidate_indices.tolist(), result.candidate_pvalues.tolist())
        )
        for index in result.filtered_indices:
            lines.append(
                f"{index}\t{index + 1}\t{probabilities[index]:.9g}\t"
                f"{pvalue_by_index[int(index)]:.9g}"
            )
        tsv_path.write_text("\n".join(lines) + "\n")
        records.append(
            {
                "chromosome": f"chr{chromosome}",
                "bins": len(contacts),
                "candidates": len(result.candidate_indices),
                "filtered_boundaries": len(result.filtered_indices),
                "gap_bins": len(result.gap_indices),
                "matrix_load_seconds": matrix_load_seconds,
                "filter_seconds": filter_seconds,
                "elapsed_seconds": time.perf_counter() - chromosome_started,
                "matrix_sha256": _sha256(matrix_path),
                "prediction_sha256": _sha256(prediction_path),
                "output_sha256": _sha256(output_path),
                "tsv_sha256": _sha256(tsv_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "method": "Python port of pinned deepTAD assemble_filterTAD.R",
        "window_size": args.window_size,
        "pvalue_cutoff": args.pvalue_cutoff,
        "resolution": args.resolution,
        "sample_prefix": args.sample_prefix,
        "indexing": "zero-based indices retained in NPZ; TSV includes both conventions",
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
