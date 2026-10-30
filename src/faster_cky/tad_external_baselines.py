"""Run and evaluate external hierarchical TAD callers on dense Hi-C matrices.

All parser outputs use zero-based half-open bin intervals.  The evaluator
reports metrics for every nontrivial domain and, separately, for parent
domains that strictly contain another reported domain.  The parent-only view
is the comparable quantity for deepTAD hierarchy experiments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .deeptad_evaluation import (
    DistanceNormalizedContacts,
    dual_boundary_peak_support,
    tad_adjusted_r_squared,
)
from .deeptad_validation_experiment import _load_peaks


Span = tuple[int, int]


def _canonicalize(spans: list[Span], matrix_size: int) -> list[Span]:
    return sorted(
        {
            (int(left), int(right))
            for left, right in spans
            if 0 <= int(left) < int(right) <= matrix_size
            and int(right) - int(left) >= 2
        }
    )


def parent_domains(spans: list[Span]) -> list[Span]:
    """Return domains that strictly contain another reported domain."""
    ordered = sorted(set(spans))
    return [
        outer
        for outer in ordered
        if any(
            outer != inner
            and outer[0] <= inner[0]
            and inner[1] <= outer[1]
            for inner in ordered
        )
    ]


def parse_ontad(path: Path, matrix_size: int) -> list[Span]:
    """Parse OnTAD's one-based inclusive positions, excluding level-0 roots."""
    spans: list[Span] = []
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 3:
                continue
            left, right, level = map(int, fields[:3])
            if level <= 0:
                continue
            spans.append((left - 1, right))
    return _canonicalize(spans, matrix_size)


def parse_jontads(path: Path, matrix_size: int) -> list[Span]:
    """Parse JOnTADS' one-based inclusive start and end columns."""
    spans: list[Span] = []
    with path.open() as handle:
        for line in handle:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 2:
                continue
            spans.append((int(float(fields[0])) - 1, int(float(fields[1]))))
    return _canonicalize(spans, matrix_size)


def parse_supertad(path: Path, matrix_size: int) -> list[Span]:
    """Parse SuperTAD's one-based inclusive boundary-bin columns."""
    spans: list[Span] = []
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 6:
                continue
            spans.append((int(fields[1]) - 1, int(fields[5])))
    return _canonicalize(spans, matrix_size)


def parse_dedoc2(path: Path, matrix_size: int) -> list[Span]:
    """Parse deDoc2 partitions, whose node identifiers are one-based."""
    spans: list[Span] = []
    with path.open() as handle:
        for line in handle:
            nodes = [int(field) for field in line.split()]
            if len(nodes) >= 2:
                spans.append((min(nodes) - 1, max(nodes)))
    return _canonicalize(spans, matrix_size)


def evaluate_domains(
    contacts: np.ndarray,
    spans: list[Span],
    *,
    peaks: np.ndarray,
    resolution: int,
    maximum_distance: int,
    peak_tolerance_bp: int,
) -> dict[str, object]:
    spans = _canonicalize(spans, len(contacts))
    parents = parent_domains(spans)
    normalized = DistanceNormalizedContacts.from_contacts(
        contacts, maximum_distance=maximum_distance
    )

    def view(selected: list[Span]) -> dict[str, object]:
        if not selected:
            return {
                "domains": 0,
                "mean_tad_adjusted_r2": None,
                "mean_contact_contrast": None,
                "dual_end_ctcf_supported": 0,
                "dual_end_ctcf_evaluated": 0,
                "dual_end_ctcf_rate": None,
            }
        adjusted = tad_adjusted_r_squared(
            contacts, selected, maximum_distance=maximum_distance
        ).mean
        contrasts = np.asarray(
            [normalized.structural_contrast(left, right) for left, right in selected]
        )
        finite = contrasts[np.isfinite(contrasts)]
        supported, evaluated = (
            dual_boundary_peak_support(
                selected,
                peaks,
                resolution=resolution,
                tolerance_bp=peak_tolerance_bp,
            )
            if len(peaks)
            else (0, 0)
        )
        return {
            "domains": len(selected),
            "mean_tad_adjusted_r2": float(adjusted),
            "mean_contact_contrast": float(np.mean(finite)) if len(finite) else None,
            "dual_end_ctcf_supported": supported,
            "dual_end_ctcf_evaluated": evaluated,
            "dual_end_ctcf_rate": supported / evaluated if evaluated else None,
        }

    return {"all_domains": view(spans), "parent_domains": view(parents)}


def _run_logged(
    command: list[str], log_path: Path, timeout_seconds: float
) -> dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w") as log:
        try:
            environment = os.environ.copy()
            matplotlib_cache = Path(".cache/matplotlib-jontads").resolve()
            matplotlib_cache.mkdir(parents=True, exist_ok=True)
            environment["MPLCONFIGDIR"] = str(matplotlib_cache)
            from faster_cky.process_control import run_with_timeout
            completed = run_with_timeout(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=timeout_seconds,
                check=False,
            )
            status = "completed" if completed.returncode == 0 else "failed"
            return {
                "status": status,
                "returncode": completed.returncode,
                "wall_seconds": time.perf_counter() - started,
                "command": command,
                "log": str(log_path),
                "timeout_seconds": timeout_seconds,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "returncode": None,
                "wall_seconds": time.perf_counter() - started,
                "command": command,
                "log": str(log_path),
                "timeout_seconds": timeout_seconds,
            }


def _method_command(
    method: str,
    matrix: Path,
    output_prefix: Path,
    chromosome: str,
    resolution: int,
    args: argparse.Namespace,
) -> tuple[list[str], Path]:
    if method == "ontad":
        return (
            [
                str(args.ontad_binary),
                str(matrix),
                "-penalty", "0.1",
                "-maxsz", str(args.maximum_tad_bp // resolution),
                "-minsz", "3",
                "-o", str(output_prefix),
            ],
            Path(str(output_prefix) + ".tad"),
        )
    if method == "dedoc2":
        return (
            [
                "java", "-jar", str(args.dedoc2_jar),
                "-inputfile", str(matrix),
                "-binsize", str(resolution // 1000),
                "-outputfile", str(output_prefix),
                "-maxTLDsize", str(args.maximum_tad_bp // 1_000_000),
            ],
            Path(str(output_prefix) + ".TAD"),
        )
    if method == "supertad-fast":
        return (
            [
                str(getattr(args, "supertad_fast_binary", args.supertad_binary)), "multi", str(matrix),
                "-h", "2", "--fast", "--step", str(args.supertad_steps),
                "--window", "5", "--chrom1", chromosome,
                "-r", str(resolution),
                "--maxbin", str(args.maximum_tad_bp // resolution),
                "-w", str(output_prefix.parent),
            ],
            output_prefix.parent / f"{matrix.name}.multifast.tsv",
        )
    if method == "supertad-binary":
        return (
            [
                str(args.supertad_binary), "binary", str(matrix),
                "--chrom1", chromosome,
                "-r", str(resolution),
                "--maxbin", str(args.maximum_tad_bp // resolution),
                "-w", str(output_prefix.parent),
            ],
            output_prefix.parent / f"{matrix.name}.binary.filter.tsv",
        )
    if method == "jontads":
        return (
            [
                str(args.jontads_python), str(args.jontads_script),
                "-F", str(matrix), "-O", str(output_prefix) + ".tad",
                "-MAXSZ", str(args.maximum_tad_bp // resolution),
                "-MINSZ", "7", "--seed", "0",
            ],
            Path(str(output_prefix) + ".tad"),
        )
    raise ValueError(f"unsupported runnable method: {method}")


def run(args: argparse.Namespace) -> dict[str, object]:
    peaks_by_chromosome = _load_peaks(args.ctcf_peaks)
    rows: list[dict[str, object]] = []
    display_resolution = args.resolution // 1000
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for chromosome_number in args.chromosomes:
        chromosome = f"chr{chromosome_number}"
        stem = f"{args.sample_prefix}_{display_resolution}k_KR.{chromosome}"
        matrix_path = args.matrix_dir / stem
        contacts = np.loadtxt(matrix_path, dtype=np.float64)
        peaks = peaks_by_chromosome.get(
            chromosome, np.empty((0, 2), dtype=np.int64)
        )
        for method in args.methods:
            method_dir = args.output_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            output_prefix = method_dir / stem
            command, output_path = _method_command(
                method,
                matrix_path,
                output_prefix,
                chromosome,
                args.resolution,
                args,
            )
            timing = _run_logged(
                command, Path(str(output_prefix) + ".log"), args.timeout_seconds
            )
            row: dict[str, object] = {
                "method": method,
                "chromosome": chromosome,
                "matrix_bins": len(contacts),
                "run": timing,
                "output": str(output_path),
            }
            if timing["status"] == "completed" and output_path.exists():
                if method == "ontad":
                    spans = parse_ontad(output_path, len(contacts))
                elif method == "dedoc2":
                    spans = parse_dedoc2(output_path, len(contacts))
                    low_path = Path(str(output_prefix) + ".window.TAD")
                    if low_path.exists():
                        spans = _canonicalize(
                            spans + parse_dedoc2(low_path, len(contacts)), len(contacts)
                        )
                elif method in {"supertad-fast", "supertad-binary"}:
                    spans = parse_supertad(output_path, len(contacts))
                else:
                    spans = parse_jontads(output_path, len(contacts))
                row["metrics"] = evaluate_domains(
                    contacts,
                    spans,
                    peaks=peaks,
                    resolution=args.resolution,
                    maximum_distance=args.maximum_distance,
                    peak_tolerance_bp=args.peak_tolerance_bp,
                )
            rows.append(row)
    result = {
        "schema_version": 1,
        "sample_prefix": args.sample_prefix,
        "resolution": args.resolution,
        "chromosomes": [f"chr{x}" for x in args.chromosomes],
        "maximum_tad_bp": args.maximum_tad_bp,
        "rows": rows,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-prefix", default="HIC002")
    parser.add_argument("--resolution", type=int, default=25_000)
    parser.add_argument("--chromosomes", nargs="+", default=["20", "21", "22"])
    parser.add_argument(
        "--methods", nargs="+", choices=["ontad", "dedoc2", "supertad-fast", "supertad-binary", "jontads"],
        default=["ontad", "dedoc2"],
    )
    parser.add_argument("--ctcf-peaks", type=Path)
    parser.add_argument("--maximum-distance", type=int, default=60)
    parser.add_argument("--peak-tolerance-bp", type=int, default=25_000)
    parser.add_argument("--maximum-tad-bp", type=int, default=10_000_000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--supertad-steps", type=int, default=20)
    parser.add_argument(
        "--ontad-binary", type=Path,
        default=Path("third_party/OnTAD/src/OnTAD"),
    )
    parser.add_argument(
        "--dedoc2-jar", type=Path, default=Path("third_party/deDoc2/deDoc2.jar")
    )
    parser.add_argument(
        "--supertad-binary", type=Path,
        default=Path("third_party/SuperTAD/build/SuperTAD"),
    )
    parser.add_argument(
        "--jontads-python", type=Path,
        default=Path("third_party/venvs/jontads/bin/python"),
    )
    parser.add_argument(
        "--jontads-script", type=Path,
        default=Path("third_party/JOnTADS/JOnTADS/JOnTADS.py"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
