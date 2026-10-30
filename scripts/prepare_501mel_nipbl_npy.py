"""Convert local .hic KR sparse dumps to memory-mapped float32 matrices.

Avoids the several-gigabyte dense text files emitted by Juicer for large
10 kb chromosomes. Coordinates remain zero-based 10 kb bins; nonfinite
values in the public dump are treated exactly as deepTAD's remove_nan.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/501mel_nipbl_hic"
OUTPUT = ROOT / "data/processed/501mel_nipbl_10kb"
ARTIFACT = ROOT / "artifacts/tad_501mel_nipbl"
SAMPLES = {
    "GSM7325027": "GSM7325027_501mel_Luc_set1_map30_10kb_normcount.hic",
    "GSM7325028": "GSM7325028_501mel_Luc_set2_map30_10kb_normcount.hic",
    "GSM7325029": "GSM7325029_501mel_NIPBL2_set1_map30_10kb_normcount.hic",
    "GSM7325030": "GSM7325030_501mel_NIPBL2_set2_map30_10kb_normcount.hic",
}
RESOLUTION = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(sample: str, chromosome: str, *, validate_text: bool = False) -> dict:
    if sample not in SAMPLES:
        raise ValueError(f"unknown sample {sample}")
    hic = RAW / SAMPLES[sample]
    destination = OUTPUT / f"{sample}_10k_KR.chr{chromosome}.npy"
    if destination.exists() and not validate_text:
        matrix = np.load(destination, mmap_mode="r")
        return {"sample": sample, "chromosome": chromosome, "bins": len(matrix),
                "output": str(destination), "status": "present"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="501mel_sparse_") as temporary:
        sparse = Path(temporary) / "observed.tsv"
        subprocess.run([
            "java", "-jar", str(ROOT / "third_party/deepTAD/prepare_data/juicer_tools.jar"),
            "dump", "observed", "KR", str(hic), f"chr{chromosome}", f"chr{chromosome}",
            "BP", str(RESOLUTION), str(sparse),
        ], check=True, stdout=subprocess.DEVNULL)
        highest_bin = -1
        records = 0
        with sparse.open() as stream:
            for line in stream:
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"invalid sparse line: {line[:100]}")
                left, right = int(parts[0]), int(parts[1])
                if left % RESOLUTION or right % RESOLUTION:
                    raise ValueError("unaligned .hic sparse coordinate")
                highest_bin = max(highest_bin, left // RESOLUTION, right // RESOLUTION)
                records += 1
        if highest_bin < 0:
            raise ValueError(f"empty sparse dump: {sample} chr{chromosome}")
        bins = highest_bin + 1
        temporary_npy = destination.with_suffix(".npy.part")
        matrix = np.lib.format.open_memmap(temporary_npy, mode="w+", dtype=np.float32,
                                           shape=(bins, bins))
        matrix[:] = 0.0
        finite_pixels = 0
        nonfinite_pixels = 0
        with sparse.open() as stream:
            for line in stream:
                left_text, right_text, value_text = line.split()
                left = int(left_text) // RESOLUTION
                right = int(right_text) // RESOLUTION
                value = float(value_text)
                if not math.isfinite(value):
                    nonfinite_pixels += 1
                    continue
                # The pinned deepTAD preprocessing writes its dense matrix
                # with np.savetxt(fmt="%g"), i.e. six significant digits.
                # Match that input exactly before float32 storage.
                rounded = float(f"{value:g}")
                matrix[left, right] = rounded
                matrix[right, left] = rounded
                finite_pixels += 1
        matrix.flush()
        del matrix
        temporary_npy.replace(destination)
    matrix = np.load(destination, mmap_mode="r")
    audit: dict[str, object] = {
        "sample": sample, "chromosome": chromosome, "resolution_bp": RESOLUTION,
        "bins": bins, "sparse_records": records, "finite_pixels": finite_pixels,
        "nonfinite_pixels_zeroed": nonfinite_pixels,
        "output": str(destination), "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }
    if validate_text:
        text_path = OUTPUT / f"{sample}_10k_KR.chr{chromosome}"
        dense = np.loadtxt(text_path, dtype=np.float32)
        if dense.shape != matrix.shape:
            raise ValueError(f"dense text shape {dense.shape} != NPY {matrix.shape}")
        absolute = np.abs(dense - matrix)
        audit["maximum_absolute_difference_from_dense_text"] = float(np.max(absolute))
        audit["different_pixels_above_1e-4"] = int(np.count_nonzero(absolute > 1e-4))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES))
    parser.add_argument("--chromosomes", nargs="+", required=True)
    parser.add_argument("--validate-text", action="store_true")
    args = parser.parse_args()
    results = [convert(sample, chromosome, validate_text=args.validate_text)
               for chromosome in args.chromosomes for sample in args.samples]
    manifest = ARTIFACT / "npy_conversion_manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else {"schema_version": 1, "records": []}
    keyed = {(row["sample"], row["chromosome"]): row for row in previous["records"]}
    keyed.update({(row["sample"], row["chromosome"]): row for row in results})
    manifest.write_text(json.dumps({"schema_version": 1, "records": list(keyed.values())}, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
