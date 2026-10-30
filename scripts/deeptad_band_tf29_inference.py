"""Run the pinned deepTAD model on precomputed normalized diagonal patches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from deeptad_tf29_inference import build_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--chromosomes", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    tf.random.set_seed(123)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    model = build_model()
    model.load_weights(args.weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for chromosome in args.chromosomes:
        path = args.patch_dir / f"{chromosome}_patches.npy"
        patches = np.load(path, mmap_mode="r")
        if patches.ndim != 3 or patches.shape[1:] != (10, 10):
            raise ValueError(f"invalid patch shape for {path}: {patches.shape}")
        started = time.perf_counter()
        probabilities = model.predict(patches[..., None], batch_size=args.batch_size, verbose=0).reshape(-1)
        output = args.output_dir / f"{chromosome}_predictions.npz"
        np.savez_compressed(output, bin_index=np.arange(len(probabilities), dtype=np.int32),
                            boundary_probability=probabilities.astype(np.float32),
                            boundary_call=(probabilities > 0.5))
        record = {"chromosome": chromosome, "bins": int(len(probabilities)), "boundary_calls": int(np.count_nonzero(probabilities > 0.5)),
                  "inference_seconds": time.perf_counter() - started, "patch_sha256": sha256(path), "prediction_sha256": sha256(output)}
        records.append(record)
        print(json.dumps(record), flush=True)
    manifest_path = args.output_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous and previous.get("weights_sha256") != sha256(args.weights):
        raise ValueError("checkpoint differs from existing prediction manifest")
    by_chromosome = {record["chromosome"]: record for record in previous.get("records", [])}
    by_chromosome.update({record["chromosome"]: record for record in records})
    manifest = {"model": "pinned public deepTAD", "weights_sha256": sha256(args.weights),
                "tensorflow": tf.__version__, "batch_size": args.batch_size, "threads": args.threads,
                "records": [by_chromosome[key] for key in sorted(by_chromosome, key=lambda name: (0, int(name[3:])) if name[3:].isdigit() else (1, name))]}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
