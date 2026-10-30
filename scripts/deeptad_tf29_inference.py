"""Deterministic inference for the pinned public deepTAD TensorFlow weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faster_cky.deeptad_monge import (  # noqa: E402
    deeptad_boundary_patches,
    normalize_deeptad_patches,
)
from faster_cky.tad_matrix_io import load_dense_matrix  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _channel_attention(features: tf.Tensor, ratio: int = 8) -> tf.Tensor:
    channels = int(features.shape[-1])
    filters = max(1, channels // ratio)
    first = layers.Dense(
        filters,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )
    second = layers.Dense(
        channels,
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )
    average = second(first(layers.GlobalAveragePooling2D()(features)))
    maximum = second(first(layers.GlobalMaxPooling2D()(features)))
    weights = layers.Activation("sigmoid")(
        layers.Reshape((1, 1, channels))(layers.Add()([average, maximum]))
    )
    return layers.Multiply()([features, weights])


def _spatial_attention(features: tf.Tensor) -> tf.Tensor:
    average = layers.Lambda(lambda value: K.mean(value, axis=3, keepdims=True))(
        features
    )
    maximum = layers.Lambda(lambda value: K.max(value, axis=3, keepdims=True))(
        features
    )
    weights = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(layers.Concatenate(axis=3)([average, maximum]))
    return layers.Multiply()([features, weights])


def build_model() -> tf.keras.Model:
    """Recreate the topology in the pinned ``cnn+transformer.ipynb``."""
    inputs = layers.Input(shape=(10, 10, 1))
    values = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(inputs)
    values = layers.MaxPooling2D((2, 2))(values)
    values = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(values)
    values = layers.MaxPooling2D((2, 2))(values)
    values = _spatial_attention(_channel_attention(values))
    values = layers.Flatten()(values)
    values = layers.Dense(128, activation="relu")(values)
    values = layers.Reshape((1, 128))(values)

    residual = layers.LayerNormalization(epsilon=1e-6)(values)
    residual = layers.MultiHeadAttention(
        num_heads=4, key_dim=128, dropout=0.1
    )(residual, residual)
    values = layers.Add()([values, residual])
    residual = layers.LayerNormalization(epsilon=1e-6)(values)
    residual = layers.Dense(512, activation="gelu")(residual)
    residual = layers.Dropout(0.1)(residual)
    residual = layers.Dense(128, activation="gelu")(residual)
    residual = layers.Dropout(0.1)(residual)
    values = layers.Add()([values, residual])

    values = layers.Flatten()(values)
    values = layers.Dense(128, activation="relu")(values)
    values = layers.Dropout(0.4)(values)
    values = layers.Dense(64, activation="relu")(values)
    values = layers.Dropout(0.4)(values)
    outputs = layers.Dense(1, activation="sigmoid")(values)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--chromosomes", nargs="+", default=["20", "21", "22"])
    parser.add_argument("--resolution", type=int, default=25_000)
    parser.add_argument("--sample-prefix", default="HIC002")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    tf.random.set_seed(123)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    model = build_model()
    model.load_weights(args.weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    display_resolution = args.resolution // 1000
    for chromosome in args.chromosomes:
        chromosome_started = time.perf_counter()
        input_stem = args.input_dir / f"{args.sample_prefix}_{display_resolution}k_KR.chr{chromosome}"
        started = time.perf_counter()
        contacts, input_path = load_dense_matrix(
            input_stem, dtype=np.float32, mmap=True
        )
        matrix_load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        patches = normalize_deeptad_patches(deeptad_boundary_patches(contacts))
        patch_seconds = time.perf_counter() - started
        started = time.perf_counter()
        probabilities = model.predict(
            patches[..., None], batch_size=args.batch_size, verbose=0
        ).reshape(-1)
        inference_seconds = time.perf_counter() - started
        output_path = args.output_dir / f"{args.sample_prefix}_{display_resolution}k_KR.chr{chromosome}.npz"
        np.savez_compressed(
            output_path,
            bin_index=np.arange(len(probabilities), dtype=np.int32),
            boundary_probability=probabilities.astype(np.float32),
            boundary_call=(probabilities > 0.5),
        )
        records.append(
            {
                "chromosome": f"chr{chromosome}",
                "bins": len(probabilities),
                "boundary_calls": int(np.count_nonzero(probabilities > 0.5)),
                "probability_minimum": float(probabilities.min()),
                "probability_maximum": float(probabilities.max()),
                "probability_mean": float(probabilities.mean()),
                "matrix_load_seconds": matrix_load_seconds,
                "patch_seconds": patch_seconds,
                "inference_seconds": inference_seconds,
                "elapsed_seconds": time.perf_counter() - chromosome_started,
                "input_path": str(input_path),
                "input_sha256": _sha256(input_path),
                "output_path": str(output_path),
                "output_sha256": _sha256(output_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "source_revision": "713780e22fb70f07520c41493f9a590ac19cb4d9",
        "weights": str(args.weights),
        "weights_sha256": _sha256(args.weights),
        "threshold": 0.5,
        "resolution": args.resolution,
        "sample_prefix": args.sample_prefix,
        "preprocessing": "public 10x10 diagonal patches and row-wise standardization; deterministic order",
        "public_notebook_differences": [
            "does not shuffle prediction patches",
            "does not discard the final incomplete batch",
            "retains raw probabilities and explicit bin indices",
        ],
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
