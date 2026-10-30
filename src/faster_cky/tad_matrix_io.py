"""Dense contact-matrix I/O shared by the TAD evaluation commands."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def resolve_dense_matrix_path(stem: str | Path) -> Path:
    """Resolve the legacy extensionless text format or the binary NPY format."""

    path = Path(stem)
    if path.exists():
        return path
    npy_path = Path(f"{path}.npy")
    if npy_path.exists():
        return npy_path
    raise FileNotFoundError(f"no dense matrix at {path} or {npy_path}")


def load_dense_matrix(
    stem: str | Path,
    *,
    dtype: np.dtype | type = np.float64,
    mmap: bool = True,
) -> tuple[np.ndarray, Path]:
    """Load a square dense matrix, memory-mapping NPY inputs when requested."""

    path = resolve_dense_matrix_path(stem)
    if path.suffix == ".npy":
        matrix = np.load(path, mmap_mode="r" if mmap else None)
        if matrix.dtype != np.dtype(dtype) and not mmap:
            matrix = matrix.astype(dtype, copy=False)
    else:
        matrix = np.loadtxt(path, dtype=dtype)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"contact matrix must be square: {path} has {matrix.shape}")
    return matrix, path
