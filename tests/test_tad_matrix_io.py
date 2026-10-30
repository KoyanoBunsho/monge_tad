from pathlib import Path

import numpy as np
import pytest

from faster_cky.tad_matrix_io import load_dense_matrix, resolve_dense_matrix_path


def test_load_dense_matrix_resolves_binary_suffix(tmp_path: Path) -> None:
    stem = tmp_path / "sample"
    expected = np.arange(9, dtype=np.float32).reshape(3, 3)
    np.save(f"{stem}.npy", expected)

    observed, path = load_dense_matrix(stem, dtype=np.float32, mmap=True)

    assert path == Path(f"{stem}.npy")
    assert isinstance(observed, np.memmap)
    np.testing.assert_array_equal(observed, expected)


def test_load_dense_matrix_rejects_non_square_input(tmp_path: Path) -> None:
    path = tmp_path / "rectangle.npy"
    np.save(path, np.zeros((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="must be square"):
        load_dense_matrix(path)


def test_resolve_dense_matrix_prefers_legacy_exact_path(tmp_path: Path) -> None:
    stem = tmp_path / "matrix"
    stem.write_text("1 0\n0 1\n")
    np.save(f"{stem}.npy", np.zeros((2, 2), dtype=np.float32))

    assert resolve_dense_matrix_path(stem) == stem
