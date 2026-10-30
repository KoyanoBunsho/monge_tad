"""Convert the published 5 kb contact maps to chromosome matrices."""
import cooler
import h5py
import numpy as np


SOURCES = {
    "gm12878": ("4DNFIXP4QG5B", "GM12878", "data/raw/contexttad/ContextTAD_reproducibility/data/1_dp_train_infer_data/annotations/gm12878_ctcf.bed"),
    "k562": ("4DNFI18UHVRO", "K562", "data/processed/ctcf_combined_5kb/K562_CTCF_combined.bed"),
    "imr90": ("4DNFIJTOIGOI", "IMR90", "data/processed/ctcf_combined_5kb/IMR90_CTCF_combined.bed"),
}

def materialize(source, chrom, destination):
    contact_map = cooler.Cooler(f"{source}::/resolutions/5000")
    start, end = contact_map.extent(chrom)
    n = int(end - start)
    weights = contact_map.bins().fetch(chrom)["weight"].to_numpy()
    temporary = destination.with_suffix(".partial.npy")
    matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(n, n))
    matrix[:] = 0
    pixels_written = 0
    with h5py.File(source) as handle:
        group = handle["resolutions/5000"]
        offsets = group["indexes/bin1_offset"]
        left, right = int(offsets[start]), int(offsets[end])
        pixels = group["pixels"]
        for offset in range(left, right, 200_000):
            stop = min(offset + 200_000, right)
            rows = pixels["bin1_id"][offset:stop] - start
            cols = pixels["bin2_id"][offset:stop] - start
            counts = pixels["count"][offset:stop]
            keep = (rows >= 0) & (cols >= 0) & (rows < n) & (cols < n)
            rows, cols, counts = rows[keep], cols[keep], counts[keep]
            values = np.nan_to_num(counts * weights[rows] * weights[cols], nan=0., posinf=0., neginf=0.).astype(np.float32)
            matrix[rows, cols] = values
            matrix[cols, rows] = values
            pixels_written += len(rows)
    matrix.flush()
    del matrix
    temporary.replace(destination)
    return {"shape": [n, n], "stored_pixels_read": pixels_written, "bytes": destination.stat().st_size,
            "normalization": "Cooler weight column", "dtype": "float32", "duplicate_policy": "last"}
