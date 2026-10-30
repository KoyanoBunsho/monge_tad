#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("usage: run_spectraltad.R MATRIX.npy CHROMOSOME RESOLUTION LEVELS OUTPUT.tsv")
}

matrix_path <- args[[1]]
chromosome <- args[[2]]
resolution <- as.integer(args[[3]])
levels <- as.integer(args[[4]])
output_path <- args[[5]]

read_numpy_matrix <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  magic <- readBin(con, "raw", n = 6)
  if (!identical(as.integer(magic), c(147L, 78L, 85L, 77L, 80L, 89L))) {
    stop("not a NumPy .npy file: ", path)
  }
  version <- readBin(con, "integer", n = 2, size = 1, signed = FALSE)
  header_size <- if (version[[1]] == 1L) 2L else 4L
  header_length <- readBin(
    con, "integer", n = 1, size = header_size, signed = FALSE,
    endian = "little"
  )
  header <- rawToChar(readBin(con, "raw", n = header_length))
  item_size <- if (grepl("['\"]<f4['\"]", header)) 4L else if (grepl("['\"]<f8['\"]", header)) 8L else NA_integer_
  if (is.na(item_size)) stop("expected little-endian float32 or float64: ", header)
  if (grepl("'fortran_order': *True", header)) stop("expected row-major NumPy input")
  shape_text <- sub(".*'shape': *\\(([^)]*)\\).*", "\\1", header)
  shape <- as.integer(strsplit(gsub("[[:space:]]", "", shape_text), ",")[[1]])
  shape <- shape[!is.na(shape)]
  if (length(shape) != 2L || shape[[1]] != shape[[2]]) {
    stop("expected a square two-dimensional matrix, got shape: ", shape_text)
  }
  n <- shape[[1]]
  count <- as.double(n) * n
  values <- readBin(con, "numeric", n = count, size = item_size, endian = "little")
  if (length(values) != count) {
    stop("truncated NumPy array: ", path)
  }
  dim(values) <- c(n, n)
  # NumPy stores rows contiguously whereas R stores columns contiguously.
  # Hi-C matrices are symmetric, so this in-memory view is unchanged.
  coordinates <- as.character(seq.int(0, by = resolution, length.out = n))
  colnames(values) <- coordinates
  rownames(values) <- coordinates
  values
}

suppressPackageStartupMessages(library(SpectralTAD))
script_file <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[[1]])
source(file.path(dirname(normalizePath(script_file)), "patch_spectraltad_silhouette.R"))
patch_spectraltad_silhouette()
contact_matrix <- read_numpy_matrix(matrix_path)

started <- proc.time()[["elapsed"]]
calls <- SpectralTAD(
  contact_matrix,
  chr = chromosome,
  levels = levels,
  resolution = resolution,
  out_format = "none"
)
elapsed <- proc.time()[["elapsed"]] - started

rows <- do.call(rbind, lapply(seq_along(calls), function(level) {
  x <- as.data.frame(calls[[level]])
  if (nrow(x) == 0L) return(NULL)
  data.frame(
    chr = as.character(x$chr),
    start = as.integer(x$start),
    end = as.integer(x$end),
    level = level
  )
}))
if (is.null(rows)) {
  rows <- data.frame(chr = character(), start = integer(), end = integer(), level = integer())
}
rows <- unique(rows)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.table(rows, output_path, quote = FALSE, sep = "\t", row.names = FALSE, col.names = FALSE)
cat(sprintf("SpectralTAD chromosome=%s bins=%d levels=%d domains=%d elapsed_seconds=%.6f\n",
            chromosome, nrow(contact_matrix), levels, nrow(rows), elapsed))
