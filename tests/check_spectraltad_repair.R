suppressPackageStartupMessages(library(SpectralTAD))
source("scripts/patch_spectraltad_silhouette.R")
original <- patch_spectraltad_silhouette()
repaired <- get(".windowedSpec", asNamespace("SpectralTAD"))

make_blocks <- function(n = 80L, width = 10L, seed = 1L) {
  set.seed(seed)
  x <- matrix(runif(n*n, 0.1, 0.2), n, n)
  x <- (x + t(x))/2
  for (i in seq(1, n, width)) x[i:(i+width-1), i:(i+width-1)] <-
    x[i:(i+width-1), i:(i+width-1)] + 10
  dimnames(x) <- rep(list(as.character(seq(0, by = 25000, length.out = n))), 2)
  x
}

# A controlled eigensystem reproduces the undefined one-cluster candidate
# without changing the contact matrix or the valid-candidate scoring code.
with_degenerate_eigenvectors <- function(f) {
  env <- new.env(parent = environment(f))
  env$get_eigs <- function(x, NEig) {
    v <- matrix(0, nrow(x), NEig)
    v[1, ] <- 1
    list(values = seq_len(NEig), vectors = v)
  }
  environment(f) <- env
  f
}
x <- make_blocks(40)
failed <- try(with_degenerate_eigenvectors(original)(x, 25000, "chrTest"), silent = TRUE)
stopifnot(inherits(failed, "try-error"), grepl("sil\\$si.summary", failed))
fixed <- with_degenerate_eigenvectors(repaired)(x, 25000, "chrTest")
stopifnot(is.data.frame(fixed))
cat("Undefined candidate: original fails, repaired function returns", nrow(fixed), "domains\n")

for (seed in seq_len(10L)) {
  x <- make_blocks(seed = seed)
  set.seed(seed)
  a <- original(x, 25000, "chrTest")
  set.seed(seed)
  b <- repaired(x, 25000, "chrTest")
  stopifnot(identical(a, b), nrow(a) > 0)
}
cat("Ten valid synthetic examples: identical domain coordinates\n")

# The fixture contains small windows only for regression testing. Benchmark
# retries still process every bin of the original complete chromosomes.
for (path in commandArgs(trailingOnly = TRUE)) {
  x <- as.matrix(read.table(path))
  dimnames(x) <- rep(list(as.character(seq(0, by = 25000, length.out = nrow(x)))), 2)
  set.seed(1)
  a <- original(x, 25000, "chr22")
  set.seed(1)
  b <- repaired(x, 25000, "chr22")
  stopifnot(identical(a, b), nrow(a) > 0)
  cat("Real contact-map window:", basename(path), "identical", nrow(a), "domains\n")
}
