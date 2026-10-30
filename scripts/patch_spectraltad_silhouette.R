# Local compatibility repair for SpectralTAD 1.28.2, applied in this R process.
# Preserve the candidate sequence and all defined scores. An undefined
# silhouette has an NA score; the existing which(diff(scores) < 0) selection
# cannot select a comparison containing NA. Do not assign a fabricated score.
patch_spectraltad_silhouette <- function() {
  ns <- asNamespace("SpectralTAD")
  if (as.character(utils::packageVersion("SpectralTAD")) != "1.28.2") {
    stop("Review the silhouette repair before using another SpectralTAD version")
  }
  original <- get(".windowedSpec", ns)
  if (isTRUE(attr(original, "undefined_silhouette_repair"))) return(invisible(original))
  target <- quote({sil = summary(cluster::silhouette(memberships, dist_sub))})[[2]]
  replacement <- quote({
    sil = cluster::silhouette(memberships, dist_sub)
    if (inherits(sil, "silhouette")) {
      sil = summary(sil)
    } else {
      # cluster::silhouette returns NA unless 2 <= number of clusters < n.
      stopifnot(length(sil) == 1L, is.na(sil))
      sil = list(si.summary = rep(NA_real_, 6L))
      message("SpectralTAD: undefined candidate silhouette retained as NA")
    }
  })
  replacements <- 0L
  rewrite <- function(node) {
    if (identical(node, target)) {
      replacements <<- replacements + 1L
      return(replacement)
    }
    if (is.call(node)) for (i in seq_along(node)) {
      if (!identical(node[[i]], quote(expr = ))) node[[i]] <- rewrite(node[[i]])
    }
    node
  }
  repaired <- original
  body(repaired) <- rewrite(body(original))
  stopifnot(replacements == 1L)
  attr(repaired, "undefined_silhouette_repair") <- TRUE
  assignInNamespace(".windowedSpec", repaired, ns = "SpectralTAD")
  invisible(original)
}
