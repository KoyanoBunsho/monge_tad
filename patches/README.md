# Corrections used for previous methods

- `robustad_corner.patch`: check the size of a contact window before comparing it with the existing 17×17 template. The existing fallback score is retained for a truncated window. The upstream license is included in `licenses/RobusTAD.LICENSE`.
- `../scripts/patch_spectraltad_silhouette.R`: represent undefined candidate silhouettes as `NA`, retaining the upstream handling of defined scores and candidate order.
- `../scripts/run_hitad_command.py`: provide the removed `scipy.stats.itemfreq` API using unique values and counts.

The experiment's validation record and checked source hashes are in `../artifacts/tad_baseline_retries_20261026/implementation.json`. Source code for the previous methods is downloaded separately and retains the authors' licenses.
