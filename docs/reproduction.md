# Reproducing the experiments

Run commands from the repository root, with the package installed as described in the README. The repository preserves the directory names used by the experiment scripts. New calculations below write to `work/`; the deposited results remain unchanged.

## 1. Recompute tables from the deposited records

```bash
python scripts/reproduce_results.py --check-files
```

No external software or large datasets are required for this step. To check without writing tables, add `--verify-only`.

`r2` is TAD-adjusted R², averaged equally over eligible genomic distances from one bin to 1.5 Mb. Contacts at each distance are assigned to the smallest containing reported domain, or to a shared group for contacts outside all reported domains. The number of eligible domains enters the degrees-of-freedom adjustment.

`quality` is the mean intra-inter difference, following ClusterTAD's definition of within-domain and adjacent-domain pixels. We evaluate outermost reported intervals to avoid counting nested domains as adjacent partitions. The deposited implementation specifies the exact pixel sets.

`ctcf_percent` is the percentage of parent domains with CTCF ChIP-seq peaks within ±10 kb of **both** endpoints. For previous methods, a parent domain strictly contains another reported domain. This measures binding near the endpoints, not physical pairing of the endpoints.

All three metrics are averaged equally over chromosomes. A method–combination aggregate is available only when all 22 autosomes complete execution and evaluation. Partial chromosome scores remain in the record but are not substituted for the full-autosome comparison.

## 2. Recreate contact matrices

Acquire the inputs in [datasets.md](datasets.md) and the deepTAD source and Juicer jar in [previous_methods.md](previous_methods.md). The CTCF peak files and boundary predictions are already included.

```bash
python scripts/prepare_benchmark_matrices.py
python scripts/prepare_501mel_nipbl_npy.py --chromosomes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22
```

The first command processes all seven combinations and all 22 autosomes by default. It checks matrix dimensions against the deposited records. The coarse-resolution inputs follow deepTAD's Juicer KR dump and `remove_nan.py` conversion, including its text precision. The 5 kb inputs use the published Cooler `weight` column and float32 storage. Do not rebalance the matrices or substitute a different processed accession.

The scripts retain prepared matrices. Dense chromosome matrices require substantial disk space and memory; this step is not part of the lightweight result-verification command.

## 3. Recompute accuracy and objective ablations

```bash
python scripts/rerun_tad_without_separators.py campaign --output-dir work/recomputed
```

The command attempts all 154 chromosome jobs with a 1,800-second limit per job. Each job uses the deposited deepTAD boundary predictions and filters, recomputes contacts and the proposed objective, and evaluates six variants: Ours, No length, Length only, With separators, Greedy, and Matched count. It checks the proposed solution against standard dynamic programming and the retained previous-ablation values.

To perform the additional numerical check of the adjusted-R² implementation, use the `benchmark` phase with `--audit-reference`. The `campaign` phase launches and records independent chromosome jobs; it resumes completed outputs rather than overwriting them.

The current objective is

```text
W[i,j] = ((right_endpoint[j-1] - left_endpoint[i]) / 4,000,000)^2
         + sum(max(1 - mean_OE[p,q], 0) for i <= p < q < j).
```

Genomic endpoints are in base pairs. The implemented length scale is converted to bins for each resolution. All atomic TADs are retained as leaves. For each tree, the output includes selected parent domains with span at most 3 Mb and a target count of `round((351/1354) * number_of_candidate_parents)`. The parent ranking combines standardized endpoint-profile dissimilarity and internal-versus-adjacent O/E contacts. The NIPBL analysis does not use the 3 Mb cap. In that analysis, the maximum contact separation used for parent ranking is 600 kb; the general benchmark uses 1.5 Mb.

The archive label `KY-W` denotes Ours. Some low-level function names retain `duration` for the genomic-length penalty. The older separator objective in `tad_configuration.py` is used for the ablation reference; the function `objective(resolution)` in the current runner disables all separator contributions.

## 4. Repeat the NIPBL analysis

```bash
python scripts/rerun_tad_nipbl_without_separators.py --output-dir work/recomputed
```

This attempts all 44 chromosome/comparison pairs. In the first comparison, parent coordinates come from GSM7325027 and are evaluated in GSM7325028 and GSM7325030. In the second, coordinates come from GSM7325028 and are evaluated in GSM7325027 and GSM7325029. The two evaluation maps in each comparison use the same parent coordinates. The original experiment used short hairpin RNA to reduce NIPBL messenger RNA by approximately 56%.

The deposited results contain 922 and 897 evaluated outermost parents. Their mean intra-inter differences change from 0.825342 to 0.646276 and from 0.792806 to 0.653845, respectively. The combined mean changes from 0.809297 to 0.650009. These are descriptive summaries of two biological replicates per condition.

## 5. Recompute total computation time

After the accuracy and biological runs above finish, install Docker and obtain the deepTAD checkpoint:

```bash
python scripts/rerun_tad_without_separator_timing.py --output-dir work/recomputed
```

This runs Ours and deepTAD serially. The shared boundary-prediction stage is included in both totals. The process includes loading prepared matrices, starting the model, predicting and filtering boundaries, computing contact features and costs, assembling domains, and writing the output. Downloading data, extracting the original archives, and evaluating accuracy are excluded.

The archive timings were measured on an Apple M4 Max with 36 GB RAM. deepTAD inference and HiTAD used Linux/amd64 Docker emulation; other methods ran directly on macOS. The Docker VM had 7.65 GiB of memory. New non-Docker baseline processes used a 20 GiB resident-memory limit. Timings from a different machine or environment should be reported separately. The complete execution records retain the scheduling and completion status of each baseline run.

See [previous_methods.md](previous_methods.md) to repeat the other methods. This release packages existing results; creating the repository does not rerun the large experiments.

## Provenance

`provenance/source.json` identifies the original working repository and export date. `provenance/copied_files.json` records original content hashes and release adaptations. `provenance/release_manifest.json` checks every deposited file except itself. The export replaces machine-specific repository roots with relative paths and adds output-directory options; it preserves numerical result values.

`provenance/python_environment.json` records the installed versions used to validate this release. `requirements-validation.txt` gives the corresponding Python dependencies. The upstream TensorFlow environment remains separate from the Python package environment.

The source runtime JSON contained cached ratios from an older measurement. The release recomputes those ratios from the current timing rows and verifies Ours/deepTAD totals against all 22 per-chromosome timing records. The measured times are unchanged.
