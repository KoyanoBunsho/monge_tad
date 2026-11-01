# MongeTAD

Code and experimental results for **Fast and Accurate Optimization for Hierarchical TAD Assembly**.

**Bunsho Koyano** · Independent researcher, Tokyo, Japan

MongeTAD extends deepTAD by keeping its predicted atomic TADs as fixed leaves and optimizing the full ordered binary tree. The objective penalizes contacts below their genomic-distance-dependent expectation and the genomic lengths of parent domains. Based on the Monge property and interval inclusion monotonicity, we apply Knuth–Yao optimization in O(m²) time for m atomic TADs.

## Quick start

Run from the repository root with Python 3.11 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/optimize_fixed_leaves.py
python -m pytest -q
```

The example compares Knuth–Yao optimization with standard O(m³) dynamic programming on a small synthetic input.

## Verify the published results

This command needs only the Python standard library. It does not download data or rerun TAD prediction:

```bash
python3 scripts/reproduce_results.py --check-files
```

It recomputes the chromosome means from the deposited records, checks them against the reported tables, and writes CSV and Markdown files to `results/reproduced/`. It verifies:

- 154 chromosome evaluations: seven combinations of cell line and resolution, each using chromosomes 1–22.
- Agreement between Knuth–Yao optimization and standard dynamic programming in all 4,601 evaluated regions.
- Accuracy and execution status for eight methods across all seven combinations. Aggregate accuracy is reported for the 42 method–combination pairs completed on all 22 autosomes; the other 14 retain their execution status.
- 44 chromosome-level biological comparisons, comprising 1,819 parent-domain comparisons across the two reciprocal evaluations.
- The ablation, matched-count, and greedy-merging results.

These checks verify the deposited outputs and their aggregation. Recomputing the metrics from the original contact maps requires the large input files and additional software described in [Reproducing the experiments](docs/reproduction.md).

## Results and data coverage

| Cell line | Resolution | Genome assembly | Hi-C input |
|---|---|---|---|
| GM12878 | 25 and 10 kb | hg19 | GSM1551551 / HIC002 |
| K562 | 25 kb | hg19 | GSM1551623 / HIC074 |
| IMR90 | 25 kb | hg19 | GSM1551605 / HIC056 |
| GM12878 | 5 kb | hg38 | 4DNFIXP4QG5B |
| K562 | 5 kb | hg38 | 4DNFI18UHVRO |
| IMR90 | 5 kb | hg38 | 4DNFIJTOIGOI |

Each row and resolution uses all 22 autosomes. The biological analysis uses the control and NIPBL-reduction samples GSM7325027–GSM7325030 from 501mel cells, also across all 22 autosomes. [Input accessions and acquisition commands](docs/datasets.md) give the exact files and normalization conventions.

Our method improves all three reported metrics over deepTAD in all seven combinations. Its total computation time is 6–13% longer at 10/25 kb and 2.25–3.54 times deepTAD's time at 5 kb. OnTAD has higher TAD-adjusted R², while our method has higher intra-inter difference in all seven combinations and higher both-endpoint CTCF support in six. SpectralTAD is faster and has higher intra-inter difference; our method has higher TAD-adjusted R² and both-endpoint CTCF support. Exact minimization does not ensure better evaluation metrics than greedy merging; those comparisons are included with the ablations.

| File | Contents |
|---|---|
| [results/quality_all_autosomes.csv](results/quality_all_autosomes.csv) | Three accuracy metrics and completion status for every method and combination |
| [results/runtime.csv](results/runtime.csv) | Computation time, execution units, and failure counts |
| [results/per_chromosome_metrics.csv](results/per_chromosome_metrics.csv) | Chromosome-level metric values, including completed parts of incomplete comparisons |
| [results/execution_records.json](results/execution_records.json) | Baseline execution commands, outcomes, and recorded timings |
| [results/tables](results/tables) | LaTeX table fragments for accuracy, computation time, NIPBL, greedy merging, and ablations |
| [artifacts/tad_no_separator_20261029](artifacts/tad_no_separator_20261029) | Current objective, chromosome outputs, selected domains, ablations, exactness checks, and biological results |
| [provenance](provenance) | Source provenance, software revisions, input identifiers, and file checksums |

The release also contains the boundary predictions, CTCF peak files, and previous outputs needed to repeat the current experiment's consistency checks. Raw Hi-C archives, chromosome contact matrices, model weights, external software, and unrelated exploratory experiments are not included.

## Implementation and evaluation

- [decoder.py](src/faster_cky/decoder.py): standard dynamic programming, Knuth–Yao optimization, and greedy merging.
- [deeptad_monge.py](src/faster_cky/deeptad_monge.py): atomic TADs, contact profiles, contact deficits, and interval costs.
- [rerun_tad_without_separators.py](scripts/rerun_tad_without_separators.py): the current objective, all-autosome accuracy experiment, and ablations.
- [tad_published_metrics.py](src/faster_cky/tad_published_metrics.py): intra-inter difference and CTCF peaks within 10 kb of both endpoints.
- [benchmark_tad_complete_pipeline.py](scripts/benchmark_tad_complete_pipeline.py): computation time from prepared contact matrices through domain output.
- [rerun_tad_nipbl_without_separators.py](scripts/rerun_tad_nipbl_without_separators.py): fixed-coordinate comparisons between control cells and cells with experimentally reduced NIPBL expression.

The metrics follow [OnTAD](https://doi.org/10.1186/s13059-019-1893-y), [ClusterTAD](https://doi.org/10.1186/s12859-017-1931-2), and [Long et al.](https://doi.org/10.1186/s13072-022-00436-9). The boundary predictor is from [deepTAD](https://doi.org/10.1093/bib/bbaf127). The 501mel analysis reuses data from [Wong et al.](https://doi.org/10.1038/s41588-025-02239-6).

See [Reproducing the experiments](docs/reproduction.md) for parameters, environments, and commands, and [Previous methods](docs/previous_methods.md) for their installation and comparison settings.
