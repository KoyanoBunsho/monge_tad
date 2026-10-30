# Previous methods

The comparison includes deepTAD, OnTAD, JOnTADS, deDoc2, RobusTAD, SpectralTAD, and HiTAD. Their code is obtained from the original authors; this repository contains adapters, the patches used in the experiment, and recorded outputs. Source revisions are recorded in [software.json](../provenance/software.json).

## Obtain the source code

```bash
python scripts/fetch_previous_methods.py
```

This creates the expected directories under `third_party/` and checks out the recorded revisions. It does not install dependencies or execute the methods. Existing directories with a different revision are left unchanged with an error.

For deepTAD, confirm that `third_party/deepTAD/Model/all_27.h5` and `third_party/deepTAD/prepare_data/juicer_tools.jar` are present. Inference uses `tensorflow/tensorflow:2.9.3` in Linux/amd64 Docker. The checkpoint is not redistributed in this repository.

Build OnTAD using its C++ compiler, libcurl, and zlib requirements:

```bash
make -C third_party/OnTAD/src
```

The expected deDoc2 jar is `third_party/deDoc2/deDoc2.jar`; Java must be on `PATH`. Its checksum is in `software.json`.

JOnTADS uses a separate Python 3.10 environment, because its recorded NumPy/Numba versions differ from the main environment:

```bash
python3.10 -m venv third_party/venvs/jontads
third_party/venvs/jontads/bin/python -m pip install -r requirements-jontads.txt
```

The adapter invokes `third_party/JOnTADS/JOnTADS/JOnTADS.py` directly. The installed versions used by the experiment are in `provenance/jontads_environment.json`.

For RobusTAD, install its dependencies in the main environment and apply the recorded correction:

```bash
python -m pip install click tqdm numba scikit-learn matplotlib importlib-resources appdirs
git -C third_party/RobusTAD apply ../../patches/robustad_corner.patch
```

The correction adds the existing 17×17 contact-window size check before indexing truncated windows. The adapter verifies the resulting source checksum before running. For 5 kb hg38 experiments, the deposited `artifacts/tad_complete_pipeline_20261024/*/robustad/boundaries.bed` files preserve the boundary choices used by the comparison. To generate new reference-panel boundaries instead, download the panel and run:

```bash
python scripts/benchmark_tad_complete_pipeline.py robustad --conditions gm12878_5kb k562_5kb imr90_5kb --output-dir work/pipeline
```

Pass `--boundary-root work/pipeline` to the baseline runner when using these new boundaries. The upstream boundary procedure uses random decoys; newly generated boundary flags can differ from the deposited flags.

For SpectralTAD, install R, Bioconductor, and **SpectralTAD 1.28.2** into `.cache/Rlib`. `scripts/run_spectraltad.R` applies the deposited correction for undefined silhouette values. Its regression example is `tests/check_spectraltad_repair.R`.

For HiTAD, build the separate environment:

```bash
docker build --platform linux/amd64 -t tad-hitad:20261022 -f scripts/Dockerfile.hitad .
```

The wrapper supplies the removed `scipy.stats.itemfreq` function required by the upstream code. TADLib is mounted from `third_party/TADLib`.

## Run the comparisons

After preparing the matrices, the following commands attempt all 22 autosomes for every combination:

```bash
python scripts/expand_tad_method_coverage.py --repeat-all-methods --methods ontad jontads dedoc2 robustad-panel robustad-no-panel spectraltad --output-root work/previous_methods
python scripts/benchmark_hitad_autosomes.py --output-dir work/hitad
```

The first command chooses the RobusTAD mode appropriate to each assembly. It enforces a 1,800-second limit per execution and a 20 GiB resident-memory limit. HiTAD processes the 22 autosomes together in one execution. The available memory inside Docker can be smaller than the machine's physical memory.

The deposited parameter choices include OnTAD `-penalty 0.1 -minsz 3`, JOnTADS `-MINSZ 7` and seed 0, and a 10 Mb maximum domain length for OnTAD, JOnTADS, and deDoc2. SpectralTAD uses three levels. RobusTAD uses its published hg38 5 kb panel for 5 kb inputs and the authors' no-panel mode for hg19 inputs. Method-specific commands are retained in `results/execution_records.json`.

A successful process must also produce a valid output. The runners distinguish timeout, memory error, runtime error, and missing execution. Accuracy is aggregated only after all 22 chromosomes complete; unsuccessful attempts remain in the records.

TADtree and SuperTAD did not complete the initial attempts within the execution limits. SuperTAD-Fast was tested separately and excluded when extrapolated computation time exceeded the study's 48-hour experiment budget. Its recorded decision is in `artifacts/tad_supertad_fast_campaign_20261026/exclusion.json`. These methods are not assigned accuracy values in the completed-comparison tables.

## Scope of the deposited timings

The archive preserves the recorded environments and execution modes, including baseline jobs scheduled concurrently to obtain accuracy first. Ours and deepTAD were subsequently remeasured serially with the current objective. Creating this release does not replace baseline timings with a new serial measurement. Use the execution records when selecting timings for a new comparison, and rerun on a common environment when measuring a new machine.
