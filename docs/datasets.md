# Dataset inputs

The exact Hi-C accessions, URLs, local filenames, and recorded file sizes are in [hic_inputs.json](../provenance/hic_inputs.json). No contact maps are downloaded during installation or result verification.

List the inputs and sizes before downloading:

```bash
python scripts/download_inputs.py --group all
```

Download the benchmark or biological inputs explicitly:

```bash
python scripts/download_inputs.py --group benchmark --download
python scripts/download_inputs.py --group nipbl --download
```

Downloads use a temporary file and can be resumed. File sizes are checked against the deposited input records. The full processed contact maps are used, followed by extraction of all 22 autosomes; sex chromosomes are outside the evaluation protocol.

## Accuracy and computation-time inputs

The hg19 samples come from [GSE63525](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525), reported by [Rao et al.](https://doi.org/10.1016/j.cell.2014.11.021): GSM1551551 (GM12878), GSM1551623 (K562), and GSM1551605 (IMR90). Juicer exports KR-normalized contacts at the resolution specified in the README.

The 5 kb experiments use the hg38 processed files [4DNFIXP4QG5B](https://data.4dnucleome.org/files-processed/4DNFIXP4QG5B/), [4DNFI18UHVRO](https://data.4dnucleome.org/files-processed/4DNFI18UHVRO/), and [4DNFIJTOIGOI](https://data.4dnucleome.org/files-processed/4DNFIJTOIGOI/). In particular, the K562 input is **4DNFI18UHVRO**. The matrix conversion uses the stored `weight` column at `/resolutions/5000`.

The three cell lines and resolutions follow the existing deepTAD evaluation inputs and the available processed 5 kb maps. They provide comparisons across cell lines and resolutions; they do not represent every public Hi-C dataset. The 25 kb deepTAD checkpoint is applied at finer resolutions without retraining.

## CTCF inputs

The six compressed peak files used by the current evaluation are included. [ctcf_inputs.json](../provenance/ctcf_inputs.json) records their source, assembly, and checksums.

| Combination | CTCF input | Assembly |
|---|---|---|
| GM12878, 25/10 kb | GSM749704 | hg19 |
| K562, 25 kb | ENCFF738TKN | hg19 |
| IMR90, 25 kb | ENCFF453XKM | hg19 |
| GM12878, 5 kb | ENCFF796WRU | hg38 |
| K562, 5 kb | ENCFF901CBP | hg38 |
| IMR90, 5 kb | ENCFF203SRF | hg38 |

We compare predicted endpoints with the peak intervals in the matching assembly using a 10 kb tolerance on each side. We do not substitute CTCF motif matches for ChIP-seq peaks.

## 501mel NIPBL inputs

[Wong et al.](https://doi.org/10.1038/s41588-025-02239-6) provide two control replicates (GSM7325027 and GSM7325028) and two replicates with experimentally reduced NIPBL expression (GSM7325029 and GSM7325030). The scripts use the published 10 kb `.hic` files, KR extraction, and all 22 autosomes. The two reciprocal comparisons are specified in [reproduction.md](reproduction.md).

## Additional downloads for previous methods

RobusTAD uses its published 177-sample hg38 5 kb reference panel:

```bash
bash scripts/download_robustad_reference_panel.sh
```

That downloader follows the original macOS experiment environment. Its URL and expected size are defined at the start of the script. The deposited baseline boundary files allow assembly to be repeated with the same boundary choices. At hg19 10/25 kb, the comparison instead uses the authors' mode without a reference panel.
