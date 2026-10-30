#!/usr/bin/env bash
set -euo pipefail

# Matched 501mel shLuc/shNIPBL_#2 Hi-C replicates from Wong et al. (2025),
# GEO GSE232299. Files are Juicer .hic, aligned to hg19 at MAPQ >= 30.
repo_root=$(cd "$(dirname "$0")/.." && pwd)
output_dir="$repo_root/data/raw/501mel_nipbl_hic"
mkdir -p "$output_dir"

samples=(
  GSM7325027_501mel_Luc_set1_map30_10kb_normcount.hic
  GSM7325028_501mel_Luc_set2_map30_10kb_normcount.hic
  GSM7325029_501mel_NIPBL2_set1_map30_10kb_normcount.hic
  GSM7325030_501mel_NIPBL2_set2_map30_10kb_normcount.hic
)
expected_bytes=(343614576 342554587 294686677 292361870)

for index in "${!samples[@]}"; do
  filename=${samples[$index]}
  accession=${filename%%_*}
  url="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7325nnn/$accession/suppl/$filename"
  destination="$output_dir/$filename"
  current_bytes=0
  if [[ -f "$destination" ]]; then
    current_bytes=$(stat -f %z "$destination")
  fi
  if [[ $current_bytes -eq ${expected_bytes[$index]} ]]; then
    echo "present: $destination"
    continue
  fi
  if [[ $current_bytes -gt ${expected_bytes[$index]} ]]; then
    echo "oversized file: $destination" >&2
    exit 1
  fi
  echo "fetching $accession: $destination"
  curl -L --fail --retry 3 --retry-delay 5 -C - -o "$destination" "$url"
  [[ $(stat -f %z "$destination") -eq ${expected_bytes[$index]} ]] || {
    echo "incomplete file: $destination" >&2
    exit 1
  }
done
