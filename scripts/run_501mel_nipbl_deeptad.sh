#!/usr/bin/env bash
set -euo pipefail

# Pilot: transfer the existing deepTAD boundary model to 501mel 10 kb maps.
# Boundary quality must be audited before any biological inference.
repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"
if (( $# )); then
  chromosomes=("$@")
else
  chromosomes=(22)
fi

input_dir=data/processed/501mel_nipbl_10kb
read -r -a samples <<< "${SAMPLE_IDS:-GSM7325027 GSM7325028 GSM7325029 GSM7325030}"
for sample in "${samples[@]}"; do
  prediction_dir="artifacts/tad_501mel_nipbl/boundary_predictions_$sample"
  filtered_dir="artifacts/tad_501mel_nipbl/filtered_boundaries_$sample"
  mkdir -p "$prediction_dir" "$filtered_dir"
  docker run --rm --platform linux/amd64 -v "$PWD:/work" -w /work tensorflow/tensorflow:2.9.3 \
    python scripts/deeptad_tf29_inference.py --input-dir "$input_dir" --output-dir "$prediction_dir" \
    --weights third_party/deepTAD/Model/all_27.h5 --chromosomes "${chromosomes[@]}" \
    --resolution 10000 --sample-prefix "$sample" --threads "${THREADS:-8}"
  PYTHONPATH=src .venv/bin/python scripts/filter_deeptad_boundaries.py \
    --matrix-dir "$input_dir" --prediction-dir "$prediction_dir" --output-dir "$filtered_dir" \
    --chromosomes "${chromosomes[@]}" --resolution 10000 --sample-prefix "$sample"
done
