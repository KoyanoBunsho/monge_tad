#!/usr/bin/env bash
set -euo pipefail

url='https://repo.cs.mcgill.ca/PUB/blanchem/RobustTAD/hg38_5kb_50_600.pkl%3fdownload=1'
path='data/raw/robustad_hg38_5kb_50_600.pkl'
expected_size=3467480869
chunk_bytes=$((64 * 1024 * 1024))
request_bytes=$((1 * 1024 * 1024))
parallel_downloads=${PARALLEL_DOWNLOADS:-8}
part_dir="$path.parts"
mkdir -p "$part_dir"

download_chunk() {
  local start=$1
  local end=$2
  local part=$3
  local expected=$((end - start + 1))
  if [[ -f "$part" ]] && [[ $(stat -f '%z' "$part") -eq "$expected" ]]; then
    return
  fi
  local partial="$part.partial"
  local actual=0
  if [[ -f "$partial" ]]; then
    actual=$(stat -f '%z' "$partial")
    if [[ "$actual" -gt "$expected" ]]; then
      rm -f "$partial"
      actual=0
    fi
  fi
  while [[ "$actual" -lt "$expected" ]]; do
    local segment_start=$((start + actual))
    local segment_end=$((segment_start + request_bytes - 1))
    ((segment_end > end)) && segment_end=$end
    local segment="$partial.next"
    rm -f "$segment"
    curl --silent --show-error --fail --location --retry 12 --retry-all-errors \
      --retry-delay 2 --connect-timeout 30 \
      --range "$segment_start-$segment_end" --output "$segment" "$url"
    local segment_expected=$((segment_end - segment_start + 1))
    local segment_actual
    segment_actual=$(stat -f '%z' "$segment")
    [[ "$segment_actual" -eq "$segment_expected" ]] || return 1
    cat "$segment" >> "$partial"
    rm -f "$segment"
    actual=$(stat -f '%z' "$partial")
  done
  mv "$partial" "$part"
}

if [[ -f "$path" ]]; then
  existing_size=$(stat -f '%z' "$path")
  if [[ "$existing_size" -eq "$expected_size" ]]; then
    .venv/bin/python -c "import pickle; pickle.load(open('$path','rb'))"
    exit 0
  fi
  complete_chunks=$((existing_size / chunk_bytes))
  for ((index = 0; index < complete_chunks; index++)); do
    part=$(printf '%s/part.%06d' "$part_dir" "$index")
    if [[ ! -f "$part" ]]; then
      dd if="$path" of="$part" bs="$chunk_bytes" skip="$index" count=1 status=none
    fi
  done
fi

pids=()
scheduled=0
index=0
for ((start = 0; start < expected_size; start += chunk_bytes)); do
  end=$((start + chunk_bytes - 1))
  ((end >= expected_size)) && end=$((expected_size - 1))
  part=$(printf '%s/part.%06d' "$part_dir" "$index")
  download_chunk "$start" "$end" "$part" &
  pids+=("$!")
  index=$((index + 1))
  if [[ ${#pids[@]} -eq "$parallel_downloads" ]] || [[ $end -eq $((expected_size - 1)) ]]; then
    failed=0
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=1
    done
    [[ "$failed" -eq 0 ]] || { echo "one or more range downloads failed" >&2; exit 1; }
    scheduled=$((scheduled + ${#pids[@]}))
    echo "RobusTAD panel: downloaded or verified $scheduled/$index scheduled chunks"
    pids=()
  fi
done

assembled="$path.assembling"
cat "$part_dir"/part.* > "$assembled"
[[ $(stat -f '%z' "$assembled") -eq "$expected_size" ]] || {
  echo "assembled size mismatch" >&2
  exit 1
}
mv "$assembled" "$path"
.venv/bin/python -c "import pickle; data=pickle.load(open('$path','rb')); assert data['resol'] == 5000"
rm -rf "$part_dir"
echo "RobusTAD reference panel verified"
