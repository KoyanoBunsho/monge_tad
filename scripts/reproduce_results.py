"""Verify and aggregate the deposited results without downloading Hi-C data."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from tad_separator_results import load_complete, CONDITIONS, METRICS

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'artifacts/tad_no_separator_20261029'
AUTOSOMES = {f'chr{i}' for i in range(1, 23)}

def read(path):
    return json.loads(path.read_text())

def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

def near(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f'{label}: {actual} != {expected}')

def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def verify_files():
    path = ROOT / 'provenance/release_manifest.json'
    manifest = read(path)
    for record in manifest['files']:
        source = ROOT / record['path']
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Release file changed or missing: {record["path"]}')
    return len(manifest['files'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/reproduced')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--check-files', action='store_true')
    args = parser.parse_args()
    files_checked = verify_files() if args.check_files else None
    summary = load_complete(CAMPAIGN)
    saved = read(CAMPAIGN / 'final_summary.json')['benchmark']
    if summary['chromosomes'] != saved['chromosomes'] or summary['regions'] != saved['regions']:
        raise ValueError('Chromosome or optimization-region counts differ')
    variants = []
    for condition, methods in summary['summary'].items():
        for method, scores in methods.items():
            for metric, score in scores.items():
                near(score, saved['summary'][condition][method][metric], (condition, method, metric))
            variants.append(dict(condition=condition, method=method, chromosomes=22, **scores))
    grouped = defaultdict(dict)
    for row in csv_rows(ROOT / 'results/per_chromosome_metrics.csv'):
        key = (row['condition'], row['method'])
        if row['chromosome'] in grouped[key]:
            raise ValueError(f'Duplicate chromosome: {key}, {row["chromosome"]}')
        grouped[key][row['chromosome']] = row
    accuracy = csv_rows(ROOT / 'results/quality_all_autosomes.csv')
    expected_methods = set(read(ROOT / 'results/manifest.json')['expected_methods'])
    if {(r['condition'], r['method']) for r in accuracy} != {(c,m) for c in CONDITIONS for m in expected_methods}:
        raise ValueError('Accuracy table does not cover all methods and conditions')
    for row in accuracy:
        key = (row['condition'], row['method'])
        if row['status'] != 'evaluated':
            if any(row[m] for m in METRICS):
                raise ValueError(f'Incomplete run has aggregate accuracy: {key}')
            continue
        records = grouped[key]
        if set(records) != AUTOSOMES or int(row['chromosomes']) != 22:
            raise ValueError(f'Accuracy does not use all 22 autosomes: {key}')
        for metric in METRICS:
            value = statistics.mean(float(r[metric]) for r in records.values())
            near(value, row[metric], (*key, metric))
            if row['method'] == 'Ours':
                near(value, summary['summary'][row['condition']]['Ours'][metric], (*key, metric))
            row[metric] = value
    runtime = csv_rows(ROOT / 'results/runtime.csv')
    saved_runtime = {(r['condition'], r['method']): r for r in read(CAMPAIGN / 'runtime.json')['rows']}
    for row in runtime:
        original = saved_runtime[(row['condition'], row['method'])]
        for key, value in original.items():
            if key not in row:
                continue
            if value is None:
                if row[key] != '':
                    raise ValueError(f'Unexpected runtime field: {key}')
            elif isinstance(value, (int,float)) and not isinstance(value, bool):
                near(row[key], value, ('runtime', row['condition'], row['method'], key))
            elif str(value) != row[key]:
                raise ValueError(f'Runtime metadata mismatch: {key}')
    ratios = []
    for condition in CONDITIONS:
        base = CAMPAIGN / 'pipeline' / condition
        boundary = read(base / 'boundary.json')['shared_boundary_seconds']
        for method, directory in [('Ours','KY-W'),('deepTAD','deepTAD')]:
            total = boundary
            for chromosome in range(1,23):
                record = read(base / 'status' / directory / f'chr{chromosome}.json')
                if record['status'] != 'completed':
                    raise ValueError('Incomplete current timing record')
                total += record['seconds']
            near(total, saved_runtime[condition,method]['complete_seconds'], ('pipeline total',condition,method))
        ours = saved_runtime[condition,'Ours']['complete_seconds']
        for method in sorted(expected_methods - {'Ours'}):
            other = saved_runtime[condition,method]
            if other['complete']:
                seconds = other['complete_seconds']
                ratios.append(dict(condition=condition,method=method,ours_seconds=ours,
                                   previous_seconds=seconds,previous_over_ours=seconds/ours,
                                   ours_over_previous=ours/seconds))
    deposited_ratios = {(r['condition'],r['method']):r for r in read(CAMPAIGN/'runtime.json')['ratios']}
    for row in ratios:
        for key in ('ours_seconds','previous_seconds','previous_over_ours','ours_over_previous'):
            near(row[key], deposited_ratios[row['condition'],row['method']][key], ('runtime ratio',key))

    # Reconstruct the two reciprocal biological comparisons from 44 files.
    parents = []
    for chromosome in range(1,23):
        for fold in (0,1):
            record = read(CAMPAIGN / 'nipbl' / f'chr{chromosome}_fold{fold}.json')
            if not record['complete'] or record['chromosome'] != f'chr{chromosome}' or record['fold'] != fold:
                raise ValueError('Incomplete biological comparison')
            parents.extend(dict(chromosome=f'chr{chromosome}', fold=fold, **r) for r in record['paired'])
    nipbl = read(CAMPAIGN / 'nipbl_summary.json')
    biology = []
    for fold in (0,1,None):
        rows = [r for r in parents if fold is None or r['fold']==fold]
        valid = [r for r in rows if r['delta'] is not None]
        for row in valid:
            near(row['nipbl']-row['control'], row['delta'], 'NIPBL difference')
        result = dict(fold='combined' if fold is None else fold, parents=len(rows), valid_pairs=len(valid),
                      **{key: statistics.mean(r[key] for r in valid) for key in ('control','nipbl','delta')})
        reference = nipbl if fold is None else next(r for r in nipbl['folds'] if r['fold']==fold)
        for key in ('parents','valid_pairs','control','nipbl','delta'):
            near(result[key], reference[key], ('NIPBL',fold,key))
        biology.append(result)
    report = dict(benchmark_chromosomes=summary['chromosomes'], exactness_regions=summary['regions'],
                  evaluated_method_conditions=sum(r['status']=='evaluated' for r in accuracy),
                  total_method_conditions=len(accuracy), biological_chromosome_comparisons=44,
                  biological_parent_comparisons=len(parents), files_checked=files_checked)
    if not args.verify_only:
        destination = args.output_dir.resolve()
        if destination == ROOT/'results' or destination == CAMPAIGN:
            parser.error('Use a separate output directory to preserve the deposited results')
        destination.mkdir(parents=True, exist_ok=True)
        for name, rows in [('accuracy',accuracy),('runtime',runtime),('runtime_ratios',ratios),('objective_variants',variants),
                           ('nipbl',biology),('nipbl_parent_pairs',parents)]:
            write_csv(destination / f'{name}.csv', rows)
        lines=['# Reproduced accuracy table','',
               '| Condition | Method | Chromosomes | Status | TAD-adjusted R² | Intra-inter difference | Both-endpoint CTCF (%) |',
               '|---|---|---:|---|---:|---:|---:|']
        for row in accuracy:
            values = [f'{row[k]:.6g}' if isinstance(row[k],float) else row[k] for k in METRICS]
            lines.append('| '+' | '.join([row['condition'],row['method'],row['chromosomes'],row['status'],*values])+' |')
        (destination/'accuracy.md').write_text('\n'.join(lines)+'\n')
        (destination/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
