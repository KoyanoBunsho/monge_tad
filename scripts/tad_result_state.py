"""Identify completed new TAD outputs and provenance-matched metric caches."""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/tad_all_chromosomes_20261022'
EXPAND = ROOT/'artifacts/tad_method_expansion_20261024'
EXTRA = ROOT/'artifacts/tad_additional_metrics_20261024'
TERMINAL = {'completed', 'timeout', 'memory_error', 'runtime_error'}
KEYS = {'OnTAD':'ontad', 'JOnTADS':'jontads', 'deDoc2':'dedoc2',
        'SuperTAD':'supertad-binary', 'SuperTAD-Fast':'supertad-fast',
        'TADtree':'tadtree', 'SpectralTAD':'spectraltad', 'HiTAD':'hitad'}


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


@lru_cache(maxsize=8192)
def _hash(path, modified, size):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(path):
    s = path.stat()
    return _hash(str(path), s.st_mtime_ns, s.st_size)


def key(condition, method):
    if method == 'RobusTAD':
        return 'robustad-panel' if condition.endswith('_5kb') else 'robustad-no-panel'
    return KEYS[method]


def records(condition, method):
    """Return all 22 required units, including missing and failed executions."""
    if method in ['Ours', 'deepTAD'] or (method in ['OnTAD','JOnTADS','deDoc2'] and not condition.endswith('_5kb')):
        return []
    result = []
    for n in range(1,23):
        chrom = f'chr{n}'
        work = (ROOT/'artifacts/tad_runtime_comparison_20261024/hitad'/condition if method == 'HiTAD'
                else EXPAND/condition/chrom/key(condition, method))
        run = work/'status.json'
        output = work/('domains.bed' if method == 'HiTAD' else 'domains.json')
        data = read(run)
        result.append(dict(condition=condition, method=method, chromosome=chrom,
                           status=data.get('status','not_run'), run_path=run, output_path=output,
                           cache_path=EXTRA/condition/key(condition,method)/(chrom+'.json')))
    return result


def fingerprint(record):
    reference = BASE/record['condition']/'evaluation'/record['chromosome']/'domains.tsv'
    protocol = [ROOT/'scripts/evaluate_tad_completed_outputs.py', ROOT/'scripts/tad_result_state.py',
                ROOT/'scripts/recompute_published_tad_metrics.py',
                ROOT/'src/faster_cky/tad_published_metrics.py',
                ROOT/'src/faster_cky/deeptad_evaluation.py']
    files = [record['output_path'], record['run_path'], reference, *protocol]
    return hashlib.sha256(json.dumps({str(p.relative_to(ROOT)):digest(p) for p in files},sort_keys=True).encode()).hexdigest()


def cached(record):
    value = read(record['cache_path'])
    if record['status'] != 'completed' or not record['output_path'].exists():
        return None
    return value if value.get('complete') and value.get('fingerprint') == fingerprint(record) else None


def finalized_panel(condition, method):
    items = records(condition, method)
    if not items or any(r['status'] not in TERMINAL for r in items):
        return None
    successful = [r for r in items if r['status'] == 'completed']
    if not successful:
        return None
    values = [cached(r) for r in successful]
    if any(v is None for v in values):
        return None
    return values
