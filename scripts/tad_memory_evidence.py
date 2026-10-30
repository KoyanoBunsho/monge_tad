"""Recognize OS memory terminations only for provenance-matched run records."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'artifacts/tad_sigkill_diagnosis_20261026/diagnosis.json'


@lru_cache(maxsize=4)
def evidence_index(path, modified_ns, size):
    data = json.loads(Path(path).read_text())
    return {(r['record'], r['record_sha256']): r for r in data.get('matched', [])
            if r.get('classification') == 'OS memory-pressure termination'}


def confirmed_memory_termination(path):
    if not EVIDENCE.exists() or not path.exists():
        return None
    info = EVIDENCE.stat()
    index = evidence_index(str(EVIDENCE), info.st_mtime_ns, info.st_size)
    relative = str(path.relative_to(ROOT))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return index.get((relative, digest))
