"""Persist resource-based comparison exclusions independently of result files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPERTAD_FAST_CAMPAIGN = ROOT / 'artifacts/tad_supertad_fast_campaign_20261026'


def supertad_fast_enabled():
    path = SUPERTAD_FAST_CAMPAIGN / 'exclusion.json'
    return not (json.loads(path.read_text()).get('excluded', False) if path.exists() else False)
