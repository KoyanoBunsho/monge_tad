"""Read the complete experiment with separator penalties removed."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "artifacts/tad_no_separator_20261029"
CONDITIONS = ("gm12878_25kb", "gm12878_10kb", "k562_25kb", "imr90_25kb",
              "gm12878_5kb", "k562_5kb", "imr90_5kb")
METRICS = ("r2", "quality", "ctcf_percent")
VARIANTS = ("Ours", "No length", "Length only", "With separators", "Greedy", "Matched count")


def load_complete(campaign=CAMPAIGN):
    campaign = Path(campaign)
    summaries = {}
    regions = 0
    for condition in CONDITIONS:
        resolution = int(condition.split("_")[1].removesuffix("kb")) * 1000
        rows = []
        for n in range(1, 23):
            path = campaign / "benchmark" / condition / f"chr{n}.json"
            value = json.loads(path.read_text())
            if not value.get("complete") or value["condition"] != condition or value["chromosome"] != f"chr{n}":
                raise ValueError(f"Incomplete or mismatched chromosome record: {path}")
            objective = value["objective"]
            if objective["name"] != "no_separator" or any(objective[k] != 0 for k in (
                    "probability_weight", "cosine_weight", "direct_separator_weight", "separator_pair_weight", "profile_weight")):
                raise ValueError(f"Incorrect objective: {path}")
            if (objective["pair_weight"] != 1 or objective["pair_transform"] != "hinge"
                    or objective["duration_scale"] != 4000000 / resolution):
                raise ValueError(f"Contact or length penalty differs from the manuscript: {path}")
            expected_selector = dict(boundary_feature="cosine_dissimilarity", boundary_reduction="minimum",
                                     boundary_weight=1.0, missing_endpoint_value=0.0,
                                     cardinality_mode="fixed_ratio_region", parent_ratio=351/1354,
                                     maximum_parent_bins=3000000//resolution)
            if value["selector"] != expected_selector:
                raise ValueError(f"Domain selection settings changed: {path}")
            for variant in VARIANTS:
                for metric in METRICS:
                    score = value["scores"][variant][metric]
                    if score is None or not math.isfinite(score):
                        raise ValueError(f"Undefined score: {path}, {variant}, {metric}")
            if not all(r["score_equal"] and r["tree_equal"] for r in value["regions"]):
                raise ValueError(f"Exactness check failed: {path}")
            if not all(r["equal"] for r in value["validation"].values()):
                raise ValueError(f"Previous ablation reproduction failed: {path}")
            regions += len(value["regions"])
            rows.append(value)
        summaries[condition] = {
            variant: {metric: mean(row["scores"][variant][metric] for row in rows) for metric in METRICS}
            for variant in VARIANTS
        }
    return dict(complete=True, chromosomes=154, regions=regions, summary=summaries)
