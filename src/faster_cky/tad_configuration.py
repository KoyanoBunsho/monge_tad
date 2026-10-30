"""Recorded objective and domain-selection settings."""
from .deeptad_pair_ablation import Objective, ParentSelector

OBJECTIVE = Objective(
    name="full",
    pair_weight=1.0,
    probability_weight=0.0,
    cosine_weight=1.0,
    duration_scale=160.0,
    pair_transform="hinge",
    separator_pair_weight=1.0,
    separator_pair_aggregation="maximum",
    direct_separator_weight=1.0,
)
SELECTOR = ParentSelector(
    boundary_feature="cosine_dissimilarity",
    boundary_reduction="minimum",
    boundary_weight=1.0,
    cardinality_mode="fixed_ratio_region",
    parent_ratio=0.259231905465288,
    maximum_parent_bins=120,
)

