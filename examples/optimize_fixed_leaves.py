"""Optimize a small synthetic example; no data download is needed."""
import numpy as np
from faster_cky.decoder import full_cky_min, knuth_cky_min
from faster_cky.deeptad_monge import add_pair_potentials, monge_hierarchy_costs, quadratic_duration_costs

# Half-open coordinates in units of 100 kb; gaps between leaves are permitted.
leaves = [(0, 2), (3, 6), (6, 8), (9, 13)]
mean_oe = np.array([[1, 1.2, .4, .1], [1.2, 1, .7, .3], [.4, .7, 1, 1.1], [.1, .3, 1.1, 1]])
length_cost = quadratic_duration_costs(13, quadratic=1 / 40**2)
cost = monge_hierarchy_costs(leaves, np.zeros(len(leaves)-1), length_cost)
cost = add_pair_potentials(cost, np.triu(np.maximum(1-mean_oe, 0), 1))
result = knuth_cky_min(cost)
reference = full_cky_min(cost)
assert np.isclose(result.score, reference.score) and result.spans == reference.spans
print("Objective value:", result.score)
print("Parent domains as half-open leaf ranges:", sorted(result.spans))
