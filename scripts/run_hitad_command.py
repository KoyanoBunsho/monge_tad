"""Run released HiTAD with the removed scipy.stats.itemfreq API restored."""
from pathlib import Path
import runpy
import sys
import numpy as np
import scipy.stats

if not hasattr(scipy.stats, "itemfreq"):
    def itemfreq(values):
        items, counts = np.unique(values, return_counts=True)
        return np.column_stack((items, counts))
    scipy.stats.itemfreq = itemfreq

program = Path(__file__).resolve().parents[1] / "third_party/TADLib/scripts/hitad"
sys.argv[0] = str(program)
runpy.run_path(str(program), run_name="__main__")
