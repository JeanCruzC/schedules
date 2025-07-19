import unittest
import numpy as np
from types import ModuleType
import sys
from pathlib import Path

for name in ["streamlit", "seaborn", "pandas"]:
    sys.modules.setdefault(name, ModuleType(name))
sys.modules.setdefault("matplotlib", ModuleType("matplotlib"))
sys.modules.setdefault("matplotlib.pyplot", ModuleType("matplotlib.pyplot"))
pywork_sched = ModuleType("pyworkforce.scheduling")
pywork_sched.MinAbsDifference = None
sys.modules.setdefault("pyworkforce.scheduling", pywork_sched)
pywork = ModuleType("pyworkforce")
pywork.scheduling = pywork_sched
sys.modules.setdefault("pyworkforce", pywork)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "generador_turnos_2025_cnx_BACKUP_F_FIRST_P_LAST (1).py"
module = ModuleType("loader")
with open(SCRIPT_PATH, "r") as fh:
    lines = []
    for line in fh:
        if line.startswith("try:"):
            break
        lines.append(line)
    code = "".join(lines)
exec(code, module.__dict__)

get_smart_start_hours = module.get_smart_start_hours
score_and_filter_patterns = module.score_and_filter_patterns
_build_pattern = module._build_pattern

class UtilTests(unittest.TestCase):
    def test_get_smart_start_hours(self):
        demand = np.zeros((7, 24), dtype=int)
        demand[:, 9:17] = 1
        hours = get_smart_start_hours(demand, step_hours=1.0, margin_hours=1)
        self.assertEqual(hours[0], 8)
        self.assertEqual(hours[-1], 17)
        self.assertEqual(len(hours), 10)

    def test_score_and_filter_patterns(self):
        demand = np.zeros((7, 24), dtype=int)
        demand[:, 8:16] = 1
        p1 = _build_pattern([0], [8], 8, 0, 0, 0, 1)
        p2 = _build_pattern([0], [8], 12, 0, 0, 0, 1)
        pats = {"p1": p1, "p2": p2}
        best = score_and_filter_patterns(pats, demand, limit=1)
        self.assertEqual(list(best.keys()), ["p1"])

if __name__ == '__main__':
    unittest.main()
