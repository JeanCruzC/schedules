import unittest
import numpy as np
import importlib.util
from pathlib import Path
from types import ModuleType
import sys

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
load_shift_patterns = module.load_shift_patterns
score_and_filter_patterns = module.score_and_filter_patterns
get_smart_start_hours = module.get_smart_start_hours

class LoaderTest(unittest.TestCase):
    def test_v1_format(self):
        data = load_shift_patterns('examples/shift_config.json', slot_duration_minutes=60)
        self.assertTrue(data)
        for arr in data.values():
            self.assertEqual(arr.shape, (7 * 24,))

    def test_v2_format(self):
        data = load_shift_patterns('examples/shift_config_v2.json', slot_duration_minutes=30)
        self.assertTrue(data)
        for arr in data.values():
            self.assertEqual(arr.shape, (7 * 48,))

    def test_max_patterns_limit(self):
        data = load_shift_patterns('examples/shift_config_v2.json', slot_duration_minutes=30, max_patterns=10)
        self.assertEqual(len(data), 10)

    def test_cross_midnight_pattern(self):
        pat = module._build_pattern([0], [4], 23, 0, 2, 2, 1)
        mat = pat.reshape(7, 24)
        self.assertEqual(mat[0, 23], 1)
        self.assertEqual(mat[1, 0], 1)
        self.assertEqual(mat[1, 1], 1)
        self.assertEqual(mat[1, 2], 1)

    def test_score_filtering(self):
        demand = np.zeros((7, 24))
        demand[0, 12] = 1
        cfg = {
            "shifts": [
                {"name": "PEAK", "pattern": {"work_days": [0], "segments": [1]}, "break": 0},
                {"name": "OFF", "pattern": {"work_days": [1], "segments": [1]}, "break": 0},
            ]
        }
        data = load_shift_patterns(
            cfg,
            start_hours=[12],
            slot_duration_minutes=60,
            demand_matrix=demand,
            keep_percentage=0.5,
        )
        self.assertEqual(len(data), 1)
        self.assertIn("PEAK_12.0_0_1", data)

    def test_smart_start_hours(self):
        demand = np.zeros((7, 24))
        demand[:, 8:11] = 5
        hours = get_smart_start_hours(demand, max_hours=5)
        self.assertTrue(8.0 in hours or 9.0 in hours)
        self.assertLessEqual(len(hours), 5)

    def test_max_patterns_per_shift(self):
        cfg = {"shifts": [{"name": "A", "pattern": {"work_days": [0], "segments": [1]}, "break": 0}]}
        data = load_shift_patterns(cfg, start_hours=[0, 1, 2], slot_duration_minutes=60, max_patterns_per_shift=2)
        self.assertLessEqual(len(data), 2)

    def test_efficiency_bonus(self):
        patterns = {
            "LONG": module._build_pattern([0], [2], 0, 0, 0, 0, 1),
            "SHORT": module._build_pattern([0], [1], 0, 0, 0, 0, 1),
        }
        demand = np.zeros((7, 24))
        demand[0, 0] = 1
        result = score_and_filter_patterns(patterns, demand, keep_percentage=0.5, efficiency_bonus=10)
        self.assertEqual(set(result.keys()), {"SHORT"})

if __name__ == '__main__':
    unittest.main()

