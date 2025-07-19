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

        # ensure patterns are deduplicated
        self.assertEqual(len(data), 30240)

    def test_max_patterns_limit(self):
        data = load_shift_patterns('examples/shift_config_v2.json', slot_duration_minutes=30, max_patterns=10)
        self.assertEqual(len(data), 10)

if __name__ == '__main__':
    unittest.main()
