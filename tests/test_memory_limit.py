import unittest
from types import ModuleType
from pathlib import Path
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

psutil_mod = ModuleType("psutil")
class VMem:
    def __init__(self, avail):
        self.available = avail
psutil_mod.virtual_memory = lambda: VMem(512 * 1024 * 1024)
sys.modules["psutil"] = psutil_mod

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

memory_limit_patterns = module.memory_limit_patterns

class MemoryLimitTest(unittest.TestCase):
    def test_dynamic_limit(self):
        limit = memory_limit_patterns(24)
        expected = (512 * 1024 * 1024) // (7 * 24)
        self.assertEqual(limit, expected)

if __name__ == "__main__":
    unittest.main()
