import unittest
import numpy as np
import importlib.util
from pathlib import Path
from types import ModuleType
import sys

class Dummy(ModuleType):
    def __getattr__(self, attr):
        return lambda *a, **k: self

for name in ["streamlit", "seaborn", "pandas"]:
    sys.modules.setdefault(name, Dummy(name))
sys.modules.setdefault("matplotlib", ModuleType("matplotlib"))
sys.modules.setdefault("matplotlib.pyplot", ModuleType("matplotlib.pyplot"))
pywork_sched = ModuleType("pyworkforce.scheduling")
pywork_sched.MinAbsDifference = None
sys.modules.setdefault("pyworkforce.scheduling", pywork_sched)
pywork = ModuleType("pyworkforce")
pywork.scheduling = pywork_sched
sys.modules.setdefault("pyworkforce", pywork)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "generador_turnos_2025_cnx_BACKUP_F_FIRST_P_LAST (1).py"
module = ModuleType("solver")
with open(SCRIPT_PATH, "r") as fh:
    file_lines = fh.readlines()

idx = next(i for i, line in enumerate(file_lines) if line.startswith("try:"))
pre_code = "".join(file_lines[:idx])

start = next(i for i, l in enumerate(file_lines) if l.startswith("def score_pattern"))
end = next(i for i, l in enumerate(file_lines) if l.startswith("def optimize_with_precision_targeting"))
code = pre_code + "".join(file_lines[start:end])
exec(code, module.__dict__)

# provide missing globals used by optimizer
module.use_ft = True
module.use_pt = True
module.agent_limit_factor = 10
module.excess_penalty = 1
module.peak_bonus = 1
module.critical_bonus = 1
module.TIME_SOLVER = 10
module.PULP_AVAILABLE = True
module.optimization_profile = "JEAN"
module.shifts_coverage = {}
import pulp
module.pulp = pulp
module.optimize_schedule_greedy = lambda a, b: ({}, "GREEDY")

load_shift_patterns = module.load_shift_patterns
optimize_chunked = module.optimize_with_phased_strategy_chunked

class ChunkedOptimizerTest(unittest.TestCase):
    def test_chunked_optimizer_runs(self):
        shifts = load_shift_patterns('examples/shift_config.json', slot_duration_minutes=60, max_patterns=4)
        demand = np.ones((7, 24), dtype=float)
        assignments, method = optimize_chunked(shifts, demand, chunk_size=2)
        self.assertIsInstance(assignments, dict)
        self.assertTrue(method.startswith('CHUNKED'))

if __name__ == '__main__':
    unittest.main()
