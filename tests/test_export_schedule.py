import sys
from types import ModuleType
from pathlib import Path
import importlib.util
import pandas as pd

# stub heavy optional modules
class _Dummy(ModuleType):
    def __getattr__(self, item):
        return lambda *a, **k: None

for name in ["streamlit", "seaborn"]:
    sys.modules.setdefault(name, _Dummy(name))
sys.modules.setdefault("matplotlib", _Dummy("matplotlib"))
sys.modules.setdefault("matplotlib.pyplot", _Dummy("matplotlib.pyplot"))
pywork_sched = ModuleType("pyworkforce.scheduling")
pywork_sched.MinAbsDifference = None
sys.modules.setdefault("pyworkforce.scheduling", pywork_sched)
pywork = ModuleType("pyworkforce")
pywork.scheduling = pywork_sched
sys.modules.setdefault("pyworkforce", pywork)

# ensure real pandas is used
if isinstance(sys.modules.get("pandas"), ModuleType) and not hasattr(sys.modules["pandas"], "DataFrame"):
    del sys.modules["pandas"]

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "generador_turnos_2025_cnx_BACKUP_F_FIRST_P_LAST (1).py"
module = ModuleType("loader_export")
code_parts = []
with open(SCRIPT_PATH, "r") as fh:
    section = []
    capture = True
    for line in fh:
        if capture and line.startswith("try:"):
            break
        section.append(line)
    code_parts.extend(section)

with open(SCRIPT_PATH, "r") as fh:
    record = False
    for line in fh:
        if line.startswith("def _extract_start_hour"):
            record = True
        if record:
            if line.startswith("def solve_in_chunks_optimized"):
                break
            code_parts.append(line)

code = "".join(code_parts)
exec(code, module.__dict__)

export_detailed_schedule = module.export_detailed_schedule
_build_pattern = module._build_pattern


def test_export_start_hour():
    shift_name = "FT_10h_4dias_8h_1dia_08.0_01234_10_10_10_8_10"
    pattern = _build_pattern([0], [8], 8.0, 0, 0, 0, 1)
    assignments = {shift_name: 1}
    coverage = {shift_name: pattern}

    excel_bytes = export_detailed_schedule(assignments, coverage)
    assert excel_bytes is not None

    from openpyxl import load_workbook
    from io import BytesIO
    wb = load_workbook(BytesIO(excel_bytes))
    sheet = wb["Horarios_Semanales"]
    # first data row after header
    horario = sheet["C2"].value
    assert horario.startswith("08:00")
