import json
import numpy as np
from itertools import permutations, combinations
from typing import Dict, List, Iterable, Union


def _build_pattern(days: Iterable[int], durations: Iterable[int], start_hour: float,
                   break_len: float, break_from_start: float, break_from_end: float) -> np.ndarray:
    """Return flattened 7x24 matrix for given days/durations."""
    pattern = np.zeros((7, 24))
    for day, dur in zip(days, durations):
        for h in range(int(dur)):
            idx = int(start_hour + h) % 24
            pattern[day, idx] = 1
        if break_len:
            b_start = int(start_hour + break_from_start) % 24
            b_end = int(start_hour + dur - break_from_end) % 24
            if b_start < b_end:
                b_hour = b_start + (b_end - b_start) // 2
            else:
                b_hour = b_start
            for b in range(int(break_len)):
                pattern[day, (b_hour + b) % 24] = 0
    return pattern.flatten()


def load_shift_patterns(cfg: Union[str, dict], *, start_hours: Iterable[float] = None,
                         break_from_start: float = 2.0, break_from_end: float = 2.0) -> Dict[str, np.ndarray]:
    """Parse JSON shift configuration and return pattern dictionary."""
    if isinstance(cfg, str):
        with open(cfg, 'r') as fh:
            data = json.load(fh)
    else:
        data = cfg

    start_hours = list(start_hours) if start_hours is not None else list(np.arange(0, 24, 0.5))
    shifts_coverage: Dict[str, np.ndarray] = {}
    for shift in data.get("shifts", []):
        name = shift.get("name", "SHIFT")
        pat = shift.get("pattern", {})
        work_days: List[int] = pat.get("work_days", [])
        segments: List[int] = pat.get("segments", [])
        brk = shift.get("break", 0)
        if not work_days or not segments:
            continue
        for days_sel in combinations(work_days, min(len(segments), len(work_days))):
            for perm in set(permutations(segments, len(days_sel))):
                for sh in start_hours:
                    pattern = _build_pattern(days_sel, perm, sh, brk, break_from_start, break_from_end)
                    day_str = ''.join(map(str, days_sel))
                    seg_str = '_'.join(map(str, perm))
                    shift_name = f"{name}_{sh:04.1f}_{day_str}_{seg_str}"
                    shifts_coverage[shift_name] = pattern
    return shifts_coverage
