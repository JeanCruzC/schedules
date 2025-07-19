import json
import numpy as np
from itertools import permutations, combinations
from typing import Dict, List, Iterable, Union


def _build_pattern(days: Iterable[int], durations: Iterable[int], start_hour: float,
                   break_len: float, break_from_start: float, break_from_end: float) -> np.ndarray:
    """Return flattened 7x24 matrix for given days/durations."""
    pattern = np.zeros((7, 24), dtype=np.int8)
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


def load_shift_patterns(
    cfg: Union[str, dict], *, start_hours: Iterable[float] | None = None,
    break_from_start: float = 2.0, break_from_end: float = 2.0,
    slot_duration_minutes: int = 30
) -> Dict[str, np.ndarray]:
    """Parse JSON shift configuration and return pattern dictionary."""
    if isinstance(cfg, str):
        with open(cfg, "r") as fh:
            data = json.load(fh)
    else:
        data = cfg

    shifts_coverage: Dict[str, np.ndarray] = {}
    for shift in data.get("shifts", []):
        name = shift.get("name", "SHIFT")
        pat = shift.get("pattern", {})
        brk = shift.get("break", 0)

        slot_min = shift.get("slot_duration_minutes", slot_duration_minutes)
        step = slot_min / 60
        sh_hours = list(start_hours) if start_hours is not None else list(np.arange(0, 24, step))

        work_days = pat.get("work_days", [])
        segments_spec = pat.get("segments", [])
        segments: List[int] = []
        for seg in segments_spec:
            if isinstance(seg, dict):
                hours = seg.get("hours")
                count = seg.get("count", 1)
                if hours is None:
                    continue
                segments.extend([int(hours)] * int(count))
            else:
                segments.append(int(seg))

        if isinstance(work_days, int):
            day_candidates = range(7)
            day_combos = combinations(day_candidates, work_days)
        else:
            day_combos = combinations(work_days, min(len(segments), len(work_days)))

        if isinstance(brk, dict):
            if brk.get("enabled", False):
                brk_len = brk.get("length_minutes", 0) / 60
                brk_start = brk.get("earliest_after_start", 0) / 60
                brk_end = brk.get("latest_before_end", 0) / 60
            else:
                brk_len = 0
                brk_start = break_from_start
                brk_end = break_from_end
        else:
            brk_len = float(brk)
            brk_start = break_from_start
            brk_end = break_from_end

        for days_sel in day_combos:
            for perm in set(permutations(segments, len(days_sel))):
                for sh in sh_hours:
                    pattern = _build_pattern(days_sel, perm, sh, brk_len, brk_start, brk_end)
                    day_str = ''.join(map(str, days_sel))
                    seg_str = '_'.join(map(str, perm))
                    shift_name = f"{name}_{sh:04.1f}_{day_str}_{seg_str}"
                    shifts_coverage[shift_name] = pattern

    return shifts_coverage
