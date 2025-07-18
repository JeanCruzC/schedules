import json
import itertools
import numpy as np
from typing import Dict, List, Iterable, Tuple

def load_config(path_or_file):
    """Load JSON configuration from a path or file-like object."""
    if hasattr(path_or_file, "read"):
        return json.load(path_or_file)
    with open(path_or_file, "r", encoding="utf-8") as f:
        return json.load(f)

def assign_segments(days: List[int], segments: List[int]) -> Iterable[Dict[int, int]]:
    """Generate all assignments of hours to the selected days."""
    if not segments:
        yield {}
        return
    first, rest = segments[0], segments[1:]
    for i, day in enumerate(days):
        for sub in assign_segments(days[:i] + days[i+1:], rest):
            sub = sub.copy()
            sub[day] = first
            yield sub

def build_pattern(assignment: Dict[int, int], slots_per_day: int, slots_per_hour: int, break_cfg: Dict) -> np.ndarray:
    """Build weekly pattern with breaks applied."""
    pattern = np.zeros((7, slots_per_day), dtype=int)
    break_len = int(break_cfg.get("duration", 0) * slots_per_hour)
    brk_start = break_cfg.get("from_start", 0)
    brk_end = break_cfg.get("from_end", 0)

    for day, hours in assignment.items():
        start_slot = 0
        end_slot = start_slot + int(hours * slots_per_hour)
        pattern[day, start_slot:end_slot] = 1
        if break_len > 0 and end_slot - start_slot > break_len:
            earliest = start_slot + int(brk_start * slots_per_hour)
            latest = end_slot - int(brk_end * slots_per_hour) - break_len
            if latest < earliest:
                brk_slot = earliest
            else:
                brk_slot = earliest + (latest - earliest) // 2
            pattern[day, brk_slot:brk_slot + break_len] = 0
    return pattern

def generate_patterns(json_path) -> Dict[str, List[int]]:
    """Generate flattened weekly patterns from configuration file."""
    cfg = load_config(json_path)
    slots_per_hour = cfg.get("slots_per_hour", 1)
    slots_per_day = cfg.get("slots_per_day", 24 * slots_per_hour)
    default_break = cfg.get("break", {})
    patterns = {}

    for shift in cfg.get("shifts", []):
        name = shift.get("name", "shift")
        days = shift.get("days", [])
        segments = shift.get("segments", [])
        brk_cfg = default_break.copy()
        brk_cfg.update(shift.get("break", {}))
        for idx, assignment in enumerate(assign_segments(days, segments)):
            patt = build_pattern(assignment, slots_per_day, slots_per_hour, brk_cfg)
            patterns[f"{name}_{idx}"] = patt.flatten().tolist()
    return patterns
