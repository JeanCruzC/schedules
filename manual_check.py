import numpy as np

# Simulate work hours for a PT shift starting at 22:00 lasting 6h
work_hours = np.array([22, 23, 0, 1, 2, 3])
start_idx = 22
end_idx = (int(work_hours[-1]) + 1) % 24
next_day = end_idx <= start_idx
horario = f"{start_idx:02d}:00-{end_idx:02d}:00" + ("+1" if next_day else "")
print(horario)
