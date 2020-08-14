import subprocess
import numpy as np

teacher_teacher_overlaps = np.linspace(0, np.pi, 50)

for o, overlap in enumerate(teacher_teacher_overlaps):
    subprocess.Popen(
        f"python experiments/main.py --tro '[{round(overlap, 4)}, 0]' --en {o}", shell=True)
