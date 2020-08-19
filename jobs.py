import subprocess
import numpy as np

teacher_teacher_overlaps = np.linspace(0, 1, 50)
rotations = np.arccos(teacher_teacher_overlaps)

for o, rotation in enumerate(rotations):
    subprocess.Popen(
        f"python experiments/main.py --tro '[{round(rotation, 4)}, 0]' --en {o}", shell=True)
