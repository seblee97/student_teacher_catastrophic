import numpy as np

with open("job_lists/relu_vs_v.txt", "w") as f:
    teacher_teacher_overlaps = np.linspace(0, 1, 50)
    rotations = np.arccos(teacher_teacher_overlaps)
    for o, rotation in enumerate(rotations):
        f.write(f"python main.py --en {o} --tro '[{round(rotation, 4)}, 0]'\n")
