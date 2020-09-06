import turibolt as bolt
import numpy as np

import sys
print(sys.path)

teacher_teacher_overlaps = np.linspace(0, 1, 50)
rotations = np.arccos(teacher_teacher_overlaps)

config = bolt.get_current_config()

# Update is_parent field for submitting child tasks
config['is_parent'] = False

print(config)

for o, rotation in enumerate(rotations):
    config[
        'command'] = 'source /mnt/miniconda/etc/profile.d/conda.sh && conda activate PY3 && export ' \
                     'PYTHONPATH="$PYTHONPATH:$PWD" && python experiments/main.py --bolt ' \
                     f"--tro '[0, {round(rotation, 4)}]' --en {o} && python utils/zip_bolt_dir.py"
    bolt.submit(config)
