# conda env
source /mnt/miniconda/etc/profile.d/conda.sh
conda activate PY3

# echo python version for debugging
python --version

pip install --upgrade pip --quiet
pip install -r bolt/bolt_requirements.txt --quiet
