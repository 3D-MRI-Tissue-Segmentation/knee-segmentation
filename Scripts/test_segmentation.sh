echo python --version

cd Segmentation_Models

python -m venv ml_env
source ml_env/bin/activate
python -m pip install -r ml_reqs.txt

pytest