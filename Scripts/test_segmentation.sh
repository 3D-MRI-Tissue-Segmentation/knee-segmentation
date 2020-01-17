cd ..
python -m venv ml_env_test
source ml_env_test/bin/activate
python -m pip install -r Segmentation/ml_reqs.txt

python -m pytest ./Segmentation/