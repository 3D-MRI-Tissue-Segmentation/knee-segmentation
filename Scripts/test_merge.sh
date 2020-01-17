python -m venv env
source env/bin/activate

python -m pip install -r Segmentation/ml_reqs.txt

python -m pip install -r UI/web_reqs.txt

pytest
