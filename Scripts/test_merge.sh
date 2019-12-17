echo python --version

python -m venv env
source env/bin/activate

python -m pip install -r Segmentation_Models/ml_reqs.txt

python -m pip install -r Web_UI/web_reqs.txt

pytest
