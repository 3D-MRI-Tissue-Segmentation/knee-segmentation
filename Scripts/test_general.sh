echo python --version

python -m venv env
source env/bin/active
python -m pip install -r reqs.txt

pytest test_setup.py