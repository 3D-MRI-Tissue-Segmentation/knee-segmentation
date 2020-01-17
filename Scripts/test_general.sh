cd ..
python -m venv env
source env/bin/activate
python -m pip install -r reqs.txt

python -m pytest test_setup.py