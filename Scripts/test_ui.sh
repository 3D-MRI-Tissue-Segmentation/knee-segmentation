echo python --version

cd ../Web_UI

python -m venv web_env
source web_env/bin/activate
python -m pip install -r web_reqs.txt

pytest