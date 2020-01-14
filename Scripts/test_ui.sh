echo python --version

python -m venv web_env
source Web_UI/web_env/bin/activate
python -m pip install -r Web_UI/web_reqs.txt --no-index

pytest ./Web_UI/