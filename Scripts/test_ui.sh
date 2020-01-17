echo python --version
cd ..

python -m venv ui_env
source UI/ui_env/bin/activate
python -m pip install -r UI/ui_reqs.txt --no-index

python -m pytest ./UI/