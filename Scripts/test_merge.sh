echo python --version

python -m venv env
source env/bin/activate

cd Segmentation_Models
python -m pip install -r ml_reqs.txt

cd ../Web_UI
python -m pip install -r web_reqs.txt

cd ..
pytest