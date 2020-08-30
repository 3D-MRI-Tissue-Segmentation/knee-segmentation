python3 main.py &&
python3 main.py --aug_strategy=random_crop &&
python3 main.py --aug_strategy=center_crop,random_flip &&
python3 main.py --aug_strategy=center_crop,random_noise &&
python3 main.py --aug_strategy=random_crop,random_flip &&
python3 main.py --aug_strategy=random_crop,random_noise &&
python3 main.py --aug_strategy=random_crop,random_flip,random_noise
