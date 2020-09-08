python3 main.py --aug_strategy=center_crop &&
python3 main.py --aug_strategy=random_crop &&
python3 main.py --aug_strategy=center_crop,random_noise && 
python3 main.py --aug_strategy=random_crop,random_noise &&
python3 main.py --model_architecture=unet++ --aug_strategy=center_crop &&
python3 main.py --model_architecture=unet++ --aug_strategy=random_crop &&
python3 main.py --model_architecture=unet++ --aug_strategy=center_crop,random_noise &&
python3 main.py --model_architecutre=unet++ --aug_strategy=random_crop,random_noise 

