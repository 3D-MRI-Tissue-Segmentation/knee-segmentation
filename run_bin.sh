python3 main.py --flagfile=config/train_tpu_unet_1_bin.cfg &&
python3 main.py --flagfile=config/train_tpu_unet_2_bin.cfg &&
python3 main.py --flagfile=config/train_tpu_unet_3_bin.cfg &&
python3 main.py --flagfile=config/train_tpu_unet_4_bin.cfg && 
python3 main.py --flagfile=config/train_tpu_unet_1_bin.cfg --aug_strategy=random_crop && 
python3 main.py --flagfile=config/train_tpu_unet_2_bin.cfg --aug_strategy=random_crop &&
python3 main.py --flagfile=config/train_tpu_unet_3_bin.cfg --aug_strategy=random_crop &&
python3 main.py --flagfile=config/train_tpu_unet_4_bin.cfg --aug_strategy=random_crop 
