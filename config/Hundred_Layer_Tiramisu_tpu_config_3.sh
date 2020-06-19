# Hundred Layer Tiramisu tpu training configuration with augumentation:
# Run date: 12.06.2020 (starting in the afternoon)

# 1. 
python3 main.py --aug_strategy=random_crop --batch_size=8 --base_learning_rate=1.0e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.5 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=lapo-tpu


# 2.
python3 main.py --aug_strategy=noise --batch_size=8 --base_learning_rate=1.0e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.5 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=lapo-tpu


# 3.
python3 main.py --aug_strategy=crop_and_noise --batch_size=8 --base_learning_rate=1.0e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.5 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=lapo-tpu

Hundred_Layer_Tiramisu_tpu_augmentation_config.sh
# Hundred Layer Tiramisu tpu training configuration with augumentation:
# Run date: 12.06.2020 (starting in the afternoon)

# 1. 
python3 main.py --aug_strategy=random_crop --batch_size=8 --base_learning_rate=1.0e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.5 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=lapo-tpu


# 2.
python3 main.py --aug_strategy=noise --batch_size=8 --base_learning_rate=1.0e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.5 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=lapo-tpu


# 3.
python3 main.py --aug_strategy=crop_and_noise --batch_size=8 --base_learning_rate=1.0e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.5 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=lapo-tpu

