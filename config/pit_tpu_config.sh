#!/bin/bash

# first config
python3 main.py --train_epochs=150 --batch_size=32 --base_learning_rate=5.0e-03 --lr_drop_ratio=0.70 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3
python3 main.py --train_epochs=150 --batch_size=32 --base_learning_rate=5.0e-03 --lr_drop_ratio=0.70 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3
python3 main.py --train_epochs=150 --batch_size=32 --base_learning_rate=5.0e-03 --lr_drop_ratio=0.70 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3

# second config
python3 main.py --train_epochs=150 --batch_size=32 --aug_strategy=random-crop --base_learning_rate=5.0e-03 --lr_drop_ratio=0.70 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3
python3 main.py --train_epochs=150 --batch_size=32 --aug_strategy=random-crop --base_learning_rate=5.0e-03 --lr_drop_ratio=0.70 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3
python3 main.py --train_epochs=150 --batch_size=32 --aug_strategy=random-crop --base_learning_rate=5.0e-03 --lr_drop_ratio=0.70 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3

# third config
python3 main.py --train_epochs=150 --batch_size=32 --aug_strategy=noise --base_learning_rate=5.0e-03 --lr_drop_ratio=0.70 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3
python3 main.py --train_epochs=150 --batch_size=32 --aug_strategy=noise --base_learning_rate=5.0e-03 --lr_drop_ratio=0.70 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3
python3 main.py --train_epochs=150 --batch_size=32 --aug_strategy=noise --base_learning_rate=5.0e-03 --lr_drop_ratio=0.70 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3

# fourth config
python3 main.py --train_epochs=150 --batch_size=32 --aug_strategy=crop_and_noise --base_learning_rate=5.0e-03 --lr_drop_ratio=0.70 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3
python3 main.py --train_epochs=150 --batch_size=32 --aug_strategy=crop_and_noise --base_learning_rate=5.0e-03 --lr_drop_ratio=0.70 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3
python3 main.py --train_epochs=150 --batch_size=32 --aug_strategy=crop_and_noise --base_learning_rate=5.0e-03 --lr_drop_ratio=0.70 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3




