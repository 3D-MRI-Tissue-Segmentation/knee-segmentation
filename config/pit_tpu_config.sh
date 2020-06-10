#!/bin/bash

# first config
python3 main.py --batch_size=32 --base_learning_rate=6.4e-04 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3

python3 main.py --batch_size=32 --base_learning_rate=6.4e-04 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=7

python3 main.py --batch_size=32 --base_learning_rate=6.4e-04 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=512 --kernel_size_initial_conv=3

# second config
python3 main.py --aug_strategy=random-crop --batch_size=32 --base_learning_rate=6.4e-04 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3

python3 main.py --aug_strategy=random-crop --batch_size=32 --base_learning_rate=6.4e-04 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=7

python3 main.py --aug_strategy=random-crop --batch_size=32 --base_learning_rate=6.4e-04 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=512 --kernel_size_initial_conv=3

# third config
python3 main.py --aug_strategy=noise --batch_size=32 --base_learning_rate=6.4e-04 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3

python3 main.py --aug_strategy=noise --batch_size=32 --base_learning_rate=6.4e-04 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=7

python3 main.py --aug_strategy=noise --batch_size=32 --base_learning_rate=6.4e-04 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=512 --kernel_size_initial_conv=3

# fourth config
python3 main.py --aug_strategy=crop_and_noise --batch_size=32 --base_learning_rate=6.4e-04 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=3

python3 main.py --aug_strategy=crop_and_noise --batch_size=32 --base_learning_rate=6.4e-04 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=256 --kernel_size_initial_conv=7

python3 main.py --aug_strategy=crop_and_noise --batch_size=32 --base_learning_rate=6.4e-04 --lr_warmup_epochs=1 --model_architecture=deeplabv3 --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=pit-tpu  --num_filters_atrous=512 --kernel_size_initial_conv=3




