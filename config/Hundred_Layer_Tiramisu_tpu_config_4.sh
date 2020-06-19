# Hundred Layer Tiramisu tpu training configuration: 
# Run date: 14.06.2020 (during the day)

# 1. 
python3 main.py --batch_size=8 --base_learning_rate=4.0e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.8 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=lapo-tpu


# 2.
python3 main.py --batch_size=8 --base_learning_rate=5.0e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.8 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=lapo-tpu


# 3.
python3 main.py --batch_size=8 --base_learning_rate=7.0e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.8 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=lapo-tpu


# 4. 
python3 main.py --batch_size=8 --base_learning_rate=9.0e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.7 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=lapo-tpu

