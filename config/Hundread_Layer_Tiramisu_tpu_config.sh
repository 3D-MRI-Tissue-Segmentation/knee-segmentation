# Hundred Layer Tiramisu tpu training configuration:

# 1. 
python3 main.py --batch_size=8 --base_learning_rate=0.5e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.5
--train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ 
--logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=lapo-tpu


# 2.
python3 main.py --batch_size=8 --base_learning_rate=1.0e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.5
--train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ 
--logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=lapo-tpu


# 3.
python3 main.py --batch_size=8 --base_learning_rate=5.0e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.4
--train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ 
--logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=lapo-tpu


# 4. 
python3 main.py --batch_size=8 --base_learning_rate=8e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.4
--train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ 
--logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=lapo-tpu

