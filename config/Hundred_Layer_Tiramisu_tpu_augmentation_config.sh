# Hundred Layer Tiramisu tpu training configuration with augumentation:

python3 main.py --aug_strategy=random_crop --batch_size=8 --multi_class=False --base_learning_rate=1.0e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.85 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=oliv-tpu

python3 main.py --batch_size=8 --multi_class=False --base_learning_rate=1.0e-06 --lr_warmup_epochs=1 --lr_drop_ratio=0.85 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=oliv-tpu

python3 main.py --aug_strategy=random_crop --batch_size=8 --multi_class=False --base_learning_rate=1.0e-05 --lr_warmup_epochs=1 --lr_drop_ratio=0.85 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=oliv-tpu

python3 main.py --batch_size=8 --multi_class=False --base_learning_rate=1.0e-05 --lr_warmup_epochs=1 --lr_drop_ratio=0.85 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=oliv-tpu

python3 main.py --aug_strategy=random_crop --batch_size=8 --multi_class=False --base_learning_rate=1.0e-04 --lr_warmup_epochs=1 --lr_drop_ratio=0.85 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=oliv-tpu

python3 main.py --batch_size=8 --multi_class=False --base_learning_rate=1.0e-04 --lr_warmup_epochs=1 --lr_drop_ratio=0.85 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=oliv-tpu

python3 main.py --aug_strategy=random_crop --batch_size=8 --multi_class=False --base_learning_rate=1.0e-03 --lr_warmup_epochs=1 --lr_drop_ratio=0.85 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=oliv-tpu
 
python3 main.py --batch_size=8 --multi_class=False --base_learning_rate=1.0e-03 --lr_warmup_epochs=1 --lr_drop_ratio=0.85 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=oliv-tpu

python3 main.py --aug_strategy=random_crop --batch_size=8 --multi_class=False --base_learning_rate=5.0e-02 --lr_warmup_epochs=1 --lr_drop_ratio=0.85 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=oliv-tpu

python3 main.py --batch_size=8 --multi_class=False --base_learning_rate=5.0e-02 --lr_warmup_epochs=1 --lr_drop_ratio=0.85 --train_epochs=40 --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=oliv-tpu

