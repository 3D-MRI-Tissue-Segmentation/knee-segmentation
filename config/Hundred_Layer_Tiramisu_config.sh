 
# Shell script for Hundred Layer Tiramisu training configuration:

# 1.
python3 main.py --batch_size=8 --base_learning_rate=1e-06 --lr_drop_ratio=0.5 --train_epocs=40--model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenges-dataset/tfrecords/ --logdir=gs://oai-challenge-dataset/checkpoints/ --tpu=lapo_tpu


# 2.




