# hundred layer tiramisu tpu training testing configuration: 
# run date: 13.06.2020 (at night)
# 1. 
python3 main.py --train=False --multi_class=False --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --weights_dir=gs://oai-challenge-dataset/checkpoints/oliv-tpu/20200618-123315/100-Layer-Tiramisu_weights.035.ckpt --logdir=gs://oai-challenge-dataset/checkpoints --tpu=lapo-tpu
'''&&
# 2.
python3 main.py --train=False --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --weights_dir=gs://oai-challenge-dataset/checkpoints/lapo-tpu/20200613-130003/100-Layer-Tiramisu_weights.040.ckpt --logdir=gs://oai-challenge-dataset/checkpoints --tpu=lapo-tpu &&
# 3.
python3 main.py --train=False --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --weights_dir=gs://oai-challenge-dataset/checkpoints/lapo-tpu/20200613-145738/100-Layer-Tiramisu_weights.040.ckpt --logdir=gs://oai-challenge-dataset/checkpoints --tpu=lapo-tpu &&
# 4.
python3 main.py --train=False --model_architecture=100-Layer-Tiramisu --tfrec_dir=gs://oai-challenge-dataset/tfrecords/ --weights_dir=gs://oai-challenge-dataset/checkpoints/lapo-tpu/20200613-165513/100-Layer-Tiramisu_weights.040.ckpt --logdir=gs://oai-challenge-dataset/checkpoints --tpu=lapo-tpu '''

