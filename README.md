# Automated Tissue Segmentation from High-Resolution 3D Steady-State MRI with Deep Learning

Joe Arrowsmith, Joonsu Gha, Lapo Rastrelli, Olivia Gallupova, Pietro Vitiello

---

### 2D Models Implemented

- [x] SegNet 
- [x] Vanilla UNet 
- [x] Attention UNet
- [x] <del> Multi-res UNet </del>
- [x] R2_UNet
- [x] R2_Attention UNet
- [x] UNet++
- [x] 100-layer Tiramisu
- [x] DeepLabv3+ 

### 3D Models Implemented

- [x] 3D UNet
- [x] Relative 3D UNet
- [x] Slice 3D UNet
- [x] VNet
- [x] Relative VNet
- [x] Slice VNet

---

### Useful Code Snippets

``` Bash
Run 3D Train

python Segmentation/model/vnet_train.py
```

``` Bash
Unit-Testing and Unit-Test Converage

python -m pytest --cov-report term-missing:skip-covered --cov=Segmentation && coverage html && open ./htmlcov.index.html
```

``` Bash
Start tensorboard on Pompeii

On pompeii: tensorboard --logdir logs --samples_per_plugin images=100

On your local machine: ssh -L 16006:127.0.0.1:6006 username@ip

Go to localhost: http://localhost:16006/
```

---

### Valid 3D Configs

Batch / GPU | Crop Size | Depth Crop Size | Num Channels | Num Conv Layers | Kernel Size
:----------:|:--------:|:---------------:|:------------:|:---------------:|:----------:
1 | 32 | 32 | 20 | 2  | (5,5,5)
1 | 64 | 64 | 32 | 2  | (3,3,3)
1 | 64 | 64 | 32 | 2  | (5,5,5)
3 | 64 | 32 | 16 | 2  | (3,3,3)