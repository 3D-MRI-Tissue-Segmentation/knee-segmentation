# Automated Tissue Segmentation from High-Resolution 3D Steady-State MRI with Deep Learning

Albert Ugwudike, Joe Arrowsmith, Joonsu Gha, Kamal Shah, Lapo Rastrelli, Olivia Gallupova, Pietro Vitiello

---

### 2D Models Implemented

- [x] Vanilla UNet 
- [x] Attention UNet
- [x] Multi-res UNet
- [x] R2_UNet
- [x] R2_Attention UNet
- [ ] 100-layer Tiramisu
- [ ] DeepLabv3+ 

### 3D Models Implemented

- [x] 3D UNet
- [x] Relative 3D UNet
- [x] Slice 3D UNet
- [x] VNet
- [ ] Relative VNet
- [ ] Slice VNet

---

### Comparison of Volume Segmentation Methods

<p align="center">Small 3D Unet Highwayless</p>

Training Loss | Training Progress
:------------:|:---------------------------:
![small-highway-less-loss](results/3d/small_highwayless_train_result_2020_03_17-08_07_29.png "Small 3D Unet Highwayless Loss") | ![small-highway-less-progress](results/3d/small_highwayless_progress.gif "Small 3D Unet Highwayless Progress")

<br />

<p align="center">Small 3D Unet</p>

Training Loss | Training Progress
:------------:|:---------------------------:
![small-3d-unet-loss](results/3d/small_3dunet_train_result_2020_03_17-09_34_10.png "Small 3D Unet Loss") | ![small-3d-unet-progress](results/3d/small_3dunet_progress.gif "Small 3D Unet Progress")

<br />

<p align="center">Small Relative 3D Unet</p>

Training Loss | Training Progress
:------------:|:---------------------------:
![small-relative-3d-unet-loss](results/3d/small_relative_3dunet_train_result_2020_03_17-11_03_20.png "Small Relative 3D Unet Loss") | ![small-relative-3d-unet-progress](results/3d/small_relative_3dunet_progress.gif "Small Relative 3D Unet Progress")

<br />

<p align="center">Small VNet</p>

Training Loss | Training Progress
:------------:|:---------------------------:
![small-vnet-loss](results/3d/small_vnet_train_result_2020_03_17-12_37_32.png "Small VNet Loss") | ![small-vnet-progress](results/3d/small_vnet_progress.gif "Small VNet Progress")

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