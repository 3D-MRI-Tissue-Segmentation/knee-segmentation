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
- [x] Relative VNet
- [x] Slice VNet

---

### Baseline Comparision of 3D Methods

| Model               | Input Shape          | Min DL  | Min Val DL | Duration (mins) |
|---------------------|----------------------|---------|------------|-----------------|
| 3D Highwayless UNet | (160, 160, 160)      | 0.77735 | 0.84669    | 86.6            |
| 3D UNet             | (160, 160, 160)      | 0.72841 | 0.4160     | 89.1            |
| 3D Relative UNet    | (160, 160, 160), (3) | 0.82768 | 0.88853    | 90.1            |
| 3D VNet             | (160, 160, 160)      | 0.37088 | 0.34238    | 89.5            |

#### Small 3D Unet Highwayless (160, 160, 160)

Training Loss | Training Progress
:------------:|:---------------------------:
![small-highway-less-loss](results/3unet_vs_vnet_baseline/small_highwayless_train_result_2020_03_17-08_07_29.png "Small 3D Unet Highwayless Loss") | ![small-highway-less-progress](results/3unet_vs_vnet_baseline/small_highwayless_progress.gif "Small 3D Unet Highwayless Progress")

> Small 3D UNet Highwayless: Loss plateaus very quickly and validation loss diverges from the loss very quickly.

<br />

#### Small 3D Unet (160, 160, 160)

Training Loss | Training Progress
:------------:|:---------------------------:
![small-3d-unet-loss](results/3unet_vs_vnet_baseline/small_3dunet_train_result_2020_03_17-09_34_10.png "Small 3D Unet Loss") | ![small-3d-unet-progress](results/3unet_vs_vnet_baseline/small_3dunet_progress.gif "Small 3D Unet Progress")

> Small 3D Unet: Very slow minimisation of the loss, the validation loss has a large variance showing the model is overfitting.

<br />

#### Small Relative 3D Unet (160, 160, 160), (3)

Training Loss | Training Progress
:------------:|:---------------------------:
![small-relative-3d-unet-loss](results/3unet_vs_vnet_baseline/small_relative_3dunet_train_result_2020_03_17-11_03_20.png "Small Relative 3D Unet Loss") | ![small-relative-3d-unet-progress](results/3unet_vs_vnet_baseline/small_relative_3dunet_progress.gif "Small Relative 3D Unet Progress")

> Small Relative 3D Unet: Slow minimisation of the loss, the validation loss quickly diverges from the loss showing the model is overfitting.

<br />

#### Small VNet (160, 160, 160)

Training Loss | Training Progress
:------------:|:---------------------------:
![small-vnet-loss](results/3unet_vs_vnet_baseline/small_vnet_train_result_2020_03_17-12_37_32.png "Small VNet Loss") | ![small-vnet-progress](results/3unet_vs_vnet_baseline/small_vnet_progress.gif "Small VNet Progress")

> Small VNet: Loss and validation loss converge together and to a low value showing good performance and generalisation abilities; this is a significant improvement on the other three model styles.

---

### Comparison of VNet Methods

|      Model     |      Input Shape     |        Loss       |      Val Loss      |     Roll Loss     |   Roll Val Loss   |  Duration / mins |
|:--------------:|:--------------------:|:-----------------:|:------------------:|:-----------------:|:-----------------:|:----------------:|
|      tiny      |     (64, 64, 64)     | 0.627 $\pm$ 0.066 |  0.684 $\pm$ 0.078 | 0.652 $\pm$ 0.071 | 0.686 $\pm$ 0.077 |  61.5 $\pm$ 5.32 |
|      tiny      |    (160, 160, 160)   | 0.773 $\pm$ 0.01  |  0.779 $\pm$ 0.019 | 0.778 $\pm$ 0.007 | 0.787 $\pm$ 0.016 | 101.8 $\pm$ 2.52 |
|      small     |    (160, 160, 160)   | 0.648 $\pm$ 0.156 |  0.676 $\pm$ 0.106 | 0.656 $\pm$ 0.152 | 0.698 $\pm$ 0.076 | 110.1 $\pm$ 4.64 |
| small_relative | (160, 160, 160), (3) | 0.653 $\pm$ 0.168 |  0.639 $\pm$ 0.176 | 0.659 $\pm$ 0.167 | 0.644 $\pm$ 0.172 | 104.6 $\pm$ 9.43 |
|      slice     |     (160, 160, 5)    | 0.546 $\pm$ 0.019 |  0.845 $\pm$ 0.054 | 0.559 $\pm$ 0.020 | 0.860 $\pm$ 0.072 |  68.6 $\pm$ 9.68 |
|      small     |    (240, 240, 160)   | 0.577 $\pm$ 0.153 |  0.657 $\pm$ 0.151 | 0.583 $\pm$ 0.151 | 0.666 $\pm$ 0.149 | 109.7 $\pm$ 0.37 |
|      large     |    (240, 240, 160)   | 0.505 $\pm$ 0.262 |  0.554 $\pm$ 0.254 | 0.508 $\pm$ 0.262 | 0.574 $\pm$ 0.243 | 129.2 $\pm$ 0.50 |
| large_relative | (240, 240, 160), (3) | 0.709 $\pm$ 0.103 |  0.880 $\pm$ 0.078 | 0.725 $\pm$ 0.094 | 0.913 $\pm$ 0.081 | 148.6 $\pm$ 0.20 |

#### Tiny VNet (64, 64, 64)

Training Loss | Training Progress
:------------:|:---------------------------:
![]() | ![]()

#### Tiny VNet (160, 160, 160)

Training Loss | Training Progress
:------------:|:---------------------------:
![]() | ![]()

#### Small VNet (160, 160, 160)

Training Loss | Training Progress
:------------:|:---------------------------:
![]() | ![]()

#### Small Relative VNet (160, 160, 160), (3)

Training Loss | Training Progress
:------------:|:---------------------------:
![]() | ![]()

#### Small Slice VNet (160, 160, 5)

Training Loss | Training Progress
:------------:|:---------------------------:
![]() | ![]()

#### Small VNet (240, 240, 160)

Training Loss | Training Progress
:------------:|:---------------------------:
![]() | ![]()

#### Large VNet (240, 240, 160)

Training Loss | Training Progress
:------------:|:---------------------------:
![]() | ![]()

#### Large Relative VNet (240, 240, 160), (3)

Training Loss | Training Progress
:------------:|:---------------------------:
![]() | ![]()

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