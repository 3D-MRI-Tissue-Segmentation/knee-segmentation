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

#### Small 3D Unet Highwayless

Training Loss | Training Progress
:------------:|:---------------------------:
![small-highway-less-loss](results/3d/small_highwayless_train_result_2020_03_17-08_07_29.png "Small 3D Unet Highwayless Loss") | ![small-highway-less-progress](results/3d/small_highwayless_progress.gif "Small 3D Unet Highwayless Progress")

> Small 3D UNet Highwayless: Loss plateaus very quickly and validation loss diverges from the loss very quickly.

<br />

#### Small 3D Unet

Training Loss | Training Progress
:------------:|:---------------------------:
![small-3d-unet-loss](results/3d/small_3dunet_train_result_2020_03_17-09_34_10.png "Small 3D Unet Loss") | ![small-3d-unet-progress](results/3d/small_3dunet_progress.gif "Small 3D Unet Progress")

> Small 3D Unet: Very slow minimisation of the loss, the validation loss has a large variance showing the model is overfitting.

<br />

#### Small Relative 3D Unet

Training Loss | Training Progress
:------------:|:---------------------------:
![small-relative-3d-unet-loss](results/3d/small_relative_3dunet_train_result_2020_03_17-11_03_20.png "Small Relative 3D Unet Loss") | ![small-relative-3d-unet-progress](results/3d/small_relative_3dunet_progress.gif "Small Relative 3D Unet Progress")

> Small Relative 3D Unet: Slow minimisation of the loss, the validation loss quickly diverges from the loss showing the model is overfitting.

<br />

#### Small VNet

Training Loss | Training Progress
:------------:|:---------------------------:
![small-vnet-loss](results/3d/small_vnet_train_result_2020_03_17-12_37_32.png "Small VNet Loss") | ![small-vnet-progress](results/3d/small_vnet_progress.gif "Small VNet Progress")

> Small VNet: Loss and validation loss converge together and to a low value showing good performance and generalisation abilities; this is a significant improvement on the other three model styles.

| Model               | Input Shape          | Min DL  | Min Val DL | Duration (mins) |
|---------------------|----------------------|---------|------------|-----------------|
| 3D Highwayless UNet | (160, 160, 160)      | 0.77735 | 0.84669    | 86.6            |
| 3D UNet             | (160, 160, 160)      | 0.72841 | 0.4160     | 89.1            |
| 3D Relative UNet    | (160, 160, 160), (3) | 0.82768 | 0.88853    | 90.1            |
| 3D VNet             | (160, 160, 160)      | 0.37088 | 0.34238    | 89.5            |

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