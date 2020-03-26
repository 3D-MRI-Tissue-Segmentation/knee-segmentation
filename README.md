# DeepSegmentation

Automatic Segmentation of Knee Cartilage from MRI using Deep Learning 

## Unit test and coverage

python -m pytest --cov-report term-missing:skip-covered --cov=Segmentation && coverage html && open ./htmlcov.index.html

### Models Implemented

- [x] Vanilla UNet 
- [x] Attention UNet
- [x] Multi-res UNet
- [x] R2_UNet
- [x] R2_Attention UNet
- [x] VNet
- [x] Tiny VNet
- [x] Small VNet
- [x] Relative VNet
- [x] 3D UNet
- [ ] 100-layer Tiramisu
- [ ] DeepLabv3+ 
