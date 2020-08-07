from Segmentation.model.Hundred_Layer_Tiramisu import Hundred_Layer_Tiramisu
from Segmentation.model.deeplabv3 import Deeplabv3
from Segmentation.model.unet import UNet

def select_model(name, num_classes, num_channels, use_2d=True, **model_kwargs):

    if name == 'Hundred_Layer_Tiramisu':
        assert 'growth_rate' in model_kwargs
        assert 'layers_per_block' in model_kwargs
        assert use_2d is True
        model_fn = Hundred_Layer_Tiramisu(model_kwargs['growth_rate'], 
                                          model_kwargs['layers_per_block'],
                                          num_channels,
                                          num_classes,
                                          **model_kwargs)

    elif name == 'Deeplabv3':
        assert 'kernel_size_initial_conv' in model_kwargs
        assert use_2d is True
        model_fn = Deeplabv3(num_classes,
                             model_kwargs['kernel_size_initial_conv'],
                             num_channels_DCNN=num_channels,
                             **model_kwargs)

    elif name == 'UNet':
        model_fn = UNet(num_channels,
                        num_classes,
                        use_2d,
                        **model_kwargs)

    else:
        raise Exception('Model name not recognised')

    return model_fn
