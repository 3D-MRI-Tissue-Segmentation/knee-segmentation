from Segmentation.model.unet import UNet, R2_UNet, Nested_UNet, Nested_UNet_v2
from Segmentation.model.segnet import SegNet
from Segmentation.model.deeplabv3 import Deeplabv3, Deeplabv3_plus
from Segmentation.model.vnet import VNet
from Segmentation.model.Hundred_Layer_Tiramisu import Hundred_Layer_Tiramisu

from absl import logging

def select_model(FLAGS, num_classes):

    if FLAGS.model_architecture == 'unet':
        model_args = [FLAGS.num_filters,
                      num_classes,
                      FLAGS.use_2d,
                      FLAGS.backbone_architecture,
                      FLAGS.num_conv,
                      FLAGS.kernel_size,
                      FLAGS.activation,
                      FLAGS.use_attention,
                      FLAGS.use_batchnorm,
                      FLAGS.use_bias,
                      FLAGS.use_dropout,
                      FLAGS.dropout_rate,
                      FLAGS.use_spatial,
                      FLAGS.channel_order]

        model_fn = UNet
    elif FLAGS.model_architecture == 'vnet':
        model_args = [FLAGS.num_filters,
                      num_classes,
                      FLAGS.use_2d,
                      FLAGS.num_conv,
                      FLAGS.kernel_size,
                      FLAGS.activation,
                      FLAGS.use_batchnorm,
                      FLAGS.dropout_rate,
                      FLAGS.use_spatial,
                      FLAGS.channel_order]
        model_fn = VNet
    elif FLAGS.model_architecture == 'r2unet':
        model_args = [FLAGS.num_filters,
                      num_classes,
                      FLAGS.use_2d,
                      FLAGS.num_conv,
                      FLAGS.kernel_size,
                      FLAGS.activation,
                      2,
                      FLAGS.use_attention,
                      FLAGS.use_batchnorm,
                      FLAGS.use_bias,
                      FLAGS.channel_order]
        model_fn = R2_UNet

    elif FLAGS.model_architecture == 'segnet':
        model_args = [FLAGS.num_filters,
                      num_classes,
                      FLAGS.backbone_architecture,
                      FLAGS.kernel_size,
                      (2, 2),
                      FLAGS.activation,
                      FLAGS.use_batchnorm,
                      FLAGS.use_bias,
                      FLAGS.use_transpose,
                      FLAGS.use_dropout,
                      FLAGS.dropout_rate,
                      FLAGS.use_spatial,
                      FLAGS.channel_order]

        model_fn = SegNet

    elif FLAGS.model_architecture == 'unet++':
        model_args = [FLAGS.num_filters,
                      num_classes,
                      FLAGS.num_conv,
                      FLAGS.kernel_size,
                      FLAGS.activation,
                      FLAGS.use_batchnorm,
                      FLAGS.use_bias,
                      FLAGS.channel_order]
        model_fn = Nested_UNet

    elif FLAGS.model_architecture == '100-Layer-Tiramisu':
        model_args = [FLAGS.growth_rate,
                      FLAGS.layers_per_block,
                      FLAGS.init_num_channels,
                      num_classes,
                      FLAGS.kernel_size,
                      FLAGS.pool_size,
                      FLAGS.activation,
                      FLAGS.dropout_rate,
                      FLAGS.strides,
                      FLAGS.padding]

        model_fn = Hundred_Layer_Tiramisu

    elif FLAGS.model_architecture == 'deeplabv3':
        model_args = [num_classes,
                      FLAGS.kernel_size_initial_conv,
                      FLAGS.num_filters_atrous,
                      FLAGS.num_filters_DCNN,
                      FLAGS.num_filters_ASPP,
                      FLAGS.kernel_size_atrous,
                      FLAGS.kernel_size_DCNN,
                      FLAGS.kernel_size_ASPP,
                      'same',
                      FLAGS.activation,
                      FLAGS.use_batchnorm,
                      FLAGS.use_bias,
                      FLAGS.channel_order,
                      FLAGS.MultiGrid,
                      FLAGS.rate_ASPP,
                      FLAGS.output_stride]

        model_fn = Deeplabv3

    elif FLAGS.model_architecture == 'deeplabv3_plus':
        model_args = [num_classes,
                      FLAGS.kernel_size_initial_conv,
                      FLAGS.num_filters_atrous,
                      FLAGS.num_filters_DCNN,
                      FLAGS.num_filters_ASPP,
                      FLAGS.kernel_size_atrous,
                      FLAGS.kernel_size_DCNN,
                      FLAGS.kernel_size_ASPP,
                      FLAGS.num_filters_final_encoder,
                      FLAGS.num_filters_from_backbone,
                      FLAGS.num_channels_UpConv,
                      FLAGS.kernel_size_UpConv,
                      (2, 2),
                      False,
                      FLAGS.use_transpose,
                      'same',
                      FLAGS.activation,
                      FLAGS.use_batchnorm,
                      FLAGS.use_bias,
                      FLAGS.channel_order,
                      FLAGS.MultiGrid,
                      FLAGS.rate_ASPP,
                      FLAGS.output_stride]

        model_fn = Deeplabv3_plus

    else:
        logging.error('The model architecture {} is not supported!'.format(FLAGS.model_architecture))

    return model_fn, model_args
