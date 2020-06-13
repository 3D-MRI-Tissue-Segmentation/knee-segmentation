import tensorflow as tf
import tensorflow.keras.layers as tfkl
from Segmentation.model.unet_build_blocks import Conv2D_Block, Up_Conv2D
from Segmentation.model.unet_build_blocks import Attention_Gate
from Segmentation.model.unet_build_blocks import Recurrent_ResConv_block
from Segmentation.model.backbone import Encoder


class UNet(tf.keras.Model):
    """ Tensorflow 2 Implementation of 'U-Net: Convolutional Networks for
    Biomedical Image Segmentation' https://arxiv.org/abs/1505.04597."""

    def __init__(self,
                 num_channels,
                 num_classes,
                 backbone_name='default',
                 num_conv_layers=2,
                 kernel_size=(3, 3),
                 nonlinearity='relu',
                 use_attention=False,
                 use_batchnorm=True,
                 use_bias=True,
                 use_dropout=False,
                 dropout_rate=0.25,
                 use_spatial_dropout=True,
                 data_format='channels_last',
                 **kwargs):

        super(UNet, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.backbone_name = backbone_name
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.nonlinearity = nonlinearity
        self.use_attention = use_attention
        self.use_batchnorm = use_batchnorm
        self.use_bias = use_bias
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_spatial_dropout = use_spatial_dropout
        self.data_format = data_format

        self.contracting_path = []

        if self.backbone_name == 'default':
            for i in range(len(self.num_channels)):
                output = self.num_channels[i]
                self.contracting_path.append(Conv2D_Block(output,
                                                          self.num_conv_layers,
                                                          self.kernel_size,
                                                          self.nonlinearity,
                                                          self.use_batchnorm,
                                                          self.use_bias,
                                                          self.use_dropout,
                                                          self.dropout_rate,
                                                          self.use_spatial_dropout,
                                                          self.data_format))
                if i != len(self.num_channels) - 1:
                    self.contracting_path.append(tfkl.MaxPooling2D())
        else:
            encoder = Encoder(weights_init='imagenet', model_architecture=self.backbone_name)
            encoder.freeze_pretrained_layers()
            self.backbone = encoder.construct_backbone()

        self.upsampling_path = []

        n = len(self.num_channels) - 2
        for i in range(n, -1, -1):
            output = self.num_channels[i]
            self.upsampling_path.append(Up_Conv2D(output,
                                                  kernel_size=(2, 2),
                                                  nonlinearity=self.nonlinearity,
                                                  use_attention=self.use_attention,
                                                  use_batchnorm=self.use_batchnorm,
                                                  use_transpose=False,
                                                  use_bias=self.use_bias,
                                                  strides=(2, 2),
                                                  data_format=self.data_format))

        self.conv_1x1 = tfkl.Conv2D(num_classes,
                                    (1, 1),
                                    activation='linear',
                                    padding='same',
                                    data_format=data_format)

    def call(self, x, training=False):
        blocks = []
        if self.backbone_name == 'default':
            for i, down in enumerate(self.contracting_path):
                x = down(x, training=training)
                if i != len(self.contracting_path) - 1:
                    blocks.append(x)
        else:
            bridge_1, bridge_2, bridge_3, bridge_4, x = self.backbone(x, training=training)
            blocks.extend([bridge_1, bridge_2, bridge_3, bridge_4])

        for j, up in enumerate(self.upsampling_path):
            if self.backbone_name in ['default']:
                x = up(x, blocks[-2 * j - 2], training=training)
            else:
                x = up(x, blocks[-j - 1], training=training)

        del blocks

        if self.backbone_name not in ['default', 'vgg16', 'vgg19']:
            x = tfkl.UpSampling2D()(x)

        x = self.conv_1x1(x)
        if self.num_classes == 1:
            output = tfkl.Activation('sigmoid')(x)
        else:
            output = tfkl.Activation('softmax')(x)
        return output

class R2_UNet(tf.keras.Model):
    """ Tensorflow 2 Implementation of 'Recurrent Residual Convolutional
    Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation'
    https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf."""

    def __init__(self,
                 num_channels,
                 num_classes,
                 num_conv_layers=2,
                 kernel_size=(3, 3),
                 nonlinearity='relu',
                 t=2,
                 use_attention=False,
                 use_batchnorm=True,
                 use_bias=True,
                 data_format='channels_last',
                 **kwargs):

        super(R2_UNet, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.nonlinearity = nonlinearity
        self.t = t
        self.use_attention = use_attention
        self.use_batchnorm = use_batchnorm
        self.use_bias = use_bias
        self.data_format = data_format

        self.contracting_path = []

        for i in range(len(self.num_channels)):
            output = self.num_channels[i]
            self.contracting_path.append(Recurrent_ResConv_block(output,
                                                                 self.kernel_size,
                                                                 self.nonlinearity,
                                                                 'same',
                                                                 (1, 1),
                                                                 self.t,
                                                                 self.use_batchnorm,
                                                                 self.data_format))
            if i != len(self.num_channels) - 1:
                self.contracting_path.append(tfkl.MaxPooling2D())

        self.upsampling_path = []

        n = len(self.num_channels) - 2
        for i in range(n, -1, -1):
            output = self.num_channels[i]
            up_conv = Up_Conv2D(output,
                                kernel_size=(2, 2),
                                nonlinearity=self.nonlinearity,
                                use_attention=self.use_attention,
                                use_batchnorm=self.use_batchnorm,
                                use_transpose=False,
                                use_bias=self.use_bias,
                                strides=(2, 2),
                                data_format=self.data_format)

            # override default conv block with recurrent-residual conv block
            up_conv.conv_block = Recurrent_ResConv_block(output,
                                                         self.kernel_size,
                                                         self.nonlinearity,
                                                         'same',
                                                         (1, 1),
                                                         self.t,
                                                         self.use_batchnorm,
                                                         self.data_format)

            self.upsampling_path.append(up_conv)

        self.conv_1x1 = tfkl.Conv2D(self.num_classes,
                                    (1, 1),
                                    activation='linear',
                                    padding='same',
                                    data_format=data_format)

    def call(self, x, training=False):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x, training=training)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)

        for j, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-2 * j - 2], training=training)

        del blocks

        x = self.conv_1x1(x)
        if self.num_classes == 1:
            output = tfkl.Activation('sigmoid')(x)
        else:
            output = tfkl.Activation('softmax')(x)

        return output

class Nested_UNet(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 num_classes,
                 num_conv_layers=2,
                 kernel_size=(3, 3),
                 nonlinearity='relu',
                 use_batchnorm=True,
                 use_bias=True,
                 data_format='channels_last',
                 **kwargs):

        super(Nested_UNet, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.nonlinearity = nonlinearity
        self.use_batchnorm = use_batchnorm
        self.use_bias = use_bias
        self.data_format = data_format

        self.conv_block_lists = []

        for i in range(len(self.num_channels)):
            output_ch = self.num_channels[i]
            conv_layer_lists = []
            num_conv_blocks = len(self.num_channels) - i

            for _ in range(num_conv_blocks):

                conv_layer_lists.append(Conv2D_Block(output_ch,
                                                     self.num_conv_layers,
                                                     self.kernel_size,
                                                     self.nonlinearity,
                                                     self.use_batchnorm,
                                                     self.use_bias,
                                                     self.data_format))

            self.conv_block_lists.append(conv_layer_lists)

        self.pool = tfkl.MaxPooling2D()
        self.up = tfkl.UpSampling2D()

        self.conv_1x1 = tfkl.Conv2D(self.num_classes,
                                    (1, 1),
                                    activation='linear',
                                    padding='same',
                                    data_format=data_format)

    def call(self, input, training=False):

        block_list = []
        x = self.conv_block_lists[0][0](input, training=training)
        block_list.append(x)
        for sum_idx in range(1, len(self.conv_block_lists)):
            left_idx = sum_idx
            right_idx = 0
            layer_list = []
            while right_idx <= sum_idx:
                if left_idx == sum_idx:
                    if left_idx == 1 and right_idx == 0:
                        x = self.conv_block_lists[left_idx][right_idx](self.pool(block_list[left_idx - 1]),
                                                                       training=training)
                    else:
                        x = self.conv_block_lists[left_idx][right_idx](self.pool(block_list[left_idx - 1][right_idx]),
                                                                       training=training)
                else:
                    if not isinstance(block_list[left_idx], list):
                        x = self.conv_block_lists[left_idx][right_idx](tfkl.concatenate([self.up(x),
                                                                                        block_list[left_idx]]
                                                                                        ),
                                                                       training=training)
                    else:
                        x = self.conv_block_lists[left_idx][right_idx](tfkl.concatenate([self.up(x),
                                                                                        block_list[left_idx][0]]
                                                                                        ),
                                                                       training=training)
                left_idx -= 1
                right_idx += 1
                layer_list.append(x)
            block_list.append(layer_list)
        output = self.conv_1x1(x)

        if self.num_classes == 1:
            output = tfkl.Activation('sigmoid')(output)
        else:
            output = tfkl.Activation('softmax')(output)

        return output
