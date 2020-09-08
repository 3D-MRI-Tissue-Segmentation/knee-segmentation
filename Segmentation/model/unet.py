import tensorflow as tf
import tensorflow.keras.layers as tfkl
from Segmentation.model.unet_build_blocks import Conv_Block, Up_Conv
from Segmentation.model.unet_build_blocks import Attention_Gate
from Segmentation.model.unet_build_blocks import Recurrent_ResConv_block
from Segmentation.model.backbone import Encoder


class UNet(tf.keras.Model):
    """ Tensorflow 2 Implementation of 'U-Net: Convolutional Networks for
    Biomedical Image Segmentation' https://arxiv.org/abs/1505.04597."""

    def __init__(self,
                 num_channels,
                 num_classes,
                 use_2d,
                 backbone_name='default',
                 num_conv_layers=2,
                 kernel_size=3,
                 activation='relu',
                 use_attention=False,
                 use_batchnorm=True,
                 use_bias=True,
                 use_dropout=False,
                 dropout_rate=0.25,
                 use_spatial_dropout=True,
                 data_format='channels_last',
                 **kwargs):

        super(UNet, self).__init__(**kwargs)

        self.backbone_name = backbone_name
        self.contracting_path = []
        self.upsampling_path = []

        if self.backbone_name == 'default':
            for i in range(len(num_channels)):
                output = num_channels[i]
                self.contracting_path.append(Conv_Block(num_channels=output,
                                                        use_2d=use_2d,
                                                        num_conv_layers=num_conv_layers,
                                                        kernel_size=kernel_size,
                                                        activation=activation,
                                                        use_batchnorm=use_batchnorm,
                                                        use_bias=use_bias,
                                                        use_dropout=use_dropout,
                                                        dropout_rate=dropout_rate,
                                                        use_spatial_dropout=use_spatial_dropout,
                                                        data_format=data_format))
                if i != len(num_channels) - 1:
                    if use_2d:
                        self.contracting_path.append(tfkl.MaxPooling2D())
                    else:
                        self.contracting_path.append(tfkl.MaxPooling3D())
        else:
            assert use_2d is True
            encoder = Encoder(weights_init='imagenet', model_architecture=backbone_name)
            encoder.freeze_pretrained_layers()
            self.backbone = encoder.construct_backbone()

        n = len(num_channels) - 2
        for i in range(n, -1, -1):
            output = num_channels[i]
            self.upsampling_path.append(Up_Conv(output,
                                                use_2d=use_2d,
                                                kernel_size=2,
                                                activation=activation,
                                                use_attention=use_attention,
                                                use_batchnorm=use_batchnorm,
                                                use_transpose=False,
                                                use_bias=use_bias,
                                                strides=2,
                                                data_format=data_format))

        if use_2d:
            self.conv_1x1 = tfkl.Conv2D(num_classes,
                                        (1, 1),
                                        activation='sigmoid' if num_classes == 1 else 'softmax',
                                        padding='same',
                                        data_format=data_format)
        else:
            self.conv_1x1 = tfkl.Conv3D(num_classes,
                                        (1, 1, 1),
                                        activation='linear' if num_classes == 1 else 'softmax',
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

        output = self.conv_1x1(x)
        return output

class R2_UNet(tf.keras.Model):
    """ Tensorflow 2 Implementation of 'Recurrent Residual Convolutional
    Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation'
    https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf."""

    def __init__(self,
                 num_channels,
                 num_classes,
                 use_2d=True,
                 num_conv_layers=2,
                 kernel_size=3,
                 activation='relu',
                 t=2,
                 use_attention=False,
                 use_batchnorm=True,
                 use_bias=True,
                 data_format='channels_last',
                 **kwargs):

        super(R2_UNet, self).__init__(**kwargs)

        self.contracting_path = []
        self.upsampling_path = []

        for i in range(len(num_channels)):
            output = num_channels[i]
            self.contracting_path.append(Recurrent_ResConv_block(num_channels=output,
                                                                 use_2d=use_2d,
                                                                 kernel_size=kernel_size,
                                                                 activation=activation,
                                                                 padding='same',
                                                                 strides=1,
                                                                 t=t,
                                                                 use_batchnorm=use_batchnorm,
                                                                 data_format=data_format))
            if i != len(num_channels) - 1:
                if use_2d:
                    self.contracting_path.append(tfkl.MaxPooling2D())
                else:
                    self.contracting_path.append(tfkl.MaxPooling3D())

        n = len(num_channels) - 2
        for i in range(n, -1, -1):
            output = num_channels[i]
            up_conv = Up_Conv(output,
                              use_2d,
                              kernel_size=2,
                              activation=activation,
                              use_attention=use_attention,
                              use_batchnorm=use_batchnorm,
                              use_transpose=False,
                              use_bias=use_bias,
                              strides=2,
                              data_format=data_format)

            # override default conv block with recurrent-residual conv block
            up_conv.conv_block = Recurrent_ResConv_block(num_channels=output,
                                                         use_2d=use_2d,
                                                         kernel_size=kernel_size,
                                                         activation=activation,
                                                         padding='same',
                                                         strides=1,
                                                         t=t,
                                                         use_batchnorm=use_batchnorm,
                                                         data_format=data_format)

            self.upsampling_path.append(up_conv)

        if use_2d:
            self.conv_1x1 = tfkl.Conv2D(filters=num_classes,
                                        kernel_size=(1, 1),
                                        activation='sigmoid' if num_classes == 1 else 'softmax',
                                        padding='same',
                                        data_format=data_format)
        else:
            self.conv_1x1 = tfkl.Conv3D(filters=num_classes,
                                        kernel_size=(1, 1, 1),
                                        activation='sigmoid' if num_classes == 1 else 'softmax',
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

        output = self.conv_1x1(x)

        return output

class Nested_UNet(tf.keras.Model):
    """ Tensorflow 2 Implementation of 'UNet++: A Nested
    U-Net Architecture for Medical Image Segmentation'
    https://arxiv.org/pdf/1807.10165.pdf """

    def __init__(self,
                 num_channels,
                 num_classes,
                 use_2d=True,
                 num_conv_layers=2,
                 kernel_size=(3, 3),
                 activation='relu',
                 use_batchnorm=True,
                 use_bias=True,
                 data_format='channels_last',
                 **kwargs):

        super(Nested_UNet, self).__init__(**kwargs)

        self.conv_block_lists = []
        self.pool = tfkl.MaxPooling2D() if use_2d else tfkl.MaxPooling3D()
        self.up = tfkl.UpSampling2D() if use_2d else tfkl.UpSampling3D()

        for i in range(len(num_channels)):
            output_ch = num_channels[i]
            conv_layer_lists = []
            num_conv_blocks = len(num_channels) - i

            for _ in range(num_conv_blocks):
                conv_layer_lists.append(Conv_Block(num_channels=output_ch,
                                                   use_2d=use_2d,
                                                   num_conv_layers=num_conv_layers,
                                                   kernel_size=kernel_size,
                                                   activation=activation,
                                                   use_batchnorm=use_batchnorm,
                                                   use_bias=use_bias,
                                                   data_format=data_format))

            self.conv_block_lists.append(conv_layer_lists)

        if use_2d:
            self.conv_1x1 = tfkl.Conv2D(num_classes,
                                        (1, 1),
                                        activation='sigmoid' if num_classes == 1 else 'softmax',
                                        padding='same',
                                        data_format=data_format)
        else:
            self.conv_1x1 = tfkl.Conv3D(num_classes,
                                        (1, 1, 1),
                                        activation='sigmoid' if num_classes == 1 else 'softmax',
                                        padding='same',
                                        data_format=data_format)

    def call(self, input, training=False):

        block_list = []
        x = self.conv_block_lists[0][0](input, training=training)
        block_list.append([x])
        for sum_idx in range(1, len(self.conv_block_lists)):
            left_idx = sum_idx
            right_idx = 0
            layer_list = []
            while right_idx <= sum_idx:
                if left_idx == sum_idx:
                    x = self.conv_block_lists[left_idx][right_idx](self.pool(block_list[left_idx - 1][right_idx]),
                                                                   training=training)
                else:
                    concat_list = [self.up(x)]
                    for idx in range(1, right_idx + 1):
                        concat_list.append(block_list[left_idx + idx - 1][-1 + idx])
                    x = self.conv_block_lists[left_idx][right_idx](tfkl.concatenate(concat_list),
                                                                   training=training)
                left_idx -= 1
                right_idx += 1
                layer_list.append(x)
            block_list.append(layer_list)
        output = self.conv_1x1(x)

        return output

class Nested_UNet_v2(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 num_classes,
                 use_2d=True,
                 num_conv_layers=2,
                 kernel_size=(3, 3),
                 activation='relu',
                 use_batchnorm=True,
                 use_bias=True,
                 data_format='channels_last',
                 **kwargs):

        super(Nested_UNet_v2, self).__init__(**kwargs)

        self.conv_block_lists = []
        self.pool = tfkl.MaxPooling2D() if use_2d else tfkl.MaxPooling3D()
        self.up = tfkl.UpSampling2D() if use_2d else tfkl.UpSampling3D()

        for i in range(len(num_channels)):
            output_ch = num_channels[i]
            conv_layer_lists = []
            num_conv_blocks = len(num_channels) - i

            for _ in range(num_conv_blocks):
                conv_layer_lists.append(Conv_Block(num_channels=output_ch,
                                                   use_2d=use_2d,
                                                   num_conv_layers=num_conv_layers,
                                                   kernel_size=kernel_size,
                                                   activation=activation,
                                                   use_batchnorm=use_batchnorm,
                                                   use_bias=use_bias,
                                                   data_format=data_format))

            self.conv_block_lists.append(conv_layer_lists)

        if use_2d:
            self.conv_1x1 = tfkl.Conv2D(num_classes,
                                        (1, 1),
                                        activation='sigmoid' if num_classes == 1 else 'softmax',
                                        padding='same',
                                        data_format=data_format)
        else:
            self.conv_1x1 = tfkl.Conv3D(num_classes,
                                        (1, 1, 1),
                                        activation='sigmoid' if num_classes == 1 else 'softmax',
                                        padding='same',
                                        data_format=data_format)

    def call(self, input, training=False):

        x = dict()
        use_x = list()
        x['0_0'] = self.conv_block_lists[0][0](input, training=training)
        last_0_name = '0_0'
        last_name = last_0_name

        for sum in range(1, len(self.conv_block_lists)):
            i, j = sum, 0
            while j <= sum:

                name = str(i) + '_' + str(j)

                if i == sum:
                    x[name] = self.conv_block_lists[i][j](self.pool(x[last_0_name]), training=training)
                    last_0_name = name

                else:
                    for temp_right in range(0, j):
                        string = str(i) + '_' + str(temp_right)
                        use_x.append(x[string])

                    use_x.append(self.up(x[last_name]))
                    x[name] = self.conv_block_lists[i][j](tfkl.concatenate(use_x), training=training)

                use_x.clear()
                last = (i, j)
                last_name = name
                i = i - 1
                j = j + 1

        output = self.conv_1x1(x[last_name])

        return output
