import tensorflow as tf
import tensorflow.keras.layers as tfkl
from Segmentation.model.unet_build_blocks import Conv2D_Block, Up_Conv2D
from Segmentation.model.unet_build_blocks import Attention_Gate
from Segmentation.model.unet_build_blocks import ResPath, MultiResBlock
from Segmentation.model.backbone import VGG16_Encoder


class UNet(tf.keras.Model):
    """ Tensorflow 2 Implementation of 'U-Net: Convolutional Networks for
    Biomedical Image Segmentation' https://arxiv.org/abs/1505.04597."""

    def __init__(self,
                 num_channels,
                 num_classes,
                 backbone='default',
                 backbone_weights=None,
                 freeze_backbone=True,
                 num_conv_layers=2,
                 kernel_size=(3, 3),
                 nonlinearity='relu',
                 use_batchnorm=True,
                 use_bias=True,
                 use_dropout=False,
                 dropout_rate=0.25,
                 use_spatial_dropout=True,
                 data_format='channels_last',
                 **kwargs):

        super(UNet, self).__init__(**kwargs)

        self.num_classes = num_classes

        # encoding blocks
        if backbone in ('vgg16', 'VGG16'):
            self.encoder = VGG16_Encoder(weights_init=backbone_weights)
            if freeze_backbone:
                self.encoder.freeze_pretrained_layers()
            # TODO(Joonsu): This code is really ugly...
            # let's find a more elegant way to implement this.
            self.conv_1 = self.encoder.conv_1
            self.conv_2 = self.encoder.conv_2
            self.conv_3 = self.encoder.conv_3
            self.conv_4 = self.encoder.conv_4
            self.conv_5 = self.encoder.conv_5

            del self.encoder

        elif backbone == 'default':
            self.conv_1 = Conv2D_Block(num_channels=num_channels,
                                       num_conv_layers=num_conv_layers,
                                       kernel_size=kernel_size,
                                       nonlinearity=nonlinearity,
                                       use_batchnorm=use_batchnorm,
                                       use_bias=use_bias,
                                       data_format=data_format)
            self.conv_2 = Conv2D_Block(num_channels=num_channels * 2,
                                       num_conv_layers=num_conv_layers,
                                       kernel_size=kernel_size,
                                       nonlinearity=nonlinearity,
                                       use_batchnorm=use_batchnorm,
                                       use_bias=use_bias,
                                       data_format=data_format)
            self.conv_3 = Conv2D_Block(num_channels=num_channels * 4,
                                       num_conv_layers=num_conv_layers,
                                       kernel_size=kernel_size,
                                       nonlinearity=nonlinearity,
                                       use_batchnorm=use_batchnorm,
                                       use_bias=use_bias,
                                       data_format=data_format)
            self.conv_4 = Conv2D_Block(num_channels=num_channels * 8,
                                       num_conv_layers=num_conv_layers,
                                       kernel_size=kernel_size,
                                       nonlinearity=nonlinearity,
                                       use_batchnorm=use_batchnorm,
                                       use_bias=use_bias,
                                       use_dropout=use_dropout,
                                       dropout_rate=dropout_rate,
                                       use_spatial_dropout=use_spatial_dropout,
                                       data_format=data_format)
            self.conv_5 = Conv2D_Block(num_channels=num_channels * 16,
                                       num_conv_layers=num_conv_layers,
                                       kernel_size=kernel_size,
                                       nonlinearity=nonlinearity,
                                       use_batchnorm=use_batchnorm,
                                       use_bias=use_bias,
                                       use_dropout=use_dropout,
                                       dropout_rate=dropout_rate,
                                       use_spatial_dropout=use_spatial_dropout,
                                       data_format=data_format)

        # decoding blocks
        self.up_5 = Up_Conv2D(num_channels * 8,
                              (2, 2),
                              nonlinearity,
                              use_batchnorm=use_batchnorm,
                              data_format=data_format)
        self.up_conv5 = Conv2D_Block(num_channels * 8,
                                     num_conv_layers,
                                     kernel_size,
                                     nonlinearity,
                                     use_batchnorm=use_batchnorm,
                                     use_bias=use_bias,
                                     data_format=data_format)

        self.up_4 = Up_Conv2D(num_channels * 4,
                              (2, 2),
                              nonlinearity,
                              use_batchnorm=use_batchnorm,
                              data_format=data_format)
        self.up_conv4 = Conv2D_Block(num_channels * 4,
                                     num_conv_layers,
                                     kernel_size,
                                     nonlinearity,
                                     use_batchnorm=use_batchnorm,
                                     use_bias=use_bias,
                                     data_format=data_format)

        self.up_3 = Up_Conv2D(num_channels * 2,
                              (2, 2),
                              nonlinearity,
                              use_batchnorm=use_batchnorm,
                              data_format=data_format)
        self.up_conv3 = Conv2D_Block(num_channels * 2,
                                     num_conv_layers,
                                     kernel_size,
                                     nonlinearity,
                                     use_batchnorm=use_batchnorm,
                                     use_bias=use_bias,
                                     data_format=data_format)

        self.up_2 = Up_Conv2D(num_channels,
                              (2, 2),
                              nonlinearity,
                              use_batchnorm=use_batchnorm,
                              data_format=data_format)
        self.up_conv2 = Conv2D_Block(num_channels,
                                     num_conv_layers,
                                     kernel_size,
                                     nonlinearity,
                                     use_batchnorm=use_batchnorm,
                                     use_bias=use_bias,
                                     data_format=data_format)

        # convolution num_channels at the output
        self.conv_1x1 = tfkl.Conv2D(num_classes,
                                    (1, 1),
                                    activation='linear',
                                    padding='same',
                                    data_format=data_format)

    def call(self, inputs, training=False):

        # encoder blocks
        # 1->64
        x1 = self.conv_1(inputs, training=training)

        # 64->128
        x2 = tfkl.MaxPooling2D(pool_size=(2, 2))(x1)
        x2 = self.conv_2(x2, training=training)

        # 128->256
        x3 = tfkl.MaxPooling2D(pool_size=(2, 2))(x2)
        x3 = self.conv_3(x3, training=training)

        # 256->512
        x4 = tfkl.MaxPooling2D(pool_size=(2, 2))(x3)
        x4 = self.conv_4(x4, training=training)

        # 512->1024
        x5 = tfkl.MaxPooling2D(pool_size=(2, 2))(x4)
        x5 = self.conv_5(x5, training=training)

        # decoder blocks
        # 1024->512
        u5 = self.up_5(x5, training=training)
        u5 = tfkl.concatenate([x4, u5], axis=3)
        u5 = self.up_conv5(u5, training=training)

        # 512->256
        u6 = self.up_4(u5, training=training)
        u6 = tfkl.concatenate([x3, u6], axis=3)
        u6 = self.up_conv4(u6, training=training)

        # 256->128
        u7 = self.up_3(u6, training=training)
        u7 = tfkl.concatenate([x2, u7], axis=3)
        u7 = self.up_conv2(u7, training=training)

        # 128->64
        u8 = self.up_2(u7, training=training)
        u8 = tfkl.concatenate([x1, u8], axis=3)
        u8 = self.up_conv2(u8, training=training)

        # logits
        u9 = self.conv_1x1(u8)

        if self.num_classes == 1:
            output = tfkl.Activation('sigmoid')(u9)

        else:
            output = tfkl.Activation('softmax')(u9)

        return output


class AttentionUNet(tf.keras.Model):
    "Tensorflow 2 Implementation of Attention UNet"

    def __init__(self,
                 num_channels,
                 num_classes,
                 num_conv_layers=2,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 pool_size=(2, 2),
                 use_bias=True,
                 padding='same',
                 nonlinearity='relu',
                 use_batchnorm=True,
                 use_transpose=True,
                 data_format='channels_last',
                 **kwargs):

        super(AttentionUNet, self).__init__(**kwargs)

        self.num_classes = num_classes

        self.conv_1 = Conv2D_Block(num_channels,
                                   num_conv_layers,
                                   kernel_size,
                                   nonlinearity,
                                   use_batchnorm=use_batchnorm,
                                   data_format=data_format)
        self.conv_2 = Conv2D_Block(num_channels * 2,
                                   num_conv_layers,
                                   kernel_size,
                                   nonlinearity,
                                   use_batchnorm=use_batchnorm,
                                   data_format=data_format)
        self.conv_3 = Conv2D_Block(num_channels * 4,
                                   num_conv_layers,
                                   kernel_size,
                                   nonlinearity,
                                   use_batchnorm=use_batchnorm,
                                   data_format=data_format)
        self.conv_4 = Conv2D_Block(num_channels * 8,
                                   num_conv_layers,
                                   kernel_size,
                                   nonlinearity,
                                   use_batchnorm=use_batchnorm,
                                   data_format=data_format)
        self.conv_5 = Conv2D_Block(num_channels * 16,
                                   num_conv_layers,
                                   kernel_size,
                                   nonlinearity,
                                   use_batchnorm=use_batchnorm,
                                   data_format=data_format)

        self.up_conv_1 = Up_Conv2D(num_channels * 8,
                                   (3, 3),
                                   nonlinearity,
                                   use_batchnorm=use_batchnorm,
                                   data_format=data_format)
        self.up_conv_2 = Up_Conv2D(num_channels * 4,
                                   (3, 3),
                                   nonlinearity,
                                   use_batchnorm=use_batchnorm,
                                   data_format=data_format)
        self.up_conv_3 = Up_Conv2D(num_channels * 2,
                                   (3, 3),
                                   nonlinearity,
                                   use_batchnorm=use_batchnorm,
                                   data_format=data_format)
        self.up_conv_4 = Up_Conv2D(num_channels,
                                   (3, 3),
                                   nonlinearity,
                                   use_batchnorm=use_batchnorm,
                                   data_format=data_format)

        self.a1 = Attention_Gate(num_channels * 8,
                                 (1, 1),
                                 nonlinearity,
                                 padding,
                                 strides,
                                 use_bias,
                                 data_format)
        self.a2 = Attention_Gate(num_channels * 4,
                                 (1, 1),
                                 nonlinearity,
                                 padding,
                                 strides,
                                 use_bias,
                                 data_format)
        self.a3 = Attention_Gate(num_channels * 2,
                                 (1, 1),
                                 nonlinearity,
                                 padding,
                                 strides,
                                 use_bias,
                                 data_format)
        self.a4 = Attention_Gate(num_channels,
                                 (1, 1),
                                 nonlinearity,
                                 padding,
                                 strides,
                                 use_bias,
                                 data_format)

        self.u1 = Conv2D_Block(num_channels * 8,
                               num_conv_layers,
                               kernel_size,
                               nonlinearity,
                               use_batchnorm=use_batchnorm,
                               use_transpose=use_transpose,
                               data_format=data_format)
        self.u2 = Conv2D_Block(num_channels * 4,
                               num_conv_layers,
                               kernel_size,
                               nonlinearity,
                               use_batchnorm=use_batchnorm,
                               use_transpose=use_transpose,
                               data_format=data_format)
        self.u3 = Conv2D_Block(num_channels * 2,
                               num_conv_layers,
                               kernel_size,
                               nonlinearity,
                               use_batchnorm=True,
                               use_transpose=use_transpose,
                               data_format=data_format)
        self.u4 = Conv2D_Block(num_channels,
                               num_conv_layers,
                               kernel_size,
                               nonlinearity,
                               use_batchnorm=use_batchnorm,
                               use_transpose=use_transpose,
                               data_format=data_format)

        self.conv_1x1 = tfkl.Conv2D(num_classes,
                                    (1, 1),
                                    activation='linear',
                                    padding='same',
                                    data_format=data_format)

    def call(self, inputs, training=False):

        # ENCODER PATH
        x1 = self.conv_1(inputs)

        pool1 = tfkl.MaxPooling2D(pool_size=(2, 2))(x1)
        x2 = self.conv_2(pool1, training=training)

        pool2 = tfkl.MaxPooling2D(pool_size=(2, 2))(x2)
        x3 = self.conv_3(pool2, training=training)

        pool3 = tfkl.MaxPooling2D(pool_size=(2, 2))(x3)
        x4 = self.conv_4(pool3, training=training)

        pool4 = tfkl.MaxPooling2D(pool_size=(2, 2))(x4)
        x5 = self.conv_5(pool4, training=training)

        # DECODER PATH
        up4 = self.up_conv_1(x5, training=training)
        a1 = self.a1(x4, up4, training=training)
        y1 = tfkl.concatenate([a1, up4])
        y1 = self.u1(y1, training=training)

        up5 = self.up_conv_2(y1, training=training)
        a2 = self.a2(x3, up5, training=training)
        y2 = tfkl.concatenate([a2, up5])
        y2 = self.u2(y2, training=training)

        up6 = self.up_conv_3(y2, training=training)
        a3 = self.a3(x2, up6, training=training)
        y3 = tfkl.concatenate([a3, up6])
        y3 = self.u3(y3, training=training)

        up7 = self.up_conv_4(y3, training=training)
        a4 = self.a4(x1, up7, training=training)
        y4 = tfkl.concatenate([a4, up7])
        y4 = self.u4(y4, training=training)
        y5 = self.conv_1x1(y4)

        if self.num_classes == 1:
            output = tfkl.Activation('sigmoid')(y5)
        else:
            output = tfkl.Activation('softmax')(y5)

        return output


class MultiResUnet(tf.keras.Model):
    "Tensorflow 2 Implementation of Multires UNet"

    def __init__(self,
                 num_channels,
                 num_classes,
                 res_path_length,
                 num_conv_layers=1,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 pool_size=(2, 2),
                 use_bias=False,
                 padding='same',
                 nonlinearity='relu',
                 use_batchnorm=True,
                 use_transpose=True,
                 data_format='channels_last',
                 **kwargs):

        super(MultiResUnet, self).__init__(**kwargs)

        # ENCODING BLOCKS
        self.mresblock_1 = MultiResBlock(num_channels,
                                         kernel_size,
                                         nonlinearity,
                                         padding='same',
                                         strides=(1, 1),
                                         data_format='channels_last')
        self.mresblock_2 = MultiResBlock(num_channels * 2,
                                         kernel_size,
                                         nonlinearity,
                                         padding='same',
                                         strides=(1, 1),
                                         data_format='channels_last')
        self.mresblock_3 = MultiResBlock(num_channels * 4,
                                         kernel_size,
                                         nonlinearity,
                                         padding='same',
                                         strides=(1, 1),
                                         data_format='channels_last')
        self.mresblock_4 = MultiResBlock(num_channels * 8,
                                         kernel_size,
                                         nonlinearity,
                                         padding='same',
                                         strides=(1, 1),
                                         data_format='channels_last')
        self.mresblock_5 = MultiResBlock(num_channels * 16,
                                         kernel_size,
                                         nonlinearity,
                                         padding='same',
                                         strides=(1, 1),
                                         data_format='channels_last')

        # DECODING BLOCKS
        self.mresblock_6 = MultiResBlock(num_channels,
                                         kernel_size,
                                         nonlinearity,
                                         padding='same',
                                         strides=(1, 1),
                                         data_format='channels_last')
        self.mresblock_7 = MultiResBlock(num_channels * 2,
                                         kernel_size,
                                         nonlinearity,
                                         padding='same',
                                         strides=(1, 1),
                                         data_format='channels_last')
        self.mresblock_8 = MultiResBlock(num_channels * 4,
                                         kernel_size,
                                         nonlinearity,
                                         padding='same',
                                         strides=(1, 1),
                                         data_format='channels_last')
        self.mresblock_9 = MultiResBlock(num_channels * 8,
                                         kernel_size,
                                         nonlinearity,
                                         padding='same',
                                         strides=(1, 1),
                                         data_format='channels_last')

        self.pool_1 = tfkl.MaxPooling2D(pool_size=(2, 2))
        self.pool_2 = tfkl.MaxPooling2D(pool_size=(2, 2))
        self.pool_3 = tfkl.MaxPooling2D(pool_size=(2, 2))
        self.pool_4 = tfkl.MaxPooling2D(pool_size=(2, 2))

        self.up_1 = Up_Conv2D(num_channels * 8,
                              kernel_size=(3, 3),
                              nonlinearity='relu',
                              use_batchnorm=True,
                              use_transpose=True,
                              strides=(2, 2),
                              data_format='channels_last')
        self.up_2 = Up_Conv2D(num_channels * 4,
                              kernel_size=(3, 3),
                              nonlinearity='relu',
                              use_batchnorm=True,
                              use_transpose=True,
                              strides=(2, 2),
                              data_format='channels_last')
        self.up_3 = Up_Conv2D(num_channels * 2,
                              kernel_size=(3, 3),
                              nonlinearity='relu',
                              use_batchnorm=True,
                              use_transpose=True,
                              strides=(2, 2),
                              data_format='channels_last')
        self.up_4 = Up_Conv2D(num_channels,
                              kernel_size=(3, 3),
                              nonlinearity='relu',
                              use_batchnorm=True,
                              use_transpose=True,
                              strides=(2, 2),
                              data_format='channels_last')

        self.respath_1 = ResPath(res_path_length,
                                 num_channels,
                                 kernel_size=(1, 1),
                                 nonlinearity='relu',
                                 padding='same',
                                 strides=(1, 1),
                                 data_format='channels_last')
        self.respath_2 = ResPath(res_path_length,
                                 num_channels * 2,
                                 kernel_size=(1, 1),
                                 nonlinearity='relu',
                                 padding='same',
                                 strides=(1, 1),
                                 data_format='channels_last')
        self.respath_3 = ResPath(res_path_length,
                                 num_channels * 4,
                                 kernel_size=(1, 1),
                                 nonlinearity='relu',
                                 padding='same',
                                 strides=(1, 1),
                                 data_format='channels_last')
        self.respath_4 = ResPath(res_path_length,
                                 num_channels * 8,
                                 kernel_size=(1, 1),
                                 nonlinearity='relu',
                                 padding='same',
                                 strides=(1, 1),
                                 data_format='channels_last')

        self.conv_1x1 = Conv2D_Block(num_classes,
                                     num_conv_layers,
                                     (1, 1),
                                     'softmax',
                                     use_batchnorm=False,
                                     data_format=data_format)

    def call(self, x, training=False):

        # ENCODER PATH

        x1 = self.mresblock_1(x, training=training)
        pool_1 = self.pool_1(x1)
        res_1 = self.respath_1(x1, training=training)

        x2 = self.mresblock_2(pool_1, training=training)
        pool_2 = self.pool_2(x2)
        res_2 = self.respath_2(x2, training=training)

        x3 = self.mresblock_3(pool_2, training=training)
        pool_3 = self.pool_3(x3)
        res_3 = self.respath_3(x3, training=training)

        x4 = self.mresblock_4(pool_3, training=training)
        pool_4 = self.pool_4(x4)
        res_4 = self.respath_4(x4, training=training)

        x5 = self.mresblock_5(pool_4, training=training)

        up6 = tf.keras.layers.concatenate([self.up_1(x5), res_4])
        x6 = self.mresblock_6(up6, training=training)

        up7 = tf.keras.layers.concatenate([self.up_2(x6), res_3])
        x7 = self.mresblock_7(up7, training=training)

        up8 = tf.keras.layers.concatenate([self.up_3(x7), res_2])
        x8 = self.mresblock_8(up8, training=training)

        up9 = tf.keras.layers.concatenate([self.up_4(x8), res_1])
        x9 = self.mresblock_9(up9, training=training)

        output = self.conv_1x1(x9, training=training)

        return output
