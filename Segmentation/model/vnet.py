import tensorflow as tf
import inspect
from Segmentation.model.vnet_build_blocks import Conv3d_ResBlock, Up_ResBlock

class VNet(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 num_classes,
                 num_conv_layers=2,
                 kernel_size=(2, 2, 2),
                 nonlinearity='relu',
                 use_batchnorm=True,
                 noise=0.0,
                 dropout_rate=0.25,
                 use_spatial_dropout=True,
                 use_slice=False,
                 slice_format=None,
                 data_format='channels_last',
                 name="vnet"):

        self.params = str(inspect.currentframe().f_locals)
        super(VNet, self).__init__(name=name)
        self.noise = noise
        self.use_slice = use_slice
        if self.use_slice:
            assert kernel_size[-1] == 1
        self.slice_format = slice_format

        block_args = {
            'num_conv_layers': num_conv_layers,
            'kernel_size': kernel_size,
            'nonlinearity': nonlinearity,
            'use_batchnorm': use_batchnorm,
            'dropout_rate': dropout_rate,
            'use_spatial_dropout': use_spatial_dropout,
            'data_format': data_format,
        }

        self.conv_1 = Conv3d_ResBlock(num_channels=num_channels, **block_args)
        self.conv_2 = Conv3d_ResBlock(num_channels=num_channels * 2, **block_args)
        self.conv_3 = Conv3d_ResBlock(num_channels=num_channels * 4, **block_args)
        self.conv_4 = Conv3d_ResBlock(num_channels=num_channels * 8, **block_args)

        self.upconv_4 = Up_ResBlock(num_channels=num_channels * 8, **block_args)
        self.upconv_3 = Up_ResBlock(num_channels=num_channels * 4, **block_args)
        self.upconv_2 = Up_ResBlock(num_channels=num_channels * 2, **block_args)
        self.upconv_1 = Up_ResBlock(num_channels=num_channels, **block_args)

        # convolution num_channels at the output
        self.conv_output = tf.keras.layers.Conv3D(filters=num_classes, kernel_size=kernel_size, activation=nonlinearity, padding='same', data_format=data_format)
        self.conv_1x1 = tf.keras.layers.Conv3D(filters=num_classes, kernel_size=(1, 1, 1), activation="sigmoid", padding='same', data_format=data_format)

    def call(self, inputs, training):
        
        tf.print(inputs.shape)

        if self.noise and training:
            inputs = tf.keras.layers.GaussianNoise(self.noise)(inputs)

        # encoder blocks
        x1, x1_before = self.conv_1(inputs, training)
        x2, x2_before = self.conv_2(x1, training)
        x3, x3_before = self.conv_3(x2, training)
        x4, x4_before = self.conv_4(x3, training)

        # decoder blocks
        u4 = self.upconv_4([x4, x4_before], training)
        u3 = self.upconv_3([u4, x3_before], training)
        u2 = self.upconv_2([u3, x2_before], training)
        u1 = self.upconv_1([u2, x1_before], training)

        output = self.conv_output(u1)
        output = self.conv_1x1(output)

        if self.use_slice:
            if self.slice_format == "mean":
                output = tf.reduce_mean(output, -2)
        return output
