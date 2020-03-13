import tensorflow as tf
from Segmentation.model.vnet_build_blocks import Conv3D_Block, Up_Conv3D


class VNet_Small(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 num_classes,
                 num_conv_layers=2,
                 kernel_size=(3, 3, 3),
                 nonlinearity='relu',
                 use_batchnorm=True,
                 dropout_rate=0.25,
                 use_spatial_dropout=True,
                 data_format='channels_last',
                 merge_connections=False,
                 output_activation=None,
                 noise=0.0001,
                 name="vnet_small"):

        super(VNet_Small, self).__init__(name=name)
        self.merge_connections = merge_connections
        self.num_classes = num_classes
        self.noise = noise

        self.conv_1 = Conv3D_Block(num_channels, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c1")
        self.conv_2 = Conv3D_Block(num_channels * 2, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c2")
        self.conv_3 = Conv3D_Block(num_channels * 4, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c3")
        self.up_3 = Up_Conv3D(num_channels * 2, (2, 2, 2), nonlinearity,
                              use_batchnorm=use_batchnorm, data_format=data_format, name="cu3")
        self.up_2 = Up_Conv3D(num_channels, (2, 2, 2), nonlinearity,
                              use_batchnorm=use_batchnorm, data_format=data_format, name="cu2")
        self.up_conv2 = Conv3D_Block(num_channels * 2, num_conv_layers, kernel_size, nonlinearity,
                                     use_batchnorm=use_batchnorm, data_format=data_format, name="upc2")
        self.up_conv1 = Conv3D_Block(num_channels, num_conv_layers, kernel_size, nonlinearity,
                                     use_batchnorm=use_batchnorm, data_format=data_format, name="upc1")

        # convolution num_channels at the output
        self.conv_output = tf.keras.layers.Conv3D(num_classes, kernel_size, activation=nonlinearity, padding='same',
                                                  data_format=data_format)
        if output_activation is None:
            output_activation = 'sigmoid'
            if num_classes > 1:
                output_activation = 'softmax'
        if num_classes == 1:
            self.conv_1x1_binary = tf.keras.layers.Conv3D(num_classes, (1, 1, 1), activation='sigmoid',
                                                          padding='same', data_format=data_format)
        else:
            self.conv_1x1 = tf.keras.layers.Conv3D(num_classes, kernel_size=(1, 1, 1), activation='softmax',
                                                   padding='same', data_format=data_format)

    def call(self, inputs, training=False):
        
        if self.noise and training:
            inputs = tf.keras.layers.GaussianNoise(self.noise)(inputs)

        # 1->64
        x1 = self.conv_1(inputs)

        # 64->128
        x2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x1)
        x2 = self.conv_2(x2)

        # 128->256
        x3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x2)
        x3 = self.conv_3(x3)

        # 256->128
        u3 = self.up_3(x3)
        if self.merge_connections:
            u3 = tf.keras.layers.concatenate([x2, u3], axis=4)
        u3 = self.up_conv2(u3)

        # 128->64
        u2 = self.up_2(u3)
        if self.merge_connections:
            u2 = tf.keras.layers.concatenate([x1, u2], axis=4)
        u2 = self.up_conv1(u2)

        output = self.conv_output(u2)

        if self.num_classes == 1:
            output = self.conv_1x1_binary(output)
        else:
            output = self.conv_1x1(output)
        return output
