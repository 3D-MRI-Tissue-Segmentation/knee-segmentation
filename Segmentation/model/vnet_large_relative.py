import tensorflow as tf
from Segmentation.model.vnet_build_blocks import Conv3D_Block, Up_Conv3D


class VNet_Large_Relative(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 num_classes,
                 num_conv_layers=3,
                 start_kernel_size=(5, 5, 5),
                 kernel_size=(3, 3, 3),
                 nonlinearity='relu',
                 use_batchnorm=True,
                 dropout_rate=0.25,
                 use_spatial_dropout=True,
                 data_format='channels_last',
                 merge_connections=True,
                 action='add',
                 output_activation=None,
                 name="vnet_large_relative"):

        super(VNet_Large_Relative, self).__init__(name=name)
        self.merge_connections = merge_connections
        self.num_classes = num_classes
        self.action = action
        assert self.action in ['add', 'multiply'], f"{self.action} not in action list"

        self.conv_1 = Conv3D_Block(num_channels, num_conv_layers, start_kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c1")
        self.conv_2 = Conv3D_Block(num_channels * 2, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c2")
        self.conv_3 = Conv3D_Block(num_channels * 4, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c3")
        self.conv_4 = Conv3D_Block(num_channels * 8, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c4")
        self.conv_5 = Conv3D_Block(num_channels * 16, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c5")
        self.up_5 = Up_Conv3D(num_channels * 8, (2, 2, 2), nonlinearity,
                              use_batchnorm=use_batchnorm, data_format=data_format, name="cu5")
        self.up_4 = Up_Conv3D(num_channels * 4, (2, 2, 2), nonlinearity,
                              use_batchnorm=use_batchnorm, data_format=data_format, name="cu4")
        self.up_3 = Up_Conv3D(num_channels * 2, (2, 2, 2), nonlinearity,
                              use_batchnorm=use_batchnorm, data_format=data_format, name="cu3")
        self.up_2 = Up_Conv3D(num_channels, (2, 2, 2), nonlinearity,
                              use_batchnorm=use_batchnorm, data_format=data_format, name="cu2")
        self.up_conv4 = Conv3D_Block(num_channels * 8, num_conv_layers, kernel_size, nonlinearity,
                                     use_batchnorm=use_batchnorm, data_format=data_format, name="upc4")
        self.up_conv3 = Conv3D_Block(num_channels * 4, num_conv_layers, kernel_size, nonlinearity,
                                     use_batchnorm=use_batchnorm, data_format=data_format, name="upc3")
        self.up_conv2 = Conv3D_Block(num_channels * 2, num_conv_layers, kernel_size, nonlinearity,
                                     use_batchnorm=use_batchnorm, data_format=data_format, name="upc2")
        self.up_conv1 = Conv3D_Block(num_channels, num_conv_layers, kernel_size, nonlinearity,
                                     use_batchnorm=use_batchnorm, data_format=data_format, name="upc1")

        self.pos_dense_1 = tf.keras.layers.Dense(16, activation="relu")
        self.pos_dense_2 = tf.keras.layers.Dense(4, activation="relu")
        self.pos_dense_3 = tf.keras.layers.Dense(1, activation="tanh")

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

    def call(self, inputs, training=True):
        image_inputs, pos_inputs = inputs

        pos_1 = self.pos_dense_1(pos_inputs)
        pos_2 = self.pos_dense_2(pos_1)
        pos_3 = self.pos_dense_3(pos_2)

        # 1->64
        x1 = self.conv_1(inputs)
        # 64->128
        x2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x1)
        x2 = self.conv_2(x2)
        # 128->256
        x3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x2)
        x3 = self.conv_3(x3)
        # 256->512
        x4 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x3)
        x4 = self.conv_4(x4)
        # 512->1024
        x5 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x4)
        x5 = self.conv_3(x5)

        # 1024->512
        u5 = self.up_5(x5)
        if self.action == 'add':
            u5 = tf.keras.layers.add([u5, pos_3])
        elif self.action == 'multiply':
            u5 = tf.keras.layers.multiply([u5, pos_3])
        if self.merge_connections:
            u5 = tf.keras.layers.concatenate([x4, u5], axis=4)
        u5 = self.up_conv4(u5)
        # 512->256
        u4 = self.up_4(u5)
        if self.merge_connections:
            u4 = tf.keras.layers.concatenate([x3, u4], axis=4)
        u4 = self.up_conv3(u4)
        # 256->128
        u3 = self.up_3(u4)
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
