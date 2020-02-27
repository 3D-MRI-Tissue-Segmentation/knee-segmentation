import tensorflow as tf
from Segmentation.model.vnet_build_blocks import Conv3D_Block, Up_Conv3D

class VNet_Tiny(tf.keras.Model):

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
                 name="vnet_tiny"):

        super(VNet_Tiny, self).__init__(name=name)
        self.merge_connections = merge_connections
        self.num_classes = num_classes

        self.conv_1 = Conv3D_Block(num_channels, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format)
        self.conv_2 = Conv3D_Block(num_channels * 2, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format)
        self.up_2 = Up_Conv3D(num_channels, (2, 2, 2), nonlinearity,
                              use_batchnorm=use_batchnorm, data_format=data_format)
        self.up_conv1 = Conv3D_Block(num_channels, num_conv_layers, kernel_size, nonlinearity,
                                     use_batchnorm=use_batchnorm, data_format=data_format)

        # convolution num_channels at the output
        self.conv_output = tf.keras.layers.Conv3D(num_classes, kernel_size, activation=nonlinearity, padding='same',
                                                  data_format=data_format)
        if output_activation is None:
            output_activation = 'sigmoid'
            if num_classes > 1:
                output_activation = 'softmax'
        # self.conv_1x1 = tf.keras.layers.Conv3D(num_classes, kernel_size, padding='same',
        #                                        data_format=data_format, activation=output_activation)
        if num_classes == 1:
            self.conv_1x1_binary = tf.keras.layers.Conv3D(num_classes, (1, 1, 1), activation='sigmoid',
                                                          padding='same', data_format=data_format)
        else:
            print("------------------------")
            print(num_classes)
            self.conv_1x1 = tf.keras.layers.Conv3D(num_classes, kernel_size=(1,1,1), activation='linear',
                                                   padding='same', data_format=data_format)

    def call(self, inputs):

        # 1->64
        x1 = self.conv_1(inputs)
        tf.print("x1:", x1.get_shape())

        # 64->128
        x2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x1)
        tf.print("x2:", x2.get_shape())
        x2 = self.conv_2(x2)
        tf.print("x2:", x2.get_shape())

        # 128->64
        u2 = self.up_2(x2)
        tf.print("u2:", u2.get_shape())
        if self.merge_connections:
            u2 = tf.keras.layers.concatenate([x1, u2], axis=4)
        tf.print("u2m:", u2.get_shape())
        u2 = self.up_conv1(u2)
        tf.print("u2:", u2.get_shape())

        u1 = self.conv_output(u2)
        tf.print("u1:", u1.get_shape())
        output = self.conv_1x1(u1)
        tf.print("output:", output.get_shape())

        if self.num_classes == 1:
            output = self.conv_1x1_binary(output)
        else:
            tf.print(self.num_classes)
            output = self.conv_1x1(output)
            tf.print("output:", output.get_shape())
            output = tf.keras.layers.Reshape((inputs.shape[1] * inputs.shape[2] * inputs.shape[3], self.num_classes))(output)
            tf.print("output:", output.get_shape())
            output = tf.keras.layers.Activation('softmax')(output)
        tf.print("output:", output.get_shape())
        return output
