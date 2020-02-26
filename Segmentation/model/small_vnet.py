import tensorflow as tf
import tensorflow.keras.backend as K

class Conv3D_Block(tf.keras.layers.Layer):

    def __init__(self,
                 num_channels,
                 num_conv_layers=1,
                 kernel_size=(2, 2, 2),
                 nonlinearity='relu',
                 use_batchnorm=False,
                 use_dropout=False,
                 dropout_rate=0.25,
                 use_spatial_dropout=False,
                 data_format='channels_last',
                 name="convolution_block"):

        super(Conv3D_Block, self).__init__(name=name)

        self.num_conv_layers = num_conv_layers
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.use_spatial_dropout = use_spatial_dropout

        self.conv = []
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=-1)
        self.activation = tf.keras.layers.Activation(nonlinearity)

        if use_spatial_dropout:
            self.dropout = tf.keras.layers.SpatialDropout3D(rate=dropout_rate)
        else:
            self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        for _ in range(num_conv_layers):
            self.conv.append(tf.keras.layers.Conv3D(num_channels, kernel_size,
                                                    padding='same', data_format=data_format))

    def call(self, inputs):
        x = inputs
        for i in range(self.num_conv_layers):
            x = self.conv[i](x)
            if self.use_batchnorm:
                x = self.batchnorm(x)
            x = self.activation(x)

        if self.use_dropout:
            x = self.dropout(x)
        outputs = x

        # tf.print("3d block:", outputs.get_shape())
        return outputs

class Up_Conv3D(tf.keras.layers.Layer):

    def __init__(self,
                 num_channels,
                 kernel_size=(2, 2, 2),
                 nonlinearity='relu',
                 use_batchnorm=False,
                 use_transpose=False,
                 strides=(2, 2, 2),
                 data_format='channels_last',
                 name="upsampling_convolution_block"):

        super(Up_Conv3D, self).__init__(name=name)

        self.use_batchnorm = use_batchnorm
        self.upsample = tf.keras.layers.UpSampling3D(size=(2, 2, 2))
        self.conv = tf.keras.layers.Conv3D(num_channels, kernel_size,
                                           padding='same', data_format=data_format)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.activation = tf.keras.layers.Activation(nonlinearity)
        self.use_transpose = use_transpose
        self.conv_transpose = tf.keras.layers.Conv3DTranspose(num_channels, kernel_size, padding='same',
                                                              strides=strides, data_format=data_format)

    def call(self, inputs):

        x = inputs
        if self.use_transpose:
            x = self.conv_transpose(x)
        else:
            x = self.upsample(x)
            x = self.conv(x)
        if self.use_batchnorm:
            x = self.batch_norm(x)
        outputs = self.activation(x)

        return outputs


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
                 name="vnet_small"):

        super(VNet_Small, self).__init__(name=name)
        self.merge_connections = merge_connections

        self.conv_1 = Conv3D_Block(num_channels, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c1")
        self.conv_2 = Conv3D_Block(num_channels * 2, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c2")
        self.conv_3 = Conv3D_Block(num_channels * 4, 1, kernel_size,
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
        self.conv_output = tf.keras.layers.Conv3D(2, kernel_size, activation=nonlinearity, padding='same',
                                                  data_format=data_format)
        self.conv_1x1 = tf.keras.layers.Conv3D(num_classes, kernel_size, padding='same',
                                               data_format=data_format)

    def call(self, inputs):

        # 1->64
        x1 = self.conv_1(inputs)
        # tf.print("x1:", x1.get_shape())

        # 64->128
        x2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x1)
        x2 = self.conv_2(x2)
        # tf.print("x2:", x2.get_shape())

        # 128->256
        x3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x2)
        x3 = self.conv_3(x3)
        # tf.print("x3:", x3.get_shape())

        # 256->128
        u3 = self.up_3(x3)
        u3 = self.up_conv2(u3)
        # tf.print("u3:", u3.get_shape())

        # 128->64
        u2 = self.up_2(x2)
        u2 = self.up_conv1(u2)
        # tf.print("u2:", u2.get_shape())

        u1 = self.conv_output(u2)
        # tf.print("u1:", u1.get_shape())
        output = self.conv_1x1(u1)

        # tf.print("output:", output.get_shape())

        return output
