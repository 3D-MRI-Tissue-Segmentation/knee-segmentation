import tensorflow as tf


class Conv3D_Block(tf.keras.layers.Layer):

    def __init__(self,
                 num_channels,
                 num_conv_layers=2,
                 kernel_size=(3, 3, 3),
                 nonlinearity='relu',
                 use_batchnorm=True,
                 use_dropout=True,
                 dropout_rate=0.25,
                 use_spatial_dropout=True,
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

    def call(self, inputs, training):
        x = inputs
        for i in range(self.num_conv_layers):
            x = self.conv[i](x)
            if self.use_batchnorm:
                x = self.batchnorm(x)
            x = self.activation(x)
            if training:
                if self.use_dropout:
                    x = self.dropout(x)
        return x


class Up_Conv3D(tf.keras.layers.Layer):

    def __init__(self,
                 num_channels,
                 kernel_size=(2, 2, 2),
                 nonlinearity='relu',
                 use_batchnorm=False,
                 use_transpose=False,
                 strides=(2, 2, 2),
                 upsample_size=(2, 2, 2),
                 data_format='channels_last',
                 name="upsampling_convolution_block"):

        super(Up_Conv3D, self).__init__(name=name)

        self.use_batchnorm = use_batchnorm
        self.upsample = tf.keras.layers.UpSampling3D(size=upsample_size)
        self.conv = tf.keras.layers.Conv3D(num_channels, kernel_size,
                                           padding='same', data_format=data_format)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.activation = tf.keras.layers.Activation(nonlinearity)
        self.use_transpose = use_transpose
        self.conv_transpose = tf.keras.layers.Conv3DTranspose(num_channels, kernel_size, padding='same',
                                                              strides=strides, data_format=data_format)

    def call(self, inputs, training):

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
