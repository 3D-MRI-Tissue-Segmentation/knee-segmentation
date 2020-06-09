import tensorflow as tf


class Conv3d_ResBlock(tf.keras.layers.Layer):
    def __init__(self,
                 num_channels,
                 kernel_size=(3, 3, 3),
                 strides=(2, 2, 2),
                 res_activation="relu",
                 name="conv_res_block",
                 **kwargs):
        super(Conv3d_ResBlock, self).__init__(name=name)

        self.conv_block = Conv3D_Block(num_channels=num_channels, kernel_size=kernel_size, **kwargs)
        self.conv_stride = tf.keras.layers.Conv3D(num_channels * 2, kernel_size=kernel_size, strides=strides, activation=res_activation, padding="same")

    def call(self, inputs, training):
        x = inputs
        x = self.conv_block(x, training=training)
        x = tf.keras.layers.add([x, inputs])
        down_x = self.conv_stride(x)
        return down_x, x


class Up_ResBlock(tf.keras.layers.Layer):
    def __init__(self,
                 num_channels,
                 kernel_size=(3, 3, 3),
                 name="upsampling_conv_res_block",
                 **kwargs):
        super(Up_ResBlock, self).__init__(name=name)
        self.num_channels = num_channels
        self.up_conv = Up_Conv3D(num_channels=num_channels, kernel_size=kernel_size, **kwargs)
        self.conv_block = Conv3D_Block(num_channels=num_channels, kernel_size=kernel_size, **kwargs)

    def call(self, inputs, training):

        # tf.print(self.num_channels)
        

        x, x_highway = inputs


        # tf.print(tf.shape(x))
        # tf.print(tf.shape(x_highway))
        # tf.print("================")
        x_res_start = self.up_conv(x, training=training)
        x_up = tf.keras.layers.concatenate([x_res_start, x_highway], axis=-1)
        x = self.conv_block(x_up)
        x = tf.keras.layers.add([x, x_res_start])
        return x


class Conv3D_Block(tf.keras.layers.Layer):

    def __init__(self,
                 num_channels,
                 num_conv_layers=2,
                 kernel_size=(3, 3, 3),
                 activation='relu',
                 use_batchnorm=True,
                 use_dropout=True,
                 dropout_rate=0.25,
                 use_spatial_dropout=True,
                 data_format='channels_last',
                 name="convolution_block",
                 **kwargs):

        super(Conv3D_Block, self).__init__(name=name)

        self.num_conv_layers = num_conv_layers
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.use_spatial_dropout = use_spatial_dropout

        self.conv = []
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=-1)
        if activation is 'prelu':
            self.activation = tf.keras.layers.PReLU()#alpha_initializer=tf.keras.initializers.Constant(value=0.25))
        else:
            self.activation = tf.keras.layers.Activation(activation)

        if use_spatial_dropout:
            self.dropout = tf.keras.layers.SpatialDropout3D(rate=dropout_rate)
        else:
            self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        for _ in range(num_conv_layers):
            self.conv.append(tf.keras.layers.Conv3D(filters=num_channels, kernel_size=kernel_size,
                                                    padding='same', data_format=data_format))

    def call(self, inputs, training):
        x = inputs
        for i in range(self.num_conv_layers):
            x = self.conv[i](x)
            if training:
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
                 activation='relu',
                 use_batchnorm=False,
                 use_transpose=False,
                 strides=(2, 2, 2),
                 upsample_size=(2, 2, 2),
                 data_format='channels_last',
                 name="upsampling_conv_block",
                 **kwargs):

        super(Up_Conv3D, self).__init__(name=name)

        self.use_batchnorm = use_batchnorm
        self.upsample = tf.keras.layers.UpSampling3D(size=upsample_size)
        self.conv = tf.keras.layers.Conv3D(filters=num_channels, kernel_size=kernel_size,
                                           padding='same', data_format=data_format)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        if activation is 'prelu':
            self.activation = tf.keras.layers.PReLU()#alpha_initializer=tf.keras.initializers.Constant(value=0.25))
        else:
            self.activation = tf.keras.layers.Activation(activation)
        self.use_transpose = use_transpose
        self.conv_transpose = tf.keras.layers.Conv3DTranspose(filters=num_channels, kernel_size=kernel_size, padding='same',
                                                              strides=strides, data_format=data_format)

    def call(self, inputs, training):
        x = inputs
        if self.use_transpose:
            x = self.conv_transpose(x)
        else:
            x = self.upsample(x)
            x = self.conv(x)
        if training:      
            if self.use_batchnorm:
                x = self.batch_norm(x)
        outputs = self.activation(x)
        return outputs
