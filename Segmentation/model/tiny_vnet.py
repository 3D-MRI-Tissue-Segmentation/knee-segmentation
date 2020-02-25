import tensorflow as tf

class Conv3D_Block(tf.keras.layers.Layer):

    def __init__(self,
                 num_channels,
                 num_conv_layers=2,
                 kernel_size=(3, 3, 3),
                 nonlinearity='relu',
                 use_batchnorm=False,
                 use_dropout=False,
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
            self.conv.append(tf.keras.layers.Conv3D(num_channels, kernel_size, padding='same', data_format=data_format))

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
        self.conv = tf.keras.layers.Conv3D(num_channels, kernel_size, padding='same', data_format=data_format)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.activation = tf.keras.layers.Activation(nonlinearity)
        self.use_transpose = use_transpose
        self.conv_transpose = tf.keras.layers.Conv3DTranspose(num_channels, kernel_size, padding='same', strides=strides, data_format=data_format)

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
                 name="vnet_tiny"):

        super(VNet_Tiny, self).__init__(name=name)

        self.conv_1 = Conv3D_Block(num_channels, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format)
        self.conv_2 = Conv3D_Block(num_channels * 2, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format)
        # self.conv_3 = Conv3D_Block(num_channels * 4, num_conv_layers, kernel_size,
        #                            nonlinearity, use_batchnorm=use_batchnorm,
        #                            data_format=data_format)
        # self.conv_4 = Conv3D_Block(num_channels * 8, num_conv_layers, kernel_size,
        #                            nonlinearity, use_batchnorm=use_batchnorm,
        #                            use_dropout=True, dropout_rate=dropout_rate, use_spatial_dropout=use_spatial_dropout,
        #                            data_format=data_format)
        # self.conv_5 = Conv3D_Block(num_channels * 16, num_conv_layers, kernel_size,
        #                            nonlinearity, use_batchnorm=use_batchnorm,
        #                            use_dropout=True, dropout_rate=dropout_rate, use_spatial_dropout=use_spatial_dropout,
        #                            data_format=data_format)

        # self.up_5 = Up_Conv3D(num_channels * 8, (2, 2, 2), nonlinearity, use_batchnorm=use_batchnorm, data_format=data_format)
        # self.up_6 = Up_Conv3D(num_channels * 4, (2, 2, 2), nonlinearity, use_batchnorm=use_batchnorm, data_format=data_format)
        # self.up_7 = Up_Conv3D(num_channels * 2, (2, 2, 2), nonlinearity, use_batchnorm=use_batchnorm, data_format=data_format)
        self.up_8 = Up_Conv3D(num_channels, (2, 2, 2), nonlinearity, use_batchnorm=use_batchnorm, data_format=data_format)

        # self.up_conv4 = Conv3D_Block(num_channels*8,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        # self.up_conv3 = Conv3D_Block(num_channels*4,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        # self.up_conv2 = Conv3D_Block(num_channels*2,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.up_conv1 = Conv3D_Block(num_channels, num_conv_layers, kernel_size, nonlinearity, use_batchnorm=use_batchnorm, data_format=data_format)

        # convolution num_channels at the output
        self.conv_output = tf.keras.layers.Conv3D(2, kernel_size, activation=nonlinearity, padding='same', data_format=data_format)
        self.conv_1x1 = tf.keras.layers.Conv3D(num_classes, kernel_size, padding='same', data_format=data_format)

    def call(self, inputs):

        # encoder blocks
        # 1->64
        x1 = self.conv_1(inputs)

        # 64->128
        x2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x1)
        x2 = self.conv_2(x2)

        # 128->256
        # x3 = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2))(x2)
        # x3 = self.conv_3(x3)

        # 256->512
        # x4 = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2))(x3)
        # x4 = self.conv_4(x4)

        # 512->1024
        # x5 = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2))(x4)
        # x5 = self.conv_5(x5)

        # decoder blocks
        # 1024->512
        # u5 = self.up_5(x5)
        # u5 = tf.keras.layers.concatenate([x4, u5], axis=3)
        # u5 = self.up_conv4(u5)

        # 512->256
        # u6 = self.up_6(u5)
        # u6 = tf.keras.layers.concatenate([x3, u6], axis=3)
        # u6 = self.up_conv3(u6)

        # 256->128
        # u7 = self.up_7(u6)
        # u7 = tf.keras.layers.concatenate([x2, u7], axis=3)
        # u7 = self.up_conv2(u7)

        # 128->64
        u8 = self.up_8(x2)
        # u8 = tf.keras.layers.concatenate([x1, u8], axis=3, name="bob")
        u8 = self.up_conv1(u8)

        u9 = self.conv_output(u8)
        output = self.conv_1x1(u9)

        return output
