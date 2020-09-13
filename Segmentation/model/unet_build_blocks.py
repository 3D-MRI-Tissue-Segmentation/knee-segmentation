import tensorflow as tf
import tensorflow.keras.layers as tfkl


class Conv_Block(tf.keras.Sequential):

    def __init__(self,
                 num_channels,
                 use_2d=True,
                 num_conv_layers=2,
                 kernel_size=3,
                 activation='relu',
                 use_batchnorm=False,
                 use_bias=True,
                 use_dropout=False,
                 dropout_rate=0.25,
                 use_spatial_dropout=True,
                 data_format='channels_last',
                 name="convolution_block",
                 **kwargs):

        super(Conv_Block, self).__init__(name=name)

        for _ in range(num_conv_layers):
            if use_2d:
                self.add(tfkl.Conv2D(num_channels,
                                     kernel_size,
                                     padding='same',
                                     use_bias=use_bias,
                                     data_format=data_format))
            else:
                self.add(tfkl.Conv3D(num_channels,
                                     kernel_size,
                                     padding='same',
                                     use_bias=use_bias,
                                     data_format=data_format))
            if use_batchnorm:
                self.add(tfkl.BatchNormalization(axis=-1 if data_format == 'channels_last' else 1,
                                                 momentum=0.95,
                                                 epsilon=0.001))
            if activation == 'prelu':
                self.add(tfkl.PReLU())
            else:
                self.add(tfkl.Activation(activation))

        if use_dropout:
            if use_spatial_dropout:
                if use_2d:
                    self.add(tfkl.SpatialDropout2D(rate=dropout_rate))
                else:
                    self.add(tfkl.SpatialDropout3D(rate=dropout_rate))
            else:
                self.add(tfkl.Dropout(rate=dropout_rate))

    def call(self, inputs, training=False):

        outputs = super(Conv_Block, self).call(inputs, training=training)

        return outputs


class Up_Conv(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 use_2d=True,
                 kernel_size=2,
                 activation='relu',
                 use_attention=False,
                 use_batchnorm=False,
                 use_transpose=False,
                 use_bias=True,
                 strides=2,
                 data_format='channels_last',
                 name="upsampling_conv_block",
                 **kwargs):

        super(Up_Conv, self).__init__(name=name)

        self.data_format = data_format
        self.use_attention = use_attention

        if use_transpose:
            if use_2d:
                self.upconv_layer = tfkl.Conv2DTranspose(num_channels,
                                                         kernel_size,
                                                         padding='same',
                                                         strides=strides,
                                                         data_format=self.data_format)
            else:
                self.upconv_layer = tfkl.Conv3DTranspose(num_channels,
                                                         kernel_size,
                                                         padding='same',
                                                         strides=strides,
                                                         data_format=self.data_format)
        else:
            if use_2d:
                self.upconv_layer = tfkl.UpSampling2D(size=strides)
            else:
                self.upconv_layer = tfkl.UpSampling3D(size=strides)

        if self.use_attention:
            self.attention = Attention_Gate(num_channels=num_channels,
                                            use_2d=use_2d,
                                            kernel_size=1,
                                            activation=activation,
                                            padding='same',
                                            strides=strides,
                                            use_bias=use_bias,
                                            data_format=self.data_format)

        self.conv = Conv_Block(num_channels=num_channels,
                               use_2d=use_2d,
                               num_conv_layers=1,
                               kernel_size=kernel_size,
                               activation=activation,
                               use_batchnorm=use_batchnorm,
                               use_dropout=False,
                               data_format=self.data_format)

        self.conv_block = Conv_Block(num_channels=num_channels,
                                     use_2d=use_2d,
                                     num_conv_layers=2,
                                     kernel_size=3,
                                     activation=activation,
                                     use_batchnorm=use_batchnorm,
                                     use_dropout=False,
                                     data_format=self.data_format)

    def call(self, inputs, bridge, training=False):

        up = self.upconv_layer(inputs)
        up = self.conv(up, training=training)
        if self.use_attention:
            up = self.attention(bridge, up, training=training)
        out = tfkl.concatenate([up, bridge], axis=-1 if self.data_format == 'channels_last' else 1)
        out = self.conv_block(out, training=training)

        return out


class Attention_Gate(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 use_2d=True,
                 kernel_size=1,
                 activation='relu',
                 padding='same',
                 strides=1,
                 use_bias=True,
                 use_batchnorm=True,
                 data_format='channels_last',
                 name='attention_gate',
                 **kwargs):

        super(Attention_Gate, self).__init__(name=name)

        self.conv_blocks = []
        self.data_format = data_format

        for _ in range(3):
            self.conv_blocks.append(Conv_Block(num_channels,
                                               use_2d=use_2d,
                                               num_conv_layers=1,
                                               kernel_size=kernel_size,
                                               activation=activation,
                                               use_batchnorm=use_batchnorm,
                                               use_dropout=False,
                                               data_format=self.data_format))

    def call(self, input_x, input_g, training=False):

        x_g = self.conv_blocks[0](input_g, training=training)
        x_l = self.conv_blocks[1](input_x, training=training)

        x = tfkl.concatenate([x_g, x_l], axis=-1 if self.data_format == 'channels_last' else 1)
        x = tfkl.Activation('relu')(x)

        x = self.conv_blocks[2](x, training=training)
        alpha = tfkl.Activation('sigmoid')(x)

        outputs = tf.math.multiply(alpha, input_x)

        return outputs


class Recurrent_Block(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 use_2d=True,
                 kernel_size=3,
                 activation='relu',
                 padding='same',
                 strides=1,
                 t=2,
                 use_batchnorm=True,
                 data_format='channels_last',
                 name='recurrent_block',
                 **kwargs):

        super(Recurrent_Block, self).__init__(name=name)

        self.t = t
        self.conv = Conv_Block(num_channels=num_channels,
                               use_2d=use_2d,
                               num_conv_layers=1,
                               kernel_size=kernel_size,
                               activation=activation,
                               use_batchnorm=use_batchnorm,
                               data_format=data_format)

    def call(self, x, training=False):

        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x, training=training)

            x1 = tfkl.Add()([x, x1])
            x1 = self.conv(x1, training=training)

        return x1


class Recurrent_ResConv_block(tf.keras.Model):
    def __init__(self,
                 num_channels,
                 use_2d=True,
                 kernel_size=3,
                 activation='relu',
                 padding='same',
                 strides=1,
                 t=2,
                 use_batchnorm=True,
                 data_format='channels_last',
                 name='res_recurrent_block',
                 **kwargs):

        super(Recurrent_ResConv_block, self).__init__(name=name)

        self.Recurrent_CNN = tf.keras.Sequential([
            Recurrent_Block(num_channels,
                            use_2d,
                            kernel_size,
                            activation,
                            padding,
                            strides,
                            t,
                            use_batchnorm,
                            data_format),
            Recurrent_Block(num_channels,
                            use_2d,
                            kernel_size,
                            activation,
                            padding,
                            strides,
                            t,
                            use_batchnorm,
                            data_format)])

        if use_2d:
            self.Conv_1x1 = tf.keras.layers.Conv2D(num_channels,
                                                   kernel_size=(1, 1),
                                                   strides=strides,
                                                   padding=padding,
                                                   data_format=data_format)
        else:
            self.Conv_1x1 = tf.keras.layers.Conv3D(num_channels,
                                                   kernel_size=(1, 1, 1),
                                                   strides=strides,
                                                   padding=padding,
                                                   data_format=data_format)

    def call(self, x):
        x = self.Conv_1x1(x)
        x1 = self.Recurrent_CNN(x)
        output = tfkl.Add()([x, x1])

        return output