import tensorflow as tf
import tensorflow.keras.layers as tfkl


class Conv2D_Block(tf.keras.Sequential):

    def __init__(self,
                 num_channels,
                 num_conv_layers=2,
                 kernel_size=(3, 3),
                 nonlinearity='relu',
                 use_batchnorm=False,
                 use_bias=True,
                 use_dropout=False,
                 dropout_rate=0.25,
                 use_spatial_dropout=True,
                 data_format='channels_last',
                 **kwargs):

        super(Conv2D_Block, self).__init__(**kwargs)

        self.num_channels = num_channels
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.nonlinearity = nonlinearity
        self.use_batchnorm = use_batchnorm
        self.use_bias = use_bias
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_spatial_dropout = use_spatial_dropout
        self.data_format = data_format

        for _ in range(self.num_conv_layers):
            self.add(tfkl.Conv2D(self.num_channels,
                                 self.kernel_size,
                                 padding='same',
                                 use_bias=self.use_bias,
                                 data_format=self.data_format))
            if self.use_batchnorm:
                self.add(tfkl.BatchNormalization(axis=-1,
                                                 momentum=0.95,
                                                 epsilon=0.001))
            self.add(tfkl.Activation(self.nonlinearity))

        if self.use_dropout:
            if self.use_spatial_dropout:
                self.add(tfkl.SpatialDropout2D(rate=self.dropout_rate))
            else:
                self.add(tfkl.Dropout(rate=self.dropout_rate))

    def call(self, inputs, training=False):

        outputs = super(Conv2D_Block, self).call(inputs, training=training)

        return outputs


class Up_Conv2D(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 kernel_size=(2, 2),
                 nonlinearity='relu',
                 use_attention=False,
                 use_batchnorm=False,
                 use_transpose=False,
                 use_bias=True,
                 strides=(2, 2),
                 data_format='channels_last',
                 **kwargs):

        super(Up_Conv2D, self).__init__(**kwargs)

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.nonlinearity = nonlinearity
        self.use_attention = use_attention
        self.use_batchnorm = use_batchnorm
        self.use_transpose = use_transpose
        self.use_bias = use_bias
        self.strides = strides
        self.data_format = data_format

        if self.use_transpose:
            self.upconv_layer = tfkl.Conv2DTranspose(self.num_channels,
                                                     self.kernel_size,
                                                     padding='same',
                                                     strides=self.strides,
                                                     data_format=self.data_format)
        else:
            self.upconv_layer = tfkl.UpSampling2D(size=self.strides)

        if self.use_attention:
            self.attention = Attention_Gate(num_channels,
                                            (1, 1),
                                            self.nonlinearity,
                                            padding='same',
                                            strides=self.strides,
                                            use_bias=self.use_bias,
                                            data_format=self.data_format)

        self.conv = Conv2D_Block(num_channels,
                                 num_conv_layers=1,
                                 kernel_size=kernel_size,
                                 nonlinearity=nonlinearity,
                                 use_batchnorm=use_batchnorm,
                                 use_dropout=False,
                                 data_format=data_format)

        self.conv_block = Conv2D_Block(num_channels,
                                       num_conv_layers=2,
                                       kernel_size=(3, 3),
                                       nonlinearity=nonlinearity,
                                       use_batchnorm=use_batchnorm,
                                       use_dropout=False,
                                       data_format=data_format)

    def call(self, inputs, bridge, training=False):

        up = self.upconv_layer(inputs)
        up = self.conv(up, training=training)
        if self.use_attention:
            up = self.attention(bridge, up, training=training)
        out = tfkl.concatenate([up, bridge], axis=-1)
        out = self.conv_block(out, training=training)

        return out


class Attention_Gate(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 kernel_size=(1, 1),
                 nonlinearity='relu',
                 padding='same',
                 strides=(1, 1),
                 use_bias=True,
                 use_batchnorm=True,
                 data_format='channels_last',
                 **kwargs):

        super(Attention_Gate, self).__init__(**kwargs)

        self.conv_blocks = []

        for _ in range(3):
            self.conv_blocks.append(Conv2D_Block(num_channels,
                                                 num_conv_layers=1,
                                                 kernel_size=kernel_size,
                                                 nonlinearity=nonlinearity,
                                                 use_batchnorm=use_batchnorm,
                                                 use_dropout=False,
                                                 data_format=data_format))

    def call(self, input_x, input_g, training=False):

        x_g = self.conv_blocks[0](input_g, training=training)
        x_l = self.conv_blocks[1](input_x, training=training)

        x = tfkl.concatenate([x_g, x_l], axis=-1)
        x = tfkl.Activation('relu')(x)

        x = self.conv_blocks[2](x, training=training)
        alpha = tfkl.Activation('sigmoid')(x)

        outputs = tf.math.multiply(alpha, input_x)

        return outputs

class Recurrent_Block(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 kernel_size=(3, 3),
                 nonlinearity='relu',
                 padding='same',
                 strides=(1, 1),
                 t=2,
                 use_batchnorm=True,
                 data_format='channels_last',
                 **kwargs):

        super(Recurrent_Block, self).__init__(**kwargs)

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.nonlinearity = nonlinearity
        self.padding = padding
        self.strides = strides
        self.t = t
        self.use_batchnorm = use_batchnorm
        self.data_format = data_format

        self.conv = Conv2D_Block(num_channels=self.num_channels,
                                 num_conv_layers=1,
                                 kernel_size=self.kernel_size,
                                 nonlinearity=self.nonlinearity,
                                 use_batchnorm=self.use_batchnorm,
                                 data_format=self.data_format)

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
                 kernel_size=(3, 3),
                 nonlinearity='relu',
                 padding='same',
                 strides=(1, 1),
                 t=2,
                 use_batchnorm=True,
                 data_format='channels_last',
                 **kwargs):

        super(Recurrent_ResConv_block, self).__init__(**kwargs)

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.nonlinearity = nonlinearity
        self.padding = padding
        self.strides = strides
        self.t = t
        self.use_batchnorm = use_batchnorm
        self.data_format = data_format

        self.Recurrent_CNN = tf.keras.Sequential([
            Recurrent_Block(self.num_channels,
                            self.kernel_size,
                            self.nonlinearity,
                            self.padding,
                            self.strides,
                            self.t,
                            self.use_batchnorm,
                            self.data_format),
            Recurrent_Block(self.num_channels,
                            self.kernel_size,
                            self.nonlinearity,
                            self.padding,
                            self.strides,
                            self.t,
                            self.use_batchnorm,
                            self.data_format)])

        self.Conv_1x1 = tf.keras.layers.Conv2D(self.num_channels,
                                               kernel_size=(1, 1),
                                               strides=self.strides,
                                               padding=self.padding,
                                               data_format=self.data_format)

    def call(self, x):
        x = self.Conv_1x1(x)
        x1 = self.Recurrent_CNN(x)
        output = tfkl.Add()([x, x1])

        return output
