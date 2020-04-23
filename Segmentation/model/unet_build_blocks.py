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


class Up_Conv2D(tf.keras.Sequential):

    def __init__(self,
                 num_channels,
                 kernel_size=(2, 2),
                 nonlinearity='relu',
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
        self.use_batchnorm = use_batchnorm
        self.use_transpose = use_transpose
        self.use_bias = use_bias
        self.strides = strides
        self.data_format = data_format

        if self.use_transpose:
            self.add(tfkl.Conv2DTranspose(self.num_channels,
                                          self.kernel_size,
                                          padding='same',
                                          strides=self.strides,
                                          data_format=self.data_format))
        else:
            self.add(tfkl.UpSampling2D(size=self.strides))
            self.add(tfkl.Conv2D(self.num_channels,
                                 self.kernel_size,
                                 padding='same',
                                 data_format=self.data_format))
        if self.use_batchnorm:
            self.add(tfkl.BatchNormalization(axis=-1,
                                             momentum=0.95,
                                             epsilon=0.001))
        self.add(tfkl.Activation(self.nonlinearity))

    def call(self, inputs, training=False):

        outputs = super(Up_Conv2D, self).call(inputs, training=training)

        return outputs


class Attention_Gate(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 kernel_size=(1, 1),
                 nonlinearity='relu',
                 padding='same',
                 strides=(1, 1),
                 use_bias=True,
                 data_format='channels_last',
                 name='attention_gate'):

        super(Attention_Gate, self).__init__()
        self.conv_1 = tfkl.Conv2D(num_channels,
                                  kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  use_bias=use_bias,
                                  data_format=data_format)
        self.batch_norm_1 = tfkl.BatchNormalization(axis=-1,
                                                    momentum=0.95,
                                                    epsilon=0.001)

        self.conv_2 = tfkl.Conv2D(num_channels,
                                  kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  use_bias=use_bias,
                                  data_format=data_format)
        self.batch_norm_2 = tfkl.BatchNormalization(axis=-1,
                                                    momentum=0.95,
                                                    epsilon=0.001)

        self.conv_3 = tfkl.Conv2D(num_channels,
                                  kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  use_bias=use_bias,
                                  data_format=data_format)
        self.batch_norm_3 = tfkl.BatchNormalization(axis=-1,
                                                    momentum=0.95,
                                                    epsilon=0.001)

    def call(self, input_x, input_g, training=False):

        x_g = self.conv_1(input_g)
        if training:
            x_g = self.batch_norm_1(x_g)

        x_l = self.conv_2(input_x)
        if training:
            x_l = self.batch_norm_2(x_l)

        x = tfkl.concatenate([x_g, x_l], axis=3)
        x = tfkl.Activation('relu')(x)

        x = self.conv_3(x)
        if training:
            x = self.batch_norm_3(x)
        alpha = tfkl.Activation('sigmoid')(x)

        outputs = tf.math.multiply(alpha, input_x)

        return outputs


class MultiResBlock(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 kernel_size=(3, 3),
                 nonlinearity='relu',
                 padding='same',
                 strides=(1, 1),
                 data_format='channels_last',
                 **kwargs):

        super(MultiResBlock, self).__init__(**kwargs)

        self.conv_1 = Conv2D_Block(num_channels,
                                   1,
                                   kernel_size=(1, 1),
                                   nonlinearity=None,
                                   use_batchnorm=False,
                                   use_dropout=False,
                                   use_spatial_dropout=False,
                                   data_format='channels_last')
        self.conv_2 = Conv2D_Block(num_channels,
                                   1,
                                   kernel_size,
                                   nonlinearity,
                                   use_batchnorm=False,
                                   use_dropout=False,
                                   use_spatial_dropout=False,
                                   data_format='channels_last')
        self.conv_3 = Conv2D_Block(num_channels,
                                   1,
                                   kernel_size,
                                   nonlinearity,
                                   use_batchnorm=False,
                                   use_dropout=False,
                                   use_spatial_dropout=False,
                                   data_format='channels_last')
        self.conv_4 = Conv2D_Block(num_channels,
                                   1,
                                   kernel_size,
                                   nonlinearity,
                                   use_batchnorm=False,
                                   use_dropout=False,
                                   use_spatial_dropout=False,
                                   data_format='channels_last')

        self.batch_1 = tfkl.BatchNormalization(axis=3)
        self.activation_1 = tfkl.Activation(nonlinearity)
        self.batch_2 = tfkl.BatchNormalization(axis=3)

    def call(self, x, training=False):

        x1 = self.conv_1(x)

        x2 = self.conv_2(x)
        x3 = self.conv_3(x1)
        x4 = self.conv_4(x3)

        out = tfkl.concatenate([x2, x3, x4])
        if training:
            out = self.batch_1(out)

        out = tfkl.concatenate([x1, out])
        out = self.activation_1(out)

        if training:
            out = self.batch_2(out)

        return out


class ResPath(tf.keras.Model):
    # TODO(Joonsu): Reimplement ResPath
    def __init__(self,
                 length,
                 num_channels,
                 kernel_size=(1, 1),
                 nonlinearity='relu',
                 padding='same',
                 strides=(1, 1),
                 data_format='channels_last',
                 **kwargs):

        super(ResPath, self).__init__(**kwargs)

        self.length = length

        self.conv = []
        self.batch_norm = []

        for _ in range(length):
            self.conv.append(Conv2D_Block(num_channels,
                                          1,
                                          kernel_size,
                                          nonlinearity=None,
                                          use_batchnorm=False,
                                          use_dropout=False,
                                          use_spatial_dropout=False,
                                          data_format='channels_last'))
            self.conv.append(Conv2D_Block(num_channels,
                                          1,
                                          kernel_size=(3, 3),
                                          nonlinearity='relu',
                                          use_batchnorm=False,
                                          use_dropout=False,
                                          use_spatial_dropout=False,
                                          data_format='channels_last'))

        for _ in range(length):
            self.batch_norm.append(tfkl.BatchNormalization(axis=3))

    def call(self, input, training=False):

        for i in range(0, self.length, 2):

            x1 = self.conv[i](input)
            x2 = self.conv[i + 1](input)

            out = tfkl.add([x1, x2])
            out = tfkl.Activation('relu')(out)
            if training:
                out = self.batch_norm[i](out)

        return out
