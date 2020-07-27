import tensorflow as tf
import tensorflow.keras.layers as tfkl

class SegNet (tf.keras.Model):
    """ Tensorflow 2 Implementation of 'SegNet: A Deep Convolutional Encoder-Decoder
    Architecture for Image Segmentation' https://arxiv.org/abs/1611.09326 """

    def __init__(self,
                 num_channels,
                 num_classes,
                 backbone='default',
                 kernel_size=(3, 3),
                 pool_size=(2, 2),
                 activation='relu',
                 use_batchnorm=True,
                 use_bias=True,
                 use_transpose=False,
                 use_dropout=False,
                 dropout_rate=0.25,
                 use_spatial_dropout=True,
                 data_format='channels_last',
                 **kwargs):

        super(SegNet, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.backbone = backbone
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        self.use_bias = use_bias
        self.use_transpose = use_transpose
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_spatial_dropout = use_spatial_dropout
        self.data_format = data_format

        self.conv_list = tf.keras.Sequential()
        for i in range(len(self.num_channels)):
            output_ch = self.num_channels[i]
            if i == 0 or i == 1:
                num_conv = 2
            else:
                num_conv = 3

            self.conv_list.add(SegNet_Conv2D_Block(output_ch,
                                                   num_conv,
                                                   self.kernel_size,
                                                   self.pool_size,
                                                   self.activation,
                                                   self.use_batchnorm,
                                                   self.use_bias,
                                                   self.use_dropout,
                                                   self.dropout_rate,
                                                   self.use_spatial_dropout,
                                                   self.data_format))

        self.up_conv_list = tf.keras.Sequential()
        n = len(self.num_channels) - 1

        for j in range(n, -1, -1):
            output = self.num_channels[j]
            if j in [n, n - 1, n - 2]:
                num_conv = 3
            else:
                num_conv = 2
            self.up_conv_list.add(segnet_Up_Conv2D_block(output,
                                                         num_conv_layers=num_conv,
                                                         kernel_size=(2, 2),
                                                         upsampling_size=(2, 2),
                                                         activation=self.activation,
                                                         use_batchnorm=self.use_batchnorm,
                                                         use_transpose=self.use_transpose,
                                                         use_bias=self.use_bias,
                                                         strides=(2, 2),
                                                         data_format=self.data_format))

        self.conv_1x1 = tfkl.Conv2D(num_classes,
                                    (1, 1),
                                    activation='linear',
                                    padding='same',
                                    data_format=data_format)

    def call(self, x, training=False):

        encoded = self.conv_list(x, training=training)
        decoded = self.up_conv_list(encoded, training=training)
        output = self.conv_1x1(decoded)

        if self.num_classes == 1:
            output = tfkl.Activation('sigmoid')(output)
        else:
            output = tfkl.Activation('softmax')(output)
        return output


class SegNet_Conv2D_Block(tf.keras.Sequential):

    def __init__(self,
                 num_channels,
                 num_conv_layers=2,
                 kernel_size=(3, 3),
                 pool_size=(2, 2),
                 activation='relu',
                 use_batchnorm=True,
                 use_bias=True,
                 use_dropout=False,
                 dropout_rate=0.25,
                 use_spatial_dropout=True,
                 data_format='channels_last',
                 **kwargs):

        super(SegNet_Conv2D_Block, self).__init__(**kwargs)

        for _ in range(num_conv_layers):
            self.add(tfkl.Conv2D(num_channels,
                                 kernel_size,
                                 padding='same',
                                 use_bias=use_bias,
                                 data_format=data_format))
            if use_batchnorm:
                self.add(tfkl.BatchNormalization(axis=-1,
                                                 momentum=0.95,
                                                 epsilon=0.001))
            self.add(tfkl.Activation(activation))

        if use_dropout:
            if use_spatial_dropout:
                self.add(tfkl.SpatialDropout2D(rate=dropout_rate))
            else:
                self.add(tfkl.Dropout(rate=dropout_rate))

        self.add(tfkl.MaxPool2D(pool_size))

    def call(self, x, training=False):

        output = super(SegNet_Conv2D_Block, self).call(x, training=training)
        return output


class segnet_Up_Conv2D_block(tf.keras.Sequential):

    def __init__(self,
                 num_channels,
                 num_conv_layers,
                 kernel_size=(3, 3),
                 upsampling_size=(2, 2),
                 activation='relu',
                 use_batchnorm=True,
                 use_transpose=False,
                 use_bias=True,
                 strides=(2, 2),
                 data_format='channels_last',
                 **kwargs):

        super(segnet_Up_Conv2D_block, self).__init__(**kwargs)

        if use_transpose:
            self.add(tfkl.Conv2DTranspose(num_channels,
                                          kernel_size,
                                          padding='same',
                                          strides=strides,
                                          data_format=data_format))
        else:
            self.add(tf.keras.layers.UpSampling2D(size=upsampling_size))

        for _ in range(num_conv_layers):
            self.add(tfkl.Conv2D(num_channels,
                                 kernel_size,
                                 padding='same',
                                 data_format=data_format))
            if use_batchnorm:
                self.add(tfkl.BatchNormalization(axis=-1,
                                                 momentum=0.95,
                                                 epsilon=0.001))
            self.add(tfkl.Activation(activation))

    def call(self, x, training=False):

        output = super(segnet_Up_Conv2D_block, self).call(x, training=training)
        return output
