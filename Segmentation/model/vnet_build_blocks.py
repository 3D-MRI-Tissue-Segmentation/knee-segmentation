import tensorflow as tf
import tensorflow.keras.layers as tfkl

class Conv3D_Block(tf.keras.Sequential):

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
                 **kwargs):

        super(Conv3D_Block, self).__init__(**kwargs)

        self.num_conv_layers = num_conv_layers
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.use_spatial_dropout = use_spatial_dropout
                
        for _ in range(self.num_conv_layers):
            self.add(tfkl.Conv3D(num_channels,
                                 kernel_size,
                                 padding='same',
                                 data_format=data_format))
            if self.use_batchnorm:
                self.add(tfkl.BatchNormalization(axis=-1,
                                                 momentum=0.95,
                                                 epsilon=0.001))
            self.add(tfkl.Activation(nonlinearity))
        
        if self.use_dropout:
            if self.use_spatial_dropout:
                self.add(tfkl.SpatialDropout3D(rate=dropout_rate))
            else:
                self.add(tfkl.Dropout(rate=dropout_rate))

    def call(self, inputs, training=False):
        
        outputs = super(Conv3D_Block, self).call(inputs, training=training)

        return outputs

class Up_Conv3D(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 kernel_size=(2, 2, 2),
                 nonlinearity='relu',
                 use_batchnorm=False,
                 use_transpose=False,
                 strides=(2, 2, 2),
                 upsample_size=(2, 2, 2),
                 data_format='channels_last',
                 **kwargs):

        super(Up_Conv3D, self).__init__(**kwargs)

        self.use_batchnorm = use_batchnorm
        self.upsample = tf.keras.layers.UpSampling3D(size=upsample_size)
        self.conv = tf.keras.layers.Conv3D(num_channels, kernel_size,
                                           padding='same', data_format=data_format)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.activation = tf.keras.layers.Activation(nonlinearity)
        self.use_transpose = use_transpose
        self.conv_transpose = tf.keras.layers.Conv3DTranspose(num_channels, kernel_size, padding='same',
                                                              strides=strides, data_format=data_format)

    def call(self, inputs, training=False):

        x = inputs
        if self.use_transpose:
            x = self.conv_transpose(x)
        else:
            x = self.upsample(x)
            x = self.conv(x)
        if self.use_batchnorm:
            if training:
                x = self.batch_norm(x)
        outputs = self.activation(x)
        return outputs
