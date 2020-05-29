import tensorflow as tf
import tensorflow.keras.layers as tfkl

class Deeplabv3(tf.keras.Model):
    """ Tensorflow 2 Implementation of """

    def __init__(self,
                 num,
                 **kwargs):
        
        super(Deeplabv3, self).__init__(**kwargs)


class astrous_spatial_pyramid_pooling():

    def __init__(self,
                 num_channels=256,
                 kernel_size=(1,3,3,3),
                 rate=(1,6,12,18),
                 padding='same',
                 use_batchnorm=True,
                 use_nonlinearity=False,
                 nonlinearity='relu',
                 use_bias=True,
                 data_format='channels_last',
                 **kwargs):
        
        self.block_list = []

        self.first_conv = tfkl.Conv2D(num_channels,
                                      kernel_size=1,
                                      padding=padding)

        for i in range(len(kernel_size)):
            self.block_list.append(aspp_block(num_channels,
                                             kernel_size[i],
                                             rate,
                                             padding,
                                             use_batchnorm,
                                             use_nonlinearity,
                                             nonlinearity,
                                             use_bias,
                                             data_format))
            
    def call(self, x, training=False):

        feature_map_size = tf.shape(x)
        output_list = []

        #Non diluted convolution
        y = tf.math.reduce_mean(x, axis=[1,2], keepdims=True)
        y = self.first_conv(y, training=training)
        output_list.append(tf.image.resize(y, (feature_map_size[1], feature_map_size[2])))

        #Series of diluted convolutions with rates (1, 6, 12, 18)
        for i, block in enumerate(self.block_list):
            output_list.append(block(x, training=training))

        #concatenate all outputs


class aspp_block(tf.keras.Sequential):

    def __init__(self,
                 num_channels=256,
                 kernel_size,
                 rate,
                 padding='same',
                 use_batchnorm=True,
                 use_nonlinearity=False,
                 nonlinearity='relu',
                 use_bias=True,
                 data_format='channels_last',
                 **kwargs):

        super(aspp_block, self).__init__(**kwargs)

        self.add(tfkl.Conv2D(num_channels,
                             kernel_size,
                             padding=padding,
                             use_bias=use_bias,
                             data_format=data_format,
                             dilation_rate=rate))
        
        if use_batchnorm:
            self.add(tfkl.BatchNormalization(axis=-1,
                                             momentum=0.95,
                                             epsilon=0.001))
        if use_nonlinearity:
            self.add(tfkl.Activation(nonlinearity))

    def call(self, x, training=False):

        output = super(aspp_block, self).call(x, training=training)
        return output

class basic_conv_block(tf.keras.Sequential):

    """ This could have been done using the aspp block, however having these
    separated makes the understanding of the code easier """

    def __init__(self,
                 num_channels,
                 num_conv_layers=2,
                 kernel_size=(3, 3),
                 padding='same',
                 nonlinearity='relu',
                 use_batchnorm=True,
                 use_nonlinearity=True,
                 use_bias=True,
                 data_format='channels_last',
                 **kwargs):

        super(basic_conv_block, self).__init__(**kwargs)

        self.add(tfkl.Conv2D(num_channels,
                             kernel_size,
                             padding=padding,
                             use_bias=use_bias,
                             data_format=data_format))
        
        if use_batchnorm:
            self.add(tfkl.BatchNormalization(axis=-1,
                                             momentum=0.95,
                                             epsilon=0.001))
        if use_nonlinearity:
            self.add(tfkl.Activation(nonlinearity))
    
    def call(self, x, training=False):

        output = super(basic_conv_block, self).call(x, training=training)
        return output
