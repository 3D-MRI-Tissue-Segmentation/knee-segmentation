import tensorflow as tf
import tensorflow.keras.layers as tfkl

class Deeplabv3(tf.keras.Model):
    """ Tensorflow 2 Implementation of """

    def __init__(self,
                 num,
                 **kwargs):
        
        super(Deeplabv3, self).__init__(**kwargs)

######################## DCNN ########################
# DCNN portion using a ResNet
class ResNet_Backbone():
    
    def __init__(self,
                 kernel_size_initial_conv,
                 num_channels=(64,256,512,1024),
                 kernel_size_blocks=(1,3),
                 padding='same',
                 nonlinearity='relu',
                 use_batchnorm=True,
                 use_nonlinearity=True,
                 use_bias=True,
                 data_format='channels_last',
                 **kwargs):

        self.first_conv = tfkl.Conv2D(num_channels[0],
                                      kernel_size_initial_conv,
                                      strides=stride,
                                      padding=padding,
                                      use_bias=use_bias,
                                      data_format=data_format))

        self.block1 = resnet_block(False,
                                   num_channels[1],
                                   kernel_size_blocks,
                                   padding,
                                   nonlinearity,
                                   use_batchnorm,
                                   use_nonlinearity,
                                   use_bias,
                                   data_format)

        self.block2 = resnet_block(True,
                                   num_channels[2],
                                   kernel_size_blocks,
                                   padding,
                                   nonlinearity,
                                   use_batchnorm,
                                   use_nonlinearity,
                                   use_bias,
                                   data_format)

        self.block3 = resnet_block(True,
                                   num_channels[3],
                                   kernel_size_blocks,
                                   padding,
                                   nonlinearity,
                                   use_batchnorm,
                                   use_nonlinearity,
                                   use_bias,
                                   data_format)

    def call(self, x, training=False):

        x = self.first_conv(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        return x


# full pre-activation residual unit
class resnet_block():

    def __init__(self,
                 use_stride,
                 num_channels,
                 kernel_size=(1,3),
                 padding='same',
                 nonlinearity='relu',
                 use_batchnorm=True,
                 use_nonlinearity=True,
                 use_bias=True,
                 data_format='channels_last',
                 **kwargs):

        self.use_stride = use_stride
        inner_num_channels = num_channels // 4

        if use_stride:
            self.input_conv = basic_conv_block(num_channels,
                                               kernel_size=1,
                                               stride=2,
                                               padding,
                                               nonlinearity,
                                               use_batchnorm,
                                               use_nonlinearity,
                                               use_bias,
                                               data_format)
            stride = 2
        
        else:
            stride = 1

        self.first_conv = basic_conv_block(inner_num_channels,
                                           kernel_size[0],
                                           stride=stride,
                                           padding,
                                           nonlinearity,
                                           use_batchnorm,
                                           use_nonlinearity,
                                           use_bias,
                                           data_format)

        self.second_conv = basic_conv_block(inner_num_channels,
                                            kernel_size[1],
                                            padding,
                                            nonlinearity,
                                            use_batchnorm,
                                            use_nonlinearity,
                                            use_bias,
                                            data_format)
                                    
        self.third_conv = basic_conv_block(num_channels,
                                           kernel_size[0],
                                           padding,
                                           nonlinearity,
                                           use_batchnorm,
                                           use_nonlinearity,
                                           use_bias,
                                           data_format)

    def call(self, x, training=False):

        if self.use_stride:
            x = self.input_conv(x, training=training)

        residual = self.first_conv(x, training=training)
        residual = self.second_conv(residual, training=training)
        residual = self.third_conv(residual, training=training)

        output = tfkl.Add([residual, x])
        return output


class basic_conv_block(tf.keras.Sequential):

    """ This could have been done using the aspp block, however having these
    separated makes the understanding of the code easier """

    def __init__(self,
                 num_channels,
                 kernel_size,
                 stride=1,
                 padding='same',
                 nonlinearity='relu',
                 use_batchnorm=True,
                 use_nonlinearity=True,
                 use_bias=True,
                 data_format='channels_last',
                 **kwargs):

        super(basic_conv_block, self).__init__(**kwargs)

        if use_batchnorm:
            self.add(tfkl.BatchNormalization(axis=-1,
                                             momentum=0.95,
                                             epsilon=0.001))
        if use_nonlinearity:
            self.add(tfkl.Activation(nonlinearity))

        self.add(tfkl.Conv2D(num_channels,
                             kernel_size,
                             strides=stride,
                             padding=padding,
                             use_bias=use_bias,
                             data_format=data_format))
    
    def call(self, x, training=False):

        output = super(basic_conv_block, self).call(x, training=training)
        return output

######################## Astrous Convolution ########################

######################## ASPP ########################
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

        self.basic_conv = tfkl.Conv2D(num_channels,
                                      kernel_size=1,
                                      padding=padding)

        for i in range(len(kernel_size)):
            self.block_list.append(aspp_block(kernel_size[i],
                                              rate[i],
                                              num_channels,
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
        y = tf.math.reduce_mean(x, axis=[1,2], keepdims=True) # ~ Average Pooling
        y = self.basic_conv(y, training=training)
        output_list.append(tf.image.resize(y, (feature_map_size[1], feature_map_size[2]))) # ~ Upsampling

        #Series of diluted convolutions with rates (1, 6, 12, 18)
        for i, block in enumerate(self.block_list):
            output_list.append(block(x, training=training))

        #concatenate all outputs
        out = tf.concat(output_list, axis=3)
        out = self.basic_conv(out, training=training)
        return out


class aspp_block(tf.keras.Sequential):

    def __init__(self,
                 kernel_size,
                 rate,
                 num_channels=256,
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
