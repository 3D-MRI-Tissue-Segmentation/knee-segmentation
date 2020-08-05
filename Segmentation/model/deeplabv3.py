import tensorflow as tf
import tensorflow.keras.layers as tfkl

class Deeplabv3_plus(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 kernel_size_initial_conv,
                 num_channels_DCNN,
                 num_channels_atrous=512,
                 num_channels_ASPP=256,
                 kernel_size_atrous=3,
                 kernel_size_DCNN=[1, 3],
                 kernel_size_ASPP=[1, 3, 3, 3],
                 num_filters_final_encoder=512,
                 num_channels_from_backcone=[128, 96],
                 num_channels_UpConv=[512, 256, 128],
                 kernel_size_UpConv=3,
                 stride_UpConv=(2, 2),
                 use_batchnorm_UpConv=False,
                 use_transpose_UpConv=False,
                 padding='same',
                 activation='relu',
                 use_batchnorm=True,
                 use_bias=True,
                 data_format='channels_last',
                 MultiGrid=[1, 2, 4],
                 rate_ASPP=[1, 6, 12, 18],
                 atrous_output_stride=16,
                 # Not adapted code for any other out stride
                 **kwargs):
        
        """ Arguments:
            kernel_size_initial_conv: the size of the kernel for the
                                      first convolution
            num_channels_DCNN: touple with the number of channels for the
                               first three blocks of the DCNN
            kernel_size_DCNN: two element touple with the kernel size of the
                              first and last convolution of the resnet_block
                              (First element) and the middle convolution
                              of the resnet_block (Second element)  """

        super(Deeplabv3_plus, self).__init__(**kwargs)

        self.num_classes = num_classes

        #ResNet backbone
        self.first_conv = tfkl.Conv2D(num_channels_DCNN[0],
                                      kernel_size_initial_conv,
                                      strides=2,
                                      padding=padding,
                                      use_bias=use_bias,
                                      data_format=data_format)

        self.block1 = resnet_block(False,
                                   num_channels_DCNN[0],
                                   kernel_size_DCNN,
                                   padding,
                                   activation,
                                   use_batchnorm,
                                   use_bias,
                                   data_format)

        self.block2 = resnet_block(True,
                                   num_channels_DCNN[1],
                                   kernel_size_DCNN,
                                   padding,
                                   activation,
                                   use_batchnorm,
                                   use_bias,
                                   data_format)

        self.block3 = resnet_block(True,
                                   num_channels_DCNN[2],
                                   kernel_size_DCNN,
                                   padding,
                                   activation,
                                   use_batchnorm,
                                   use_bias,
                                   data_format)

        #Atrous components
        self.atrous_conv = Atrous_conv(num_channels_atrous,
                                       kernel_size_atrous,
                                       MultiGrid,
                                       padding,
                                       use_batchnorm,
                                       'linear',
                                       use_bias,
                                       data_format,
                                       atrous_output_stride)

        self.aspp_term = atrous_spatial_pyramid_pooling(num_channels_ASPP,
                                                        kernel_size_ASPP,
                                                        rate_ASPP,
                                                        padding,
                                                        use_batchnorm,
                                                        'linear',
                                                        use_bias,
                                                        data_format)

        #Final convolution of encoder
        self.final_encoder_conv = aspp_block(1,
                                             1,
                                             num_filters_final_encoder,
                                             padding,
                                             use_batchnorm,
                                             'linear',
                                             use_bias,
                                             data_format)

        #Decoder
        self.decoder_term = Decoder(num_channels_from_backcone=num_channels_from_backcone,
                                    num_channels_UpConv=num_channels_UpConv,
                                    kernel_size_UpConv=kernel_size_UpConv,
                                    stride_UpConv=stride_UpConv,
                                    use_batchnorm_UpConv=use_batchnorm_UpConv,
                                    use_transpose_UpConv=use_transpose_UpConv,
                                    use_bias=use_bias,
                                    padding=padding,
                                    data_format=data_format)

        self.output_conv = tfkl.Conv2D(num_classes,
                                       1,
                                       activation='linear',
                                       padding='same',
                                       data_format=data_format)

    def call(self, x, training=False):
        
        ###Encoder
        before_final_stride = self.first_conv(x, training=training)  # output stride 2

        before_final_stride = self.block1(before_final_stride, training=training)  # output stride 2
        before_final_stride = self.block2(before_final_stride, training=training)  # output stride 4
        atrous_out = self.block3(before_final_stride, training=training)  # output stride 8

        atrous_out = self.atrous_conv(atrous_out, training=training)
        out = self.aspp_term(atrous_out, training=training)
        out = self.final_encoder_conv(out, training=training)

        ###Decoder
        out = self.decoder_term(atrous_out ,out, before_final_stride, training=training)

        out = self.output_conv(out, training=training)
        if self.num_classes == 1:
            out = tfkl.Activation('sigmoid')(out)
        else:
            out = tfkl.Activation('softmax')(out)
        
        # Upsample to same size as the input
        # print(f"Input Shape: {x.shape}, Out Shape: {decoder_out.shape}")
        # input_size = tf.shape(x)[1:3]
        # decoder_out = tf.image.resize(decoder_out, input_size)

        return out


class Deeplabv3(tf.keras.Sequential):
    """ Tensorflow 2 Implementation of """
    def __init__(self,
                 num_classes,
                 kernel_size_initial_conv,
                 num_channels_atrous,
                 num_channels_DCNN=[256, 512, 1024],
                 num_channels_ASPP=256,
                 kernel_size_atrous=3,
                 kernel_size_DCNN=[1, 3],
                 kernel_size_ASPP=[1, 3, 3, 3],
                 padding='same',
                 activation='relu',
                 use_batchnorm=True,
                 use_bias=True,
                 data_format='channels_last',
                 MultiGrid=[1, 2, 4],
                 rate_ASPP=[1, 6, 12, 18],
                 atrous_output_stride=16,
                 # Not adapted code for any other out stride
                 **kwargs):
        
        """ Arguments:
            kernel_size_initial_conv: the size of the kernel for the
                                      first convolution
            num_channels_DCNN: touple with the number of channels for the
                               first three blocks of the DCNN
            kernel_size_DCNN: two element touple with the kernel size of the
                              first and last convolution of the resnet_block
                              (First element) and the middle convolution
                              of the resnet_block (Second element)  """

        super(Deeplabv3, self).__init__(**kwargs)

        self.num_classes = num_classes

        self.add(ResNet_Backbone(kernel_size_initial_conv,
                                 num_channels_DCNN,
                                 kernel_size_DCNN,
                                 padding,
                                 activation,
                                 use_batchnorm,
                                 use_bias,
                                 False,
                                 data_format))

        self.add(Atrous_conv(num_channels_atrous,
                             kernel_size_atrous,
                             MultiGrid,
                             padding,
                             use_batchnorm,
                             'linear',
                             use_bias,
                             data_format,
                             atrous_output_stride))

        self.add(atrous_spatial_pyramid_pooling(num_channels_ASPP,
                                                kernel_size_ASPP,
                                                rate_ASPP,
                                                padding,
                                                use_batchnorm,
                                                'linear',
                                                use_bias,
                                                data_format))

        self.add(aspp_block(1,
                            1,
                            num_classes,
                            padding,
                            use_batchnorm,
                            'linear',
                            use_bias,
                            data_format))

    def call(self, x, training=False):

        out = super(Deeplabv3, self).call(x, training=training)
        if self.num_classes == 1:
            out = tfkl.Activation('sigmoid')(out)
        else:
            out = tfkl.Activation('softmax')(out)
        
        # Upsample to same size as the input
        input_size = tf.shape(x)[1:3]
        out = tf.image.resize(out, input_size)

        return out

class ResNet_Backbone(tf.keras.Model):
    def __init__(self,
                 kernel_size_initial_conv,
                 num_channels=[256, 512, 1024],
                 kernel_size_blocks=[1, 3],
                 padding='same',
                 activation='relu',
                 use_batchnorm=True,
                 use_bias=True,
                 use_pooling=False,
                 data_format='channels_last',
                 **kwargs):
        
        super(ResNet_Backbone, self).__init__(**kwargs)
        self.first_conv = tfkl.Conv2D(num_channels[0],
                                      kernel_size_initial_conv,
                                      strides=2,
                                      padding=padding,
                                      use_bias=use_bias,
                                      data_format=data_format)

        self.max_pool = tfkl.MaxPool2D(pool_size=(2, 2),
                                       padding='valid')
        
        self.use_pooling = use_pooling

        self.block1 = resnet_block(False,
                                   num_channels[0],
                                   kernel_size_blocks,
                                   padding,
                                   activation,
                                   use_batchnorm,
                                   use_bias,
                                   data_format)
        self.block2 = resnet_block(True,
                                   num_channels[1],
                                   kernel_size_blocks,
                                   padding,
                                   activation,
                                   use_batchnorm,
                                   use_bias,
                                   data_format)

        self.block3 = resnet_block(True,
                                   num_channels[2],
                                   kernel_size_blocks,
                                   padding,
                                   activation,
                                   use_batchnorm,
                                   use_bias,
                                   data_format)

    def call(self, x, training=False):

        x = self.first_conv(x, training=training)  # output stride 2
        if self.use_pooling:
            x = self.max_pool(x)  # output stride 4

        x = self.block1(x, training=training)  # output stride 2 or 4
        x = self.block2(x, training=training)  # output stride 4 or 8
        x = self.block3(x, training=training)  # output stride 8 or 16
        return x


# full pre-activation residual unit
class resnet_block(tf.keras.Model):

    def __init__(self,
                 use_stride,
                 num_channels,
                 kernel_size=[1, 3],
                 padding='same',
                 activation='relu',
                 use_batchnorm=True,
                 use_bias=True,
                 data_format='channels_last',
                 **kwargs):
        
        super(resnet_block, self).__init__(**kwargs)
        self.use_stride = use_stride
        inner_num_channels = num_channels // 4

        if use_stride:
            self.input_conv = basic_conv_block(num_channels,
                                               1,
                                               2,
                                               padding,
                                               activation,
                                               use_batchnorm,
                                               use_bias,
                                               data_format)
            stride = 2

        else:
            stride = 1

        self.first_conv = basic_conv_block(inner_num_channels,
                                           kernel_size[0],
                                           stride,
                                           padding,
                                           activation,
                                           use_batchnorm,
                                           use_bias,
                                           data_format)

        self.second_conv = basic_conv_block(inner_num_channels,
                                            kernel_size[1],
                                            1,
                                            padding,
                                            activation,
                                            use_batchnorm,
                                            use_bias,
                                            data_format)
                             
        self.third_conv = basic_conv_block(num_channels,
                                           kernel_size[0],
                                           1,
                                           padding,
                                           activation,
                                           use_batchnorm,
                                           use_bias,
                                           data_format)

    def call(self, x, training=False):
        
        residual = self.first_conv(x, training=training)

        if self.use_stride:
            x = self.input_conv(x, training=training)

        residual = self.second_conv(residual, training=training)
        residual = self.third_conv(residual, training=training)

        output = tfkl.Add()([residual, x])
        return output


class basic_conv_block(tf.keras.Sequential):

    def __init__(self,
                 num_channels,
                 kernel_size,
                 stride=1,
                 padding='same',
                 activation='relu',
                 use_batchnorm=True,
                 use_bias=True,
                 data_format='channels_last',
                 rate=1,
                 **kwargs):

        super(basic_conv_block, self).__init__(**kwargs)

        if use_batchnorm:
            self.add(tfkl.BatchNormalization(axis=-1,
                                             momentum=0.95,
                                             epsilon=0.001))
        self.add(tfkl.Activation(activation))

        self.add(tfkl.Conv2D(num_channels,
                             kernel_size,
                             strides=stride,
                             padding=padding,
                             use_bias=use_bias,
                             data_format=data_format,
                             dilation_rate=rate))
    
    def call(self, x, training=False):

        output = super(basic_conv_block, self).call(x, training=training)
        return output

# ####################### Atrous Convolution ####################### #
class Atrous_conv(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 kernel_size=3,
                 MultiGrid=[1, 2, 4],
                 padding='same',
                 use_batchnorm=True,
                 activation='linear',
                 use_bias=True,
                 data_format='channels_last',
                 output_stride=16,
                 **kwargs):

        super(Atrous_conv, self).__init__(**kwargs)

        if output_stride == 16:
            multiplier = 2
        else:
            multiplier = 1
        
        self.first_conv = basic_conv_block(num_channels,
                                           kernel_size,
                                           1,
                                           padding,
                                           activation,
                                           use_batchnorm,
                                           use_bias,
                                           data_format,
                                           rate=int(multiplier * MultiGrid[0]))

        self.second_conv = basic_conv_block(num_channels,
                                            kernel_size,
                                            1,
                                            padding,
                                            activation,
                                            use_batchnorm,
                                            use_bias,
                                            data_format,
                                            rate=int(multiplier * MultiGrid[1]))

        self.third_conv = basic_conv_block(num_channels,
                                           kernel_size,
                                           1,
                                           padding,
                                           activation,
                                           use_batchnorm,
                                           use_bias,
                                           data_format,
                                           rate=int(multiplier * MultiGrid[2]))

    def call(self, x, training=False):

        x = self.first_conv(x, training)
        x = self.second_conv(x, training)
        x = self.third_conv(x, training)
        return x


# ####################### ASPP ####################### #
class atrous_spatial_pyramid_pooling(tf.keras.Model):

    def __init__(self,
                 num_channels=256,
                 kernel_size=[1, 3, 3, 3],
                 rate=[1, 6, 12, 18],
                 padding='same',
                 use_batchnorm=True,
                 activation='linear',
                 use_bias=True,
                 data_format='channels_last',
                 **kwargs):
        
        super(atrous_spatial_pyramid_pooling, self).__init__(**kwargs)
        self.block_list = []

        self.basic_conv1 = tfkl.Conv2D(num_channels,
                                       kernel_size=1,
                                       padding=padding)

        self.basic_conv2 = tfkl.Conv2D(num_channels,
                                       kernel_size=1,
                                       padding=padding)

        for i in range(len(kernel_size)):
            self.block_list.append(aspp_block(kernel_size[i],
                                              rate[i],
                                              num_channels,
                                              padding,
                                              use_batchnorm,
                                              activation,
                                              use_bias,
                                              data_format))
            
    def call(self, x, training=False):

        feature_map_size = tf.shape(x)
        output_list = []

        # Non diluted convolution
        y = tf.math.reduce_mean(x, axis=[1, 2], keepdims=True)  # ~ Average Pooling
        y = self.basic_conv1(y, training=training)
        output_list.append(tf.image.resize(y, (feature_map_size[1], feature_map_size[2])))  # ~ Upsampling

        # Series of diluted convolutions with rates (1, 6, 12, 18)
        for i, block in enumerate(self.block_list):
            output_list.append(block(x, training=training))

        # concatenate all outputs
        out = tf.concat(output_list, axis=3)
        out = self.basic_conv2(out, training=training)
        return out


class aspp_block(tf.keras.Sequential):

    def __init__(self,
                 kernel_size,
                 rate,
                 num_channels=256,
                 padding='same',
                 use_batchnorm=True,
                 activation='linear',
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

        self.add(tfkl.Activation(activation))

    def call(self, x, training=False):

        output = super(aspp_block, self).call(x, training=training)
        return output

# ####################### Decoder ####################### #
class Decoder(tf.keras.Model):

    def __init__(self,
                 num_channels_from_backcone=[48],
                 num_channels_UpConv=[512, 256, 128],
                 kernel_size_UpConv=3,
                 stride_UpConv=(2, 2),
                 use_batchnorm_UpConv=False,
                 use_transpose_UpConv=False,
                 use_bias=True,
                 padding='same',
                 data_format='channels_last',
                 **kwargs):

        super(Decoder, self).__init__(**kwargs)

        self.first_conv1x1 = tfkl.Conv2D(num_channels_from_backcone[0],
                                         kernel_size=1,
                                         padding=padding,
                                         data_format=data_format)

        self.second_conv1x1 = tfkl.Conv2D(num_channels_from_backcone[1],
                                          kernel_size=1,
                                          padding=padding,
                                          data_format=data_format)

        self.conv1 = Up_Conv2D(num_channels_conv=num_channels_UpConv[0],
                               kernel_size=kernel_size_UpConv,
                               use_batchnorm=use_batchnorm_UpConv,
                               use_transpose=use_transpose_UpConv,
                               strides=stride_UpConv)

        self.conv2 = Up_Conv2D(num_channels_conv=num_channels_UpConv[1],
                               kernel_size=kernel_size_UpConv,
                               use_batchnorm=use_batchnorm_UpConv,
                               use_transpose=use_transpose_UpConv,
                               strides=stride_UpConv)

        self.conv3 = Up_Conv2D(num_channels_conv=num_channels_UpConv[2],
                               kernel_size=kernel_size_UpConv,
                               use_batchnorm=use_batchnorm_UpConv,
                               use_transpose=use_transpose_UpConv,
                               strides=stride_UpConv)

    def call(self, in_atrous, in_encoder, in_DCNN, training=False):

        in_atrous = self.first_conv1x1(in_atrous, training=training)
        in_DCNN = self.second_conv1x1(in_DCNN, training=training)

        out = tf.concat([in_atrous, in_encoder], axis=3)
        out = self.conv1(out, training=training)

        out = tf.concat([in_DCNN, out], axis=3)
        out = self.conv2(out, training=training)
        out = self.conv3(out, training=training)

        return out


class Up_Conv2D(tf.keras.Sequential):

    def __init__(self,
                 num_channels_conv,
                 num_channels_UpConv=256,
                 kernel_size=3,
                 activation='relu',
                 use_batchnorm=False,
                 use_transpose=False,
                 use_bias=True,
                 strides=(2, 2),
                 padding='same',
                 data_format='channels_last',
                 **kwargs):

        super(Up_Conv2D, self).__init__(**kwargs)

        if use_transpose:
            self.add(tfkl.Conv2DTranspose(num_channels_UpConv,
                                          kernel_size,
                                          padding='same',
                                          strides=strides,
                                          data_format=data_format))
        else:
            self.add(tfkl.UpSampling2D(size=strides))

        self.add(aspp_block(kernel_size=kernel_size,
                            rate=1,
                            num_channels=num_channels_conv,
                            padding=padding,
                            use_batchnorm=use_batchnorm,
                            activation=activation,
                            use_bias=use_bias,
                            data_format=data_format))

    def call(self, x, training=False):

        out = super(Up_Conv2D, self).call(x, training=training)
        return out


