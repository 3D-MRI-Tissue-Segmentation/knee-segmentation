import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, SpatialDropout2D, concatenate, add
from tensorflow.keras import Model, Input

class Conv2D_Block(tf.keras.Sequential):

    def __init__(self,
                 num_channels,
                 num_conv_layers=2,
                 kernel_size=(3,3),
                 nonlinearity='relu',
                 use_batchnorm = False,
                 use_dropout = False,
                 dropout_rate = 0.25, 
                 use_spatial_dropout = True,
                 data_format='channels_last',
                 **kwargs):

        super(Conv2D_Block, self).__init__(**kwargs)

        self.num_channels = num_channels
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.nonlinearity = nonlinearity
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_spatial_dropout = use_spatial_dropout
        self.data_format = data_format

        for i in range(self.num_conv_layers):
            self.add(tf.keras.layers.Conv2D(self.num_channels, self.kernel_size, padding='same', data_format=self.data_format))
            if self.use_batchnorm:
              self.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001))
            self.add(tf.keras.layers.Activation(self.nonlinearity))

        if self.use_dropout:
          if self.use_spatial_dropout:
            self.add(tf.keras.layers.SpatialDropout2D(rate=self.dropout_rate))
          else:
            self.add(tf.keras.layers.Dropout(rate=self.dropout_rate))

    def call(self, inputs, training=False):

        outputs = super(Conv2D_Block, self).call(inputs, training=training)

        return outputs

class Up_Conv2D(tf.keras.Sequential):

    def __init__(self, 
                 num_channels,
                 kernel_size=(2,2),
                 nonlinearity='relu',
                 use_batchnorm = False,
                 use_transpose = False,
                 strides=(2,2),
                 data_format='channels_last',
                 **kwargs):

        super(Up_Conv2D, self).__init__(**kwargs)

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.nonlinearity = nonlinearity
        self.use_batchnorm = use_batchnorm
        self.use_transpose = use_transpose
        self.strides = strides
        self.data_format = data_format

        if self.use_transpose:
          self.add(tf.keras.layers.Conv2DTranspose(self.num_channels, self.kernel_size, padding='same', strides=self.strides, data_format=self.data_format))
        else:
          self.add(tf.keras.layers.UpSampling2D(size=self.strides))
          self.add(tf.keras.layers.Conv2D(self.num_channels, self.kernel_size, padding='same', data_format=self.data_format))
        if self.use_batchnorm:
          self.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001))
        self.add(tf.keras.layers.Activation(self.nonlinearity))

    def call(self, inputs, training=False):
        
        outputs = super(Up_Conv2D, self).call(inputs, training=training)

        return outputs

class Attention_Gate(tf.keras.Model):

    def __init__(self,
                num_channels,
                kernel_size = (1,1),
                nonlinearity='relu',
                padding='same',
                strides=(1,1),
                use_bias=False,
                data_format='channels_last',
                name='attention_gate'):

        super(Attention_Gate, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides, padding=padding, use_bias=use_bias, data_format=data_format)
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001)
        
        self.conv_2 = tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides, padding=padding, use_bias=use_bias, data_format=data_format)
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001)

        self.conv_3 = tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides, padding=padding, use_bias=use_bias, data_format=data_format)
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001)
        
    def call(self, input_x, input_g, training=False):

        x_g = self.conv_1(input_g)
        if training:
            x_g = self.batch_norm_1(x_g)

        x_l = self.conv_2(input_x)
        if training:
            x_l = self.batch_norm_2(x_l)

        x = tf.keras.layers.concatenate([x_g, x_l], axis=3)
        x = tf.keras.layers.Activation('relu')(x)

        x = self.conv_3(x)
        if training:
            x = self.batch_norm_3(x)
        alpha = tf.keras.layers.Activation('sigmoid')(x)

        outputs = tf.math.multiply(alpha, input_x)

        return outputs


class MultiResBlock(tf.keras.Model):

    def __init__(self, 
                num_channels, 
                kernel_size=(3,3),
                nonlinearity='relu', 
                padding='same',
                strides=(1,1),
                data_format='channels_last',
                **kwargs):

        super(MultiResBlock, self).__init__(**kwargs)

        self.conv_1 = Conv2D_Block(num_channels, 1, kernel_size=(1,1), nonlinearity=None, use_batchnorm=False, use_dropout=False, use_spatial_dropout=False, data_format='channels_last')
        self.conv_2 = Conv2D_Block(num_channels, 1, kernel_size, nonlinearity, use_batchnorm=False, use_dropout=False, use_spatial_dropout=False, data_format='channels_last')
        self.conv_3 = Conv2D_Block(num_channels, 1, kernel_size, nonlinearity, use_batchnorm=False, use_dropout=False, use_spatial_dropout=False, data_format='channels_last')
        self.conv_4 = Conv2D_Block(num_channels, 1, kernel_size, nonlinearity, use_batchnorm=False, use_dropout=False, use_spatial_dropout=False, data_format='channels_last')
        
        self.batch_1 = tf.keras.layers.BatchNormalization(axis=3)
        self.activation_1 = tf.keras.layers.Activation(nonlinearity)
        self.batch_2 = tf.keras.layers.BatchNormalization(axis=3)

    def call (self, x, training=False):

        x1 = self.conv_1(x)

        x2 = self.conv_2(x)
        x3 = self.conv_3(x1)
        x4 = self.conv_4(x3)

        out = tf.keras.layers.concatenate([x2, x3, x4])
        if training:
            out = self.batch_1(out)

        out = tf.keras.layers.concatenate([x1, out])
        out = self.activation_1(out)
        
        if training:
            out = self.batch_2(out)

        return out


class ResPath(tf.keras.layers.Layer):

    def __init__(self,
                length,
                num_channels,
                kernel_size=(1,1),
                nonlinearity='relu',
                padding='same',
                strides=(1,1),
                data_format='channels_last',
                name="ResPath"
                ):

        super(ResPath, self).__init__()

        self.length = length

        self.conv = []
        self.batch_norm = []

        for _ in range(length):
            self.conv.append(Conv2D_Block(num_channels, 1, kernel_size, nonlinearity=None, use_batchnorm=False, use_dropout=False, use_spatial_dropout=False, data_format='channels_last'))
            self.conv.append(Conv2D_Block(num_channels, 1, kernel_size=(3,3), nonlinearity='relu', use_batchnorm=False, use_dropout=False, use_spatial_dropout=False, data_format='channels_last'))

        for _ in range(length):
            self.batch_norm.append(tf.keras.layers.BatchNormalization(axis=3))
        
    def call(self, input, training=False):
        
        for i in range (0, self.length, 2):
            
            x1 = self.conv[i](input)
            x2 = self.conv[i+1](input)

            out = tf.keras.layers.add([x1,x2])
            out = tf.keras.layers.Activation('relu')(out)
            if training:
                out = self.batch_norm[i](out)

        return out

class UNet(tf.keras.Model):
    """ Tensorflow 2 Implementation of 'U-Net: Convolutional Networks for Biomedical Image Segmentation'
    https://arxiv.org/abs/1505.04597."""

    def __init__(self, 
                 num_channels,
                 num_classes,
                 num_conv_layers=2,
                 kernel_size=(3,3),
                 nonlinearity='relu',
                 use_batchnorm=True,
                 dropout_rate = 0.00, 
                 use_spatial_dropout = False,
                 data_format='channels_last'):

        super(UNet, self).__init__()

        self.num_classes = num_classes

        #encoding blocks
        self.conv_1 = Conv2D_Block(num_channels,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.conv_2 = Conv2D_Block(num_channels*2,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.conv_3 = Conv2D_Block(num_channels*4,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.conv_4 = Conv2D_Block(num_channels*8,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,use_dropout=True,dropout_rate=dropout_rate,use_spatial_dropout=use_spatial_dropout,data_format=data_format)
        self.conv_5 = Conv2D_Block(num_channels*16,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,use_dropout=True,dropout_rate=dropout_rate,use_spatial_dropout=use_spatial_dropout,data_format=data_format)
        
        #decoding blocks
        self.up_5 = Up_Conv2D(num_channels*8,(2,2),nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.up_6 = Up_Conv2D(num_channels*4,(2,2),nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.up_7 = Up_Conv2D(num_channels*2,(2,2),nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.up_8 = Up_Conv2D(num_channels,(2,2),nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)

        self.up_conv4 = Conv2D_Block(num_channels*8,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.up_conv3 = Conv2D_Block(num_channels*4,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.up_conv2 = Conv2D_Block(num_channels*2,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.up_conv1 = Conv2D_Block(num_channels,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)

        #convolution num_channels at the output
        self.conv_1x1 = tf.keras.layers.Conv2D(num_classes, (1,1), activation='linear', padding='same', data_format=data_format)

    def call(self, inputs,training=False):
        
        ##encoder blocks
        #1->64
        x1 = self.conv_1(inputs,training=training)
        
        #64->128
        x2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x1)
        x2 = self.conv_2(x2,training=training)
        
        #128->256
        x3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x2)
        x3 = self.conv_3(x3,training=training)

        #256->512
        x4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x3)
        x4 = self.conv_4(x4,training=training)

        #512->1024
        x5 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x4)
        x5 = self.conv_5(x5,training=training)

        #decoder blocks
        #1024->512
        u5 = self.up_5(x5,training=training)
        u5 = tf.keras.layers.concatenate([x4, u5], axis=3)
        u5 = self.up_conv4(u5,training=training)

        #512->256
        u6 = self.up_6(u5,training=training)
        u6 = tf.keras.layers.concatenate([x3, u6], axis=3)
        u6 = self.up_conv3(u6,training=training)

        #256->128
        u7 = self.up_7(u6,training=training)
        u7 = tf.keras.layers.concatenate([x2, u7], axis=3)
        u7 = self.up_conv2(u7,training=training)

        #128->64
        u8 = self.up_8(u7,training=training)
        u8 = tf.keras.layers.concatenate([x1, u8], axis=3)
        u8 = self.up_conv1(u8,training=training)
        u9 = self.conv_1x1(u8)

        if self.num_classes == 1:
            output = tf.keras.layers.Activation('sigmoid')(u9)

        else:
            output = tf.keras.layers.Activation('softmax')(u9)

        return output

class AttentionUNet_v1(tf.keras.Model):
    "Tensorflow 2 Implementation of Attention UNet"

    def __init__(self, 
                 num_channels,
                 num_classes,
                 num_conv_layers=1,
                 kernel_size=(3,3),
                 strides=(1,1),
                 pool_size=(2,2),
                 use_bias=False,
                 padding='same',
                 nonlinearity='relu',
                 use_batchnorm = True,
                 use_transpose = True,
                 data_format='channels_last',
                 name="attention_unet_v1"):

        super(AttentionUNet_v1, self).__init__(name=name)

        self.num_classes = num_classes

        self.conv_1 = Conv2D_Block(num_channels,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)
        self.conv_2 = Conv2D_Block(num_channels*2,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)
        self.conv_3 = Conv2D_Block(num_channels*4,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)
        self.conv_4 = Conv2D_Block(num_channels*8,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)
        self.conv_5 = Conv2D_Block(num_channels*16,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)

        self.up_conv_1 = Up_Conv2D(num_channels*8,(3,3),nonlinearity,use_batchnorm=True,data_format=data_format)
        self.up_conv_2 = Up_Conv2D(num_channels*4,(3,3),nonlinearity,use_batchnorm=True,data_format=data_format)
        self.up_conv_3 = Up_Conv2D(num_channels*2,(3,3),nonlinearity,use_batchnorm=True,data_format=data_format)
        self.up_conv_4 = Up_Conv2D(num_channels,(3,3),nonlinearity,use_batchnorm=True,data_format=data_format)

        self.a1 = Attention_Gate(num_channels*8,(1,1),nonlinearity,padding,strides,use_bias,data_format)
        self.a2 = Attention_Gate(num_channels*4,(1,1),nonlinearity,padding,strides,use_bias,data_format)
        self.a3 = Attention_Gate(num_channels*2,(1,1),nonlinearity,padding,strides,use_bias,data_format)
        self.a4 = Attention_Gate(num_channels,(1,1),nonlinearity,padding,strides,use_bias,data_format)

        self.u1 = Conv2D_Block(num_channels*8,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)
        self.u2 = Conv2D_Block(num_channels*4,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)
        self.u3 = Conv2D_Block(num_channels*2,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)
        self.u4 = Conv2D_Block(num_channels,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)

        self.conv_1x1 = tf.keras.layers.Conv2D(num_classes, (1,1), activation='linear', padding='same', data_format=data_format)

    def call(self, inputs, training=False):

        #ENCODER PATH
        x1 = self.conv_1(inputs)
        
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
        x2 = self.conv_2(pool1, training=training)
        
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x2)
        x3 = self.conv_3(pool2, training=training)
        
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x3)
        x4 = self.conv_4(pool3, training=training)
        
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x4)
        x5 = self.conv_5(pool4, training=training)

        #DECODER PATH
        up4 = self.up_conv_1(x5, training=training)
        a1 = self.a1(x4, up4, training=training)
        y1 = tf.keras.layers.concatenate([a1, up4])
        y1 = self.u1(y1, training=training)

        up5 = self.up_conv_2(y1, training=training)
        a2 = self.a2(x3, up5, training=training)
        y2 = tf.keras.layers.concatenate([a2, up5])
        y2 = self.u2(y2, training=training)

        up6 = self.up_conv_3(y2, training=training)
        a3 = self.a3(x2, up6, training=training)
        y3 = tf.keras.layers.concatenate([a3, up6])
        y3 = self.u3(y3, training=training)

        up7 = self.up_conv_4(y3, training=training)
        a4 = self.a4(x1, up7, training=training)
        y4 = tf.keras.layers.concatenate([a4, up7])
        y4 = self.u4(y4, training=training)
        y5 = self.conv_1x1(y4)

        if self.num_classes == 1:
            output = tf.keras.layers.Activation('sigmoid')(y5)
        else:
            output = tf.keras.layers.Activation('softmax')(y5)

        return output

class MultiResUnet(tf.keras.Model):
    "Tensorflow 2 Implementation of Multires UNet"

    def __init__(self,
                num_channels,
                num_classes,
                res_path_length,
                num_conv_layers=1,
                kernel_size=(3,3),
                strides=(1,1),
                pool_size=(2,2),
                use_bias=False,
                padding='same',
                nonlinearity='relu',
                use_batchnorm = True,
                use_transpose = True,
                data_format='channels_last',
                **kwargs):
                
        
        super(MultiResUnet, self).__init__(**kwargs)

        # ENCODING BLOCKS
        self.mresblock_1 = MultiResBlock(num_channels, kernel_size, nonlinearity, padding='same', strides=(1,1), data_format='channels_last')
        self.mresblock_2 = MultiResBlock(num_channels*2, kernel_size, nonlinearity, padding='same', strides=(1,1), data_format='channels_last')
        self.mresblock_3 = MultiResBlock(num_channels*4, kernel_size, nonlinearity, padding='same', strides=(1,1), data_format='channels_last')
        self.mresblock_4 = MultiResBlock(num_channels*8, kernel_size, nonlinearity, padding='same', strides=(1,1), data_format='channels_last')
        self.mresblock_5 = MultiResBlock(num_channels*16, kernel_size, nonlinearity, padding='same', strides=(1,1), data_format='channels_last')
        
        # DECODING BLOCKS
        self.mresblock_6 = MultiResBlock(num_channels, kernel_size, nonlinearity, padding='same', strides=(1,1), data_format='channels_last')
        self.mresblock_7 = MultiResBlock(num_channels*2, kernel_size, nonlinearity, padding='same', strides=(1,1), data_format='channels_last')
        self.mresblock_8 = MultiResBlock(num_channels*4, kernel_size, nonlinearity, padding='same', strides=(1,1), data_format='channels_last')
        self.mresblock_9 = MultiResBlock(num_channels*8, kernel_size, nonlinearity, padding='same', strides=(1,1), data_format='channels_last')

        self.pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.pool_4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.up_1 = Up_Conv2D(num_channels*8, kernel_size=(3,3), nonlinearity='relu', use_batchnorm=True, use_transpose=True, strides=(2,2), data_format='channels_last')
        self.up_2 = Up_Conv2D(num_channels*4, kernel_size=(3,3), nonlinearity='relu', use_batchnorm=True, use_transpose=True, strides=(2,2), data_format='channels_last')
        self.up_3 = Up_Conv2D(num_channels*2, kernel_size=(3,3), nonlinearity='relu', use_batchnorm=True, use_transpose=True, strides=(2,2), data_format='channels_last')
        self.up_4 = Up_Conv2D(num_channels, kernel_size=(3,3), nonlinearity='relu', use_batchnorm=True, use_transpose=True, strides=(2,2), data_format='channels_last')

        self.respath_1 = ResPath(res_path_length, num_channels,kernel_size=(1,1), nonlinearity='relu', padding='same', strides=(1,1), data_format='channels_last')
        self.respath_2 = ResPath(res_path_length, num_channels*2,kernel_size=(1,1), nonlinearity='relu', padding='same', strides=(1,1), data_format='channels_last')
        self.respath_3 = ResPath(res_path_length, num_channels*4,kernel_size=(1,1), nonlinearity='relu', padding='same', strides=(1,1), data_format='channels_last')
        self.respath_4 = ResPath(res_path_length, num_channels*8,kernel_size=(1,1), nonlinearity='relu', padding='same', strides=(1,1), data_format='channels_last')

        self.conv_1x1 = Conv2D_Block(num_classes,num_conv_layers,(1,1),'softmax',use_batchnorm=False,data_format=data_format)
       
    def call(self, x, training=False):
        
        #ENCODER PATH

        x1 = self.mresblock_1(x, training=training)
        pool_1 = self.pool_1(x1)
        res_1 = self.respath_1(x1, training=training)

        x2 = self.mresblock_2(pool_1, training=training)
        pool_2 = self.pool_2(x2)
        res_2 = self.respath_2(x2, training=training)

        x3 = self.mresblock_3(pool_2, training=training)
        pool_3 = self.pool_3(x3)
        res_3 = self.respath_3(x3, training=training)

        x4 = self.mresblock_4(pool_3, training=training)
        pool_4 = self.pool_4(x4)
        res_4 = self.respath_4(x4, training=training)

        x5 = self.mresblock_5(pool_4, training=training)

        up6 = tf.keras.layers.concatenate([self.up_1(x5), res_4])
        x6 =  self.mresblock_6(up6, training=training)
        
        up7 = tf.keras.layers.concatenate([self.up_2(x6), res_3])
        x7 =  self.mresblock_7(up7, training=training)

        up8 = tf.keras.layers.concatenate([self.up_3(x7), res_2])
        x8 =  self.mresblock_8(up8, training=training)

        up9 = tf.keras.layers.concatenate([self.up_4(x8), res_1])
        x9 =  self.mresblock_9(up9, training=training)

        output = self.conv_1x1(x9, training=training)

        return output
     

