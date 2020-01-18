import tensorflow as tf 
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, SpatialDropout2D, concatenate
#from tensorflow.keras import Model, Input

class Conv2D_Block(tf.keras.Model):

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
                 name="convolution_block"):

        super(Conv2D_Block, self).__init__(name=name)

        self.num_conv_layers = num_conv_layers
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.use_spatial_dropout = use_spatial_dropout

        self.conv = []
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=-1)
        self.activation = tf.keras.layers.Activation(nonlinearity)

        if use_spatial_dropout:
            self.dropout = tf.keras.layers.SpatialDropout2D(rate=dropout_rate)
        else:
            self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        for _ in range(num_conv_layers):
            self.conv.append(tf.keras.layers.Conv2D(num_channels, kernel_size, padding='same', data_format=data_format))
        
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
        
class Up_Conv2D(tf.keras.Model):

    def __init__(self, 
                 num_channels,
                 kernel_size=(2,2),
                 nonlinearity='relu',
                 use_batchnorm = False,
                 data_format='channels_last',
                 name="upsampling_convolution_block"):

        super(Up_Conv2D, self).__init__(name=name)

        self.use_batchnorm = use_batchnorm
        
        self.upsample = tf.keras.layers.UpSampling2D(size=(2,2))
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size, padding='same', data_format=data_format)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.activation = tf.keras.layers.Activation(nonlinearity)
        
    def call(self, inputs):
        
        x = inputs
        x = self.upsample(x)
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.batch_norm(x)
        outputs = self.activation(x)

        return outputs


class UNet(tf.keras.Model):
    """ Tensorflow 2 Implementation of 'U-Net: Convolutional Networks for Biomedical Image Segmentation'
    https://arxiv.org/abs/1505.04597."""

    def __init__(self, 
                 num_channels,
                 num_classes,
                 num_conv_layers=2,
                 kernel_size=(3,3),
                 nonlinearity='relu',
                 dropout_rate = 0.25, 
                 use_spatial_dropout = True,
                 data_format='channels_last',
                 name="unet"):

        super(UNet, self).__init__(name=name)

        self.conv_1 = Conv2D_Block(64,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)
        self.conv_2 = Conv2D_Block(128,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)
        self.conv_3 = Conv2D_Block(256,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)
        self.conv_4 = Conv2D_Block(512,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,use_dropout=True,dropout_rate=dropout_rate,use_spatial_dropout=use_spatial_dropout,data_format=data_format)
        self.conv_5 = Conv2D_Block(1024,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,use_dropout=True,dropout_rate=dropout_rate,use_spatial_dropout=use_spatial_dropout,data_format=data_format)
    
        self.up_5 = Up_Conv2D(512,(2,2),nonlinearity,use_batchnorm=True,data_format=data_format)
        self.up_6 = Up_Conv2D(256,(2,2),nonlinearity,use_batchnorm=True,data_format=data_format)
        self.up_7 = Up_Conv2D(128,(2,2),nonlinearity,use_batchnorm=True,data_format=data_format)
        self.up_8 = Up_Conv2D(64,(2,2),nonlinearity,use_batchnorm=True,data_format=data_format)

        self.up_conv4 = Conv2D_Block(512,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)
        self.up_conv3 = Conv2D_Block(256,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)
        self.up_conv2 = Conv2D_Block(128,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)
        self.up_conv1 = Conv2D_Block(64,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=True,data_format=data_format)

        #convolution filters at the output
        self.conv_output = tf.keras.layers.Conv2D(2, kernel_size, activation = nonlinearity, padding='same', data_format=data_format)
        self.conv_1x1 = tf.keras.layers.Conv2D(num_classes, kernel_size, padding='same', data_format=data_format)

        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2,2)) 

    def call(self, inputs):
        
        ##encoder blocks
        #1->64
        x1 = self.conv_1(inputs)
        
        #64->128
        x2 = self.maxpool(x1)
        x2 = self.conv_2(x2)
        
        #128->256
        x3 = self.maxpool(x2)
        x3 = self.conv_3(x3)

        #256->512
        x4 = self.maxpool(x3)
        x4 = self.conv_4(x4)

        #512->1024
        x5 = self.maxpool(x4)
        x5 = self.conv_5(x5)
        
        #decoder blocks
        #1024->512
        u5 = self.up_5(x5)
        u5 = tf.keras.layers.concatenate([x4, u5], axis=3)
        u5 = self.up_conv4(u5)

        #512->256
        u6 = self.up_6(u5)
        u6 = tf.keras.layers.concatenate([x3, u6], axis=3)
        u6 = self.up_conv3(u6)

        #256->128
        u7 = self.up_7(u6)
        u7 = tf.keras.layers.concatenate([x2, u7], axis=3)
        u7 = self.up_conv2(u7)

        #128->64
        u8 = self.up_8(u7)
        u8 = tf.keras.layers.concatenate([x1, u8], axis=3)
        u8 = self.up_conv1(u8)

        u9 = self.conv_output(u8)
        output = self.conv_1x1(u9)

        return output


#This is the old build_unet function written in functional API. Don't delete until we test the UNet on actual data 
#Build UNet using tf.keras Functional API
"""
def build_unet(num_classes,
            input_size = (256,256,1), 
            num_channels=64, 
            kernel_size=3, 
            non_linearity=tf.keras.activations.relu,
            dropout_rate=0.25,  
            data_format='channels_last',
            use_spatial_dropout=True):

    inputs = Input(input_size)
    conv1 = Conv2D(num_channels, kernel_size, activation = non_linearity, padding='same', data_format=data_format)(inputs)
    conv1 = Conv2D(num_channels, kernel_size, activation = non_linearity, padding = 'same', data_format=data_format)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(num_channels*2, kernel_size, activation = non_linearity, padding = 'same', data_format=data_format)(pool1)
    conv2 = Conv2D(num_channels*2, kernel_size, activation = non_linearity, padding = 'same', data_format=data_format)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(num_channels*4, kernel_size, activation = non_linearity, padding = 'same', data_format=data_format)(pool2)
    conv3 = Conv2D(num_channels*4, kernel_size, activation = non_linearity, padding = 'same', data_format=data_format)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(num_channels*8, kernel_size, activation = non_linearity, padding = 'same', data_format=data_format)(pool3)
    conv4 = Conv2D(num_channels*8, kernel_size, activation = non_linearity, padding = 'same', data_format=data_format)(conv4)
    
    if use_spatial_dropout:
        drop4 = SpatialDropout2D(dropout_rate)(conv4)
    else:
        drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(num_channels*16, kernel_size, activation = non_linearity, padding = 'same', data_format=data_format)(pool4)
    conv5 = Conv2D(num_channels*16, kernel_size, activation = non_linearity, padding = 'same', data_format=data_format)(conv5)
    
    if use_spatial_dropout:
        drop5 = SpatialDropout2D(dropout_rate)(conv5)
    else:
        drop5 = Dropout(dropout_rate)(conv5)
    
    up6 = Conv2D(512, 2, activation = non_linearity, padding = 'same', data_format=data_format)(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = non_linearity, padding = 'same', data_format=data_format)(merge6)
    conv6 = Conv2D(512, 3, activation = non_linearity, padding = 'same', data_format=data_format)(conv6)

    up7 = Conv2D(256, 2, activation = non_linearity, padding = 'same', data_format=data_format)(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = non_linearity, padding = 'same', data_format=data_format)(merge7)
    conv7 = Conv2D(256, 3, activation = non_linearity, padding = 'same', data_format=data_format)(conv7)

    up8 = Conv2D(128, 2, activation = non_linearity, padding = 'same', data_format=data_format)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = non_linearity, padding = 'same', data_format=data_format)(merge8)
    conv8 = Conv2D(128, 3, activation = non_linearity, padding = 'same', data_format=data_format)(conv8)

    up9 = Conv2D(64, 2, activation = non_linearity, padding = 'same', data_format=data_format)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = non_linearity, padding = 'same', data_format=data_format)(merge9)
    conv9 = Conv2D(64, 3, activation = non_linearity, padding = 'same', data_format=data_format)(conv9)
    conv9 = Conv2D(2, 3, activation = non_linearity, padding = 'same', data_format=data_format)(conv9)
    conv10 = Conv2D(num_classes, 1, data_format=data_format)(conv9)

    model = Model(inputs = inputs, outputs = conv10)    

    return model
"""