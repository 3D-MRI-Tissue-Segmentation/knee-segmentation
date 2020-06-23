import tensorflow as tf
import tensorflow.keras.layers as tfkl
from Segmentation.model.vnet_build_blocks import Conv3D_Block, Up_Conv3D

class VNet(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 num_classes,
                 num_conv_layers=2,
                 kernel_size=(3, 3, 3),
                 nonlinearity='relu',
                 use_batchnorm=True,
                 dropout_rate=0.25,
                 use_spatial_dropout=True,
                 data_format='channels_last',
                 **kwargs):

        super(VNet, self).__init__(**kwargs)
        
        self.num_classes = num_classes

        self.conv_1 = Conv3D_Block(num_channels,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.conv_2 = Conv3D_Block(num_channels*2,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.conv_3 = Conv3D_Block(num_channels*4,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.conv_4 = Conv3D_Block(num_channels*8,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,use_dropout=True,dropout_rate=dropout_rate,use_spatial_dropout=use_spatial_dropout,data_format=data_format)
        self.conv_5 = Conv3D_Block(num_channels*16,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,use_dropout=True,dropout_rate=dropout_rate,use_spatial_dropout=use_spatial_dropout,data_format=data_format)

        self.up_5 = Up_Conv3D(num_channels*8,(2,2,2),nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.up_6 = Up_Conv3D(num_channels*4,(2,2,2),nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.up_7 = Up_Conv3D(num_channels*2,(2,2,2),nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.up_8 = Up_Conv3D(num_channels,(2,2,2),nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)

        self.up_conv4 = Conv3D_Block(num_channels*8,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.up_conv3 = Conv3D_Block(num_channels*4,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.up_conv2 = Conv3D_Block(num_channels*2,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.up_conv1 = Conv3D_Block(num_channels,num_conv_layers,kernel_size,nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)

        # convolution num_channels at the output
        # self.conv_output = tf.keras.layers.Conv3D(2, kernel_size, activation=nonlinearity, padding='same', data_format=data_format)
        self.conv_1x1 = tf.keras.layers.Conv3D(num_classes, kernel_size, padding='same', data_format=data_format)

    def call(self, inputs, training=False):

        # encoder blocks
        # 1->64
        x1 = self.conv_1(inputs, training=training)

        # 64->128
        x2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x1)
        x2 = self.conv_2(x2, training=training)

        # 128->256
        x3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x2)
        x3 = self.conv_3(x3, training=training)

        # 256->512
        x4 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x3)
        x4 = self.conv_4(x4, training=training)

        # 512->1024
        x5 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x4)
        x5 = self.conv_5(x5, training=training)

        # decoder blocks
        # 1024->512
        u5 = self.up_5(x5, training=training)
        u5 = tf.keras.layers.concatenate([x4, u5], axis=-1)
        u5 = self.up_conv4(u5, training=training)

        # 512->256
        u6 = self.up_6(u5, training=training)
        u6 = tf.keras.layers.concatenate([x3, u6], axis=-1)
        u6 = self.up_conv3(u6, training=training)

        # 256->128
        u7 = self.up_7(u6, training=training)
        u7 = tf.keras.layers.concatenate([x2, u7], axis=-1)
        u7 = self.up_conv2(u7, training=training)

        # 128->64
        u8 = self.up_8(u7, training=training)
        u8 = tf.keras.layers.concatenate([x1, u8], axis=-1)
        u8 = self.up_conv1(u8, training=training)

        # u9 = self.conv_output(u8)
        output = self.conv_1x1(u8)

        if self.num_classes == 1:
            output = tfkl.Activation('sigmoid')(output)
        else:
            output = tfkl.Activation('softmax')(output)

        return output
