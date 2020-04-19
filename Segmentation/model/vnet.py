import tensorflow as tf
import inspect
from Segmentation.model.vnet_build_blocks import Conv3d_ResBlock, Up_ResBlock

class VNet(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 num_classes,
                 num_conv_layers=2,
                 kernel_size=(3, 3, 3),
                 nonlinearity='relu',
                 use_batchnorm=True,
                 # dropout_rate=0.25,
                 # use_spatial_dropout=True,
                 merge_connections=True,
                 data_format='channels_last',
                 name="vnet"):
        self.params = str(inspect.currentframe().f_locals)
        super(VNet, self).__init__(name=name)
        self.merge_connections = merge_connections

        self.conv_1 = Conv3d_ResBlock(num_channels=num_channels,num_conv_layers=num_conv_layers,kernel_size=kernel_size,nonlinearity=nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.conv_2 = Conv3d_ResBlock(num_channels=num_channels*2,num_conv_layers=num_conv_layers,kernel_size=kernel_size,nonlinearity=nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.conv_3 = Conv3d_ResBlock(num_channels=num_channels*4,num_conv_layers=num_conv_layers,kernel_size=kernel_size,nonlinearity=nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.conv_4 = Conv3d_ResBlock(num_channels=num_channels*8,num_conv_layers=num_conv_layers,kernel_size=kernel_size,nonlinearity=nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.conv_5 = Conv3d_ResBlock(num_channels=num_channels*16,num_conv_layers=num_conv_layers,kernel_size=kernel_size,nonlinearity=nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)

        self.upconv_4 = Up_ResBlock(num_channels=num_channels*8,kernel_size=(2,2,2),nonlinearity=nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.upconv_3 = Up_ResBlock(num_channels=num_channels*4,kernel_size=(2,2,2),nonlinearity=nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.upconv_2 = Up_ResBlock(num_channels=num_channels*2,kernel_size=(2,2,2),nonlinearity=nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)
        self.upconv_1 = Up_ResBlock(num_channels=num_channels,kernel_size=(2,2,2),nonlinearity=nonlinearity,use_batchnorm=use_batchnorm,data_format=data_format)

        # convolution num_channels at the output
        self.conv_output = tf.keras.layers.Conv3D(filters=num_classes, kernel_size=kernel_size, activation=nonlinearity, padding='same', data_format=data_format)
        self.conv_1x1 = tf.keras.layers.Conv3D(filters=num_classes, kernel_size=kernel_size, padding='same', data_format=data_format)

    def call(self, inputs):

        # encoder blocks
        x1 = self.conv_1(inputs)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)
        x4 = self.conv_4(x3)
        
        # lowest layer
        x5 = self.conv_5(x4)

        # decoder blocks
        if self.merge_connections:
            x5 = tf.keras.layers.concatenate([x5, x4], axis=-1)
        u4 = self.upconv_4(x5)

        if self.merge_connections:
            u4 = tf.keras.layers.concatenate([u4, x3], axis=-1)
        u3 = self.upconv_3(u4)

        if self.merge_connections:
            u3 = tf.keras.layers.concatenate([u3, x2], axis=-1)
        u2 = self.upconv_2(u3)

        if self.merge_connections:
            u2 = tf.keras.layers.concatenate([u2, x1], axis=-1)
        u1 = self.upconv_1(u2)

        output = self.conv_output(u1)

        print(output.shape)
        output = self.conv_1x1(output)
        print("final", output.shape)
        return output
