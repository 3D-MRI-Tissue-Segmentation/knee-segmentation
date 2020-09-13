import tensorflow as tf
import tensorflow.keras.layers as tfkl
import inspect
from Segmentation.model.vnet_build_blocks import Conv_ResBlock, Up_ResBlock

class VNet(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 num_classes,
                 use_2d=True,
                 num_conv_layers=2,
                 kernel_size=3,
                 activation='prelu',
                 use_batchnorm=True,
                 noise=0.0,
                 dropout_rate=0.25,
                 use_spatial_dropout=True,
                 name="vnet",
                 **kwargs):

        self.params = str(inspect.currentframe().f_locals)
        super(VNet, self).__init__(name=name)
        self.noise = noise

        block_args = {
            'use_2d': use_2d,
            'num_conv_layers': num_conv_layers,
            'kernel_size': kernel_size,
            'activation': activation,
            'use_batchnorm': use_batchnorm,
            'dropout_rate': dropout_rate,
            'use_spatial_dropout': use_spatial_dropout,
        }

        self.contracting_path = []

        for i in range(len(num_channels)):
            output_ch = num_channels[i]
            self.contracting_path.append(Conv_ResBlock(output_ch,
                                                       **block_args,
                                                       **kwargs))

        self.upsampling_path = []
        n = len(num_channels) - 1
        for i in range(n, -1, -1):
            output_ch = num_channels[i]
            self.upsampling_path.append(Up_ResBlock(output_ch,
                                                    num_conv_layers,
                                                    use_2d,
                                                    kernel_size,
                                                    **kwargs))

        if use_2d:
            self.conv_output = tfkl.Conv2D(num_classes,
                                           kernel_size=kernel_size,
                                           activation=None,
                                           padding='same')
        else:
            self.conv_output = tfkl.Conv3D(num_classes,
                                           kernel_size=kernel_size,
                                           activation=None,
                                           padding='same')
        if activation == 'prelu':
            self.activation = tfkl.PReLU()  # alpha_initializer=tf.keras.initializers.Constant(value=0.25))
        else:
            self.activation = tfkl.Activation(activation)

        if use_2d:
            self.conv_1x1 = tfkl.Conv2D(filters=num_classes,
                                        kernel_size=(1, 1),
                                        padding='same')
        else:
            self.conv_1x1 = tfkl.Conv3D(filters=num_classes,
                                        kernel_size=(1, 1, 1),
                                        padding='same')

        self.output_act = tfkl.Activation('sigmoid' if num_classes == 1 else 'softmax')

    def call(self, x, training):

        if self.noise and training:
            x = tfkl.GaussianNoise(self.noise)(x)

        blocks = []
        # encoder blocks
        for _, down in enumerate(self.contracting_path):
            x, x_before = down(x, training=training)
            blocks.append(x_before)

        # decoder blocks
        for j, up in enumerate(self.upsampling_path):
            x = up([x, blocks[-j - 1]], training=training)

        output = self.conv_output(x)
        output = self.activation(output)

        output = self.conv_1x1(output)
        output = self.output_act(output)
        return output
