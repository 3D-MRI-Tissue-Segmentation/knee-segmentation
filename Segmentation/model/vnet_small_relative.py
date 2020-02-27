import tensorflow as tf
from Segmentation.model.vnet_build_blocks import Conv3D_Block, Up_Conv3D


class VNet_Small_Relative(tf.keras.Model):

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
                 merge_connections=False,
                 num_compressors=3,
                 compressor_filters=3,
                 name="vnet_small_relative"):

        super(VNet_Small_Relative, self).__init__(name=name)
        self.merge_connections = merge_connections

        self.conv_1 = Conv3D_Block(num_channels, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c1")
        self.conv_2 = Conv3D_Block(num_channels * 2, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c2")
        self.conv_3 = Conv3D_Block(num_channels * 4, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c3")

        self.up_3 = Up_Conv3D(num_channels * 2, (2, 2, 2), nonlinearity,
                              use_batchnorm=use_batchnorm, data_format=data_format, name="cu3")
        self.up_2 = Up_Conv3D(num_channels, (2, 2, 2), nonlinearity,
                              use_batchnorm=use_batchnorm, data_format=data_format, name="cu2")
        self.up_conv2 = Conv3D_Block(num_channels * 2, num_conv_layers, kernel_size, nonlinearity,
                                     use_batchnorm=use_batchnorm, data_format=data_format, name="upc2")
        self.up_conv1 = Conv3D_Block(num_channels, num_conv_layers, kernel_size, nonlinearity,
                                     use_batchnorm=use_batchnorm, data_format=data_format, name="upc1")

        # compression section
        self.compressors = []
        for idx, i in enumerate(range(num_compressors)):
            cfilters = compressor_filters
            if num_compressors == (idx + 1):
                cfilters = 1
            self.compressors.append(
                tf.keras.layers.Conv3D(cfilters, kernel_size=3, strides=2, padding="valid",
                                       activation=nonlinearity, data_format=data_format,
                                       name=f"compessor_{idx}")
            )

        self.comp_dense_1 = tf.keras.layers.Dense(16, activation="relu")
        self.comp_dense_2 = tf.keras.layers.Dense(4, activation="relu")
        self.comp_dense_3 = tf.keras.layers.Dense(1, activation="tanh")

        # convolution num_channels at the output
        self.conv_output = tf.keras.layers.Conv3D(2, kernel_size, activation=nonlinearity, padding='same',
                                                  data_format=data_format)
        self.conv_1x1 = tf.keras.layers.Conv3D(num_classes, kernel_size, padding='same',
                                               data_format=data_format)

    def call(self, inputs, training=True):
        image_inputs, pos_inputs = inputs

        # decoder
        # 1->64
        x1 = self.conv_1(image_inputs)
        tf.print("x1:", x1.get_shape())

        # 64->128
        x2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x1)
        x2 = self.conv_2(x2)
        tf.print("x2:", x2.get_shape())

        # 128->256
        x3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x2)
        x3 = self.conv_3(x3)
        tf.print("x3:", x3.get_shape())

        # compressor

        asd = 0
        comp = x3
        for i in self.compressors:
            comp = i(comp)
            tf.print(f"compressor {asd}:", comp.get_shape())
            asd += 1

        comp = tf.keras.layers.Flatten()(comp)
        tf.print("flat shape:", comp.get_shape())
        comp = self.comp_dense_1(comp)
        tf.print("flat shape:", comp.get_shape())
        comp = self.comp_dense_2(comp)
        tf.print("flat shape:", comp.get_shape())
        tf.print("pos:", pos_inputs.get_shape())
        tf.print("====================")
        comp_cat = tf.concat([comp, pos_inputs], axis=-1)
        tf.print("comp_cat:", comp_cat.get_shape())
        comp = self.comp_dense_3(comp)
        tf.print("flat shape:", comp.get_shape())
        assert comp.get_shape()[-1] == 1

        # encoder
        # 256->128
        tf.print(x3.get_shape())

        u3 = self.up_3(x3)

        tf.print("===============")

        tf.print(u3[0, :3, 0, 0])
        tf.print(u3[1, :3, 0, 0])
        tf.print(u3[2, :3, 0, 0])

        u3 = tf.keras.layers.add([u3, comp])

        tf.print("===============")

        tf.print(u3[0, :3, 0, 0])
        tf.print(u3[1, :3, 0, 0])
        tf.print(u3[2, :3, 0, 0])

        tf.print("===============")

        tf.print(u3.get_shape())

        if self.merge_connections:
            u3 = tf.keras.layers.concatenate([x2, u3], axis=4)
        u3 = self.up_conv2(u3)
        tf.print("u3:", u3.get_shape())

        # 128->64
        u2 = self.up_2(x2)
        tf.print("before u2:", u2.get_shape())
        if self.merge_connections:
            u2 = tf.keras.layers.concatenate([x1, u2], axis=4)
        tf.print("after u2:", u2.get_shape())
        u2 = self.up_conv1(u2)
        tf.print("u2:", u2.get_shape())

        u1 = self.conv_output(u2)
        tf.print("u1:", u1.get_shape())
        output = self.conv_1x1(u1)

        tf.print("output:", output.get_shape())
        return output
