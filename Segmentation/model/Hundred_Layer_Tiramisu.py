import tensorflow as tf
import tensorflow.keras.layers as tfkl

'''The implementation of the 100 layer Tiramisu Network follows
directly from the publication found at https://arxiv.org/pdf/1611.09326.pdf'''

class Hundred_Layer_Tiramisu(tf.keras.Model):
    def __init__(self,
                 growth_rate,
                 layers_per_block,
                 num_channels,
                 num_classes,
                 kernel_size=(3, 3),
                 pool_size=(2, 2),
                 activation='relu',
                 dropout_rate=0.2,
                 strides=(2, 2),
                 padding='same',
                 use_dropout=False,
                 use_concat=True,
                 **kwargs):

        super(Hundred_Layer_Tiramisu, self).__init__(**kwargs)

        self.growth_rate = growth_rate
        self.layers_per_block = layers_per_block
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.strides = strides
        self.padding = padding
        self.use_dropout = use_dropout
        self.use_concat = use_concat

        self.conv_3x3 = tfkl.Conv2D(self.num_channels,
                                    kernel_size,
                                    padding='same')
        self.dense_block_list = []
        self.up_transition_list = []   

        self.conv_1x1 = tfkl.Conv2D(filters=num_classes,
                                    kernel_size=(1, 1),
                                    padding='same')

        layers_counter = 0
        num_filters = num_channels

        print(len(self.layers_per_block))

        for idx in range(0, len(self.layers_per_block)):
            print(idx)
            num_conv_layers = layers_per_block[idx]
            self.dense_block_list.append(dense_layer(num_conv_layers,
                                                     growth_rate,
                                                     kernel_size,
                                                     dropout_rate,
                                                     activation,
                                                     use_dropout=False,
                                                     use_concat=True))

            layers_counter = layers_counter + num_conv_layers
            num_filters = num_channels + layers_counter * growth_rate

            if idx != len(self.layers_per_block)-1:
                self.dense_block_list.append(down_transition(num_channels=num_filters,
                                                             kernel_size=(1, 1),
                                                             pool_size=(2, 2),
                                                             dropout_rate=0.2,
                                                             activation='relu',
                                                             use_dropout=False))

        for idx in range(len(self.layers_per_block) - 1, 0, -1):
            num_conv_layers = layers_per_block[idx - 1]
            num_filters = num_conv_layers * growth_rate
            self.up_transition_list.append(up_transition(num_conv_layers,
                                                         num_channels=num_filters,
                                                         growth_rate=self.growth_rate,
                                                         kernel_size=(3, 3),
                                                         strides=(2, 2),
                                                         padding='same',
                                                         use_concat=False))

    def call(self, inputs, training=False):
        blocks = []
        x = self.conv_3x3(inputs)
        for i, down in enumerate(self.dense_block_list):
            x = down(x, training=training)
            if i % 2 == 0 and i != len(self.dense_block_list)-1:
                blocks.append(x)

        for i, up in enumerate(self.up_transition_list):
            x = up(x, blocks[- i-1], training=training)

        x = self.conv_1x1(x)
        if self.num_classes == 1:
            output = tfkl.Activation('sigmoid')(x)
        else:
            output = tfkl.Activation('softmax')(x)
        return output

'''------------------------------------------------------------------'''

class conv_layer(tf.keras.Sequential):

    def __init__(self,
                 num_channels,
                 kernel_size=(3, 3),
                 dropout_rate=0.2,
                 activation='relu',
                 use_dropout=False,
                 **kwargs):

        super(conv_layer, self).__init__(**kwargs)

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_dropout = use_dropout

        self.add(tfkl.BatchNormalization(axis=-1,
                                         momentum=0.95,
                                         epsilon=0.001))

        self.add(tfkl.Activation(self.activation))

        self.add(tfkl.Conv2D(self.num_channels,
                             self.kernel_size,
                             padding='same',
                             activation=None, 
                             use_bias=True))

        if use_dropout:
            self.add(tfkl.Dropout(rate=self.dropout_rate))

    def call(self, inputs, training=False):

        outputs = super(conv_layer, self).call(inputs, training=training)    
        return outputs

'''-----------------------------------------------------------------'''

class dense_layer(tf.keras.Sequential):

    def __init__(self,
                 num_conv_layers,
                 growth_rate,
                 kernel_size=(3, 3),
                 dropout_rate=0.2,
                 activation='relu',
                 use_dropout=False,
                 use_concat=True,
                 **kwargs):

        super(dense_layer, self).__init__(**kwargs)

        self.num_conv_layers = num_conv_layers
        self.growth_rate = growth_rate
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_dropout = use_dropout
        self.use_concat = use_concat

        self.conv_list = []
        for layer in range(num_conv_layers):
            self.conv_list.append(conv_layer(num_channels=self.growth_rate,
                                             kernel_size=self.kernel_size,
                                             dropout_rate=self.dropout_rate,
                                             activation=self.activation,
                                             use_dropout=self.use_dropout))

    def call(self, inputs, training=False):
        dense_output = []
        x = inputs
        for i, conv in enumerate(self.conv_list):
            out = conv(x, training=training)
            x = tfkl.concatenate([x, out], axis=-1)
            dense_output.append(out)

        x = tfkl.concatenate(dense_output, axis=-1)

        if self.use_concat:
            x = tfkl.concatenate([x, inputs], axis=-1)

        outputs = x
        return outputs

'''-----------------------------------------------------------------'''

class down_transition(tf.keras.Sequential):

    def __init__(self,
                 num_channels,
                 kernel_size=(1, 1),
                 pool_size=(2, 2),
                 dropout_rate=0.2,
                 activation='relu',
                 use_dropout=False,
                 **kwargs):

        super(down_transition, self).__init__(**kwargs)

        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_dropout = use_dropout

        self.add(tfkl.BatchNormalization(axis=-1,
                                         momentum=0.95,
                                         epsilon=0.001))
        self.add(tfkl.Activation(activation))
        self.add(tfkl.Conv2D(num_channels, kernel_size, padding='same'))

        if use_dropout:
            self.add(tfkl.Dropout(rate=self.dropout_rate))

        self.add(tfkl.MaxPooling2D(pool_size))
    
    def call(self, inputs, training=False):

        outputs = super(down_transition, self).call(inputs, training=training)

        return outputs

'''-----------------------------------------------------------------'''

class up_transition(tf.keras.Model):

    def __init__(self,
                 num_conv_layers,
                 num_channels,
                 growth_rate,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 padding='same',
                 activation='relu',
                 use_concat=False,
                 **kwargs):
        
        super(up_transition, self).__init__(**kwargs)

        self.num_conv_layers = num_conv_layers
        self.num_channels = num_channels
        self.growth_rate = growth_rate
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_concat = use_concat

        self.up_conv = tfkl.Conv2DTranspose(num_channels,
                                            kernel_size,
                                            strides,
                                            padding)

        self.dense_block = dense_layer(num_conv_layers,
                                       growth_rate, 
                                       kernel_size,
                                       strides,
                                       activation,
                                       use_concat=self.use_concat)

    def call(self, inputs, bridge, training=False):
        
        up = self.up_conv(inputs, training=training)
        db_up = self.dense_block(up, training=training)
        c_up = tfkl.concatenate([db_up, bridge], axis=3)

        return c_up
