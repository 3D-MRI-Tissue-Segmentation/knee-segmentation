import tensorflow as tf 

class UNet(tf.keras.Model):
    """ Keras Implementation of 'U-Net: Convolutional Networks for Biomedical Image Segmentation'
    https://arxiv.org/abs/1505.04597."""

    def __init__(self, 
                 num_channels,
                 num_classes,
                 kernel_size,
                 nonlinearity=tf.keras.activations.relu,
                 dropout_rate = 0.25, 
                 use_spatial_dropout = True,
                 data_format='channels_last',
                 name="unet"):

        super(UNet, self).__init__(name=name)

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.nonlinearity = nonlinearity
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.data_format = data_format

        self.Conv_1 = tf.keras.layers.Conv2D(num_channels, kernel_size, activation = nonlinearity, padding='same', data_format=data_format)
        self.Conv_2 = tf.keras.layers.Conv2D(num_channels*2, kernel_size, activation = nonlinearity, padding='same', data_format=data_format)
        self.Conv_3 = tf.keras.layers.Conv2D(num_channels*4, kernel_size, activation = nonlinearity, padding='same', data_format=data_format)
        self.Conv_4 = tf.keras.layers.Conv2D(num_channels*8, kernel_size, activation = nonlinearity, padding='same', data_format=data_format)
        self.Conv_5 = tf.keras.layers.Conv2D(num_channels*16, kernel_size, activation = nonlinearity, padding='same', data_format=data_format)
        self.Conv_6 = tf.keras.layers.Conv2D(num_channels*8, 2, activation = nonlinearity, padding='same', data_format=data_format)
        self.Conv_7 = tf.keras.layers.Conv2D(num_channels*4, 2, activation = nonlinearity, padding='same', data_format=data_format)
        self.Conv_8 = tf.keras.layers.Conv2D(num_channels*2, 2, activation = nonlinearity, padding='same', data_format=data_format)
        self.Conv_9 = tf.keras.layers.Conv2D(num_channels, 2, activation = nonlinearity, padding='same', data_format=data_format)
        self.Conv_10 = tf.keras.layers.Conv2D(2, kernel_size, activation = nonlinearity, padding='same', data_format=data_format)
        self.Conv_11 = tf.keras.layers.Conv2D(num_classes, 1, padding='same', data_format=data_format)

        self.DownSample = tf.keras.layers.MaxPooling2D(pool_size=(2,2)) 
        self.UpSample = tf.keras.layers.UpSampling2D(pool_size=(2,2))

        if use_spatial_dropout:
            self.Dropout = tf.keras.layers.SpatialDropout2D(rate=dropout_rate)
        else:
            self.Dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs):

        ##encoder blocks 
        #first block
        x1 = self.Conv_1(inputs)
        x1 = self.Conv_1(x1)
        p1 = self.DownSample(x1)
        #second block
        x2 = self.Conv_2(p1)
        x2 = self.Conv_2(x2)
        p2 = self.DownSample(x2)
        #third block
        x3 = self.Conv_3(p2)
        x3 = self.Conv_3(x3)
        p3 = self.DownSample(x3)
        #fourth block
        x4 = self.Conv_4(p3)
        x4 = self.Conv_4(x4)
        d4 = self.Dropout(x4)
        p4 = self.DownSample(d4)
        #fifth block
        x5 = self.Conv_5(p4)
        x5 = self.Conv_5(x5)
        d5 = self.Dropout(x5)

        ##decoder blocks
        #first block 
        x6 = self.UpSample(d5)
        x6 = self.Conv_6(x6)
        m6 = tf.keras.layers.concatenate([d4, x6], axis=3)
        x6 = self.Conv_4(m6)
        x6 = self.Conv_4(x6)
        #second block
        x7 = self.UpSample(x6)
        x7 = self.Conv_7(x7)
        m7 = tf.keras.layers.concatenate([x3, x7], axis=3)
        x7 = self.Conv_3(m7)
        x7 = self.Conv_3(x7)
        #third block
        x8 = self.UpSample(x7)
        x8 = self.Conv_8(x8)
        m8 = tf.keras.layers.concatenate([x2, x8], axis=3)
        x8 = self.Conv_2(m8)
        x8 = self.Conv_2(x8)
        #fourth block
        x9 = self.UpSample(x8)
        x9 = self.Conv_9(x9)
        m9 = tf.keras.layers.concatenate([x1, x9], axis=3)
        x9 = self.Conv_1(m9)
        x9 = self.Conv_1(x9)
        #output block
        x10 = self.Conv_10(x9)
        
        return self.Conv_11(x10)