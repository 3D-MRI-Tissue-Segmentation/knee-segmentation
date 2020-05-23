import tensorflow as tf
from tensorflow.keras.applications import VGG16


class VGG16_Encoder(object):

    def __init__(self,
                 weights_init='imagenet'):

        self.weights_init = weights_init
        self.model = VGG16(weights=weights_init, include_top=False)
        self.layer_list = dict([(layer.name, layer)
                                for layer in self.model.layers])

        # layers to be constructed
        self.conv_1 = self.construct_conv_block([self.layer_list['block1_conv1'],
                                                self.layer_list['block1_conv2']])
        self.conv_2 = self.construct_conv_block([self.layer_list['block2_conv1'],
                                                self.layer_list['block2_conv2']])
        self.conv_3 = self.construct_conv_block([self.layer_list['block3_conv1'],
                                                self.layer_list['block3_conv2'],
                                                self.layer_list['block3_conv3']])
        self.conv_4 = self.construct_conv_block([self.layer_list['block4_conv1'],
                                                self.layer_list['block4_conv2'],
                                                self.layer_list['block4_conv3']])
        self.conv_5 = self.construct_conv_block([self.layer_list['block5_conv1'],
                                                self.layer_list['block5_conv2'],
                                                self.layer_list['block5_conv3']])

    def construct_conv_block(self, layers):
        model = tf.keras.Sequential()
        for i in range(len(layers)):
            model.add(layers[i])
        return model

    def freeze_pretrained_layers(self):
        for layer in self.model.layers:
            layer.trainable = False
