import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19, ResNet50


class Encoder(object):
    def __init__(self):

        self.model = None
        self.conv_list = []

    def get_conv_block(self, idx):
        return self.conv_list[idx]

    def construct_conv_block(self, layers):
        model = tf.keras.Sequential()
        for i in range(len(layers)):
            model.add(layers[i])
        return model

    def freeze_pretrained_layers(self):
        for layer in self.model.layers:
            layer.trainable = False

class VGG16_Encoder(Encoder):

    def __init__(self,
                 weights_init='imagenet'):

        super(VGG16_Encoder, self).__init__()

        self.weights_init = weights_init
        self.model = VGG16(weights=self.weights_init, include_top=False)
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

        self.conv_list = [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5]

class VGG19_Encoder(VGG16_Encoder):

    def __init__(self,
                 weights_init='imagenet'):
        super(VGG19_Encoder, self).__init__()

        self.model = VGG19(weights=self.weights_init, include_top=False)
        self.layer_list = dict([(layer.name, layer)
                                for layer in self.model.layers])

        # block 3 to 5 are different from VGG16
        self.conv_3 = self.construct_conv_block([self.layer_list['block3_conv1'],
                                                 self.layer_list['block3_conv2'],
                                                 self.layer_list['block3_conv3'],
                                                 self.layer_list['block3_conv4']])
        self.conv_4 = self.construct_conv_block([self.layer_list['block4_conv1'],
                                                 self.layer_list['block4_conv2'],
                                                 self.layer_list['block4_conv3'],
                                                 self.layer_list['block4_conv4']])
        self.conv_5 = self.construct_conv_block([self.layer_list['block5_conv1'],
                                                 self.layer_list['block5_conv2'],
                                                 self.layer_list['block5_conv3'],
                                                 self.layer_list['block5_conv4']])

        self.conv_list = [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5]

class ResNet50_Encoder(Encoder):

    def __init__(self,
                 weights_init='imagenet'):

        self.weights_init = weights_init

        self.model = ResNet50(weights=self.weights_init, include_top=False)

        self.conv_1 = self.construct_conv_block(self.model.layers[1:6])
        self.conv_2 = self.construct_conv_block(self.model.layers[7:39])
        self.conv_3 = self.construct_conv_block(self.model.layers[39:81])
        self.conv_4 = self.construct_conv_block(self.model.layers[81:143])
        self.conv_5 = self.construct_conv_block(self.model.layers[143:])

        self.conv_list = [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5]