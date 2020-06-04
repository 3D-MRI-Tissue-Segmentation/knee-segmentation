import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19, ResNet50


class Encoder(object):
    def __init__(self,
                 weights_init,
                 model_architecture='vgg16'):

        self.weights_init = weights_init
        if model_architecture == 'vgg16':
            self.model = VGG16(weights=self.weights_init, include_top=False)
            self.layer_list = dict([(layer.name, layer) for layer in self.model.layers])
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

        elif model_architecture == 'vgg19':
            self.model = VGG19(weights=self.weights_init, include_top=False)
            self.layer_list = dict([(layer.name, layer) for layer in self.model.layers])
            self.conv_1 = self.construct_conv_block([self.layer_list['block1_conv1'],
                                                     self.layer_list['block1_conv2']])
            self.conv_2 = self.construct_conv_block([self.layer_list['block2_conv1'],
                                                    self.layer_list['block2_conv2']])
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

        elif model_architecture == 'resnet50':
            self.model = ResNet50(weights=self.weights_init, include_top=False)
            self.layer_list = dict([(layer.name, layer) for layer in self.model.layers])
            self.conv_1 = self.construct_conv_block(self.model.layers[1:6])
            self.conv_2 = self.construct_conv_block(self.model.layers[7:39])
            self.conv_3 = self.construct_conv_block(self.model.layers[39:81])
            self.conv_4 = self.construct_conv_block(self.model.layers[81:143])
            self.conv_5 = self.construct_conv_block(self.model.layers[143:])

        self.conv_list = [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5]

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
