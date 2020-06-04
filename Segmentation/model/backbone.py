import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.models import Model

class Encoder(object):
    def __init__(self,
                 weights_init,
                 model_architecture='vgg16'):

        self.weights_init = weights_init
        if model_architecture == 'vgg16':
            self.model = VGG16(weights=self.weights_init, include_top=False)
            self.bridge_list = [2, 5, 9, 13, 17]

        elif model_architecture == 'vgg19':
            self.model = VGG19(weights=self.weights_init, include_top=False)
            self.bridge_list = [2, 5, 10, 15, 20]

        elif model_architecture == 'resnet50':
            self.model = ResNet50(weights=self.weights_init, include_top=False)
            self.bridge_list = [4, 38, 80, 142, -1]

        elif model_architecture == 'resnet50v2':
            self.model = ResNet50V2(weights=self.weights_init, include_top=False)
            self.bridge_list = [2, 27, 62, 108, -1] 

        elif model_architecture == 'resnet101':
            self.model = ResNet101(weghts=self.weights_init, include_top=False)
            self.bridge_list = [4, 38, 80, 312, -1]

        elif model_architecture == 'resnet101v2':
            self.model = ResNet101V2(weights=self.weights_init, include_top=False)
            self.bridge_list = [2, 27, 62, 328, -1]

        elif model_architecture == 'resnet152':
            self.model = ResNet152(weights=self.weights_init, include_top=False)
            self.bridge_list = [4, 38, 120, 482, -1]

        elif model_architecture == 'resnet152v2':
            self.model = ResNet152V2(weights=self.weights_init, include_top=False)
            self.bridge_list = [2, 27, 117, 515, -1]
        
    def construct_backbone(self):
        output_list = []
        for _, layer_idx in enumerate(self.bridge_list):
            layer_output = self.model.layers[layer_idx].output
            output_list.append(layer_output)

        backbone = Model(inputs=self.model.input, outputs=output_list)

        return backbone

    def freeze_pretrained_layers(self):
        for layer in self.model.layers:
            layer.trainable = False
