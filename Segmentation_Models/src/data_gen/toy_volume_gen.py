import numpy as np
from random import randint

class Toy_Volume:
    def __init__(self, n_classes, width, height, depth, colour_channels=3):
        self.init_check(n_classes, width, height, depth, colour_channels)
        self.n_classes = n_classes
        self.width = width
        self.height = height
        self.depth = depth
        self.colour_channels = colour_channels
        self.class_colours = Toy_Volume.get_class_colours(n_classes, colour_channels)
        #self.image = self.get_empty_array()
        #self.one_hot_array = self.get_empty_array(colour_channels=self.n_classes)


    def init_check(self, n_classes, width, height, depth, colour_channels):
        pass

    @staticmethod
    def get_class_colours(n_classes, colour_channels):
        from src.data_gen.toy_image_gen.Toy_Image import get_class_colours
        return get_class_colours(n_classes, colour_channels)


    def get_empty_array(self, colour_channels):
        pass


def get_test_volumes(n_volumes, n_reps, n_classes, 
                     width, height, depth, colour_channels):
    #volumes, one_hots = [], []
    volumes, one_hots = None, None

    return volumes, one_hots