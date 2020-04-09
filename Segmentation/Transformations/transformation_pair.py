import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import random

class Transformations2D():
    def __init__(self, channels, angle, translation, output_side):
        self.pair = []
        self.dims = dict()
        self.angle = angle
        self.translation = translation
        self.side = output_side
        self.center = (0, 0)

        self.rotated_pair = []
        self.transformed_pair = np.zeros((2, output_side, output_side, channels))
        self.report = dict(transformation="None", angle=0, translation="None")

    def rotation(self, scale=1.0):
        teta = random.randint(-self.angle, self.angle)
        self.report['angle'] = teta

        M = cv2.getRotationMatrix2D(self.center, teta, scale)
        for img_counter, img in enumerate(self.pair):
            self.rotated_pair[img_counter] = cv2.warpAffine(img, M, (self.dims['width'], self.dims['height']))

    def croppingTranslation(self, pair="None", move="None"):
        if type(pair) == str:
            pair = self.pair
        if move == "None":
            move = random.randint(0, self.translation)
        else:
            self.report['translation'] = 0

        which = random.randint(0, 3)
        where = np.zeros(4, dtype='int8')
        where[which] = 1
        if (which == 0 or which == 1) and self.report['translation'] != 0:
            self.report['translation'] = f"x by {move}"
        if (which == 2 or which == 3) and self.report['translation'] != 0:
            self.report['translation'] = f"y by {move}"
        
        x_cropLimit_low = int(self.center[0] - self.side/2 + where[0]*move - where[1]*move)
        x_cropLimit_high = int(self.center[0] + self.side/2 + where[0]*move - where[1]*move)
        y_cropLimit_low = int(self.center[1] - self.side/2 + where[2]*move - where[3]*move)
        y_cropLimit_high = int(self.center[1] + self.side/2 + where[2]*move - where[3]*move)

        for img_counter, img in enumerate(pair):
            self.transformed_pair[img_counter] = img[y_cropLimit_low:y_cropLimit_high, x_cropLimit_low:x_cropLimit_high, 0:self.dims['channels']+1]

    def random_transform(self, pair, dimensions):
        self.pair = np.asarray(pair)
        self.dims = dimensions
        self.center = (int(math.floor(self.dims['width']/2)), int(math.floor(self.dims['height']/2)))
        self.rotated_pair = np.zeros((2, self.dims['height'], self.dims['width'], self.dims['channels']))

        which_transform = random.randint(0, 1)
        if which_transform == 0:
            self.rotation()
            self.croppingTranslation(self.rotated_pair, 0)
            self.report['transformation'] = 'rotation'
        elif which_transform == 1:
            self.croppingTranslation()
            self.report['transformation'] = 'translation'

    def getTransformedPair(self):
        returning = [self.transformed_pair, self.report]
        return returning

from Segmentation.utils.data_loader import dataset_generator, get_multiclass
from Segmentation.utils.training_utils import visualise_multi_class

train_ds = dataset_generator('./Data/train_2d', batch_size=5, shuffle=True)
valid_ds = dataset_generator('./Data/valid_"d', batch_size=5)

transform = Transformations2D()

for step, (images, labels) in enumerate(train_ds):

    print('step: {}'.format(step))
    labels = get_multiclass(labels)

    

    
    