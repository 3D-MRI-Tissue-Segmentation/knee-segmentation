import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
import math
import random

class Data_Generation():
    def __init__(self, dataset, dimensions):
        self.dataset = np.asarray(dataset)
        self.dims = dimensions
        self.center = (int(math.floor(self.dims['width']/2)), int(math.floor(self.dims['height']/2)))
        # self.center = (self.dims['width']/2, self.dims['height']/2)
        self.transformed_dataset = dict()
        self.augmented_data = list()
        self.corrupted_data = list()

    def rotations(self, angle, scale=1.0, use_initial_data=True):
        data = self.dataset if use_initial_data else self.transformed_dataset['translated']
        rotated_dataset = np.zeros(((2*self.dims['slices']+1)*angle, self.dims['height'], self.dims['width'], self.dims['channels']))
        img_counter = 0
        for angle in range(-angle, angle+1):
            M = cv2.getRotationMatrix2D(self.center, angle, scale)
            for img in data:
                rotated_dataset[img_counter] = cv2.warpAffine(img, M, (self.dims['width'], self.dims['height']))
                img_counter = img_counter + 1
        self.transformed_dataset['rotated'] = rotated_dataset

    def cropping(self, cropped_side, translation=0, use_initial_data=True):
        data = self.dataset if use_initial_data else self.transformed_dataset['rotated']
        translated_dataset = list()

        x_cropLimit_low = self.center[0] - int(cropped_side/2)
        x_cropLimit_high = self.center[0] + int(cropped_side/2)
        y_cropLimit_low = self.center[1] - int(cropped_side/2)
        y_cropLimit_high = self.center[1] + int(cropped_side/2)

        for move in range(-translation, translation+1):
            cropped_area = [[x_cropLimit_low+move, x_cropLimit_high+move, y_cropLimit_low, y_cropLimit_high, 0, self.dims['channels']+1],
                            [x_cropLimit_low-move, x_cropLimit_high-move, y_cropLimit_low, y_cropLimit_high, 0, self.dims['channels']+1],
                            [x_cropLimit_low, x_cropLimit_high, y_cropLimit_low+move, y_cropLimit_high+move, 0, self.dims['channels']+1],
                            [x_cropLimit_low, x_cropLimit_high, y_cropLimit_low-move, y_cropLimit_high+move, 0, self.dims['channels']+1]]
            for img in data:
                for crop in range(0, len(cropped_area)):
                    translated_dataset.append(img[cropped_area[crop][0]:cropped_area[crop][1], cropped_area[crop][2]:cropped_area[crop][3], cropped_area[crop][4]:cropped_area[crop][5]])
        self.transformed_dataset['translated'] = translated_dataset

    def shuffleAugmentedData(self):
        self.augmented_data = self.transformed_dataset['rotated'] + self.transformed_dataset['translated']
        random.shuffle(self.augmented_data)
    
    def corruption(self, percentage=100, use_initial_data=True):
        if (use_initial_data == False):
            self.shuffleAugmentedData
        data = self.dataset if use_initial_data else self.augmented_data
        for img in data[0:(percentage/100)*len(data)]:
            self.addNoise(image)
            self.enhance(image, 3, 100)
        

    def addNoise(image):
        return np.add(image, abs(np.random.normal(255/2, 10, image.shape)))

    def enhance(image, contrast, brightness):
        #contrast > 1 increases contrast
        #0 < contrast < 1 decreases the contrast
        #-127< brightness < 127 is a good brigthness range
        assert contrast>0, "The contrast variable can't be lower or equal to zero"
        return contrast*image + brightness

    def getTransformedData(self):
        return self.transformed_dataset

dims = {
    'slices': 2,
    'width':960,
    'height': 640,
    'channels':3,
}

dataset = (cv2.imread('../../Data/Tests_data/try/house1.jpg'), cv2.imread('../../Data/Tests_data/try/house2.jpg'))
dataset = list(dataset)
dataset[1] = dataset[1][0:640,0:960,:]
dataset = tuple(dataset)
assert np.shape(dataset[0]) == np.shape(dataset[1])
cv2.imshow('image',dataset[0])
print(np.shape(dataset[0]))
print(np.shape(dataset[1]))
# cv2.waitKey(0)

data_gen = Data_Generation(dataset, dims)
data_gen.rotations(10)
data_gen.cropping(288, 15)

transformed_data = data_gen.getTransformedData()
plt.imshow((transformed_data['translated'][9])/255)
plt.show()
# cv2.waitKey(0)