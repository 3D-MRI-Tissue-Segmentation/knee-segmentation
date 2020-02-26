import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import h5py
import math
import random

class Transformations3D():
    def __init__(self, max_angle, max_translation, output_size):
        self.pair = []
        self.dims = []
        self.center = [(0, 0), (0,0), (0,0)]
        self.angle = max_angle
        self.translation = max_translation
        self.output_size = output_size

        self.transformed_pair = np.zeros((2, output_size[0], output_size[1], output_size[2]))
        self.report = dict(XY_rotation=0, YZ_rotation=0, XZ_rotation=0, X_translation=0, Y_translation=0, Z_translation=0)

    def XY_rotation(self, scale=1.0):
        teta = random.randint(-self.angle, self.angle)
        self.report['XY_rotation'] = teta

        M = cv2.getRotationMatrix2D(self.center[0], teta, scale)
        for render_counter, render in enumerate(self.pair):
            for z in range(0, self.dims[2]):
                self.pair[render_counter, :, :, z] = cv2.warpAffine(render[:,:,z], M, (self.dims[0], self.dims[1]))
    
    def YZ_rotation(self, scale=1.0):
        teta = random.randint(-self.angle, self.angle)
        self.report['YZ_rotation'] = teta

        M = cv2.getRotationMatrix2D(self.center[1], teta, scale)
        for render_counter, render in enumerate(self.pair):
            for x in range(0, self.dims[0]):
                self.pair[render_counter, x, :, :] = cv2.warpAffine(render[x,:,:], M, (self.dims[1], self.dims[2]))
    
    def XZ_rotation(self, scale=1.0):
        teta = random.randint(-self.angle, self.angle)
        self.report['XZ_rotation'] = teta

        M = cv2.getRotationMatrix2D(self.center[2], teta, scale)
        for render_counter, render in enumerate(self.pair):
            for y in range(0, self.dims[1]):
                self.pair[render_counter, :, y, :] = cv2.warpAffine(render[:,y,:], M, (self.dims[0], self.dims[2]))

    def carvingTranslation(self):
        move = [0, 0, 0]
        which = random.randint(0, 2)
        move[which] = random.randint(-self.translation, self.translation)
        # move[0] = ranrandom.randint(-self.translation, self.translation) #translation in the x
        # move[1] = ranrandom.randint(-self.translation, self.translation) #translation in the y
        # move[2] = ranrandom.randint(-self.translation, self.translation) #translation in the z
        self.report['X_translation'] = move[0]
        self.report['Y_translation'] = move[1]
        self.report['Z_translation'] = move[2]

        x_lower = int(self.dims[0]/2 - self.output_size[0]/2 + move[0])
        y_lower = int(self.dims[1]/2 - self.output_size[1]/2 + move[1])
        z_lower = int(self.dims[2]/2 - self.output_size[2]/2 + move[2])

        x_boundary = range(x_lower, (x_lower + self.output_size[0]))
        y_boundary = range(y_lower, (y_lower + self.output_size[1]))
        z_boundary = range(z_lower, (z_lower + self.output_size[2]))

        for render_counter, render in enumerate(self.pair):
            self.transformed_pair[render_counter] = render[x_boundary, y_boundary, z_boundary]

    def random_transform(self, pair):
        self.pair = np.asarray(pair)
        self.dims = self.pair[0].shape
        self.center[0] = (int(math.floor(self.dims[0]/2)), int(math.floor(self.dims[1]/2))) #center for frontal slice
        self.center[1] = (int(math.floor(self.dims[2]/2)), int(math.floor(self.dims[1]/2))) #center for lateral slice
        self.center[2] = (int(math.floor(self.dims[0]/2)), int(math.floor(self.dims[2]/2))) #center for top slice

        self.XY_rotation()
        self.YZ_rotation()
        self.XZ_rotation()
        self.carvingTranslation()

    def getTransformedPair(self):
        returning = [self.transformed_pair, self.report]
        return returning

class gabba():
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

class Image3D():
    def __init__(self):
        self.image = []
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        self.dims = tuple()
        self.voxels_arr = []

    def make_mesh(self, image_3d, lower_boundary=-1000, higher_boundary=1000):
        self.image = np.asarray(image_3d)
        self.dims = self.image.shape
        self.voxels_arr = np.zeros((self.dims[1], self.dims[2], self.dims[0]), dtype=bool)

        for xx in range(0, self.dims[0]):
            for yy in range(0, self.dims[1]):
                for zz in range(0, self.dims[2]):
                    if (self.image[xx, yy, zz] > lower_boundary and self.image[xx, yy, zz] < higher_boundary):
                        self.voxels_arr[yy, zz, xx] = 1

    def getMesh(self, image_3d, lower_boundary=-1000, higher_boundary=1000):
        self.make_mesh(image_3d, lower_boundary, higher_boundary)
        self.ax.voxels(self.voxels_arr)
        return self.ax

    def show_mesh(self):
        self.ax.voxels(self.voxels_arr)
        plt.show()

dims = {
    'slices': 16,
    'width':16,
    'height': 16,
    'channels':3,
}


with h5py.File("../../Data/Tests_data/3d-mnist/full_dataset_vectors.h5", "r") as hf:
    print(hf)   
    X_train = hf["X_train"][:]
    y_train = hf["y_train"][:]  
    X_test = hf["X_test"][:]
    y_test = hf["y_test"][:]

#Reshaping the 3D mnist
X_train = np.reshape(X_train, (X_train.shape[0], 16, 16, 16))
X_test = np.reshape(X_test, (X_test.shape[0], 16, 16, 16))
assert X_train.shape == (X_train.shape[0], 16, 16, 16), f"X_train's shape is {X_train.shape} != ({X_train.shape[0]}, 16, 16, 16)"

#Testing the transformations
dataset = [X_train[100], X_train[200]]
# dataset = np.reshape(dataset, (16, 16, 16, 1))
output_size = (16, 16, 16, 1)
transformation = Transformations3D(10, 0, output_size)

transformation.random_transform(dataset)
new_dataset = transformation.getTransformedPair()[0]
transformation_report = transformation.getTransformedPair()[1]

#display results
disp3D = Image3D()

# disp3D.make_mesh(dataset[0], 0)
# disp3D.show_mesh()

# disp3D.make_mesh(new_dataset[0], 0)
# disp3D.show_mesh()

fig = plt.figure()

# axes0 = fig.add_subplot(1,2,1, projection='3d')
# fig2.axes.append(axes0)
# fig2.add_axes(axes0)

axes0 = fig.add_subplot(1,2,1, projection='3d')
axes0 = disp3D.getMesh(dataset[0], 0)

axes1 = fig.add_subplot(1,2,2, projection='3d')
axes1 = disp3D.getMesh(new_dataset[0], 0)

plt.show()

# fig, axes = plt.subplots(1,2, projection='3d')
# axes[0] = disp3D.getMesh(dataset[0])
# axes[1] = disp3D.getMesh(new_dataset[0])
# plt.show()


