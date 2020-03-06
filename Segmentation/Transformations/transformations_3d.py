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
                self.pair[render_counter, :, :, z] = cv2.warpAffine(render[:,:,z], M, (self.dims[1], self.dims[0]))
    
    def YZ_rotation(self, scale=1.0):
        teta = random.randint(-self.angle, self.angle)
        self.report['YZ_rotation'] = teta

        M = cv2.getRotationMatrix2D(self.center[1], teta, scale)
        for render_counter, render in enumerate(self.pair):
            for x in range(0, self.dims[1]):
                self.pair[render_counter, :, x, :] = cv2.warpAffine(render[:,x,:], M, (self.dims[2], self.dims[0]))
    
    def XZ_rotation(self, scale=1.0):
        teta = random.randint(-self.angle, self.angle)
        self.report['XZ_rotation'] = teta

        M = cv2.getRotationMatrix2D(self.center[2], teta, scale)
        for render_counter, render in enumerate(self.pair):
            for y in range(0, self.dims[0]):
                self.pair[render_counter, y, :, :] = cv2.warpAffine(render[y,:,:], M, (self.dims[2], self.dims[1]))

    def carvingTranslation(self):
        move = [0, 0, 0]
        if(self.translation != 0):
            random_range = random.choice(((-self.translation,-1), (1,self.translation)))
            which = random.randint(0, 2)
            move[which] = random.randint(random_range[0], random_range[1])

        # move[0] = ranrandom.randint(-self.translation, self.translation) #translation in the x
        # move[1] = ranrandom.randint(-self.translation, self.translation) #translation in the y
        # move[2] = ranrandom.randint(-self.translation, self.translation) #translation in the z

        self.report['X_translation'] = move[1]
        self.report['Y_translation'] = move[0]
        self.report['Z_translation'] = move[2]

        row_low = int(self.dims[0]/2 - self.output_size[0]/2 + move[0])
        column_low = int(self.dims[1]/2 - self.output_size[1]/2 + move[1])
        z_low = int(self.dims[2]/2 - self.output_size[2]/2 + move[2])

        row_up = row_low + self.output_size[0]
        column_up = column_low + self.output_size[1]
        z_up = z_low + self.output_size[2]

        for render_counter, render in enumerate(self.pair):
            self.transformed_pair[render_counter] = render[row_low:row_up, column_low:column_up, z_low:z_up]

    def random_transform(self, pair):
        self.pair = np.asarray(pair)
        self.dims = self.pair[0].shape

        assert self.translation <= (min(self.dims) - self.output_size[0])/2, f"You cannot translate by more than the available pixels"

        self.center[0] = (int(math.floor(self.dims[1]/2)), int(math.floor(self.dims[0]/2))) #center for frontal slice
        self.center[1] = (int(math.floor(self.dims[2]/2)), int(math.floor(self.dims[0]/2))) #center for lateral slice
        self.center[2] = (int(math.floor(self.dims[1]/2)), int(math.floor(self.dims[2]/2))) #center for top slice

        self.XY_rotation()
        self.YZ_rotation()
        self.XZ_rotation()
        self.carvingTranslation()

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

if __name__=="__main__":
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

    #cube
    cube = np.zeros((150,160,160))
    cube[70:90,70:90,70:90] = 1

    #Testing the transformations
    dataset = [cube, cube]
    output_size = (40, 40, 40, 1)

    transformation = Transformations3D(10, 2, output_size)
    transformation.random_transform(dataset)
    new_dataset = transformation.getTransformedPair()[0]
    transformation_report = transformation.getTransformedPair()[1]

    print(transformation_report)

    #display results
    disp3D = Image3D()
    disp3D.make_mesh(new_dataset[1], 0)
    disp3D.show_mesh()
