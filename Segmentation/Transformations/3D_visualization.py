import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import h5py

class Image3D():
    def __init__(self):
        self.image = []
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        self.dims = self.image.shape
        self.voxels_arr = np.zeros((self.dims[1], self.dims[2], self.dims[0]), dtype=bool)

    def make_mesh(self, image_3d, lower_boundary=-1000, higher_boundary=1000):
        self.image = np.asarray(image_3d)

        for xx in range(0, self.dims[0]):
            for yy in range(0, self.dims[1]):
                for zz in range(0, self.dims[2]):
                    if (self.image[xx, yy, zz] > lower_boundary and self.image[xx, yy, zz] < higher_boundary):
                        self.voxels_arr[yy, zz, xx] = 1

    def getMesh(self):
        return self.ax

    def show_mesh(self):
        self.ax.voxels(self.voxels_arr)
        plt.show()


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

#Representing the 3D dataset
print(y_train[8000])
mash = Image3D(X_train[8000])
mash.make_mesh(lower_boundary=0)
mash.show_mesh()