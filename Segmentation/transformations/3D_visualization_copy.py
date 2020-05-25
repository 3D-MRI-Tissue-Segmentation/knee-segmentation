#This script has just commented added things that might be useful in the future 
#but the clean visualization class can be found in the 3D_visualization.py script
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import h5py

class Image3D():
    def __init__(self, image_3d=None):
        self.image = np.asarray(image_3d)
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        self.dims = self.image.shape
        self.voxels_arr = np.zeros((self.dims[1], self.dims[2], self.dims[0]), dtype=bool)

    def showEmptyFig(self):
        self.ax.voxels(self.voxels_arr)
        plt.show()

    def make_mash(self):
        # x = range(0, self.dims[1])
        # y = range(0, self.dims[2])
        # z = range(0, self.dims[0])
        # X, Y, Z = np.meshgrid(x, y, z)
        # assert Y.shape == X.shape
        # assert Z.shape == X.shape
        # assert self.voxels_arr.shape == X.shape

        for xx in range(0, self.dims[0]):
            for yy in range(0, self.dims[1]):
                for zz in range(0, self.dims[2]):
                    if self.image[xx, yy, zz] != 0:
                        self.voxels_arr[yy, zz, xx] = 1



        # x = range(0, self.dims[0])
        # y = range(0, self.dims[2])
        # X, Y = np.meshgrid(x, y)
        # Z = np.zeros(np.shape(X))
        # for i in range(0, np.shape(X)[0]):
        #     for j in range(0, np.shape(X)[1]):
        #         for z in range(0, self.dims[1]):
        #             if self.image[i, z, j] != 0:
        #                 Z[i, j] = z
        # self.ax.plot_surface(X, Y, Z)

        # X, Y, Z, val = list(), list(), list(), list()
        # for x in range(0, self.dims[0]):
        #     for y in range(0, self.dims[1]):
        #         for z in range(0, self.dims[2]):
        #             if self.image[x, y, z] != 0:
        #                 X.append(x)
        #                 Y.append(y)
        #                 Z.append(z)
        #                 val.append(self.image[x, y, z])
        # self.ax.scatter(Y, Z, X, val, s=40, marker='s')

        # X, Y = np.meshgrid(X, Y)
        # Z = Z , np.ones(np.shape(Z))
        # print(np.shape(Z))
        # self.ax.plot_surface(X, Y, Z)

        # x, y, z = np.indices(self.dims)
        # self.ax.voxels(x, y, z, self.image[x, y, z])

        # x, y, z = np.indices(self.dims)
        # self.ax.scatter(x, y, z, self.image[x, y, z])

        # self.fig2 = go.Figure(data=[go.Mesh3d(x=X, y=Z, z=Y, color='lightpink', opacity=0.50)])



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

print(y_train[8000])

#Representing the 3D dataset
mash = Image3D(X_train[8000])
mash.make_mash()
mash.showEmptyFig()

#understand meshgrid
# nx, ny = (3, 2)
# x = np.linspace(0, 1, nx)
# y = np.linspace(0, 1, ny)
# xv, yv = np.meshgrid(x, y)
# print (x)
# print(xv)