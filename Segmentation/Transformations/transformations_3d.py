import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py


class Transformations():
    def __init__(self, dataset, tetax, tetay, tetaz):
        self.dataset = np.asarray(dataset)
        self.teta = [tetax, tetay, tetaz]
    
    def rotations():
        pass

class Image3D():
    def __init__(self, image_3d):
        self.image = image_3d

    def make_mash():
        pass

    def show():
        plt.subplot(1, 2, 1, projection='3d')



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
