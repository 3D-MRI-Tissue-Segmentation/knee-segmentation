import tensorflow as tf
import numpy as np
from Segmentation.model.unet import UNet
from Segmentation.model.vnet import VNet
from Segmentation.utils.data_loader import create_OAI_dataset
import matplotlib.pyplot as plt

dataset_size = 10
batch_size = 5
input_shape = (256, 256, 1)
num_classes = 3
output_shape = (256, 256, num_classes)

x, y = create_OAI_dataset('./Data/train', get_slices=True)
print(x.shape)
print(y.shape)

dataset_size = len(x)
batch_size = 1
input_shape = (384, 384, 160, 1)
num_classes = 6
output_shape = (384,384, 160, num_classes)

num_filters = 64 #number of filters at the start

features = tf.random.normal((dataset_size,) + input_shape)
labels = tf.random.normal((dataset_size,) + output_shape)
model = UNet(num_filters, num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['acc'])
model.fit(x,y,validation_split=0.2,epochs=100,batch_size=batch_size)