import tensorflow as tf
import numpy as np
from Segmentation.model.unet import UNet, AttentionUNet_v1
from Segmentation.utils.data_loader import create_OAI_dataset
import matplotlib.pyplot as plt

x, y = create_OAI_dataset('./Data/train')
print(x.shape)
print(y.shape)

dataset_size = len(x)
batch_size = 1
input_shape = (384, 384, 1)
num_classes = 6
output_shape = (384,384, num_classes)
num_filters = 64 #number of filters at the start

#features = tf.random.normal((dataset_size,) + input_shape)
#labels = tf.random.normal((dataset_size,) + output_shape)
model = UNet(num_filters, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['acc'])
model.fit(x,y,validation_split=0.2,epochs=100,batch_size=batch_size)
