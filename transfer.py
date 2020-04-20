import tensorflow as tf
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.layers import Input 

input_tensor = Input(shape=(288,288,1)) 
vgg_model= VGG16(weights='imagenet', include_top=False)
vgg_model.summary()

