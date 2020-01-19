import tensorflow as tf
from Segmentation.model.unet import UNet, AttentionUNet_v1

dataset_size = 10
batch_size = 5
input_shape = (256, 256, 1)
num_classes = 3
output_shape = (256,256, num_classes)
num_filters = 64 #number of filters at the start

features = tf.random.normal((dataset_size,) + input_shape)
labels = tf.random.normal((dataset_size,) + output_shape)
#model = UNet(num_filters, num_classes)
#y = model(features)
#model.summary()
#print(y.shape)

attention_model = AttentionUNet_v1(num_filters, num_classes)
y2 = attention_model(features)
#model.summary()
print(y2.shape)
