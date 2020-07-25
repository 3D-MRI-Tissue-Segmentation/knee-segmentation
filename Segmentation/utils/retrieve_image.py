import tensorflow as tf 
from tensorflow.python.summary.summary_iterator import summary_iterator
import os
from typing import List
import imageio

img_tag = ["Whole Validation - All Slices", "", ""]
def extract_images_from_event(event_filename: str, image_tags: List[str], ouput_dir:str):
    count = 0
    for event in tf.compat.v1.train.summary_iterator(event_filename):
        count += 1
        for v in event.summary.value:
            if v.tag in image_tags:
                if v.HasField('tensor'):  # event for images using tensor field
                    s = v.tensor.string_val[2]  # first elements are W and H
                    tf_img = tf.image.decode_image(s)  # [H, W, C]
                    np_img = tf_img.numpy()
                    imageio.imwrite(ouput_dir + str(count) + ".png", np_img)
