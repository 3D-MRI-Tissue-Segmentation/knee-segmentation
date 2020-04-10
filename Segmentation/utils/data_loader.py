import h5py
import numpy as np
import os.path
import random
import matplotlib.pyplot as plt
import math
import tensorflow as tf

#TODO(Joonsu): Integrate the function with Tensorflow Dataset 
def create_OAI_dataset(data_folder, get_train=True, start=1, end=61, get_slices=True, save=False):
    
    #img_list = []
    seg_list = []
    
    for i in range(start,end):
        for j in range(2):
            if i <= 9:
                if get_train:
                    fname_img = 'train_00{}_V0{}.im'.format(i, j)
                    fname_seg = 'train_00{}_V0{}.seg'.format(i, j)
                else:
                    fname_img = 'valid_00{}_V0{}.im'.format(i, j)
                    fname_seg = 'valid_00{}_V0{}.seg'.format(i, j)
            else:
                if get_train:
                    fname_img = 'train_0{}_V0{}.im'.format(i, j)
                    fname_seg = 'train_0{}_V0{}.seg'.format(i, j)
                else:
                    fname_img = 'valid_0{}_V0{}.im'.format(i, j)
                    fname_seg = 'valid_0{}_V0{}.seg'.format(i, j)
            
            img_filepath = os.path.join(data_folder, fname_img)
            seg_filepath = os.path.join(data_folder, fname_seg)

            with h5py.File(img_filepath,'r') as hf:
                img = np.array(hf['data'])
            with h5py.File(seg_filepath,'r') as hf:
                seg = np.array(hf['data'])
            
            if get_slices:
                img = np.rollaxis(img, 2, 0)
                seg = np.rollaxis(seg, 2, 0) 
            
            seg = seg[:,48:336,48:336,:]
            seg_temp = np.zeros((160,288,288,1),dtype=np.int8)
            seg_sum = np.sum(seg, axis=3)
            seg_temp[seg_sum == 0] = 1
            seg = np.concatenate([seg,seg_temp], axis=3)

            img = np.expand_dims(img, axis=3)    
            img_list.append(img)
            seg_list.append(seg)

        print('{} out of {} datasets have been processed'.format(i, end-start))

    x = np.asarray(img_list)
    y = np.asarray(seg_list)

    if get_slices:
        x = np.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
        y = np.reshape(y, (y.shape[0]*y.shape[1], y.shape[2], y.shape[3], y.shape[4]))

    if save:
        if get_train:
            fname_img_npy = os.path.join(data_folder, 'x_train.npy')
            fname_seg_npy = os.path.join(data_folder, 'y_train.npy')
        else:
            fname_img_npy = os.path.join(data_folder, 'x_valid.npy')
            fname_seg_npy = os.path.join(data_folder, 'y_valid.npy')

        np.save(fname_img_npy, x)
        np.save(fname_seg_npy, y)

    return x,y

def get_slices(path_in, path_out, extension):
    
    image_3d = os.listdir(path_in)
    idx = 1
    file_list = []

    for image in image_3d:
        if image.endswith(extension):
            file_list.append(image)
    
    for image in file_list:
        # create training samples
        img_path = os.path.join(path_in, str(image))
        with h5py.File(img_path, 'r') as hf:
            img = np.array(hf['data'])

        img_shape = img.shape

        for channel in range(img_shape[2]):
            if len(img_shape) == 3:
                img_slice = img[:,:,channel]
                img_slice = np.expand_dims(img_slice,axis=2)
            elif len(img_shape) == 4:
                img_slice = img[:,:,channel,:]
            
            img_slice = img_slice[48:336,48:336,:]

            name_out = "img_" + str(idx)
            save_samples = os.path.join(path_out, name_out)
            np.save(save_samples, img_slice)
            idx += 1

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                x_set, 
                y_set,
                batch_size=4, 
                shuffle=True, 
                multi_class=True):

        self.x_set = x_set
        self.y_set = y_set
        self.x = os.listdir(x_set) 
        self.y = os.listdir(y_set)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.multi_class = multi_class
        self.idx_list = np.arange(start=1, stop=len(self.x)+1)
        
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        
        indexes = self.idx_list[idx*self.batch_size:(idx+1)*self.batch_size]
        # generate data
        X, y = self.data_generator(indexes)

        return X, y

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.idx_list)

    def data_generator(self, indexes):
        
        # Initialization
        X = np.empty((self.batch_size, 288, 288, 1))
        
        if self.multi_class:
            Y = np.empty((self.batch_size, 288,288, 7))
        else:
            Y = np.empty((self.batch_size, 288,288, 1))

        for i, idx in enumerate(indexes):

            img = np.load(self.x_set + 'img_' + str(idx) + '.npy')
            X[i,:] = img

            seg = np.load(self.y_set + 'img_' + str(idx) + '.npy')

            if self.multi_class:
                seg = self._get_multiclass(seg)
            else:
                seg = np.sum(seg, axis=2)
                seg = np.expand_dims(seg, axis=2)
                seg[seg != 0] = 1

            Y[i,:] = seg

        return X, Y

    def _get_multiclass(self, label):
        #label shape
        #(height,width,channels)
        
        height = label.shape[0]
        width = label.shape[1]
        channels = label.shape[2]

        background = np.zeros((height, width, 1))
        label_sum = np.sum(label, axis=2)
        background[label_sum == 0] = 1
                    
        label = np.concatenate((label, background), axis = 2)
        
        return label 

def get_multiclass(label):

    #label shape
    #(batch_size, height,width,channels)
    
    batch_size = label.shape[0]
    height = label.shape[1]
    width = label.shape[2]
    channels = label.shape[3]

    background = np.zeros((batch_size, height, width, 1))
    label_sum = np.sum(label, axis=3)
    background[label_sum == 0] = 1
                
    label = np.concatenate((label, background), axis = 3)
    
    return label 

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))        

def write_to_tfrecord(data_path,tfrecords_filename):
    
    folders = os.listdir(data_path)

    sample_idx = folders.index("samples")
    samples_path = data_path + str(folders[sample_idx])

    labels_idx = folders.index("labels")
    labels_path = data_path + str(folders[labels_idx]) 

    samples_in = os.listdir(samples_path)
    labels_in = os.listdir(labels_path) 

    print(len(samples_in))
    filename_pairs = [(samples_in[i], labels_in[i]) for i in range(0, len(samples_in))] 

    with tf.io.TFRecordWriter(tfrecords_filename) as writer:
        for img_path, labels_path in filename_pairs:
            
            img = np.load(data_path + 'samples' + '/' + img_path)
            label = np.load(data_path + 'labels' + '/' + labels_path)
            
            height = img.shape[0]
            width = img.shape[1]

            img_raw = img.tostring()
            label_raw = label.tostring()

            feature = {
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'image_raw': _bytes_feature(img_raw),
                'label_raw': _bytes_feature(label_raw)
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def read_tfrecord(tfrecords_filename):

    raw_image_dataset = tf.data.TFRecordDataset(tfrecords_filename)

    features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, features)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    for image_features in parsed_image_dataset:
        image_raw = tf.io.decode_raw(image_features['image_raw'],tf.float32).numpy()
        image_raw = np.reshape(image_raw, (image_features['height'], image_features['width']))
        plt.imshow(image_raw)
        plt.show()

    return raw_image_dataset

def dataset_generator(data_path, batch_size, shuffle=False):
    
    sample_path = os.path.join(data_path, 'samples')
    label_path = os.path.join(data_path, 'labels')
    num_files = len(os.listdir(sample_path))

    def generator():
        
        for i in range(num_files):
            
            sample = np.load(sample_path + '/' + 'img_' + str(i+1) + '.npy')
            label = np.load(label_path + '/' + 'img_' + str(i+1) + '.npy')

            yield sample, label

    dataset = tf.data.Dataset.from_generator(generator, (tf.float64, tf.int8))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000).repeat()
    else:
        dataset = dataset.repeat()
    
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

