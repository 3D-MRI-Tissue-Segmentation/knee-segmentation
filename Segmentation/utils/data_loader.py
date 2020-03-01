import h5py
import numpy as np
import os.path
import random
import matplotlib.pyplot as plt
import math
import tensorflow as tf

#TODO(Joonsu): Integrate the function with Tensorflow Dataset 
def create_OAI_dataset(data_folder, get_train=True, start=1, end=61, get_slices=True, save=False):
    
    img_list = []
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

    return x, y

def train_generator(data_path, batch_size = 10, multi_class = False):
    
    folders = os.listdir(data_path)

    sample_idx = folders.index("samples")
    samples_path = data_path + str(folders[sample_idx])

    labels_idx = folders.index("labels")
    labels_path = data_path + str(folders[labels_idx]) 

    samples_in = os.listdir(samples_path)
    labels_in = os.listdir(labels_path)    
    
    # Loop forever so the generator never terminates
    while True: 

        samples = []
        labels = []

        count = 0
        
        while count < batch_size:

            rand_idx = random.randint(0, len(samples_in) - 1)

            sample = np.load(samples_path + "/" + samples_in[rand_idx])
            sample = sample[48:336,48:336]
            samples.append(sample)
            
            label = np.load(labels_path + "/" + labels_in[rand_idx])
            label = label[48:336,48:336,:]

            if multi_class == True:
                background = np.zeros((label.shape[0], label.shape[1], 1))
                for i in range(label.shape[0]):
                    for j in range(label.shape[1]):
                        sum = np.sum(label[i,j,:])
                        if sum == 0:
                            background[i][j] = 1
                        else:
                            background[i][j] = 0
                            
                label = np.concatenate((label, background), axis = 2)
                label = np.reshape(label, (label.shape[0]*label.shape[1], 7))
                labels.append(label)

            else:
                label = np.sum(label, axis = 2)
                labels.append(label)
            
            X_ = np.array(samples)
            Y_ = np.array(labels)

            X_ = np.expand_dims(X_, axis=3)
            if not multi_class:
                Y_ = np.expand_dims(Y_, axis=3)
                        
            count += 1

        yield (X_, Y_)

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
            elif len(img_shape) == 4:
                img_slice = img[:,:,channel,:]
            
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
            Y = np.empty((self.batch_size, 288*288, 7))
        else:
            Y = np.empty((self.batch_size, 288,288, 1))

        for i, idx in enumerate(indexes):

            img = np.load(self.x_set + 'img_' + str(idx) + '.npy')
            img = img[48:336,48:336]
            img = np.expand_dims(img, axis=2)
            X[i,:] = img

            seg = np.load(self.y_set + 'img_' + str(idx) + '.npy')
            seg = seg[48:336,48:336,:]

            if self.multi_class:
                seg_1d = self.get_multiclass(seg)
                Y[i,:] = seg_1d
            else:
                seg = np.sum(seg, axis=2)
                seg = np.expand_dims(seg, axis=2)
                Y[i,:] = seg
       
        return X, Y

    def get_multiclass(self, label):
        
        background = np.zeros((label.shape[0], label.shape[1], 1))
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                sum = np.sum(label[i,j,:])
                if sum == 0:
                    background[i][j] = 1
                else:
                    background[i][j] = 0
                    
        label = np.concatenate((label, background), axis = 2)
        label = np.reshape(label, (label.shape[0]*label.shape[1], label.shape[2]))
        
        return label
        