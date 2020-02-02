import h5py
import numpy as np
import os.path

def create_OAI_dataset(data_folder, get_slices=True, save=False):
    
    img_list = []
    seg_list = []

    for i in range(1,61):
        for j in range(2):
            
            if i <= 9:
                fname_img = 'train_00{}_V0{}.im'.format(i, j)
                fname_seg = 'train_00{}_V0{}.seg'.format(i, j)
            else:
                fname_img = 'train_0{}_V0{}.im'.format(i, j)
                fname_seg = 'train_0{}_V0{}.seg'.format(i, j)
            
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

        print('%d out of 60 datasets have been processed' %i)

    x = np.asarray(img_list)
    y = np.asarray(seg_list)

    if get_slices:
        x = np.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
        y = np.reshape(y, (y.shape[0]*y.shape[1], y.shape[2], y.shape[3], y.shape[4]))

    if save:
        fname_img_npy = os.path.join(data_folder, 'x_train.npy')
        np.save(fname_img_npy, x)
        
        fname_seg_npy = os.path.join(data_folder, 'y_train.npy')
        np.save(fname_seg_npy, y)

    return x, y

