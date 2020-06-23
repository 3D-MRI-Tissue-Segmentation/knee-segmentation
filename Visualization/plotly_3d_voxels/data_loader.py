
# Load numpy data etc.
import os
import os.path
import numpy as np
from toy_volume_gen_class import Toy_Volume
from random import randint


EXTENSIONS = ['.npy', '.NPY']

def is_acceptable(filename):
    return any(filename.endswith(extension) for extension in EXTENSIONS)

def make_toy_ds(opt):
    print('Making toy dataset')
    d = opt.toy_dataset
    # data = np.floor(np.random.rand(d,d,d)*2)
    # data = data > 0

    n_reps, n_classes = 4, 3
    width, height, depth = d,d,d
    colour_channels = 1

    td = Toy_Volume(n_classes, width, height, depth, colour_channels)

    for rep in range(n_reps):
        for colour_idx in range(n_classes):
            #td.set_colour_to_random_xyz(colour_idx)
            x, y, z = td.get_random_xyz()
            rand_x_len = randint(1, int(td.width/4))
            rand_y_len = randint(1, int(td.height/4))
            rand_z_len = randint(1, int(td.depth/4))
            rnd_i = randint(0, 1)
            if rnd_i == 0:
                td.set_rect_cuboid_to_xyz(x, y, z, 
                                        rand_x_len, rand_y_len, rand_z_len, 
                                        colour_idx)
            elif rnd_i == 1:
                td.set_ellipsoid_to_xyz(x, y, z,
                                        rand_x_len, rand_y_len, rand_z_len, 
                                        colour_idx)

    data = td.volume
    data = data[:,:,:,1]
    # data = data[1:2,:,:,:]
    return data

def get_paths(dataroot):
    data_paths = []
    for root, dir, fnames in sorted(os.walk(dataroot)):
        for fname in fnames:
            if is_acceptable(fname):
                data_path = os.path.join(root,fname)
                data_paths.append(data_path)
    return data_paths 

def load_data(opt):
    data = []

    # Get filenames
    # Read in all numpy arrays in curr dir unless 'filename' was specified
    if not opt.file_name:         # if no filename given
        assert os.path.isdir(opt.dataroot), '%s is not a valid directory' % opt.dataroot

        if not (opt.dataroot_left or opt.file_name_left) or not (opt.dataroot_right or opt.file_name_right):
            data_paths = get_paths(opt.dataroot)
        else:

            left = opt.dataroot_left if opt.dataroot_left else opt.file_name_left
            right = opt.dataroot_right if opt.dataroot_right else opt.file_name_right

            data_paths_left = get_paths(left) if opt.dataroot_left else opt.file_name_left
            data_paths_right = get_paths(right) if opt.dataroot_right else opt.file_name_right
            
    else: 
        data_paths = opt.file_name
        


    # Load data
    # Make toy dataset if no files found or opt set
    if opt.toy_dataset:          
        data = make_toy_ds(opt)
    else:
        if not (left and right):
            assert data_paths, 'The directory %s possibly does contain files with valid extensions %s' % (opt.dataroot, EXTENSIONS)        

            data = []
            if not (type(data_paths) == list):
                data = np.load(data_paths)
            else:
                for i, path in enumerate(data_paths):
                    should_get_loaded = np.mod(i,opt.slider_interval) == 0
                    if should_get_loaded or i==0:
                        data.append(np.load(path))

            print('Loaded data_paths', data_paths)
            return data


        else:

            assert data_paths_left, 'The directory %s may not contain files with valid extensions %s' % (data_paths_left, EXTENSIONS)        
            assert data_paths_right, 'The directory %s may not contain files with valid extensions %s' % (data_paths_right, EXTENSIONS)        
            data_left = []
            data_right = []

            if not (type(data_paths_left) == list) or opt.file_name_left:

                data_left = np.load(data_paths_left)
            else:
                for i, path in enumerate(data_paths_left):
                    should_get_loaded = np.mod(i,opt.slider_interval) == 0
                    if should_get_loaded or i==0:
                        data_left.append(np.load(path))

            if not (type(data_paths_right) == list) or opt.file_name_right:
                data_right = np.load(data_paths_right)
            else:
                for i, path in enumerate(data_paths_right):
                    should_get_loaded = np.mod(i,opt.slider_interval) == 0
                    if should_get_loaded or i==0:
                        data_right.append(np.load(path))
            
            print('Loaded data paths', data_paths_left, data_paths_right)
            
            num_in_left = len(data_paths_left) if (type(data_paths_left) == list) else 1
            num_in_right = len(data_paths_right) if (type(data_paths_right) == list) else 1
            if opt.file_name_left or opt.file_name_right:
                num_in = max(num_in_left, num_in_right)
            else:
                num_in = min(num_in_left, num_in_right)
            

            print('num_in_left',num_in_left)
            print('num_in_right',num_in_right)
            print('num_in', num_in)
            
            return data_left, data_right, num_in

    

