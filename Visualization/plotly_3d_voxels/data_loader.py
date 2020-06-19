
# Load numpy data etc.
import os
import os.path
import numpy as np
from toy_volume_gen_class import Toy_Volume
from random import randint


EXTENSIONS = ['.npy', '.NPY']

def is_acceptable(filename):
    return any(filename.endswith(extension) for extension in EXTENSIONS)

def load_data(opt):
    data_paths = []
    data = []

    # Read in all numpy arrays in curr dir unless 'filename' was specified
    if not opt.file_name:         # if no filename given
        assert os.path.isdir(opt.dataroot), '%s is not a valid directory' % opt.dataroot

        if not opt.dataroot_left or not opt.dataroot_right:

            for root, dir, fnames in sorted(os.walk(opt.dataroot)):
                for fname in fnames:
                    if is_acceptable(fname):
                        data_path = os.path.join(root,fname)
                        data_paths.append(data_path)
        else:

            assert os.path.isdir(opt.dataroot_left), '%s is not a valid directory' % opt.dataroot_left
            assert os.path.isdir(opt.dataroot_right), '%s is not a valid directory' % opt.dataroot_right

            data_paths_left = []
            data_paths_right = []
            for root, dir, fnames in sorted(os.walk(opt.dataroot_left)):
                for fname in fnames:
                    if is_acceptable(fname):
                        data_path = os.path.join(root,fname)
                        data_paths_left.append(data_path)
            for root, dir, fnames in sorted(os.walk(opt.dataroot_right)):
                for fname in fnames:
                    if is_acceptable(fname):
                        data_path = os.path.join(root,fname)
                        data_paths_right.append(data_path)
    else: 
        data_paths = opt.file_name
        
    # Make toy dataset if no files found or opt set
    if opt.toy_dataset:          
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
        
    else:
        if not opt.dataroot_left or not opt.dataroot_right:
            assert data_paths, 'The directory %s may not contain files with valid extensions %s' % (opt.dataroot, EXTENSIONS)        
            data = []
            if len(data_paths) == 1:
                data = np.load(data_paths)
            else:
                for i, path in enumerate(data_paths):
                    should_get_loaded = np.mod(i,opt.slider_interval) == 0
                    if should_get_loaded or i==0:
                        data.append(np.load(path))

            return data


        else:
            assert data_paths_left, 'The directory %s may not contain files with valid extensions %s' % (opt.data_paths_left, EXTENSIONS)        
            assert data_paths_right, 'The directory %s may not contain files with valid extensions %s' % (opt.data_paths_right, EXTENSIONS)        
            data_left = []
            data_right = []
            if len(data_paths_left) == 1:
                data_left = np.load(data_paths_left)
            else:
                for i, path in enumerate(data_paths_left):
                    should_get_loaded = np.mod(i,opt.slider_interval) == 0
                    if should_get_loaded or i==0:
                        data_left.append(np.load(path))
            if len(data_paths_right) == 1:
                data_right = np.load(data_paths_right)
            else:
                for i, path in enumerate(data_paths_right):
                    should_get_loaded = np.mod(i,opt.slider_interval) == 0
                    if should_get_loaded or i==0:
                        data_right.append(np.load(path))
            
            return data_left, data_right

    

