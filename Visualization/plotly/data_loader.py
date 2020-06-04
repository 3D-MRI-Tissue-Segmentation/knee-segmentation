
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

        for root, dir, fnames in sorted(os.walk(opt.dataroot)):
            for fname in fnames:
                if is_acceptable(fname):
                    data_path = os.path.join(root,fname)
                    data_paths.append(data_path)
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
        colour_channels = 3

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
        
    else:
        assert data_paths, 'The directory %s does not contain files with valid extensions %s' % (opt.dataroot, EXTENSIONS)
        print("data_paths", data_paths)
        data = np.load(data_paths)
        

    return data

