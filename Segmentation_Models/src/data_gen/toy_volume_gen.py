import numpy as np
from random import randint

class Toy_Volume:
    def __init__(self, n_classes, width, height, depth, colour_channels=3):
        self.init_check(n_classes, width, height, depth, colour_channels)
        self.n_classes = n_classes
        self.width = width
        self.height = height
        self.depth = depth
        self.colour_channels = colour_channels
        self.class_colours = Toy_Volume.get_class_colours(n_classes, colour_channels)
        self.volume = self.get_empty_array()
        self.one_hot_array = self.get_empty_array(channels=self.n_classes)

    def init_check(self, n_classes, width, height, depth, colour_channels):
        assert type(n_classes) is int, "n_classes must be of type int"
        assert n_classes > 0, "Need at least one class"
        assert width > 0, "Need postive width"
        assert height > 0, "Need positive height"
        assert depth > 0, "Need positive depth"
        assert (colour_channels == 3) or (colour_channels == 1), "Either RGB or grayscale"

    @staticmethod
    def get_class_colours(n_classes, colour_channels):
        """ Generates random colours to be visualised with and returns the list """
        classes = []
        for class_idx in range(n_classes):
            count = 0
            valid = False
            while( not valid ):
                colour = Toy_Volume.get_random_colour(colour_channels)
                if colour not in classes:
                    classes.append(colour)
                    valid = True
        return classes
    
    @staticmethod
    def get_random_colour(colour_channels):
        """ Returns a random colour """
        if colour_channels == 1:
            return [randint(0,255)]
        return [randint(0,255)/255,randint(0,255)/255,randint(0,255)/255]
        
    def get_empty_array(self, channels=None):
        """ Empty starting array """
        if channels is None:
            channels = self.colour_channels
        return np.zeros([self.width, self.height, self.depth, channels], dtype=float)

    def get_random_xyz(self):
        x = randint(0, self.width-1)
        y = randint(0, self.height-1)
        z = randint(0, self.depth-1)
        return x, y, z

    def set_colour_to_xyz(self, x, y, z, colour_idx):
        """ Sets the colour for a specific pixel """
        if self.colour_channels == 1:
            self.volume[x][y][z][0] = self.class_colours[colour_idx][0]
        else:
            self.volume[x][y][z][0] = self.class_colours[colour_idx][0]
            self.volume[x][y][z][1] = self.class_colours[colour_idx][1]
            self.volume[x][y][z][2] = self.class_colours[colour_idx][2]
        self.one_hot_array[x][y][z][:] = 0
        self.one_hot_array[x][y][z][colour_idx] = 1

    def set_colour_to_random_xyz(self, colour_idx):
        self.set_colour_to_xyz(*self.get_random_xyz(), colour_idx)

    def get_volume_cube_range(self, x, y, z, length):
        assert type(length) is int, "length must be an int, it should be half the width of the object"
        (x_min, x_max) = self.get_axis_range(x, length, self.width)
        (y_min, y_max) = self.get_axis_range(y, length, self.height)
        (z_min, z_max) = self.get_axis_range(z, length, self.depth)
        return (x_min, x_max), (y_min, y_max), (z_min, z_max)

    def get_axis_range(self, axis_pos, axis_length, frame_length):
        inputs = (axis_pos, axis_length)
        (axis_min, axis_max) = (self.get_shape_range_min(*inputs), self.get_shape_range_max(*inputs, frame_length))
        return (axis_min, axis_max)

    def get_shape_range_min(self, axis_pos, length):
        assert type(length) is int, "length must be an int"
        temp_min = axis_pos - length 
        range_min = temp_min if temp_min > 0 else 0
        return range_min

    def get_shape_range_max(self, axis_pos, length, frame_length):
        assert type(length) is int, "length must be an int"
        temp_max = axis_pos + length 
        range_max = temp_max if temp_max < (frame_length - 1) else frame_length
        return range_max

    def set_cube_to_xyz(self, x, y, z, length, colour_idx):
        (x_min, x_max) = self.get_axis_range(x, length, self.width)
        (y_min, y_max) = self.get_axis_range(y, length, self.height)
        (z_min, z_max) = self.get_axis_range(z, length, self.depth)
        for x_ in range(x_min, x_max):
            for y_ in range(y_min, y_max):
                for z_ in range(z_min, z_max):
                    self.set_colour_to_xyz(x_, y_, z_, colour_idx)
    

def get_test_volumes(n_volumes, n_reps, n_classes, 
                     width, height, depth, colour_channels):
    #volumes, one_hots = [], []
    volumes, one_hots = None, None

    return volumes, one_hots

def plot_volume(volume):
    voxel = volume[:,:,:,0] > 0
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel, facecolors=volume, linewidth=0.5)
    plt.show()


def rgb_to_hex(rgb):
    assert type(rgb) is list
    assert len(rgb) == 3
    assert all((0 <= col < 256 and type(col) is int) for col in rgb), "The colours must be an int from 0 to 255"
    return '#%02x%02x%02x' % tuple(rgb)

if __name__ == "__main__":
    n_reps, n_classes = 2, 3
    width, height, depth = 20, 20, 20
    colour_channels = 3

    td = Toy_Volume(n_classes, width, height, depth, colour_channels)

    for rep in range(n_reps):
        for colour_idx in range(n_classes):
            #td.set_colour_to_random_xyz(colour_idx)
            x, y, z = td.get_random_xyz()
            rand_length = randint(1, int(td.width/8))
            td.set_cube_to_xyz(x, y, z, rand_length, colour_idx)

    plot_volume(td.volume)