
##########################################################################################################
# Joe's toy_volume_gen.py script below so i can use the volume for trial
##########################################################################################################

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

    def set_rect_cuboid_to_xyz(self, x, y, z, 
                               x_length, y_length, z_length, 
                               colour_idx):
        (x_min, x_max) = self.get_axis_range(x, x_length, self.width)
        (y_min, y_max) = self.get_axis_range(y, y_length, self.height)
        (z_min, z_max) = self.get_axis_range(z, z_length, self.depth)
        for x_ in range(x_min, x_max):
            for y_ in range(y_min, y_max):
                for z_ in range(z_min, z_max):
                    self.set_colour_to_xyz(x_, y_, z_, colour_idx)                 

    def set_cube_to_xyz(self, x, y, z, length, colour_idx):
        self.set_rect_cuboid_to_xyz(x, y, z, length, length, length, colour_idx)
    
    def is_in_sphere(self, x, y, z, centre, radius):
        return self.is_in_ellipsoid(x, y, z, centre, radius, radius, radius)

    def is_in_ellipsoid(self, x, y, z, centre, x_radius, y_radius, z_radius):
        x_centre, y_centre, z_centre = centre
        if ((x_centre-x)**2)/x_radius**2 + ((y_centre-y)**2)/y_radius**2 + ((z_centre-z)**2)/z_radius**2 < 1:
            return True
        return False

    def set_sphere_to_xyz(self, x, y, z, radius, colour_idx):
        self.set_ellipsoid_to_xyz(x, y, z, radius, radius, radius, colour_idx)

    def set_ellipsoid_to_xyz(self, x, y, z, x_radius, y_radius, z_radius, colour_idx):
        (x_min, x_max) = self.get_axis_range(x, x_radius, self.width)
        (y_min, y_max) = self.get_axis_range(y, y_radius, self.height)
        (z_min, z_max) = self.get_axis_range(z, z_radius, self.depth)
        for x_ in range(x_min, x_max):
            for y_ in range(y_min, y_max):
                for z_ in range(z_min, z_max):
                    if self.is_in_ellipsoid(x_, y_, z_, (x, y, z), x_radius, y_radius, z_radius):
                        self.set_colour_to_xyz(x_, y_, z_, colour_idx)


def get_test_volumes(n_volumes, n_reps, n_classes, 
                     width, height, depth, colour_channels):
    #volumes, one_hots = [], []
    volumes, one_hots = None, None

    return volumes, one_hots

def plot_volume(volume, show=True):
    voxel = volume[:,:,:,0] > 0
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel, facecolors=volume, linewidth=0.5)
    if show:
        plt.show()


def rgb_to_hex(rgb):
    assert type(rgb) is list
    assert len(rgb) == 3
    assert all((0 <= col < 256 and type(col) is int) for col in rgb), "The colours must be an int from 0 to 255"
    return '#%02x%02x%02x' % tuple(rgb)

if __name__ == "__main__":
    n_reps, n_classes = 4, 3
    width, height, depth = 100, 100, 100
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



##########################################################################################################
# End joe's toy_volume_gen.py (copied as is)
##########################################################################################################




