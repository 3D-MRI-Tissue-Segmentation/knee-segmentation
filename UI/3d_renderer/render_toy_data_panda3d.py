# Renders generate toy data output in Panda3d from random voxel generated data,
# which is a 4D numpy array of 3 spatial dimensions and 1 color

# Rought outline
# - constrain viewing region to volume_array size + some margin
# - add a node at each voxel location and store the color
# - Turn every node into a voxel with that color on each face 


from direct.showbase.ShowBase import ShowBase
from direct.task.Task import Task
from math import pi, sin, cos
import numpy as np
from random import randint
from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import *
from direct.interval.IntervalGlobal import *
from panda3d.core import lookAt
from panda3d.core import GeomVertexFormat, GeomVertexData
from panda3d.core import Geom, GeomTriangles, GeomVertexWriter
from panda3d.core import Texture, GeomNode
from panda3d.core import PerspectiveLens
from panda3d.core import CardMaker
from panda3d.core import Light, Spotlight
from panda3d.core import TextNode
from panda3d.core import LVector3
import sys
import os




##########################################################################################################
# Joe's toy_volume_gen.py script below so i can use the volume for trial
##########################################################################################################

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








base = ShowBase()
spin_rate = 400     # If simulation is spinning, higher num = slower; go to create_voxel to control rotation
def initializeSim():   
    """ Sets world settings like text, cam position etc. """
    base.disableMouse()
    base.camera.setPos(0, -500, 0)

# You can't normalize inline so this is a helper function
def normalized(*args):
    myVec = LVector3(*args)
    myVec.normalize()
    return myVec

def makeSquare(x1, y1, z1, x2, y2, z2, color):       # This is straight copied off the Panda3d 'procedural cube' example https://github.com/panda3d/panda3d/blob/master/samples/procedural-cube/main.py#L11
    
    format = GeomVertexFormat.getV3n3cpt2()
    vdata = GeomVertexData('square', format, Geom.UHDynamic)

    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')
    color_setter = GeomVertexWriter(vdata, 'color')
    texcoord = GeomVertexWriter(vdata, 'texcoord')

    # make sure we draw the sqaure in the right plane
    if x1 != x2:
        vertex.addData3(x1, y1, z1)
        vertex.addData3(x2, y1, z1)
        vertex.addData3(x2, y2, z2)
        vertex.addData3(x1, y2, z2)

        normal.addData3(normalized(2 * x1 - 1, 2 * y1 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y1 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y2 - 1, 2 * z2 - 1))
        normal.addData3(normalized(2 * x1 - 1, 2 * y2 - 1, 2 * z2 - 1))

    else:
        vertex.addData3(x1, y1, z1)
        vertex.addData3(x2, y2, z1)
        vertex.addData3(x2, y2, z2)
        vertex.addData3(x1, y1, z2)

        normal.addData3(normalized(2 * x1 - 1, 2 * y1 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y2 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y2 - 1, 2 * z2 - 1))
        normal.addData3(normalized(2 * x1 - 1, 2 * y1 - 1, 2 * z2 - 1))

    # adding different colors to the vertex for visibility
    if np.sum(color) > 0:
        a = 1  # Alpha transparency
    else: a = 0

    # Handling channel size 1 or 3
    if np.size(color) == 1:     
        for i in range(4):      # Have to run this 4 times cuz each vertex on the square has a color setting
            color_setter.addData2f(color,a)
    elif np.size(color) == 3:
        r, g, b = color[0], color[1], color[2]
        for i in range(4): 
            color_setter.addData4f(r,g,b,a)


    texcoord.addData2f(0.0, 1.0)
    texcoord.addData2f(0.0, 0.0)
    texcoord.addData2f(1.0, 0.0)
    texcoord.addData2f(1.0, 1.0)

    # Quads aren't directly supported by the Geom interface
    # you might be interested in the CardMaker class if you are
    # interested in rectangle though
    tris = GeomTriangles(Geom.UHDynamic)
    tris.addVertices(0, 1, 3)
    tris.addVertices(1, 2, 3)

    square = Geom(vdata)
    square.addPrimitive(tris)
    return square


def getFacesNode(x,y,z, color): 
    """ Returns node with 6 cube face sides """
    node_name = 'faces' + str(x+y+z)
    faces_node = GeomNode(node_name)

    # Note: it isn't particularly efficient to make every face as a separate Geom.
    # instead, it would be better to create one Geom holding all of the faces.
    # Since xyz are centers of the voxel, getting the vertices means we treat the xyz's as the origin

    x,y,z = 2*x, 2*y, 2*z
    square0 = makeSquare(x-1, y-1, z-1, x+1, y-1, z+1, color)
    square1 = makeSquare(x-1, y+1, z-1, x+1, y+1, z+1, color)
    square2 = makeSquare(x-1, y+1, z+1, x+1, y-1, z+1, color)
    square3 = makeSquare(x-1, y+1, z-1, x+1, y-1, z-1, color)
    square4 = makeSquare(x-1, y-1, z-1, x-1, y+1, z+1, color)
    square5 = makeSquare(x+1, y-1, z-1, x+1, y+1, z+1, color)

    faces_node.addGeom(square0)
    faces_node.addGeom(square1)
    faces_node.addGeom(square2)
    faces_node.addGeom(square3)
    faces_node.addGeom(square4)
    faces_node.addGeom(square5)

    return faces_node


def create_voxel(x,y,z, color):    # Where xyz are the center of the voxel
    """ Creates and renders voxel """
    # All you have to do is get all 6 faces of the voxel and attach them to a node

    if np.sum(color) > 0:
        faces_node = getFacesNode(x,y,z, color)    # returns Geoms 6 squares
        cube = render.attachNewNode(faces_node)

        # OpenGl by default only draws "front faces" (polygons whose vertices are
        # specified CCW).
        cube.setTwoSided(True)

        # cube.hprInterval(spin_rate, (360, 360, 360)).loop()         # This rotates each cube lol




# Having the program generate all voxels as cubes makes it too bulky, 
# only going to generate the squares on surface of the geometry. 
def getSurfaceVoxels(volume):
    """ Returns the voxels on surface of shapes """
    volume_surface = np.zeros(np.shape(volume)) 
    width, height, depth, channels = np.shape(volume)

    in_shape = False    # Either 1 in or 0 out of a shape
    last_color = volume[0,0,0,:]        # Randomly initializing last_color
    # Pass along z dimension from start (bottom-top)
    for x in range(width):
        for y in range(height):
            for z in range(depth): 
                curr_voxel = volume[x,y,z,:]    # Color of current voxel
                past_state = in_shape

                if np.any(curr_voxel >0):       # Check if in shape
                    in_shape = True
                    last_color = curr_voxel
                elif np.sum(curr_voxel) == 0: in_shape = False

                
                if in_shape and (not past_state):       # Empty --> shape
                    volume_surface[x,y,z,:] = curr_voxel
                    last_color = curr_voxel
                    
                if (not in_shape) and past_state:       # Shape --> empty
                    volume_surface[x,y,z-1,:] = last_color

                if past_state and (np.all(last_color) != np.all(curr_voxel)):      # Shape --> shape
                    in_shape = True
                    volume_surface[x,y,z,:] = curr_voxel
                    last_color = curr_voxel


    in_shape = 0
    # Pass along y dimension from start (bottom-top)
    for z in range(depth):
        for x in range(width):
            for y in range(height): 
                curr_voxel = volume[x,y,z,:]    # Color of current voxel
                past_state = in_shape

                if np.any(curr_voxel >0):       # Check if in shape
                    in_shape = True
                    last_color = curr_voxel
                elif np.sum(curr_voxel) == 0: in_shape = False

                
                if in_shape and (not past_state):       # Empty --> shape
                    volume_surface[x,y,z,:] = curr_voxel
                    last_color = curr_voxel
                    
                if (not in_shape) and past_state:       # Shape --> empty
                    volume_surface[x,y-1,z,:] = last_color

                if past_state and (np.all(last_color) != np.all(curr_voxel)):      # Shape --> shape
                    in_shape = True
                    volume_surface[x,y,z,:] = curr_voxel
                    last_color = curr_voxel
    
    in_shape = 0
    # Pass along x dimension from start (bottom-top)
    for y in range(height):
        for z in range(depth):
            for x in range(width): 
                curr_voxel = volume[x,y,z,:]    # Color of current voxel
                past_state = in_shape

                if np.any(curr_voxel >0):       # Check if in shape
                    in_shape = True
                    last_color = curr_voxel
                elif np.sum(curr_voxel) == 0: in_shape = False

                
                if in_shape and (not past_state):       # Empty --> shape
                    volume_surface[x,y,z,:] = curr_voxel
                    last_color = curr_voxel
                    
                if (not in_shape) and past_state:       # Shape --> empty
                    volume_surface[x-1,y,z,:] = last_color

                if past_state and (np.all(last_color) != np.all(curr_voxel)):      # Shape --> shape
                    in_shape = True
                    volume_surface[x,y,z,:] = curr_voxel
                    last_color = curr_voxel

    return volume_surface



if __name__ == "__main__":
    initializeSim()
    volume = td.volume          # Input 4D array of xyz coords and corresponding color
    width, height, depth, channels = np.shape(volume)
    volume_surface = getSurfaceVoxels(volume)

    # Procedural voxel generation
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                color = volume_surface[x,y,z,:]
                np.reshape(color, channels)
                create_voxel(x,y,z,color)

    base.run()
