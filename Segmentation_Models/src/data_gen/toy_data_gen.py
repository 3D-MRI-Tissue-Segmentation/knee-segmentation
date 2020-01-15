import numpy as np
from random import randint

class Toy_Image:
    def __init__(self, n_classes, width, height, depth=3):
        self.init_check(n_classes, width, height, depth)
        self.n_classes = n_classes
        self.width = width
        self.height = height
        self.depth = depth
        self.class_colours = self.get_class_colours()
        self.image = self.get_empty_array()


    def init_check(self, n_classes, width, height, depth):
        assert type(n_classes) is int, "n_classes must be of type int"
        assert n_classes > 0, "Need at least one class"
        assert width > 0, "Need postive width"
        assert height > 0, "Need positive height"
        assert (depth == 3) or (depth == 1), "Either RGB or grayscale"
    

    def get_class_colours(self):
        """ Generates random colours to be visualised with and returns the list """
        classes = []
        for class_idx in range(self.n_classes):
            count = 0
            valid = False
            while( not valid ):
                colour = self.get_random_colour()
                if colour not in classes:
                    classes.append(colour)
                    valid = True
        return classes
        

    def get_random_colour(self):
        """ Returns a random colour """
        if self.depth == 1:
            return [randint(0,255)]
        return [randint(0,255),randint(0,255),randint(0,255)]
    
    def get_empty_array(self):
        """ Empty starting array """
        return np.zeros([self.width, self.height, self.depth], dtype=int)
    
    def get_random_xy(self):
        x = randint(0, self.width-1)
        y = randint(0, self.height-1)
        return x, y

    def set_colour_to_xy(self, x, y, colour_idx):
        """ Sets the colour for a specific pixel """
        if self.depth == 1:
            self.image[x][y][0] = self.class_colours[colour_idx][0]
        else:
            self.image[x][y][0] = self.class_colours[colour_idx][0]
            self.image[x][y][1] = self.class_colours[colour_idx][1]
            self.image[x][y][2] = self.class_colours[colour_idx][2]
    
    
    def get_shape_square_range(self, x, y, length):
        assert type(length) is int, "length must be an int, it should be half the width of the object"
        (x_min, x_max) = self.get_axis_range(x, length)
        (y_min, y_max) = self.get_axis_range(y, length)
        return (x_min, x_max), (y_min, y_max)

    def get_axis_range(self, axis_pos, axis_length):
        inputs = (axis_pos, axis_length)
        (axis_min, axis_max) = (self.get_shape_range_min(*inputs), self.get_shape_range_max(*inputs))
        return (axis_min, axis_max)

    def get_shape_range_min(self, axis_pos, length):
        assert type(length) is int, "length must be an int"
        temp_min = axis_pos - length 
        range_min = temp_min if temp_min > 0 else 0
        return range_min

    def get_shape_range_max(self, axis_pos, length):
        assert type(length) is int, "length must be an int"
        temp_max = axis_pos + length 
        range_max = temp_max if temp_max < (self.width - 1) else self.width
        return range_max

    def set_rect_to_xy(self, x, y, x_length, y_length, colour_idx):
        (x_min, x_max) = self.get_axis_range(x, x_length)
        (y_min, y_max) = self.get_axis_range(y, y_length)
        for x_ in range(x_min, x_max):
            for y_ in range(y_min, y_max):
                self.set_colour_to_xy(x_, y_, colour_idx)

    def set_square_to_xy(self, x, y, length, colour_idx):
        self.set_rect_to_xy(x, y, length, length, colour_idx)

    def is_in_circle(self, x, y, centre, radius):
        return self.is_in_oval(x, y, centre, radius, radius)

    def is_in_oval(self, x, y, centre, x_radius, y_radius):
        x_centre, y_centre = centre
        if ((x_centre-x)**2)/x_radius**2 + ((y_centre-y)**2)/y_radius**2 < 1:
            return True
        return False

    def set_circle_to_xy(self, x, y, radius, colour_idx):
        (x_min, x_max), (y_min, y_max) = self.get_shape_square_range(x, y, radius)
        for x_ in range(x_min, x_max):
            for y_ in range(y_min, y_max):
                if self.is_in_circle(x_, y_, (x, y), radius):
                    self.set_colour_to_xy(x_, y_, colour_idx)
    
    def set_oval_to_xy(self, x, y, x_radius, y_radius, colour_idx):
        (x_min, x_max) = self.get_axis_range(x, x_radius)
        (y_min, y_max) = self.get_axis_range(y, y_radius)
        for x_ in range(x_min, x_max):
            for y_ in range(y_min, y_max):
                if self.is_in_oval(x_, y_, (x, y), x_radius, y_radius):
                    self.set_colour_to_xy(x_, y_, colour_idx)



        

if __name__ == "__main__":
    n_reps = 4
    n_classes = 3
    width, height = 400, 400
    depth = 3
    td = Toy_Image(n_classes, width, height, depth)

    for rep in range(n_reps):
        for colour_idx in range(n_classes):
            x,y = td.get_random_xy()
            rand_width = randint(1, int(td.width/8))
            rand_height = randint(1, int(td.height/8))
            
            td.set_oval_to_xy(x, y, rand_width, rand_height, colour_idx)
            # rnd_i = randint(0, 2)
            # if rnd_i == 0:
            #     td.set_square_to_xy(x, y, rand_width, colour_idx)
            # elif rnd_i == 1:
            #     td.set_circle_to_xy(x, y, rand_width, colour_idx)
            # elif rnd_i == 2:
            #     td.set_rect_to_xy(x, y, rand_width, rand_height, colour_idx)

    import matplotlib.pyplot as plt
    plt.imshow(td.image, cmap='jet')
    plt.show()
    