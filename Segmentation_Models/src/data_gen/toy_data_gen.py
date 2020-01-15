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
        
    def set_square_to_xy(self, x, y, length, colour_idx):
        width = x + length
        height = y + length
        max_width = width if width < self.width - 1 else self.width
        max_height = height if height < self.height - 1 else self.height
        for x_ in range(x, max_width):
            for y_ in range(y, max_height):
                self.set_colour_to_xy(x_, y_, colour_idx)

        

if __name__ == "__main__":
    n_reps = 5
    n_classes = 3
    width, height = 40, 40
    depth = 3
    td = Toy_Image(n_classes, width, height, depth)

    for rep in range(n_reps):
        for colour_idx in range(n_classes):
            x,y = td.get_random_xy()
            rand_width = randint(1, int(td.width/4))
            td.set_square_to_xy(x, y, rand_width, colour_idx)

    import matplotlib.pyplot as plt
    plt.imshow(td.image, cmap='jet')
    plt.show()
    