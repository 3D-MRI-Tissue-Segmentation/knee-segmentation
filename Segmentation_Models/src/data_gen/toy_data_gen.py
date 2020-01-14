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
        if self.depth == 1:
            return [randint(0,255)]
        return [randint(0,255),randint(0,255),randint(0,255)]
    
    def get_empty_array(self):
        return np.zeros([self.width, self.height, self.depth], dtype=int)
    

    def set_colour_to_xy(self, x, y, colour_idx):
        if self.depth == 1:
            self.image[x][y][0] = self.class_colours[colour_idx][0]
        else:
            self.image[x][y][0] = self.class_colours[colour_idx][0]
            self.image[x][y][1] = self.class_colours[colour_idx][1]
            self.image[x][y][2] = self.class_colours[colour_idx][2]
        
    def set_color_to_random_xy(self, colour_idx):
        x = randint(0, self.width-1)
        y = randint(0, self.height-1)
        self.set_colour_to_xy(x, y, colour_idx)
    

if __name__ == "__main__":
    n_reps = 10
    n_classes = 100
    td = Toy_Image(n_classes, 40, 40, 3)

    for rep in range(n_reps):
        for i in range(n_classes):
            td.set_color_to_random_xy(i)

    import matplotlib.pyplot as plt
    plt.imshow(td.image, cmap='jet')
    plt.show()
    