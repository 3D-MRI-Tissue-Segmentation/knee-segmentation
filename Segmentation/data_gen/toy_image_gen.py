import numpy as np
from random import randint

class Toy_Image:
    def __init__(self, n_classes, width, height, colour_channels=3):
        self.init_check(n_classes, width, height, colour_channels)
        self.n_classes = n_classes
        self.width = width
        self.height = height
        self.colour_channels = colour_channels
        self.class_colours = Toy_Image.get_class_colours(n_classes, colour_channels)
        self.image = self.get_empty_array()
        self.one_hot_array = self.get_empty_array(channels=self.n_classes)

    def init_check(self, n_classes, width, height, colour_channels):
        assert type(n_classes) is int, "n_classes must be of type int"
        assert n_classes > 0, "Need at least one class"
        assert width > 0, "Need postive width"
        assert height > 0, "Need positive height"
        assert (colour_channels == 3) or (colour_channels == 1), "Either RGB or grayscale"

    @staticmethod
    def get_class_colours(n_classes, colour_channels):
        """ Generates random colours to be visualised with and returns the list """
        classes = []
        for class_idx in range(n_classes):
            count = 0
            valid = False
            while(not valid):
                colour = Toy_Image.get_random_colour(colour_channels)
                if colour not in classes:
                    classes.append(colour)
                    valid = True
        return classes

    @staticmethod
    def get_random_colour(colour_channels):
        """ Returns a random colour """
        if colour_channels == 1:
            return [randint(1, 255)]
        return [randint(1, 255), randint(1, 255), randint(1, 255)]

    def get_colour_from_idx(self, colour_idx):
        return self.class_colours[colour_idx]

    def get_empty_array(self, channels=None):
        """ Empty starting array """
        if channels is None:
            channels = self.colour_channels
        return np.zeros([self.width, self.height, channels], dtype=np.float32)

    def get_random_xy(self):
        x = randint(0, self.width - 1)
        y = randint(0, self.height - 1)
        return x, y

    def set_colour_to_xy(self, x, y, colour_idx):
        """ Sets the colour for a specific pixel """
        if self.colour_channels == 1:
            self.image[x, y, 0] = self.class_colours[colour_idx][0]
        else:
            self.image[x, y, 0] = self.class_colours[colour_idx][0]
            self.image[x, y, 1] = self.class_colours[colour_idx][1]
            self.image[x, y, 2] = self.class_colours[colour_idx][2]
        self.one_hot_array[x, y, :] = 0
        self.one_hot_array[x, y, colour_idx] = 1

    def set_colour_to_random_xy(self, colour_idx):
        self.set_colour_to_xy(*self.get_random_xy(), colour_idx)

    def get_shape_square_range(self, x, y, length):
        assert type(length) is int, "length must be an int, it should be half the width of the object"
        (x_min, x_max) = self.get_axis_range(x, length, self.width)
        (y_min, y_max) = self.get_axis_range(y, length, self.height)
        return (x_min, x_max), (y_min, y_max)

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

    def set_rect_to_xy(self, x, y, x_length, y_length, colour_idx):
        (x_min, x_max) = self.get_axis_range(x, x_length, self.width)
        (y_min, y_max) = self.get_axis_range(y, y_length, self.height)
        for x_ in range(x_min, x_max):
            for y_ in range(y_min, y_max):
                self.set_colour_to_xy(x_, y_, colour_idx)

    def set_square_to_xy(self, x, y, length, colour_idx):
        self.set_rect_to_xy(x, y, length, length, colour_idx)

    def is_in_circle(self, x, y, centre, radius):
        return self.is_in_elipse(x, y, centre, radius, radius)

    def is_in_ellipse(self, x, y, centre, x_radius, y_radius):
        x_centre, y_centre = centre
        if ((x_centre - x)**2) / x_radius**2 + ((y_centre - y)**2) / y_radius**2 < 1:
            return True
        return False

    def set_circle_to_xy(self, x, y, radius, colour_idx):
        self.set_ellipse_to_xy(x, y, radius, radius, colour_idx)

    def set_ellipse_to_xy(self, x, y, x_radius, y_radius, colour_idx):
        (x_min, x_max) = self.get_axis_range(x, x_radius, self.width)
        (y_min, y_max) = self.get_axis_range(y, y_radius, self.height)
        for x_ in range(x_min, x_max):
            for y_ in range(y_min, y_max):
                if self.is_in_ellipse(x_, y_, (x, y), x_radius, y_radius):
                    self.set_colour_to_xy(x_, y_, colour_idx)


def get_test_image(n_reps, n_classes,
                   image_width, image_height, image_depth):
    td = Toy_Image(n_classes,
                   image_width, image_height, image_depth)
    for rep in range(n_reps):
        for colour_idx in range(n_classes):
            x, y = td.get_random_xy()
            rand_width = randint(1, int(td.width / 8))
            rand_height = randint(1, int(td.height / 8))
            rnd_i = randint(0, 3)
            if rnd_i == 0:
                td.set_square_to_xy(x, y, rand_width, colour_idx)
            elif rnd_i == 1:
                td.set_circle_to_xy(x, y, rand_width, colour_idx)
            elif rnd_i == 2:
                td.set_rect_to_xy(x, y, rand_width, rand_height, colour_idx)
            elif rnd_i == 3:
                td.set_ellipse_to_xy(x, y, rand_width, rand_height, colour_idx)
    return td.image, td.one_hot_array


def get_test_images(n_images, n_reps, n_classes,
                    image_width, image_height, colour_channels):
    images, one_hots = [], []
    for i in range(n_images):
        image, one_hot = get_test_image(n_reps, n_classes,
                                        image_width, image_height, colour_channels)
        images.append(image)
        one_hots.append(one_hot)
    return images, one_hots

if __name__ == "__main__":
    n_reps = 4
    n_classes = 5
    width, height = 400, 400
    colour_channels = 3
    image, one_hot = get_test_image(n_reps, n_classes,
                                    width, height, colour_channels)

    import matplotlib.pyplot as plt
    plt.imshow(image, cmap='jet')
    plt.show()
