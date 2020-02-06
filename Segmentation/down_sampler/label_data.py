import numpy as np
import matplotlib.pyplot as plt

class Label:
    def __init__(self, train_images, image_height,
                 num_images=None, start_point=0, three_dim=True):
        assert image_height == train_images.shape[1], "Unexpected shape for train image"
        self.train_images = train_images
        self.num_images = num_images
        self.start_point = start_point
        self.three_dim = three_dim
        self.coords = []

        self.image_counter = self.start_point
        self.ax = None
        self.im = None

        self.click_start = None
        if self.three_dim:
            self.layer_idx = 0
            self.layer_bounds = (0, image_height)

        self.done = False

    def onclick(self, event):
        self.click_start = (int(event.xdata), int(event.ydata))

    @staticmethod
    def calc_drag_distance(start, end):
        x_start, y_start = start
        x_end, y_end = end
        return ((x_end - x_start)**2 + (y_end - y_start)**2)**0.5

    def onrelease(self, event):
        end = (int(event.xdata), int(event.ydata))
        dist = Label.calc_drag_distance(self.click_start, end)
        if self.three_dim:
            self.coords.append([self.image_counter, dist,
                                *self.click_start, self.layer_idx])
        else:
            self.coords.append([self.image_counter, dist,
                                *self.click_start])
        self.image_counter += 1

        end = False
        if self.num_images is not None:
            if self.image_counter >= (self.num_images + self.start_point):
                end = True
                plt.close()
        if not end:
            self.update_image()

    def onarrowpress(self, event):
        update = False
        if event.key == "up":
            if self.layer_bounds[0] <= self.layer_idx < (self.layer_bounds[1] - 1):
                self.layer_idx += 1
                update = True
        elif event.key == "down":
            if self.layer_bounds[0] < self.layer_idx <= (self.layer_bounds[1] - 1):
                self.layer_idx -= 1
                update = True
        if update:
            self.update_image()

    def update_image(self):
        if self.three_dim:
            self.im.set_data(self.train_images[self.image_counter][self.layer_idx, :, :])
            self.ax.set_title(f"Image being labeled: {self.image_counter}, layer: {self.layer_idx}")
        else:
            self.im.set_data(self.train_images[self.image_counter])
            self.ax.set_title(f"Image being labeled: {self.image_counter}")
        plt.draw()

    def reset(self):
        self.fig = plt.figure()
        self.ax = self.fig.gca()
        if self.three_dim:
            self.im = self.ax.imshow(self.train_images[self.image_counter][self.layer_idx, :, :])
        else:
            self.im = self.ax.imshow(self.train_images[self.image_counter])

    def start_loop(self):
        self.update_image()
        if self.three_dim:
            self.im.set_data(self.train_images[self.image_counter][self.layer_idx, :, :])
        else:
            self.im.set_data(self.train_images[self.image_counter])
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        if self.three_dim:
            self.fig.canvas.mpl_connect('key_press_event', self.onarrowpress)
        plt.show()

    def main(self):
        self.reset()
        self.start_loop()


def mnist_3d_main():
    import h5py
    with h5py.File("./Data/Tests_data/full_dataset_vectors.h5") as hf:
        data = hf["X_train"][:]

    data = np.reshape(data, (data.shape[0], 16, 16, 16))

    label = Label(data, 16)
    label.main()

    for i in label.coords:
        print(i)

    import pandas as pd

    df = pd.DataFrame(np.asarray(label.coords),
                      columns=["idx", "radius", "x", "y", "z"])
    df.to_csv("./Data/Tests_data/mnist_3d_centre_test_data.csv", index=False)


def mnist_2d_main():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    data = x_train

    label = Label(data, 28, three_dim=False)
    label.main()

    for i in label.coords:
        print(i)

    import pandas as pd

    df = pd.DataFrame(np.asarray(label.coords),
                      columns=["idx", "radius", "x", "y"])
    df.to_csv("./Data/Tests_data/mnist_2d_centre_test_data.csv", index=False)


if __name__ == "__main__":
    mnist_2d_main()
