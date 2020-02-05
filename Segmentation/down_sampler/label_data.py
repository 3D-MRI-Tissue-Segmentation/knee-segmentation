import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Label:
    def __init__(self, train_images, image_height,
                 num_images=None, start_point=0,
                 data=tf.keras.datasets.mnist):
        self.num_images = num_images
        self.start_point = start_point
        self.data = data
        self.coords = []

        self.train_images = train_images

        self.image_counter = self.start_point
        self.ax = None
        self.im = None

        self.click_start = None
        self.layer_idx = 0
        self.layer_bounds = (0, image_height)

        assert image_height == train_images.shape[1], "Unexpected shape for train image"

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
        self.coords.append([self.image_counter, dist,
                            *self.click_start, self.layer_idx])
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
        self.im.set_data(self.train_images[self.image_counter][self.layer_idx, :, :])
        self.ax.set_title(f"Image being labeled: {self.image_counter}, layer: {self.layer_idx}")
        plt.draw()

    def reset(self):
        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.im = self.ax.imshow(self.train_images[self.image_counter][self.layer_idx, :, :])

    def start_loop(self):
        self.update_image()
        self.im.set_data(self.train_images[self.image_counter][self.layer_idx, :, :])
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
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
    df.to_csv("./Data/mnist_3d_centre_test_data.csv", index=False)


if __name__ == "__main__":
    mnist_3d_main()
    
