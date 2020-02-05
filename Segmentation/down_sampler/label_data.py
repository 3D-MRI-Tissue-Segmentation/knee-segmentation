import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Label:
    def __init__(self, num_images=None, start_point=0,
                 data=tf.keras.datasets.mnist):
        self.num_images = num_images
        self.start_point = start_point
        self.data = data
        self.coords = []
        self.fig = plt.figure()

        (train_images, train_labels), (test_images, test_labels) = self.data.load_data()

        self.train_images = train_images

        self.image_counter = self.start_point
        self.ax = None

    def onclick(self, event):
        self.coords.append([self.image_counter, (int(event.xdata), int(event.ydata))])
        self.image_counter += 1

        end = False
        if self.num_images is not None:
            if self.image_counter >= (self.num_images + self.start_point):
                print("ENDING")
                plt.close()
                end = True
        if not end:
            self.update_image()
            plt.show()

    def update_image(self):
        self.ax.imshow(self.train_images[self.image_counter])
        self.ax.set_title(f"Image being labeled: {self.image_counter}")

    def start_loop(self):
        self.update_image()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def main(self):
        self.ax = self.fig.add_subplot(111)
        self.start_loop()


if __name__ == "__main__":
    label = Label(start_point=4)
    label.main()
    print(label.coords)
