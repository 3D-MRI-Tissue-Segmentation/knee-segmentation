import numpy as np
import matplotlib.pyplot as plt

class MRI_Label:
    def __init__(self, train_images,
                 num_images=None, start_point=0):
        self.train_images = train_images
        self.num_images = num_images
        self.start_point = start_point
        self.coords = []

        self.image_counter = self.start_point
        self.ax = None
        self.im_yz = None
        self.im_xz = None
        self.im_xy = None

        self.click_start = None
        self.idx = [0, 0, 0]
        # oai bounds =(384, 384, 160)
        self.bounds = (train_images.shape[1],  # height (y) (vertically up)
                       train_images.shape[2],  # width (x) (outside to inside leg)
                       train_images.shape[3])  # depth (z) (front to back)
        self.done = False

        self.focus = 0

    def onclick(self, event):
        print(event.inaxes == self.ax[0][0])  # how to determine which you are in
        self.click_start = (int(event.xdata), int(event.ydata))

    @staticmethod
    def calc_drag_distance(start, end):
        x_start, y_start = start
        x_end, y_end = end
        return ((x_end - x_start)**2 + (y_end - y_start)**2)**0.5

    def onrelease(self, event):
        end = (int(event.xdata), int(event.ydata))
        dist = MRI_Label.calc_drag_distance(self.click_start, end)

        # need to move to an update function

        # self.coords.append([self.image_counter, dist,
        #                     *self.click_start, self.idx[0]])
        # self.image_counter += 1

        # end = False
        # if self.num_images is not None:
        #     if self.image_counter >= (self.num_images + self.start_point):
        #         end = True
        #         plt.close()
        # if not end:
        #     self.update_yz()

    def onarrowpress(self, event):
        if event.key == "0":
            self.focus = 0
        elif event.key == "1":
            self.focus = 1
        elif event.key == "2":
            self.focus = 2

        target_ax = self.focus
        update = False

        if event.key == "up":
            if 0 <= self.idx[target_ax] < (self.bounds[target_ax] - 1):
                self.idx[target_ax] += 1
                update = True
        elif event.key == "down":
            if 0 < self.idx[target_ax] <= (self.bounds[target_ax] - 1):
                self.idx[target_ax] -= 1
                update = True
        if update:
            print(self.image_counter, self.idx)
            if target_ax == 0:
                self.update_yz()
            elif target_ax == 1:
                self.update_xz()
            elif target_ax == 2:
                self.update_xy()
            else:
                raise IndexError(f"target_ax value of {target_ax} unexpected.")

    def update_yz(self):
        plt.sca(self.ax[0][0])
        im = self.train_images[self.image_counter, self.idx[0], :, :]
        print("u yz", im.shape)
        print(im.max(), im.min())
        self.im_yz.set_data(im)
        self.ax[0][0].set_title(f"YZ plane - Image being labeled: {self.image_counter}\nlayer: {self.idx[0]}")
        plt.draw()
        print(plt.gca())
        print(self.ax[0][0] == plt.gca())

    def update_xz(self):
        plt.sca(self.ax[0][1])
        im = self.train_images[self.image_counter, :, self.idx[1], :]
        print("u xz", im.shape)
        print(im.max(), im.min())
        self.im_xz.set_data(im)
        self.ax[0][1].set_title(f"XZ plane - Image being labeled: {self.image_counter}\nlayer: {self.idx[1]}")
        plt.draw()
        print(plt.gca())
        print(self.ax[0][1] == plt.gca())

    def update_xy(self):
        plt.sca(self.ax[1][1])
        im = self.train_images[self.image_counter, :, :, self.idx[2]]
        print("u xy", im.shape)
        print(im.max(), im.min())
        self.im_xy.set_data(im)
        self.ax[1][1].set_title(f"XY plane - Image being labeled: {self.image_counter}\nlayer: {self.idx[2]}")
        plt.draw()
        print(plt.gca())
        print(self.ax[1][1] == plt.gca())

    def reset(self):
        self.fig, self.ax = plt.subplots(2, 2)
        # set the image to be one from the middle of the data to avoid an image of zeros which results in an empty colourmap.
        self.im_yz = self.ax[0][0].imshow(self.train_images[self.image_counter, int(self.bounds[0] / 2), :, :], cmap='jet', aspect='auto')
        self.im_xz = self.ax[0][1].imshow(self.train_images[self.image_counter, :, int(self.bounds[1] / 2), :], cmap='jet', aspect='auto')
        self.im_xy = self.ax[1][1].imshow(self.train_images[self.image_counter, :, :, int(self.bounds[2] / 2)], cmap='jet', aspect='auto')

    def start_loop(self):
        self.update_yz()
        self.update_xz()
        self.update_xy()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.fig.canvas.mpl_connect('key_press_event', self.onarrowpress)
        self.fig.show()
        plt.show()

    def main(self):
        self.reset()
        self.start_loop()


def mnist_3d_main():
    import h5py
    with h5py.File("./Data/Tests_data/full_dataset_vectors.h5") as hf:
        data = hf["X_train"][:]

    data = np.reshape(data, (data.shape[0], 16, 16, 16))

    label = MRI_Label(data, 16)
    label.main()

    for i in label.coords:
        print(i)

    print("========== Annotation complete ==========")
    save_loc = "./Data/Tests_data/mri_mnist_3d_centre_test_data.csv"
    print(f"Save location: {save_loc}")
    write = input("Do you wish to write: (Enter 'y' to write): ")

    if (write == "y") or (write == "Y"):
        import pandas as pd
        df = pd.DataFrame(np.asarray(label.coords),
                          columns=["idx", "radius", "x", "y", "z"])
        df.to_csv(save_loc, index=False)


def mri_3d_main(img_name='./Data/train/train_001_V00.im'):
    import h5py
    with h5py.File(img_name, 'r') as hf:
        img = np.array(hf['data'])

    #plt.ion()

    print(img.shape)
    img = np.reshape(img, (1, *img.shape))
    print(img.shape)

    label = MRI_Label(img)
    label.main()

    # im1 = img[0, 3, :, :]
    # im2 = img[0, :, 0, :]
    # im3 = img[0, :, :, 0]

    # print(im1.shape)
    # print(im2.shape)
    # print(im3.shape)

    # print(im1.max())

    # plt.imshow(im1)
    # # plt.imshow(im2)
    # # plt.imshow(im3)
    # plt.show()


if __name__ == "__main__":
    mri_3d_main()
