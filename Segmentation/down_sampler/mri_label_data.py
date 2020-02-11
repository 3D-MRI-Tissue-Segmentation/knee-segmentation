import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.patches as patches
from matplotlib.widgets import Button

class Click:

    def __init__(self):
        self.bound = None

    def click_start(self, ax_start, start):
        self.start = start
        self.ax_start = ax_start

    def click_end(self, ax_end, end):
        if self.ax_start != ax_end:
            return None
        self.end = end
        self.bound = (self.start, self.end)


class MRI_Label:

    def __init__(self, train_images,
                 num_images=None, start_point=0, image_str_list=None):
        self.train_images = train_images
        self.image_str_list = image_str_list

        self.num_images = num_images
        self.start_point = start_point
        self.coords = []

        self.image_counter = self.start_point
        if self.image_str_list is not None:
            assert self.start_point <= len(image_str_list) - 1, "Start point must be less than or equal to the len of strings"
            self.next_image_for_str()
        self.ax = None
        self.im_yz = None
        self.im_xz = None
        self.im_xy = None

        self.prev_3d_patch = [None, None, None]  # Stores the previous 3d patch used
        self.prev_patch = [None, None, None]  # Stores the previous patch used
        self.focus_area = [Click(), Click(), Click()]  # Create click object for each of the axes
        self.idx = [0, 0, 0]  # idx of the tangental axis
        self.bounds = (self.train_images.shape[1],  # height (y) (vertically up)
                       self.train_images.shape[2],  # width (x) (outside to inside leg)
                       self.train_images.shape[3])  # depth (z) (front to back)
        # oai bounds =(384, 384, 160)
        self.focus = 0  # Target axis that is being modified (0, 1, or 2)

    def onclick(self, event):
        if event.inaxes not in [self.ax[0][0], self.ax[0][1], self.ax[1][1]]:
            return
        click_start_loc = (int(event.xdata), int(event.ydata))
        if event.inaxes == self.ax[0][0]:
            self.focus_area[0].click_start(self.ax[0][0],
                                           click_start_loc)
        elif event.inaxes == self.ax[0][1]:
            self.focus_area[1].click_start(self.ax[0][1],
                                           click_start_loc)
        elif event.inaxes == self.ax[1][1]:
            self.focus_area[2].click_start(self.ax[1][1],
                                           click_start_loc)

    @staticmethod
    def calc_drag_distance(start, end):
        x_start, y_start = start
        x_end, y_end = end
        return ((x_end - x_start)**2 + (y_end - y_start)**2)**0.5

    def onrelease(self, event):
        if event.inaxes not in [self.ax[0][0], self.ax[0][1], self.ax[1][1]]:
            return
        click_end_loc = (int(event.xdata), int(event.ydata))
        if event.inaxes == self.ax[0][0]:
            self.focus_area[0].click_end(self.ax[0][0],
                                         click_end_loc)
            self.update_yz(False)
        elif event.inaxes == self.ax[0][1]:
            self.focus_area[1].click_end(self.ax[0][1],
                                         click_end_loc)
            self.update_xz(False)
        elif event.inaxes == self.ax[1][1]:
            self.focus_area[2].click_end(self.ax[1][1],
                                         click_end_loc)
            self.update_xy(False)

    def next_image_for_str(self):
        """ gets the next image in the string list """
        assert self.image_str_list, "Image string must be true for this option"
        import h5py
        with h5py.File(self.image_str_list[self.image_counter], 'r') as hf:
            img = np.array(hf['data'])
        self.train_images = np.reshape(img, (1, *img.shape))

    def calculate_cube(self):
        yz_bound = self.focus_area[0].bound
        xz_bound = self.focus_area[1].bound
        xy_bound = self.focus_area[2].bound
        x_ = [xz_bound[0][1], xz_bound[1][1], xy_bound[0][1], xy_bound[1][1]]
        y_ = [yz_bound[0][1], yz_bound[1][1], xy_bound[0][0], xy_bound[1][0]]
        z_ = [xz_bound[0][0], xz_bound[1][0], yz_bound[0][0], yz_bound[1][0]]
        x_min = min(x_)
        x_max = max(x_)
        y_min = min(y_)
        y_max = max(y_)
        z_min = min(z_)
        z_max = max(z_)
        return x_min, x_max, y_min, y_max, z_min, z_max

    def next_vol(self, event=None):
        """ update when current image is annonated """
        print("DONE")
        x_min, x_max, y_min, y_max, z_min, z_max = self.calculate_cube()
        self.coords.append(["a", "b", "c"])

        self.image_counter += 1
        end = False
        if self.image_str_list is not None:
            if self.image_counter >= (len(self.image_str_list) + self.start_point):
                end = True
                plt.close()
            else:
                self.next_image_for_str()
        else:
            if self.image_counter >= (self.num_images + self.start_point):
                end = True
                plt.close()
        if not end:
            self.update_yz()
            self.update_xz()
            self.update_xy()

    def onkeypress(self, event):
        if event.key == "1":
            self.focus = 0
        elif event.key == "2":
            self.focus = 1
        elif event.key == "3":
            self.focus = 2
        elif event.key == "enter":
            self.next_vol()

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
            if target_ax == 0:
                self.update_yz()
            elif target_ax == 1:
                self.update_xz()
            elif target_ax == 2:
                self.update_xy()
            else:
                raise IndexError(f"target_ax value of {target_ax} unexpected.")

    def update_patch(self, ax_idx, ax):
        if self.focus_area[ax_idx].bound is not None:
            b = self.focus_area[ax_idx].bound
            delta_y = b[1][1] - b[0][1]
            detla_x = b[1][0] - b[0][0]
            rect = patches.Rectangle(b[0], detla_x, delta_y,
                                     linewidth=1, edgecolor='r', facecolor='none')
            if rect != self.prev_patch[ax_idx]:
                if self.prev_patch[ax_idx] is not None:
                    self.prev_patch[ax_idx].remove()
                self.prev_patch[ax_idx] = rect
                ax.add_patch(rect)
                self.update_3d_patch(ax_idx)

    def update_3d_patch(self, ax_idx):
        if self.focus_area[ax_idx].bound is not None:
            b = self.focus_area[ax_idx].bound
            delta_y = b[1][1] - b[0][1]
            detla_x = b[1][0] - b[0][0]

            b_t = (b[0][1], b[0][0])
            rect = patches.Rectangle(b_t, delta_y, detla_x,
                                     linewidth=1, edgecolor='r', facecolor='none')  # need to use the transpose to visualise
            if rect != self.prev_3d_patch[ax_idx]:
                if self.prev_3d_patch[ax_idx] is not None:
                    self.prev_3d_patch[ax_idx].remove()
                self.prev_3d_patch[ax_idx] = rect
                self.ax[1][0].add_patch(rect)
                z = self.idx[ax_idx]
                if ax_idx == 0:
                    zdir = "x"
                elif ax_idx == 1:
                    zdir = "y"
                elif ax_idx == 2:
                    zdir = "z"
                art3d.pathpatch_2d_to_3d(rect, z=z, zdir=zdir)

                # print(all(self.prev_3d_patch))

                if all(self.prev_3d_patch):
                    #  updating transparent cube
                    x_min, x_max, y_min, y_max, z_min, z_max = self.calculate_cube()

                    vertices = np.array([[x_min, y_min, z_min],
                                         [x_min, y_min, z_max],
                                         [x_min, y_max, z_max],
                                         [x_min, y_max, z_min],
                                         [x_max, y_max, z_min],
                                         [x_max, y_min, z_min],
                                         [x_max, y_min, z_max],
                                         [x_max, y_max, z_max],
                                         ])
                    sides = [
                        [vertices[0], vertices[1], vertices[2], vertices[3]],
                        [vertices[0], vertices[3], vertices[4], vertices[5]],
                        [vertices[0], vertices[1], vertices[6], vertices[5]],
                        [vertices[2], vertices[3], vertices[4], vertices[7]],
                        [vertices[4], vertices[5], vertices[6], vertices[7]],
                        [vertices[1], vertices[2], vertices[7], vertices[6]],
                    ]
                    self.ax[1][0].scatter3D(vertices[:, 0],
                                            vertices[:, 1],
                                            vertices[:, 2])
                    self.ax[1][0].add_collection3d(art3d.Poly3DCollection(sides,
                                                                          edgecolors='b', facecolors='w',
                                                                          linewidths=1, alpha=0.2))

    def update_yz(self, update_image=True):
        plt.sca(self.ax[0][0])
        if update_image:
            img_count = 0
            if self.image_str_list is None:
                img_count = self.image_counter
            im = self.train_images[img_count, self.idx[0], :, :]
            self.im_yz.set_data(im)
            self.ax[0][0].set_title(f"YZ plane - Image being labeled: {self.image_counter}\nlayer: {self.idx[0]}")
        self.update_patch(0, self.ax[0][0])
        plt.draw()

    def update_xz(self, update_image=True):
        plt.sca(self.ax[0][1])
        if update_image:
            img_count = 0
            if self.image_str_list is None:
                img_count = self.image_counter
            im = self.train_images[img_count, :, self.idx[1], :]
            self.im_xz.set_data(im)
            self.ax[0][1].set_title(f"XZ plane - Image being labeled: {self.image_counter}\nlayer: {self.idx[1]}")
        self.update_patch(1, self.ax[0][1])
        plt.draw()

    def update_xy(self, update_image=True):
        plt.sca(self.ax[1][1])
        if update_image:
            img_count = 0
            if self.image_str_list is None:
                img_count = self.image_counter
            im = self.train_images[img_count, :, :, self.idx[2]]
            self.im_xy.set_data(im)
            self.ax[1][1].set_title(f"XY plane - Image being labeled: {self.image_counter}\nlayer: {self.idx[2]}")
        self.update_patch(2, self.ax[1][1])
        plt.draw()

    def reset(self):
        self.ax = [[None, None], [None, None]]
        # self.fig, self.ax = plt.subplots(2, 2)
        self.fig = plt.figure()
        self.ax[0][0] = self.fig.add_subplot(221)
        self.ax[0][0].set_xlabel('z plane')
        self.ax[0][0].set_ylabel('y plane')
        self.ax[0][1] = self.fig.add_subplot(222)
        self.ax[0][1].set_xlabel('z plane')
        self.ax[0][1].set_ylabel('x plane')
        self.ax[1][0] = self.fig.add_subplot(223, projection='3d')
        self.ax[1][0].set_xlim(0, self.bounds[0])
        self.ax[1][0].set_ylim(0, self.bounds[1])
        self.ax[1][0].set_zlim(0, self.bounds[2])
        self.ax[1][0].set_xlabel('x plane')
        self.ax[1][0].set_ylabel('y plane')
        self.ax[1][0].set_zlabel('z plane')
        self.ax[1][1] = self.fig.add_subplot(224)
        self.ax[1][1].set_xlabel('y plane')
        self.ax[1][1].set_ylabel('x plane')
        # set the image to be one from the middle of the data to avoid an image of zeros which results in an empty colourmap.
        self.im_yz = self.ax[0][0].imshow(self.train_images[self.image_counter, int(self.bounds[0] / 2), :, :],
                                          cmap='jet', aspect='auto')
        self.im_xz = self.ax[0][1].imshow(self.train_images[self.image_counter, :, int(self.bounds[1] / 2), :],
                                          cmap='jet', aspect='auto')
        self.im_xy = self.ax[1][1].imshow(self.train_images[self.image_counter, :, :, int(self.bounds[2] / 2)],
                                          cmap='jet', aspect='auto')
        axnext = plt.axes([0.9, 0.01, 0.05, 0.075])
        self.bnext = Button(axnext, "Next")

    def start_loop(self):
        self.update_yz()
        self.update_xz()
        self.update_xy()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        self.bnext.on_clicked(self.next_vol)
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
    image_list = []
    for i in range(2):
        i_str = str(i).zfill(2)
        image_list.append(f"./Data/train/train_001_V{i_str}.im")
    print(image_list)
    label = MRI_Label(None, image_str_list=image_list)
    label.main()


if __name__ == "__main__":
    mri_3d_main()
