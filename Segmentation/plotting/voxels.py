import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_volume(volume, show=False):
    voxel = volume[:, :, :, 0] > 0
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel, facecolors=volume, linewidth=0.5)
    if show:
        plt.show()
    else:
        plt.savefig(f"test_vol")
        plt.close('all')

def plot_slice(vol_slice, show=False):
    fig = plt.figure()
    plt.imshow(vol_slice, cmap="gray")
    if show:
        plt.show()
    else:
        plt.savefig(f"test_slice")
        plt.close('all')
