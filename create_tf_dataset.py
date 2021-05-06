from Segmentation.utils.data_loader import create_OAI_dataset
import os

def create_tfrecords(folder="train", use_2d=False, crop_size=None, mid_folders=""):
    train = (folder == 'train')
    str_dim = "" if use_2d else "_3d"

    if use_2d and crop_size is None:
        crop_size = 144

    if not os.path.exists(f'./Data{mid_folders}/tfrecords/'):
        os.makedirs(f'./Data{mid_folders}/tfrecords/')
    if not os.path.exists(f'./Data{mid_folders}/tfrecords/{folder}{str_dim}'):
        os.makedirs(f'./Data{mid_folders}/tfrecords/{folder}{str_dim}')

    create_OAI_dataset(data_folder=f"./Data{mid_folders}/" + folder,
                       tfrecord_directory=f"./Data{mid_folders}/tfrecords/" + folder + str_dim,
                       get_train=train,
                       use_2d=use_2d,
                       crop_size=crop_size)
    
if __name__ == "__main__":
    # create_tfrecords("train", mid_folders='/mnt')
    create_tfrecords("valid", mid_folders='/mnt')
