from Segmentation.utils.data_loader import create_OAI_dataset
import os

def create_tfrecords(folder="train", use_2d=False, crop_size=None):
    train = (folder == 'train')
    str_dim = "" if use_2d else "_3d"

    if use_2d and crop_size is None:
        crop_size = 144

    if not os.path.exists('./Data/tfrecords/'):
        os.makedirs('./Data/tfrecords/')
    if not os.path.exists(f'./Data/tfrecords/{folder}{str_dim}'):
        os.makedirs(f'./Data/tfrecords/{folder}{str_dim}')

    create_OAI_dataset(data_folder="./Data/" + folder,
                       tfrecord_directory="./Data/tfrecords/" + folder + str_dim,
                       get_train=train,
                       use_2d=use_2d,
                       crop_size=crop_size)
    
if __name__ == "__main__":
    create_tfrecords("train")
    create_tfrecords("valid")
