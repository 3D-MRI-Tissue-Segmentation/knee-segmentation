from Segmentation.utils.data_loader import create_OAI_dataset

if __name__ == "__main__":
    folder = 'valid'
    use_2d = True

    train = (folder == 'train')
    str_dim = "" if use_2d else "_3d"

    create_OAI_dataset(data_folder="./Data/" + folder,
                       tfrecord_directory="./Data/tfrecords/" + folder + str_dim,
                       get_train=train,
                       use_2d=use_2d)
