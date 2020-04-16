from Segmentation.utils.data_loader import create_OAI_dataset

if __name__=="__main__":
    folder = 'valid'
    train = (folder == 'train')

    create_OAI_dataset(data_folder="./Data/" + folder,
                       tfrecord_directory="./Data/tf_" + folder + "_3d",
                       get_train=train,
                       use_2d=False)
