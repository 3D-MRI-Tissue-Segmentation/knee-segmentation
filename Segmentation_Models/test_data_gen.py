def test_random_data_gen():
    from src.data_gen.toy_data_gen import Toy_Image

    n_reps = 10
    n_classes = 100
    test_ti = Toy_Image(n_classes, 40, 40, 3)

    for rep in range(n_reps):
        for i in range(n_classes):
            test_ti.set_colour_to_random_xy(i)

    import matplotlib.pyplot as plt
    plt.imshow(test_ti.image, cmap='jet')
    img_path = "./Data/Tests_data/random_image.png"
    plt.savefig(img_path)
    import os
    assert os.path.isfile(img_path), "file does not exist"
    os.remove(img_path)