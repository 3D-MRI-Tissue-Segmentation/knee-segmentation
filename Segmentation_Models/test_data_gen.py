def test_random_data_gen():
    from Segmentation_Models.src.data_gen.toy_data_gen import Toy_Image

    n_reps = 10
    n_classes = 100
    test_ti = Toy_Image(n_classes, 40, 40, 3)

    for rep in range(n_reps):
        for i in range(n_classes):
            test_ti.set_color_to_random_xy(i)

    import matplotlib.pyplot as plt
    plt.imshow(test_ti.image, cmap='jet')
    plt.savefig("Data/Tests_data/random_image.png")