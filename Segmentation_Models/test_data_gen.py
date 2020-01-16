def test_random_data_gen_2D():
    from src.data_gen.toy_data_gen import Toy_Image

    n_reps = 10
    n_classes = 100
    test_ti = Toy_Image(n_classes, 40, 40, 3)

    for rep in range(n_reps):
        for i in range(n_classes):
            test_ti.set_colour_to_random_xy(i)

    import matplotlib.pyplot as plt
    plt.imshow(test_ti.image, cmap='jet')
    img_path = "./Data/Tests_data/random_image_2d.png"
    plt.savefig(img_path)
    import os
    assert os.path.isfile(img_path), "file does not exist"
    os.remove(img_path)


def test_random_shapes_gen_2D():
    from src.data_gen.toy_data_gen import Toy_Image
    from random import randint

    n_reps = 10
    n_classes = 6
    width = 400
    height = 400

    test_ti = Toy_Image(n_classes, width, height, 3)

    for rep in range(n_reps):
        for colour_idx in range(n_classes):
            x,y = test_ti.get_random_xy()
            rand_width = randint(1, int(test_ti.width/8))
            rand_height = randint(1, int(test_ti.height/8))
            rnd_i = randint(0, 3)
            if rnd_i == 0:
                test_ti.set_square_to_xy(x, y, rand_width, colour_idx)
            elif rnd_i == 1:
                test_ti.set_circle_to_xy(x, y, rand_width, colour_idx)
            elif rnd_i == 2:
                test_ti.set_rect_to_xy(x, y, rand_width, rand_height, colour_idx)
            elif rnd_i == 3:
                test_ti.set_oval_to_xy(x, y, rand_width, rand_height, colour_idx)

    import matplotlib.pyplot as plt
    plt.imshow(test_ti.image, cmap='jet')
    img_path = "./Data/Tests_data/random_shapes_image_2d.png"
    plt.savefig(img_path)
    import os
    assert os.path.isfile(img_path), "file does not exist"
    os.remove(img_path)


def test_get_test_images():
    from src.data_gen.toy_data_gen import get_test_images

    n_images, n_reps, n_classes = 10, 4, 5
    width, height, depth = 400, 400, 3

    images, one_hots = get_test_images(n_images, n_reps, n_classes,
                                            width, height, depth)

    assert len(images) == n_images
    assert len(one_hots) == n_images

    import matplotlib.pyplot as plt
    import os

    for image in images:
        plt.imshow(image, cmap='jet')
        img_path = "./Data/Tests_data/random_shapes_images_2d.png"
        plt.savefig(img_path)
        assert os.path.isfile(img_path), "file does not exist"
        os.remove(img_path)
