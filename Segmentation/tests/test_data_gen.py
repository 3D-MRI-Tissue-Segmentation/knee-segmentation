def test_random_data_gen_2D():
    from Segmentation.data_gen.toy_image_gen import Toy_Image

    n_reps, n_classes = 10, 20
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
    from Segmentation.data_gen.toy_image_gen import Toy_Image
    from random import randint

    n_reps, n_classes = 10, 6
    width, height = 400, 400
    colour_channels = 3

    test_ti = Toy_Image(n_classes, width, height, colour_channels)

    for rep in range(n_reps):
        for colour_idx in range(n_classes):
            x,y = test_ti.get_random_xy()
            rand_width = randint(1, int(test_ti.width/8))
            rand_height = randint(1, int(test_ti.height/8))
            rnd_i = randint(0, 4)
            if rnd_i == 0:
                test_ti.set_square_to_xy(x, y, rand_width, colour_idx)
            elif rnd_i == 1:
                test_ti.set_circle_to_xy(x, y, rand_width, colour_idx)
            elif rnd_i == 2:
                test_ti.set_rect_to_xy(x, y, rand_width, rand_height, colour_idx)
            elif rnd_i == 3:
                test_ti.set_ellipse_to_xy(x, y, rand_width, rand_height, colour_idx)
            elif rnd_i == 4:
                test_ti.set_colour_to_random_xy(colour_idx)
            else:
                raise Exception(f"Invalid {rnd_i}")

    import matplotlib.pyplot as plt
    plt.imshow(test_ti.image, cmap='jet')
    img_path = "./Data/Tests_data/random_shapes_image_2d.png"
    plt.savefig(img_path)
    import os
    assert os.path.isfile(img_path), "file does not exist"
    os.remove(img_path)


def test_get_test_images():
    from Segmentation.data_gen.toy_image_gen import get_test_images

    n_images, n_reps, n_classes = 10, 4, 5
    width, height = 400, 400
    colour_channels = 3

    images, one_hots = get_test_images(n_images, n_reps, n_classes,
                                            width, height, colour_channels)

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


def test_random_volumes_gen():
    from Segmentation.data_gen.toy_volume_gen import Toy_Volume
    from random import randint

    n_reps, n_classes = 10, 6
    width, height, depth = 40, 40, 40
    colour_channels = 3

    test_tv = Toy_Volume(n_classes, width, height, depth, colour_channels)

    for rep in range(n_reps):
        for colour_idx in range(n_classes):
            x, y, z = test_tv.get_random_xyz()
            rand_x_len = randint(1, int(test_tv.width/4))
            rand_y_len = randint(1, int(test_tv.height/4))
            rand_z_len = randint(1, int(test_tv.depth/4))
            rnd_i = randint(0, 4)
            if rnd_i == 0:
                test_tv.set_rect_cuboid_to_xyz(x, y, z, 
                                               rand_x_len, rand_y_len, rand_z_len, 
                                               colour_idx)
            elif rnd_i == 1:
                test_tv.set_ellipsoid_to_xyz(x, y, z,
                                             rand_x_len, rand_y_len, rand_z_len, 
                                             colour_idx)
            elif rnd_i == 2:
                test_tv.set_cube_to_xyz(x, y, z, 
                                        rand_x_len, 
                                        colour_idx)
            elif rnd_i == 3:
                test_tv.set_sphere_to_xyz(x, y, z,
                                          rand_x_len, 
                                          colour_idx)
            elif rnd_i == 4:
                test_tv.set_colour_to_random_xyz(colour_idx)
            else:
                raise Exception(f"Invalid {rnd_i}")

    from Segmentation.data_gen.toy_volume_gen import plot_volume
    plot_volume(test_tv.volume, False)

def test_get_test_volumes():
    from Segmentation.data_gen.toy_volume_gen import get_test_volumes

    n_volumes, n_reps, n_classes = 10, 4, 5
    width, height, depth = 40, 40, 20
    colour_channels = 3

    volumes, one_hots = get_test_volumes(n_volumes, n_reps, n_classes,
                                         width, height, depth, colour_channels)

    assert len(volumes) == n_volumes
    assert len(one_hots) == n_volumes

    import matplotlib.pyplot as plt
    import os

    from Segmentation.data_gen.toy_volume_gen import plot_volume
    for volume in volumes:
        plot_volume(volume, False)
    