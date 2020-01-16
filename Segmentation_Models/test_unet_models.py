def test_standard_unet():
    from src.data_gen.toy_data_gen import get_test_image

    n_images, n_reps, n_classes = 10, 4, 5
    width, height, depth = 400, 400, 3
    images, one_hots = get_test_image(n_images, n_reps, n_classes, 
                                    width, height, depth)

    from src.model.unet import UNet

