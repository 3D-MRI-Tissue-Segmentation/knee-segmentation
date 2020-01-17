def test_standard_unet():
    from src.data_gen.toy_image_gen import get_test_images

    n_images, n_reps, n_classes = 10, 4, 5
    width, height = 400, 400
    colour_channels = 3
    images, one_hots = get_test_images(n_images, n_reps, n_classes, 
                                       width, height, colour_channels)

    from src.model.unet import UNet

