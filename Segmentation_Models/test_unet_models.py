def test_standard_unet():
    from src.data_gen.toy_data_gen import get_test_image

    n_reps, n_classes = 4, 5
    width, height, depth = 400, 400, 3
    image, one_hot = get_test_image(n_reps, n_classes, 
                                    width, height, depth)

    from src.model.unet import UNet

