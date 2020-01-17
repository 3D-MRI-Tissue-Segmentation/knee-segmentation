def test_standard_vnet():
    from Segmentation.data_gen.toy_volume_gen import get_test_volumes

    n_volumes, n_reps, n_classes = 10, 4, 5
    width, height, depth = 10, 10, 10
    colour_channels = 3
    volumes, one_hots = get_test_volumes(n_volumes, n_reps, n_classes, 
                                         width, height, depth, colour_channels)

    ## from Segmentation.model.vnet import VNet
