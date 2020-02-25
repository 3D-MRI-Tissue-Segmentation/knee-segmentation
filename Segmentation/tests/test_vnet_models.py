def test_standard_vnet():
    from Segmentation.data_gen.toy_volume_gen import get_test_volumes

    n_volumes, n_reps, n_classes = 1, 4, 2
    width, height, depth = 288, 288, 160
    colour_channels = 3
    volumes, one_hots = get_test_volumes(n_volumes, n_reps, n_classes,
                                         width, height, depth, colour_channels)

    from Segmentation.model.vnet import VNet
    vnet = VNet(colour_channels, n_classes)

    from tensorflow.keras.optimizers import Adam
    from Segmentation.utils.training_utils import tversky_loss
    vnet.compile(optimizer=Adam(0.001),
                 loss=tversky_loss,
                 metrics=['categorical_crossentropy'],
                 experimental_run_tf_function=False)

    print(volumes.shape)
    print(one_hots.shape)

    vnet.fit(volumes, one_hots, epochs=1)

    pred = vnet.predict(volumes)

    print(pred.shape)
    print(volumes.shape)
    print(one_hots.shape)

    assert pred.shape == one_hots.shape


def test_tiny_vnet():
    from Segmentation.data_gen.toy_volume_gen import get_test_volumes

    n_volumes, n_reps, n_classes = 300, 4, 5
    width, height, depth = 28, 28, 16
    colour_channels = 3
    volumes, one_hots = get_test_volumes(n_volumes, n_reps, n_classes,
                                         width, height, depth, colour_channels)

    from Segmentation.model.tiny_vnet import VNet_Tiny as VNet
    vnet = VNet(colour_channels, n_classes)

    from tensorflow.keras.optimizers import Adam
    from Segmentation.utils.training_utils import tversky_loss
    vnet.compile(optimizer=Adam(0.00001),
                 loss=tversky_loss,
                 metrics=['categorical_crossentropy'],
                 experimental_run_tf_function=False)

    print(volumes.shape)
    print(one_hots.shape)

    vnet.fit(volumes, one_hots, epochs=5)

    pred = vnet.predict(volumes)

    print(pred.shape)
    print(volumes.shape)
    print(one_hots.shape)

    assert pred.shape == one_hots.shape

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.getcwd())

    # test_standard_vnet()
    test_tiny_vnet()
