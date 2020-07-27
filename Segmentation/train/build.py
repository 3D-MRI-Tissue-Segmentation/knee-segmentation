from Segmentation.model.vnet_old import VNet


def build_model(num_channels, num_classes, name, **kwargs):
    """
    Builds standard vnet for 3D
    """
    model = VNet(num_channels, num_classes, name=name, **kwargs)
    return model
