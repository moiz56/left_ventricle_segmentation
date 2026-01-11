from monai.networks.nets import AttentionUnet


def attention_unet(in_channels=1, out_channels=2):
    return AttentionUnet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2)
    )

def attention_unet_extended_channels(in_channels=1, out_channels=2):
    return AttentionUnet(
    spatial_dims=2,
    in_channels=in_channels,
    out_channels=out_channels,         # binary mask
    channels=(64, 128, 256, 512, 1048),
    strides=(2, 2, 2, 2),
    dropout=0.5
)

MODEL_REGISTRY = {
    "attention_unet": attention_unet,
    "attention_unet_extended": attention_unet_extended_channels
}

def get_model(name, in_channels=1, out_channels=2):
    """
    Returns the model instance given its name.
    Raises ValueError if model name is invalid.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](in_channels=in_channels, out_channels=out_channels)
