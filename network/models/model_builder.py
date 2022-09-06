from network.encoders import *
from network.targets import *
from network.models.model import Model

def build_model(args):
    """
    Build universial encoders representations models.
    The combinations of different embedding, encoders,
    and targets layers yield pretrained models of different
    properties.
    We could select suitable one for downstream tasks.
    """

    encoder = None
    target = None

    model = Model(args, encoder, target)

    return model
