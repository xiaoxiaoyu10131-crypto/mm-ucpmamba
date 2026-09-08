"""Public UCPMamba model entry."""

from src.backbones.convlstm import ConvLSTM
from src.backbones.utae import RecUNet


class UCPMamba(RecUNet):
    """RecUNet with MSVSS ConvLSTM and deepest-only phenology skip attention."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("temporal_encoder_class", ConvLSTM)
        kwargs.setdefault("skip_aggregation", "phenology")
        kwargs.setdefault("decoder_norm", "batch")
        super().__init__(*args, **kwargs)
