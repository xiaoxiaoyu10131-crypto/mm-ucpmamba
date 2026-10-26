"""1016 异步融合主路的无 VMamba 消融，仅替换共享时序单元。"""

from src.backbones.mm_ucpmamba_async import MMUCPMambaAsync
from src.backbones.uconvlstm import PlainConvLSTM


class MMUCPMambaNoVMamba(MMUCPMambaAsync):
    """保留七个 P-MTSA 和原投影，使用普通单向 ConvLSTM。"""

    def __init__(self, *args, **kwargs):
        if "temporal_encoder_class" in kwargs:
            raise ValueError("the no-VMamba model requires PlainConvLSTM")
        super().__init__(*args, temporal_encoder_class=PlainConvLSTM, **kwargs)
