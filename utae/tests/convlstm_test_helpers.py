"""在 CPU 上加载真实 ConvLSTM 门控，隔离 CUDA 空间算子的测试夹具。"""

import importlib.util
import sys
import types

import pytest
from torch import nn

from test_mm_ucpmamba_mid import ROOT


class CpuVSS(nn.Module):
    """测试专用可微空间算子，不替换实际 ConvLSTM 递推与门控。"""

    def __init__(self, dim, **kwargs):
        super().__init__()
        self.projection = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        return self.projection(x)


@pytest.fixture
def real_convlstm(mid_module, monkeypatch):
    vmamba = types.ModuleType("src.backbones.vmamba.vmamba")
    vmamba.VSSBlock = nn.Identity
    vmamba.LayerNorm2d = nn.Identity
    vmamba.Linear2d = nn.Linear
    sys.modules[vmamba.__name__] = vmamba
    name = "src.backbones.convlstm"
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/backbones/convlstm.py")
    convlstm = importlib.util.module_from_spec(spec)
    sys.modules[name] = convlstm
    spec.loader.exec_module(convlstm)
    monkeypatch.setattr(convlstm, "MSVSS", CpuVSS)
    return convlstm
