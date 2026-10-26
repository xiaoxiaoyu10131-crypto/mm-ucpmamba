"""沿用 1016 完整 A+B 配置，仅将共享 ConvLSTM 改为 L-TAE。"""

import sys

from run_mm_ucpmamba_async import make_config as make_async_config
from train_semantic import main


def make_config(arguments):
    return make_async_config(arguments, model_name="mm-uptae")


if __name__ == "__main__":
    main(make_config(sys.argv[1:]))
