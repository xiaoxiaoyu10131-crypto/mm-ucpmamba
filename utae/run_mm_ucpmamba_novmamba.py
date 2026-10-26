"""沿用 1016 配置，仅移除共享 ConvLSTM 候选状态中的 VMamba。"""

import sys

from run_mm_ucpmamba_async import make_config as make_async_config
from train_semantic import main


def make_config(arguments):
    return make_async_config(arguments, model_name="mm-ucpmamba-novmamba")


if __name__ == "__main__":
    main(make_config(sys.argv[1:]))
