import pytest
import torch
import torch.nn as nn

from src.learning.weight_init import (
    find_final_batchnorm,
    reset_final_batchnorm_identity,
)


class ToyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.earlier = nn.BatchNorm2d(20)
        self.head = nn.Sequential(nn.Conv2d(8, 20, 1), nn.BatchNorm2d(20))


def test_reset_final_batchnorm_changes_only_last_matching_bn():
    model = ToyClassifier()
    earlier_bn = model.earlier
    final_bn = model.head[1]
    with torch.no_grad():
        earlier_bn.weight.fill_(3)
        earlier_bn.bias.fill_(4)
        final_bn.weight.fill_(0.25)
        final_bn.bias.fill_(-2)
        final_bn.running_mean.fill_(5)
        final_bn.running_var.fill_(6)

    name = reset_final_batchnorm_identity(model, num_features=20)

    assert name == "head.1"
    torch.testing.assert_close(final_bn.weight, torch.ones_like(final_bn.weight))
    torch.testing.assert_close(final_bn.bias, torch.zeros_like(final_bn.bias))
    torch.testing.assert_close(final_bn.running_mean, torch.full_like(final_bn.running_mean, 5))
    torch.testing.assert_close(final_bn.running_var, torch.full_like(final_bn.running_var, 6))
    torch.testing.assert_close(earlier_bn.weight, torch.full_like(earlier_bn.weight, 3))
    torch.testing.assert_close(earlier_bn.bias, torch.full_like(earlier_bn.bias, 4))


def test_final_batchnorm_lookup_fails_clearly_when_classifier_bn_is_absent():
    model = nn.Sequential(nn.Conv2d(3, 8, 1), nn.BatchNorm2d(8))

    assert find_final_batchnorm(model, num_features=20) == (None, None)
    with pytest.raises(ValueError, match="num_features=20"):
        reset_final_batchnorm_identity(model, num_features=20)
