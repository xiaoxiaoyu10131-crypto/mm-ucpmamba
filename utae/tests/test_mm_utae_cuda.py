"""默认宽度与 128x128 输入的真实 CUDA 测试，不代表正式训练性能。"""

import pytest
import torch

from src.backbones.mm_utae import MMUTAE
from src.mm_utae_experiment import mm_utae_loss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.parametrize("head,auxiliary", [("legacy", False), ("logits", True)])
def test_default_architecture_cuda_forward_backward_and_cpu_equivalence(head, auxiliary):
    torch.manual_seed(83)
    torch.set_num_threads(2)
    model = MMUTAE(head_mode=head, auxiliary=auxiliary).cuda()
    inputs = [torch.randn(2, length, channels, 128, 128, device="cuda")
              for length, channels in [(2, 10), (3, 3), (2, 3)]]
    dates = {"batch_positions_s2": torch.tensor([[0, 30], [0, 30]], device="cuda"),
             "batch_positions_s1a": torch.tensor([[5, 20, 45], [5, 20, 45]], device="cuda"),
             "batch_positions_s1d": torch.tensor([[15, 40], [15, 40]], device="cuda")}
    target = torch.randint(0, 20, (2, 128, 128), device="cuda")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=19)
    torch.cuda.reset_peak_memory_stats()
    result = model(*inputs, **dates, return_aux=auxiliary)
    output, loss = mm_utae_loss(result, target, criterion, 0.5 if auxiliary else 0)
    assert output.shape == (2, 20, 128, 128)
    assert torch.isfinite(output).all() and torch.isfinite(loss)
    loss.backward()
    for branch in (model.s2_branch, model.s1a_branch, model.s1d_branch):
        grad = branch.temporal_encoder.attention_heads.Q.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.cuda.synchronize()
    print(f"CUDA {head}: loss={loss.item():.6f}, "
          f"peak_allocated={torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
    model.eval()
    with torch.no_grad():
        cuda_output = model(*inputs, **dates).cpu()
        cpu_model = MMUTAE(head_mode=head, auxiliary=auxiliary).eval()
        cpu_model.load_state_dict({key: value.cpu() for key, value in model.state_dict().items()})
        cpu_output = cpu_model(*[value.cpu() for value in inputs],
                              **{key: value.cpu() for key, value in dates.items()})
    torch.testing.assert_close(cuda_output, cpu_output, atol=2e-5, rtol=2e-4)
