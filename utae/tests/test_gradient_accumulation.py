import torch

from src.learning.gradient_accumulation import (
    accumulation_window_size,
    is_accumulation_boundary,
)


def test_accumulation_boundaries_include_partial_tail():
    sizes = [accumulation_window_size(i, 5, 2) for i in range(5)]
    boundaries = [is_accumulation_boundary(i, 5, 2) for i in range(5)]

    assert sizes == [2, 2, 2, 2, 1]
    assert boundaries == [False, True, False, True, True]


def test_two_micro_batches_match_one_effective_batch_update():
    full_batch_model = torch.nn.Linear(1, 1, bias=False)
    accumulated_model = torch.nn.Linear(1, 1, bias=False)
    accumulated_model.load_state_dict(full_batch_model.state_dict())
    full_optimizer = torch.optim.SGD(full_batch_model.parameters(), lr=0.1)
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.1)
    inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    targets = 2 * inputs

    full_optimizer.zero_grad()
    torch.nn.functional.mse_loss(full_batch_model(inputs), targets).backward()
    full_optimizer.step()

    accumulated_optimizer.zero_grad()
    micro_batches = list(zip(inputs.chunk(2), targets.chunk(2)))
    for i, (micro_inputs, micro_targets) in enumerate(micro_batches):
        loss = torch.nn.functional.mse_loss(
            accumulated_model(micro_inputs), micro_targets
        )
        window_size = accumulation_window_size(i, len(micro_batches), 2)
        (loss / window_size).backward()
        if is_accumulation_boundary(i, len(micro_batches), 2):
            accumulated_optimizer.step()
            accumulated_optimizer.zero_grad()

    assert torch.allclose(
        accumulated_model.weight, full_batch_model.weight, atol=1e-7
    )
