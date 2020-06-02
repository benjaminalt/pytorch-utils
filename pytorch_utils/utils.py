import torch

T = 100


def differentiable_len(trajectory_tensor: torch.Tensor) -> torch.Tensor:
    has_batch_dim = len(trajectory_tensor.size()) == 3
    if not has_batch_dim:
        trajectory_tensor = trajectory_tensor.unsqueeze(0)
    not_eos_binary = differentiable_where(trajectory_tensor[:, :, 0], torch.tensor(0.5), torch.tensor(0.0), torch.tensor(1.0))
    length = torch.sum(not_eos_binary, dim=1)
    if not has_batch_dim:
        return length.squeeze()
    return length


def differentiable_where(t: torch.Tensor, threshold: torch.Tensor, value_if: torch.Tensor, value_else: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid((t - threshold) * T) * (value_if - value_else) + value_else