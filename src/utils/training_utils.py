import torch
from typing import Optional
from torch.nn.functional import cross_entropy


def mse_loss(inputs: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None):
    if mask is None:
        mask = torch.ones(
            (targets.shape[0], targets.shape[1]),
            device=inputs.device,
            requires_grad=False,
            dtype=torch.int64,
        )
    losses = torch.mean(torch.square(inputs - targets), dim=-1)
    losses = losses * mask
    loss = torch.sum(losses) / torch.sum(mask)
    return loss


def masked_mean(tensor, mask):
    return torch.sum(tensor * mask[:, :, None], dim=[0, 1]) / torch.sum(mask)


def masked_std(tensor, mask):
    mean = masked_mean(tensor, mask)
    return torch.sqrt(torch.sum(tensor ** 2 * mask[:, :, None], dim=[0, 1]) / torch.sum(mask) - mean ** 2)


def get_stat(z, mask):
    if mask is None:
        mask = torch.ones(
            (z.shape[0], z.shape[1]),
            device=z.device,
            requires_grad=False,
            dtype=torch.int64,
        )
    mean = masked_mean(z, mask)
    std = masked_std(z, mask)
    return torch.mean(mean), torch.mean(std)


def reconstruction_loss(target, prediction_scores, mask):
    if mask is None:
        return cross_entropy(
            input=prediction_scores.view(-1, prediction_scores.shape[-1]),
            target=target.view(-1),
        )

    ce_losses = cross_entropy(
        input=prediction_scores.view(-1, prediction_scores.shape[-1]),
        target=target.view(-1),
        reduce=False,
    )
    ce_losses = ce_losses * mask.reshape(-1)
    ce_loss = torch.sum(ce_losses) / torch.sum(mask)
    return ce_loss