import os
import torch
from torch import nn, FloatTensor


class EncNormalizer(nn.Module):
    def __init__(self, enc_path: str):
        super().__init__()

        if os.path.exists(enc_path):
            self._load_state_dict(enc_path)
        else:
            print(f"Warning: EncNormalizer state dict not found at {enc_path}")

    def _load_state_dict(self, enc_path: str):
        state_dict = torch.load(enc_path, map_location='cuda')

        self.enc_mean = nn.Parameter(
            state_dict['enc_mean'][None, None, :],
            requires_grad=False
        )
        self.enc_std = nn.Parameter(
            state_dict['enc_std'][None, None, :],
            requires_grad=False
        )

    def forward(self, *args, **kwargs):
        return nn.Identity()(*args, **kwargs)

    def normalize(self, encoding: FloatTensor) -> FloatTensor:
        enc_std = torch.where(self.enc_std < 1e-5, 1, self.enc_std)
        return (encoding - self.enc_mean) / enc_std

    def denormalize(self, pred_x_0: FloatTensor) -> FloatTensor:
        return pred_x_0 * self.enc_std + self.enc_mean
