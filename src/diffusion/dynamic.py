import torch
from torch import Tensor
from typing import Dict


class DynamicSDE:
    def __init__(self, scheduler, T: float):
        """Construct a Variance Preserving SDE."""
        self.T = T
        self.scheduler = scheduler

    def marginal_params(self, t: Tensor) -> Dict[str, Tensor]:
        mu, std = self.scheduler.params(t)
        return {
            "mu": mu,
            "std": std
        }

    def marginal(self, x_0: Tensor, t: Tensor) -> Dict[str, Tensor]:
        """
        Calculate marginal q(x_t|x_0)'s mean and std
        """
        params = self.marginal_params(t)
        mu, std = params["mu"], params["std"]
        noise = torch.randn_like(x_0)
        x_t = x_0 * mu + noise * std
        score = -noise / params["std"]
        return {
            "x_t": x_t,
            "noise": noise,
            "mu": mu,
            "std": std,
            "score": score,
        }

    def reverse_params(self, x_t, t, score, ode_sampling=False):
        beta_t = self.scheduler.beta_t(t)
        drift_sde = (-1) / 2 * beta_t[:, None, None] * x_t
        diffuson_sde = torch.sqrt(beta_t)
        
        if ode_sampling:
            drift = drift_sde - (1 / 2) * beta_t[:, None, None] * score
            diffusion = 0
        else:
            drift = drift_sde - beta_t[:, None, None] * score
            diffusion = diffuson_sde
        return drift, diffusion
