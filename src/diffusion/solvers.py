import torch
from functools import partial


class EulerDiffEqSolver:
    def __init__(self, dynamic, score_fn, ode_sampling=False):
        self.dynamic = dynamic
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling

    def step(self, x_t, t, next_t, **kwargs):
        """
        Implement reverse SDE/ODE Euler solver
        """

        """
        x_mean = deterministic part
        x = x_mean + noise (yet another noise sampling)
        """
        dt = (next_t - t).view(-1, 1, 1)
        noise = torch.randn_like(x_t)
        score_output = self.score_fn(x_t=x_t, t=t, **kwargs)
        drift, diffusion = self.dynamic.reverse_params(x_t, t, score_output["score"], self.ode_sampling)
        x_mean = x_t + drift * dt
        x = x_mean + diffusion.view(-1, 1, 1) * torch.sqrt(-dt) * noise
        return {
            "x": x,
            "x_mean": x_mean,
            "x_0": score_output["x_0"],
        }


class HeunSolver:
    def __init__(self, dynamic, score_fn, ode_sampling=False):
        self.dynamic = dynamic
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling

    def step(self, x_t, t, next_t, **kwargs):
        """
        Implement reverse SDE/ODE Euler solver
        """

        """
        x_mean = deterministic part
        x = x_mean + noise (yet another noise sampling)
        """
        dt = (next_t - t).view(-1, 1, 1)
        drift, _, score_output = self.dynamic.reverse_params(x_t, t, partial(self.score_fn, **kwargs), ode_sampling=True)
        x_next = x_t + drift * dt

        if next_t[0] > 10e-5:
            drift_next, _, score_output = self.dynamic.reverse_params(x_next, next_t, partial(self.score_fn, **kwargs), ode_sampling=True)
            x_next = x_t + dt / 2 * (drift + drift_next)
        x_mean = x_next
        return {
            "x": x_mean,
            "x_mean": x_mean,
            "x_0": score_output["x_0"]
        }


class EdmSolver:
    def __init__(self, dynamic, score_fn, ode_sampling=False):
        self.dynamic = dynamic
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling

    def step(self, x_t, t, next_t, labels=None):
        """
        Implement reverse SDE/ODE Euler solver
        """

        """
        x_mean = deterministic part
        x = x_mean + noise (yet another noise sampling)
        """
        dt = (next_t - t).view(-1, 1, 1)
        drift, _ = self.dynamic.reverse_params(x_t, t, self.score_fn, self.ode_sampling)
        x_next = x_t + drift * dt

        if next_t[0] > 10e-5:  # TODO make vector processing
            drift_next, _ = self.dynamic.reverse_params(x_next, next_t, self.score_fn, self.ode_sampling)
            x_next = x_t + dt / 2 * (drift + drift_next)
        x_mean = x_next
        return {
            "x": x_mean,
            "x_mean": x_mean
        }


class DDIMSolver:
    def __init__(self, dynamic, score_fn, ode_sampling=False):
        self.dynamic = dynamic
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling

    def q_x_t_reverse(self, x_t, x_0, t, next_t):
        alpha_t = torch.clip(self.dynamic.marginal_params(t)["mu"] ** 2, min=0, max=1)
        alpha_t_1 = torch.clip(self.dynamic.marginal_params(next_t)["mu"] ** 2, min=0, max=1)

        sigma_t = torch.zeros_like(alpha_t)

        noise_t = (x_t - torch.sqrt(alpha_t) * x_0) / torch.sqrt(1 - alpha_t)
        mu = torch.sqrt(alpha_t_1) * x_0 + \
             torch.sqrt(1 - alpha_t_1 - sigma_t ** 2) * noise_t
        std = sigma_t
        return mu, std

    def step(self, x_t, t, next_t=None, **kwargs):
        """
        Implement reverse SDE/ODE Euler solver
        """

        """
        x_mean = deterministic part
        x = x_mean + noise (yet another noise sampling)
        """
        noise = torch.randn_like(x_t)
        x_0 = self.score_fn(x_t=x_t, t=t, **kwargs)["x_0"]
        mu, std = self.q_x_t_reverse(x_t, x_0, t, next_t)
        x = mu + std * noise
        return {
            "x": x,
            "x_mean": mu,
            "x_0": x_0,
        }


class DDPMSolver:
    def __init__(self, dynamic, score_fn, ode_sampling=False):
        self.dynamic = dynamic
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling

    def q_x_t_reverse(self, x_t, x_0, t, next_t=None, ):
        alpha_t = torch.clip(self.dynamic.marginal_params(t)["mu"] ** 2, min=0, max=1)
        alpha_t_1 = torch.clip(self.dynamic.marginal_params(next_t)["mu"] ** 2, min=0, max=1)
        beta_t = 1 - alpha_t / alpha_t_1

        mu = torch.sqrt(alpha_t_1) * beta_t / (1 - alpha_t) * x_0 + \
             torch.sqrt(1 - beta_t) * (1 - alpha_t_1) / (1 - alpha_t) * x_t
        std = torch.sqrt((1 - alpha_t_1) / (1 - alpha_t) * beta_t)
        return mu, std

    def step(self, x_t, t, next_t=None, **kwargs):
        """
        Implement reverse SDE/ODE Euler solver
        """

        """
        x_mean = deterministic part
        x = x_mean + noise (yet another noise sampling)
        """
        noise = torch.randn_like(x_t)
        x_0 = self.score_fn(x_t=x_t, t=t, **kwargs)["x_0"]
        mu, std = self.q_x_t_reverse(x_t, x_0, t, next_t)
        x = mu + std * noise
        return {
            "x": x,
            "x_mean": mu,
            "x_0": x_0,
        }
