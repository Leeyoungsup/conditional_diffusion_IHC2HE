import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.distributed import get_rank


class GaussianDiffusion(nn.Module):
    def __init__(self, dtype: torch.dtype, model, betas: np.ndarray, w: float, v: float, device: torch.device):
        super().__init__()
        self.dtype = dtype
        self.model = model.to(device)
        self.model.dtype = self.dtype
        self.betas = torch.tensor(betas, dtype=self.dtype)
        self.w = w
        self.v = v
        self.T = len(betas)
        self.device = device
        self.alphas = 1 - self.betas
        self.log_alphas = torch.log(self.alphas)

        self.log_alphas_bar = torch.cumsum(self.log_alphas, dim=0)
        self.alphas_bar = torch.exp(self.log_alphas_bar)

        self.log_alphas_bar_prev = F.pad(
            self.log_alphas_bar[:-1], [1, 0], 'constant', 0)
        self.alphas_bar_prev = torch.exp(self.log_alphas_bar_prev)
        self.log_one_minus_alphas_bar_prev = torch.log(
            1.0 - self.alphas_bar_prev)

        self.log_sqrt_alphas = 0.5 * self.log_alphas
        self.sqrt_alphas = torch.exp(self.log_sqrt_alphas)

        self.log_sqrt_alphas_bar = 0.5 * self.log_alphas_bar
        self.sqrt_alphas_bar = torch.exp(self.log_sqrt_alphas_bar)
        self.log_one_minus_alphas_bar = torch.log(1.0 - self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.exp(
            0.5 * self.log_one_minus_alphas_bar)

        self.tilde_betas = self.betas * \
            torch.exp(self.log_one_minus_alphas_bar_prev -
                      self.log_one_minus_alphas_bar)
        self.log_tilde_betas_clipped = torch.log(
            torch.cat((self.tilde_betas[1].view(-1), self.tilde_betas[1:]), 0))
        self.mu_coef_x0 = self.betas * \
            torch.exp(0.5 * self.log_alphas_bar_prev -
                      self.log_one_minus_alphas_bar)
        self.mu_coef_xt = torch.exp(
            0.5 * self.log_alphas + self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)
        self.vars = torch.cat((self.tilde_betas[1:2], self.betas[1:]), 0)
        self.coef1 = torch.exp(-self.log_sqrt_alphas)
        self.coef2 = self.coef1 * self.betas / self.sqrt_one_minus_alphas_bar
        self.sqrt_recip_alphas_bar = torch.exp(-self.log_sqrt_alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.exp(
            self.log_one_minus_alphas_bar - self.log_sqrt_alphas_bar)

    @staticmethod
    def _extract(coef: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        assert t.shape[0] == x_shape[0]
        neo_shape = torch.ones_like(torch.tensor(x_shape))
        neo_shape[0] = x_shape[0]
        neo_shape = neo_shape.tolist()
        coef = coef.to(t.device)
        chosen = coef[t]
        chosen = chosen.to(t.device)
        return chosen.reshape(neo_shape)

    def q_mean_variance(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
        var = self._extract(1.0 - self.sqrt_alphas_bar, t, x_0.shape)
        return mean, var

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eps = torch.randn_like(x_0, requires_grad=False)
        return self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 \
            + self._extract(self.sqrt_one_minus_alphas_bar,
                            t, x_0.shape) * eps, eps

    def q_posterior_mean_variance(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = self._extract(self.mu_coef_x0, t, x_0.shape) * x_0 \
            + self._extract(self.mu_coef_xt, t, x_t.shape) * x_t
        posterior_var_max = self._extract(self.tilde_betas, t, x_t.shape)
        log_posterior_var_min = self._extract(
            self.log_tilde_betas_clipped, t, x_t.shape)
        log_posterior_var_max = self._extract(
            torch.log(self.betas), t, x_t.shape)
        log_posterior_var = self.v * log_posterior_var_max + \
            (1 - self.v) * log_posterior_var_min
        neo_posterior_var = torch.exp(log_posterior_var)
        return posterior_mean, posterior_var_max, neo_posterior_var

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, cemb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C = x_t.shape[:2]
        assert t.shape == (B,)

        pred_eps_cond = self.model(x_t, mask, t, cemb)
        pred_eps_uncond = self.model(
            x_t, torch.zeros_like(mask), t, torch.zeros_like(cemb))
        pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond

        p_mean = self._predict_xt_prev_mean_from_eps(
            x_t, t.type(dtype=torch.long), pred_eps)
        p_var = self._extract(self.vars, t.type(dtype=torch.long), x_t.shape)
        return p_mean, p_var

    def _predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return self._extract(coef=self.sqrt_recip_alphas_bar, t=t, x_shape=x_t.shape) * x_t \
            - self._extract(coef=self.sqrt_one_minus_alphas_bar,
                            t=t, x_shape=x_t.shape) * eps

    def _predict_xt_prev_mean_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return self._extract(coef=self.coef1, t=t, x_shape=x_t.shape) * x_t \
            - self._extract(coef=self.coef2, t=t, x_shape=x_t.shape) * eps

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.p_mean_variance(x_t, t, mask, cemb)
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0
        return mean + torch.sqrt(var) * noise

    def sample(self, shape: tuple, mask: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        local_rank = 0
        if local_rank == 0:
            print('Start generating...')
        x_t = torch.randn(shape, device=self.device)
        tlist = torch.ones([x_t.shape[0]], device=self.device) * self.T
        for _ in tqdm(range(self.T), dynamic_ncols=True, disable=(local_rank % torch.cuda.device_count() != 0)):
            tlist -= 1
            with torch.no_grad():
                x_t = self.p_sample(x_t, tlist, mask, cemb)
        x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print('Ending sampling process...')
        return x_t

    def ddim_p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, prevt: torch.Tensor, eta: float, mask: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        B, C = x_t.shape[:2]
        assert t.shape == (B,)

        pred_eps_cond = self.model(x_t, mask, t, cemb)
        pred_eps_uncond = self.model(
            x_t, torch.zeros_like(mask), t, torch.zeros_like(cemb))
        pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond

        alphas_bar_t = self._extract(
            coef=self.alphas_bar, t=t, x_shape=x_t.shape)
        alphas_bar_prev = self._extract(
            coef=self.alphas_bar_prev, t=prevt + 1, x_shape=x_t.shape)
        sigma = eta * torch.sqrt((1 - alphas_bar_prev) / (1 -
                                 alphas_bar_t) * (1 - alphas_bar_t / alphas_bar_prev))
        p_var = sigma ** 2
        coef_eps = 1 - alphas_bar_prev - p_var
        coef_eps[coef_eps < 0] = 0
        coef_eps = torch.sqrt(coef_eps)
        p_mean = torch.sqrt(alphas_bar_prev) * (x_t - torch.sqrt(1 - alphas_bar_t)
                                                * pred_eps) / torch.sqrt(alphas_bar_t) + coef_eps * pred_eps
        return p_mean, p_var

    def ddim_p_sample(self, x_t: torch.Tensor, t: torch.Tensor, prevt: torch.Tensor, eta: float, mask: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.ddim_p_mean_variance(x_t, t.type(
            dtype=torch.long), prevt.type(dtype=torch.long), eta, mask, cemb)
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0
        return mean + torch.sqrt(var) * noise

    def ddim_sample(self, shape: tuple, num_steps: int, eta: float, select: str, mask: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        local_rank = 0
        if local_rank == 0:
            print('Start generating(ddim)...')

        if select == 'linear':
            tseq = list(np.linspace(0, self.T-1, num_steps).astype(int))
        elif select == 'quadratic':
            tseq = list(
                (np.linspace(0, np.sqrt(self.T), num_steps-1)**2).astype(int))
            tseq.insert(0, 0)
            tseq[-1] = self.T - 1
        else:
            raise NotImplementedError(
                f'There is no ddim discretization method called "{select}"')

        x_t = torch.randn(shape, device=self.device)
        tlist = torch.zeros([x_t.shape[0]], device=self.device)
        for i in tqdm(range(num_steps), dynamic_ncols=True, disable=(local_rank % torch.cuda.device_count() != 0)):
            with torch.no_grad():
                tlist = tlist * 0 + tseq[-1-i]
                if i != num_steps - 1:
                    prevt = torch.ones_like(
                        tlist, device=self.device) * tseq[-2-i]
                else:
                    prevt = - torch.ones_like(tlist, device=self.device)
                x_t = self.ddim_p_sample(x_t, tlist, prevt, eta, mask, cemb)
                torch.cuda.empty_cache()
        x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print('ending sampling process(ddim)...')
        return x_t

    def trainloss(self, x_0: torch.Tensor, mask: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        t = torch.randint(self.T, size=(x_0.shape[0],), device=self.device)
        x_t, eps = self.q_sample(x_0, t)
        pred_eps = self.model(x_t, mask, t, cemb)
        loss = F.mse_loss(pred_eps, eps, reduction='mean')
        return loss
