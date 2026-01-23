import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F


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
        
        # ⭐ 수치 안정성을 위한 epsilon
        self.eps = 1e-8
        
        self.alphas = 1 - self.betas
        self.log_alphas = torch.log(self.alphas.clamp(min=self.eps))

        self.log_alphas_bar = torch.cumsum(self.log_alphas, dim=0)
        self.alphas_bar = torch.exp(self.log_alphas_bar).clamp(min=self.eps, max=1.0)

        self.log_alphas_bar_prev = F.pad(self.log_alphas_bar[:-1], [1, 0], 'constant', 0)
        self.alphas_bar_prev = torch.exp(self.log_alphas_bar_prev).clamp(min=self.eps, max=1.0)
        self.log_one_minus_alphas_bar_prev = torch.log((1.0 - self.alphas_bar_prev).clamp(min=self.eps))

        # calculate parameters for q(x_t|x_{t-1})
        self.log_sqrt_alphas = 0.5 * self.log_alphas
        self.sqrt_alphas = torch.exp(self.log_sqrt_alphas).clamp(min=self.eps)

        # calculate parameters for q(x_t|x_0)
        self.log_sqrt_alphas_bar = 0.5 * self.log_alphas_bar
        self.sqrt_alphas_bar = torch.exp(self.log_sqrt_alphas_bar).clamp(min=self.eps)
        self.log_one_minus_alphas_bar = torch.log((1.0 - self.alphas_bar).clamp(min=self.eps))
        self.sqrt_one_minus_alphas_bar = torch.exp(0.5 * self.log_one_minus_alphas_bar).clamp(min=self.eps)

        # calculate parameters for q(x_{t-1}|x_t,x_0)
        self.tilde_betas = (self.betas * 
            torch.exp(self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar)).clamp(min=self.eps)
        self.log_tilde_betas_clipped = torch.log(
            torch.cat((self.tilde_betas[1].view(-1), self.tilde_betas[1:]), 0).clamp(min=self.eps))
        
        self.mu_coef_x0 = (self.betas * 
            torch.exp(0.5 * self.log_alphas_bar_prev - self.log_one_minus_alphas_bar)).clamp(min=self.eps)
        self.mu_coef_xt = torch.exp(
            0.5 * self.log_alphas + self.log_one_minus_alphas_bar_prev - 
            self.log_one_minus_alphas_bar).clamp(min=self.eps)
        
        self.vars = torch.cat((self.tilde_betas[1:2], self.betas[1:]), 0).clamp(min=self.eps)
        self.coef1 = torch.exp(-self.log_sqrt_alphas).clamp(min=self.eps)
        self.coef2 = (self.coef1 * self.betas / self.sqrt_one_minus_alphas_bar).clamp(min=self.eps)
        
        # calculate parameters for predicted x_0
        self.sqrt_recip_alphas_bar = torch.exp(-self.log_sqrt_alphas_bar).clamp(min=self.eps)
        self.sqrt_recipm1_alphas_bar = torch.exp(
            self.log_one_minus_alphas_bar - self.log_sqrt_alphas_bar).clamp(min=self.eps)

    @staticmethod
    def _extract(coef: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        Extract coefficients at timestep t
        """
        assert t.shape[0] == x_shape[0]
        
        neo_shape = torch.ones_like(torch.tensor(x_shape))
        neo_shape[0] = x_shape[0]
        neo_shape = neo_shape.tolist()
        coef = coef.to(t.device)
        
        # ⭐ Clamp timestep to valid range
        t_clamped = torch.clamp(t, 0, len(coef) - 1)
        chosen = coef[t_clamped]
        chosen = chosen.to(t.device)
        return chosen.reshape(neo_shape)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sample from q(x_t|x_0) with numerical stability
        """
        eps = torch.randn_like(x_0, requires_grad=False)
        
        sqrt_alphas_bar = self._extract(self.sqrt_alphas_bar, t, x_0.shape)
        sqrt_one_minus_alphas_bar = self._extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape)
        
        # ⭐ Ensure numerical stability
        sqrt_alphas_bar = torch.clamp(sqrt_alphas_bar, min=self.eps, max=1.0)
        sqrt_one_minus_alphas_bar = torch.clamp(sqrt_one_minus_alphas_bar, min=self.eps, max=1.0)
        
        x_t = sqrt_alphas_bar * x_0 + sqrt_one_minus_alphas_bar * eps
        
        # ⭐ Check for NaN
        if torch.isnan(x_t).any():
            print(f"⚠️ NaN in q_sample!")
            print(f"  t: {t[0].item()}")
            print(f"  sqrt_alphas_bar: {sqrt_alphas_bar.mean():.6f}")
            print(f"  x_0 range: [{x_0.min():.4f}, {x_0.max():.4f}]")
        
        return x_t, eps

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, **model_kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        calculate the parameters of p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)

        # Handle mask concatenation
        if 'mask' in model_kwargs:
            mask = model_kwargs.get('mask')
            kwargs_cond = {k: v for k, v in model_kwargs.items() if k != 'mask'}
            x_cond = torch.cat([x_t, mask], dim=1)
            pred_eps_cond = self.model(x_cond, t, **kwargs_cond)

            zero_mask = torch.zeros_like(mask)
            x_uncond = torch.cat([x_t, zero_mask], dim=1)
            kwargs_uncond = kwargs_cond.copy()
            pred_eps_uncond = self.model(x_uncond, t, **kwargs_uncond)
        else:
            kwargs_cond = model_kwargs.copy()
            pred_eps_cond = self.model(x_t, t, **kwargs_cond)
            kwargs_uncond = kwargs_cond.copy()
            pred_eps_uncond = self.model(x_t, t, **kwargs_uncond)

        # ⭐ Classifier-free guidance with gradient clipping
        pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
        pred_eps = torch.clamp(pred_eps, -10.0, 10.0)  # Prevent explosion

        # ⭐ NaN checks
        assert not torch.isnan(x_t).any(), f"NaN in x_t at t={t[0]}"
        assert not torch.isnan(pred_eps).any(), f"NaN in pred_eps at t={t[0]}"
        
        p_mean = self._predict_xt_prev_mean_from_eps(x_t, t.type(dtype=torch.long), pred_eps)
        p_var = self._extract(self.vars, t.type(dtype=torch.long), x_t.shape)
        
        return p_mean, p_var

    def _predict_xt_prev_mean_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        coef1 = self._extract(coef=self.coef1, t=t, x_shape=x_t.shape)
        coef2 = self._extract(coef=self.coef2, t=t, x_shape=x_t.shape)
        return coef1 * x_t - coef2 * eps

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, **model_kwargs) -> torch.Tensor:
        """
        sample x_{t-1} from p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        
        mean, var = self.p_mean_variance(x_t, t, **model_kwargs)
        
        # ⭐ Clamp variance
        var = torch.clamp(var, min=self.eps)
        
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0
        
        sample = mean + torch.sqrt(var) * noise
        
        # ⭐ Clamp output
        sample = torch.clamp(sample, -10.0, 10.0)
        
        return sample

    def sample(self, shape: tuple, **model_kwargs) -> torch.Tensor:
        """
        sample images from p_{theta}
        """
        local_rank = 0
        if local_rank == 0:
            print('Start generating...')
        if model_kwargs is None:
            model_kwargs = {}
        
        x_t = torch.randn(shape, device=self.device)
        tlist = torch.ones([x_t.shape[0]], device=self.device) * self.T
        
        for _ in tqdm(range(self.T), dynamic_ncols=True, disable=(local_rank % torch.cuda.device_count() != 0)):
            tlist -= 1
            with torch.no_grad():
                x_t = self.p_sample(x_t, tlist, **model_kwargs)
        
        x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print('ending sampling process...')
        return x_t

    def ddim_p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, prevt: torch.Tensor, 
                            eta: float, **model_kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        DDIM sampling with numerical stability
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)

        # Handle mask
        if 'mask' in model_kwargs:
            mask = model_kwargs.get('mask')
            kwargs_cond = {k: v for k, v in model_kwargs.items() if k != 'mask'}
            x_cond = torch.cat([x_t, mask], dim=1)
            pred_eps_cond = self.model(x_cond, t, **kwargs_cond)

            zero_mask = torch.zeros_like(mask)
            x_uncond = torch.cat([x_t, zero_mask], dim=1)
            kwargs_uncond = kwargs_cond.copy()
            pred_eps_uncond = self.model(x_uncond, t, **kwargs_uncond)
        else:
            kwargs_cond = model_kwargs.copy()
            pred_eps_cond = self.model(x_t, t, **kwargs_cond)
            kwargs_uncond = kwargs_cond.copy()
            pred_eps_uncond = self.model(x_t, t, **kwargs_uncond)

        # ⭐ Guidance with clamping
        pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
        pred_eps = torch.clamp(pred_eps, -10.0, 10.0)

        # ⭐ Extract with clamping
        alphas_bar_t = self._extract(coef=self.alphas_bar, t=t, x_shape=x_t.shape)
        alphas_bar_prev = self._extract(coef=self.alphas_bar_prev, t=prevt + 1, x_shape=x_t.shape)
        
        alphas_bar_t = torch.clamp(alphas_bar_t, min=self.eps, max=1.0)
        alphas_bar_prev = torch.clamp(alphas_bar_prev, min=self.eps, max=1.0)
        
        # ⭐ Variance calculation with clamping
        sigma = eta * torch.sqrt(
            ((1 - alphas_bar_prev) / (1 - alphas_bar_t) * (1 - alphas_bar_t / alphas_bar_prev)).clamp(min=self.eps)
        )
        p_var = (sigma ** 2).clamp(min=self.eps)
        
        coef_eps = (1 - alphas_bar_prev - p_var).clamp(min=0.0)
        coef_eps = torch.sqrt(coef_eps)
        
        # ⭐ Mean calculation
        sqrt_alphas_bar_t = torch.sqrt(alphas_bar_t.clamp(min=self.eps))
        sqrt_alphas_bar_prev = torch.sqrt(alphas_bar_prev.clamp(min=self.eps))
        sqrt_one_minus_alphas_bar_t = torch.sqrt((1 - alphas_bar_t).clamp(min=self.eps))
        
        p_mean = (sqrt_alphas_bar_prev * (x_t - sqrt_one_minus_alphas_bar_t * pred_eps) / 
                 sqrt_alphas_bar_t + coef_eps * pred_eps)
        
        # ⭐ Clamp output
        p_mean = torch.clamp(p_mean, -10.0, 10.0)
        
        return p_mean, p_var

    def ddim_p_sample(self, x_t: torch.Tensor, t: torch.Tensor, prevt: torch.Tensor, 
                     eta: float, **model_kwargs) -> torch.Tensor:
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        
        mean, var = self.ddim_p_mean_variance(x_t, t.type(dtype=torch.long), 
                                              prevt.type(dtype=torch.long), eta, **model_kwargs)
        
        var = torch.clamp(var, min=self.eps)
        
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0
        
        sample = mean + torch.sqrt(var) * noise
        sample = torch.clamp(sample, -10.0, 10.0)
        
        return sample

    def ddim_sample(self, shape: tuple, num_steps: int, eta: float, select: str, **model_kwargs) -> torch.Tensor:
        local_rank = 0
        if local_rank == 0:
            print('Start generating(ddim)...')
        if model_kwargs is None:
            model_kwargs = {}
        
        if select == 'linear':
            tseq = list(np.linspace(0, self.T-1, num_steps).astype(int))
        elif select == 'quadratic':
            tseq = list((np.linspace(0, np.sqrt(self.T), num_steps-1)**2).astype(int))
            tseq.insert(0, 0)
            tseq[-1] = self.T - 1
        else:
            raise NotImplementedError(f'Unknown ddim discretization: "{select}"')

        x_t = torch.randn(shape, device=self.device)
        tlist = torch.zeros([x_t.shape[0]], device=self.device)
        
        for i in tqdm(range(num_steps), dynamic_ncols=True, disable=(local_rank % torch.cuda.device_count() != 0)):
            with torch.no_grad():
                tlist = tlist * 0 + tseq[-1-i]
                if i != num_steps - 1:
                    prevt = torch.ones_like(tlist, device=self.device) * tseq[-2-i]
                else:
                    prevt = -torch.ones_like(tlist, device=self.device)
                
                x_t = self.ddim_p_sample(x_t, tlist, prevt, eta, **model_kwargs)
                torch.cuda.empty_cache()
        
        x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print('ending sampling process(ddim)...')
        return x_t

    def trainloss(self, x_0: torch.Tensor, **model_kwargs) -> torch.Tensor:
        """
        Calculate loss with numerical stability checks
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # ⭐ Input validation
        if torch.isnan(x_0).any() or torch.isinf(x_0).any():
            print("⚠️ NaN/Inf in input x_0")
            return torch.tensor(float('nan'), device=x_0.device, requires_grad=True)
        
        # Random timestep
        t = torch.randint(self.T, size=(x_0.shape[0],), device=self.device)
        t = torch.clamp(t, 0, self.T - 1)
        
        # Forward diffusion
        x_t, eps = self.q_sample(x_0, t)
        
        # ⭐ Check after q_sample
        if torch.isnan(x_t).any() or torch.isnan(eps).any():
            print(f"⚠️ NaN after q_sample at t={t[0].item()}")
            return torch.tensor(float('nan'), device=x_0.device, requires_grad=True)
        
        # Prepare input
        if 'mask' in model_kwargs:
            mask = model_kwargs.get('mask')
            if torch.isnan(mask).any() or torch.isinf(mask).any():
                print("⚠️ NaN/Inf in mask")
                return torch.tensor(float('nan'), device=x_0.device, requires_grad=True)
            
            kwargs = {k: v for k, v in model_kwargs.items() if k != 'mask'}
            x_input = torch.cat([x_t, mask], dim=1)
        else:
            kwargs = model_kwargs
            x_input = x_t
        
        # ⭐ Model forward
        pred_eps = self.model(x_input, t, **kwargs)
        
        # ⭐ Check model output
        if torch.isnan(pred_eps).any() or torch.isinf(pred_eps).any():
            print(f"⚠️ NaN/Inf in model output at t={t[0].item()}")
            print(f"  pred_eps range: [{pred_eps.min():.4f}, {pred_eps.max():.4f}]")
            return torch.tensor(float('nan'), device=x_0.device, requires_grad=True)
        
        # ⭐ Clamp predictions
        pred_eps = torch.clamp(pred_eps, -10.0, 10.0)
        
        # Calculate loss
        loss = F.mse_loss(pred_eps, eps, reduction='mean')
        
        # ⭐ Final check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN/Inf in final loss")
            return torch.tensor(float('nan'), device=x_0.device, requires_grad=True)
        
        return loss