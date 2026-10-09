"""Safety-net model: image-conditioned Diffusion Policy (DDPM).

Design choices for maximum portability (must run on Iluvatar CoreX with
minimal adaptation):
  - pure PyTorch, no custom CUDA ops, no flash-attn, no compile
  - small CNN vision encoders (no pretrained weights required)
  - language conditioning via a learned per-task embedding (10 fixed
    instruction templates); instruction -> task_index mapping happens at
    eval time by string match. CLIP-text embedding is the upgrade path if
    paraphrases ever appear.
  - predicts an action chunk (horizon x 7) via DDPM denoising
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionPolicy(nn.Module):
    def __init__(self, n_tasks=10, state_dim=8, action_dim=7, horizon=16,
                 obs_dim=256, hidden=256, n_diffusion_steps=100, task_emb_dim=64):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_steps = n_diffusion_steps

        self.task_emb = nn.Embedding(n_tasks, task_emb_dim)
        self.vis_main = VisionEncoder(128)
        self.vis_wrist = VisionEncoder(128)
        self.obs_fuse = nn.Sequential(
            nn.Linear(128 + 128 + state_dim + task_emb_dim, obs_dim), nn.ReLU(),
            nn.Linear(obs_dim, obs_dim), nn.ReLU(),
        )
        self.time_emb = SinusoidalTimeEmb(hidden)
        self.den = nn.Sequential(
            nn.Linear(obs_dim + horizon * action_dim + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, horizon * action_dim),
        )

        betas = torch.linspace(1e-4, 2e-2, n_diffusion_steps, dtype=torch.float64)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("abar", abar.float())
        self.register_buffer("abar_prev", torch.cat([torch.ones(1, dtype=torch.float32), abar[:-1].float()]))

    def _obs(self, images, state, task_index):
        vm = self.vis_main(images["main"])
        vw = self.vis_wrist(images["wrist"])
        te = self.task_emb(task_index)
        x = torch.cat([vm, vw, state, te], dim=-1)
        return self.obs_fuse(x)

    def loss(self, images, state, task_index, action_chunk):
        B = action_chunk.shape[0]
        a0 = action_chunk.reshape(B, -1)
        t = torch.randint(0, self.n_steps, (B,), device=a0.device)
        eps = torch.randn_like(a0)
        ab = self.abar[t][:, None]
        xt = ab.sqrt() * a0 + (1 - ab).sqrt() * eps
        obs = self._obs(images, state, task_index)
        pred = self.den(torch.cat([obs, xt, self.time_emb(t)], dim=-1))
        return F.mse_loss(pred, eps)

    @torch.no_grad()
    def predict(self, images, state, task_index, n_steps=None):
        """DDPM reverse sampling -> (B, horizon, action_dim)."""
        n_steps = n_steps or self.n_steps
        B = state.shape[0]
        device = state.device
        obs = self._obs(images, state, task_index)
        x = torch.randn(B, self.horizon * self.action_dim, device=device)
        for i in reversed(range(n_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            pred = self.den(torch.cat([obs, x, self.time_emb(t)], dim=-1))
            ab = self.abar[i]
            abp = self.abar_prev[i]
            beta = self.betas[i]
            coef = (1 - abp) / (1 - ab)
            mean = (1.0 / self.alphas[i].sqrt()) * (x - (beta / (1 - ab).sqrt()) * pred)
            if i > 0:
                x = mean + (beta * coef).sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.view(B, self.horizon, self.action_dim)
