"""
A simple and readable PyTorch rewrite of `VAE_CNN_300_samples.py`.

What this script does:
1) Loads x/y/frequency/S11 data from .npy files
2) Builds a small VAE for 12x12 unit-cell reconstruction
3) Builds an MLP predictor for S11 using [z_mean, frequency] as input
4) Trains both tasks jointly (reconstruction + KL + weighted S11 MSE)

This keeps the original model logic, but uses modern and clean PyTorch code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    x_path: str = "x_train.npy"
    y_path: str = "y_train.npy"
    freq_path: str = "fre_train.npy"
    s11_path: str = "S11_train.npy"

    latent_dim: int = 8
    freq_dim: int = 401
    s11_dim: int = 401

    batch_size: int = 128
    epochs: int = 200
    lr: float = 1e-3
    s11_loss_weight: float = 5.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Dataset
# -----------------------------
class MetasurfaceDataset(Dataset):
    def __init__(self, x_path: str, y_path: str, freq_path: str, s11_path: str) -> None:
        # Convert to float32 for stable training and lower memory usage.
        self.x = np.load(x_path).astype(np.float32)      # [N, 12, 12]
        self.y = np.load(y_path).astype(np.float32)      # [N, 12, 12]
        self.freq = np.load(freq_path).astype(np.float32)  # [N, 401]
        self.s11 = np.load(s11_path).astype(np.float32)  # [N, 401]

        if not (len(self.x) == len(self.y) == len(self.freq) == len(self.s11)):
            raise ValueError("All input arrays must have the same sample count.")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x[idx])
        y = torch.from_numpy(self.y[idx])
        freq = torch.from_numpy(self.freq[idx])
        s11 = torch.from_numpy(self.s11[idx])
        return x, y, freq, s11


# -----------------------------
# Model blocks
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2),  # 12x12 -> 6x6
        )
        self.fc = nn.Linear(32 * 6 * 6, 128)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # [B, 12, 12] -> [B, 1, 12, 12]
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        h = F.selu(self.fc(h))
        z_mean = self.fc_mean(h)
        z_logvar = self.fc_logvar(h)
        return z_mean, z_logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 32 * 6 * 6)
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 6x6 -> 12x12
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.selu(self.fc1(z))
        h = F.selu(self.fc2(h))
        h = h.view(-1, 32, 6, 6)
        y = self.conv(h)
        return y.squeeze(1)  # [B, 1, 12, 12] -> [B, 12, 12]


class Predictor(nn.Module):
    def __init__(self, latent_dim: int, freq_dim: int, s11_dim: int) -> None:
        super().__init__()
        in_dim = latent_dim + freq_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, s11_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_mean: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z_mean, freq], dim=1)
        return self.net(h)


class VAEWithPredictor(nn.Module):
    def __init__(self, latent_dim: int, freq_dim: int, s11_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.predictor = Predictor(latent_dim, freq_dim, s11_dim)

    @staticmethod
    def reparameterize(z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        # Keep the same scale idea as original code (epsilon multiplied by 0.1).
        eps = 0.1 * torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_logvar) * eps

    def forward(self, x: torch.Tensor, freq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        y_hat = self.decoder(z)
        s11_hat = self.predictor(z_mean, freq)  # follow original logic
        return y_hat, s11_hat, z_mean, z_logvar


# -----------------------------
# Loss
# -----------------------------
def kl_divergence(z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
    # Mean KL over batch.
    return -0.5 * torch.mean(torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1))


def compute_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    s11_hat: torch.Tensor,
    s11_true: torch.Tensor,
    z_mean: torch.Tensor,
    z_logvar: torch.Tensor,
    s11_loss_weight: float,
) -> Tuple[torch.Tensor, dict]:
    recon_loss = F.binary_cross_entropy(y_hat, y_true)
    kl_loss = kl_divergence(z_mean, z_logvar)
    s11_loss = F.mse_loss(s11_hat, s11_true)

    total = recon_loss + kl_loss + s11_loss_weight * s11_loss
    parts = {
        "recon": recon_loss.item(),
        "kl": kl_loss.item(),
        "s11": s11_loss.item(),
        "total": total.item(),
    }
    return total, parts


# -----------------------------
# Train
# -----------------------------
def train(cfg: Config) -> None:
    dataset = MetasurfaceDataset(cfg.x_path, cfg.y_path, cfg.freq_path, cfg.s11_path)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = VAEWithPredictor(cfg.latent_dim, cfg.freq_dim, cfg.s11_dim).to(cfg.device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr)

    print(f"Device: {cfg.device}")
    print(f"Samples: {len(dataset)}")

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        running = {"recon": 0.0, "kl": 0.0, "s11": 0.0, "total": 0.0}

        for x, y, freq, s11 in loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            freq = freq.to(cfg.device)
            s11 = s11.to(cfg.device)

            optimizer.zero_grad()
            y_hat, s11_hat, z_mean, z_logvar = model(x, freq)
            loss, parts = compute_loss(y_hat, y, s11_hat, s11, z_mean, z_logvar, cfg.s11_loss_weight)
            loss.backward()
            optimizer.step()

            for k in running:
                running[k] += parts[k]

        n_batches = len(loader)
        avg = {k: v / n_batches for k, v in running.items()}
        print(
            f"Epoch {epoch:04d}/{cfg.epochs} | "
            f"total={avg['total']:.6f} recon={avg['recon']:.6f} kl={avg['kl']:.6f} s11={avg['s11']:.6f}"
        )

    # Save weights (similar to original script behavior).
    torch.save(model.encoder.state_dict(), "encoder_weights.pt")
    torch.save(model.predictor.state_dict(), "predictor_weights.pt")
    torch.save(model.decoder.state_dict(), "decoder_weights.pt")
    torch.save(model.state_dict(), "vae_with_predictor_weights.pt")
    print("Saved weights: encoder/predictor/decoder/full model")


if __name__ == "__main__":
    config = Config()
    train(config)
