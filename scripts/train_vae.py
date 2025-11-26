# scripts/train_vae.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from io_utils.caves_pt_utils import iter_cave_paths, load_raw_cave, get_3d_volume_from_raw, OUTLIER_FILES


class CaveVolumeDataset(Dataset):
    """
    Loads full_volume (1,D,H,W) from .pt, squeezes to (D,H,W),
    then resamples to fixed size (D_target,H_target,W_target) and
    returns as float tensor in {0,1} with shape (1,Dt,Ht,Wt).
    """

    def __init__(
        self,
        caves_root: Path,
        target_shape: Tuple[int, int, int] = (32, 32, 32),
        max_samples: int | None = None,
        skip_outliers: bool = True,
    ):
        self.paths = list(iter_cave_paths(caves_root, skip_outliers=skip_outliers))
        if max_samples is not None:
            self.paths = self.paths[:max_samples]
        self.target_shape = target_shape

    def __len__(self) -> int:
        return len(self.paths)

    def _resample(self, v3d: torch.Tensor) -> torch.Tensor:
        """
        v3d: (D,H,W) uint8 -> resampled (Dt,Ht,Wt) float32
        """
        Dt, Ht, Wt = self.target_shape
        # (1,1,D,H,W)
        x = v3d.unsqueeze(0).unsqueeze(0).float()
        x = F.interpolate(
            x,
            size=(Dt, Ht, Wt),
            mode="trilinear",
            align_corners=False,
        )
        # threshold-ish but keep as float in [0,1]
        x = torch.clamp(x, 0.0, 1.0)
        return x  # (1,1,Dt,Ht,Wt)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        obj = load_raw_cave(p)
        v3d = get_3d_volume_from_raw(obj)  # (D,H,W) uint8 {0,1}
        x = self._resample(v3d)            # (1,1,Dt,Ht,Wt)
        return x.squeeze(0)                # (1,Dt,Ht,Wt)


# --- 3D VAE model --- #

class Encoder3D(nn.Module):
    def __init__(self, z_dim: int = 128):
        super().__init__()
        # Input: (B,1,32,32,32)
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=2, padding=1),  # -> 16^3
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), # -> 8^3
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),# -> 4^3
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4 * 4, z_dim)

    def forward(self, x):
        h = self.conv(x)        # (B,128,4,4,4)
        h = h.flatten(1)        # (B, 128*4*4*4)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder3D(nn.Module):
    def __init__(self, z_dim: int = 128):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 128 * 4 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),  # 4 -> 8
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),   # 8 -> 16
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),   # 16 -> 32
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),           # keep 32^3
        )

    def forward(self, z):
        h = self.fc(z)                     # (B,128*4*4*4)
        h = h.view(-1, 128, 4, 4, 4)
        x_recon_logits = self.deconv(h)    # (B,1,32,32,32)
        return x_recon_logits


class VAE3D(nn.Module):
    def __init__(self, z_dim: int = 128):
        super().__init__()
        self.encoder = Encoder3D(z_dim=z_dim)
        self.decoder = Decoder3D(z_dim=z_dim)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


def loss_vae(recon_logits, x, mu, logvar):
    # x and logits: (B,1,32,32,32)
    recon_loss = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="sum")
    # KL divergence (per standard VAE)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld, recon_loss, kld


def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = total_rec = total_kld = 0.0
    n = 0
    for x in loader:
        x = x.to(device)  # (B,1,32,32,32)
        opt.zero_grad()
        logits, mu, logvar = model(x)
        loss, rec, kld = loss_vae(logits, x, mu, logvar)
        loss.backward()
        opt.step()
        bs = x.size(0)
        total_loss += loss.item()
        total_rec += rec.item()
        total_kld += kld.item()
        n += bs
    return total_loss / n, total_rec / n, total_kld / n


def main():
    ap = argparse.ArgumentParser(description="Train 3D VAE on cave .pt dataset.")
    ap.add_argument("--caves-root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--z-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caves_root = Path(args.caves_root)

    ds = CaveVolumeDataset(caves_root, target_shape=(32, 32, 32), max_samples=args.max_samples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = VAE3D(z_dim=args.z_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Dataset size: {len(ds)} caves")
    for ep in range(1, args.epochs + 1):
        loss, rec, kld = train_epoch(model, dl, opt, device)
        print(f"Epoch {ep}: loss={loss:.4f}, recon={rec:.4f}, kld={kld:.4f}")

    out_path = Path(f"vae3d_z{args.z_dim}.pt")
    torch.save(model.state_dict(), out_path)
    print(f"Saved model weights to {out_path}")


if __name__ == "__main__":
    main()
