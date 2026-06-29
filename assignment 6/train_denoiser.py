"""
Denoising Autoencoder on MNIST
==============================
Trains a convolutional autoencoder to remove Gaussian noise from handwritten
digit images. The network never sees clean->clean; it learns the mapping
   noisy image  ->  clean image
which forces it to capture the underlying structure of digits rather than
memorising pixels.

Author: (project script)
"""

import os
import glob
import time
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_ROOT   = "/tmp/extracted/mnist_png"
OUT_DIR     = "/home/claude/denoising_project"
NOISE_STD   = 0.5      # std-dev of Gaussian noise added to [0,1] images
N_TRAIN     = 20000    # subset of the 60k for faster CPU training
N_TEST      = 2000
BATCH_SIZE  = 128
EPOCHS      = 12
LR          = 1e-3

# ----------------------------------------------------------------------
# Dataset: loads PNGs, returns (noisy, clean) pairs
# ----------------------------------------------------------------------
class MNISTDenoiseDataset(Dataset):
    def __init__(self, root, split, limit=None, noise_std=0.5):
        self.noise_std = noise_std
        pattern = os.path.join(root, split, "*", "*.png")
        files = sorted(glob.glob(pattern))
        random.shuffle(files)
        if limit is not None:
            files = files[:limit]
        self.files = files
        print(f"  {split}: {len(self.files)} images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        # clean image scaled to [0,1], shape (1,28,28)
        clean = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        clean = clean.unsqueeze(0)
        # add Gaussian noise, then clip back to valid range
        noise = torch.randn_like(clean) * self.noise_std
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        return noisy, clean

# ----------------------------------------------------------------------
# Model: a small convolutional autoencoder
# ----------------------------------------------------------------------
class ConvDenoiser(nn.Module):
    """
    Encoder compresses 28x28 -> 7x7 feature maps (the 'bottleneck'),
    decoder reconstructs back to 28x28. Skip-free, fully convolutional.
    """
    def __init__(self):
        super().__init__()
        # ---- Encoder ----
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
        )
        # ---- Decoder ----
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2,
                               padding=1, output_padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, stride=2,
                               padding=1, output_padding=1),   # 14 -> 28
            nn.Sigmoid(),   # output pixels in [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# ----------------------------------------------------------------------
# Metric: PSNR (peak signal-to-noise ratio), higher = better
# ----------------------------------------------------------------------
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(1.0 / mse).item()

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def main():
    print("Loading data...")
    train_ds = MNISTDenoiseDataset(DATA_ROOT, "training", N_TRAIN, NOISE_STD)
    test_ds  = MNISTDenoiseDataset(DATA_ROOT, "testing",  N_TEST,  NOISE_STD)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = ConvDenoiser().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "test_loss": [], "test_psnr": []}

    print("\nStarting training...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        # ---- train ----
        model.train()
        run_loss = 0.0
        for noisy, clean in train_dl:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            optimizer.zero_grad()
            out = model(noisy)
            loss = criterion(out, clean)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * noisy.size(0)
        train_loss = run_loss / len(train_ds)

        # ---- evaluate ----
        model.eval()
        test_loss, test_psnr = 0.0, 0.0
        with torch.no_grad():
            for noisy, clean in test_dl:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                out = model(noisy)
                test_loss += criterion(out, clean).item() * noisy.size(0)
                test_psnr += psnr(out, clean) * noisy.size(0)
        test_loss /= len(test_ds)
        test_psnr /= len(test_ds)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_psnr"].append(test_psnr)

        dt = time.time() - t0
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"train_loss {train_loss:.5f} | "
              f"test_loss {test_loss:.5f} | "
              f"test_PSNR {test_psnr:.2f} dB | {dt:.1f}s")

    # ---- save model + history ----
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "denoiser.pth"))
    np.savez(os.path.join(OUT_DIR, "history.npz"), **history)
    print("\nSaved model -> denoiser.pth")

    # ---- baseline: PSNR of noisy images vs clean (to show improvement) ----
    model.eval()
    base_psnr = 0.0
    with torch.no_grad():
        for noisy, clean in test_dl:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            base_psnr += psnr(noisy, clean) * noisy.size(0)
    base_psnr /= len(test_ds)
    print(f"\nNoisy-vs-clean baseline PSNR : {base_psnr:.2f} dB")
    print(f"Denoised-vs-clean final PSNR : {history['test_psnr'][-1]:.2f} dB")
    print(f"Improvement                  : +{history['test_psnr'][-1]-base_psnr:.2f} dB")

    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"Baseline (noisy) PSNR: {base_psnr:.2f} dB\n")
        f.write(f"Denoised PSNR: {history['test_psnr'][-1]:.2f} dB\n")
        f.write(f"Improvement: +{history['test_psnr'][-1]-base_psnr:.2f} dB\n")

if __name__ == "__main__":
    main()
