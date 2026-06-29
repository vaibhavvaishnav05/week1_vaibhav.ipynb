"""
Generate visualizations for the denoising autoencoder:
  1. results_grid.png  -> noisy / denoised / clean comparison
  2. training_curves.png -> loss and PSNR over epochs
"""
import os
import glob
import random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR   = "/home/claude/denoising_project"
DATA_ROOT = "/tmp/extracted/mnist_png"
NOISE_STD = 0.5
DEVICE    = torch.device("cpu")

# --- same model definition ---
class ConvDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = ConvDenoiser().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "denoiser.pth"), map_location=DEVICE))
model.eval()

# --- pick one test image per digit 0-9 ---
samples = []
for d in range(10):
    files = glob.glob(os.path.join(DATA_ROOT, "testing", str(d), "*.png"))
    f = random.choice(files)
    clean = torch.from_numpy(np.array(Image.open(f).convert("L"), dtype=np.float32) / 255.0)
    clean = clean.unsqueeze(0)
    noise = torch.randn_like(clean) * NOISE_STD
    noisy = torch.clamp(clean + noise, 0, 1)
    samples.append((noisy, clean))

# --- run model ---
with torch.no_grad():
    noisy_batch = torch.stack([s[0] for s in samples]).to(DEVICE)
    denoised = model(noisy_batch).cpu()

# --- results grid: 3 rows (noisy / denoised / clean) x 10 cols ---
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
row_titles = ["Noisy\n(input)", "Denoised\n(output)", "Clean\n(target)"]
for col in range(10):
    imgs = [samples[col][0][0], denoised[col][0], samples[col][1][0]]
    for row in range(3):
        ax = axes[row, col]
        ax.imshow(imgs[row].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(row_titles[row], fontsize=11, rotation=0,
                          ha="right", va="center", labelpad=40)
fig.suptitle(f"Denoising Autoencoder on MNIST  (Gaussian noise, σ={NOISE_STD})",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "results_grid.png"), dpi=130, bbox_inches="tight")
plt.close()
print("saved results_grid.png")

# --- training curves ---
h = np.load(os.path.join(OUT_DIR, "history.npz"))
epochs = range(1, len(h["train_loss"]) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(epochs, h["train_loss"], "o-", label="Train loss")
ax1.plot(epochs, h["test_loss"], "s-", label="Test loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE loss")
ax1.set_title("Reconstruction Loss"); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(epochs, h["test_psnr"], "d-", color="green")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("PSNR (dB)")
ax2.set_title("Test PSNR (higher = better)"); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=130, bbox_inches="tight")
plt.close()
print("saved training_curves.png")
