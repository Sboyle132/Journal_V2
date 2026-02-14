# %% [markdown]
# # DBN-RBM Autoencoder for Hyperspectral Anomaly Detection
# Adapted from WaveletCNN methodology
#
# ## ⚠️ IMPORTANT: Fix NumPy Compatibility First!
#
# **If you see "RuntimeError: Numpy is not available" or NumPy warnings:**
#
# ```bash
# pip uninstall numpy
# pip install "numpy<2.0"
# # Then restart your kernel/Python session
# ```
#
# ## Requirements
# ```bash
# pip install "numpy<2.0"  # CRITICAL - Must be version 1.x
# pip install torch torchvision tqdm scikit-learn joblib matplotlib seaborn
# pip install hypso  # For HYPSO data loading
# ```

# %% [markdown]
# ## Imports

# %%
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
import random
import joblib
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from hypso.load import load_l1a_nc_cube, load_l1b_nc_cube, load_l1c_nc_cube, load_l1d_nc_cube

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# %% [markdown]
# ## Device Setup

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(DEVICE)
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached GPU memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# %% [markdown]
# ## Model Components

# %%
class GaussianRBM(nn.Module):
    def __init__(self, n_visible, n_hidden, learning_rate=0.01, k=1, sigma=1.0):
        super(GaussianRBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.sigma = sigma
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        activation = F.linear(v, self.W, self.h_bias)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, h):
        activation = F.linear(h, self.W.t(), self.v_bias)
        return activation, activation

    def forward(self, v):
        p_h, _ = self.sample_h(v)
        return p_h

    def free_energy(self, v):
        vbias_term = torch.matmul(v, self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)
        return -hidden_term - vbias_term

    def contrastive_divergence(self, v0):
        p_h0, h0 = self.sample_h(v0)
        h_k = h0
        for _ in range(self.k):
            v_k, _ = self.sample_v(h_k)
            p_h_k, h_k = self.sample_h(v_k)
        positive_grad = torch.matmul(p_h0.t(), v0)
        negative_grad = torch.matmul(p_h_k.t(), v_k)
        return positive_grad, negative_grad, v_k


class BernoulliRBM(nn.Module):
    def __init__(self, n_visible, n_hidden, learning_rate=0.01, k=1):
        super(BernoulliRBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        activation = F.linear(v, self.W, self.h_bias)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, h):
        activation = F.linear(h, self.W.t(), self.v_bias)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def forward(self, v):
        p_h, _ = self.sample_h(v)
        return p_h

    def contrastive_divergence(self, v0):
        p_h0, h0 = self.sample_h(v0)
        h_k = h0
        for _ in range(self.k):
            p_v_k, v_k = self.sample_v(h_k)
            p_h_k, h_k = self.sample_h(v_k)
        positive_grad = torch.matmul(p_h0.t(), v0)
        negative_grad = torch.matmul(p_h_k.t(), p_v_k)
        return positive_grad, negative_grad, p_v_k


class DBNAutoencoder(nn.Module):
    def __init__(self, n_visible=120, n_hidden=13, use_tied_weights=False):
        super(DBNAutoencoder, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.use_tied_weights = use_tied_weights
        self.encoder = nn.Linear(n_visible, n_hidden)
        if use_tied_weights:
            self.decoder = None
            self.decoder_bias = nn.Parameter(torch.zeros(n_visible))
        else:
            self.decoder = nn.Linear(n_hidden, n_visible)
            self.decoder_bias = None

    def encode(self, x):
        return torch.sigmoid(self.encoder(x))

    def decode(self, z):
        if self.use_tied_weights:
            return F.linear(z, self.encoder.weight.t(), self.decoder_bias)
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def initialize_from_rbm(self, rbm):
        with torch.no_grad():
            self.encoder.weight.copy_(rbm.W)
            self.encoder.bias.copy_(rbm.h_bias)
            if self.use_tied_weights:
                self.decoder_bias.copy_(rbm.v_bias)
            else:
                self.decoder.weight.copy_(rbm.W.t())
                self.decoder.bias.copy_(rbm.v_bias)
        print("✅ Autoencoder initialized with pre-trained RBM weights")

# %% [markdown]
# ## Dataset and Data Loading

# %%
class PixelDataset(Dataset):
    def __init__(self, pixels, transform=None):
        self.pixels = torch.FloatTensor(pixels)
        self.transform = transform

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        pixel = self.pixels[idx]
        if self.transform:
            pixel = self.transform(pixel)
        return pixel


def load_hypso_image(image_path):
    basename = os.path.basename(image_path)
    if 'l1d' in basename:
        return load_l1d_nc_cube(image_path)
    elif 'l1c' in basename:
        return load_l1c_nc_cube(image_path)
    elif 'l1b' in basename:
        return load_l1b_nc_cube(image_path)
    return load_l1a_nc_cube(image_path)


def load_water_mask(label_path, water_class_label=2):
    labels = np.fromfile(label_path, dtype=np.uint8).reshape((598, 1092)) - 1
    return (labels == water_class_label)


def sample_pixels_from_image(image, n_samples=100000, random_seed=42, water_mask=None):
    rng = np.random.default_rng(random_seed)
    H, W, C = image.shape
    pixels_flat = image.reshape(-1, C)

    if water_mask is not None:
        water_mask_flat = water_mask.reshape(-1)
        water_indices = np.where(water_mask_flat)[0]
        if len(water_indices) == 0:
            raise ValueError("No water pixels found in the mask!")
        print(f"Water mask applied: {len(water_indices)} / {len(pixels_flat)} pixels are water "
              f"({100*len(water_indices)/len(pixels_flat):.1f}%)")
        n_samples = min(n_samples, len(water_indices))
        sampled_indices = rng.choice(water_indices, size=n_samples, replace=False)
    else:
        n_samples = min(n_samples, pixels_flat.shape[0])
        sampled_indices = rng.choice(pixels_flat.shape[0], size=n_samples, replace=False)

    return pixels_flat[sampled_indices]


def preprocess_pixels(pixels, fit_scaler=True, scaler=None):
    if fit_scaler:
        scaler = StandardScaler()
        normalized_pixels = scaler.fit_transform(pixels)
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit_scaler=False")
        normalized_pixels = scaler.transform(pixels)
    return normalized_pixels, scaler

# %% [markdown]
# ## Training Functions

# %%
def train_rbm(rbm, data_loader, n_epochs=50, learning_rate=0.01,
              momentum=0.5, weight_decay=0.0001, device='cuda'):
    rbm = rbm.to(device)
    velocity_W = torch.zeros_like(rbm.W)
    velocity_h = torch.zeros_like(rbm.h_bias)
    velocity_v = torch.zeros_like(rbm.v_bias)
    loss_history = []

    print(f"Training RBM: {rbm.n_visible} -> {rbm.n_hidden}")

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        batch_count = 0
        current_momentum = momentum if epoch < 5 else 0.9

        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            batch = batch.to(device)
            batch_size = batch.shape[0]
            pos_grad, neg_grad, reconstruction = rbm.contrastive_divergence(batch)
            grad_W = (pos_grad - neg_grad) / batch_size
            grad_h = torch.mean(rbm.sample_h(batch)[0] - rbm.sample_h(reconstruction)[0], dim=0)
            grad_v = torch.mean(batch - reconstruction, dim=0)
            velocity_W = current_momentum * velocity_W + learning_rate * grad_W
            velocity_h = current_momentum * velocity_h + learning_rate * grad_h
            velocity_v = current_momentum * velocity_v + learning_rate * grad_v
            with torch.no_grad():
                rbm.W += velocity_W - weight_decay * rbm.W
                rbm.h_bias += velocity_h
                rbm.v_bias += velocity_v
            error = torch.mean((batch - reconstruction) ** 2)
            epoch_loss += error.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Reconstruction Error: {avg_loss:.6f}")

    print(f"✅ RBM training complete. Final loss: {loss_history[-1]:.6f}")
    return loss_history


def train_autoencoder(model, data_loader, n_epochs=50, learning_rate=0.001,
                      weight_decay=0.0005, device='cuda', patience=10):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0

    print("\nFine-tuning Autoencoder")

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction, z = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.6f} - LR: {current_lr:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"✅ Autoencoder fine-tuning complete. Best loss: {best_loss:.6f}")
    return loss_history

# %% [markdown]
# ## Anomaly Detection

# %%
def compute_reconstruction_error(model, image, scaler, device='cuda', batch_size=4096, water_mask=None):
    """
    Compute per-pixel reconstruction RMSE and latent codes.

    Returns
    -------
    rmse_map          : (H, W)        euclidean reconstruction error per pixel
    reconstructed_image: (H, W, C)
    latent_codes      : (H, W, n_hidden)  hidden-layer activations from the encoder
    """
    model.eval()
    model = model.to(device)
    H, W, C = image.shape
    n_hidden = model.n_hidden
    pixels_flat = image.reshape(-1, C)
    rmse_per_pixel    = np.zeros(H * W,          dtype=np.float32)
    reconstructions_flat = np.zeros_like(pixels_flat)
    latent_flat       = np.zeros((H * W, n_hidden), dtype=np.float32)

    def _run_batches(pixels_tensor):
        recons, codes = [], []
        with torch.no_grad():
            for i in tqdm(range(0, len(pixels_tensor), batch_size), desc="Computing reconstruction"):
                batch = pixels_tensor[i:i+batch_size].to(device)
                recon, z = model(batch)
                recons.append(recon.cpu())
                codes.append(z.cpu())
        return (torch.cat(recons, dim=0).numpy(),
                torch.cat(codes,  dim=0).numpy())

    if water_mask is not None:
        water_mask_flat = water_mask.reshape(-1)
        water_indices   = np.where(water_mask_flat)[0]
        print(f"Computing reconstruction for {len(water_indices):,} water pixels only...")
        water_pixels = pixels_flat[water_indices]
        water_norm, _ = preprocess_pixels(water_pixels, fit_scaler=False, scaler=scaler)
        recons, codes  = _run_batches(torch.FloatTensor(water_norm))
        recons_orig    = scaler.inverse_transform(recons)
        # Euclidean distance per pixel (L2 across bands)
        rmse_per_pixel[water_indices]  = np.sqrt(np.sum((water_pixels - recons_orig) ** 2, axis=1))
        reconstructions_flat[water_indices] = recons_orig
        latent_flat[water_indices]     = codes
        rmse_per_pixel[~water_mask_flat] = 0.0
    else:
        pixels_norm, _ = preprocess_pixels(pixels_flat, fit_scaler=False, scaler=scaler)
        recons, codes  = _run_batches(torch.FloatTensor(pixels_norm))
        recons_orig    = scaler.inverse_transform(recons)
        rmse_per_pixel = np.sqrt(np.sum((pixels_flat - recons_orig) ** 2, axis=1))
        reconstructions_flat = recons_orig
        latent_flat    = codes

    return (rmse_per_pixel.reshape(H, W),
            reconstructions_flat.reshape(H, W, C),
            latent_flat.reshape(H, W, n_hidden))


def create_anomaly_mask(rmse_map, threshold_percentile=95, water_mask=None):
    if water_mask is not None:
        water_rmse_values = rmse_map[water_mask]
        if len(water_rmse_values) == 0:
            raise ValueError("No water pixels to compute threshold!")
        threshold = np.percentile(water_rmse_values, threshold_percentile)
    else:
        threshold = np.percentile(rmse_map, threshold_percentile)

    anomaly_mask = (rmse_map > threshold).astype(np.uint8)
    if water_mask is not None:
        anomaly_mask[~water_mask] = 0
    return anomaly_mask, threshold


# %% [markdown]
# ## DSW Anomaly Detector

# %%
def _opt_novel_dsw(con, R, C):
    """
    Optimised Double Sliding Window (DSW) anomaly detector.
    Translated from MATLAB → C → Python (SBO 22.11.2023), vectorised inner loops.

    Parameters
    ----------
    con : dict  with keys height, width, win_dif, win_size, code_size
    R   : (height*width,)          scalar relevance per pixel (e.g. reconstruction error)
    C   : (height*width, code_size) code vector per pixel     (e.g. latent activations)
    """
    import math

    height    = con['height']
    width     = con['width']
    dif       = con['win_dif']
    size      = con['win_size']
    code      = con['code_size']

    win_total        = size + 2 * dif
    NEIGHBOURS       = win_total * win_total - size * size
    under_halfway    = dif + (size - 1) // 2
    BOUNDARY_TOP_LEFT     = win_total - dif   # exclusive upper bound of inner region
    BOUNDARY_BOTTOM_RIGHT = dif               # inclusive lower bound of inner region
    padding          = dif + (size - 1) // 2

    # Build the outer-ring mask once (True = neighbour pixel, False = inner window)
    window_mask = np.zeros((win_total, win_total), dtype=bool)
    s1 = slice(0, dif)
    s2 = slice(win_total - dif, win_total)
    window_mask[s1, :] = True
    window_mask[s2, :] = True
    window_mask[:, s1] = True
    window_mask[:, s2] = True

    anomaly_scores = np.zeros((height, width), dtype=np.float32)

    Rshaped      = R.reshape(height, width)
    Cshaped      = C.reshape(height, width, code)
    padded_data  = np.pad(Rshaped,  ((padding, padding), (padding, padding)),
                          mode='constant', constant_values=-1)
    padded_codes = np.pad(Cshaped,  ((padding, padding), (padding, padding), (0, 0)),
                          mode='constant', constant_values=0)

    for j in range(padding, height + padding):
        for i in range(padding, width + padding):
            window   = padded_data [j - padding : j + padding + 1,
                                    i - padding : i + padding + 1]
            window_c = padded_codes[j - padding : j + padding + 1,
                                    i - padding : i + padding + 1, :]

            mid_val = window[padding, padding]
            if mid_val == -1:
                continue

            # ── Mean of neighbours ────────────────────────────────────────
            valid   = window_mask & (window != -1)
            count   = int(np.sum(valid))
            if count == 0:
                continue
            r_total = float(np.sum(window[valid])) + (NEIGHBOURS - count)
            r_mean  = r_total / count

            # ── Standard deviation ────────────────────────────────────────
            work = window.copy()
            work[window == -1] = r_mean
            comp     = work - r_mean
            std_dev  = math.sqrt(float(np.sum(comp[window_mask] ** 2)) / count)

            # ── Weights  (zero if outlier neighbour or zero-value) ────────
            weight = np.where(np.abs(window - r_mean) > std_dev, 0.0,
                              np.where(window != 0, 1.0 / (window + 1e-12), 0.0))
            weight[~window_mask] = 0.0
            weight[window == -1] = 0.0

            # ── Code distance (L2 from centre code) ───────────────────────
            diff      = window_c - window_c[under_halfway, under_halfway]
            code_dist = np.sqrt(np.sum(diff ** 2, axis=2))

            # ── Final score ───────────────────────────────────────────────
            anomaly_scores[j - padding, i - padding] = (
                float(np.sum(code_dist[window_mask] * weight[window_mask]))
                * mid_val / count
            )

    return anomaly_scores


def compute_dsw_anomaly_scores(rmse_map, latent_codes, water_mask=None,
                                win_dif=3, win_size=1):
    """
    Run the DSW anomaly detector using the DBN's outputs.

    R (relevance) = per-pixel euclidean reconstruction error  (rmse_map)
    C (codes)     = per-pixel hidden-layer activations        (latent_codes)

    Non-water pixels are set to -1 so DSW treats them as missing/border.

    Parameters
    ----------
    rmse_map     : (H, W)           reconstruction error from compute_reconstruction_error
    latent_codes : (H, W, n_hidden) hidden activations from compute_reconstruction_error
    water_mask   : (H, W) bool      if provided, land/cloud pixels are excluded
    win_dif      : int              ring width  (gap between inner & outer window)
    win_size     : int              inner window size (1 = single centre pixel)

    Returns
    -------
    dsw_scores : (H, W)  raw DSW anomaly score per pixel
    """
    H, W        = rmse_map.shape
    n_hidden    = latent_codes.shape[2]

    # Use RMSE as R; mask non-water pixels with -1 so DSW ignores them
    R = rmse_map.copy().astype(np.float32)
    if water_mask is not None:
        R[~water_mask] = -1.0

    con = {
        'height':    H,
        'width':     W,
        'win_dif':   win_dif,
        'win_size':  win_size,
        'code_size': n_hidden,
    }

    print(f"Running DSW  (win_size={win_size}, win_dif={win_dif}, "
          f"code_size={n_hidden})  on {H}×{W} image...")
    dsw_scores = _opt_novel_dsw(con, R.reshape(-1), latent_codes.reshape(-1, n_hidden))

    if water_mask is not None:
        dsw_scores[~water_mask] = 0.0

    return dsw_scores

# %% [markdown]
# ## Visualization

# %%
def visualize_water_mask(image, water_mask, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    H, W, C = image.shape
    r_band, g_band, b_band = min(60, C-1), min(40, C-1), min(20, C-1)
    rgb = np.stack([image[:, :, r_band], image[:, :, g_band], image[:, :, b_band]], axis=-1)
    rgb_min, rgb_max = rgb.min(), rgb.max()
    rgb_normalized = np.clip((rgb - rgb_min) / (rgb_max - rgb_min + 1e-8), 0, 1)
    axes[0].imshow(rgb_normalized)
    axes[0].set_title(f'False Color RGB (Bands {r_band}, {g_band}, {b_band})', fontsize=14)
    axes[0].axis('off')
    water_rgb = np.zeros((H, W, 3))
    water_rgb[water_mask] = [0.2, 0.4, 0.8]
    water_rgb[~water_mask] = [0.5, 0.5, 0.5]
    axes[1].imshow(water_rgb)
    axes[1].set_title('Water Mask (Blue = Water, Gray = Land/Cloud)', fontsize=14)
    axes[1].axis('off')
    n_water = np.sum(water_mask)
    fig.text(0.5, 0.02, f'Water pixels: {n_water:,} / {water_mask.size:,} ({100*n_water/water_mask.size:.1f}%)',
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Water mask visualization saved to {save_path}")
    plt.show()


def visualize_results(original_image, rmse_map, anomaly_mask, threshold,
                      band_index=60, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    im0 = axes[0, 0].imshow(original_image[:, :, band_index], cmap='gray')
    axes[0, 0].set_title(f'Original Image (Band {band_index})', fontsize=14)
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    im1 = axes[0, 1].imshow(rmse_map, cmap='hot')
    axes[0, 1].set_title('Reconstruction RMSE Map', fontsize=14)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    im2 = axes[1, 0].imshow(anomaly_mask, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Anomaly Mask (Threshold: {threshold:.4f})', fontsize=14)
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, ticks=[0, 1])
    overlay = original_image[:, :, band_index].copy()
    overlay_rgb = np.stack([overlay, overlay, overlay], axis=-1)
    overlay_rgb = (overlay_rgb - overlay_rgb.min()) / (overlay_rgb.max() - overlay_rgb.min() + 1e-8)
    overlay_rgb[anomaly_mask == 1] = [1, 0, 0]
    axes[1, 1].imshow(overlay_rgb)
    axes[1, 1].set_title('Anomalies Overlay (Red)', fontsize=14)
    axes[1, 1].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Results saved to {save_path}")
    plt.show()


def plot_training_history(rbm_losses, ae_losses, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(rbm_losses, linewidth=2)
    axes[0].set_title('RBM Pre-training Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Reconstruction Error')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ae_losses, linewidth=2, color='orange')
    axes[1].set_title('Autoencoder Fine-tuning Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## Save and Load Functions

# %%
def save_model_and_config(model, scaler, config, save_dir, model_name="dbn_autoencoder"):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save({'model_state_dict': model.state_dict(), 'config': config}, model_path)
    scaler_path = os.path.join(save_dir, f"{model_name}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    config_path = os.path.join(save_dir, f"{model_name}_config.txt")
    with open(config_path, 'w') as f:
        f.write(f"DBN Autoencoder Configuration\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    print(f"✅ Model saved to {model_path}")
    print(f"✅ Scaler saved to {scaler_path}")
    print(f"✅ Config saved to {config_path}")


def load_model_and_scaler(model_path, scaler_path, device='cuda'):
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    model = DBNAutoencoder(
        n_visible=config['n_visible'],
        n_hidden=config['n_hidden'],
        use_tied_weights=config['use_tied_weights']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    scaler = joblib.load(scaler_path)
    print(f"✅ Model loaded from {model_path}")
    print(f"✅ Scaler loaded from {scaler_path}")
    return model, scaler, config

# %% [markdown]
# ## Configuration and File Paths

# %%
# ============================================================================
# CONFIGURATION AND FILE PATHS
# ============================================================================

# Define your data paths
raw_data_path = r'/home/samb/Coding/WaveletCNN/hsi_data/HYPSORAWDATA'
labels_path   = r'/home/samb/Coding/WaveletCNN/hsi_data/HYPSOLABELS'

# Choose data processing level: 'l1a', 'l1b', 'l1c', or 'l1d'
DATA_LEVEL = 'l1b'

# Water mask settings
USE_WATER_MASK   = True
WATER_CLASS_LABEL = 2  # 0=Cloud, 1=Land, 2=Sea/Water

# ── Full scene lists (base names, without level suffix) ───────────────────────
raw_data_files_h1 = [
    'adriatic_2024-07-18T08-53-45Z',           #  0
    'aeronetgalata_2025-01-02T08-18-16Z',       #  1
    'annapolis_2024-09-08T15-34-29Z',           #  2
    'aquawatchplymouth_2025-03-04T10-37-14Z',   #  3
    'barbados_2025-03-04T13-54-23Z',            #  4
    'capetown_2024-11-30T08-25-39Z',            #  5
    'casablanca_2024-11-19T10-27-26Z',          #  6
    'dubai_2024-11-21T06-24-16Z',               #  7
    'erie_2024-09-18T15-53-53Z',                #  8
    'eurasianplate1_2024-12-30T08-20-49Z',      #  9
    'eurasianplate3_2024-12-14T08-49-37Z',      # 10
    'flindersIsland_2024-11-09T23-33-38Z',      # 11
    'florida_2024-05-21T15-51-31Z',             # 12
    'florida_2024-05-30T15-34-21Z',             # 13
    'fredrikstad_2024-09-10T09-54-25Z',         # 14
    'frohavetnorth_2024-06-15T10-48-13Z',       # 15
    'greenbay_2024-10-01T16-22-32Z',            # 16
    'grizzlybay_2024-07-18T18-20-54Z',          # 17
    'hawkebay_2024-12-23T21-43-24Z',            # 18
    'kakhovka_2024-11-24T07-57-14Z',            # 19
    'kemigawa_2025-01-30T01-03-42Z',            # 20
    'kemigawa_2025-02-27T01-02-47Z',            # 21
    'kvaloya_2024-06-17T10-00-18Z',             # 22
    'lagunaDeTerminos_2024-12-06T16-07-50Z',    # 23
    'longisland3_2024-07-15T14-49-22Z',         # 24
    'longisland_2024-06-25T14-52-40Z',          # 25
    'longisland_2024-08-26T14-51-29Z',          # 26
    'malpasdam_2025-02-27T23-12-04Z',           # 27
    'marmara_2024-04-16T08-16-38Z',             # 28
    'mvco_2025-03-02T14-55-47Z',               # 29
    'plocan_2024-12-28T11-00-05Z',              # 30
    'rogaland_2024-06-01T09-56-49Z',            # 31
    'sanfrancisco_2024-05-23T18-12-14Z',        # 32
    'sicilychannel_2024-12-28T09-23-44Z',       # 33
    'solbergstrand_2024-07-08T09-39-39Z',       # 34
    'tasmania_2024-12-04T23-39-49Z',            # 35
    'tenerife_2025-01-17T11-17-36Z',            # 36
    'trondheim_2024-09-11T09-27-22Z',           # 37
    'trondheim_2024-09-11T11-00-43Z',           # 38
    'trondheim_2024-09-14T09-42-48Z',           # 39
]

raw_data_files_h2 = [
    'aeronetgalata_2025-01-02T08-52-34Z',       #  0
    'aeronetgloria_2025-01-09T11-17-15Z',        #  1
    'aquawatchgrippsland_2025-03-09T00-16-17Z',  #  2
    'aquawatchlakehume_2025-03-08T00-09-52Z',    #  3
    'aquawatchmoreton_2025-01-22T00-11-33Z',     #  4
    'aquawatchspencer_2025-01-03T01-17-50Z',     #  5
    'ariake_2025-02-11T02-05-25Z',               #  6
    'blanca_2025-02-04T14-31-12Z',               #  7
    'bluenile_2025-01-25T08-23-16Z',             #  8
    'falklandsatlantic_2025-03-03T14-11-51Z',    #  9
    'frohavet_2025-02-25T11-26-39Z',             # 10
    'gobabeb_2025-02-02T09-24-52Z',              # 11
    'goddard_2025-01-09T16-07-16Z',              # 12
    'grizzlybay_2025-01-22T19-11-18Z',           # 13
    'gulfofcalifornia_2025-01-14T18-19-49Z',     # 14
    'gulfofcampeche_2025-03-03T17-04-38Z',       # 15
    'hypernetEstuary_2024-12-28T11-29-35Z',      # 16
    'image61N6E_2025-03-13T11-27-56Z',           # 17
    'image65N10E_2025-03-12T11-20-52Z',          # 18
    'kemigawa_2025-01-22T01-31-13Z',             # 19
    'lacrau_2024-12-26T11-15-54Z',               # 20
    'menindee_2025-01-02T01-10-11Z',             # 21
    'moby_2025-01-08T20-54-59Z',                 # 22
    'mvco_2025-02-05T15-52-26Z',                 # 23
    'sanjorgegulf_2025-03-08T14-39-52Z',         # 24
    'section7platform_2025-02-02T09-06-35Z',     # 25
    'sicilychannel_2025-01-10T09-48-59Z',        # 26
    'timaru_2025-01-21T22-39-53Z',               # 27
    'wilmington_2025-03-08T15-55-06Z',           # 28
    'zeebrugge_2025-03-08T11-00-54Z',            # 29
]

# ── Pick your training scene ──────────────────────────────────────────────────
# DATASET_MODE controls which list TRAINING_IMAGE_INDEX refers to:
#   "h1"   → index into raw_data_files_h1 (0-39)
#   "h2"   → index into raw_data_files_h2 (0-29)
#   "h1h2" → index into h1 + h2 combined  (0-69)
DATASET_MODE = "h1h2"

if DATASET_MODE == "h1":
    raw_data_files = raw_data_files_h1
elif DATASET_MODE == "h2":
    raw_data_files = raw_data_files_h2
elif DATASET_MODE == "h1h2":
    raw_data_files = raw_data_files_h1 + raw_data_files_h2
else:
    raise ValueError(f"Unknown DATASET_MODE: {DATASET_MODE}")

# ↓↓ CHANGE THIS to train on a different scene (see index comments above) ↓↓
TRAINING_IMAGE_INDEX = 1  # 37 = trondheim_2024-09-11T09-27-22Z  (in h1h2 mode)

# Resolve paths
training_image_basename = raw_data_files[TRAINING_IMAGE_INDEX]
training_image_filename  = f"{training_image_basename}-{DATA_LEVEL}.nc"
training_image_path      = os.path.join(raw_data_path, training_image_filename)
training_label_filename  = f"{training_image_basename}_labels.dat"
training_label_path      = os.path.join(labels_path, training_label_filename)

print(f"Selected [{TRAINING_IMAGE_INDEX}]: {training_image_filename}")
print(f"Data level: {DATA_LEVEL}")
if not os.path.exists(training_image_path):
    raise FileNotFoundError(f"Image not found: {training_image_path}")
if USE_WATER_MASK:
    if os.path.exists(training_label_path):
        print(f"Label file:  {training_label_filename}")
    else:
        print("⚠️ Label file not found — water masking disabled.")
        USE_WATER_MASK = False

# Model configuration
config = {
    'image_path':         training_image_path,
    'label_path':         training_label_path if USE_WATER_MASK else None,
    'use_water_mask':     USE_WATER_MASK,
    'water_class_label':  WATER_CLASS_LABEL,
    'data_level':         DATA_LEVEL,
    'n_bands':            120,
    'n_samples_train':    100000,
    'n_visible':          120,
    'n_hidden':           13,
    'use_tied_weights':   False,
    'rbm_epochs':         50,
    'rbm_learning_rate':  0.01,
    'rbm_k':              1,
    'rbm_momentum':       0.5,
    'rbm_weight_decay':   0.0001,
    'ae_epochs':          50,
    'ae_learning_rate':   0.001,
    'ae_weight_decay':    0.0005,
    'ae_patience':        10,
    'batch_size':         2048,
    'random_seed':        42,
    'threshold_percentile': 95,
    # DSW parameters
    'dsw_win_dif':   3,   # ring width  — larger = more context, slower
    'dsw_win_size':  1,   # inner window (1 = single centre pixel)
    'save_dir':           'results/dbn_anomaly_detection',
    'model_name':         'dbn_autoencoder_hypso',
}

os.makedirs(config['save_dir'], exist_ok=True)

print()
print("=" * 80)
print("DBN-RBM AUTOENCODER FOR HYPERSPECTRAL ANOMALY DETECTION")
print("=" * 80)
print("\nConfiguration:")
for key, value in config.items():
    print(f"  {key}: {value}")
print()

# %% [markdown]
# ## Main Execution Pipeline

# %%
def main():
    print("\n" + "=" * 80)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 80)

    print(f"Loading image: {os.path.basename(config['image_path'])}")
    start_time = time.time()
    image = load_hypso_image(config['image_path'])
    print(f"Image shape: {image.shape}  ({time.time()-start_time:.2f}s)")

    water_mask = None
    if config['use_water_mask'] and config['label_path'] is not None:
        print(f"\nLoading water mask: {os.path.basename(config['label_path'])}")
        water_mask = load_water_mask(config['label_path'], config['water_class_label'])
        n_water = int(np.sum(water_mask))
        print(f"Water pixels: {n_water:,} / {water_mask.size:,} ({100*n_water/water_mask.size:.1f}%)")
        mask_viz_path = os.path.join(config['save_dir'], f"{config['model_name']}_water_mask_visualization.png")
        visualize_water_mask(image, water_mask, save_path=mask_viz_path)

    print(f"\nSampling {config['n_samples_train']:,} {'water ' if water_mask is not None else ''}pixels...")
    sampled_pixels = sample_pixels_from_image(
        image, n_samples=config['n_samples_train'],
        random_seed=config['random_seed'], water_mask=water_mask
    )
    print(f"Sampled pixels shape: {sampled_pixels.shape}")

    print("\nNormalizing pixels...")
    normalized_pixels, scaler = preprocess_pixels(sampled_pixels, fit_scaler=True)
    print(f"Mean: {normalized_pixels.mean():.6f}  Std: {normalized_pixels.std():.6f}")

    train_dataset = PixelDataset(normalized_pixels)
    train_loader  = DataLoader(train_dataset, batch_size=config['batch_size'],
                               shuffle=True, num_workers=0, pin_memory=False)
    print(f"DataLoader: {len(train_dataset):,} samples")

    # ── RBM Pre-training ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 2: PRE-TRAINING RBM")
    print("=" * 80)
    rbm = GaussianRBM(n_visible=config['n_visible'], n_hidden=config['n_hidden'],
                      learning_rate=config['rbm_learning_rate'], k=config['rbm_k'])
    rbm_losses = train_rbm(rbm, train_loader, n_epochs=config['rbm_epochs'],
                           learning_rate=config['rbm_learning_rate'],
                           momentum=config['rbm_momentum'],
                           weight_decay=config['rbm_weight_decay'], device=device)

    # ── Autoencoder Fine-tuning ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 3: BUILDING AND FINE-TUNING AUTOENCODER")
    print("=" * 80)
    autoencoder = DBNAutoencoder(n_visible=config['n_visible'], n_hidden=config['n_hidden'],
                                  use_tied_weights=config['use_tied_weights'])
    autoencoder.initialize_from_rbm(rbm)
    ae_losses = train_autoencoder(autoencoder, train_loader, n_epochs=config['ae_epochs'],
                                   learning_rate=config['ae_learning_rate'],
                                   weight_decay=config['ae_weight_decay'],
                                   device=device, patience=config['ae_patience'])
    plot_path = os.path.join(config['save_dir'], f"{config['model_name']}_training_history.png")
    plot_training_history(rbm_losses, ae_losses, save_path=plot_path)

    # ── Save ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 4: SAVING MODEL")
    print("=" * 80)
    save_model_and_config(autoencoder, scaler, config, config['save_dir'], config['model_name'])

    # ── Anomaly Detection ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 5: ANOMALY DETECTION")
    print("=" * 80)
    rmse_map, _, latent_codes = compute_reconstruction_error(
        autoencoder, image, scaler, device=device,
        batch_size=config['batch_size'], water_mask=water_mask
    )

    if water_mask is not None:
        water_rmse = rmse_map[water_mask]
        print(f"RMSE (water) — Mean: {water_rmse.mean():.6f}  Std: {water_rmse.std():.6f}  "
              f"Min: {water_rmse.min():.6f}  Max: {water_rmse.max():.6f}  "
              f"Median: {np.median(water_rmse):.6f}")
    else:
        print(f"RMSE — Mean: {rmse_map.mean():.6f}  Std: {rmse_map.std():.6f}")

    # ── DSW scoring ───────────────────────────────────────────────────────────
    print("\nRunning DSW anomaly scoring with latent codes...")
    dsw_scores = compute_dsw_anomaly_scores(
        rmse_map, latent_codes,
        water_mask=water_mask,
        win_dif=config['dsw_win_dif'],
        win_size=config['dsw_win_size'],
    )

    # Threshold on DSW scores (same percentile approach, water pixels only)
    print(f"\nThresholding DSW scores ({config['threshold_percentile']}th percentile)...")
    anomaly_mask, threshold = create_anomaly_mask(
        dsw_scores,
        threshold_percentile=config['threshold_percentile'],
        water_mask=water_mask
    )
    n_anomalies = int(np.sum(anomaly_mask))
    denom = int(np.sum(water_mask)) if water_mask is not None else anomaly_mask.size
    print(f"DSW threshold: {threshold:.6f}")
    print(f"Anomalies:     {n_anomalies:,} / {denom:,} ({100*n_anomalies/denom:.2f}%)")

    # ── Visualize ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 6: VISUALIZATION")
    print("=" * 80)

    # Full comparison: RMSE map | DSW score map | anomaly mask | overlay
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    band_index = 60

    im0 = axes[0, 0].imshow(rmse_map, cmap='hot')
    axes[0, 0].set_title('Reconstruction Error (Euclidean distance)', fontsize=13)
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(dsw_scores, cmap='hot')
    axes[0, 1].set_title('DSW Anomaly Score (latent code + RMSE)', fontsize=13)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(anomaly_mask, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Anomaly Mask — DSW threshold: {threshold:.4f}', fontsize=13)
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, ticks=[0, 1])

    overlay = image[:, :, band_index].copy().astype(np.float32)
    overlay_rgb = np.stack([overlay, overlay, overlay], axis=-1)
    overlay_rgb = (overlay_rgb - overlay_rgb.min()) / (overlay_rgb.max() - overlay_rgb.min() + 1e-8)
    overlay_rgb[anomaly_mask == 1] = [1, 0, 0]
    axes[1, 1].imshow(overlay_rgb)
    axes[1, 1].set_title('Anomalies Overlay — Red (Band 60)', fontsize=13)
    axes[1, 1].axis('off')

    plt.tight_layout()
    results_path = os.path.join(config['save_dir'], f"{config['model_name']}_dsw_results.png")
    plt.savefig(results_path, dpi=150, bbox_inches='tight')
    print(f"✅ Results saved to {results_path}")
    plt.show()

    # Save arrays
    rmse_path  = os.path.join(config['save_dir'], f"{config['model_name']}_rmse_map.npy")
    dsw_path   = os.path.join(config['save_dir'], f"{config['model_name']}_dsw_scores.npy")
    mask_path  = os.path.join(config['save_dir'], f"{config['model_name']}_anomaly_mask.npy")
    codes_path = os.path.join(config['save_dir'], f"{config['model_name']}_latent_codes.npy")
    np.save(rmse_path,  rmse_map)
    np.save(dsw_path,   dsw_scores)
    np.save(mask_path,  anomaly_mask)
    np.save(codes_path, latent_codes)
    print(f"✅ RMSE map saved:      {rmse_path}")
    print(f"✅ DSW scores saved:    {dsw_path}")
    print(f"✅ Anomaly mask saved:  {mask_path}")
    print(f"✅ Latent codes saved:  {codes_path}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {config['save_dir']}")
    return autoencoder, scaler, config, rmse_map, dsw_scores, anomaly_mask

# %% [markdown]
# ## Inference on New Images

# %%
def detect_anomalies_in_new_image(image_path, model_path, scaler_path,
                                   threshold_percentile=95, device='cuda',
                                   label_path=None, water_class_label=2):
    print(f"\n{'='*80}\nDETECTING ANOMALIES IN NEW IMAGE\n{'='*80}")
    model, scaler, config = load_model_and_scaler(model_path, scaler_path, device)
    print(f"\nLoading image: {os.path.basename(image_path)}")
    image = load_hypso_image(image_path)
    print(f"Image shape: {image.shape}")

    water_mask = None
    if label_path is not None and os.path.exists(label_path):
        water_mask = load_water_mask(label_path, water_class_label)
        n_water = int(np.sum(water_mask))
        print(f"Water pixels: {n_water:,} / {water_mask.size:,} ({100*n_water/water_mask.size:.1f}%)")
        visualize_water_mask(image, water_mask,
                             save_path=image_path.replace('.nc', '_water_mask_visualization.png'))

    rmse_map, _ = compute_reconstruction_error(model, image, scaler, device=device,
                                                batch_size=4096, water_mask=water_mask)
    anomaly_mask, threshold = create_anomaly_mask(rmse_map, threshold_percentile, water_mask)

    n_anomalies = int(np.sum(anomaly_mask))
    denom = int(np.sum(water_mask)) if water_mask is not None else anomaly_mask.size
    if water_mask is not None:
        water_rmse = rmse_map[water_mask]
        print(f"RMSE (water) — Mean: {water_rmse.mean():.6f}  Std: {water_rmse.std():.6f}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Anomalies: {n_anomalies:,} / {denom:,} ({100*n_anomalies/denom:.2f}%)")

    visualize_results(image, rmse_map, anomaly_mask, threshold, band_index=60,
                      save_path=image_path.replace('.nc', '_anomaly_detection.png'))
    return rmse_map, anomaly_mask, threshold

# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    autoencoder, scaler, config, rmse_map, dsw_scores, anomaly_mask = main()
    print("\n✅ Training and inference complete!")
    print(f"Model saved to: {config['save_dir']}")

# %%
# # Apply to a new image (uncomment to use):
# rmse_new, mask_new, thresh_new = detect_anomalies_in_new_image(
#     image_path  = 'path/to/new/image.nc',
#     model_path  = 'results/dbn_anomaly_detection/dbn_autoencoder_hypso.pth',
#     scaler_path = 'results/dbn_anomaly_detection/dbn_autoencoder_hypso_scaler.pkl',
#     threshold_percentile = 95,
#     device      = device,
# )