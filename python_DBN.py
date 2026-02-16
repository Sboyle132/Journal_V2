# %% [markdown]
# # MAWDBN – Hyperspectral Anomaly Detection on ABU Dataset
#
# Implementation of the Modified Adaptive Weight DBN (MAWDBN) from:
#   Gundersen, Boyle, Orlandic – "An Improved Adaptive Weighted Deep Belief
#   Network Autoencoder for Hyperspectral Images", WHISPERS 2023.
#
# Two anomaly scores are computed and evaluated independently:
#   1. **RMSE**      – per-pixel reconstruction error from the DBN alone (no
#                      spatial post-processing)
#   2. **MAWDBN**    – DSW anomaly score β using the AWDBN weight formula
#                      (Eq. 2) with penalty factor pf = 0, i.e.:
#                        inlier  neighbour  →  wt_j = r_p / r_n_j
#                        outlier neighbour  →  wt_j = 0  (pf=0)
#
# ABU Dataset Format (.mat files):
#   'data' : (H, W, C)  float32  hyperspectral cube
#   'map'  : (H, W)     uint8    ground-truth anomaly mask  (1 = anomaly)
#
# Requirements:
#   pip install "numpy<2.0"   # CRITICAL – must be 1.x
#   pip install torch torchvision tqdm scikit-learn joblib matplotlib scipy

# %% [markdown]
# ## Imports

# %%
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
import random
import joblib
import scipy.io as sio


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
    print(f"  Device name : {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"  Allocated   : {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Cached      : {torch.cuda.memory_reserved()  / 1e9:.2f} GB")

# %% [markdown]
# ## Model Components

# %%
class GaussianRBM(nn.Module):
    """
    Single Gaussian-Bernoulli RBM.
    Visible units are real-valued (Gaussian); hidden units are Bernoulli.
    Trained with CD-1 (k=1 Gibbs steps).
    """
    def __init__(self, n_visible, n_hidden, k=1):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        self.k         = k
        self.W      = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        p = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p, torch.bernoulli(p)

    def sample_v(self, h):
        # Gaussian visible: reconstruct as mean (no noise added)
        act = F.linear(h, self.W.t(), self.v_bias)
        return act, act

    def forward(self, v):
        return self.sample_h(v)[0]

    def contrastive_divergence(self, v0):
        p_h0, h0 = self.sample_h(v0)
        h_k = h0
        for _ in range(self.k):
            v_k, _    = self.sample_v(h_k)
            p_hk, h_k = self.sample_h(v_k)
        pos = torch.matmul(p_h0.t(), v0)
        neg = torch.matmul(p_hk.t(), v_k)
        return pos, neg, v_k


class DBNAutoencoder(nn.Module):
    """
    3-layer DBN autoencoder: n_visible → n_hidden → n_visible.
    Paper: first and last layers have nb units, middle layer has 13 units.
    Encoder uses sigmoid activation (Bernoulli hidden units).
    Decoder is linear (Gaussian reconstruction).
    """
    def __init__(self, n_visible, n_hidden, use_tied_weights=False):
        super().__init__()
        self.n_visible        = n_visible
        self.n_hidden         = n_hidden
        self.use_tied_weights = use_tied_weights
        self.encoder = nn.Linear(n_visible, n_hidden)
        if use_tied_weights:
            self.decoder      = None
            self.decoder_bias = nn.Parameter(torch.zeros(n_visible))
        else:
            self.decoder      = nn.Linear(n_hidden, n_visible)
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
        print("  ✅ Autoencoder initialised from RBM weights")

# %% [markdown]
# ## Dataset

# %%
class PixelDataset(Dataset):
    def __init__(self, pixels):
        self.pixels = torch.FloatTensor(pixels)

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        return self.pixels[idx]

# %% [markdown]
# ## Data Loading

# %%
def load_abu_mat(mat_path):
    """
    Load an ABU .mat file.

    Returns
    -------
    image   : (H, W, C) float32
    gt_mask : (H, W)    uint8   (1 = anomaly), or None if the key is absent
    n_bands : int
    """
    mat = sio.loadmat(mat_path)

    # ── Data cube ─────────────────────────────────────────────────────────────
    if 'data' in mat:
        image = mat['data'].astype(np.float32)
    else:
        candidates = {k: v for k, v in mat.items()
                      if not k.startswith('_') and isinstance(v, np.ndarray)
                      and v.ndim == 3}
        if not candidates:
            raise KeyError(f"No 3-D array in {mat_path}. Keys: {list(mat.keys())}")
        key   = max(candidates, key=lambda k: candidates[k].size)
        print(f"  ⚠️  'data' key missing; using '{key}'")
        image = candidates[key].astype(np.float32)

    # Guard against (C, H, W) layout
    if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0))

    # ── Ground-truth mask ─────────────────────────────────────────────────────
    gt_mask = None
    if 'map' in mat:
        gt_mask = mat['map'].astype(np.uint8).squeeze()
        if gt_mask.shape != image.shape[:2]:
            gt_mask = gt_mask.T

    H, W, C = image.shape
    n_anom  = int(np.sum(gt_mask)) if gt_mask is not None else 0
    print(f"  Shape   : {image.shape}  (H × W × C)")
    if gt_mask is not None:
        print(f"  GT mask : {n_anom:,} anomaly pixels  ({100*n_anom/(H*W):.2f}%)")
    return image, gt_mask, C


def preprocess_pixels(pixels, fit_scaler=True, scaler=None):
    if fit_scaler:
        scaler     = StandardScaler()
        normalised = scaler.fit_transform(pixels)
    else:
        if scaler is None:
            raise ValueError("scaler must be supplied when fit_scaler=False")
        normalised = scaler.transform(pixels)
    return normalised, scaler

# %% [markdown]
# ## Training

# %%
def train_rbm(rbm, loader, n_epochs, learning_rate, momentum, weight_decay, device):
    """
    CD-1 pre-training with momentum (0.5 for first 5 epochs, then 0.9).
    Paper: step ratio 0.01, momentum schedule as above, weight decay 0.0002.
    """
    rbm = rbm.to(device)
    vel_W = torch.zeros_like(rbm.W)
    vel_h = torch.zeros_like(rbm.h_bias)
    vel_v = torch.zeros_like(rbm.v_bias)
    history = []

    print(f"  RBM  {rbm.n_visible} → {rbm.n_hidden}")
    for epoch in range(n_epochs):
        total, count = 0.0, 0
        mom = momentum if epoch < 5 else 0.9

        for batch in tqdm(loader, desc=f"  RBM epoch {epoch+1}/{n_epochs}", leave=False):
            batch = batch.to(device)
            N     = batch.shape[0]
            pos, neg, recon = rbm.contrastive_divergence(batch)

            gW = (pos - neg) / N
            gh = torch.mean(rbm.sample_h(batch)[0] - rbm.sample_h(recon)[0], dim=0)
            gv = torch.mean(batch - recon, dim=0)

            vel_W = mom * vel_W + learning_rate * gW
            vel_h = mom * vel_h + learning_rate * gh
            vel_v = mom * vel_v + learning_rate * gv

            with torch.no_grad():
                rbm.W      += vel_W - weight_decay * rbm.W
                rbm.h_bias += vel_h
                rbm.v_bias += vel_v

            total += torch.mean((batch - recon) ** 2).item()
            count += 1

        avg = total / count
        history.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:>3}/{n_epochs}  recon-err: {avg:.6f}")

    print(f"  ✅ RBM done.  Final loss: {history[-1]:.6f}")
    return history


def train_autoencoder(model, loader, n_epochs, learning_rate,
                      weight_decay, device, patience):
    """
    Fine-tune the autoencoder end-to-end with gradient descent + backpropagation.
    Uses Adam optimiser (paper uses GD; Adam is a well-understood improvement).
    """
    model = model.to(device)
    model.train()
    opt       = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                  patience=5, verbose=True)
    criterion = nn.MSELoss()
    history, best, wait = [], float('inf'), 0

    print("  Autoencoder fine-tuning")
    for epoch in range(n_epochs):
        total, count = 0.0, 0
        for batch in tqdm(loader, desc=f"  AE epoch {epoch+1}/{n_epochs}", leave=False):
            batch = batch.to(device)
            opt.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            opt.step()
            total += loss.item()
            count += 1

        avg = total / count
        history.append(avg)
        scheduler.step(avg)
        lr = opt.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:>3}/{n_epochs}  loss: {avg:.6f}  lr: {lr:.2e}")

        if avg < best:
            best, wait = avg, 0
        else:
            wait += 1
        if wait >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break

    print(f"  ✅ AE done.  Best loss: {best:.6f}")
    return history

# %% [markdown]
# ## Reconstruction Error & Latent Codes

# %%
def compute_reconstruction_error(model, image, scaler, device, batch_size):
    """
    Forward every pixel through the trained autoencoder.

    Reconstruction error r per pixel is the RMSE between the original
    spectrum X and the reconstructed spectrum X_hat (paper notation, Eq. 1
    context):
        r = sqrt( mean( (X - X_hat)^2 ) )   [mean over spectral bands]

    Returns
    -------
    rmse_map     : (H, W)           per-pixel RMSE in original spectral space
    latent_codes : (H, W, n_hidden) encoder activations  C  (paper notation)
    """
    model.eval()
    model = model.to(device)
    H, W, C  = image.shape
    n_hidden = model.n_hidden

    pixels_flat     = image.reshape(-1, C)
    pixels_norm, _  = preprocess_pixels(pixels_flat, fit_scaler=False, scaler=scaler)
    tensor          = torch.FloatTensor(pixels_norm)

    recons_list, codes_list = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(tensor), batch_size), desc="  Forward pass"):
            b = tensor[i:i+batch_size].to(device)
            r, z = model(b)
            recons_list.append(r.cpu())
            codes_list.append(z.cpu())

    recons_norm  = torch.cat(recons_list).numpy()
    codes        = torch.cat(codes_list).numpy()

    # Inverse-transform to original spectral scale before computing error
    recons_orig  = scaler.inverse_transform(recons_norm)

    # RMSE per pixel: sqrt(mean((X - X_hat)^2))  — mean over C bands
    rmse_flat = np.sqrt(np.mean((pixels_flat - recons_orig) ** 2, axis=1))

    return rmse_flat.reshape(H, W), codes.reshape(H, W, n_hidden)

# %% [markdown]
# ## MAWDBN DSW Detector
#
# Implements Equations 1–3 from the paper using the MAWDBN weight formula.
#
# AWDBN weight (Eq. 2):
#   wt_j   = 1/r_n_j        if (r_n_j - mu_n_r) < sigma_n_r   [inlier]
#   wt_j   = pf/r_n_j       otherwise                          [outlier]
#
# MAWDBN weight (Eq. 5) — multiplies both branches by r_p:
#   wt_j^M = r_p / r_n_j          [inlier]
#   wt_j^M = r_p * pf / r_n_j     [outlier]
#
# DSW window layout (paper notation):
#   winner  = inner window size (pixels, set by dsw_winner in config)
#   wouter  = outer window size (pixels, set by dsw_wouter in config)
#   ring width on each side = (wouter - winner) / 2
#
# The inner window defines the neighbourhood of the Pixel Under Test (PUT).
# The outer ring (between winner and wouter) provides the k background
# neighbours used to compute the anomaly score.

# %%
def _mawdbn_dsw(R, C, winner, wouter, pf):
    """
    MAWDBN anomaly score map using the DSW detector.

    Implements the full algorithm (steps 5–9 from the paper) for every pixel:
      1. Select k neighbours from the outer ring using DSW
      2. Compute d_j  (Eq. 1): L2 distance between neighbour code and PUT code
      3. Compute wt_j (Eq. 5): inlier/outlier split using mu_n and sigma_n:
            inlier  (r_n_j - mu_n_r) < sigma_n_r  →  wt_j^M = r_p / r_n_j
            outlier                                →  wt_j^M = r_p * pf / r_n_j
      4. Compute β    (Eq. 3): (1/k) * sum(wt_j * d_j)

    Parameters
    ----------
    R      : (H, W)           RMSE map (r in paper notation)
    C      : (H, W, n_hidden) latent code map (C in paper notation)
    winner : int              inner window size in pixels (must be odd)
    wouter : int              outer window size in pixels (must be odd, > winner)
    pf     : float            penalty factor for outlier neighbours (0 < pf < 1)

    Returns
    -------
    beta_map : (H, W)  anomaly score β per pixel
    """
    import math

    assert winner % 2 == 1 and wouter % 2 == 1, "winner and wouter must be odd"
    assert wouter > winner, "wouter must be larger than winner"

    H, W     = R.shape
    n_hidden = C.shape[2]

    half_in  = winner  // 2   # half-width of inner window
    half_out = wouter  // 2   # half-width of outer window (= padding needed)
    padding  = half_out

    # k = number of outer-ring neighbour cells
    k = wouter * wouter - winner * winner

    # Outer-ring mask: True for cells in [wouter] but NOT in [winner]
    ring = np.ones((wouter, wouter), dtype=bool)
    ring[half_out - half_in : half_out + half_in + 1,
         half_out - half_in : half_out + half_in + 1] = False

    pad_R = np.pad(R, ((padding, padding), (padding, padding)),
                   mode='constant', constant_values=-1)   # -1 = border sentinel
    pad_C = np.pad(C, ((padding, padding), (padding, padding), (0, 0)),
                   mode='constant', constant_values=0)

    beta_map = np.zeros((H, W), dtype=np.float32)

    for j in range(padding, H + padding):
        for i in range(padding, W + padding):
            win_R = pad_R[j - half_out : j + half_out + 1,
                          i - half_out : i + half_out + 1]
            win_C = pad_C[j - half_out : j + half_out + 1,
                          i - half_out : i + half_out + 1]

            r_p = win_R[half_out, half_out]   # RMSE of the PUT
            if r_p == -1:
                continue                        # border pixel — skip

            # ── Neighbour RMSE values (outer ring only) ────────────────────
            rn    = win_R[ring]       # shape (k,) — may contain -1 sentinels
            valid = rn != -1
            if not np.any(valid):
                continue

            rn_valid = rn[valid]
            mu_n  = float(np.mean(rn_valid))
            sig_n = float(np.std(rn_valid))

            # ── MAWDBN weights  (Eq. 5) ───────────────────────────────────
            # inlier  (r_n_j - mu_n_r) < sigma_n_r  →  wt_j^M = r_p / r_n_j
            # outlier                                →  wt_j^M = r_p * pf / r_n_j
            # Border sentinels (valid=False) get weight 0.
            inlier = valid & ((rn - mu_n) < sig_n)
            wt = np.where(inlier,
                          r_p / (rn + 1e-12),
                          r_p * pf / (rn + 1e-12))
            wt[~valid] = 0.0

            # ── Code distances  (Eq. 1):  d_j = ||c_n_j - c_p||_2 ─────────
            c_p    = win_C[half_out, half_out]          # PUT code vector
            c_ring = win_C[ring]                         # shape (k, n_hidden)
            d      = np.sqrt(np.sum((c_ring - c_p) ** 2, axis=1))

            # ── Anomaly score  (Eq. 3):  β = (1/k) * sum(wt_j * d_j) ──────
            # Use k_valid (actual neighbour count) in denominator to handle
            # border pixels where fewer than k neighbours are available
            k_valid = int(np.sum(valid))
            beta_map[j - padding, i - padding] = float(np.sum(wt * d)) / k_valid

    return beta_map


def compute_mawdbn_scores(rmse_map, latent_codes, winner, wouter, pf):
    """
    Wrapper: run the MAWDBN DSW detector over the full image.

    Parameters
    ----------
    rmse_map     : (H, W)           RMSE per pixel
    latent_codes : (H, W, n_hidden) encoder activations
    winner       : int              inner window size (odd integer, e.g. 1)
    wouter       : int              outer window size (odd integer, > winner)
    pf           : float            penalty factor for outlier neighbours (0 < pf < 1)

    Returns
    -------
    beta_map : (H, W)  MAWDBN anomaly score
    """
    H, W = rmse_map.shape
    print(f"  MAWDBN DSW  winner={winner}  wouter={wouter}  pf={pf}  "
          f"k={(wouter**2 - winner**2)}  image={H}×{W}")
    return _mawdbn_dsw(rmse_map.astype(np.float32), latent_codes, winner, wouter, pf)

# %% [markdown]
# ## Evaluation & Thresholding

# %%
def evaluate_auc(score_map, gt_mask, label):
    """Compute AUC-ROC of a continuous score map vs the binary GT mask."""
    auc         = roc_auc_score(gt_mask.reshape(-1).astype(int),
                                score_map.reshape(-1))
    fpr, tpr, _ = roc_curve(gt_mask.reshape(-1).astype(int),
                             score_map.reshape(-1))
    print(f"  {label:<12}  AUC-ROC = {auc:.4f}")
    return auc, fpr, tpr


def threshold_score_map(score_map, percentile):
    """Binary mask at the given percentile of the score map."""
    thresh = np.percentile(score_map, percentile)
    return (score_map > thresh).astype(np.uint8), thresh

# %% [markdown]
# ## Visualisation

# %%
def plot_training_history(rbm_losses, ae_losses, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(rbm_losses, lw=2)
    axes[0].set_title('RBM Pre-training Loss (CD-1)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Recon Error')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ae_losses, lw=2, color='orange')
    axes[1].set_title('Autoencoder Fine-tuning Loss')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_results(image, rmse_map, rmse_mask, beta_map, beta_mask,
                 gt_mask, scene_name, save_path=None):
    """
    8-panel figure:
      Row 0: False-colour RGB         |  Ground truth
      Row 1: RMSE score map           |  RMSE binary mask
      Row 2: MAWDBN β score map       |  MAWDBN binary mask
      Row 3: RMSE overlay             |  MAWDBN overlay
    Overlay: red = predicted anomaly, blue = false negative (missed)
    """
    H, W, C = image.shape
    rb = min(60, C-1); gb = min(40, C-1); bb = min(20, C-1)
    rgb = np.stack([image[:,:,rb], image[:,:,gb], image[:,:,bb]], axis=-1)
    rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8), 0, 1)

    def make_overlay(mask):
        out = rgb.copy()
        out[mask == 1] = [1, 0, 0]
        if gt_mask is not None:
            out[(gt_mask == 1) & (mask == 0)] = [0, 0.4, 1.0]
        return out

    fig, axes = plt.subplots(4, 2, figsize=(14, 22))
    fig.suptitle(f"MAWDBN Anomaly Detection — {scene_name}",
                 fontsize=15, fontweight='bold')

    axes[0,0].imshow(rgb)
    axes[0,0].set_title(f'False Colour (B{rb}/{gb}/{bb})')
    if gt_mask is not None:
        axes[0,1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
        axes[0,1].set_title(f'Ground Truth  ({int(gt_mask.sum()):,} anomaly px)')
    else:
        axes[0,1].set_title('Ground Truth — not available')

    im1 = axes[1,0].imshow(rmse_map, cmap='hot')
    axes[1,0].set_title('RMSE map  r = √mean((X−X̂)²)')
    plt.colorbar(im1, ax=axes[1,0], fraction=0.046)
    axes[1,1].imshow(rmse_mask, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1,1].set_title('RMSE anomaly mask')

    im2 = axes[2,0].imshow(beta_map, cmap='hot')
    axes[2,0].set_title('MAWDBN β score  (Eq. 3, weights Eq. 5)')
    plt.colorbar(im2, ax=axes[2,0], fraction=0.046)
    axes[2,1].imshow(beta_mask, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[2,1].set_title('MAWDBN anomaly mask')

    axes[3,0].imshow(make_overlay(rmse_mask))
    axes[3,0].set_title('RMSE overlay  (red = pred, blue = miss)')
    axes[3,1].imshow(make_overlay(beta_mask))
    axes[3,1].set_title('MAWDBN overlay  (red = pred, blue = miss)')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Figure → {save_path}")
    plt.show()


def plot_roc_curves(roc_results, scene_name, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, (auc, fpr, tpr) in roc_results.items():
        ax.plot(fpr, tpr, lw=2, label=f"{label}  AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title(f'ROC Curves — {scene_name}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ ROC → {save_path}")
    plt.show()

# %% [markdown]
# ## Save / Load

# %%
def save_model(model, scaler, config, save_dir, model_name):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'config': config},
               os.path.join(save_dir, f"{model_name}.pth"))
    joblib.dump(scaler, os.path.join(save_dir, f"{model_name}_scaler.pkl"))
    with open(os.path.join(save_dir, f"{model_name}_config.txt"), 'w') as f:
        f.write(f"Date: {datetime.now()}\n\n")
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
    print(f"  ✅ Model → {save_dir}/{model_name}.pth")


def load_model(model_path, scaler_path, device):
    ckpt   = torch.load(model_path, map_location=device)
    cfg    = ckpt['config']
    model  = DBNAutoencoder(n_visible=cfg['n_visible'], n_hidden=cfg['n_hidden'],
                            use_tied_weights=cfg['use_tied_weights'])
    model.load_state_dict(ckpt['model_state_dict'])
    model  = model.to(device).eval()
    scaler = joblib.load(scaler_path)
    print(f"  ✅ Loaded {model_path}")
    return model, scaler, cfg

# %% [markdown]
# ## Configuration
#
# **Change `SCENE_INDEX` to select a scene.**
# All ABU scene names are listed; missing .mat files are caught at startup
# (single-scene run) or skipped silently (batch run).

# %%
ABU_DATASET_DIR = 'ABU_DATASET'

# Complete ABU scene list — leave every entry in, even if the file is absent
ABU_SCENES = [
    'abu-airport-1',   #  0
    'abu-airport-2',   #  1
    'abu-airport-3',   #  2
    'abu-airport-4',   #  3
    'abu-beach-1',     #  4
    'abu-beach-2',     #  5
    'abu-beach-3',     #  6
    'abu-beach-4',     #  7
    'abu-urban-1',     #  8
    'abu-urban-2',     #  9
    'abu-urban-3',     # 10
    'abu-urban-4',     # 11
    'abu-urban-5',     # 12
]

# ↓↓ Change this index to select a scene ↓↓
SCENE_INDEX = 1

# ── Resolve path; error early if the selected scene is missing ────────────────
scene_name = ABU_SCENES[SCENE_INDEX]
mat_path   = os.path.join(ABU_DATASET_DIR, f'{scene_name}.mat')

if not os.path.exists(mat_path):
    available = [s for s in ABU_SCENES
                 if os.path.exists(os.path.join(ABU_DATASET_DIR, f'{s}.mat'))]
    raise FileNotFoundError(
        f"Scene [{SCENE_INDEX}] '{scene_name}' not found at: {mat_path}\n"
        f"Available scenes: {available}"
    )

# Peek at the file to get band count
print(f"Loading scene [{SCENE_INDEX}]: {scene_name}")
_img, _, N_BANDS = load_abu_mat(mat_path)
del _img

config = {
    # ── Data ──────────────────────────────────────────────────────────────────
    'scene_name':        scene_name,
    'mat_path':          mat_path,
    'n_bands':           N_BANDS,
    'n_samples_train':   50000,      # cap; uses all pixels if scene is smaller

    # ── Model architecture ────────────────────────────────────────────────────
    # Paper: 3-layer DBN, first/last = nb units, middle = 13 units (fixed).
    'n_visible':         N_BANDS,
    'n_hidden':          13,         # fixed at 13 per paper
    'use_tied_weights':  False,

    # ── RBM pre-training (CD-1) ───────────────────────────────────────────────
    # Paper: step ratio 0.01, momentum 0.5→0.9, weight decay 0.0002
    'rbm_epochs':        50,
    'rbm_learning_rate': 0.01,       # "step ratio" in paper
    'rbm_k':             1,          # CD-1
    'rbm_momentum':      0.5,        # switches to 0.9 after 5 epochs
    'rbm_weight_decay':  0.0002,     # paper: 0.0002

    # ── Autoencoder fine-tuning ───────────────────────────────────────────────
    'ae_epochs':         50,
    'ae_learning_rate':  0.001,
    'ae_weight_decay':   0.0002,
    'ae_patience':       10,

    # ── Inference ─────────────────────────────────────────────────────────────
    'batch_size':            2048,
    'random_seed':           42,
    'threshold_percentile':  95,

    # ── DSW window parameters (paper notation) ────────────────────────────────
    # winner : inner window size in pixels (odd integer)
    #          anomalies are assumed to fit inside this window
    # wouter : outer window size in pixels (odd integer, > winner)
    #          the ring between winner and wouter provides the k neighbours
    # pf     : penalty factor for outlier neighbours (0 < pf < 1)
    #          outlier = neighbour where (r_n_j - mu_n_r) >= sigma_n_r
    # k = wouter² - winner²   neighbours used per pixel
    'dsw_winner': 1,    # inner window  (1 = single centre pixel)
    'dsw_wouter': 7,    # outer window  (7×7, ring of 48 neighbours)
    'dsw_pf':     0,  # penalty factor  (paper: value between 0 and 1)
}
os.makedirs(config['save_dir'] if 'save_dir' in config else
            os.path.join('results', 'dbn_abu', scene_name), exist_ok=True)
config['save_dir']   = os.path.join('results', 'dbn_abu', scene_name)
config['model_name'] = f'mawdbn_{scene_name}'
os.makedirs(config['save_dir'], exist_ok=True)

print()
print("=" * 70)
print(f"  MAWDBN  ·  ABU  ·  {scene_name.upper()}")
print("=" * 70)
for k, v in config.items():
    print(f"  {k:<22}: {v}")
print()

# %% [markdown]
# ## Main Pipeline

# %%
def main():
    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 1 · LOAD DATA")
    print("="*70)
    image, gt_mask, _ = load_abu_mat(config['mat_path'])
    H, W, C = image.shape

    # All pixels are used for training (fully unsupervised, as in the paper)
    rng      = np.random.default_rng(config['random_seed'])
    n_train  = min(config['n_samples_train'], H * W)
    idx      = rng.choice(H * W, size=n_train, replace=False)
    train_px = image.reshape(-1, C)[idx]
    print(f"  Training pixels: {n_train:,}  ({100*n_train/(H*W):.1f}% of image)")

    norm_px, scaler = preprocess_pixels(train_px, fit_scaler=True)
    print(f"  Normalised — mean: {norm_px.mean():.4f}  std: {norm_px.std():.4f}")

    loader = DataLoader(PixelDataset(norm_px),
                        batch_size=config['batch_size'],
                        shuffle=True, num_workers=0, pin_memory=False)

    # ── Step 2: RBM pre-training (CD-1) ───────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 2 · RBM PRE-TRAINING  (CD-1)")
    print("="*70)
    rbm = GaussianRBM(n_visible=config['n_visible'],
                      n_hidden=config['n_hidden'],
                      k=config['rbm_k'])
    rbm_losses = train_rbm(rbm, loader,
                            n_epochs=config['rbm_epochs'],
                            learning_rate=config['rbm_learning_rate'],
                            momentum=config['rbm_momentum'],
                            weight_decay=config['rbm_weight_decay'],
                            device=device)

    # ── Step 3: Autoencoder fine-tuning ───────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 3 · AUTOENCODER FINE-TUNING")
    print("="*70)
    ae = DBNAutoencoder(n_visible=config['n_visible'],
                        n_hidden=config['n_hidden'],
                        use_tied_weights=config['use_tied_weights'])
    ae.initialize_from_rbm(rbm)
    ae_losses = train_autoencoder(ae, loader,
                                   n_epochs=config['ae_epochs'],
                                   learning_rate=config['ae_learning_rate'],
                                   weight_decay=config['ae_weight_decay'],
                                   device=device,
                                   patience=config['ae_patience'])
    plot_training_history(rbm_losses, ae_losses,
        save_path=os.path.join(config['save_dir'],
                               f"{config['model_name']}_training.png"))

    # ── Step 4: Save model ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 4 · SAVE MODEL")
    print("="*70)
    save_model(ae, scaler, config, config['save_dir'], config['model_name'])

    # ── Step 5: RMSE + latent codes (encode & decode full image) ──────────────
    print("\n" + "="*70)
    print("STEP 5 · RECONSTRUCTION ERROR  r = √mean((X−X̂)²)  +  LATENT CODES C")
    print("="*70)
    rmse_map, latent_codes = compute_reconstruction_error(
        ae, image, scaler, device=device, batch_size=config['batch_size']
    )
    print(f"  RMSE — mean: {rmse_map.mean():.6f}  "
          f"std: {rmse_map.std():.6f}  "
          f"max: {rmse_map.max():.6f}")

    # ── Step 6: MAWDBN anomaly score β ────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 6 · MAWDBN ANOMALY SCORE  β  (Eq. 3, weights Eq. 5)")
    print("="*70)
    beta_map = compute_mawdbn_scores(
        rmse_map, latent_codes,
        winner=config['dsw_winner'],
        wouter=config['dsw_wouter'],
        pf=config['dsw_pf'],
    )

    # ── Step 7: Threshold both maps ───────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 7 · THRESHOLDING")
    print("="*70)
    pct = config['threshold_percentile']
    rmse_mask, rmse_thresh = threshold_score_map(rmse_map, pct)
    beta_mask, beta_thresh = threshold_score_map(beta_map, pct)
    print(f"  RMSE   threshold ({pct}th pct): {rmse_thresh:.6f} "
          f"→ {int(rmse_mask.sum()):,} anomaly pixels")
    print(f"  MAWDBN threshold ({pct}th pct): {beta_thresh:.6f}  "
          f"→ {int(beta_mask.sum()):,} anomaly pixels")

    # ── Step 8: AUC-ROC evaluation ────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 8 · EVALUATION  (AUC-ROC vs ground truth)")
    print("="*70)
    roc_results = {}
    if gt_mask is not None:
        auc_rmse, fpr_rmse, tpr_rmse = evaluate_auc(rmse_map, gt_mask, 'RMSE')
        auc_beta, fpr_beta, tpr_beta = evaluate_auc(beta_map, gt_mask, 'MAWDBN')
        roc_results['RMSE']   = (auc_rmse, fpr_rmse, tpr_rmse)
        roc_results['MAWDBN'] = (auc_beta, fpr_beta, tpr_beta)
        plot_roc_curves(roc_results, config['scene_name'],
            save_path=os.path.join(config['save_dir'],
                                   f"{config['model_name']}_roc.png"))
    else:
        print("  ⚠️  No ground-truth mask — skipping AUC.")

    # ── Step 9: Visualise ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 9 · VISUALISATION")
    print("="*70)
    plot_results(image, rmse_map, rmse_mask, beta_map, beta_mask, gt_mask,
                 config['scene_name'],
                 save_path=os.path.join(config['save_dir'],
                                        f"{config['model_name']}_results.png"))

    # ── Step 10: Save arrays ──────────────────────────────────────────────────
    sd, mn = config['save_dir'], config['model_name']
    np.save(f"{sd}/{mn}_rmse_map.npy",     rmse_map)
    np.save(f"{sd}/{mn}_beta_map.npy",     beta_map)
    np.save(f"{sd}/{mn}_rmse_mask.npy",    rmse_mask)
    np.save(f"{sd}/{mn}_beta_mask.npy",    beta_mask)
    np.save(f"{sd}/{mn}_latent_codes.npy", latent_codes)
    print(f"  ✅ Arrays saved → {sd}/")

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)

    return ae, scaler, config, rmse_map, beta_map, rmse_mask, beta_mask, gt_mask, roc_results

# %% [markdown]
# ## Batch Runner  (all scenes)
#
# Trains and evaluates every ABU scene in sequence.
# Missing .mat files are silently skipped; the full list stays intact.

# %%
def run_all_scenes(abu_dir=ABU_DATASET_DIR):
    """
    Iterate over all ABU_SCENES, skip missing files, print AUC summary table.
    """
    table = {}
    for idx, scene in enumerate(ABU_SCENES):
        mat = os.path.join(abu_dir, f'{scene}.mat')
        if not os.path.exists(mat):
            print(f"\n⚠️  [{idx:>2}] {scene:<20}  — .mat not found, skipping.")
            table[scene] = {'AUC_RMSE': None, 'AUC_MAWDBN': None}
            continue

        print(f"\n{'='*70}")
        print(f"  SCENE [{idx}]: {scene}")
        print('='*70)

        _img, _, _nb = load_abu_mat(mat)
        del _img
        config.update({
            'scene_name':   scene,
            'mat_path':     mat,
            'n_bands':      _nb,
            'n_visible':    _nb,
            # n_hidden stays fixed at 13 per paper
            'save_dir':     os.path.join('results', 'dbn_abu', scene),
            'model_name':   f'mawdbn_{scene}',
        })
        os.makedirs(config['save_dir'], exist_ok=True)

        try:
            *_, roc = main()
            table[scene] = {
                'AUC_RMSE':   roc.get('RMSE',   (None,))[0],
                'AUC_MAWDBN': roc.get('MAWDBN', (None,))[0],
            }
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            table[scene] = {'AUC_RMSE': None, 'AUC_MAWDBN': None}

    print("\n" + "="*62)
    print("  BATCH RESULTS")
    print("="*62)
    print(f"  {'Scene':<22}  {'AUC-RMSE':>10}  {'AUC-MAWDBN':>11}")
    print("  " + "-"*47)
    for scene, r in table.items():
        rs = f"{r['AUC_RMSE']:.4f}"   if r['AUC_RMSE']   is not None else "    N/A"
        ms = f"{r['AUC_MAWDBN']:.4f}" if r['AUC_MAWDBN'] is not None else "    N/A"
        print(f"  {scene:<22}  {rs:>10}  {ms:>11}")
    return table

# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    ae, scaler, config, rmse_map, beta_map, rmse_mask, beta_mask, gt_mask, roc_results = main()
    print("\n✅ Done!")

    # ── Benchmark every scene (uncomment to use) ──────────────────────────────
    # table = run_all_scenes()