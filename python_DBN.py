# %% [markdown]
# # DBN-RBM Autoencoder for Hyperspectral Anomaly Detection
# Adapted from WaveletCNN methodology
#
# ## ⚠️ IMPORTANT: Fix NumPy Compatibility First!
# 
# **If you see "RuntimeError: Numpy is not available" or NumPy warnings:**
# 
# ```bash
# # Run this in your environment:
# pip uninstall numpy
# pip install "numpy<2.0"
# 
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

# Import your HYPSO data loading functions
from hypso.load import load_l1a_nc_cube, load_l1b_nc_cube, load_l1c_nc_cube, load_l1d_nc_cube

# Set random seeds for reproducibility
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
# Check CUDA availability
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
# ============================================================================
# MODEL COMPONENTS
# ============================================================================

# %% [markdown]
# ### Gaussian RBM

# %%
class GaussianRBM(nn.Module):
    """
    Gaussian-Bernoulli Restricted Boltzmann Machine
    For continuous input data (spectral bands)
    """
    def __init__(self, n_visible, n_hidden, learning_rate=0.01, k=1, sigma=1.0):
        super(GaussianRBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k  # CD-k steps
        self.sigma = sigma  # Standard deviation for Gaussian visible units
        
        # Initialize weights and biases
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        
    def sample_h(self, v):
        """Sample hidden units given visible units"""
        activation = F.linear(v, self.W, self.h_bias)
        p_h_given_v = torch.sigmoid(activation)
        sample = torch.bernoulli(p_h_given_v)
        return p_h_given_v, sample
    
    def sample_v(self, h):
        """Sample visible units given hidden units (Gaussian)"""
        activation = F.linear(h, self.W.t(), self.v_bias)
        # For Gaussian visible units, we return the mean (no sampling during reconstruction)
        return activation, activation
    
    def forward(self, v):
        """Forward pass - returns hidden probabilities"""
        p_h, _ = self.sample_h(v)
        return p_h
    
    def free_energy(self, v):
        """Compute free energy for visible units"""
        vbias_term = torch.matmul(v, self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)
        return -hidden_term - vbias_term
    
    def contrastive_divergence(self, v0):
        """Perform k steps of Contrastive Divergence"""
        # Positive phase
        p_h0, h0 = self.sample_h(v0)
        
        # Negative phase - Gibbs sampling
        h_k = h0
        for _ in range(self.k):
            v_k, _ = self.sample_v(h_k)
            p_h_k, h_k = self.sample_h(v_k)
        
        # Compute gradients
        positive_grad = torch.matmul(p_h0.t(), v0)
        negative_grad = torch.matmul(p_h_k.t(), v_k)
        
        # Return gradients and reconstruction for monitoring
        return positive_grad, negative_grad, v_k


# %% [markdown]
# ### Bernoulli RBM

# %%
class BernoulliRBM(nn.Module):
    """
    Bernoulli-Bernoulli Restricted Boltzmann Machine
    For binary/normalized hidden representations
    """
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
        sample = torch.bernoulli(p_h_given_v)
        return p_h_given_v, sample
    
    def sample_v(self, h):
        activation = F.linear(h, self.W.t(), self.v_bias)
        p_v_given_h = torch.sigmoid(activation)
        sample = torch.bernoulli(p_v_given_h)
        return p_v_given_h, sample
    
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


# %% [markdown]
# ### DBN Autoencoder

# %%
class DBNAutoencoder(nn.Module):
    """
    Deep Belief Network Autoencoder
    Encoder: 120 -> 13 (using pre-trained RBM)
    Decoder: 13 -> 120 (tied weights)
    """
    def __init__(self, n_visible=120, n_hidden=13, use_tied_weights=False):
        super(DBNAutoencoder, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.use_tied_weights = use_tied_weights
        
        # Encoder (will be initialized with RBM weights)
        self.encoder = nn.Linear(n_visible, n_hidden)
        
        # Decoder
        if use_tied_weights:
            # Tied weights: decoder weights are transpose of encoder
            # But we need separate decoder bias for reconstruction
            self.decoder = None
            self.decoder_bias = nn.Parameter(torch.zeros(n_visible))
        else:
            self.decoder = nn.Linear(n_hidden, n_visible)
            self.decoder_bias = None
        
    def encode(self, x):
        """Encode input to latent representation"""
        return torch.sigmoid(self.encoder(x))
    
    def decode(self, z):
        """Decode latent representation to reconstruction"""
        if self.use_tied_weights:
            # Use transposed encoder weights with decoder bias
            return F.linear(z, self.encoder.weight.t(), self.decoder_bias)
        else:
            return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass: encode then decode"""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z
    
    def initialize_from_rbm(self, rbm):
        """Initialize encoder weights from pre-trained RBM"""
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
# ============================================================================
# DATASET AND DATA LOADING
# ============================================================================

class PixelDataset(Dataset):
    """
    Dataset for randomly sampled pixels from a hyperspectral image
    """
    def __init__(self, pixels, transform=None):
        """
        Args:
            pixels: numpy array of shape (n_pixels, n_bands)
            transform: optional transform to apply
        """
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
    """
    Load HYPSO image based on file extension
    """
    basename = os.path.basename(image_path)
    if 'l1d' in basename:
        img = load_l1d_nc_cube(image_path)
    elif 'l1c' in basename:
        img = load_l1c_nc_cube(image_path)
    elif 'l1b' in basename:
        img = load_l1b_nc_cube(image_path)
    else:
        img = load_l1a_nc_cube(image_path)
    return img


def load_water_mask(label_path, water_class_label=2):
    """
    Load water mask from label file
    
    Args:
        label_path: path to .dat label file
        water_class_label: integer label for water class (default: 2)
    
    Returns:
        water_mask: boolean numpy array of shape (H, W) where True = water
    """
    # Load labels (values are 1, 2, 3, so subtract 1 to get 0, 1, 2)
    labels = np.fromfile(label_path, dtype=np.uint8).reshape((598, 1092)) - 1
    water_mask = (labels == water_class_label)
    return water_mask


def sample_pixels_from_image(image, n_samples=100000, random_seed=42, water_mask=None):
    """
    Randomly sample pixels from an image, optionally filtered by water mask
    
    Args:
        image: numpy array of shape (H, W, C)
        n_samples: number of pixels to sample
        random_seed: random seed for reproducibility
        water_mask: optional boolean array of shape (H, W) where True = water
    
    Returns:
        pixels: numpy array of shape (n_samples, C)
    """
    rng = np.random.default_rng(random_seed)
    H, W, C = image.shape
    
    # Flatten spatial dimensions
    pixels_flat = image.reshape(-1, C)
    
    if water_mask is not None:
        # Apply water mask - only sample from water pixels
        water_mask_flat = water_mask.reshape(-1)
        water_indices = np.where(water_mask_flat)[0]
        
        if len(water_indices) == 0:
            raise ValueError("No water pixels found in the mask!")
        
        print(f"Water mask applied: {len(water_indices)} / {len(pixels_flat)} pixels are water ({100*len(water_indices)/len(pixels_flat):.1f}%)")
        
        # Sample from water pixels only
        n_available = len(water_indices)
        n_samples = min(n_samples, n_available)
        
        sampled_water_indices = rng.choice(water_indices, size=n_samples, replace=False)
        sampled_pixels = pixels_flat[sampled_water_indices]
    else:
        # Sample from all pixels
        n_available = pixels_flat.shape[0]
        n_samples = min(n_samples, n_available)
        
        indices = rng.choice(n_available, size=n_samples, replace=False)
        sampled_pixels = pixels_flat[indices]
    
    return sampled_pixels


def preprocess_pixels(pixels, fit_scaler=True, scaler=None):
    """
    Normalize pixels to zero mean and unit variance
    
    Args:
        pixels: numpy array of shape (n_pixels, n_bands)
        fit_scaler: whether to fit a new scaler
        scaler: existing scaler to use (if fit_scaler=False)
    
    Returns:
        normalized_pixels: normalized pixel array
        scaler: the fitted or provided scaler
    """
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
# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_rbm(rbm, data_loader, n_epochs=50, learning_rate=0.01, 
              momentum=0.5, weight_decay=0.0001, device='cuda'):
    """
    Train an RBM using Contrastive Divergence
    
    Args:
        rbm: RBM model instance
        data_loader: DataLoader with training data
        n_epochs: number of training epochs
        learning_rate: initial learning rate
        momentum: momentum coefficient
        weight_decay: L2 regularization
        device: device to train on
    
    Returns:
        loss_history: list of reconstruction errors per epoch
    """
    rbm = rbm.to(device)
    
    # Initialize velocity for momentum
    velocity_W = torch.zeros_like(rbm.W)
    velocity_h = torch.zeros_like(rbm.h_bias)
    velocity_v = torch.zeros_like(rbm.v_bias)
    
    loss_history = []
    
    print(f"Training RBM: {rbm.n_visible} -> {rbm.n_hidden}")
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        # Increase momentum after 5 epochs
        current_momentum = momentum if epoch < 5 else 0.9
        
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            batch = batch.to(device)
            batch_size = batch.shape[0]
            
            # Contrastive divergence
            pos_grad, neg_grad, reconstruction = rbm.contrastive_divergence(batch)
            
            # Compute gradients
            grad_W = (pos_grad - neg_grad) / batch_size
            grad_h = torch.mean(rbm.sample_h(batch)[0] - rbm.sample_h(reconstruction)[0], dim=0)
            grad_v = torch.mean(batch - reconstruction, dim=0)
            
            # Update velocities with momentum
            velocity_W = current_momentum * velocity_W + learning_rate * grad_W
            velocity_h = current_momentum * velocity_h + learning_rate * grad_h
            velocity_v = current_momentum * velocity_v + learning_rate * grad_v
            
            # Update parameters
            with torch.no_grad():
                rbm.W += velocity_W - weight_decay * rbm.W
                rbm.h_bias += velocity_h
                rbm.v_bias += velocity_v
            
            # Compute reconstruction error
            error = torch.mean((batch - reconstruction) ** 2)
            epoch_loss += error.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Reconstruction Error: {avg_loss:.6f}")
    
    print(f"✅ RBM training complete. Final loss: {loss_history[-1]:.6f}")
    return loss_history


# %% [markdown]
# ### Autoencoder Fine-tuning

# %%
def train_autoencoder(model, data_loader, n_epochs=50, learning_rate=0.001,
                     weight_decay=0.0005, device='cuda', patience=10):
    """
    Fine-tune the autoencoder end-to-end using backpropagation
    
    Args:
        model: DBNAutoencoder instance
        data_loader: DataLoader with training data
        n_epochs: number of training epochs
        learning_rate: learning rate
        weight_decay: L2 regularization
        device: device to train on
        patience: early stopping patience
    
    Returns:
        loss_history: list of losses per epoch
    """
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
        
        # Early stopping
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
# ============================================================================
# ANOMALY DETECTION
# ============================================================================

def compute_reconstruction_error(model, image, scaler, device='cuda', batch_size=4096, water_mask=None):
    """
    Compute pixel-wise reconstruction error (RMSE) for an entire image
    
    Args:
        model: trained DBNAutoencoder
        image: numpy array of shape (H, W, C)
        scaler: fitted StandardScaler
        device: device to use
        batch_size: batch size for processing
        water_mask: optional boolean array of shape (H, W) to filter inference
    
    Returns:
        rmse_map: numpy array of shape (H, W) with RMSE per pixel
        reconstructed_image: numpy array of shape (H, W, C)
    """
    model.eval()
    model = model.to(device)
    
    H, W, C = image.shape
    pixels_flat = image.reshape(-1, C)
    
    # Initialize output arrays
    rmse_per_pixel = np.zeros(H * W)
    reconstructions_flat = np.zeros_like(pixels_flat)
    
    if water_mask is not None:
        # Only process water pixels
        water_mask_flat = water_mask.reshape(-1)
        water_indices = np.where(water_mask_flat)[0]
        
        print(f"Computing reconstruction for {len(water_indices)} water pixels only...")
        
        # Extract water pixels
        water_pixels = pixels_flat[water_indices]
        
        # Normalize
        water_pixels_normalized, _ = preprocess_pixels(water_pixels, fit_scaler=False, scaler=scaler)
        water_pixels_tensor = torch.FloatTensor(water_pixels_normalized)
        
        # Process in batches without multiprocessing to avoid NumPy issues
        reconstructions = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(water_pixels_tensor), batch_size), desc="Computing reconstruction"):
                batch = water_pixels_tensor[i:i+batch_size].to(device)
                recon, _ = model(batch)
                reconstructions.append(recon.cpu())
        
        reconstructions = torch.cat(reconstructions, dim=0).numpy()
        
        # Inverse transform to original scale
        reconstructions_original = scaler.inverse_transform(reconstructions)
        
        # Compute RMSE per water pixel
        squared_error = (water_pixels - reconstructions_original) ** 2
        rmse_water = np.sqrt(np.mean(squared_error, axis=1))
        
        # Place results back into full arrays
        rmse_per_pixel[water_indices] = rmse_water
        reconstructions_flat[water_indices] = reconstructions_original
        
        # Set non-water pixels to zero RMSE
        rmse_per_pixel[~water_mask_flat] = 0.0
        
    else:
        # Process all pixels
        # Normalize
        pixels_normalized, _ = preprocess_pixels(pixels_flat, fit_scaler=False, scaler=scaler)
        pixels_tensor = torch.FloatTensor(pixels_normalized)
        
        # Process in batches without multiprocessing to avoid NumPy issues
        reconstructions = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(pixels_tensor), batch_size), desc="Computing reconstruction"):
                batch = pixels_tensor[i:i+batch_size].to(device)
                recon, _ = model(batch)
                reconstructions.append(recon.cpu())
        
        reconstructions = torch.cat(reconstructions, dim=0).numpy()
        
        # Inverse transform to original scale
        reconstructions_original = scaler.inverse_transform(reconstructions)
        
        # Compute RMSE per pixel
        squared_error = (pixels_flat - reconstructions_original) ** 2
        rmse_per_pixel = np.sqrt(np.mean(squared_error, axis=1))
        
        reconstructions_flat = reconstructions_original
    
    # Reshape to image dimensions
    rmse_map = rmse_per_pixel.reshape(H, W)
    reconstructed_image = reconstructions_flat.reshape(H, W, C)
    
    return rmse_map, reconstructed_image


# %%
def create_anomaly_mask(rmse_map, threshold_percentile=95, water_mask=None):
    """
    Create binary anomaly mask based on RMSE threshold
    
    Args:
        rmse_map: numpy array of RMSE values
        threshold_percentile: percentile to use as threshold (e.g., 95)
        water_mask: optional boolean array to compute threshold only on water pixels
    
    Returns:
        anomaly_mask: binary mask (1 = anomaly, 0 = normal)
        threshold: the RMSE threshold used
    """
    if water_mask is not None:
        # Compute threshold only on water pixels
        water_rmse_values = rmse_map[water_mask]
        if len(water_rmse_values) == 0:
            raise ValueError("No water pixels to compute threshold!")
        threshold = np.percentile(water_rmse_values, threshold_percentile)
    else:
        # Compute threshold on all pixels
        threshold = np.percentile(rmse_map, threshold_percentile)
    
    anomaly_mask = (rmse_map > threshold).astype(np.uint8)
    
    if water_mask is not None:
        # Set non-water pixels to 0 (not anomalies)
        anomaly_mask[~water_mask] = 0
    
    return anomaly_mask, threshold


# %% [markdown]
# ## Visualization

# %%
# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_water_mask(image, water_mask, save_path=None):
    """
    Visualize original image (false color RGB) and water mask side-by-side
    
    Args:
        image: numpy array of shape (H, W, C)
        water_mask: boolean numpy array of shape (H, W)
        save_path: path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create false color RGB from bands (e.g., bands 60, 40, 20 for RGB)
    # Normalize to 0-1 range for display
    H, W, C = image.shape
    
    # Select bands for RGB visualization (adjust these if needed)
    r_band = min(60, C-1)  # Red
    g_band = min(40, C-1)  # Green
    b_band = min(20, C-1)  # Blue
    
    rgb = np.stack([
        image[:, :, r_band],
        image[:, :, g_band],
        image[:, :, b_band]
    ], axis=-1)
    
    # Normalize to 0-1
    rgb_min = rgb.min()
    rgb_max = rgb.max()
    if rgb_max > rgb_min:
        rgb_normalized = (rgb - rgb_min) / (rgb_max - rgb_min)
    else:
        rgb_normalized = rgb
    
    # Clip to valid range
    rgb_normalized = np.clip(rgb_normalized, 0, 1)
    
    # Display false color RGB
    axes[0].imshow(rgb_normalized)
    axes[0].set_title(f'False Color RGB (Bands {r_band}, {g_band}, {b_band})', fontsize=14)
    axes[0].axis('off')
    
    # Display water mask
    # Create a color overlay: water = blue, non-water = gray
    water_rgb = np.zeros((H, W, 3))
    water_rgb[water_mask] = [0.2, 0.4, 0.8]  # Blue for water
    water_rgb[~water_mask] = [0.5, 0.5, 0.5]  # Gray for non-water
    
    axes[1].imshow(water_rgb)
    axes[1].set_title('Water Mask (Blue = Water, Gray = Land/Cloud)', fontsize=14)
    axes[1].axis('off')
    
    # Add statistics text
    n_water = np.sum(water_mask)
    n_total = water_mask.size
    pct_water = 100 * n_water / n_total
    
    fig.text(0.5, 0.02, f'Water pixels: {n_water:,} / {n_total:,} ({pct_water:.1f}%)', 
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Water mask visualization saved to {save_path}")
    
    plt.show()


def visualize_results(original_image, rmse_map, anomaly_mask, threshold, 
                     band_index=60, save_path=None):
    """
    Visualize original image, RMSE map, and anomaly mask
    
    Args:
        original_image: numpy array of shape (H, W, C)
        rmse_map: numpy array of shape (H, W)
        anomaly_mask: binary numpy array of shape (H, W)
        threshold: RMSE threshold value
        band_index: which spectral band to display
        save_path: path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original image (single band)
    im0 = axes[0, 0].imshow(original_image[:, :, band_index], cmap='gray')
    axes[0, 0].set_title(f'Original Image (Band {band_index})', fontsize=14)
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # RMSE map
    im1 = axes[0, 1].imshow(rmse_map, cmap='hot')
    axes[0, 1].set_title('Reconstruction RMSE Map', fontsize=14)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Anomaly mask
    im2 = axes[1, 0].imshow(anomaly_mask, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Anomaly Mask (Threshold: {threshold:.4f})', fontsize=14)
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, ticks=[0, 1])
    
    # Overlay
    overlay = original_image[:, :, band_index].copy()
    overlay_rgb = np.stack([overlay, overlay, overlay], axis=-1)
    overlay_rgb = (overlay_rgb - overlay_rgb.min()) / (overlay_rgb.max() - overlay_rgb.min())
    overlay_rgb[anomaly_mask == 1] = [1, 0, 0]  # Red for anomalies
    
    axes[1, 1].imshow(overlay_rgb)
    axes[1, 1].set_title('Anomalies Overlay (Red)', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Results saved to {save_path}")
    
    plt.show()


# %%
def plot_training_history(rbm_losses, ae_losses, save_path=None):
    """
    Plot training loss history for RBM and Autoencoder
    """
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
# ============================================================================
# SAVE AND LOAD FUNCTIONS
# ============================================================================

def save_model_and_config(model, scaler, config, save_dir, model_name="dbn_autoencoder"):
    """
    Save trained model, scaler, and configuration
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, model_path)
    
    # Save scaler
    scaler_path = os.path.join(save_dir, f"{model_name}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # Save config as text
    config_path = os.path.join(save_dir, f"{model_name}_config.txt")
    with open(config_path, 'w') as f:
        f.write(f"DBN Autoencoder Configuration\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    print(f"✅ Model saved to {model_path}")
    print(f"✅ Scaler saved to {scaler_path}")
    print(f"✅ Config saved to {config_path}")


# %%
def load_model_and_scaler(model_path, scaler_path, device='cuda'):
    """
    Load trained model and scaler
    """
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

# Define your data paths (UPDATE THESE TO MATCH YOUR SYSTEM)
raw_data_path = r'/home/samb/Coding/WaveletCNN/hsi_data/HYPSORAWDATA'
labels_path = r'/home/samb/Coding/WaveletCNN/hsi_data/HYPSOLABELS'  # Path to label files

# Choose data processing level: 'l1a', 'l1b', 'l1c', or 'l1d'
DATA_LEVEL = 'l1b'  # Options: 'l1a', 'l1b', 'l1c', 'l1d'

# Water mask settings
USE_WATER_MASK = True  # Set to True to train only on water pixels
WATER_CLASS_LABEL = 2  # In your labels: 0=Cloud, 1=Land, 2=Sea/Water

# Choose which image to use for training (index into the list below)
TRAINING_IMAGE_INDEX = 0  # Change this to select different training image

# Available images - same as your WaveletCNN code (base names, will add level suffix)
raw_data_files_h1 = [
    'trondheim_2024-09-11T09-27-22Z',      # Index 0
    'adriatic_2024-07-18T08-53-45Z',       # Index 1
    'florida_2024-05-21T15-51-31Z',        # Index 2
    'solbergstrand_2024-07-08T09-39-39Z',  # Index 3
    'annapolis_2024-09-08T15-34-29Z',      # Index 4
    'trondheim_2024-09-14T09-42-48Z',      # Index 5
    # Add more files as needed (without -l1a.nc suffix)
]

raw_data_files_h2 = [
    'aeronetgalata_2025-01-02T08-52-34Z',  # Index 0
    'moby_2025-01-08T20-54-59Z',           # Index 1
    'mvco_2025-02-05T15-52-26Z',           # Index 2
    # Add more files as needed (without -l1a.nc suffix)
]

# Choose dataset: "h1", "h2", or "h1h2"
DATASET_MODE = "h1"  # Options: "h1", "h2", "h1h2"

if DATASET_MODE == "h1":
    raw_data_files = raw_data_files_h1
elif DATASET_MODE == "h2":
    raw_data_files = raw_data_files_h2
elif DATASET_MODE == "h1h2":
    raw_data_files = raw_data_files_h1 + raw_data_files_h2
else:
    raise ValueError(f"Unknown DATASET_MODE: {DATASET_MODE}")

# Get the training image path with proper level suffix
training_image_basename = raw_data_files[TRAINING_IMAGE_INDEX]
training_image_filename = f"{training_image_basename}-{DATA_LEVEL}.nc"
training_image_path = os.path.join(raw_data_path, training_image_filename)

# Get corresponding label file for water mask
training_label_filename = f"{training_image_basename}_labels.dat"
training_label_path = os.path.join(labels_path, training_label_filename)

print(f"Selected training image: {training_image_filename}")
print(f"Data level: {DATA_LEVEL}")
if USE_WATER_MASK:
    print(f"Using water mask from: {training_label_filename}")
    if not os.path.exists(training_label_path):
        print(f"⚠️ WARNING: Label file not found at {training_label_path}")
        print(f"   Water masking will be disabled.")
        USE_WATER_MASK = False

# Model configuration
config = {
    # Data parameters
    'image_path': training_image_path,
    'label_path': training_label_path if USE_WATER_MASK else None,
    'use_water_mask': USE_WATER_MASK,
    'water_class_label': WATER_CLASS_LABEL,
    'data_level': DATA_LEVEL,
    'n_bands': 120,
    'n_samples_train': 100000,  # Number of pixels to sample for training
        
        # Model parameters
        'n_visible': 120,
        'n_hidden': 13,
        'use_tied_weights': False,  # Untied for better flexibility and simpler code
        
        # RBM training parameters
        'rbm_epochs': 50,
        'rbm_learning_rate': 0.01,
        'rbm_k': 1,  # CD-k steps
        'rbm_momentum': 0.5,
        'rbm_weight_decay': 0.0001,
        
        # Autoencoder fine-tuning parameters
        'ae_epochs': 50,
        'ae_learning_rate': 0.001,
        'ae_weight_decay': 0.0005,
        'ae_patience': 10,
        
        # Training parameters
        'batch_size': 2048,
        'random_seed': 42,
        
        # Anomaly detection parameters
        'threshold_percentile': 95,
        
        # Output
    'save_dir': 'results/dbn_anomaly_detection',
    'model_name': 'dbn_autoencoder_hypso'
}

# Verify the training image exists
if not os.path.exists(config['image_path']):
    print(f"❌ ERROR: Training image not found at: {config['image_path']}")
    print(f"\nPlease update the paths at the top of the script:")
    print(f"  - raw_data_path: {raw_data_path}")
    print(f"  - DATA_LEVEL: {DATA_LEVEL}")
    print(f"  - TRAINING_IMAGE_INDEX: {TRAINING_IMAGE_INDEX}")
    print(f"  - Available files: {raw_data_files}")
    raise FileNotFoundError(f"Training image not found: {config['image_path']}")

# Create output directory
os.makedirs(config['save_dir'], exist_ok=True)

print("=" * 80)
print("DBN-RBM AUTOENCODER FOR HYPERSPECTRAL ANOMALY DETECTION")
print("=" * 80)
print(f"\nConfiguration:")
for key, value in config.items():
    print(f"  {key}: {value}")
print()


# %% [markdown]
# ## Main Execution Pipeline

# %%
def main():
    """
    Complete pipeline for training DBN autoencoder and detecting anomalies
    Uses the config dictionary defined above
    """
    
    #
    # ========================================================================
    # LOAD AND PREPROCESS DATA
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 80)
    
    # Load image
    print(f"Loading image: {os.path.basename(config['image_path'])}")
    start_time = time.time()
    image = load_hypso_image(config['image_path'])
    print(f"Image shape: {image.shape}")
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    
    # Load water mask if enabled
    water_mask = None
    if config['use_water_mask'] and config['label_path'] is not None:
        print(f"\nLoading water mask from: {os.path.basename(config['label_path'])}")
        water_mask = load_water_mask(config['label_path'], config['water_class_label'])
        n_water = np.sum(water_mask)
        n_total = water_mask.size
        print(f"Water pixels: {n_water} / {n_total} ({100*n_water/n_total:.1f}%)")
        
        # Visualize water mask
        print("\nVisualizing water mask...")
        mask_viz_path = os.path.join(config['save_dir'], f"{config['model_name']}_water_mask_visualization.png")
        visualize_water_mask(image, water_mask, save_path=mask_viz_path)
    
    # Sample pixels for training
    if config['use_water_mask'] and water_mask is not None:
        print(f"\nSampling {config['n_samples_train']} WATER pixels for training...")
    else:
        print(f"\nSampling {config['n_samples_train']} pixels for training...")
    
    sampled_pixels = sample_pixels_from_image(
        image, 
        n_samples=config['n_samples_train'],
        random_seed=config['random_seed'],
        water_mask=water_mask
    )
    print(f"Sampled pixels shape: {sampled_pixels.shape}")
    
    # Normalize pixels
    print("\nNormalizing pixels...")
    normalized_pixels, scaler = preprocess_pixels(sampled_pixels, fit_scaler=True)
    print(f"Normalized pixels - Mean: {normalized_pixels.mean():.6f}, Std: {normalized_pixels.std():.6f}")
    
    # Create dataset and dataloader
    train_dataset = PixelDataset(normalized_pixels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid NumPy multiprocessing issues
        pin_memory=False  # Disable pin_memory when using CPU or if having issues
    )
    print(f"Created DataLoader with {len(train_dataset)} samples")
    
    #
    # ========================================================================
    # TRAIN RBM
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 2: PRE-TRAINING RBM")
    print("=" * 80)
    
    rbm = GaussianRBM(
        n_visible=config['n_visible'],
        n_hidden=config['n_hidden'],
        learning_rate=config['rbm_learning_rate'],
        k=config['rbm_k']
    )
    
    rbm_losses = train_rbm(
        rbm=rbm,
        data_loader=train_loader,
        n_epochs=config['rbm_epochs'],
        learning_rate=config['rbm_learning_rate'],
        momentum=config['rbm_momentum'],
        weight_decay=config['rbm_weight_decay'],
        device=device
    )
    
    #
    # ========================================================================
    # BUILD AND FINE-TUNE AUTOENCODER
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 3: BUILDING AND FINE-TUNING AUTOENCODER")
    print("=" * 80)
    
    autoencoder = DBNAutoencoder(
        n_visible=config['n_visible'],
        n_hidden=config['n_hidden'],
        use_tied_weights=config['use_tied_weights']
    )
    
    # Initialize with RBM weights
    autoencoder.initialize_from_rbm(rbm)
    
    # Fine-tune
    ae_losses = train_autoencoder(
        model=autoencoder,
        data_loader=train_loader,
        n_epochs=config['ae_epochs'],
        learning_rate=config['ae_learning_rate'],
        weight_decay=config['ae_weight_decay'],
        device=device,
        patience=config['ae_patience']
    )
    
    # Plot training history
    plot_path = os.path.join(config['save_dir'], f"{config['model_name']}_training_history.png")
    plot_training_history(rbm_losses, ae_losses, save_path=plot_path)
    
    #
    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 4: SAVING MODEL")
    print("=" * 80)
    
    save_model_and_config(
        model=autoencoder,
        scaler=scaler,
        config=config,
        save_dir=config['save_dir'],
        model_name=config['model_name']
    )
    
    #
    # ========================================================================
    # ANOMALY DETECTION
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 5: ANOMALY DETECTION")
    print("=" * 80)
    
    print("Computing reconstruction error for entire image...")
    rmse_map, reconstructed_image = compute_reconstruction_error(
        model=autoencoder,
        image=image,
        scaler=scaler,
        device=device,
        batch_size=config['batch_size'],
        water_mask=water_mask
    )
    
    if config['use_water_mask'] and water_mask is not None:
        # Compute statistics only on water pixels
        water_rmse = rmse_map[water_mask]
        print(f"RMSE Statistics (water pixels only):")
        print(f"  Mean: {water_rmse.mean():.6f}")
        print(f"  Std: {water_rmse.std():.6f}")
        print(f"  Min: {water_rmse.min():.6f}")
        print(f"  Max: {water_rmse.max():.6f}")
        print(f"  Median: {np.median(water_rmse):.6f}")
    else:
        print(f"RMSE Statistics:")
        print(f"  Mean: {rmse_map.mean():.6f}")
        print(f"  Std: {rmse_map.std():.6f}")
        print(f"  Min: {rmse_map.min():.6f}")
        print(f"  Max: {rmse_map.max():.6f}")
        print(f"  Median: {np.median(rmse_map):.6f}")
    
    # Create anomaly mask
    print(f"\nCreating anomaly mask (threshold: {config['threshold_percentile']}th percentile)...")
    anomaly_mask, threshold = create_anomaly_mask(
        rmse_map,
        threshold_percentile=config['threshold_percentile'],
        water_mask=water_mask
    )
    
    n_anomalies = np.sum(anomaly_mask)
    total_pixels = anomaly_mask.size
    anomaly_percentage = 100 * n_anomalies / total_pixels
    
    print(f"Threshold: {threshold:.6f}")
    print(f"Anomalies detected: {n_anomalies} / {total_pixels} ({anomaly_percentage:.2f}%)")
    
    #
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 6: VISUALIZATION")
    print("=" * 80)
    
    results_path = os.path.join(config['save_dir'], f"{config['model_name']}_results.png")
    visualize_results(
        original_image=image,
        rmse_map=rmse_map,
        anomaly_mask=anomaly_mask,
        threshold=threshold,
        band_index=60,
        save_path=results_path
    )
    
    # Save RMSE map and anomaly mask
    rmse_path = os.path.join(config['save_dir'], f"{config['model_name']}_rmse_map.npy")
    mask_path = os.path.join(config['save_dir'], f"{config['model_name']}_anomaly_mask.npy")
    
    np.save(rmse_path, rmse_map)
    np.save(mask_path, anomaly_mask)
    
    print(f"✅ RMSE map saved to {rmse_path}")
    print(f"✅ Anomaly mask saved to {mask_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {config['save_dir']}")
    print(f"  - Model: {config['model_name']}.pth")
    print(f"  - Scaler: {config['model_name']}_scaler.pkl")
    print(f"  - Config: {config['model_name']}_config.txt")
    print(f"  - Training history: {config['model_name']}_training_history.png")
    print(f"  - Results visualization: {config['model_name']}_results.png")
    print(f"  - RMSE map: {config['model_name']}_rmse_map.npy")
    print(f"  - Anomaly mask: {config['model_name']}_anomaly_mask.npy")
    print()
    
    return autoencoder, scaler, config, rmse_map, anomaly_mask


# %% [markdown]
# ## Inference Function for New Images

# %%
# ============================================================================
# INFERENCE FUNCTION FOR NEW IMAGES
# ============================================================================

def detect_anomalies_in_new_image(image_path, model_path, scaler_path, 
                                  threshold_percentile=95, device='cuda',
                                  label_path=None, water_class_label=2):
    """
    Apply trained model to detect anomalies in a new image
    
    Args:
        image_path: path to new hyperspectral image
        model_path: path to saved model
        scaler_path: path to saved scaler
        threshold_percentile: percentile for anomaly threshold
        device: device to use
        label_path: optional path to label file for water masking
        water_class_label: label value for water class
    
    Returns:
        rmse_map: reconstruction error map
        anomaly_mask: binary anomaly mask
        threshold: RMSE threshold used
    """
    print(f"\n{'='*80}")
    print(f"DETECTING ANOMALIES IN NEW IMAGE")
    print(f"{'='*80}")
    
    # Load model and scaler
    model, scaler, config = load_model_and_scaler(model_path, scaler_path, device)
    
    # Load image
    print(f"\nLoading image: {os.path.basename(image_path)}")
    image = load_hypso_image(image_path)
    print(f"Image shape: {image.shape}")
    
    # Load water mask if provided
    water_mask = None
    if label_path is not None and os.path.exists(label_path):
        print(f"\nLoading water mask from: {os.path.basename(label_path)}")
        water_mask = load_water_mask(label_path, water_class_label)
        n_water = np.sum(water_mask)
        n_total = water_mask.size
        print(f"Water pixels: {n_water} / {n_total} ({100*n_water/n_total:.1f}%)")
        
        # Visualize water mask
        print("\nVisualizing water mask...")
        mask_viz_path = image_path.replace('.nc', '_water_mask_visualization.png')
        visualize_water_mask(image, water_mask, save_path=mask_viz_path)
    
    # Compute reconstruction error
    print("\nComputing reconstruction error...")
    rmse_map, reconstructed_image = compute_reconstruction_error(
        model=model,
        image=image,
        scaler=scaler,
        device=device,
        batch_size=4096,
        water_mask=water_mask
    )
    
    # Create anomaly mask
    print(f"\nCreating anomaly mask (threshold: {threshold_percentile}th percentile)...")
    anomaly_mask, threshold = create_anomaly_mask(
        rmse_map,
        threshold_percentile=threshold_percentile,
        water_mask=water_mask
    )
    
    n_anomalies = np.sum(anomaly_mask)
    if water_mask is not None:
        n_total = np.sum(water_mask)  # Only count water pixels
    else:
        n_total = anomaly_mask.size
    anomaly_percentage = 100 * n_anomalies / n_total if n_total > 0 else 0
    
    print(f"\nResults:")
    if water_mask is not None:
        water_rmse = rmse_map[water_mask]
        print(f"  RMSE (water only) - Mean: {water_rmse.mean():.6f}, Std: {water_rmse.std():.6f}")
    else:
        print(f"  RMSE - Mean: {rmse_map.mean():.6f}, Std: {rmse_map.std():.6f}")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Anomalies: {n_anomalies} / {n_total} ({anomaly_percentage:.2f}%)")
    
    # Visualize
    save_path = image_path.replace('.nc', '_anomaly_detection.png')
    visualize_results(
        original_image=image,
        rmse_map=rmse_map,
        anomaly_mask=anomaly_mask,
        threshold=threshold,
        band_index=60,
        save_path=save_path
    )
    
    return rmse_map, anomaly_mask, threshold


# %% [markdown]
# ## Example Usage

# %%
# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# %%
if __name__ == "__main__":
    # Train model and detect anomalies
    autoencoder, scaler, config, rmse_map, anomaly_mask = main()
    
    print("\n✅ Training and inference complete!")
    print(f"Model saved to: {config['save_dir']}")

# %% [markdown]
# ## Apply to New Images (Optional)
# 
# Uncomment and run this cell to apply the trained model to a different image:

# %%
# # Example: Apply to a new image
# new_image_path = 'path/to/new/image.nc'
# model_path = 'results/dbn_anomaly_detection/dbn_autoencoder_hypso.pth'
# scaler_path = 'results/dbn_anomaly_detection/dbn_autoencoder_hypso_scaler.pkl'
# 
# rmse_new, mask_new, thresh_new = detect_anomalies_in_new_image(
#     image_path=new_image_path,
#     model_path=model_path,
#     scaler_path=scaler_path,
#     threshold_percentile=95,
#     device=device
# )