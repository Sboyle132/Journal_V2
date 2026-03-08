# %% [markdown]
# # HYPSO Hyperspectral Anomaly Detection
# Trains a DBN autoencoder on sea pixels, computes RMSE and β anomaly maps,
# displays false colour + RMSE + β for both FP32 (A) and HybridRaw (D) variants.
#
# Variant A: fp32 AE — fp32 RMSE + fp32 codes → fp32 DSW
# Variant D: int8 AE — fp32 RMSE (dequantised) + raw int8 codes → int8 DSW (int64 distances)

# ═══════════════════════════════════════════════════════════════════════════════
# %% CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

# Path to the capture directory, e.g. '/data/caspiansea2/caspiansea2_2025-10-31T07-37-34Z'
# The script will locate the -l1a.nc and processing-temp/sea-land-cloud.labels automatically.
CAPTURE_DIR = './data/caspiansea2_2025-08-23T07-36-40Z'

# Label values (after the -1 offset applied by the loader):
#   2 = Cloud, 1 = Land, 0 = Sea  — matches the WaveletCNN script convention
SEA_LABEL = 0

# False-colour band indices (0-based) for display
FC_RED   = 60
FC_GREEN = 40
FC_BLUE  = 20

# HYPSO fixed image shape
HYPSO_H = 598
HYPSO_W = 1092
HYPSO_C = 120

# Model
N_HIDDEN = 13

# RBM
RBM_EPOCHS        = 50
RBM_LEARNING_RATE = 0.01
RBM_K             = 1
RBM_MOMENTUM      = 0.5
RBM_WEIGHT_DECAY  = 0.0002

# Autoencoder
AE_EPOCHS        = 50
AE_LEARNING_RATE = 0.001
AE_WEIGHT_DECAY  = 0.0002
AE_PATIENCE      = 15

# Training samples (None = use all sea pixels)
N_SAMPLES_TRAIN = 50000
BATCH_SIZE      = 2048

# DSW — fixed window, configure here
DSW_WINNER = 1    # inner square half-size (1 = centre pixel only excluded)
DSW_WOUTER = 9    # outer square size

RANDOM_SEED = 42

# ═══════════════════════════════════════════════════════════════════════════════
# %% IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import os, random, warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

from hypso.load import load_l1a_nc_cube

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, losses
from tensorflow.keras.regularizers import l2 as _l2

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"  GPU(s): {[g.name for g in gpus]}")
else:
    print("  No GPU — running on CPU.")
print(f"  TF {tf.__version__}  |  DSW w={DSW_WINNER}, W={DSW_WOUTER}")


# ═══════════════════════════════════════════════════════════════════════════════
# %% UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_hypso_paths(capture_dir):
    """
    Given a capture directory like:
        /data/caspiansea2/caspiansea2_2025-10-31T07-37-34Z
    Returns:
        l1a_path   — path to the -l1a.nc file
        labels_path — path to processing-temp/sea-land-cloud.labels
    """
    capture_name = os.path.basename(capture_dir.rstrip('/'))
    l1a_path     = os.path.join(capture_dir, f'{capture_name}-l1a.nc')
    labels_path  = os.path.join(capture_dir, 'processing-temp', 'sea-land-cloud.labels')
    assert os.path.exists(l1a_path),    f"L1A not found: {l1a_path}"
    assert os.path.exists(labels_path), f"Labels not found: {labels_path}"
    return l1a_path, labels_path


def load_hypso_image(l1a_path):
    """Load L1A cube via hypso library → (H, W, C) float32."""
    img = load_l1a_nc_cube(l1a_path)
    return img.astype(np.float32)


def load_hypso_labels(labels_path):
    """
    Load sea-land-cloud.labels: raw uint8, values 0/1/2 (already 0-based).
    0=Cloud, 1=Land, 2=Sea
    Returns (598, 1092) int array.
    """
    raw = np.fromfile(labels_path, dtype=np.uint8).reshape((HYPSO_H, HYPSO_W))
    return raw.astype(np.int32)


def false_colour(image, r=FC_RED, g=FC_GREEN, b=FC_BLUE):
    rgb = np.stack([image[:, :, r], image[:, :, g], image[:, :, b]], axis=-1)
    mn, mx = rgb.min(), rgb.max()
    return np.clip((rgb - mn) / (mx - mn + 1e-8), 0, 1).astype(np.float32)


def preprocess_pixels(pixels, fit_scaler=True, scaler=None):
    if fit_scaler:
        scaler     = StandardScaler()
        normalised = scaler.fit_transform(pixels)
    else:
        normalised = scaler.transform(pixels)
    return normalised.astype(np.float32), scaler


# ═══════════════════════════════════════════════════════════════════════════════
# %% RBM
# ═══════════════════════════════════════════════════════════════════════════════

class GaussianRBM_NP:
    def __init__(self, n_visible, n_hidden, k=1):
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        self.k         = k
        rng = np.random.default_rng(RANDOM_SEED)
        self.W      = rng.standard_normal((n_hidden, n_visible)).astype(np.float32) * 0.01
        self.h_bias = np.zeros(n_hidden,  dtype=np.float32)
        self.v_bias = np.zeros(n_visible, dtype=np.float32)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _sample_h(self, v):
        p = self._sigmoid(v @ self.W.T + self.h_bias)
        return p, (np.random.rand(*p.shape) < p).astype(np.float32)

    def _sample_v(self, h):
        mean = h @ self.W + self.v_bias
        return mean, mean

    def train(self, data, n_epochs, lr, momentum, weight_decay, batch_size=2048):
        vel_W = np.zeros_like(self.W)
        vel_h = np.zeros_like(self.h_bias)
        vel_v = np.zeros_like(self.v_bias)
        history = []
        N = len(data)
        for epoch in range(n_epochs):
            idx   = np.random.permutation(N)
            total = 0.0; count = 0
            mom   = momentum if epoch < 5 else 0.9
            for start in range(0, N, batch_size):
                batch    = data[idx[start:start + batch_size]]
                p_h0, h0 = self._sample_h(batch)
                h_k = h0
                for _ in range(self.k):
                    v_k, _    = self._sample_v(h_k)
                    p_hk, h_k = self._sample_h(v_k)
                nb  = batch.shape[0]
                gW  = (p_h0.T @ batch - p_hk.T @ v_k) / nb
                gh  = np.mean(p_h0 - p_hk, axis=0)
                gv  = np.mean(batch - v_k,  axis=0)
                vel_W = mom * vel_W + lr * gW
                vel_h = mom * vel_h + lr * gh
                vel_v = mom * vel_v + lr * gv
                self.W      += vel_W - weight_decay * self.W
                self.h_bias += vel_h
                self.v_bias += vel_v
                total += np.mean((batch - v_k) ** 2); count += 1
            history.append(total / count)
        print(f"  RBM {self.n_visible}→{self.n_hidden}  epochs={n_epochs}  "
              f"final loss: {history[-1]:.6f}")
        return history


# ═══════════════════════════════════════════════════════════════════════════════
# %% AUTOENCODER
# ═══════════════════════════════════════════════════════════════════════════════

def build_autoencoder_fp32(n_visible, n_hidden):
    wd  = AE_WEIGHT_DECAY
    inp = keras.Input(shape=(n_visible,), name='spectrum')
    z   = layers.Dense(n_hidden, activation='sigmoid',
                       kernel_regularizer=_l2(wd / 2), name='encoder_dense')(inp)
    out = layers.Dense(n_visible, activation='linear',
                       kernel_regularizer=_l2(wd / 2), name='decoder_dense')(z)
    return keras.Model(inputs=inp, outputs=out, name='ae_fp32')


def build_encoder_only(autoencoder):
    return keras.Model(inputs=autoencoder.input,
                       outputs=autoencoder.get_layer('encoder_dense').output,
                       name='encoder')


def initialize_from_rbm(model, rbm):
    model.get_layer('encoder_dense').set_weights([rbm.W.T, rbm.h_bias])
    model.get_layer('decoder_dense').set_weights([rbm.W,   rbm.v_bias])


def train_autoencoder(model, X_train, name='ae'):
    model.compile(optimizer=optimizers.Adam(learning_rate=AE_LEARNING_RATE),
                  loss=losses.MeanSquaredError())
    history = model.fit(
        X_train, X_train,
        epochs=AE_EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.0, shuffle=True, verbose=0,
        callbacks=[
            callbacks.EarlyStopping(monitor='loss', patience=AE_PATIENCE,
                                    restore_best_weights=True, verbose=0),
            callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5),
        ],
    )
    n_run = len(history.history['loss'])
    print(f"  AE [{name}]  epochs={n_run}/{AE_EPOCHS}  "
          f"best loss: {min(history.history['loss']):.6f}")
    return history.history


# ═══════════════════════════════════════════════════════════════════════════════
# %% PTQ EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_tflite_ptq(model, rep_data, name):
    def rep_dataset():
        for i in range(min(200, len(rep_data))):
            yield [rep_data[i:i+1].astype(np.float32)]
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations             = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset    = rep_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type      = tf.int8
    conv.inference_output_type     = tf.int8
    tflite_model = conv.convert()
    path = f'{name}_ptq_int8.tflite'
    with open(path, 'wb') as f:
        f.write(tflite_model)
    print(f"  PTQ AE  → {path}  ({os.path.getsize(path)/1024:.1f} KB)")
    return path


def export_tflite_encoder_ptq(ae_model, rep_data, name):
    enc = build_encoder_only(ae_model)
    def rep_dataset():
        for i in range(min(200, len(rep_data))):
            yield [rep_data[i:i+1].astype(np.float32)]
    conv = tf.lite.TFLiteConverter.from_keras_model(enc)
    conv.optimizations             = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset    = rep_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type      = tf.int8
    conv.inference_output_type     = tf.int8
    tflite_model = conv.convert()
    path = f'{name}_ptq_int8_encoder.tflite'
    with open(path, 'wb') as f:
        f.write(tflite_model)
    print(f"  PTQ Enc → {path}  ({os.path.getsize(path)/1024:.1f} KB)")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# %% INFERENCE — FP32
# ═══════════════════════════════════════════════════════════════════════════════

def compute_maps_fp32(ae_model, encoder, image, sea_mask, scaler):
    """
    Returns:
        rmse_fp32  (H, W) — fp32, NaN outside sea mask
        codes_fp32 (H, W, D) — fp32, NaN outside sea mask
    """
    H, W, C   = image.shape
    pixels    = image.reshape(-1, C)
    flat_mask = sea_mask.reshape(-1)

    norm, _ = preprocess_pixels(pixels[flat_mask], fit_scaler=False, scaler=scaler)
    recons  = ae_model.predict(norm,   batch_size=BATCH_SIZE, verbose=0)
    codes   = encoder.predict(norm,    batch_size=BATCH_SIZE, verbose=0)
    orig    = scaler.inverse_transform(recons).astype(np.float32)
    rmse_sea = np.sqrt(np.mean((pixels[flat_mask] - orig) ** 2, axis=1))

    n_hidden   = codes.shape[1]
    rmse_map   = np.full((H * W,), np.nan, dtype=np.float32)
    codes_map  = np.full((H * W, n_hidden), np.nan, dtype=np.float32)
    rmse_map[flat_mask]    = rmse_sea
    codes_map[flat_mask]   = codes
    return rmse_map.reshape(H, W), codes_map.reshape(H, W, n_hidden)


# ═══════════════════════════════════════════════════════════════════════════════
# %% INFERENCE — INT8 TFLITE
# ═══════════════════════════════════════════════════════════════════════════════

def _run_tflite_interpreter(interp, inp_det, out_det, batch_norm, inp_scale, inp_zp):
    results = []
    for px in batch_norm:
        q = np.clip(np.round(px / inp_scale) + inp_zp, -128, 127).astype(np.int8)
        interp.set_tensor(inp_det['index'], q[np.newaxis])
        interp.invoke()
        out = interp.get_tensor(out_det['index'])
        out_scale, out_zp = out_det['quantization']
        results.append(((out.astype(np.float32) - out_zp) * out_scale)[0])
    return np.array(results, dtype=np.float32)


def compute_maps_tflite(tflite_ae_path, tflite_enc_path, image, sea_mask, scaler):
    """
    Single int8 AE pass over sea pixels only.

    Returns:
        rmse_quant     (H, W)    fp32  — NaN outside sea mask
        codes_dq_fp32  (H, W, D) fp32  — dequantised codes, NaN outside sea mask
        codes_raw_i8   (H, W, D) int8  — raw int8 codes, 0 outside sea mask
    """
    H, W, C   = image.shape
    pixels    = image.reshape(-1, C)
    flat_mask = sea_mask.reshape(-1)
    sea_pixels = pixels[flat_mask]

    norm, _ = preprocess_pixels(sea_pixels, fit_scaler=False, scaler=scaler)

    interp_ae  = tf.lite.Interpreter(model_path=tflite_ae_path)
    interp_ae.allocate_tensors()
    inp_ae = interp_ae.get_input_details()[0]
    out_ae = interp_ae.get_output_details()[0]
    ae_in_s, ae_in_zp = inp_ae['quantization']

    interp_enc = tf.lite.Interpreter(model_path=tflite_enc_path)
    interp_enc.allocate_tensors()
    inp_enc = interp_enc.get_input_details()[0]
    out_enc = interp_enc.get_output_details()[0]
    enc_in_s,  enc_in_zp  = inp_enc['quantization']
    enc_out_s, enc_out_zp = out_enc['quantization']

    recons_list, codes_dq_list, codes_raw_list = [], [], []

    for start in tqdm(range(0, len(norm), BATCH_SIZE), desc="  TFLite infer", leave=False):
        batch = norm[start:start + BATCH_SIZE]

        # AE reconstruction
        recons_list.append(_run_tflite_interpreter(
            interp_ae, inp_ae, out_ae, batch, ae_in_s, ae_in_zp))

        # Encoder codes
        batch_dq, batch_raw = [], []
        for px in batch:
            q = np.clip(np.round(px / enc_in_s) + enc_in_zp, -128, 127).astype(np.int8)
            interp_enc.set_tensor(inp_enc['index'], q[np.newaxis])
            interp_enc.invoke()
            raw = interp_enc.get_tensor(out_enc['index'])[0]   # int8
            dq  = (raw.astype(np.float32) - enc_out_zp) * enc_out_s
            batch_dq.append(dq)
            batch_raw.append(raw.copy())
        codes_dq_list.append(np.array(batch_dq,  dtype=np.float32))
        codes_raw_list.append(np.array(batch_raw, dtype=np.int8))

    recons_norm   = np.concatenate(recons_list)
    codes_dq_all  = np.concatenate(codes_dq_list)
    codes_raw_all = np.concatenate(codes_raw_list)

    recons_orig = scaler.inverse_transform(recons_norm).astype(np.float32)
    rmse_sea    = np.sqrt(np.mean((sea_pixels - recons_orig) ** 2, axis=1))

    n_hidden = codes_dq_all.shape[1]

    rmse_map      = np.full((H * W,),          np.nan, dtype=np.float32)
    codes_dq_map  = np.full((H * W, n_hidden), np.nan, dtype=np.float32)
    codes_raw_map = np.zeros((H * W, n_hidden),        dtype=np.int8)

    rmse_map[flat_mask]      = rmse_sea
    codes_dq_map[flat_mask]  = codes_dq_all
    codes_raw_map[flat_mask] = codes_raw_all

    return (rmse_map.reshape(H, W),
            codes_dq_map.reshape(H, W, n_hidden),
            codes_raw_map.reshape(H, W, n_hidden))


# ═══════════════════════════════════════════════════════════════════════════════
# %% DSW β DETECTORS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_beta_map_fp32(R, C_codes, sea_mask, winner, wouter):
    """
    fp32 DSW. Only uses valid sea pixels within the window.
    Pixels with fewer than 2 valid neighbours in the ring get β=NaN.
    """
    half_in  = winner // 2
    half_out = wouter // 2
    H, W     = R.shape
    beta     = np.full((H, W), np.nan, dtype=np.float32)

    sea_coords = np.argwhere(sea_mask)
    for (j, i) in tqdm(sea_coords, desc="  DSW (A)", leave=False):
        r0 = max(j - half_out, 0);  r1 = min(j + half_out + 1, H)
        c0 = max(i - half_out, 0);  c1 = min(i + half_out + 1, W)

        win_R    = R[r0:r1, c0:c1]
        win_C    = C_codes[r0:r1, c0:c1]
        win_sea  = sea_mask[r0:r1, c0:c1]
        cj, ci   = j - r0, i - c0

        win_h, win_w = win_R.shape
        ri  = np.arange(win_h)[:, None]
        ci_ = np.arange(win_w)[None, :]

        inner  = (np.abs(ri - cj) <= half_in) & (np.abs(ci_ - ci) <= half_in)
        centre = (ri == cj) & (ci_ == ci)
        ring   = (~inner | centre) & ~centre & win_sea

        rn = win_R[ring]
        if rn.size < 2:
            continue

        mu_n   = rn.mean()
        sig_n  = rn.std()
        inlier = np.abs(rn - mu_n) <= sig_n

        r_p   = R[j, i]
        wt    = np.where(inlier, r_p / (rn + 1e-12), 0.0).astype(np.float32)
        c_p   = C_codes[j, i]
        c_rng = win_C[ring]
        d     = np.sqrt(np.sum((c_rng - c_p) ** 2, axis=-1))

        beta[j, i] = float(np.sum(wt * d)) / rn.size

    return beta


def compute_beta_map_int8(R, C_codes_i8, sea_mask, winner, wouter):
    """
    int8 DSW. Squared distances in int64, weights and β in fp32.
    Only uses valid sea pixels within the window.
    """
    half_in  = winner // 2
    half_out = wouter // 2
    H, W     = R.shape
    beta     = np.full((H, W), np.nan, dtype=np.float32)

    sea_coords = np.argwhere(sea_mask)
    for (j, i) in tqdm(sea_coords, desc="  DSW (D)", leave=False):
        r0 = max(j - half_out, 0);  r1 = min(j + half_out + 1, H)
        c0 = max(i - half_out, 0);  c1 = min(i + half_out + 1, W)

        win_R   = R[r0:r1, c0:c1]
        win_C   = C_codes_i8[r0:r1, c0:c1]
        win_sea = sea_mask[r0:r1, c0:c1]
        cj, ci  = j - r0, i - c0

        win_h, win_w = win_R.shape
        ri  = np.arange(win_h)[:, None]
        ci_ = np.arange(win_w)[None, :]

        inner  = (np.abs(ri - cj) <= half_in) & (np.abs(ci_ - ci) <= half_in)
        centre = (ri == cj) & (ci_ == ci)
        ring   = (~inner | centre) & ~centre & win_sea

        rn = win_R[ring]
        if rn.size < 2:
            continue

        mu_n   = rn.mean()
        sig_n  = rn.std()
        inlier = np.abs(rn - mu_n) <= sig_n

        r_p   = R[j, i]
        wt    = np.where(inlier, r_p / (rn + 1e-12), 0.0).astype(np.float32)

        c_p   = C_codes_i8[j, i].astype(np.int64)
        c_rng = win_C[ring].astype(np.int64)
        d     = np.sqrt(np.sum((c_rng - c_p) ** 2, axis=-1).astype(np.float32))

        beta[j, i] = float(np.sum(wt * d)) / rn.size

    return beta


# ═══════════════════════════════════════════════════════════════════════════════
# %% DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def display_results(image, sea_mask, rmse_a, beta_a, rmse_d, beta_d, title=''):
    """
    Displays inline:
      Row 0: False colour  |  Sea mask
      Row 1: RMSE-A        |  β-A
      Row 2: RMSE-D        |  β-D
    NaN pixels (outside sea mask) shown in grey.
    """
    fc = false_colour(image)

    def masked_img(arr):
        """Return RGBA image with NaN pixels set to transparent grey."""
        normed = arr.copy()
        valid  = ~np.isnan(normed)
        if valid.any():
            vmin, vmax = np.nanpercentile(normed, 1), np.nanpercentile(normed, 99)
            normed = np.clip((normed - vmin) / (vmax - vmin + 1e-12), 0, 1)
        rgba   = plt.cm.hot(normed)
        rgba[~valid, :3] = 0.25
        rgba[~valid,  3] = 1.0
        return rgba

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle(f"HYPSO Anomaly Detection — FP32 (A) vs HybridRaw (D)\n{title}",
                 fontsize=13, fontweight='bold')

    def show(ax, img, title, cmap=None):
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    show(axes[0, 0], fc,                            'False Colour (RGB)')
    show(axes[0, 1], sea_mask.astype(np.uint8),     'Sea Mask',  cmap='gray')
    show(axes[1, 0], masked_img(rmse_a),             'RMSE — FP32 (A)')
    show(axes[1, 1], masked_img(beta_a),             'β Score — FP32 (A)')
    show(axes[2, 0], masked_img(rmse_d),             'RMSE — HybridRaw (D)')
    show(axes[2, 1], masked_img(beta_d),             'β Score — HybridRaw (D)')

    plt.tight_layout()
    plt.show()
    print("  Displayed.")


# ═══════════════════════════════════════════════════════════════════════════════
# %% MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

# ── Resolve paths from capture directory ──────────────────────────────────────
l1a_path, labels_path = resolve_hypso_paths(CAPTURE_DIR)
capture_name = os.path.basename(CAPTURE_DIR.rstrip('/'))
print(f"Capture : {capture_name}")
print(f"L1A     : {l1a_path}")
print(f"Labels  : {labels_path}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading L1A cube …")
image   = load_hypso_image(l1a_path)
H, W, C = image.shape
print(f"  Image: {H}×{W}×{C}")

raw_labels = np.fromfile(labels_path, dtype=np.uint8).reshape((HYPSO_H, HYPSO_W))
print(f"  Raw label unique values: {np.unique(raw_labels)}")
label_arr = raw_labels.astype(np.int32)   # no offset — values are already 0-based
for v in np.unique(label_arr):
    print(f"    label {v}: {(label_arr == v).sum():,} pixels")

sea_mask  = (label_arr == SEA_LABEL)
n_sea     = sea_mask.sum()
print(f"  Sea pixels (SEA_LABEL={SEA_LABEL}): {n_sea:,} / {H*W:,}  ({100*n_sea/(H*W):.1f}%)")
if n_sea == 0:
    raise RuntimeError(f"No sea pixels found — raw unique values are {np.unique(raw_labels)}, adjust SEA_LABEL accordingly.")

# ── Training data ─────────────────────────────────────────────────────────────
sea_pixels = image.reshape(-1, C)[sea_mask.reshape(-1)]
n_train    = min(N_SAMPLES_TRAIN, n_sea) if N_SAMPLES_TRAIN else n_sea
rng        = np.random.default_rng(RANDOM_SEED)
idx        = rng.choice(n_sea, size=n_train, replace=False)
train_px   = sea_pixels[idx]
norm_px, scaler = preprocess_pixels(train_px, fit_scaler=True)
print(f"  Training on {n_train:,} sea pixels")

# ── RBM ───────────────────────────────────────────────────────────────────────
set_seed(RANDOM_SEED)
rbm = GaussianRBM_NP(n_visible=C, n_hidden=N_HIDDEN, k=RBM_K)
rbm.train(norm_px, n_epochs=RBM_EPOCHS, lr=RBM_LEARNING_RATE,
          momentum=RBM_MOMENTUM, weight_decay=RBM_WEIGHT_DECAY,
          batch_size=BATCH_SIZE)

# ── FP32 AE ───────────────────────────────────────────────────────────────────
set_seed(RANDOM_SEED)
ae_fp32 = build_autoencoder_fp32(C, N_HIDDEN)
initialize_from_rbm(ae_fp32, rbm)
train_autoencoder(ae_fp32, norm_px, name='ae_fp32')
enc_fp32 = build_encoder_only(ae_fp32)

print("  Computing FP32 maps …")
rmse_fp32, codes_fp32 = compute_maps_fp32(ae_fp32, enc_fp32, image, sea_mask, scaler)

# ── Int8 AE (PTQ) ─────────────────────────────────────────────────────────────
set_seed(RANDOM_SEED)
ae_ptq = build_autoencoder_fp32(C, N_HIDDEN)
ae_ptq._name = 'ae_ptq'
initialize_from_rbm(ae_ptq, rbm)
train_autoencoder(ae_ptq, norm_px, name='ae_ptq')

tflite_ae_path  = export_tflite_ptq(ae_ptq, norm_px[:200], f'{capture_name}_ae_ptq')
tflite_enc_path = export_tflite_encoder_ptq(ae_ptq, norm_px[:200], f'{capture_name}_ae_ptq')

print("  Computing Int8 maps …")
rmse_quant, codes_quant_fp32, codes_quant_raw_i8 = compute_maps_tflite(
    tflite_ae_path, tflite_enc_path, image, sea_mask, scaler)

# ── DSW β maps ────────────────────────────────────────────────────────────────
print(f"  Computing β maps (DSW w={DSW_WINNER}, W={DSW_WOUTER}) …")
beta_a = compute_beta_map_fp32(rmse_fp32,  codes_fp32,        sea_mask, DSW_WINNER, DSW_WOUTER)
beta_d = compute_beta_map_int8(rmse_quant, codes_quant_raw_i8, sea_mask, DSW_WINNER, DSW_WOUTER)

# ── Display ───────────────────────────────────────────────────────────────────
display_results(image, sea_mask, rmse_fp32, beta_a, rmse_quant, beta_d, title=capture_name)

print("\nDONE.")