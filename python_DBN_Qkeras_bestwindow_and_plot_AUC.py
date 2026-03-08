# %% [markdown]
# MAWDBN – Hyperspectral Anomaly Detection on ABU Dataset
#
# Compares four variants — all use a single int8 AE pass for B/C/D, DSW arithmetic always fp32:
#   A) FP32      — fp32 network  → fp32 RMSE (type)  + fp32 codes (type)
#   B) Quant     — int8 network  → fp32 RMSE (type)  + fp32 codes (type, dequantised)
#   C) HybridQ   — int8 network  → fp32 RMSE (type)  + fp32 codes (type, dequantised)   [same as B — to be distinguished by window]
#   D) HybridRaw — int8 network  → fp32 RMSE (type)  + raw int8 codes cast to fp32 (no dequantisation)
#
# Both networks (FP32 and PTQ int8) are trained once per scene from the same RBM init.
# DSW window search: all legal (winner, wouter) pairs, penalty factor fixed at 0.
# Per scene, the best β-AUC window is selected per variant independently.

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

ABU_DATASET_DIR = 'ABU_DATASET'
RESULTS_DIR     = 'new_results'

# Quantisation
QUANTIZATION_MODE = 'ptq'       # only 'ptq' supported in this script
QUANT_BITS        = 8
QUANT_INTEGER     = 2
QUANT_ACT_BITS    = 8
QUANT_ACT_INT     = 1

# Model
N_HIDDEN = 13

# RBM (CD-1)
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

# Training data
N_SAMPLES_TRAIN = 50000
BATCH_SIZE      = 2048

# DSW window search — all legal (winner, wouter) pairs will be generated
# from: inner ∈ {1,3,5}, outer ∈ {3,5,7,9,11,13}, with wouter > winner+1
# (legal = wouter > winner so a ring of neighbours actually exists)
DSW_INNER_SIZES = [1, 3, 5]
DSW_OUTER_SIZES = [3, 5, 7, 9, 11, 13]
DSW_PF          = 0   # penalty factor fixed at 0

THRESHOLD_PERCENTILE = 95
RANDOM_SEED          = 42

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import os, random, warnings
from datetime import datetime
from itertools import product as iproduct

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io as sio
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, losses
from tensorflow.keras.regularizers import l2 as _l2

ABU_SCENES = [
    'abu-airport-1', 'abu-airport-2', 'abu-airport-3', 'abu-airport-4',
    'abu-beach-1',   'abu-beach-2',   'abu-beach-3',   'abu-beach-4',
    'abu-urban-1',   'abu-urban-2',   'abu-urban-3',   'abu-urban-4',
    'abu-urban-5',
]

# Build legal window pairs: wouter > winner (ring must have pixels)
DSW_WINDOW_PAIRS = sorted(
    [(wi, wo) for wi, wo in iproduct(DSW_INNER_SIZES, DSW_OUTER_SIZES) if wo > wi],
    key=lambda x: (x[0], x[1])
)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"  GPU(s): {[g.name for g in gpus]}")
else:
    print("  No GPU — running on CPU.")
print(f"  TF {tf.__version__}  |  quant={QUANTIZATION_MODE}")
print(f"  Window pairs to search ({len(DSW_WINDOW_PAIRS)}): {DSW_WINDOW_PAIRS}")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_abu_mat(mat_path):
    mat = sio.loadmat(mat_path)
    if 'data' in mat:
        image = mat['data'].astype(np.float32)
    else:
        candidates = {k: v for k, v in mat.items()
                      if not k.startswith('_') and isinstance(v, np.ndarray)
                      and v.ndim == 3}
        key   = max(candidates, key=lambda k: candidates[k].size)
        image = candidates[key].astype(np.float32)

    if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0))

    gt_mask = None
    if 'map' in mat:
        gt_mask = mat['map'].astype(np.uint8).squeeze()
        if gt_mask.shape != image.shape[:2]:
            gt_mask = gt_mask.T

    return image, gt_mask


def preprocess_pixels(pixels, fit_scaler=True, scaler=None):
    if fit_scaler:
        scaler     = StandardScaler()
        normalised = scaler.fit_transform(pixels)
    else:
        normalised = scaler.transform(pixels)
    return normalised.astype(np.float32), scaler


# ═══════════════════════════════════════════════════════════════════════════════
# RBM
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
# AUTOENCODER
# ═══════════════════════════════════════════════════════════════════════════════

def build_autoencoder_fp32(n_visible, n_hidden):
    wd  = AE_WEIGHT_DECAY
    inp = keras.Input(shape=(n_visible,), name='spectrum')
    z   = layers.Dense(n_hidden, activation='sigmoid',
                       kernel_regularizer=_l2(wd/2),
                       name='encoder_dense')(inp)
    out = layers.Dense(n_visible, activation='linear',
                       kernel_regularizer=_l2(wd/2),
                       name='decoder_dense')(z)
    return keras.Model(inputs=inp, outputs=out, name='ae_fp32')


def build_encoder_only(autoencoder):
    return keras.Model(inputs=autoencoder.input,
                       outputs=autoencoder.get_layer('encoder_dense').output,
                       name='encoder')


def initialize_from_rbm(model, rbm):
    model.get_layer('encoder_dense').set_weights([rbm.W.T, rbm.h_bias])
    model.get_layer('decoder_dense').set_weights([rbm.W,   rbm.v_bias])


def train_autoencoder(model, X_train, save_dir, model_name):
    model.compile(optimizer=optimizers.Adam(learning_rate=AE_LEARNING_RATE),
                  loss=losses.MeanSquaredError())
    ckpt = os.path.join(save_dir, f'{model_name}.h5')
    history = model.fit(
        X_train, X_train,
        epochs=AE_EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.0, shuffle=True, verbose=0,
        callbacks=[
            callbacks.EarlyStopping(monitor='loss', patience=AE_PATIENCE,
                                    restore_best_weights=True, verbose=0),
            callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5),
            callbacks.ModelCheckpoint(filepath=ckpt, monitor='loss',
                                      save_best_only=True, save_format='h5', verbose=0),
        ],
    )
    n_run = len(history.history['loss'])
    print(f"  AE [{model.name}]  epochs={n_run}/{AE_EPOCHS}  "
          f"best loss: {min(history.history['loss']):.6f}")
    return history.history


# ═══════════════════════════════════════════════════════════════════════════════
# PTQ EXPORT & TFLITE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def export_tflite_ptq(model, rep_data, save_dir, model_name):
    """Export full autoencoder as int8 TFLite."""
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
    path = os.path.join(save_dir, f'{model_name}_ptq_int8.tflite')
    with open(path, 'wb') as f:
        f.write(tflite_model)
    print(f"  PTQ AE  → {path}  ({os.path.getsize(path)/1024:.1f} KB)")
    return path


def export_tflite_encoder_ptq(keras_ae, rep_data, save_dir, model_name):
    """Export only the encoder half as int8 TFLite."""
    enc = build_encoder_only(keras_ae)
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
    path = os.path.join(save_dir, f'{model_name}_ptq_int8_encoder.tflite')
    with open(path, 'wb') as f:
        f.write(tflite_model)
    print(f"  PTQ Enc → {path}  ({os.path.getsize(path)/1024:.1f} KB)")
    return path


def _run_tflite_interpreter(interp, inp_det, out_det, batch_norm):
    """Run a TFLite interpreter on a float32 normalised batch, return float32 output."""
    inp_scale, inp_zp = inp_det['quantization']
    out_scale, out_zp = out_det['quantization']
    results = []
    for px in batch_norm:
        q = np.clip(np.round(px / inp_scale) + inp_zp, -128, 127).astype(np.int8)
        interp.set_tensor(inp_det['index'], q[np.newaxis])
        interp.invoke()
        o = interp.get_tensor(out_det['index'])
        results.append(((o.astype(np.float32) - out_zp) * out_scale)[0])
    return np.array(results, dtype=np.float32)


def compute_maps_tflite(tflite_ae_path, tflite_enc_path, image, scaler):
    """
    Single int8 AE pass over the full image.

    Returns:
        rmse_quant      (H, W)    fp32  — RMSE from dequantised int8 AE output vs original input
        codes_dq_fp32   (H, W, D) fp32  — encoder codes dequantised to fp32  (for variant B)
        codes_dq_i8     (H, W, D) int8  — dequantised values rounded/clipped back to int8  (for variant C)
        codes_raw_i8    (H, W, D) int8  — raw int8 encoder output, no dequantisation  (for variant D)
    """
    H, W, C     = image.shape
    pixels_flat = image.reshape(-1, C)
    pixels_norm, _ = preprocess_pixels(pixels_flat, fit_scaler=False, scaler=scaler)

    interp_ae  = tf.lite.Interpreter(model_path=tflite_ae_path)
    interp_ae.allocate_tensors()
    inp_ae = interp_ae.get_input_details()[0]
    out_ae = interp_ae.get_output_details()[0]

    interp_enc = tf.lite.Interpreter(model_path=tflite_enc_path)
    interp_enc.allocate_tensors()
    inp_enc = interp_enc.get_input_details()[0]
    out_enc = interp_enc.get_output_details()[0]
    enc_in_s,  enc_in_zp  = inp_enc['quantization']
    enc_out_s, enc_out_zp = out_enc['quantization']

    recons_list, codes_dq_fp32_list, codes_dq_i8_list, codes_raw_i8_list = [], [], [], []
    for start in tqdm(range(0, H*W, BATCH_SIZE), desc="  TFLite infer", leave=False):
        batch = pixels_norm[start:start + BATCH_SIZE]

        # AE: dequantised reconstruction for RMSE
        recons_list.append(_run_tflite_interpreter(interp_ae, inp_ae, out_ae, batch))

        # Encoder: three code representations
        batch_dq_fp32, batch_dq_i8, batch_raw_i8 = [], [], []
        for px in batch:
            q = np.clip(np.round(px / enc_in_s) + enc_in_zp, -128, 127).astype(np.int8)
            interp_enc.set_tensor(inp_enc['index'], q[np.newaxis])
            interp_enc.invoke()
            raw = interp_enc.get_tensor(out_enc['index'])[0]           # int8
            dq  = (raw.astype(np.float32) - enc_out_zp) * enc_out_s   # fp32 dequantised
            batch_dq_fp32.append(dq)
            batch_dq_i8.append(np.clip(np.round(dq), -128, 127).astype(np.int8))
            batch_raw_i8.append(raw.copy())                             # raw int8
        codes_dq_fp32_list.append(np.array(batch_dq_fp32, dtype=np.float32))
        codes_dq_i8_list.append(np.array(batch_dq_i8,   dtype=np.int8))
        codes_raw_i8_list.append(np.array(batch_raw_i8,  dtype=np.int8))

    recons_norm     = np.concatenate(recons_list)
    codes_dq_fp32   = np.concatenate(codes_dq_fp32_list)
    codes_dq_i8     = np.concatenate(codes_dq_i8_list)
    codes_raw_i8    = np.concatenate(codes_raw_i8_list)

    recons_orig = scaler.inverse_transform(recons_norm).astype(np.float32)
    rmse_flat   = np.sqrt(np.mean((pixels_flat - recons_orig) ** 2, axis=1))
    n_hidden    = codes_dq_fp32.shape[1]
    return (rmse_flat.reshape(H, W),
            codes_dq_fp32.reshape(H, W, n_hidden),
            codes_dq_i8.reshape(H, W, n_hidden),
            codes_raw_i8.reshape(H, W, n_hidden))


# ═══════════════════════════════════════════════════════════════════════════════
# FP32 RECONSTRUCTION & CODES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_maps_fp32(model, encoder, image, scaler):
    """
    Returns:
        rmse_fp32  (H, W)
        codes_fp32 (H, W, D)
    """
    H, W, C = image.shape
    pixels  = image.reshape(-1, C)
    norm, _ = preprocess_pixels(pixels, fit_scaler=False, scaler=scaler)
    recons  = model.predict(norm,   batch_size=BATCH_SIZE, verbose=0)
    codes   = encoder.predict(norm, batch_size=BATCH_SIZE, verbose=0)
    orig    = scaler.inverse_transform(recons).astype(np.float32)
    rmse_f  = np.sqrt(np.mean((pixels - orig) ** 2, axis=1))
    return rmse_f.reshape(H, W), codes.reshape(H, W, encoder.output_shape[-1])


# ═══════════════════════════════════════════════════════════════════════════════
# DSW β DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def compute_beta_map_int8(R, C_codes_i8, winner, wouter):
    """
    MAWDBN β anomaly score with int8 codes.
    Squared distances computed in int64 to avoid overflow.
    Weights and final score remain fp32.
    C_codes_i8: (H, W, D) dtype=int8
    """
    half_in  = winner // 2
    half_out = wouter // 2
    H, W     = R.shape
    beta     = np.zeros((H, W), dtype=np.float32)

    for j in range(H):
        r0 = max(j - half_out, 0)
        r1 = min(j + half_out + 1, H)
        for i in range(W):
            c0 = max(i - half_out, 0)
            c1 = min(i + half_out + 1, W)

            win_R = R[r0:r1, c0:c1]
            win_C = C_codes_i8[r0:r1, c0:c1]
            cj    = j - r0
            ci    = i - c0

            win_h, win_w = win_R.shape
            ri  = np.arange(win_h)[:, None]
            ci_ = np.arange(win_w)[None, :]

            inner_mask  = (np.abs(ri - cj) <= half_in) & (np.abs(ci_ - ci) <= half_in)
            centre_mask = (ri == cj) & (ci_ == ci)
            ring = (~inner_mask) | centre_mask
            ring[cj, ci] = False

            rn = win_R[ring]
            if rn.size == 0:
                continue

            mu_n   = rn.mean()
            sig_n  = rn.std()
            inlier = np.abs(rn - mu_n) <= sig_n

            r_p = R[j, i]
            wt  = np.where(inlier, r_p / (rn + 1e-12), 0.0).astype(np.float32)

            c_p   = C_codes_i8[j, i].astype(np.int64)
            c_rng = win_C[ring].astype(np.int64)
            # squared distance in int64, then sqrt to fp32
            d = np.sqrt(np.sum((c_rng - c_p) ** 2, axis=-1).astype(np.float32))

            beta[j, i] = float(np.sum(wt * d)) / rn.size

    return beta


def compute_beta_map(R, C_codes, winner, wouter):
    """
    MAWDBN β anomaly score — fp32 codes, fp32 throughout.

      d_j    = ||c_nj - c_p||_2
      wt_j   = r_p / r_nj   if |r_nj - mu_n| <= sigma_n  (inlier, pf=0)
             = 0             otherwise
      beta_p = (1/k) * sum(wt_j * d_j)
    """
    half_in  = winner // 2
    half_out = wouter // 2
    H, W     = R.shape
    beta     = np.zeros((H, W), dtype=np.float32)

    for j in range(H):
        r0 = max(j - half_out, 0)
        r1 = min(j + half_out + 1, H)
        for i in range(W):
            c0 = max(i - half_out, 0)
            c1 = min(i + half_out + 1, W)

            win_R = R[r0:r1, c0:c1]
            win_C = C_codes[r0:r1, c0:c1]
            cj    = j - r0
            ci    = i - c0

            win_h, win_w = win_R.shape
            ri  = np.arange(win_h)[:, None]
            ci_ = np.arange(win_w)[None, :]

            # Ring mask: outside inner square but not the centre pixel
            inner_mask = (np.abs(ri - cj) <= half_in) & (np.abs(ci_ - ci) <= half_in)
            centre_mask = (ri == cj) & (ci_ == ci)
            ring = (~inner_mask) | centre_mask   # invert: keep what's outside inner
            ring[cj, ci] = False                 # always exclude centre itself

            rn = win_R[ring]
            if rn.size == 0:
                continue

            mu_n   = rn.mean()
            sig_n  = rn.std()
            inlier = np.abs(rn - mu_n) <= sig_n

            r_p = R[j, i]
            wt  = np.where(inlier, r_p / (rn + 1e-12), 0.0)

            c_p   = C_codes[j, i]
            c_rng = win_C[ring]
            d     = np.sqrt(np.sum((c_rng - c_p) ** 2, axis=-1))

            beta[j, i] = float(np.sum(wt * d)) / rn.size

    return beta


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_auc_roc(score_map, gt_mask):
    y_true  = gt_mask.reshape(-1).astype(int)
    y_score = score_map.reshape(-1)
    auc     = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(auc), fpr, tpr


# ═══════════════════════════════════════════════════════════════════════════════
# PER-SCENE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_scene(scene_name, mat_path):
    """
    Train both FP32 and quantised networks once, then evaluate all window
    configurations for four variant combinations:

      A  FP32     : fp32 RMSE  + fp32 codes
      B  Quant    : int8 RMSE  + int8 codes (dequantised)     — original quant version
      C  HybridQ  : fp32 RMSE  + int8 codes (dequantised)
      D  HybridRaw: fp32 RMSE  + raw int8 codes (not dequantised)

    Returns a dict with per-variant best-window AUC results and ROC curve.
    """
    set_seed(RANDOM_SEED)
    save_dir = os.path.join(RESULTS_DIR, scene_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  {scene_name.upper()}")
    print(f"{'═'*70}")

    image, gt_mask = load_abu_mat(mat_path)
    H, W, C = image.shape

    n_train  = min(N_SAMPLES_TRAIN, H * W) if N_SAMPLES_TRAIN else H * W
    rng      = np.random.default_rng(RANDOM_SEED)
    idx      = rng.choice(H * W, size=n_train, replace=False)
    train_px = image.reshape(-1, C)[idx]
    norm_px, scaler = preprocess_pixels(train_px, fit_scaler=True)
    print(f"  Image: {H}×{W}×{C}  |  train px: {n_train:,}")

    # ── RBM ──────────────────────────────────────────────────────────────────
    rbm = GaussianRBM_NP(n_visible=C, n_hidden=N_HIDDEN, k=RBM_K)
    rbm.train(norm_px, n_epochs=RBM_EPOCHS, lr=RBM_LEARNING_RATE,
              momentum=RBM_MOMENTUM, weight_decay=RBM_WEIGHT_DECAY,
              batch_size=BATCH_SIZE)

    # ── FP32 autoencoder ─────────────────────────────────────────────────────
    ae_fp32 = build_autoencoder_fp32(C, N_HIDDEN)
    initialize_from_rbm(ae_fp32, rbm)
    train_autoencoder(ae_fp32, norm_px, save_dir, f'ae_fp32_{scene_name}')
    enc_fp32 = build_encoder_only(ae_fp32)
    rmse_fp32, codes_fp32 = compute_maps_fp32(ae_fp32, enc_fp32, image, scaler)

    # ── Quantised autoencoder (PTQ) ───────────────────────────────────────────
    set_seed(RANDOM_SEED)
    ae_ptq = build_autoencoder_fp32(C, N_HIDDEN)   # same architecture, PTQ at export
    ae_ptq._name = 'ae_ptq'
    initialize_from_rbm(ae_ptq, rbm)
    train_autoencoder(ae_ptq, norm_px, save_dir, f'ae_ptq_{scene_name}')

    tflite_ae_path  = export_tflite_ptq(ae_ptq, norm_px[:200], save_dir,
                                         f'ae_ptq_{scene_name}')
    tflite_enc_path = export_tflite_encoder_ptq(ae_ptq, norm_px[:200], save_dir,
                                                 f'ae_ptq_{scene_name}')
    rmse_quant, codes_quant_fp32, codes_quant_dq_i8, codes_quant_raw_i8 = compute_maps_tflite(
        tflite_ae_path, tflite_enc_path, image, scaler)

    print(f"\n  Searching {len(DSW_WINDOW_PAIRS)} window configurations …")

    # ── Window search ─────────────────────────────────────────────────────────
    # A: fp32 network  — fp32 RMSE, fp32 codes                          → fp32 DSW
    # B: int8 network  — fp32 RMSE (dequantised), fp32 codes (dequantised) → fp32 DSW
    # C: int8 network  — fp32 RMSE (dequantised), int8 codes (dequantised, re-clipped) → int8 DSW
    # D: int8 network  — fp32 RMSE (dequantised), raw int8 codes           → int8 DSW
    variants = {
        'A_fp32':     {'rmse': rmse_fp32,  'codes': codes_fp32,         'int8_codes': False, 'label': 'FP32'},
        'B_quant':    {'rmse': rmse_quant, 'codes': codes_quant_fp32,   'int8_codes': False, 'label': 'Quant'},
        'C_hybridq':  {'rmse': rmse_quant, 'codes': codes_quant_dq_i8,  'int8_codes': True,  'label': 'HybridQ'},
        'D_hybridraw':{'rmse': rmse_quant, 'codes': codes_quant_raw_i8, 'int8_codes': True,  'label': 'HybridRaw'},
    }

    # Initialise trackers
    for vk in variants:
        variants[vk].update({
            'best_auc':        -1.0,
            'best_window':     None,
            'best_fpr':        None,
            'best_tpr':        None,
            'all_auc':         {},   # (wi,wo) -> auc
        })

    if gt_mask is None:
        print("  No ground truth — skipping AUC evaluation.")
        return {'scene': scene_name, 'gt': False}

    for wi, wo in tqdm(DSW_WINDOW_PAIRS, desc="  Windows", leave=True):
        beta_a = compute_beta_map(rmse_fp32,  codes_fp32,            wi, wo)
        beta_b = compute_beta_map(rmse_quant, codes_quant_fp32,      wi, wo)
        beta_c = compute_beta_map_int8(rmse_quant, codes_quant_dq_i8,  wi, wo)
        beta_d = compute_beta_map_int8(rmse_quant, codes_quant_raw_i8, wi, wo)

        maps = {'A_fp32': beta_a, 'B_quant': beta_b,
                'C_hybridq': beta_c, 'D_hybridraw': beta_d}
        for vk, bmap in maps.items():
            auc, fpr, tpr = compute_auc_roc(bmap, gt_mask)
            variants[vk]['all_auc'][(wi, wo)] = auc
            if auc > variants[vk]['best_auc']:
                variants[vk]['best_auc']    = auc
                variants[vk]['best_window'] = (wi, wo)
                variants[vk]['best_fpr']    = fpr
                variants[vk]['best_tpr']    = tpr

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n  ┌{'─'*60}┐")
    print(f"  │ {'Scene':<20} {'Best AUC':>10}  {'Window':>12}  {'Label':<10}│")
    print(f"  ├{'─'*60}┤")
    for vk, v in variants.items():
        wi, wo = v['best_window']
        print(f"  │ {scene_name:<20} {v['best_auc']:>10.4f}  "
              f"({'w='+str(wi)+',W='+str(wo):>12})  {v['label']:<10}│")
    print(f"  └{'─'*60}┘")

    # ── Per-scene ROC plot ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    colours = {'A_fp32': '#1f77b4', 'B_quant': '#d62728',
               'C_hybridq': '#2ca02c', 'D_hybridraw': '#9467bd'}
    for vk, v in variants.items():
        wi, wo = v['best_window']
        ax.plot(v['best_fpr'], v['best_tpr'], lw=2, color=colours[vk],
                label=f"{v['label']}  AUC={v['best_auc']:.4f}  (w={wi},W={wo})")
    ax.plot([0,1],[0,1],'k:',lw=0.8)
    ax.set_xlim(0,1); ax.set_ylim(0,1.01)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate',  fontsize=11)
    ax.set_title(f"Best-window ROC — {scene_name}", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    roc_path = os.path.join(save_dir, f'{scene_name}_best_roc.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ROC → {roc_path}")

    # ── RMSE AUC (no DSW) ────────────────────────────────────────────────────
    auc_rmse_fp32,  _, _ = compute_auc_roc(rmse_fp32,  gt_mask)
    auc_rmse_quant, _, _ = compute_auc_roc(rmse_quant, gt_mask)

    return {
        'scene':          scene_name,
        'gt':             True,
        'variants':       variants,
        'auc_rmse_fp32':  auc_rmse_fp32,
        'auc_rmse_quant': auc_rmse_quant,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results):
    valid = [r for r in results if r.get('gt')]
    if not valid:
        print("No valid results."); return

    bdr = '═' * 130
    div = '─' * 130

    vkeys   = ['A_fp32', 'B_quant', 'C_hybridq', 'D_hybridraw']
    vlabels = {'A_fp32': 'FP32', 'B_quant': 'Quant',
               'C_hybridq': 'HybridQ', 'D_hybridraw': 'HybridRaw'}

    def fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A"

    print()
    print(bdr)
    print(f"  MAWDBN BEST-WINDOW β-AUC SUMMARY  |  dynamic window search  |  pf={DSW_PF}")
    print(bdr)
    header = f"  {'Scene':<22}  {'RMSE-FP32':>9}  {'RMSE-Qt':>9}"
    for vk in vkeys:
        header += f"  {vlabels[vk]:>10}  {'window':>10}"
    print(header)
    print(div)

    type_groups = {
        'AIRPORT': [r for r in valid if 'airport' in r['scene']],
        'BEACH':   [r for r in valid if 'beach'   in r['scene']],
        'URBAN':   [r for r in valid if 'urban'   in r['scene']],
    }

    avg_by_variant  = {vk: [] for vk in vkeys}
    avg_rmse_fp32   = []
    avg_rmse_quant  = []

    for tname, grp in type_groups.items():
        if not grp: continue
        for r in grp:
            vs   = r['variants']
            line = f"  {r['scene']:<22}  {fmt(r['auc_rmse_fp32']):>9}  {fmt(r['auc_rmse_quant']):>9}"
            for vk in vkeys:
                wi, wo = vs[vk]['best_window']
                line  += f"  {fmt(vs[vk]['best_auc']):>10}  (w={wi},W={wo})"
                avg_by_variant[vk].append(vs[vk]['best_auc'])
            avg_rmse_fp32.append(r['auc_rmse_fp32'])
            avg_rmse_quant.append(r['auc_rmse_quant'])
            print(line)
        print(div)

        grp_rmse_fp32  = [r['auc_rmse_fp32']  for r in grp]
        grp_rmse_quant = [r['auc_rmse_quant'] for r in grp]
        avg_line = (f"  {'  AVG '+tname+' ('+str(len(grp))+')':<22}"
                    f"  {np.mean(grp_rmse_fp32):>9.4f}  {np.mean(grp_rmse_quant):>9.4f}")
        for vk in vkeys:
            grp_aucs = [r['variants'][vk]['best_auc'] for r in grp]
            avg_line += f"  {np.mean(grp_aucs):>10.4f}  {'(avg)':>10}"
        print(avg_line)
        print(div)

    overall_line = (f"  {'  OVERALL ('+str(len(valid))+')':<22}"
                    f"  {np.mean(avg_rmse_fp32):>9.4f}  {np.mean(avg_rmse_quant):>9.4f}")
    for vk in vkeys:
        overall_line += f"  {np.mean(avg_by_variant[vk]):>10.4f}  {'(avg)':>10}"
    print(overall_line)
    print(bdr)

    all_a = np.mean(avg_by_variant['A_fp32'])
    print()
    print(f"  QUANTISATION IMPACT (vs FP32):")
    for vk in ['B_quant', 'C_hybridq', 'D_hybridraw']:
        delta = np.mean(avg_by_variant[vk]) - all_a
        print(f"    {vlabels[vk]:<12} Δ : {delta:+.4f}  "
              f"({'improved' if delta > 0 else 'degraded'})")
    print()

    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(RESULTS_DIR, f'summary_{ts}.txt')
    with open(path, 'w') as f:
        f.write("MAWDBN BEST-WINDOW β-AUC SUMMARY\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Window pairs searched: {DSW_WINDOW_PAIRS}\n\n")
        for r in valid:
            vs = r['variants']
            f.write(f"{r['scene']}\n")
            f.write(f"  {'RMSE-FP32':<12}  AUC={r['auc_rmse_fp32']:.4f}  (no DSW)\n")
            f.write(f"  {'RMSE-Quant':<12}  AUC={r['auc_rmse_quant']:.4f}  (no DSW)\n")
            for vk in vkeys:
                wi, wo = vs[vk]['best_window']
                f.write(f"  {vlabels[vk]:<12}  AUC={vs[vk]['best_auc']:.4f}  "
                        f"best_window=({wi},{wo})\n")
            f.write("\n")
        f.write("\nOVERALL AVERAGES\n")
        f.write(f"  {'RMSE-FP32':<12}  {np.mean(avg_rmse_fp32):.4f}\n")
        f.write(f"  {'RMSE-Quant':<12}  {np.mean(avg_rmse_quant):.4f}\n")
        for vk in vkeys:
            f.write(f"  {vlabels[vk]:<12}  {np.mean(avg_by_variant[vk]):.4f}\n")
    print(f"  Summary saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# COLLECTIVE AVERAGED ROC PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_collective_roc(results):
    """
    2×2 figure — one panel per variant.
    Each panel: thin per-scene lines (coloured by scene type) +
                bold dataset-average curve.
    """
    valid = [r for r in results if r.get('gt')]
    if not valid:
        print("  No ROC data."); return

    fpr_grid = np.linspace(0, 1, 500)

    vkeys   = ['A_fp32', 'B_quant', 'C_hybridq', 'D_hybridraw']
    vlabels = {'A_fp32': 'FP32', 'B_quant': 'Quant',
               'C_hybridq': 'HybridQ', 'D_hybridraw': 'HybridRaw'}
    colours = {'A_fp32': '#1f77b4', 'B_quant': '#d62728',
               'C_hybridq': '#2ca02c', 'D_hybridraw': '#9467bd'}
    type_colours = {'airport': '#4878cf', 'beach': '#f58231', 'urban': '#3cb44b'}

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharey=True, sharex=True)
    axes_flat = axes.flatten()
    fig.suptitle(
        "Collective Best-Window ROC — ABU Dataset\n"
        "Bold = dataset average  |  thin lines = individual scenes",
        fontsize=13, fontweight='bold')

    for ax, vk in zip(axes_flat, vkeys):
        tprs_interp, aucs = [], []
        for r in valid:
            fpr = r['variants'][vk]['best_fpr']
            tpr = r['variants'][vk]['best_tpr']
            auc = r['variants'][vk]['best_auc']
            sc_type = next((t for t in type_colours if t in r['scene']), 'urban')
            ax.plot(fpr, tpr, lw=0.8, alpha=0.45, color=type_colours[sc_type])
            tprs_interp.append(np.interp(fpr_grid, fpr, tpr))
            aucs.append(auc)

        mean_tpr = np.mean(tprs_interp, axis=0)
        avg_auc  = np.mean(aucs)
        ax.plot(fpr_grid, mean_tpr, lw=3, color=colours[vk])
        ax.plot([0, 1], [0, 1], 'k:', lw=0.8)

        type_patches = [mpatches.Patch(color=c, label=t.capitalize(), alpha=0.6)
                        for t, c in type_colours.items()]
        avg_handle = plt.Line2D([0], [0], color=colours[vk], lw=3,
                                label=f"Avg AUC={avg_auc:.4f}")
        ax.legend(handles=[avg_handle] + type_patches, fontsize=9, loc='lower right')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate',  fontsize=11)
        ax.set_title(f"{vk.split('_')[0]}  —  {vlabels[vk]}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'collective_roc_all_variants.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Collective ROC → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# AUC BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════

def plot_auc_barchart(results):
    """
    Horizontal grouped bar chart: best-window β AUC per scene for all four variants.
    """
    valid = [r for r in results if r.get('gt')]
    if not valid: return

    vkeys   = ['A_fp32', 'B_quant', 'C_hybridq', 'D_hybridraw']
    vlabels = {'A_fp32': 'FP32', 'B_quant': 'Quant',
               'C_hybridq': 'HybridQ', 'D_hybridraw': 'HybridRaw'}
    colours = {'A_fp32': '#1f77b4', 'B_quant': '#d62728',
               'C_hybridq': '#2ca02c', 'D_hybridraw': '#9467bd'}
    hatches = {'A_fp32': '', 'B_quant': '///', 'C_hybridq': 'xxx', 'D_hybridraw': '...'}

    scenes = [r['scene'] for r in valid]
    n      = len(scenes)
    y      = np.arange(n)
    h      = 0.18
    offsets = [-1.5*h, -0.5*h, 0.5*h, 1.5*h]

    type_colour = {'airport': '#4878cf', 'beach': '#f58231', 'urban': '#3cb44b'}

    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.75)))

    for vk, off in zip(vkeys, offsets):
        aucs = [r['variants'][vk]['best_auc'] for r in valid]
        bars = ax.barh(y + off, aucs, h, color=colours[vk],
                       hatch=hatches[vk], alpha=0.80,
                       label=vlabels[vk], edgecolor='white', linewidth=0.4)
        for bar, v in zip(bars, aucs):
            ax.text(max(v - 0.001, 0.5),
                    bar.get_y() + bar.get_height() / 2,
                    f'{v:.4f}', va='center', ha='right',
                    fontsize=6.5, color='white', fontweight='bold')

    last_type = None
    for i, s in enumerate(scenes):
        t = next((x for x in type_colour if x in s), 'urban')
        if last_type and t != last_type:
            ax.axhline(y[i] - 0.5, color='gray', lw=0.8, ls='--', alpha=0.6)
        last_type = t

    for vk in vkeys:
        avg = np.mean([r['variants'][vk]['best_auc'] for r in valid])
        ax.axvline(avg, color=colours[vk], lw=1.2, ls=':',
                   label=f"{vlabels[vk]} avg {avg:.4f}")

    ax.set_yticks(y)
    ax.set_yticklabels([s.replace('abu-', '') for s in scenes], fontsize=10)
    ax.set_xlabel('Best-Window β AUC-ROC', fontsize=11)
    xmin = min(r['variants'][vk]['best_auc'] for r in valid for vk in vkeys) - 0.03
    ax.set_xlim(xmin, 1.005)
    ax.set_title('MAWDBN Best-Window β AUC — FP32 / Quant / HybridQ / HybridRaw',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.25)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'auc_barchart.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Bar chart → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# A vs D FOCUSED PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_roc_a_vs_d(results):
    """
    One ROC plot per scene (A vs D) saved to the scene's own directory.
    Plus one averaged ROC plot saved to RESULTS_DIR.
    """
    valid = [r for r in results if r.get('gt')]
    if not valid:
        print("  No ROC data for A vs D plot."); return

    fpr_grid = np.linspace(0, 1, 500)
    colour_a = '#1f77b4'
    colour_d = '#9467bd'

    tprs_a, tprs_d, aucs_a, aucs_d = [], [], [], []

    for r in valid:
        scene_name = r['scene']
        save_dir   = os.path.join(RESULTS_DIR, scene_name)
        os.makedirs(save_dir, exist_ok=True)

        fpr_a = r['variants']['A_fp32']['best_fpr']
        tpr_a = r['variants']['A_fp32']['best_tpr']
        auc_a = r['variants']['A_fp32']['best_auc']
        wi_a, wo_a = r['variants']['A_fp32']['best_window']

        fpr_d = r['variants']['D_hybridraw']['best_fpr']
        tpr_d = r['variants']['D_hybridraw']['best_tpr']
        auc_d = r['variants']['D_hybridraw']['best_auc']
        wi_d, wo_d = r['variants']['D_hybridraw']['best_window']

        tprs_a.append(np.interp(fpr_grid, fpr_a, tpr_a))
        tprs_d.append(np.interp(fpr_grid, fpr_d, tpr_d))
        aucs_a.append(auc_a)
        aucs_d.append(auc_d)

        # Per-scene plot
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr_a, tpr_a, lw=2, color=colour_a,
                label=f"FP32       AUC={auc_a:.4f}  (w={wi_a},W={wo_a})")
        ax.plot(fpr_d, tpr_d, lw=2, color=colour_d, ls='--',
                label=f"HybridRaw  AUC={auc_d:.4f}  (w={wi_d},W={wo_d})")
        ax.plot([0, 1], [0, 1], 'k:', lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate',  fontsize=11)
        ax.set_title(f"A vs D — {scene_name}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        path = os.path.join(save_dir, f'{scene_name}_a_vs_d_roc.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  A vs D → {path}")

    # Average plot
    avg_a = np.mean(aucs_a)
    avg_d = np.mean(aucs_d)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_grid, np.mean(tprs_a, axis=0), lw=2.5, color=colour_a,
            label=f"FP32       avg AUC={avg_a:.4f}")
    ax.plot(fpr_grid, np.mean(tprs_d, axis=0), lw=2.5, color=colour_d, ls='--',
            label=f"HybridRaw  avg AUC={avg_d:.4f}")
    ax.plot([0, 1], [0, 1], 'k:', lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate',  fontsize=11)
    ax.set_title("A vs D — Dataset Average (13 scenes)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'a_vs_d_average_roc.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  A vs D average → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    for scene in ABU_SCENES:
        mat = os.path.join(ABU_DATASET_DIR, f'{scene}.mat')
        if not os.path.exists(mat):
            print(f"  {scene}: .mat not found — skipping.")
            continue
        try:
            results.append(run_scene(scene, mat))
        except Exception as e:
            import traceback
            print(f"  {scene}: FAILED — {e}")
            traceback.print_exc()

    if results:
        print_summary(results)
        plot_auc_barchart(results)
        plot_collective_roc(results)
        plot_roc_a_vs_d(results)

    print(f"\nDONE — {len(results)} scene(s) processed.")