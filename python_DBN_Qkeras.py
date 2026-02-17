# %% [markdown]
# MAWDBN – Hyperspectral Anomaly Detection on ABU Dataset
# Fixed DSW window: winner=1, wouter=9, pf=0  (globally optimal from grid search)
#
# Runs float32 baseline AND quantised model side-by-side for all 13 ABU scenes.
# Outputs:
#   - Per-scene console summary (one line each)
#   - Final AUC table grouped by scene type
#   - Collective ROC curves (all scenes + dataset average) saved to PNG

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

ABU_DATASET_DIR   = 'ABU_DATASET'
RESULTS_DIR       = 'results'

# Quantisation mode: 'qkeras_qat' | 'ptq'
QUANTIZATION_MODE = 'ptq'
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

# DSW — fixed, no search
DSW_WINNER = 1
DSW_WOUTER = 9
DSW_PF     = 0

THRESHOLD_PERCENTILE = 95
RANDOM_SEED          = 42

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import os, random, warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, losses
from tensorflow.keras.regularizers import l2 as _l2

try:
    import qkeras
    from qkeras import QDense, QActivation
    from qkeras.utils import model_save_quantized_weights
    QKERAS_AVAILABLE = True
except ImportError:
    QKERAS_AVAILABLE = False
    warnings.warn("QKeras not found — 'qkeras_qat' mode will raise an error.")

QUANTIZATION_MODE = QUANTIZATION_MODE.lower()

ABU_SCENES = [
    'abu-airport-1', 'abu-airport-2', 'abu-airport-3', 'abu-airport-4',
    'abu-beach-1',   'abu-beach-2',   'abu-beach-3',   'abu-beach-4',
    'abu-urban-1',   'abu-urban-2',   'abu-urban-3',   'abu-urban-4',
    'abu-urban-5',
]

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"  GPU(s): {[g.name for g in gpus]}")
else:
    print("  No GPU — running on CPU.")
print(f"  TF {tf.__version__}  |  quant={QUANTIZATION_MODE}  |  DSW w={DSW_WINNER},W={DSW_WOUTER},pf={DSW_PF}")


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

def build_autoencoder(n_visible, n_hidden, mode):
    wd  = AE_WEIGHT_DECAY
    inp = keras.Input(shape=(n_visible,), name='spectrum')
    if mode == 'qkeras_qat':
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras required for mode='qkeras_qat'.")
        qs  = f"quantized_bits({QUANT_BITS},{QUANT_INTEGER})"
        qas = f"quantized_sigmoid({QUANT_ACT_BITS},{QUANT_ACT_INT})"
        z   = QDense(n_hidden, kernel_quantizer=qs, bias_quantizer=qs,
                     kernel_regularizer=_l2(wd/2), use_bias=True,
                     name='encoder_dense')(inp)
        z   = QActivation(qas, name='encoder_act')(z)
        out = QDense(n_visible, kernel_quantizer=qs, bias_quantizer=qs,
                     kernel_regularizer=_l2(wd/2), use_bias=True,
                     activation='linear', name='decoder_dense')(z)
    else:
        z   = layers.Dense(n_hidden, activation='sigmoid',
                           kernel_regularizer=_l2(wd/2),
                           name='encoder_dense')(inp)
        out = layers.Dense(n_visible, activation='linear',
                           kernel_regularizer=_l2(wd/2),
                           name='decoder_dense')(z)
    return keras.Model(inputs=inp, outputs=out, name=f'dbn_{mode}')


def build_encoder_only(autoencoder, mode):
    layer = 'encoder_act' if mode == 'qkeras_qat' else 'encoder_dense'
    return keras.Model(inputs=autoencoder.input,
                       outputs=autoencoder.get_layer(layer).output,
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
    print(f"  PTQ → {path}  ({os.path.getsize(path)/1024:.1f} KB)")
    return path


def export_tflite_encoder(keras_ae, rep_data, save_dir, model_name, mode):
    enc = build_encoder_only(keras_ae, mode=mode)
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
    return path


def compute_recon_and_codes_tflite(tflite_ae_path, n_hidden, image, scaler):
    H, W, C   = image.shape
    N         = H * W
    pixels_flat = image.reshape(-1, C)
    pixels_norm, _ = preprocess_pixels(pixels_flat, fit_scaler=False, scaler=scaler)

    enc_path  = tflite_ae_path.replace('_ptq_int8.tflite', '_ptq_int8_encoder.tflite')

    interp_ae  = tf.lite.Interpreter(model_path=tflite_ae_path)
    interp_ae.allocate_tensors()
    inp_det    = interp_ae.get_input_details()[0]
    out_det    = interp_ae.get_output_details()[0]
    inp_scale, inp_zp   = inp_det['quantization']
    out_scale, out_zp   = out_det['quantization']

    interp_enc = tf.lite.Interpreter(model_path=enc_path)
    interp_enc.allocate_tensors()
    enc_inp    = interp_enc.get_input_details()[0]
    enc_out    = interp_enc.get_output_details()[0]
    enc_in_s, enc_in_zp   = enc_inp['quantization']
    enc_out_s, enc_out_zp = enc_out['quantization']

    recons_list, codes_list = [], []
    for start in tqdm(range(0, N, BATCH_SIZE), desc="  TFLite inference", leave=False):
        batch = pixels_norm[start:start + BATCH_SIZE]
        q_ae  = np.clip(np.round(batch / inp_scale) + inp_zp, -128, 127).astype(np.int8)
        q_enc = np.clip(np.round(batch / enc_in_s)  + enc_in_zp, -128, 127).astype(np.int8)

        batch_recons = []
        for px in q_ae:
            interp_ae.set_tensor(inp_det['index'], px[np.newaxis])
            interp_ae.invoke()
            o = interp_ae.get_tensor(out_det['index'])
            batch_recons.append(((o.astype(np.float32) - out_zp) * out_scale)[0])

        batch_codes = []
        for px in q_enc:
            interp_enc.set_tensor(enc_inp['index'], px[np.newaxis])
            interp_enc.invoke()
            o = interp_enc.get_tensor(enc_out['index'])
            batch_codes.append(((o.astype(np.float32) - enc_out_zp) * enc_out_s)[0])

        recons_list.append(np.array(batch_recons, dtype=np.float32))
        codes_list.append(np.array(batch_codes,   dtype=np.float32))

    recons_norm = np.concatenate(recons_list)
    codes_all   = np.concatenate(codes_list)
    recons_orig = scaler.inverse_transform(recons_norm).astype(np.float32)
    rmse_flat   = np.sqrt(np.mean((pixels_flat - recons_orig) ** 2, axis=1))
    return rmse_flat.reshape(H, W), codes_all.reshape(H, W, n_hidden)


# ═══════════════════════════════════════════════════════════════════════════════
# RECONSTRUCTION ERROR & LATENT CODES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_recon_and_codes(model, encoder, image, scaler):
    H, W, C   = image.shape
    pixels    = image.reshape(-1, C)
    norm, _   = preprocess_pixels(pixels, fit_scaler=False, scaler=scaler)
    recons    = model.predict(norm,   batch_size=BATCH_SIZE, verbose=0)
    codes     = encoder.predict(norm, batch_size=BATCH_SIZE, verbose=0)
    orig      = scaler.inverse_transform(recons).astype(np.float32)
    rmse_flat = np.sqrt(np.mean((pixels - orig) ** 2, axis=1))
    return rmse_flat.reshape(H, W), codes.reshape(H, W, encoder.output_shape[-1])


# ═══════════════════════════════════════════════════════════════════════════════
# DSW DETECTOR  (fixed w=1, W=9, pf=0)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_beta_map(R, C_codes):
    """
    MAWDBN β anomaly score via DSW with fixed winner=1, wouter=9, pf=0.

    For every pixel p, neighbours are drawn from the ring between the
    1×1 inner square and the 9×9 outer square (k=80 on interior pixels).

      d_j    = ||c_nj - c_p||_2
      wt_j   = r_p / r_nj          if |r_nj - mu_n| <= sigma_n  (inlier)
             = 0                    otherwise  (pf=0 makes outlier weight zero)
      beta_p = (1/k) * sum(wt_j * d_j)
    """
    winner, wouter = DSW_WINNER, DSW_WOUTER
    half_in  = winner  // 2   # = 0  (inner square is just the centre pixel)
    half_out = wouter  // 2   # = 4

    H, W  = R.shape
    beta  = np.zeros((H, W), dtype=np.float32)

    for j in tqdm(range(H), desc="  DSW", leave=False):
        r0 = max(j - half_out, 0)
        r1 = min(j + half_out + 1, H)
        for i in range(W):
            c0 = max(i - half_out, 0)
            c1 = min(i + half_out + 1, W)

            win_R = R[r0:r1, c0:c1]
            win_C = C_codes[r0:r1, c0:c1]
            cj, ci = j - r0, i - c0

            win_h, win_w = win_R.shape
            ri = np.arange(win_h)[:, None]
            ci_ = np.arange(win_w)[None, :]
            # Ring: outside inner square AND not the centre pixel itself.
            # With winner=1 the inner square is just the centre pixel,
            # so ring = everything except centre.
            ring = ~(
                (np.abs(ri - cj) <= half_in) &
                (np.abs(ci_ - ci) <= half_in)
            ) | ((ri != cj) | (ci_ != ci))
            # Simpler: ring = not centre (since half_in=0)
            ring = (ri != cj) | (ci_ != ci)

            rn = win_R[ring]
            if rn.size == 0:
                continue

            mu_n  = rn.mean()
            sig_n = rn.std()
            inlier = np.abs(rn - mu_n) <= sig_n  # pf=0 → outliers contribute 0

            r_p   = R[j, i]
            wt    = np.where(inlier, r_p / (rn + 1e-12), 0.0)

            c_p   = C_codes[j, i]
            c_rng = win_C[ring]
            d     = np.sqrt(np.sum((c_rng - c_p) ** 2, axis=1))

            beta[j, i] = float(np.sum(wt * d)) / rn.size

    return beta


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_auc_roc(score_map, gt_mask):
    y_true = gt_mask.reshape(-1).astype(int)
    y_score = score_map.reshape(-1)
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc, fpr, tpr


def threshold_map(score_map, percentile=THRESHOLD_PERCENTILE):
    thresh = np.percentile(score_map, percentile)
    return (score_map > thresh).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# PER-SCENE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_scene(scene_name, mat_path):
    set_seed(RANDOM_SEED)
    save_dir = os.path.join(RESULTS_DIR, 'mawdbn_fixed', scene_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n── {scene_name.upper()} {'─'*50}")
    image, gt_mask = load_abu_mat(mat_path)
    H, W, C = image.shape

    n_train  = min(N_SAMPLES_TRAIN, H * W) if N_SAMPLES_TRAIN else H * W
    rng      = np.random.default_rng(RANDOM_SEED)
    idx      = rng.choice(H * W, size=n_train, replace=False)
    train_px = image.reshape(-1, C)[idx]
    norm_px, scaler = preprocess_pixels(train_px, fit_scaler=True)
    print(f"  {H}×{W}×{C}  |  {n_train:,} px")

    # RBM (shared init)
    rbm = GaussianRBM_NP(n_visible=C, n_hidden=N_HIDDEN, k=RBM_K)
    rbm_losses = rbm.train(norm_px, n_epochs=RBM_EPOCHS, lr=RBM_LEARNING_RATE,
                           momentum=RBM_MOMENTUM, weight_decay=RBM_WEIGHT_DECAY,
                           batch_size=BATCH_SIZE)

    # Float32 model
    ae_none = build_autoencoder(C, N_HIDDEN, mode='none')
    initialize_from_rbm(ae_none, rbm)
    train_autoencoder(ae_none, norm_px, save_dir, f'mawdbn_none_{scene_name}')
    enc_none = build_encoder_only(ae_none, mode='none')
    rmse_none, codes_none = compute_recon_and_codes(ae_none, enc_none, image, scaler)

    # Quantised model
    set_seed(RANDOM_SEED)
    ae_quant = build_autoencoder(C, N_HIDDEN, mode=QUANTIZATION_MODE)
    initialize_from_rbm(ae_quant, rbm)
    train_autoencoder(ae_quant, norm_px, save_dir, f'mawdbn_{QUANTIZATION_MODE}_{scene_name}')

    if QUANTIZATION_MODE == 'ptq':
        tflite_path = export_tflite_ptq(ae_quant, norm_px[:200], save_dir,
                                        f'mawdbn_{QUANTIZATION_MODE}_{scene_name}')
        export_tflite_encoder(ae_quant, norm_px[:200], save_dir,
                              f'mawdbn_{QUANTIZATION_MODE}_{scene_name}',
                              mode=QUANTIZATION_MODE)
        rmse_quant, codes_quant = compute_recon_and_codes_tflite(
            tflite_path, N_HIDDEN, image, scaler)
    else:
        enc_quant = build_encoder_only(ae_quant, mode=QUANTIZATION_MODE)
        rmse_quant, codes_quant = compute_recon_and_codes(
            ae_quant, enc_quant, image, scaler)

    # DSW β maps
    beta_none  = compute_beta_map(rmse_none,  codes_none)
    beta_quant = compute_beta_map(rmse_quant, codes_quant)

    # AUC
    result = {'scene': scene_name}
    if gt_mask is not None:
        auc_rn, fpr_rn, tpr_rn = compute_auc_roc(rmse_none,  gt_mask)
        auc_bn, fpr_bn, tpr_bn = compute_auc_roc(beta_none,  gt_mask)
        auc_rq, fpr_rq, tpr_rq = compute_auc_roc(rmse_quant, gt_mask)
        auc_bq, fpr_bq, tpr_bq = compute_auc_roc(beta_quant, gt_mask)

        print(f"  RMSE  f32={auc_rn:.4f}  qt={auc_rq:.4f}  Δ={auc_rq-auc_rn:+.4f}"
              f"  |  β  f32={auc_bn:.4f}  qt={auc_bq:.4f}  Δ={auc_bq-auc_bn:+.4f}")

        result.update({
            'auc_rmse_none':  auc_rn, 'fpr_rmse_none':  fpr_rn, 'tpr_rmse_none':  tpr_rn,
            'auc_rmse_quant': auc_rq, 'fpr_rmse_quant': fpr_rq, 'tpr_rmse_quant': tpr_rq,
            'auc_beta_none':  auc_bn, 'fpr_beta_none':  fpr_bn, 'tpr_beta_none':  tpr_bn,
            'auc_beta_quant': auc_bq, 'fpr_beta_quant': fpr_bq, 'tpr_beta_quant': tpr_bq,
            # Score maps kept for the illustrated comparison figure
            'image': image, 'gt_mask': gt_mask,
            'rmse_none': rmse_none, 'beta_none': beta_none,
            'rmse_quant': rmse_quant, 'beta_quant': beta_quant,
        })

        plot_scene_roc(result,
                       save_path=os.path.join(save_dir, f'{scene_name}_roc.png'))
        plot_scene_comparison(result,
                              save_path=os.path.join(save_dir, f'{scene_name}_comparison.png'))
    else:
        print("  No ground truth — AUC skipped.")

    return result



# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def _false_colour(image):
    H, W, C = image.shape
    rb = min(60, C-1); gb = min(40, C-1); bb = min(20, C-1)
    rgb = np.stack([image[:,:,rb], image[:,:,gb], image[:,:,bb]], axis=-1)
    return np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8), 0, 1)


def _overlay(rgb, mask, gt_mask):
    out = rgb.copy()
    out[mask == 1] = [1, 0, 0]
    if gt_mask is not None:
        out[(gt_mask == 1) & (mask == 0)] = [0, 0.4, 1.0]
    return out


def plot_scene_roc(r, save_path):
    """
    Per-scene ROC: four curves on one axes.
      RMSE float32 / RMSE quant / β float32 / β quant
    """
    qm  = QUANTIZATION_MODE
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, fpr_k, tpr_k, auc_k, col, ls in [
        ('RMSE  float32', 'fpr_rmse_none',  'tpr_rmse_none',  'auc_rmse_none',  '#1f77b4', '-'),
        (f'RMSE  {qm}',   'fpr_rmse_quant', 'tpr_rmse_quant', 'auc_rmse_quant', '#aec7e8', '--'),
        ('β     float32', 'fpr_beta_none',  'tpr_beta_none',  'auc_beta_none',  '#d62728', '-'),
        (f'β     {qm}',   'fpr_beta_quant', 'tpr_beta_quant', 'auc_beta_quant', '#f4a460', '--'),
    ]:
        ax.plot(r[fpr_k], r[tpr_k], lw=2, color=col, ls=ls,
                label=f"{label}  AUC={r[auc_k]:.4f}")
    ax.plot([0,1],[0,1],'k:',lw=0.8)
    ax.set_xlim(0,1); ax.set_ylim(0,1.01)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate',  fontsize=11)
    ax.set_title(f"ROC — {r['scene']}  |  DSW w={DSW_WINNER}, W={DSW_WOUTER}",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_scene_comparison(r, save_path):
    """
    Illustrated 3×4 comparison figure per scene:
      Col 0 — float32 score map
      Col 1 — quantised score map
      Col 2 — float32 detection overlay  (red=detected, blue=missed)
      Col 3 — quantised detection overlay

      Row 0 — false colour + ground truth
      Row 1 — RMSE maps & overlays
      Row 2 — β maps & overlays
    """
    qm      = QUANTIZATION_MODE
    image   = r['image']
    gt      = r['gt_mask']
    rgb     = _false_colour(image)

    thresh_rn  = threshold_map(r['rmse_none'])
    thresh_rq  = threshold_map(r['rmse_quant'])
    thresh_bn  = threshold_map(r['beta_none'])
    thresh_bq  = threshold_map(r['beta_quant'])

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle(
        f"float32 vs {qm.upper()}  |  {r['scene']}  |  DSW w={DSW_WINNER}, W={DSW_WOUTER}\n"
        "red = detected anomaly   |   blue = missed ground-truth",
        fontsize=13, fontweight='bold')

    def show(ax, img, title, cmap='viridis', vmin=None, vmax=None):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10); ax.axis('off')

    # Row 0: scene overview
    show(axes[0,0], rgb, 'False colour')
    if gt is not None:
        show(axes[0,1], gt, f'Ground truth  ({int(gt.sum()):,} px)', cmap='gray', vmin=0, vmax=1)
    axes[0,2].axis('off'); axes[0,3].axis('off')

    # Row 1: RMSE
    show(axes[1,0], r['rmse_none'],  'RMSE — float32',  cmap='hot')
    show(axes[1,1], r['rmse_quant'], f'RMSE — {qm}',    cmap='hot')
    show(axes[1,2], _overlay(rgb, thresh_rn, gt), 'RMSE overlay — float32')
    show(axes[1,3], _overlay(rgb, thresh_rq, gt), f'RMSE overlay — {qm}')

    # Row 2: β
    show(axes[2,0], r['beta_none'],  'β — float32',  cmap='hot')
    show(axes[2,1], r['beta_quant'], f'β — {qm}',    cmap='hot')
    show(axes[2,2], _overlay(rgb, thresh_bn, gt), 'β overlay — float32')
    show(axes[2,3], _overlay(rgb, thresh_bq, gt), f'β overlay — {qm}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_auc_barchart(results, save_path):
    """
    Horizontal grouped bar chart: β AUC per scene, float32 vs quantised.
    Scenes are sorted by type (airport / beach / urban) with dividers.
    Bars are annotated with their AUC value.
    """
    valid = [r for r in results if 'auc_beta_none' in r]
    if not valid: return

    qm     = QUANTIZATION_MODE
    scenes = [r['scene'] for r in valid]
    bn     = [r['auc_beta_none']  for r in valid]
    bq     = [r['auc_beta_quant'] for r in valid]

    n   = len(scenes)
    y   = np.arange(n)
    h   = 0.35

    type_colour = {'airport': '#4878cf', 'beach': '#f58231', 'urban': '#3cb44b'}
    bar_colours = [type_colour.get(next((t for t in type_colour if t in s), 'urban'), '#888')
                   for s in scenes]

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.55)))

    bars_n = ax.barh(y + h/2, bn, h, color=bar_colours, alpha=0.85, label='float32')
    bars_q = ax.barh(y - h/2, bq, h, color=bar_colours, alpha=0.45,
                     hatch='///', label=qm, edgecolor='white', linewidth=0.4)

    # Value annotations
    for bar, v in zip(bars_n, bn):
        ax.text(max(v - 0.001, 0.84), bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', ha='right', fontsize=7.5, color='white', fontweight='bold')
    for bar, v in zip(bars_q, bq):
        ax.text(max(v - 0.001, 0.84), bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', ha='right', fontsize=7.5, color='white', fontweight='bold')

    # Type dividers
    last_type = None
    for i, s in enumerate(scenes):
        t = next((x for x in type_colour if x in s), 'urban')
        if last_type and t != last_type:
            ax.axhline(y[i] - 0.5, color='gray', lw=0.8, ls='--', alpha=0.6)
        last_type = t

    # Overall average lines
    ax.axvline(np.mean(bn), color='black',  lw=1.2, ls=':', label=f'f32 avg {np.mean(bn):.4f}')
    ax.axvline(np.mean(bq), color='dimgray', lw=1.2, ls=':', label=f'{qm} avg {np.mean(bq):.4f}')

    # Type legend patches
    import matplotlib.patches as mpatches
    handles, labels = ax.get_legend_handles_labels()
    type_patches = [mpatches.Patch(color=c, label=t.capitalize())
                    for t, c in type_colour.items()]
    ax.legend(handles=handles + type_patches, fontsize=9, loc='lower right')

    ax.set_yticks(y)
    ax.set_yticklabels([s.replace('abu-','') for s in scenes], fontsize=10)
    ax.set_xlabel('AUC-ROC (β score)', fontsize=11)
    ax.set_xlim(min(min(bn), min(bq)) - 0.02, 1.005)
    ax.set_title(
        f'MAWDBN β AUC — float32 vs {qm.upper()}  |  DSW w={DSW_WINNER}, W={DSW_WOUTER}',
        fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results):
    valid = [r for r in results if 'auc_rmse_none' in r]
    if not valid:
        print("No valid results."); return

    qm   = QUANTIZATION_MODE
    W    = 22
    div  = "  " + "─" * 88
    bdr  = "═" * 92

    def fmt(v):  return f"{v:.4f}" if v is not None else "  N/A"
    def dlt(a, b): return f"{b-a:+.4f}" if None not in (a, b) else "  N/A"
    def row(label, rn, rq, bn, bq, indent="  "):
        return (f"{indent}{label:<{W}}  "
                f"{fmt(rn):>8}  {fmt(rq):>8}  {dlt(rn,rq):>8}  "
                f"{fmt(bn):>8}  {fmt(bq):>8}  {dlt(bn,bq):>8}")
    def gavg(grp, key):
        vs = [r[key] for r in grp if r.get(key) is not None]
        return sum(vs)/len(vs) if vs else None

    types = {
        'AIRPORT': [r for r in valid if 'airport' in r['scene']],
        'BEACH':   [r for r in valid if 'beach'   in r['scene']],
        'URBAN':   [r for r in valid if 'urban'   in r['scene']],
    }

    lines = []
    def pr(s=""): print(s); lines.append(s)

    pr(); pr(bdr)
    pr(f"  MAWDBN AUC SUMMARY  |  float32 vs {qm.upper()}  |  DSW w={DSW_WINNER}, W={DSW_WOUTER}")
    pr(bdr)
    pr(f"  {'Scene':<{W}}  {'RMSE-f32':>8}  {'RMSE-qt':>8}  {'ΔRMSE':>8}  "
       f"{'β-f32':>8}  {'β-qt':>8}  {'Δβ':>8}")
    pr(div)

    for tname, grp in types.items():
        if not grp: continue
        for r in grp:
            pr(row(r['scene'],
                   r['auc_rmse_none'], r['auc_rmse_quant'],
                   r['auc_beta_none'], r['auc_beta_quant']))
        pr(div)
        avg_rn = gavg(grp, 'auc_rmse_none');  avg_rq = gavg(grp, 'auc_rmse_quant')
        avg_bn = gavg(grp, 'auc_beta_none');  avg_bq = gavg(grp, 'auc_beta_quant')
        pr(row(f"  AVG {tname} ({len(grp)} scenes)", avg_rn, avg_rq, avg_bn, avg_bq, indent=""))
        pr(div)

    all_rn = gavg(valid, 'auc_rmse_none');  all_rq = gavg(valid, 'auc_rmse_quant')
    all_bn = gavg(valid, 'auc_beta_none');  all_bq = gavg(valid, 'auc_beta_quant')
    pr(row(f"  OVERALL ({len(valid)} scenes)", all_rn, all_rq, all_bn, all_bq, indent=""))
    pr(bdr)

    pr()
    pr(f"  QUANTISATION IMPACT  ({qm} vs float32):")
    if all_rn and all_rq:
        pr(f"    RMSE Δ : {all_rq-all_rn:+.4f}  ({'improved' if all_rq>all_rn else 'degraded'})")
    if all_bn and all_bq:
        pr(f"    β    Δ : {all_bq-all_bn:+.4f}  ({'improved' if all_bq>all_bn else 'degraded'})")
    pr()

    best = max(valid, key=lambda r: r.get('auc_beta_none', 0))
    worst = min(valid, key=lambda r: r.get('auc_beta_none', 1))
    pr(f"  β-f32 best  : {best['scene']}  {best['auc_beta_none']:.4f}")
    pr(f"  β-f32 worst : {worst['scene']}  {worst['auc_beta_none']:.4f}")

    # Save
    out_dir = os.path.join(RESULTS_DIR, 'mawdbn_fixed')
    os.makedirs(out_dir, exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'summary_{qm}_{ts}.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Summary saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# COLLECTIVE ROC PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_collective_roc(results):
    """
    One figure with two panels:
      Left  — β (MAWDBN) ROC curves for every scene, colour-coded by type,
               plus a bold dataset-average curve for float32 and quantised.
      Right — same for RMSE baseline.

    Each scene is a thin line; averages are computed by interpolating all
    scene FPR arrays onto a common grid then averaging TPR.
    """
    valid = [r for r in results if 'fpr_beta_none' in r]
    if not valid:
        print("  No ROC data to plot."); return

    qm = QUANTIZATION_MODE
    fpr_grid = np.linspace(0, 1, 500)

    def interp_tpr(fpr, tpr):
        return np.interp(fpr_grid, fpr, tpr)

    def avg_curve(fprs, tprs):
        """Mean TPR over a common FPR grid."""
        tprs_i = [interp_tpr(f, t) for f, t in zip(fprs, tprs)]
        return fpr_grid, np.mean(tprs_i, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"Collective ROC — ABU Dataset  |  float32 vs {qm.upper()}"
        f"  |  DSW w={DSW_WINNER}, W={DSW_WOUTER}",
        fontsize=13, fontweight='bold')

    panels = [
        ('β (MAWDBN)', 'fpr_beta_none', 'tpr_beta_none',
                       'fpr_beta_quant', 'tpr_beta_quant',
                       'auc_beta_none', 'auc_beta_quant'),
        ('RMSE',       'fpr_rmse_none', 'tpr_rmse_none',
                       'fpr_rmse_quant', 'tpr_rmse_quant',
                       'auc_rmse_none', 'auc_rmse_quant'),
    ]

    for ax, (metric, fkn, tkn, fkq, tkq, akn, akq) in zip(axes, panels):

        avg_auc_none  = np.mean([r[akn] for r in valid])
        avg_auc_quant = np.mean([r[akq] for r in valid])

        fpr_avg_n, tpr_avg_n = avg_curve(
            [r[fkn] for r in valid], [r[tkn] for r in valid])
        fpr_avg_q, tpr_avg_q = avg_curve(
            [r[fkq] for r in valid], [r[tkq] for r in valid])

        ax.plot(fpr_avg_n, tpr_avg_n, color='#1f77b4', lw=2.5,
                label=f"float32  AUC={avg_auc_none:.4f}")
        ax.plot(fpr_avg_q, tpr_avg_q, color='#d62728', lw=2.5, ls='--',
                label=f"{qm}     AUC={avg_auc_quant:.4f}")

        ax.plot([0, 1], [0, 1], 'k:', lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate',  fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_dir = os.path.join(RESULTS_DIR, 'mawdbn_fixed')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'roc_collective_{qm}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ROC chart → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
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
        out_dir = os.path.join(RESULTS_DIR, 'mawdbn_fixed')
        os.makedirs(out_dir, exist_ok=True)

        print_summary(results)
        plot_auc_barchart(results,
                          save_path=os.path.join(out_dir, f'auc_barchart_{QUANTIZATION_MODE}.png'))
        plot_collective_roc(results)

    print(f"\nDONE — {len(results)} scene(s) processed.")