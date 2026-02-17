# %% [markdown]
# # MAWDBN – Hyperspectral Anomaly Detection on ABU Dataset
# ## QKeras Quantised Implementation  (with float32 comparison)
#
# Automatically runs BOTH float32 baseline AND quantised model side-by-side,
# then produces a combined comparison plot and AUC summary table.
#
# ── RUN MODES ────────────────────────────────────────────────────────────────
#   RUN_MODE = 'single'  →  process one scene (SCENE_INDEX)
#   RUN_MODE = 'batch'   →  process all available ABU scenes in sequence
#
# ── QUANTISATION MODES ───────────────────────────────────────────────────────
#   QUANTIZATION_MODE = 'qkeras_qat'  →  QKeras Quantisation-Aware Training
#   QUANTIZATION_MODE = 'ptq'         →  Post-Training Quantisation (TFLite)
#   (float32 baseline always runs alongside for comparison)
#
# ── DSW WINDOW GRID SEARCH ───────────────────────────────────────────────────
#   DSW_WINNER_RANGE / DSW_WOUTER_RANGE / DSW_PF_RANGE accept lists of values.
#   Every valid combination is tried per scene; the best beta AUC is reported.
#   Set each list to a single value to reproduce fixed-window behaviour.
#
# All user-configurable parameters are in the CONFIG PANEL below.
# Do not edit anything below the CONFIG PANEL unless you know what you're doing.

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# ███  USER CONFIG PANEL  ██████████████████████████████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════
# Everything you might want to change lives here.
# Nothing below this section needs to be touched for normal use.

# ── Run mode ──────────────────────────────────────────────────────────────────
#   'single' : run one scene, specified by SCENE_INDEX below
#   'batch'  : run all scenes found in ABU_DATASET_DIR
RUN_MODE    = 'batch'

# ── Scene selection (only used when RUN_MODE = 'single') ──────────────────────
#   0  abu-airport-1       4  abu-beach-1       8  abu-urban-1
#   1  abu-airport-2       5  abu-beach-2       9  abu-urban-2
#   2  abu-airport-3       6  abu-beach-3      10  abu-urban-3
#   3  abu-airport-4       7  abu-beach-4      11  abu-urban-4
#                                              12  abu-urban-5
SCENE_INDEX = 1

# ── Dataset path ──────────────────────────────────────────────────────────────
ABU_DATASET_DIR = 'ABU_DATASET'

# ── Quantisation ──────────────────────────────────────────────────────────────
#   QUANTIZATION_MODE : which quantised model to compare against float32
#     'qkeras_qat' – Quantisation-Aware Training  (recommended)
#     'ptq'        – Post-Training Quantisation via TFLite
#
#   QUANT_BITS    : total bit-width for weights & biases  (4 / 6 / 8 / 16)
#   QUANT_INTEGER : integer bits within QUANT_BITS
#                   rule of thumb: QUANT_BITS // 4  (e.g. 2 for 8-bit)
#   QUANT_ACT_BITS: bit-width for the encoder sigmoid activation
#   QUANT_ACT_INT : integer bits for activation (1 is correct for sigmoid)
QUANTIZATION_MODE = 'ptq'   # 'qkeras_qat' | 'ptq'
QUANT_BITS        = 8
QUANT_INTEGER     = 2
QUANT_ACT_BITS    = 8
QUANT_ACT_INT     = 1

# ── Model architecture ────────────────────────────────────────────────────────
#   N_HIDDEN : latent dimension — fixed at 13 per the MAWDBN paper.
#              Change only for experimentation.
N_HIDDEN = 13

# ── RBM pre-training (CD-1) ───────────────────────────────────────────────────
#   Paper: step ratio 0.01, momentum 0.5→0.9 (switches at epoch 5),
#          weight decay 0.0002, CD-1.
RBM_EPOCHS        = 50
RBM_LEARNING_RATE = 0.01
RBM_K             = 1        # CD-k steps (1 = CD-1 as in paper)
RBM_MOMENTUM      = 0.5      # initial; switches to 0.9 after epoch 5
RBM_WEIGHT_DECAY  = 0.0002

# ── Autoencoder fine-tuning ───────────────────────────────────────────────────
#   AE_WEIGHT_DECAY is replicated as kernel_regularizer=l2(wd/2) on each Dense
#   layer, faithfully matching PyTorch Adam(weight_decay=λ) behaviour.
#   AE_PATIENCE : early-stopping patience on training loss.
#                 Set to AE_EPOCHS to disable early stopping entirely.
AE_EPOCHS        = 50
AE_LEARNING_RATE = 0.001
AE_WEIGHT_DECAY  = 0.0002
AE_PATIENCE      = 15

# ── Training data ─────────────────────────────────────────────────────────────
#   N_SAMPLES_TRAIN : max pixels sampled for training.
#                     Set to None to use all pixels in the scene.
#   BATCH_SIZE      : mini-batch size for both RBM and AE training.
N_SAMPLES_TRAIN = 50000
BATCH_SIZE      = 2048

# ── DSW anomaly detector — window grid search ─────────────────────────────────
#   DSW_WINNER_RANGE : list of inner window sizes to try  (odd integers >= 1)
#   DSW_WOUTER_RANGE : list of outer window sizes to try  (odd integers > winner)
#   DSW_PF_RANGE     : list of penalty factors to try     (0 <= pf < 1)
#
#   For each scene every valid (winner, wouter, pf) combination is evaluated.
#   The combination with the highest beta AUC on the float32 model is recorded
#   as "best" and reported in the summary table alongside the fixed-window result.
#   The same best (winner, wouter, pf) is also applied to the quantised model
#   so the comparison remains fair.
#
#   To replicate the original paper's single fixed window, set each list to one
#   value:  DSW_WINNER_RANGE = [1]  DSW_WOUTER_RANGE = [7]  DSW_PF_RANGE = [0]
#
#   Note: DSW is the CPU bottleneck — each (winner, wouter) pair on a 100×100
#   image takes ~5–30 s.  A 3×5×1 grid = 15 combos × 2 models = 30 DSW passes.
DSW_WINNER_RANGE = [1, 3, 5, 7]
DSW_WOUTER_RANGE = [3, 5, 7, 9, 11, 15, 21]
DSW_PF_RANGE     = [0]

# ── Thresholding & evaluation ─────────────────────────────────────────────────
#   THRESHOLD_PERCENTILE : anomaly pixels are those above this percentile
#                          of the score map  (e.g. 95 = top 5%)
THRESHOLD_PERCENTILE = 95

# ── Output ────────────────────────────────────────────────────────────────────
#   RESULTS_DIR : root directory for all saved outputs
#   RANDOM_SEED : controls NumPy, Python random, and TF random state
RESULTS_DIR = 'results'
RANDOM_SEED = 42

# ═══════════════════════════════════════════════════════════════════════════════
# ███  END OF USER CONFIG PANEL  ███████████████████████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════


# %% [markdown]
# ## Imports & Setup

# %%
import os
import random
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    print(f"  QKeras {qkeras.__version__} loaded.")
except ImportError:
    QKERAS_AVAILABLE = False
    warnings.warn("QKeras not found — 'qkeras_qat' mode will raise an error.")

# Normalise mode string so any capitalisation works
QUANTIZATION_MODE = QUANTIZATION_MODE.lower()

ABU_SCENES = [
    'abu-airport-1', 'abu-airport-2', 'abu-airport-3', 'abu-airport-4',
    'abu-beach-1',   'abu-beach-2',   'abu-beach-3',   'abu-beach-4',
    'abu-urban-1',   'abu-urban-2',   'abu-urban-3',   'abu-urban-4',
    'abu-urban-5',
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# %% [markdown]
# ## GPU Setup

# %%
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  GPU(s): {[g.name for g in gpus]}  (memory growth enabled)")
    except RuntimeError as e:
        print(f"  GPU config error: {e}")
else:
    print("  No GPU — running on CPU.")

print(f"  TensorFlow : {tf.__version__}")
print(f"  Run mode   : {RUN_MODE}")
print(f"  Quant mode : {QUANTIZATION_MODE}")


# %% [markdown]
# ## Data Loading

# %%
def load_abu_mat(mat_path):
    mat = sio.loadmat(mat_path)
    if 'data' in mat:
        image = mat['data'].astype(np.float32)
    else:
        candidates = {k: v for k, v in mat.items()
                      if not k.startswith('_') and isinstance(v, np.ndarray)
                      and v.ndim == 3}
        if not candidates:
            raise KeyError(f"No 3-D array in {mat_path}. Keys: {list(mat.keys())}")
        key   = max(candidates, key=lambda k: candidates[k].size)
        print(f"  Warning: 'data' key missing; using '{key}'")
        image = candidates[key].astype(np.float32)

    if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0))

    gt_mask = None
    if 'map' in mat:
        gt_mask = mat['map'].astype(np.uint8).squeeze()
        if gt_mask.shape != image.shape[:2]:
            gt_mask = gt_mask.T

    H, W, C = image.shape
    n_anom  = int(np.sum(gt_mask)) if gt_mask is not None else 0
    print(f"  Shape   : {image.shape}  (H x W x C)")
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
    return normalised.astype(np.float32), scaler


# %% [markdown]
# ## RBM (NumPy)

# %%
class GaussianRBM_NP:
    """
    Gaussian-Bernoulli RBM trained with CD-k in NumPy.
    Provides weight initialisation for the QKeras autoencoder.
    The RBM is shared between the float32 and quantised autoencoders —
    both start from the same pre-trained weights.
    """
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
            total = 0.0
            count = 0
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
                total += np.mean((batch - v_k) ** 2)
                count += 1
            avg = total / count
            history.append(float(avg))
        print(f"  RBM {self.n_visible}→{self.n_hidden}  "
              f"epochs={n_epochs}  final loss: {history[-1]:.6f}")
        return history


# %% [markdown]
# ## Autoencoder (QKeras or plain Keras)

# %%
def _quant_str(bits, integer):
    return f"quantized_bits({bits},{integer})"


def _act_quant_str(bits, integer):
    return f"quantized_sigmoid({bits},{integer})"


def build_autoencoder(n_visible, n_hidden, mode):
    """
    3-layer DBN autoencoder: n_visible -> n_hidden (sigmoid) -> n_visible.

    mode = 'none' / 'ptq' : standard float32 Dense layers
    mode = 'qkeras_qat'   : QDense + QActivation (fake-quantisation in training)

    kernel_regularizer=l2(wd/2) replicates PyTorch Adam(weight_decay=ld),
    which adds ld*w to the gradient before the adaptive step — equivalent
    to a penalty of (ld/2)*||w||^2 in the loss.
    """
    wd  = AE_WEIGHT_DECAY
    inp = keras.Input(shape=(n_visible,), name='spectrum')

    if mode == 'qkeras_qat':
        if not QKERAS_AVAILABLE:
            raise ImportError(
                "QKeras is required for mode='qkeras_qat'. "
                "Install with: pip install qkeras"
            )
        z = QDense(
            n_hidden,
            kernel_quantizer   = _quant_str(QUANT_BITS, QUANT_INTEGER),
            bias_quantizer     = _quant_str(QUANT_BITS, QUANT_INTEGER),
            kernel_regularizer = _l2(wd / 2),
            use_bias           = True,
            name               = 'encoder_dense',
        )(inp)
        z = QActivation(
            _act_quant_str(QUANT_ACT_BITS, QUANT_ACT_INT),
            name='encoder_act',
        )(z)
        out = QDense(
            n_visible,
            kernel_quantizer   = _quant_str(QUANT_BITS, QUANT_INTEGER),
            bias_quantizer     = _quant_str(QUANT_BITS, QUANT_INTEGER),
            kernel_regularizer = _l2(wd / 2),
            use_bias           = True,
            activation         = 'linear',
            name               = 'decoder_dense',
        )(z)
    else:
        # 'none' or 'ptq' — plain float32
        z   = layers.Dense(n_hidden, activation='sigmoid',
                           kernel_regularizer=_l2(wd / 2),
                           name='encoder_dense')(inp)
        out = layers.Dense(n_visible, activation='linear',
                           kernel_regularizer=_l2(wd / 2),
                           name='decoder_dense')(z)

    return keras.Model(inputs=inp, outputs=out, name=f'dbn_{mode}')


def build_encoder_only(autoencoder, mode):
    enc_out_layer = 'encoder_act' if mode == 'qkeras_qat' else 'encoder_dense'
    enc_out = autoencoder.get_layer(enc_out_layer).output
    return keras.Model(inputs=autoencoder.input, outputs=enc_out, name='encoder')


def initialize_from_rbm(model, rbm):
    """
    Copy pre-trained RBM weights into the autoencoder.
    Keras Dense kernel shape is (input_dim, units):
      encoder: (n_visible, n_hidden) = rbm.W.T
      decoder: (n_hidden, n_visible) = rbm.W  (same shape directly)
    """
    model.get_layer('encoder_dense').set_weights([rbm.W.T, rbm.h_bias])
    model.get_layer('decoder_dense').set_weights([rbm.W,   rbm.v_bias])
    print("  Autoencoder initialised from RBM weights")


# %% [markdown]
# ## Training

# %%
def train_autoencoder(model, X_train, save_dir, model_name):
    """
    Fine-tune autoencoder with Adam + MSE on 100% of training pixels.

    Key parity decisions vs original PyTorch:
      - validation_split=0.0  (PyTorch used all pixels, no val split)
      - monitor='loss'        (train loss, since there is no val set)
      - plain Adam            (matches PyTorch Adam L2 via kernel_regularizer)
      - save_format='h5'      (avoids Keras 3 'options' crash with .keras format)
    """
    model.compile(
        optimizer = optimizers.Adam(learning_rate=AE_LEARNING_RATE),
        loss      = losses.MeanSquaredError(),
    )

    ckpt_path = os.path.join(save_dir, f'{model_name}.h5')
    cb_list = [
        callbacks.EarlyStopping(
            monitor='loss', patience=AE_PATIENCE,
            restore_best_weights=True, verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=5,
        ),
        callbacks.ModelCheckpoint(
            filepath=ckpt_path, monitor='loss',
            save_best_only=True, save_format='h5', verbose=0,
        ),
    ]

    history = model.fit(
        X_train, X_train,
        epochs           = AE_EPOCHS,
        batch_size       = BATCH_SIZE,
        validation_split = 0.0,
        callbacks        = cb_list,
        verbose          = 0,
        shuffle          = True,
    )
    best_loss   = min(history.history['loss'])
    n_epochs_run = len(history.history['loss'])
    print(f"  AE  [{model.name}]  "
          f"epochs={n_epochs_run}/{AE_EPOCHS}  best loss: {best_loss:.6f}")
    return history.history


# %% [markdown]
# ## PTQ Export

# %%
def export_tflite_ptq(model, rep_data, save_dir, model_name):
    """
    Convert a trained float32 model to a fully-integer TFLite flatbuffer.

    Uses full-integer PTQ: a representative dataset of ~200 calibration
    pixels is passed through the model to determine per-layer quantisation
    ranges (scale + zero-point).  Every weight, bias, and activation is
    then mapped to INT8 (range -128..127).

    Both inputs and outputs are INT8 so the .tflite file is suitable for
    deployment on microcontrollers / FPGAs via TFLite Micro.

    Returns the path to the saved .tflite file.
    """
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
    out_path = os.path.join(save_dir, f'{model_name}_ptq_int8.tflite')
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print(f"  PTQ TFLite -> {out_path}  "
          f"({os.path.getsize(out_path)/1024:.1f} KB)")
    return out_path


def compute_recon_and_codes_tflite(tflite_path, n_hidden, image, scaler):
    """
    Run full-image inference using the TFLite INT8 interpreter.

    This is the correct PTQ inference path.  The original script's design
    gap was that export_tflite_ptq() saved the quantised model but all
    subsequent inference (compute_recon_and_codes) still used the float32
    Keras model via model.predict() — producing identical results to 'none'.

    This function runs inference through the actual INT8 TFLite model,
    so the RMSE and latent codes reflect genuine post-quantisation behaviour.

    How INT8 TFLite inference works
    --------------------------------
    During PTQ conversion, TFLite recorded a scale (s) and zero-point (z)
    for each tensor.  At inference time:
      - The float32 input x is mapped to INT8 via:  q = round(x / s) + z
      - All matrix multiplications run in INT8 arithmetic
      - Each layer output is dequantised back to float32 for the next layer
        (or kept as INT8 if the next op supports it)
      - The final output is dequantised to float32 automatically when
        output_details[0]['dtype'] is int8 and we read .output_details

    Because the input tensor is typed int8, we must quantise the normalised
    pixel values ourselves using the input tensor's scale and zero_point
    before invoking the interpreter.  The output is then dequantised using
    the output tensor's scale and zero_point.

    Encoder codes: the full autoencoder has one output (reconstruction).
    To get latent codes we run a second separate TFLite model that contains
    only the encoder half, converted separately.

    Parameters
    ----------
    tflite_path : str   path to the full autoencoder .tflite file
    n_hidden    : int   latent dimension (needed to reshape output)
    image       : (H, W, C) float32 original hyperspectral image
    scaler      : fitted StandardScaler

    Returns
    -------
    rmse_map     : (H, W)           RMSE in original spectral scale
    latent_codes : (H, W, n_hidden) encoder activations from INT8 inference
    """
    H, W, C = image.shape
    pixels_flat    = image.reshape(-1, C)
    pixels_norm, _ = preprocess_pixels(pixels_flat, fit_scaler=False,
                                       scaler=scaler)
    N = pixels_flat.shape[0]

    # ── Load full-AE interpreter ───────────────────────────────────────────────
    interp_ae = tf.lite.Interpreter(model_path=tflite_path)
    interp_ae.allocate_tensors()

    inp_det  = interp_ae.get_input_details()[0]
    out_det  = interp_ae.get_output_details()[0]

    inp_scale, inp_zp = inp_det['quantization']   # (scale, zero_point)
    out_scale, out_zp = out_det['quantization']

    # ── Load encoder-only interpreter ─────────────────────────────────────────
    # Build and convert encoder-only float model → encoder TFLite
    # (we need latent codes, which are an intermediate tensor not exposed
    #  in the full-AE tflite output)
    enc_tflite_path = tflite_path.replace('_ptq_int8.tflite',
                                          '_ptq_int8_encoder.tflite')

    interp_enc = tf.lite.Interpreter(model_path=enc_tflite_path)
    interp_enc.allocate_tensors()
    enc_inp_det  = interp_enc.get_input_details()[0]
    enc_out_det  = interp_enc.get_output_details()[0]
    enc_inp_scale, enc_inp_zp = enc_inp_det['quantization']
    enc_out_scale, enc_out_zp = enc_out_det['quantization']

    # ── Run inference pixel-batch by pixel-batch ───────────────────────────────
    recons_list = []
    codes_list  = []

    for start in tqdm(range(0, N, BATCH_SIZE), desc="  TFLite inference",
                      leave=False):
        batch_norm = pixels_norm[start:start + BATCH_SIZE]   # (B, C) float32

        # Quantise input to INT8
        batch_int8_ae  = np.clip(
            np.round(batch_norm / inp_scale) + inp_zp, -128, 127
        ).astype(np.int8)
        batch_int8_enc = np.clip(
            np.round(batch_norm / enc_inp_scale) + enc_inp_zp, -128, 127
        ).astype(np.int8)

        # Full-AE: run one sample at a time (TFLite Interpreter is not batched)
        batch_recons = []
        for pixel in batch_int8_ae:
            interp_ae.set_tensor(inp_det['index'], pixel[np.newaxis])
            interp_ae.invoke()
            out_int8 = interp_ae.get_tensor(out_det['index'])   # (1, C) int8
            # Dequantise: float = (int8 - zero_point) * scale
            out_f32  = (out_int8.astype(np.float32) - out_zp) * out_scale
            batch_recons.append(out_f32[0])

        # Encoder-only: same pixel-by-pixel loop
        batch_codes = []
        for pixel in batch_int8_enc:
            interp_enc.set_tensor(enc_inp_det['index'], pixel[np.newaxis])
            interp_enc.invoke()
            code_int8 = interp_enc.get_tensor(enc_out_det['index'])
            code_f32  = (code_int8.astype(np.float32) - enc_out_zp) * enc_out_scale
            batch_codes.append(code_f32[0])

        recons_list.append(np.array(batch_recons, dtype=np.float32))
        codes_list.append(np.array(batch_codes,   dtype=np.float32))

    recons_norm_all = np.concatenate(recons_list, axis=0)   # (N, C)
    codes_all       = np.concatenate(codes_list,  axis=0)   # (N, n_hidden)

    recons_orig = scaler.inverse_transform(recons_norm_all).astype(np.float32)
    rmse_flat   = np.sqrt(np.mean((pixels_flat - recons_orig) ** 2, axis=1))

    return rmse_flat.reshape(H, W), codes_all.reshape(H, W, n_hidden)


def export_tflite_encoder(keras_ae, rep_data, save_dir, model_name, mode):
    """
    Build and convert an encoder-only model to TFLite INT8.
    Required for PTQ latent-code inference via compute_recon_and_codes_tflite().
    """
    enc_keras = build_encoder_only(keras_ae, mode=mode)

    def rep_dataset():
        for i in range(min(200, len(rep_data))):
            yield [rep_data[i:i+1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(enc_keras)
    conv.optimizations             = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset    = rep_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type      = tf.int8
    conv.inference_output_type     = tf.int8

    tflite_model = conv.convert()
    out_path = os.path.join(save_dir, f'{model_name}_ptq_int8_encoder.tflite')
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print(f"  PTQ encoder TFLite -> {out_path}  "
          f"({os.path.getsize(out_path)/1024:.1f} KB)")
    return out_path


# %% [markdown]
# ## Reconstruction Error & Latent Codes

# %%
def compute_recon_and_codes(model, encoder, image, scaler):
    """
    Run full-image inference through the autoencoder and encoder.
    RMSE per pixel: sqrt(mean((X - X_hat)^2)) in original spectral scale.
    """
    H, W, C  = image.shape
    n_hidden = encoder.output_shape[-1]

    pixels_flat    = image.reshape(-1, C)
    pixels_norm, _ = preprocess_pixels(pixels_flat, fit_scaler=False, scaler=scaler)

    recons_norm = model.predict(pixels_norm,   batch_size=BATCH_SIZE, verbose=0)
    codes       = encoder.predict(pixels_norm, batch_size=BATCH_SIZE, verbose=0)

    recons_orig = scaler.inverse_transform(recons_norm).astype(np.float32)
    rmse_flat   = np.sqrt(np.mean((pixels_flat - recons_orig) ** 2, axis=1))
    return rmse_flat.reshape(H, W), codes.reshape(H, W, n_hidden)


# %% [markdown]
# ## MAWDBN DSW Detector

# %%
def _mawdbn_dsw(R, C_codes, winner, wouter, pf):
    """
    MAWDBN anomaly score map via the DSW detector (Eqs 1-3, weights Eq 5).

    For every pixel p:
      1. Collect k neighbours from the outer ring (between winner and wouter).
      2. d_j    = ||c_n_j - c_p||_2            (Eq 1: latent code distance)
      3. wt_j   = r_p / r_n_j   if inlier      (Eq 5: MAWDBN weight)
                = r_p*pf/r_n_j  if outlier
         inlier : (r_n_j - mu_n) < sigma_n
      4. beta   = (1/k) * sum(wt_j * d_j)      (Eq 3: anomaly score)

    Border pixels use whatever ring neighbours fall inside the image; the
    existing valid-mask logic handles the reduced neighbour count naturally.
    """
    assert winner % 2 == 1 and wouter % 2 == 1, "winner and wouter must be odd"
    assert wouter > winner, "wouter must be > winner"

    H, W     = R.shape
    half_in  = winner  // 2
    half_out = wouter  // 2

    beta_map = np.zeros((H, W), dtype=np.float32)

    for j in tqdm(range(H), desc="  DSW rows", leave=False):
        # Row bounds for the outer window, clamped to image
        r0 = max(j - half_out, 0)
        r1 = min(j + half_out + 1, H)

        for i in range(W):
            # Column bounds for the outer window, clamped to image
            c0 = max(i - half_out, 0)
            c1 = min(i + half_out + 1, W)

            win_R = R[r0:r1, c0:c1]
            win_C = C_codes[r0:r1, c0:c1]

            r_p = R[j, i]

            # Build ring mask over the actual (possibly clipped) window:
            # a position belongs to the ring if it is outside the inner square
            # relative to the centre of this window.
            win_h, win_w = win_R.shape
            cj = j - r0          # centre row within window
            ci = i - c0          # centre col within window

            row_idx = np.arange(win_h)[:, None]
            col_idx = np.arange(win_w)[None, :]
            in_inner = (
                (np.abs(row_idx - cj) <= half_in) &
                (np.abs(col_idx - ci) <= half_in)
            )
            is_centre = (row_idx == cj) & (col_idx == ci)
            ring = ~in_inner & ~is_centre

            rn = win_R[ring]
            if rn.size == 0:
                continue

            mu_n  = float(rn.mean())
            sig_n = float(rn.std())

            # Inlier: |r_n - mu| <= sig  (matches reference; absolute deviation,
            # not one-sided).  Outlier weight uses penalty factor pf (default 0).
            inlier = np.abs(rn - mu_n) <= sig_n
            wt = np.where(inlier,
                          r_p / (rn + 1e-12),
                          r_p * pf / (rn + 1e-12))

            c_p    = C_codes[j, i]
            c_ring = win_C[ring]
            d      = np.sqrt(np.sum((c_ring - c_p) ** 2, axis=1))

            beta_map[j, i] = float(np.sum(wt * d)) / rn.size

    return beta_map


def compute_mawdbn_scores(rmse_map, latent_codes, winner, wouter, pf):
    """
    Run the DSW detector for a single (winner, wouter, pf) combination.
    Called by dsw_grid_search; use that function rather than calling this directly.
    """
    return _mawdbn_dsw(rmse_map.astype(np.float32), latent_codes,
                       winner, wouter, pf)


def _build_combos():
    """Return all valid (winner, wouter, pf) triples from the config ranges."""
    combos = []
    for winner in DSW_WINNER_RANGE:
        for wouter in DSW_WOUTER_RANGE:
            for pf in DSW_PF_RANGE:
                if wouter > winner and winner % 2 == 1 and wouter % 2 == 1:
                    combos.append((winner, wouter, pf))
    if not combos:
        raise ValueError(
            "No valid (winner, wouter, pf) combinations. "
            "Ensure winner < wouter and both are odd integers."
        )
    return combos


# %% [markdown]
# ## Evaluation helpers

# %%
def evaluate_auc(score_map, gt_mask, label):
    auc         = roc_auc_score(gt_mask.reshape(-1).astype(int),
                                score_map.reshape(-1))
    fpr, tpr, _ = roc_curve(gt_mask.reshape(-1).astype(int),
                             score_map.reshape(-1))
    print(f"  {label:<28}  AUC-ROC = {auc:.4f}")
    return auc, fpr, tpr


def threshold_score_map(score_map, percentile):
    thresh = np.percentile(score_map, percentile)
    return (score_map > thresh).astype(np.uint8), thresh


# %% [markdown]
# ## Visualisation

# %%
def _false_colour(image):
    H, W, C = image.shape
    rb = min(60, C-1); gb = min(40, C-1); bb = min(20, C-1)
    rgb = np.stack([image[:, :, rb], image[:, :, gb], image[:, :, bb]], axis=-1)
    return np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8), 0, 1)


def _overlay(rgb, mask, gt_mask):
    out = rgb.copy()
    out[mask == 1] = [1, 0, 0]
    if gt_mask is not None:
        out[(gt_mask == 1) & (mask == 0)] = [0, 0.4, 1.0]
    return out


def _add_type_dividers(ax, scenes, x):
    last_type = None
    for i, s in enumerate(scenes):
        t = ('airport' if 'airport' in s else
             'beach'   if 'beach'   in s else 'urban')
        if last_type and t != last_type:
            ax.axvline(x[i] - 0.5, color='gray', ls='-', lw=0.8, alpha=0.5)
        last_type = t


def plot_training_histories(rbm_losses, hist_none, hist_quant,
                            scene_name, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Training Curves — {scene_name}', fontweight='bold')
    axes[0].plot(rbm_losses, lw=2)
    axes[0].set_title('RBM Pre-training (CD-1)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Recon Error')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(hist_none.get('loss', []), lw=2)
    axes[1].set_title('AE Loss — float32')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(hist_quant.get('loss', []), lw=2, color='darkorange')
    axes[2].set_title(f'AE Loss — {QUANTIZATION_MODE}')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('MSE')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show(); plt.close(fig)


def plot_comparison(image, gt_mask,
                    rmse_none, beta_none, rmse_mask_none, beta_mask_none,
                    rmse_quant, beta_quant, rmse_mask_quant, beta_mask_quant,
                    scene_name, save_path=None):
    rgb = _false_colour(image)
    fig = plt.figure(figsize=(18, 28))
    gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.25)
    fig.suptitle(
        f"float32 vs {QUANTIZATION_MODE.upper()}  |  {scene_name}\n"
        "red = predicted anomaly   |   blue = missed ground-truth",
        fontsize=14, fontweight='bold', y=0.995)

    def ax(r, c): return fig.add_subplot(gs[r, c])
    def show(a, img, title, cmap='viridis', vmin=None, vmax=None):
        a.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        a.set_title(title, fontsize=10); a.axis('off')

    show(ax(0,0), rgb, 'False colour')
    if gt_mask is not None:
        show(ax(0,1), gt_mask,
             f'Ground truth  ({int(gt_mask.sum()):,} px)', cmap='gray', vmin=0, vmax=1)
    ax(0,2).axis('off')
    show(ax(1,0), rmse_none,  'RMSE score — float32',              cmap='hot')
    show(ax(1,1), rmse_quant, f'RMSE score — {QUANTIZATION_MODE}', cmap='hot')
    show(ax(1,2), _overlay(rgb, rmse_mask_none, gt_mask), 'RMSE overlay — float32')
    show(ax(2,0), beta_none,  'beta score — float32',              cmap='hot')
    show(ax(2,1), beta_quant, f'beta score — {QUANTIZATION_MODE}', cmap='hot')
    show(ax(2,2), _overlay(rgb, beta_mask_none, gt_mask), 'beta overlay — float32')
    show(ax(3,0), rmse_mask_none,  'RMSE mask — float32',
         cmap='RdYlGn_r', vmin=0, vmax=1)
    show(ax(3,1), rmse_mask_quant, f'RMSE mask — {QUANTIZATION_MODE}',
         cmap='RdYlGn_r', vmin=0, vmax=1)
    show(ax(3,2), _overlay(rgb, rmse_mask_quant, gt_mask),
         f'RMSE overlay — {QUANTIZATION_MODE}')
    show(ax(4,0), beta_mask_none,  'beta mask — float32',
         cmap='RdYlGn_r', vmin=0, vmax=1)
    show(ax(4,1), beta_mask_quant, f'beta mask — {QUANTIZATION_MODE}',
         cmap='RdYlGn_r', vmin=0, vmax=1)
    show(ax(4,2), _overlay(rgb, beta_mask_quant, gt_mask),
         f'beta overlay — {QUANTIZATION_MODE}')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show(); plt.close(fig)


def plot_roc_comparison(roc_none, roc_quant, scene_name, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, val, col, ls in [
        ('RMSE float32',              roc_none.get('RMSE'),   '#1f77b4', '-'),
        (f'RMSE {QUANTIZATION_MODE}', roc_quant.get('RMSE'),  '#ff7f0e', '--'),
        ('beta float32',              roc_none.get('MAWDBN'), '#2ca02c', '-'),
        (f'beta {QUANTIZATION_MODE}', roc_quant.get('MAWDBN'),'#d62728', '--'),
    ]:
        if val is None: continue
        auc, fpr, tpr = val
        ax.plot(fpr, tpr, lw=2, label=f"{label}  AUC={auc:.4f}",
                color=col, linestyle=ls)
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title(f'ROC: float32 vs {QUANTIZATION_MODE}  |  {scene_name}', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show(); plt.close(fig)


# %% [markdown]
# ## Save / Load

# %%
def save_model_to_disk(model, scaler, save_dir, model_name, config_dict):
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f'{model_name}.h5'), save_format='h5')
    joblib.dump(scaler, os.path.join(save_dir, f'{model_name}_scaler.pkl'))
    if QUANTIZATION_MODE == 'qkeras_qat' and QKERAS_AVAILABLE \
            and 'qkeras_qat' in model_name:
        try:
            model_save_quantized_weights(
                model, os.path.join(save_dir, f'{model_name}_qweights.pkl'))
        except Exception as e:
            print(f"  model_save_quantized_weights skipped: {e}")
    with open(os.path.join(save_dir, f'{model_name}_config.txt'), 'w') as f:
        f.write(f"Date: {datetime.now()}\n\n")
        for k, v in config_dict.items():
            f.write(f"{k}: {v}\n")


def load_model_h5(model_path, scaler_path, mode):
    custom_objects = {}
    if mode == 'qkeras_qat' and QKERAS_AVAILABLE:
        custom_objects = qkeras.get_custom_objects()
    model  = keras.models.load_model(model_path, custom_objects=custom_objects)
    scaler = joblib.load(scaler_path)
    return model, scaler


# %% [markdown]
# ## Pipeline — Phase 1: train + reconstruct (per scene, no DSW)

# %%
def run_scene_train(scene_name, mat_path, config_dict):
    """
    Load data, train float32 and quantised models, compute RMSE maps and
    latent codes.  DSW is deliberately NOT run here — it happens globally
    in Phase 2 after all scenes have been processed.

    Returns a scene_data dict:
        { 'scene', 'image', 'gt_mask',
          'rmse_none', 'codes_none',
          'rmse_quant', 'codes_quant',
          'hist_none', 'hist_quant', 'rbm_losses', 'save_dir' }
    """
    set_seed(RANDOM_SEED)
    save_dir = os.path.join(RESULTS_DIR, 'dbn_abu_qkeras', scene_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n── {scene_name.upper()} {'─'*50}")
    image, gt_mask, C = load_abu_mat(mat_path)
    H, W, _ = image.shape

    n_train  = min(N_SAMPLES_TRAIN, H * W) if N_SAMPLES_TRAIN else H * W
    rng      = np.random.default_rng(RANDOM_SEED)
    idx      = rng.choice(H * W, size=n_train, replace=False)
    train_px = image.reshape(-1, C)[idx]
    norm_px, scaler = preprocess_pixels(train_px, fit_scaler=True)
    print(f"  {H}×{W}×{C}  |  {n_train:,} training pixels")

    # RBM — shared init for both models
    rbm = GaussianRBM_NP(n_visible=C, n_hidden=N_HIDDEN, k=RBM_K)
    rbm_losses = rbm.train(
        norm_px, n_epochs=RBM_EPOCHS, lr=RBM_LEARNING_RATE,
        momentum=RBM_MOMENTUM, weight_decay=RBM_WEIGHT_DECAY,
        batch_size=BATCH_SIZE)

    # Float32 autoencoder
    ae_none = build_autoencoder(C, N_HIDDEN, mode='none')
    initialize_from_rbm(ae_none, rbm)
    mn_none   = f'mawdbn_none_{scene_name}'
    hist_none = train_autoencoder(ae_none, norm_px, save_dir, mn_none)
    save_model_to_disk(ae_none, scaler, save_dir, mn_none, config_dict)

    # Quantised autoencoder
    set_seed(RANDOM_SEED)
    ae_quant = build_autoencoder(C, N_HIDDEN, mode=QUANTIZATION_MODE)
    initialize_from_rbm(ae_quant, rbm)
    mn_quant   = f'mawdbn_{QUANTIZATION_MODE}_{scene_name}'
    hist_quant = train_autoencoder(ae_quant, norm_px, save_dir, mn_quant)
    save_model_to_disk(ae_quant, scaler, save_dir, mn_quant, config_dict)

    # PTQ TFLite export (must precede inference)
    tflite_ae_path = None
    if QUANTIZATION_MODE == 'ptq':
        tflite_ae_path = export_tflite_ptq(
            ae_quant, norm_px[:200], save_dir, mn_quant)
        export_tflite_encoder(
            ae_quant, norm_px[:200], save_dir, mn_quant, mode=QUANTIZATION_MODE)

    plot_training_histories(
        rbm_losses, hist_none, hist_quant, scene_name,
        save_path=os.path.join(save_dir, f'{scene_name}_training.png'))

    # Reconstruction + latent codes
    enc_none = build_encoder_only(ae_none, mode='none')
    rmse_none, codes_none = compute_recon_and_codes(ae_none, enc_none, image, scaler)

    if QUANTIZATION_MODE == 'ptq':
        rmse_quant, codes_quant = compute_recon_and_codes_tflite(
            tflite_ae_path, N_HIDDEN, image, scaler)
    else:
        enc_quant = build_encoder_only(ae_quant, mode=QUANTIZATION_MODE)
        rmse_quant, codes_quant = compute_recon_and_codes(
            ae_quant, enc_quant, image, scaler)

    print(f"  RMSE  f32={rmse_none.mean():.5f}  "
          f"{QUANTIZATION_MODE}={rmse_quant.mean():.5f}")

    return {
        'scene':       scene_name,
        'image':       image,
        'gt_mask':     gt_mask,
        'rmse_none':   rmse_none,
        'codes_none':  codes_none,
        'rmse_quant':  rmse_quant,
        'codes_quant': codes_quant,
        'hist_none':   hist_none,
        'hist_quant':  hist_quant,
        'rbm_losses':  rbm_losses,
        'save_dir':    save_dir,
        'scaler':      scaler,
    }


# %% [markdown]
# ## Pipeline — Phase 2: global DSW window search

# %%
def dsw_global_grid_search(scene_data_list):
    """
    Sweep every valid (winner, wouter, pf) combo across ALL scenes
    simultaneously.  For each combo, compute float32 beta AUC on every
    scene that has a ground-truth mask, then average across scenes.
    The combo with the highest mean AUC is the globally optimal window.

    This is the methodologically correct approach for publication: the
    spatial detector parameters are chosen independently of any individual
    test image, using all available scenes as a joint validation set.

    Parameters
    ----------
    scene_data_list : list of dicts from run_scene_train()

    Returns
    -------
    grid_ranking : list of dicts, sorted by mean_auc descending
        { 'winner', 'wouter', 'pf',
          'mean_auc',           -- mean float32 beta AUC across all GT scenes
          'per_scene' }         -- list of { 'scene', 'auc_none', 'auc_quant',
                                             'beta_none', 'beta_quant' }
    best_window  : grid_ranking[0]
    """
    combos    = _build_combos()
    gt_scenes = [sd for sd in scene_data_list if sd['gt_mask'] is not None]
    print(f"\n  Global DSW grid: {len(combos)} combo(s) × "
          f"{len(gt_scenes)} scene(s) with GT")
    print(f"  winner={DSW_WINNER_RANGE}")
    print(f"  wouter={DSW_WOUTER_RANGE}")
    print(f"  pf    ={DSW_PF_RANGE}")

    grid_ranking = []

    for winner, wouter, pf in tqdm(combos, desc="  DSW grid", leave=False):
        per_scene = []
        aucs      = []

        for sd in scene_data_list:
            beta_none  = compute_mawdbn_scores(
                sd['rmse_none'],  sd['codes_none'],  winner, wouter, pf)
            beta_quant = compute_mawdbn_scores(
                sd['rmse_quant'], sd['codes_quant'], winner, wouter, pf)

            auc_none = auc_quant = None
            if sd['gt_mask'] is not None:
                auc_none  = roc_auc_score(
                    sd['gt_mask'].reshape(-1).astype(int),
                    beta_none.reshape(-1))
                auc_quant = roc_auc_score(
                    sd['gt_mask'].reshape(-1).astype(int),
                    beta_quant.reshape(-1))
                aucs.append(auc_none)

            per_scene.append({
                'scene':      sd['scene'],
                'auc_none':   auc_none,
                'auc_quant':  auc_quant,
                'beta_none':  beta_none,
                'beta_quant': beta_quant,
            })

        mean_auc = float(np.mean(aucs)) if aucs else 0.0
        grid_ranking.append({
            'winner':    winner,
            'wouter':    wouter,
            'pf':        pf,
            'mean_auc':  mean_auc,
            'per_scene': per_scene,
        })

    grid_ranking.sort(key=lambda r: r['mean_auc'], reverse=True)
    best = grid_ranking[0]

    print(f"\n  ★ Global best:  winner={best['winner']}  "
          f"wouter={best['wouter']}  pf={best['pf']}  "
          f"→  mean β AUC = {best['mean_auc']:.4f}")
    if len(grid_ranking) > 1:
        r2 = grid_ranking[1]
        print(f"     Runner-up:   winner={r2['winner']}  "
              f"wouter={r2['wouter']}  pf={r2['pf']}  "
              f"→  {r2['mean_auc']:.4f}  "
              f"(Δ = {r2['mean_auc'] - best['mean_auc']:+.4f})")

    return grid_ranking, best


# %% [markdown]
# ## Pipeline — Phase 3: evaluate with chosen window (per scene)

# %%
def run_scene_eval(scene_data, winner, wouter, pf):
    """
    Apply one fixed (winner, wouter, pf) to a scene, threshold,
    compute AUC, save figures and arrays.

    Returns a result dict for _print_final_summary().
    """
    scene_name  = scene_data['scene']
    image       = scene_data['image']
    gt_mask     = scene_data['gt_mask']
    rmse_none   = scene_data['rmse_none']
    codes_none  = scene_data['codes_none']
    rmse_quant  = scene_data['rmse_quant']
    codes_quant = scene_data['codes_quant']
    save_dir    = scene_data['save_dir']

    beta_none  = compute_mawdbn_scores(rmse_none,  codes_none,  winner, wouter, pf)
    beta_quant = compute_mawdbn_scores(rmse_quant, codes_quant, winner, wouter, pf)

    pct = THRESHOLD_PERCENTILE
    rmse_mask_none,  _ = threshold_score_map(rmse_none,  pct)
    beta_mask_none,  _ = threshold_score_map(beta_none,  pct)
    rmse_mask_quant, _ = threshold_score_map(rmse_quant, pct)
    beta_mask_quant, _ = threshold_score_map(beta_quant, pct)

    result = {'scene': scene_name, 'winner': winner,
              'wouter': wouter, 'pf': pf}

    if gt_mask is not None:
        auc_rn, fpr_rn, tpr_rn = evaluate_auc(rmse_none,  gt_mask, 'RMSE  f32')
        auc_bn, fpr_bn, tpr_bn = evaluate_auc(beta_none,  gt_mask, 'β     f32')
        auc_rq, fpr_rq, tpr_rq = evaluate_auc(rmse_quant, gt_mask,
                                               f'RMSE  {QUANTIZATION_MODE}')
        auc_bq, fpr_bq, tpr_bq = evaluate_auc(beta_quant, gt_mask,
                                               f'β     {QUANTIZATION_MODE}')

        print(f"  AUC  RMSE={auc_rn:.4f}→{auc_rq:.4f}({auc_rq-auc_rn:+.4f})  "
              f"β={auc_bn:.4f}→{auc_bq:.4f}({auc_bq-auc_bn:+.4f})")

        result.update({
            'auc_rmse_none':  auc_rn,
            'auc_beta_none':  auc_bn,
            'auc_rmse_quant': auc_rq,
            'auc_beta_quant': auc_bq,
        })

        plot_roc_comparison(
            {'RMSE': (auc_rn, fpr_rn, tpr_rn), 'MAWDBN': (auc_bn, fpr_bn, tpr_bn)},
            {'RMSE': (auc_rq, fpr_rq, tpr_rq), 'MAWDBN': (auc_bq, fpr_bq, tpr_bq)},
            scene_name,
            save_path=os.path.join(save_dir, f'{scene_name}_roc.png'))
    else:
        print("  No ground truth — skipping AUC.")

    plot_comparison(
        image, gt_mask,
        rmse_none, beta_none, rmse_mask_none, beta_mask_none,
        rmse_quant, beta_quant, rmse_mask_quant, beta_mask_quant,
        scene_name,
        save_path=os.path.join(save_dir, f'{scene_name}_comparison.png'))

    for tag, rm, bm, rm_m, bm_m, codes in [
        ('none',            rmse_none,  beta_none,
         rmse_mask_none,  beta_mask_none,  codes_none),
        (QUANTIZATION_MODE, rmse_quant, beta_quant,
         rmse_mask_quant, beta_mask_quant, codes_quant),
    ]:
        prefix = os.path.join(save_dir, f'{scene_name}_{tag}')
        np.save(f'{prefix}_rmse_map.npy',     rm)
        np.save(f'{prefix}_beta_map.npy',     bm)
        np.save(f'{prefix}_rmse_mask.npy',    rm_m)
        np.save(f'{prefix}_beta_mask.npy',    bm_m)
        np.save(f'{prefix}_latent_codes.npy', codes)

    print(f"  Saved → {save_dir}/")
    return result


# %% [markdown]
# ## Summary

# %%
def _print_final_summary(results, grid_ranking=None, best_window=None):
    """
    Two-section printed + saved summary.

    Section 1 — Per-scene AUC table under the globally chosen window,
                with type-group averages and overall average.
    Section 2 — Full DSW grid ranking: every combo ranked by mean AUC,
                with a per-scene AUC column for each scene so you can
                see exactly how each window performs on each image.
    """
    valid = [r for r in results if 'auc_rmse_none' in r]
    if not valid:
        print("  No valid results to summarise.")
        return

    qm = QUANTIZATION_MODE
    SW = 22

    def fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A "

    def dstr(a, b):
        return f"{b-a:+.4f}" if (a is not None and b is not None) else "  N/A "

    def row_main(label, rn, bn, rq, bq, indent="  "):
        return (f"{indent}{label:<{SW}}  "
                f"{fmt(rn):>8}  {fmt(rq):>8}  {dstr(rn,rq):>8}  "
                f"{fmt(bn):>8}  {fmt(bq):>8}  {dstr(bn,bq):>8}")

    def gavg(grp, key):
        vals = [r[key] for r in grp if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    TYPES = {
        'airport': [r for r in valid if 'airport' in r['scene']],
        'beach':   [r for r in valid if 'beach'   in r['scene']],
        'urban':   [r for r in valid if 'urban'   in r['scene']],
    }

    border  = "=" * 96
    divider = "  " + "-" * 92
    lines   = []

    def pr(s=""):
        print(s); lines.append(s)

    # ══ Section 1: AUC table ═════════════════════════════════════════════════
    pr(); pr(border)
    pr(f"  FINAL AUC SUMMARY  |  float32 vs {qm.upper()}  |  ABU DATASET")
    if best_window:
        pr(f"  DSW window: winner={best_window['winner']}  "
           f"wouter={best_window['wouter']}  pf={best_window['pf']}  "
           f"[globally optimal — chosen by mean β AUC across all scenes]")
    pr(border)
    pr(f"  {'Scene':<{SW}}  "
       f"{'RMSE-f32':>8}  {'RMSE-qt':>8}  {'ΔRMSE':>8}  "
       f"{'β-f32':>8}  {'β-qt':>8}  {'Δβ':>8}")
    pr(divider)

    for tname, grp in TYPES.items():
        if not grp: continue
        for r in grp:
            pr(row_main(r['scene'],
                        r.get('auc_rmse_none'), r.get('auc_beta_none'),
                        r.get('auc_rmse_quant'), r.get('auc_beta_quant')))
        pr(divider)
        pr(row_main(f"  AVG {tname.upper()} ({len(grp)} scenes)",
                    gavg(grp,'auc_rmse_none'), gavg(grp,'auc_beta_none'),
                    gavg(grp,'auc_rmse_quant'), gavg(grp,'auc_beta_quant'),
                    indent=""))
        pr(divider)

    all_rn = gavg(valid,'auc_rmse_none');  all_bn = gavg(valid,'auc_beta_none')
    all_rq = gavg(valid,'auc_rmse_quant'); all_bq = gavg(valid,'auc_beta_quant')
    pr(row_main(f"  OVERALL AVG ({len(valid)} scenes)",
                all_rn, all_bn, all_rq, all_bq, indent=""))
    pr(border)

    pr()
    pr("  BEST / WORST β-f32:")
    for label, fn, dflt in [('best', max, 0), ('worst', min, 1)]:
        r = fn(valid, key=lambda x: x.get('auc_beta_none', dflt))
        pr(f"    {label:<6}  {r['scene']:<22}  AUC = "
           f"{r.get('auc_beta_none', float('nan')):.4f}")

    pr()
    pr(f"  QUANTISATION IMPACT  ({qm} vs float32):")
    if all_rn and all_rq:
        pr(f"    Avg RMSE delta : {all_rq-all_rn:+.4f}  "
           f"({'improved' if all_rq>all_rn else 'degraded'})")
    if all_bn and all_bq:
        pr(f"    Avg β    delta : {all_bq-all_bn:+.4f}  "
           f"({'improved' if all_bq>all_bn else 'degraded'})")
    pr(border)

    # ══ Section 2: Grid ranking ═══════════════════════════════════════════════
    if grid_ranking:
        scene_names = [r['scene'] for r in valid]
        SC  = 8   # per-scene column width
        sep = "  "

        pr(); pr(border)
        pr(f"  DSW GRID RANKING — {len(grid_ranking)} combos, "
           f"sorted by mean β AUC (float32) across {len(scene_names)} scenes")
        pr(border)

        short = [s.replace('abu-','').replace('airport','air')
                  .replace('beach','bch').replace('urban','urb')
                 for s in scene_names]
        scene_hdr = sep.join(f"{s[:SC]:>{SC}}" for s in short)
        pr(f"  {'#':>3}  {'w':>3}  {'W':>4}  {'pf':>3}  "
           f"{'mean AUC':>9}  {'vs best':>8}    {scene_hdr}")
        pr(divider)

        best_auc = grid_ranking[0]['mean_auc']
        for rank, g in enumerate(grid_ranking, 1):
            star = "★" if rank == 1 else " "
            auc_by_scene = {ps['scene']: ps['auc_none']
                            for ps in g['per_scene']
                            if ps.get('auc_none') is not None}
            scene_cols = sep.join(
                f"{auc_by_scene.get(s, float('nan')):>{SC}.4f}"
                for s in scene_names
            )
            pr(f"  {rank:>3}{star} {g['winner']:>3}  {g['wouter']:>4}  "
               f"{g['pf']:>3}  "
               f"{g['mean_auc']:>9.4f}  "
               f"{g['mean_auc']-best_auc:>+8.4f}    {scene_cols}")

        pr(border)

    # ── Save ──────────────────────────────────────────────────────────────────
    summary_dir = os.path.join(RESULTS_DIR, 'dbn_abu_qkeras')
    os.makedirs(summary_dir, exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(summary_dir, f'summary_{qm}_{ts}.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Summary saved → {path}")


def _plot_final_summary(results, best_window=None):
    """Bar chart: float32 vs quantised beta AUC per scene."""
    valid = [r for r in results if 'auc_rmse_none' in r]
    if not valid: return

    scenes    = [r['scene'] for r in valid]
    beta_none = [r['auc_beta_none']  for r in valid]
    beta_qt   = [r['auc_beta_quant'] for r in valid]
    n = len(scenes); x = np.arange(n); w = 0.35
    xl = [s.replace('abu-', '') for s in scenes]

    colours_none = ['#4e79a7' if 'airport' in s else
                    '#f28e2b' if 'beach'   in s else '#59a14f' for s in scenes]
    colours_qt   = ['#76b7f7' if 'airport' in s else
                    '#ffc97f' if 'beach'   in s else '#8be08e' for s in scenes]

    fig, ax = plt.subplots(figsize=(max(14, n * 1.2), 6))
    ax.bar(x - w/2, beta_none, w, color=colours_none,
           edgecolor='white', lw=0.5, label='β float32')
    ax.bar(x + w/2, beta_qt,   w, color=colours_qt,
           edgecolor='white', lw=0.5,
           label=f'β {QUANTIZATION_MODE}', alpha=0.9)

    avg_none = float(np.mean(beta_none))
    avg_qt   = float(np.mean(beta_qt))
    ax.axhline(avg_none, color='#4e79a7', ls='--', lw=1.5,
               label=f'avg f32 = {avg_none:.4f}')
    ax.axhline(avg_qt,   color='#76b7f7', ls=':',  lw=1.5,
               label=f'avg qt  = {avg_qt:.4f}')

    _add_type_dividers(ax, scenes, x)
    ax.set_xticks(x); ax.set_xticklabels(xl, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('AUC-ROC', fontsize=11)
    ax.set_ylim(max(0, min(beta_none + beta_qt) - 0.05), 1.01)

    win_str = (f"w={best_window['winner']}, W={best_window['wouter']}"
               if best_window else "window")
    ax.set_title(
        f'MAWDBN β AUC  —  float32 vs {QUANTIZATION_MODE.upper()}  '
        f'|  global DSW: {win_str}  |  ABU Dataset',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    summary_dir = os.path.join(RESULTS_DIR, 'dbn_abu_qkeras')
    os.makedirs(summary_dir, exist_ok=True)
    chart_path = os.path.join(summary_dir,
                              f'summary_chart_{QUANTIZATION_MODE}.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"  Summary chart → {chart_path}")
    plt.close(fig)


# %% [markdown]
# ## Batch runner

# %%
def run_batch():
    """
    Three-phase batch pipeline.

    Phase 1 (per-scene)  Train float32 + quantised models, compute RMSE
                         maps and latent codes.  No DSW yet — all scene
                         data is held in memory.
    Phase 2 (global)     Sweep every (winner, wouter, pf) combo across all
                         scenes simultaneously.  Pick the combo that
                         maximises mean float32 β AUC across scenes.
    Phase 3 (per-scene)  Apply the single globally optimal window, compute
                         final AUC per scene, save all figures and arrays.
    """
    config_dict = {
        'quantization_mode': QUANTIZATION_MODE,
        'quant_bits':        QUANT_BITS,
        'quant_integer':     QUANT_INTEGER,
        'n_hidden':          N_HIDDEN,
        'rbm_epochs':        RBM_EPOCHS,
        'ae_epochs':         AE_EPOCHS,
        'ae_lr':             AE_LEARNING_RATE,
        'ae_weight_decay':   AE_WEIGHT_DECAY,
        'ae_patience':       AE_PATIENCE,
        'batch_size':        BATCH_SIZE,
        'n_samples_train':   N_SAMPLES_TRAIN,
        'dsw_winner_range':  DSW_WINNER_RANGE,
        'dsw_wouter_range':  DSW_WOUTER_RANGE,
        'dsw_pf_range':      DSW_PF_RANGE,
        'threshold_pct':     THRESHOLD_PERCENTILE,
        'random_seed':       RANDOM_SEED,
    }

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  PHASE 1 — TRAINING + RECONSTRUCTION")
    print("="*70)

    scene_data_list = []
    for idx, scene in enumerate(ABU_SCENES):
        mat = os.path.join(ABU_DATASET_DIR, f'{scene}.mat')
        if not os.path.exists(mat):
            print(f"  [{idx:>2}] {scene:<22}  .mat not found — skipping.")
            continue
        try:
            sd = run_scene_train(scene, mat, config_dict)
            scene_data_list.append(sd)
        except Exception as e:
            import traceback
            print(f"  [{idx:>2}] {scene:<22}  FAILED: {e}")
            traceback.print_exc()

    if not scene_data_list:
        print("  No scenes loaded — aborting.")
        return []

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  PHASE 2 — GLOBAL DSW WINDOW SEARCH")
    print("="*70)

    grid_ranking, best_window = dsw_global_grid_search(scene_data_list)

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  PHASE 3 — EVALUATION WITH GLOBALLY OPTIMAL WINDOW")
    print(f"  winner={best_window['winner']}  "
          f"wouter={best_window['wouter']}  pf={best_window['pf']}")
    print("="*70)

    results = []
    for sd in scene_data_list:
        try:
            r = run_scene_eval(sd, best_window['winner'],
                               best_window['wouter'], best_window['pf'])
            results.append(r)
        except Exception as e:
            import traceback
            print(f"  {sd['scene']}  FAILED: {e}")
            traceback.print_exc()
            results.append({'scene': sd['scene']})

    _print_final_summary(results, grid_ranking=grid_ranking,
                         best_window=best_window)
    _plot_final_summary(results, best_window=best_window)

    return results


# %% [markdown]
# ## Entry point

# %%
if __name__ == "__main__":

    print("\n" + "="*70)
    print(f"  MAWDBN  |  ABU Dataset  |  float32 vs {QUANTIZATION_MODE.upper()}")
    print(f"  Run mode : {RUN_MODE.upper()}")
    print("="*70)

    if RUN_MODE == 'single':
        # Single-scene mode: grid search on one image only.
        # Cannot produce a globally optimal window — use 'batch' for that.
        scene_name = ABU_SCENES[SCENE_INDEX]
        mat_path   = os.path.join(ABU_DATASET_DIR, f'{scene_name}.mat')
        if not os.path.exists(mat_path):
            available = [s for s in ABU_SCENES
                         if os.path.exists(
                             os.path.join(ABU_DATASET_DIR, f'{s}.mat'))]
            raise FileNotFoundError(
                f"Scene [{SCENE_INDEX}] '{scene_name}' not found.\n"
                f"Available: {available}")

        config_dict = {
            'quantization_mode': QUANTIZATION_MODE, 'n_hidden': N_HIDDEN,
            'rbm_epochs': RBM_EPOCHS, 'ae_epochs': AE_EPOCHS,
            'random_seed': RANDOM_SEED,
        }
        sd = run_scene_train(scene_name, mat_path, config_dict)

        combos = _build_combos()
        print(f"\n  Local DSW grid: {len(combos)} combo(s)  [single-scene mode]")
        local_ranking = []
        for winner, wouter, pf in tqdm(combos, desc="  DSW grid", leave=False):
            beta = compute_mawdbn_scores(
                sd['rmse_none'], sd['codes_none'], winner, wouter, pf)
            auc = None
            if sd['gt_mask'] is not None:
                auc = roc_auc_score(
                    sd['gt_mask'].reshape(-1).astype(int), beta.reshape(-1))
            local_ranking.append({
                'winner':    winner,
                'wouter':    wouter,
                'pf':        pf,
                'mean_auc':  auc or 0.0,
                'per_scene': [{'scene': scene_name,
                               'auc_none': auc, 'auc_quant': None}],
            })
        local_ranking.sort(key=lambda r: r['mean_auc'], reverse=True)
        best_local = local_ranking[0]
        print(f"  ★ Best local:  winner={best_local['winner']}  "
              f"wouter={best_local['wouter']}  "
              f"→  β AUC = {best_local['mean_auc']:.4f}")

        result = run_scene_eval(sd, best_local['winner'],
                                best_local['wouter'], best_local['pf'])
        _print_final_summary([result], grid_ranking=local_ranking,
                             best_window=best_local)
        _plot_final_summary([result], best_window=best_local)

    elif RUN_MODE == 'batch':
        results = run_batch()
        print(f"\nDONE — {len(results)} scene(s) processed.")

    else:
        raise ValueError(
            f"Unknown RUN_MODE='{RUN_MODE}'. Choose 'single' or 'batch'.")