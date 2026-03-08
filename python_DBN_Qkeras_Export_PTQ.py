# %% [markdown]
# MAWDBN – Hyperspectral Anomaly Detection on ABU Dataset
# PCA variant: reduces ABU 205-band images to 120 bands before training.
# The trained model, TFLite export, and all validation data are therefore
# 120-band — matching HYPSO native band count directly.
#
# PCA is fit on a random subsample of pixels from each scene independently,
# then applied to the full image before the RBM/AE training pipeline runs.
# The PCA object is saved alongside the model for reproducibility.

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

ABU_DATASET_DIR   = 'ABU_DATASET'
RESULTS_DIR       = 'results'

QUANTIZATION_MODE = 'ptq'
QUANT_BITS        = 8
QUANT_INTEGER     = 2
QUANT_ACT_BITS    = 8
QUANT_ACT_INT     = 1

N_PCA_BANDS = 120   # target band count; HYPSO native band count

N_HIDDEN = 13

RBM_EPOCHS        = 50
RBM_LEARNING_RATE = 0.01
RBM_K             = 1
RBM_MOMENTUM      = 0.5
RBM_WEIGHT_DECAY  = 0.0002

AE_EPOCHS        = 50
AE_LEARNING_RATE = 0.001
AE_WEIGHT_DECAY  = 0.0002
AE_PATIENCE      = 15

N_SAMPLES_TRAIN      = 50000
BATCH_SIZE           = 2048
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
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, losses
from tensorflow.keras.regularizers import l2 as _l2
try:
    import qkeras
    from qkeras import QDense, QActivation
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
print(f"  TF {tf.__version__}  |  quant={QUANTIZATION_MODE}  |  PCA -> {N_PCA_BANDS} bands")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed=RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


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


def apply_pca(image, n_components=N_PCA_BANDS, seed=RANDOM_SEED):
    """
    Fit PCA on a random subsample of pixels from the image.
    Returns (image_reduced [H, W, n_components], pca_object).
    If the image already has <= n_components bands, returns unchanged with pca=None.
    """
    H, W, C = image.shape
    if C <= n_components:
        print(f"  PCA skipped — image already has {C} bands (<= {n_components})")
        return image, None
    flat = image.reshape(-1, C)
    n    = min(N_SAMPLES_TRAIN, H * W)
    rng  = np.random.default_rng(seed)
    idx  = rng.choice(H * W, size=n, replace=False)
    pca  = PCA(n_components=n_components, random_state=seed)
    pca.fit(flat[idx])
    reduced      = pca.transform(flat).reshape(H, W, n_components).astype(np.float32)
    var_explained = float(pca.explained_variance_ratio_.sum()) * 100
    print(f"  PCA: {C} -> {n_components} bands  ({var_explained:.1f}% variance retained)")
    return reduced, pca


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
        print(f"  RBM {self.n_visible}->{self.n_hidden}  epochs={n_epochs}  "
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
    print(f"  PTQ -> {path}  ({os.path.getsize(path)/1024:.1f} KB)")
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


def compute_recon_and_codes_tflite(tflite_ae_path, n_hidden, image_pca, scaler):
    """
    Run TFLite inference on PCA-reduced image.
    image_pca is already [H, W, 120] — no further PCA needed here.
    RMSE is computed in PCA space (not original DN space) since the model
    was trained on PCA-reduced data.
    """
    H, W, C   = image_pca.shape
    N         = H * W
    pixels_flat = image_pca.reshape(-1, C)
    pixels_norm, _ = preprocess_pixels(pixels_flat, fit_scaler=False, scaler=scaler)

    enc_path = tflite_ae_path.replace('_ptq_int8.tflite', '_ptq_int8_encoder.tflite')

    interp_ae  = tf.lite.Interpreter(model_path=tflite_ae_path)
    interp_ae.allocate_tensors()
    inp_det  = interp_ae.get_input_details()[0]
    out_det  = interp_ae.get_output_details()[0]
    inp_scale, inp_zp = inp_det['quantization']
    out_scale, out_zp = out_det['quantization']

    interp_enc = tf.lite.Interpreter(model_path=enc_path)
    interp_enc.allocate_tensors()
    enc_inp = interp_enc.get_input_details()[0]
    enc_out = interp_enc.get_output_details()[0]
    enc_in_s, enc_in_zp   = enc_inp['quantization']
    enc_out_s, enc_out_zp = enc_out['quantization']

    recons_list, codes_list = [], []
    for start in tqdm(range(0, N, BATCH_SIZE), desc="  TFLite inference", leave=False):
        batch = pixels_norm[start:start + BATCH_SIZE]
        q_ae  = np.clip(np.round(batch / inp_scale) + inp_zp,   -128, 127).astype(np.int8)
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

    # RMSE in PCA-normalised space — model was trained here
    rmse_flat = np.sqrt(np.mean((pixels_norm - recons_norm) ** 2, axis=1))
    return rmse_flat.reshape(H, W), codes_all.reshape(H, W, n_hidden)


def compute_recon_and_codes(model, encoder, image_pca, scaler):
    H, W, C = image_pca.shape
    pixels  = image_pca.reshape(-1, C)
    norm, _ = preprocess_pixels(pixels, fit_scaler=False, scaler=scaler)
    recons  = model.predict(norm,   batch_size=BATCH_SIZE, verbose=0)
    codes   = encoder.predict(norm, batch_size=BATCH_SIZE, verbose=0)
    rmse_flat = np.sqrt(np.mean((norm - recons) ** 2, axis=1))
    return rmse_flat.reshape(H, W), codes.reshape(H, W, encoder.output_shape[-1])


# ═══════════════════════════════════════════════════════════════════════════════
# FIXED-POINT EXPORT  (inspect_ptq.py compatible)
# ═══════════════════════════════════════════════════════════════════════════════

EXPORT_DIR = 'fixed_export'

def quantise_multiplier(M, n_bits=31):
    if M == 0.0:
        return 0, 0
    import math
    exponent   = math.floor(math.log2(M))
    mantissa   = M / (2 ** exponent)
    multiplier = int(round(mantissa * (2 ** n_bits)))
    while multiplier >= (2 ** n_bits):
        multiplier //= 2
        exponent   += 1
    shift = n_bits - exponent
    assert 0 < multiplier < (2 ** n_bits)
    return multiplier, shift


def export_scaler_rom(out_dir, scaler, n_bands, scene):
    hpp_path = os.path.join(out_dir, 'scaler_rom.hpp')
    mean  = scaler.mean_.astype(np.float64)
    scale = scaler.scale_.astype(np.float64)

    def fmt_array(name, values, n):
        lines = [f'static const float {name}[{n}] = {{']
        per_row = 4
        for i in range(0, n, per_row):
            chunk = values[i:i + per_row]
            row   = ',  '.join(f'{v:.9e}f' for v in chunk)
            comma = ',' if i + per_row < n else ''
            lines.append(f'    {row}{comma}')
        lines.append('};')
        return '\n'.join(lines)

    with open(hpp_path, 'w') as f:
        f.write(f'/*\n')
        f.write(f' * scaler_rom.hpp  --  StandardScaler parameters for {scene}\n')
        f.write(f' * Generated by mawdbn_pca.py  --  DO NOT EDIT\n')
        f.write(f' * n_bands = {n_bands}  (PCA-reduced to {N_PCA_BANDS})\n')
        f.write(f' */\n\n')
        f.write(f'#pragma once\n\n')
        f.write(f'#define SCALER_N_BANDS  {n_bands}\n\n')
        f.write(fmt_array('SCALER_MEAN',  mean,  n_bands))
        f.write('\n\n')
        f.write(fmt_array('SCALER_SCALE', scale, n_bands))
        f.write('\n')


def export_fixed_point(scene, tflite_path, scaler, image_pca, gt_mask, pca):
    """
    Export everything inspect_ptq.py would produce, but driven directly
    from the trained model here rather than as a separate script run.
    Writes to fixed_export/<scene>/.
    """
    import json

    # Load quant params from the TFLite flatbuffer
    try:
        interp = tf.lite.Interpreter(
            model_path=tflite_path,
            experimental_op_resolver_type=
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    except Exception:
        interp = tf.lite.Interpreter(model_path=tflite_path)

    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    interp.set_tensor(inp_det['index'],
                      np.zeros([1] + list(inp_det['shape'][1:]), dtype=np.int8))
    interp.invoke()

    tidx   = {t['name']: t['index'] for t in interp.get_tensor_details()}
    tquant = {t['name']: t['quantization_parameters']
              for t in interp.get_tensor_details()}

    def get(name):
        raw = interp.get_tensor(tidx[name]).copy()
        q   = tquant[name]
        sc  = float(q['scales'][0])    if len(q['scales'])      > 0 else 0.0
        zp  = int(q['zero_points'][0]) if len(q['zero_points']) > 0 else 0
        return raw, sc, zp

    def find(must_have, must_not=None):
        must_not = must_not or []
        return next(n for n in tidx
                    if all(m in n for m in must_have)
                    and all(x not in n for x in must_not))

    enc_w_name   = find(['encoder_dense', 'MatMul'],  ['BiasAdd', ';'])
    enc_b_name   = find(['encoder_dense', 'BiasAdd'], ['Sigmoid', ';'])
    enc_act_name = find(['encoder_dense', 'MatMul', 'BiasAdd'])
    sig_name     = find(['encoder_dense', 'Sigmoid'])
    dec_w_name   = find(['decoder_dense', 'MatMul'],  ['BiasAdd', ';'])
    dec_b_name   = find(['decoder_dense', 'BiasAdd'], [';'])

    raw_enc_w, enc_w_sc, _            = get(enc_w_name)
    raw_enc_b, _,        _            = get(enc_b_name)
    raw_dec_w, dec_w_sc, _            = get(dec_w_name)
    raw_dec_b, _,        _            = get(dec_b_name)
    _,         enc_act_sc, enc_act_zp = get(enc_act_name)
    _,         sig_sc,     sig_zp     = get(sig_name)

    out_det = interp.get_output_details()[0]
    inp_q   = inp_det['quantization_parameters']
    out_q   = out_det['quantization_parameters']

    input_scale  = float(inp_q['scales'][0])
    input_zp     = int(inp_q['zero_points'][0])
    output_scale = float(out_q['scales'][0])
    output_zp    = int(out_q['zero_points'][0])

    enc_w = raw_enc_w.astype(np.int8)
    enc_b = raw_enc_b.flatten().astype(np.int32)
    dec_w = raw_dec_w.astype(np.int8)
    dec_b = raw_dec_b.flatten().astype(np.int32)

    n_hidden = len(enc_b)
    n_input  = len(dec_b)

    M1 = (input_scale * enc_w_sc) / enc_act_sc
    M2 = (sig_sc      * dec_w_sc) / output_scale
    M1_mult, M1_shift = quantise_multiplier(M1)
    M2_mult, M2_shift = quantise_multiplier(M2)

    # Sigmoid LUT
    sigmoid_lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        x = (float(i - 128) - enc_act_zp) * enc_act_sc
        x = max(-30.0, min(30.0, x))
        q = int(round((1.0 / (1.0 + np.exp(-x))) / sig_sc)) + sig_zp
        sigmoid_lut[i] = np.int8(max(-128, min(127, q)))

    H, W, C = image_pca.shape
    N       = H * W
    flat    = image_pca.reshape(-1, C).astype(np.float32)

    out_dir = os.path.join(EXPORT_DIR, scene)
    val_dir = os.path.join(out_dir, 'validation')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Model files
    enc_w.tofile(                                    os.path.join(out_dir, 'enc_w.bin'))
    enc_b.tofile(                                    os.path.join(out_dir, 'enc_b.bin'))
    dec_w.tofile(                                    os.path.join(out_dir, 'dec_w.bin'))
    dec_b.tofile(                                    os.path.join(out_dir, 'dec_b.bin'))
    sigmoid_lut.tofile(                              os.path.join(out_dir, 'sigmoid_lut.bin'))
    scaler.mean_.astype(np.float32).tofile(          os.path.join(out_dir, 'scaler_mean.bin'))
    scaler.scale_.astype(np.float32).tofile(         os.path.join(out_dir, 'scaler_scale.bin'))

    # BIP image — float32 [H*W, 120], ready for direct C kernel loading
    flat.tofile(os.path.join(out_dir, 'image_pca120.bip'))

    # PCA components
    if pca is not None:
        np.savez(os.path.join(out_dir, 'pca_components.npz'),
                 components=pca.components_.astype(np.float32),
                 mean=pca.mean_.astype(np.float32),
                 explained_variance=pca.explained_variance_.astype(np.float32),
                 explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32))

    # Scaler ROM header
    export_scaler_rom(out_dir, scaler, C, scene)

    # Config JSON — flat, no nesting
    config = {
        "scene":             scene,
        "height":            H,
        "width":             W,
        "n_bands":           C,
        "n_pixels":          N,
        "n_input":           n_input,
        "n_hidden":          n_hidden,
        "input_scale_f32":   input_scale,
        "input_zp":          input_zp,
        "enc_mult":          M1_mult,
        "enc_shift":         M1_shift,
        "enc_act_zp":        enc_act_zp,
        "sigmoid_zp":        sig_zp,
        "dec_mult":          M2_mult,
        "dec_shift":         M2_shift,
        "output_scale_f32":  output_scale,
        "output_zp":         output_zp,
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Validation data
    norm = scaler.transform(flat).astype(np.float32)
    flat.tofile(os.path.join(val_dir, 'image_raw.bin'))
    norm.tofile(os.path.join(val_dir, 'image_norm.bin'))

    gt_flat = gt_mask.reshape(-1).astype(np.uint8) if gt_mask is not None \
              else np.zeros(N, dtype=np.uint8)
    gt_flat.tofile(os.path.join(val_dir, 'gt_mask.bin'))

    # T3 RMSE scores using integer arithmetic
    half1 = np.int64(1) << np.int64(M1_shift - 1)
    half2 = np.int64(1) << np.int64(M2_shift - 1)
    inp_q_all = np.clip(
        np.round(norm / input_scale).astype(np.int64) + input_zp,
        -128, 127).astype(np.int8)
    enc_acc = (inp_q_all.astype(np.int64) - input_zp) \
              @ enc_w.T.astype(np.int64) + enc_b.astype(np.int64)
    enc_req = np.clip(
        ((enc_acc * np.int64(M1_mult) + half1) >> np.int64(M1_shift)) + enc_act_zp,
        -128, 127).astype(np.int8)
    hidden  = sigmoid_lut[enc_req.astype(np.int32) + 128].astype(np.int8)
    dec_acc = (hidden.astype(np.int64) - sig_zp) \
              @ dec_w.T.astype(np.int64) + dec_b.astype(np.int64)
    out_q   = np.clip(
        ((dec_acc * np.int64(M2_mult) + half2) >> np.int64(M2_shift)) + output_zp,
        -128, 127).astype(np.int8)
    out_f   = (out_q.astype(np.float32) - output_zp) * output_scale
    rmse_t3 = np.sqrt(np.mean((norm - out_f) ** 2, axis=1)).astype(np.float32)
    rmse_t3.tofile(os.path.join(val_dir, 'rmse_t3.bin'))

    auc_val = None
    if gt_mask is not None and gt_flat.sum() > 0:
        auc_val = float(roc_auc_score(gt_flat.astype(int), rmse_t3))
        print(f"  Export T3 AUC: {auc_val:.6f}")

    print(f"  Exported to {out_dir}/")
    return auc_val


# ═══════════════════════════════════════════════════════════════════════════════
# DSW DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

DSW_WINNER = 1
DSW_WOUTER = 9
DSW_PF     = 0

def compute_beta_map(R, C_codes):
    half_in  = DSW_WINNER // 2
    half_out = DSW_WOUTER // 2
    H, W  = R.shape
    beta  = np.zeros((H, W), dtype=np.float32)
    for j in tqdm(range(H), desc="  DSW", leave=False):
        r0 = max(j - half_out, 0); r1 = min(j + half_out + 1, H)
        for i in range(W):
            c0 = max(i - half_out, 0); c1 = min(i + half_out + 1, W)
            win_R = R[r0:r1, c0:c1]
            win_C = C_codes[r0:r1, c0:c1]
            cj, ci = j - r0, i - c0
            win_h, win_w = win_R.shape
            ri  = np.arange(win_h)[:, None]
            ci_ = np.arange(win_w)[None, :]
            ring = (ri != cj) | (ci_ != ci)
            rn = win_R[ring]
            if rn.size == 0:
                continue
            mu_n  = rn.mean(); sig_n = rn.std()
            inlier = np.abs(rn - mu_n) <= sig_n
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
    y_true  = gt_mask.reshape(-1).astype(int)
    y_score = score_map.reshape(-1)
    auc     = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc, fpr, tpr

def threshold_map(score_map, percentile=THRESHOLD_PERCENTILE):
    return (score_map > np.percentile(score_map, percentile)).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# PER-SCENE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_scene(scene_name, mat_path):
    set_seed(RANDOM_SEED)
    save_dir = os.path.join(RESULTS_DIR, 'mawdbn_fixed', scene_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n-- {scene_name.upper()} {'─'*50}")

    # ── Load + PCA ────────────────────────────────────────────────────────────
    image, gt_mask = load_abu_mat(mat_path)
    H_orig, W_orig, C_orig = image.shape
    print(f"  Original: {H_orig}x{W_orig}x{C_orig}")

    image_pca, pca = apply_pca(image, n_components=N_PCA_BANDS)
    H, W, C = image_pca.shape
    print(f"  PCA:      {H}x{W}x{C}")

    # ── Training data (on PCA-reduced image) ──────────────────────────────────
    n_train  = min(N_SAMPLES_TRAIN, H * W)
    rng      = np.random.default_rng(RANDOM_SEED)
    idx      = rng.choice(H * W, size=n_train, replace=False)
    train_px = image_pca.reshape(-1, C)[idx]
    norm_px, scaler = preprocess_pixels(train_px, fit_scaler=True)
    print(f"  Training on {n_train:,} px  |  {C} bands")

    # ── RBM ───────────────────────────────────────────────────────────────────
    rbm = GaussianRBM_NP(n_visible=C, n_hidden=N_HIDDEN, k=RBM_K)
    rbm.train(norm_px, n_epochs=RBM_EPOCHS, lr=RBM_LEARNING_RATE,
              momentum=RBM_MOMENTUM, weight_decay=RBM_WEIGHT_DECAY,
              batch_size=BATCH_SIZE)

    # ── Float32 model ─────────────────────────────────────────────────────────
    ae_none = build_autoencoder(C, N_HIDDEN, mode='none')
    initialize_from_rbm(ae_none, rbm)
    train_autoencoder(ae_none, norm_px, save_dir, f'mawdbn_none_{scene_name}')
    enc_none = build_encoder_only(ae_none, mode='none')
    rmse_none, codes_none = compute_recon_and_codes(ae_none, enc_none, image_pca, scaler)

    # ── Quantised model ───────────────────────────────────────────────────────
    set_seed(RANDOM_SEED)
    ae_quant = build_autoencoder(C, N_HIDDEN, mode=QUANTIZATION_MODE)
    initialize_from_rbm(ae_quant, rbm)
    train_autoencoder(ae_quant, norm_px, save_dir, f'mawdbn_{QUANTIZATION_MODE}_{scene_name}')

    tflite_path = export_tflite_ptq(ae_quant, norm_px[:200], save_dir,
                                    f'mawdbn_{QUANTIZATION_MODE}_{scene_name}')
    export_tflite_encoder(ae_quant, norm_px[:200], save_dir,
                          f'mawdbn_{QUANTIZATION_MODE}_{scene_name}',
                          mode=QUANTIZATION_MODE)
    rmse_quant, codes_quant = compute_recon_and_codes_tflite(
        tflite_path, N_HIDDEN, image_pca, scaler)

    # ── Fixed-point export ────────────────────────────────────────────────────
    export_fixed_point(scene_name, tflite_path, scaler, image_pca, gt_mask, pca)

    # ── DSW ───────────────────────────────────────────────────────────────────
    beta_none  = compute_beta_map(rmse_none,  codes_none)
    beta_quant = compute_beta_map(rmse_quant, codes_quant)

    # ── AUC ───────────────────────────────────────────────────────────────────
    result = {'scene': scene_name}
    if gt_mask is not None:
        auc_rn, fpr_rn, tpr_rn = compute_auc_roc(rmse_none,  gt_mask)
        auc_bn, fpr_bn, tpr_bn = compute_auc_roc(beta_none,  gt_mask)
        auc_rq, fpr_rq, tpr_rq = compute_auc_roc(rmse_quant, gt_mask)
        auc_bq, fpr_bq, tpr_bq = compute_auc_roc(beta_quant, gt_mask)

        print(f"  RMSE  f32={auc_rn:.4f}  qt={auc_rq:.4f}  d={auc_rq-auc_rn:+.4f}"
              f"  |  b  f32={auc_bn:.4f}  qt={auc_bq:.4f}  d={auc_bq-auc_bn:+.4f}")

        result.update({
            'auc_rmse_none':  auc_rn, 'fpr_rmse_none':  fpr_rn, 'tpr_rmse_none':  tpr_rn,
            'auc_rmse_quant': auc_rq, 'fpr_rmse_quant': fpr_rq, 'tpr_rmse_quant': tpr_rq,
            'auc_beta_none':  auc_bn, 'fpr_beta_none':  fpr_bn, 'tpr_beta_none':  tpr_bn,
            'auc_beta_quant': auc_bq, 'fpr_beta_quant': fpr_bq, 'tpr_beta_quant': tpr_bq,
            'image': image_pca, 'gt_mask': gt_mask,
            'rmse_none': rmse_none, 'beta_none': beta_none,
            'rmse_quant': rmse_quant, 'beta_quant': beta_quant,
        })
    else:
        print("  No ground truth — AUC skipped.")

    result['codes_none']  = codes_none
    result['codes_quant'] = codes_quant
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
    qm  = QUANTIZATION_MODE
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, fpr_k, tpr_k, auc_k, col, ls in [
        ('RMSE  float32',  'fpr_rmse_none',  'tpr_rmse_none',  'auc_rmse_none',  '#1f77b4', '-'),
        (f'RMSE  {qm}',    'fpr_rmse_quant', 'tpr_rmse_quant', 'auc_rmse_quant', '#aec7e8', '--'),
        ('beta  float32',  'fpr_beta_none',  'tpr_beta_none',  'auc_beta_none',  '#d62728', '-'),
        (f'beta  {qm}',    'fpr_beta_quant', 'tpr_beta_quant', 'auc_beta_quant', '#f4a460', '--'),
    ]:
        ax.plot(r[fpr_k], r[tpr_k], lw=2, color=col, ls=ls,
                label=f"{label}  AUC={r[auc_k]:.4f}")
    ax.plot([0,1],[0,1],'k:',lw=0.8)
    ax.set_xlim(0,1); ax.set_ylim(0,1.01)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate',  fontsize=11)
    ax.set_title(f"ROC -- {r['scene']}  (PCA {N_PCA_BANDS} bands)", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_scene_comparison(r, save_path):
    qm    = QUANTIZATION_MODE
    image = r['image']; gt = r['gt_mask']
    rgb   = _false_colour(image)
    thresh_rn = threshold_map(r['rmse_none'])
    thresh_rq = threshold_map(r['rmse_quant'])
    thresh_bn = threshold_map(r['beta_none'])
    thresh_bq = threshold_map(r['beta_quant'])
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle(f"float32 vs {qm.upper()}  |  {r['scene']}  (PCA {N_PCA_BANDS} bands)\n"
                 "red=detected  blue=missed GT", fontsize=13, fontweight='bold')
    def show(ax, img, title, cmap='viridis'):
        ax.imshow(img, cmap=cmap); ax.set_title(title, fontsize=10); ax.axis('off')
    show(axes[0,0], rgb, 'False colour')
    if gt is not None:
        show(axes[0,1], gt, f'GT ({int(gt.sum()):,} px)', cmap='gray')
    axes[0,2].axis('off'); axes[0,3].axis('off')
    show(axes[1,0], r['rmse_none'],  'RMSE float32', cmap='hot')
    show(axes[1,1], r['rmse_quant'], f'RMSE {qm}',   cmap='hot')
    show(axes[1,2], _overlay(rgb, thresh_rn, gt), 'RMSE overlay f32')
    show(axes[1,3], _overlay(rgb, thresh_rq, gt), f'RMSE overlay {qm}')
    show(axes[2,0], r['beta_none'],  'beta float32', cmap='hot')
    show(axes[2,1], r['beta_quant'], f'beta {qm}',   cmap='hot')
    show(axes[2,2], _overlay(rgb, thresh_bn, gt), 'beta overlay f32')
    show(axes[2,3], _overlay(rgb, thresh_bq, gt), f'beta overlay {qm}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_auc_barchart(results, save_path):
    valid = [r for r in results if 'auc_beta_none' in r]
    if not valid: return
    qm     = QUANTIZATION_MODE
    scenes = [r['scene'] for r in valid]
    bn     = [r['auc_beta_none']  for r in valid]
    bq     = [r['auc_beta_quant'] for r in valid]
    n = len(scenes); y = np.arange(n); h = 0.35
    type_colour = {'airport': '#4878cf', 'beach': '#f58231', 'urban': '#3cb44b'}
    bar_colours = [type_colour.get(next((t for t in type_colour if t in s), 'urban'), '#888')
                   for s in scenes]
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.55)))
    bars_n = ax.barh(y + h/2, bn, h, color=bar_colours, alpha=0.85, label='float32')
    bars_q = ax.barh(y - h/2, bq, h, color=bar_colours, alpha=0.45,
                     hatch='///', label=qm, edgecolor='white', linewidth=0.4)
    for bar, v in zip(bars_n, bn):
        ax.text(max(v-0.001, 0.84), bar.get_y()+bar.get_height()/2,
                f'{v:.4f}', va='center', ha='right', fontsize=7.5, color='white', fontweight='bold')
    for bar, v in zip(bars_q, bq):
        ax.text(max(v-0.001, 0.84), bar.get_y()+bar.get_height()/2,
                f'{v:.4f}', va='center', ha='right', fontsize=7.5, color='white', fontweight='bold')
    ax.axvline(np.mean(bn), color='black',   lw=1.2, ls=':', label=f'f32 avg {np.mean(bn):.4f}')
    ax.axvline(np.mean(bq), color='dimgray', lw=1.2, ls=':', label=f'{qm} avg {np.mean(bq):.4f}')
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace('abu-','') for s in scenes], fontsize=10)
    ax.set_xlabel('AUC-ROC (beta score)', fontsize=11)
    ax.set_xlim(min(min(bn), min(bq)) - 0.02, 1.005)
    ax.set_title(f'MAWDBN beta AUC  |  float32 vs {qm.upper()}  |  PCA {N_PCA_BANDS} bands',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
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
    qm  = QUANTIZATION_MODE
    W   = 22
    div = "  " + "-" * 88
    bdr = "=" * 92
    def fmt(v):    return f"{v:.4f}" if v is not None else "  N/A"
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
    pr(f"  MAWDBN AUC SUMMARY  |  float32 vs {qm.upper()}  |  PCA {N_PCA_BANDS} bands")
    pr(bdr)
    pr(f"  {'Scene':<{W}}  {'RMSE-f32':>8}  {'RMSE-qt':>8}  {'dRMSE':>8}  "
       f"{'b-f32':>8}  {'b-qt':>8}  {'db':>8}")
    pr(div)
    for tname, grp in types.items():
        if not grp: continue
        for r in grp:
            pr(row(r['scene'],
                   r['auc_rmse_none'], r['auc_rmse_quant'],
                   r['auc_beta_none'], r['auc_beta_quant']))
        pr(div)
        avg_rn = gavg(grp,'auc_rmse_none'); avg_rq = gavg(grp,'auc_rmse_quant')
        avg_bn = gavg(grp,'auc_beta_none'); avg_bq = gavg(grp,'auc_beta_quant')
        pr(row(f"  AVG {tname} ({len(grp)})", avg_rn, avg_rq, avg_bn, avg_bq, indent=""))
        pr(div)
    all_rn = gavg(valid,'auc_rmse_none'); all_rq = gavg(valid,'auc_rmse_quant')
    all_bn = gavg(valid,'auc_beta_none'); all_bq = gavg(valid,'auc_beta_quant')
    pr(row(f"  OVERALL ({len(valid)} scenes)", all_rn, all_rq, all_bn, all_bq, indent=""))
    pr(bdr)
    out_dir = os.path.join(RESULTS_DIR, 'mawdbn_fixed')
    os.makedirs(out_dir, exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'summary_pca{N_PCA_BANDS}_{qm}_{ts}.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Summary saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results = []
    for scene in ABU_SCENES:
        mat = os.path.join(ABU_DATASET_DIR, f'{scene}.mat')
        if not os.path.exists(mat):
            print(f"  {scene}: .mat not found -- skipping.")
            continue
        try:
            results.append(run_scene(scene, mat))
        except Exception as e:
            import traceback
            print(f"  {scene}: FAILED -- {e}")
            traceback.print_exc()

    if results:
        out_dir = os.path.join(RESULTS_DIR, 'mawdbn_fixed')
        os.makedirs(out_dir, exist_ok=True)
        print_summary(results)
        plot_auc_barchart(
            results,
            save_path=os.path.join(out_dir, f'auc_barchart_pca{N_PCA_BANDS}_{QUANTIZATION_MODE}.png'))

    print(f"\nDONE -- {len(results)} scene(s) processed.")