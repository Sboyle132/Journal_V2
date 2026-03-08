# %% [markdown]
"""
inspect_ptq.py — Three-Track PTQ Validation for MAWDBN DBN (PCA variant)

Loads PCA-reduced image and scaler directly from fixed_export/<scene>/
produced by mawdbn_pca.py. Does NOT reload the raw .mat file or refit
the scaler — everything is read from the already-exported files so that
validation uses exactly the same data the C kernel will see.

Track 1 — TFLite black-box:   invoke() only. Ground truth.
Track 2 — QuantisedDBN class:  Extracts weights/params from TFLite flatbuffer.
Track 3 — Manual reimpl:       Standalone functions, independent reimplementation.

Checks:
  T1 == T2  ->  class correctly reproduces TFLite inference at every stage
  T2 == T3  ->  standalone reimplementation matches the class exactly

RMSE convention: normalised space (norm - dequantised_recon).
  After PCA + StandardScaler each band has unit variance, so normalised
  RMSE weights all PCA components equally. Raw DN RMSE is dominated by
  high-variance components and degrades anomaly detection performance.

USAGE:
  RUN_MODE = 'single'  — run SCENE_INDEX only
  RUN_MODE = 'all'     — run all ABU scenes in sequence
"""

RUN_MODE    = 'all'
SCENE_INDEX = 0

RESULTS_DIR     = 'results/mawdbn_fixed'
EXPORT_DIR      = 'fixed_export'
N_SAMPLES_TRAIN = 50000
RANDOM_SEED     = 42

ABU_SCENES = [
    'abu-airport-1', 'abu-airport-2', 'abu-airport-3', 'abu-airport-4',
    'abu-beach-1',   'abu-beach-2',   'abu-beach-3',   'abu-beach-4',
    'abu-urban-1',   'abu-urban-2',   'abu-urban-3',   'abu-urban-4',
    'abu-urban-5',
]

import os, sys, random, json
from collections import OrderedDict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def set_seed(s=RANDOM_SEED):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)

# ══════════════════════════════════════════════════════════════════════════════
# Data helpers — load from fixed_export, not from raw .mat
# ══════════════════════════════════════════════════════════════════════════════

def load_scene_data(scene):
    """
    Load PCA-reduced image, scaler, GT mask, and config from
    fixed_export/<scene>/ produced by mawdbn_pca.py.

    Returns:
        image_pca  float32 [H, W, C]   PCA-reduced image (120 bands)
        flat       float32 [N, C]      same, flattened
        scaler     StandardScaler      reconstructed from saved mean/scale
        gt_mask    uint8  [H, W] or None
        config     dict               parsed config.json
        H, W, C    int
    """
    export_dir = os.path.join(EXPORT_DIR, scene)

    cfg_path = os.path.join(export_dir, 'config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found: {cfg_path}\n"
                                f"Run mawdbn_pca.py first.")
    with open(cfg_path) as f:
        config = json.load(f)

    H = config['height']
    W = config['width']
    C = config['n_input']
    N = H * W

    bip_path = os.path.join(export_dir, 'image_pca120.bip')
    if not os.path.exists(bip_path):
        raise FileNotFoundError(f"image_pca120.bip not found: {bip_path}")
    flat = np.fromfile(bip_path, dtype=np.float32).reshape(N, C)
    image_pca = flat.reshape(H, W, C)

    mean_path  = os.path.join(export_dir, 'scaler_mean.bin')
    scale_path = os.path.join(export_dir, 'scaler_scale.bin')
    scaler = StandardScaler()
    scaler.mean_  = np.fromfile(mean_path,  dtype=np.float32).astype(np.float64)
    scaler.scale_ = np.fromfile(scale_path, dtype=np.float32).astype(np.float64)
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = C
    scaler.n_samples_seen_ = N_SAMPLES_TRAIN

    gt_path = os.path.join(export_dir, 'validation', 'gt_mask.bin')
    if os.path.exists(gt_path):
        gt_flat = np.fromfile(gt_path, dtype=np.uint8)
        gt_mask = gt_flat.reshape(H, W) if gt_flat.sum() > 0 else None
    else:
        gt_mask = None

    print(f"  Loaded {scene}: {H}x{W}x{C}  GT: {'yes' if gt_mask is not None else 'none'}")
    return image_pca, flat, scaler, gt_mask, config, H, W, C


def find_tflite(scene):
    p = os.path.join(RESULTS_DIR, scene,
                     f'mawdbn_ptq_{scene}_ptq_int8.tflite')
    if not os.path.exists(p):
        raise FileNotFoundError(f"PTQ TFLite not found: {p}\n"
                                f"Run mawdbn_pca.py first.")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# Track 1 — TFLite black-box
# ══════════════════════════════════════════════════════════════════════════════

class TFLiteModel:
    def __init__(self, tflite_path):
        self.interp = tf.lite.Interpreter(model_path=tflite_path)
        self.interp.allocate_tensors()
        self.inp_det = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]
        self.input_scale  = float(self.inp_det['quantization_parameters']['scales'][0])
        self.input_zp     = int(self.inp_det['quantization_parameters']['zero_points'][0])
        self.output_scale = float(self.out_det['quantization_parameters']['scales'][0])
        self.output_zp    = int(self.out_det['quantization_parameters']['zero_points'][0])

    def __call__(self, pixel_norm):
        q = np.clip(
            np.round(pixel_norm / self.input_scale) + self.input_zp,
            -128, 127).astype(np.int8)
        self.interp.set_tensor(self.inp_det['index'], q[np.newaxis])
        self.interp.invoke()
        out = self.interp.get_tensor(self.out_det['index'])[0]
        return (out.astype(np.float32) - self.output_zp) * self.output_scale

    def batch(self, norm, flat, scaler):
        """RMSE in normalised space — norm vs dequantised reconstruction."""
        recon = np.zeros_like(norm)
        for i in range(len(norm)):
            recon[i] = self(norm[i])
        return np.sqrt(np.mean((norm - recon) ** 2, axis=1))


# ══════════════════════════════════════════════════════════════════════════════
# Fixed-point multiplier
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Track 2 — QuantisedDBN class
# ══════════════════════════════════════════════════════════════════════════════

class QuantisedDBN:
    def __init__(self, tflite_path):
        self._load_params(tflite_path)
        self._build_sigmoid_lut()

    def _load_params(self, tflite_path):
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

        enc_w_name   = find(['encoder_dense', 'MatMul'],             ['BiasAdd', ';'])
        enc_b_name   = find(['encoder_dense', 'BiasAdd'],            ['Sigmoid', ';'])
        enc_act_name = find(['encoder_dense', 'MatMul', 'BiasAdd'])
        sig_name     = find(['encoder_dense', 'Sigmoid'])
        dec_w_name   = find(['decoder_dense', 'MatMul'],             ['BiasAdd', ';'])
        dec_b_name   = find(['decoder_dense', 'BiasAdd'],            [';'])

        raw_enc_w, enc_w_sc, _            = get(enc_w_name)
        raw_enc_b, enc_b_sc, _            = get(enc_b_name)
        raw_dec_w, dec_w_sc, _            = get(dec_w_name)
        raw_dec_b, dec_b_sc, _            = get(dec_b_name)
        _,         enc_act_sc, enc_act_zp = get(enc_act_name)
        _,         sig_sc,     sig_zp     = get(sig_name)

        out_det = interp.get_output_details()[0]
        inp_q   = inp_det['quantization_parameters']
        out_q   = out_det['quantization_parameters']

        self.input_scale   = float(inp_q['scales'][0])
        self.input_zp      = int(inp_q['zero_points'][0])
        self.enc_w_scale   = enc_w_sc
        self.enc_b_scale   = enc_b_sc
        self.enc_act_scale = enc_act_sc
        self.enc_act_zp    = enc_act_zp
        self.sigmoid_scale = sig_sc
        self.sigmoid_zp    = sig_zp
        self.dec_w_scale   = dec_w_sc
        self.dec_b_scale   = dec_b_sc
        self.output_scale  = float(out_q['scales'][0])
        self.output_zp     = int(out_q['zero_points'][0])

        self.enc_w = raw_enc_w.astype(np.int8)
        self.enc_b = raw_enc_b.flatten().astype(np.int32)
        self.dec_w = raw_dec_w.astype(np.int8)
        self.dec_b = raw_dec_b.flatten().astype(np.int32)

        self.n_hidden = len(self.enc_b)
        self.n_input  = len(self.dec_b)

        self.M1 = (self.input_scale   * self.enc_w_scale) / self.enc_act_scale
        self.M2 = (self.sigmoid_scale * self.dec_w_scale) / self.output_scale
        self.M1_mult, self.M1_shift = quantise_multiplier(self.M1)
        self.M2_mult, self.M2_shift = quantise_multiplier(self.M2)

    def _build_sigmoid_lut(self):
        lut = np.zeros(256, dtype=np.int8)
        for i in range(256):
            x = (float(i - 128) - self.enc_act_zp) * self.enc_act_scale
            x = max(-30.0, min(30.0, x))
            q = int(round((1.0 / (1.0 + np.exp(-x))) / self.sigmoid_scale)) + self.sigmoid_zp
            lut[i] = np.int8(max(-128, min(127, q)))
        self.sigmoid_lut = lut

    def quantise_input(self, pixel_norm):
        q = np.round(pixel_norm / self.input_scale).astype(np.int64) + self.input_zp
        return np.clip(q, -128, 127).astype(np.int8)

    def encoder_matmul(self, inp_q):
        acc = self.enc_b.copy().astype(np.int64)
        acc += self.enc_w.astype(np.int64) @ (inp_q.astype(np.int64) - self.input_zp)
        return acc.astype(np.int32)

    def encoder_requantise(self, enc_acc):
        q = np.round(enc_acc.astype(np.float64) * self.M1).astype(np.int64) + self.enc_act_zp
        return np.clip(q, -128, 127).astype(np.int8)

    def sigmoid(self, enc_req):
        return self.sigmoid_lut[enc_req.astype(np.int32) + 128].astype(np.int8)

    def decoder_matmul(self, hidden):
        acc = self.dec_b.copy().astype(np.int64)
        acc += self.dec_w.astype(np.int64) @ (hidden.astype(np.int64) - self.sigmoid_zp)
        return acc.astype(np.int32)

    def decoder_requantise(self, dec_acc):
        q = np.round(dec_acc.astype(np.float64) * self.M2).astype(np.int64) + self.output_zp
        return np.clip(q, -128, 127).astype(np.int8)

    def dequantise_output(self, out_int8):
        return (out_int8.astype(np.float32) - self.output_zp) * self.output_scale

    def forward(self, pixel_norm):
        s = OrderedDict()
        s['input_int8'] = self.quantise_input(pixel_norm)
        s['enc_acc']    = self.encoder_matmul(s['input_int8'])
        s['enc_req']    = self.encoder_requantise(s['enc_acc'])
        s['hidden']     = self.sigmoid(s['enc_req'])
        s['dec_acc']    = self.decoder_matmul(s['hidden'])
        s['out_int8']   = self.decoder_requantise(s['dec_acc'])
        s['out_f32']    = self.dequantise_output(s['out_int8'])
        return s

    def batch(self, norm, flat, scaler):
        """RMSE in normalised space — norm vs dequantised reconstruction."""
        inp_q   = np.clip(
            np.round(norm / self.input_scale).astype(np.int64) + self.input_zp,
            -128, 127).astype(np.int8)
        enc_acc = (inp_q.astype(np.int64) - self.input_zp) \
                  @ self.enc_w.T.astype(np.int64) + self.enc_b.astype(np.int64)
        enc_req = np.clip(
            np.round(enc_acc.astype(np.float64) * self.M1).astype(np.int64) + self.enc_act_zp,
            -128, 127).astype(np.int8)
        hidden  = self.sigmoid_lut[enc_req.astype(np.int32) + 128].astype(np.int8)
        dec_acc = (hidden.astype(np.int64) - self.sigmoid_zp) \
                  @ self.dec_w.T.astype(np.int64) + self.dec_b.astype(np.int64)
        out_q   = np.clip(
            np.round(dec_acc.astype(np.float64) * self.M2).astype(np.int64) + self.output_zp,
            -128, 127).astype(np.int8)
        out_f   = (out_q.astype(np.float32) - self.output_zp) * self.output_scale
        return np.sqrt(np.mean((norm - out_f) ** 2, axis=1))


# ══════════════════════════════════════════════════════════════════════════════
# Track 3 — Standalone reimplementation
# ══════════════════════════════════════════════════════════════════════════════

def t3_quantise_input(pixel_norm, input_scale, input_zp):
    q = np.round(pixel_norm / input_scale).astype(np.int64) + input_zp
    return np.clip(q, -128, 127).astype(np.int8)

def t3_encoder_matmul(inp_q, enc_w, enc_b, input_zp):
    acc = enc_b.copy().astype(np.int64)
    for i in range(len(enc_b)):
        for j in range(len(inp_q)):
            acc[i] += int(enc_w[i, j]) * (int(inp_q[j]) - input_zp)
    return acc.astype(np.int32)

def t3_encoder_requantise(enc_acc, M1_mult, M1_shift, enc_act_zp):
    half = np.int64(1) << np.int64(M1_shift - 1)
    q = (enc_acc.astype(np.int64) * np.int64(M1_mult) + half) >> np.int64(M1_shift)
    return np.clip(q + enc_act_zp, -128, 127).astype(np.int8)

def t3_sigmoid(enc_req, sigmoid_lut, enc_act_zp):
    return sigmoid_lut[enc_req.astype(np.int32) + 128].astype(np.int8)

def t3_decoder_matmul(hidden, dec_w, dec_b, sigmoid_zp):
    acc = dec_b.copy().astype(np.int64)
    for i in range(len(dec_b)):
        for j in range(len(hidden)):
            acc[i] += int(dec_w[i, j]) * (int(hidden[j]) - sigmoid_zp)
    return acc.astype(np.int32)

def t3_decoder_requantise(dec_acc, M2_mult, M2_shift, output_zp):
    half = np.int64(1) << np.int64(M2_shift - 1)
    q = (dec_acc.astype(np.int64) * np.int64(M2_mult) + half) >> np.int64(M2_shift)
    return np.clip(q + output_zp, -128, 127).astype(np.int8)

def t3_dequantise_output(out_int8, output_scale, output_zp):
    return (out_int8.astype(np.float32) - output_zp) * output_scale

def t3_forward(pixel_norm, model):
    s = OrderedDict()
    s['input_int8'] = t3_quantise_input(pixel_norm, model.input_scale, model.input_zp)
    s['enc_acc']    = t3_encoder_matmul(s['input_int8'], model.enc_w, model.enc_b, model.input_zp)
    s['enc_req']    = t3_encoder_requantise(s['enc_acc'], model.M1_mult, model.M1_shift, model.enc_act_zp)
    s['hidden']     = t3_sigmoid(s['enc_req'], model.sigmoid_lut, model.enc_act_zp)
    s['dec_acc']    = t3_decoder_matmul(s['hidden'], model.dec_w, model.dec_b, model.sigmoid_zp)
    s['out_int8']   = t3_decoder_requantise(s['dec_acc'], model.M2_mult, model.M2_shift, model.output_zp)
    s['out_f32']    = t3_dequantise_output(s['out_int8'], model.output_scale, model.output_zp)
    return s

def t3_batch(norm, flat, scaler, model):
    """RMSE in normalised space — norm vs dequantised reconstruction."""
    inp_q   = np.clip(
        np.round(norm / model.input_scale).astype(np.int64) + model.input_zp,
        -128, 127).astype(np.int8)
    enc_acc = (inp_q.astype(np.int64) - model.input_zp) \
              @ model.enc_w.T.astype(np.int64) + model.enc_b.astype(np.int64)
    half1   = np.int64(1) << np.int64(model.M1_shift - 1)
    enc_req = np.clip(
        ((enc_acc.astype(np.int64) * np.int64(model.M1_mult) + half1) >> np.int64(model.M1_shift))
        + model.enc_act_zp, -128, 127).astype(np.int8)
    hidden  = model.sigmoid_lut[enc_req.astype(np.int32) + 128].astype(np.int8)
    dec_acc = (hidden.astype(np.int64) - model.sigmoid_zp) \
              @ model.dec_w.T.astype(np.int64) + model.dec_b.astype(np.int64)
    half2   = np.int64(1) << np.int64(model.M2_shift - 1)
    out_q   = np.clip(
        ((dec_acc.astype(np.int64) * np.int64(model.M2_mult) + half2) >> np.int64(model.M2_shift))
        + model.output_zp, -128, 127).astype(np.int8)
    out_f   = (out_q.astype(np.float32) - model.output_zp) * model.output_scale
    return np.sqrt(np.mean((norm - out_f) ** 2, axis=1))


# ══════════════════════════════════════════════════════════════════════════════
# Comparison utilities
# ══════════════════════════════════════════════════════════════════════════════

def print_section(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)

def cmp_int(la, a, lb, b, n=8):
    d  = a.astype(np.int64) - b.astype(np.int64)
    mm = int(np.sum(d != 0))
    print(f"\n  -- {la} vs {lb}")
    print(f"     Mismatches={mm}/{len(a)}  MaxAbsDiff={int(np.abs(d).max())}")
    for i in range(min(n, len(a))):
        mk = " <-" if d[i] != 0 else ""
        print(f"     [{i:3d}]  {int(a[i]):6d}  {int(b[i]):6d}  diff={int(d[i]):+d}{mk}")
    return mm

def cmp_float(la, a, lb, b, n=8):
    d    = a.astype(np.float64) - b.astype(np.float64)
    rmse = float(np.sqrt(np.mean(d ** 2)))
    print(f"\n  -- {la} vs {lb}")
    print(f"     RMSE={rmse:.6e}  MaxAbsErr={float(np.abs(d).max()):.6e}")
    for i in range(min(n, len(a))):
        print(f"     [{i:3d}]  {float(a[i]):+14.6f}  {float(b[i]):+14.6f}  "
              f"diff={float(d[i]):+.6e}")
    return rmse

def arr_summary(a):
    return f"min={a.min():.4f}  max={a.max():.4f}  mean={a.mean():.4f}  std={a.std():.4f}"

def compute_auc(scores, gt):
    if gt is None or int(gt.sum()) == 0:
        return None
    return float(roc_auc_score(gt.reshape(-1).astype(int), scores.reshape(-1)))


# ══════════════════════════════════════════════════════════════════════════════
# Per-pixel intermediate export
# ══════════════════════════════════════════════════════════════════════════════

def export_pixel_intermediates(px_dir, label, px_idx, model, flat, scaler):
    """
    Run T3 on one pixel and write every intermediate to px_dir/.
    RMSE convention: normalised space (norm vs dequantised recon).
    rmse_raw is also computed and stored for reference but is not used
    by the C testbench.
    """
    os.makedirs(px_dir, exist_ok=True)

    px_orig = flat[px_idx].astype(np.float32)
    px_norm = scaler.transform(px_orig[np.newaxis])[0].astype(np.float32)

    inp_q   = t3_quantise_input(px_norm, model.input_scale, model.input_zp)
    enc_acc = t3_encoder_matmul(inp_q, model.enc_w, model.enc_b, model.input_zp)
    enc_req = t3_encoder_requantise(enc_acc, model.M1_mult, model.M1_shift, model.enc_act_zp)
    hidden  = t3_sigmoid(enc_req, model.sigmoid_lut, model.enc_act_zp)
    dec_acc = t3_decoder_matmul(hidden, model.dec_w, model.dec_b, model.sigmoid_zp)
    out_q   = t3_decoder_requantise(dec_acc, model.M2_mult, model.M2_shift, model.output_zp)
    out_f32 = t3_dequantise_output(out_q, model.output_scale, model.output_zp)

    rmse_norm = float(np.sqrt(np.mean((px_norm - out_f32) ** 2)))
    px_recon  = scaler.inverse_transform(out_f32[np.newaxis])[0].astype(np.float32)
    rmse_raw  = float(np.sqrt(np.mean((px_orig - px_recon) ** 2)))

    inp_q.astype(np.int8).tofile(      os.path.join(px_dir, 'inp_q.bin'))
    enc_acc.astype(np.int64).tofile(   os.path.join(px_dir, 'enc_acc.bin'))
    enc_req.astype(np.int8).tofile(    os.path.join(px_dir, 'enc_req.bin'))
    hidden.astype(np.int8).tofile(     os.path.join(px_dir, 'hidden.bin'))
    dec_acc.astype(np.int64).tofile(   os.path.join(px_dir, 'dec_acc.bin'))
    out_q.astype(np.int8).tofile(      os.path.join(px_dir, 'out_q.bin'))
    out_f32.astype(np.float32).tofile( os.path.join(px_dir, 'out_f32.bin'))
    np.array([rmse_norm], dtype=np.float32).tofile(os.path.join(px_dir, 'rmse_norm.bin'))
    np.array([rmse_raw],  dtype=np.float32).tofile(os.path.join(px_dir, 'rmse_raw.bin'))

    info = {
        "label":     label,
        "px_idx":    px_idx,
        "rmse_norm": rmse_norm,
        "rmse_raw":  rmse_raw,
        "shapes": {
            "inp_q":   {"dtype": "int8",    "shape": [model.n_input]},
            "enc_acc": {"dtype": "int64",   "shape": [model.n_hidden]},
            "enc_req": {"dtype": "int8",    "shape": [model.n_hidden]},
            "hidden":  {"dtype": "int8",    "shape": [model.n_hidden]},
            "dec_acc": {"dtype": "int64",   "shape": [model.n_input]},
            "out_q":   {"dtype": "int8",    "shape": [model.n_input]},
            "out_f32": {"dtype": "float32", "shape": [model.n_input]},
        }
    }
    with open(os.path.join(px_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"    {label:<12}  px={px_idx:5d}  "
          f"RMSE_norm={rmse_norm:.4f}  RMSE_raw={rmse_raw:.2f}")


def export_pixel_refs(scene, model, flat, scaler, gt_mask, H, W):
    """Write px_centre and px_anomaly into fixed_export/<scene>/validation/."""
    val_dir = os.path.join(EXPORT_DIR, scene, 'validation')
    os.makedirs(val_dir, exist_ok=True)
    N = H * W
    print("  Exporting pixel intermediates...")
    export_pixel_intermediates(
        os.path.join(val_dir, 'px_centre'), 'centre',
        N // 2, model, flat, scaler)
    if gt_mask is not None:
        gt_flat = gt_mask.reshape(-1)
        pos = np.argwhere(gt_flat)
        if len(pos) > 0:
            anomaly_idx = int(pos[len(pos) // 2][0])
            export_pixel_intermediates(
                os.path.join(val_dir, 'px_anomaly'), 'anomaly',
                anomaly_idx, model, flat, scaler)


def export_validation(scene, model, flat, scaler, gt_mask, H, W, C):
    """
    Write validation files into fixed_export/<scene>/validation/:
      image_raw.bin   float32 [N, n_input]   raw PCA-reduced pixels
      gt_mask.bin     uint8   [N]             anomaly ground truth
      rmse_t3.bin     float32 [N]             T3 RMSE in normalised space
    """
    val_dir = os.path.join(EXPORT_DIR, scene, 'validation')
    os.makedirs(val_dir, exist_ok=True)
    N = H * W

    flat_f32 = flat.astype(np.float32)
    norm_f32 = scaler.transform(flat_f32).astype(np.float32)

    flat_f32.tofile(os.path.join(val_dir, 'image_raw.bin'))

    if gt_mask is not None:
        gt_flat = gt_mask.reshape(-1).astype(np.uint8)
    else:
        gt_flat = np.zeros(N, dtype=np.uint8)
    gt_flat.tofile(os.path.join(val_dir, 'gt_mask.bin'))

    print("  Running T3 full-image pass for rmse_t3.bin...")
    rmse_all = t3_batch(norm_f32, flat_f32, scaler, model)
    rmse_all.astype(np.float32).tofile(os.path.join(val_dir, 'rmse_t3.bin'))

    if gt_flat.sum() > 0:
        auc_val = float(roc_auc_score(gt_flat.astype(int), rmse_all))
        print(f"  T3 AUC (normalised RMSE): {auc_val:.6f}")

    print(f"  Wrote image_raw.bin  gt_mask.bin  rmse_t3.bin  ->  {val_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    set_seed()
    scene = ABU_SCENES[SCENE_INDEX]
    print_section(f"PTQ THREE-TRACK VALIDATION  |  {scene}  (PCA variant)")

    image_pca, flat, scaler, gt_mask, config, H, W, C = load_scene_data(scene)
    norm = scaler.transform(flat).astype(np.float32)

    tflite_path = find_tflite(scene)
    print(f"  TFLite: {os.path.relpath(tflite_path)}")

    print_section("MODEL LOADING")
    t1_model = TFLiteModel(tflite_path)
    t2_model = QuantisedDBN(tflite_path)

    print(f"  N_INPUT={t2_model.n_input}  N_HIDDEN={t2_model.n_hidden}")
    print(f"  input_scale={t2_model.input_scale:.6e}  input_zp={t2_model.input_zp}")
    print(f"  M1={t2_model.M1:.6e}  M1_mult={t2_model.M1_mult}  M1_shift={t2_model.M1_shift}")
    print(f"  M2={t2_model.M2:.6e}  M2_mult={t2_model.M2_mult}  M2_shift={t2_model.M2_shift}")
    print(f"  sigmoid_zp={t2_model.sigmoid_zp}  output_scale={t2_model.output_scale:.6e}")

    # ── Export validation data ────────────────────────────────────────────────
    print_section("VALIDATION DATA EXPORT")
    export_validation(scene, t2_model, flat, scaler, gt_mask, H, W, C)

    # ── Export pixel intermediates ────────────────────────────────────────────
    print_section("PIXEL INTERMEDIATE EXPORT")
    export_pixel_refs(scene, t2_model, flat, scaler, gt_mask, H, W)

    # ── Single-pixel layer-by-layer ───────────────────────────────────────────
    test_pixels = [('Centre', H * W // 2)]
    if gt_mask is not None and gt_mask.sum() > 0:
        pos = np.argwhere(gt_mask.reshape(-1))
        test_pixels.append(('Anomaly', int(pos[len(pos) // 2][0])))

    for px_label, px_idx in test_pixels:
        print_section(f"LAYER-BY-LAYER -- {px_label} pixel #{px_idx}")
        px_orig = flat[px_idx].astype(np.float32)
        px_norm = scaler.transform(px_orig[np.newaxis])[0].astype(np.float32)

        t1_out  = t1_model(px_norm)
        t1_rmse = float(np.sqrt(np.mean((px_norm - t1_out) ** 2)))
        print(f"\n  Track 1 (TFLite)   RMSE_norm={t1_rmse:.6f}")

        t2_inp  = t2_model.quantise_input(px_norm)
        t2_eacc = t2_model.encoder_matmul(t2_inp)
        t2_ereq = t2_model.encoder_requantise(t2_eacc)
        t2_hid  = t2_model.sigmoid(t2_ereq)
        t2_dacc = t2_model.decoder_matmul(t2_hid)
        t2_out8 = t2_model.decoder_requantise(t2_dacc)
        t2_out  = t2_model.dequantise_output(t2_out8)
        t2_rmse = float(np.sqrt(np.mean((px_norm - t2_out) ** 2)))
        t1t2_ok = np.allclose(t1_out, t2_out, atol=1e-7)
        print(f"  Track 2 (class)    RMSE_norm={t2_rmse:.6f}  T1==T2: {'OK' if t1t2_ok else 'DIFF'}")

        t3_inp  = t3_quantise_input(px_norm, t2_model.input_scale, t2_model.input_zp)
        t3_eacc = t3_encoder_matmul(t3_inp, t2_model.enc_w, t2_model.enc_b, t2_model.input_zp)
        t3_ereq = t3_encoder_requantise(t3_eacc, t2_model.M1_mult, t2_model.M1_shift, t2_model.enc_act_zp)
        t3_hid  = t3_sigmoid(t3_ereq, t2_model.sigmoid_lut, t2_model.enc_act_zp)
        t3_dacc = t3_decoder_matmul(t3_hid, t2_model.dec_w, t2_model.dec_b, t2_model.sigmoid_zp)
        t3_out8 = t3_decoder_requantise(t3_dacc, t2_model.M2_mult, t2_model.M2_shift, t2_model.output_zp)
        t3_out  = t3_dequantise_output(t3_out8, t2_model.output_scale, t2_model.output_zp)
        t3_rmse = float(np.sqrt(np.mean((px_norm - t3_out) ** 2)))
        t2t3_ok = np.allclose(t2_out, t3_out, atol=1e-7)
        print(f"  Track 3 (manual)   RMSE_norm={t3_rmse:.6f}  T2==T3: {'OK' if t2t3_ok else 'DIFF'}")

        print("\n  -- T2 vs T3 stage comparison --")
        cmp_int  ("T2 input_int8", t2_inp,  "T3 input_int8", t3_inp)
        cmp_float("T2 enc_acc",    t2_eacc.astype(np.float64), "T3 enc_acc", t3_eacc.astype(np.float64))
        cmp_int  ("T2 enc_req",    t2_ereq, "T3 enc_req",    t3_ereq)
        cmp_int  ("T2 hidden",     t2_hid,  "T3 hidden",     t3_hid)
        cmp_float("T2 dec_acc",    t2_dacc.astype(np.float64), "T3 dec_acc", t3_dacc.astype(np.float64))
        cmp_int  ("T2 out_int8",   t2_out8, "T3 out_int8",   t3_out8)
        cmp_float("T2 out_f32",    t2_out.astype(np.float64),  "T3 out_f32", t3_out.astype(np.float64))

    # ── Full batch ────────────────────────────────────────────────────────────
    print_section("FULL-IMAGE BATCH COMPARISON")
    MAX_PX = 5000
    N_tot  = H * W
    if N_tot > MAX_PX:
        sub_idx = np.random.default_rng(RANDOM_SEED + 1).choice(N_tot, MAX_PX, replace=False)
    else:
        sub_idx = np.arange(N_tot)
    sub_flat = flat[sub_idx].astype(np.float32)
    sub_norm = scaler.transform(sub_flat).astype(np.float32)

    r1 = t1_model.batch(sub_norm, sub_flat, scaler)
    r2 = t2_model.batch(sub_norm, sub_flat, scaler)
    r3 = t3_batch(sub_norm, sub_flat, scaler, t2_model)

    print(f"  T1: {arr_summary(r1)}")
    print(f"  T2: {arr_summary(r2)}")
    print(f"  T3: {arr_summary(r3)}")
    d12 = float(np.sqrt(np.mean((r1 - r2) ** 2)))
    d23 = float(np.sqrt(np.mean((r2 - r3) ** 2)))
    d13 = float(np.sqrt(np.mean((r1 - r3) ** 2)))
    print(f"\n  T1 vs T2: {d12:.4e}  T2 vs T3: {d23:.4e}  T1 vs T3: {d13:.4e}")

    if gt_mask is not None and gt_mask.sum() > 0:
        sub_gt = gt_mask.reshape(-1)[sub_idx]
        print(f"  AUC T1={compute_auc(r1,sub_gt):.6f}  "
              f"T2={compute_auc(r2,sub_gt):.6f}  T3={compute_auc(r3,sub_gt):.6f}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print_section("FINAL VERDICT")
    px_n  = scaler.transform(flat[H * W // 2][np.newaxis])[0].astype(np.float32)
    t1_f  = t1_model(px_n)
    t2_f  = t2_model.forward(px_n)
    t3_f  = t3_forward(px_n, t2_model)
    t1t2  = np.allclose(t1_f, t2_f['out_f32'], atol=1e-7)
    mm_inp  = int(np.sum(t2_f['input_int8'] != t3_f['input_int8']))
    mm_ereq = int(np.sum(t2_f['enc_req']    != t3_f['enc_req']))
    mm_hid  = int(np.sum(t2_f['hidden']     != t3_f['hidden']))
    mm_out  = int(np.sum(t2_f['out_int8']   != t3_f['out_int8']))
    r_out   = float(np.sqrt(np.mean((t2_f['out_f32'] - t3_f['out_f32']) ** 2)))
    EPS     = 1e-5

    checks = [
        ("T1 == T2  (TFLite vs class)",    t1t2,           "match" if t1t2 else "MISMATCH"),
        ("Input quantisation  T2==T3",     mm_inp  == 0,   f"mm={mm_inp}"),
        ("Encoder requantise  T2==T3",     mm_ereq == 0,   f"mm={mm_ereq}"),
        ("Sigmoid LUT         T2==T3",     mm_hid  == 0,   f"mm={mm_hid}"),
        ("Decoder requantise  T2==T3",     mm_out  == 0,   f"mm={mm_out}"),
        ("Output float32      T2==T3",     r_out   < EPS,  f"RMSE={r_out:.2e}"),
        ("Batch RMSE          T2==T3",     d23     < EPS,  f"RMSE={d23:.2e}"),
    ]
    all_pass = True
    for name, ok, detail in checks:
        all_pass = all_pass and ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<44}  {detail}")
    print()
    print("  ALL PASS" if all_pass else "  CHECKS FAILED")
    return all_pass


# ══════════════════════════════════════════════════════════════════════════════
# main_all
# ══════════════════════════════════════════════════════════════════════════════

def main_all():
    import traceback
    verdicts = []
    for idx, scene in enumerate(ABU_SCENES):
        global SCENE_INDEX
        SCENE_INDEX = idx
        print(f"\n{'=' * 78}")
        print(f"  SCENE {idx+1}/{len(ABU_SCENES)}: {scene}")
        print(f"{'=' * 78}")
        try:
            ok = main()
            verdicts.append((scene, ok))
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            verdicts.append((scene, False))

    print(f"\n{'=' * 78}")
    print("  ALL-SCENES SUMMARY")
    print(f"{'=' * 78}")
    for scene, ok in verdicts:
        print(f"  {'PASS' if ok else 'FAIL'}  {scene}")
    passed = sum(1 for _, ok in verdicts if ok)
    print(f"  {passed}/{len(verdicts)} scenes passed")


if __name__ == '__main__':
    if RUN_MODE == 'all':
        main_all()
    else:
        main()