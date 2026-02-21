# %% [markdown]
"""
inspect_ptq.py — Three-Track PTQ Validation for MAWDBN DBN

Track 1 — TFLite black-box:   invoke() only. Ground truth.
Track 2 — QuantisedDBN class:  Extracts weights/params from the TFLite flatbuffer
                               and exposes each layer stage as a separate callable
                               method (quantise_input, encoder_matmul, etc.).
                               T1 and T2 run identical arithmetic — T2 just makes
                               each intermediate value inspectable.
Track 3 — Manual reimpl:       Standalone functions that independently rewrite the
                               same int8 arithmetic as T2, for cross-validation.

Checks:
  T1 == T2  →  class correctly reproduces TFLite inference at every stage
  T2 == T3  →  standalone reimplementation matches the class exactly

USAGE: Set SCENE_INDEX (0-12) and run.
"""

SCENE_INDEX     = 3
RESULTS_DIR     = 'results/mawdbn_fixed'
ABU_DATASET_DIR = 'ABU_DATASET'
N_SAMPLES_TRAIN = 50000
RANDOM_SEED     = 42

ABU_SCENES = [
    'abu-airport-1', 'abu-airport-2', 'abu-airport-3', 'abu-airport-4',
    'abu-beach-1',   'abu-beach-2',   'abu-beach-3',   'abu-beach-4',
    'abu-urban-1',   'abu-urban-2',   'abu-urban-3',   'abu-urban-4',
    'abu-urban-5',
]

import os, sys, random
from collections import OrderedDict
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def set_seed(s=RANDOM_SEED):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)

# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_abu_mat(mat_path):
    mat = sio.loadmat(mat_path)
    if 'data' in mat:
        image = mat['data'].astype(np.float32)
    else:
        cands = {k: v for k, v in mat.items()
                 if not k.startswith('_') and isinstance(v, np.ndarray) and v.ndim == 3}
        image = cands[max(cands, key=lambda k: cands[k].size)].astype(np.float32)
    if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0))
    gt = None
    if 'map' in mat:
        gt = mat['map'].astype(np.uint8).squeeze()
        if gt.shape != image.shape[:2]:
            gt = gt.T
    return image, gt

def fit_scaler(image):
    H, W, C = image.shape
    n   = min(N_SAMPLES_TRAIN, H * W)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(H * W, size=n, replace=False)
    sc  = StandardScaler()
    sc.fit(image.reshape(-1, C)[idx])
    return sc

def find_tflite(scene):
    p = os.path.join(RESULTS_DIR, scene, f'mawdbn_ptq_{scene}_ptq_int8.tflite')
    if not os.path.exists(p):
        raise FileNotFoundError(f"PTQ TFLite not found: {p}")
    return p

# ══════════════════════════════════════════════════════════════════════════════
# Track 1 — TFLite black-box
# ══════════════════════════════════════════════════════════════════════════════

class TFLiteModel:
    """Thin wrapper around the PTQ TFLite interpreter. Black-box inference only."""

    def __init__(self, tflite_path):
        self.interp = tf.lite.Interpreter(model_path=tflite_path)
        self.interp.allocate_tensors()
        self.inp_det = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]
        self.input_scale  = float(self.inp_det['quantization_parameters']['scales'][0])
        self.input_zp     = int(self.inp_det['quantization_parameters']['zero_points'][0])
        self.output_scale = float(self.out_det['quantization_parameters']['scales'][0])
        self.output_zp    = int(self.out_det['quantization_parameters']['zero_points'][0])

    def __call__(self, pixel_norm: np.ndarray) -> np.ndarray:
        """Run one normalised float32 pixel through TFLite. Returns float32 reconstruction."""
        q = np.clip(
            np.round(pixel_norm / self.input_scale) + self.input_zp,
            -128, 127).astype(np.int8)
        self.interp.set_tensor(self.inp_det['index'], q[np.newaxis])
        self.interp.invoke()
        out = self.interp.get_tensor(self.out_det['index'])[0]
        return (out.astype(np.float32) - self.output_zp) * self.output_scale

    def batch(self, norm: np.ndarray, flat: np.ndarray, scaler) -> np.ndarray:
        """Run N pixels, return per-pixel RMSE in original (unscaled) space."""
        recon = np.zeros_like(norm)
        for i in range(len(norm)):
            recon[i] = self(norm[i])
        orig = scaler.inverse_transform(recon).astype(np.float32)
        return np.sqrt(np.mean((flat - orig) ** 2, axis=1))

# ══════════════════════════════════════════════════════════════════════════════
# Fixed-point multiplier helper
# ══════════════════════════════════════════════════════════════════════════════

def quantise_multiplier(M: float, n_bits: int = 31):
    """
    Convert a float scalar M into (int32 multiplier, int shift) such that:

        M ≈ multiplier * 2^(-shift)

    This is the standard TFLite method for eliminating floating-point from
    the requantisation step.  The approach:

      1. Express M as  mantissa * 2^exponent  where 0.5 <= mantissa < 1.0
         (i.e. normalise M into the top half of [0,1)).
      2. Scale mantissa up to an (n_bits)-bit integer:
             multiplier = round(mantissa * 2^n_bits)
      3. The shift is then:  shift = n_bits - exponent
         so that  multiplier * 2^(-shift) = mantissa * 2^exponent = M.

    At inference time the operation is:
        result = (acc * multiplier) >> shift          # pure integer

    n_bits=31 matches TFLite's own implementation (32-bit signed multiply,
    top bit reserved for sign, giving 31 usable bits of precision).
    """
    if M == 0.0:
        return 0, 0

    # Decompose M into mantissa in [0.5, 1.0) and a power-of-two exponent.
    import math
    exponent = math.floor(math.log2(M))   # M = mantissa * 2^exponent
    mantissa = M / (2 ** exponent)        # mantissa in [0.5, 1.0)  (or 1.0 exact)

    # Scale mantissa to an n_bits integer
    multiplier = int(round(mantissa * (2 ** n_bits)))

    # If rounding pushed multiplier to or beyond 2^n_bits, renormalise
    while multiplier >= (2 ** n_bits):
        multiplier //= 2
        exponent   += 1

    shift = n_bits - exponent
    assert 0 < multiplier < (2 ** n_bits), f"multiplier out of range: {multiplier}"
    return multiplier, shift


# ══════════════════════════════════════════════════════════════════════════════
# Track 2 — QuantisedDBN class
# One method per layer stage. Parameters extracted from the TFLite flatbuffer.
# ══════════════════════════════════════════════════════════════════════════════

class QuantisedDBN:
    """
    Track 2: Quantised DBN with one method per layer stage.

    Weights, biases and quantisation parameters are extracted directly from
    the TFLite flatbuffer — the same values TFLite uses internally.  Each
    layer method performs the identical int8 arithmetic that TFLite runs,
    but exposes the intermediate tensors so they can be inspected and
    compared against T1 and T3.
    """

    def __init__(self, tflite_path: str):
        self._load_params(tflite_path)
        self._build_sigmoid_lut()

    # ── Parameter loading ─────────────────────────────────────────────────────

    def _load_params(self, tflite_path: str):
        # Use reference kernel (no XNNPACK) so intermediate tensors are visible
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

        # enc_w: (n_hidden, n_input)   dec_w: (n_input, n_hidden)
        self.enc_w = raw_enc_w.astype(np.int8)
        self.enc_b = raw_enc_b.flatten().astype(np.int32)
        self.dec_w = raw_dec_w.astype(np.int8)
        self.dec_b = raw_dec_b.flatten().astype(np.int32)

        self.n_hidden = len(self.enc_b)
        self.n_input  = len(self.dec_b)

        assert self.enc_w.shape == (self.n_hidden, self.n_input), \
            f"enc_w shape {self.enc_w.shape} != ({self.n_hidden}, {self.n_input})"
        assert self.dec_w.shape == (self.n_input, self.n_hidden), \
            f"dec_w shape {self.dec_w.shape} != ({self.n_input}, {self.n_hidden})"

        self.M1 = (self.input_scale   * self.enc_w_scale) / self.enc_act_scale
        self.M2 = (self.sigmoid_scale * self.dec_w_scale) / self.output_scale

        # Fixed-point representation of M1 and M2 for T3.
        # Each float scalar M is expressed as (int32 multiplier, int shift) such that:
        #   M ≈ multiplier * 2^(-shift)
        # This allows requantisation using only integer multiply + right-shift.
        self.M1_mult, self.M1_shift = quantise_multiplier(self.M1)
        self.M2_mult, self.M2_shift = quantise_multiplier(self.M2)

    def _build_sigmoid_lut(self):
        # Index i maps to int8 enc_act value (i - 128).
        # Float value = (enc_act_int8 - enc_act_zp) * enc_act_scale
        # LUT is indexed by enc_req + 128.
        lut = np.zeros(256, dtype=np.int8)
        for i in range(256):
            enc_act_val = i - 128          # int8 encoder activation value
            x_float = (float(enc_act_val) - self.enc_act_zp) * self.enc_act_scale
            x_float = max(-30.0, min(30.0, x_float))
            sig     = 1.0 / (1.0 + np.exp(-x_float))
            q       = int(round(sig / self.sigmoid_scale)) + self.sigmoid_zp
            lut[i]  = np.int8(max(-128, min(127, q)))
        self.sigmoid_lut = lut

    # ── Layer methods ─────────────────────────────────────────────────────────

    def quantise_input(self, pixel_norm: np.ndarray) -> np.ndarray:
        """float32 → int8"""
        q = np.round(pixel_norm / self.input_scale).astype(np.int64) + self.input_zp
        return np.clip(q, -128, 127).astype(np.int8)

    def encoder_matmul(self, inp_q: np.ndarray) -> np.ndarray:
        """int8 input → int32 accumulator"""
        acc = self.enc_b.copy().astype(np.int64)
        acc += self.enc_w.astype(np.int64) @ (inp_q.astype(np.int64) - self.input_zp)
        return acc.astype(np.int32)

    def encoder_requantise(self, enc_acc: np.ndarray) -> np.ndarray:
        """int32 accumulator → int8 enc_act"""
        q = np.round(enc_acc.astype(np.float64) * self.M1).astype(np.int64) + self.enc_act_zp
        return np.clip(q, -128, 127).astype(np.int8)

    def sigmoid(self, enc_req: np.ndarray) -> np.ndarray:
        """int8 enc_act → int8 hidden via LUT. Index = enc_req + 128."""
        idx = enc_req.astype(np.int32) + 128
        return self.sigmoid_lut[idx].astype(np.int8)

    def decoder_matmul(self, hidden: np.ndarray) -> np.ndarray:
        """int8 hidden → int32 accumulator"""
        acc = self.dec_b.copy().astype(np.int64)
        acc += self.dec_w.astype(np.int64) @ (hidden.astype(np.int64) - self.sigmoid_zp)
        return acc.astype(np.int32)

    def decoder_requantise(self, dec_acc: np.ndarray) -> np.ndarray:
        """int32 accumulator → int8 output"""
        q = np.round(dec_acc.astype(np.float64) * self.M2).astype(np.int64) + self.output_zp
        return np.clip(q, -128, 127).astype(np.int8)

    def dequantise_output(self, out_int8: np.ndarray) -> np.ndarray:
        """int8 → float32"""
        return (out_int8.astype(np.float32) - self.output_zp) * self.output_scale

    def forward(self, pixel_norm: np.ndarray) -> OrderedDict:
        """Full forward pass, returns all intermediates."""
        s = OrderedDict()
        s['input_int8'] = self.quantise_input(pixel_norm)
        s['enc_acc']    = self.encoder_matmul(s['input_int8'])
        s['enc_req']    = self.encoder_requantise(s['enc_acc'])
        s['hidden']     = self.sigmoid(s['enc_req'])
        s['dec_acc']    = self.decoder_matmul(s['hidden'])
        s['out_int8']   = self.decoder_requantise(s['dec_acc'])
        s['out_f32']    = self.dequantise_output(s['out_int8'])
        return s

    def batch(self, norm: np.ndarray, flat: np.ndarray, scaler) -> np.ndarray:
        """Vectorised batch → per-pixel RMSE."""
        inp_q   = np.clip(
            np.round(norm / self.input_scale).astype(np.int64) + self.input_zp,
            -128, 127).astype(np.int8)
        enc_acc = (inp_q.astype(np.int64) - self.input_zp) \
                  @ self.enc_w.T.astype(np.int64) + self.enc_b.astype(np.int64)
        enc_req = np.clip(
            np.round(enc_acc.astype(np.float64) * self.M1).astype(np.int64) + self.enc_act_zp,
            -128, 127).astype(np.int8)
        hidden  = self.sigmoid_lut[
            enc_req.astype(np.int32) + 128].astype(np.int8)
        dec_acc = (hidden.astype(np.int64) - self.sigmoid_zp) \
                  @ self.dec_w.T.astype(np.int64) + self.dec_b.astype(np.int64)
        out_q   = np.clip(
            np.round(dec_acc.astype(np.float64) * self.M2).astype(np.int64) + self.output_zp,
            -128, 127).astype(np.int8)
        out_f   = (out_q.astype(np.float32) - self.output_zp) * self.output_scale
        orig    = scaler.inverse_transform(out_f).astype(np.float32)
        return np.sqrt(np.mean((flat - orig) ** 2, axis=1))

# ══════════════════════════════════════════════════════════════════════════════
# Track 3 — Standalone reimplementation (independent functions, no class)
# Independently rewrites the same int8 arithmetic as T2.
# Compared against T2 stage-by-stage to validate correctness.
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
    # Pure integer requantisation — no floating point.
    # Computes: round(enc_acc * M1) + enc_act_zp
    # where M1 = M1_mult * 2^(-M1_shift).
    # Adding 2^(shift-1) before the right-shift achieves round-to-nearest.
    half = np.int64(1) << np.int64(M1_shift - 1)
    q = (enc_acc.astype(np.int64) * np.int64(M1_mult) + half) >> np.int64(M1_shift)
    return np.clip(q + enc_act_zp, -128, 127).astype(np.int8)

def t3_sigmoid(enc_req, sigmoid_lut, enc_act_zp):
    idx = enc_req.astype(np.int32) + 128
    return sigmoid_lut[idx].astype(np.int8)

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

def t3_forward(pixel_norm, model: QuantisedDBN) -> OrderedDict:
    s = OrderedDict()
    s['input_int8'] = t3_quantise_input(pixel_norm, model.input_scale, model.input_zp)
    s['enc_acc']    = t3_encoder_matmul(s['input_int8'], model.enc_w, model.enc_b, model.input_zp)
    s['enc_req']    = t3_encoder_requantise(s['enc_acc'], model.M1_mult, model.M1_shift, model.enc_act_zp)
    s['hidden']     = t3_sigmoid(s['enc_req'], model.sigmoid_lut, model.enc_act_zp)
    s['dec_acc']    = t3_decoder_matmul(s['hidden'], model.dec_w, model.dec_b, model.sigmoid_zp)
    s['out_int8']   = t3_decoder_requantise(s['dec_acc'], model.M2_mult, model.M2_shift, model.output_zp)
    s['out_f32']    = t3_dequantise_output(s['out_int8'], model.output_scale, model.output_zp)
    return s

def t3_batch(norm, flat, scaler, model: QuantisedDBN) -> np.ndarray:
    inp_q   = np.clip(
        np.round(norm / model.input_scale).astype(np.int64) + model.input_zp,
        -128, 127).astype(np.int8)
    enc_acc = (inp_q.astype(np.int64) - model.input_zp) \
              @ model.enc_w.T.astype(np.int64) + model.enc_b.astype(np.int64)
    half1   = np.int64(1) << np.int64(model.M1_shift - 1)
    enc_req = np.clip(
        ((enc_acc.astype(np.int64) * np.int64(model.M1_mult) + half1) >> np.int64(model.M1_shift))
        + model.enc_act_zp, -128, 127).astype(np.int8)
    hidden  = model.sigmoid_lut[
        enc_req.astype(np.int32) + 128].astype(np.int8)
    dec_acc = (hidden.astype(np.int64) - model.sigmoid_zp) \
              @ model.dec_w.T.astype(np.int64) + model.dec_b.astype(np.int64)
    half2   = np.int64(1) << np.int64(model.M2_shift - 1)
    out_q   = np.clip(
        ((dec_acc.astype(np.int64) * np.int64(model.M2_mult) + half2) >> np.int64(model.M2_shift))
        + model.output_zp, -128, 127).astype(np.int8)
    out_f   = (out_q.astype(np.float32) - model.output_zp) * model.output_scale
    orig    = scaler.inverse_transform(out_f).astype(np.float32)
    return np.sqrt(np.mean((flat - orig) ** 2, axis=1))

# ══════════════════════════════════════════════════════════════════════════════
# Comparison utilities
# ══════════════════════════════════════════════════════════════════════════════

def print_section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def cmp_int(la, a, lb, b, n=8):
    d  = a.astype(np.int64) - b.astype(np.int64)
    mm = int(np.sum(d != 0))
    print(f"\n  ── {la} vs {lb}")
    print(f"     Mismatches={mm}/{len(a)}  MaxAbsDiff={int(np.abs(d).max())}")
    for i in range(min(n, len(a))):
        mk = " ←" if d[i] != 0 else ""
        print(f"     [{i:3d}]  {int(a[i]):6d}  {int(b[i]):6d}  diff={int(d[i]):+d}{mk}")
    return mm

def cmp_float(la, a, lb, b, n=8):
    d    = a.astype(np.float64) - b.astype(np.float64)
    rmse = float(np.sqrt(np.mean(d ** 2)))
    print(f"\n  ── {la} vs {lb}")
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
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    set_seed()
    scene = ABU_SCENES[SCENE_INDEX]
    print_section(f"PTQ THREE-TRACK VALIDATION  |  {scene}  (index {SCENE_INDEX})")
    print("  Track 1 = TFLite black-box")
    print("  Track 2 = QuantisedDBN class (layer-by-layer)")
    print("  Track 3 = Manual standalone reimplementation\n")

    mat_path = os.path.join(ABU_DATASET_DIR, f'{scene}.mat')
    if not os.path.exists(mat_path):
        print(f"ERROR: {mat_path} not found"); sys.exit(1)

    image, gt_mask = load_abu_mat(mat_path)
    H, W, C = image.shape
    flat    = image.reshape(-1, C)
    print(f"  Image: {H}×{W}×{C}   GT mask: {'yes' if gt_mask is not None else 'none'}")

    tflite_path = find_tflite(scene)
    print(f"  TFLite: {os.path.relpath(tflite_path)}")

    print("\n  Fitting scaler...")
    scaler = fit_scaler(image)

    # ── Instantiate models ────────────────────────────────────────────────────
    print_section("MODEL LOADING")
    print("  Loading Track 1 (TFLite)...")
    t1_model = TFLiteModel(tflite_path)

    print("  Loading Track 2 (QuantisedDBN)...")
    t2_model = QuantisedDBN(tflite_path)

    print(f"\n  N_INPUT={t2_model.n_input}  N_HIDDEN={t2_model.n_hidden}")
    print(f"  input_scale={t2_model.input_scale:.6e}   input_zp={t2_model.input_zp}")
    print(f"  enc_w_scale={t2_model.enc_w_scale:.6e}")
    print(f"  enc_act_scale={t2_model.enc_act_scale:.6e}  enc_act_zp={t2_model.enc_act_zp}")
    print(f"  sigmoid_scale={t2_model.sigmoid_scale:.6e}  sigmoid_zp={t2_model.sigmoid_zp}")
    print(f"  dec_w_scale={t2_model.dec_w_scale:.6e}")
    print(f"  output_scale={t2_model.output_scale:.6e}   output_zp={t2_model.output_zp}")
    print(f"  M1={t2_model.M1:.6e}  M2={t2_model.M2:.6e}")

    # ── Single-pixel layer-by-layer ───────────────────────────────────────────
    test_pixels = [('Centre', H * W // 2)]
    if gt_mask is not None and gt_mask.sum() > 0:
        pos = np.argwhere(gt_mask.reshape(-1))
        test_pixels.append(('Anomaly', int(pos[len(pos) // 2][0])))

    for px_label, px_idx in test_pixels:
        print_section(f"LAYER-BY-LAYER — {px_label} pixel #{px_idx}")
        px_orig = flat[px_idx].astype(np.float32)
        px_norm = scaler.transform(px_orig[np.newaxis])[0].astype(np.float32)

        # Track 1 — black box
        t1_out  = t1_model(px_norm)
        t1_rmse = float(np.sqrt(np.mean(
            (px_orig - scaler.inverse_transform(t1_out[np.newaxis])[0]) ** 2)))
        print(f"\n  Track 1 (TFLite)   RMSE={t1_rmse:.6f}")
        print(f"  out_f32[0:5] = {[round(float(x), 6) for x in t1_out[:5]]}")

        # Track 2 — each layer method called individually
        t2_inp  = t2_model.quantise_input(px_norm)
        t2_eacc = t2_model.encoder_matmul(t2_inp)
        t2_ereq = t2_model.encoder_requantise(t2_eacc)
        t2_hid  = t2_model.sigmoid(t2_ereq)
        t2_dacc = t2_model.decoder_matmul(t2_hid)
        t2_out8 = t2_model.decoder_requantise(t2_dacc)
        t2_out  = t2_model.dequantise_output(t2_out8)

        t2_rmse = float(np.sqrt(np.mean(
            (px_orig - scaler.inverse_transform(t2_out[np.newaxis])[0]) ** 2)))
        t1t2_ok = np.allclose(t1_out, t2_out, atol=1e-7)
        print(f"\n  Track 2 (class)    RMSE={t2_rmse:.6f}  T1==T2: {'✓' if t1t2_ok else '✗'}")
        print(f"  out_f32[0:5] = {[round(float(x), 6) for x in t2_out[:5]]}")

        # Track 3 — standalone functions called individually
        t3_inp  = t3_quantise_input(px_norm, t2_model.input_scale, t2_model.input_zp)
        t3_eacc = t3_encoder_matmul(t3_inp, t2_model.enc_w, t2_model.enc_b, t2_model.input_zp)
        t3_ereq = t3_encoder_requantise(t3_eacc, t2_model.M1_mult, t2_model.M1_shift, t2_model.enc_act_zp)
        t3_hid  = t3_sigmoid(t3_ereq, t2_model.sigmoid_lut, t2_model.enc_act_zp)
        t3_dacc = t3_decoder_matmul(t3_hid, t2_model.dec_w, t2_model.dec_b, t2_model.sigmoid_zp)
        t3_out8 = t3_decoder_requantise(t3_dacc, t2_model.M2_mult, t2_model.M2_shift, t2_model.output_zp)
        t3_out  = t3_dequantise_output(t3_out8, t2_model.output_scale, t2_model.output_zp)

        t3_rmse = float(np.sqrt(np.mean(
            (px_orig - scaler.inverse_transform(t3_out[np.newaxis])[0]) ** 2)))
        t2t3_ok = np.allclose(t2_out, t3_out, atol=1e-7)
        print(f"\n  Track 3 (manual)   RMSE={t3_rmse:.6f}  T2==T3: {'✓' if t2t3_ok else '✗'}")
        print(f"  out_f32[0:5] = {[round(float(x), 6) for x in t3_out[:5]]}")

        # Layer-by-layer T2 vs T3
        print("\n  ── T2 (class) vs T3 (manual) ────────────────────────────────────")
        cmp_int  ("T2 input_int8",  t2_inp,  "T3 input_int8",  t3_inp)
        cmp_float("T2 enc_acc",     t2_eacc.astype(np.float64),
                  "T3 enc_acc",     t3_eacc.astype(np.float64))
        cmp_int  ("T2 enc_req",     t2_ereq, "T3 enc_req",     t3_ereq)
        cmp_int  ("T2 hidden",      t2_hid,  "T3 hidden",      t3_hid)
        cmp_float("T2 dec_acc",     t2_dacc.astype(np.float64),
                  "T3 dec_acc",     t3_dacc.astype(np.float64))
        cmp_int  ("T2 out_int8",    t2_out8, "T3 out_int8",    t3_out8)
        cmp_float("T2 out_f32",     t2_out.astype(np.float64),
                  "T3 out_f32",     t3_out.astype(np.float64))

    # ── Self-consistency: T3 scalar vs T3 vectorised ──────────────────────────
    print_section("SELF-CONSISTENCY  (T3 scalar vs T3 vectorised)")
    N_CHK    = min(100, H * W)
    chk_idx  = np.random.default_rng(RANDOM_SEED + 99).choice(H * W, N_CHK, replace=False)
    chk_norm = scaler.transform(flat[chk_idx]).astype(np.float32)

    scalar_out = np.zeros((N_CHK, t2_model.n_input), dtype=np.float32)
    for i in range(N_CHK):
        scalar_out[i] = t3_forward(chk_norm[i], t2_model)['out_f32']

    # Recompute vectorised output for comparison (uses T3 fixed-point path)
    _iq  = np.clip(np.round(chk_norm / t2_model.input_scale).astype(np.int64)
                   + t2_model.input_zp, -128, 127).astype(np.int8)
    _ea  = (_iq.astype(np.int64) - t2_model.input_zp) \
           @ t2_model.enc_w.T.astype(np.int64) + t2_model.enc_b.astype(np.int64)
    _h1  = np.int64(1) << np.int64(t2_model.M1_shift - 1)
    _er  = np.clip(
           (((_ea.astype(np.int64) * np.int64(t2_model.M1_mult) + _h1) >> np.int64(t2_model.M1_shift))
           + t2_model.enc_act_zp), -128, 127).astype(np.int8)
    _hq  = t2_model.sigmoid_lut[
           _er.astype(np.int32) + 128].astype(np.int8)
    _da  = (_hq.astype(np.int64) - t2_model.sigmoid_zp) \
           @ t2_model.dec_w.T.astype(np.int64) + t2_model.dec_b.astype(np.int64)
    _h2  = np.int64(1) << np.int64(t2_model.M2_shift - 1)
    _dr  = np.clip(
           (((_da.astype(np.int64) * np.int64(t2_model.M2_mult) + _h2) >> np.int64(t2_model.M2_shift))
           + t2_model.output_zp), -128, 127).astype(np.int8)
    vec_out = (_dr.astype(np.float32) - t2_model.output_zp) * t2_model.output_scale

    diff       = np.abs(scalar_out - vec_out)
    n_mismatch = int(np.sum(diff > 1e-10))
    print(f"  Pixels: {N_CHK}  Max diff: {diff.max():.2e}  Mismatches: {n_mismatch}")
    print(f"  {'✓ scalar == vectorised' if n_mismatch == 0 else '✗ MISMATCH'}")

    # ── Full batch comparison ─────────────────────────────────────────────────
    print_section("FULL-IMAGE BATCH COMPARISON")
    MAX_PX = 5000
    N_tot  = H * W
    if N_tot > MAX_PX:
        print(f"  Subsampling {N_tot:,} → {MAX_PX}")
        sub_idx = np.random.default_rng(RANDOM_SEED + 1).choice(N_tot, MAX_PX, replace=False)
    else:
        sub_idx = np.arange(N_tot)
    sub_flat = flat[sub_idx].astype(np.float32)
    sub_norm = scaler.transform(sub_flat).astype(np.float32)

    print("  Running Track 1 (TFLite)...")
    r1 = t1_model.batch(sub_norm, sub_flat, scaler)
    print("  Running Track 2 (QuantisedDBN class)...")
    r2 = t2_model.batch(sub_norm, sub_flat, scaler)
    print("  Running Track 3 (manual)...")
    r3 = t3_batch(sub_norm, sub_flat, scaler, t2_model)

    print(f"\n  Track 1 (TFLite): {arr_summary(r1)}")
    print(f"  Track 2 (class):  {arr_summary(r2)}")
    print(f"  Track 3 (manual): {arr_summary(r3)}")

    d12 = float(np.sqrt(np.mean((r1 - r2) ** 2)))
    d23 = float(np.sqrt(np.mean((r2 - r3) ** 2)))
    d13 = float(np.sqrt(np.mean((r1 - r3) ** 2)))
    print(f"\n  T1 vs T2 (TFLite vs class):  {d12:.4e}  ← should be ~0")
    print(f"  T2 vs T3 (class vs manual):  {d23:.4e}  ← should be ~0")
    print(f"  T1 vs T3 (TFLite vs manual): {d13:.4e}  ← should be ~0")

    if gt_mask is not None and gt_mask.sum() > 0:
        sub_gt = gt_mask.reshape(-1)[sub_idx]
        a1 = compute_auc(r1, sub_gt)
        a2 = compute_auc(r2, sub_gt)
        a3 = compute_auc(r3, sub_gt)
        print(f"\n  AUC — T1={a1:.6f}  T2={a2:.6f}  T3={a3:.6f}")

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
        ("T1 == T2  (TFLite vs class)",       t1t2,           "match" if t1t2 else "MISMATCH"),
        ("Input quantisation  T2==T3",         mm_inp  == 0,   f"mismatches={mm_inp}"),
        ("Encoder requantise  T2==T3",         mm_ereq == 0,   f"mismatches={mm_ereq}"),
        ("Sigmoid LUT         T2==T3",         mm_hid  == 0,   f"mismatches={mm_hid}"),
        ("Decoder requantise  T2==T3",         mm_out  == 0,   f"mismatches={mm_out}"),
        ("Output float32      T2==T3",         r_out   < EPS,  f"RMSE={r_out:.2e}"),
        ("Scalar==vectorised  T3 internal",    n_mismatch == 0,f"mismatches={n_mismatch}"),
        ("Batch RMSE          T2==T3",         d23     < EPS,  f"RMSE={d23:.2e}"),
    ]

    all_pass = True
    for name, ok, detail in checks:
        all_pass = all_pass and ok
        print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name:<44}  {detail}")

    print()
    if all_pass:
        print("  ✓ ALL CHECKS PASSED")
        print("    T1 (TFLite) == T2 (QuantisedDBN class): inference is identical.")
        print("    T2 == T3 (standalone): manual reimplementation is bit-exact.")
    else:
        print("  ✗ CHECKS FAILED — review mismatches above.")
    print()

if __name__ == '__main__':
    main()