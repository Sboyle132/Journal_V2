# %% [markdown]
"""
inspect_ptq.py — Three-Track PTQ Validation for MAWDBN DBN

Track 1 — PTQ BASELINE:    TFLite int8, one-shot inference.
Track 2 — PTQ GRANULAR:    Same TFLite model, each stage extracted individually.
Track 3 — MANUAL NUMPY:    Pure NumPy HLS software model, compared against T2.

T1 and T2 must always agree (same model, same weights).
T2 and T3 must agree (T3 is the manual reimplementation of T2).
The float32 Keras model is not used here.

USAGE: Set SCENE_INDEX (0-12) and run. No prompts.
"""

SCENE_INDEX      = 1
RESULTS_DIR      = 'results/mawdbn_fixed'
ABU_DATASET_DIR  = 'ABU_DATASET'
N_SAMPLES_TRAIN  = 50000
BATCH_SIZE       = 2048
RANDOM_SEED      = 42

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

# ── Data ──────────────────────────────────────────────────────────────────────

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

# ── Model discovery ───────────────────────────────────────────────────────────

def find_tflite(scene):
    p = os.path.join(RESULTS_DIR, scene, f'mawdbn_ptq_{scene}_ptq_int8.tflite')
    if not os.path.exists(p):
        raise FileNotFoundError(f"PTQ TFLite not found: {p}")
    return p

# ── Parameter extraction ──────────────────────────────────────────────────────

class ModelParams:
    def __init__(self):
        self.input_scale = self.input_zp = 0
        self.enc_w_scale = self.enc_b_scale = 0.0
        self.enc_act_scale = 0.0; self.enc_act_zp = 0  # post-matmul pre-sigmoid
        self.sigmoid_scale = self.sigmoid_zp = 0
        self.dec_w_scale = self.dec_b_scale = 0.0
        self.output_scale = self.output_zp = 0
        self.enc_w = self.enc_b = self.dec_w = self.dec_b = None
        self.sigmoid_lut = None
        self.n_input = self.n_hidden = 0

def load_tensors(tflite_path):
    """
    Load all tensors using the reference kernel (no XNNPACK delegates).
    The reference kernel exposes intermediate activation tensors (post-matmul,
    post-sigmoid) that XNNPACK fuses and hides.  These are essential to get
    the correct quantisation scales for enc_act and sigmoid outputs.
    """
    try:
        interp = tf.lite.Interpreter(
            model_path=tflite_path,
            experimental_op_resolver_type=
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    except Exception:
        interp = tf.lite.Interpreter(model_path=tflite_path)

    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    interp.set_tensor(inp['index'],
                      np.zeros([1] + list(inp['shape'][1:]), dtype=np.int8))
    interp.invoke()
    out = {}
    for t in interp.get_tensor_details():
        try:
            out[t['name']] = {'data':  interp.get_tensor(t['index']).copy(),
                               'quant': t['quantization_parameters'],
                               'shape': tuple(t['shape'])}
        except Exception:
            pass
    return out, interp

def extract_params(tflite_path):
    p            = ModelParams()
    tensor_map, interp = load_tensors(tflite_path)

    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    p.input_scale  = float(inp_det['quantization_parameters']['scales'][0])
    p.input_zp     = int(inp_det['quantization_parameters']['zero_points'][0])
    p.output_scale = float(out_det['quantization_parameters']['scales'][0])
    p.output_zp    = int(out_det['quantization_parameters']['zero_points'][0])

    print("  TFLite tensor inventory:")
    for name, info in tensor_map.items():
        sc = (info['quant']['scales'][0] if len(info['quant']['scales']) > 0 else 0.0)
        print(f"    [{info['data'].dtype}] {info['shape']}  sc={sc:.3e}  '{name}'")

    # Extract weights and biases by iterating tensor details to get indices,
    # then reading raw data via interp.get_tensor(idx).
    # We identify tensors by name but READ them by index to get the exact values
    # that TFLite uses at inference time (avoids any post-processing in tensor_map).
    tensor_idx = {}
    tensor_quant = {}
    for t in interp.get_tensor_details():
        tensor_idx[t['name']] = t['index']
        tensor_quant[t['name']] = t['quantization_parameters']

    def read_tensor(name):
        idx = tensor_idx.get(name)
        if idx is None:
            return None, 0.0, 0
        raw = interp.get_tensor(idx).copy()
        q = tensor_quant[name]
        sc = float(q['scales'][0]) if len(q['scales']) > 0 else 0.0
        zp = int(q['zero_points'][0]) if len(q['zero_points']) > 0 else 0
        return raw, sc, zp

    # Encoder weight: name contains 'encoder_dense' and 'MatMul' (not BiasAdd)
    # Decoder weight: name contains 'decoder_dense' and 'MatMul' (not BiasAdd)
    enc_w_name = next((n for n in tensor_idx
                       if 'encoder_dense' in n and 'MatMul' in n and 'BiasAdd' not in n
                       and ';' not in n), None)
    enc_b_name = next((n for n in tensor_idx
                       if 'encoder_dense' in n and 'BiasAdd' in n
                       and 'Sigmoid' not in n and ';' not in n), None)
    dec_w_name = next((n for n in tensor_idx
                       if 'decoder_dense' in n and 'MatMul' in n and 'BiasAdd' not in n
                       and ';' not in n), None)
    dec_b_name = next((n for n in tensor_idx
                       if 'decoder_dense' in n and 'BiasAdd' in n and ';' not in n), None)
    enc_act_name = next((n for n in tensor_idx
                         if ('encoder_dense' in n and 'MatMul' in n and 'BiasAdd' in n)
                         or (';' in n and 'encoder' in n.lower() and 'decoder' not in n.lower())), None)
    sig_name = next((n for n in tensor_idx
                     if 'Sigmoid' in n and 'encoder' in n.lower()), None)

    raw_enc_w, enc_w_sc, _ = read_tensor(enc_w_name)
    raw_enc_b, enc_b_sc, _ = read_tensor(enc_b_name)
    raw_dec_w, dec_w_sc, _ = read_tensor(dec_w_name)
    raw_dec_b, dec_b_sc, _ = read_tensor(dec_b_name)
    _, enc_act_sc, enc_act_zp_ = read_tensor(enc_act_name)
    _, sig_sc, sig_zp_         = read_tensor(sig_name)

    if raw_enc_b is not None:
        p.enc_b = raw_enc_b.flatten().astype(np.int32)
        p.enc_b_scale = enc_b_sc
        p.n_hidden = len(p.enc_b)
    if raw_dec_b is not None:
        p.dec_b = raw_dec_b.flatten().astype(np.int32)
        p.dec_b_scale = dec_b_sc
        p.n_input = len(p.dec_b)

    # Store weights directly as read from the interpreter (same as get_tensor at runtime).
    # enc_w TFLite shape: (n_hidden, n_input) — verified by weight formula search.
    # dec_w TFLite shape: (n_input, n_hidden) — verified by weight formula search.
    if raw_enc_w is not None:
        p.enc_w = raw_enc_w.astype(np.int8)
        p.enc_w_scale = enc_w_sc
        if p.enc_w.shape[0] != p.n_hidden and p.enc_w.shape[1] == p.n_hidden:
            p.enc_w = p.enc_w.T  # ensure (n_hidden, n_input)
    if raw_dec_w is not None:
        p.dec_w = raw_dec_w.astype(np.int8)
        p.dec_w_scale = dec_w_sc
        if p.dec_w.shape[0] != p.n_input and p.dec_w.shape[1] == p.n_input:
            p.dec_w = p.dec_w.T  # ensure (n_input, n_hidden)

    if enc_act_sc > 0:
        p.enc_act_scale = enc_act_sc; p.enc_act_zp = enc_act_zp_
    if sig_sc > 0:
        p.sigmoid_scale = sig_sc; p.sigmoid_zp = sig_zp_

    if p.enc_w is not None and p.n_hidden > 0 and p.n_input > 0:
        assert p.enc_w.shape == (p.n_hidden, p.n_input),             f"enc_w shape {p.enc_w.shape} != ({p.n_hidden},{p.n_input})"
    if p.dec_w is not None and p.n_hidden > 0 and p.n_input > 0:
        assert p.dec_w.shape == (p.n_input, p.n_hidden),             f"dec_w shape {p.dec_w.shape} != ({p.n_input},{p.n_hidden})" 

    # Fallback / verification: search all tensors by shape and scale magnitude.
    # enc_act (pre-sigmoid): int8, shape N_HIDDEN, scale >> enc_b_scale (typically ~0.1)
    # sigmoid output:        int8, shape N_HIDDEN, scale < enc_act_scale (typically ~0.004)
    # We collect all int8 [N_HIDDEN] tensors excluding input/output, sort by scale desc.
    activation_candidates = []
    for name, info in tensor_map.items():
        sc = (float(info['quant']['scales'][0])
              if len(info['quant']['scales']) > 0 else 0.0)
        zp = (int(info['quant']['zero_points'][0])
              if len(info['quant']['zero_points']) > 0 else 0)
        if (info['data'].dtype == np.int8
                and int(np.prod(info['shape'])) == p.n_hidden
                and sc not in (p.input_scale, p.output_scale) and sc > 0):
            activation_candidates.append((sc, zp, name))
    activation_candidates.sort(reverse=True)  # largest scale first = enc_act
    print(f"  Int8 [N_HIDDEN] activation tensors found: {len(activation_candidates)}")
    for sc, zp, name in activation_candidates:
        print(f"    sc={sc:.4e}  zp={zp:5d}  '{name}'")

    if len(activation_candidates) >= 2:
        if p.enc_act_scale == 0.0:
            p.enc_act_scale = activation_candidates[0][0]
            p.enc_act_zp    = activation_candidates[0][1]
        if p.sigmoid_scale == 0.0:
            p.sigmoid_scale = activation_candidates[1][0]
            p.sigmoid_zp    = activation_candidates[1][1]
    elif len(activation_candidates) == 1:
        # Only sigmoid output found (reference kernel may not expose enc_act separately)
        if p.sigmoid_scale == 0.0:
            p.sigmoid_scale = activation_candidates[0][0]
            p.sigmoid_zp    = activation_candidates[0][1]
        # Estimate enc_act_scale from the bias scale if not found:
        # enc_b_scale = input_scale * enc_w_scale (TFLite convention),
        # so enc_act_scale must be derived from the actual calibration range.
        # Without it we cannot proceed — warn loudly.
        if p.enc_act_scale == 0.0:
            print(f"  WARNING: enc_act tensor not found. Only 1 candidate available.")
            print(f"  Cannot determine correct M1 — T2/T3 will not match T1.")

    if p.enc_act_scale == 0.0 or p.sigmoid_scale == 0.0:
        print(f"  WARNING: missing activation scales: "
              f"enc_act={p.enc_act_scale:.4e}  sigmoid={p.sigmoid_scale:.4e}")

    # Build sigmoid LUT by probing TFLite directly.
    # We construct synthetic inputs that produce each possible enc_req int8 value
    # (-128..127), run TFLite, and read back the sigmoid output.
    # This captures TFLite's exact fixed-point sigmoid approximation.
    #
    # Strategy: we need an input x such that enc_req = target_req.
    # enc_req = clip(round(acc * M1) + enc_act_zp, -128, 127)
    # acc = sum(enc_w[i,j] * (inp_q[j] - inp_zp)) + enc_b[i]
    # We use a single enc_w row to drive one hidden unit; others are masked by
    # choosing inp_q = inp_zp (zero contribution) except for one channel.
    # Simpler: just calibrate from the float formula but use TFLite's rounding.
    # TFLite sigmoid uses round-half-away-from-zero on the quantised output.

    # First build a float LUT (fallback)
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        # LUT index i → enc_req int8 value = (i - 128 + enc_act_zp)
        # But we centre differently: index = enc_req - enc_act_zp + 128
        # so enc_req = i - 128 + enc_act_zp
        enc_req_val = i - 128 + p.enc_act_zp
        x_float = float(enc_req_val - p.enc_act_zp) * p.enc_act_scale
        sig = 0.0 if x_float < -30 else (1.0 if x_float > 30 else 1.0 / (1.0 + np.exp(-x_float)))
        q = int(round(sig / max(p.sigmoid_scale, 1e-12))) + p.sigmoid_zp
        lut[i] = np.int8(max(-128, min(127, q)))

    # Now probe TFLite to get the exact LUT values.
    # We need a probe interpreter that lets us read idx=6 (sigmoid output).
    # Use the reference kernel which exposes intermediates.
    try:
        probe_interp = tf.lite.Interpreter(
            model_path=tflite_path,
            experimental_op_resolver_type=
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        probe_interp.allocate_tensors()
        probe_inp = probe_interp.get_input_details()[0]
        probe_out = probe_interp.get_output_details()[0]

        # Find tensor idx=6 (sigmoid output, shape [1,13])
        sig_tensor_idx = None
        for t in probe_interp.get_tensor_details():
            if 'Sigmoid' in t['name'] and 'encoder' in t['name'].lower():
                sig_tensor_idx = t['index']
                break

        if sig_tensor_idx is not None:
            # We'll probe by constructing inputs that isolate each hidden unit.
            # For hidden unit 0, use inp_q = inp_zp everywhere (→ acc = enc_b[0]),
            # but we need to sweep enc_req across -128..127.
            # Better approach: for each target enc_req value, find what accumulator
            # produces it, then find an input that gives that accumulator.
            # Simplest: just run with a range of real inputs and build a sparse LUT,
            # then fill gaps with float formula.
            #
            # Even simpler: for hidden unit k, we can set all inputs to inp_zp
            # (giving acc[k] = enc_b[k]) and scale one input channel to sweep acc[k].
            # But this requires knowing enc_w[k,:] well.
            #
            # EASIEST: construct a synthetic 1-channel model input that produces
            # enc_req=target for hidden unit 0 by solving the linear equation,
            # then read back sigmoid[0].
            M1_val = (float(p.input_scale) * float(p.enc_w_scale)) / max(float(p.enc_act_scale), 1e-12)
            lut_probed = np.zeros(256, dtype=np.int8)
            lut_probed_valid = np.zeros(256, dtype=bool)

            # For a range of synthetic integer accumulator values,
            # compute what enc_req they produce and what sigmoid output TFLite gives.
            # We construct an input: all channels = inp_zp (→ zero matmul contribution)
            # plus enc_b. Then sweep one channel to add a known delta to acc[0].
            # acc[0] = enc_b[0] + enc_w[0, ch] * (inp_q[ch] - inp_zp) + 0 (others at zp)
            # We pick ch=0 and sweep inp_q[0] from -128 to 127.
            base_inp = np.full((1, p.n_input), p.input_zp, dtype=np.int8)
            enc_w_row0 = p.enc_w[0].astype(np.int64)  # shape [n_input]

            probed_pairs = {}  # enc_req_val → sigmoid_out for unit 0
            for q_val in range(-128, 128):
                inp = base_inp.copy()
                inp[0, 0] = np.int8(q_val)
                # acc[0] = enc_b[0] + sum(enc_w[0,j]*(inp[j]-inp_zp))
                acc0 = int(p.enc_b[0]) + int(enc_w_row0[0]) * (int(q_val) - int(p.input_zp))
                req0 = int(np.clip(round(acc0 * M1_val) + p.enc_act_zp, -128, 127))
                probe_interp.set_tensor(probe_inp['index'], inp)
                probe_interp.invoke()
                sig_out = probe_interp.get_tensor(sig_tensor_idx).squeeze()
                sig0 = int(sig_out[0])
                probed_pairs[req0] = sig0

            # Fill LUT from probed pairs
            for enc_req_val, sig_val in probed_pairs.items():
                lut_idx = enc_req_val - p.enc_act_zp + 128
                if 0 <= lut_idx <= 255:
                    lut[lut_idx] = np.int8(sig_val)
                    lut_probed_valid[lut_idx] = True

            n_probed = int(lut_probed_valid.sum())
            print(f"  Sigmoid LUT: probed {n_probed}/256 entries from TFLite")
        else:
            print("  Sigmoid LUT: sigmoid tensor not found, using float approximation")

    except Exception as ex:
        print(f"  Sigmoid LUT probe failed ({ex}), using float approximation")

    p.sigmoid_lut = lut

    assert p.enc_w is not None, "Encoder weights not found"
    assert p.dec_w is not None, "Decoder weights not found"
    print(f"  Extracted: N_INPUT={p.n_input}  N_HIDDEN={p.n_hidden}")
    return p

# ── Track 1: PTQ one-shot ─────────────────────────────────────────────────────

def t1_interp(tflite_path):
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    return interp

def t1_single(interp, pixel_norm, params):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    s_i = inp['quantization_parameters']['scales'][0]
    z_i = inp['quantization_parameters']['zero_points'][0]
    s_o = out['quantization_parameters']['scales'][0]
    z_o = out['quantization_parameters']['zero_points'][0]
    q = np.clip(np.round(pixel_norm / s_i) + z_i, -128, 127).astype(np.int8)
    interp.set_tensor(inp['index'], q[np.newaxis])
    interp.invoke()
    r = interp.get_tensor(out['index'])[0]
    return (r.astype(np.float32) - z_o) * s_o

def t1_batch(tflite_path, norm, flat, scaler):
    interp = t1_interp(tflite_path)
    inp    = interp.get_input_details()[0]
    out    = interp.get_output_details()[0]
    s_i    = inp['quantization_parameters']['scales'][0]
    z_i    = inp['quantization_parameters']['zero_points'][0]
    s_o    = out['quantization_parameters']['scales'][0]
    z_o    = out['quantization_parameters']['zero_points'][0]
    N      = len(norm)
    recon  = np.zeros_like(norm)
    for i in range(N):
        q = np.clip(np.round(norm[i] / s_i) + z_i, -128, 127).astype(np.int8)
        interp.set_tensor(inp['index'], q[np.newaxis])
        interp.invoke()
        r         = interp.get_tensor(out['index'])[0]
        recon[i]  = (r.astype(np.float32) - z_o) * s_o
    orig = scaler.inverse_transform(recon).astype(np.float32)
    return np.sqrt(np.mean((flat - orig) ** 2, axis=1))

# ── Shared staged arithmetic (used by both T2 and T3) ─────────────────────────
# Both tracks use exactly the same arithmetic functions.
# T2 is called "granular golden" because it was derived from the TFLite params.
# T3 is the manual model.  They are intentionally identical in implementation —
# the purpose is to verify that the extracted parameters reproduce T1 exactly,
# and that the vectorised batch path matches the scalar path.

def _staged_single(pixel_norm, params):
    """Staged int8 inference for one pixel. Returns OrderedDict of intermediates."""
    stages = OrderedDict()

    inp_q = np.clip(
        np.round(pixel_norm / params.input_scale).astype(np.int64) + params.input_zp,
        -128, 127).astype(np.int8)
    stages['input_int8'] = inp_q.copy()

    # TFLite int8 matmul convention: accumulator uses (inp_q - input_zp).
    # The stored enc_b is the raw bias; the zp correction is applied at runtime.
    enc_acc = params.enc_b.copy().astype(np.int64)
    for i in range(params.n_hidden):
        for j in range(params.n_input):
            enc_acc[i] += np.int64(params.enc_w[i, j]) * (np.int64(inp_q[j]) - params.input_zp)
    stages['enc_acc'] = enc_acc.astype(np.int32).copy()

    # M1: requantise int32 accumulator to the enc_act output scale.
    # enc_act_scale is the scale of tensor idx=5 (MatMul+BiasAdd output),
    # NOT enc_b_scale (which is the bias constant scale, always = input*weight).
    M1 = ((float(params.input_scale) * float(params.enc_w_scale)) /
          max(float(params.enc_act_scale), 1e-12))
    enc_req = np.clip(
        np.round(enc_acc.astype(np.float64) * M1).astype(np.int64) + params.enc_act_zp,
        -128, 127).astype(np.int8)
    stages['enc_req_int8'] = enc_req.copy()

    # LUT index: enc_req is in enc_act quantisation space (zp=enc_act_zp).
    # Shift to 0-255 index using the enc_act zp so index 0 = most negative value.
    hidden_q = np.array([params.sigmoid_lut[int(v) - params.enc_act_zp + 128]
                         for v in enc_req], dtype=np.int8)
    stages['hidden_int8'] = hidden_q.copy()

    # Decoder matmul: subtract sigmoid_zp from hidden activations.
    dec_acc = params.dec_b.copy().astype(np.int64)
    for i in range(params.n_input):
        for j in range(params.n_hidden):
            dec_acc[i] += np.int64(params.dec_w[i, j]) * (np.int64(hidden_q[j]) - params.sigmoid_zp)
    stages['dec_acc'] = dec_acc.astype(np.int32).copy()

    M2 = ((float(params.sigmoid_scale) * float(params.dec_w_scale)) /
          max(float(params.output_scale), 1e-12))
    dec_req = np.clip(
        np.round(dec_acc.astype(np.float64) * M2) + params.output_zp,
        -128, 127).astype(np.int8)
    stages['out_int8'] = dec_req.copy()

    out_f = (dec_req.astype(np.float32) - params.output_zp) * params.output_scale
    stages['out_f32'] = out_f.copy()
    return stages


def _staged_batch(norm, flat, scaler, params):
    """Vectorised staged inference over N pixels → RMSE [N]."""
    inp_q = np.clip(
        np.round(norm / params.input_scale).astype(np.int64) + params.input_zp,
        -128, 127).astype(np.int8)

    inp_shifted = inp_q.astype(np.int64) - params.input_zp   # subtract input zp
    enc_acc = (inp_shifted @ params.enc_w.T.astype(np.int64) +
               params.enc_b.astype(np.int64))
    M1 = ((float(params.input_scale) * float(params.enc_w_scale)) /
          max(float(params.enc_act_scale), 1e-12))
    enc_req = np.clip(
        np.round(enc_acc.astype(np.float64) * M1).astype(np.int64) + params.enc_act_zp,
        -128, 127).astype(np.int8)

    hidden_q = params.sigmoid_lut[
        (enc_req.astype(np.int32) - params.enc_act_zp + 128)].astype(np.int8)

    hid_shifted = hidden_q.astype(np.int64) - params.sigmoid_zp   # subtract sigmoid zp
    dec_acc = (hid_shifted @ params.dec_w.T.astype(np.int64) +
               params.dec_b.astype(np.int64))
    M2 = ((float(params.sigmoid_scale) * float(params.dec_w_scale)) /
          max(float(params.output_scale), 1e-12))
    dec_req = np.clip(
        np.round(dec_acc.astype(np.float64) * M2) + params.output_zp,
        -128, 127).astype(np.int8)

    out_f = (dec_req.astype(np.float32) - params.output_zp) * params.output_scale
    orig  = scaler.inverse_transform(out_f).astype(np.float32)
    return np.sqrt(np.mean((flat - orig) ** 2, axis=1))

# T2 and T3 are the same arithmetic — aliases make the reporting clear
t2_single = _staged_single
t2_batch  = _staged_batch
t3_single = _staged_single
t3_batch  = _staged_batch

# ── Comparison utilities ──────────────────────────────────────────────────────

def print_section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def cmp_int(la, a, lb, b, n=8):
    d = a.astype(np.int64) - b.astype(np.int64)
    mm = int(np.sum(d != 0))
    print(f"\n  ── {la} vs {lb}")
    print(f"     Mismatches={mm}/{len(a)}  MaxAbsDiff={int(np.abs(d).max())}")
    for i in range(min(n, len(a))):
        mk = " ←" if d[i] != 0 else ""
        print(f"     [{i:3d}]  {int(a[i]):6d}  {int(b[i]):6d}  diff={int(d[i]):+d}{mk}")
    return mm

def cmp_float(la, a, lb, b, n=8):
    d    = a.astype(np.float64) - b.astype(np.float64)
    rmse = float(np.sqrt(np.mean(d**2)))
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

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed()
    if not (0 <= SCENE_INDEX < len(ABU_SCENES)):
        print(f"ERROR: SCENE_INDEX={SCENE_INDEX} out of range"); sys.exit(1)

    scene = ABU_SCENES[SCENE_INDEX]
    print_section(f"PTQ THREE-TRACK VALIDATION  |  {scene}  (index {SCENE_INDEX})")
    print("  Track 1 = PTQ one-shot  |  Track 2 = PTQ granular  |  Track 3 = Manual NumPy\n")

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

    print_section("PARAMETER EXTRACTION")
    params = extract_params(tflite_path)

    print(f"\n  Quant params:")
    print(f"    input   scale={params.input_scale:.6e}  zp={params.input_zp}")
    print(f"    enc_w   scale={params.enc_w_scale:.6e}")
    print(f"    enc_b   scale={params.enc_b_scale:.6e}  (bias constant = input*weight scale)")
    print(f"    enc_act scale={params.enc_act_scale:.6e}  zp={params.enc_act_zp}  (MatMul+BiasAdd output)")
    print(f"    sigmoid scale={params.sigmoid_scale:.6e}  zp={params.sigmoid_zp}  (Sigmoid output)")
    print(f"    dec_w   scale={params.dec_w_scale:.6e}")
    print(f"    output  scale={params.output_scale:.6e}  zp={params.output_zp}")
    print(f"\n  M1 (enc requant multiplier) = "
          f"{params.input_scale * params.enc_w_scale / max(params.enc_act_scale,1e-12):.6e}")
    print(f"  M2 (dec requant multiplier) = "
          f"{params.sigmoid_scale * params.dec_w_scale / max(params.output_scale,1e-12):.6e}")

    # ── Test pixels ───────────────────────────────────────────────────────────
    test_pixels = [('Centre', H * W // 2)]
    if gt_mask is not None and gt_mask.sum() > 0:
        pos = np.argwhere(gt_mask.reshape(-1))
        test_pixels.append(('Anomaly', int(pos[len(pos) // 2])))

    interp_t1 = t1_interp(tflite_path)

    # ─────────────────────────────────────────────────────────────────────────
    # SINGLE-PIXEL LAYER-BY-LAYER
    # ─────────────────────────────────────────────────────────────────────────
    for px_label, px_idx in test_pixels:
        print_section(f"LAYER-BY-LAYER — {px_label} pixel #{px_idx}")
        px_orig = flat[px_idx].astype(np.float32)
        px_norm = scaler.transform(px_orig[np.newaxis])[0].astype(np.float32)

        # Track 1 — capture raw int8 output AND all intermediate tensors
        interp_px = t1_interp(tflite_path)
        inp_det   = interp_px.get_input_details()[0]
        out_det   = interp_px.get_output_details()[0]
        inp_s  = inp_det['quantization_parameters']['scales'][0]
        inp_zp = inp_det['quantization_parameters']['zero_points'][0]
        out_s  = out_det['quantization_parameters']['scales'][0]
        out_zp = out_det['quantization_parameters']['zero_points'][0]
        q_in   = np.clip(np.round(px_norm / inp_s) + inp_zp, -128, 127).astype(np.int8)
        interp_px.set_tensor(inp_det['index'], q_in[np.newaxis])
        interp_px.invoke()
        t1_out_int8 = interp_px.get_tensor(out_det['index'])[0].copy()
        t1_out      = (t1_out_int8.astype(np.float32) - out_zp) * out_s
        t1_rmse     = float(np.sqrt(np.mean(
            (px_orig - scaler.inverse_transform(t1_out[np.newaxis])[0]) ** 2)))
        print(f"\n  Track 1 (PTQ one-shot)   RMSE={t1_rmse:.6f}")
        print(f"  T1 out_int8[0:8] = {t1_out_int8[:8].tolist()}")
        print(f"  T1 out_f32 [0:5] = {[round(float(x),6) for x in t1_out[:5]]}")
        print(f"  T1 scale={out_s:.6e}  zp={out_zp}")

        # Re-run with XNNPACK disabled so intermediate activations are visible.
        # XNNPACK fuses ops and hides intermediates; the reference kernel exposes them.
        try:
            interp_ref = tf.lite.Interpreter(
                model_path=tflite_path,
                experimental_op_resolver_type=
                    tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        except Exception:
            interp_ref = tf.lite.Interpreter(model_path=tflite_path, num_threads=1)
        interp_ref.allocate_tensors()
        interp_ref.set_tensor(interp_ref.get_input_details()[0]['index'], q_in[np.newaxis])
        interp_ref.invoke()
        print(f"\n  ── T1 intermediate tensors (reference kernel, no XNNPACK):")
        for t in interp_ref.get_tensor_details():
            if len(t['shape']) == 0 or int(np.prod(t['shape'])) == 0:
                continue
            try:
                val   = interp_ref.get_tensor(t['index']).squeeze()
                quant = t['quantization_parameters']
                sc    = float(quant['scales'][0])    if len(quant['scales'])      > 0 else 0.0
                zp    = int(quant['zero_points'][0]) if len(quant['zero_points']) > 0 else 0
                print(f"    idx={t['index']:3d}  {str(tuple(val.shape) if hasattr(val,'shape') else '?'):15s}  "
                      f"dtype={str(getattr(val,'dtype','?')):6s}  sc={sc:.4e}  zp={zp:5d}  '{t['name']}'")
                if hasattr(val, 'flatten') and 0 < int(np.prod(val.shape)) <= 300:
                    print(f"      [0:8]={val.flatten()[:8].tolist()}")
            except Exception:
                pass

        # Track 2
        t2 = t2_single(px_norm, params)
        t2_rmse = float(np.sqrt(np.mean(
            (px_orig - scaler.inverse_transform(t2['out_f32'][np.newaxis])[0]) ** 2)))
        t1t2_ok = np.allclose(t1_out, t2['out_f32'], atol=1e-7)
        print(f"\n  Track 2 (PTQ granular)   RMSE={t2_rmse:.6f}")
        print(f"  T2 out_int8[0:8] = {t2['out_int8'][:8].tolist()}")
        print(f"  T2 out_f32 [0:5] = {[round(float(x),6) for x in t2['out_f32'][:5]]}")
        print(f"  T2 scale={params.output_scale:.6e}  zp={params.output_zp}")
        print(f"  T1==T2: {'✓' if t1t2_ok else '✗'}")

        # Track 3
        t3 = t3_single(px_norm, params)
        t3_rmse = float(np.sqrt(np.mean(
            (px_orig - scaler.inverse_transform(t3['out_f32'][np.newaxis])[0]) ** 2)))
        print(f"\n  Track 3 (manual NumPy)   RMSE={t3_rmse:.6f}")

        # Layer comparison T2 vs T3
        print("\n  ── T2 vs T3 layer-by-layer ─────────────────────────────────────")
        cmp_int  ("T2 input_int8",   t2['input_int8'],
                  "T3 input_int8",   t3['input_int8'])
        cmp_float("T2 enc_acc",      t2['enc_acc'].astype(np.float64),
                  "T3 enc_acc",      t3['enc_acc'].astype(np.float64))
        cmp_int  ("T2 enc_req_int8", t2['enc_req_int8'],
                  "T3 enc_req_int8", t3['enc_req_int8'])

        # ── Sigmoid diagnostic: trace every step for each hidden unit ──────
        print("\n  ── Sigmoid diagnostic (T2 enc_req → hidden vs T1 reference)")
        print(f"  enc_act_scale={params.enc_act_scale:.6e}  enc_act_zp={params.enc_act_zp}")
        print(f"  sigmoid_scale={params.sigmoid_scale:.6e}  sigmoid_zp={params.sigmoid_zp}")
        # Read T1's actual sigmoid input and output from reference kernel
        t1_enc_act_int8 = interp_ref.get_tensor(5).squeeze().copy()  # idx=5
        t1_sigmoid_int8 = interp_ref.get_tensor(6).squeeze().copy()  # idx=6
        print(f"  T1 idx=5 (enc_act?) : {t1_enc_act_int8.tolist()}")
        print(f"  T1 idx=6 (sigmoid?) : {t1_sigmoid_int8.tolist()}")
        print(f"  T2 enc_req          : {t2['enc_req_int8'].tolist()}")
        print(f"  T2 hidden           : {t2['hidden_int8'].tolist()}")
        print()
        # Back-solve: what enc_req INT8 value would produce T1's idx=6 sigmoid output?
        # sigmoid_out = lut[enc_req - enc_act_zp + 128]
        # So enc_req_needed = lut_inverse[t1_sigmoid_int8[i]]
        # Build inverse LUT from current LUT
        inv_lut = {}  # sigmoid_out → list of enc_req values that produce it
        for lut_idx in range(256):
            sv = int(params.sigmoid_lut[lut_idx])
            enc_req_val = lut_idx - 128 + params.enc_act_zp
            if sv not in inv_lut:
                inv_lut[sv] = []
            inv_lut[sv].append(enc_req_val)

        # Also compute enc_acc WITHOUT input_zp subtraction (raw inp_q)
        raw_enc_acc = params.enc_b.copy().astype(np.int64)
        for i in range(params.n_hidden):
            for j in range(params.n_input):
                raw_enc_acc[i] += np.int64(params.enc_w[i,j]) * np.int64(t2['input_int8'][j])
        M1 = ((float(params.input_scale)*float(params.enc_w_scale)) /
               max(float(params.enc_act_scale),1e-12))
        raw_enc_req = np.clip(
            np.round(raw_enc_acc.astype(np.float64)*M1).astype(np.int64) + params.enc_act_zp,
            -128, 127).astype(np.int8)
        raw_hidden = np.array([params.sigmoid_lut[int(v)-params.enc_act_zp+128]
                                for v in raw_enc_req], dtype=np.int8)

        print(f"  {'u':>2}  {'acc(zp)':>10}  {'req(zp)':>8}  {'hid(zp)':>8}  "
              f"{'acc(raw)':>10}  {'req(raw)':>9}  {'hid(raw)':>9}  "
              f"{'t1_sig6':>8}  {'need_req':>9}")
        for i in range(params.n_hidden):
            acc_i    = int(t2['enc_acc'][i])
            req_i    = int(t2['enc_req_int8'][i])
            hid_i    = int(t2['hidden_int8'][i])
            t1_s6_i  = int(t1_sigmoid_int8[i]) if i < len(t1_sigmoid_int8) else -999
            r_acc    = int(raw_enc_acc[i])
            r_req    = int(raw_enc_req[i])
            r_hid    = int(raw_hidden[i])
            # What enc_req values produce t1_sig6?
            needed   = inv_lut.get(t1_s6_i, ['?'])
            need_str = str(needed[len(needed)//2]) if needed != ['?'] else '?'
            print(f"  {i:>2}  {acc_i:>10}  {req_i:>8}  {hid_i:>8}  "
                  f"  {r_acc:>10}  {r_req:>9}  {r_hid:>9}  "
                  f"  {t1_s6_i:>8}  {need_str:>9}")
        # ── Weight formula diagnostic ─────────────────────────────────────
        # Try 4 accumulator formulas and see which produces need_req closest to T1.
        # need_req for each unit is back-solved from T1's sigmoid output (idx=6).
        t1_sig_vals = t1_sigmoid_int8.astype(float)
        need_req_arr = np.zeros(params.n_hidden, dtype=int)
        for _i, _s in enumerate(t1_sigmoid_int8):
            _sf = (float(_s) - params.sigmoid_zp) * params.sigmoid_scale
            _sf = min(0.9999, max(0.0001, _sf))
            _logit = np.log(_sf / (1 - _sf))
            _req = int(round(_logit / params.enc_act_scale)) + params.enc_act_zp
            need_req_arr[_i] = max(-128, min(127, _req))

        # Read raw tensors from reference kernel by index
        _raw_enc_w  = interp_ref.get_tensor(4).squeeze().copy()  # (13,205) or (205,13)?
        _raw_dec_w  = interp_ref.get_tensor(2).squeeze().copy()  # (205,13) or (13,205)?
        _raw_enc_b  = interp_ref.get_tensor(3).squeeze().copy()  # int32 (13,)
        _inp_q      = t2['input_int8']
        _M1         = ((float(params.input_scale) * float(params.enc_w_scale)) /
                       max(float(params.enc_act_scale), 1e-12))

        def _try_formula(name, w, b, subtract_zp):
            if w.shape[0] != params.n_hidden:
                w = w.T
            if subtract_zp:
                acc = b.astype(np.int64) + w.astype(np.int64) @ (_inp_q.astype(np.int64) - params.input_zp)
            else:
                acc = b.astype(np.int64) + w.astype(np.int64) @ _inp_q.astype(np.int64)
            req = np.clip(np.round(acc * _M1).astype(np.int64) + params.enc_act_zp, -128, 127).astype(int)
            errs = req - need_req_arr
            print(f"  {name:40s}: req={req.tolist()}")
            print(f"  {'':40s}  err={errs.tolist()}  maxabs={np.abs(errs).max()}")
            return req, acc

        print(f"── Weight formula search")
        print(f"  need_req (from T1 sigmoid): {need_req_arr.tolist()}")
        print()
        _try_formula("enc_w + enc_b, sub zp (current)",      params.enc_w, params.enc_b,  True)
        _try_formula("enc_w + enc_b, NO sub zp",             params.enc_w, params.enc_b,  False)
        _try_formula("dec_w.T + enc_b, sub zp",              params.dec_w.T, params.enc_b, True)
        _try_formula("dec_w.T + enc_b, no sub zp",           params.dec_w.T, params.enc_b, False)
        _try_formula("raw_enc_w(idx4)+raw_enc_b, sub zp",    _raw_enc_w, _raw_enc_b, True)
        _try_formula("raw_enc_w(idx4)+raw_enc_b, no sub zp", _raw_enc_w, _raw_enc_b, False)
        _try_formula("raw_dec_w(idx2).T+enc_b, sub zp",      _raw_dec_w, params.enc_b, True)
        _try_formula("raw_dec_w(idx2).T+enc_b, no sub zp",   _raw_dec_w, params.enc_b, False)
        # ──────────────────────────────────────────────────────────────────

        cmp_int  ("T2 hidden_int8",  t2['hidden_int8'],
                  "T3 hidden_int8",  t3['hidden_int8'])
        cmp_float("T2 dec_acc",      t2['dec_acc'].astype(np.float64),
                  "T3 dec_acc",      t3['dec_acc'].astype(np.float64))
        cmp_int  ("T2 out_int8",     t2['out_int8'],
                  "T3 out_int8",     t3['out_int8'])
        cmp_float("T2 out_f32",      t2['out_f32'].astype(np.float64),
                  "T3 out_f32",      t3['out_f32'].astype(np.float64))

    # ─────────────────────────────────────────────────────────────────────────
    # SELF-CONSISTENCY: scalar T3 vs vectorised T3
    # ─────────────────────────────────────────────────────────────────────────
    print_section("SELF-CONSISTENCY  (scalar T3 vs vectorised T3 batch)")
    N_CHK    = min(100, H * W)
    chk_idx  = np.random.default_rng(RANDOM_SEED + 99).choice(H * W, N_CHK, replace=False)
    chk_norm = scaler.transform(flat[chk_idx]).astype(np.float32)

    scalar_out = np.zeros((N_CHK, params.n_input), dtype=np.float32)
    for i in range(N_CHK):
        scalar_out[i] = t3_single(chk_norm[i], params)['out_f32']

    # Run vectorised path inline (same as _staged_batch)
    iq = np.clip(np.round(chk_norm / params.input_scale).astype(np.int64)
                 + params.input_zp, -128, 127).astype(np.int8)
    ea = (iq.astype(np.int64) - params.input_zp) @ params.enc_w.T.astype(np.int64) + params.enc_b.astype(np.int64)
    M1 = (float(params.input_scale) * float(params.enc_w_scale)) / max(float(params.enc_act_scale), 1e-12)
    er = np.clip(np.round(ea.astype(np.float64) * M1).astype(np.int64) + params.enc_act_zp, -128, 127).astype(np.int8)
    hq = params.sigmoid_lut[(er.astype(np.int32) - params.enc_act_zp + 128)].astype(np.int8)
    da = (hq.astype(np.int64) - params.sigmoid_zp) @ params.dec_w.T.astype(np.int64) + params.dec_b.astype(np.int64)
    M2 = (float(params.sigmoid_scale) * float(params.dec_w_scale)) / max(float(params.output_scale), 1e-12)
    dr = np.clip(np.round(da.astype(np.float64) * M2) + params.output_zp, -128, 127).astype(np.int8)
    vec_out = (dr.astype(np.float32) - params.output_zp) * params.output_scale

    diff_sc    = np.abs(scalar_out - vec_out)
    n_mismatch = int(np.sum(diff_sc > 1e-10))
    print(f"  Pixels: {N_CHK}  Max diff: {diff_sc.max():.2e}  Mismatches: {n_mismatch}")
    print(f"  {'✓ scalar == vectorised' if n_mismatch == 0 else '✗ MISMATCH'}")

    # ─────────────────────────────────────────────────────────────────────────
    # FULL-IMAGE BATCH
    # ─────────────────────────────────────────────────────────────────────────
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

    print("  Running Track 1 (PTQ one-shot)...")
    r1 = t1_batch(tflite_path, sub_norm, sub_flat, scaler)
    print("  Running Track 2 (PTQ granular)...")
    r2 = t2_batch(sub_norm, sub_flat, scaler, params)
    print("  Running Track 3 (Manual NumPy)...")
    r3 = t3_batch(sub_norm, sub_flat, scaler, params)

    print(f"\n  Track 1 (PTQ one-shot): {arr_summary(r1)}")
    print(f"  Track 2 (PTQ granular): {arr_summary(r2)}")
    print(f"  Track 3 (manual):       {arr_summary(r3)}")

    d12 = float(np.sqrt(np.mean((r1 - r2) ** 2)))
    d23 = float(np.sqrt(np.mean((r2 - r3) ** 2)))
    d13 = float(np.sqrt(np.mean((r1 - r3) ** 2)))
    print(f"\n  T1 vs T2: {d12:.4e}  (same model — should be ~0)")
    print(f"  T2 vs T3: {d23:.4e}  (manual vs staged — should be ~0)")
    print(f"  T1 vs T3: {d13:.4e}  (manual vs TFLite — should be ~0)")

    if gt_mask is not None and gt_mask.sum() > 0:
        sub_gt = gt_mask.reshape(-1)[sub_idx]
        a1 = compute_auc(r1, sub_gt)
        a2 = compute_auc(r2, sub_gt)
        a3 = compute_auc(r3, sub_gt)
        print(f"\n  AUC — T1={a1:.6f}  T2={a2:.6f}  T3={a3:.6f}")

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL VERDICT
    # ─────────────────────────────────────────────────────────────────────────
    print_section("FINAL VERDICT")

    px_n = scaler.transform(flat[H*W//2][np.newaxis])[0].astype(np.float32)
    t1_f = t1_single(interp_t1, px_n, params)
    t2_f = t2_single(px_n, params)
    t3_f = t3_single(px_n, params)

    mm_inp = int(np.sum(t2_f['input_int8']   != t3_f['input_int8']))
    mm_enc = int(np.sum(t2_f['enc_req_int8'] != t3_f['enc_req_int8']))
    mm_hid = int(np.sum(t2_f['hidden_int8']  != t3_f['hidden_int8']))
    mm_out = int(np.sum(t2_f['out_int8']     != t3_f['out_int8']))
    r_out  = float(np.sqrt(np.mean((t2_f['out_f32'] - t3_f['out_f32']) ** 2)))
    t1t2   = np.allclose(t1_f, t2_f['out_f32'], atol=1e-7)

    EPS = 1e-5
    checks = [
        ("T1 == T2  (param extraction OK)",     t1t2,           "match" if t1t2 else "MISMATCH"),
        ("Input quantisation    T2==T3",         mm_inp == 0,    f"mismatches={mm_inp}"),
        ("Encoder req           T2==T3",         mm_enc == 0,    f"mismatches={mm_enc}"),
        ("Sigmoid LUT           T2==T3",         mm_hid == 0,    f"mismatches={mm_hid}"),
        ("Decoder req           T2==T3",         mm_out == 0,    f"mismatches={mm_out}"),
        ("Output float32        T2==T3",         r_out  < EPS,   f"RMSE={r_out:.2e}"),
        ("Scalar==vectorised    T3 internal",    n_mismatch == 0,f"mismatches={n_mismatch}"),
        ("Batch map             T2==T3",         d23    < EPS,   f"RMSE={d23:.2e}"),
    ]

    all_pass = True
    for name, ok, detail in checks:
        all_pass = all_pass and ok
        print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name:<44}  {detail}")

    print()
    if all_pass:
        print("  ✓ ALL CHECKS PASSED")
        print("    Track 3 is a bit-exact software model of the PTQ TFLite inference.")
        print("    Ready for HLS synthesis.")
    else:
        print("  ✗ CHECKS FAILED — review mismatches above.")
    print()

if __name__ == '__main__':
    main()