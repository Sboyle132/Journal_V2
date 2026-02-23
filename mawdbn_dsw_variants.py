"""
mawdbn_dsw_variants.py
Two new DSW variants alongside the original compute_beta_map().
Import from python_DBN_Qkeras.py:
    from mawdbn_dsw_variants import *
Then call run_scene_variants() instead of run_scene().
Requires run_scene() to store codes_none and codes_quant in result dict.
"""
import numpy as np
from tqdm import tqdm
from python_DBN_Qkeras import *

def compute_beta_map_sq(R_mse, C_codes):
    """Variant A: squared error + squared code distance. No sqrt anywhere."""
    half_out = DSW_WOUTER // 2
    H, W = R_mse.shape
    beta = np.zeros((H, W), dtype=np.float32)
    for j in tqdm(range(H), desc="  DSW-A", leave=False):
        for i in range(W):
            r0,r1 = max(j-half_out,0), min(j+half_out+1,H)
            c0,c1 = max(i-half_out,0), min(i+half_out+1,W)
            win_R = R_mse[r0:r1, c0:c1]
            win_C = C_codes[r0:r1, c0:c1]
            cj, ci = j-r0, i-c0
            ri  = np.arange(win_R.shape[0])[:,None]
            ci_ = np.arange(win_R.shape[1])[None,:]
            ring = (ri!=cj)|(ci_!=ci)
            rn = win_R[ring]
            if not rn.size: continue
            mu, sig = rn.mean(), rn.std()
            inlier = np.abs(rn - mu) <= sig
            wt = np.where(inlier, R_mse[j,i] / (rn + 1e-12), 0.0)
            c_p = C_codes[j,i].astype(np.float32)
            d_sq = np.sum((win_C[ring].astype(np.float32) - c_p)**2, axis=1)
            beta[j,i] = np.sum(wt * d_sq) / rn.size
    return beta


def compute_beta_map_sqrtre_sqdist(R_mse, C_codes):
    """Variant B: RMSE (sqrt of MSE) + squared code distance."""
    half_out = DSW_WOUTER // 2
    R = np.sqrt(R_mse).astype(np.float32)
    H, W = R.shape
    beta = np.zeros((H, W), dtype=np.float32)
    for j in tqdm(range(H), desc="  DSW-B", leave=False):
        for i in range(W):
            r0,r1 = max(j-half_out,0), min(j+half_out+1,H)
            c0,c1 = max(i-half_out,0), min(i+half_out+1,W)
            win_R = R[r0:r1, c0:c1]
            win_C = C_codes[r0:r1, c0:c1]
            cj, ci = j-r0, i-c0
            ri  = np.arange(win_R.shape[0])[:,None]
            ci_ = np.arange(win_R.shape[1])[None,:]
            ring = (ri!=cj)|(ci_!=ci)
            rn = win_R[ring]
            if not rn.size: continue
            mu, sig = rn.mean(), rn.std()
            inlier = np.abs(rn - mu) <= sig
            wt = np.where(inlier, R[j,i] / (rn + 1e-12), 0.0)
            c_p = C_codes[j,i].astype(np.float32)
            d_sq = np.sum((win_C[ring].astype(np.float32) - c_p)**2, axis=1)
            beta[j,i] = np.sum(wt * d_sq) / rn.size
    return beta

def get_int8_codes_tflite(tflite_ae_path, image, scaler):
    """
    Run the TFLite encoder and return raw int8 codes — no dequantisation.
    The zero-point is subtracted so values are centred (zero-point removed)
    but scale is NOT applied, keeping everything in integer space.
    Returns (H, W, n_codes) int8 array.
    """
    import tensorflow as tf
    H, W, C = image.shape
    pixels  = image.reshape(-1, C)
    norm, _ = preprocess_pixels(pixels, fit_scaler=False, scaler=scaler)

    enc_path = tflite_ae_path.replace('_ptq_int8.tflite', '_ptq_int8_encoder.tflite')
    interp   = tf.lite.Interpreter(model_path=enc_path)
    interp.allocate_tensors()
    inp_d  = interp.get_input_details()[0]
    out_d  = interp.get_output_details()[0]
    in_s, in_zp   = inp_d['quantization']
    _, out_zp     = out_d['quantization']   # scale intentionally ignored

    codes = []
    for px in norm:
        q = np.clip(np.round(px / in_s) + in_zp, -128, 127).astype(np.int8)
        interp.set_tensor(inp_d['index'], q[np.newaxis])
        interp.invoke()
        raw = interp.get_tensor(out_d['index'])[0]       # int8
        codes.append((raw.astype(np.int16) - out_zp).astype(np.int16))  # zp-subtracted

    codes = np.array(codes, dtype=np.int16)
    n_codes = codes.shape[1]
    return codes.reshape(H, W, n_codes)


def compute_beta_map_int8codes(R_mse, C_int8):
    """
    Variant C: RMSE (sqrt of MSE) + squared integer code distance.
    Codes are raw int8 (zero-point subtracted), never dequantised.
    The scale factor on code distances is constant across all pixels
    so it cancels in the relative weight computation — AUC is unaffected
    by it, but absolute beta values will differ from float variants.
    """
    half_out = DSW_WOUTER // 2
    R = np.sqrt(R_mse).astype(np.float32)
    H, W = R.shape
    beta = np.zeros((H, W), dtype=np.float32)
    for j in tqdm(range(H), desc="  DSW-C", leave=False):
        for i in range(W):
            r0,r1 = max(j-half_out,0), min(j+half_out+1,H)
            c0,c1 = max(i-half_out,0), min(i+half_out+1,W)
            win_R  = R[r0:r1, c0:c1]
            win_C  = C_int8[r0:r1, c0:c1].astype(np.float32)
            cj, ci = j-r0, i-c0
            ri  = np.arange(win_R.shape[0])[:,None]
            ci_ = np.arange(win_R.shape[1])[None,:]
            ring = (ri!=cj)|(ci_!=ci)
            rn = win_R[ring]
            if not rn.size: continue
            mu, sig = rn.mean(), rn.std()
            inlier = np.abs(rn - mu) <= sig
            wt  = np.where(inlier, R[j,i] / (rn + 1e-12), 0.0)
            c_p = win_C[cj, ci]
            d_sq = np.sum((win_C[ring] - c_p)**2, axis=1)
            beta[j,i] = np.sum(wt * d_sq) / rn.size
    return beta


# ═══════════════════════════════════════════════════════════════════════════════
# Variant D — fully integer
#   - Input requantised into output scale space
#   - Squared error computed in that integer space
#   - int8 codes (zero-point subtracted)
# ═══════════════════════════════════════════════════════════════════════════════

def get_int8_mse_and_codes_tflite(tflite_ae_path, image, scaler):
    """
    Run full AE and encoder in integer space.

    Steps per pixel:
      1. normalise (float scaler)
      2. quantise → inp_q  (input scale)
      3. run AE   → recon_q (output scale)
      4. requantise inp_q into output scale:
             inp_req = clip(round((inp_q - in_zp) * in_s / out_s) + out_zp)
      5. squared error: mean((inp_req - recon_q)^2)  — integer space
      6. encoder codes: zp-subtracted int8

    Returns
    -------
    mse_map   : (H, W) float32   integer squared error per pixel
    codes_map : (H, W, n_codes)  int16  zero-point-subtracted codes
    """
    import tensorflow as tf
    H, W, C = image.shape
    pixels  = image.reshape(-1, C)
    norm, _ = preprocess_pixels(pixels, fit_scaler=False, scaler=scaler)

    interp_ae = tf.lite.Interpreter(model_path=tflite_ae_path)
    interp_ae.allocate_tensors()
    inp_d = interp_ae.get_input_details()[0]
    out_d = interp_ae.get_output_details()[0]
    in_s,  in_zp  = inp_d['quantization']
    out_s, out_zp = out_d['quantization']

    enc_path = tflite_ae_path.replace('_ptq_int8.tflite', '_ptq_int8_encoder.tflite')
    interp_enc = tf.lite.Interpreter(model_path=enc_path)
    interp_enc.allocate_tensors()
    enc_inp_d = interp_enc.get_input_details()[0]
    enc_out_d = interp_enc.get_output_details()[0]
    enc_in_s, enc_in_zp = enc_inp_d['quantization']
    _,        enc_out_zp = enc_out_d['quantization']

    mse_list   = []
    codes_list = []

    for px_norm in norm:
        # quantise input to AE input scale
        inp_q = np.clip(np.round(px_norm / in_s) + in_zp, -128, 127).astype(np.int8)

        # run AE → int8 reconstruction in output scale
        interp_ae.set_tensor(inp_d['index'], inp_q[np.newaxis])
        interp_ae.invoke()
        recon_q = interp_ae.get_tensor(out_d['index'])[0]  # int8, output scale

        # requantise input into output scale space
        inp_req = np.clip(
            np.round((inp_q.astype(np.float32) - in_zp) * in_s / out_s) + out_zp,
            -128, 127
        ).astype(np.int8)

        # integer squared error in output scale space
        diff = inp_req.astype(np.int32) - recon_q.astype(np.int32)
        mse_list.append(float(np.mean(diff ** 2)))

        # encoder: int8 codes, zp subtracted
        enc_q = np.clip(np.round(px_norm / enc_in_s) + enc_in_zp, -128, 127).astype(np.int8)
        interp_enc.set_tensor(enc_inp_d['index'], enc_q[np.newaxis])
        interp_enc.invoke()
        raw_code = interp_enc.get_tensor(enc_out_d['index'])[0]
        codes_list.append((raw_code.astype(np.int16) - enc_out_zp))

    mse_map   = np.array(mse_list,   dtype=np.float32).reshape(H, W)
    codes_map = np.array(codes_list, dtype=np.int16).reshape(H, W, -1)
    return mse_map, codes_map


def compute_beta_map_int8_full(R_mse_int, C_int8):
    """Variant D: integer MSE + integer code distance. RMSE applied for inlier test."""
    half_out = DSW_WOUTER // 2
    R = np.sqrt(R_mse_int).astype(np.float32)
    H, W = R.shape
    beta = np.zeros((H, W), dtype=np.float32)
    for j in tqdm(range(H), desc="  DSW-D", leave=False):
        for i in range(W):
            r0,r1 = max(j-half_out,0), min(j+half_out+1,H)
            c0,c1 = max(i-half_out,0), min(i+half_out+1,W)
            win_R = R[r0:r1, c0:c1]
            win_C = C_int8[r0:r1, c0:c1].astype(np.float32)
            cj, ci = j-r0, i-c0
            ri  = np.arange(win_R.shape[0])[:,None]
            ci_ = np.arange(win_R.shape[1])[None,:]
            ring = (ri!=cj)|(ci_!=ci)
            rn = win_R[ring]
            if not rn.size: continue
            mu, sig = rn.mean(), rn.std()
            inlier = np.abs(rn - mu) <= sig
            wt  = np.where(inlier, R[j,i] / (rn + 1e-12), 0.0)
            c_p  = win_C[cj, ci]
            d_sq = np.sum((win_C[ring] - c_p)**2, axis=1)
            beta[j,i] = np.sum(wt * d_sq) / rn.size
    return beta

def get_int8_mse_direct_and_codes_tflite(tflite_ae_path, image, scaler):
    """
    Variant E: quantise the normalised float input DIRECTLY into output scale
    space — no intermediate input-scale quantisation, no requantisation step.
    Both sides of the error comparison are in identical integer units with no
    intermediate rounding from scale conversion.

    Steps per pixel:
      1. normalise (float scaler)
      2. run AE with input scale quantisation → recon_q  (model unaffected)
      3. quantise float DIRECTLY to output scale → inp_req
      4. squared error: mean((inp_req - recon_q)^2)
      5. encoder codes: zp-subtracted int8
    """
    import tensorflow as tf
    H, W, C = image.shape
    pixels  = image.reshape(-1, C)
    norm, _ = preprocess_pixels(pixels, fit_scaler=False, scaler=scaler)

    interp_ae = tf.lite.Interpreter(model_path=tflite_ae_path)
    interp_ae.allocate_tensors()
    inp_d = interp_ae.get_input_details()[0]
    out_d = interp_ae.get_output_details()[0]
    in_s,  in_zp  = inp_d['quantization']
    out_s, out_zp = out_d['quantization']

    enc_path = tflite_ae_path.replace('_ptq_int8.tflite', '_ptq_int8_encoder.tflite')
    interp_enc = tf.lite.Interpreter(model_path=enc_path)
    interp_enc.allocate_tensors()
    enc_inp_d = interp_enc.get_input_details()[0]
    enc_out_d = interp_enc.get_output_details()[0]
    enc_in_s, enc_in_zp = enc_inp_d['quantization']
    _,        enc_out_zp = enc_out_d['quantization']

    mse_list   = []
    codes_list = []

    for px_norm in norm:
        # run AE with correct input-scale quantisation
        inp_q = np.clip(np.round(px_norm / in_s) + in_zp, -128, 127).astype(np.int8)
        interp_ae.set_tensor(inp_d['index'], inp_q[np.newaxis])
        interp_ae.invoke()
        recon_q = interp_ae.get_tensor(out_d['index'])[0]  # int8, output scale

        # quantise float DIRECTLY to output scale — no requantisation rounding
        inp_req = np.clip(np.round(px_norm / out_s) + out_zp, -128, 127).astype(np.int8)

        diff = inp_req.astype(np.int32) - recon_q.astype(np.int32)
        mse_list.append(float(np.mean(diff ** 2)))

        # encoder codes
        enc_q = np.clip(np.round(px_norm / enc_in_s) + enc_in_zp, -128, 127).astype(np.int8)
        interp_enc.set_tensor(enc_inp_d['index'], enc_q[np.newaxis])
        interp_enc.invoke()
        raw_code = interp_enc.get_tensor(enc_out_d['index'])[0]
        codes_list.append((raw_code.astype(np.int16) - enc_out_zp))

    mse_map   = np.array(mse_list,   dtype=np.float32).reshape(H, W)
    codes_map = np.array(codes_list, dtype=np.int16).reshape(H, W, -1)
    return mse_map, codes_map

def compute_beta_map_int8_full(R_mse_int, C_int8):
    """Variant D: integer MSE + integer code distance. RMSE applied for inlier test."""
    half_out = DSW_WOUTER // 2
    R = np.sqrt(R_mse_int).astype(np.float32)
    H, W = R.shape
    beta = np.zeros((H, W), dtype=np.float32)
    for j in tqdm(range(H), desc="  DSW-D", leave=False):
        for i in range(W):
            r0,r1 = max(j-half_out,0), min(j+half_out+1,H)
            c0,c1 = max(i-half_out,0), min(i+half_out+1,W)
            win_R = R[r0:r1, c0:c1]
            win_C = C_int8[r0:r1, c0:c1].astype(np.float32)
            cj, ci = j-r0, i-c0
            ri  = np.arange(win_R.shape[0])[:,None]
            ci_ = np.arange(win_R.shape[1])[None,:]
            ring = (ri!=cj)|(ci_!=ci)
            rn = win_R[ring]
            if not rn.size: continue
            mu, sig = rn.mean(), rn.std()
            inlier = np.abs(rn - mu) <= sig
            wt  = np.where(inlier, R[j,i] / (rn + 1e-12), 0.0)
            c_p  = win_C[cj, ci]
            d_sq = np.sum((win_C[ring] - c_p)**2, axis=1)
            beta[j,i] = np.sum(wt * d_sq) / rn.size
    return beta

def run_scene_variants(scene_name, mat_path):
    result = run_scene(scene_name, mat_path)

    codes_none  = result['codes_none']
    codes_quant = result['codes_quant']
    mse_none    = result['rmse_none']  ** 2
    mse_quant   = result['rmse_quant'] ** 2

    # Rebuild scaler identically to run_scene
    image = result['image']
    H, W, C = image.shape
    n_train  = min(N_SAMPLES_TRAIN, H * W) if N_SAMPLES_TRAIN else H * W
    rng      = np.random.default_rng(RANDOM_SEED)
    idx      = rng.choice(H * W, size=n_train, replace=False)
    _, scaler = preprocess_pixels(image.reshape(-1, C)[idx], fit_scaler=True)

    tflite_path = os.path.join(RESULTS_DIR, 'mawdbn_fixed', scene_name,
                               f'mawdbn_{QUANTIZATION_MODE}_{scene_name}_ptq_int8.tflite')

    print(f"  [variants] A+B (float codes)...")
    result['beta_va_none']  = compute_beta_map_sq(mse_none,  codes_none)
    result['beta_vb_none']  = compute_beta_map_sqrtre_sqdist(mse_none,  codes_none)
    result['beta_va_quant'] = compute_beta_map_sq(mse_quant, codes_quant)
    result['beta_vb_quant'] = compute_beta_map_sqrtre_sqdist(mse_quant, codes_quant)

    print(f"  [variants] C (int8 codes, float RMSE)...")
    codes_int8 = get_int8_codes_tflite(tflite_path, image, scaler)
    result['beta_vc_quant'] = compute_beta_map_int8codes(mse_quant, codes_int8)

    print(f"  [variants] D (int8 codes, integer MSE via requantise)...")
    mse_int, codes_int8_d = get_int8_mse_and_codes_tflite(tflite_path, image, scaler)
    result['beta_vd_quant'] = compute_beta_map_int8_full(mse_int, codes_int8_d)

    print(f"  [variants] E (int8 codes, integer MSE direct to output scale)...")
    mse_int_e, codes_int8_e = get_int8_mse_direct_and_codes_tflite(tflite_path, image, scaler)
    result['beta_ve_quant'] = compute_beta_map_int8_full(mse_int_e, codes_int8_e)

    gt = result.get('gt_mask')
    if gt is not None:
        for k in ('beta_va_none','beta_vb_none','beta_va_quant','beta_vb_quant',
                  'beta_vc_quant','beta_vd_quant','beta_ve_quant'):
            auc, fpr, tpr = compute_auc_roc(result[k], gt)
            result[f'auc_{k}'] = auc
        print(f"  orig  f32={result['auc_beta_none']:.4f}  qt={result['auc_beta_quant']:.4f}")
        print(f"  Var-A f32={result['auc_beta_va_none']:.4f}  qt={result['auc_beta_va_quant']:.4f}  (sq/sq)")
        print(f"  Var-B f32={result['auc_beta_vb_none']:.4f}  qt={result['auc_beta_vb_quant']:.4f}  (rmse/sqdist float)")
        print(f"  Var-C             qt={result['auc_beta_vc_quant']:.4f}                    (rmse/int8-sqdist)")
        print(f"  Var-D             qt={result['auc_beta_vd_quant']:.4f}                    (int-mse/int8-sqdist requantise)")
        print(f"  Var-E             qt={result['auc_beta_ve_quant']:.4f}                    (int-mse/int8-sqdist direct)")
    return result