/*
 * dbn_inout.cpp  —  Quantised DBN HLS kernel
 *
 * Top function: dbn_inout()
 * Mirrors the structure of the float hsi_inout() kernel exactly:
 *   load_weights  -> load_bram -> load_input -> encDec -> store_result
 *
 * Integer arithmetic throughout the network (int8 weights, int32 accumulators,
 * fixed-point multiply-shift requantisation, int8 LUT sigmoid).
 * Float only at the boundary: scaler normalisation on input (ROM constants),
 * dequantise + RMSE on output.
 *
 * Scaler parameters (SCALER_MEAN, SCALER_SCALE) are compile-time ROM constants
 * from scaler_rom.hpp — no AXI port, no BRAM load, HLS sees them at synthesis.
 *
 * Types follow TFLite int8 PTQ spec exactly:
 *   dat_tacc  ap_int<32>   accumulator
 *   dat_tmul  ap_uint<32>  requantise multiplier
 *   zp args   ap_int<8>    zero points
 *   shift args ap_uint<8>  shift values
 *   height/width ap_uint<16>
 *   requantise intermediate ap_int<64> local only — 32x32->64, shift back to 32
 *
 * RMSE convention: normalised space (px_norm - recon_norm).
 *   After PCA + StandardScaler each band has unit variance; normalised RMSE
 *   weights all components equally and matches the Python T3 reference scores.
 */

#include "dbn_inout.hpp"

/* ══════════════════════════════════════════════════════════════════════════
 * IO — load weights into on-chip BRAM
 * ══════════════════════════════════════════════════════════════════════════ */
static void load_weights(
    dat_tw1 W_enc[MAX_CODES * MAX_BANDS],
    dat_tw2 W_dec[MAX_BANDS * MAX_CODES],
    dat_tw1 W_enc_i[MAX_CODES * MAX_BANDS],
    dat_tw2 W_dec_i[MAX_BANDS * MAX_CODES])
{
load_weights:
    for (int i = 0; i < MAX_CODES * MAX_BANDS; i++) {
        #pragma HLS PIPELINE
        W_enc_i[i] = W_enc[i];
        W_dec_i[i] = W_dec[i];
    }
}

/* ── Load biases + LUT into on-chip BRAM ────────────────────────────────────
 * Scaler is now a ROM (scaler_rom.hpp) — no load needed.
 * ─────────────────────────────────────────────────────────────────────────── */
static void load_bram(
    dat_tb1  B_enc[MAX_CODES],
    dat_tb2  B_dec[MAX_BANDS],
    dat_tc1  LUT[LUT_SIZE],
    dat_tb1  B_enc_i[MAX_CODES],
    dat_tb2  B_dec_i[MAX_BANDS],
    dat_tc1  LUT_i[LUT_SIZE])
{
load_B_enc:
    for (int i = 0; i < MAX_CODES; i++) {
        #pragma HLS PIPELINE
        B_enc_i[i] = B_enc[i];
    }
load_B_dec:
    for (int i = 0; i < MAX_BANDS; i++) {
        #pragma HLS PIPELINE
        B_dec_i[i] = B_dec[i];
    }
load_LUT:
    for (int i = 0; i < LUT_SIZE; i++) {
        #pragma HLS PIPELINE
        LUT_i[i] = LUT[i];
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * IO — stream input pixels
 * ══════════════════════════════════════════════════════════════════════════ */
static void load_input(
    float                hsi_in[BUFFER_SIZE],
    hls::stream<float>  &inStream,
    ap_uint<16> height, ap_uint<16> width)
{
stream_in:
    for (int i = 0; i < width * height * MAX_BANDS; i++) {
        #pragma HLS loop_tripcount max=MAX_WIDTH*MAX_HEIGHT*MAX_BANDS
        #pragma HLS PIPELINE
        inStream.write(hsi_in[i]);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * IO — store RMSE results
 * ══════════════════════════════════════════════════════════════════════════ */
static void store_result(
    dat_tre              SCORE[BUFFER_OUT],
    hls::stream<dat_tre> &score_stream,
    ap_uint<16> height, ap_uint<16> width)
{
mem_wr:
    for (int i = 0; i < width * height; i++) {
        #pragma HLS loop_tripcount max=MAX_WIDTH*MAX_HEIGHT
        #pragma HLS PIPELINE
        SCORE[i] = score_stream.read();
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * Autoencoder pipeline
 * ══════════════════════════════════════════════════════════════════════════ */

/* ── read_pixels
 * Reads one pixel (MAX_BANDS floats) from stream, applies StandardScaler
 * using ROM constants SCALER_MEAN / SCALER_SCALE from scaler_rom.hpp.
 * Output is normalised pixel only — raw copy no longer needed since RMSE
 * is computed in normalised space.
 * ─────────────────────────────────────────────────────────────────────────── */
static void read_pixels(
    float               pix_norm[MAX_BANDS],
    hls::stream<float> &hsi_in)
{
read_outer:
    for (int r = 0; r < MAX_BANDS; r++) {
        #pragma HLS loop_tripcount max=MAX_BANDS
        #pragma HLS PIPELINE
        float raw   = hsi_in.read();
        pix_norm[r] = (raw - SCALER_MEAN[r]) / SCALER_SCALE[r];
    }
}

/* ── requantise
 * TFLite multiply-shift pattern.
 * acc is int32, mult is uint32.
 * Intermediate product needs int64 — local only, not loop-carried.
 * Maps to 2 DSP48s. Result clipped to int8.
 * ─────────────────────────────────────────────────────────────────────────── */
static ap_int<8> requantise(dat_tacc acc, ap_uint<32> mult, ap_uint<8> shift, ap_int<8> zp)
{
    #pragma HLS INLINE
    ap_int<64> product = (ap_int<64>)acc * (ap_int<64>)mult;
    ap_int<64> rounded = (product + ((ap_int<64>)1 << (shift - 1))) >> shift;
    ap_int<32> result  = (ap_int<32>)rounded + (ap_int<32>)zp;
    if      (result >  127) result =  127;
    else if (result < -128) result = -128;
    return (ap_int<8>)result;
}

/* ── encode_layer
 * Integer matmul (int32 acc) + requantise + LUT sigmoid.
 * ─────────────────────────────────────────────────────────────────────────── */
static void encode_layer(
    float       pix_norm[MAX_BANDS],
    dat_tc1     H1[MAX_CODES],
    dat_tw1     W_enc[MAX_CODES * MAX_BANDS],
    dat_tb1     B_enc[MAX_CODES],
    dat_tc1     LUT[LUT_SIZE],
    float        input_scale,
    ap_int<8>    input_zp,
    ap_uint<32>  enc_mult,
    ap_uint<8>   enc_shift,
    ap_int<8>    enc_act_zp)
{
    ap_int<8> inp_q[MAX_BANDS];
    #pragma HLS ARRAY_PARTITION variable=inp_q complete
quantise_input:
    for (int k = 0; k < MAX_BANDS; k++) {
        #pragma HLS loop_tripcount max=MAX_BANDS
        #pragma HLS PIPELINE
        int q = (int)hls::roundf(pix_norm[k] / input_scale) + input_zp;
        if      (q >  127) q =  127;
        else if (q < -128) q = -128;
        inp_q[k] = (ap_int<8>)q;
    }

RBM1_exterior:
    for (int j = 0; j < MAX_CODES; j++) {
        #pragma HLS loop_tripcount max=MAX_CODES
        dat_tacc acc = (dat_tacc)B_enc[j];
RBM1_interior:
        for (int k = 0; k < MAX_BANDS; k++) {
            #pragma HLS loop_tripcount max=MAX_BANDS
            acc += (dat_tacc)W_enc[j * MAX_BANDS + k]
                 * ((dat_tacc)inp_q[k] - (dat_tacc)input_zp);
        }
        ap_int<8> enc_req = requantise(acc, enc_mult, enc_shift, enc_act_zp);
        H1[j] = LUT[(int)enc_req + 128];
    }
}

/* ── decode_layer
 * Integer matmul (int32 acc) + requantise -> int8 reconstruction.
 * Dequantise to normalised space, compute RMSE vs normalised input pixel.
 * Writes one float score to stream.
 * ─────────────────────────────────────────────────────────────────────────── */
static void decode_layer(
    dat_tc1              H1[MAX_CODES],
    float                pix_norm[MAX_BANDS],
    hls::stream<dat_tre> &hsi_out,
    dat_tw2              W_dec[MAX_BANDS * MAX_CODES],
    dat_tb2              B_dec[MAX_BANDS],
    ap_int<8>            sigmoid_zp,
    ap_uint<32>          dec_mult,
    ap_uint<8>           dec_shift,
    float                output_scale,
    ap_int<8>            output_zp)
{
    float score = 0.0f;

RBM2_exterior:
    for (int m = 0; m < MAX_BANDS; m++) {
        #pragma HLS loop_tripcount max=MAX_BANDS

        dat_tacc acc = (dat_tacc)B_dec[m];
RBM2_interior:
        for (int n = 0; n < MAX_CODES; n++) {
            #pragma HLS loop_tripcount max=MAX_CODES
            acc += (dat_tacc)W_dec[m * MAX_CODES + n]
                 * ((dat_tacc)H1[n] - (dat_tacc)sigmoid_zp);
        }

        ap_int<8> out_q = requantise(acc, dec_mult, dec_shift, output_zp);

        /* dequantise to normalised space, compute diff vs normalised input */
        float recon_norm = ((float)(int)out_q - (float)output_zp) * output_scale;
        float diff = pix_norm[m] - recon_norm;
        score += diff * diff;
    }

    score = hls::sqrt(score / (float)MAX_BANDS);
    hsi_out.write(score);
}

/* ── encDec
 * Pixel loop.
 * ─────────────────────────────────────────────────────────────────────────── */
static void encDec(
    dat_tw1              W_enc[MAX_CODES * MAX_BANDS],
    dat_tw2              W_dec[MAX_BANDS * MAX_CODES],
    dat_tb1              B_enc[MAX_CODES],
    dat_tb2              B_dec[MAX_BANDS],
    dat_tc1              LUT[LUT_SIZE],
    hls::stream<float>  &hsi_in,
    hls::stream<dat_tre> &hsi_out,
    ap_uint<16> height, ap_uint<16> width,
    float input_scale, ap_int<8> input_zp,
    ap_uint<32> enc_mult, ap_uint<8> enc_shift, ap_int<8> enc_act_zp,
    ap_int<8> sigmoid_zp,
    ap_uint<32> dec_mult, ap_uint<8> dec_shift,
    float output_scale, ap_int<8> output_zp)
{
    float    pix_norm[MAX_BANDS];
    dat_tc1  H1[MAX_CODES];
    dat_tw1  W_enc_i[MAX_CODES * MAX_BANDS];
    dat_tw2  W_dec_i[MAX_BANDS * MAX_CODES];
    dat_tb1  B_enc_i[MAX_CODES];
    dat_tb2  B_dec_i[MAX_BANDS];
    dat_tc1  LUT_i[LUT_SIZE];

    load_weights(W_enc, W_dec, W_enc_i, W_dec_i);
    load_bram(B_enc, B_dec, LUT, B_enc_i, B_dec_i, LUT_i);

HSI_AD_LOOP:
    for (int i = 0; i < width * height; i++) {
        #pragma HLS loop_tripcount max=MAX_WIDTH*MAX_HEIGHT

        read_pixels (pix_norm, hsi_in);
        encode_layer(pix_norm, H1, W_enc_i, B_enc_i, LUT_i,
                     input_scale, input_zp, enc_mult, enc_shift, enc_act_zp);
        decode_layer(H1, pix_norm, hsi_out, W_dec_i, B_dec_i,
                     sigmoid_zp, dec_mult, dec_shift, output_scale, output_zp);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * Top function
 * ══════════════════════════════════════════════════════════════════════════ */
void dbn_inout(
    dat_tw1      W_enc [MAX_CODES * MAX_BANDS],
    dat_tb1      B_enc [MAX_CODES],
    dat_tw2      W_dec [MAX_BANDS * MAX_CODES],
    dat_tb2      B_dec [MAX_BANDS],
    dat_tc1      LUT   [LUT_SIZE],
    float        hsi_in  [BUFFER_SIZE],
    dat_tre      hsi_out [BUFFER_OUT],
    ap_uint<16>  height,
    ap_uint<16>  width,
    float        input_scale,
    ap_int<8>    input_zp,
    ap_uint<32>  enc_mult,
    ap_uint<8>   enc_shift,
    ap_int<8>    enc_act_zp,
    ap_int<8>    sigmoid_zp,
    ap_uint<32>  dec_mult,
    ap_uint<8>   dec_shift,
    float        output_scale,
    ap_int<8>    output_zp)
{
    /* ── AXI interfaces ──────────────────────────────────────────────────── */
    #pragma HLS INTERFACE m_axi port=W_enc   bundle=aximm1
    #pragma HLS INTERFACE m_axi port=B_enc   bundle=aximm2
    #pragma HLS INTERFACE m_axi port=W_dec   bundle=aximm3
    #pragma HLS INTERFACE m_axi port=B_dec   bundle=aximm4
    #pragma HLS INTERFACE m_axi port=LUT     bundle=aximm5
    #pragma HLS INTERFACE m_axi port=hsi_in  bundle=aximm6
    #pragma HLS INTERFACE m_axi port=hsi_out bundle=aximm6

    /* ── Control scalars — AXI-lite ─────────────────────────────────────── */
    #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS INTERFACE s_axilite port=height
    #pragma HLS INTERFACE s_axilite port=width
    #pragma HLS INTERFACE s_axilite port=input_scale
    #pragma HLS INTERFACE s_axilite port=input_zp
    #pragma HLS INTERFACE s_axilite port=enc_mult
    #pragma HLS INTERFACE s_axilite port=enc_shift
    #pragma HLS INTERFACE s_axilite port=enc_act_zp
    #pragma HLS INTERFACE s_axilite port=sigmoid_zp
    #pragma HLS INTERFACE s_axilite port=dec_mult
    #pragma HLS INTERFACE s_axilite port=dec_shift
    #pragma HLS INTERFACE s_axilite port=output_scale
    #pragma HLS INTERFACE s_axilite port=output_zp

    /* ── Internal streams ───────────────────────────────────────────────── */
    static hls::stream<float,   MAX_READS + 1> hsi_stream  ("hsi_stream");
    static hls::stream<dat_tre, MAX_WIDTH + 1> score_stream ("score_stream");

    #pragma HLS DATAFLOW

    load_input  (hsi_in, hsi_stream, height, width);
    encDec      (W_enc, W_dec, B_enc, B_dec, LUT,
                 hsi_stream, score_stream, height, width,
                 input_scale, input_zp,
                 enc_mult, enc_shift, enc_act_zp,
                 sigmoid_zp,
                 dec_mult, dec_shift,
                 output_scale, output_zp);
    store_result(hsi_out, score_stream, height, width);
}