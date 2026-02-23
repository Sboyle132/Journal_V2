/*
 * dbn_inout.cpp  —  Quantised DBN HLS kernel
 *
 * Top function: dbn_inout()
 * Mirrors the structure of the float hsi_inout() kernel exactly:
 *   load_weights  → load_bram → load_input → encDec → store_result
 *
 * Integer arithmetic throughout the network (int8 weights, int64 accumulators,
 * fixed-point multiply-shift requantisation, int8 LUT sigmoid).
 * Float only at the boundary: scaler normalisation on input,
 * dequantise + inverse-scale + RMSE on output.
 *
 * All pragma names and loop labels follow the original conventions.
 */

#include "dbn_inout.hpp"

/* ══════════════════════════════════════════════════════════════════════════
 * IO — load weights into on-chip BRAM
 * Same pattern as original load_weights()
 * ════════════════════════════════════════════════════════════════════════ */
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

/* ── Load biases + LUT + scaler into on-chip BRAM ───────────────────────
 * Extends the original load_bram() to also bring in LUT and scaler.
 * These are small enough to sit entirely on-chip for the duration.
 * ──────────────────────────────────────────────────────────────────────── */
static void load_bram(
    dat_tb1  B_enc[MAX_CODES],
    dat_tb2  B_dec[MAX_BANDS],
    dat_tc1  LUT[LUT_SIZE],
    float    scaler_mean[MAX_BANDS],
    float    scaler_scale[MAX_BANDS],
    dat_tb1  B_enc_i[MAX_CODES],
    dat_tb2  B_dec_i[MAX_BANDS],
    dat_tc1  LUT_i[LUT_SIZE],
    float    sc_mean_i[MAX_BANDS],
    float    sc_scale_i[MAX_BANDS])
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
load_scaler:
    for (int i = 0; i < MAX_BANDS; i++) {
        #pragma HLS PIPELINE
        sc_mean_i[i]  = scaler_mean[i];
        sc_scale_i[i] = scaler_scale[i];
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * IO — stream input pixels
 * Same as original load_input() but reads flat float array [H*W*bands]
 * row-major. Streams one float per beat (IN_SIZE=1).
 * ════════════════════════════════════════════════════════════════════════ */
static void load_input(
    float                hsi_in[BUFFER_SIZE],
    hls::stream<float>  &inStream,
    int height, int width)
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
 * Same as original store_result()
 * ════════════════════════════════════════════════════════════════════════ */
static void store_result(
    dat_tre              SCORE[BUFFER_OUT],
    hls::stream<dat_tre> &score_stream,
    int height, int width)
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
 * Mirrors original encDec() structure:
 *   read_pixels → encode_layer → decode_layer
 * ════════════════════════════════════════════════════════════════════════ */

/* ── read_pixels
 * Reads one pixel (MAX_BANDS floats) from stream, applies StandardScaler,
 * stores normalised result in pix[]. Keeps a raw copy in pix_raw[] for
 * the RMSE calculation after reconstruction.
 * ──────────────────────────────────────────────────────────────────────── */
static void read_pixels(
    float               pix_norm[MAX_BANDS],
    float               pix_raw [MAX_BANDS],
    hls::stream<float> &hsi_in,
    float               sc_mean [MAX_BANDS],
    float               sc_scale[MAX_BANDS])
{
read_outer:
    for (int r = 0; r < MAX_BANDS; r++) {
        #pragma HLS loop_tripcount max=MAX_BANDS
        #pragma HLS PIPELINE
        float raw      = hsi_in.read();
        pix_raw[r]     = raw;
        pix_norm[r]    = (raw - sc_mean[r]) / sc_scale[r];
    }
}

/* ── encode_layer
 * Integer matmul + fixed-point requantise + LUT sigmoid.
 * Replaces the float sigmoid in the original encode_layer().
 *
 *   acc[j] = B_enc[j] + sum_k( W_enc[k*MAX_CODES+j] * (pix_norm_q[k] - input_zp) )
 *   enc_req = clip( ((acc * enc_mult + 2^(enc_shift-1)) >> enc_shift) + enc_act_zp )
 *   H1[j]  = LUT[ enc_req + 128 ]
 * ──────────────────────────────────────────────────────────────────────── */
static void encode_layer(
    float       pix_norm[MAX_BANDS],
    dat_tc1     H1[MAX_CODES],
    dat_tw1     W_enc[MAX_CODES * MAX_BANDS],
    dat_tb1     B_enc[MAX_CODES],
    dat_tc1     LUT[LUT_SIZE],
    float       input_scale,
    int         input_zp,
    ap_uint<32> enc_mult,
    int         enc_shift,
    int         enc_act_zp)
{
    /* Step 1: quantise input pixel to int8 */
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

    /* Step 2: matmul + requantise + LUT */
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

        /* fixed-point requantise */
        ap_int<64> product = (ap_int<64>)acc * (ap_int<64>)(ap_uint<32>)enc_mult;
        ap_int<64> half    = (ap_int<64>)1 << (enc_shift - 1);
        ap_int<64> rounded = (product + half) >> enc_shift;
        ap_int<64> result  = rounded + (ap_int<64>)enc_act_zp;
        if      (result >  127) result =  127;
        else if (result < -128) result = -128;
        ap_int<8> enc_req = (ap_int<8>)result;

        /* sigmoid via LUT — index = enc_req + 128 maps [-128,127] -> [0,255] */
        H1[j] = LUT[(int)enc_req + 128];
    }
}

/* ── decode_layer
 * Integer matmul + fixed-point requantise → int8 reconstruction.
 * Dequantise and inverse-scale to raw DN, then compute RMSE vs original.
 * Writes one float RMSE score to hsi_out stream.
 *
 *   acc[m] = B_dec[m] + sum_n( W_dec[m*MAX_CODES+n] * (H1[n] - sigmoid_zp) )
 *   out_q  = clip( ((acc * dec_mult + 2^(dec_shift-1)) >> dec_shift) + output_zp )
 *   recon  = (out_q - output_zp) * output_scale * sc_scale[m] + sc_mean[m]
 *   RMSE   = sqrt( mean( (raw - recon)^2 ) )
 * ──────────────────────────────────────────────────────────────────────── */
static void decode_layer(
    dat_tc1              H1[MAX_CODES],
    float                pix_raw[MAX_BANDS],
    hls::stream<dat_tre> &hsi_out,
    dat_tw2              W_dec[MAX_BANDS * MAX_CODES],
    dat_tb2              B_dec[MAX_BANDS],
    float                sc_mean[MAX_BANDS],
    float                sc_scale[MAX_BANDS],
    int                  sigmoid_zp,
    ap_uint<32>          dec_mult,
    int                  dec_shift,
    float                output_scale,
    int                  output_zp)
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

        /* fixed-point requantise */
        ap_int<64> product = (ap_int<64>)acc * (ap_int<64>)(ap_uint<32>)dec_mult;
        ap_int<64> half    = (ap_int<64>)1 << (dec_shift - 1);
        ap_int<64> rounded = (product + half) >> dec_shift;
        ap_int<64> result  = rounded + (ap_int<64>)output_zp;
        if      (result >  127) result =  127;
        else if (result < -128) result = -128;
        ap_int<8> out_q = (ap_int<8>)result;

        /* dequantise to normalised space, then inverse-scale to raw DN */
        float recon_norm = ((float)(int)out_q - (float)output_zp) * output_scale;
        float recon_raw  = recon_norm * sc_scale[m] + sc_mean[m];

        /* accumulate squared error */
        float diff = pix_raw[m] - recon_raw;
        score += diff * diff;
    }

    score = hls::sqrt(score / (float)MAX_BANDS);
    hsi_out.write(score);
}

/* ── encDec
 * Pixel loop — same structure as original encDec().
 * Loads weights/biases/LUT/scaler into local BRAM, then processes
 * each pixel: read → encode → decode → emit score.
 * ──────────────────────────────────────────────────────────────────────── */
static void encDec(
    dat_tw1              W_enc[MAX_CODES * MAX_BANDS],
    dat_tw2              W_dec[MAX_BANDS * MAX_CODES],
    dat_tb1              B_enc[MAX_CODES],
    dat_tb2              B_dec[MAX_BANDS],
    dat_tc1              LUT[LUT_SIZE],
    float                sc_mean[MAX_BANDS],
    float                sc_scale[MAX_BANDS],
    hls::stream<float>  &hsi_in,
    hls::stream<dat_tre> &hsi_out,
    int height, int width,
    float input_scale, int input_zp,
    ap_uint<32> enc_mult, int enc_shift, int enc_act_zp,
    int sigmoid_zp,
    ap_uint<32> dec_mult, int dec_shift,
    float output_scale, int output_zp)
{
    /* On-chip buffers */
    float    pix_norm[MAX_BANDS];
    float    pix_raw [MAX_BANDS];
    dat_tc1  H1[MAX_CODES];
    dat_tw1  W_enc_i[MAX_CODES * MAX_BANDS];
    dat_tw2  W_dec_i[MAX_BANDS * MAX_CODES];
    dat_tb1  B_enc_i[MAX_CODES];
    dat_tb2  B_dec_i[MAX_BANDS];
    dat_tc1  LUT_i[LUT_SIZE];
    float    sc_mean_i[MAX_BANDS];
    float    sc_scale_i[MAX_BANDS];

    load_weights(W_enc, W_dec, W_enc_i, W_dec_i);
    load_bram(B_enc, B_dec, LUT, sc_mean, sc_scale,
              B_enc_i, B_dec_i, LUT_i, sc_mean_i, sc_scale_i);

HSI_AD_LOOP:
    for (int i = 0; i < width * height; i++) {
        #pragma HLS loop_tripcount max=MAX_WIDTH*MAX_HEIGHT

        read_pixels (pix_norm, pix_raw, hsi_in, sc_mean_i, sc_scale_i);
        encode_layer(pix_norm, H1, W_enc_i, B_enc_i, LUT_i,
                     input_scale, input_zp, enc_mult, enc_shift, enc_act_zp);
        decode_layer(H1, pix_raw, hsi_out, W_dec_i, B_dec_i,
                     sc_mean_i, sc_scale_i,
                     sigmoid_zp, dec_mult, dec_shift, output_scale, output_zp);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * Top function
 * ════════════════════════════════════════════════════════════════════════ */
void dbn_inout(
    dat_tw1  W_enc [MAX_CODES * MAX_BANDS],
    dat_tb1  B_enc [MAX_CODES],
    dat_tw2  W_dec [MAX_BANDS * MAX_CODES],
    dat_tb2  B_dec [MAX_BANDS],
    dat_tc1  LUT   [LUT_SIZE],
    float    scaler_mean  [MAX_BANDS],
    float    scaler_scale [MAX_BANDS],
    float    hsi_in  [BUFFER_SIZE],
    dat_tre  hsi_out [BUFFER_OUT],
    int      height,
    int      width,
    float    input_scale,
    int      input_zp,
    unsigned int enc_mult,
    int      enc_shift,
    int      enc_act_zp,
    int      sigmoid_zp,
    unsigned int dec_mult,
    int      dec_shift,
    float    output_scale,
    int      output_zp)
{
    /* ── AXI interfaces ─────────────────────────────────────────────────── */
    /* Weights and biases — separate bundles to allow parallel access      */
    #pragma HLS INTERFACE m_axi port=W_enc        bundle=aximm1
    #pragma HLS INTERFACE m_axi port=B_enc        bundle=aximm2
    #pragma HLS INTERFACE m_axi port=W_dec        bundle=aximm3
    #pragma HLS INTERFACE m_axi port=B_dec        bundle=aximm4
    /* LUT and scaler — share a bundle (small, loaded once, sequential)   */
    #pragma HLS INTERFACE m_axi port=LUT          bundle=aximm5
    #pragma HLS INTERFACE m_axi port=scaler_mean  bundle=aximm5
    #pragma HLS INTERFACE m_axi port=scaler_scale bundle=aximm5
    /* Image data — separate bundle for sustained bandwidth               */
    #pragma HLS INTERFACE m_axi port=hsi_in       bundle=aximm6
    #pragma HLS INTERFACE m_axi port=hsi_out      bundle=aximm6

    /* ── Control scalars — AXI-lite ────────────────────────────────────── */
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
    encDec      (W_enc, W_dec, B_enc, B_dec, LUT, scaler_mean, scaler_scale,
                 hsi_stream, score_stream, height, width,
                 input_scale, input_zp,
                 (ap_uint<32>)enc_mult, enc_shift, enc_act_zp,
                 sigmoid_zp,
                 (ap_uint<32>)dec_mult, dec_shift,
                 output_scale, output_zp);
    store_result(hsi_out, score_stream, height, width);
}
