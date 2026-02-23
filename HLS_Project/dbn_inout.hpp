#pragma once
#ifndef dbn_inout_h
#define dbn_inout_h

#include "params.hpp"
#include <hls_vector.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <ap_int.h>
#include <ap_fixed.h>

/* ══════════════════════════════════════════════════════════════════════════
 * Type definitions
 *
 * Network data path — pure integer, matching the exported quantised model.
 * Scaler and RMSE stay float: one float normalise at input,
 * one float dequantise + inverse-scale + RMSE at output.
 * ════════════════════════════════════════════════════════════════════════ */

typedef float        dat_tin;   /* raw HSI input  [float DN]               */
typedef ap_int<8>    dat_tw1;   /* encoder weights [int8]                  */
typedef ap_int<8>    dat_tw2;   /* decoder weights [int8]                  */
typedef ap_int<32>   dat_tb1;   /* encoder biases  [int32]                 */
typedef ap_int<32>   dat_tb2;   /* decoder biases  [int32]                 */
typedef ap_int<8>    dat_tc1;   /* hidden activations / LUT entries [int8] */
typedef ap_int<64>   dat_tacc;  /* matmul accumulator [int64]              */
typedef ap_uint<32>  dat_tmul;  /* requantise multiplier [uint32]          */
typedef float        dat_tre;   /* RMSE output [float]                     */

/* ── Top-level function declaration ────────────────────────────────────── */
void dbn_inout(
    /* Weights and biases — m_axi */
    dat_tw1  W_enc [MAX_CODES * MAX_BANDS],   /* enc_w  [n_hidden, n_input] */
    dat_tb1  B_enc [MAX_CODES],               /* enc_b  [n_hidden]          */
    dat_tw2  W_dec [MAX_BANDS * MAX_CODES],   /* dec_w  [n_input, n_hidden] */
    dat_tb2  B_dec [MAX_BANDS],               /* dec_b  [n_input]           */
    /* Sigmoid LUT — m_axi, loaded once into BRAM */
    dat_tc1  LUT   [LUT_SIZE],
    /* Scaler — m_axi, loaded once into BRAM */
    float    scaler_mean  [MAX_BANDS],
    float    scaler_scale [MAX_BANDS],
    /* Image input — m_axi */
    float    hsi_in  [BUFFER_SIZE],           /* [H*W*n_bands] row-major    */
    /* RMSE output — m_axi */
    dat_tre  hsi_out [BUFFER_OUT],            /* [H*W] float RMSE scores    */
    /* Dimensions — s_axilite */
    int      height,
    int      width,
    /* Quantisation scalars — s_axilite */
    float    input_scale,
    int      input_zp,
    unsigned int enc_mult,
    int      enc_shift,
    int      enc_act_zp,
    int      sigmoid_zp,
    unsigned int dec_mult,
    int      dec_shift,
    float    output_scale,
    int      output_zp
);

#endif /* dbn_inout_h */
