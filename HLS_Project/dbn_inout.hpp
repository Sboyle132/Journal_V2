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
 * Type definitions — follow TFLite int8 PTQ spec exactly
 *
 *   dat_tw1/2  ap_int<8>    weights  — always int8 by PTQ definition
 *   dat_tb1/2  ap_int<32>   biases   — TFLite stores as int32, pre-scaled
 *   dat_tc1    ap_int<8>    hidden activations and LUT entries
 *   dat_tacc   ap_int<32>   matmul accumulator — int32 sufficient per TFLite spec
 *   dat_tmul   ap_uint<32>  requantise multiplier — full 32-bit normalised fraction
 *   dat_tre    float        RMSE output — float boundary only
 *
 * Zero points (input_zp, enc_act_zp, sigmoid_zp, output_zp): ap_int<8>
 * Shifts (enc_shift, dec_shift): ap_uint<8>
 * Dimensions (height, width): ap_uint<16>
 *
 * Requantise intermediate: ap_int<64> LOCAL only inside requantise()
 *   int32 acc x uint32 mult -> 64-bit product, >> shift -> back to int32
 *   Not loop-carried; maps to 2 DSP48s, not a wide accumulator.
 *
 * Scaler parameters (SCALER_MEAN, SCALER_SCALE) are compile-time ROM
 * constants from scaler_rom.hpp — no function parameters needed.
 * ══════════════════════════════════════════════════════════════════════════ */

typedef float        dat_tin;   /* raw HSI input  [float DN]               */
typedef ap_int<8>    dat_tw1;   /* encoder weights [int8]                  */
typedef ap_int<8>    dat_tw2;   /* decoder weights [int8]                  */
typedef ap_int<32>   dat_tb1;   /* encoder biases  [int32]                 */
typedef ap_int<32>   dat_tb2;   /* decoder biases  [int32]                 */
typedef ap_int<8>    dat_tc1;   /* hidden activations / LUT entries [int8] */
typedef ap_int<32>   dat_tacc;  /* matmul accumulator [int32]              */
typedef ap_uint<32>  dat_tmul;  /* requantise multiplier [uint32]          */
typedef float        dat_tre;   /* RMSE output [float]                     */

/* ── Top-level function declaration ──────────────────────────────────────── */
void dbn_inout(
    /* Weights and biases — m_axi */
    dat_tw1      W_enc [MAX_CODES * MAX_BANDS],
    dat_tb1      B_enc [MAX_CODES],
    dat_tw2      W_dec [MAX_BANDS * MAX_CODES],
    dat_tb2      B_dec [MAX_BANDS],
    /* Sigmoid LUT — m_axi, loaded once into BRAM */
    dat_tc1      LUT   [LUT_SIZE],
    /* Image input / output — m_axi */
    float        hsi_in  [BUFFER_SIZE],
    dat_tre      hsi_out [BUFFER_OUT],
    /* Dimensions — s_axilite */
    ap_uint<16>  height,
    ap_uint<16>  width,
    /* Quantisation scalars — s_axilite */
    float        input_scale,
    ap_int<8>    input_zp,
    ap_uint<32>  enc_mult,
    ap_uint<8>   enc_shift,
    ap_int<8>    enc_act_zp,
    ap_int<8>    sigmoid_zp,
    ap_uint<32>  dec_mult,
    ap_uint<8>   dec_shift,
    float        output_scale,
    ap_int<8>    output_zp
);

#endif /* dbn_inout_h */