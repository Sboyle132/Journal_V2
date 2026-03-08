#ifndef params_h
#define params_h

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef CEILING
#define CEILING(x,y) (((x) + (y) - 1) / (y))
#endif

/* ── Scene dimensions ────────────────────────────────────────────────────── */
#define MAX_BANDS   120     /* PCA-reduced; matches HYPSO native band count  */
#define MAX_CODES   13
#define MAX_WIDTH   100
#define MAX_HEIGHT  100
#define IN_SIZE     1

/* ── Derived ─────────────────────────────────────────────────────────────── */
#define MAX_READS   CEILING(MAX_BANDS, IN_SIZE)
#define BUFFER_SIZE (MAX_WIDTH * MAX_HEIGHT * MAX_BANDS)
#define BUFFER_OUT  (MAX_WIDTH * MAX_HEIGHT)

/* ── Sigmoid LUT ─────────────────────────────────────────────────────────── */
#define LUT_SIZE    256

/* ── Scaler ROM — StandardScaler parameters baked in at synthesis time ───── */
#include "scaler_rom.hpp"

#endif /* params_h */