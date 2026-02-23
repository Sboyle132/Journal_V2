#pragma once
#ifndef params_h
#define params_h

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* ── Utility ─────────────────────────────────────────────────────────────── */
#ifndef CEILING
#define CEILING(x,y) (((x) + (y) - 1) / (y))
#endif

/* ── Scene: abu-airport-1 ────────────────────────────────────────────────── */
#define MAX_BANDS   205
#define MAX_CODES   13
#define MAX_WIDTH   100
#define MAX_HEIGHT  100
#define IN_SIZE     1   /* bands streamed per beat; raise for throughput */

/* ── Derived ─────────────────────────────────────────────────────────────── */
#define MAX_READS   CEILING(MAX_BANDS, IN_SIZE)
#define BUFFER_SIZE (MAX_WIDTH * MAX_HEIGHT * MAX_BANDS)
#define BUFFER_OUT  (MAX_WIDTH * MAX_HEIGHT)

/* ── Sigmoid LUT ─────────────────────────────────────────────────────────── */
#define LUT_SIZE    256

#endif /* params_h */
