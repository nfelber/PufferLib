#pragma once

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifndef QUAD_MESHING_ENABLE_ASSERTS
#define QUAD_MESHING_ENABLE_ASSERTS 1
#endif

#if QUAD_MESHING_ENABLE_ASSERTS
#define QM_ASSERT(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "quad_meshing assert failed: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
        abort(); \
    } \
} while (0)
#else
#define QM_ASSERT(expr) do { (void)sizeof(expr); } while (0)
#endif

static int rand_range(unsigned int* rng, int max_exclusive) {
    return rand_r(rng) % max_exclusive;
}

static inline float clampf(float v, float lo, float hi) {
    return fmaxf(lo, fminf(hi, v));
}


static char* read_file_bytes(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    QM_ASSERT(f != NULL);
    QM_ASSERT(fseek(f, 0, SEEK_END) == 0);
    long size = ftell(f);
    QM_ASSERT(size > 0);
    QM_ASSERT(fseek(f, 0, SEEK_SET) == 0);
    char* buffer = (char*)calloc((size_t)size + 1, 1);
    QM_ASSERT(buffer != NULL);
    size_t read = fread(buffer, 1, (size_t)size, f);
    fclose(f);
    QM_ASSERT(read == (size_t)size);
    buffer[size] = '\0';
    if (out_size) *out_size = (size_t)size;
    return buffer;
}

