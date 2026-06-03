/* Simple type-safe dynamic array macros
 */

#pragma once

#include "helpers.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define DEFINE_VECTOR(type, name) \
typedef struct { \
    type *data; \
    size_t size; \
    size_t capacity; \
} name; \
\
void name##_init(name *v) { \
    v->data = NULL; \
    v->size = 0; \
    v->capacity = 0; \
} \
\
void name##_free(name *v) { \
    free(v->data); \
    v->data = NULL; \
    v->size = 0; \
    v->capacity = 0; \
} \
\
void name##_reserve(name *v, size_t cap) { \
    if (cap > v->capacity) { \
        v->capacity = cap; \
        v->data = (type*)realloc(v->data, cap * sizeof(type)); \
        QM_ASSERT(v->data); \
    } \
} \
\
void name##_resize(name *v, size_t size) { \
    name##_reserve(v, size); \
    v->size = size; \
} \
\
void name##_push(name *v, type value) { \
    if (v->size == v->capacity) { \
        size_t new_cap = v->capacity ? v->capacity * 2 : 4; \
        name##_reserve(v, new_cap); \
    } \
    v->data[v->size++] = value; \
} \
\
void name##_remove_swap(name *v, size_t idx) { \
    QM_ASSERT(idx < v->size); \
    v->data[idx] = v->data[v->size - 1]; \
    v->size--; \
} \
\
void name##_remove(name *v, size_t idx) { \
    QM_ASSERT(idx < v->size); \
    memmove(&v->data[idx], \
            &v->data[idx + 1], \
            (v->size - idx - 1) * sizeof(type)); \
    v->size--; \
} \
\
void name##_remove_range(name *v, size_t from, size_t to) { \
    QM_ASSERT(from <= to); \
    QM_ASSERT(to <= v->size); \
    memmove(&v->data[from], \
            &v->data[to + 1], \
            (v->size - to - 1) * sizeof(type)); \
    v->size -= (to + 1 - from); \
}

// Define common array types
DEFINE_VECTOR(int, IntArray)
DEFINE_VECTOR(float, FloatArray)
DEFINE_VECTOR(bool, BoolArray)
DEFINE_VECTOR(char, ByteArray)
