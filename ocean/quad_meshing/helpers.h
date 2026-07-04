#pragma once

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <dirent.h>

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

static bool str_has_suffix(const char* str, const char* suffix) {
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    return str_len >= suffix_len && strcmp(str + str_len - suffix_len, suffix) == 0;
}

static char* path_join(const char* folder, const char* name) {
    size_t folder_len = strlen(folder);
    size_t name_len = strlen(name);
    bool needs_sep = folder_len > 0 && folder[folder_len - 1] != '/';
    char* path = (char*)calloc(folder_len + needs_sep + name_len + 1, sizeof(char));
    QM_ASSERT(path != NULL);
    memcpy(path, folder, folder_len);
    if (needs_sep) path[folder_len] = '/';
    memcpy(path + folder_len + needs_sep, name, name_len);
    return path;
}

static int compare_strings(const void* a, const void* b) {
    const char* sa = *(const char* const*)a;
    const char* sb = *(const char* const*)b;
    return strcmp(sa, sb);
}

static const char** list_files_with_suffix(const char* folder, const char* suffix, int* out_count) {
    DIR* dir = opendir(folder);
    QM_ASSERT(dir != NULL);

    int count = 0;
    int capacity = 16;
    const char** paths = (const char**)calloc(capacity, sizeof(const char*));
    QM_ASSERT(paths != NULL);

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (!str_has_suffix(entry->d_name, suffix)) continue;
        if (count == capacity) {
            capacity *= 2;
            paths = (const char**)realloc(paths, capacity * sizeof(const char*));
            QM_ASSERT(paths != NULL);
        }
        paths[count++] = path_join(folder, entry->d_name);
    }
    closedir(dir);

    qsort(paths, count, sizeof(const char*), compare_strings);
    *out_count = count;
    return paths;
}
