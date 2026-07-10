#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QM3_ASSERT(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "quad_meshing_3d assert failed: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
        abort(); \
    } \
} while (0)

#define QMSURF3D_VERSION 1u
#define QMSURF3D_SECTION_INFO 1u
#define QMSURF3D_SECTION_VERTICES 2u
#define QMSURF3D_SECTION_TRIANGLES 3u
#define QMSURF3D_SECTION_VERTEX_NORMALS 4u
#define QMSURF3D_SECTION_FACE_NORMALS 5u
#define QMSURF3D_SECTION_TRIANGLE_NEIGHBORS 6u
#define QMSURF3D_SECTION_FACE_DIR_U 7u
#define QMSURF3D_SECTION_FACE_DIR_V 8u
#define QMSURF3D_SECTION_SAMPLES 9u
#define QMSURF3D_SECTION_SHARP_EDGES 10u

typedef struct {
    float x;
    float y;
    float z;
} Qm3Vec3;

typedef struct {
    uint32_t a;
    uint32_t b;
    uint32_t c;
} Qm3Tri;

typedef struct {
    int32_t a;
    int32_t b;
    int32_t c;
} Qm3TriNeighbors;

typedef struct {
    float total_area;
    float sample_density;
    float sharp_dihedral_radians;
    uint32_t sample_count;
} Qm3SurfaceInfo;

typedef struct {
    Qm3Vec3 p;
    Qm3Vec3 n;
    uint32_t tri;
    uint32_t reserved;
} Qm3SurfaceSample;

typedef struct {
    uint32_t a;
    uint32_t b;
    int32_t f0;
    int32_t f1;
    float angle;
} Qm3SharpEdge;

typedef struct {
    uint32_t type;
    uint32_t count;
    uint32_t elem_size;
    uint32_t reserved;
} Qm3SectionHeader;

typedef struct {
    Qm3SurfaceInfo info;
    Qm3Vec3* vertices;
    Qm3Tri* triangles;
    Qm3Vec3* vertex_normals;
    Qm3Vec3* face_normals;
    Qm3TriNeighbors* triangle_neighbors;
    Qm3Vec3* face_dir_u;
    Qm3Vec3* face_dir_v;
    Qm3SurfaceSample* samples;
    Qm3SharpEdge* sharp_edges;
    uint32_t vertex_count;
    uint32_t triangle_count;
    uint32_t sample_count;
    uint32_t sharp_edge_count;
    Qm3Vec3 bounds_min;
    Qm3Vec3 bounds_max;
} Qm3Surface;

static inline Qm3Vec3 qm3_add(Qm3Vec3 a, Qm3Vec3 b) {
    return (Qm3Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Qm3Vec3 qm3_sub(Qm3Vec3 a, Qm3Vec3 b) {
    return (Qm3Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline Qm3Vec3 qm3_scale(Qm3Vec3 a, float s) {
    return (Qm3Vec3){a.x * s, a.y * s, a.z * s};
}

static inline float qm3_dot(Qm3Vec3 a, Qm3Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float qm3_len(Qm3Vec3 a) {
    return sqrtf(qm3_dot(a, a));
}

static inline Qm3Vec3 qm3_normalize(Qm3Vec3 a) {
    float len = qm3_len(a);
    if (len <= 1e-20f) return (Qm3Vec3){0.0f, 1.0f, 0.0f};
    return qm3_scale(a, 1.0f / len);
}

static void qm3_read_exact(FILE* f, void* dst, size_t size) {
    QM3_ASSERT(fread(dst, 1, size, f) == size);
}

static void qm3_surface_init(Qm3Surface* surface) {
    memset(surface, 0, sizeof(*surface));
    surface->bounds_min = (Qm3Vec3){INFINITY, INFINITY, INFINITY};
    surface->bounds_max = (Qm3Vec3){-INFINITY, -INFINITY, -INFINITY};
}

static void qm3_surface_free(Qm3Surface* surface) {
    free(surface->vertices);
    free(surface->triangles);
    free(surface->vertex_normals);
    free(surface->face_normals);
    free(surface->triangle_neighbors);
    free(surface->face_dir_u);
    free(surface->face_dir_v);
    free(surface->samples);
    free(surface->sharp_edges);
    qm3_surface_init(surface);
}

static void* qm3_read_section_array(FILE* f, uint32_t count, uint32_t elem_size, uint32_t expected_elem_size) {
    QM3_ASSERT(elem_size == expected_elem_size);
    size_t bytes = (size_t)count * (size_t)elem_size;
    if (bytes == 0) return NULL;
    void* data = malloc(bytes);
    QM3_ASSERT(data != NULL);
    qm3_read_exact(f, data, bytes);
    return data;
}

static void qm3_surface_update_bounds(Qm3Surface* surface) {
    surface->bounds_min = (Qm3Vec3){INFINITY, INFINITY, INFINITY};
    surface->bounds_max = (Qm3Vec3){-INFINITY, -INFINITY, -INFINITY};
    for (uint32_t i = 0; i < surface->vertex_count; ++i) {
        Qm3Vec3 p = surface->vertices[i];
        surface->bounds_min.x = fminf(surface->bounds_min.x, p.x);
        surface->bounds_min.y = fminf(surface->bounds_min.y, p.y);
        surface->bounds_min.z = fminf(surface->bounds_min.z, p.z);
        surface->bounds_max.x = fmaxf(surface->bounds_max.x, p.x);
        surface->bounds_max.y = fmaxf(surface->bounds_max.y, p.y);
        surface->bounds_max.z = fmaxf(surface->bounds_max.z, p.z);
    }
}

static float qm3_surface_diag(const Qm3Surface* surface) {
    return qm3_len(qm3_sub(surface->bounds_max, surface->bounds_min));
}

static void qm3_surface_load(Qm3Surface* surface, const char* path) {
    FILE* f = fopen(path, "rb");
    QM3_ASSERT(f != NULL);

    qm3_surface_free(surface);

    char magic[8];
    qm3_read_exact(f, magic, sizeof(magic));
    QM3_ASSERT(memcmp(magic, "QMSURF3D", sizeof(magic)) == 0);

    uint32_t version = 0;
    uint32_t section_count = 0;
    qm3_read_exact(f, &version, sizeof(version));
    qm3_read_exact(f, &section_count, sizeof(section_count));
    QM3_ASSERT(version == QMSURF3D_VERSION);

    int saw_info = 0;
    int saw_vertices = 0;
    int saw_triangles = 0;
    int saw_vertex_normals = 0;
    int saw_face_normals = 0;
    int saw_neighbors = 0;
    int saw_dir_u = 0;
    int saw_dir_v = 0;
    int saw_samples = 0;
    int saw_sharp_edges = 0;

    for (uint32_t i = 0; i < section_count; ++i) {
        Qm3SectionHeader header;
        qm3_read_exact(f, &header, sizeof(header));
        QM3_ASSERT(header.reserved == 0);

        if (header.type == QMSURF3D_SECTION_INFO) {
            QM3_ASSERT(header.count == 1 && header.elem_size == sizeof(Qm3SurfaceInfo));
            qm3_read_exact(f, &surface->info, sizeof(surface->info));
            saw_info = 1;
        } else if (header.type == QMSURF3D_SECTION_VERTICES) {
            surface->vertices = (Qm3Vec3*)qm3_read_section_array(f, header.count, header.elem_size, sizeof(Qm3Vec3));
            surface->vertex_count = header.count;
            saw_vertices = 1;
        } else if (header.type == QMSURF3D_SECTION_TRIANGLES) {
            surface->triangles = (Qm3Tri*)qm3_read_section_array(f, header.count, header.elem_size, sizeof(Qm3Tri));
            surface->triangle_count = header.count;
            saw_triangles = 1;
        } else if (header.type == QMSURF3D_SECTION_VERTEX_NORMALS) {
            surface->vertex_normals = (Qm3Vec3*)qm3_read_section_array(f, header.count, header.elem_size, sizeof(Qm3Vec3));
            saw_vertex_normals = 1;
        } else if (header.type == QMSURF3D_SECTION_FACE_NORMALS) {
            surface->face_normals = (Qm3Vec3*)qm3_read_section_array(f, header.count, header.elem_size, sizeof(Qm3Vec3));
            saw_face_normals = 1;
        } else if (header.type == QMSURF3D_SECTION_TRIANGLE_NEIGHBORS) {
            surface->triangle_neighbors = (Qm3TriNeighbors*)qm3_read_section_array(f, header.count, header.elem_size, sizeof(Qm3TriNeighbors));
            saw_neighbors = 1;
        } else if (header.type == QMSURF3D_SECTION_FACE_DIR_U) {
            surface->face_dir_u = (Qm3Vec3*)qm3_read_section_array(f, header.count, header.elem_size, sizeof(Qm3Vec3));
            saw_dir_u = 1;
        } else if (header.type == QMSURF3D_SECTION_FACE_DIR_V) {
            surface->face_dir_v = (Qm3Vec3*)qm3_read_section_array(f, header.count, header.elem_size, sizeof(Qm3Vec3));
            saw_dir_v = 1;
        } else if (header.type == QMSURF3D_SECTION_SAMPLES) {
            surface->samples = (Qm3SurfaceSample*)qm3_read_section_array(f, header.count, header.elem_size, sizeof(Qm3SurfaceSample));
            surface->sample_count = header.count;
            saw_samples = 1;
        } else if (header.type == QMSURF3D_SECTION_SHARP_EDGES) {
            surface->sharp_edges = (Qm3SharpEdge*)qm3_read_section_array(f, header.count, header.elem_size, sizeof(Qm3SharpEdge));
            surface->sharp_edge_count = header.count;
            saw_sharp_edges = 1;
        } else {
            QM3_ASSERT(fseek(f, (long)((size_t)header.count * (size_t)header.elem_size), SEEK_CUR) == 0);
        }
    }

    fclose(f);

    QM3_ASSERT(saw_info && saw_vertices && saw_triangles && saw_vertex_normals && saw_face_normals);
    QM3_ASSERT(saw_neighbors && saw_dir_u && saw_dir_v && saw_samples && saw_sharp_edges);
    QM3_ASSERT(surface->vertex_count > 0 && surface->triangle_count > 0);
    QM3_ASSERT(surface->sample_count == surface->info.sample_count);
    qm3_surface_update_bounds(surface);
}
