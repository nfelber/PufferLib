#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int* data;
    int size;
    int cap;
} IntVec;

typedef struct {
    int from;
    int to;
    int tri;
    float w;
} GraphEdge;

typedef struct {
    GraphEdge* data;
    int size;
    int cap;
} EdgeVec;

typedef struct {
    int node;
    float dist;
} HeapItem;

typedef struct {
    HeapItem* data;
    int size;
    int cap;
} MinHeap;

typedef struct {
    Vector3* data;
    int size;
    int cap;
} Vec3Vec;

typedef struct {
    int tri;
    Vector3 a;
    Vector3 b;
} LoopSegment;

typedef struct {
    LoopSegment* data;
    int size;
    int cap;
} LoopSegmentVec;

typedef struct {
    float* data;
    int size;
    int cap;
} FloatVec;

typedef struct {
    int tri;
    int edge;
    float t0;
    float t1;
    int side;
} LoopPortal;

typedef struct {
    LoopPortal* data;
    int size;
    int cap;
} LoopPortalVec;

typedef struct {
    Vector3 p;
    int tri;
    float distance;
    unsigned int stamp;
    bool in_radius;
    bool disabled;
} SurfaceSample;

typedef struct {
    Vector3* vertices;
    int vertex_count;
    int* tris;
    int* tri_neighbors;
    int tri_count;
    float* tri_cdf;
    float total_area;
    float max_edge_length;
    BoundingBox bounds;
} SurfaceMesh;

typedef struct {
    long long qx;
    long long qy;
    long long qz;
    int unique_idx;
    int next;
} WeldEntry;

typedef struct {
    int* tri_sample_offsets;
    int* tri_sample_ids;
    int tri_count;
    int sample_count;
} SurfaceTopo;

typedef struct {
    int a;
    int b;
    int tri;
    int edge;
} TriEdge;

typedef struct {
    int tri;
    int entry_edge;
    int parent;
    float key;
    float offset;
    float t0;
    float t1;
    int source_vertex;
    Vector2 tri2d[3];
    Vector2 source2;
} GeoWindow;

typedef struct {
    GeoWindow* data;
    int size;
    int cap;
} GeoWindowVec;

typedef struct {
    int window;
    float key;
} WindowHeapItem;

typedef struct {
    WindowHeapItem* data;
    int size;
    int cap;
} WindowHeap;

typedef struct {
    GeoWindowVec windows;
    WindowHeap heap;
    IntVec* state_windows;
    unsigned int* state_stamp;
    IntVec touched_states;
    float* best_vertex;
    unsigned int* vertex_stamp;
    IntVec touched_vertices;
    unsigned char* pseudo_source_vertex;
    Vector2* tri2d_base;
    unsigned char* tri2d_valid;
    unsigned int query_id;
    int state_count;
    int vertex_count;
    int windows_pushed;
    int windows_popped;
    int pseudo_sources;
    int touched_state_count;
    int max_state_windows;
} GeodesicPathContext;

typedef struct {
    Vector3* nodes;
    int node_count;
    int node_cap;
    int* edge_offsets;
    int* edge_to;
    int* edge_tri;
    float* edge_weight;
    int edge_count;
    int* tri_node_offsets;
    int* tri_node_ids;
    int* node_tri_offsets;
    int* node_tri_ids;
    int tri_count;
    float spacing;
} PropGraph;

typedef struct {
    float* dist;
    int* prev;
    int* prev_tri;
    unsigned int* stamp;
    unsigned char* state;
    unsigned int* tri_stamp;
    int* reached_tris;
    int reached_tri_count;
    int reached_tri_cap;
    unsigned int query_id;
    MinHeap heap;
    int vertex_count;
    int tri_count;
} FMMContext;

enum {
    FMM_FAR = 0,
    FMM_TRIAL = 1,
    FMM_ACCEPTED = 2,
};

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void* checked_realloc(void* ptr, size_t bytes) {
    void* out = realloc(ptr, bytes);
    if (out == NULL) {
        fprintf(stderr, "out of memory allocating %zu bytes\n", bytes);
        exit(1);
    }
    return out;
}

static void int_vec_push(IntVec* v, int value) {
    if (v->size == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 16;
        v->data = (int*)checked_realloc(v->data, (size_t)v->cap * sizeof(int));
    }
    v->data[v->size++] = value;
}

static void int_vec_free(IntVec* v) {
    free(v->data);
    memset(v, 0, sizeof(*v));
}

static void edge_vec_push(EdgeVec* v, GraphEdge value) {
    if (v->size == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 1024;
        v->data = (GraphEdge*)checked_realloc(v->data, (size_t)v->cap * sizeof(GraphEdge));
    }
    v->data[v->size++] = value;
}

static void geo_window_vec_push(GeoWindowVec* v, GeoWindow value) {
    if (v->size == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 1024;
        v->data = (GeoWindow*)checked_realloc(v->data, (size_t)v->cap * sizeof(GeoWindow));
    }
    v->data[v->size++] = value;
}

static void vec3_vec_push(Vec3Vec* v, Vector3 value) {
    if (v->size == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 64;
        v->data = (Vector3*)checked_realloc(v->data, (size_t)v->cap * sizeof(Vector3));
    }
    v->data[v->size++] = value;
}

static void vec3_vec_clear(Vec3Vec* v) {
    v->size = 0;
}

static void vec3_vec_free(Vec3Vec* v) {
    free(v->data);
    memset(v, 0, sizeof(*v));
}

static void loop_segment_vec_push(LoopSegmentVec* v, LoopSegment value) {
    if (Vector3DistanceSqr(value.a, value.b) <= 1e-14f) return;
    if (v->size == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 64;
        v->data = (LoopSegment*)checked_realloc(v->data, (size_t)v->cap * sizeof(LoopSegment));
    }
    v->data[v->size++] = value;
}

static void loop_segment_vec_clear(LoopSegmentVec* v) {
    v->size = 0;
}

static void loop_segment_vec_free(LoopSegmentVec* v) {
    free(v->data);
    memset(v, 0, sizeof(*v));
}

static void float_vec_push(FloatVec* v, float value) {
    if (v->size == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 8;
        v->data = (float*)checked_realloc(v->data, (size_t)v->cap * sizeof(float));
    }
    v->data[v->size++] = value;
}

static int compare_floats(const void* pa, const void* pb) {
    float a = *(const float*)pa;
    float b = *(const float*)pb;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static void float_vec_sort_unique(FloatVec* v) {
    if (v->size <= 1) return;
    qsort(v->data, (size_t)v->size, sizeof(float), compare_floats);
    int out = 0;
    for (int i = 0; i < v->size; ++i) {
        float value = Clamp(v->data[i], 0.0f, 1.0f);
        if (out > 0 && fabsf(value - v->data[out - 1]) <= 1e-5f) continue;
        v->data[out++] = value;
    }
    v->size = out;
}

static void float_vec_free(FloatVec* v) {
    free(v->data);
    memset(v, 0, sizeof(*v));
}

static void loop_portal_vec_push(LoopPortalVec* v, LoopPortal value) {
    if (value.t1 <= value.t0 + 1e-5f || value.side == 0) return;
    if (v->size == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 64;
        v->data = (LoopPortal*)checked_realloc(v->data, (size_t)v->cap * sizeof(LoopPortal));
    }
    v->data[v->size++] = value;
}

static void loop_portal_vec_free(LoopPortalVec* v) {
    free(v->data);
    memset(v, 0, sizeof(*v));
}

static void heap_push(MinHeap* h, HeapItem item) {
    if (h->size == h->cap) {
        h->cap = h->cap ? h->cap * 2 : 1024;
        h->data = (HeapItem*)checked_realloc(h->data, (size_t)h->cap * sizeof(HeapItem));
    }
    int i = h->size++;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->data[p].dist <= item.dist) break;
        h->data[i] = h->data[p];
        i = p;
    }
    h->data[i] = item;
}

static HeapItem heap_pop(MinHeap* h) {
    HeapItem out = h->data[0];
    HeapItem item = h->data[--h->size];
    int i = 0;
    while (true) {
        int l = 2 * i + 1;
        int r = l + 1;
        if (l >= h->size) break;
        int c = (r < h->size && h->data[r].dist < h->data[l].dist) ? r : l;
        if (h->data[c].dist >= item.dist) break;
        h->data[i] = h->data[c];
        i = c;
    }
    if (h->size > 0) h->data[i] = item;
    return out;
}

static void window_heap_push(WindowHeap* h, WindowHeapItem item) {
    if (h->size == h->cap) {
        h->cap = h->cap ? h->cap * 2 : 1024;
        h->data = (WindowHeapItem*)checked_realloc(h->data, (size_t)h->cap * sizeof(WindowHeapItem));
    }
    int i = h->size++;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->data[p].key <= item.key) break;
        h->data[i] = h->data[p];
        i = p;
    }
    h->data[i] = item;
}

static WindowHeapItem window_heap_pop(WindowHeap* h) {
    WindowHeapItem out = h->data[0];
    WindowHeapItem item = h->data[--h->size];
    int i = 0;
    while (true) {
        int l = 2 * i + 1;
        int r = l + 1;
        if (l >= h->size) break;
        int c = (r < h->size && h->data[r].key < h->data[l].key) ? r : l;
        if (h->data[c].key >= item.key) break;
        h->data[i] = h->data[c];
        i = c;
    }
    if (h->size > 0) h->data[i] = item;
    return out;
}

static float frand01(unsigned int* rng) {
    return rand_r(rng) / ((float)RAND_MAX + 1.0f);
}

static float triangle_area(Vector3 a, Vector3 b, Vector3 c) {
    return 0.5f * Vector3Length(Vector3CrossProduct(Vector3Subtract(b, a), Vector3Subtract(c, a)));
}

static Vector3 triangle_sample(Vector3 a, Vector3 b, Vector3 c, unsigned int* rng) {
    float u = frand01(rng);
    float v = frand01(rng);
    float su = sqrtf(u);
    float b0 = 1.0f - su;
    float b1 = su * (1.0f - v);
    float b2 = su * v;
    return (Vector3){
        b0 * a.x + b1 * b.x + b2 * c.x,
        b0 * a.y + b1 * b.y + b2 * c.y,
        b0 * a.z + b1 * b.z + b2 * c.z,
    };
}

static int tri_local_index(const SurfaceMesh* surface, int tri, int vertex) {
    for (int i = 0; i < 3; ++i) {
        if (surface->tris[3 * tri + i] == vertex) return i;
    }
    return -1;
}

static void barycentric3(Vector3 p, Vector3 a, Vector3 b, Vector3 c, float* u, float* v, float* w) {
    Vector3 v0 = Vector3Subtract(b, a);
    Vector3 v1 = Vector3Subtract(c, a);
    Vector3 v2 = Vector3Subtract(p, a);
    float d00 = Vector3DotProduct(v0, v0);
    float d01 = Vector3DotProduct(v0, v1);
    float d11 = Vector3DotProduct(v1, v1);
    float d20 = Vector3DotProduct(v2, v0);
    float d21 = Vector3DotProduct(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    if (fabsf(denom) < 1e-20f) {
        *u = 1.0f;
        *v = 0.0f;
        *w = 0.0f;
        return;
    }
    *v = (d11 * d20 - d01 * d21) / denom;
    *w = (d00 * d21 - d01 * d20) / denom;
    *u = 1.0f - *v - *w;
}

static Vector2 barycentric_to_2d(float u, float v, float w, Vector2 a, Vector2 b, Vector2 c) {
    return (Vector2){
        u * a.x + v * b.x + w * c.x,
        u * a.y + v * b.y + w * c.y,
    };
}

static float cross2v(Vector2 a, Vector2 b) {
    return a.x * b.y - a.y * b.x;
}

static bool segment_intersect_2d(Vector2 p, Vector2 q, Vector2 a, Vector2 b, float* out_t, float* out_u) {
    Vector2 r = Vector2Subtract(q, p);
    Vector2 s = Vector2Subtract(b, a);
    float den = cross2v(r, s);
    if (fabsf(den) < 1e-8f) return false;
    Vector2 ap = Vector2Subtract(a, p);
    float t = cross2v(ap, s) / den;
    float u = cross2v(ap, r) / den;
    if (t < -1e-5f || t > 1.0f + 1e-5f || u < -1e-5f || u > 1.0f + 1e-5f) return false;
    *out_t = Clamp(t, 0.0f, 1.0f);
    *out_u = Clamp(u, 0.0f, 1.0f);
    return true;
}

static bool point_on_segment_2d(Vector2 p, Vector2 a, Vector2 b, float* out_t) {
    Vector2 ab = Vector2Subtract(b, a);
    float len2 = Vector2DotProduct(ab, ab);
    if (len2 < 1e-20f) return false;
    float t = Vector2DotProduct(Vector2Subtract(p, a), ab) / len2;
    if (t < -1e-5f || t > 1.0f + 1e-5f) return false;
    Vector2 q = Vector2Add(a, Vector2Scale(ab, t));
    if (Vector2Distance(q, p) > 1e-5f) return false;
    *out_t = Clamp(t, 0.0f, 1.0f);
    return true;
}

static bool segment_portal_hit_2d(Vector2 p, Vector2 q, Vector2 a, Vector2 b, float min_t, float* out_t, float* out_u) {
    float t, u;
    if (segment_intersect_2d(p, q, a, b, &t, &u) && t >= min_t - 1e-4f) {
        *out_t = t;
        *out_u = u;
        return true;
    }

    float ta, tb;
    bool has_a = point_on_segment_2d(a, p, q, &ta) && ta >= min_t - 1e-4f;
    bool has_b = point_on_segment_2d(b, p, q, &tb) && tb >= min_t - 1e-4f;
    if (!has_a && !has_b) return false;

    if (has_a && (!has_b || ta <= tb)) {
        *out_t = ta;
        *out_u = 0.0f;
    } else {
        *out_t = tb;
        *out_u = 1.0f;
    }
    return true;
}

static bool robust_portal_hit_2d(Vector2 p, Vector2 q, Vector2 a, Vector2 b, float u0, float u1, float* out_u) {
    float lt, lu;
    if (segment_portal_hit_2d(p, q, a, b, -1.0f, &lt, &lu)) {
        *out_u = Clamp(lu, fminf(u0, u1), fmaxf(u0, u1));
        return true;
    }

    Vector2 r = Vector2Subtract(q, p);
    Vector2 s = Vector2Subtract(b, a);
    float den = cross2v(r, s);
    if (fabsf(den) > 1e-10f) {
        Vector2 ap = Vector2Subtract(a, p);
        lu = cross2v(ap, r) / den;
        *out_u = Clamp(lu, fminf(u0, u1), fmaxf(u0, u1));
        return true;
    }

    float len2 = Vector2DotProduct(s, s);
    if (len2 <= 1e-20f) return false;
    float up = Vector2DotProduct(Vector2Subtract(p, a), s) / len2;
    float uq = Vector2DotProduct(Vector2Subtract(q, a), s) / len2;
    float lo = fminf(u0, u1);
    float hi = fmaxf(u0, u1);
    if (up >= lo && up <= hi) *out_u = up;
    else if (uq >= lo && uq <= hi) *out_u = uq;
    else {
        float mid = 0.5f * (up + uq);
        *out_u = Clamp(mid, lo, hi);
    }
    return true;
}

static int cdf_sample_triangle(const SurfaceMesh* surface, unsigned int* rng) {
    float x = frand01(rng) * surface->total_area;
    int lo = 0;
    int hi = surface->tri_count - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (surface->tri_cdf[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static void surface_free(SurfaceMesh* surface) {
    free(surface->vertices);
    free(surface->tris);
    free(surface->tri_neighbors);
    free(surface->tri_cdf);
    memset(surface, 0, sizeof(*surface));
}

static uint64_t hash_quantized_vertex(long long x, long long y, long long z) {
    uint64_t h = 1469598103934665603ull;
    h = (h ^ (uint64_t)x) * 1099511628211ull;
    h = (h ^ (uint64_t)y) * 1099511628211ull;
    h = (h ^ (uint64_t)z) * 1099511628211ull;
    return h;
}

static int pow2_at_least(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

static float surface_diag(BoundingBox b) {
    return Vector3Distance(b.min, b.max);
}

static int* build_triangle_neighbors(const SurfaceMesh* surface);

static Matrix normalize_model_transform(BoundingBox bounds) {
    Vector3 size = Vector3Subtract(bounds.max, bounds.min);
    float longest = fmaxf(size.x, fmaxf(size.y, size.z));
    float scale = longest > 1e-12f ? 1.0f / longest : 1.0f;
    Vector3 center = Vector3Scale(Vector3Add(bounds.min, bounds.max), 0.5f);

    return (Matrix){
        scale, 0.0f, 0.0f, -center.x * scale,
        0.0f, scale, 0.0f, -center.y * scale,
        0.0f, 0.0f, scale, -center.z * scale,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
}

static SurfaceMesh surface_from_model(const Model* model) {
    SurfaceMesh surface = {0};

    int raw_vertex_count = 0;
    int tri_count = 0;
    for (int m = 0; m < model->meshCount; ++m) {
        const Mesh* mesh = &model->meshes[m];
        raw_vertex_count += mesh->vertexCount;
        tri_count += mesh->triangleCount;
    }
    if (raw_vertex_count == 0 || tri_count == 0) {
        fprintf(stderr, "model has no triangle geometry\n");
        exit(1);
    }

    Vector3* raw_vertices = (Vector3*)calloc((size_t)raw_vertex_count, sizeof(Vector3));
    int* raw_tris = (int*)calloc((size_t)tri_count * 3, sizeof(int));
    int* raw_to_unique = (int*)calloc((size_t)raw_vertex_count, sizeof(int));
    surface.vertices = (Vector3*)calloc((size_t)raw_vertex_count, sizeof(Vector3));
    surface.tris = (int*)calloc((size_t)tri_count * 3, sizeof(int));
    surface.tri_cdf = (float*)calloc((size_t)tri_count, sizeof(float));
    if (!raw_vertices || !raw_tris || !raw_to_unique || !surface.vertices || !surface.tris || !surface.tri_cdf) {
        fprintf(stderr, "out of memory while copying mesh\n");
        exit(1);
    }

    int v_base = 0;
    int t_base = 0;
    BoundingBox raw_bounds = {
        .min = {FLT_MAX, FLT_MAX, FLT_MAX},
        .max = {-FLT_MAX, -FLT_MAX, -FLT_MAX},
    };
    for (int m = 0; m < model->meshCount; ++m) {
        const Mesh* mesh = &model->meshes[m];
        for (int i = 0; i < mesh->vertexCount; ++i) {
            Vector3 v = {
                mesh->vertices[3 * i + 0],
                mesh->vertices[3 * i + 1],
                mesh->vertices[3 * i + 2],
            };
            v = Vector3Transform(v, model->transform);
            raw_vertices[v_base + i] = v;
            raw_bounds.min.x = fminf(raw_bounds.min.x, v.x);
            raw_bounds.min.y = fminf(raw_bounds.min.y, v.y);
            raw_bounds.min.z = fminf(raw_bounds.min.z, v.z);
            raw_bounds.max.x = fmaxf(raw_bounds.max.x, v.x);
            raw_bounds.max.y = fmaxf(raw_bounds.max.y, v.y);
            raw_bounds.max.z = fmaxf(raw_bounds.max.z, v.z);
        }

        for (int i = 0; i < mesh->triangleCount; ++i) {
            for (int k = 0; k < 3; ++k) {
                int idx = mesh->indices ? mesh->indices[3 * i + k] : 3 * i + k;
                raw_tris[3 * (t_base + i) + k] = v_base + idx;
            }
        }
        v_base += mesh->vertexCount;
        t_base += mesh->triangleCount;
    }

    float diag = surface_diag(raw_bounds);
    float weld_eps = fmaxf(diag * 1e-6f, 1e-7f);
    float weld_eps2 = weld_eps * weld_eps;
    int bucket_count = pow2_at_least(raw_vertex_count * 2);
    int* buckets = (int*)malloc((size_t)bucket_count * sizeof(int));
    WeldEntry* entries = (WeldEntry*)calloc((size_t)raw_vertex_count, sizeof(WeldEntry));
    if (!buckets || !entries) {
        fprintf(stderr, "out of memory while welding mesh vertices\n");
        exit(1);
    }
    for (int i = 0; i < bucket_count; ++i) buckets[i] = -1;

    int unique_count = 0;
    int entry_count = 0;
    for (int i = 0; i < raw_vertex_count; ++i) {
        Vector3 v = raw_vertices[i];
        long long qx = llroundf((v.x - raw_bounds.min.x) / weld_eps);
        long long qy = llroundf((v.y - raw_bounds.min.y) / weld_eps);
        long long qz = llroundf((v.z - raw_bounds.min.z) / weld_eps);
        int bucket = (int)(hash_quantized_vertex(qx, qy, qz) & (uint64_t)(bucket_count - 1));
        int found = -1;
        for (int e = buckets[bucket]; e != -1; e = entries[e].next) {
            WeldEntry entry = entries[e];
            if (entry.qx != qx || entry.qy != qy || entry.qz != qz) continue;
            if (Vector3DistanceSqr(v, surface.vertices[entry.unique_idx]) <= weld_eps2) {
                found = entry.unique_idx;
                break;
            }
        }
        if (found < 0) {
            found = unique_count;
            surface.vertices[unique_count++] = v;
            entries[entry_count] = (WeldEntry){
                .qx = qx,
                .qy = qy,
                .qz = qz,
                .unique_idx = found,
                .next = buckets[bucket],
            };
            buckets[bucket] = entry_count++;
        }
        raw_to_unique[i] = found;
    }

    for (int i = 0; i < tri_count * 3; ++i) surface.tris[i] = raw_to_unique[raw_tris[i]];
    surface.vertices = (Vector3*)checked_realloc(surface.vertices, (size_t)unique_count * sizeof(Vector3));

    surface.vertex_count = unique_count;
    surface.tri_count = tri_count;
    surface.bounds.min = (Vector3){FLT_MAX, FLT_MAX, FLT_MAX};
    surface.bounds.max = (Vector3){-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (int i = 0; i < unique_count; ++i) {
        Vector3 v = surface.vertices[i];
        surface.bounds.min.x = fminf(surface.bounds.min.x, v.x);
        surface.bounds.min.y = fminf(surface.bounds.min.y, v.y);
        surface.bounds.min.z = fminf(surface.bounds.min.z, v.z);
        surface.bounds.max.x = fmaxf(surface.bounds.max.x, v.x);
        surface.bounds.max.y = fmaxf(surface.bounds.max.y, v.y);
        surface.bounds.max.z = fmaxf(surface.bounds.max.z, v.z);
    }

    float total = 0.0f;
    float max_edge_length = 0.0f;
    for (int i = 0; i < tri_count; ++i) {
        int ia = surface.tris[3 * i + 0];
        int ib = surface.tris[3 * i + 1];
        int ic = surface.tris[3 * i + 2];
        max_edge_length = fmaxf(max_edge_length, Vector3Distance(surface.vertices[ia], surface.vertices[ib]));
        max_edge_length = fmaxf(max_edge_length, Vector3Distance(surface.vertices[ib], surface.vertices[ic]));
        max_edge_length = fmaxf(max_edge_length, Vector3Distance(surface.vertices[ic], surface.vertices[ia]));
        total += triangle_area(surface.vertices[ia], surface.vertices[ib], surface.vertices[ic]);
        surface.tri_cdf[i] = total;
    }
    if (total <= 0.0f) {
        fprintf(stderr, "model has zero surface area\n");
        exit(1);
    }
    surface.total_area = total;
    surface.max_edge_length = max_edge_length;
    surface.tri_neighbors = build_triangle_neighbors(&surface);

    free(raw_vertices);
    free(raw_tris);
    free(raw_to_unique);
    free(buckets);
    free(entries);

    return surface;
}

static SurfaceSample* sample_surface_points(const SurfaceMesh* surface, int count, unsigned int* rng) {
    SurfaceSample* samples = (SurfaceSample*)calloc((size_t)count, sizeof(SurfaceSample));
    if (!samples) {
        fprintf(stderr, "out of memory while sampling points\n");
        exit(1);
    }

    for (int i = 0; i < count; ++i) {
        int tri = cdf_sample_triangle(surface, rng);
        int ia = surface->tris[3 * tri + 0];
        int ib = surface->tris[3 * tri + 1];
        int ic = surface->tris[3 * tri + 2];
        samples[i].p = triangle_sample(surface->vertices[ia], surface->vertices[ib], surface->vertices[ic], rng);
        samples[i].tri = tri;
    }
    return samples;
}

static SurfaceTopo build_surface_topo(const SurfaceMesh* surface, const SurfaceSample* samples, int sample_count) {
    SurfaceTopo topo = {0};
    topo.tri_count = surface->tri_count;
    topo.sample_count = sample_count;
    topo.tri_sample_offsets = (int*)calloc((size_t)surface->tri_count + 1, sizeof(int));
    topo.tri_sample_ids = (int*)malloc((size_t)sample_count * sizeof(int));
    if (!topo.tri_sample_offsets || !topo.tri_sample_ids) {
        fprintf(stderr, "out of memory while building topology\n");
        exit(1);
    }

    for (int i = 0; i < sample_count; ++i) topo.tri_sample_offsets[samples[i].tri + 1]++;
    for (int tri = 0; tri < surface->tri_count; ++tri) {
        topo.tri_sample_offsets[tri + 1] += topo.tri_sample_offsets[tri];
    }
    int* cursor = (int*)malloc((size_t)surface->tri_count * sizeof(int));
    if (!cursor) {
        fprintf(stderr, "out of memory while building topology cursor\n");
        exit(1);
    }
    memcpy(cursor, topo.tri_sample_offsets, (size_t)surface->tri_count * sizeof(int));
    for (int i = 0; i < sample_count; ++i) topo.tri_sample_ids[cursor[samples[i].tri]++] = i;
    free(cursor);
    return topo;
}

static void surface_topo_free(SurfaceTopo* topo, const SurfaceMesh* surface) {
    (void)surface;
    free(topo->tri_sample_offsets);
    free(topo->tri_sample_ids);
    memset(topo, 0, sizeof(*topo));
}

static int compare_tri_edges(const void* pa, const void* pb) {
    const TriEdge* a = (const TriEdge*)pa;
    const TriEdge* b = (const TriEdge*)pb;
    if (a->a != b->a) return a->a < b->a ? -1 : 1;
    if (a->b != b->b) return a->b < b->b ? -1 : 1;
    return 0;
}

static int* build_triangle_neighbors(const SurfaceMesh* surface) {
    int* neighbors = (int*)malloc((size_t)surface->tri_count * 3 * sizeof(int));
    TriEdge* edges = (TriEdge*)malloc((size_t)surface->tri_count * 3 * sizeof(TriEdge));
    if (!neighbors || !edges) {
        fprintf(stderr, "out of memory while building triangle adjacency\n");
        exit(1);
    }
    for (int i = 0; i < surface->tri_count * 3; ++i) neighbors[i] = -1;

    for (int tri = 0; tri < surface->tri_count; ++tri) {
        int v[3] = {
            surface->tris[3 * tri + 0],
            surface->tris[3 * tri + 1],
            surface->tris[3 * tri + 2],
        };
        for (int e = 0; e < 3; ++e) {
            int a = v[(e + 1) % 3];
            int b = v[(e + 2) % 3];
            if (a > b) {
                int tmp = a;
                a = b;
                b = tmp;
            }
            edges[3 * tri + e] = (TriEdge){.a = a, .b = b, .tri = tri, .edge = e};
        }
    }

    qsort(edges, (size_t)surface->tri_count * 3, sizeof(TriEdge), compare_tri_edges);
    int edge_count = surface->tri_count * 3;
    for (int i = 0; i < edge_count;) {
        int j = i + 1;
        while (j < edge_count && edges[j].a == edges[i].a && edges[j].b == edges[i].b) ++j;
        if (j == i + 2) {
            TriEdge e0 = edges[i];
            TriEdge e1 = edges[i + 1];
            neighbors[3 * e0.tri + e0.edge] = e1.tri;
            neighbors[3 * e1.tri + e1.edge] = e0.tri;
        }
        i = j;
    }

    free(edges);
    return neighbors;
}

static int prop_add_node(PropGraph* graph, Vector3 p) {
    if (graph->node_count == graph->node_cap) {
        graph->node_cap = graph->node_cap ? graph->node_cap * 2 : 1024;
        graph->nodes = (Vector3*)checked_realloc(graph->nodes, (size_t)graph->node_cap * sizeof(Vector3));
    }
    int idx = graph->node_count++;
    graph->nodes[idx] = p;
    return idx;
}

static void prop_add_node_to_tri(IntVec* tri_nodes_tmp, IntVec* node_tris_tmp, int tri, int node) {
    IntVec* tri_nodes = &tri_nodes_tmp[tri];
    for (int i = 0; i < tri_nodes->size; ++i) {
        if (tri_nodes->data[i] == node) return;
    }
    int_vec_push(tri_nodes, node);
    int_vec_push(&node_tris_tmp[node], tri);
}

static void prop_add_undirected(PropGraph* graph, EdgeVec* edges, int a, int b, int tri) {
    if (a == b) return;
    float w = Vector3Distance(graph->nodes[a], graph->nodes[b]);
    edge_vec_push(edges, (GraphEdge){.from = a, .to = b, .tri = tri, .w = w});
    edge_vec_push(edges, (GraphEdge){.from = b, .to = a, .tri = tri, .w = w});
}

static int compare_graph_edges(const void* pa, const void* pb) {
    const GraphEdge* a = (const GraphEdge*)pa;
    const GraphEdge* b = (const GraphEdge*)pb;
    if (a->from != b->from) return a->from < b->from ? -1 : 1;
    if (a->to != b->to) return a->to < b->to ? -1 : 1;
    if (a->tri != b->tri) return a->tri < b->tri ? -1 : 1;
    if (a->w < b->w) return -1;
    if (a->w > b->w) return 1;
    return 0;
}

static PropGraph build_prop_graph(const SurfaceMesh* surface, float spacing) {
    PropGraph graph = {0};
    graph.tri_count = surface->tri_count;
    graph.spacing = spacing;
    IntVec* tri_nodes_tmp = (IntVec*)calloc((size_t)surface->tri_count, sizeof(IntVec));
    IntVec* node_tris_tmp = NULL;
    EdgeVec edges = {0};
    if (!tri_nodes_tmp) {
        fprintf(stderr, "out of memory while allocating propagation triangles\n");
        exit(1);
    }

    for (int i = 0; i < surface->vertex_count; ++i) {
        prop_add_node(&graph, surface->vertices[i]);
    }
    node_tris_tmp = (IntVec*)calloc((size_t)graph.node_cap, sizeof(IntVec));
    if (!node_tris_tmp) {
        fprintf(stderr, "out of memory while allocating propagation node triangles\n");
        exit(1);
    }
    for (int tri = 0; tri < surface->tri_count; ++tri) {
        for (int i = 0; i < 3; ++i) {
            prop_add_node_to_tri(tri_nodes_tmp, node_tris_tmp, tri, surface->tris[3 * tri + i]);
        }
    }

    TriEdge* tri_edges = (TriEdge*)calloc((size_t)surface->tri_count * 3, sizeof(TriEdge));
    if (!tri_edges) {
        fprintf(stderr, "out of memory while building edge Steiner points\n");
        exit(1);
    }
    for (int tri = 0; tri < surface->tri_count; ++tri) {
        int v[3] = {
            surface->tris[3 * tri + 0],
            surface->tris[3 * tri + 1],
            surface->tris[3 * tri + 2],
        };
        for (int e = 0; e < 3; ++e) {
            int a = v[e];
            int b = v[(e + 1) % 3];
            if (a > b) {
                int tmp = a;
                a = b;
                b = tmp;
            }
            tri_edges[3 * tri + e] = (TriEdge){.a = a, .b = b, .tri = tri};
        }
    }
    qsort(tri_edges, (size_t)surface->tri_count * 3, sizeof(TriEdge), compare_tri_edges);

    int edge_count = surface->tri_count * 3;
    for (int i = 0; i < edge_count;) {
        int j = i + 1;
        while (j < edge_count && tri_edges[j].a == tri_edges[i].a && tri_edges[j].b == tri_edges[i].b) ++j;

        int a = tri_edges[i].a;
        int b = tri_edges[i].b;
        Vector3 pa = surface->vertices[a];
        Vector3 pb = surface->vertices[b];
        float len = Vector3Distance(pa, pb);
        int segments = spacing > 1e-8f ? (int)ceilf(len / spacing) : 1;
        if (segments < 1) segments = 1;

        for (int s = 1; s < segments; ++s) {
            float t = s / (float)segments;
            Vector3 p = Vector3Add(Vector3Scale(pa, 1.0f - t), Vector3Scale(pb, t));
            int node = prop_add_node(&graph, p);
            if (graph.node_count > graph.node_cap / 2) {
                node_tris_tmp = (IntVec*)checked_realloc(node_tris_tmp, (size_t)graph.node_cap * sizeof(IntVec));
            }
            node_tris_tmp[node] = (IntVec){0};
            for (int k = i; k < j; ++k) prop_add_node_to_tri(tri_nodes_tmp, node_tris_tmp, tri_edges[k].tri, node);
        }
        i = j;
    }
    free(tri_edges);

    for (int tri = 0; tri < surface->tri_count; ++tri) {
        IntVec nodes = tri_nodes_tmp[tri];
        for (int a = 0; a < nodes.size; ++a) {
            for (int b = a + 1; b < nodes.size; ++b) {
                prop_add_undirected(&graph, &edges, nodes.data[a], nodes.data[b], tri);
            }
        }
    }

    qsort(edges.data, edges.size, sizeof(GraphEdge), compare_graph_edges);
    int unique_edges = 0;
    for (int i = 0; i < edges.size; ++i) {
        if (unique_edges > 0 &&
            edges.data[i].from == edges.data[unique_edges - 1].from &&
            edges.data[i].to == edges.data[unique_edges - 1].to &&
            edges.data[i].tri == edges.data[unique_edges - 1].tri) {
            if (edges.data[i].w < edges.data[unique_edges - 1].w) edges.data[unique_edges - 1].w = edges.data[i].w;
            continue;
        }
        edges.data[unique_edges++] = edges.data[i];
    }
    graph.edge_count = unique_edges;
    graph.edge_offsets = (int*)calloc((size_t)graph.node_count + 1, sizeof(int));
    graph.edge_to = (int*)malloc((size_t)unique_edges * sizeof(int));
    graph.edge_tri = (int*)malloc((size_t)unique_edges * sizeof(int));
    graph.edge_weight = (float*)malloc((size_t)unique_edges * sizeof(float));
    if (!graph.edge_offsets || !graph.edge_to || !graph.edge_tri || !graph.edge_weight) {
        fprintf(stderr, "out of memory while finalizing propagation CSR\n");
        exit(1);
    }
    for (int i = 0; i < unique_edges; ++i) graph.edge_offsets[edges.data[i].from + 1]++;
    for (int i = 0; i < graph.node_count; ++i) graph.edge_offsets[i + 1] += graph.edge_offsets[i];
    int* cursor = (int*)malloc((size_t)graph.node_count * sizeof(int));
    if (!cursor) {
        fprintf(stderr, "out of memory while finalizing propagation cursor\n");
        exit(1);
    }
    memcpy(cursor, graph.edge_offsets, (size_t)graph.node_count * sizeof(int));
    for (int i = 0; i < unique_edges; ++i) {
        int pos = cursor[edges.data[i].from]++;
        graph.edge_to[pos] = edges.data[i].to;
        graph.edge_tri[pos] = edges.data[i].tri;
        graph.edge_weight[pos] = edges.data[i].w;
    }
    free(cursor);
    free(edges.data);

    graph.tri_node_offsets = (int*)calloc((size_t)surface->tri_count + 1, sizeof(int));
    graph.node_tri_offsets = (int*)calloc((size_t)graph.node_count + 1, sizeof(int));
    if (!graph.tri_node_offsets || !graph.node_tri_offsets) {
        fprintf(stderr, "out of memory while flattening propagation lists\n");
        exit(1);
    }
    for (int tri = 0; tri < surface->tri_count; ++tri) graph.tri_node_offsets[tri + 1] = tri_nodes_tmp[tri].size;
    for (int node = 0; node < graph.node_count; ++node) graph.node_tri_offsets[node + 1] = node_tris_tmp[node].size;
    for (int tri = 0; tri < surface->tri_count; ++tri) graph.tri_node_offsets[tri + 1] += graph.tri_node_offsets[tri];
    for (int node = 0; node < graph.node_count; ++node) graph.node_tri_offsets[node + 1] += graph.node_tri_offsets[node];
    graph.tri_node_ids = (int*)malloc((size_t)graph.tri_node_offsets[surface->tri_count] * sizeof(int));
    graph.node_tri_ids = (int*)malloc((size_t)graph.node_tri_offsets[graph.node_count] * sizeof(int));
    if (!graph.tri_node_ids || !graph.node_tri_ids) {
        fprintf(stderr, "out of memory while storing flat propagation lists\n");
        exit(1);
    }
    for (int tri = 0; tri < surface->tri_count; ++tri) {
        memcpy(&graph.tri_node_ids[graph.tri_node_offsets[tri]], tri_nodes_tmp[tri].data, (size_t)tri_nodes_tmp[tri].size * sizeof(int));
    }
    for (int node = 0; node < graph.node_count; ++node) {
        memcpy(&graph.node_tri_ids[graph.node_tri_offsets[node]], node_tris_tmp[node].data, (size_t)node_tris_tmp[node].size * sizeof(int));
    }

    for (int tri = 0; tri < surface->tri_count; ++tri) free(tri_nodes_tmp[tri].data);
    for (int node = 0; node < graph.node_count; ++node) free(node_tris_tmp[node].data);
    free(tri_nodes_tmp);
    free(node_tris_tmp);

    return graph;
}

static void prop_graph_free(PropGraph* graph) {
    free(graph->nodes);
    free(graph->edge_offsets);
    free(graph->edge_to);
    free(graph->edge_tri);
    free(graph->edge_weight);
    free(graph->tri_node_offsets);
    free(graph->tri_node_ids);
    free(graph->node_tri_offsets);
    free(graph->node_tri_ids);
    memset(graph, 0, sizeof(*graph));
}

static FMMContext fmm_context_create(int vertex_count, int tri_count) {
    FMMContext ctx = {0};
    ctx.vertex_count = vertex_count;
    ctx.tri_count = tri_count;
    ctx.dist = (float*)malloc((size_t)vertex_count * sizeof(float));
    ctx.prev = (int*)malloc((size_t)vertex_count * sizeof(int));
    ctx.prev_tri = (int*)malloc((size_t)vertex_count * sizeof(int));
    ctx.stamp = (unsigned int*)calloc((size_t)vertex_count, sizeof(unsigned int));
    ctx.state = (unsigned char*)calloc((size_t)vertex_count, sizeof(unsigned char));
    ctx.tri_stamp = (unsigned int*)calloc((size_t)tri_count, sizeof(unsigned int));
    ctx.query_id = 1;
    if (!ctx.dist || !ctx.prev || !ctx.prev_tri || !ctx.stamp || !ctx.state || !ctx.tri_stamp) {
        fprintf(stderr, "out of memory while allocating FMM context\n");
        exit(1);
    }
    return ctx;
}

static void fmm_context_free(FMMContext* ctx) {
    free(ctx->dist);
    free(ctx->prev);
    free(ctx->prev_tri);
    free(ctx->stamp);
    free(ctx->state);
    free(ctx->tri_stamp);
    free(ctx->reached_tris);
    free(ctx->heap.data);
    memset(ctx, 0, sizeof(*ctx));
}

static void fmm_mark_triangle(FMMContext* ctx, int tri) {
    if (tri < 0 || tri >= ctx->tri_count || ctx->tri_stamp[tri] == ctx->query_id) return;
    ctx->tri_stamp[tri] = ctx->query_id;
    if (ctx->reached_tri_count == ctx->reached_tri_cap) {
        ctx->reached_tri_cap = ctx->reached_tri_cap ? ctx->reached_tri_cap * 2 : 128;
        ctx->reached_tris = (int*)checked_realloc(ctx->reached_tris, (size_t)ctx->reached_tri_cap * sizeof(int));
    }
    ctx->reached_tris[ctx->reached_tri_count++] = tri;
}

static float fmm_get_dist(const FMMContext* ctx, int v) {
    return ctx->stamp[v] == ctx->query_id ? ctx->dist[v] : INFINITY;
}

static unsigned char fmm_get_state(const FMMContext* ctx, int v) {
    return ctx->stamp[v] == ctx->query_id ? ctx->state[v] : FMM_FAR;
}

static void fmm_set_vertex(FMMContext* ctx, int v, float dist, unsigned char state, int prev, int prev_tri) {
    ctx->stamp[v] = ctx->query_id;
    ctx->dist[v] = dist;
    ctx->state[v] = state;
    ctx->prev[v] = prev;
    ctx->prev_tri[v] = prev_tri;
}

static void fmm_push_trial(FMMContext* ctx, int v, float dist, int prev, int prev_tri) {
    if (dist >= fmm_get_dist(ctx, v)) return;
    fmm_set_vertex(ctx, v, dist, FMM_TRIAL, prev, prev_tri);
    heap_push(&ctx->heap, (HeapItem){.node = v, .dist = dist});
}

static float sample_distance_from_prop(const PropGraph* graph, const SurfaceSample* sample, const FMMContext* ctx) {
    int begin = graph->tri_node_offsets[sample->tri];
    int end = graph->tri_node_offsets[sample->tri + 1];
    float best = INFINITY;
    for (int i = begin; i < end; ++i) {
        int node = graph->tri_node_ids[i];
        float d = fmm_get_dist(ctx, node);
        if (isfinite(d)) best = fminf(best, d + Vector3Distance(sample->p, graph->nodes[node]));
    }
    return best;
}

static int compute_source_and_measure_steiner(
    const PropGraph* graph,
    const SurfaceTopo* topo,
    SurfaceSample* samples,
    int sample_count,
    int source,
    float radius,
    FMMContext* ctx,
    double* out_ms
) {
    SurfaceSample src = samples[source];
    if (++ctx->query_id == 0) {
        memset(ctx->stamp, 0, (size_t)ctx->vertex_count * sizeof(unsigned int));
        memset(ctx->tri_stamp, 0, (size_t)ctx->tri_count * sizeof(unsigned int));
        ctx->query_id = 1;
    }
    ctx->heap.size = 0;
    ctx->reached_tri_count = 0;

    double t0 = now_seconds();

    float stop_radius = radius + graph->spacing;
    fmm_mark_triangle(ctx, src.tri);
    for (int i = graph->tri_node_offsets[src.tri]; i < graph->tri_node_offsets[src.tri + 1]; ++i) {
        int node = graph->tri_node_ids[i];
        float d = Vector3Distance(src.p, graph->nodes[node]);
        if (d <= stop_radius) fmm_push_trial(ctx, node, d, -1, src.tri);
    }

    while (ctx->heap.size > 0) {
        HeapItem item = heap_pop(&ctx->heap);
        if (item.dist != fmm_get_dist(ctx, item.node)) continue;
        if (item.dist > stop_radius) break;
        fmm_set_vertex(ctx, item.node, item.dist, FMM_ACCEPTED, ctx->prev[item.node], ctx->prev_tri[item.node]);

        for (int i = graph->node_tri_offsets[item.node]; i < graph->node_tri_offsets[item.node + 1]; ++i) {
            fmm_mark_triangle(ctx, graph->node_tri_ids[i]);
        }

        for (int e = graph->edge_offsets[item.node]; e < graph->edge_offsets[item.node + 1]; ++e) {
            int to = graph->edge_to[e];
            if (fmm_get_state(ctx, to) == FMM_ACCEPTED) continue;
            float nd = item.dist + graph->edge_weight[e];
            if (nd <= stop_radius) fmm_push_trial(ctx, to, nd, item.node, graph->edge_tri[e]);
        }
    }

    int in_radius = 0;
    float radius2 = radius * radius;
    for (int ti = 0; ti < ctx->reached_tri_count; ++ti) {
        int reached_tri = ctx->reached_tris[ti];
        int begin = topo->tri_sample_offsets[reached_tri];
        int end = topo->tri_sample_offsets[reached_tri + 1];
        for (int si = begin; si < end; ++si) {
            int i = topo->tri_sample_ids[si];
            if (samples[i].disabled) continue;
            if (samples[i].stamp == ctx->query_id) continue;
            samples[i].stamp = ctx->query_id;
            samples[i].distance = INFINITY;
            samples[i].in_radius = false;
            Vector3 delta = Vector3Subtract(samples[i].p, src.p);
            if (Vector3DotProduct(delta, delta) > radius2) continue;
            float d = (i == source) ? 0.0f : sample_distance_from_prop(graph, &samples[i], ctx);
            samples[i].distance = d;
            samples[i].in_radius = d <= radius;
            if (samples[i].in_radius) ++in_radius;
        }
    }

    double t1 = now_seconds();
    *out_ms = 1000.0 * (t1 - t0);
    return in_radius;
}

static int pick_sample_screen(
    const SurfaceSample* samples,
    int sample_count,
    Camera3D camera,
    Vector2 mouse,
    bool require_in_radius,
    unsigned int query_id
) {
    float best_d2 = 12.0f * 12.0f;
    int best = -1;
    for (int i = 0; i < sample_count; ++i) {
        if (samples[i].disabled) continue;
        if (require_in_radius && !(samples[i].stamp == query_id && samples[i].in_radius)) continue;
        Vector2 p = GetWorldToScreen(samples[i].p, camera);
        float dx = p.x - mouse.x;
        float dy = p.y - mouse.y;
        float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best = i;
        }
    }
    return best;
}

static float distance_point_segment_2d(Vector2 p, Vector2 a, Vector2 b) {
    Vector2 ab = Vector2Subtract(b, a);
    float len2 = Vector2DotProduct(ab, ab);
    if (len2 <= 1e-20f) return Vector2Distance(p, a);
    float t = Clamp(Vector2DotProduct(Vector2Subtract(p, a), ab) / len2, 0.0f, 1.0f);
    Vector2 q = Vector2Add(a, Vector2Scale(ab, t));
    return Vector2Distance(p, q);
}

static float distance_point_segment_3d(Vector3 p, Vector3 a, Vector3 b) {
    Vector3 ab = Vector3Subtract(b, a);
    float len2 = Vector3DotProduct(ab, ab);
    if (len2 <= 1e-20f) return Vector3Distance(p, a);
    float t = Clamp(Vector3DotProduct(Vector3Subtract(p, a), ab) / len2, 0.0f, 1.0f);
    Vector3 q = Vector3Add(a, Vector3Scale(ab, t));
    return Vector3Distance(p, q);
}

static Vector2 lerp2(Vector2 a, Vector2 b, float t) {
    return Vector2Add(a, Vector2Scale(Vector2Subtract(b, a), t));
}

static bool clip_segment_by_values(float f0, float f1, float* lo, float* hi) {
    const float eps = -1e-6f;
    if (f0 >= eps && f1 >= eps) return true;
    if (f0 < eps && f1 < eps) return false;
    float denom = f0 - f1;
    if (fabsf(denom) < 1e-12f) return false;
    float t = f0 / denom;
    if (f0 < eps) *lo = fmaxf(*lo, t);
    else *hi = fminf(*hi, t);
    return *lo <= *hi + 1e-6f;
}

static float visible_cone_value(Vector2 source, Vector2 a, Vector2 b, Vector2 p, int side) {
    Vector2 sa = Vector2Subtract(a, source);
    Vector2 sb = Vector2Subtract(b, source);
    Vector2 sp = Vector2Subtract(p, source);
    if (side == 0) return cross2v(sa, sp);
    if (side == 1) return -cross2v(sb, sp);
    float source_side = cross2v(Vector2Subtract(b, a), Vector2Subtract(source, a));
    float point_side = cross2v(Vector2Subtract(b, a), Vector2Subtract(p, a));
    return -source_side * point_side;
}

static bool geo_window_clip_edge(const GeoWindow* w, int edge, float* out_t0, float* out_t1) {
    if (w->entry_edge < 0) {
        *out_t0 = 0.0f;
        *out_t1 = 1.0f;
        return true;
    }
    Vector2 entry_a = w->tri2d[(w->entry_edge + 1) % 3];
    Vector2 entry_b = w->tri2d[(w->entry_edge + 2) % 3];
    Vector2 a = lerp2(entry_a, entry_b, w->t0);
    Vector2 b = lerp2(entry_a, entry_b, w->t1);
    Vector2 edge_a = w->tri2d[(edge + 1) % 3];
    Vector2 edge_b = w->tri2d[(edge + 2) % 3];
    float cone = cross2v(Vector2Subtract(a, w->source2), Vector2Subtract(b, w->source2));
    if (fabsf(cone) < 1e-10f) return false;

    float lo = 0.0f;
    float hi = 1.0f;
    float sign = cone > 0.0f ? 1.0f : -1.0f;
    float f0 = sign * visible_cone_value(w->source2, a, b, edge_a, 0);
    float f1 = sign * visible_cone_value(w->source2, a, b, edge_b, 0);
    if (!clip_segment_by_values(f0, f1, &lo, &hi)) return false;
    f0 = sign * visible_cone_value(w->source2, a, b, edge_a, 1);
    f1 = sign * visible_cone_value(w->source2, a, b, edge_b, 1);
    if (!clip_segment_by_values(f0, f1, &lo, &hi)) return false;
    f0 = visible_cone_value(w->source2, a, b, edge_a, 2);
    f1 = visible_cone_value(w->source2, a, b, edge_b, 2);
    if (!clip_segment_by_values(f0, f1, &lo, &hi)) return false;

    *out_t0 = Clamp(lo, 0.0f, 1.0f);
    *out_t1 = Clamp(hi, 0.0f, 1.0f);
    return *out_t1 >= *out_t0 - 1e-6f;
}

static bool geo_window_sees_point(const GeoWindow* w, Vector2 p) {
    if (w->entry_edge < 0) return true;
    Vector2 entry_a = w->tri2d[(w->entry_edge + 1) % 3];
    Vector2 entry_b = w->tri2d[(w->entry_edge + 2) % 3];
    Vector2 a = lerp2(entry_a, entry_b, w->t0);
    Vector2 b = lerp2(entry_a, entry_b, w->t1);
    float cone = cross2v(Vector2Subtract(a, w->source2), Vector2Subtract(b, w->source2));
    if (fabsf(cone) < 1e-10f) return false;
    float sign = cone > 0.0f ? 1.0f : -1.0f;
    return sign * visible_cone_value(w->source2, a, b, p, 0) >= -1e-5f &&
        sign * visible_cone_value(w->source2, a, b, p, 1) >= -1e-5f &&
        visible_cone_value(w->source2, a, b, p, 2) >= -1e-5f;
}

static bool geo_window_dominated(const GeoWindowVec* windows, const IntVec* state_windows, int state, float t0, float t1, float key) {
    const IntVec* list = &state_windows[state];
    for (int i = 0; i < list->size; ++i) {
        const GeoWindow* w = &windows->data[list->data[i]];
        if (w->t0 <= t0 + 1e-5f && w->t1 >= t1 - 1e-5f && w->key <= key + 1e-5f) return true;
    }
    return false;
}

static float triangle_vertex_angle(const SurfaceMesh* surface, int tri, int local) {
    Vector3 p = surface->vertices[surface->tris[3 * tri + local]];
    Vector3 a = surface->vertices[surface->tris[3 * tri + (local + 1) % 3]];
    Vector3 b = surface->vertices[surface->tris[3 * tri + (local + 2) % 3]];
    Vector3 pa = Vector3Normalize(Vector3Subtract(a, p));
    Vector3 pb = Vector3Normalize(Vector3Subtract(b, p));
    return acosf(Clamp(Vector3DotProduct(pa, pb), -1.0f, 1.0f));
}

static bool should_seed_pseudo_source_vertex(const SurfaceMesh* surface, const PropGraph* graph, int vid) {
    int begin = graph->node_tri_offsets[vid];
    int end = graph->node_tri_offsets[vid + 1];
    int incident = end - begin;
    if (incident <= 0) return false;

    float angle_sum = 0.0f;
    int boundary_edges = 0;
    for (int i = begin; i < end; ++i) {
        int tri = graph->node_tri_ids[i];
        int local = tri_local_index(surface, tri, vid);
        if (local < 0) continue;
        angle_sum += triangle_vertex_angle(surface, tri, local);
        for (int e = 0; e < 3; ++e) {
            int a = surface->tris[3 * tri + (e + 1) % 3];
            int b = surface->tris[3 * tri + (e + 2) % 3];
            if ((a == vid || b == vid) && surface->tri_neighbors[3 * tri + e] < 0) ++boundary_edges;
        }
    }

    return boundary_edges > 0 || angle_sum > 2.0f * PI + 1e-3f;
}

static bool window_chain_uses_pseudo_source(const GeoWindowVec* windows, int wi) {
    for (int cur = wi; cur >= 0; cur = windows->data[cur].parent) {
        if (windows->data[cur].source_vertex >= 0) return true;
    }
    return false;
}

static bool triangle_to_2d(const SurfaceMesh* surface, int tri, Vector2 out[3]) {
    Vector3 p0 = surface->vertices[surface->tris[3 * tri + 0]];
    Vector3 p1 = surface->vertices[surface->tris[3 * tri + 1]];
    Vector3 p2 = surface->vertices[surface->tris[3 * tri + 2]];
    float l01 = Vector3Distance(p0, p1);
    if (l01 < 1e-8f) return false;
    float x2 = Vector3DotProduct(Vector3Subtract(p2, p0), Vector3Scale(Vector3Subtract(p1, p0), 1.0f / l01));
    float y2 = sqrtf(fmaxf(0.0f, Vector3DistanceSqr(p0, p2) - x2 * x2));
    if (y2 < 1e-8f) return false;
    out[0] = (Vector2){0.0f, 0.0f};
    out[1] = (Vector2){l01, 0.0f};
    out[2] = (Vector2){x2, y2};
    return true;
}

static GeodesicPathContext geodesic_path_context_create(const SurfaceMesh* surface, const PropGraph* graph) {
    GeodesicPathContext ctx = {0};
    ctx.state_count = surface->tri_count * 4;
    ctx.vertex_count = surface->vertex_count;
    ctx.query_id = 1;
    ctx.state_windows = (IntVec*)calloc((size_t)ctx.state_count, sizeof(IntVec));
    ctx.state_stamp = (unsigned int*)calloc((size_t)ctx.state_count, sizeof(unsigned int));
    ctx.best_vertex = (float*)malloc((size_t)ctx.vertex_count * sizeof(float));
    ctx.vertex_stamp = (unsigned int*)calloc((size_t)ctx.vertex_count, sizeof(unsigned int));
    ctx.pseudo_source_vertex = (unsigned char*)calloc((size_t)ctx.vertex_count, sizeof(unsigned char));
    ctx.tri2d_base = (Vector2*)calloc((size_t)surface->tri_count * 3, sizeof(Vector2));
    ctx.tri2d_valid = (unsigned char*)calloc((size_t)surface->tri_count, sizeof(unsigned char));
    if (!ctx.state_windows || !ctx.state_stamp || !ctx.best_vertex || !ctx.vertex_stamp ||
        !ctx.pseudo_source_vertex || !ctx.tri2d_base || !ctx.tri2d_valid) {
        fprintf(stderr, "out of memory while allocating geodesic path context\n");
        exit(1);
    }

    for (int tri = 0; tri < surface->tri_count; ++tri) {
        ctx.tri2d_valid[tri] = triangle_to_2d(surface, tri, &ctx.tri2d_base[3 * tri]) ? 1 : 0;
    }
    for (int v = 0; v < surface->vertex_count; ++v) {
        ctx.pseudo_source_vertex[v] = should_seed_pseudo_source_vertex(surface, graph, v) ? 1 : 0;
    }
    return ctx;
}

static void geodesic_path_context_free(GeodesicPathContext* ctx) {
    for (int i = 0; i < ctx->state_count; ++i) int_vec_free(&ctx->state_windows[i]);
    free(ctx->state_windows);
    free(ctx->state_stamp);
    int_vec_free(&ctx->touched_states);
    free(ctx->best_vertex);
    free(ctx->vertex_stamp);
    int_vec_free(&ctx->touched_vertices);
    free(ctx->pseudo_source_vertex);
    free(ctx->tri2d_base);
    free(ctx->tri2d_valid);
    free(ctx->windows.data);
    free(ctx->heap.data);
    memset(ctx, 0, sizeof(*ctx));
}

static void geodesic_path_context_begin(GeodesicPathContext* ctx) {
    if (++ctx->query_id == 0) {
        memset(ctx->state_stamp, 0, (size_t)ctx->state_count * sizeof(unsigned int));
        memset(ctx->vertex_stamp, 0, (size_t)ctx->vertex_count * sizeof(unsigned int));
        ctx->query_id = 1;
    }
    for (int i = 0; i < ctx->touched_states.size; ++i) ctx->state_windows[ctx->touched_states.data[i]].size = 0;
    ctx->touched_states.size = 0;
    ctx->touched_vertices.size = 0;
    ctx->windows.size = 0;
    ctx->heap.size = 0;
    ctx->windows_pushed = 0;
    ctx->windows_popped = 0;
    ctx->pseudo_sources = 0;
    ctx->touched_state_count = 0;
    ctx->max_state_windows = 0;
}

static IntVec* geodesic_state_list(GeodesicPathContext* ctx, int state) {
    if (ctx->state_stamp[state] != ctx->query_id) {
        ctx->state_stamp[state] = ctx->query_id;
        ctx->state_windows[state].size = 0;
        int_vec_push(&ctx->touched_states, state);
        ctx->touched_state_count = ctx->touched_states.size;
    }
    return &ctx->state_windows[state];
}

static float geodesic_get_best_vertex(const GeodesicPathContext* ctx, int vertex) {
    return ctx->vertex_stamp[vertex] == ctx->query_id ? ctx->best_vertex[vertex] : INFINITY;
}

static void geodesic_set_best_vertex(GeodesicPathContext* ctx, int vertex, float value) {
    if (ctx->vertex_stamp[vertex] != ctx->query_id) {
        ctx->vertex_stamp[vertex] = ctx->query_id;
        int_vec_push(&ctx->touched_vertices, vertex);
    }
    ctx->best_vertex[vertex] = value;
}

static bool copy_precomputed_tri2d(const GeodesicPathContext* ctx, int tri, Vector2 out[3]) {
    if (!ctx->tri2d_valid[tri]) return false;
    out[0] = ctx->tri2d_base[3 * tri + 0];
    out[1] = ctx->tri2d_base[3 * tri + 1];
    out[2] = ctx->tri2d_base[3 * tri + 2];
    return true;
}

static void geodesic_push_window(GeodesicPathContext* ctx, GeoWindow window, int state, float priority) {
    int wi = ctx->windows.size;
    geo_window_vec_push(&ctx->windows, window);
    IntVec* list = geodesic_state_list(ctx, state);
    int_vec_push(list, wi);
    if (list->size > ctx->max_state_windows) ctx->max_state_windows = list->size;
    window_heap_push(&ctx->heap, (WindowHeapItem){.window = wi, .key = priority});
    ++ctx->windows_pushed;
}

static int local_edge_between(const SurfaceMesh* surface, int tri, int va, int vb) {
    for (int e = 0; e < 3; ++e) {
        int a = surface->tris[3 * tri + (e + 1) % 3];
        int b = surface->tris[3 * tri + (e + 2) % 3];
        if ((a == va && b == vb) || (a == vb && b == va)) return e;
    }
    return -1;
}

static bool unfold_neighbor_triangle(
    const SurfaceMesh* surface,
    int prev_tri,
    int edge,
    int curr_tri,
    const Vector2 prev2d[3],
    Vector2 curr2d[3]
) {
    int shared0 = surface->tris[3 * prev_tri + (edge + 1) % 3];
    int shared1 = surface->tris[3 * prev_tri + (edge + 2) % 3];
    int prev_u = tri_local_index(surface, prev_tri, shared0);
    int prev_v = tri_local_index(surface, prev_tri, shared1);
    int curr_u = tri_local_index(surface, curr_tri, shared0);
    int curr_v = tri_local_index(surface, curr_tri, shared1);
    if (prev_u < 0 || prev_v < 0 || curr_u < 0 || curr_v < 0) return false;

    int curr_w = 3 - curr_u - curr_v;
    int w_vid = surface->tris[3 * curr_tri + curr_w];
    Vector2 u2 = prev2d[prev_u];
    Vector2 v2 = prev2d[prev_v];
    Vector2 uv = Vector2Subtract(v2, u2);
    float len = Vector2Length(uv);
    if (len < 1e-8f) return false;

    Vector3 u3 = surface->vertices[shared0];
    Vector3 v3 = surface->vertices[shared1];
    Vector3 w3 = surface->vertices[w_vid];
    float duw = Vector3Distance(u3, w3);
    float dvw = Vector3Distance(v3, w3);
    float x = (duw * duw - dvw * dvw + len * len) / (2.0f * len);
    float h = sqrtf(fmaxf(0.0f, duw * duw - x * x));
    Vector2 ex = Vector2Scale(uv, 1.0f / len);
    Vector2 perp = (Vector2){-ex.y, ex.x};
    float side = cross2v(uv, Vector2Subtract(prev2d[edge], u2));
    if (side > 0.0f) h = -h;

    curr2d[curr_u] = u2;
    curr2d[curr_v] = v2;
    curr2d[curr_w] = Vector2Add(u2, Vector2Add(Vector2Scale(ex, x), Vector2Scale(perp, h)));
    return true;
}

static Vector2 point_in_window_2d(const SurfaceMesh* surface, const GeoWindow* w, Vector3 p) {
    float u, v, q;
    barycentric3(p,
        surface->vertices[surface->tris[3 * w->tri + 0]],
        surface->vertices[surface->tris[3 * w->tri + 1]],
        surface->vertices[surface->tris[3 * w->tri + 2]], &u, &v, &q);
    return barycentric_to_2d(u, v, q, w->tri2d[0], w->tri2d[1], w->tri2d[2]);
}

static void vec3_vec_push_unique(Vec3Vec* path, Vector3 p) {
    if (path->size == 0 || Vector3Distance(path->data[path->size - 1], p) > 1e-7f) vec3_vec_push(path, p);
}

static bool append_window_segment_backtrace(
    const SurfaceMesh* surface,
    const GeoWindowVec* windows,
    int start,
    int wi,
    Vector3 start_p,
    Vector3 end_p,
    Vec3Vec* path,
    LoopSegmentVec* segments,
    bool append
) {
    Vec3Vec rev = {0};
    LoopSegmentVec rev_segments = {0};
    Vector3 cur_p = end_p;
    vec3_vec_push(&rev, end_p);

    for (int cur = wi; cur != start; cur = windows->data[cur].parent) {
        const GeoWindow* w = &windows->data[cur];
        if (w->entry_edge < 0 || w->parent < 0) {
            vec3_vec_free(&rev);
            loop_segment_vec_free(&rev_segments);
            return false;
        }

        Vector2 cur2 = point_in_window_2d(surface, w, cur_p);
        int a = (w->entry_edge + 1) % 3;
        int b = (w->entry_edge + 2) % 3;
        float lu;
        if (!robust_portal_hit_2d(w->source2, cur2, w->tri2d[a], w->tri2d[b], w->t0, w->t1, &lu)) {
            vec3_vec_free(&rev);
            loop_segment_vec_free(&rev_segments);
            return false;
        }

        int va = surface->tris[3 * w->tri + a];
        int vb = surface->tris[3 * w->tri + b];
        Vector3 q = Vector3Add(
            Vector3Scale(surface->vertices[va], 1.0f - lu),
            Vector3Scale(surface->vertices[vb], lu));
        loop_segment_vec_push(&rev_segments, (LoopSegment){.tri = w->tri, .a = q, .b = cur_p});
        vec3_vec_push_unique(&rev, q);
        cur_p = q;
    }

    loop_segment_vec_push(&rev_segments, (LoopSegment){.tri = windows->data[start].tri, .a = start_p, .b = cur_p});
    vec3_vec_push_unique(&rev, start_p);
    if (!append) vec3_vec_clear(path);
    for (int i = rev.size - 1; i >= 0; --i) vec3_vec_push_unique(path, rev.data[i]);
    if (segments) {
        for (int i = rev_segments.size - 1; i >= 0; --i) loop_segment_vec_push(segments, rev_segments.data[i]);
    }
    vec3_vec_free(&rev);
    loop_segment_vec_free(&rev_segments);
    return true;
}

static bool append_window_chain_path(
    const SurfaceMesh* surface,
    const SurfaceSample* samples,
    int source,
    const GeoWindowVec* windows,
    int wi,
    Vector3 end_p,
    int end_tri,
    Vec3Vec* path,
    LoopSegmentVec* segments,
    bool append
) {
    if (wi < 0) return false;
    int source_vertex = windows->data[wi].source_vertex;
    int start = wi;
    while (windows->data[start].parent >= 0 && windows->data[windows->data[start].parent].source_vertex == source_vertex) {
        start = windows->data[start].parent;
    }

    if (source_vertex >= 0) {
        int parent = windows->data[start].parent;
        if (parent < 0) return false;
        Vector3 vertex_p = surface->vertices[source_vertex];
        if (!append_window_chain_path(surface, samples, source, windows, parent, vertex_p, windows->data[parent].tri, path, segments, append)) return false;
        append = true;
    }

    (void)end_tri;
    Vector3 start_p = source_vertex >= 0 ? surface->vertices[source_vertex] : samples[source].p;
    return append_window_segment_backtrace(surface, windows, start, wi, start_p, end_p, path, segments, append);
}

static bool build_geodesic_path(
    const SurfaceMesh* surface,
    const PropGraph* graph,
    const SurfaceSample* samples,
    int source,
    int target,
    const FMMContext* ctx,
    GeodesicPathContext* path_ctx,
    Vec3Vec* path,
    LoopSegmentVec* segments
) {
    vec3_vec_clear(path);
    if (segments) loop_segment_vec_clear(segments);
    if (source < 0 || target < 0) return false;
    if (!(samples[target].stamp == ctx->query_id && samples[target].in_radius)) return false;

    if (target == source) {
        vec3_vec_push(path, samples[source].p);
        return true;
    }

    if (samples[source].tri == samples[target].tri) {
        vec3_vec_push(path, samples[source].p);
        vec3_vec_push(path, samples[target].p);
        if (segments) loop_segment_vec_push(segments, (LoopSegment){.tri = samples[source].tri, .a = samples[source].p, .b = samples[target].p});
        return true;
    }

    geodesic_path_context_begin(path_ctx);

    GeoWindow root = {0};
    root.tri = samples[source].tri;
    root.entry_edge = -1;
    root.parent = -1;
    root.key = 0.0f;
    root.offset = 0.0f;
    root.t0 = 0.0f;
    root.t1 = 1.0f;
    root.source_vertex = -1;
    if (!copy_precomputed_tri2d(path_ctx, root.tri, root.tri2d)) return false;
    float su, sv, sw;
    barycentric3(samples[source].p,
        surface->vertices[surface->tris[3 * root.tri + 0]],
        surface->vertices[surface->tris[3 * root.tri + 1]],
        surface->vertices[surface->tris[3 * root.tri + 2]], &su, &sv, &sw);
    root.source2 = barycentric_to_2d(su, sv, sw, root.tri2d[0], root.tri2d[1], root.tri2d[2]);
    geodesic_push_window(path_ctx, root, root.tri * 4, 0.0f);

    float best_target = (samples[target].stamp == ctx->query_id && isfinite(samples[target].distance)) ? samples[target].distance * 1.0001f + 1e-6f : INFINITY;
    int best_target_window = -1;
    while (path_ctx->heap.size > 0) {
        WindowHeapItem item = window_heap_pop(&path_ctx->heap);
        ++path_ctx->windows_popped;
        if (item.key >= best_target) break;
        GeoWindow w = path_ctx->windows.data[item.window];

        if (w.tri == samples[target].tri) {
            float tu, tv, tw;
            barycentric3(samples[target].p,
                surface->vertices[surface->tris[3 * w.tri + 0]],
                surface->vertices[surface->tris[3 * w.tri + 1]],
                surface->vertices[surface->tris[3 * w.tri + 2]], &tu, &tv, &tw);
            Vector2 target2 = barycentric_to_2d(tu, tv, tw, w.tri2d[0], w.tri2d[1], w.tri2d[2]);
            if (!geo_window_sees_point(&w, target2)) continue;
            float d = w.offset + Vector2Distance(w.source2, target2);
            bool better = d < best_target - 1e-6f;
            if (!better && fabsf(d - best_target) <= 1e-6f && best_target_window >= 0) {
                better = window_chain_uses_pseudo_source(&path_ctx->windows, best_target_window) && !window_chain_uses_pseudo_source(&path_ctx->windows, item.window);
            }
            if (better) {
                best_target = d;
                best_target_window = item.window;
            }
        }

        for (int edge = 0; edge < 3; ++edge) {
            if (edge == w.entry_edge) continue;
            int next_tri = surface->tri_neighbors[3 * w.tri + edge];
            if (next_tri < 0) continue;

            int a = (edge + 1) % 3;
            int b = (edge + 2) % 3;
            float clipped_t0;
            float clipped_t1;
            if (!geo_window_clip_edge(&w, edge, &clipped_t0, &clipped_t1)) continue;

            int va = surface->tris[3 * w.tri + a];
            int vb = surface->tris[3 * w.tri + b];
            int endpoint_vids[2] = {va, vb};
            float endpoint_ts[2] = {clipped_t0, clipped_t1};
            Vector2 endpoint_points[2] = {w.tri2d[a], w.tri2d[b]};
            for (int pi = 0; pi < 2; ++pi) {
                if ((pi == 0 && endpoint_ts[pi] > 1e-5f) || (pi == 1 && endpoint_ts[pi] < 1.0f - 1e-5f)) continue;
                int vid = endpoint_vids[pi];
                if (!path_ctx->pseudo_source_vertex[vid]) continue;
                float vd = w.offset + Vector2Distance(w.source2, endpoint_points[pi]);
                if (vd >= geodesic_get_best_vertex(path_ctx, vid) - 1e-6f || vd >= best_target) continue;
                geodesic_set_best_vertex(path_ctx, vid, vd);
                ++path_ctx->pseudo_sources;

                int begin = graph->node_tri_offsets[vid];
                int end = graph->node_tri_offsets[vid + 1];
                for (int ti = begin; ti < end; ++ti) {
                    int seed_tri = graph->node_tri_ids[ti];
                    GeoWindow vw = {0};
                    vw.tri = seed_tri;
                    vw.entry_edge = -1;
                    vw.parent = item.window;
                    vw.key = vd;
                    vw.offset = vd;
                    vw.t0 = 0.0f;
                    vw.t1 = 1.0f;
                    vw.source_vertex = vid;
                    if (!copy_precomputed_tri2d(path_ctx, seed_tri, vw.tri2d)) continue;
                    int local = tri_local_index(surface, seed_tri, vid);
                    if (local < 0) continue;
                    vw.source2 = vw.tri2d[local];
                    float priority = vd + Vector3Distance(surface->vertices[vid], samples[target].p);
                    geodesic_push_window(path_ctx, vw, seed_tri * 4, priority);
                }
            }

            if (clipped_t1 < clipped_t0 + 1e-6f) continue;

            Vector2 clipped_a = lerp2(w.tri2d[a], w.tri2d[b], clipped_t0);
            Vector2 clipped_b = lerp2(w.tri2d[a], w.tri2d[b], clipped_t1);
            float key = w.offset + distance_point_segment_2d(w.source2, clipped_a, clipped_b);
            if (key >= best_target) continue;

            int next_entry = local_edge_between(surface, next_tri, va, vb);
            if (next_entry < 0) continue;
            int next_state = next_tri * 4 + next_entry + 1;

            GeoWindow nw = {0};
            nw.tri = next_tri;
            nw.entry_edge = next_entry;
            nw.parent = item.window;
            nw.key = key;
            nw.offset = w.offset;
            nw.source_vertex = w.source_vertex;
            nw.source2 = w.source2;
            if (!unfold_neighbor_triangle(surface, w.tri, edge, next_tri, w.tri2d, nw.tri2d)) continue;
            int na = surface->tris[3 * next_tri + (next_entry + 1) % 3];
            int nb = surface->tris[3 * next_tri + (next_entry + 2) % 3];
            if (na == va && nb == vb) {
                nw.t0 = clipped_t0;
                nw.t1 = clipped_t1;
            } else {
                nw.t0 = 1.0f - clipped_t1;
                nw.t1 = 1.0f - clipped_t0;
            }
            geodesic_state_list(path_ctx, next_state);
            if (geo_window_dominated(&path_ctx->windows, path_ctx->state_windows, next_state, nw.t0, nw.t1, key)) continue;
            Vector3 edge_a3 = Vector3Add(Vector3Scale(surface->vertices[va], 1.0f - clipped_t0), Vector3Scale(surface->vertices[vb], clipped_t0));
            Vector3 edge_b3 = Vector3Add(Vector3Scale(surface->vertices[va], 1.0f - clipped_t1), Vector3Scale(surface->vertices[vb], clipped_t1));
            float priority = key + distance_point_segment_3d(samples[target].p, edge_a3, edge_b3);
            geodesic_push_window(path_ctx, nw, next_state, priority);
        }
    }

    bool ok = false;
    if (best_target_window >= 0) {
        Vec3Vec tmp_path = {0};
        LoopSegmentVec tmp_segments = {0};
        ok = append_window_chain_path(surface, samples, source, &path_ctx->windows, best_target_window, samples[target].p, samples[target].tri, &tmp_path, segments ? &tmp_segments : NULL, false);
        if (ok) {
            vec3_vec_clear(path);
            for (int i = 0; i < tmp_path.size; ++i) vec3_vec_push(path, tmp_path.data[i]);
            if (segments) {
                loop_segment_vec_clear(segments);
                for (int i = 0; i < tmp_segments.size; ++i) loop_segment_vec_push(segments, tmp_segments.data[i]);
            }
        }
        vec3_vec_free(&tmp_path);
        loop_segment_vec_free(&tmp_segments);
        if (!ok) {
            fprintf(stderr, "continuous geodesic path emission failed: source_tri=%d target_tri=%d windows=%d popped=%d distance=%.8f\n",
                samples[source].tri, samples[target].tri, path_ctx->windows.size, path_ctx->windows_popped, best_target);
        }
    } else {
        fprintf(stderr, "continuous geodesic propagation found no visible target window: source_tri=%d target_tri=%d windows=%d popped=%d\n",
            samples[source].tri, samples[target].tri, path_ctx->windows.size, path_ctx->windows_popped);
    }
    return ok;
}

static void draw_path_color(const Vec3Vec* path, float radius, Color color) {
    if (path->size < 2) return;
    float r = fmaxf(radius * 0.0015f, 0.00025f);
    for (int i = 0; i < path->size - 1; ++i) {
        DrawLine3D(path->data[i], path->data[i + 1], color);
        DrawSphere(path->data[i], r, color);
    }
    DrawSphere(path->data[path->size - 1], r, color);
}

static void draw_geodesic_path(const Vec3Vec* path, float radius) {
    draw_path_color(path, radius, (Color){255, 80, 220, 220});
}

enum {
    LOOP_EDGE_BLOCKED = 1,
    LOOP_EDGE_LEFT = 2,
    LOOP_EDGE_RIGHT = 4,
};

typedef struct {
    int disabled_samples;
    int left_tris;
    int right_tris;
    float left_area;
    float right_area;
    int chosen_side;
    int classification_conflicts;
    int flood_conflicts;
} LoopRemovalStats;

typedef struct {
    unsigned char* flags;
    unsigned char* boundary_tri;
    unsigned char* side_mark;
    unsigned char* conflict_tri;
    LoopPortal* portals;
    int tri_count;
    int portal_count;
} LoopDebugData;

static void loop_debug_free(LoopDebugData* debug) {
    free(debug->flags);
    free(debug->boundary_tri);
    free(debug->side_mark);
    free(debug->conflict_tri);
    free(debug->portals);
    memset(debug, 0, sizeof(*debug));
}

static void loop_debug_capture(
    LoopDebugData* debug,
    const unsigned char* flags,
    const unsigned char* boundary_tri,
    const unsigned char* side_mark,
    const unsigned char* conflict_tri,
    const LoopPortalVec* portals,
    int tri_count
) {
    loop_debug_free(debug);
    debug->tri_count = tri_count;
    debug->flags = (unsigned char*)malloc((size_t)tri_count * 3 * sizeof(unsigned char));
    debug->boundary_tri = (unsigned char*)malloc((size_t)tri_count * sizeof(unsigned char));
    debug->side_mark = (unsigned char*)malloc((size_t)tri_count * sizeof(unsigned char));
    debug->conflict_tri = (unsigned char*)malloc((size_t)tri_count * sizeof(unsigned char));
    debug->portal_count = portals ? portals->size : 0;
    debug->portals = debug->portal_count > 0 ? (LoopPortal*)malloc((size_t)debug->portal_count * sizeof(LoopPortal)) : NULL;
    if (!debug->flags || !debug->boundary_tri || !debug->side_mark || !debug->conflict_tri) {
        fprintf(stderr, "out of memory while capturing loop debug data\n");
        exit(1);
    }
    if (debug->portal_count > 0 && !debug->portals) {
        fprintf(stderr, "out of memory while capturing loop portal debug data\n");
        exit(1);
    }
    memcpy(debug->flags, flags, (size_t)tri_count * 3 * sizeof(unsigned char));
    memcpy(debug->boundary_tri, boundary_tri, (size_t)tri_count * sizeof(unsigned char));
    memcpy(debug->side_mark, side_mark, (size_t)tri_count * sizeof(unsigned char));
    memcpy(debug->conflict_tri, conflict_tri, (size_t)tri_count * sizeof(unsigned char));
    if (debug->portal_count > 0) memcpy(debug->portals, portals->data, (size_t)debug->portal_count * sizeof(LoopPortal));
}

static Vector2 surface_point_to_tri2d(const SurfaceMesh* surface, int tri, Vector3 p, const Vector2 tri2d[3]) {
    float u, v, w;
    barycentric3(p,
        surface->vertices[surface->tris[3 * tri + 0]],
        surface->vertices[surface->tris[3 * tri + 1]],
        surface->vertices[surface->tris[3 * tri + 2]], &u, &v, &w);
    return barycentric_to_2d(u, v, w, tri2d[0], tri2d[1], tri2d[2]);
}

static void loop_mark_vertex_fan_boundary(const SurfaceMesh* surface, const PropGraph* graph, int vertex, unsigned char* boundary_tri) {
    if (!graph || vertex < 0 || vertex >= surface->vertex_count) return;
    for (int i = graph->node_tri_offsets[vertex]; i < graph->node_tri_offsets[vertex + 1]; ++i) {
        boundary_tri[graph->node_tri_ids[i]] = 1;
    }
}

static void loop_mark_point_vertex_fan_boundary(
    const SurfaceMesh* surface,
    const PropGraph* graph,
    int tri,
    Vector3 p,
    unsigned char* boundary_tri
) {
    float u, v, w;
    barycentric3(p,
        surface->vertices[surface->tris[3 * tri + 0]],
        surface->vertices[surface->tris[3 * tri + 1]],
        surface->vertices[surface->tris[3 * tri + 2]], &u, &v, &w);
    float b[3] = {u, v, w};
    for (int i = 0; i < 3; ++i) {
        if (b[i] >= 1.0f - 1e-5f) loop_mark_vertex_fan_boundary(surface, graph, surface->tris[3 * tri + i], boundary_tri);
    }
}

static void loop_mark_segment_blocked_edges(
    const SurfaceMesh* surface,
    const PropGraph* graph,
    const LoopSegment* seg,
    unsigned char* flags,
    unsigned char* boundary_tri,
    int* conflicts
) {
    Vector2 tri2d[3];
    if (!triangle_to_2d(surface, seg->tri, tri2d)) return;
    Vector2 a2 = surface_point_to_tri2d(surface, seg->tri, seg->a, tri2d);
    Vector2 b2 = surface_point_to_tri2d(surface, seg->tri, seg->b, tri2d);
    Vector2 dir = Vector2Subtract(b2, a2);
    if (Vector2LengthSqr(dir) < 1e-14f) return;

    boundary_tri[seg->tri] = 1;
    loop_mark_point_vertex_fan_boundary(surface, graph, seg->tri, seg->a, boundary_tri);
    loop_mark_point_vertex_fan_boundary(surface, graph, seg->tri, seg->b, boundary_tri);
    for (int edge = 0; edge < 3; ++edge) {
        int ea = (edge + 1) % 3;
        int eb = (edge + 2) % 3;
        float t, u;
        if (!segment_intersect_2d(a2, b2, tri2d[ea], tri2d[eb], &t, &u)) continue;
        if (u <= 1e-5f || u >= 1.0f - 1e-5f) {
            int vertex = surface->tris[3 * seg->tri + (u <= 1e-5f ? ea : eb)];
            loop_mark_vertex_fan_boundary(surface, graph, vertex, boundary_tri);
            continue;
        }

        flags[3 * seg->tri + edge] |= LOOP_EDGE_BLOCKED;
        int nb = surface->tri_neighbors[3 * seg->tri + edge];
        if (nb >= 0) {
            int va = surface->tris[3 * seg->tri + (edge + 1) % 3];
            int vb = surface->tris[3 * seg->tri + (edge + 2) % 3];
            int nb_edge = local_edge_between(surface, nb, va, vb);
            if (nb_edge >= 0) flags[3 * nb + nb_edge] |= LOOP_EDGE_BLOCKED;
        }
    }
    (void)conflicts;
}

static bool loop_nearest_segment_side_in_tri(
    const SurfaceMesh* surface,
    const LoopSegmentVec* loop_segments,
    int tri,
    Vector2 p2,
    const Vector2 tri2d[3],
    int* out_side
) {
    float best = INFINITY;
    float best_side = 0.0f;
    for (int i = 0; i < loop_segments->size; ++i) {
        const LoopSegment* seg = &loop_segments->data[i];
        if (seg->tri != tri) continue;
        Vector2 a2 = surface_point_to_tri2d(surface, tri, seg->a, tri2d);
        Vector2 b2 = surface_point_to_tri2d(surface, tri, seg->b, tri2d);
        Vector2 dir = Vector2Subtract(b2, a2);
        if (Vector2LengthSqr(dir) < 1e-14f) continue;
        float d = distance_point_segment_2d(p2, a2, b2);
        if (d < best) {
            best = d;
            best_side = cross2v(dir, Vector2Subtract(p2, a2));
        }
    }
    if (!isfinite(best) || fabsf(best_side) <= 1e-7f) return false;
    *out_side = best_side > 0.0f ? 1 : -1;
    return true;
}

typedef struct {
    Vector2 a;
    Vector2 b;
} LocalCutSegment;

typedef struct {
    Vector2 p;
    int side;
    int edge;
    float t0;
    float t1;
    bool portal;
    int comp;
} LocalClassPoint;

static bool point_inside_triangle_2d(Vector2 p, const Vector2 tri2d[3]) {
    for (int edge = 0; edge < 3; ++edge) {
        Vector2 a = tri2d[(edge + 1) % 3];
        Vector2 b = tri2d[(edge + 2) % 3];
        if (cross2v(Vector2Subtract(b, a), Vector2Subtract(p, a)) < -1e-6f) return false;
    }
    return true;
}

static void local_class_point_push(LocalClassPoint** points, int* count, int* cap, LocalClassPoint value) {
    if (*count == *cap) {
        *cap = *cap ? *cap * 2 : 32;
        *points = (LocalClassPoint*)checked_realloc(*points, (size_t)*cap * sizeof(LocalClassPoint));
    }
    value.comp = -1;
    (*points)[(*count)++] = value;
}

static void local_cut_segment_push(LocalCutSegment** cuts, int* count, int* cap, LocalCutSegment value) {
    if (Vector2DistanceSqr(value.a, value.b) <= 1e-14f) return;
    if (*count == *cap) {
        *cap = *cap ? *cap * 2 : 16;
        *cuts = (LocalCutSegment*)checked_realloc(*cuts, (size_t)*cap * sizeof(LocalCutSegment));
    }
    (*cuts)[(*count)++] = value;
}

static bool local_connection_crosses_cut(Vector2 a, Vector2 b, const LocalCutSegment* cuts, int cut_count) {
    for (int i = 0; i < cut_count; ++i) {
        float t, u;
        if (!segment_intersect_2d(a, b, cuts[i].a, cuts[i].b, &t, &u)) continue;
        if (t > 1e-5f && t < 1.0f - 1e-5f && u > -1e-5f && u < 1.0f + 1e-5f) return true;
    }
    return false;
}

static bool add_local_side_seed(
    LocalClassPoint** points,
    int* point_count,
    int* point_cap,
    Vector2 mid,
    Vector2 normal,
    int side,
    float eps,
    const Vector2 tri2d[3]
) {
    for (int i = 0; i < 8; ++i) {
        Vector2 p = Vector2Add(mid, Vector2Scale(normal, eps));
        if (point_inside_triangle_2d(p, tri2d)) {
            local_class_point_push(points, point_count, point_cap, (LocalClassPoint){.p = p, .side = side, .edge = -1, .portal = false});
            return true;
        }
        eps *= 0.5f;
    }
    return false;
}

static int local_component_side(const LocalClassPoint* points, int point_count, int comp, int* conflicts, int tri) {
    int side = 0;
    for (int i = 0; i < point_count; ++i) {
        if (points[i].comp != comp || points[i].side == 0) continue;
        if (side != 0 && side != points[i].side) {
            fprintf(stderr, "local loop component conflict: tri=%d comp=%d\n", tri, comp);
            ++*conflicts;
            return 0;
        }
        side = points[i].side;
    }
    return side;
}

static void loop_classify_interval_portals_for_tri(
    const SurfaceMesh* surface,
    const LoopSegmentVec* loop_segments,
    int tri,
    const unsigned char* boundary_tri,
    unsigned char* flags,
    LoopPortalVec* portals,
    int* conflicts
) {
    Vector2 tri2d[3];
    if (!triangle_to_2d(surface, tri, tri2d)) return;

    FloatVec edge_params[3] = {0};
    for (int edge = 0; edge < 3; ++edge) {
        float_vec_push(&edge_params[edge], 0.0f);
        float_vec_push(&edge_params[edge], 1.0f);
    }

    LocalCutSegment* cuts = NULL;
    int cut_count = 0;
    int cut_cap = 0;
    LocalClassPoint* points = NULL;
    int point_count = 0;
    int point_cap = 0;

    float tri_scale = fmaxf(Vector2Distance(tri2d[0], tri2d[1]), fmaxf(Vector2Distance(tri2d[1], tri2d[2]), Vector2Distance(tri2d[2], tri2d[0])));
    float seed_eps = fmaxf(tri_scale * 1e-4f, 1e-7f);

    for (int i = 0; i < 3; ++i) {
        Vector2 corner = Vector2Add(
            Vector2Scale(tri2d[i], 1.0f - 2.0f * seed_eps / fmaxf(tri_scale, 1e-8f)),
            Vector2Scale(Vector2Add(tri2d[(i + 1) % 3], tri2d[(i + 2) % 3]), seed_eps / fmaxf(tri_scale, 1e-8f)));
        if (point_inside_triangle_2d(corner, tri2d)) {
            local_class_point_push(&points, &point_count, &point_cap, (LocalClassPoint){.p = corner, .side = 0, .edge = -1, .portal = false});
        }
    }

    for (int i = 0; i < loop_segments->size; ++i) {
        const LoopSegment* seg = &loop_segments->data[i];
        if (seg->tri != tri) continue;
        Vector2 a2 = surface_point_to_tri2d(surface, tri, seg->a, tri2d);
        Vector2 b2 = surface_point_to_tri2d(surface, tri, seg->b, tri2d);
        Vector2 dir = Vector2Subtract(b2, a2);
        float len = Vector2Length(dir);
        if (len <= 1e-8f) continue;
        local_cut_segment_push(&cuts, &cut_count, &cut_cap, (LocalCutSegment){.a = a2, .b = b2});

        for (int edge = 0; edge < 3; ++edge) {
            int ea = (edge + 1) % 3;
            int eb = (edge + 2) % 3;
            float t, u;
            if (!segment_intersect_2d(a2, b2, tri2d[ea], tri2d[eb], &t, &u)) continue;
            if (u > 1e-5f && u < 1.0f - 1e-5f) float_vec_push(&edge_params[edge], u);
        }

        Vector2 mid = Vector2Scale(Vector2Add(a2, b2), 0.5f);
        Vector2 n = Vector2Scale((Vector2){-dir.y, dir.x}, 1.0f / len);
        add_local_side_seed(&points, &point_count, &point_cap, mid, n, 1, seed_eps, tri2d);
        add_local_side_seed(&points, &point_count, &point_cap, mid, Vector2Scale(n, -1.0f), 2, seed_eps, tri2d);
    }

    if (cut_count == 0) goto done;

    int same_cross_edge = -1;
    int crossed_edge_count = 0;
    for (int edge = 0; edge < 3; ++edge) {
        float_vec_sort_unique(&edge_params[edge]);
        if (edge_params[edge].size > 2) {
            same_cross_edge = edge;
            ++crossed_edge_count;
        }
    }

    int same_edge_opposite_side = 0;
    if (crossed_edge_count == 1 && edge_params[same_cross_edge].size >= 4) {
        int nearest = 0;
        if (loop_nearest_segment_side_in_tri(surface, loop_segments, tri, tri2d[same_cross_edge], tri2d, &nearest)) {
            same_edge_opposite_side = nearest > 0 ? 1 : 2;
        }
    }

    for (int edge = 0; edge < 3; ++edge) {
        int nb = surface->tri_neighbors[3 * tri + edge];
        if (nb < 0 || boundary_tri[nb]) continue;
        int a = (edge + 1) % 3;
        int b = (edge + 2) % 3;
        for (int i = 0; i < edge_params[edge].size - 1; ++i) {
            float t0 = edge_params[edge].data[i];
            float t1 = edge_params[edge].data[i + 1];
            if (t1 <= t0 + 1e-5f) continue;
            float tm = 0.5f * (t0 + t1);
            Vector2 p = lerp2(tri2d[a], tri2d[b], tm);
            local_class_point_push(&points, &point_count, &point_cap, (LocalClassPoint){.p = p, .side = 0, .edge = edge, .t0 = t0, .t1 = t1, .portal = true});
        }
    }

    int* queue = (int*)malloc((size_t)fmaxf((float)point_count, 1.0f) * sizeof(int));
    if (!queue) {
        fprintf(stderr, "out of memory while classifying local loop portals\n");
        exit(1);
    }
    int comp_count = 0;
    for (int start = 0; start < point_count; ++start) {
        if (points[start].comp >= 0) continue;
        int head = 0;
        int tail = 0;
        points[start].comp = comp_count;
        queue[tail++] = start;
        while (head < tail) {
            int pidx = queue[head++];
            for (int j = 0; j < point_count; ++j) {
                if (points[j].comp >= 0) continue;
                if (local_connection_crosses_cut(points[pidx].p, points[j].p, cuts, cut_count)) continue;
                points[j].comp = comp_count;
                queue[tail++] = j;
            }
        }
        ++comp_count;
    }
    free(queue);

    int* comp_side = (int*)calloc((size_t)fmaxf((float)comp_count, 1.0f), sizeof(int));
    if (!comp_side) {
        fprintf(stderr, "out of memory while labeling local loop components\n");
        exit(1);
    }
    for (int comp = 0; comp < comp_count; ++comp) comp_side[comp] = local_component_side(points, point_count, comp, conflicts, tri);

    for (int i = 0; i < point_count; ++i) {
        if (!points[i].portal) continue;
        int side = points[i].comp >= 0 ? comp_side[points[i].comp] : 0;
        if (same_edge_opposite_side != 0 && points[i].edge != same_cross_edge) side = same_edge_opposite_side;
        if (side == 0) {
            int nearest = 0;
            if (loop_nearest_segment_side_in_tri(surface, loop_segments, tri, points[i].p, tri2d, &nearest)) side = nearest > 0 ? 1 : 2;
        }
        if (side == 0) {
            fprintf(stderr, "loop portal interval unlabeled: tri=%d edge=%d t=[%.6f, %.6f]\n", tri, points[i].edge, points[i].t0, points[i].t1);
            ++*conflicts;
            continue;
        }
        flags[3 * tri + points[i].edge] |= side == 1 ? LOOP_EDGE_LEFT : LOOP_EDGE_RIGHT;
        loop_portal_vec_push(portals, (LoopPortal){.tri = tri, .edge = points[i].edge, .t0 = points[i].t0, .t1 = points[i].t1, .side = side});
    }
    free(comp_side);

done:
    for (int edge = 0; edge < 3; ++edge) float_vec_free(&edge_params[edge]);
    free(cuts);
    free(points);
}

static void loop_classify_interval_portals(
    const SurfaceMesh* surface,
    const LoopSegmentVec* loop_segments,
    const unsigned char* boundary_tri,
    unsigned char* flags,
    LoopPortalVec* portals,
    int* conflicts
) {
    for (int tri = 0; tri < surface->tri_count; ++tri) {
        if (boundary_tri[tri]) loop_classify_interval_portals_for_tri(surface, loop_segments, tri, boundary_tri, flags, portals, conflicts);
    }
}

static float surface_triangle_area(const SurfaceMesh* surface, int tri) {
    return triangle_area(
        surface->vertices[surface->tris[3 * tri + 0]],
        surface->vertices[surface->tris[3 * tri + 1]],
        surface->vertices[surface->tris[3 * tri + 2]]);
}

static const char* loop_debug_mode_name(int mode) {
    switch (mode) {
        case 0: return "off";
        case 1: return "boundary";
        case 2: return "portals";
        case 3: return "flood";
        case 4: return "conflicts";
        case 5: return "all";
        default: return "unknown";
    }
}

static Vector3 triangle_normal_offset(const SurfaceMesh* surface, int tri, float amount) {
    Vector3 a = surface->vertices[surface->tris[3 * tri + 0]];
    Vector3 b = surface->vertices[surface->tris[3 * tri + 1]];
    Vector3 c = surface->vertices[surface->tris[3 * tri + 2]];
    Vector3 n = Vector3CrossProduct(Vector3Subtract(b, a), Vector3Subtract(c, a));
    if (Vector3LengthSqr(n) <= 1e-20f) return (Vector3){0.0f, 0.0f, 0.0f};
    return Vector3Scale(Vector3Normalize(n), amount);
}

static void draw_loop_debug_triangle(const SurfaceMesh* surface, int tri, Color color, float offset) {
    Vector3 o = triangle_normal_offset(surface, tri, offset);
    Vector3 a = Vector3Add(surface->vertices[surface->tris[3 * tri + 0]], o);
    Vector3 b = Vector3Add(surface->vertices[surface->tris[3 * tri + 1]], o);
    Vector3 c = Vector3Add(surface->vertices[surface->tris[3 * tri + 2]], o);
    DrawTriangle3D(a, b, c, color);
}

static void draw_loop_debug_edge(const SurfaceMesh* surface, int tri, int edge, Color color, float offset) {
    Vector3 o = triangle_normal_offset(surface, tri, offset);
    int a = surface->tris[3 * tri + (edge + 1) % 3];
    int b = surface->tris[3 * tri + (edge + 2) % 3];
    DrawLine3D(Vector3Add(surface->vertices[a], o), Vector3Add(surface->vertices[b], o), color);
}

static void draw_loop_debug_edge_interval(const SurfaceMesh* surface, const LoopPortal* portal, Color color, float offset) {
    Vector3 o = triangle_normal_offset(surface, portal->tri, offset);
    int a = surface->tris[3 * portal->tri + (portal->edge + 1) % 3];
    int b = surface->tris[3 * portal->tri + (portal->edge + 2) % 3];
    Vector3 p0 = Vector3Add(Vector3Scale(surface->vertices[a], 1.0f - portal->t0), Vector3Scale(surface->vertices[b], portal->t0));
    Vector3 p1 = Vector3Add(Vector3Scale(surface->vertices[a], 1.0f - portal->t1), Vector3Scale(surface->vertices[b], portal->t1));
    DrawLine3D(Vector3Add(p0, o), Vector3Add(p1, o), color);
}

static void draw_loop_debug(const SurfaceMesh* surface, const LoopDebugData* debug, int mode, int chosen_side, float marker_scale) {
    if (mode <= 0 || !debug->flags || debug->tri_count != surface->tri_count) return;
    float off = fmaxf(marker_scale * 0.00025f, 1e-5f);

    bool show_boundary = mode == 1 || mode == 5;
    bool show_portals = mode == 2 || mode == 5;
    bool show_flood = mode == 3 || mode == 5;
    bool show_conflicts = mode == 4 || mode == 5;

    if (show_flood) {
        for (int tri = 0; tri < surface->tri_count; ++tri) {
            if (debug->side_mark[tri] == 1) draw_loop_debug_triangle(surface, tri, chosen_side == 1 ? (Color){50, 230, 120, 95} : (Color){40, 140, 255, 55}, off);
            else if (debug->side_mark[tri] == 2) draw_loop_debug_triangle(surface, tri, chosen_side == 2 ? (Color){50, 230, 120, 95} : (Color){255, 170, 40, 55}, off);
        }
    }

    if (show_boundary) {
        for (int tri = 0; tri < surface->tri_count; ++tri) {
            if (debug->boundary_tri[tri]) draw_loop_debug_triangle(surface, tri, (Color){180, 180, 190, 80}, off * 1.5f);
        }
    }

    if (show_conflicts) {
        for (int tri = 0; tri < surface->tri_count; ++tri) {
            if (debug->conflict_tri[tri]) draw_loop_debug_triangle(surface, tri, (Color){255, 40, 40, 180}, off * 2.0f);
        }
    }

    if (show_portals) {
        for (int tri = 0; tri < surface->tri_count; ++tri) {
            for (int edge = 0; edge < 3; ++edge) {
                unsigned char f = debug->flags[3 * tri + edge];
                if (f & LOOP_EDGE_BLOCKED) draw_loop_debug_edge(surface, tri, edge, (Color){255, 40, 40, 255}, off * 3.0f);
            }
        }
        for (int i = 0; i < debug->portal_count; ++i) {
            Color color = debug->portals[i].side == 1 ? (Color){40, 140, 255, 255} : (Color){255, 190, 40, 255};
            draw_loop_debug_edge_interval(surface, &debug->portals[i], color, off * 3.5f);
        }
    }
}

static void loop_enqueue_tri(
    int tri,
    int side,
    int* queue,
    int* tail,
    unsigned char* side_mark,
    const unsigned char* boundary_tri,
    unsigned char* conflict_tri,
    int* flood_conflicts
) {
    if (tri < 0 || boundary_tri[tri]) return;
    unsigned char mark = side == 1 ? 1 : 2;
    unsigned char other = side == 1 ? 2 : 1;
    if (side_mark[tri] == mark) return;
    if (side_mark[tri] == other) {
        fprintf(stderr, "loop flood conflict: tri=%d reached by both sides\n", tri);
        if (conflict_tri) conflict_tri[tri] = 1;
        ++*flood_conflicts;
        return;
    }
    side_mark[tri] = mark;
    queue[(*tail)++] = tri;
}

static bool loop_flood_pop(
    const SurfaceMesh* surface,
    const unsigned char* flags,
    const unsigned char* boundary_tri,
    int side,
    int* queue,
    int* head,
    int* tail,
    unsigned char* side_mark,
    unsigned char* conflict_tri,
    float* area,
    int* count,
    int* flood_conflicts
) {
    if (*head >= *tail) return false;
    int tri = queue[(*head)++];
    *area += surface_triangle_area(surface, tri);
    ++*count;
    for (int edge = 0; edge < 3; ++edge) {
        if (flags[3 * tri + edge] & LOOP_EDGE_BLOCKED) continue;
        int nb = surface->tri_neighbors[3 * tri + edge];
        if (nb < 0 || boundary_tri[nb]) continue;

        int va = surface->tris[3 * tri + (edge + 1) % 3];
        int vb = surface->tris[3 * tri + (edge + 2) % 3];
        int nb_edge = local_edge_between(surface, nb, va, vb);
        if (nb_edge >= 0 && (flags[3 * nb + nb_edge] & LOOP_EDGE_BLOCKED)) continue;

        loop_enqueue_tri(nb, side, queue, tail, side_mark, boundary_tri, conflict_tri, flood_conflicts);
    }
    return true;
}

static int classify_boundary_sample_side_local(
    const SurfaceMesh* surface,
    const LoopSegmentVec* loop_segments,
    int tri,
    Vector3 sample,
    int* conflicts
) {
    Vector2 tri2d[3];
    if (!triangle_to_2d(surface, tri, tri2d)) return 0;

    LocalCutSegment* cuts = NULL;
    int cut_count = 0;
    int cut_cap = 0;
    LocalClassPoint* points = NULL;
    int point_count = 0;
    int point_cap = 0;

    float tri_scale = fmaxf(Vector2Distance(tri2d[0], tri2d[1]), fmaxf(Vector2Distance(tri2d[1], tri2d[2]), Vector2Distance(tri2d[2], tri2d[0])));
    float seed_eps = fmaxf(tri_scale * 1e-4f, 1e-7f);
    float inset = seed_eps / fmaxf(tri_scale, 1e-8f);
    for (int i = 0; i < 3; ++i) {
        Vector2 corner = Vector2Add(Vector2Scale(tri2d[i], 1.0f - 2.0f * inset), Vector2Scale(Vector2Add(tri2d[(i + 1) % 3], tri2d[(i + 2) % 3]), inset));
        if (point_inside_triangle_2d(corner, tri2d)) local_class_point_push(&points, &point_count, &point_cap, (LocalClassPoint){.p = corner, .side = 0, .edge = -1, .portal = false});
    }

    for (int i = 0; i < loop_segments->size; ++i) {
        const LoopSegment* seg = &loop_segments->data[i];
        if (seg->tri != tri) continue;
        Vector2 a2 = surface_point_to_tri2d(surface, tri, seg->a, tri2d);
        Vector2 b2 = surface_point_to_tri2d(surface, tri, seg->b, tri2d);
        Vector2 dir = Vector2Subtract(b2, a2);
        float len = Vector2Length(dir);
        if (len <= 1e-8f) continue;
        local_cut_segment_push(&cuts, &cut_count, &cut_cap, (LocalCutSegment){.a = a2, .b = b2});
        Vector2 mid = Vector2Scale(Vector2Add(a2, b2), 0.5f);
        Vector2 n = Vector2Scale((Vector2){-dir.y, dir.x}, 1.0f / len);
        add_local_side_seed(&points, &point_count, &point_cap, mid, n, 1, seed_eps, tri2d);
        add_local_side_seed(&points, &point_count, &point_cap, mid, Vector2Scale(n, -1.0f), 2, seed_eps, tri2d);
    }

    int sample_idx = point_count;
    local_class_point_push(&points, &point_count, &point_cap, (LocalClassPoint){.p = surface_point_to_tri2d(surface, tri, sample, tri2d), .side = 0, .edge = -1, .portal = false});

    int* queue = (int*)malloc((size_t)fmaxf((float)point_count, 1.0f) * sizeof(int));
    if (!queue) {
        fprintf(stderr, "out of memory while classifying boundary sample\n");
        exit(1);
    }

    int comp_count = 0;
    for (int start = 0; start < point_count; ++start) {
        if (points[start].comp >= 0) continue;
        int head = 0;
        int tail = 0;
        points[start].comp = comp_count;
        queue[tail++] = start;
        while (head < tail) {
            int pidx = queue[head++];
            for (int j = 0; j < point_count; ++j) {
                if (points[j].comp >= 0) continue;
                if (local_connection_crosses_cut(points[pidx].p, points[j].p, cuts, cut_count)) continue;
                points[j].comp = comp_count;
                queue[tail++] = j;
            }
        }
        ++comp_count;
    }

    int side = points[sample_idx].comp >= 0 ? local_component_side(points, point_count, points[sample_idx].comp, conflicts, tri) : 0;
    free(queue);
    free(cuts);
    free(points);
    return side;
}

static LoopRemovalStats remove_samples_inside_loop(
    const SurfaceMesh* surface,
    const PropGraph* graph,
    const SurfaceTopo* topo,
    SurfaceSample* samples,
    const LoopSegmentVec* loop_segments,
    LoopDebugData* debug
) {
    LoopRemovalStats stats = {0};
    if (loop_segments->size <= 0) return stats;

    unsigned char* flags = (unsigned char*)calloc((size_t)surface->tri_count * 3, sizeof(unsigned char));
    unsigned char* boundary_tri = (unsigned char*)calloc((size_t)surface->tri_count, sizeof(unsigned char));
    unsigned char* side_mark = (unsigned char*)calloc((size_t)surface->tri_count, sizeof(unsigned char));
    unsigned char* conflict_tri = (unsigned char*)calloc((size_t)surface->tri_count, sizeof(unsigned char));
    int* queue_left = (int*)malloc((size_t)surface->tri_count * sizeof(int));
    int* queue_right = (int*)malloc((size_t)surface->tri_count * sizeof(int));
    LoopPortalVec portals = {0};
    if (!flags || !boundary_tri || !side_mark || !conflict_tri || !queue_left || !queue_right) {
        fprintf(stderr, "out of memory while removing loop interior\n");
        exit(1);
    }

    for (int i = 0; i < loop_segments->size; ++i) {
        loop_mark_segment_blocked_edges(surface, graph, &loop_segments->data[i], flags, boundary_tri, &stats.classification_conflicts);
    }
    loop_classify_interval_portals(surface, loop_segments, boundary_tri, flags, &portals, &stats.classification_conflicts);

    int left_head = 0, left_tail = 0;
    int right_head = 0, right_tail = 0;
    for (int i = 0; i < portals.size; ++i) {
        LoopPortal p = portals.data[i];
        int nb = surface->tri_neighbors[3 * p.tri + p.edge];
        if (nb < 0 || boundary_tri[nb]) continue;
        if (p.side == 1) loop_enqueue_tri(nb, 1, queue_left, &left_tail, side_mark, boundary_tri, conflict_tri, &stats.flood_conflicts);
        else if (p.side == 2) loop_enqueue_tri(nb, 2, queue_right, &right_tail, side_mark, boundary_tri, conflict_tri, &stats.flood_conflicts);
    }

    bool prefer_left = true;
    while (left_head < left_tail || right_head < right_tail) {
        bool popped = false;
        if (prefer_left) {
            popped = loop_flood_pop(surface, flags, boundary_tri, 1, queue_left, &left_head, &left_tail, side_mark, conflict_tri, &stats.left_area, &stats.left_tris, &stats.flood_conflicts);
            if (!popped) loop_flood_pop(surface, flags, boundary_tri, 2, queue_right, &right_head, &right_tail, side_mark, conflict_tri, &stats.right_area, &stats.right_tris, &stats.flood_conflicts);
        } else {
            popped = loop_flood_pop(surface, flags, boundary_tri, 2, queue_right, &right_head, &right_tail, side_mark, conflict_tri, &stats.right_area, &stats.right_tris, &stats.flood_conflicts);
            if (!popped) loop_flood_pop(surface, flags, boundary_tri, 1, queue_left, &left_head, &left_tail, side_mark, conflict_tri, &stats.left_area, &stats.left_tris, &stats.flood_conflicts);
        }
        prefer_left = !prefer_left;

        bool left_done = left_head >= left_tail;
        bool right_done = right_head >= right_tail;
        if (left_done && stats.left_tris > 0 && stats.right_area >= stats.left_area) break;
        if (right_done && stats.right_tris > 0 && stats.left_area >= stats.right_area) break;
    }

    if (stats.left_tris <= 0 && stats.right_tris <= 0) {
        fprintf(stderr, "loop removal failed: no flood seed portals\n");
    } else if (stats.right_tris <= 0 || (stats.left_tris > 0 && stats.left_area <= stats.right_area)) {
        stats.chosen_side = 1;
    } else {
        stats.chosen_side = 2;
    }

    if (stats.chosen_side != 0) {
        for (int tri = 0; tri < surface->tri_count; ++tri) {
            int begin = topo->tri_sample_offsets[tri];
            int end = topo->tri_sample_offsets[tri + 1];
            if (!boundary_tri[tri]) {
                if (side_mark[tri] != stats.chosen_side) continue;
                for (int si = begin; si < end; ++si) {
                    int sample_id = topo->tri_sample_ids[si];
                    if (!samples[sample_id].disabled) {
                        samples[sample_id].disabled = true;
                        ++stats.disabled_samples;
                    }
                }
            } else {
                for (int si = begin; si < end; ++si) {
                    int sample_id = topo->tri_sample_ids[si];
                    if (samples[sample_id].disabled) continue;
                    int side = classify_boundary_sample_side_local(surface, loop_segments, tri, samples[sample_id].p, &stats.classification_conflicts);
                    if (side == stats.chosen_side) {
                        samples[sample_id].disabled = true;
                        ++stats.disabled_samples;
                    }
                }
            }
        }
    }

    printf("loop removal: segments=%d disabled=%d chosen=%s left_tris=%d right_tris=%d left_area=%.8f right_area=%.8f classify_conflicts=%d flood_conflicts=%d\n",
        loop_segments->size, stats.disabled_samples,
        stats.chosen_side == 1 ? "left" : (stats.chosen_side == 2 ? "right" : "none"),
        stats.left_tris, stats.right_tris, stats.left_area, stats.right_area,
        stats.classification_conflicts, stats.flood_conflicts);
    fflush(stdout);

    if (debug) loop_debug_capture(debug, flags, boundary_tri, side_mark, conflict_tri, &portals, surface->tri_count);

    free(flags);
    free(boundary_tri);
    free(side_mark);
    free(conflict_tri);
    free(queue_left);
    free(queue_right);
    loop_portal_vec_free(&portals);
    return stats;
}

static void print_query_stats(int source, float radius, int in_radius, double query_ms) {
    printf("query source=%d radius=%.8f in_radius=%d fmm_ms=%.3f\n",
        source, radius, in_radius, query_ms);
    fflush(stdout);
}

static void print_path_stats(int source, int target, bool ok, int path_points, double path_ms, const GeodesicPathContext* path_ctx) {
    printf("path source=%d target=%d ok=%d points=%d path_ms=%.3f windows=%d popped=%d pseudo=%d touched_states=%d max_state_windows=%d\n",
        source, target, ok ? 1 : 0, path_points, path_ms,
        path_ctx->windows_pushed, path_ctx->windows_popped, path_ctx->pseudo_sources,
        path_ctx->touched_state_count, path_ctx->max_state_windows);
    fflush(stdout);
}

static void update_orbit_camera(Camera3D* camera, float min_dist, float max_dist) {
    Vector3 offset = Vector3Subtract(camera->position, camera->target);
    float dist = Vector3Length(offset);
    if (dist < min_dist) dist = min_dist;

    Vector2 mouse_delta = GetMouseDelta();

    bool picking_modifier = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL) ||
        IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !picking_modifier) {
        const float rotate_speed = 0.005f;
        float yaw = atan2f(offset.x, offset.z);
        float pitch = asinf(Clamp(offset.y / dist, -1.0f, 1.0f));

        yaw -= mouse_delta.x * rotate_speed;
        pitch += mouse_delta.y * rotate_speed;
        pitch = Clamp(pitch, -1.52f, 1.52f);

        float cp = cosf(pitch);
        offset = (Vector3){
            dist * cp * sinf(yaw),
            dist * sinf(pitch),
            dist * cp * cosf(yaw),
        };
        camera->position = Vector3Add(camera->target, offset);
    }

    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
        Vector3 forward = Vector3Normalize(Vector3Subtract(camera->target, camera->position));
        Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, camera->up));
        Vector3 up = Vector3Normalize(Vector3CrossProduct(right, forward));
        float view_height = 2.0f * dist * tanf(DEG2RAD * camera->fovy * 0.5f);
        float pan_scale = view_height / fmaxf((float)GetScreenHeight(), 1.0f);
        Vector3 pan = Vector3Add(
            Vector3Scale(right, -mouse_delta.x * pan_scale),
            Vector3Scale(up, mouse_delta.y * pan_scale)
        );
        camera->position = Vector3Add(camera->position, pan);
        camera->target = Vector3Add(camera->target, pan);
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) {
        offset = Vector3Subtract(camera->position, camera->target);
        dist = Vector3Length(offset);
        float new_dist = Clamp(dist * expf(-wheel * 0.12f), min_dist, max_dist);
        if (dist > 1e-8f) {
            camera->position = Vector3Add(camera->target, Vector3Scale(offset, new_dist / dist));
        }
    }

    camera->up = (Vector3){0.0f, 1.0f, 0.0f};
}

static void draw_point_crosses(
    const SurfaceSample* samples,
    int sample_count,
    Camera3D camera,
    float size,
    bool only_in_radius,
    unsigned int query_id,
    Color color
) {
    Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, camera.up));
    Vector3 up = Vector3Normalize(Vector3CrossProduct(right, forward));
    right = Vector3Scale(right, size);
    up = Vector3Scale(up, size);

    rlBegin(RL_LINES);
    rlColor4ub(color.r, color.g, color.b, color.a);
    for (int i = 0; i < sample_count; ++i) {
        if (samples[i].disabled) continue;
        bool current_in_radius = samples[i].stamp == query_id && samples[i].in_radius;
        if (only_in_radius != current_in_radius) continue;
        Vector3 p = samples[i].p;
        Vector3 a = Vector3Subtract(p, right);
        Vector3 b = Vector3Add(p, right);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
        a = Vector3Subtract(p, up);
        b = Vector3Add(p, up);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
    }
    rlEnd();
}

static Color distance_color(float distance, float radius) {
    if (!isfinite(distance)) return (Color){50, 55, 65, 90};
    float t = radius > 1e-8f ? Clamp(distance / radius, 0.0f, 1.0f) : 0.0f;

    Color near = (Color){40, 220, 255, 230};
    Color mid = (Color){80, 240, 100, 230};
    Color far = (Color){255, 210, 60, 230};
    Color out = (Color){230, 70, 70, 180};

    Color a;
    Color b;
    float u;
    if (t < 0.5f) {
        a = near;
        b = mid;
        u = t * 2.0f;
    } else if (t < 1.0f) {
        a = mid;
        b = far;
        u = (t - 0.5f) * 2.0f;
    } else {
        a = far;
        b = out;
        u = 1.0f;
    }

    return (Color){
        (unsigned char)(a.r + (b.r - a.r) * u),
        (unsigned char)(a.g + (b.g - a.g) * u),
        (unsigned char)(a.b + (b.b - a.b) * u),
        (unsigned char)(a.a + (b.a - a.a) * u),
    };
}

static void draw_distance_point_crosses(
    const SurfaceSample* samples,
    int sample_count,
    Camera3D camera,
    float size,
    float radius,
    unsigned int query_id
) {
    Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, camera.up));
    Vector3 up = Vector3Normalize(Vector3CrossProduct(right, forward));
    right = Vector3Scale(right, size);
    up = Vector3Scale(up, size);

    rlBegin(RL_LINES);
    for (int i = 0; i < sample_count; ++i) {
        if (samples[i].disabled) continue;
        Color color = distance_color(samples[i].stamp == query_id ? samples[i].distance : INFINITY, radius);
        rlColor4ub(color.r, color.g, color.b, color.a);
        Vector3 p = samples[i].p;
        Vector3 a = Vector3Subtract(p, right);
        Vector3 b = Vector3Add(p, right);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
        a = Vector3Subtract(p, up);
        b = Vector3Add(p, up);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
    }
    rlEnd();
}

static void draw_vertex_crosses(
    const Vector3* vertices,
    int vertex_count,
    Camera3D camera,
    float size,
    Color color
) {
    Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, camera.up));
    Vector3 up = Vector3Normalize(Vector3CrossProduct(right, forward));
    right = Vector3Scale(right, size);
    up = Vector3Scale(up, size);

    rlBegin(RL_LINES);
    rlColor4ub(color.r, color.g, color.b, color.a);
    for (int i = 0; i < vertex_count; ++i) {
        Vector3 p = vertices[i];
        Vector3 a = Vector3Subtract(p, right);
        Vector3 b = Vector3Add(p, right);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
        a = Vector3Subtract(p, up);
        b = Vector3Add(p, up);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
    }
    rlEnd();
}

static void usage(const char* exe) {
    fprintf(stderr,
        "Usage: %s MESH_PATH [sample_count] [radius] [seed] [steiner_spacing]\n"
        "  sample_count   default: 5000\n"
        "  radius         default: 5%% of mesh bounding-box diagonal\n"
        "  seed           default: 1\n"
        "  steiner_spacing default: radius / 10\n",
        exe
    );
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char* mesh_path = argv[1];
    int sample_count = argc > 2 ? atoi(argv[2]) : 5000;
    unsigned int rng = argc > 4 ? (unsigned int)strtoul(argv[4], NULL, 10) : 1u;
    if (sample_count <= 0) {
        usage(argv[0]);
        return 1;
    }

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(1280, 900, "Quad Meshing 3D Geodesic Candidate PoC");
    SetTargetFPS(144);

    Model model = LoadModel(mesh_path);
    if (model.meshCount == 0) {
        fprintf(stderr, "failed to load model: %s\n", mesh_path);
        CloseWindow();
        return 1;
    }
    model.transform = normalize_model_transform(GetModelBoundingBox(model));

    SurfaceMesh surface = surface_from_model(&model);
    float diag = surface_diag(surface.bounds);
    float radius = argc > 3 ? (float)atof(argv[3]) : 0.05f * diag;
    if (radius <= 0.0f) radius = 0.05f * diag;
    float steiner_spacing = argc > 5 ? (float)atof(argv[5]) : radius / 10.0f;
    if (steiner_spacing <= 0.0f) steiner_spacing = radius / 10.0f;

    double preprocess_t0 = now_seconds();
    SurfaceSample* samples = sample_surface_points(&surface, sample_count, &rng);
    SurfaceTopo topo = build_surface_topo(&surface, samples, sample_count);
    steiner_spacing = fmaxf(steiner_spacing, 1e-4f);
    PropGraph prop_graph = build_prop_graph(&surface, steiner_spacing);
    FMMContext fmm = fmm_context_create(prop_graph.node_count, surface.tri_count);
    GeodesicPathContext path_ctx = geodesic_path_context_create(&surface, &prop_graph);
    double preprocess_t1 = now_seconds();

    int source = rand_r(&rng) % sample_count;
    int target = -1;
    Vec3Vec geodesic_path = {0};
    Vec3Vec loop_draw_path = {0};
    LoopSegmentVec path_segments = {0};
    LoopSegmentVec loop_segments = {0};
    LoopRemovalStats loop_stats = {0};
    LoopDebugData loop_debug = {0};
    int loop_first_source = -1;
    int loop_saved_paths = 0;
    int disabled_samples = 0;
    int loop_debug_mode = 0;
    double query_ms = 0.0;
    double path_ms = -1.0;
    bool path_ok = false;
    int in_radius = compute_source_and_measure_steiner(&prop_graph, &topo, samples, sample_count, source, radius, &fmm, &query_ms);
    printf("preprocess_ms=%.3f\n", 1000.0 * (preprocess_t1 - preprocess_t0));
    fflush(stdout);
    print_query_stats(source, radius, in_radius, query_ms);

    Vector3 center = Vector3Scale(Vector3Add(surface.bounds.min, surface.bounds.max), 0.5f);
    float camera_dist = fmaxf(diag * 1.8f, 1.0f);
    Camera3D camera = {0};
    camera.position = Vector3Add(center, (Vector3){camera_dist, camera_dist * 0.6f, camera_dist});
    camera.target = center;
    camera.up = (Vector3){0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    bool show_mesh = true;
    bool mesh_opaque = false;
    bool show_all_points = true;
    bool show_distance_colors = false;
    bool show_graph_vertices = false;
    float marker_scale = diag > 1e-8f ? diag : 1.0f;
    float point_radius = marker_scale * 0.0025f;
    float hit_point_radius = marker_scale * 0.0040f;
    float source_radius = marker_scale * 0.0006f;

    while (!WindowShouldClose()) {
        update_orbit_camera(&camera, fmaxf(diag * 0.02f, 0.01f), fmaxf(diag * 20.0f, 10.0f));

        if (IsKeyPressed(KEY_R) || IsKeyPressed(KEY_SPACE)) {
            do {
                source = rand_r(&rng) % sample_count;
            } while (samples[source].disabled && disabled_samples < sample_count);
            target = -1;
            loop_first_source = -1;
            loop_saved_paths = 0;
            path_ms = -1.0;
            path_ok = false;
            vec3_vec_clear(&geodesic_path);
            vec3_vec_clear(&loop_draw_path);
            loop_segment_vec_clear(&loop_segments);
            in_radius = compute_source_and_measure_steiner(&prop_graph, &topo, samples, sample_count, source, radius, &fmm, &query_ms);
            print_query_stats(source, radius, in_radius, query_ms);
        }
        if (IsKeyPressed(KEY_M)) show_mesh = !show_mesh;
        if (IsKeyPressed(KEY_O)) mesh_opaque = !mesh_opaque;
        if (IsKeyPressed(KEY_P)) show_all_points = !show_all_points;
        if (IsKeyPressed(KEY_D)) show_distance_colors = !show_distance_colors;
        if (IsKeyPressed(KEY_V)) show_graph_vertices = !show_graph_vertices;
        if (IsKeyPressed(KEY_N)) loop_debug_mode = (loop_debug_mode + 1) % 6;
        if (IsKeyPressed(KEY_B)) loop_debug_mode = loop_debug_mode == 0 ? 5 : loop_debug_mode - 1;
        if (IsKeyPressed(KEY_UP)) {
            radius *= 1.1f;
            target = -1;
            path_ms = -1.0;
            path_ok = false;
            vec3_vec_clear(&geodesic_path);
            in_radius = compute_source_and_measure_steiner(&prop_graph, &topo, samples, sample_count, source, radius, &fmm, &query_ms);
            print_query_stats(source, radius, in_radius, query_ms);
        }
        if (IsKeyPressed(KEY_DOWN)) {
            radius /= 1.1f;
            target = -1;
            path_ms = -1.0;
            path_ok = false;
            vec3_vec_clear(&geodesic_path);
            in_radius = compute_source_and_measure_steiner(&prop_graph, &topo, samples, sample_count, source, radius, &fmm, &query_ms);
            print_query_stats(source, radius, in_radius, query_ms);
        }

        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            bool ctrl = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL);
            bool shift = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
            if (ctrl) {
                int picked = pick_sample_screen(samples, sample_count, camera, GetMousePosition(), false, fmm.query_id);
                if (picked >= 0) {
                    source = picked;
                    target = -1;
                    loop_first_source = picked;
                    loop_saved_paths = 0;
                    path_ms = -1.0;
                    path_ok = false;
                    vec3_vec_clear(&geodesic_path);
                    vec3_vec_clear(&loop_draw_path);
                    loop_segment_vec_clear(&loop_segments);
                    in_radius = compute_source_and_measure_steiner(&prop_graph, &topo, samples, sample_count, source, radius, &fmm, &query_ms);
                    print_query_stats(source, radius, in_radius, query_ms);
                }
            } else if (shift) {
                int picked = pick_sample_screen(samples, sample_count, camera, GetMousePosition(), true, fmm.query_id);
                if (picked >= 0) {
                    target = picked;
                    double path_t0 = now_seconds();
                    path_ok = build_geodesic_path(&surface, &prop_graph, samples, source, target, &fmm, &path_ctx, &geodesic_path, &path_segments);
                    double path_t1 = now_seconds();
                    path_ms = 1000.0 * (path_t1 - path_t0);
                    print_path_stats(source, target, path_ok, geodesic_path.size, path_ms, &path_ctx);
                    if (path_ok) {
                        if (loop_first_source < 0) loop_first_source = source;
                        for (int i = 0; i < geodesic_path.size; ++i) vec3_vec_push_unique(&loop_draw_path, geodesic_path.data[i]);
                        for (int i = 0; i < path_segments.size; ++i) loop_segment_vec_push(&loop_segments, path_segments.data[i]);
                        ++loop_saved_paths;
                        source = target;

                        if (loop_saved_paths >= 3 && loop_first_source >= 0 && !samples[loop_first_source].disabled) {
                            float old_radius = radius;
                            int close_target = loop_first_source;
                            float close_radius = fmaxf(diag, old_radius);
                            in_radius = compute_source_and_measure_steiner(&prop_graph, &topo, samples, sample_count, source, close_radius, &fmm, &query_ms);
                            while (!(samples[close_target].stamp == fmm.query_id && samples[close_target].in_radius) && close_radius < diag * 16.0f) {
                                close_radius *= 2.0f;
                                in_radius = compute_source_and_measure_steiner(&prop_graph, &topo, samples, sample_count, source, close_radius, &fmm, &query_ms);
                            }
                            double close_t0 = now_seconds();
                            bool close_ok = build_geodesic_path(&surface, &prop_graph, samples, source, close_target, &fmm, &path_ctx, &geodesic_path, &path_segments);
                            double close_t1 = now_seconds();
                            path_ms = 1000.0 * (close_t1 - close_t0);
                            path_ok = close_ok;
                            print_path_stats(source, close_target, close_ok, geodesic_path.size, path_ms, &path_ctx);
                            if (close_ok) {
                                for (int i = 0; i < geodesic_path.size; ++i) vec3_vec_push_unique(&loop_draw_path, geodesic_path.data[i]);
                                for (int i = 0; i < path_segments.size; ++i) loop_segment_vec_push(&loop_segments, path_segments.data[i]);
                                loop_stats = remove_samples_inside_loop(&surface, &prop_graph, &topo, samples, &loop_segments, &loop_debug);
                                loop_debug_mode = loop_stats.flood_conflicts > 0 || loop_stats.classification_conflicts > 0 ? 4 : 3;
                                disabled_samples += loop_stats.disabled_samples;
                                if (samples[source].disabled) {
                                    for (int i = 0; i < sample_count; ++i) {
                                        if (!samples[i].disabled) {
                                            source = i;
                                            break;
                                        }
                                    }
                                }
                                target = -1;
                                loop_first_source = -1;
                                loop_saved_paths = 0;
                                loop_segment_vec_clear(&loop_segments);
                            } else {
                                fprintf(stderr, "loop closure path failed; keeping open loop for debugging\n");
                            }
                            in_radius = compute_source_and_measure_steiner(&prop_graph, &topo, samples, sample_count, source, old_radius, &fmm, &query_ms);
                            print_query_stats(source, old_radius, in_radius, query_ms);
                        } else {
                            target = -1;
                            in_radius = compute_source_and_measure_steiner(&prop_graph, &topo, samples, sample_count, source, radius, &fmm, &query_ms);
                            print_query_stats(source, radius, in_radius, query_ms);
                        }
                    }
                }
            }
        }

        BeginDrawing();
        ClearBackground((Color){9, 13, 18, 255});

        BeginMode3D(camera);
        if (show_mesh) {
            if (mesh_opaque) {
                DrawModel(model, (Vector3){0, 0, 0}, 1.0f, WHITE);
            } else {
                DrawModelWires(model, (Vector3){0, 0, 0}, 1.0f, (Color){80, 100, 120, 120});
            }
        }

        draw_loop_debug(&surface, &loop_debug, loop_debug_mode, loop_stats.chosen_side, marker_scale);

        if (show_graph_vertices) {
            draw_vertex_crosses(surface.vertices, surface.vertex_count, camera, point_radius, (Color){80, 90, 110, 160});
        }

        if (show_all_points && show_distance_colors) {
            draw_distance_point_crosses(samples, sample_count, camera, point_radius, radius, fmm.query_id);
        } else if (show_all_points) {
            draw_point_crosses(samples, sample_count, camera, point_radius, false, fmm.query_id, (Color){130, 140, 150, 140});
        }
        if (!show_distance_colors) {
            draw_point_crosses(samples, sample_count, camera, hit_point_radius, true, fmm.query_id, (Color){40, 220, 120, 230});
        }
        draw_path_color(&loop_draw_path, radius, (Color){255, 170, 40, 230});
        draw_geodesic_path(&geodesic_path, radius);
        DrawSphere(samples[source].p, source_radius, (Color){255, 80, 60, 255});
        if (target >= 0) DrawSphere(samples[target].p, source_radius * 0.8f, (Color){255, 80, 220, 255});
        DrawSphereWires(samples[source].p, radius, 24, 12, (Color){255, 210, 80, 70});
        DrawBoundingBox(surface.bounds, (Color){90, 140, 220, 80});
        EndMode3D();

        DrawRectangle(12, 12, 920, 262, (Color){0, 0, 0, 170});
        DrawText(TextFormat("mesh: %s", mesh_path), 24, 24, 18, RAYWHITE);
        DrawText(TextFormat("triangles: %d | prop nodes: %d | samples: %d active / %d total | steiner spacing: %.4f",
            surface.tri_count, prop_graph.node_count, sample_count - disabled_samples, sample_count, prop_graph.spacing), 24, 50, 18, RAYWHITE);
        DrawText(TextFormat("source: %d | target: %d | radius: %.6f | in radius: %d", source, target, radius, in_radius), 24, 76, 18, RAYWHITE);
        DrawText(TextFormat("radius-bounded local FMM query: %.3f ms", query_ms), 24, 102, 18, (Color){120, 255, 160, 255});
        DrawText(path_ms >= 0.0 ? TextFormat("point-to-point geodesic path: %.3f ms | ok: %d | points: %d | windows: %d/%d", path_ms, path_ok ? 1 : 0, geodesic_path.size, path_ctx.windows_popped, path_ctx.windows_pushed) : "point-to-point geodesic path: n/a", 24, 128, 18, (Color){255, 130, 220, 255});
        DrawText(TextFormat("loop paths: %d/3 | loop segments: %d | last disabled: %d | side: %s | conflicts: %d/%d",
            loop_saved_paths, loop_segments.size, loop_stats.disabled_samples,
            loop_stats.chosen_side == 1 ? "left" : (loop_stats.chosen_side == 2 ? "right" : "none"),
            loop_stats.classification_conflicts, loop_stats.flood_conflicts), 24, 154, 18, (Color){255, 190, 90, 255});
        DrawText(TextFormat("loop debug: %s | N/B cycle | boundary gray, blocked red, left blue, right yellow, conflicts red fill", loop_debug_mode_name(loop_debug_mode)), 24, 180, 18, (Color){180, 210, 255, 255});
        DrawText(TextFormat("preprocessing excluded from query timing: %.3f ms", 1000.0 * (preprocess_t1 - preprocess_t0)), 24, 206, 18, (Color){180, 190, 200, 255});
        DrawText("controls: left-drag orbit | right-drag pan | Ctrl+left reset loop/source | Shift+left add path | wheel zoom | R/Space random", 24, 232, 18, (Color){180, 190, 200, 255});

        EndDrawing();
    }

    fmm_context_free(&fmm);
    geodesic_path_context_free(&path_ctx);
    vec3_vec_free(&geodesic_path);
    vec3_vec_free(&loop_draw_path);
    loop_segment_vec_free(&path_segments);
    loop_segment_vec_free(&loop_segments);
    loop_debug_free(&loop_debug);
    prop_graph_free(&prop_graph);
    surface_topo_free(&topo, &surface);
    free(samples);
    surface_free(&surface);
    UnloadModel(model);
    CloseWindow();
    return 0;
}
