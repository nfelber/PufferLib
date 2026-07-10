#pragma once

#include "geodesic.h"

typedef struct { float x, y; } Qm3Vec2;

typedef struct {
    int tri;
    int entry_edge;
    int parent;
    float key;
    float offset;
    float t0;
    float t1;
    int source_vertex;
    Qm3Vec2 tri2d[3];
    Qm3Vec2 source2;
} Qm3GeoWindow;

typedef struct {
    Qm3GeoWindow* data;
    int size;
    int cap;
} Qm3GeoWindowArray;

typedef struct { int window; float key; } Qm3WindowHeapItem;

typedef struct {
    Qm3WindowHeapItem* data;
    int size;
    int cap;
} Qm3WindowHeap;

typedef struct {
    Qm3GeoWindowArray windows;
    Qm3WindowHeap heap;
    Qm3IntArray* state_windows;
    unsigned int* state_stamp;
    Qm3IntArray touched_states;
    float* best_vertex;
    unsigned int* vertex_stamp;
    Qm3IntArray touched_vertices;
    unsigned char* pseudo_source_vertex;
    Qm3Vec2* tri2d_base;
    unsigned char* tri2d_valid;
    unsigned int query_id;
    int state_count;
    int vertex_count;
    int windows_pushed;
    int windows_popped;
    int pseudo_sources;
} Qm3ContinuousContext;

static Qm3Vec2 qm3_v2_add(Qm3Vec2 a, Qm3Vec2 b) { return (Qm3Vec2){a.x + b.x, a.y + b.y}; }
static Qm3Vec2 qm3_v2_sub(Qm3Vec2 a, Qm3Vec2 b) { return (Qm3Vec2){a.x - b.x, a.y - b.y}; }
static Qm3Vec2 qm3_v2_scale(Qm3Vec2 a, float s) { return (Qm3Vec2){a.x * s, a.y * s}; }
static float qm3_v2_dot(Qm3Vec2 a, Qm3Vec2 b) { return a.x * b.x + a.y * b.y; }
static float qm3_v2_cross(Qm3Vec2 a, Qm3Vec2 b) { return a.x * b.y - a.y * b.x; }
static float qm3_v2_len(Qm3Vec2 a) { return sqrtf(qm3_v2_dot(a, a)); }
static float qm3_v2_dist(Qm3Vec2 a, Qm3Vec2 b) { return qm3_v2_len(qm3_v2_sub(a, b)); }
static Qm3Vec2 qm3_v2_lerp(Qm3Vec2 a, Qm3Vec2 b, float t) { return qm3_v2_add(a, qm3_v2_scale(qm3_v2_sub(b, a), t)); }

static void qm3_geo_window_push(Qm3GeoWindowArray* a, Qm3GeoWindow v) {
    if (a->size == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 1024;
        a->data = (Qm3GeoWindow*)qm3_checked_realloc(a->data, (size_t)a->cap * sizeof(Qm3GeoWindow));
    }
    a->data[a->size++] = v;
}

static void qm3_window_heap_push(Qm3WindowHeap* h, Qm3WindowHeapItem item) {
    if (h->size == h->cap) {
        h->cap = h->cap ? h->cap * 2 : 1024;
        h->data = (Qm3WindowHeapItem*)qm3_checked_realloc(h->data, (size_t)h->cap * sizeof(Qm3WindowHeapItem));
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

static Qm3WindowHeapItem qm3_window_heap_pop(Qm3WindowHeap* h) {
    Qm3WindowHeapItem out = h->data[0];
    Qm3WindowHeapItem item = h->data[--h->size];
    int i = 0;
    for (;;) {
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

static bool qm3_triangle_to_2d(const Qm3Surface* s, int tri, Qm3Vec2 out[3]) {
    Qm3Tri t = s->triangles[tri];
    Qm3Vec3 p0 = s->vertices[t.a], p1 = s->vertices[t.b], p2 = s->vertices[t.c];
    float l01 = qm3_distance(p0, p1);
    if (l01 < 1e-8f) return false;
    Qm3Vec3 e01 = qm3_scale(qm3_sub(p1, p0), 1.0f / l01);
    float x2 = qm3_dot(qm3_sub(p2, p0), e01);
    float y2 = sqrtf(fmaxf(0.0f, qm3_dot(qm3_sub(p2, p0), qm3_sub(p2, p0)) - x2 * x2));
    if (y2 < 1e-8f) return false;
    out[0] = (Qm3Vec2){0, 0}; out[1] = (Qm3Vec2){l01, 0}; out[2] = (Qm3Vec2){x2, y2};
    return true;
}

static void qm3_barycentric3(Qm3Vec3 p, Qm3Vec3 a, Qm3Vec3 b, Qm3Vec3 c, float* u, float* v, float* w) {
    Qm3Vec3 v0 = qm3_sub(b, a), v1 = qm3_sub(c, a), v2 = qm3_sub(p, a);
    float d00 = qm3_dot(v0, v0), d01 = qm3_dot(v0, v1), d11 = qm3_dot(v1, v1);
    float d20 = qm3_dot(v2, v0), d21 = qm3_dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    if (fabsf(denom) < 1e-20f) { *u = 1; *v = 0; *w = 0; return; }
    *v = (d11 * d20 - d01 * d21) / denom;
    *w = (d00 * d21 - d01 * d20) / denom;
    *u = 1.0f - *v - *w;
}

static Qm3Vec2 qm3_bary_to_2d(float u, float v, float w, Qm3Vec2 a, Qm3Vec2 b, Qm3Vec2 c) {
    return (Qm3Vec2){u * a.x + v * b.x + w * c.x, u * a.y + v * b.y + w * c.y};
}

static Qm3Vec2 qm3_surface_point_to_window_2d(const Qm3Surface* s, const Qm3GeoWindow* w, Qm3Vec3 p) {
    Qm3Tri t = s->triangles[w->tri];
    float u, v, q;
    qm3_barycentric3(p, s->vertices[t.a], s->vertices[t.b], s->vertices[t.c], &u, &v, &q);
    return qm3_bary_to_2d(u, v, q, w->tri2d[0], w->tri2d[1], w->tri2d[2]);
}

static int qm3_tri_local_index(const Qm3Surface* s, int tri, int vertex) {
    Qm3Tri t = s->triangles[tri];
    if ((int)t.a == vertex) return 0;
    if ((int)t.b == vertex) return 1;
    if ((int)t.c == vertex) return 2;
    return -1;
}

static int qm3_local_edge_between(const Qm3Surface* s, int tri, int va, int vb) {
    Qm3Tri t = s->triangles[tri];
    int v[3] = {(int)t.a, (int)t.b, (int)t.c};
    for (int e = 0; e < 3; ++e) {
        int a = v[(e + 1) % 3], b = v[(e + 2) % 3];
        if ((a == va && b == vb) || (a == vb && b == va)) return e;
    }
    return -1;
}

static bool qm3_unfold_neighbor_triangle(const Qm3Surface* s, int prev_tri, int edge, int curr_tri, const Qm3Vec2 prev2d[3], Qm3Vec2 curr2d[3]) {
    Qm3Tri pt = s->triangles[prev_tri];
    int pv[3] = {(int)pt.a, (int)pt.b, (int)pt.c};
    int shared0 = pv[(edge + 1) % 3], shared1 = pv[(edge + 2) % 3];
    int prev_u = qm3_tri_local_index(s, prev_tri, shared0), prev_v = qm3_tri_local_index(s, prev_tri, shared1);
    int curr_u = qm3_tri_local_index(s, curr_tri, shared0), curr_v = qm3_tri_local_index(s, curr_tri, shared1);
    if (prev_u < 0 || prev_v < 0 || curr_u < 0 || curr_v < 0) return false;
    int curr_w = 3 - curr_u - curr_v;
    Qm3Tri ct = s->triangles[curr_tri];
    int cv[3] = {(int)ct.a, (int)ct.b, (int)ct.c};
    Qm3Vec2 u2 = prev2d[prev_u], v2 = prev2d[prev_v];
    Qm3Vec2 uv = qm3_v2_sub(v2, u2);
    float len = qm3_v2_len(uv);
    if (len < 1e-8f) return false;
    float duw = qm3_distance(s->vertices[shared0], s->vertices[cv[curr_w]]);
    float dvw = qm3_distance(s->vertices[shared1], s->vertices[cv[curr_w]]);
    float x = (duw * duw - dvw * dvw + len * len) / (2.0f * len);
    float h = sqrtf(fmaxf(0.0f, duw * duw - x * x));
    Qm3Vec2 ex = qm3_v2_scale(uv, 1.0f / len), perp = (Qm3Vec2){-ex.y, ex.x};
    float side = qm3_v2_cross(uv, qm3_v2_sub(prev2d[edge], u2));
    if (side > 0.0f) h = -h;
    curr2d[curr_u] = u2; curr2d[curr_v] = v2; curr2d[curr_w] = qm3_v2_add(u2, qm3_v2_add(qm3_v2_scale(ex, x), qm3_v2_scale(perp, h)));
    return true;
}

static bool qm3_clip_values(float f0, float f1, float* lo, float* hi) {
    const float eps = -1e-6f;
    if (f0 >= eps && f1 >= eps) return true;
    if (f0 < eps && f1 < eps) return false;
    float denom = f0 - f1;
    if (fabsf(denom) < 1e-12f) return false;
    float t = f0 / denom;
    if (f0 < eps) *lo = fmaxf(*lo, t); else *hi = fminf(*hi, t);
    return *lo <= *hi + 1e-6f;
}

static float qm3_visible_cone_value(Qm3Vec2 source, Qm3Vec2 a, Qm3Vec2 b, Qm3Vec2 p, int side) {
    if (side == 0) return qm3_v2_cross(qm3_v2_sub(a, source), qm3_v2_sub(p, source));
    if (side == 1) return -qm3_v2_cross(qm3_v2_sub(b, source), qm3_v2_sub(p, source));
    float source_side = qm3_v2_cross(qm3_v2_sub(b, a), qm3_v2_sub(source, a));
    float point_side = qm3_v2_cross(qm3_v2_sub(b, a), qm3_v2_sub(p, a));
    return -source_side * point_side;
}

static bool qm3_window_clip_edge(const Qm3GeoWindow* w, int edge, float* out_t0, float* out_t1) {
    if (w->entry_edge < 0) { *out_t0 = 0.0f; *out_t1 = 1.0f; return true; }
    Qm3Vec2 ea = w->tri2d[(w->entry_edge + 1) % 3], eb = w->tri2d[(w->entry_edge + 2) % 3];
    Qm3Vec2 a = qm3_v2_lerp(ea, eb, w->t0), b = qm3_v2_lerp(ea, eb, w->t1);
    Qm3Vec2 edge_a = w->tri2d[(edge + 1) % 3], edge_b = w->tri2d[(edge + 2) % 3];
    float cone = qm3_v2_cross(qm3_v2_sub(a, w->source2), qm3_v2_sub(b, w->source2));
    if (fabsf(cone) < 1e-10f) return false;
    float lo = 0.0f, hi = 1.0f, sign = cone > 0.0f ? 1.0f : -1.0f;
    if (!qm3_clip_values(sign * qm3_visible_cone_value(w->source2, a, b, edge_a, 0), sign * qm3_visible_cone_value(w->source2, a, b, edge_b, 0), &lo, &hi)) return false;
    if (!qm3_clip_values(sign * qm3_visible_cone_value(w->source2, a, b, edge_a, 1), sign * qm3_visible_cone_value(w->source2, a, b, edge_b, 1), &lo, &hi)) return false;
    if (!qm3_clip_values(qm3_visible_cone_value(w->source2, a, b, edge_a, 2), qm3_visible_cone_value(w->source2, a, b, edge_b, 2), &lo, &hi)) return false;
    *out_t0 = fminf(fmaxf(lo, 0.0f), 1.0f); *out_t1 = fminf(fmaxf(hi, 0.0f), 1.0f);
    return *out_t1 >= *out_t0 - 1e-6f;
}

static bool qm3_window_sees_point(const Qm3GeoWindow* w, Qm3Vec2 p) {
    if (w->entry_edge < 0) return true;
    Qm3Vec2 ea = w->tri2d[(w->entry_edge + 1) % 3], eb = w->tri2d[(w->entry_edge + 2) % 3];
    Qm3Vec2 a = qm3_v2_lerp(ea, eb, w->t0), b = qm3_v2_lerp(ea, eb, w->t1);
    float cone = qm3_v2_cross(qm3_v2_sub(a, w->source2), qm3_v2_sub(b, w->source2));
    if (fabsf(cone) < 1e-10f) return false;
    float sign = cone > 0.0f ? 1.0f : -1.0f;
    return sign * qm3_visible_cone_value(w->source2, a, b, p, 0) >= -1e-5f &&
           sign * qm3_visible_cone_value(w->source2, a, b, p, 1) >= -1e-5f &&
           qm3_visible_cone_value(w->source2, a, b, p, 2) >= -1e-5f;
}

static bool qm3_window_dominated(const Qm3GeoWindowArray* windows, const Qm3IntArray* state_windows, int state, float t0, float t1, float key) {
    const Qm3IntArray* list = &state_windows[state];
    for (int i = 0; i < list->size; ++i) {
        const Qm3GeoWindow* w = &windows->data[list->data[i]];
        if (w->t0 <= t0 + 1e-5f && w->t1 >= t1 - 1e-5f && w->key <= key + 1e-5f) return true;
    }
    return false;
}

static float qm3_v2_dist_point_segment(Qm3Vec2 p, Qm3Vec2 a, Qm3Vec2 b) {
    Qm3Vec2 ab = qm3_v2_sub(b, a);
    float denom = qm3_v2_dot(ab, ab);
    if (denom <= 1e-12f) return qm3_v2_dist(p, a);
    float t = fminf(fmaxf(qm3_v2_dot(qm3_v2_sub(p, a), ab) / denom, 0.0f), 1.0f);
    return qm3_v2_dist(p, qm3_v2_lerp(a, b, t));
}

static bool qm3_segment_intersect_2d(Qm3Vec2 p, Qm3Vec2 q, Qm3Vec2 a, Qm3Vec2 b, float* out_t, float* out_u) {
    Qm3Vec2 r = qm3_v2_sub(q, p);
    Qm3Vec2 s = qm3_v2_sub(b, a);
    float den = qm3_v2_cross(r, s);
    if (fabsf(den) < 1e-8f) return false;
    Qm3Vec2 ap = qm3_v2_sub(a, p);
    float t = qm3_v2_cross(ap, s) / den;
    float u = qm3_v2_cross(ap, r) / den;
    if (t < -1e-5f || t > 1.0f + 1e-5f || u < -1e-5f || u > 1.0f + 1e-5f) return false;
    *out_t = fminf(fmaxf(t, 0.0f), 1.0f);
    *out_u = fminf(fmaxf(u, 0.0f), 1.0f);
    return true;
}

static bool qm3_point_on_segment_2d(Qm3Vec2 p, Qm3Vec2 a, Qm3Vec2 b, float* out_t) {
    Qm3Vec2 ab = qm3_v2_sub(b, a);
    float len2 = qm3_v2_dot(ab, ab);
    if (len2 < 1e-20f) return false;
    float t = qm3_v2_dot(qm3_v2_sub(p, a), ab) / len2;
    if (t < -1e-5f || t > 1.0f + 1e-5f) return false;
    if (qm3_v2_dist(qm3_v2_lerp(a, b, t), p) > 1e-5f) return false;
    *out_t = fminf(fmaxf(t, 0.0f), 1.0f);
    return true;
}

static bool qm3_segment_portal_hit_2d(Qm3Vec2 p, Qm3Vec2 q, Qm3Vec2 a, Qm3Vec2 b, float min_t, float* out_t, float* out_u) {
    float t, u;
    if (qm3_segment_intersect_2d(p, q, a, b, &t, &u) && t >= min_t - 1e-4f) {
        *out_t = t;
        *out_u = u;
        return true;
    }
    float ta, tb;
    bool has_a = qm3_point_on_segment_2d(a, p, q, &ta) && ta >= min_t - 1e-4f;
    bool has_b = qm3_point_on_segment_2d(b, p, q, &tb) && tb >= min_t - 1e-4f;
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

static bool qm3_robust_portal_hit_2d(Qm3Vec2 p, Qm3Vec2 q, Qm3Vec2 a, Qm3Vec2 b, float u0, float u1, float* out_u) {
    float lt, lu;
    if (qm3_segment_portal_hit_2d(p, q, a, b, -1.0f, &lt, &lu)) {
        *out_u = fminf(fmaxf(lu, fminf(u0, u1)), fmaxf(u0, u1));
        return true;
    }
    Qm3Vec2 r = qm3_v2_sub(q, p);
    Qm3Vec2 s = qm3_v2_sub(b, a);
    float den = qm3_v2_cross(r, s);
    if (fabsf(den) > 1e-10f) {
        lu = qm3_v2_cross(qm3_v2_sub(a, p), r) / den;
        *out_u = fminf(fmaxf(lu, fminf(u0, u1)), fmaxf(u0, u1));
        return true;
    }
    float len2 = qm3_v2_dot(s, s);
    if (len2 <= 1e-20f) return false;
    float up = qm3_v2_dot(qm3_v2_sub(p, a), s) / len2;
    float uq = qm3_v2_dot(qm3_v2_sub(q, a), s) / len2;
    float lo = fminf(u0, u1);
    float hi = fmaxf(u0, u1);
    if (up >= lo && up <= hi) *out_u = up;
    else if (uq >= lo && uq <= hi) *out_u = uq;
    else *out_u = fminf(fmaxf(0.5f * (up + uq), lo), hi);
    return true;
}

static bool qm3_window_chain_uses_pseudo_source(const Qm3GeoWindowArray* windows, int wi) {
    for (int cur = wi; cur >= 0; cur = windows->data[cur].parent) {
        if (windows->data[cur].source_vertex >= 0) return true;
    }
    return false;
}

static Qm3ContinuousContext qm3_continuous_context_create(const Qm3Surface* s, const Qm3PropGraph* graph) {
    Qm3ContinuousContext ctx = {0};
    ctx.state_count = (int)s->triangle_count * 4; ctx.vertex_count = (int)s->vertex_count; ctx.query_id = 1;
    ctx.state_windows = (Qm3IntArray*)calloc((size_t)ctx.state_count, sizeof(Qm3IntArray));
    ctx.state_stamp = (unsigned int*)calloc((size_t)ctx.state_count, sizeof(unsigned int));
    ctx.best_vertex = (float*)malloc((size_t)ctx.vertex_count * sizeof(float));
    ctx.vertex_stamp = (unsigned int*)calloc((size_t)ctx.vertex_count, sizeof(unsigned int));
    ctx.pseudo_source_vertex = (unsigned char*)calloc((size_t)ctx.vertex_count, sizeof(unsigned char));
    ctx.tri2d_base = (Qm3Vec2*)calloc((size_t)s->triangle_count * 3, sizeof(Qm3Vec2));
    ctx.tri2d_valid = (unsigned char*)calloc((size_t)s->triangle_count, sizeof(unsigned char));
    QM3_ASSERT(ctx.state_windows && ctx.state_stamp && ctx.best_vertex && ctx.vertex_stamp && ctx.pseudo_source_vertex && ctx.tri2d_base && ctx.tri2d_valid);
    for (uint32_t tri = 0; tri < s->triangle_count; ++tri) ctx.tri2d_valid[tri] = qm3_triangle_to_2d(s, (int)tri, &ctx.tri2d_base[3 * tri]) ? 1 : 0;
    for (uint32_t v = 0; v < s->vertex_count; ++v) {
        int begin = graph->node_tri_offsets[v], end = graph->node_tri_offsets[v + 1];
        float angle_sum = 0.0f; int boundary = 0;
        for (int i = begin; i < end; ++i) {
            int tri = graph->node_tri_ids[i]; int local = qm3_tri_local_index(s, tri, (int)v);
            if (local < 0) continue;
            Qm3Tri t = s->triangles[tri]; int tv[3] = {(int)t.a, (int)t.b, (int)t.c};
            Qm3Vec3 p = s->vertices[v], a = s->vertices[tv[(local + 1) % 3]], b = s->vertices[tv[(local + 2) % 3]];
            angle_sum += acosf(fminf(fmaxf(qm3_dot(qm3_normalize(qm3_sub(a, p)), qm3_normalize(qm3_sub(b, p))), -1.0f), 1.0f));
            for (int e = 0; e < 3; ++e) if ((tv[(e + 1) % 3] == (int)v || tv[(e + 2) % 3] == (int)v) && ((int*)&s->triangle_neighbors[tri])[e] < 0) boundary++;
        }
        ctx.pseudo_source_vertex[v] = (boundary > 0 || angle_sum > 2.0f * (float)M_PI + 1e-3f) ? 1 : 0;
    }
    return ctx;
}

static void qm3_continuous_context_free(Qm3ContinuousContext* ctx) {
    for (int i = 0; i < ctx->state_count; ++i) qm3_int_free(&ctx->state_windows[i]);
    free(ctx->state_windows); free(ctx->state_stamp); qm3_int_free(&ctx->touched_states); free(ctx->best_vertex); free(ctx->vertex_stamp); qm3_int_free(&ctx->touched_vertices);
    free(ctx->pseudo_source_vertex); free(ctx->tri2d_base); free(ctx->tri2d_valid); free(ctx->windows.data); free(ctx->heap.data); memset(ctx, 0, sizeof(*ctx));
}

static void qm3_continuous_context_begin(Qm3ContinuousContext* ctx) {
    if (++ctx->query_id == 0) { memset(ctx->state_stamp, 0, (size_t)ctx->state_count * sizeof(unsigned int)); memset(ctx->vertex_stamp, 0, (size_t)ctx->vertex_count * sizeof(unsigned int)); ctx->query_id = 1; }
    for (int i = 0; i < ctx->touched_states.size; ++i) ctx->state_windows[ctx->touched_states.data[i]].size = 0;
    ctx->touched_states.size = 0; ctx->touched_vertices.size = 0; ctx->windows.size = 0; ctx->heap.size = 0; ctx->windows_pushed = 0; ctx->windows_popped = 0; ctx->pseudo_sources = 0;
}

static Qm3IntArray* qm3_geo_state_list(Qm3ContinuousContext* ctx, int state) {
    if (ctx->state_stamp[state] != ctx->query_id) { ctx->state_stamp[state] = ctx->query_id; ctx->state_windows[state].size = 0; qm3_int_push(&ctx->touched_states, state); }
    return &ctx->state_windows[state];
}

static float qm3_geo_best_vertex(const Qm3ContinuousContext* ctx, int v) { return ctx->vertex_stamp[v] == ctx->query_id ? ctx->best_vertex[v] : INFINITY; }
static void qm3_geo_set_best_vertex(Qm3ContinuousContext* ctx, int v, float d) { if (ctx->vertex_stamp[v] != ctx->query_id) { ctx->vertex_stamp[v] = ctx->query_id; qm3_int_push(&ctx->touched_vertices, v); } ctx->best_vertex[v] = d; }

static bool qm3_copy_tri2d(const Qm3ContinuousContext* ctx, int tri, Qm3Vec2 out[3]) { if (!ctx->tri2d_valid[tri]) return false; out[0] = ctx->tri2d_base[3 * tri]; out[1] = ctx->tri2d_base[3 * tri + 1]; out[2] = ctx->tri2d_base[3 * tri + 2]; return true; }

static void qm3_geo_push_window(Qm3ContinuousContext* ctx, Qm3GeoWindow w, int state, float priority) {
    int wi = ctx->windows.size; qm3_geo_window_push(&ctx->windows, w); qm3_int_push(qm3_geo_state_list(ctx, state), wi); qm3_window_heap_push(&ctx->heap, (Qm3WindowHeapItem){wi, priority}); ctx->windows_pushed++;
}

static bool qm3_seed_source_root_in_tri(const Qm3Surface* s, Qm3ContinuousContext* ctx, Qm3Vec3 source_p, int source_tri) {
    if (source_tri < 0) return false;
    Qm3GeoWindow root = {.tri = source_tri, .entry_edge = -1, .parent = -1, .key = 0, .offset = 0, .t0 = 0, .t1 = 1, .source_vertex = -1};
    if (!qm3_copy_tri2d(ctx, root.tri, root.tri2d)) return false;
    Qm3Tri st = s->triangles[source_tri];
    float su, sv, sw;
    qm3_barycentric3(source_p, s->vertices[st.a], s->vertices[st.b], s->vertices[st.c], &su, &sv, &sw);
    root.source2 = qm3_bary_to_2d(su, sv, sw, root.tri2d[0], root.tri2d[1], root.tri2d[2]);
    qm3_geo_push_window(ctx, root, source_tri * 4, 0.0f);
    return true;
}

static bool qm3_seed_source_roots(
    const Qm3Surface* s,
    const Qm3PropGraph* graph,
    Qm3ContinuousContext* ctx,
    Qm3Vec3 source_p,
    int source_tri
) {
    qm3_continuous_context_begin(ctx);
    bool seeded = false;
    Qm3SourceSupport source = qm3_source_support(s, source_p, source_tri, 1e-5f);
    if (source.source_vertex >= 0 && source.source_vertex < graph->node_count) {
        int begin = graph->node_tri_offsets[source.source_vertex];
        int end = graph->node_tri_offsets[source.source_vertex + 1];
        for (int i = begin; i < end; ++i) {
            seeded |= qm3_seed_source_root_in_tri(s, ctx, source_p, graph->node_tri_ids[i]);
        }
    } else {
        for (int i = 0; i < source.tri_count; ++i) {
            seeded |= qm3_seed_source_root_in_tri(s, ctx, source_p, source.tris[i]);
        }
    }
    if (!seeded) seeded = qm3_seed_source_root_in_tri(s, ctx, source_p, source_tri);
    return seeded;
}

static bool qm3_build_shared_source_windows(
    const Qm3Surface* s,
    const Qm3PropGraph* graph,
    Qm3ContinuousContext* ctx,
    Qm3Vec3 source_p,
    int source_tri,
    float stop_distance
) {
    if (!isfinite(stop_distance) || stop_distance <= 0.0f) return false;
    if (!qm3_seed_source_roots(s, graph, ctx, source_p, source_tri)) return false;
    while (ctx->heap.size > 0) {
        Qm3WindowHeapItem item = qm3_window_heap_pop(&ctx->heap);
        ctx->windows_popped++;
        if (item.key >= stop_distance) break;
        Qm3GeoWindow w = ctx->windows.data[item.window];

        for (int edge = 0; edge < 3; ++edge) {
            if (edge == w.entry_edge) continue;
            int next_tri = ((int*)&s->triangle_neighbors[w.tri])[edge];
            if (next_tri < 0) continue;
            float t0, t1;
            if (!qm3_window_clip_edge(&w, edge, &t0, &t1)) continue;

            Qm3Tri wt = s->triangles[w.tri];
            int wv[3] = {(int)wt.a, (int)wt.b, (int)wt.c};
            int va = wv[(edge + 1) % 3], vb = wv[(edge + 2) % 3];
            int endpoint_vids[2] = {va, vb};
            float endpoint_ts[2] = {t0, t1};
            Qm3Vec2 endpoint_points[2] = {w.tri2d[(edge + 1) % 3], w.tri2d[(edge + 2) % 3]};
            for (int pi = 0; pi < 2; ++pi) {
                if ((pi == 0 && endpoint_ts[pi] > 1e-5f) || (pi == 1 && endpoint_ts[pi] < 1.0f - 1e-5f)) continue;
                int vid = endpoint_vids[pi];
                if (!ctx->pseudo_source_vertex[vid]) continue;
                float vd = w.offset + qm3_v2_dist(w.source2, endpoint_points[pi]);
                if (vd >= qm3_geo_best_vertex(ctx, vid) - 1e-6f || vd >= stop_distance) continue;
                qm3_geo_set_best_vertex(ctx, vid, vd);
                ctx->pseudo_sources++;
                for (int ti = graph->node_tri_offsets[vid]; ti < graph->node_tri_offsets[vid + 1]; ++ti) {
                    int seed_tri = graph->node_tri_ids[ti];
                    Qm3GeoWindow vw = {.tri = seed_tri, .entry_edge = -1, .parent = item.window, .key = vd, .offset = vd, .t0 = 0, .t1 = 1, .source_vertex = vid};
                    if (!qm3_copy_tri2d(ctx, seed_tri, vw.tri2d)) continue;
                    int local = qm3_tri_local_index(s, seed_tri, vid);
                    if (local < 0) continue;
                    vw.source2 = vw.tri2d[local];
                    qm3_geo_push_window(ctx, vw, seed_tri * 4, vd);
                }
            }

            if (t1 < t0 + 1e-6f) continue;
            Qm3Vec2 ca = qm3_v2_lerp(w.tri2d[(edge + 1) % 3], w.tri2d[(edge + 2) % 3], t0);
            Qm3Vec2 cb = qm3_v2_lerp(w.tri2d[(edge + 1) % 3], w.tri2d[(edge + 2) % 3], t1);
            float key = w.offset + qm3_v2_dist_point_segment(w.source2, ca, cb);
            if (key >= stop_distance) continue;
            int next_entry = qm3_local_edge_between(s, next_tri, va, vb);
            if (next_entry < 0) continue;
            int next_state = next_tri * 4 + next_entry + 1;
            Qm3GeoWindow nw = {.tri = next_tri, .entry_edge = next_entry, .parent = item.window, .key = key, .offset = w.offset, .source_vertex = w.source_vertex, .source2 = w.source2};
            if (!qm3_unfold_neighbor_triangle(s, w.tri, edge, next_tri, w.tri2d, nw.tri2d)) continue;
            Qm3Tri nt = s->triangles[next_tri];
            int nv[3] = {(int)nt.a, (int)nt.b, (int)nt.c};
            int na = nv[(next_entry + 1) % 3], nb = nv[(next_entry + 2) % 3];
            if (na == va && nb == vb) { nw.t0 = t0; nw.t1 = t1; }
            else { nw.t0 = 1.0f - t1; nw.t1 = 1.0f - t0; }
            qm3_geo_state_list(ctx, next_state);
            if (qm3_window_dominated(&ctx->windows, ctx->state_windows, next_state, nw.t0, nw.t1, key)) continue;
            qm3_geo_push_window(ctx, nw, next_state, key);
        }
    }
    return true;
}

static int qm3_find_best_target_window(
    const Qm3Surface* s,
    const Qm3ContinuousContext* ctx,
    Qm3Vec3 target_p,
    int target_tri,
    float stop_distance,
    float* out_distance
) {
    if (target_tri < 0) return -1;
    float best_target = isfinite(stop_distance) ? stop_distance * 1.0001f + 1e-6f : INFINITY;
    int best_wi = -1;
    for (int state_offset = 0; state_offset < 4; ++state_offset) {
        int state = target_tri * 4 + state_offset;
        if (state < 0 || state >= ctx->state_count || ctx->state_stamp[state] != ctx->query_id) continue;
        const Qm3IntArray* list = &ctx->state_windows[state];
        for (int i = 0; i < list->size; ++i) {
            int wi = list->data[i];
            const Qm3GeoWindow* w = &ctx->windows.data[wi];
            Qm3Vec2 target2 = qm3_surface_point_to_window_2d(s, w, target_p);
            if (!qm3_window_sees_point(w, target2)) continue;
            float d = w->offset + qm3_v2_dist(w->source2, target2);
            bool better = d < best_target - 1e-6f;
            if (!better && fabsf(d - best_target) <= 1e-6f && best_wi >= 0) {
                better = qm3_window_chain_uses_pseudo_source(&ctx->windows, best_wi) && !qm3_window_chain_uses_pseudo_source(&ctx->windows, wi);
            }
            if (better) {
                best_target = d;
                best_wi = wi;
            }
        }
    }
    if (best_wi >= 0 && out_distance) *out_distance = best_target;
    return best_wi;
}

static bool qm3_append_window_backtrace(
    const Qm3Surface* s,
    const Qm3GeoWindowArray* windows,
    int start,
    int wi,
    Qm3Vec3 start_p,
    Qm3Vec3 end_p,
    Qm3Path* path,
    Qm3PathSegmentArray* segments
) {
    Qm3Path rev = {0};
    Qm3PathSegmentArray rev_segments = {0};
    Qm3Vec3 cur_p = end_p;
    qm3_path_push(&rev, end_p);
    for (int cur = wi; cur != start; cur = windows->data[cur].parent) {
        const Qm3GeoWindow* w = &windows->data[cur];
        if (w->entry_edge < 0 || w->parent < 0) { qm3_path_free(&rev); qm3_path_segment_free(&rev_segments); return false; }
        Qm3Vec2 cur2 = qm3_surface_point_to_window_2d(s, w, cur_p); int a = (w->entry_edge + 1) % 3, b = (w->entry_edge + 2) % 3;
        Qm3Vec2 pa = w->tri2d[a], pb = w->tri2d[b];
        float lu;
        if (!qm3_robust_portal_hit_2d(w->source2, cur2, pa, pb, w->t0, w->t1, &lu)) { qm3_path_free(&rev); qm3_path_segment_free(&rev_segments); return false; }
        Qm3Tri t = s->triangles[w->tri]; int tv[3] = {(int)t.a, (int)t.b, (int)t.c};
        Qm3Vec3 q = qm3_add(qm3_scale(s->vertices[tv[a]], 1.0f - lu), qm3_scale(s->vertices[tv[b]], lu));
        qm3_path_segment_push(&rev_segments, (Qm3PathSegment){.tri = w->tri, .a = q, .b = cur_p});
        qm3_path_push(&rev, q); cur_p = q;
    }
    qm3_path_segment_push(&rev_segments, (Qm3PathSegment){.tri = windows->data[start].tri, .a = start_p, .b = cur_p});
    qm3_path_push(&rev, start_p); qm3_path_clear(path);
    for (int i = (int)rev.count - 1; i >= 0; --i) qm3_path_push(path, rev.points[i]);
    if (segments) {
        qm3_path_segment_clear(segments);
        for (int i = (int)rev_segments.count - 1; i >= 0; --i) qm3_path_segment_push(segments, rev_segments.data[i]);
    }
    qm3_path_free(&rev);
    qm3_path_segment_free(&rev_segments);
    return true;
}

static bool qm3_append_window_chain_path(
    const Qm3Surface* s,
    Qm3Vec3 root_source,
    const Qm3GeoWindowArray* windows,
    int wi,
    Qm3Vec3 end_p,
    Qm3Path* path,
    Qm3PathSegmentArray* segments
) {
    if (wi < 0) return false; int source_vertex = windows->data[wi].source_vertex; int start = wi;
    while (windows->data[start].parent >= 0 && windows->data[windows->data[start].parent].source_vertex == source_vertex) start = windows->data[start].parent;
    if (source_vertex >= 0) {
        int parent = windows->data[start].parent; if (parent < 0) return false; Qm3Path prefix = {0}; Qm3PathSegmentArray prefix_segments = {0};
        if (!qm3_append_window_chain_path(s, root_source, windows, parent, s->vertices[source_vertex], &prefix, &prefix_segments)) { qm3_path_free(&prefix); qm3_path_segment_free(&prefix_segments); return false; }
        Qm3Path suffix = {0}; Qm3PathSegmentArray suffix_segments = {0}; bool ok = qm3_append_window_backtrace(s, windows, start, wi, s->vertices[source_vertex], end_p, &suffix, &suffix_segments);
        qm3_path_clear(path); for (uint32_t i = 0; i < prefix.count; ++i) qm3_path_push(path, prefix.points[i]); for (uint32_t i = 1; ok && i < suffix.count; ++i) qm3_path_push(path, suffix.points[i]);
        if (segments) {
            qm3_path_segment_clear(segments);
            for (uint32_t i = 0; i < prefix_segments.count; ++i) qm3_path_segment_push(segments, prefix_segments.data[i]);
            for (uint32_t i = 0; ok && i < suffix_segments.count; ++i) qm3_path_segment_push(segments, suffix_segments.data[i]);
        }
        qm3_path_free(&prefix); qm3_path_free(&suffix); qm3_path_segment_free(&prefix_segments); qm3_path_segment_free(&suffix_segments); return ok;
    }
    return qm3_append_window_backtrace(s, windows, start, wi, root_source, end_p, path, segments);
}
