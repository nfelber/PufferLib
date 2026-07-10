#pragma once

#include "mesh3d.h"

#include <float.h>

typedef struct {
    int from;
    int to;
    int tri;
    float w;
} Qm3PropEdgeTmp;

typedef struct {
    Qm3PropEdgeTmp* data;
    int size;
    int cap;
} Qm3PropEdgeTmpArray;

typedef struct {
    int* data;
    int size;
    int cap;
} Qm3IntArray;

typedef struct {
    Qm3Vec3* nodes;
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
} Qm3PropGraph;

typedef struct {
    int* tri_sample_offsets;
    int* tri_sample_ids;
    int tri_count;
    int sample_count;
} Qm3SurfaceTopo;

typedef struct {
    int node;
    float dist;
} Qm3HeapItem;

typedef struct {
    Qm3HeapItem* data;
    int size;
    int cap;
} Qm3MinHeap;

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
    Qm3MinHeap heap;
    int vertex_count;
    int tri_count;
} Qm3FMMContext;

enum {
    QM3_FMM_FAR = 0,
    QM3_FMM_TRIAL = 1,
    QM3_FMM_ACCEPTED = 2,
};

static void qm3_int_push(Qm3IntArray* a, int v) {
    if (a->size == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 8;
        a->data = (int*)qm3_checked_realloc(a->data, (size_t)a->cap * sizeof(int));
    }
    a->data[a->size++] = v;
}

static void qm3_int_free(Qm3IntArray* a) {
    free(a->data);
    memset(a, 0, sizeof(*a));
}

static void qm3_prop_edge_tmp_push(Qm3PropEdgeTmpArray* a, Qm3PropEdgeTmp v) {
    if (a->size == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 1024;
        a->data = (Qm3PropEdgeTmp*)qm3_checked_realloc(a->data, (size_t)a->cap * sizeof(Qm3PropEdgeTmp));
    }
    a->data[a->size++] = v;
}

static void qm3_heap_push(Qm3MinHeap* h, Qm3HeapItem item) {
    if (h->size == h->cap) {
        h->cap = h->cap ? h->cap * 2 : 256;
        h->data = (Qm3HeapItem*)qm3_checked_realloc(h->data, (size_t)h->cap * sizeof(Qm3HeapItem));
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

static Qm3HeapItem qm3_heap_pop(Qm3MinHeap* h) {
    Qm3HeapItem out = h->data[0];
    Qm3HeapItem item = h->data[--h->size];
    int i = 0;
    for (;;) {
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

static void qm3_barycentric3f(Qm3Vec3 p, Qm3Vec3 a, Qm3Vec3 b, Qm3Vec3 c, float* u, float* v, float* w) {
    Qm3Vec3 v0 = qm3_sub(b, a), v1 = qm3_sub(c, a), v2 = qm3_sub(p, a);
    float d00 = qm3_dot(v0, v0), d01 = qm3_dot(v0, v1), d11 = qm3_dot(v1, v1);
    float d20 = qm3_dot(v2, v0), d21 = qm3_dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    if (fabsf(denom) < 1e-20f) { *u = 1.0f; *v = 0.0f; *w = 0.0f; return; }
    *v = (d11 * d20 - d01 * d21) / denom;
    *w = (d00 * d21 - d01 * d20) / denom;
    *u = 1.0f - *v - *w;
}

typedef struct {
    int source_vertex;
    int tri_count;
    int tris[2];
} Qm3SourceSupport;

static Qm3SourceSupport qm3_source_support(const Qm3Surface* surface, Qm3Vec3 p, int source_tri, float tol) {
    Qm3SourceSupport support = {.source_vertex = -1, .tri_count = 0, .tris = {-1, -1}};
    if (source_tri < 0 || source_tri >= (int)surface->triangle_count) return support;

    Qm3Tri tri = surface->triangles[source_tri];
    uint32_t verts[3] = {tri.a, tri.b, tri.c};
    float bary[3];
    qm3_barycentric3f(p, surface->vertices[tri.a], surface->vertices[tri.b], surface->vertices[tri.c], &bary[0], &bary[1], &bary[2]);

    int near_zero = 0;
    for (int i = 0; i < 3; ++i) if (bary[i] <= tol) near_zero++;
    if (near_zero >= 2) {
        int vertex_local = 0;
        if (bary[1] > bary[vertex_local]) vertex_local = 1;
        if (bary[2] > bary[vertex_local]) vertex_local = 2;
        support.source_vertex = (int)verts[vertex_local];
        return support;
    }

    support.tris[support.tri_count++] = source_tri;
    if (near_zero == 1) {
        for (int edge = 0; edge < 3; ++edge) {
            if (bary[edge] > tol) continue;
            int neighbor = ((int*)&surface->triangle_neighbors[source_tri])[edge];
            if (neighbor >= 0) support.tris[support.tri_count++] = neighbor;
            break;
        }
    }
    return support;
}

static Qm3SurfaceTopo qm3_surface_topo_build(const Qm3Surface* surface) {
    Qm3SurfaceTopo topo = {0};
    topo.tri_count = (int)surface->triangle_count;
    topo.sample_count = (int)surface->sample_count;
    topo.tri_sample_offsets = (int*)calloc((size_t)surface->triangle_count + 1, sizeof(int));
    topo.tri_sample_ids = (int*)malloc((size_t)surface->sample_count * sizeof(int));
    QM3_ASSERT(topo.tri_sample_offsets && topo.tri_sample_ids);
    for (uint32_t i = 0; i < surface->sample_count; ++i) topo.tri_sample_offsets[surface->samples[i].tri + 1]++;
    for (uint32_t i = 0; i < surface->triangle_count; ++i) topo.tri_sample_offsets[i + 1] += topo.tri_sample_offsets[i];
    int* cursor = (int*)malloc((size_t)surface->triangle_count * sizeof(int));
    QM3_ASSERT(cursor);
    memcpy(cursor, topo.tri_sample_offsets, (size_t)surface->triangle_count * sizeof(int));
    for (uint32_t i = 0; i < surface->sample_count; ++i) topo.tri_sample_ids[cursor[surface->samples[i].tri]++] = (int)i;
    free(cursor);
    return topo;
}

static void qm3_surface_topo_free(Qm3SurfaceTopo* topo) {
    free(topo->tri_sample_offsets);
    free(topo->tri_sample_ids);
    memset(topo, 0, sizeof(*topo));
}

static int qm3_prop_add_node(Qm3PropGraph* graph, Qm3Vec3 p) {
    if (graph->node_count == graph->node_cap) {
        graph->node_cap = graph->node_cap ? graph->node_cap * 2 : 1024;
        graph->nodes = (Qm3Vec3*)qm3_checked_realloc(graph->nodes, (size_t)graph->node_cap * sizeof(Qm3Vec3));
    }
    int idx = graph->node_count++;
    graph->nodes[idx] = p;
    return idx;
}

static void qm3_prop_add_node_to_tri(Qm3IntArray* tri_nodes, Qm3IntArray* node_tris, int tri, int node) {
    for (int i = 0; i < tri_nodes[tri].size; ++i) if (tri_nodes[tri].data[i] == node) return;
    qm3_int_push(&tri_nodes[tri], node);
    qm3_int_push(&node_tris[node], tri);
}

static void qm3_prop_add_undirected(Qm3PropGraph* graph, Qm3PropEdgeTmpArray* edges, int a, int b, int tri) {
    if (a == b) return;
    float w = qm3_distance(graph->nodes[a], graph->nodes[b]);
    qm3_prop_edge_tmp_push(edges, (Qm3PropEdgeTmp){a, b, tri, w});
    qm3_prop_edge_tmp_push(edges, (Qm3PropEdgeTmp){b, a, tri, w});
}

static int qm3_prop_edge_cmp(const void* pa, const void* pb) {
    const Qm3PropEdgeTmp* a = (const Qm3PropEdgeTmp*)pa;
    const Qm3PropEdgeTmp* b = (const Qm3PropEdgeTmp*)pb;
    if (a->from != b->from) return a->from < b->from ? -1 : 1;
    if (a->to != b->to) return a->to < b->to ? -1 : 1;
    if (a->tri != b->tri) return a->tri < b->tri ? -1 : 1;
    return a->w < b->w ? -1 : a->w > b->w;
}

static Qm3PropGraph qm3_prop_graph_build(const Qm3Surface* surface, float spacing) {
    Qm3PropGraph graph = {0};
    graph.tri_count = (int)surface->triangle_count;
    graph.spacing = spacing;
    Qm3IntArray* tri_nodes = (Qm3IntArray*)calloc(surface->triangle_count, sizeof(Qm3IntArray));
    Qm3IntArray* node_tris = NULL;
    Qm3PropEdgeTmpArray edges = {0};
    QM3_ASSERT(tri_nodes);

    for (uint32_t i = 0; i < surface->vertex_count; ++i) qm3_prop_add_node(&graph, surface->vertices[i]);
    node_tris = (Qm3IntArray*)calloc((size_t)graph.node_cap, sizeof(Qm3IntArray));
    QM3_ASSERT(node_tris);
    for (uint32_t tri = 0; tri < surface->triangle_count; ++tri) {
        Qm3Tri t = surface->triangles[tri];
        qm3_prop_add_node_to_tri(tri_nodes, node_tris, tri, (int)t.a);
        qm3_prop_add_node_to_tri(tri_nodes, node_tris, tri, (int)t.b);
        qm3_prop_add_node_to_tri(tri_nodes, node_tris, tri, (int)t.c);
    }

    for (uint32_t tri = 0; tri < surface->triangle_count; ++tri) {
        Qm3Tri t = surface->triangles[tri];
        uint32_t v[3] = {t.a, t.b, t.c};
        for (int e = 0; e < 3; ++e) {
            uint32_t a = v[(e + 1) % 3];
            uint32_t b = v[(e + 2) % 3];
            if (a > b) { uint32_t tmp = a; a = b; b = tmp; }
            /* Add Steiner points once per canonical edge by only processing from the lower adjacent triangle. */
            int nb = ((int*)&surface->triangle_neighbors[tri])[e];
            if (nb >= 0 && nb < (int)tri) continue;
            float len = qm3_distance(surface->vertices[a], surface->vertices[b]);
            int segments = spacing > 1e-8f ? (int)ceilf(len / spacing) : 1;
            if (segments < 1) segments = 1;
            for (int s = 1; s < segments; ++s) {
                float u = s / (float)segments;
                Qm3Vec3 p = qm3_add(qm3_scale(surface->vertices[a], 1.0f - u), qm3_scale(surface->vertices[b], u));
                int node = qm3_prop_add_node(&graph, p);
                if (graph.node_count > graph.node_cap / 2) {
                    node_tris = (Qm3IntArray*)qm3_checked_realloc(node_tris, (size_t)graph.node_cap * sizeof(Qm3IntArray));
                }
                node_tris[node] = (Qm3IntArray){0};
                qm3_prop_add_node_to_tri(tri_nodes, node_tris, (int)tri, node);
                if (nb >= 0) qm3_prop_add_node_to_tri(tri_nodes, node_tris, nb, node);
            }
        }
    }

    for (uint32_t tri = 0; tri < surface->triangle_count; ++tri) {
        Qm3IntArray nodes = tri_nodes[tri];
        for (int a = 0; a < nodes.size; ++a) for (int b = a + 1; b < nodes.size; ++b) qm3_prop_add_undirected(&graph, &edges, nodes.data[a], nodes.data[b], (int)tri);
    }

    qsort(edges.data, (size_t)edges.size, sizeof(Qm3PropEdgeTmp), qm3_prop_edge_cmp);
    int unique_edges = 0;
    for (int i = 0; i < edges.size; ++i) {
        if (unique_edges > 0 && edges.data[i].from == edges.data[unique_edges - 1].from && edges.data[i].to == edges.data[unique_edges - 1].to && edges.data[i].tri == edges.data[unique_edges - 1].tri) {
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
    QM3_ASSERT(graph.edge_offsets && graph.edge_to && graph.edge_tri && graph.edge_weight);
    for (int i = 0; i < unique_edges; ++i) graph.edge_offsets[edges.data[i].from + 1]++;
    for (int i = 0; i < graph.node_count; ++i) graph.edge_offsets[i + 1] += graph.edge_offsets[i];
    int* cursor = (int*)malloc((size_t)graph.node_count * sizeof(int));
    QM3_ASSERT(cursor);
    memcpy(cursor, graph.edge_offsets, (size_t)graph.node_count * sizeof(int));
    for (int i = 0; i < unique_edges; ++i) {
        int pos = cursor[edges.data[i].from]++;
        graph.edge_to[pos] = edges.data[i].to;
        graph.edge_tri[pos] = edges.data[i].tri;
        graph.edge_weight[pos] = edges.data[i].w;
    }
    free(cursor);
    free(edges.data);

    graph.tri_node_offsets = (int*)calloc((size_t)surface->triangle_count + 1, sizeof(int));
    graph.node_tri_offsets = (int*)calloc((size_t)graph.node_count + 1, sizeof(int));
    QM3_ASSERT(graph.tri_node_offsets && graph.node_tri_offsets);
    for (uint32_t tri = 0; tri < surface->triangle_count; ++tri) graph.tri_node_offsets[tri + 1] = tri_nodes[tri].size;
    for (int node = 0; node < graph.node_count; ++node) graph.node_tri_offsets[node + 1] = node_tris[node].size;
    for (uint32_t tri = 0; tri < surface->triangle_count; ++tri) graph.tri_node_offsets[tri + 1] += graph.tri_node_offsets[tri];
    for (int node = 0; node < graph.node_count; ++node) graph.node_tri_offsets[node + 1] += graph.node_tri_offsets[node];
    graph.tri_node_ids = (int*)malloc((size_t)graph.tri_node_offsets[surface->triangle_count] * sizeof(int));
    graph.node_tri_ids = (int*)malloc((size_t)graph.node_tri_offsets[graph.node_count] * sizeof(int));
    QM3_ASSERT(graph.tri_node_ids && graph.node_tri_ids);
    for (uint32_t tri = 0; tri < surface->triangle_count; ++tri) memcpy(&graph.tri_node_ids[graph.tri_node_offsets[tri]], tri_nodes[tri].data, (size_t)tri_nodes[tri].size * sizeof(int));
    for (int node = 0; node < graph.node_count; ++node) memcpy(&graph.node_tri_ids[graph.node_tri_offsets[node]], node_tris[node].data, (size_t)node_tris[node].size * sizeof(int));

    for (uint32_t tri = 0; tri < surface->triangle_count; ++tri) qm3_int_free(&tri_nodes[tri]);
    for (int node = 0; node < graph.node_count; ++node) qm3_int_free(&node_tris[node]);
    free(tri_nodes);
    free(node_tris);
    return graph;
}

static void qm3_prop_graph_free(Qm3PropGraph* graph) {
    free(graph->nodes); free(graph->edge_offsets); free(graph->edge_to); free(graph->edge_tri); free(graph->edge_weight);
    free(graph->tri_node_offsets); free(graph->tri_node_ids); free(graph->node_tri_offsets); free(graph->node_tri_ids);
    memset(graph, 0, sizeof(*graph));
}

static Qm3FMMContext qm3_fmm_create(int vertex_count, int tri_count) {
    Qm3FMMContext ctx = {0};
    ctx.vertex_count = vertex_count; ctx.tri_count = tri_count; ctx.query_id = 1;
    ctx.dist = (float*)malloc((size_t)vertex_count * sizeof(float));
    ctx.prev = (int*)malloc((size_t)vertex_count * sizeof(int));
    ctx.prev_tri = (int*)malloc((size_t)vertex_count * sizeof(int));
    ctx.stamp = (unsigned int*)calloc((size_t)vertex_count, sizeof(unsigned int));
    ctx.state = (unsigned char*)calloc((size_t)vertex_count, sizeof(unsigned char));
    ctx.tri_stamp = (unsigned int*)calloc((size_t)tri_count, sizeof(unsigned int));
    QM3_ASSERT(ctx.dist && ctx.prev && ctx.prev_tri && ctx.stamp && ctx.state && ctx.tri_stamp);
    return ctx;
}

static void qm3_fmm_free(Qm3FMMContext* ctx) {
    free(ctx->dist); free(ctx->prev); free(ctx->prev_tri); free(ctx->stamp); free(ctx->state); free(ctx->tri_stamp); free(ctx->reached_tris); free(ctx->heap.data);
    memset(ctx, 0, sizeof(*ctx));
}

static void qm3_fmm_mark_triangle(Qm3FMMContext* ctx, int tri) {
    if (tri < 0 || tri >= ctx->tri_count || ctx->tri_stamp[tri] == ctx->query_id) return;
    ctx->tri_stamp[tri] = ctx->query_id;
    if (ctx->reached_tri_count == ctx->reached_tri_cap) {
        ctx->reached_tri_cap = ctx->reached_tri_cap ? ctx->reached_tri_cap * 2 : 128;
        ctx->reached_tris = (int*)qm3_checked_realloc(ctx->reached_tris, (size_t)ctx->reached_tri_cap * sizeof(int));
    }
    ctx->reached_tris[ctx->reached_tri_count++] = tri;
}

static float qm3_fmm_get_dist(const Qm3FMMContext* ctx, int v) { return ctx->stamp[v] == ctx->query_id ? ctx->dist[v] : INFINITY; }
static unsigned char qm3_fmm_get_state(const Qm3FMMContext* ctx, int v) { return ctx->stamp[v] == ctx->query_id ? ctx->state[v] : QM3_FMM_FAR; }

static void qm3_fmm_set_vertex(Qm3FMMContext* ctx, int v, float dist, unsigned char state, int prev, int prev_tri) {
    ctx->stamp[v] = ctx->query_id; ctx->dist[v] = dist; ctx->state[v] = state; ctx->prev[v] = prev; ctx->prev_tri[v] = prev_tri;
}

static void qm3_fmm_push_trial(Qm3FMMContext* ctx, int v, float dist, int prev, int prev_tri) {
    if (dist >= qm3_fmm_get_dist(ctx, v)) return;
    qm3_fmm_set_vertex(ctx, v, dist, QM3_FMM_TRIAL, prev, prev_tri);
    qm3_heap_push(&ctx->heap, (Qm3HeapItem){v, dist});
}

static float qm3_sample_distance_from_prop(const Qm3PropGraph* graph, const Qm3SurfaceSample* sample, const Qm3FMMContext* ctx) {
    int begin = graph->tri_node_offsets[sample->tri];
    int end = graph->tri_node_offsets[sample->tri + 1];
    float best = INFINITY;
    for (int i = begin; i < end; ++i) {
        int node = graph->tri_node_ids[i];
        float d = qm3_fmm_get_dist(ctx, node);
        if (isfinite(d)) best = fminf(best, d + qm3_distance(sample->p, graph->nodes[node]));
    }
    return best;
}

static int qm3_geodesic_candidate_query(
    const Qm3Surface* surface,
    const Qm3PropGraph* graph,
    const Qm3SurfaceTopo* topo,
    Qm3Vec3 source_pos,
    int source_tri,
    float radius,
    Qm3FMMContext* ctx,
    uint32_t** out_candidates,
    uint32_t* out_count,
    uint32_t* out_cap
) {
    *out_count = 0;
    if (++ctx->query_id == 0) {
        memset(ctx->stamp, 0, (size_t)ctx->vertex_count * sizeof(unsigned int));
        memset(ctx->tri_stamp, 0, (size_t)ctx->tri_count * sizeof(unsigned int));
        ctx->query_id = 1;
    }
    ctx->heap.size = 0;
    ctx->reached_tri_count = 0;
    float stop_radius = radius + graph->spacing;

    Qm3SourceSupport source = qm3_source_support(surface, source_pos, source_tri, 1e-5f);
    if (source.source_vertex >= 0 && source.source_vertex < graph->node_count) {
        qm3_fmm_push_trial(ctx, source.source_vertex, 0.0f, -1, -1);
        for (int i = graph->node_tri_offsets[source.source_vertex]; i < graph->node_tri_offsets[source.source_vertex + 1]; ++i) {
            qm3_fmm_mark_triangle(ctx, graph->node_tri_ids[i]);
        }
    } else {
        if (source.tri_count == 0) return 0;
        for (int ti = 0; ti < source.tri_count; ++ti) {
            int tri = source.tris[ti];
            int begin = graph->tri_node_offsets[tri];
            int end = graph->tri_node_offsets[tri + 1];
            for (int i = begin; i < end; ++i) {
                int node = graph->tri_node_ids[i];
                qm3_fmm_push_trial(ctx, node, qm3_distance(source_pos, graph->nodes[node]), -1, tri);
            }
            qm3_fmm_mark_triangle(ctx, tri);
        }
    }

    while (ctx->heap.size > 0) {
        Qm3HeapItem item = qm3_heap_pop(&ctx->heap);
        if (item.dist != qm3_fmm_get_dist(ctx, item.node)) continue;
        if (item.dist > stop_radius) break;
        qm3_fmm_set_vertex(ctx, item.node, item.dist, QM3_FMM_ACCEPTED, ctx->prev[item.node], ctx->prev_tri[item.node]);
        for (int i = graph->node_tri_offsets[item.node]; i < graph->node_tri_offsets[item.node + 1]; ++i) qm3_fmm_mark_triangle(ctx, graph->node_tri_ids[i]);
        for (int e = graph->edge_offsets[item.node]; e < graph->edge_offsets[item.node + 1]; ++e) {
            int to = graph->edge_to[e];
            if (qm3_fmm_get_state(ctx, to) == QM3_FMM_ACCEPTED) continue;
            float nd = item.dist + graph->edge_weight[e];
            if (nd <= stop_radius) qm3_fmm_push_trial(ctx, to, nd, item.node, graph->edge_tri[e]);
        }
    }

    int in_radius = 0;
    float radius2 = radius * radius;
    for (int ti = 0; ti < ctx->reached_tri_count; ++ti) {
        int tri = ctx->reached_tris[ti];
        for (int si = topo->tri_sample_offsets[tri]; si < topo->tri_sample_offsets[tri + 1]; ++si) {
            uint32_t sample_id = (uint32_t)topo->tri_sample_ids[si];
            Qm3SurfaceSample* sample = &surface->samples[sample_id];
            Qm3Vec3 delta = qm3_sub(sample->p, source_pos);
            if (qm3_dot(delta, delta) > radius2) continue;
            float d = qm3_sample_distance_from_prop(graph, sample, ctx);
            if (!(d <= radius)) continue;
            if (*out_count == *out_cap) {
                *out_cap = *out_cap ? *out_cap * 2 : 256;
                *out_candidates = (uint32_t*)qm3_checked_realloc(*out_candidates, (size_t)*out_cap * sizeof(uint32_t));
            }
            (*out_candidates)[(*out_count)++] = sample_id;
            ++in_radius;
        }
    }
    return in_radius;
}

static int qm3_best_sample_prop_node(
    const Qm3Surface* surface,
    const Qm3PropGraph* graph,
    const Qm3FMMContext* ctx,
    uint32_t sample_id,
    float* out_distance
) {
    if (sample_id >= surface->sample_count) return -1;
    const Qm3SurfaceSample* sample = &surface->samples[sample_id];
    int begin = graph->tri_node_offsets[sample->tri];
    int end = graph->tri_node_offsets[sample->tri + 1];
    int best_node = -1;
    float best = INFINITY;
    for (int i = begin; i < end; ++i) {
        int node = graph->tri_node_ids[i];
        float d = qm3_fmm_get_dist(ctx, node);
        if (!isfinite(d)) continue;
        d += qm3_distance(sample->p, graph->nodes[node]);
        if (d < best) {
            best = d;
            best_node = node;
        }
    }
    if (best_node >= 0 && out_distance) *out_distance = best;
    return best_node;
}
