#pragma once

#include "qmsurface.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void* qm3_checked_realloc(void* ptr, size_t bytes) {
    void* out = realloc(ptr, bytes);
    QM3_ASSERT(out != NULL || bytes == 0);
    return out;
}

static float qm3_distance(Qm3Vec3 a, Qm3Vec3 b) {
    return qm3_len(qm3_sub(a, b));
}

typedef struct {
    Qm3Vec3* points;
    uint32_t count;
    uint32_t cap;
    float length;
} Qm3Path;

typedef struct {
    int tri;
    Qm3Vec3 a;
    Qm3Vec3 b;
} Qm3PathSegment;

typedef struct {
    Qm3PathSegment* data;
    uint32_t count;
    uint32_t cap;
} Qm3PathSegmentArray;

static void qm3_path_clear(Qm3Path* path) {
    path->count = 0;
    path->length = 0.0f;
}

static void qm3_path_free(Qm3Path* path) {
    free(path->points);
    memset(path, 0, sizeof(*path));
}

static void qm3_path_push(Qm3Path* path, Qm3Vec3 p) {
    if (path->count > 0 && qm3_distance(path->points[path->count - 1], p) <= 1e-7f) return;
    if (path->count == path->cap) {
        path->cap = path->cap ? path->cap * 2 : 64;
        path->points = (Qm3Vec3*)qm3_checked_realloc(path->points, (size_t)path->cap * sizeof(Qm3Vec3));
    }
    if (path->count > 0) path->length += qm3_distance(path->points[path->count - 1], p);
    path->points[path->count++] = p;
}

static void qm3_path_push_raw(Qm3Path* path, Qm3Vec3 p) {
    if (path->count == path->cap) {
        path->cap = path->cap ? path->cap * 2 : 64;
        path->points = (Qm3Vec3*)qm3_checked_realloc(path->points, (size_t)path->cap * sizeof(Qm3Vec3));
    }
    path->points[path->count++] = p;
}

static void qm3_path_segment_clear(Qm3PathSegmentArray* a) { a->count = 0; }

static void qm3_path_segment_free(Qm3PathSegmentArray* a) { free(a->data); memset(a, 0, sizeof(*a)); }

static void qm3_path_segment_push(Qm3PathSegmentArray* a, Qm3PathSegment v) {
    if (a->count == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 64;
        a->data = (Qm3PathSegment*)qm3_checked_realloc(a->data, (size_t)a->cap * sizeof(Qm3PathSegment));
    }
    a->data[a->count++] = v;
}

typedef struct {
    Qm3Vec3 pos;
    Qm3Vec3 normal;
    uint32_t sample_id;
    uint32_t surface_vertex;
    int32_t surface_tri;
    uint16_t degree;
    uint16_t open_edges;
    int32_t frontier_index;
    bool disabled;
} Qm3MeshVertex;

typedef struct {
    uint32_t a;
    uint32_t b;
    uint32_t path_offset;
    uint32_t path_count;
    uint32_t segment_offset;
    uint32_t segment_count;
    float path_length;
    uint8_t face_count;
    bool disabled;
} Qm3MeshEdge;

typedef struct {
    Qm3MeshVertex* vertices;
    Qm3MeshEdge* edges;
    int32_t* neighbors;
    int32_t* neighbor_edges;
    uint32_t* frontier;
    Qm3Path edge_path_points;
    Qm3PathSegmentArray edge_path_segments;
    uint32_t vertex_count;
    uint32_t vertex_cap;
    uint32_t edge_count;
    uint32_t edge_cap;
    uint32_t frontier_count;
    uint32_t frontier_cap;
    uint32_t max_degree;
} Qm3Mesh;

static void qm3_mesh_init(Qm3Mesh* mesh, uint32_t max_degree) {
    memset(mesh, 0, sizeof(*mesh));
    mesh->max_degree = max_degree > 0 ? max_degree : 16;
}

static void qm3_mesh_reset(Qm3Mesh* mesh) {
    mesh->vertex_count = 0;
    mesh->edge_count = 0;
    mesh->frontier_count = 0;
    qm3_path_clear(&mesh->edge_path_points);
    qm3_path_segment_clear(&mesh->edge_path_segments);
}

static void qm3_mesh_free(Qm3Mesh* mesh) {
    free(mesh->vertices);
    free(mesh->edges);
    free(mesh->neighbors);
    free(mesh->neighbor_edges);
    free(mesh->frontier);
    qm3_path_free(&mesh->edge_path_points);
    qm3_path_segment_free(&mesh->edge_path_segments);
    qm3_mesh_init(mesh, mesh->max_degree);
}

static void qm3_mesh_reserve_vertices(Qm3Mesh* mesh, uint32_t cap) {
    if (cap <= mesh->vertex_cap) return;
    mesh->vertices = (Qm3MeshVertex*)qm3_checked_realloc(mesh->vertices, (size_t)cap * sizeof(Qm3MeshVertex));
    mesh->neighbors = (int32_t*)qm3_checked_realloc(mesh->neighbors, (size_t)cap * mesh->max_degree * sizeof(int32_t));
    mesh->neighbor_edges = (int32_t*)qm3_checked_realloc(mesh->neighbor_edges, (size_t)cap * mesh->max_degree * sizeof(int32_t));
    for (uint32_t v = mesh->vertex_cap; v < cap; ++v) {
        for (uint32_t i = 0; i < mesh->max_degree; ++i) {
            mesh->neighbors[(size_t)v * mesh->max_degree + i] = -1;
            mesh->neighbor_edges[(size_t)v * mesh->max_degree + i] = -1;
        }
    }
    mesh->vertex_cap = cap;
}

static void qm3_mesh_reserve_edges(Qm3Mesh* mesh, uint32_t cap) {
    if (cap <= mesh->edge_cap) return;
    mesh->edges = (Qm3MeshEdge*)qm3_checked_realloc(mesh->edges, (size_t)cap * sizeof(Qm3MeshEdge));
    mesh->edge_cap = cap;
}

static void qm3_mesh_reserve_frontier(Qm3Mesh* mesh, uint32_t cap) {
    if (cap <= mesh->frontier_cap) return;
    mesh->frontier = (uint32_t*)qm3_checked_realloc(mesh->frontier, (size_t)cap * sizeof(uint32_t));
    mesh->frontier_cap = cap;
}

static bool qm3_mesh_vertex_is_frontier(const Qm3Mesh* mesh, uint32_t v) {
    const Qm3MeshVertex* vertex = &mesh->vertices[v];
    return !vertex->disabled && vertex->open_edges > 0 && vertex->degree < mesh->max_degree;
}

static void qm3_mesh_update_frontier_vertex(Qm3Mesh* mesh, uint32_t v) {
    Qm3MeshVertex* vertex = &mesh->vertices[v];
    if (vertex->frontier_index < 0 && qm3_mesh_vertex_is_frontier(mesh, v)) {
        qm3_mesh_reserve_frontier(mesh, mesh->frontier_count + 1);
        vertex->frontier_index = (int32_t)mesh->frontier_count;
        mesh->frontier[mesh->frontier_count++] = v;
    } else if (vertex->frontier_index >= 0 && !qm3_mesh_vertex_is_frontier(mesh, v)) {
        uint32_t remove = (uint32_t)vertex->frontier_index;
        uint32_t swapped = mesh->frontier[mesh->frontier_count - 1];
        mesh->frontier[remove] = swapped;
        mesh->vertices[swapped].frontier_index = (int32_t)remove;
        mesh->frontier_count--;
        vertex->frontier_index = -1;
    }
}

static uint32_t qm3_mesh_add_vertex(Qm3Mesh* mesh, Qm3Vec3 pos, Qm3Vec3 normal, uint32_t sample_id, uint32_t surface_vertex, int32_t surface_tri) {
    if (mesh->vertex_count == mesh->vertex_cap) {
        uint32_t new_cap = mesh->vertex_cap ? mesh->vertex_cap * 2 : 64;
        qm3_mesh_reserve_vertices(mesh, new_cap);
    }
    uint32_t idx = mesh->vertex_count++;
    mesh->vertices[idx] = (Qm3MeshVertex){
        .pos = pos,
        .normal = normal,
        .sample_id = sample_id,
        .surface_vertex = surface_vertex,
        .surface_tri = surface_tri,
        .degree = 0,
        .open_edges = 0,
        .frontier_index = -1,
        .disabled = false,
    };
    for (uint32_t i = 0; i < mesh->max_degree; ++i) {
        mesh->neighbors[(size_t)idx * mesh->max_degree + i] = -1;
        mesh->neighbor_edges[(size_t)idx * mesh->max_degree + i] = -1;
    }
    return idx;
}

static int32_t qm3_mesh_edge_index(const Qm3Mesh* mesh, uint32_t a, uint32_t b) {
    if (a >= mesh->vertex_count || b >= mesh->vertex_count) return -1;
    const Qm3MeshVertex* va = &mesh->vertices[a];
    for (uint32_t i = 0; i < va->degree; ++i) {
        size_t nidx = (size_t)a * mesh->max_degree + i;
        int32_t eidx = mesh->neighbor_edges[nidx];
        if (eidx < 0 || mesh->edges[eidx].disabled) continue;
        if ((uint32_t)mesh->neighbors[nidx] == b) return eidx;
    }
    return -1;
}

static uint32_t qm3_mesh_add_edge_with_path(
    Qm3Mesh* mesh,
    uint32_t a,
    uint32_t b,
    uint8_t face_count,
    const Qm3Vec3* path_points,
    uint32_t path_count,
    const Qm3PathSegment* path_segments,
    uint32_t segment_count,
    float path_length
) {
    QM3_ASSERT(a != b);
    int32_t existing = qm3_mesh_edge_index(mesh, a, b);
    if (existing >= 0) return (uint32_t)existing;

    QM3_ASSERT(mesh->vertices[a].degree < mesh->max_degree);
    QM3_ASSERT(mesh->vertices[b].degree < mesh->max_degree);
    if (mesh->edge_count == mesh->edge_cap) {
        uint32_t new_cap = mesh->edge_cap ? mesh->edge_cap * 2 : 64;
        qm3_mesh_reserve_edges(mesh, new_cap);
    }

    uint32_t eidx = mesh->edge_count++;
    uint32_t path_offset = mesh->edge_path_points.count;
    for (uint32_t i = 0; i < path_count; ++i) qm3_path_push_raw(&mesh->edge_path_points, path_points[i]);
    uint32_t stored_path_count = mesh->edge_path_points.count - path_offset;
    uint32_t segment_offset = mesh->edge_path_segments.count;
    for (uint32_t i = 0; i < segment_count; ++i) qm3_path_segment_push(&mesh->edge_path_segments, path_segments[i]);
    mesh->edges[eidx] = (Qm3MeshEdge){
        .a = a,
        .b = b,
        .path_offset = path_offset,
        .path_count = stored_path_count,
        .segment_offset = segment_offset,
        .segment_count = mesh->edge_path_segments.count - segment_offset,
        .path_length = path_length,
        .face_count = face_count,
        .disabled = false,
    };
    if (face_count < 2) {
        mesh->vertices[a].open_edges++;
        mesh->vertices[b].open_edges++;
    }

    uint32_t adeg = mesh->vertices[a].degree++;
    uint32_t bdeg = mesh->vertices[b].degree++;
    size_t anidx = (size_t)a * mesh->max_degree + adeg;
    size_t bnidx = (size_t)b * mesh->max_degree + bdeg;
    mesh->neighbors[anidx] = (int32_t)b;
    mesh->neighbor_edges[anidx] = (int32_t)eidx;
    mesh->neighbors[bnidx] = (int32_t)a;
    mesh->neighbor_edges[bnidx] = (int32_t)eidx;

    qm3_mesh_update_frontier_vertex(mesh, a);
    qm3_mesh_update_frontier_vertex(mesh, b);
    return eidx;
}

static uint32_t qm3_mesh_vertex_for_sample(
    Qm3Mesh* mesh,
    const Qm3Surface* surface,
    int32_t* sample_to_graph,
    uint32_t sample_id
) {
    QM3_ASSERT(sample_id < surface->sample_count);
    if (sample_to_graph[sample_id] >= 0) return (uint32_t)sample_to_graph[sample_id];

    const Qm3SurfaceSample* sample = &surface->samples[sample_id];
    uint32_t graph_vertex = qm3_mesh_add_vertex(
        mesh,
        sample->p,
        sample->n,
        sample_id,
        UINT32_MAX,
        (int32_t)sample->tri
    );
    sample_to_graph[sample_id] = (int32_t)graph_vertex;
    return graph_vertex;
}

static void qm3_mesh_build_from_frontier_edges(Qm3Mesh* mesh, const Qm3Surface* surface) {
    qm3_mesh_reset(mesh);
    qm3_mesh_reserve_edges(mesh, surface->frontier_edge_count);
    qm3_mesh_reserve_vertices(mesh, surface->frontier_edge_count * 2);
    qm3_mesh_reserve_frontier(mesh, surface->frontier_edge_count * 2);

    int32_t* sample_to_graph = (int32_t*)malloc((size_t)surface->sample_count * sizeof(int32_t));
    QM3_ASSERT(sample_to_graph != NULL || surface->sample_count == 0);
    for (uint32_t i = 0; i < surface->sample_count; ++i) sample_to_graph[i] = -1;

    for (uint32_t i = 0; i < surface->frontier_edge_count; ++i) {
        uint32_t sample_a = surface->frontier_edges[2u * i];
        uint32_t sample_b = surface->frontier_edges[2u * i + 1u];
        uint32_t a = qm3_mesh_vertex_for_sample(mesh, surface, sample_to_graph, sample_a);
        uint32_t b = qm3_mesh_vertex_for_sample(mesh, surface, sample_to_graph, sample_b);
        Qm3Vec3 points[2] = {surface->samples[sample_a].p, surface->samples[sample_b].p};
        Qm3PathSegment segments[2];
        uint32_t segment_count = 0;
        int tri_a = (int)surface->samples[sample_a].tri;
        int tri_b = (int)surface->samples[sample_b].tri;
        if (tri_a >= 0) segments[segment_count++] = (Qm3PathSegment){.tri = tri_a, .a = points[0], .b = points[1]};
        if (tri_b >= 0 && tri_b != tri_a) segments[segment_count++] = (Qm3PathSegment){.tri = tri_b, .a = points[0], .b = points[1]};
        qm3_mesh_add_edge_with_path(mesh, a, b, 1, points, 2, segments, segment_count, qm3_distance(points[0], points[1]));
    }

    free(sample_to_graph);
}
