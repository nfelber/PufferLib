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
    uint8_t n;
    uint32_t vertices[4];
    uint32_t edges[4];
    float area;
    float quality;
    bool disabled;
} Qm3MeshFace;

typedef struct {
    Qm3MeshVertex* vertices;
    Qm3MeshEdge* edges;
    Qm3MeshFace* faces;
    int32_t* neighbors;
    int32_t* neighbor_edges;
    uint32_t* frontier;
    Qm3Path edge_path_points;
    Qm3PathSegmentArray edge_path_segments;
    uint32_t vertex_count;
    uint32_t vertex_cap;
    uint32_t edge_count;
    uint32_t edge_cap;
    uint32_t face_count;
    uint32_t face_cap;
    uint32_t quad_count;
    uint32_t tri_count;
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
    mesh->face_count = 0;
    mesh->quad_count = 0;
    mesh->tri_count = 0;
    mesh->frontier_count = 0;
    qm3_path_clear(&mesh->edge_path_points);
    qm3_path_segment_clear(&mesh->edge_path_segments);
}

static void qm3_mesh_free(Qm3Mesh* mesh) {
    free(mesh->vertices);
    free(mesh->edges);
    free(mesh->faces);
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

static void qm3_mesh_reserve_faces(Qm3Mesh* mesh, uint32_t cap) {
    if (cap <= mesh->face_cap) return;
    mesh->faces = (Qm3MeshFace*)qm3_checked_realloc(mesh->faces, (size_t)cap * sizeof(Qm3MeshFace));
    mesh->face_cap = cap;
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

static float qm3_mesh_face_area(const Qm3Mesh* mesh, const uint32_t* verts, uint32_t n) {
    if (n < 3) return 0.0f;
    Qm3Vec3 p0 = mesh->vertices[verts[0]].pos;
    float area = 0.0f;
    for (uint32_t i = 1; i + 1 < n; ++i) {
        Qm3Vec3 a = qm3_sub(mesh->vertices[verts[i]].pos, p0);
        Qm3Vec3 b = qm3_sub(mesh->vertices[verts[i + 1]].pos, p0);
        Qm3Vec3 c = (Qm3Vec3){
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        };
        area += 0.5f * qm3_len(c);
    }
    return area;
}

static bool qm3_face_has_same_vertices(const Qm3MeshFace* face, const uint32_t* verts, uint32_t n) {
    if (face->disabled || face->n != n) return false;
    for (uint32_t i = 0; i < n; ++i) {
        bool found = false;
        for (uint32_t j = 0; j < n; ++j) {
            if (face->vertices[j] == verts[i]) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

static bool qm3_mesh_register_face(Qm3Mesh* mesh, const uint32_t* verts, uint32_t n) {
    QM3_ASSERT(n == 3 || n == 4);
    for (uint32_t i = 0; i < n; ++i) {
        if (verts[i] >= mesh->vertex_count || mesh->vertices[verts[i]].disabled) return false;
        for (uint32_t j = i + 1; j < n; ++j) if (verts[i] == verts[j]) return false;
    }
    for (uint32_t fi = 0; fi < mesh->face_count; ++fi) {
        if (qm3_face_has_same_vertices(&mesh->faces[fi], verts, n)) return false;
    }

    uint32_t edges[4];
    for (uint32_t i = 0; i < n; ++i) {
        int32_t eidx = qm3_mesh_edge_index(mesh, verts[i], verts[(i + 1) % n]);
        if (eidx < 0) return false;
        Qm3MeshEdge* edge = &mesh->edges[eidx];
        if (edge->disabled || edge->face_count >= 2) return false;
        edges[i] = (uint32_t)eidx;
    }

    if (mesh->face_count == mesh->face_cap) {
        uint32_t new_cap = mesh->face_cap ? mesh->face_cap * 2 : 64;
        qm3_mesh_reserve_faces(mesh, new_cap);
    }

    Qm3MeshFace* face = &mesh->faces[mesh->face_count++];
    memset(face, 0, sizeof(*face));
    face->n = (uint8_t)n;
    face->area = qm3_mesh_face_area(mesh, verts, n);
    face->quality = 0.0f;
    face->disabled = false;
    for (uint32_t i = 0; i < n; ++i) {
        face->vertices[i] = verts[i];
        face->edges[i] = edges[i];
        Qm3MeshEdge* edge = &mesh->edges[edges[i]];
        edge->face_count++;
        if (edge->face_count == 2) {
            QM3_ASSERT(mesh->vertices[edge->a].open_edges > 0);
            QM3_ASSERT(mesh->vertices[edge->b].open_edges > 0);
            mesh->vertices[edge->a].open_edges--;
            mesh->vertices[edge->b].open_edges--;
        }
    }
    if (n == 4) mesh->quad_count++;
    else mesh->tri_count++;

    for (uint32_t i = 0; i < n; ++i) qm3_mesh_update_frontier_vertex(mesh, verts[i]);
    return true;
}

static bool qm3_mesh_edge_is_face_boundary(const Qm3MeshFace* face, uint32_t eidx) {
    if (!face || face->disabled) return false;
    for (uint32_t i = 0; i < face->n; ++i) if (face->edges[i] == eidx) return true;
    return false;
}

static bool qm3_mesh_vertex_is_face_boundary(const Qm3MeshFace* face, uint32_t vidx) {
    if (!face || face->disabled) return false;
    for (uint32_t i = 0; i < face->n; ++i) if (face->vertices[i] == vidx) return true;
    return false;
}

static void qm3_mesh_remove_neighbor_edge(Qm3Mesh* mesh, uint32_t v, uint32_t eidx) {
    Qm3MeshVertex* vertex = &mesh->vertices[v];
    for (uint32_t i = 0; i < vertex->degree; ++i) {
        size_t nidx = (size_t)v * mesh->max_degree + i;
        if (mesh->neighbor_edges[nidx] != (int32_t)eidx) continue;
        uint32_t last = vertex->degree - 1;
        size_t last_idx = (size_t)v * mesh->max_degree + last;
        mesh->neighbors[nidx] = mesh->neighbors[last_idx];
        mesh->neighbor_edges[nidx] = mesh->neighbor_edges[last_idx];
        mesh->neighbors[last_idx] = -1;
        mesh->neighbor_edges[last_idx] = -1;
        vertex->degree--;
        return;
    }
}

static void qm3_mesh_disable_edge(Qm3Mesh* mesh, uint32_t eidx) {
    if (eidx >= mesh->edge_count) return;
    Qm3MeshEdge* edge = &mesh->edges[eidx];
    if (edge->disabled) return;
    uint32_t a = edge->a;
    uint32_t b = edge->b;
    if (edge->face_count < 2) {
        if (mesh->vertices[a].open_edges > 0) mesh->vertices[a].open_edges--;
        if (mesh->vertices[b].open_edges > 0) mesh->vertices[b].open_edges--;
    }
    qm3_mesh_remove_neighbor_edge(mesh, a, eidx);
    qm3_mesh_remove_neighbor_edge(mesh, b, eidx);
    edge->disabled = true;
    if (mesh->vertices[a].degree == 0) mesh->vertices[a].disabled = true;
    if (mesh->vertices[b].degree == 0) mesh->vertices[b].disabled = true;
    qm3_mesh_update_frontier_vertex(mesh, a);
    qm3_mesh_update_frontier_vertex(mesh, b);
}

static uint32_t qm3_mesh_detect_triangles(const Qm3Mesh* mesh, uint32_t u, uint32_t v, uint32_t* out_cycles, uint32_t max_cycles) {
    uint32_t count = 0;
    int32_t uv = qm3_mesh_edge_index(mesh, u, v);
    if (uv < 0 || mesh->edges[uv].face_count >= 2) return count;
    for (uint32_t i = 0; i < mesh->vertices[u].degree; ++i) {
        uint32_t a = (uint32_t)mesh->neighbors[(size_t)u * mesh->max_degree + i];
        if (a == v) continue;
        int32_t ua = qm3_mesh_edge_index(mesh, u, a);
        int32_t av = qm3_mesh_edge_index(mesh, a, v);
        if (ua < 0 || av < 0) continue;
        if (mesh->edges[ua].face_count >= 2 || mesh->edges[av].face_count >= 2) continue;
        if (count < max_cycles) {
            out_cycles[count * 3 + 0] = u;
            out_cycles[count * 3 + 1] = a;
            out_cycles[count * 3 + 2] = v;
            count++;
        }
    }
    return count;
}

static uint32_t qm3_mesh_detect_quads(const Qm3Mesh* mesh, uint32_t u, uint32_t v, uint32_t* out_cycles, uint32_t max_cycles) {
    uint32_t count = 0;
    int32_t uv = qm3_mesh_edge_index(mesh, u, v);
    if (uv < 0 || mesh->edges[uv].face_count >= 2) return count;
    for (uint32_t i = 0; i < mesh->vertices[u].degree; ++i) {
        uint32_t a = (uint32_t)mesh->neighbors[(size_t)u * mesh->max_degree + i];
        if (a == v) continue;
        for (uint32_t j = 0; j < mesh->vertices[v].degree; ++j) {
            uint32_t b = (uint32_t)mesh->neighbors[(size_t)v * mesh->max_degree + j];
            if (b == u || b == a) continue;
            int32_t ua = qm3_mesh_edge_index(mesh, u, a);
            int32_t vb = qm3_mesh_edge_index(mesh, v, b);
            int32_t ab = qm3_mesh_edge_index(mesh, a, b);
            if (ua < 0 || vb < 0 || ab < 0) continue;
            if (mesh->edges[ua].face_count >= 2 || mesh->edges[vb].face_count >= 2 || mesh->edges[ab].face_count >= 2) continue;
            if (count < max_cycles) {
                out_cycles[count * 4 + 0] = u;
                out_cycles[count * 4 + 1] = a;
                out_cycles[count * 4 + 2] = b;
                out_cycles[count * 4 + 3] = v;
                count++;
            }
        }
    }
    return count;
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
        qm3_mesh_add_edge_with_path(mesh, a, b, 0, points, 2, segments, segment_count, qm3_distance(points[0], points[1]));
    }

    free(sample_to_graph);
}
