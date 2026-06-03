#pragma once

#include "memory.h"
#include "geometry.h"
#include "helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    Vec2 pos; 
    unsigned char degree;
    unsigned char open_edges;
    int frontier_index;
} MeshVertex;

DEFINE_VECTOR(MeshVertex, MeshVertexArray)

typedef struct {
    int a;
    int b;
    unsigned char face_count;
    bool face_orientation;
} MeshEdge;

DEFINE_VECTOR(MeshEdge, MeshEdgeArray)

typedef struct {
    MeshVertexArray vertices;
    MeshEdgeArray edges;
    IntArray neighbors;
    IntArray neighbor_edges;
    IntArray frontier;
    int max_degree;
} QuadMesh;

typedef enum {
    MESH_VALID_OK = 0,
    MESH_VALID_SAME_VERTEX,
    MESH_VALID_EDGE_EXISTS,
    MESH_VALID_DEGREE_FULL,
    MESH_VALID_INTERSECT,
    MESH_VALID_WEDGE_BLOCKED,
} MeshValidReason;

/** Allocates mesh buffers with capacity/degree limits. */
void mesh_init(QuadMesh* mesh, int max_degree) {
    MeshVertexArray_init(&mesh->vertices);
    MeshEdgeArray_init(&mesh->edges);
    IntArray_init(&mesh->neighbors);
    IntArray_init(&mesh->neighbor_edges);
    IntArray_init(&mesh->frontier);
    mesh->max_degree = max_degree;
}

/** Resets counters for a new episode. */
void mesh_reset(QuadMesh* mesh) {
    MeshVertexArray_resize(&mesh->vertices, 0);
    MeshEdgeArray_resize(&mesh->edges, 0);
    IntArray_resize(&mesh->neighbors, 0);
    IntArray_resize(&mesh->neighbor_edges, 0);
    IntArray_resize(&mesh->frontier, 0);
}

/** Frees mesh buffers. */
void mesh_free(QuadMesh* mesh) {
    MeshVertexArray_free(&mesh->vertices);
    MeshEdgeArray_free(&mesh->edges);
    IntArray_free(&mesh->neighbors);
    IntArray_free(&mesh->neighbor_edges);
    IntArray_free(&mesh->frontier);
}

static bool mesh_is_edge_canonical(const QuadMesh* mesh, int a, int b) {
    Vec2 pa = mesh->vertices.data[a].pos;
    Vec2 pb = mesh->vertices.data[b].pos;
    return (pa.y < pb.y || (pa.y == pb.y && pa.x <= pb.x));
}

static inline int mesh_neighbor_idx(const QuadMesh* mesh, int v, int slot) {
    return v * mesh->max_degree + slot;
}

/** Sets a boundary edge face side based on polygon orientation. */
void mesh_set_boundary_edge_face(QuadMesh* mesh, int edge_idx, int a, int b, bool boundary_ccw) {
    MeshEdge* e = &mesh->edges.data[edge_idx];
    e->face_orientation = (a == e->a) == boundary_ccw;
    e->face_count = 1;
}

static bool mesh_edge_face_orientation_from_vertex(const QuadMesh* mesh, int edge_idx, int source) {
    MeshEdge e = mesh->edges.data[edge_idx];
    if (source == e.a) return e.face_orientation;
    if (source == e.b) return !e.face_orientation;
    QM_ASSERT(!"source must be an endpoint of edge");
    return e.face_orientation;
}

/** Returns edge index or -1 if missing. */
int mesh_edge_index(const QuadMesh* mesh, int a, int b) {
    for (int i = 0; i < mesh->vertices.data[a].degree; i++) {
        if (mesh->neighbors.data[mesh_neighbor_idx(mesh, a, i)] == b) {
            return mesh->neighbor_edges.data[mesh_neighbor_idx(mesh, a, i)];
        }
    }
    return -1;
}

static MeshValidReason mesh_validate_edge(const QuadMesh* mesh, int source, Vec2 target_pos, int target_idx) {
    Vec2 sp = mesh->vertices.data[source].pos;
    for (int i = 0; i < mesh->edges.size; i++) {
        MeshEdge e = mesh->edges.data[i];
        if (e.a == source || e.b == source || e.a == target_idx || e.b == target_idx) continue;
        Vec2 q1 = mesh->vertices.data[e.a].pos;
        Vec2 q2 = mesh->vertices.data[e.b].pos;
        if (segments_intersect(sp, target_pos, q1, q2)) return MESH_VALID_INTERSECT;
    }
    Vec2 tvec = sub2(target_pos, sp);
    float best_angle = 1e9f;
    int best_edge = -1;
    for (int i = 0; i < mesh->vertices.data[source].degree; i++) {
        int nidx = mesh_neighbor_idx(mesh, source, i);
        int nvidx = mesh->neighbors.data[nidx];
        int eidx = mesh->neighbor_edges.data[nidx];
        if (mesh->edges.data[eidx].face_count != 1) continue;
        Vec2 np = mesh->vertices.data[nvidx].pos;
        Vec2 evec = sub2(np, sp);
        float ang = atan2f(cross2(evec, tvec), dot2(evec, tvec));
        if (ang <= 0.0f) ang += 2.0f * M_PI;
        if (ang < best_angle) {
            best_angle = ang;
            best_edge = eidx;
        }
    }
    if (best_edge >= 0) {
        int ccw_face = mesh_edge_face_orientation_from_vertex(mesh, best_edge, source);
        if (!ccw_face) return MESH_VALID_WEDGE_BLOCKED;
    }
    return MESH_VALID_OK;
}

/** Adds a vertex and returns its index. */
int mesh_add_vertex(QuadMesh* mesh, Vec2 p) {
    int idx = mesh->vertices.size;
    MeshVertexArray_push(&mesh->vertices, (MeshVertex){.pos = p, .degree = 0, .open_edges = 0, .frontier_index = -1});
    // Push is smart about memory reallocation
    for (int i=0; i<mesh->max_degree; ++i) {
        IntArray_push(&mesh->neighbors, -1);
        IntArray_push(&mesh->neighbor_edges, -1);
    }
    return idx;
}

/** Returns 1 if vertex has open edges and degree < max_degree. */
int mesh_vertex_is_frontier(const QuadMesh* mesh, int v) {
    return (mesh->vertices.data[v].open_edges > 0 && mesh->vertices.data[v].degree < mesh->max_degree) || mesh->vertices.data[v].degree == 0;
}

/** Rebuilds the frontier locally after a vertex might have changed status. */
void mesh_update_frontier_vertex(const QuadMesh* mesh, int v) {
    // If vertex became a frontier member, add it
    if (mesh->vertices.data[v].frontier_index < 0 && mesh_vertex_is_frontier(mesh, v)) {
        mesh->vertices.data[v].frontier_index = mesh->frontier.size;
        IntArray_push(&mesh->frontier, v);
    }
    // If vertex should no longer be a frontier member, remove it
    else if (mesh->vertices.data[v].frontier_index >= 0 && !mesh_vertex_is_frontier(mesh, v)) {
        int swapped = mesh->frontier.data[mesh->frontier.size - 1];
        IntArray_remove_swap(&mesh->frontier, mesh->vertices.data[v].frontier_index);
        // Swap indices as well
        mesh->vertices.data[swapped].frontier_index = mesh->vertices.data[v].frontier_index;
        mesh->vertices.data[v].frontier_index = -1;
    }
    // Else, do nothing
}

/** Returns 1 if edge exists. */
int mesh_edge_exists(const QuadMesh* mesh, int a, int b) {
    return mesh_edge_index(mesh, a, b) >= 0;
}

/** Adds an edge and returns its index. */
int mesh_add_edge(QuadMesh* mesh, int a, int b) {
    QM_ASSERT(a != b);
    // Make sure edge is stored in canonical direction
    if (!mesh_is_edge_canonical(mesh, a, b)) {
        int tmp = a;
        a = b;
        b = tmp;
    }

    // If edge exists, return existing index
    if (mesh_edge_exists(mesh, a, b)) return mesh_edge_index(mesh, a, b);

    // Add edge
    int idx = mesh->edges.size;
    MeshEdgeArray_push(&mesh->edges, (MeshEdge){.a = a, .b = b, .face_count = 0, .face_orientation = 0});
    mesh->vertices.data[a].open_edges += 1;
    mesh->vertices.data[b].open_edges += 1;
    QM_ASSERT(mesh->vertices.data[a].degree < mesh->max_degree);
    QM_ASSERT(mesh->vertices.data[b].degree < mesh->max_degree);
    int adeg = mesh->vertices.data[a].degree++;
    size_t anidx = mesh_neighbor_idx(mesh, a, adeg);
    mesh->neighbors.data[anidx] = b;
    mesh->neighbor_edges.data[anidx] = idx;
    int bdeg = mesh->vertices.data[b].degree++;
    size_t bnidx = mesh_neighbor_idx(mesh, b, bdeg);
    mesh->neighbors.data[bnidx] = a;
    mesh->neighbor_edges.data[bnidx] = idx;

    // Update frontier
    mesh_update_frontier_vertex(mesh, a);
    mesh_update_frontier_vertex(mesh, b);

    return idx;
}

/** Detects all 3-cycles involving edge (u,v). */
int mesh_detect_triangles(
    const QuadMesh* mesh, int u, int v, int* out_cycles, int max_cycles
) {
    int count = 0;
    int uv = mesh_edge_index(mesh, u, v);
    if (uv < 0 || mesh->edges.data[uv].face_count >= 2) return count;
    for (int i = 0; i < mesh->vertices.data[u].degree; i++) {
        int a = mesh->neighbors.data[mesh_neighbor_idx(mesh, u, i)];
        if (a == v) continue;
        int ua = mesh_edge_index(mesh, u, a);
        int av = mesh_edge_index(mesh, a, v);
        if (ua < 0 || av < 0) continue;
        if (mesh->edges.data[ua].face_count >= 2 ||
            mesh->edges.data[av].face_count >= 2) {
            continue;
        }
        if (count < max_cycles) {
            out_cycles[count * 3 + 0] = u;
            out_cycles[count * 3 + 1] = a;
            out_cycles[count * 3 + 2] = v;
            count++;
        }
    }
    return count;
}

/** Detects all 4-cycles involving edge (u,v). */
int mesh_detect_quads(
    const QuadMesh* mesh, int u, int v, int* out_cycles, int max_cycles
) {
    int count = 0;
    int uv = mesh_edge_index(mesh, u, v);
    if (uv < 0 || mesh->edges.data[uv].face_count >= 2) return count;
    for (int i = 0; i < mesh->vertices.data[u].degree; i++) {
        int a = mesh->neighbors.data[mesh_neighbor_idx(mesh, u, i)];
        if (a == v) continue;
        for (int j = 0; j < mesh->vertices.data[v].degree; j++) {
            int b = mesh->neighbors.data[mesh_neighbor_idx(mesh, v, j)];
            if (b == u || b == a) continue;
            int ua = mesh_edge_index(mesh, u, a);
            int vb = mesh_edge_index(mesh, v, b);
            int ab = mesh_edge_index(mesh, a, b);
            if (ua < 0 || vb < 0 || ab < 0) continue;
            if (mesh->edges.data[ua].face_count >= 2 ||
                mesh->edges.data[vb].face_count >= 2 ||
                mesh->edges.data[ab].face_count >= 2) {
                continue;
            }
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

/** Registers a face cycle and updates edge face counts/sides. */
bool mesh_register_face(QuadMesh* mesh, const int* verts, int n) {
    QM_ASSERT(n <= 4);

    Vec2 face[4];
    for (int i = 0; i < n; ++i)
        face[i] = mesh->vertices.data[verts[i]].pos;
    bool face_ccw = polygon_is_ccw(face, n);

    MeshEdge* edges[4];
    int face_orientations[4];
    for (int i = 0; i < n; ++i) {
        int u = verts[i];
        int v = verts[(i + 1) % n];
        int eidx = mesh_edge_index(mesh, u, v);
        QM_ASSERT(eidx != -1);
        MeshEdge* e = &mesh->edges.data[eidx];
        if (e->face_count == 2) return false;
        edges[i] = e;
        face_orientations[i] = (u == e->a) != face_ccw;
        // Return if edge is already part of a face on the same side
        if (e->face_count == 1 && e->face_orientation == face_orientations[i]) return false;
    }

    for (int i = 0; i < n; ++i) {
        MeshEdge* e = edges[i];
        ++e->face_count;
        if (e->face_count == 1) {
            e->face_orientation = face_orientations[i];
        } else if (e->face_count == 2) {
            mesh->vertices.data[e->a].open_edges -= 1;
            mesh->vertices.data[e->b].open_edges -= 1;
        }
    }
    
    // Local frontier update
    for (int i = 0; i < n; ++i) mesh_update_frontier_vertex(mesh, verts[i]);

    return true;
}


/** Validates an existing target; returns reason. */
MeshValidReason mesh_validate_existing_target(const QuadMesh* mesh, int source, int target) {
    if (target == source) return MESH_VALID_SAME_VERTEX;
    if (mesh_edge_exists(mesh, source, target)) return MESH_VALID_EDGE_EXISTS;
    if (mesh->vertices.data[source].degree >= mesh->max_degree) return MESH_VALID_DEGREE_FULL;
    if (mesh->vertices.data[target].degree >= mesh->max_degree) return MESH_VALID_DEGREE_FULL;
    Vec2 tp = mesh->vertices.data[target].pos;
    return mesh_validate_edge(mesh, source, tp, target);
}

/** Validates a candidate target position; returns reason. */
MeshValidReason mesh_validate_candidate_target(const QuadMesh* mesh, int source, Vec2 target_pos) {
    if (mesh->vertices.data[source].degree >= mesh->max_degree) return MESH_VALID_DEGREE_FULL;
    return mesh_validate_edge(mesh, source, target_pos, -1);
}

/** Returns a reason string for debugging. */
const char* mesh_valid_reason_str(MeshValidReason r) {
    switch (r) {
        case MESH_VALID_OK: return "valid";
        case MESH_VALID_SAME_VERTEX: return "same-vertex";
        case MESH_VALID_EDGE_EXISTS: return "edge-exists";
        case MESH_VALID_DEGREE_FULL: return "degree-full";
        case MESH_VALID_INTERSECT: return "edge-intersection";
        case MESH_VALID_WEDGE_BLOCKED: return "wedge-blocked";
        default: return "invalid";
    }
}

