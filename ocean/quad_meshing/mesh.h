#pragma once

#include "memory.h"
#include "geometry.h"
#include "helpers.h"
#include "ugrid.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct {
    Vec2 pos; 
    unsigned char degree;
    unsigned char open_edges;
    int frontier_index;
    float frontier_quality;
    bool frontier_quality_dirty;
    bool disabled;
} MeshVertex;

DEFINE_VECTOR(MeshVertex, MeshVertexArray)

typedef struct {
    int a;
    int b;
    unsigned char face_count;
    bool face_orientation_cw;
    bool disabled;
} MeshEdge;

DEFINE_VECTOR(MeshEdge, MeshEdgeArray)

typedef struct {
    MeshVertexArray vertices;
    MeshEdgeArray edges;
    IntArray neighbors;
    IntArray neighbor_edges;
    IntArray frontier;
    int max_degree;
    UGrid edge_grid;
    UGridCellIteratorArray grid_it;
    float intersection_tol;
} QuadMesh;

typedef enum {
    MESH_VALID_OK = 0,
    MESH_VALID_SAME_VERTEX,
    MESH_VALID_EDGE_EXISTS,
    MESH_VALID_DEGREE_FULL,
    MESH_VALID_INTERSECT,
    MESH_VALID_WEDGE_BLOCKED,
    MESH_VALID_BOUNDARY_VIOLATION,
    MESH_VALID_TOO_FAR,
} MeshValidReason;

/** Allocates mesh buffers with capacity/degree limits. */
void mesh_init(
    QuadMesh* mesh,
    int max_degree,
    unsigned int grid_res,
    float grid_cell_size,
    unsigned int grid_cell_cap,
    float intersection_tol
) {
    MeshVertexArray_init(&mesh->vertices);
    MeshEdgeArray_init(&mesh->edges);
    IntArray_init(&mesh->neighbors);
    IntArray_init(&mesh->neighbor_edges);
    IntArray_init(&mesh->frontier);
    ugrid_init(&mesh->edge_grid, grid_res, grid_cell_size, grid_cell_cap);
    UGridCellIteratorArray_init(&mesh->grid_it);
    mesh->max_degree = max_degree;
    mesh->intersection_tol = intersection_tol;
}

/** Resets counters for a new episode. */
void mesh_reset(QuadMesh* mesh) {
    MeshVertexArray_resize(&mesh->vertices, 0);
    MeshEdgeArray_resize(&mesh->edges, 0);
    IntArray_resize(&mesh->neighbors, 0);
    IntArray_resize(&mesh->neighbor_edges, 0);
    IntArray_resize(&mesh->frontier, 0);
    ugrid_reset(&mesh->edge_grid);
    UGridCellIteratorArray_resize(&mesh->grid_it, 0);
}

/** Frees mesh buffers. */
void mesh_free(QuadMesh* mesh) {
    MeshVertexArray_free(&mesh->vertices);
    MeshEdgeArray_free(&mesh->edges);
    IntArray_free(&mesh->neighbors);
    IntArray_free(&mesh->neighbor_edges);
    IntArray_free(&mesh->frontier);
    ugrid_free(&mesh->edge_grid);
    UGridCellIteratorArray_free(&mesh->grid_it);
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
    e->face_orientation_cw = (a == e->a) == boundary_ccw;
    e->face_count = 1;
}

static bool mesh_edge_face_orientation_from_vertex(const QuadMesh* mesh, int edge_idx, int source) {
    MeshEdge e = mesh->edges.data[edge_idx];
    return (source == e.a) == e.face_orientation_cw;
}

/** Returns edge index or -1 if missing. */
int mesh_edge_index(const QuadMesh* mesh, int a, int b) {
    for (int i = 0; i < mesh->vertices.data[a].degree; i++) {
        int nidx = mesh_neighbor_idx(mesh, a, i);
        int eidx = mesh->neighbor_edges.data[nidx];
        if (mesh->edges.data[eidx].disabled) continue;
        if (mesh->neighbors.data[nidx] == b) {
            return eidx;
        }
    }
    return -1;
}

static MeshValidReason mesh_validate_edge(const QuadMesh* mesh, int source, Vec2 target_pos, int target_idx) {
    Vec2 source_pos = mesh->vertices.data[source].pos;
    Vec2 tvec = sub2(target_pos, source_pos);
    float best_angle = 1e9f;
    int best_edge = -1;
    for (int i = 0; i < mesh->vertices.data[source].degree; i++) {
        int nidx = mesh_neighbor_idx(mesh, source, i);
        int nvidx = mesh->neighbors.data[nidx];
        int eidx = mesh->neighbor_edges.data[nidx];
        if (mesh->edges.data[eidx].disabled) continue;
        Vec2 np = mesh->vertices.data[nvidx].pos;
        Vec2 evec = sub2(np, source_pos);
        float ang = atan2f(cross2(evec, tvec), dot2(evec, tvec));
        if (fabs(ang) < 1e-2) return MESH_VALID_WEDGE_BLOCKED;
        if (mesh->edges.data[eidx].face_count != 1) continue;
        if (ang <= 0.0f) ang += 2.0f * M_PI;
        if (ang < best_angle) {
            best_angle = ang;
            best_edge = eidx;
        }
    }
    if (best_edge >= 0) {
        bool cw_face = mesh_edge_face_orientation_from_vertex(mesh, best_edge, source);
        if (!cw_face) return MESH_VALID_WEDGE_BLOCKED;
    }

    // for (int i = 0; i < mesh->edges.size; i++) {
    //     int a = mesh->edges.data[i].a;
    //     int b = mesh->edges.data[i].b;
    //     if (a == source || b == source || a == target_idx || b == target_idx) continue;
    //     Vec2 q1 = mesh->vertices.data[a].pos;
    //     Vec2 q2 = mesh->vertices.data[b].pos;
    //     if (segments_intersect(source_pos, target_pos, q1, q2)) return MESH_VALID_INTERSECT;
    // }

    // Intersection test
    UGridCellIteratorArray_resize(&mesh->grid_it, 0);
    ugrid_segment_query_dda(&mesh->edge_grid, source_pos, target_pos, &mesh->grid_it);
    for (int i=0; i<mesh->grid_it.size; ++i) {
        UGridCellIterator* it = &mesh->grid_it.data[i];
        for (int eidx; (eidx = ugrid_cell_it_next(it)) != -1;) {
            if (mesh->edges.data[eidx].disabled) continue;
            int a = mesh->edges.data[eidx].a;
            int b = mesh->edges.data[eidx].b;
            if (a == source || b == source || a == target_idx || b == target_idx) continue;
            Vec2 q1 = mesh->vertices.data[a].pos;
            Vec2 q2 = mesh->vertices.data[b].pos;
            if (segments_intersect(source_pos, target_pos, q1, q2, mesh->intersection_tol)) return MESH_VALID_INTERSECT;
        }
    }

    return MESH_VALID_OK;
}

/** Adds a vertex and returns its index. */
int mesh_add_vertex(QuadMesh* mesh, Vec2 p) {
    int idx = mesh->vertices.size;
    MeshVertexArray_push(&mesh->vertices, (MeshVertex){
        .pos = p,
        .degree = 0,
        .open_edges = 0,
        .frontier_index = -1,
        .frontier_quality = 0.0f,
        .frontier_quality_dirty = true,
        .disabled = false,
    });
    // Push is smart about memory reallocation
    for (int i=0; i<mesh->max_degree; ++i) {
        IntArray_push(&mesh->neighbors, -1);
        IntArray_push(&mesh->neighbor_edges, -1);
    }
    return idx;
}

/** Returns 1 if vertex has open edges and degree < max_degree. */
int mesh_vertex_is_frontier(const QuadMesh* mesh, int v) {
    if (mesh->vertices.data[v].disabled) return 0;
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
    MeshEdgeArray_push(&mesh->edges, (MeshEdge){.a = a, .b = b, .face_count = 0, .face_orientation_cw = false, .disabled = false});
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
    mesh->vertices.data[a].frontier_quality_dirty = true;
    mesh->vertices.data[b].frontier_quality_dirty = true;

    // Update frontier
    mesh_update_frontier_vertex(mesh, a);
    mesh_update_frontier_vertex(mesh, b);

    // Update edge grid
    Vec2 from = mesh->vertices.data[a].pos;
    Vec2 to   = mesh->vertices.data[b].pos;
    UGridCellIteratorArray_resize(&mesh->grid_it, 0);
    ugrid_segment_query(&mesh->edge_grid, from, to, mesh->intersection_tol, &mesh->grid_it);
    for (int i=0; i<mesh->grid_it.size; ++i) {
        ugrid_place(&mesh->edge_grid, mesh->grid_it.data[i].cell, idx);
    }

    return idx;
}

static void mesh_remove_neighbor_edge(QuadMesh* mesh, int v, int eidx) {
    MeshVertex* vertex = &mesh->vertices.data[v];
    for (int i = 0; i < vertex->degree; ++i) {
        int nidx = mesh_neighbor_idx(mesh, v, i);
        if (mesh->neighbor_edges.data[nidx] != eidx) continue;

        int last = vertex->degree - 1;
        int last_idx = mesh_neighbor_idx(mesh, v, last);
        mesh->neighbors.data[nidx] = mesh->neighbors.data[last_idx];
        mesh->neighbor_edges.data[nidx] = mesh->neighbor_edges.data[last_idx];
        mesh->neighbors.data[last_idx] = -1;
        mesh->neighbor_edges.data[last_idx] = -1;
        vertex->degree--;
        return;
    }
}

void mesh_disable_edge(QuadMesh* mesh, int eidx) {
    MeshEdge* e = &mesh->edges.data[eidx];
    if (e->disabled) return;

    int a = e->a;
    int b = e->b;

    if (e->face_count < 2) {
        QM_ASSERT(mesh->vertices.data[a].open_edges > 0);
        QM_ASSERT(mesh->vertices.data[b].open_edges > 0);
        mesh->vertices.data[a].open_edges -= 1;
        mesh->vertices.data[b].open_edges -= 1;
    }

    mesh_remove_neighbor_edge(mesh, a, eidx);
    mesh_remove_neighbor_edge(mesh, b, eidx);
    mesh->vertices.data[a].frontier_quality_dirty = true;
    mesh->vertices.data[b].frontier_quality_dirty = true;
    e->disabled = true;

    if (mesh->vertices.data[a].degree == 0) mesh->vertices.data[a].disabled = true;
    if (mesh->vertices.data[b].degree == 0) mesh->vertices.data[b].disabled = true;

    mesh_update_frontier_vertex(mesh, a);
    mesh_update_frontier_vertex(mesh, b);
}

static bool mesh_edge_is_face_boundary(const QuadMesh* mesh, int eidx, const int* verts, int n) {
    MeshEdge e = mesh->edges.data[eidx];
    for (int i = 0; i < n; ++i) {
        int u = verts[i];
        int v = verts[(i + 1) % n];
        if ((e.a == u && e.b == v) || (e.a == v && e.b == u)) return true;
    }
    return false;
}

void mesh_disable_edges_inside_face(QuadMesh* mesh, const int* verts, int n) {
    QM_ASSERT(n <= 4);

    Vec2 face[4];
    float min_x = INFINITY;
    float min_y = INFINITY;
    float max_x = -INFINITY;
    float max_y = -INFINITY;
    for (int i = 0; i < n; ++i) {
        face[i] = mesh->vertices.data[verts[i]].pos;
        min_x = fminf(min_x, face[i].x);
        min_y = fminf(min_y, face[i].y);
        max_x = fmaxf(max_x, face[i].x);
        max_y = fmaxf(max_y, face[i].y);
    }

    UGridCellIteratorArray_resize(&mesh->grid_it, 0);
    ugrid_aabb_query(&mesh->edge_grid, min_x, min_y, max_x, max_y, &mesh->grid_it);

    BoolArray seen;
    BoolArray_init(&seen);
    BoolArray_resize(&seen, mesh->edges.size);
    for (int i = 0; i < seen.size; ++i) seen.data[i] = false;

    for (int i = 0; i < mesh->grid_it.size; ++i) {
        UGridCellIterator* it = &mesh->grid_it.data[i];
        for (int eidx; (eidx = ugrid_cell_it_next(it)) != -1;) {
            if (seen.data[eidx]) continue;
            seen.data[eidx] = true;

            MeshEdge* e = &mesh->edges.data[eidx];
            if (e->disabled) continue;
            if (mesh_edge_is_face_boundary(mesh, eidx, verts, n)) continue;

            Vec2 a = mesh->vertices.data[e->a].pos;
            Vec2 b = mesh->vertices.data[e->b].pos;
            Vec2 mid = scalmul2(add2(a, b), 0.5f);
            if (point_in_polygon(mid, face, n, 0.0)) {
                mesh_disable_edge(mesh, eidx);
            }
        }
    }

    BoolArray_free(&seen);
}

/** Maybe useful at some point **/
int mesh_distance_A_star(const QuadMesh* mesh, int u, int v, int max_dist) {
    // TODO: implement
    return -1; // distance between vertices is > max_dist
}

/** Returns the index of the n-th neighbor of v along the frontier assuming a ring-shaped frontier **/
int mesh_ring_frontier_neighbor(const QuadMesh* mesh, int v, int n) {
    QM_ASSERT(mesh->vertices.data[v].frontier_index >= 0);
    bool neg_n = (n < 0);
    n = neg_n ? -n : n;
    int prev = -1;
    int curr = v;
    for (int i=0; i<n; ++i) {
        for (int j=0; j<mesh->vertices.data[curr].degree; ++j) {
            int nidx = mesh->neighbors.data[mesh_neighbor_idx(mesh, curr, j)];
            if (nidx != prev && mesh->vertices.data[nidx].frontier_index >= 0) {
                if (prev == -1 && neg_n) {
                    // Skip first frontier neighbor if negative n
                    neg_n = false;
                    continue;
                }
                prev = curr;
                curr = nidx;
                break;
            }
        }
    }
    return curr;
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
        if (eidx < 0) return false; // Edge was disabled
        MeshEdge* e = &mesh->edges.data[eidx];
        if (e->face_count == 2) return false;
        edges[i] = e;
        face_orientations[i] = (u == e->a) != face_ccw;
        // Return if edge is already part of a face on the same side
        if (e->face_count == 1 && e->face_orientation_cw == face_orientations[i]) return false;
    }

    for (int i = 0; i < n; ++i) {
        MeshEdge* e = edges[i];
        ++e->face_count;
        if (e->face_count == 1) {
            e->face_orientation_cw = face_orientations[i];
        } else if (e->face_count == 2) {
            mesh->vertices.data[e->a].open_edges -= 1;
            mesh->vertices.data[e->b].open_edges -= 1;
        }
    }
    for (int i = 0; i < n; ++i) mesh->vertices.data[verts[i]].frontier_quality_dirty = true;
    
    // Local frontier update
    for (int i = 0; i < n; ++i) mesh_update_frontier_vertex(mesh, verts[i]);

    return true;
}


/** Validates an existing target; returns reason. */
MeshValidReason mesh_validate_existing_target(const QuadMesh* mesh, int source, int target, bool boundary_mode, float max_distance) {
    if (target == source) return MESH_VALID_SAME_VERTEX;
    if (mesh_edge_exists(mesh, source, target)) return MESH_VALID_EDGE_EXISTS;
    if (mesh->vertices.data[source].degree >= mesh->max_degree) return MESH_VALID_DEGREE_FULL;
    if (mesh->vertices.data[target].degree >= mesh->max_degree) return MESH_VALID_DEGREE_FULL;
    Vec2 tp = mesh->vertices.data[target].pos;
    if (max_distance > 0.0f) {
        Vec2 sp = mesh->vertices.data[source].pos;
        if (sqrd_norm2(sub2(tp, sp)) > max_distance * max_distance) return MESH_VALID_TOO_FAR;
    }
    if (boundary_mode) {
        int l3 = mesh_ring_frontier_neighbor(mesh, source, -3);
        int r3 = mesh_ring_frontier_neighbor(mesh, source,  3);
        if (target != l3 && target != r3) return MESH_VALID_BOUNDARY_VIOLATION;
    }
    return mesh_validate_edge(mesh, source, tp, target);
}

/** Validates a candidate target position; returns reason. */
MeshValidReason mesh_validate_candidate_target(const QuadMesh* mesh, int source, Vec2 target_pos, bool boundary_mode) {
    if (boundary_mode) {
        int l = mesh_ring_frontier_neighbor(mesh, source, -1);
        int r = mesh_ring_frontier_neighbor(mesh, source,  1);
        if (mesh->vertices.data[l].degree >= mesh->max_degree || mesh->vertices.data[r].degree >= mesh->max_degree) return MESH_VALID_DEGREE_FULL;
        MeshValidReason vr = mesh_validate_edge(mesh, l, target_pos, -1);
        if (vr != MESH_VALID_OK) return vr;
        return mesh_validate_edge(mesh, r, target_pos, -1);
    } else {
        if (mesh->vertices.data[source].degree >= mesh->max_degree) return MESH_VALID_DEGREE_FULL;
        return mesh_validate_edge(mesh, source, target_pos, -1);
    }
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
        case MESH_VALID_BOUNDARY_VIOLATION: return "boundary-violation";
        case MESH_VALID_TOO_FAR: return "too-far";
        default: return "invalid";
    }
}

/** Exports the mesh to an OBJ file (vertices + used edges as lines). */
void mesh_dump_obj(const QuadMesh* mesh, const char* filename) {
    FILE* f = fopen(filename, "w");
    QM_ASSERT(f != NULL);
    fprintf(f, "# QuadMeshing OBJ export\n");
    for (int i = 0; i < mesh->vertices.size; i++) {
        MeshVertex* v = &mesh->vertices.data[i];
        fprintf(f, "v %.8f %.8f 0\n", v->pos.x, v->pos.y);
    }
    for (int i = 0; i < mesh->edges.size; i++) {
        MeshEdge* e = &mesh->edges.data[i];
        if (e->disabled) continue;
        if (e->face_count == 0) continue;
        fprintf(f, "l %d %d\n", e->a + 1, e->b + 1);
    }
    fclose(f);
}
