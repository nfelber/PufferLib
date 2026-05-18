#include "mesh.h"
#include "geometry.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static bool mesh_edge_is_canonical_vertices(const QuadMesh* mesh, int a, int b) {
    Vector2 pa = {mesh->vertices[a].x, mesh->vertices[a].y};
    Vector2 pb = {mesh->vertices[b].x, mesh->vertices[b].y};
    return (pa.y < pb.y || (pa.y == pb.y && pa.x <= pb.x));
}

static inline int mesh_neighbor_idx(const QuadMesh* mesh, int v, int slot) {
    return v * mesh->max_degree_limit + slot;
}

void mesh_set_boundary_edge_face(QuadMesh* mesh, int edge_idx, int a, int b, int boundary_ccw) {
    int face_ccw_edge_dir = boundary_ccw ? 0 : 1;
    MeshEdge* e = &mesh->edges[edge_idx];
    int face_ccw = (a == e->a && b == e->b) ? face_ccw_edge_dir : !face_ccw_edge_dir;
    e->face_side = face_ccw ? 1 : 0;
}

static int mesh_edge_face_side_from_vertex(const QuadMesh* mesh, int edge_idx, int source) {
    MeshEdge e = mesh->edges[edge_idx];
    int face_ccw = e.face_side ? 1 : 0;
    if (source == e.a) return face_ccw;
    if (source == e.b) return !face_ccw;
    QM_ASSERT(!"source must be an endpoint of edge");
    return face_ccw;
}

static int mesh_would_form_triangle(const QuadMesh* mesh, int source, int target) {
    for (int i = 0; i < mesh->vertices[source].degree; i++) {
        int a = mesh->neighbors[mesh_neighbor_idx(mesh, source, i)];
        for (int j = 0; j < mesh->vertices[target].degree; j++) {
            if (mesh->neighbors[mesh_neighbor_idx(mesh, target, j)] == a) return 1;
        }
    }
    return 0;
}

static MeshValidReason mesh_validate_edge(const QuadMesh* mesh, int source, Vector2 target_pos, int target_idx) {
    Vector2 sp = {mesh->vertices[source].x, mesh->vertices[source].y};
    (void)target_idx;
    if (mesh->num_edges > 0) {
        for (int i = 0; i < mesh->num_edges; i++) {
            MeshEdge e = mesh->edges[i];
            if (e.a == source || e.b == source || e.a == target_idx || e.b == target_idx) continue;
            Vector2 q1 = {mesh->vertices[e.a].x, mesh->vertices[e.a].y};
            Vector2 q2 = {mesh->vertices[e.b].x, mesh->vertices[e.b].y};
            if (segments_intersect(sp, target_pos, q1, q2)) return MESH_VALID_INTERSECT;
        }
    }
    Vector2 tvec = v2_sub(target_pos, sp);
    float best_angle = 1e9f;
    int best_edge = -1;
    for (int i = 0; i < mesh->vertices[source].degree; i++) {
        int n = mesh->neighbors[mesh_neighbor_idx(mesh, source, i)];
        int eidx = mesh_edge_index(mesh, source, n);
        if (eidx < 0) continue;
        if (mesh->edges[eidx].face_count != 1) continue;
        Vector2 evec = v2_sub((Vector2){mesh->vertices[n].x, mesh->vertices[n].y}, sp);
        float ang = atan2f(v2_cross(evec, tvec), v2_dot(evec, tvec));
        if (ang <= 0.0f) ang += 2.0f * PI;
        if (ang < best_angle) {
            best_angle = ang;
            best_edge = eidx;
        }
    }
    if (best_edge >= 0) {
        int ccw_ok = mesh_edge_face_side_from_vertex(mesh, best_edge, source);
        if (ccw_ok) return MESH_VALID_WEDGE_BLOCKED;
    }
    return MESH_VALID_OK;
}

void mesh_init(QuadMesh* mesh, int max_vertices, int max_edges, int max_degree_limit) {
    QM_ASSERT(max_vertices > 0);
    QM_ASSERT(max_edges > 0);
    QM_ASSERT(max_degree_limit > 0);
    if (mesh->vertices == NULL || mesh->max_vertices != max_vertices) {
        free(mesh->vertices);
        mesh->vertices = (MeshVertex*)calloc((size_t)max_vertices, sizeof(MeshVertex));
    }
    if (mesh->edges == NULL || mesh->max_edges != max_edges) {
        free(mesh->edges);
        mesh->edges = (MeshEdge*)calloc((size_t)max_edges, sizeof(MeshEdge));
    }
    size_t neighbor_count = (size_t)max_vertices * (size_t)max_degree_limit;
    if (mesh->neighbors == NULL || mesh->max_vertices != max_vertices || mesh->max_degree_limit != max_degree_limit) {
        free(mesh->neighbors);
        mesh->neighbors = (int*)calloc(neighbor_count, sizeof(int));
    }
    if (mesh->neighbor_edges == NULL || mesh->max_vertices != max_vertices || mesh->max_degree_limit != max_degree_limit) {
        free(mesh->neighbor_edges);
        mesh->neighbor_edges = (int*)calloc(neighbor_count, sizeof(int));
    }
    QM_ASSERT(mesh->vertices != NULL);
    QM_ASSERT(mesh->edges != NULL);
    QM_ASSERT(mesh->neighbors != NULL);
    QM_ASSERT(mesh->neighbor_edges != NULL);
    mesh->max_vertices = max_vertices;
    mesh->max_edges = max_edges;
    mesh->max_degree_limit = max_degree_limit;
    mesh_reset(mesh);
}

void mesh_reset(QuadMesh* mesh) {
    mesh->num_vertices = 0;
    mesh->num_edges = 0;
    mesh->num_quads = 0;
}

void mesh_free(QuadMesh* mesh) {
    free(mesh->vertices);
    free(mesh->edges);
    free(mesh->neighbors);
    free(mesh->neighbor_edges);
    memset(mesh, 0, sizeof(*mesh));
}

int mesh_add_vertex(QuadMesh* mesh, Vector2 p) {
    if (mesh->num_vertices >= mesh->max_vertices) {
        QM_ASSERT(!"max_vertices exceeded");
        return -1;
    }
    int idx = mesh->num_vertices++;
    mesh->vertices[idx] = (MeshVertex){.x = p.x, .y = p.y, .degree = 0, .open_edges = 0};
    return idx;
}

int mesh_add_edge(QuadMesh* mesh, int a, int b) {
    if (mesh->num_edges >= mesh->max_edges) {
        QM_ASSERT(!"max_edges exceeded");
        return -1;
    }
    QM_ASSERT(a >= 0 && a < mesh->num_vertices);
    QM_ASSERT(b >= 0 && b < mesh->num_vertices);
    QM_ASSERT(a != b);
    if (!mesh_edge_is_canonical_vertices(mesh, a, b)) {
        int tmp = a;
        a = b;
        b = tmp;
    }
    int idx = mesh->num_edges++;
    mesh->edges[idx] = (MeshEdge){.a = a, .b = b, .face_count = 0, .face_side = 0};
    mesh->vertices[a].open_edges += 1;
    mesh->vertices[b].open_edges += 1;
    QM_ASSERT(mesh->vertices[a].degree < mesh->max_degree_limit);
    QM_ASSERT(mesh->vertices[b].degree < mesh->max_degree_limit);
    int apos = mesh->vertices[a].degree++;
    mesh->neighbors[mesh_neighbor_idx(mesh, a, apos)] = b;
    mesh->neighbor_edges[mesh_neighbor_idx(mesh, a, apos)] = idx;
    int bpos = mesh->vertices[b].degree++;
    mesh->neighbors[mesh_neighbor_idx(mesh, b, bpos)] = a;
    mesh->neighbor_edges[mesh_neighbor_idx(mesh, b, bpos)] = idx;
    return idx;
}

int mesh_edge_index(const QuadMesh* mesh, int a, int b) {
    for (int i = 0; i < mesh->vertices[a].degree; i++) {
        if (mesh->neighbors[mesh_neighbor_idx(mesh, a, i)] == b) {
            return mesh->neighbor_edges[mesh_neighbor_idx(mesh, a, i)];
        }
    }
    return -1;
}

int mesh_edge_exists(const QuadMesh* mesh, int a, int b) {
    return mesh_edge_index(mesh, a, b) >= 0;
}

int mesh_is_frontier(const QuadMesh* mesh, int v) {
    return mesh->vertices[v].open_edges > 0 && mesh->vertices[v].degree < mesh->max_degree_limit;
}

int mesh_detect_triangles(
    const QuadMesh* mesh, int u, int v, int* out_cycles, int* out_lengths, int max_cycles
) {
    int count = 0;
    for (int i = 0; i < mesh->vertices[u].degree; i++) {
        int a = mesh->neighbors[mesh_neighbor_idx(mesh, u, i)];
        if (a == v) continue;
        int uv = mesh_edge_index(mesh, u, v);
        int ua = mesh_edge_index(mesh, u, a);
        int av = mesh_edge_index(mesh, a, v);
        if (uv < 0 || ua < 0 || av < 0) continue;
        if (mesh->edges[uv].face_count >= 2 ||
            mesh->edges[ua].face_count >= 2 ||
            mesh->edges[av].face_count >= 2) {
            continue;
        }
        if (count < max_cycles) {
            out_lengths[count] = 3;
            out_cycles[count * 4 + 0] = u;
            out_cycles[count * 4 + 1] = a;
            out_cycles[count * 4 + 2] = v;
            out_cycles[count * 4 + 3] = -1;
            count++;
        }
    }
    return count;
}

int mesh_detect_quads(
    const QuadMesh* mesh, int u, int v, int* out_cycles, int* out_lengths, int max_cycles
) {
    int count = 0;
    int uv = mesh_edge_index(mesh, u, v);
    if (uv < 0 || mesh->edges[uv].face_count >= 2) return count;
    for (int i = 0; i < mesh->vertices[u].degree; i++) {
        int a = mesh->neighbors[mesh_neighbor_idx(mesh, u, i)];
        if (a == v) continue;
        for (int j = 0; j < mesh->vertices[v].degree; j++) {
            int b = mesh->neighbors[mesh_neighbor_idx(mesh, v, j)];
            if (b == u || b == a) continue;
            int ua = mesh_edge_index(mesh, u, a);
            int vb = mesh_edge_index(mesh, v, b);
            int ab = mesh_edge_index(mesh, a, b);
            if (ua < 0 || vb < 0 || ab < 0) continue;
            if (mesh->edges[ua].face_count >= 2 ||
                mesh->edges[vb].face_count >= 2 ||
                mesh->edges[ab].face_count >= 2) {
                continue;
            }
            if (count < max_cycles) {
                out_lengths[count] = 4;
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

void mesh_register_face(QuadMesh* mesh, const int* verts, int n) {
    QM_ASSERT(n<=4);
    Vector2 face[4];
    for (int i = 0; i < n; ++i)
        face[i] = (Vector2){mesh->vertices[verts[i]].x, mesh->vertices[verts[i]].y};
    bool face_ccw = polygon_is_ccw(face, n);
    for (int i = 0; i < n; ++i) {
        int u = verts[i];
        int v = verts[(i + 1) % n];
        int eidx = mesh_edge_index(mesh, u, v);
        QM_ASSERT(eidx != -1);
        MeshEdge* e = &mesh->edges[eidx];
        QM_ASSERT(e->face_count < 2);
        ++e->face_count;
        if (e->face_count == 1) {
            e->face_side = ((u == e->a) == face_ccw) ? 1 : 0;
        } else if (e->face_count == 2) {
            mesh->vertices[e->a].open_edges -= 1;
            mesh->vertices[e->b].open_edges -= 1;
        }
    }
}


MeshValidReason mesh_validate_existing_target(const QuadMesh* mesh, int source, int target) {
    if (target == source) return MESH_VALID_SAME_VERTEX;
    if (target >= mesh->num_vertices) return MESH_VALID_OUT_OF_BOUNDS;
    if (mesh_edge_exists(mesh, source, target)) return MESH_VALID_EDGE_EXISTS;
    if (mesh->vertices[source].degree >= mesh->max_degree_limit) return MESH_VALID_DEGREE_FULL;
    if (mesh->vertices[target].degree >= mesh->max_degree_limit) return MESH_VALID_DEGREE_FULL;
    if (mesh_would_form_triangle(mesh, source, target)) return MESH_VALID_TRIANGLE;
    Vector2 tp = {mesh->vertices[target].x, mesh->vertices[target].y};
    return mesh_validate_edge(mesh, source, tp, target);
}

MeshValidReason mesh_validate_candidate_target(const QuadMesh* mesh, int source, Vector2 target_pos) {
    if (mesh->num_vertices >= mesh->max_vertices) return MESH_VALID_CAPACITY;
    if (mesh->vertices[source].degree >= mesh->max_degree_limit) return MESH_VALID_DEGREE_FULL;
    return mesh_validate_edge(mesh, source, target_pos, -1);
}

const char* mesh_valid_reason_str(MeshValidReason r) {
    switch (r) {
        case MESH_VALID_OK: return "valid";
        case MESH_VALID_SAME_VERTEX: return "same-vertex";
        case MESH_VALID_OUT_OF_BOUNDS: return "out-of-bounds";
        case MESH_VALID_EDGE_EXISTS: return "edge-exists";
        case MESH_VALID_DEGREE_FULL: return "degree-full";
        case MESH_VALID_TRIANGLE: return "triangle-closure";
        case MESH_VALID_INTERSECT: return "edge-intersection";
        case MESH_VALID_WEDGE_BLOCKED: return "wedge-blocked";
        case MESH_VALID_CAPACITY: return "max-vertices";
        default: return "invalid";
    }
}
