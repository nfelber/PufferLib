#pragma once

#include "qmsurface.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    Qm3Vec3 pos;
    Qm3Vec3 normal;
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
    uint8_t face_count;
    bool disabled;
} Qm3MeshEdge;

typedef struct {
    Qm3MeshVertex* vertices;
    Qm3MeshEdge* edges;
    int32_t* neighbors;
    int32_t* neighbor_edges;
    uint32_t* frontier;
    uint32_t vertex_count;
    uint32_t vertex_cap;
    uint32_t edge_count;
    uint32_t edge_cap;
    uint32_t frontier_count;
    uint32_t frontier_cap;
    uint32_t max_degree;
} Qm3Mesh;

static void* qm3_checked_realloc(void* ptr, size_t bytes) {
    void* out = realloc(ptr, bytes);
    QM3_ASSERT(out != NULL || bytes == 0);
    return out;
}

static void qm3_mesh_init(Qm3Mesh* mesh, uint32_t max_degree) {
    memset(mesh, 0, sizeof(*mesh));
    mesh->max_degree = max_degree > 0 ? max_degree : 16;
}

static void qm3_mesh_reset(Qm3Mesh* mesh) {
    mesh->vertex_count = 0;
    mesh->edge_count = 0;
    mesh->frontier_count = 0;
}

static void qm3_mesh_free(Qm3Mesh* mesh) {
    free(mesh->vertices);
    free(mesh->edges);
    free(mesh->neighbors);
    free(mesh->neighbor_edges);
    free(mesh->frontier);
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

static uint32_t qm3_mesh_add_vertex(Qm3Mesh* mesh, Qm3Vec3 pos, Qm3Vec3 normal, uint32_t surface_vertex, int32_t surface_tri) {
    if (mesh->vertex_count == mesh->vertex_cap) {
        uint32_t new_cap = mesh->vertex_cap ? mesh->vertex_cap * 2 : 64;
        qm3_mesh_reserve_vertices(mesh, new_cap);
    }
    uint32_t idx = mesh->vertex_count++;
    mesh->vertices[idx] = (Qm3MeshVertex){
        .pos = pos,
        .normal = normal,
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

static uint32_t qm3_mesh_add_edge(Qm3Mesh* mesh, uint32_t a, uint32_t b, uint8_t face_count) {
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
    mesh->edges[eidx] = (Qm3MeshEdge){
        .a = a,
        .b = b,
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

static int32_t qm3_surface_vertex_incident_tri(const Qm3Surface* surface, uint32_t surface_vertex) {
    for (uint32_t i = 0; i < surface->sharp_edge_count; ++i) {
        Qm3SharpEdge e = surface->sharp_edges[i];
        if (e.a != surface_vertex && e.b != surface_vertex) continue;
        if (e.f0 >= 0) return e.f0;
        if (e.f1 >= 0) return e.f1;
    }
    return -1;
}

static uint32_t qm3_mesh_vertex_for_surface_vertex(
    Qm3Mesh* mesh,
    const Qm3Surface* surface,
    int32_t* surface_to_graph,
    uint32_t surface_vertex
) {
    QM3_ASSERT(surface_vertex < surface->vertex_count);
    if (surface_to_graph[surface_vertex] >= 0) return (uint32_t)surface_to_graph[surface_vertex];

    int32_t tri = qm3_surface_vertex_incident_tri(surface, surface_vertex);
    uint32_t graph_vertex = qm3_mesh_add_vertex(
        mesh,
        surface->vertices[surface_vertex],
        surface->vertex_normals[surface_vertex],
        surface_vertex,
        tri
    );
    surface_to_graph[surface_vertex] = (int32_t)graph_vertex;
    return graph_vertex;
}

static void qm3_mesh_build_from_sharp_edges(Qm3Mesh* mesh, const Qm3Surface* surface) {
    qm3_mesh_reset(mesh);
    qm3_mesh_reserve_edges(mesh, surface->sharp_edge_count);
    qm3_mesh_reserve_vertices(mesh, surface->sharp_edge_count * 2);
    qm3_mesh_reserve_frontier(mesh, surface->sharp_edge_count * 2);

    int32_t* surface_to_graph = (int32_t*)malloc((size_t)surface->vertex_count * sizeof(int32_t));
    QM3_ASSERT(surface_to_graph != NULL);
    for (uint32_t i = 0; i < surface->vertex_count; ++i) surface_to_graph[i] = -1;

    for (uint32_t i = 0; i < surface->sharp_edge_count; ++i) {
        Qm3SharpEdge edge = surface->sharp_edges[i];
        uint32_t a = qm3_mesh_vertex_for_surface_vertex(mesh, surface, surface_to_graph, edge.a);
        uint32_t b = qm3_mesh_vertex_for_surface_vertex(mesh, surface, surface_to_graph, edge.b);
        qm3_mesh_add_edge(mesh, a, b, 1);
    }

    free(surface_to_graph);
}
