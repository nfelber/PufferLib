#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "raylib.h"

#ifndef QUAD_MESHING_ENABLE_ASSERTS
#define QUAD_MESHING_ENABLE_ASSERTS 1
#endif

#if QUAD_MESHING_ENABLE_ASSERTS
#define QM_ASSERT(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "quad_meshing assert failed: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
        abort(); \
    } \
} while (0)
#else
#define QM_ASSERT(expr) do { (void)sizeof(expr); } while (0)
#endif

typedef struct {
    float x;
    float y;
    unsigned char degree;
    unsigned char open_edges;
} MeshVertex;

typedef struct {
    int a;
    int b;
    unsigned char face_count;
    unsigned char face_side;
} MeshEdge;

typedef struct {
    MeshVertex* vertices;
    MeshEdge* edges;
    int* neighbors;
    int* neighbor_edges;
    int num_vertices;
    int num_edges;
    int num_quads;
    int max_degree_limit;
    int max_vertices;
    int max_edges;
} QuadMesh;

typedef enum {
    MESH_VALID_OK = 0,
    MESH_VALID_SAME_VERTEX,
    MESH_VALID_OUT_OF_BOUNDS,
    MESH_VALID_EDGE_EXISTS,
    MESH_VALID_DEGREE_FULL,
    MESH_VALID_TRIANGLE,
    MESH_VALID_INTERSECT,
    MESH_VALID_WEDGE_BLOCKED,
    MESH_VALID_CAPACITY,
} MeshValidReason;

/** Allocates mesh buffers and sets capacity/degree limits. */
void mesh_init(QuadMesh* mesh, int max_vertices, int max_edges, int max_degree_limit);

/** Resets counters for a new episode. */
void mesh_reset(QuadMesh* mesh);

/** Frees mesh buffers. */
void mesh_free(QuadMesh* mesh);

/** Adds a vertex and returns its index, or -1 on capacity. */
int mesh_add_vertex(QuadMesh* mesh, Vector2 p);

/** Adds an edge and returns its index, or -1 on capacity. */
int mesh_add_edge(QuadMesh* mesh, int a, int b);

/** Returns edge index or -1 if missing. */
int mesh_edge_index(const QuadMesh* mesh, int a, int b);

/** Returns 1 if edge exists. */
int mesh_edge_exists(const QuadMesh* mesh, int a, int b);

/** Returns 1 if vertex has open edges and degree < max_degree. */
int mesh_is_frontier(const QuadMesh* mesh, int v);

/** Detects all 3-cycles involving edge (u,v). */
int mesh_detect_triangles(
    const QuadMesh* mesh, int u, int v, int* out_cycles, int* out_lengths, int max_cycles);

/** Detects all 4-cycles involving edge (u,v). */
int mesh_detect_quads(
    const QuadMesh* mesh, int u, int v, int* out_cycles, int* out_lengths, int max_cycles);

/** Registers a face cycle and updates edge face counts/sides. */
void mesh_register_face(QuadMesh* mesh, const int* verts, int n);

/** Validates an existing target; returns reason. */
MeshValidReason mesh_validate_existing_target(
    const QuadMesh* mesh, int source, int target);

/** Validates a candidate target position; returns reason. */
MeshValidReason mesh_validate_candidate_target(
    const QuadMesh* mesh, int source, Vector2 target_pos);

/** Returns a reason string for debugging. */
const char* mesh_valid_reason_str(MeshValidReason r);

/** Sets a boundary edge face side based on polygon orientation. */
void mesh_set_boundary_edge_face(QuadMesh* mesh, int edge_idx, int a, int b, int boundary_ccw);
