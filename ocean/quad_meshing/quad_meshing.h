#pragma once

#include <stdbool.h>
#include "raylib.h"
#include "mesh.h"

typedef struct { float score, n; } Log;

typedef enum {
    BOUNDARY_SQUARE = 0,
    BOUNDARY_CIRCLE = 1,
} BoundaryType;

typedef struct {
    Log log;
    int num_agents;
    unsigned int rng;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;

    int max_steps;
    int obs_buffer_size;
    int max_degree;
    int max_vertices;
    int max_edges;
    int max_frontier;
    int max_existing_targets;
    int max_candidates;
    int max_targets;
    int max_quads;
    int max_boundary_points;

    int obs_vertex_offset;
    int obs_edge_offset;
    int obs_frontier_offset;
    int obs_source_mask_offset;
    int obs_existing_target_offset;
    int obs_candidate_offset;
    int obs_target_mask_offset;
    int obs_size;

    BoundaryType boundary_type;
    int boundary_points;
    Vector2* boundary_poly;
    int boundary_poly_count;

    int candidate_rings;
    int candidate_angles;
    float candidate_radius_min;
    float candidate_radius_max;
    Vector2* candidate_offsets;
    int candidate_count;

    QuadMesh mesh;

    int* frontier;
    int num_frontier;

    int steps;
    int last_source_slot;
    int last_target_slot;
    int last_target_is_candidate;
    Vector2 last_target_pos;
    int last_valid;
    int last_quad_completed;
    int last_done;
    float last_reward;
    int ui_pending_source;
    int ui_selecting_target;

    float reward_step;
    float reward_edge;
    float reward_quad;
    float reward_complete;
    float reward_invalid;

    int render_width;
    int render_height;
    bool render_show_frontier;
    bool render_show_candidates;
    bool render_show_indices;
    float render_line_thickness;
    float render_point_radius;
    float render_candidate_radius;
    Camera2D camera;
    float camera_zoom;
} QuadMeshingEnv;

/** Allocates env buffers and computes observation layout. */
void quad_meshing_init(QuadMeshingEnv* env);

/** Resets the environment state and observation buffers. */
void c_reset(QuadMeshingEnv* env);

/** Advances one environment step using the current action buffer. */
void c_step(QuadMeshingEnv* env);

/** Renders the current environment state (raylib). */
void c_render(QuadMeshingEnv* env);

/** Releases any rendering resources. */
void c_close(QuadMeshingEnv* env);
