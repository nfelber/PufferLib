#include "quad_meshing.h"
#include <assert.h>
#include <stdlib.h>

// ACTION BUFFER
#define NUM_ATNS 2
#define ACT_SIZES {1, 1}
#define NUM_SUBSTEPS 1

// OBSERVATION BUFFER (in bytes)
// - substep [1]
// - frontier size [2]
// - max degree [2]
// - frontier vertices [MAX_FRONTIER_SIZE * 8]
// - frontier neighbors [MAX_FRONTIER_SIZE * MAX_DEGREE * 2]
// - frontier validity mask [MAX_FRONTIER_SIZE]
// - valid new vertex candidates count [2]
// - valid new vertex candidates [MAX_NEW_CANDIDATES * 8]
#define OBS_TENSOR_T ByteTensor // Pack observations in raw bytes buffer
#define MAX_FRONTIER_SIZE 1024 // Assume less than 2^16
#define MAX_DEGREE 8
#define MAX_NEW_CANDIDATES 256 // Assume less than 2^16

#define OBS_SIZE ( \
  1 + \
  2 + \
  2 + \
  MAX_FRONTIER_SIZE * 8 + \
  MAX_FRONTIER_SIZE * MAX_DEGREE * 2 + \
  MAX_FRONTIER_SIZE + \
  2 \
  MAX_NEW_CANDIDATES * 8 \
)

// ENV
#define Env QuadMeshingEnv

#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;

    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    env->obs_buffer_size = (int)dict_get(kwargs, "obs_buffer_size")->value;
    env->max_degree = (int)dict_get(kwargs, "max_degree")->value;
    env->max_vertices = (int)dict_get(kwargs, "max_vertices")->value;
    env->max_edges = (int)dict_get(kwargs, "max_edges")->value;
    env->max_frontier = (int)dict_get(kwargs, "max_frontier")->value;
    env->max_candidates = (int)dict_get(kwargs, "max_candidates")->value;
    env->max_boundary_points = (int)dict_get(kwargs, "max_boundary_points")->value;

    DictItem* boundary_item = dict_get_unsafe(kwargs, "boundary_paths");
    QM_ASSERT(boundary_item != NULL);
    env->boundary_paths = (const char**)boundary_item->ptr;
    env->boundary_path_count = (int)boundary_item->value;
    QM_ASSERT(env->boundary_path_count > 0);

    env->candidate_rings = (int)dict_get(kwargs, "candidate_rings")->value;
    env->candidate_angles = (int)dict_get(kwargs, "candidate_angles")->value;
    env->candidate_radius_min = (float)dict_get(kwargs, "candidate_radius_min")->value;
    env->candidate_radius_max = (float)dict_get(kwargs, "candidate_radius_max")->value;

    env->reward_step = (float)dict_get(kwargs, "reward_step")->value;
    env->reward_edge = (float)dict_get(kwargs, "reward_edge")->value;
    env->reward_quad = (float)dict_get(kwargs, "reward_quad")->value;
    env->reward_complete = (float)dict_get(kwargs, "reward_complete")->value;
    env->reward_invalid = (float)dict_get(kwargs, "reward_invalid")->value;

    env->render_target_fps = (int)dict_get(kwargs, "render_target_fps")->value;
    env->render_width = (int)dict_get(kwargs, "render_width")->value;
    env->render_height = (int)dict_get(kwargs, "render_height")->value;
    env->render_show_frontier = dict_get(kwargs, "render_show_frontier")->value > 0.5;
    env->render_show_candidates = dict_get(kwargs, "render_show_candidates")->value > 0.5;
    env->render_show_indices = dict_get(kwargs, "render_show_indices")->value > 0.5;
    env->render_line_thickness = (float)dict_get(kwargs, "render_line_thickness")->value;
    env->render_point_radius = (float)dict_get(kwargs, "render_point_radius")->value;
    env->render_candidate_radius = (float)dict_get(kwargs, "render_candidate_radius")->value;

    env->rng ^= (unsigned int)dict_get(kwargs, "seed")->value;

    quad_meshing_init(env, MAX_DEGREE, MAX_FRONTIER_SIZE, MAX_NEW_CANDIDATES);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "n", log->n);
}

