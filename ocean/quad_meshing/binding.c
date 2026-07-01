#include "quad_meshing.h"
#include <assert.h>
// ACTION BUFFER
#define NUM_ATNS 1
#define ACT_SIZES {1}
// #define NUM_SUBSTEPS 1

// OBSERVATION BUFFER (in bytes)
// - substep [1]
// - frontier size [2]
// - max degree [2]
// - frontier vertices [MAX_FRONTIER_SIZE * 8]
// - frontier neighbors [MAX_FRONTIER_SIZE * MAX_DEGREE * 3]
// - source index [2] (substep 1 only)
// - frontier validity mask [MAX_FRONTIER_SIZE]
// - valid new vertex candidates count [2]
// - valid new vertex candidates [MAX_NEW_CANDIDATES * 8]
#define OBS_TENSOR_T ByteTensor // Pack observations in raw bytes buffer
#define MAX_FRONTIER_SIZE 256 // Assume less than 2^16
#define MAX_DEGREE 8
#define MAX_NEW_CANDIDATES 768 // Assume less than 2^16

#define OBS_SIZE ( \
  1 + \
  2 + \
  2 + \
  MAX_FRONTIER_SIZE * 8 + \
  MAX_FRONTIER_SIZE * MAX_DEGREE * 3 + \
  2 + \
  MAX_FRONTIER_SIZE + \
  2 + \
  MAX_NEW_CANDIDATES * 8 \
)

// ENV
#define Env QuadMeshingEnv

#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;

    env->episode_max_length = (int)dict_get(kwargs, "episode_max_length")->value;
    env->candidate_rings = (int)dict_get(kwargs, "candidate_rings")->value;
    env->candidate_angles = (int)dict_get(kwargs, "candidate_angles")->value;
    env->candidate_radius_min = (float)dict_get(kwargs, "candidate_radius_min")->value;
    env->candidate_radius_max = (float)dict_get(kwargs, "candidate_radius_max")->value;
    env->target_quad_area = (float)dict_get(kwargs, "target_quad_area")->value;

    DictItem* boundary_item = dict_get_unsafe(kwargs, "boundary_paths");
    QM_ASSERT(boundary_item != NULL);
    env->boundary_paths = (const char**)boundary_item->ptr;
    env->boundary_count = (int)boundary_item->value;
    QM_ASSERT(env->boundary_count > 0);

    env->boundary_mode = dict_get(kwargs, "boundary_mode")->value > 0.5;
    env->reward_invalid = (float)dict_get(kwargs, "reward_invalid")->value;
    env->reward_incomplete = (float)dict_get(kwargs, "reward_incomplete")->value;

    env->render_target_fps = (int)dict_get(kwargs, "render_target_fps")->value;
    env->render_width = (int)dict_get(kwargs, "render_width")->value;
    env->render_height = (int)dict_get(kwargs, "render_height")->value;
    env->render_show_frontier = dict_get(kwargs, "render_show_frontier")->value > 0.5;
    env->render_show_candidates = dict_get(kwargs, "render_show_candidates")->value > 0.5;
    env->render_line_thickness = (float)dict_get(kwargs, "render_line_thickness")->value;
    env->render_point_radius = (float)dict_get(kwargs, "render_point_radius")->value;
    env->render_candidate_radius = (float)dict_get(kwargs, "render_candidate_radius")->value;

    env->rng ^= (unsigned int)dict_get(kwargs, "seed")->value;

    env->max_degree = (int)dict_get(kwargs, "max_degree")->value;
    env->grid_res = (float)dict_get(kwargs, "grid_res")->value;
    env->grid_cell_size = (float)dict_get(kwargs, "grid_cell_size")->value;
    env->grid_cell_cap = (float)dict_get(kwargs, "grid_cell_cap")->value;
    env->intersection_tol = (float)dict_get(kwargs, "intersection_tol")->value;

    quad_meshing_init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}
