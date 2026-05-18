#include "quad_meshing.h"
#include <assert.h>

#define NUM_ATNS 2
#define ACT_SIZES {DEFAULT_MAX_FRONTIER, DEFAULT_MAX_TARGETS}
#define OBS_TENSOR_T FloatTensor
#define OBS_SIZE DEFAULT_OBS_SIZE
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
    env->max_existing_targets = (int)dict_get(kwargs, "max_existing_targets")->value;
    env->max_candidates = (int)dict_get(kwargs, "max_candidates")->value;
    env->max_quads = (int)dict_get(kwargs, "max_quads")->value;
    env->max_boundary_points = (int)dict_get(kwargs, "max_boundary_points")->value;

    env->boundary_type = (BoundaryType)(int)dict_get(kwargs, "boundary_type")->value;
    env->boundary_points = (int)dict_get(kwargs, "boundary_points")->value;
    if (env->boundary_points < 4) env->boundary_points = 4;
    if (env->boundary_points > env->max_boundary_points) env->boundary_points = env->max_boundary_points;
    if (env->boundary_points > env->max_vertices) env->boundary_points = env->max_vertices;

    env->candidate_rings = (int)dict_get(kwargs, "candidate_rings")->value;
    env->candidate_angles = (int)dict_get(kwargs, "candidate_angles")->value;
    env->candidate_radius_min = (float)dict_get(kwargs, "candidate_radius_min")->value;
    env->candidate_radius_max = (float)dict_get(kwargs, "candidate_radius_max")->value;

    env->reward_step = (float)dict_get(kwargs, "reward_step")->value;
    env->reward_edge = (float)dict_get(kwargs, "reward_edge")->value;
    env->reward_quad = (float)dict_get(kwargs, "reward_quad")->value;
    env->reward_complete = (float)dict_get(kwargs, "reward_complete")->value;
    env->reward_invalid = (float)dict_get(kwargs, "reward_invalid")->value;

    env->render_width = (int)dict_get(kwargs, "render_width")->value;
    env->render_height = (int)dict_get(kwargs, "render_height")->value;
    env->render_show_frontier = dict_get(kwargs, "render_show_frontier")->value > 0.5;
    env->render_show_candidates = dict_get(kwargs, "render_show_candidates")->value > 0.5;
    env->render_show_indices = dict_get(kwargs, "render_show_indices")->value > 0.5;
    env->render_line_thickness = (float)dict_get(kwargs, "render_line_thickness")->value;
    env->render_point_radius = (float)dict_get(kwargs, "render_point_radius")->value;
    env->render_candidate_radius = (float)dict_get(kwargs, "render_candidate_radius")->value;

    env->rng ^= (unsigned int)dict_get(kwargs, "seed")->value;

    quad_meshing_init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "n", log->n);
}
