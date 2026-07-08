#include "quad_meshing.h"
#include "mesh.h"
#include "raylib.h"

#include <stdio.h>
#include <stdlib.h>

int main() {
    QuadMeshingEnv env = {0};
    env.episode_max_length_ratio = 1.5;
    static const char* shape_paths[] = {
      "resources/quad_meshing/shapes/dolphin.qmshape"
    };
    env.shape_paths = shape_paths;
    env.shape_count = (int)(sizeof(shape_paths) / sizeof(shape_paths[0]));
    env.boundary_mode = false;
    env.export_obj = false;
    env.export_obj_path = "quad_meshing.obj";
    env.candidate_rings = 10;
    env.candidate_angles = 64;
    env.target_edge_length_ratio = 1.0;
    env.candidate_radius_min_ratio = 0.5;
    env.candidate_radius_max_ratio = 2.0;
    env.reward_invalid = -0.05;
    env.reward_incomplete = -1.0;
    env.reward_triangle = -0.5;
    env.base_quad_reward = 0.0;
    env.potential_beta = 0.15;
    env.potential_gamma = 1.0;
    env.frontier_quality_weight = 1.0;
    env.frontier_edge_length_weight = 1.0;
    env.frontier_alignment_weight = 1.0;
    env.frontier_angle_weight = 1.0;
    env.frontier_size_pressure_weight = 1.0;
    env.degree_pressure_weight = 1.0;
    env.safe_frontier_size_ratio = 0.9;
    env.safe_degree_ratio = 0.7;
    env.render_width = 1200;
    env.render_height = 900;
    env.render_target_fps = 144;
    env.render_show_frontier = true;
    env.render_show_candidates = true;
    env.render_show_cross_field = false;
    env.render_line_thickness = 2.0f;
    env.render_point_radius = 4.0f;
    env.render_candidate_radius = 3.0f;
    env.rng = 12345;

    unsigned int rng = 42;
    double key_repeat_cd = 0.01;

    env.max_frontier = 256;
    env.max_degree = 8;
    env.grid_res = 32;
    env.grid_cell_size = 1.0 / (float)env.grid_res;
    env.grid_cell_cap = 8;
    env.intersection_tol = 1e-3;

    quad_meshing_init(&env);
    env.observations = (unsigned char*)calloc(1 + 4 + 4 + env.max_frontier * (8 + 3 * env.max_degree + 1) + 2 + env.candidate_angles*env.candidate_rings * 8, sizeof(unsigned char));
    env.actions = (float*)calloc(2, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);

    SerialObsBuffer obs = {.sb = env.observations};

    env.ui_pending_source = -1;
    double last_key_repeat = GetTime();
    while (!WindowShouldClose()) {
        float pick_radius = env.camera.zoom > 0 ? 10.0f / env.camera.zoom : 0.03f;
        if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
            env.ui_pending_source = -1;
        }
        if (IsKeyDown(KEY_TAB) && GetTime() - last_key_repeat > key_repeat_cd) {
            last_key_repeat = GetTime();

            // Pick source + target at random
            QuadMesh mesh;
            mesh_init(&mesh, 0, env.grid_res, env.grid_cell_size, env.grid_cell_cap, env.intersection_tol);
            deserialize_obs_frontier(&obs, &mesh);

            if (env.ui_pending_source < 0) {
                // Substep
                env.actions[0] = rand_range(&rng, mesh.frontier.size);
                env.ui_pending_source = env.actions[0];
                // c_substep(&env, 0);
                c_step(&env);
            } else {
                BoolArray validity_mask;
                BoolArray_init(&validity_mask);
                deserialize_obs_validity_mask(&obs, &mesh, &validity_mask);
                Vec2Array new_candidates;
                Vec2Array_init(&new_candidates);
                deserialize_obs_new_candidates(&obs, &mesh, &new_candidates);

                IntArray valid_targets;
                IntArray_init(&valid_targets);

                for (int i = 0; i < mesh.vertices.size; i++) {
                    if (validity_mask.data[i]) IntArray_push(&valid_targets, i);
                }

                for (int i = 0; i < new_candidates.size; i++) {
                    IntArray_push(&valid_targets, mesh.frontier.size + i);
                }

                // Step
                if (valid_targets.size > 0) {
                    // env.actions[1] = rand_range(&rng, valid_targets.size);
                    env.actions[0] = rand_range(&rng, valid_targets.size);
                } else {
                    // env.actions[1] = 0;
                    env.actions[0] = 0;
                }
                env.ui_pending_source = -1;
                c_step(&env);

                BoolArray_free(&validity_mask);
                Vec2Array_free(&new_candidates);
                IntArray_free(&valid_targets);
            }

            mesh_free(&mesh);
        }

        if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
            env.ui_pending_source = -1;
        }

        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && IsWindowReady()) {
            Vector2 vmouse_world = GetScreenToWorld2D(GetMousePosition(), env.camera);
            Vec2 mouse_world = {vmouse_world.x, -vmouse_world.y};
            // uint8_t substep = deserialize_obs_substep(&obs);
            // printf("substep: %d\n", substep);
            QuadMesh mesh;
            mesh_init(&mesh, 0, env.grid_res, env.grid_cell_size, env.grid_cell_cap, env.intersection_tol);
            deserialize_obs_frontier(&obs, &mesh);
            if (env.ui_pending_source < 0) {
                float best = 1e9f;
                int best_idx = -1;
                for (int i = 0; i < mesh.frontier.size; i++) {
                    int vidx = mesh.frontier.data[i];
                    Vec2 p = mesh.vertices.data[vidx].pos;
                    float d = sqrd_norm2(sub2(mouse_world, p));
                    if (d < best) {
                        best = d;
                        best_idx = i;
                    }
                }
                if (best_idx >= 0 && best <= pick_radius*pick_radius) {
                    env.ui_pending_source = best_idx;
                    env.actions[0] = best_idx;
                    // c_substep(&env, 0);
                    c_step(&env);
                }
            } else {
                BoolArray validity_mask;
                BoolArray_init(&validity_mask);
                deserialize_obs_validity_mask(&obs, &mesh, &validity_mask);
                Vec2Array new_candidates;
                Vec2Array_init(&new_candidates);
                deserialize_obs_new_candidates(&obs, &mesh, &new_candidates);

                float best = 1e9f;
                int best_target = -1;
                int last_bounday_action_idx = 0;

                for (int i = 0; i < mesh.frontier.size; i++) {
                    int vidx = mesh.frontier.data[i];
                    Vec2 p = mesh.vertices.data[vidx].pos;
                    float d = sqrd_norm2(sub2(mouse_world, p));
                    if (validity_mask.data[i]) {
                        if (d < best) {
                            best = d;
                            best_target = last_bounday_action_idx;
                        }
                        ++last_bounday_action_idx;
                    }
                }

                for (int i = 0; i < new_candidates.size; i++) {
                    Vec2 cp = new_candidates.data[i];
                    float d = sqrd_norm2(sub2(mouse_world, cp));
                    if (d < best) {
                        best = d;
                        best_target = last_bounday_action_idx + i;
                    }
                }

                if (best_target >= 0 && best <= pick_radius*pick_radius) {
                    // env.actions[0] = env.ui_pending_source;
                    // env.actions[1] = best_target;
                    env.actions[0] = best_target;
                    c_step(&env);
                    printf("reward: %f\n", env.rewards[0]);
                }

                env.ui_pending_source = -1;

                BoolArray_free(&validity_mask);
                Vec2Array_free(&new_candidates);
            }
            mesh_free(&mesh);
        }

        c_render(&env);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
