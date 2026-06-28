#include "quad_meshing.h"
#include "mesh.h"
#include "raylib.h"

#include <stdio.h>
#include <stdlib.h>

int main() {
    QuadMeshingEnv env = {0};
    env.episode_max_length = 2048;
    static const char* boundary_paths[] = {
      // "resources/quad_meshing/boundaries/square.json"
      "resources/quad_meshing/boundaries/dolphin.json"
      // "resources/quad_meshing/boundaries/rat-18.json",
      // "resources/quad_meshing/boundaries/cup-13.json",
      // "resources/quad_meshing/boundaries/chopper-12.json",
      // "resources/quad_meshing/boundaries/classic-8.json",
      // "resources/quad_meshing/boundaries/dog-12.json",
      // "resources/quad_meshing/boundaries/Bone-11.json",
      // "resources/quad_meshing/boundaries/stef-19.json",
      // "resources/quad_meshing/boundaries/bell-19.json",
      // "resources/quad_meshing/boundaries/fork-18.json",
      // "resources/quad_meshing/boundaries/pencil-11.json"
    };
    env.boundary_paths = boundary_paths;
    env.boundary_count = (int)(sizeof(boundary_paths) / sizeof(boundary_paths[0]));
    env.boundary_mode = true;
    env.candidate_rings = 12;
    env.candidate_angles = 64;
    // env.candidate_radius_min = 0.15;
    // env.candidate_radius_max = 0.20;
    // env.target_quad_area = 0.015625;
    env.candidate_radius_min = 0.025;
    env.candidate_radius_max = 0.07;
    env.target_quad_area = 0.0009;
    env.reward_invalid = -0.1f;
    env.render_width = 1200;
    env.render_height = 900;
    env.render_target_fps = 144;
    env.render_show_frontier = true;
    env.render_show_candidates = true;
    env.render_line_thickness = 2.0f;
    env.render_point_radius = 4.0f;
    env.render_candidate_radius = 3.0f;
    env.rng = 12345;

    unsigned int rng = 42;
    double key_repeat_cd = 0.01;

    const int max_frontier = 1024;
    env.max_degree = 16;
    env.grid_res = 32;
    env.grid_cell_size = 1.0 / (float)env.grid_res;
    env.grid_cell_cap = 8;
    env.intersection_tol = 1e-3;

    quad_meshing_init(&env);
    env.observations = (unsigned char*)calloc(1 + 4 + max_frontier * (8 + 3 * env.max_degree + 1) + 2 + env.candidate_angles*env.candidate_rings * 8, sizeof(unsigned char));
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
                        best_idx = vidx;
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
