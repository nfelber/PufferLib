#include "quad_meshing.h"
#include "constants.h"
#include "geometry.h"
#include "mesh.h"

#include <stdlib.h>
#include <time.h>
#include <math.h>

static inline float clampf(float v, float lo, float hi) {
    return fmaxf(lo, fminf(hi, v));
}

static void compute_obs_layout(QuadMeshingEnv* env) {
    env->max_targets = env->max_existing_targets + env->max_candidates;
    int vertex_size = env->max_vertices * OBS_VERTEX_STRIDE;
    int edge_size = env->max_edges * OBS_EDGE_STRIDE;
    int frontier_size = env->max_frontier;
    int source_mask_size = env->max_frontier;
    int existing_target_size = env->max_existing_targets;
    int candidate_size = env->max_candidates * 2;
    int target_mask_size = env->max_frontier * env->max_targets;

    env->obs_vertex_offset = OBS_HEADER_SIZE;
    env->obs_edge_offset = env->obs_vertex_offset + vertex_size;
    env->obs_frontier_offset = env->obs_edge_offset + edge_size;
    env->obs_source_mask_offset = env->obs_frontier_offset + frontier_size;
    env->obs_existing_target_offset = env->obs_source_mask_offset + source_mask_size;
    env->obs_candidate_offset = env->obs_existing_target_offset + existing_target_size;
    env->obs_target_mask_offset = env->obs_candidate_offset + candidate_size;
    env->obs_size = env->obs_target_mask_offset + target_mask_size;
}

void quad_meshing_init(QuadMeshingEnv* env) {
    QM_ASSERT(env->max_vertices > 0);
    QM_ASSERT(env->max_edges > 0);
    QM_ASSERT(env->max_frontier > 0);
    QM_ASSERT(env->max_existing_targets > 0);
    QM_ASSERT(env->max_candidates > 0);
    QM_ASSERT(env->max_quads > 0);
    QM_ASSERT(env->max_degree > 0);
    QM_ASSERT(env->max_boundary_points > 0);
    QM_ASSERT(env->max_existing_targets <= env->max_vertices);
    QM_ASSERT(env->max_boundary_points <= env->max_vertices);
    QM_ASSERT(env->max_vertices <= DEFAULT_MAX_VERTICES);
    QM_ASSERT(env->max_edges <= DEFAULT_MAX_EDGES);
    QM_ASSERT(env->max_frontier <= DEFAULT_MAX_FRONTIER);
    QM_ASSERT(env->max_existing_targets <= DEFAULT_MAX_EXISTING_TARGETS);
    QM_ASSERT(env->max_candidates <= DEFAULT_MAX_CANDIDATES);
    QM_ASSERT(env->max_quads <= DEFAULT_MAX_QUADS);
    QM_ASSERT(env->max_degree <= DEFAULT_MAX_DEGREE);
    compute_obs_layout(env);
    QM_ASSERT(env->obs_buffer_size >= env->obs_size);
    QM_ASSERT(env->obs_size <= DEFAULT_OBS_SIZE);

    mesh_init(&env->mesh, env->max_vertices, env->max_edges, env->max_degree);

    if (env->boundary_poly == NULL) {
        env->boundary_poly = (Vector2*)calloc((size_t)env->max_boundary_points, sizeof(Vector2));
    }
    if (env->candidate_offsets == NULL) {
        env->candidate_offsets = (Vector2*)calloc((size_t)env->max_candidates, sizeof(Vector2));
    }
    if (env->frontier == NULL) {
        env->frontier = (int*)calloc((size_t)env->max_frontier, sizeof(int));
    }
    QM_ASSERT(env->boundary_poly != NULL);
    QM_ASSERT(env->candidate_offsets != NULL);
    QM_ASSERT(env->frontier != NULL);
}
static void build_boundary(QuadMeshingEnv* env) {
    env->boundary_poly_count = 0;
    QM_ASSERT(env->boundary_points <= env->max_boundary_points);
    QM_ASSERT(env->boundary_points <= env->max_vertices);
    if (env->boundary_type == BOUNDARY_CIRCLE) {
        int n = env->boundary_points;
        for (int i = 0; i < n; i++) {
            float a = (2.0f * PI * i) / n;
            env->boundary_poly[env->boundary_poly_count++] = (Vector2){
                0.5f + 0.45f * cosf(a),
                0.5f + 0.45f * sinf(a),
            };
        }
    } else {
        int per_side = env->boundary_points / 4;
        if (per_side < 2) per_side = 2;
        for (int i = 0; i < per_side; i++) {
            float t = i / (float)(per_side - 1);
            env->boundary_poly[env->boundary_poly_count++] = (Vector2){0.1f + 0.8f * t, 0.1f};
        }
        for (int i = 1; i < per_side; i++) {
            float t = i / (float)(per_side - 1);
            env->boundary_poly[env->boundary_poly_count++] = (Vector2){0.9f, 0.1f + 0.8f * t};
        }
        for (int i = 1; i < per_side; i++) {
            float t = i / (float)(per_side - 1);
            env->boundary_poly[env->boundary_poly_count++] = (Vector2){0.9f - 0.8f * t, 0.9f};
        }
        for (int i = 1; i < per_side - 1; i++) {
            float t = i / (float)(per_side - 1);
            env->boundary_poly[env->boundary_poly_count++] = (Vector2){0.1f, 0.9f - 0.8f * t};
        }
    }
}

static void init_candidates(QuadMeshingEnv* env) {
    env->candidate_count = 0;
    int rings = env->candidate_rings;
    int angles = env->candidate_angles;
    for (int r = 1; r <= rings; r++) {
        float frac = r / (float)rings;
        float radius = env->candidate_radius_min +
            frac * (env->candidate_radius_max - env->candidate_radius_min);
        for (int a = 0; a < angles; a++) {
            if (env->candidate_count >= env->max_candidates) return;
            float theta = (2.0f * PI * a) / angles;
            float rr = radius;
            env->candidate_offsets[env->candidate_count++] = (Vector2){
                rr * cosf(theta),
                rr * sinf(theta),
            };
        }
    }
}

static int random_int_range(unsigned int* rng, int max_exclusive) {
    if (max_exclusive <= 0) return 0;
    *rng = (*rng ^ (*rng << 13)) ^ (*rng >> 17) ^ (*rng << 5);
    return (int)(*rng % (unsigned int)max_exclusive);
}

static void update_frontier(QuadMeshingEnv* env) {
    env->num_frontier = 0;
    for (int i = 0; i < env->mesh.num_vertices; i++) {
        if (mesh_is_frontier(&env->mesh, i)) {
            if (env->num_frontier < env->max_frontier) {
                env->frontier[env->num_frontier++] = i;
            }
        }
    }
}

static void write_observation(QuadMeshingEnv* env) {
    QM_ASSERT(env->obs_buffer_size >= env->obs_size);
    float* obs = env->observations;
    for (int i = 0; i < env->obs_size; i++) obs[i] = 0.0f;

    obs[0] = (float)env->mesh.num_vertices;
    obs[1] = (float)env->mesh.num_edges;
    obs[2] = (float)env->num_frontier;
    obs[3] = (float)env->mesh.num_quads;
    obs[4] = (float)env->steps;
    obs[5] = (float)env->max_steps;
    obs[6] = env->last_reward;
    obs[7] = (float)env->last_valid;
    obs[8] = (float)env->last_source_slot;
    obs[9] = (float)env->last_target_slot;
    obs[10] = (float)env->last_target_is_candidate;
    obs[11] = (float)env->last_quad_completed;
    obs[12] = (float)env->last_done;
    obs[13] = (float)env->candidate_count;
    obs[14] = (float)env->max_vertices;
    obs[15] = (float)env->max_edges;
    obs[16] = (float)env->max_frontier;
    obs[17] = (float)env->max_targets;

    int idx = env->obs_vertex_offset;
    for (int i = 0; i < env->max_vertices; i++) {
        if (i < env->mesh.num_vertices) {
            MeshVertex v = env->mesh.vertices[i];
            obs[idx++] = v.x;
            obs[idx++] = v.y;
            obs[idx++] = (float)v.degree;
            obs[idx++] = (float)v.open_edges;
            obs[idx++] = 0.0f;
            obs[idx++] = 1.0f;
        } else {
            idx += OBS_VERTEX_STRIDE;
        }
    }

    idx = env->obs_edge_offset;
    for (int i = 0; i < env->max_edges; i++) {
        if (i < env->mesh.num_edges) {
            MeshEdge e = env->mesh.edges[i];
            Vector2 a = {env->mesh.vertices[e.a].x, env->mesh.vertices[e.a].y};
            Vector2 b = {env->mesh.vertices[e.b].x, env->mesh.vertices[e.b].y};
            obs[idx++] = a.x;
            obs[idx++] = a.y;
            obs[idx++] = b.x;
            obs[idx++] = b.y;
        } else {
            idx += OBS_EDGE_STRIDE;
        }
    }

    idx = env->obs_frontier_offset;
    for (int i = 0; i < env->max_frontier; i++) {
        obs[idx++] = (i < env->num_frontier) ? (float)env->frontier[i] : -1.0f;
    }

    idx = env->obs_source_mask_offset;
    for (int i = 0; i < env->max_frontier; i++) {
        obs[idx++] = (i < env->num_frontier) ? 1.0f : 0.0f;
    }

    idx = env->obs_existing_target_offset;
    for (int i = 0; i < env->max_existing_targets; i++) {
        obs[idx++] = (i < env->mesh.num_vertices) ? (float)i : -1.0f;
    }

    idx = env->obs_candidate_offset;
    for (int i = 0; i < env->max_candidates; i++) {
        if (i < env->candidate_count) {
            obs[idx++] = env->candidate_offsets[i].x;
            obs[idx++] = env->candidate_offsets[i].y;
        } else {
            idx += 2;
        }
    }

    idx = env->obs_target_mask_offset;
    for (int s = 0; s < env->max_frontier; s++) {
        int source = (s < env->num_frontier) ? env->frontier[s] : -1;
        for (int t = 0; t < env->max_targets; t++) {
            float valid = 0.0f;
            if (source >= 0) {
                if (t < env->max_existing_targets) {
                    int target = t;
                    if (mesh_validate_existing_target(&env->mesh, source, target) == MESH_VALID_OK) {
                        valid = 1.0f;
                    }
                } else {
                    int cidx = t - env->max_existing_targets;
                    if (cidx < env->candidate_count) {
                        Vector2 tp = v2_add(
                            (Vector2){env->mesh.vertices[source].x, env->mesh.vertices[source].y},
                            env->candidate_offsets[cidx]
                        );
                        if (mesh_validate_candidate_target(&env->mesh, source, tp) == MESH_VALID_OK) {
                            valid = 1.0f;
                        }
                    }
                }
            }
            obs[idx++] = valid;
        }
    }
}

void c_reset(QuadMeshingEnv* env) {
    env->steps = 0;
    env->num_frontier = 0;
    env->last_source_slot = -1;
    env->last_target_slot = -1;
    env->last_target_is_candidate = 0;
    env->last_target_pos = (Vector2){0};
    env->last_valid = 0;
    env->last_quad_completed = 0;
    env->last_done = 0;
    env->last_reward = 0.0f;

    QM_ASSERT(env->observations != NULL);
    QM_ASSERT(env->mesh.vertices != NULL);
    mesh_reset(&env->mesh);
    build_boundary(env);
    int boundary_ccw = polygon_is_ccw(env->boundary_poly, env->boundary_poly_count);
    for (int i = 0; i < env->boundary_poly_count; i++) {
        mesh_add_vertex(&env->mesh, env->boundary_poly[i]);
    }
    for (int i = 0; i < env->boundary_poly_count; i++) {
        int a = i;
        int b = (i + 1) % env->boundary_poly_count;
        int eidx = mesh_add_edge(&env->mesh, a, b);
        QM_ASSERT(eidx >= 0);
        env->mesh.edges[eidx].face_count = 1;
        mesh_set_boundary_edge_face(&env->mesh, eidx, a, b, boundary_ccw);
    }

    init_candidates(env);
    update_frontier(env);
    write_observation(env);

    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
}

void c_step(QuadMeshingEnv* env) {
    env->rewards[0] = env->reward_step;
    env->terminals[0] = 0.0f;
    env->last_valid = 0;
    env->last_quad_completed = 0;
    env->last_done = 0;
    env->last_reward = env->reward_step;
    env->last_source_slot = -1;
    env->last_target_slot = -1;
    env->last_target_is_candidate = 0;
    env->last_target_pos = (Vector2){0};

    int source_slot = (int)env->actions[0];
    int target_slot = (int)env->actions[1];
    if (source_slot < 0 || source_slot >= env->num_frontier) {
        env->rewards[0] += env->reward_invalid;
        env->last_reward = env->rewards[0];
        env->steps++;
        write_observation(env);
        return;
    }

    int source = env->frontier[source_slot];
    env->last_source_slot = source_slot;
    env->last_target_slot = target_slot;

    int valid = 0;
    int created_vertex = 0;
    int target_vertex = -1;
    Vector2 target_pos = {0};

    if (target_slot >= 0 && target_slot < env->max_existing_targets) {
        target_vertex = target_slot;
        if (mesh_validate_existing_target(&env->mesh, source, target_vertex) == MESH_VALID_OK) {
            target_pos = (Vector2){env->mesh.vertices[target_vertex].x, env->mesh.vertices[target_vertex].y};
            valid = 1;
        }
    } else if (target_slot >= env->max_existing_targets &&
        target_slot < env->max_existing_targets + env->candidate_count) {
        int cidx = target_slot - env->max_existing_targets;
        target_pos = v2_add(
            (Vector2){env->mesh.vertices[source].x, env->mesh.vertices[source].y},
            env->candidate_offsets[cidx]
        );
        if (mesh_validate_candidate_target(&env->mesh, source, target_pos) == MESH_VALID_OK) {
            valid = 1;
            created_vertex = 1;
        }
    }

    if (!valid) {
        env->rewards[0] += env->reward_invalid;
        env->last_reward = env->rewards[0];
        env->steps++;
        write_observation(env);
        return;
    }

    if (created_vertex) {
        target_vertex = mesh_add_vertex(&env->mesh, target_pos);
        if (target_vertex < 0) {
            env->rewards[0] += env->reward_invalid;
            env->last_reward = env->rewards[0];
            env->steps++;
            write_observation(env);
            return;
        }
    }

    env->last_target_is_candidate = created_vertex;
    env->last_target_pos = target_pos;
    env->last_valid = 1;

    int new_edge_idx = mesh_add_edge(&env->mesh, source, target_vertex);
    if (new_edge_idx < 0) {
        env->rewards[0] += env->reward_invalid;
        env->last_reward = env->rewards[0];
        env->steps++;
        write_observation(env);
        return;
    }
    env->rewards[0] += env->reward_edge;

    int cycles[64] = {0};
    int lengths[16] = {0};
    int face_added = 0;
    int cycle_count = mesh_detect_quads(&env->mesh, source, target_vertex, cycles, lengths, 16);
    for (int i = 0; i < cycle_count; i++) {
        int len = lengths[i];
        int* verts = &cycles[i * 4];
        mesh_register_face(&env->mesh, verts, len);
        face_added = 1;
        if (len == 4 && env->mesh.num_quads < env->max_quads) {
            env->mesh.num_quads++;
        }
    }
    cycle_count = mesh_detect_triangles(&env->mesh, source, target_vertex, cycles, lengths, 16);
    for (int i = 0; i < cycle_count; i++) {
        int len = lengths[i];
        int* verts = &cycles[i * 4];
        mesh_register_face(&env->mesh, verts, len);
        face_added = 1;
    }
    if (face_added) {
        env->rewards[0] += env->reward_quad;
        env->last_quad_completed = 1;
    }

    update_frontier(env);
    env->steps++;

    int done = (env->num_frontier == 0) || (env->steps >= env->max_steps);
    if (done) {
        env->rewards[0] += env->reward_complete;
        env->terminals[0] = 1.0f;
        env->last_done = 1;
    }

    env->last_reward = env->rewards[0];
    env->log.score += env->rewards[0];
    env->log.n += 1.0f;

    write_observation(env);
}

void c_render(QuadMeshingEnv* env) {
    if (!IsWindowReady()) {
        InitWindow(env->render_width, env->render_height, "PufferLib Quad Meshing");
        SetTargetFPS(60);
        env->camera.offset = (Vector2){env->render_width / 2.0f, env->render_height / 2.0f};
        env->camera.target = (Vector2){0.5f, 0.5f};
        env->camera.rotation = 0.0f;
        env->camera_zoom = fminf(env->render_width, env->render_height) * 0.9f;
        env->camera.zoom = env->camera_zoom;
    }

    if (IsKeyPressed(KEY_F)) env->render_show_frontier = !env->render_show_frontier;
    if (IsKeyPressed(KEY_C)) env->render_show_candidates = !env->render_show_candidates;
    if (IsKeyPressed(KEY_I)) env->render_show_indices = !env->render_show_indices;

    float zoom_delta = GetMouseWheelMove();
    if (zoom_delta != 0.0f) {
        env->camera.zoom = clampf(env->camera.zoom * (1.0f + zoom_delta * 0.1f), 50.0f, 5000.0f);
    }
    if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
        Vector2 delta = GetMouseDelta();
        env->camera.target.x -= delta.x / env->camera.zoom;
        env->camera.target.y -= delta.y / env->camera.zoom;
    }

    float line_thickness = env->render_line_thickness / env->camera.zoom;
    float point_radius = env->render_point_radius / env->camera.zoom;
    float candidate_radius = env->render_candidate_radius / env->camera.zoom;

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    BeginMode2D(env->camera);

    for (int i = 0; i < env->boundary_poly_count; i++) {
        Vector2 a = env->boundary_poly[i];
        Vector2 b = env->boundary_poly[(i + 1) % env->boundary_poly_count];
        DrawLineEx(a, b, line_thickness, (Color){60, 120, 120, 255});
    }

    for (int i = 0; i < env->mesh.num_edges; i++) {
        MeshEdge e = env->mesh.edges[i];
        Vector2 a = {env->mesh.vertices[e.a].x, env->mesh.vertices[e.a].y};
        Vector2 b = {env->mesh.vertices[e.b].x, env->mesh.vertices[e.b].y};
        Color c = (Color){220, 80, 80, 255};
        if (e.face_count == 1) c = (Color){255, 200, 80, 255};
        if (e.face_count >= 2) c = (Color){80, 200, 120, 255};
        DrawLineEx(a, b, line_thickness, c);
    }

    int selecting_target = env->ui_selecting_target && env->ui_pending_source >= 0;
    int source_slot = selecting_target ? env->ui_pending_source : -1;
    int source = (source_slot >= 0 && source_slot < env->num_frontier) ? env->frontier[source_slot] : -1;

    for (int i = 0; i < env->mesh.num_vertices; i++) {
        Vector2 p = {env->mesh.vertices[i].x, env->mesh.vertices[i].y};
        Color c = (Color){240, 240, 240, 255};
        if (!selecting_target) {
            if (mesh_is_frontier(&env->mesh, i)) c = (Color){0, 140, 255, 255};
        } else if (source >= 0) {
            MeshValidReason r = mesh_validate_existing_target(&env->mesh, source, i);
            c = (r == MESH_VALID_OK) ? (Color){0, 220, 120, 255} : (Color){220, 80, 80, 255};
        }
        DrawCircleV(p, point_radius, c);
        if (env->render_show_indices) {
            DrawText(TextFormat("%d", i), p.x + 0.005f, p.y + 0.005f, 10, RAYWHITE);
        }
    }

    if (selecting_target && env->render_show_candidates && source >= 0) {
        Vector2 s = {env->mesh.vertices[source].x, env->mesh.vertices[source].y};
        for (int i = 0; i < env->candidate_count; i++) {
            Vector2 cp = v2_add(s, env->candidate_offsets[i]);
            MeshValidReason r = mesh_validate_candidate_target(&env->mesh, source, cp);
            Color c = (r == MESH_VALID_OK) ? (Color){0, 220, 120, 220} : (Color){220, 80, 80, 220};
            DrawCircleV(cp, candidate_radius, c);
        }
    }

    EndMode2D();

    if (selecting_target && source >= 0) {
        Vector2 mouse = GetMousePosition();
        Vector2 mouse_world = GetScreenToWorld2D(mouse, env->camera);
        float hover_radius = env->camera.zoom > 0 ? 10.0f / env->camera.zoom : 0.03f;
        float best = 1e9f;
        int best_target = -1;
        int best_is_candidate = 0;
        int best_candidate_idx = -1;
        for (int i = 0; i < env->candidate_count; i++) {
            Vector2 cp = v2_add(
                (Vector2){env->mesh.vertices[source].x, env->mesh.vertices[source].y},
                env->candidate_offsets[i]
            );
            float d = v2_len(v2_sub(mouse_world, cp));
            if (d < best) {
                best = d;
                best_is_candidate = 1;
                best_candidate_idx = i;
                best_target = env->max_existing_targets + i;
            }
        }
        for (int i = 0; i < env->mesh.num_vertices; i++) {
            Vector2 p = {env->mesh.vertices[i].x, env->mesh.vertices[i].y};
            float d = v2_len(v2_sub(mouse_world, p));
            if (d < best) {
                best = d;
                best_is_candidate = 0;
                best_target = i;
            }
        }
        if (best_target >= 0 && best <= hover_radius) {
            MeshValidReason r = MESH_VALID_OUT_OF_BOUNDS;
            if (best_is_candidate) {
                Vector2 tp = v2_add(
                    (Vector2){env->mesh.vertices[source].x, env->mesh.vertices[source].y},
                    env->candidate_offsets[best_candidate_idx]
                );
                r = mesh_validate_candidate_target(&env->mesh, source, tp);
            } else {
                r = mesh_validate_existing_target(&env->mesh, source, best_target);
            }
            DrawText(TextFormat("Target: %s", mesh_valid_reason_str(r)), 20, 20, 20, RAYWHITE);
        }
    }
    EndDrawing();
}

void c_close(QuadMeshingEnv* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    free(env->boundary_poly);
    free(env->candidate_offsets);
    free(env->frontier);
    mesh_free(&env->mesh);
}

int main() {
    QuadMeshingEnv env = {0};
    env.num_agents = 1;
    env.max_steps = 2048;
    env.max_degree = DEFAULT_MAX_DEGREE;
    env.max_vertices = DEFAULT_MAX_VERTICES;
    env.max_edges = DEFAULT_MAX_EDGES;
    env.max_frontier = DEFAULT_MAX_FRONTIER;
    env.max_existing_targets = DEFAULT_MAX_EXISTING_TARGETS;
    env.max_candidates = DEFAULT_MAX_CANDIDATES;
    env.max_quads = DEFAULT_MAX_QUADS;
    env.max_boundary_points = DEFAULT_MAX_BOUNDARY_POINTS;
    env.obs_buffer_size = DEFAULT_OBS_SIZE;
    env.boundary_type = BOUNDARY_SQUARE;
    env.boundary_points = 24;
    env.candidate_rings = 4;
    env.candidate_angles = 16;
    env.candidate_radius_min = 0.03f;
    env.candidate_radius_max = 0.20f;
    env.reward_step = -0.001f;
    env.reward_edge = 0.01f;
    env.reward_quad = 0.25f;
    env.reward_complete = 2.0f;
    env.reward_invalid = -0.05f;
    env.render_width = 1200;
    env.render_height = 900;
    env.render_show_frontier = true;
    env.render_show_candidates = true;
    env.render_show_indices = false;
    env.render_line_thickness = 2.0f;
    env.render_point_radius = 4.0f;
    env.render_candidate_radius = 3.0f;
    env.rng = 12345;

    quad_meshing_init(&env);
    env.observations = (float*)calloc((size_t)env.obs_size, sizeof(float));
    env.actions = (float*)calloc(2, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    c_reset(&env);
    c_render(&env);

    int pending_source = -1;
    while (!WindowShouldClose()) {
        float pick_radius = env.camera.zoom > 0 ? 10.0f / env.camera.zoom : 0.03f;
        env.ui_pending_source = pending_source;
        env.ui_selecting_target = pending_source >= 0;
        if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
            pending_source = -1;
            env.last_source_slot = -1;
            env.ui_pending_source = -1;
            env.ui_selecting_target = 0;
        }
        if (IsKeyPressed(KEY_TAB)) {
            if (env.num_frontier > 0) {
                int start = random_int_range(&env.rng, env.num_frontier);
                int picked_source = -1;
                for (int i = 0; i < env.num_frontier; i++) {
                    int idx = (start + i) % env.num_frontier;
                    int source = env.frontier[idx];
                    if (!mesh_is_frontier(&env.mesh, source)) continue;
                    int* targets = (int*)calloc((size_t)env.max_targets, sizeof(int));
                    QM_ASSERT(targets != NULL);
                    int target_count = 0;
                    for (int t = 0; t < env.max_existing_targets; t++) {
                        if (mesh_validate_existing_target(&env.mesh, source, t) == MESH_VALID_OK) {
                            targets[target_count++] = t;
                        }
                    }
                    for (int c = 0; c < env.candidate_count; c++) {
                        Vector2 tp = v2_add(
                            (Vector2){env.mesh.vertices[source].x, env.mesh.vertices[source].y},
                            env.candidate_offsets[c]
                        );
                        if (mesh_validate_candidate_target(&env.mesh, source, tp) == MESH_VALID_OK) {
                            targets[target_count++] = env.max_existing_targets + c;
                        }
                    }
                    if (target_count > 0) {
                        picked_source = idx;
                        int target_idx = targets[random_int_range(&env.rng, target_count)];
                        env.actions[0] = (float)picked_source;
                        env.actions[1] = (float)target_idx;
                        c_step(&env);
                        pending_source = -1;
                        env.last_source_slot = -1;
                        env.ui_pending_source = -1;
                        env.ui_selecting_target = 0;
                        free(targets);
                        break;
                    }
                    free(targets);
                }
                if (picked_source < 0) {
                    pending_source = -1;
                    env.ui_pending_source = -1;
                    env.ui_selecting_target = 0;
                }
            }
        }

        if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
            pending_source = -1;
            env.last_source_slot = -1;
            env.ui_pending_source = -1;
            env.ui_selecting_target = 0;
        }

        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && IsWindowReady()) {
            Vector2 mouse_world = GetScreenToWorld2D(GetMousePosition(), env.camera);
            if (pending_source < 0) {
                float best = 1e9f;
                int best_slot = -1;
                for (int i = 0; i < env.num_frontier; i++) {
                    int vidx = env.frontier[i];
                    Vector2 p = {env.mesh.vertices[vidx].x, env.mesh.vertices[vidx].y};
                    float d = v2_len(v2_sub(mouse_world, p));
                    if (d < best) {
                        best = d;
                        best_slot = i;
                    }
                }
                if (best_slot >= 0 && best <= pick_radius) {
                    pending_source = best_slot;
                    env.last_source_slot = pending_source;
                    env.last_target_slot = -1;
                }
            } else {
                int source = env.frontier[pending_source];
                Vector2 s = {env.mesh.vertices[source].x, env.mesh.vertices[source].y};
                float best = 1e9f;
                int best_target = -1;

                for (int i = 0; i < env.candidate_count; i++) {
                    Vector2 cp = v2_add(s, env.candidate_offsets[i]);
                    float d = v2_len(v2_sub(mouse_world, cp));
                    if (d < best) {
                        best = d;
                        best_target = env.max_existing_targets + i;
                    }
                }

                if (best > pick_radius) {
                    for (int i = 0; i < env.mesh.num_vertices; i++) {
                        Vector2 p = {env.mesh.vertices[i].x, env.mesh.vertices[i].y};
                        float d = v2_len(v2_sub(mouse_world, p));
                        if (d < best) {
                            best = d;
                            best_target = i;
                        }
                    }
                }

                if (best_target >= 0 && best <= pick_radius) {
                    env.actions[0] = (float)pending_source;
                    env.actions[1] = (float)best_target;
                    c_step(&env);
                }
                pending_source = -1;
                env.last_source_slot = -1;
                env.ui_pending_source = -1;
                env.ui_selecting_target = 0;
            }
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
