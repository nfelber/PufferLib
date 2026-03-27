#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#include "geometry.h"
#include "rendering.h"

// Only use floats!
typedef struct {
    float perf;           // Recommended 0-1 normalized single real number perf metric
    float score;          // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    float n;              // Required as the last field 
} Log;

typedef struct {
    Log log;                     // Required field
    float* observations;         // Required field. Ensure type matches in .py and .c
    float* actions;              // Required field. Ensure type matches in .py and .c
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field

    // Env config
    int episode_max_length;
    Polygon2D starting_boundary;
    float target_quad_area;
    float action_radius;
    float observation_radius;
    int observation_density;

    // Env state
    Polygon2D boundary;
    Polygon2D quad;
    Mesh2D mesh;
    int active_vertex;
    int episode_length;
    float episode_return;
} QuadMeshing;

void init(QuadMeshing* env) {
    // Square boundary
    const int n = 8;
    const int N = 4*n;
    Vec2Array_init(&env->starting_boundary.vertices);
    Vec2Array_reserve(&env->starting_boundary.vertices, N);
    Vec2 dir = {1.0 / n, 0.0};
    Vec2 pos = {0.0, 0.0};
    for (int s=0; s<4; ++s) {
        for (int i=0; i<n; ++i) {
            Vec2Array_push(&env->starting_boundary.vertices, pos);
            pos = add2(pos, dir);
        }
        const float x = dir.x;
        dir.x = dir.y;
        dir.y = -x;
    }
    env->starting_boundary.isCCW = is_polygon_ccw(env->starting_boundary);    

    // Allocate boundary
    Vec2Array_init(&env->boundary.vertices);
    Vec2Array_reserve(&env->boundary.vertices, N);
    env->boundary.isCCW = env->starting_boundary.isCCW;

    // Allocate quad
    Vec2Array_init(&env->quad.vertices);
    Vec2Array_resize(&env->quad.vertices, 4);
    env->quad.isCCW = true;

    env->episode_max_length = 2*(n*n);    
    env->target_quad_area = 1.0 / (n*n);    
    env->active_vertex = 0;    
}

void add_log(QuadMeshing* env) {
    env->log.perf += (env->episode_length < env->episode_max_length) ?
        env->episode_return / polygon2D_area(env->starting_boundary) : 0;
    env->log.score += env->episode_return;
    env->log.episode_length += env->episode_length;
    env->log.episode_return += env->episode_return;
    env->log.n++;
}

static Frame2D compute_active_local_frame(QuadMeshing* env) {
    Vec2 v_prev = Polygon2D_neighbor(env->boundary, env->active_vertex, -1);
    Vec2 v      = env->boundary.vertices.data[env->active_vertex];
    Vec2 v_next = Polygon2D_neighbor(env->boundary, env->active_vertex, 1);

    Vec2 e1 = safe_normalize(sub2(v_prev, v));
    Vec2 e2 = safe_normalize(sub2(v_next, v));

    Vec2 bis = add2(e1, e2);

    // 90 deg angle
    if (norm2(bis) < 1e-6) {
        bis = (Vec2){ -e1.y, e1.x };
    }

    bis = safe_normalize(bis);

    // Build orthonormal frame
    Frame2D f;
    f.origin = v;
    f.y = bis;
    f.x = (Vec2){ bis.y, -bis.x };

    if (!env->boundary.isCCW) {
        f.x = scalmul2(f.x, -1.0);
    }

    return f;
}

void compute_observations(QuadMeshing* env) {
    assert(env->boundary.vertices.size > 0);

    // Determine active vertex
    env->active_vertex = 0;
    float angleMin = polygonInteriorAngle(env->boundary, 0);
    for (int i=1; i<env->boundary.vertices.size; ++i) {
        const float angle = polygonInteriorAngle(env->boundary, i);
        if (angle < angleMin) {
            angleMin = angle;
            env->active_vertex = i;
        }
    }

    // Make SDF observations
    Frame2D frame = compute_active_local_frame(env);
    int N = env->observation_density;

    int obs_idx = 0;
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            Vec2 query_local = {
                env->observation_radius * ((x + 0.5f) / N * 2.0 - 1.0),
                env->observation_radius * ((y + 0.5f) / N * 2.0 - 1.0)
            };
            Vec2 query_world = local_to_world(frame, query_local);
            env->observations[obs_idx++] = clampf(eval_polygon2D_sdf(env->boundary, query_world), -1.0, 1.0);
        }
    }
}

float compute_reward(QuadMeshing* env, Polygon2D quad) {
    assert(quad.vertices.size == 4);

    // Element quality
    float Emin2 = FLT_MAX;
    float angleMin = FLT_MAX;
    float angleMax = FLT_MIN;
    for (int i=quad.vertices.size-1, j=0; j<quad.vertices.size; i=j, ++j) {
        Segment2D edge = {quad.vertices.data[i], quad.vertices.data[j]};
        Emin2 = fmin(Emin2, segment2D_sqrd_length(edge));
        const float angle = polygonInteriorAngle(quad, j);
        angleMin = fmin(angleMin, angle);
        angleMax = fmax(angleMax, angle);
    }
    const float Emin = sqrtf(Emin2);
    const float Dmax = sqrtf(fmax(
      segment2D_sqrd_length((Segment2D){quad.vertices.data[0], quad.vertices.data[2]}),
      segment2D_sqrd_length((Segment2D){quad.vertices.data[1], quad.vertices.data[3]})
    ));
    const float eq = sqrtf(sqrtf(2) * Emin * angleMin / (Dmax * angleMax));

    // Density quality
    const float alpha = 4096.0 * 10.0;
    const float A = polygon2D_area(quad);
    const float Ad = A - env->target_quad_area;
    float dq = 1.0 / (1.0 + alpha * Ad*Ad);

    return eq * dq;
}

void c_reset(QuadMeshing* env) {
    // Reset episode metrics
    env->episode_length = 0;
    env->episode_return = 0.0;

    // Reset boundary
    Vec2Array_resize(&env->boundary.vertices, env->starting_boundary.vertices.size);
    memcpy(env->boundary.vertices.data, env->starting_boundary.vertices.data,
           env->starting_boundary.vertices.size * sizeof(Vec2));

    // Reset mesh
    mesh2D_free(&env->mesh);

    // Starting observation 
    compute_observations(env);
}

void c_step(QuadMeshing* env) {
    env->episode_length++;
    env->rewards[0] = 0;
    env->terminals[0] = 0;

    const float action_kind = env->actions[0];
    const float action_angle = env->actions[1];
    const float action_radius = env->actions[2];

    bool action_valid = false;
    if (action_kind < -0.5) {
        // Close left
        env->quad.vertices.data[0] = Polygon2D_neighbor(env->boundary, env->active_vertex, -2);
        env->quad.vertices.data[1] = Polygon2D_neighbor(env->boundary, env->active_vertex, -1);
        env->quad.vertices.data[2] = env->boundary.vertices.data[env->active_vertex];
        env->quad.vertices.data[3] = Polygon2D_neighbor(env->boundary, env->active_vertex,  1);
        const Segment2D new_edge = {env->quad.vertices.data[0], env->quad.vertices.data[3]};

        action_valid = !polygon2D_segment_intersect(env->boundary, new_edge, NULL, 1e-6);
        if (action_valid) {
            if (env->active_vertex == 0) {
                env->boundary.vertices.data[0] = env->boundary.vertices.data[env->boundary.vertices.size-2];
                env->boundary.vertices.size -= 2;
            } else {
                Vec2Array_remove_range(&env->boundary.vertices, env->active_vertex-1, env->active_vertex);
            }
        }
    } else if (action_kind > 0.5) {
        // Close right
        env->quad.vertices.data[0] = Polygon2D_neighbor(env->boundary, env->active_vertex, -1);
        env->quad.vertices.data[1] = env->boundary.vertices.data[env->active_vertex];
        env->quad.vertices.data[2] = Polygon2D_neighbor(env->boundary, env->active_vertex,  1);
        env->quad.vertices.data[3] = Polygon2D_neighbor(env->boundary, env->active_vertex,  2);
        const Segment2D new_edge = {env->quad.vertices.data[0], env->quad.vertices.data[3]};

        action_valid = !polygon2D_segment_intersect(env->boundary, new_edge, NULL, 1e-6);
        if (action_valid) {
            if (env->active_vertex == env->boundary.vertices.size-1) {
                env->boundary.vertices.data[0] = env->boundary.vertices.data[env->boundary.vertices.size-2];
                env->boundary.vertices.size -= 2;
            } else {
                Vec2Array_remove_range(&env->boundary.vertices, env->active_vertex, env->active_vertex+1);
            }
        }
    } else {
        env->quad.vertices.data[0] = Polygon2D_neighbor(env->boundary, env->active_vertex, -1);
        env->quad.vertices.data[1] = env->boundary.vertices.data[env->active_vertex];
        env->quad.vertices.data[2] = Polygon2D_neighbor(env->boundary, env->active_vertex,  1);
        const float t = 0.5 * (1.0 + action_angle);
        const float r = env->action_radius * action_radius;
        const Vec2 dir = slerp2(
            sub2(env->quad.vertices.data[2], env->quad.vertices.data[1]),
            sub2(env->quad.vertices.data[0], env->quad.vertices.data[1]),
            t, false
        );
        env->quad.vertices.data[3] = add2(env->quad.vertices.data[1], scalmul2(dir, r));

        const Segment2D new_edge1 = {env->quad.vertices.data[0], env->quad.vertices.data[3]};
        const Segment2D new_edge2 = {env->quad.vertices.data[2], env->quad.vertices.data[3]};

        if (!polygon2D_segment_intersect(env->boundary, new_edge1, NULL, 1e-6) &&
            !polygon2D_segment_intersect(env->boundary, new_edge2, NULL, 1e-6) &&
            eval_polygon2D_sdf(env->boundary, env->quad.vertices.data[3]) < 0.0)
        {
            action_valid = true;
            env->boundary.vertices.data[env->active_vertex] = env->quad.vertices.data[3];
        }
    }

    env->rewards[0] = action_valid ? compute_reward(env, env->quad) : -0.1;
    env->log.score += env->rewards[0];

    // Add latest quad to mesh
    // TODO: Only when rendering
    if (action_valid) {
      // TODO: deduplicate vertices
      size_t v0 = mesh2D_add_vertex(&env->mesh, env->quad.vertices.data[0]);
      size_t v1 = mesh2D_add_vertex(&env->mesh, env->quad.vertices.data[1]);
      size_t v2 = mesh2D_add_vertex(&env->mesh, env->quad.vertices.data[2]);
      size_t v3 = mesh2D_add_vertex(&env->mesh, env->quad.vertices.data[3]);
      mesh2D_add_edge(&env->mesh, v0, v1);
      mesh2D_add_edge(&env->mesh, v1, v2);
      mesh2D_add_edge(&env->mesh, v2, v3);
      mesh2D_add_edge(&env->mesh, v3, v0);
    }

    // TODO: don't ignore last quad / tri
    if (env->boundary.vertices.size <= 4 || env->episode_length == env->episode_max_length) {
        add_log(env);
        c_reset(env);
        env->terminals[0] = 1;
    }

    compute_observations(env);
}

void c_render(QuadMeshing* env) {
    if (!IsWindowReady()) {
        InitWindow(1080, 720, "PufferLib QuadMeshing");
        SetTargetFPS(4);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }

    BeginDrawing();
    ClearBackground(RAYWHITE);

    // ------------------------------------------------------------
    // Compute bounding box (boundary + mesh)
    // ------------------------------------------------------------
    Vec2 min = { FLT_MAX, FLT_MAX };
    Vec2 max = { -FLT_MAX, -FLT_MAX };

    // Boundary
    for (int i = 0; i < env->boundary.vertices.size; i++) {
        Vec2 v = env->boundary.vertices.data[i];
        if (v.x < min.x) min.x = v.x;
        if (v.y < min.y) min.y = v.y;
        if (v.x > max.x) max.x = v.x;
        if (v.y > max.y) max.y = v.y;
    }

    // Mesh vertices
    for (size_t i = 0; i < env->mesh.vertices.size; i++) {
        Vec2 v = env->mesh.vertices.data[i];
        if (v.x < min.x) min.x = v.x;
        if (v.y < min.y) min.y = v.y;
        if (v.x > max.x) max.x = v.x;
        if (v.y > max.y) max.y = v.y;
    }

    RenderContext ctx = compute_render_context(min, max);

    if (IsKeyDown(KEY_S)) {
        draw_sdf(&env->boundary, &ctx);
    }

    draw_mesh(&env->mesh, &ctx);
    draw_boundary(&env->boundary, &ctx);

    // Active vertex
    DrawCircleV(world_to_screen(env->boundary.vertices.data[env->active_vertex], &ctx), 8.0, RED);

    // SDF grid
    Frame2D frame = compute_active_local_frame(env);
    int N = env->observation_density;
    int obs_idx = 0;
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            Vec2 query_local = {
                env->observation_radius * ((x + 0.5f) / N * 2.0 - 1.0),
                env->observation_radius * ((y + 0.5f) / N * 2.0 - 1.0)
            };
            Vec2 query_world = local_to_world(frame, query_local);
            float d = env->observations[obs_idx++];
            Color c = {
                (unsigned char)(255 * clamp01(0.5 - 5.0*d * 0.1)),
                (unsigned char)(255 * clamp01(0.5 + 5.0*d * 0.4)),
                (unsigned char)(255 * clamp01(0.5 - 5.0*d * 0.7)),
                255
            };
            DrawCircleV(world_to_screen(query_world, &ctx), 3.0, c);
        }
    }

    DrawText(TextFormat("S: SDF | ESC: Quit | Score: %f", env->log.score), 10, 10, 20, DARKGRAY);

    EndDrawing();
}

void c_close(QuadMeshing* env) {
    Vec2Array_free(&env->starting_boundary.vertices);
    Vec2Array_free(&env->boundary.vertices);
    Vec2Array_free(&env->quad.vertices);
    mesh2D_free(&env->mesh);
    if (IsWindowReady()) {
        CloseWindow();
    }
}
