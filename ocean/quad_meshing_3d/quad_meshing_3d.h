#pragma once

#include "../quad_meshing/serialization.h"
#include "continuous_geodesic.h"
#include "qmsurface.h"

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef MAX_FRONTIER_SIZE
#define MAX_FRONTIER_SIZE 2048
#endif

#ifndef MAX_DEGREE
#define MAX_DEGREE 16
#endif

#ifndef MAX_TARGETS
#define MAX_TARGETS 2048
#endif

#ifndef OBS_SIZE
#define OBS_SIZE ( \
  1 + \
  4 + \
  2 + \
  2 + \
  MAX_FRONTIER_SIZE * 24 + \
  MAX_FRONTIER_SIZE * MAX_DEGREE * 2 + \
  2 + \
  2 + \
  2 + \
  MAX_TARGETS * 29 \
)
#endif

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef enum {
    QM3_CANDIDATE_VALID = 0,
    QM3_CANDIDATE_UNREACHABLE = 1,
    QM3_CANDIDATE_INTERSECT = 2,
    QM3_CANDIDATE_DISABLED = 3,
    QM3_CANDIDATE_SAME_VERTEX = 4,
    QM3_CANDIDATE_EXISTING_EDGE = 5,
    QM3_CANDIDATE_MAX_DEGREE = 6,
    QM3_CANDIDATE_TRIANGLE = 7,
} Qm3CandidateValidity;

typedef enum {
    QM3_TARGET_NEW_SAMPLE = 0,
    QM3_TARGET_EXISTING_FRONTIER = 1,
} Qm3TargetKind;

typedef enum {
    QM3_PHASE_SOURCE = 0,
    QM3_PHASE_TARGET = 1,
} Qm3Phase;

typedef enum {
    QM3_ACTION_NONE = 0,
    QM3_ACTION_SOURCE_SELECTED = 1,
    QM3_ACTION_TARGET_COMMITTED = 2,
    QM3_ACTION_INVALID_SOURCE = 3,
    QM3_ACTION_INVALID_TARGET = 4,
    QM3_ACTION_NO_VALID_TARGET = 5,
} Qm3ActionResult;

typedef enum {
    QM3_COST_OVERLAY_OFF = 0,
    QM3_COST_OVERLAY_EDGE_LENGTH = 1,
    QM3_COST_OVERLAY_PATH_CHORD = 2,
    QM3_COST_OVERLAY_ALIGNMENT = 3,
    QM3_COST_OVERLAY_ANGLE = 4,
} Qm3CostOverlayMode;

typedef struct {
    Qm3TargetKind kind;
    uint32_t graph_vertex;
    uint32_t sample_id;
    uint32_t path_offset;
    uint32_t path_count;
    uint32_t segment_offset;
    uint32_t segment_count;
    int32_t intersect_edge;
    int32_t intersect_segment;
    float path_length;
    Qm3CandidateValidity validity;
    unsigned char path_ok;
} Qm3TargetCandidate;

typedef struct {
    uint32_t edge;
    Qm3Vec2 a;
    Qm3Vec2 b;
} Qm3GraphTriSegment;

typedef struct {
    Qm3GraphTriSegment* data;
    uint32_t count;
    uint32_t cap;
} Qm3GraphTriSegmentArray;

typedef struct {
    uint32_t graph_vertex;
    uint32_t incident_edge;
    Qm3Vec2 p;
} Qm3GraphTriVertex;

typedef struct {
    Qm3GraphTriVertex* data;
    uint32_t count;
    uint32_t cap;
} Qm3GraphTriVertexArray;

enum {
    QM3_LOOP_EDGE_BLOCKED = 1,
    QM3_LOOP_EDGE_LEFT = 2,
    QM3_LOOP_EDGE_RIGHT = 4,
};

typedef struct {
    uint32_t disabled_samples;
    uint32_t left_tris;
    uint32_t right_tris;
    float left_area;
    float right_area;
    int chosen_side;
    uint32_t classification_conflicts;
    uint32_t flood_conflicts;
} Qm3LoopRemovalStats;

typedef struct {
    unsigned char* flags;
    unsigned char* boundary_tri;
    unsigned char* side_mark;
    unsigned char* conflict_tri;
    uint32_t tri_count;
} Qm3LoopDebugData;

typedef struct {
    int num_agents;
    Log log;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned int rng;

    Qm3Surface surface;
    Qm3Mesh mesh;
    Qm3SurfaceTopo surface_topo;
    Qm3PropGraph prop_graph;
    Qm3FMMContext fmm;
    Qm3ContinuousContext continuous_ctx;
    Qm3GraphTriSegmentArray* graph_tri_segments;
    Qm3GraphTriVertexArray* graph_tri_vertices;
    int32_t* sample_to_graph;
    unsigned char* sample_disabled;
    uint32_t graph_tri_segment_count;
    uint32_t graph_tri_vertex_count;
    bool geodesic_ready;
    Qm3Phase phase;
    Qm3ActionResult last_action_result;
    Qm3TargetCandidate* candidates;
    uint32_t candidate_count;
    uint32_t candidate_cap;
    Qm3Path candidate_path_points;
    Qm3PathSegmentArray candidate_path_segments;
    uint32_t continuous_paths_ok;
    uint32_t valid_candidate_count;
    uint32_t candidate_existing_count;
    uint32_t candidate_sample_count;
    uint32_t invalid_unreachable_count;
    uint32_t invalid_intersect_count;
    double timing_target_query_ms;
    double timing_path_build_ms;
    double timing_intersection_ms;
    double timing_loop_removal_ms;
    uint32_t timing_intersection_tests;
    uint32_t disabled_sample_count;
    Qm3LoopRemovalStats loop_stats;
    Qm3LoopDebugData loop_debug;
    int loop_debug_mode;
    int32_t source_frontier_idx;
    int32_t target_candidate_idx;
    int32_t debug_candidate_idx;
    int episode_length;
    float episode_return;
    int episode_max_length;
    Qm3Path selected_path;
    bool selected_path_ok;
    const char** shape_paths;
    int shape_count;
    int loaded_shape;
    int max_degree;
    float candidate_radius_ratio;
    float candidate_radius;
    float geodesic_steiner_spacing_ratio;
    float target_edge_length_ratio;
    float target_edge_length;
    float target_quad_area;
    float starting_frontier_edge_length;
    float episode_max_length_ratio;
    bool prevent_triangles;

    float reward_invalid;
    float reward_incomplete;
    float reward_triangle;
    float base_quad_reward;
    float potential_beta;
    float potential_gamma;
    float frontier_quality_weight;
    float frontier_edge_length_weight;
    float frontier_alignment_weight;
    float frontier_angle_weight;
    float frontier_size_pressure_weight;
    float degree_pressure_weight;
    float safe_frontier_size_ratio;
    float safe_degree_ratio;
    float potential;
    float last_face_area;
    float last_area_quality;
    float last_element_quality;
    float last_planarity_quality;
    float last_curvature_quality;
    float last_frontier_potential;
    float last_reward;

    int render_target_fps;
    int render_width;
    int render_height;
    bool render_show_mesh;
    bool render_mesh_opaque;
    bool render_show_samples;
    bool render_show_vertices;
    bool render_show_normals;
    bool render_show_cross_field;
    bool render_show_frontier;
    bool render_show_graph;
    bool render_show_graph_paths;
    bool render_show_candidates;
    bool render_debug_validity;
    Qm3CostOverlayMode render_cost_overlay_mode;
    float render_point_radius;
    float render_normal_length;
    float render_cross_field_length;

    Camera3D camera;
} QuadMeshing3DEnv;

static Vector3 qm3_v3(Qm3Vec3 v) {
    return (Vector3){v.x, v.y, v.z};
}

static double qm3_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static const char* qm3_candidate_validity_str(Qm3CandidateValidity reason) {
    switch (reason) {
        case QM3_CANDIDATE_VALID: return "valid";
        case QM3_CANDIDATE_UNREACHABLE: return "unreachable";
        case QM3_CANDIDATE_INTERSECT: return "intersect";
        case QM3_CANDIDATE_DISABLED: return "disabled";
        case QM3_CANDIDATE_SAME_VERTEX: return "same_vertex";
        case QM3_CANDIDATE_EXISTING_EDGE: return "existing_edge";
        case QM3_CANDIDATE_MAX_DEGREE: return "max_degree";
        case QM3_CANDIDATE_TRIANGLE: return "triangle";
        default: return "unknown";
    }
}

static Color qm3_candidate_validity_color(Qm3CandidateValidity reason) {
    switch (reason) {
        case QM3_CANDIDATE_VALID: return (Color){50, 235, 125, 230};
        case QM3_CANDIDATE_UNREACHABLE: return (Color){255, 100, 50, 240};
        case QM3_CANDIDATE_INTERSECT: return (Color){190, 90, 255, 240};
        case QM3_CANDIDATE_DISABLED: return (Color){130, 130, 130, 220};
        case QM3_CANDIDATE_SAME_VERTEX: return (Color){255, 210, 70, 240};
        case QM3_CANDIDATE_EXISTING_EDGE: return (Color){255, 150, 70, 240};
        case QM3_CANDIDATE_MAX_DEGREE: return (Color){255, 70, 120, 240};
        case QM3_CANDIDATE_TRIANGLE: return (Color){255, 120, 35, 240};
        default: return (Color){220, 220, 220, 200};
    }
}

static Vector3 qm3_candidate_position(const QuadMeshing3DEnv* env, const Qm3TargetCandidate* candidate) {
    if (candidate->kind == QM3_TARGET_EXISTING_FRONTIER && candidate->graph_vertex < env->mesh.vertex_count) {
        return qm3_v3(env->mesh.vertices[candidate->graph_vertex].pos);
    }
    if (candidate->sample_id < env->surface.sample_count) return qm3_v3(env->surface.samples[candidate->sample_id].p);
    return (Vector3){0.0f, 0.0f, 0.0f};
}

static const char* qm3_candidate_kind_str(Qm3TargetKind kind) {
    switch (kind) {
        case QM3_TARGET_NEW_SAMPLE: return "sample";
        case QM3_TARGET_EXISTING_FRONTIER: return "existing";
        default: return "unknown";
    }
}

static const char* qm3_phase_str(Qm3Phase phase) {
    switch (phase) {
        case QM3_PHASE_SOURCE: return "source";
        case QM3_PHASE_TARGET: return "target";
        default: return "unknown";
    }
}

static const char* qm3_action_result_str(Qm3ActionResult result) {
    switch (result) {
        case QM3_ACTION_NONE: return "none";
        case QM3_ACTION_SOURCE_SELECTED: return "source_selected";
        case QM3_ACTION_TARGET_COMMITTED: return "target_committed";
        case QM3_ACTION_INVALID_SOURCE: return "invalid_source";
        case QM3_ACTION_INVALID_TARGET: return "invalid_target";
        case QM3_ACTION_NO_VALID_TARGET: return "no_valid_target";
        default: return "unknown";
    }
}

static const char* qm3_loop_debug_mode_str(int mode) {
    switch (mode) {
        case 0: return "off";
        case 1: return "boundary";
        case 2: return "cuts";
        case 3: return "flood";
        case 4: return "conflicts";
        case 5: return "disabled";
        case 6: return "all";
        default: return "unknown";
    }
}

static const char* qm3_cost_overlay_mode_str(Qm3CostOverlayMode mode) {
    switch (mode) {
        case QM3_COST_OVERLAY_OFF: return "off";
        case QM3_COST_OVERLAY_EDGE_LENGTH: return "open-edge length";
        case QM3_COST_OVERLAY_PATH_CHORD: return "path/chord";
        case QM3_COST_OVERLAY_ALIGNMENT: return "cross-field alignment";
        case QM3_COST_OVERLAY_ANGLE: return "angle";
        default: return "unknown";
    }
}

static uint32_t qm3_clamp_action_index(float action) {
    if (!isfinite(action) || action < 0.0f) return 0;
    return (uint32_t)action;
}

static int32_t qm3_pick_frontier_screen(const QuadMeshing3DEnv* env, Vector2 mouse, float radius_px) {
    float best_d2 = radius_px * radius_px;
    int32_t best = -1;
    for (uint32_t i = 0; i < env->mesh.frontier_count; ++i) {
        uint32_t vidx = env->mesh.frontier[i];
        if (vidx >= env->mesh.vertex_count) continue;
        const Qm3MeshVertex* v = &env->mesh.vertices[vidx];
        if (v->disabled) continue;
        Vector2 p = GetWorldToScreen(qm3_v3(v->pos), env->camera);
        float dx = p.x - mouse.x;
        float dy = p.y - mouse.y;
        float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best = (int32_t)i;
        }
    }
    return best;
}

static int32_t qm3_pick_candidate_screen(const QuadMeshing3DEnv* env, Vector2 mouse, bool require_valid, float radius_px) {
    float best_d2 = radius_px * radius_px;
    int32_t best = -1;
    for (uint32_t i = 0; i < env->candidate_count; ++i) {
        if (require_valid && !env->candidates[i].path_ok) continue;
        if (env->candidates[i].sample_id >= env->surface.sample_count) continue;
        Vector2 p = GetWorldToScreen(qm3_candidate_position(env, &env->candidates[i]), env->camera);
        float dx = p.x - mouse.x;
        float dy = p.y - mouse.y;
        float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best = (int32_t)i;
        }
    }
    return best;
}

static void qm3_update_orbit_camera(Camera3D* camera, float min_dist, float max_dist) {
    Vector3 offset = Vector3Subtract(camera->position, camera->target);
    float dist = Vector3Length(offset);
    if (dist < min_dist) dist = min_dist;

    Vector2 mouse_delta = GetMouseDelta();
    bool picking_modifier = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL) ||
        IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !picking_modifier) {
        const float rotate_speed = 0.005f;
        float yaw = atan2f(offset.x, offset.z);
        float pitch = asinf(Clamp(offset.y / dist, -1.0f, 1.0f));

        yaw -= mouse_delta.x * rotate_speed;
        pitch += mouse_delta.y * rotate_speed;
        pitch = Clamp(pitch, -1.52f, 1.52f);

        float cp = cosf(pitch);
        offset = (Vector3){
            dist * cp * sinf(yaw),
            dist * sinf(pitch),
            dist * cp * cosf(yaw),
        };
        camera->position = Vector3Add(camera->target, offset);
    }

    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
        Vector3 forward = Vector3Normalize(Vector3Subtract(camera->target, camera->position));
        Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, camera->up));
        Vector3 up = Vector3Normalize(Vector3CrossProduct(right, forward));
        float view_height = 2.0f * dist * tanf(DEG2RAD * camera->fovy * 0.5f);
        float pan_scale = view_height / fmaxf((float)GetScreenHeight(), 1.0f);
        Vector3 pan = Vector3Add(
            Vector3Scale(right, -mouse_delta.x * pan_scale),
            Vector3Scale(up, mouse_delta.y * pan_scale)
        );
        camera->position = Vector3Add(camera->position, pan);
        camera->target = Vector3Add(camera->target, pan);
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) {
        offset = Vector3Subtract(camera->position, camera->target);
        dist = Vector3Length(offset);
        float new_dist = Clamp(dist * expf(-wheel * 0.12f), min_dist, max_dist);
        if (dist > 1e-8f) camera->position = Vector3Add(camera->target, Vector3Scale(offset, new_dist / dist));
    }

    camera->up = (Vector3){0.0f, 1.0f, 0.0f};
}

static void qm3_draw_active_sample_crosses(const QuadMeshing3DEnv* env, Camera3D camera, float size, Color color) {
    Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, camera.up));
    Vector3 up = Vector3Normalize(Vector3CrossProduct(right, forward));
    right = Vector3Scale(right, size);
    up = Vector3Scale(up, size);

    rlBegin(RL_LINES);
    rlColor4ub(color.r, color.g, color.b, color.a);
    for (uint32_t i = 0; i < env->surface.sample_count; ++i) {
        if (env->sample_disabled && env->sample_disabled[i]) continue;
        Vector3 p = qm3_v3(env->surface.samples[i].p);
        Vector3 a = Vector3Subtract(p, right);
        Vector3 b = Vector3Add(p, right);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
        a = Vector3Subtract(p, up);
        b = Vector3Add(p, up);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
    }
    rlEnd();
}

static void qm3_draw_disabled_sample_crosses(const QuadMeshing3DEnv* env, Camera3D camera, float size, Color color) {
    if (!env->sample_disabled) return;
    Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, camera.up));
    Vector3 up = Vector3Normalize(Vector3CrossProduct(right, forward));
    right = Vector3Scale(right, size);
    up = Vector3Scale(up, size);

    rlBegin(RL_LINES);
    rlColor4ub(color.r, color.g, color.b, color.a);
    for (uint32_t i = 0; i < env->surface.sample_count; ++i) {
        if (!env->sample_disabled[i]) continue;
        Vector3 p = qm3_v3(env->surface.samples[i].p);
        Vector3 a = Vector3Subtract(p, right);
        Vector3 b = Vector3Add(p, right);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
        a = Vector3Subtract(p, up);
        b = Vector3Add(p, up);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
    }
    rlEnd();
}

static void qm3_draw_mesh(const Qm3Surface* surface, bool opaque) {
    if (opaque) {
        for (uint32_t i = 0; i < surface->triangle_count; ++i) {
            Qm3Tri tri = surface->triangles[i];
            Vector3 a = qm3_v3(surface->vertices[tri.a]);
            Vector3 b = qm3_v3(surface->vertices[tri.b]);
            Vector3 c = qm3_v3(surface->vertices[tri.c]);
            DrawTriangle3D(a, b, c, (Color){170, 185, 205, 190});
        }
    }

    Color wire = opaque ? (Color){25, 35, 45, 180} : (Color){85, 110, 135, 140};
    for (uint32_t i = 0; i < surface->triangle_count; ++i) {
        Qm3Tri tri = surface->triangles[i];
        Vector3 a = qm3_v3(surface->vertices[tri.a]);
        Vector3 b = qm3_v3(surface->vertices[tri.b]);
        Vector3 c = qm3_v3(surface->vertices[tri.c]);
        DrawLine3D(a, b, wire);
        DrawLine3D(b, c, wire);
        DrawLine3D(c, a, wire);
    }
}

static Qm3Vec3 qm3_triangle_normal_offset(const Qm3Surface* surface, int tri, float amount) {
    Qm3Tri t = surface->triangles[tri];
    Qm3Vec3 ab = qm3_sub(surface->vertices[t.b], surface->vertices[t.a]);
    Qm3Vec3 ac = qm3_sub(surface->vertices[t.c], surface->vertices[t.a]);
    Qm3Vec3 n = (Qm3Vec3){ab.y * ac.z - ab.z * ac.y, ab.z * ac.x - ab.x * ac.z, ab.x * ac.y - ab.y * ac.x};
    if (qm3_dot(n, n) <= 1e-20f) return (Qm3Vec3){0.0f, 0.0f, 0.0f};
    return qm3_scale(qm3_normalize(n), amount);
}

static void qm3_draw_loop_debug_triangle(const Qm3Surface* surface, int tri, Color color, float offset) {
    Qm3Tri t = surface->triangles[tri];
    Qm3Vec3 o = qm3_triangle_normal_offset(surface, tri, offset);
    DrawTriangle3D(qm3_v3(qm3_add(surface->vertices[t.a], o)), qm3_v3(qm3_add(surface->vertices[t.b], o)), qm3_v3(qm3_add(surface->vertices[t.c], o)), color);
}

static void qm3_draw_loop_debug_edge(const Qm3Surface* surface, int tri, int edge, Color color, float offset) {
    Qm3Tri t = surface->triangles[tri];
    uint32_t verts[3] = {t.a, t.b, t.c};
    Qm3Vec3 o = qm3_triangle_normal_offset(surface, tri, offset);
    DrawLine3D(qm3_v3(qm3_add(surface->vertices[verts[(edge + 1) % 3]], o)), qm3_v3(qm3_add(surface->vertices[verts[(edge + 2) % 3]], o)), color);
}

static void qm3_draw_loop_debug(const QuadMeshing3DEnv* env, float marker_scale) {
    int mode = env->loop_debug_mode;
    const Qm3LoopDebugData* debug = &env->loop_debug;
    if (mode <= 0 || !debug->flags || debug->tri_count != env->surface.triangle_count) return;
    float off = fmaxf(marker_scale * 0.00005f, 2e-6f);
    if (mode == 3 || mode == 6) {
        for (uint32_t tri = 0; tri < env->surface.triangle_count; ++tri) {
            if (debug->side_mark[tri] == 1) qm3_draw_loop_debug_triangle(&env->surface, (int)tri, env->loop_stats.chosen_side == 1 ? (Color){50, 230, 120, 95} : (Color){40, 140, 255, 55}, off);
            else if (debug->side_mark[tri] == 2) qm3_draw_loop_debug_triangle(&env->surface, (int)tri, env->loop_stats.chosen_side == 2 ? (Color){50, 230, 120, 95} : (Color){255, 170, 40, 55}, off);
        }
    }
    if (mode == 1 || mode == 6) {
        for (uint32_t tri = 0; tri < env->surface.triangle_count; ++tri) if (debug->boundary_tri[tri]) qm3_draw_loop_debug_triangle(&env->surface, (int)tri, (Color){180, 180, 190, 80}, off * 1.5f);
    }
    if (mode == 4 || mode == 6) {
        for (uint32_t tri = 0; tri < env->surface.triangle_count; ++tri) if (debug->conflict_tri[tri]) qm3_draw_loop_debug_triangle(&env->surface, (int)tri, (Color){255, 40, 40, 180}, off * 2.0f);
    }
    if (mode == 2 || mode == 6) {
        for (uint32_t tri = 0; tri < env->surface.triangle_count; ++tri) {
            for (int edge = 0; edge < 3; ++edge) {
                unsigned char f = debug->flags[3 * tri + edge];
                if (f & QM3_LOOP_EDGE_BLOCKED) qm3_draw_loop_debug_edge(&env->surface, (int)tri, edge, (Color){255, 40, 40, 255}, off * 3.0f);
                else if ((f & QM3_LOOP_EDGE_LEFT) && (f & QM3_LOOP_EDGE_RIGHT)) qm3_draw_loop_debug_edge(&env->surface, (int)tri, edge, (Color){255, 40, 255, 255}, off * 3.0f);
                else if (f & QM3_LOOP_EDGE_LEFT) qm3_draw_loop_debug_edge(&env->surface, (int)tri, edge, (Color){40, 140, 255, 255}, off * 3.0f);
                else if (f & QM3_LOOP_EDGE_RIGHT) qm3_draw_loop_debug_edge(&env->surface, (int)tri, edge, (Color){255, 190, 40, 255}, off * 3.0f);
            }
        }
    }
    if (mode == 5 || mode == 6) {
        qm3_draw_disabled_sample_crosses(env, env->camera, marker_scale * 0.0020f, (Color){255, 70, 70, 230});
    }
}

static void qm3_draw_vertices(const Qm3Surface* surface, Camera3D camera, float size, Color color) {
    Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, camera.up));
    Vector3 up = Vector3Normalize(Vector3CrossProduct(right, forward));
    right = Vector3Scale(right, size);
    up = Vector3Scale(up, size);

    rlBegin(RL_LINES);
    rlColor4ub(color.r, color.g, color.b, color.a);
    for (uint32_t i = 0; i < surface->vertex_count; ++i) {
        Vector3 p = qm3_v3(surface->vertices[i]);
        Vector3 a = Vector3Subtract(p, right);
        Vector3 b = Vector3Add(p, right);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
        a = Vector3Subtract(p, up);
        b = Vector3Add(p, up);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
    }
    rlEnd();
}

static void qm3_draw_normals(const Qm3Surface* surface, float length) {
    for (uint32_t i = 0; i < surface->vertex_count; ++i) {
        Vector3 p = qm3_v3(surface->vertices[i]);
        Vector3 n = Vector3Scale(qm3_v3(surface->vertex_normals[i]), length);
        DrawLine3D(p, Vector3Add(p, n), (Color){80, 220, 255, 190});
    }
}

static void qm3_draw_cross_field(const Qm3Surface* surface, float length) {
    for (uint32_t i = 0; i < surface->triangle_count; ++i) {
        Qm3Tri tri = surface->triangles[i];
        Qm3Vec3 center = qm3_scale(qm3_add(qm3_add(surface->vertices[tri.a], surface->vertices[tri.b]), surface->vertices[tri.c]), 1.0f / 3.0f);
        Qm3Vec3 nudge = qm3_scale(surface->face_normals[i], length * 0.03f);
        Vector3 p = qm3_v3(qm3_add(center, nudge));
        Vector3 u = Vector3Scale(qm3_v3(surface->face_dir_u[i]), length);
        Vector3 v = Vector3Scale(qm3_v3(surface->face_dir_v[i]), length);
        DrawLine3D(Vector3Subtract(p, u), Vector3Add(p, u), (Color){255, 210, 70, 175});
        DrawLine3D(Vector3Subtract(p, v), Vector3Add(p, v), (Color){255, 210, 70, 175});
    }
}

static void qm3_draw_graph_edges(const Qm3Mesh* mesh, float offset) {
    for (uint32_t i = 0; i < mesh->edge_count; ++i) {
        Qm3MeshEdge e = mesh->edges[i];
        if (e.disabled) continue;
        Qm3MeshVertex a = mesh->vertices[e.a];
        Qm3MeshVertex b = mesh->vertices[e.b];
        Qm3Vec3 an = qm3_normalize(a.normal);
        Qm3Vec3 bn = qm3_normalize(b.normal);
        Vector3 pa = qm3_v3(qm3_add(a.pos, qm3_scale(an, offset)));
        Vector3 pb = qm3_v3(qm3_add(b.pos, qm3_scale(bn, offset)));
        Color color = e.face_count == 0 ? (Color){230, 80, 80, 255} : (Color){255, 190, 70, 255};
        if (e.face_count >= 2) color = (Color){70, 220, 120, 255};
        DrawLine3D(pa, pb, color);
    }
}

static void qm3_draw_graph_path_segments(const Qm3Mesh* mesh, float y_offset, Color color) {
    if (mesh->edge_path_segments.count == 0) return;
    rlBegin(RL_LINES);
    rlColor4ub(color.r, color.g, color.b, color.a);
    for (uint32_t ei = 0; ei < mesh->edge_count; ++ei) {
        Qm3MeshEdge edge = mesh->edges[ei];
        if (edge.disabled || edge.segment_count == 0) continue;
        uint32_t end = edge.segment_offset + edge.segment_count;
        if (end > mesh->edge_path_segments.count) end = mesh->edge_path_segments.count;
        for (uint32_t si = edge.segment_offset; si < end; ++si) {
            Qm3Vec3 a3 = mesh->edge_path_segments.data[si].a;
            Qm3Vec3 b3 = mesh->edge_path_segments.data[si].b;
            Vector3 a = qm3_v3((Qm3Vec3){a3.x, a3.y + y_offset, a3.z});
            Vector3 b = qm3_v3((Qm3Vec3){b3.x, b3.y + y_offset, b3.z});
            rlVertex3f(a.x, a.y, a.z);
            rlVertex3f(b.x, b.y, b.z);
        }
    }
    rlEnd();
}

static void qm3_draw_graph_edge_highlight(const Qm3Mesh* mesh, int32_t edge_idx, float offset, Color color) {
    if (edge_idx < 0 || (uint32_t)edge_idx >= mesh->edge_count) return;
    Qm3MeshEdge e = mesh->edges[edge_idx];
    if (e.disabled) return;
    Qm3MeshVertex a = mesh->vertices[e.a];
    Qm3MeshVertex b = mesh->vertices[e.b];
    Vector3 pa = qm3_v3(qm3_add(a.pos, qm3_scale(qm3_normalize(a.normal), offset)));
    Vector3 pb = qm3_v3(qm3_add(b.pos, qm3_scale(qm3_normalize(b.normal), offset)));
    DrawLine3D(pa, pb, color);
    DrawSphere(pa, fmaxf(offset * 0.45f, 1e-5f), color);
    DrawSphere(pb, fmaxf(offset * 0.45f, 1e-5f), color);
}

static void qm3_draw_mesh_vertex_crosses(const Qm3Mesh* mesh, Camera3D camera, float size, float offset) {
    Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, camera.up));
    Vector3 up = Vector3Normalize(Vector3CrossProduct(right, forward));
    right = Vector3Scale(right, size);
    up = Vector3Scale(up, size);

    rlBegin(RL_LINES);
    for (uint32_t i = 0; i < mesh->vertex_count; ++i) {
        const Qm3MeshVertex* vertex = &mesh->vertices[i];
        if (vertex->disabled) continue;
        Color color = vertex->frontier_index >= 0 ? (Color){30, 170, 255, 255} : (Color){230, 230, 230, 210};
        rlColor4ub(color.r, color.g, color.b, color.a);
        Vector3 p = qm3_v3(qm3_add(vertex->pos, qm3_scale(qm3_normalize(vertex->normal), offset)));
        Vector3 a = Vector3Subtract(p, right);
        Vector3 b = Vector3Add(p, right);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
        a = Vector3Subtract(p, up);
        b = Vector3Add(p, up);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
    }
    rlEnd();
}

static float qm3_candidate_overlay_cost(const QuadMeshing3DEnv* env, const Qm3TargetCandidate* candidate);

static Color qm3_cost_overlay_color(float cost) {
    cost = fminf(fmaxf(cost, 0.0f), 1.0f);
    Color low = (Color){50, 235, 125, 245};
    Color mid = (Color){255, 210, 70, 245};
    Color high = (Color){255, 60, 55, 245};
    Color a = cost < 0.5f ? low : mid;
    Color b = cost < 0.5f ? mid : high;
    float t = cost < 0.5f ? cost * 2.0f : (cost - 0.5f) * 2.0f;
    return (Color){
        (unsigned char)((1.0f - t) * a.r + t * b.r),
        (unsigned char)((1.0f - t) * a.g + t * b.g),
        (unsigned char)((1.0f - t) * a.b + t * b.b),
        245,
    };
}

static void qm3_draw_candidate_subset_crosses(
    const QuadMeshing3DEnv* env,
    const Qm3TargetCandidate* candidates,
    uint32_t sample_count,
    Camera3D camera,
    float size,
    Color sample_color,
    Color existing_color
) {
    Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, camera.up));
    Vector3 up = Vector3Normalize(Vector3CrossProduct(right, forward));
    right = Vector3Scale(right, size);
    up = Vector3Scale(up, size);

    rlBegin(RL_LINES);
    for (uint32_t i = 0; i < sample_count; ++i) {
        if (!candidates[i].path_ok) continue;
        if (candidates[i].sample_id >= env->surface.sample_count) continue;
        Color color = candidates[i].kind == QM3_TARGET_EXISTING_FRONTIER ? existing_color : sample_color;
        if (env->render_cost_overlay_mode != QM3_COST_OVERLAY_OFF) {
            color = qm3_cost_overlay_color(qm3_candidate_overlay_cost(env, &candidates[i]));
        }
        rlColor4ub(color.r, color.g, color.b, color.a);
        Vector3 p = qm3_candidate_position(env, &candidates[i]);
        Vector3 a = Vector3Subtract(p, right);
        Vector3 b = Vector3Add(p, right);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
        a = Vector3Subtract(p, up);
        b = Vector3Add(p, up);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
    }
    rlEnd();
}

static void qm3_draw_candidate_validity_crosses(
    const QuadMeshing3DEnv* env,
    const Qm3TargetCandidate* candidates,
    uint32_t sample_count,
    Camera3D camera,
    float size
) {
    Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    Vector3 right = Vector3Scale(Vector3Normalize(Vector3CrossProduct(forward, camera.up)), size);
    Vector3 up = Vector3Scale(Vector3Normalize(Vector3CrossProduct(right, forward)), size);
    rlBegin(RL_LINES);
    for (uint32_t i = 0; i < sample_count; ++i) {
        if (candidates[i].sample_id >= env->surface.sample_count) continue;
        Color color = qm3_candidate_validity_color(candidates[i].validity);
        rlColor4ub(color.r, color.g, color.b, color.a);
        Vector3 p = qm3_candidate_position(env, &candidates[i]);
        Vector3 a = Vector3Subtract(p, right);
        Vector3 b = Vector3Add(p, right);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
        a = Vector3Subtract(p, up);
        b = Vector3Add(p, up);
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
    }
    rlEnd();
}

static void qm3_draw_path(const Qm3Path* path, float offset, Color color) {
    if (path->count < 2) return;
    rlBegin(RL_LINES);
    rlColor4ub(color.r, color.g, color.b, color.a);
    for (uint32_t i = 0; i + 1 < path->count; ++i) {
        Qm3Vec3 a3 = path->points[i];
        Qm3Vec3 b3 = path->points[i + 1];
        Vector3 a = qm3_v3((Qm3Vec3){a3.x, a3.y + offset, a3.z});
        Vector3 b = qm3_v3((Qm3Vec3){b3.x, b3.y + offset, b3.z});
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
    }
    rlEnd();
}

static void qm3_draw_path_segments(const Qm3PathSegmentArray* segments, uint32_t offset, uint32_t count, float y_offset, Color color) {
    if (count == 0 || offset >= segments->count) return;
    uint32_t end = offset + count;
    if (end > segments->count) end = segments->count;
    rlBegin(RL_LINES);
    rlColor4ub(color.r, color.g, color.b, color.a);
    for (uint32_t i = offset; i < end; ++i) {
        Qm3Vec3 a3 = segments->data[i].a;
        Qm3Vec3 b3 = segments->data[i].b;
        Vector3 a = qm3_v3((Qm3Vec3){a3.x, a3.y + y_offset, a3.z});
        Vector3 b = qm3_v3((Qm3Vec3){b3.x, b3.y + y_offset, b3.z});
        rlVertex3f(a.x, a.y, a.z);
        rlVertex3f(b.x, b.y, b.z);
    }
    rlEnd();
}

static void qm3_candidate_reserve(QuadMeshing3DEnv* env, uint32_t cap) {
    if (cap <= env->candidate_cap) return;
    env->candidates = (Qm3TargetCandidate*)qm3_checked_realloc(env->candidates, (size_t)cap * sizeof(Qm3TargetCandidate));
    env->candidate_cap = cap;
}

static void qm3_candidate_push(QuadMeshing3DEnv* env, Qm3TargetKind kind, uint32_t graph_vertex, uint32_t sample_id, Qm3CandidateValidity validity) {
    if (sample_id >= env->surface.sample_count) return;
    if (env->candidate_count == env->candidate_cap) qm3_candidate_reserve(env, env->candidate_cap ? env->candidate_cap * 2 : 256);
    env->candidates[env->candidate_count++] = (Qm3TargetCandidate){
        .kind = kind,
        .graph_vertex = graph_vertex,
        .sample_id = sample_id,
        .intersect_edge = -1,
        .intersect_segment = -1,
        .validity = validity,
    };
    if (kind == QM3_TARGET_EXISTING_FRONTIER) env->candidate_existing_count++;
    else env->candidate_sample_count++;
}

static void qm3_graph_tri_segment_push(Qm3GraphTriSegmentArray* a, Qm3GraphTriSegment v) {
    if (a->count == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 4;
        a->data = (Qm3GraphTriSegment*)qm3_checked_realloc(a->data, (size_t)a->cap * sizeof(Qm3GraphTriSegment));
    }
    a->data[a->count++] = v;
}

static void qm3_graph_tri_vertex_push(Qm3GraphTriVertexArray* a, Qm3GraphTriVertex v) {
    if (a->count == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 4;
        a->data = (Qm3GraphTriVertex*)qm3_checked_realloc(a->data, (size_t)a->cap * sizeof(Qm3GraphTriVertex));
    }
    a->data[a->count++] = v;
}

static void qm3_graph_tri_segments_free(Qm3GraphTriSegmentArray** buckets, uint32_t tri_count) {
    if (!*buckets) return;
    for (uint32_t i = 0; i < tri_count; ++i) free((*buckets)[i].data);
    free(*buckets);
    *buckets = NULL;
}

static void qm3_graph_tri_vertices_free(Qm3GraphTriVertexArray** buckets, uint32_t tri_count) {
    if (!*buckets) return;
    for (uint32_t i = 0; i < tri_count; ++i) free((*buckets)[i].data);
    free(*buckets);
    *buckets = NULL;
}

static void qm3_ensure_graph_tri_buckets(QuadMeshing3DEnv* env) {
    if (!env->graph_tri_segments) {
        env->graph_tri_segments = (Qm3GraphTriSegmentArray*)calloc((size_t)env->surface.triangle_count, sizeof(Qm3GraphTriSegmentArray));
        QM3_ASSERT(env->graph_tri_segments != NULL || env->surface.triangle_count == 0);
    }
    if (!env->graph_tri_vertices) {
        env->graph_tri_vertices = (Qm3GraphTriVertexArray*)calloc((size_t)env->surface.triangle_count, sizeof(Qm3GraphTriVertexArray));
        QM3_ASSERT(env->graph_tri_vertices != NULL || env->surface.triangle_count == 0);
    }
}

static bool qm3_point_to_base_tri2d(const Qm3Surface* surface, const Qm3ContinuousContext* ctx, int tri, Qm3Vec3 p, Qm3Vec2* out) {
    if (tri < 0 || tri >= (int)surface->triangle_count || !ctx->tri2d_valid[tri]) return false;
    Qm3Tri t = surface->triangles[tri];
    float u, v, w;
    qm3_barycentric3(p, surface->vertices[t.a], surface->vertices[t.b], surface->vertices[t.c], &u, &v, &w);
    *out = qm3_bary_to_2d(u, v, w, ctx->tri2d_base[3 * tri], ctx->tri2d_base[3 * tri + 1], ctx->tri2d_base[3 * tri + 2]);
    return true;
}

static float qm3_v2_point_segment_dist_sq(Qm3Vec2 p, Qm3Vec2 a, Qm3Vec2 b) {
    Qm3Vec2 ab = qm3_v2_sub(b, a);
    float denom = qm3_v2_dot(ab, ab);
    if (denom <= 1e-12f) {
        Qm3Vec2 d = qm3_v2_sub(p, a);
        return qm3_v2_dot(d, d);
    }
    float t = fminf(fmaxf(qm3_v2_dot(qm3_v2_sub(p, a), ab) / denom, 0.0f), 1.0f);
    Qm3Vec2 q = qm3_v2_lerp(a, b, t);
    Qm3Vec2 d = qm3_v2_sub(p, q);
    return qm3_v2_dot(d, d);
}

static bool qm3_segments_intersect_2d(Qm3Vec2 a, Qm3Vec2 b, Qm3Vec2 c, Qm3Vec2 d, float tol) {
    Qm3Vec2 ab = qm3_v2_sub(b, a);
    Qm3Vec2 cd = qm3_v2_sub(d, c);
    float den = qm3_v2_cross(ab, cd);
    Qm3Vec2 ca = qm3_v2_sub(c, a);
    if (fabsf(den) > 1e-12f) {
        float t = qm3_v2_cross(ca, cd) / den;
        float u = qm3_v2_cross(ca, ab) / den;
        if (t >= -tol && t <= 1.0f + tol && u >= -tol && u <= 1.0f + tol) return true;
    }
    float tol_sq = tol * tol;
    return qm3_v2_point_segment_dist_sq(a, c, d) <= tol_sq ||
           qm3_v2_point_segment_dist_sq(b, c, d) <= tol_sq ||
           qm3_v2_point_segment_dist_sq(c, a, b) <= tol_sq ||
           qm3_v2_point_segment_dist_sq(d, a, b) <= tol_sq;
}

static bool qm3_segment_intersect_2d_params(Qm3Vec2 p, Qm3Vec2 q, Qm3Vec2 a, Qm3Vec2 b, float* out_t, float* out_u) {
    Qm3Vec2 r = qm3_v2_sub(q, p);
    Qm3Vec2 s = qm3_v2_sub(b, a);
    float den = qm3_v2_cross(r, s);
    if (fabsf(den) < 1e-8f) return false;
    Qm3Vec2 ap = qm3_v2_sub(a, p);
    float t = qm3_v2_cross(ap, s) / den;
    float u = qm3_v2_cross(ap, r) / den;
    if (t < -1e-2f || t > 1.0f + 1e-2f || u < -1e-2f || u > 1.0f + 1e-2f) return false;
    if (out_t) *out_t = fminf(fmaxf(t, 0.0f), 1.0f);
    if (out_u) *out_u = fminf(fmaxf(u, 0.0f), 1.0f);
    return true;
}

typedef struct { Qm3Vec2 a; Qm3Vec2 b; } Qm3LocalCutSegment;
typedef struct {
    Qm3Vec2 p;
    int side;
    int comp;
} Qm3LocalClassPoint;

static void qm3_local_cut_segment_push(Qm3LocalCutSegment** cuts, int* count, int* cap, Qm3LocalCutSegment value) {
    if (qm3_v2_dot(qm3_v2_sub(value.a, value.b), qm3_v2_sub(value.a, value.b)) <= 1e-14f) return;
    if (*count == *cap) {
        *cap = *cap ? *cap * 2 : 16;
        *cuts = (Qm3LocalCutSegment*)qm3_checked_realloc(*cuts, (size_t)*cap * sizeof(Qm3LocalCutSegment));
    }
    (*cuts)[(*count)++] = value;
}

static void qm3_local_class_point_push(Qm3LocalClassPoint** points, int* count, int* cap, Qm3LocalClassPoint value) {
    if (*count == *cap) {
        *cap = *cap ? *cap * 2 : 32;
        *points = (Qm3LocalClassPoint*)qm3_checked_realloc(*points, (size_t)*cap * sizeof(Qm3LocalClassPoint));
    }
    value.comp = -1;
    (*points)[(*count)++] = value;
}

static bool qm3_point_inside_triangle_2d(Qm3Vec2 p, const Qm3Vec2 tri2d[3]) {
    for (int edge = 0; edge < 3; ++edge) {
        Qm3Vec2 a = tri2d[(edge + 1) % 3];
        Qm3Vec2 b = tri2d[(edge + 2) % 3];
        if (qm3_v2_cross(qm3_v2_sub(b, a), qm3_v2_sub(p, a)) < -1e-6f) return false;
    }
    return true;
}

static bool qm3_add_local_side_seed(Qm3LocalClassPoint** points, int* count, int* cap, Qm3Vec2 mid, Qm3Vec2 normal, int side, float eps, const Qm3Vec2 tri2d[3]) {
    for (int i = 0; i < 8; ++i) {
        Qm3Vec2 p = qm3_v2_add(mid, qm3_v2_scale(normal, eps));
        if (qm3_point_inside_triangle_2d(p, tri2d)) {
            qm3_local_class_point_push(points, count, cap, (Qm3LocalClassPoint){.p = p, .side = side});
            return true;
        }
        eps *= 0.5f;
    }
    return false;
}

static bool qm3_local_connection_crosses_cut(Qm3Vec2 a, Qm3Vec2 b, const Qm3LocalCutSegment* cuts, int cut_count) {
    for (int i = 0; i < cut_count; ++i) {
        float t, u;
        if (!qm3_segment_intersect_2d_params(a, b, cuts[i].a, cuts[i].b, &t, &u)) continue;
        if (t > 1e-5f && t < 1.0f - 1e-5f && u > -1e-5f && u < 1.0f + 1e-5f) return true;
    }
    return false;
}

static int qm3_local_component_side(const Qm3LocalClassPoint* points, int point_count, int comp, uint32_t* conflicts) {
    int side = 0;
    for (int i = 0; i < point_count; ++i) {
        if (points[i].comp != comp || points[i].side == 0) continue;
        if (side != 0 && side != points[i].side) {
            (*conflicts)++;
            return 0;
        }
        side = points[i].side;
    }
    return side;
}

static float qm3_surface_triangle_area(const Qm3Surface* surface, int tri) {
    Qm3Tri t = surface->triangles[tri];
    Qm3Vec3 ab = qm3_sub(surface->vertices[t.b], surface->vertices[t.a]);
    Qm3Vec3 ac = qm3_sub(surface->vertices[t.c], surface->vertices[t.a]);
    Qm3Vec3 cross = (Qm3Vec3){ab.y * ac.z - ab.z * ac.y, ab.z * ac.x - ab.x * ac.z, ab.x * ac.y - ab.y * ac.x};
    return 0.5f * qm3_len(cross);
}

static bool qm3_candidate_segments_intersect_graph(
    const QuadMeshing3DEnv* env,
    uint32_t source_vidx,
    uint32_t target_vidx,
    const Qm3PathSegmentArray* segments,
    uint32_t* out_tests,
    int32_t* out_edge,
    int32_t* out_segment
) {
    if (out_edge) *out_edge = -1;
    if (out_segment) *out_segment = -1;
    if (!env->graph_tri_segments || segments->count == 0) return false;
    float diag = qm3_surface_diag(&env->surface);
    float tol = fmaxf(diag * 1e-5f, 1e-6f);
    uint32_t tests = 0;
    for (uint32_t si = 0; si < segments->count; ++si) {
        const Qm3PathSegment* seg = &segments->data[si];
        if (seg->tri < 0 || seg->tri >= (int)env->surface.triangle_count) continue;
        if (qm3_distance(seg->a, seg->b) < tol) continue;
        Qm3Vec2 a, b;
        if (!qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, seg->tri, seg->a, &a)) continue;
        if (!qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, seg->tri, seg->b, &b)) continue;
        const Qm3GraphTriVertexArray* vertex_bucket = env->graph_tri_vertices ? &env->graph_tri_vertices[seg->tri] : NULL;
        if (vertex_bucket) {
            float tol_sq = tol * tol;
            for (uint32_t i = 0; i < vertex_bucket->count; ++i) {
                const Qm3GraphTriVertex* gv = &vertex_bucket->data[i];
                if (gv->graph_vertex == source_vidx) continue;
                if (target_vidx != UINT32_MAX && gv->graph_vertex == target_vidx) continue;
                const Qm3MeshVertex* mv = &env->mesh.vertices[gv->graph_vertex];
                if (mv->disabled) continue;
                tests++;
                if (qm3_v2_point_segment_dist_sq(gv->p, a, b) <= tol_sq) {
                    if (out_edge) *out_edge = (int32_t)gv->incident_edge;
                    if (out_segment) *out_segment = (int32_t)si;
                    if (out_tests) *out_tests += tests;
                    return true;
                }
            }
        }
        const Qm3GraphTriSegmentArray* bucket = &env->graph_tri_segments[seg->tri];
        for (uint32_t i = 0; i < bucket->count; ++i) {
            const Qm3GraphTriSegment* graph_seg = &bucket->data[i];
            Qm3MeshEdge e = env->mesh.edges[graph_seg->edge];
            if (e.disabled || e.a == source_vidx || e.b == source_vidx) continue;
            if (target_vidx != UINT32_MAX && (e.a == target_vidx || e.b == target_vidx)) continue;
            tests++;
            if (qm3_segments_intersect_2d(a, b, graph_seg->a, graph_seg->b, tol)) {
                if (out_edge) *out_edge = (int32_t)graph_seg->edge;
                if (out_segment) *out_segment = (int32_t)si;
                if (out_tests) *out_tests += tests;
                return true;
            }
        }
    }
    if (out_tests) *out_tests += tests;
    return false;
}

static bool qm3_support_tri_contains_vertex(const Qm3Surface* surface, int tri, uint32_t vertex) {
    Qm3Tri t = surface->triangles[tri];
    return t.a == vertex || t.b == vertex || t.c == vertex;
}

static void qm3_support_push_tri(Qm3IntArray* support, int tri) {
    for (int i = 0; i < support->size; ++i) if (support->data[i] == tri) return;
    qm3_int_push(support, tri);
}

static void qm3_point_support_tris(const Qm3Surface* surface, Qm3Vec3 p, int base_tri, Qm3IntArray* support) {
    support->size = 0;
    if (base_tri < 0 || base_tri >= (int)surface->triangle_count) return;

    const float tol = 1e-5f;
    Qm3Tri t = surface->triangles[base_tri];
    uint32_t verts[3] = {t.a, t.b, t.c};
    float bary[3];
    qm3_barycentric3(p, surface->vertices[t.a], surface->vertices[t.b], surface->vertices[t.c], &bary[0], &bary[1], &bary[2]);

    qm3_support_push_tri(support, base_tri);
    int near_zero = 0;
    for (int i = 0; i < 3; ++i) if (bary[i] <= tol) near_zero++;
    if (near_zero == 0) return;

    if (near_zero == 1) {
        for (int edge = 0; edge < 3; ++edge) {
            if (bary[edge] > tol) continue;
            int neighbor = ((int*)&surface->triangle_neighbors[base_tri])[edge];
            if (neighbor >= 0) qm3_support_push_tri(support, neighbor);
            return;
        }
    }

    int vertex_local = 0;
    if (bary[1] > bary[vertex_local]) vertex_local = 1;
    if (bary[2] > bary[vertex_local]) vertex_local = 2;
    uint32_t vertex = verts[vertex_local];
    for (int qi = 0; qi < support->size; ++qi) {
        int tri = support->data[qi];
        Qm3Tri curr = surface->triangles[tri];
        uint32_t cv[3] = {curr.a, curr.b, curr.c};
        for (int edge = 0; edge < 3; ++edge) {
            uint32_t a = cv[(edge + 1) % 3];
            uint32_t b = cv[(edge + 2) % 3];
            if (a != vertex && b != vertex) continue;
            int neighbor = ((int*)&surface->triangle_neighbors[tri])[edge];
            if (neighbor >= 0 && qm3_support_tri_contains_vertex(surface, neighbor, vertex)) {
                qm3_support_push_tri(support, neighbor);
            }
        }
    }
}

static int qm3_surface_local_edge_between(const Qm3Surface* surface, int tri, uint32_t va, uint32_t vb) {
    if (tri < 0 || tri >= (int)surface->triangle_count) return -1;
    Qm3Tri t = surface->triangles[tri];
    uint32_t v[3] = {t.a, t.b, t.c};
    for (int edge = 0; edge < 3; ++edge) {
        uint32_t a = v[(edge + 1) % 3];
        uint32_t b = v[(edge + 2) % 3];
        if ((a == va && b == vb) || (a == vb && b == va)) return edge;
    }
    return -1;
}

static void qm3_loop_debug_free(Qm3LoopDebugData* debug) {
    free(debug->flags);
    free(debug->boundary_tri);
    free(debug->side_mark);
    free(debug->conflict_tri);
    memset(debug, 0, sizeof(*debug));
}

static void qm3_loop_debug_capture(Qm3LoopDebugData* debug, const unsigned char* flags, const unsigned char* boundary_tri, const unsigned char* side_mark, const unsigned char* conflict_tri, uint32_t tri_count) {
    qm3_loop_debug_free(debug);
    debug->tri_count = tri_count;
    debug->flags = (unsigned char*)malloc((size_t)tri_count * 3 * sizeof(unsigned char));
    debug->boundary_tri = (unsigned char*)malloc((size_t)tri_count * sizeof(unsigned char));
    debug->side_mark = (unsigned char*)malloc((size_t)tri_count * sizeof(unsigned char));
    debug->conflict_tri = (unsigned char*)malloc((size_t)tri_count * sizeof(unsigned char));
    QM3_ASSERT(debug->flags && debug->boundary_tri && debug->side_mark && debug->conflict_tri);
    memcpy(debug->flags, flags, (size_t)tri_count * 3 * sizeof(unsigned char));
    memcpy(debug->boundary_tri, boundary_tri, (size_t)tri_count * sizeof(unsigned char));
    memcpy(debug->side_mark, side_mark, (size_t)tri_count * sizeof(unsigned char));
    memcpy(debug->conflict_tri, conflict_tri, (size_t)tri_count * sizeof(unsigned char));
}

static void qm3_loop_mark_vertex_fan_boundary(const Qm3Surface* surface, const Qm3PropGraph* graph, int vertex, unsigned char* boundary_tri) {
    if (!graph || vertex < 0 || vertex >= (int)surface->vertex_count) return;
    for (int i = graph->node_tri_offsets[vertex]; i < graph->node_tri_offsets[vertex + 1]; ++i) boundary_tri[graph->node_tri_ids[i]] = 1;
}

static void qm3_loop_mark_point_vertex_fan_boundary(const Qm3Surface* surface, const Qm3PropGraph* graph, int tri, Qm3Vec3 p, unsigned char* boundary_tri) {
    Qm3Tri t = surface->triangles[tri];
    float b[3];
    qm3_barycentric3(p, surface->vertices[t.a], surface->vertices[t.b], surface->vertices[t.c], &b[0], &b[1], &b[2]);
    uint32_t verts[3] = {t.a, t.b, t.c};
    for (int i = 0; i < 3; ++i) if (b[i] >= 1.0f - 1e-5f) qm3_loop_mark_vertex_fan_boundary(surface, graph, (int)verts[i], boundary_tri);
}

static void qm3_loop_update_local_vertex_side(int tri, int local_vertex, int side, float dist2, unsigned char* vertex_side, float* vertex_dist2) {
    if (side == 0) return;
    int idx = 3 * tri + local_vertex;
    if (dist2 < vertex_dist2[idx]) {
        vertex_dist2[idx] = dist2;
        vertex_side[idx] = (unsigned char)side;
    }
}

static void qm3_loop_mark_segment_blocked_edges(const QuadMeshing3DEnv* env, const Qm3PathSegment* seg, unsigned char* flags, unsigned char* vertex_side, float* vertex_dist2, unsigned char* boundary_tri) {
    if (seg->tri < 0 || seg->tri >= (int)env->surface.triangle_count || !env->continuous_ctx.tri2d_valid[seg->tri]) return;
    Qm3Vec2 tri2d[3] = {
        env->continuous_ctx.tri2d_base[3 * seg->tri + 0],
        env->continuous_ctx.tri2d_base[3 * seg->tri + 1],
        env->continuous_ctx.tri2d_base[3 * seg->tri + 2],
    };
    Qm3Vec2 a2, b2;
    if (!qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, seg->tri, seg->a, &a2)) return;
    if (!qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, seg->tri, seg->b, &b2)) return;
    Qm3Vec2 dir = qm3_v2_sub(b2, a2);
    if (qm3_v2_dot(dir, dir) < 1e-14f) return;

    boundary_tri[seg->tri] = 1;
    qm3_loop_mark_point_vertex_fan_boundary(&env->surface, &env->prop_graph, seg->tri, seg->a, boundary_tri);
    qm3_loop_mark_point_vertex_fan_boundary(&env->surface, &env->prop_graph, seg->tri, seg->b, boundary_tri);
    Qm3Tri st = env->surface.triangles[seg->tri];
    uint32_t verts[3] = {st.a, st.b, st.c};
    for (int edge = 0; edge < 3; ++edge) {
        int ea = (edge + 1) % 3;
        int eb = (edge + 2) % 3;
        float t, u;
        if (!qm3_segment_intersect_2d_params(a2, b2, tri2d[ea], tri2d[eb], &t, &u)) continue;
        if (u <= 1e-5f || u >= 1.0f - 1e-5f) {
            qm3_loop_mark_vertex_fan_boundary(&env->surface, &env->prop_graph, (int)verts[u <= 1e-5f ? ea : eb], boundary_tri);
            continue;
        }
        flags[3 * seg->tri + edge] |= QM3_LOOP_EDGE_BLOCKED;
        Qm3Vec2 hit = qm3_v2_lerp(tri2d[ea], tri2d[eb], u);
        int side_a = qm3_v2_cross(dir, qm3_v2_sub(tri2d[ea], hit)) > 0.0f ? 1 : 2;
        int side_b = qm3_v2_cross(dir, qm3_v2_sub(tri2d[eb], hit)) > 0.0f ? 1 : 2;

        qm3_loop_update_local_vertex_side(seg->tri, ea, side_a, qm3_v2_dot(qm3_v2_sub(tri2d[ea], hit), qm3_v2_sub(tri2d[ea], hit)), vertex_side, vertex_dist2);
        qm3_loop_update_local_vertex_side(seg->tri, eb, side_b, qm3_v2_dot(qm3_v2_sub(tri2d[eb], hit), qm3_v2_sub(tri2d[eb], hit)), vertex_side, vertex_dist2);
    }
}

static void qm3_loop_enqueue_tri(int tri, int side, int* queue, int* tail, unsigned char* side_mark, const unsigned char* boundary_tri, unsigned char* conflict_tri, uint32_t* flood_conflicts) {
    if (tri < 0 || boundary_tri[tri]) return;
    unsigned char mark = side == 1 ? 1 : 2;
    unsigned char other = side == 1 ? 2 : 1;
    if (side_mark[tri] == mark) return;
    if (side_mark[tri] == other) {
        conflict_tri[tri] = 1;
        (*flood_conflicts)++;
        return;
    }
    side_mark[tri] = mark;
    queue[(*tail)++] = tri;
}

static void qm3_loop_seed_cut_edges_from_vertices(const QuadMeshing3DEnv* env, const Qm3IntArray* boundary_list, const unsigned char* vertex_side, const unsigned char* boundary_tri, unsigned char* flags, int* queue_left, int* left_tail, int* queue_right, int* right_tail, unsigned char* side_mark, unsigned char* conflict_tri, uint32_t* classification_conflicts, uint32_t* flood_conflicts) {
    for (int bi = 0; bi < boundary_list->size; ++bi) {
        uint32_t tri = (uint32_t)boundary_list->data[bi];
        for (int edge = 0; edge < 3; ++edge) {
            if (flags[3 * tri + edge] & QM3_LOOP_EDGE_BLOCKED) continue;
            int a = (edge + 1) % 3;
            int b = (edge + 2) % 3;
            bool has_left = vertex_side[3 * tri + a] == 1 || vertex_side[3 * tri + b] == 1;
            bool has_right = vertex_side[3 * tri + a] == 2 || vertex_side[3 * tri + b] == 2;
            if (!has_left && !has_right) continue;
            if (has_left && has_right) {
                flags[3 * tri + edge] |= QM3_LOOP_EDGE_LEFT | QM3_LOOP_EDGE_RIGHT;
                conflict_tri[tri] = 1;
                (*classification_conflicts)++;
                continue;
            }
            int side = has_left ? 1 : 2;
            flags[3 * tri + edge] |= side == 1 ? QM3_LOOP_EDGE_LEFT : QM3_LOOP_EDGE_RIGHT;
            int nb = ((int*)&env->surface.triangle_neighbors[tri])[edge];
            if (nb < 0 || boundary_tri[nb]) continue;
            if (side == 1) qm3_loop_enqueue_tri(nb, 1, queue_left, left_tail, side_mark, boundary_tri, conflict_tri, flood_conflicts);
            else qm3_loop_enqueue_tri(nb, 2, queue_right, right_tail, side_mark, boundary_tri, conflict_tri, flood_conflicts);
        }
    }
}

static bool qm3_loop_flood_pop(const QuadMeshing3DEnv* env, const unsigned char* flags, const unsigned char* boundary_tri, int side, int* queue, int* head, int* tail, unsigned char* side_mark, unsigned char* conflict_tri, float* area, uint32_t* count, Qm3IntArray* reached, uint32_t* flood_conflicts) {
    if (*head >= *tail) return false;
    int tri = queue[(*head)++];
    *area += qm3_surface_triangle_area(&env->surface, tri);
    (*count)++;
    qm3_int_push(reached, tri);
    Qm3Tri t = env->surface.triangles[tri];
    uint32_t verts[3] = {t.a, t.b, t.c};
    for (int edge = 0; edge < 3; ++edge) {
        if (flags[3 * tri + edge] & QM3_LOOP_EDGE_BLOCKED) continue;
        int nb = ((int*)&env->surface.triangle_neighbors[tri])[edge];
        if (nb < 0 || boundary_tri[nb]) continue;
        int nb_edge = qm3_surface_local_edge_between(&env->surface, nb, verts[(edge + 1) % 3], verts[(edge + 2) % 3]);
        if (nb_edge >= 0 && (flags[3 * nb + nb_edge] & QM3_LOOP_EDGE_BLOCKED)) continue;
        qm3_loop_enqueue_tri(nb, side, queue, tail, side_mark, boundary_tri, conflict_tri, flood_conflicts);
    }
    return true;
}

static int qm3_classify_boundary_sample_side_local(const QuadMeshing3DEnv* env, const Qm3PathSegmentArray* loop_segments, int tri, Qm3Vec3 sample, uint32_t* conflicts) {
    if (tri < 0 || tri >= (int)env->surface.triangle_count || !env->continuous_ctx.tri2d_valid[tri]) return 0;
    Qm3Vec2 tri2d[3] = {
        env->continuous_ctx.tri2d_base[3 * tri + 0],
        env->continuous_ctx.tri2d_base[3 * tri + 1],
        env->continuous_ctx.tri2d_base[3 * tri + 2],
    };
    Qm3LocalCutSegment* cuts = NULL;
    int cut_count = 0, cut_cap = 0;
    Qm3LocalClassPoint* points = NULL;
    int point_count = 0, point_cap = 0;
    float tri_scale = fmaxf(qm3_v2_dist(tri2d[0], tri2d[1]), fmaxf(qm3_v2_dist(tri2d[1], tri2d[2]), qm3_v2_dist(tri2d[2], tri2d[0])));
    float seed_eps = fmaxf(tri_scale * 1e-4f, 1e-7f);
    float inset = seed_eps / fmaxf(tri_scale, 1e-8f);
    for (int i = 0; i < 3; ++i) {
        Qm3Vec2 corner = qm3_v2_add(qm3_v2_scale(tri2d[i], 1.0f - 2.0f * inset), qm3_v2_scale(qm3_v2_add(tri2d[(i + 1) % 3], tri2d[(i + 2) % 3]), inset));
        if (qm3_point_inside_triangle_2d(corner, tri2d)) qm3_local_class_point_push(&points, &point_count, &point_cap, (Qm3LocalClassPoint){.p = corner, .side = 0});
    }
    for (uint32_t i = 0; i < loop_segments->count; ++i) {
        const Qm3PathSegment* seg = &loop_segments->data[i];
        if (seg->tri != tri) continue;
        Qm3Vec2 a2, b2;
        if (!qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, tri, seg->a, &a2)) continue;
        if (!qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, tri, seg->b, &b2)) continue;
        Qm3Vec2 dir = qm3_v2_sub(b2, a2);
        float len = qm3_v2_len(dir);
        if (len <= 1e-8f) continue;
        qm3_local_cut_segment_push(&cuts, &cut_count, &cut_cap, (Qm3LocalCutSegment){.a = a2, .b = b2});
        Qm3Vec2 mid = qm3_v2_scale(qm3_v2_add(a2, b2), 0.5f);
        Qm3Vec2 n = qm3_v2_scale((Qm3Vec2){-dir.y, dir.x}, 1.0f / len);
        qm3_add_local_side_seed(&points, &point_count, &point_cap, mid, n, 1, seed_eps, tri2d);
        qm3_add_local_side_seed(&points, &point_count, &point_cap, mid, qm3_v2_scale(n, -1.0f), 2, seed_eps, tri2d);
    }
    Qm3Vec2 sample2;
    if (!qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, tri, sample, &sample2)) {
        free(cuts); free(points); return 0;
    }
    int sample_idx = point_count;
    qm3_local_class_point_push(&points, &point_count, &point_cap, (Qm3LocalClassPoint){.p = sample2, .side = 0});
    int* queue = (int*)malloc((size_t)fmaxf((float)point_count, 1.0f) * sizeof(int));
    QM3_ASSERT(queue != NULL);
    int comp_count = 0;
    for (int start = 0; start < point_count; ++start) {
        if (points[start].comp >= 0) continue;
        int head = 0, tail = 0;
        points[start].comp = comp_count;
        queue[tail++] = start;
        while (head < tail) {
            int pidx = queue[head++];
            for (int j = 0; j < point_count; ++j) {
                if (points[j].comp >= 0) continue;
                if (qm3_local_connection_crosses_cut(points[pidx].p, points[j].p, cuts, cut_count)) continue;
                points[j].comp = comp_count;
                queue[tail++] = j;
            }
        }
        comp_count++;
    }
    int side = points[sample_idx].comp >= 0 ? qm3_local_component_side(points, point_count, points[sample_idx].comp, conflicts) : 0;
    free(queue); free(cuts); free(points);
    return side;
}

static void qm3_rebuild_sample_to_graph(QuadMeshing3DEnv* env) {
    free(env->sample_to_graph);
    env->sample_to_graph = NULL;
    if (env->surface.sample_count == 0) return;
    env->sample_to_graph = (int32_t*)malloc((size_t)env->surface.sample_count * sizeof(int32_t));
    QM3_ASSERT(env->sample_to_graph != NULL);
    for (uint32_t i = 0; i < env->surface.sample_count; ++i) env->sample_to_graph[i] = -1;
    for (uint32_t vi = 0; vi < env->mesh.vertex_count; ++vi) {
        const Qm3MeshVertex* v = &env->mesh.vertices[vi];
        if (v->disabled || v->sample_id >= env->surface.sample_count) continue;
        if (env->sample_disabled && env->sample_disabled[v->sample_id]) continue;
        env->sample_to_graph[v->sample_id] = (int32_t)vi;
    }
}

static void qm3_reset_sample_disabled(QuadMeshing3DEnv* env) {
    free(env->sample_disabled);
    env->sample_disabled = NULL;
    env->disabled_sample_count = 0;
    if (env->surface.sample_count == 0) return;
    env->sample_disabled = (unsigned char*)calloc((size_t)env->surface.sample_count, sizeof(unsigned char));
    QM3_ASSERT(env->sample_disabled != NULL);
}

static void qm3_build_face_loop_segments(const Qm3Mesh* mesh, const Qm3MeshFace* face, Qm3PathSegmentArray* out) {
    qm3_path_segment_clear(out);
    if (!face || face->disabled) return;
    for (uint32_t i = 0; i < face->n; ++i) {
        uint32_t u = face->vertices[i];
        uint32_t eidx = face->edges[i];
        if (eidx >= mesh->edge_count) continue;
        const Qm3MeshEdge* edge = &mesh->edges[eidx];
        if (edge->segment_count == 0) continue;
        uint32_t begin = edge->segment_offset;
        uint32_t end = edge->segment_offset + edge->segment_count;
        if (end > mesh->edge_path_segments.count) end = mesh->edge_path_segments.count;
        if (edge->a == u) {
            for (uint32_t si = begin; si < end; ++si) qm3_path_segment_push(out, mesh->edge_path_segments.data[si]);
        } else {
            for (uint32_t si = end; si > begin; --si) {
                Qm3PathSegment seg = mesh->edge_path_segments.data[si - 1];
                Qm3Vec3 tmp = seg.a;
                seg.a = seg.b;
                seg.b = tmp;
                qm3_path_segment_push(out, seg);
            }
        }
    }
}

static Qm3LoopRemovalStats qm3_remove_samples_inside_loop(QuadMeshing3DEnv* env, const Qm3PathSegmentArray* loop_segments) {
    Qm3LoopRemovalStats stats = {0};
    if (loop_segments->count == 0 || !env->sample_disabled) return stats;
    Qm3IntArray boundary_list = {0};
    Qm3IntArray reached_left = {0};
    Qm3IntArray reached_right = {0};
    uint32_t tri_count = env->surface.triangle_count;
    unsigned char* flags = (unsigned char*)calloc((size_t)tri_count * 3, sizeof(unsigned char));
    unsigned char* vertex_side = (unsigned char*)calloc((size_t)tri_count * 3, sizeof(unsigned char));
    float* vertex_dist2 = (float*)malloc((size_t)tri_count * 3 * sizeof(float));
    unsigned char* boundary_tri = (unsigned char*)calloc((size_t)tri_count, sizeof(unsigned char));
    unsigned char* side_mark = (unsigned char*)calloc((size_t)tri_count, sizeof(unsigned char));
    unsigned char* conflict_tri = (unsigned char*)calloc((size_t)tri_count, sizeof(unsigned char));
    int* queue_left = (int*)malloc((size_t)tri_count * sizeof(int));
    int* queue_right = (int*)malloc((size_t)tri_count * sizeof(int));
    QM3_ASSERT(flags && vertex_side && vertex_dist2 && boundary_tri && side_mark && conflict_tri && queue_left && queue_right);
    for (uint32_t i = 0; i < tri_count * 3; ++i) vertex_dist2[i] = INFINITY;

    for (uint32_t i = 0; i < loop_segments->count; ++i) {
        qm3_loop_mark_segment_blocked_edges(env, &loop_segments->data[i], flags, vertex_side, vertex_dist2, boundary_tri);
    }
    for (uint32_t tri = 0; tri < tri_count; ++tri) if (boundary_tri[tri]) qm3_int_push(&boundary_list, (int)tri);

    int left_head = 0, left_tail = 0, right_head = 0, right_tail = 0;
    qm3_loop_seed_cut_edges_from_vertices(env, &boundary_list, vertex_side, boundary_tri, flags, queue_left, &left_tail, queue_right, &right_tail, side_mark, conflict_tri, &stats.classification_conflicts, &stats.flood_conflicts);

    bool left_seeded = left_tail > 0;
    bool right_seeded = right_tail > 0;
    if (!left_seeded && right_seeded) {
        stats.chosen_side = 1;
    } else if (!right_seeded && left_seeded) {
        stats.chosen_side = 2;
    } else {
        bool prefer_left = true;
        while (left_head < left_tail || right_head < right_tail) {
            bool popped = false;
            if (prefer_left) {
                popped = qm3_loop_flood_pop(env, flags, boundary_tri, 1, queue_left, &left_head, &left_tail, side_mark, conflict_tri, &stats.left_area, &stats.left_tris, &reached_left, &stats.flood_conflicts);
                if (!popped) qm3_loop_flood_pop(env, flags, boundary_tri, 2, queue_right, &right_head, &right_tail, side_mark, conflict_tri, &stats.right_area, &stats.right_tris, &reached_right, &stats.flood_conflicts);
            } else {
                popped = qm3_loop_flood_pop(env, flags, boundary_tri, 2, queue_right, &right_head, &right_tail, side_mark, conflict_tri, &stats.right_area, &stats.right_tris, &reached_right, &stats.flood_conflicts);
                if (!popped) qm3_loop_flood_pop(env, flags, boundary_tri, 1, queue_left, &left_head, &left_tail, side_mark, conflict_tri, &stats.left_area, &stats.left_tris, &reached_left, &stats.flood_conflicts);
            }
            prefer_left = !prefer_left;
            bool left_done = left_head >= left_tail;
            bool right_done = right_head >= right_tail;
            if (left_done && stats.right_area >= stats.left_area) break;
            if (right_done && stats.left_area >= stats.right_area) break;
        }

        if (stats.right_tris <= 0 && stats.left_tris > 0) stats.chosen_side = 2;
        else if (stats.left_tris <= 0 && stats.right_tris > 0) stats.chosen_side = 1;
        else if (stats.left_tris > 0 && stats.left_area <= stats.right_area) stats.chosen_side = 1;
        else if (stats.right_tris > 0) stats.chosen_side = 2;
    }

    if (stats.chosen_side != 0) {
        Qm3IntArray* chosen_tris = stats.chosen_side == 1 ? &reached_left : &reached_right;
        for (int ti = 0; ti < chosen_tris->size; ++ti) {
            uint32_t tri = (uint32_t)chosen_tris->data[ti];
            int begin = env->surface_topo.tri_sample_offsets[tri];
            int end = env->surface_topo.tri_sample_offsets[tri + 1];
            for (int si = begin; si < end; ++si) {
                uint32_t sample_id = (uint32_t)env->surface_topo.tri_sample_ids[si];
                if (!env->sample_disabled[sample_id]) {
                    env->sample_disabled[sample_id] = 1;
                    if (env->sample_to_graph) env->sample_to_graph[sample_id] = -1;
                    env->disabled_sample_count++;
                    stats.disabled_samples++;
                }
            }
        }
        for (int bi = 0; bi < boundary_list.size; ++bi) {
            uint32_t tri = (uint32_t)boundary_list.data[bi];
            int begin = env->surface_topo.tri_sample_offsets[tri];
            int end = env->surface_topo.tri_sample_offsets[tri + 1];
            for (int si = begin; si < end; ++si) {
                uint32_t sample_id = (uint32_t)env->surface_topo.tri_sample_ids[si];
                if (env->sample_disabled[sample_id]) continue;
                int side = qm3_classify_boundary_sample_side_local(env, loop_segments, (int)tri, env->surface.samples[sample_id].p, &stats.classification_conflicts);
                if (side == stats.chosen_side) {
                    env->sample_disabled[sample_id] = 1;
                    if (env->sample_to_graph) env->sample_to_graph[sample_id] = -1;
                    env->disabled_sample_count++;
                    stats.disabled_samples++;
                }
            }
        }
    }

    qm3_loop_debug_capture(&env->loop_debug, flags, boundary_tri, side_mark, conflict_tri, tri_count);
    free(flags); free(vertex_side); free(vertex_dist2); free(boundary_tri); free(side_mark); free(conflict_tri); free(queue_left); free(queue_right);
    qm3_int_free(&boundary_list); qm3_int_free(&reached_left); qm3_int_free(&reached_right);
    return stats;
}

static void qm3_prune_graph_after_loop_removal(QuadMeshing3DEnv* env, const Qm3MeshFace* face) {
    if (!env->sample_disabled) return;
    for (uint32_t vi = 0; vi < env->mesh.vertex_count; ++vi) {
        Qm3MeshVertex* v = &env->mesh.vertices[vi];
        if (v->disabled || v->sample_id >= env->surface.sample_count) continue;
        if (!env->sample_disabled[v->sample_id]) continue;
        if (qm3_mesh_vertex_is_face_boundary(face, vi)) continue;
        v->disabled = true;
        if (v->frontier_index >= 0) qm3_mesh_update_frontier_vertex(&env->mesh, vi);
    }
    for (uint32_t ei = 0; ei < env->mesh.edge_count; ++ei) {
        Qm3MeshEdge* e = &env->mesh.edges[ei];
        if (e->disabled || qm3_mesh_edge_is_face_boundary(face, ei)) continue;
        if (env->mesh.vertices[e->a].disabled || env->mesh.vertices[e->b].disabled) qm3_mesh_disable_edge(&env->mesh, ei);
    }
    for (uint32_t vi = 0; vi < env->mesh.vertex_count; ++vi) qm3_mesh_update_frontier_vertex(&env->mesh, vi);
}

static void qm3_insert_graph_tri_vertex(QuadMeshing3DEnv* env, uint32_t vi) {
    if (vi >= env->mesh.vertex_count) return;
    qm3_ensure_graph_tri_buckets(env);
    Qm3MeshVertex vertex = env->mesh.vertices[vi];
    if (vertex.disabled || vertex.degree == 0 || vertex.surface_tri < 0 || vertex.surface_tri >= (int32_t)env->surface.triangle_count) return;
    int32_t incident_edge = -1;
    for (uint32_t i = 0; i < vertex.degree; ++i) {
        size_t nidx = (size_t)vi * env->mesh.max_degree + i;
        int32_t eidx = env->mesh.neighbor_edges[nidx];
        if (eidx >= 0 && !env->mesh.edges[eidx].disabled) {
            incident_edge = eidx;
            break;
        }
    }
    if (incident_edge < 0) return;
    Qm3IntArray support = {0};
    qm3_point_support_tris(&env->surface, vertex.pos, vertex.surface_tri, &support);
    for (int i = 0; i < support.size; ++i) {
        int tri = support.data[i];
        Qm3Vec2 p;
        if (!qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, tri, vertex.pos, &p)) continue;
        qm3_graph_tri_vertex_push(&env->graph_tri_vertices[tri], (Qm3GraphTriVertex){
            .graph_vertex = vi,
            .incident_edge = (uint32_t)incident_edge,
            .p = p,
        });
        env->graph_tri_vertex_count++;
    }
    qm3_int_free(&support);
}

static void qm3_insert_graph_tri_edge_segments(QuadMeshing3DEnv* env, uint32_t ei) {
    if (ei >= env->mesh.edge_count) return;
    qm3_ensure_graph_tri_buckets(env);
    Qm3MeshEdge edge = env->mesh.edges[ei];
    if (edge.disabled || edge.segment_count == 0) return;
    uint32_t end = edge.segment_offset + edge.segment_count;
    if (end > env->mesh.edge_path_segments.count) end = env->mesh.edge_path_segments.count;
    for (uint32_t si = edge.segment_offset; si < end; ++si) {
        const Qm3PathSegment* seg = &env->mesh.edge_path_segments.data[si];
        if (seg->tri < 0 || seg->tri >= (int)env->surface.triangle_count) continue;
        Qm3Vec2 a, b;
        if (!qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, seg->tri, seg->a, &a)) continue;
        if (!qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, seg->tri, seg->b, &b)) continue;
        qm3_graph_tri_segment_push(&env->graph_tri_segments[seg->tri], (Qm3GraphTriSegment){
            .edge = ei,
            .a = a,
            .b = b,
        });
        env->graph_tri_segment_count++;
    }
}

static void qm3_build_graph_tri_segments(QuadMeshing3DEnv* env) {
    qm3_graph_tri_segments_free(&env->graph_tri_segments, env->surface.triangle_count);
    qm3_graph_tri_vertices_free(&env->graph_tri_vertices, env->surface.triangle_count);
    qm3_ensure_graph_tri_buckets(env);
    env->graph_tri_segment_count = 0;
    env->graph_tri_vertex_count = 0;

    for (uint32_t vi = 0; vi < env->mesh.vertex_count; ++vi) {
        qm3_insert_graph_tri_vertex(env, vi);
    }

    for (uint32_t ei = 0; ei < env->mesh.edge_count; ++ei) {
        qm3_insert_graph_tri_edge_segments(env, ei);
    }
}

static int32_t qm3_next_valid_candidate(const QuadMeshing3DEnv* env, int32_t start, int32_t step) {
    if (env->candidate_count == 0 || env->valid_candidate_count == 0) return -1;
    int32_t idx = start;
    int32_t count = (int32_t)env->candidate_count;
    for (uint32_t i = 0; i < env->candidate_count; ++i) {
        idx %= count;
        if (idx < 0) idx += count;
        if (env->candidates[idx].path_ok) return idx;
        idx += step;
    }
    return -1;
}

static void qm3_select_debug_candidate(QuadMeshing3DEnv* env, int32_t candidate_idx) {
    if (env->candidate_count == 0) {
        env->debug_candidate_idx = -1;
        return;
    }
    int32_t count = (int32_t)env->candidate_count;
    candidate_idx %= count;
    if (candidate_idx < 0) candidate_idx += count;
    env->debug_candidate_idx = candidate_idx;
}

static void qm3_compute_frontier_hop_distances(const QuadMeshing3DEnv* env, int source_fidx, int* dist, int* queue) {
    const Qm3Mesh* mesh = &env->mesh;
    for (uint32_t i = 0; i < mesh->frontier_count; ++i) dist[i] = -1;
    if (source_fidx < 0 || (uint32_t)source_fidx >= mesh->frontier_count) return;

    dist[source_fidx] = 0;
    queue[0] = source_fidx;
    uint32_t head = 0;
    uint32_t tail = 1;
    while (head < tail) {
        int fidx = queue[head++];
        uint32_t vidx = mesh->frontier[fidx];
        const Qm3MeshVertex* vertex = &mesh->vertices[vidx];
        int next_dist = dist[fidx] + 1;

        for (uint32_t i = 0; i < vertex->degree; ++i) {
            size_t nidx = (size_t)vidx * mesh->max_degree + i;
            int32_t nvidx = mesh->neighbors[nidx];
            int32_t eidx = mesh->neighbor_edges[nidx];
            if (nvidx < 0 || eidx < 0) continue;
            if ((uint32_t)nvidx >= mesh->vertex_count || (uint32_t)eidx >= mesh->edge_count) continue;
            int32_t nfidx = mesh->vertices[nvidx].frontier_index;
            if (nfidx < 0 || (uint32_t)nfidx >= mesh->frontier_count || dist[nfidx] >= 0) continue;
            if (mesh->edges[eidx].disabled || mesh->edges[eidx].face_count == 2) continue;

            dist[nfidx] = next_dist;
            queue[tail++] = nfidx;
        }
    }
}

static bool qm3_prevent_triangle_target_allowed(const QuadMeshing3DEnv* env, uint32_t target_fidx, const int* dist) {
    const Qm3Mesh* mesh = &env->mesh;
    if (target_fidx >= mesh->frontier_count) return false;
    int target_dist = dist[target_fidx];

    if (target_dist < 0) return false;
    if (target_dist % 2 != 0) return true;
    if (target_dist != 2) return false;

    uint32_t target = mesh->frontier[target_fidx];
    const Qm3MeshVertex* vertex = &mesh->vertices[target];
    for (uint32_t i = 0; i < vertex->degree; ++i) {
        size_t nidx = (size_t)target * mesh->max_degree + i;
        int32_t nvidx = mesh->neighbors[nidx];
        int32_t eidx = mesh->neighbor_edges[nidx];
        if (nvidx < 0 || eidx < 0) continue;
        if ((uint32_t)nvidx >= mesh->vertex_count || (uint32_t)eidx >= mesh->edge_count) continue;
        int32_t nfidx = mesh->vertices[nvidx].frontier_index;
        if (nfidx < 0 || (uint32_t)nfidx >= mesh->frontier_count || dist[nfidx] != 2) continue;
        if (mesh->edges[eidx].disabled || mesh->edges[eidx].face_count == 2) continue;
        return true;
    }

    return false;
}

static void qm3_compute_all_continuous_paths(QuadMeshing3DEnv* env) {
    qm3_path_clear(&env->candidate_path_points);
    qm3_path_segment_clear(&env->candidate_path_segments);
    env->continuous_paths_ok = 0;
    env->valid_candidate_count = 0;
    env->invalid_unreachable_count = 0;
    env->invalid_intersect_count = 0;
    env->timing_path_build_ms = 0.0;
    env->timing_intersection_ms = 0.0;
    env->timing_loop_removal_ms = 0.0;
    env->timing_intersection_tests = 0;
    if (env->candidate_count == 0) return;
    for (uint32_t i = 0; i < env->candidate_count; ++i) {
        env->candidates[i].path_offset = env->candidate_path_points.count;
        env->candidates[i].path_count = 0;
        env->candidates[i].segment_offset = env->candidate_path_segments.count;
        env->candidates[i].segment_count = 0;
        env->candidates[i].path_length = 0.0f;
        env->candidates[i].path_ok = 0;
        env->candidates[i].intersect_edge = -1;
        env->candidates[i].intersect_segment = -1;
    }
    if (env->source_frontier_idx < 0 || (uint32_t)env->source_frontier_idx >= env->mesh.frontier_count) return;
    uint32_t source_vidx = env->mesh.frontier[env->source_frontier_idx];
    Qm3MeshVertex source = env->mesh.vertices[source_vidx];
    if (source.surface_tri < 0) {
        env->invalid_unreachable_count = env->candidate_count;
        return;
    }

    float* stop_distances = (float*)malloc((size_t)env->candidate_count * sizeof(float));
    QM3_ASSERT(stop_distances != NULL);
    float max_stop_distance = 0.0f;
    uint32_t path_candidate_count = 0;
    for (uint32_t i = 0; i < env->candidate_count; ++i) {
        stop_distances[i] = INFINITY;
        if (env->candidates[i].validity != QM3_CANDIDATE_VALID) continue;
        float stop_distance = INFINITY;
        qm3_best_sample_prop_node(&env->surface, &env->prop_graph, &env->fmm, env->candidates[i].sample_id, &stop_distance);
        stop_distances[i] = stop_distance;
        path_candidate_count++;
        if (isfinite(stop_distance)) max_stop_distance = fmaxf(max_stop_distance, stop_distance * 1.0001f + 1e-6f);
    }
    if (path_candidate_count == 0) {
        free(stop_distances);
        return;
    }

    Qm3Path tmp = {0};
    Qm3PathSegmentArray tmp_segments = {0};
    double path_start = qm3_time_ms();
    bool shared_ok = qm3_build_shared_source_windows(
        &env->surface,
        &env->prop_graph,
        &env->continuous_ctx,
        source.pos,
        source.surface_tri,
        max_stop_distance
    );
    env->timing_path_build_ms += qm3_time_ms() - path_start;
    if (!shared_ok) {
        env->invalid_unreachable_count += path_candidate_count;
        free(stop_distances);
        qm3_path_free(&tmp);
        qm3_path_segment_free(&tmp_segments);
        return;
    }

    for (uint32_t i = 0; i < env->candidate_count; ++i) {
        Qm3TargetCandidate* candidate = &env->candidates[i];
        if (candidate->validity != QM3_CANDIDATE_VALID) continue;
        uint32_t sample_id = candidate->sample_id;
        const Qm3SurfaceSample* sample = &env->surface.samples[sample_id];
        double emit_start = qm3_time_ms();
        qm3_path_clear(&tmp);
        qm3_path_segment_clear(&tmp_segments);
        bool ok = false;
        if (source.surface_tri == (int32_t)sample->tri) {
            qm3_path_push(&tmp, source.pos);
            qm3_path_push(&tmp, sample->p);
            qm3_path_segment_push(&tmp_segments, (Qm3PathSegment){.tri = source.surface_tri, .a = source.pos, .b = sample->p});
            ok = true;
        } else if (isfinite(stop_distances[i])) {
            int wi = qm3_find_best_target_window(&env->surface, &env->continuous_ctx, sample->p, (int)sample->tri, stop_distances[i], NULL);
            ok = wi >= 0 && qm3_append_window_chain_path(&env->surface, source.pos, &env->continuous_ctx.windows, wi, sample->p, &tmp, &tmp_segments);
        }
        env->timing_path_build_ms += qm3_time_ms() - emit_start;
        candidate->path_offset = env->candidate_path_points.count;
        candidate->segment_offset = env->candidate_path_segments.count;
        if (ok && tmp.count >= 2 && tmp_segments.count > 0) {
            env->continuous_paths_ok++;
            for (uint32_t j = 0; j < tmp_segments.count; ++j) qm3_path_segment_push(&env->candidate_path_segments, tmp_segments.data[j]);
            candidate->segment_count = tmp_segments.count;
            int32_t intersect_edge = -1;
            int32_t intersect_segment = -1;
            double intersect_start = qm3_time_ms();
            bool intersects = qm3_candidate_segments_intersect_graph(env, source_vidx, candidate->graph_vertex, &tmp_segments, &env->timing_intersection_tests, &intersect_edge, &intersect_segment);
            env->timing_intersection_ms += qm3_time_ms() - intersect_start;
            if (intersects) {
                candidate->validity = QM3_CANDIDATE_INTERSECT;
                candidate->intersect_edge = intersect_edge;
                candidate->intersect_segment = intersect_segment;
                env->invalid_intersect_count++;
                continue;
            }
            for (uint32_t j = 0; j < tmp.count; ++j) qm3_path_push(&env->candidate_path_points, tmp.points[j]);
            candidate->path_count = tmp.count;
            candidate->path_length = tmp.length;
            candidate->path_ok = 1;
            env->valid_candidate_count++;
        } else {
            candidate->validity = QM3_CANDIDATE_UNREACHABLE;
            env->invalid_unreachable_count++;
        }
    }
    free(stop_distances);
    qm3_path_free(&tmp);
    qm3_path_segment_free(&tmp_segments);
    env->debug_candidate_idx = env->candidate_count > 0 ? 0 : -1;
}

static void qm3_compute_source_candidates(QuadMeshing3DEnv* env) {
    env->candidate_count = 0;
    env->valid_candidate_count = 0;
    env->candidate_existing_count = 0;
    env->candidate_sample_count = 0;
    env->continuous_paths_ok = 0;
    env->invalid_unreachable_count = 0;
    env->invalid_intersect_count = 0;
    env->timing_target_query_ms = 0.0;
    env->timing_path_build_ms = 0.0;
    env->timing_intersection_ms = 0.0;
    env->timing_loop_removal_ms = 0.0;
    env->timing_intersection_tests = 0;
    env->target_candidate_idx = -1;
    env->debug_candidate_idx = -1;
    env->selected_path_ok = false;
    qm3_path_clear(&env->selected_path);
    if (env->source_frontier_idx < 0 || (uint32_t)env->source_frontier_idx >= env->mesh.frontier_count) return;

    uint32_t source_vidx = env->mesh.frontier[env->source_frontier_idx];
    const Qm3MeshVertex* source_vertex = &env->mesh.vertices[source_vidx];
    Qm3Vec3 source_pos = source_vertex->pos;
    float radius = env->candidate_radius;
    if (radius <= 0.0f) {
        float diag = qm3_surface_diag(&env->surface);
        if (diag <= 1e-8f) diag = 1.0f;
        float ratio = env->candidate_radius_ratio > 0.0f ? env->candidate_radius_ratio : 0.05f;
        radius = diag * ratio;
        env->candidate_radius = radius;
    }

    if (!env->geodesic_ready) return;
    double query_start = qm3_time_ms();
    uint32_t* query_samples = NULL;
    uint32_t query_count = 0;
    uint32_t query_cap = 0;
    qm3_geodesic_candidate_query(
        &env->surface,
        &env->prop_graph,
        &env->surface_topo,
        source_pos,
        source_vertex->surface_tri,
        radius,
        &env->fmm,
        &query_samples,
        &query_count,
        &query_cap
    );

    if (!env->sample_to_graph && env->surface.sample_count > 0) qm3_rebuild_sample_to_graph(env);

    int* frontier_hop_distance = NULL;
    int* frontier_hop_queue = NULL;
    if (env->prevent_triangles && env->mesh.frontier_count > 0) {
        frontier_hop_distance = (int*)malloc((size_t)env->mesh.frontier_count * sizeof(int));
        frontier_hop_queue = (int*)malloc((size_t)env->mesh.frontier_count * sizeof(int));
        QM3_ASSERT(frontier_hop_distance != NULL && frontier_hop_queue != NULL);
        qm3_compute_frontier_hop_distances(env, env->source_frontier_idx, frontier_hop_distance, frontier_hop_queue);
    }

    for (uint32_t i = 0; i < query_count; ++i) {
        uint32_t sample_id = query_samples[i];
        if (sample_id >= env->surface.sample_count) continue;
        if (env->sample_disabled && env->sample_disabled[sample_id]) continue;
        if (env->sample_to_graph && env->sample_to_graph[sample_id] >= 0) continue;
        Qm3CandidateValidity validity = QM3_CANDIDATE_VALID;
        if (source_vertex->disabled) validity = QM3_CANDIDATE_DISABLED;
        else if (source_vertex->degree >= env->mesh.max_degree) validity = QM3_CANDIDATE_MAX_DEGREE;
        qm3_candidate_push(env, QM3_TARGET_NEW_SAMPLE, UINT32_MAX, sample_id, validity);
    }

    for (uint32_t fi = 0; fi < env->mesh.frontier_count; ++fi) {
        uint32_t target_vidx = env->mesh.frontier[fi];
        if (target_vidx >= env->mesh.vertex_count) continue;
        if (target_vidx == source_vidx) continue;
        const Qm3MeshVertex* target_vertex = &env->mesh.vertices[target_vidx];
        if (target_vertex->sample_id >= env->surface.sample_count) continue;
        float distance = INFINITY;
        qm3_best_sample_prop_node(&env->surface, &env->prop_graph, &env->fmm, target_vertex->sample_id, &distance);
        if (!(distance <= radius)) continue;
        Qm3CandidateValidity validity = QM3_CANDIDATE_VALID;
        if (source_vertex->disabled || target_vertex->disabled) validity = QM3_CANDIDATE_DISABLED;
        else if (qm3_mesh_edge_index(&env->mesh, source_vidx, target_vidx) >= 0) validity = QM3_CANDIDATE_EXISTING_EDGE;
        else if (source_vertex->degree >= env->mesh.max_degree || target_vertex->degree >= env->mesh.max_degree) validity = QM3_CANDIDATE_MAX_DEGREE;
        else if (frontier_hop_distance && !qm3_prevent_triangle_target_allowed(env, fi, frontier_hop_distance)) validity = QM3_CANDIDATE_TRIANGLE;
        qm3_candidate_push(env, QM3_TARGET_EXISTING_FRONTIER, target_vidx, target_vertex->sample_id, validity);
    }
    free(frontier_hop_distance);
    free(frontier_hop_queue);
    free(query_samples);
    env->timing_target_query_ms = qm3_time_ms() - query_start;
    qm3_compute_all_continuous_paths(env);
    env->target_candidate_idx = qm3_next_valid_candidate(env, 0, 1);
    env->debug_candidate_idx = env->target_candidate_idx >= 0 ? env->target_candidate_idx : (env->candidate_count > 0 ? 0 : -1);
}

static void qm3_update_selected_path(QuadMeshing3DEnv* env) {
    env->selected_path_ok = false;
    qm3_path_clear(&env->selected_path);
    if (env->source_frontier_idx < 0 || (uint32_t)env->source_frontier_idx >= env->mesh.frontier_count) return;
    if (env->target_candidate_idx < 0 || (uint32_t)env->target_candidate_idx >= env->candidate_count) return;

    uint32_t target = (uint32_t)env->target_candidate_idx;
    const Qm3TargetCandidate* candidate = &env->candidates[target];
    if (!candidate->path_ok) return;
    uint32_t offset = candidate->path_offset;
    uint32_t count = candidate->path_count;
    for (uint32_t i = 0; i < count; ++i) qm3_path_push(&env->selected_path, env->candidate_path_points.points[offset + i]);
    env->selected_path.length = candidate->path_length;
    env->selected_path_ok = count >= 2;
}

static void qm3_select_target_candidate(QuadMeshing3DEnv* env, int32_t target_candidate_idx) {
    if (env->candidate_count == 0 || env->valid_candidate_count == 0) {
        env->target_candidate_idx = -1;
        env->selected_path_ok = false;
        qm3_path_clear(&env->selected_path);
        return;
    }
    int32_t step = target_candidate_idx >= env->target_candidate_idx ? 1 : -1;
    if (env->target_candidate_idx < 0) step = 1;
    env->target_candidate_idx = qm3_next_valid_candidate(env, target_candidate_idx, step);
    qm3_update_selected_path(env);
}

static int32_t qm3_nth_valid_candidate(const QuadMeshing3DEnv* env, uint32_t slot) {
    for (uint32_t i = 0; i < env->candidate_count; ++i) {
        if (!env->candidates[i].path_ok) continue;
        if (slot == 0) return (int32_t)i;
        slot--;
    }
    return -1;
}

static void qm3_clear_target_selection(QuadMeshing3DEnv* env) {
    env->candidate_count = 0;
    env->valid_candidate_count = 0;
    env->candidate_existing_count = 0;
    env->candidate_sample_count = 0;
    env->continuous_paths_ok = 0;
    env->invalid_unreachable_count = 0;
    env->invalid_intersect_count = 0;
    env->source_frontier_idx = -1;
    env->target_candidate_idx = -1;
    env->debug_candidate_idx = -1;
    env->selected_path_ok = false;
    qm3_path_clear(&env->selected_path);
}

static float qm3_compute_face_reward(QuadMeshing3DEnv* env, Qm3MeshFace* face);
static float qm3_compute_frontier_potential(QuadMeshing3DEnv* env);
static float qm3_edge_chord_length(const Qm3Mesh* mesh, const Qm3MeshEdge* edge);

static void qm3_add_reward(QuadMeshing3DEnv* env, float reward) {
    if (env->rewards) env->rewards[0] += reward;
    env->last_reward += reward;
}

static bool qm3_commit_target_candidate(QuadMeshing3DEnv* env, uint32_t candidate_idx) {
    if (env->source_frontier_idx < 0 || (uint32_t)env->source_frontier_idx >= env->mesh.frontier_count) {
        env->last_action_result = QM3_ACTION_INVALID_SOURCE;
        return false;
    }
    if (candidate_idx >= env->candidate_count) {
        env->last_action_result = QM3_ACTION_INVALID_TARGET;
        return false;
    }
    Qm3TargetCandidate* candidate = &env->candidates[candidate_idx];
    if (!candidate->path_ok || candidate->path_count < 2 || candidate->segment_count == 0) {
        env->last_action_result = QM3_ACTION_INVALID_TARGET;
        return false;
    }

    uint32_t source_vidx = env->mesh.frontier[env->source_frontier_idx];
    uint32_t target_vidx = candidate->graph_vertex;
    bool inserted_vertex = false;
    if (candidate->kind == QM3_TARGET_NEW_SAMPLE) {
        if (!env->sample_to_graph && env->surface.sample_count > 0) qm3_rebuild_sample_to_graph(env);
        if (candidate->sample_id >= env->surface.sample_count || !env->sample_to_graph) {
            env->last_action_result = QM3_ACTION_INVALID_TARGET;
            return false;
        }
        uint32_t before_vertices = env->mesh.vertex_count;
        target_vidx = qm3_mesh_vertex_for_sample(&env->mesh, &env->surface, env->sample_to_graph, candidate->sample_id);
        inserted_vertex = env->mesh.vertex_count > before_vertices;
    } else if (target_vidx >= env->mesh.vertex_count) {
        env->last_action_result = QM3_ACTION_INVALID_TARGET;
        return false;
    }

    if (source_vidx == target_vidx || qm3_mesh_edge_index(&env->mesh, source_vidx, target_vidx) >= 0) {
        env->last_action_result = QM3_ACTION_INVALID_TARGET;
        return false;
    }
    if (env->mesh.vertices[source_vidx].degree >= env->mesh.max_degree || env->mesh.vertices[target_vidx].degree >= env->mesh.max_degree) {
        env->last_action_result = QM3_ACTION_INVALID_TARGET;
        return false;
    }

    uint32_t path_end = candidate->path_offset + candidate->path_count;
    uint32_t segment_end = candidate->segment_offset + candidate->segment_count;
    if (path_end > env->candidate_path_points.count || segment_end > env->candidate_path_segments.count) {
        env->last_action_result = QM3_ACTION_INVALID_TARGET;
        return false;
    }

    uint32_t before_edges = env->mesh.edge_count;
    uint32_t edge_idx = qm3_mesh_add_edge_with_path(
        &env->mesh,
        source_vidx,
        target_vidx,
        0,
        &env->candidate_path_points.points[candidate->path_offset],
        candidate->path_count,
        &env->candidate_path_segments.data[candidate->segment_offset],
        candidate->segment_count,
        candidate->path_length
    );
    if (env->mesh.edge_count > before_edges) {
        qm3_insert_graph_tri_edge_segments(env, edge_idx);
        if (inserted_vertex) qm3_insert_graph_tri_vertex(env, target_vidx);

        Qm3PathSegmentArray loop_segments = {0};
        uint32_t cycles[64];
        uint32_t registered = 0;
        uint32_t quad_count = qm3_mesh_detect_quads(&env->mesh, source_vidx, target_vidx, cycles, 16);
        for (uint32_t i = 0; i < quad_count && registered < 2; ++i) {
            uint32_t before_faces = env->mesh.face_count;
            if (qm3_mesh_register_face(&env->mesh, &cycles[i * 4], 4)) {
                registered++;
                double loop_start = qm3_time_ms();
                Qm3MeshFace* face = &env->mesh.faces[before_faces];
                qm3_add_reward(env, 0.5f * qm3_compute_face_reward(env, face));
                qm3_build_face_loop_segments(&env->mesh, face, &loop_segments);
                env->loop_stats = qm3_remove_samples_inside_loop(env, &loop_segments);
                qm3_prune_graph_after_loop_removal(env, face);
                env->timing_loop_removal_ms = qm3_time_ms() - loop_start;
            }
        }
        uint32_t tri_count = qm3_mesh_detect_triangles(&env->mesh, source_vidx, target_vidx, cycles, 16);
        for (uint32_t i = 0; i < tri_count && registered < 2; ++i) {
            uint32_t before_faces = env->mesh.face_count;
            if (qm3_mesh_register_face(&env->mesh, &cycles[i * 3], 3)) {
                registered++;
                double loop_start = qm3_time_ms();
                Qm3MeshFace* face = &env->mesh.faces[before_faces];
                qm3_add_reward(env, qm3_compute_face_reward(env, face));
                qm3_build_face_loop_segments(&env->mesh, face, &loop_segments);
                env->loop_stats = qm3_remove_samples_inside_loop(env, &loop_segments);
                qm3_prune_graph_after_loop_removal(env, face);
                env->timing_loop_removal_ms = qm3_time_ms() - loop_start;
            }
        }
        qm3_path_segment_free(&loop_segments);
    }

    env->last_action_result = QM3_ACTION_TARGET_COMMITTED;
    env->phase = QM3_PHASE_SOURCE;
    qm3_clear_target_selection(env);
    float new_potential = qm3_compute_frontier_potential(env);
    qm3_add_reward(env, env->potential_beta * (env->potential_gamma * new_potential - env->potential));
    env->potential = new_potential;
    env->last_frontier_potential = new_potential;
    return true;
}

static void qm3_select_source_frontier(QuadMeshing3DEnv* env, int32_t source_frontier_idx) {
    if (env->mesh.frontier_count == 0) {
        qm3_clear_target_selection(env);
        env->phase = QM3_PHASE_SOURCE;
        env->last_action_result = QM3_ACTION_INVALID_SOURCE;
        return;
    }
    if (source_frontier_idx < 0) source_frontier_idx = 0;
    env->source_frontier_idx = source_frontier_idx % (int32_t)env->mesh.frontier_count;
    qm3_compute_source_candidates(env);
    qm3_update_selected_path(env);
    env->phase = QM3_PHASE_TARGET;
    env->last_action_result = env->valid_candidate_count > 0 ? QM3_ACTION_SOURCE_SELECTED : QM3_ACTION_NO_VALID_TARGET;
}

static void quad_meshing_3d_init(QuadMeshing3DEnv* env) {
    qm3_surface_init(&env->surface);
    qm3_mesh_init(&env->mesh, env->max_degree > 0 ? (uint32_t)env->max_degree : 16);
    env->loaded_shape = -1;
    env->source_frontier_idx = -1;
    env->target_candidate_idx = -1;
    env->debug_candidate_idx = -1;
    env->phase = QM3_PHASE_SOURCE;
    env->last_action_result = QM3_ACTION_NONE;
    env->geodesic_ready = false;
}

static void qm3_update_episode_max_length(QuadMeshing3DEnv* env) {
    float target_area = fmaxf(env->target_quad_area, 1e-12f);
    float ratio = env->episode_max_length_ratio > 0.0f ? env->episode_max_length_ratio : 1.5f;
    env->episode_max_length = (int)ceilf((env->surface.info.total_area / target_area) * ratio);
    if (env->episode_max_length < 1) env->episode_max_length = 1;
}

static void qm3_update_target_scale(QuadMeshing3DEnv* env) {
    float total = 0.0f;
    uint32_t count = 0;
    for (uint32_t i = 0; i < env->mesh.edge_count; ++i) {
        Qm3MeshEdge* edge = &env->mesh.edges[i];
        if (edge->disabled) continue;
        total += qm3_edge_chord_length(&env->mesh, edge);
        count++;
    }
    float avg = count > 0 ? total / (float)count : sqrtf(fmaxf(env->surface.info.total_area, 1e-12f));
    env->starting_frontier_edge_length = avg;
    float ratio = env->target_edge_length_ratio > 0.0f ? env->target_edge_length_ratio : 1.0f;
    env->target_edge_length = fmaxf(avg * ratio, 1e-6f);
    env->target_quad_area = env->target_edge_length * env->target_edge_length;
}

static void qm3_add_log(QuadMeshing3DEnv* env) {
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->episode_length;
    env->log.n += 1.0f;
}

static Qm3Vec3 qm3_cross(Qm3Vec3 a, Qm3Vec3 b) {
    return (Qm3Vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

static float qm3_compute_area_quality(float area, float target_area) {
    float log_ratio = logf(fmaxf(area / fmaxf(target_area, 1e-12f), 1e-6f));
    return expf(-2.0f * log_ratio * log_ratio);
}

static float qm3_compute_edge_length_quality(float edge_length, float target_edge_length) {
    float log_ratio = logf(fmaxf(edge_length / fmaxf(target_edge_length, 1e-12f), 1e-6f));
    return expf(-8.0f * log_ratio * log_ratio);
}

static float qm3_open_edge_length_cost_from_chord(float chord, float target_edge_length) {
    return 1.0f - qm3_compute_edge_length_quality(chord, target_edge_length);
}

static float qm3_path_chord_cost(float path, float chord, float target_edge_length) {
    float log_diff = logf(fmaxf(1.0f + (path - chord) / target_edge_length, 1e-12f));
    return 1.0f - expf(-32.0f * log_diff * log_diff);
}

static float qm3_right_angle_cost(float angle) {
    float target = 0.5f * PI;
    if (angle >= target) return 0.0f;
    return fminf((target - angle) / target, 1.0f);
}

static float qm3_edge_chord_length(const Qm3Mesh* mesh, const Qm3MeshEdge* edge) {
    return qm3_distance(mesh->vertices[edge->a].pos, mesh->vertices[edge->b].pos);
}

static Qm3Vec3 qm3_project_tangent(Qm3Vec3 v, Qm3Vec3 n) {
    return qm3_sub(v, qm3_scale(n, qm3_dot(v, n)));
}

static float qm3_face_planarity_quality(const Qm3Mesh* mesh, const Qm3MeshFace* face, float target_edge_length) {
    if (face->n < 4) return 1.0f;
    Qm3Vec3 p0 = mesh->vertices[face->vertices[0]].pos;
    Qm3Vec3 p1 = mesh->vertices[face->vertices[1]].pos;
    Qm3Vec3 p2 = mesh->vertices[face->vertices[2]].pos;
    Qm3Vec3 n = qm3_cross(qm3_sub(p1, p0), qm3_sub(p2, p0));
    float nlen = qm3_len(n);
    if (nlen <= 1e-12f) return 0.0f;
    n = qm3_scale(n, 1.0f / nlen);
    float max_dist = 0.0f;
    for (uint32_t i = 0; i < face->n; ++i) {
        Qm3Vec3 p = mesh->vertices[face->vertices[i]].pos;
        max_dist = fmaxf(max_dist, fabsf(qm3_dot(qm3_sub(p, p0), n)));
    }
    float ratio = max_dist / fmaxf(target_edge_length, 1e-12f);
    return expf(-2.0f * ratio * ratio);
}

static float qm3_face_element_quality(const Qm3Mesh* mesh, const Qm3MeshFace* face) {
    if (face->n != 4) return 0.0f;

    float emin2 = INFINITY;
    float angle_min = INFINITY;
    float angle_max = 0.0f;
    for (uint32_t i = 0; i < 4; ++i) {
        Qm3Vec3 p = mesh->vertices[face->vertices[i]].pos;
        Qm3Vec3 prev = qm3_sub(mesh->vertices[face->vertices[(i + 3) % 4]].pos, p);
        Qm3Vec3 next = qm3_sub(mesh->vertices[face->vertices[(i + 1) % 4]].pos, p);
        float plen = qm3_len(prev);
        float nlen = qm3_len(next);
        if (plen <= 1e-12f || nlen <= 1e-12f) return 0.0f;
        emin2 = fminf(emin2, nlen * nlen);
        float c = fminf(fmaxf(qm3_dot(prev, next) / (plen * nlen), -1.0f), 1.0f);
        float angle = acosf(c);
        angle_min = fminf(angle_min, angle);
        angle_max = fmaxf(angle_max, angle);
    }

    Qm3Vec3 d02 = qm3_sub(mesh->vertices[face->vertices[0]].pos, mesh->vertices[face->vertices[2]].pos);
    Qm3Vec3 d13 = qm3_sub(mesh->vertices[face->vertices[1]].pos, mesh->vertices[face->vertices[3]].pos);
    float dmax = sqrtf(fmaxf(qm3_dot(d02, d02), qm3_dot(d13, d13)));
    if (dmax <= 1e-12f || angle_max <= 1e-12f) return 0.0f;
    return sqrtf(fmaxf(0.0f, sqrtf(2.0f) * sqrtf(emin2) * angle_min / (dmax * angle_max)));
}

static float qm3_face_curvature_quality(const Qm3Mesh* mesh, const Qm3MeshFace* face, float target_edge_length) {
    if (face->n == 0) return 0.0f;
    float quality = 0.0f;
    for (uint32_t i = 0; i < face->n; ++i) {
        const Qm3MeshEdge* edge = &mesh->edges[face->edges[i]];
        float chord = qm3_edge_chord_length(mesh, edge);
        float path = edge->path_length > 0.0f ? edge->path_length : chord;
        quality += 1.0f - qm3_path_chord_cost(path, chord, target_edge_length);
    }
    return quality / (float)face->n;
}

static float qm3_compute_face_quality(QuadMeshing3DEnv* env, Qm3MeshFace* face) {
    float element_quality = qm3_face_element_quality(&env->mesh, face);
    float area_quality = qm3_compute_area_quality(face->area, env->target_quad_area);
    float planarity_quality = qm3_face_planarity_quality(&env->mesh, face, env->target_edge_length);
    float curvature_quality = qm3_face_curvature_quality(&env->mesh, face, env->target_edge_length);
    float quality = element_quality * area_quality * planarity_quality * curvature_quality;

    face->quality = quality;
    env->last_face_area = face->area;
    env->last_area_quality = area_quality;
    env->last_element_quality = element_quality;
    env->last_planarity_quality = planarity_quality;
    env->last_curvature_quality = curvature_quality;
    return quality;
}

static float qm3_compute_face_reward(QuadMeshing3DEnv* env, Qm3MeshFace* face) {
    if (face->n == 3) return env->reward_triangle;
    float q = qm3_compute_face_quality(env, face);
    return env->base_quad_reward + (1.0f - env->base_quad_reward) * q;
}

static float qm3_pressure_ratio(float value, float safe_ratio, float max_value) {
    float denom = (1.0f - safe_ratio) * max_value;
    if (denom <= 0.0f) return value > max_value ? 1.0f : 0.0f;
    float pressure = fmaxf(0.0f, (value - safe_ratio * max_value) / denom);
    return pressure * pressure;
}

static int qm3_edge_representative_tri(const Qm3Mesh* mesh, const Qm3MeshEdge* edge) {
    if (edge->segment_count > 0 && edge->segment_offset < mesh->edge_path_segments.count) {
        int tri = mesh->edge_path_segments.data[edge->segment_offset].tri;
        if (tri >= 0) return tri;
    }
    return mesh->vertices[edge->a].surface_tri;
}

static float qm3_alignment_cost_from_dir_and_tri(const QuadMeshing3DEnv* env, Qm3Vec3 dir, int tri);

static float qm3_open_edge_alignment_cost(const QuadMeshing3DEnv* env, const Qm3MeshEdge* edge) {
    int tri = qm3_edge_representative_tri(&env->mesh, edge);
    if (tri < 0 || tri >= (int)env->surface.triangle_count) return 0.0f;
    Qm3Vec3 a = env->mesh.vertices[edge->a].pos;
    Qm3Vec3 b = env->mesh.vertices[edge->b].pos;
    Qm3Vec3 dir = qm3_sub(b, a);
    return qm3_alignment_cost_from_dir_and_tri(env, dir, tri);
}

static float qm3_alignment_cost_from_dir_and_tri(const QuadMeshing3DEnv* env, Qm3Vec3 dir, int tri) {
    if (tri < 0 || tri >= (int)env->surface.triangle_count) return 0.0f;
    float len = qm3_len(dir);
    if (len <= 1e-12f) return 1.0f;
    dir = qm3_scale(dir, 1.0f / len);
    float du = qm3_dot(dir, qm3_normalize(env->surface.face_dir_u[tri]));
    float dv = qm3_dot(dir, qm3_normalize(env->surface.face_dir_v[tri]));
    float s = fmaxf(du * du, dv * dv);
    return fminf(fmaxf(4.0f * s * (1.0f - s), 0.0f), 1.0f);
}

static float qm3_new_edge_endpoint_angle_cost(const Qm3Mesh* mesh, uint32_t vidx, Qm3Vec3 other_pos) {
    if (vidx >= mesh->vertex_count) return -1.0f;
    const Qm3MeshVertex* vertex = &mesh->vertices[vidx];
    Qm3Vec3 normal = qm3_normalize(vertex->normal);
    Qm3Vec3 new_edge = qm3_project_tangent(qm3_sub(other_pos, vertex->pos), normal);
    float new_len = qm3_len(new_edge);
    if (new_len <= 1e-12f) return 1.0f;

    float worst_cost = 0.0f;
    bool saw_open_edge = false;
    for (uint32_t i = 0; i < vertex->degree; ++i) {
        size_t ni = (size_t)vidx * mesh->max_degree + i;
        int32_t ei = mesh->neighbor_edges[ni];
        if (ei < 0 || mesh->edges[ei].disabled || mesh->edges[ei].face_count >= 2) continue;
        int32_t neighbor = mesh->neighbors[ni];
        if (neighbor < 0 || (uint32_t)neighbor >= mesh->vertex_count) continue;
        Qm3Vec3 existing = qm3_project_tangent(qm3_sub(mesh->vertices[neighbor].pos, vertex->pos), normal);
        float existing_len = qm3_len(existing);
        if (existing_len <= 1e-12f) continue;
        float c = fminf(fmaxf(qm3_dot(new_edge, existing) / (new_len * existing_len), -1.0f), 1.0f);
        float angle = acosf(c);
        float cost = qm3_right_angle_cost(angle);
        worst_cost = fmaxf(worst_cost, cost);
        saw_open_edge = true;
    }
    return saw_open_edge ? worst_cost : -1.0f;
}

static float qm3_vertex_open_angle_cost(const QuadMeshing3DEnv* env, uint32_t vidx) {
    const Qm3Mesh* mesh = &env->mesh;
    const Qm3MeshVertex* vertex = &mesh->vertices[vidx];
    float worst_cost = 0.0f;
    uint32_t count = 0;
    for (uint32_t i = 0; i < vertex->degree; ++i) {
        size_t ni = (size_t)vidx * mesh->max_degree + i;
        int32_t ei = mesh->neighbor_edges[ni];
        if (ei < 0 || mesh->edges[ei].disabled || mesh->edges[ei].face_count >= 2) continue;
        Qm3Vec3 ai = qm3_sub(mesh->vertices[mesh->neighbors[ni]].pos, vertex->pos);
        ai = qm3_project_tangent(ai, qm3_normalize(vertex->normal));
        float ail = qm3_len(ai);
        if (ail <= 1e-12f) continue;
        count++;
        for (uint32_t j = i + 1; j < vertex->degree; ++j) {
            size_t nj = (size_t)vidx * mesh->max_degree + j;
            int32_t ej = mesh->neighbor_edges[nj];
            if (ej < 0 || mesh->edges[ej].disabled || mesh->edges[ej].face_count >= 2) continue;
            Qm3Vec3 aj = qm3_sub(mesh->vertices[mesh->neighbors[nj]].pos, vertex->pos);
            aj = qm3_project_tangent(aj, qm3_normalize(vertex->normal));
            float ajl = qm3_len(aj);
            if (ajl <= 1e-12f) continue;
            float c = fminf(fmaxf(qm3_dot(ai, aj) / (ail * ajl), -1.0f), 1.0f);
            worst_cost = fmaxf(worst_cost, qm3_right_angle_cost(acosf(c)));
        }
    }
    if (count < 2) return 0.0f;
    return worst_cost;
}

static Qm3Vec3 qm3_candidate_target_pos3(const QuadMeshing3DEnv* env, const Qm3TargetCandidate* candidate) {
    if (candidate->kind == QM3_TARGET_EXISTING_FRONTIER && candidate->graph_vertex < env->mesh.vertex_count) {
        return env->mesh.vertices[candidate->graph_vertex].pos;
    }
    if (candidate->sample_id < env->surface.sample_count) return env->surface.samples[candidate->sample_id].p;
    return (Qm3Vec3){0.0f, 0.0f, 0.0f};
}

static float qm3_candidate_chord_length(const QuadMeshing3DEnv* env, const Qm3TargetCandidate* candidate) {
    if (env->source_frontier_idx < 0 || (uint32_t)env->source_frontier_idx >= env->mesh.frontier_count) return 0.0f;
    uint32_t source_vidx = env->mesh.frontier[env->source_frontier_idx];
    return qm3_distance(env->mesh.vertices[source_vidx].pos, qm3_candidate_target_pos3(env, candidate));
}

static int qm3_candidate_representative_tri(const QuadMeshing3DEnv* env, const Qm3TargetCandidate* candidate) {
    if (candidate->segment_count > 0 && candidate->segment_offset < env->candidate_path_segments.count) {
        int tri = env->candidate_path_segments.data[candidate->segment_offset].tri;
        if (tri >= 0) return tri;
    }
    if (candidate->sample_id < env->surface.sample_count) return (int)env->surface.samples[candidate->sample_id].tri;
    return -1;
}

static float qm3_candidate_alignment_cost(const QuadMeshing3DEnv* env, const Qm3TargetCandidate* candidate) {
    if (env->source_frontier_idx < 0 || (uint32_t)env->source_frontier_idx >= env->mesh.frontier_count) return 0.0f;
    int tri = qm3_candidate_representative_tri(env, candidate);
    if (tri < 0 || tri >= (int)env->surface.triangle_count) return 0.0f;
    uint32_t source_vidx = env->mesh.frontier[env->source_frontier_idx];
    Qm3Vec3 dir = qm3_sub(qm3_candidate_target_pos3(env, candidate), env->mesh.vertices[source_vidx].pos);
    return qm3_alignment_cost_from_dir_and_tri(env, dir, tri);
}

static float qm3_candidate_endpoint_angle_cost(const QuadMeshing3DEnv* env, uint32_t vidx, Qm3Vec3 other_pos) {
    return qm3_new_edge_endpoint_angle_cost(&env->mesh, vidx, other_pos);
}

static float qm3_candidate_angle_cost(const QuadMeshing3DEnv* env, const Qm3TargetCandidate* candidate) {
    if (env->source_frontier_idx < 0 || (uint32_t)env->source_frontier_idx >= env->mesh.frontier_count) return 0.0f;
    uint32_t source_vidx = env->mesh.frontier[env->source_frontier_idx];
    Qm3Vec3 source_pos = env->mesh.vertices[source_vidx].pos;
    Qm3Vec3 target_pos = qm3_candidate_target_pos3(env, candidate);
    float total = 0.0f;
    float count = 0.0f;
    float source_cost = qm3_candidate_endpoint_angle_cost(env, source_vidx, target_pos);
    if (source_cost >= 0.0f) {
        total += source_cost;
        count += 1.0f;
    }
    if (candidate->kind == QM3_TARGET_EXISTING_FRONTIER && candidate->graph_vertex < env->mesh.vertex_count) {
        float target_cost = qm3_candidate_endpoint_angle_cost(env, candidate->graph_vertex, source_pos);
        if (target_cost >= 0.0f) {
            total += target_cost;
            count += 1.0f;
        }
    }
    return count > 0.0f ? total / count : 0.0f;
}

static float qm3_candidate_overlay_cost(const QuadMeshing3DEnv* env, const Qm3TargetCandidate* candidate) {
    float chord = qm3_candidate_chord_length(env, candidate);
    switch (env->render_cost_overlay_mode) {
        case QM3_COST_OVERLAY_EDGE_LENGTH:
            return qm3_open_edge_length_cost_from_chord(chord, env->target_edge_length);
        case QM3_COST_OVERLAY_PATH_CHORD: {
            float path = candidate->path_length > 0.0f ? candidate->path_length : chord;
            return qm3_path_chord_cost(path, chord, env->target_edge_length);
        }
        case QM3_COST_OVERLAY_ALIGNMENT:
            return qm3_candidate_alignment_cost(env, candidate);
        case QM3_COST_OVERLAY_ANGLE:
            return qm3_candidate_angle_cost(env, candidate);
        case QM3_COST_OVERLAY_OFF:
        default:
            return 0.0f;
    }
}

typedef struct {
    SerialBuffer sb;
} Qm3SerialObsBuffer;

static void qm3_serialize_vec3(SerialBuffer* sb, Qm3Vec3 v) {
    serialize_float(sb, v.x);
    serialize_float(sb, v.y);
    serialize_float(sb, v.z);
}

static Qm3Vec3 qm3_candidate_target_normal(const QuadMeshing3DEnv* env, const Qm3TargetCandidate* candidate) {
    if (candidate->kind == QM3_TARGET_EXISTING_FRONTIER && candidate->graph_vertex < env->mesh.vertex_count) {
        return env->mesh.vertices[candidate->graph_vertex].normal;
    }
    if (candidate->sample_id < env->surface.sample_count) return env->surface.samples[candidate->sample_id].n;
    return (Qm3Vec3){0.0f, 0.0f, 1.0f};
}

static uint16_t qm3_u16_or_max(uint32_t value) {
    return value < UINT16_MAX ? (uint16_t)value : UINT16_MAX;
}

static uint32_t qm3_valid_target_count_capped(const QuadMeshing3DEnv* env) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < env->candidate_count && count < MAX_TARGETS; ++i) {
        if (env->candidates[i].path_ok) count++;
    }
    return count;
}

static int qm3_rand_range(unsigned int* rng, int max_exclusive) {
    return rand_r(rng) % max_exclusive;
}

static void qm3_compute_observations(QuadMeshing3DEnv* env) {
    if (!env->observations) return;
    memset(env->observations, 0, OBS_SIZE);
    QM3_ASSERT(env->mesh.frontier_count <= MAX_FRONTIER_SIZE);
    QM3_ASSERT(env->mesh.max_degree <= MAX_DEGREE);

    Qm3SerialObsBuffer obs = {.sb = {.data = env->observations, .pos = 0}};
    SerialBuffer* sb = &obs.sb;

    serialize_u8(sb, (uint8_t)env->phase);
    serialize_float(sb, env->target_edge_length);
    serialize_u16(sb, qm3_u16_or_max(env->mesh.frontier_count));
    serialize_u16(sb, qm3_u16_or_max(env->mesh.max_degree));

    for (uint32_t i = 0; i < MAX_FRONTIER_SIZE; ++i) {
        if (i < env->mesh.frontier_count) {
            uint32_t vidx = env->mesh.frontier[i];
            const Qm3MeshVertex* v = &env->mesh.vertices[vidx];
            qm3_serialize_vec3(sb, v->pos);
            qm3_serialize_vec3(sb, v->normal);
        } else {
            qm3_serialize_vec3(sb, (Qm3Vec3){0.0f, 0.0f, 0.0f});
            qm3_serialize_vec3(sb, (Qm3Vec3){0.0f, 0.0f, 0.0f});
        }
    }

    for (uint32_t i = 0; i < MAX_FRONTIER_SIZE; ++i) {
        uint32_t written = 0;
        if (i < env->mesh.frontier_count) {
            uint32_t vidx = env->mesh.frontier[i];
            const Qm3MeshVertex* v = &env->mesh.vertices[vidx];
            for (uint32_t j = 0; j < v->degree && written < MAX_DEGREE; ++j) {
                size_t nidx = (size_t)vidx * env->mesh.max_degree + j;
                int32_t nvidx = env->mesh.neighbors[nidx];
                int32_t eidx = env->mesh.neighbor_edges[nidx];
                if (nvidx < 0 || eidx < 0) continue;
                if (env->mesh.edges[eidx].disabled || env->mesh.edges[eidx].face_count >= 2) continue;
                int32_t nfidx = env->mesh.vertices[nvidx].frontier_index;
                if (nfidx < 0) continue;
                serialize_u16(sb, qm3_u16_or_max((uint32_t)nfidx));
                written++;
            }
        }
        for (; written < MAX_DEGREE; ++written) serialize_u16(sb, UINT16_MAX);
    }

    uint16_t suggested = env->mesh.frontier_count > 0
        ? (uint16_t)qm3_rand_range(&env->rng, (int)env->mesh.frontier_count)
        : UINT16_MAX;
    serialize_u16(sb, suggested);
    serialize_u16(sb, env->source_frontier_idx >= 0 ? (uint16_t)env->source_frontier_idx : UINT16_MAX);

    if (env->phase != QM3_PHASE_TARGET) {
        serialize_u16(sb, 0);
        return;
    }

    uint32_t target_count = qm3_valid_target_count_capped(env);
    serialize_u16(sb, qm3_u16_or_max(target_count));
    uint32_t written = 0;
    for (uint32_t i = 0; i < env->candidate_count && written < target_count; ++i) {
        const Qm3TargetCandidate* candidate = &env->candidates[i];
        if (!candidate->path_ok) continue;
        qm3_serialize_vec3(sb, qm3_candidate_target_pos3(env, candidate));
        qm3_serialize_vec3(sb, qm3_candidate_target_normal(env, candidate));
        serialize_float(sb, candidate->path_length);
        serialize_u8(sb, (uint8_t)candidate->kind);
        written++;
    }
}

static float qm3_compute_frontier_potential(QuadMeshing3DEnv* env) {
    if (env->mesh.frontier_count == 0) return 0.0f;
    float edge_length_cost = 0.0f;
    float alignment_cost = 0.0f;
    float angle_cost = 0.0f;
    uint32_t open_edges = 0;
    for (uint32_t ei = 0; ei < env->mesh.edge_count; ++ei) {
        Qm3MeshEdge* edge = &env->mesh.edges[ei];
        if (edge->disabled || edge->face_count >= 2) continue;
        float chord = qm3_edge_chord_length(&env->mesh, edge);
        float path = edge->path_length > 0.0f ? edge->path_length : chord;
        edge_length_cost += qm3_open_edge_length_cost_from_chord(chord, env->target_edge_length);
        edge_length_cost += qm3_path_chord_cost(path, chord, env->target_edge_length);
        alignment_cost += qm3_open_edge_alignment_cost(env, edge);
        open_edges++;
    }
    for (uint32_t fi = 0; fi < env->mesh.frontier_count; ++fi) {
        angle_cost += qm3_vertex_open_angle_cost(env, env->mesh.frontier[fi]);
    }

    float edge_term = open_edges > 0 ? edge_length_cost / (2.0f * (float)open_edges) : 0.0f;
    float alignment_term = open_edges > 0 ? alignment_cost / (float)open_edges : 0.0f;
    float angle_term = angle_cost / (float)env->mesh.frontier_count;
    float weight_sum = env->frontier_edge_length_weight + env->frontier_alignment_weight + env->frontier_angle_weight;
    float frontier_cost = weight_sum > 0.0f
        ? (env->frontier_edge_length_weight * edge_term + env->frontier_alignment_weight * alignment_term + env->frontier_angle_weight * angle_term) / weight_sum
        : 0.0f;
    float max_frontier = fmaxf((float)(env->surface.sample_count > 0 ? env->surface.sample_count : env->mesh.frontier_count), 1.0f);
    float highest_degree = 0.0f;
    for (uint32_t i = 0; i < env->mesh.vertex_count; ++i) highest_degree = fmaxf(highest_degree, (float)env->mesh.vertices[i].degree);

    return -env->frontier_quality_weight * frontier_cost
        - env->frontier_size_pressure_weight * qm3_pressure_ratio((float)env->mesh.frontier_count, env->safe_frontier_size_ratio, max_frontier)
        - env->degree_pressure_weight * qm3_pressure_ratio(highest_degree, env->safe_degree_ratio, (float)env->mesh.max_degree);
}

static void qm3_free_geodesic_state(QuadMeshing3DEnv* env) {
    qm3_graph_tri_segments_free(&env->graph_tri_segments, env->surface.triangle_count);
    qm3_graph_tri_vertices_free(&env->graph_tri_vertices, env->surface.triangle_count);
    env->graph_tri_segment_count = 0;
    env->graph_tri_vertex_count = 0;
    if (env->geodesic_ready) {
        qm3_fmm_free(&env->fmm);
        qm3_continuous_context_free(&env->continuous_ctx);
    }
    qm3_prop_graph_free(&env->prop_graph);
    qm3_surface_topo_free(&env->surface_topo);
    env->geodesic_ready = false;
}

static void qm3_build_geodesic_state(QuadMeshing3DEnv* env) {
    qm3_free_geodesic_state(env);
    float diag = qm3_surface_diag(&env->surface);
    if (diag <= 1e-8f) diag = 1.0f;
    float radius_ratio = env->candidate_radius_ratio > 0.0f ? env->candidate_radius_ratio : 0.05f;
    float spacing_ratio = env->geodesic_steiner_spacing_ratio > 0.0f ? env->geodesic_steiner_spacing_ratio : radius_ratio / 10.0f;
    float spacing = fmaxf(diag * spacing_ratio, 1e-4f);
    env->surface_topo = qm3_surface_topo_build(&env->surface);
    env->prop_graph = qm3_prop_graph_build(&env->surface, spacing);
    env->fmm = qm3_fmm_create(env->prop_graph.node_count, env->surface.triangle_count);
    env->continuous_ctx = qm3_continuous_context_create(&env->surface, &env->prop_graph);
    qm3_build_graph_tri_segments(env);
    env->geodesic_ready = true;
}

static void quad_meshing_3d_load_shape(QuadMeshing3DEnv* env, int shape_idx) {
    QM3_ASSERT(shape_idx >= 0 && shape_idx < env->shape_count);
    qm3_surface_load(&env->surface, env->shape_paths[shape_idx]);
    qm3_mesh_build_from_frontier_edges(&env->mesh, &env->surface);
    qm3_reset_sample_disabled(env);
    memset(&env->loop_stats, 0, sizeof(env->loop_stats));
    qm3_loop_debug_free(&env->loop_debug);
    qm3_rebuild_sample_to_graph(env);
    qm3_build_geodesic_state(env);
    qm3_update_target_scale(env);
    qm3_update_episode_max_length(env);
    env->potential = qm3_compute_frontier_potential(env);
    env->last_frontier_potential = env->potential;
    env->candidate_radius = 0.0f;
    qm3_clear_target_selection(env);
    env->phase = QM3_PHASE_SOURCE;
    env->loaded_shape = shape_idx;
}

void c_reset(QuadMeshing3DEnv* env) {
    env->episode_length = 0;
    env->episode_return = 0.0f;
    if (env->shape_count > 0 && env->loaded_shape < 0) quad_meshing_3d_load_shape(env, 0);
    else {
        qm3_mesh_build_from_frontier_edges(&env->mesh, &env->surface);
        qm3_reset_sample_disabled(env);
        memset(&env->loop_stats, 0, sizeof(env->loop_stats));
        qm3_loop_debug_free(&env->loop_debug);
        qm3_rebuild_sample_to_graph(env);
        if (env->geodesic_ready) qm3_build_graph_tri_segments(env);
        qm3_update_target_scale(env);
        qm3_update_episode_max_length(env);
    }
    env->potential = qm3_compute_frontier_potential(env);
    env->last_frontier_potential = env->potential;
    env->last_reward = 0.0f;
    env->last_face_area = 0.0f;
    env->last_area_quality = 0.0f;
    env->last_element_quality = 0.0f;
    env->last_planarity_quality = 0.0f;
    env->last_curvature_quality = 0.0f;
    env->candidate_radius = 0.0f;
    qm3_clear_target_selection(env);
    env->phase = QM3_PHASE_SOURCE;
    env->last_action_result = QM3_ACTION_NONE;
    if (env->rewards) env->rewards[0] = 0.0f;
    if (env->terminals) env->terminals[0] = 0.0f;
    qm3_compute_observations(env);
}

void c_step(QuadMeshing3DEnv* env) {
    if (env->rewards) env->rewards[0] = 0.0f;
    if (env->terminals) env->terminals[0] = 0.0f;
    env->last_reward = 0.0f;

    if (env->phase == QM3_PHASE_SOURCE) {
        uint32_t source_slot = qm3_clamp_action_index(env->actions ? env->actions[0] : 0.0f);
        if (source_slot < env->mesh.frontier_count) qm3_select_source_frontier(env, (int32_t)source_slot);
        else {
            env->last_action_result = QM3_ACTION_INVALID_SOURCE;
            qm3_add_reward(env, env->reward_invalid);
        }
    } else {
        env->episode_length++;
        uint32_t target_slot = qm3_clamp_action_index(env->actions ? env->actions[0] : 0.0f);
        int32_t candidate_idx = qm3_nth_valid_candidate(env, target_slot);
        if (candidate_idx >= 0) qm3_commit_target_candidate(env, (uint32_t)candidate_idx);
        else {
            env->last_action_result = QM3_ACTION_INVALID_TARGET;
            qm3_add_reward(env, env->reward_invalid);
            qm3_clear_target_selection(env);
            env->phase = QM3_PHASE_SOURCE;
        }
    }

    bool terminal = env->episode_length >= env->episode_max_length || env->mesh.frontier_count == 0;
    if (terminal) {
        float reward = env->rewards ? env->rewards[0] : env->last_reward;
        if (env->mesh.frontier_count != 0) reward = env->reward_incomplete;
        env->episode_return += reward;
        qm3_add_log(env);
        c_reset(env);
        if (env->rewards) env->rewards[0] = reward;
        if (env->terminals) env->terminals[0] = 1.0f;
        env->last_reward = reward;
    } else {
        env->episode_return += env->rewards ? env->rewards[0] : env->last_reward;
        if (env->rewards) env->last_reward = env->rewards[0];
        qm3_compute_observations(env);
    }
}

void c_render(QuadMeshing3DEnv* env) {
    Qm3Surface* surface = &env->surface;
    if (surface->vertex_count == 0) return;

    float diag = qm3_surface_diag(surface);
    if (diag <= 1e-8f) diag = 1.0f;

    if (!IsWindowReady()) {
        SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
        InitWindow(env->render_width > 0 ? env->render_width : 1280, env->render_height > 0 ? env->render_height : 900, "PufferLib Quad Meshing 3D");
        SetTargetFPS(env->render_target_fps > 0 ? env->render_target_fps : 144);

        Vector3 center = Vector3Scale(Vector3Add(qm3_v3(surface->bounds_min), qm3_v3(surface->bounds_max)), 0.5f);
        float dist = fmaxf(diag * 1.8f, 1.0f);
        env->camera.position = Vector3Add(center, (Vector3){dist, dist * 0.6f, dist});
        env->camera.target = center;
        env->camera.up = (Vector3){0.0f, 1.0f, 0.0f};
        env->camera.fovy = 45.0f;
        env->camera.projection = CAMERA_PERSPECTIVE;
    }

    qm3_update_orbit_camera(&env->camera, fmaxf(diag * 0.02f, 0.01f), fmaxf(diag * 20.0f, 10.0f));

    if (IsKeyPressed(KEY_M)) env->render_show_mesh = !env->render_show_mesh;
    if (IsKeyPressed(KEY_O)) env->render_mesh_opaque = !env->render_mesh_opaque;
    if (IsKeyPressed(KEY_P)) env->render_show_samples = !env->render_show_samples;
    if (IsKeyPressed(KEY_V)) env->render_show_vertices = !env->render_show_vertices;
    if (IsKeyPressed(KEY_N)) env->render_show_normals = !env->render_show_normals;
    if (IsKeyPressed(KEY_X)) env->render_show_cross_field = !env->render_show_cross_field;
    if (IsKeyPressed(KEY_F)) env->render_show_frontier = !env->render_show_frontier;
    if (IsKeyPressed(KEY_G)) env->render_show_graph = !env->render_show_graph;
    if (IsKeyPressed(KEY_Y)) env->render_show_graph_paths = !env->render_show_graph_paths;
    if (IsKeyPressed(KEY_C)) env->render_show_candidates = !env->render_show_candidates;
    if (IsKeyPressed(KEY_I)) env->render_debug_validity = !env->render_debug_validity;
    if (IsKeyPressed(KEY_K)) env->render_cost_overlay_mode = (Qm3CostOverlayMode)((env->render_cost_overlay_mode + 1) % 5);
    if (IsKeyPressed(KEY_L)) env->loop_debug_mode = env->loop_debug_mode == 0 ? 6 : 0;
    if (IsKeyPressed(KEY_R)) c_reset(env);
    if (IsKeyPressed(KEY_SPACE) && env->phase == QM3_PHASE_TARGET && env->target_candidate_idx >= 0) {
        env->episode_length++;
        env->last_reward = 0.0f;
        qm3_commit_target_candidate(env, (uint32_t)env->target_candidate_idx);
        env->episode_return += env->last_reward;
        if (env->episode_length >= env->episode_max_length || env->mesh.frontier_count == 0) {
            qm3_add_log(env);
            c_reset(env);
        }
    }
    if (IsKeyPressed(KEY_UP)) {
        env->candidate_radius *= 1.1f;
        qm3_compute_source_candidates(env);
        qm3_update_selected_path(env);
    }
    if (IsKeyPressed(KEY_DOWN)) {
        env->candidate_radius /= 1.1f;
        qm3_compute_source_candidates(env);
        qm3_update_selected_path(env);
    }

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        bool ctrl = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL);
        bool shift = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
        Vector2 mouse = GetMousePosition();
        if (ctrl && !shift) {
            int32_t picked = qm3_pick_frontier_screen(env, mouse, 14.0f);
            if (picked >= 0) qm3_select_source_frontier(env, picked);
        } else if (shift && !ctrl) {
            bool require_valid = !env->render_debug_validity;
            int32_t picked = qm3_pick_candidate_screen(env, mouse, require_valid, 14.0f);
            if (picked >= 0) {
                qm3_select_debug_candidate(env, picked);
                if (env->candidates[picked].path_ok) qm3_select_target_candidate(env, picked);
                else qm3_update_selected_path(env);
            }
        }
    }

    float point_size = env->render_point_radius > 0.0f ? env->render_point_radius : diag * 0.0015f;
    float normal_length = env->render_normal_length > 0.0f ? env->render_normal_length : diag * 0.035f;
    float cross_length = env->render_cross_field_length > 0.0f ? env->render_cross_field_length : diag * 0.0025f;

    BeginDrawing();
    ClearBackground((Color){9, 13, 18, 255});

    BeginMode3D(env->camera);
    if (env->render_show_mesh) qm3_draw_mesh(surface, env->render_mesh_opaque);
    qm3_draw_loop_debug(env, diag);
    if (env->render_show_vertices) qm3_draw_vertices(surface, env->camera, point_size, (Color){110, 130, 160, 170});
    if (env->render_show_samples) qm3_draw_active_sample_crosses(env, env->camera, point_size, (Color){130, 145, 155, 135});
    if (env->render_show_normals) qm3_draw_normals(surface, normal_length);
    if (env->render_show_cross_field) qm3_draw_cross_field(surface, cross_length);
    if (env->render_show_graph) {
        float offset = fmaxf(diag * 0.0002f, 2e-6f);
        if (env->render_show_graph_paths) qm3_draw_graph_path_segments(&env->mesh, 0.0f, (Color){80, 235, 255, 255});
        else qm3_draw_graph_edges(&env->mesh, offset);
        if (!env->render_show_graph_paths) qm3_draw_mesh_vertex_crosses(&env->mesh, env->camera, point_size * 1.1f, offset * 1.5f);
    }
    if (env->render_debug_validity) {
        qm3_draw_candidate_validity_crosses(env, env->candidates, env->candidate_count, env->camera, point_size * 1.25f);
    } else if (env->render_show_candidates) {
        qm3_draw_candidate_subset_crosses(env, env->candidates, env->candidate_count, env->camera, point_size * 1.2f, (Color){50, 235, 125, 230}, (Color){80, 190, 255, 245});
    }
    if (env->selected_path_ok) {
        qm3_draw_path(&env->selected_path, 0.0f, (Color){255, 80, 220, 255});
    }
    if (env->render_debug_validity && env->debug_candidate_idx >= 0 && (uint32_t)env->debug_candidate_idx < env->candidate_count) {
        uint32_t dbg = (uint32_t)env->debug_candidate_idx;
        const Qm3TargetCandidate* candidate = &env->candidates[dbg];
        Qm3CandidateValidity reason = candidate->validity;
        Color reason_color = qm3_candidate_validity_color(reason);
        uint32_t seg_offset = candidate->segment_offset;
        uint32_t seg_count = candidate->segment_count;
        float debug_offset = 0.0f;
        qm3_draw_path_segments(&env->candidate_path_segments, seg_offset, seg_count, debug_offset, reason_color);
        if (reason == QM3_CANDIDATE_INTERSECT) {
            int32_t local_seg = candidate->intersect_segment;
            if (local_seg >= 0) qm3_draw_path_segments(&env->candidate_path_segments, seg_offset + (uint32_t)local_seg, 1, 0.0f, (Color){255, 245, 70, 255});
            qm3_draw_graph_edge_highlight(&env->mesh, candidate->intersect_edge, 0.0f, (Color){255, 40, 40, 255});
        }
    }
    if (env->source_frontier_idx >= 0 && (uint32_t)env->source_frontier_idx < env->mesh.frontier_count) {
        uint32_t source_vidx = env->mesh.frontier[env->source_frontier_idx];
        Qm3MeshVertex source = env->mesh.vertices[source_vidx];
        Vector3 source_pos = qm3_v3(source.pos);
        DrawSphere(source_pos, fmaxf(point_size * 1.2f, diag * 0.0018f), (Color){255, 70, 55, 255});
        DrawSphereWires(source_pos, env->candidate_radius, 32, 12, (Color){255, 210, 80, 60});
    }
    if (env->target_candidate_idx >= 0 && (uint32_t)env->target_candidate_idx < env->candidate_count) {
        const Qm3TargetCandidate* candidate = &env->candidates[env->target_candidate_idx];
        DrawSphere(qm3_candidate_position(env, candidate), fmaxf(point_size, diag * 0.0015f), (Color){255, 80, 220, 255});
    }
    if (env->render_debug_validity && env->debug_candidate_idx >= 0 && (uint32_t)env->debug_candidate_idx < env->candidate_count) {
        const Qm3TargetCandidate* candidate = &env->candidates[env->debug_candidate_idx];
        Color color = qm3_candidate_validity_color(candidate->validity);
        Vector3 p = qm3_candidate_position(env, candidate);
        DrawSphere(p, fmaxf(point_size * 1.3f, diag * 0.002f), color);
        DrawSphereWires(p, fmaxf(point_size * 2.0f, diag * 0.003f), 12, 8, color);
    }
    DrawBoundingBox((BoundingBox){qm3_v3(surface->bounds_min), qm3_v3(surface->bounds_max)}, (Color){90, 140, 220, 80});
    EndMode3D();

    DrawRectangle(12, 12, 1160, 250, (Color){0, 0, 0, 170});
    DrawText(TextFormat("QMSURF3D | vertices: %u | triangles: %u | samples: %u | frontier_edges: %u | prop nodes: %d",
        surface->vertex_count, surface->triangle_count, surface->sample_count, surface->frontier_edge_count, env->prop_graph.node_count), 24, 24, 18, RAYWHITE);
    DrawText(TextFormat("area: %.6f | sample density: %.2f | sharp dihedral: %.2f deg",
        surface->info.total_area, surface->info.sample_density, surface->info.sharp_dihedral_radians * RAD2DEG), 24, 50, 18, RAYWHITE);
    DrawText(TextFormat("phase: %s | step: %d / %d | return: %.3f | frontier: %u | edges: %u | faces: %u (quads %u tris %u) | graph vertices: %u | last: %s",
        qm3_phase_str(env->phase),
        env->episode_length,
        env->episode_max_length,
        env->episode_return,
        env->mesh.frontier_count,
        env->mesh.edge_count,
        env->mesh.face_count,
        env->mesh.quad_count,
        env->mesh.tri_count,
        env->mesh.vertex_count,
        qm3_action_result_str(env->last_action_result)), 24, 76, 18, (Color){120, 220, 255, 255});
    DrawText(TextFormat("source: %d / %u | radius: %.5f | candidates: %u (samples %u existing %u) valid: %u | continuous: %u | target: %d | path: %s %.5f (%u pts)",
        env->source_frontier_idx, env->mesh.frontier_count, env->candidate_radius, env->candidate_count,
        env->candidate_sample_count,
        env->candidate_existing_count,
        env->valid_candidate_count,
        env->continuous_paths_ok,
        env->target_candidate_idx,
        env->selected_path_ok ? "ok" : "n/a",
        env->selected_path.length,
        env->selected_path.count), 24, 102, 18, (Color){255, 210, 100, 255});
    DrawText(TextFormat("invalid: unreachable %u | intersect %u | disabled samples %u | loop L:%s last %u side %s conf %u/%u",
        env->invalid_unreachable_count,
        env->invalid_intersect_count,
        env->disabled_sample_count,
        qm3_loop_debug_mode_str(env->loop_debug_mode),
        env->loop_stats.disabled_samples,
        env->loop_stats.chosen_side == 1 ? "left" : (env->loop_stats.chosen_side == 2 ? "right" : "none"),
        env->loop_stats.classification_conflicts,
        env->loop_stats.flood_conflicts), 24, 128, 18, (Color){180, 210, 255, 255});
    DrawText(TextFormat("reward: last %.4f | face area %.6f | area q %.3f | element q %.3f | planarity q %.3f | curvature q %.3f | frontier potential %.4f | target edge %.5f area %.6f",
        env->last_reward,
        env->last_face_area,
        env->last_area_quality,
        env->last_element_quality,
        env->last_planarity_quality,
        env->last_curvature_quality,
        env->last_frontier_potential,
        env->target_edge_length,
        env->target_quad_area), 24, 154, 18, (Color){210, 235, 180, 255});
    DrawText(TextFormat("draw: debug I:%s | cost K:%s | mesh M:%s | opaque O:%s | samples P:%s | vertices V:%s | normals N:%s | cross X:%s | graph G:%s | graph mode Y:%s | candidates C:%s",
        env->render_debug_validity ? "on" : "off",
        qm3_cost_overlay_mode_str(env->render_cost_overlay_mode),
        env->render_show_mesh ? "on" : "off",
        env->render_mesh_opaque ? "on" : "off",
        env->render_show_samples ? "on" : "off",
        env->render_show_vertices ? "on" : "off",
        env->render_show_normals ? "on" : "off",
        env->render_show_cross_field ? "on" : "off",
        env->render_show_graph ? "on" : "off",
        env->render_show_graph_paths ? "paths" : "edges",
        env->render_show_candidates ? "on" : "off"), 24, 180, 18, (Color){180, 210, 255, 255});
    if (env->render_debug_validity && env->debug_candidate_idx >= 0 && (uint32_t)env->debug_candidate_idx < env->candidate_count) {
        uint32_t dbg = (uint32_t)env->debug_candidate_idx;
        const Qm3TargetCandidate* candidate = &env->candidates[dbg];
        uint32_t sample_id = candidate->sample_id;
        Qm3CandidateValidity reason = candidate->validity;
        int32_t edge = candidate->intersect_edge;
        int32_t seg = candidate->intersect_segment;
        DrawText(TextFormat("debug candidate: %u / %u | kind: %s | graph_vertex: %u | sample: %u | tri: %u | reason: %s | path_segments: %u | hit_edge: %d | hit_segment: %d",
            dbg,
            env->candidate_count,
            qm3_candidate_kind_str(candidate->kind),
            candidate->graph_vertex,
            sample_id,
            sample_id < surface->sample_count ? surface->samples[sample_id].tri : 0,
            qm3_candidate_validity_str(reason),
            candidate->segment_count,
            edge,
            seg), 24, 206, 18, qm3_candidate_validity_color(reason));
    } else {
        DrawText("debug candidate: off (I toggles validity overlay)", 24, 206, 18, (Color){150, 160, 175, 255});
    }
    DrawText("controls: Ctrl+LMB source | Shift+LMB target | Space commit | R reset | K cost overlay | G graph | Y graph mode | L loop debug | Up/Down radius", 24, 232, 18, (Color){180, 190, 200, 255});

    int footer_y = GetScreenHeight() - 42;
    DrawRectangle(12, footer_y - 8, 920, 34, (Color){0, 0, 0, 170});
    DrawText(TextFormat("timing: targets %.3f ms | geodesic paths %.3f ms | intersections %.3f ms (%u tests) | loop %.3f ms | total %.3f ms",
        env->timing_target_query_ms,
        env->timing_path_build_ms,
        env->timing_intersection_ms,
        env->timing_intersection_tests,
        env->timing_loop_removal_ms,
        env->timing_target_query_ms + env->timing_path_build_ms + env->timing_intersection_ms + env->timing_loop_removal_ms),
        24, footer_y, 18, (Color){210, 235, 255, 255});

    EndDrawing();
}

void c_close(QuadMeshing3DEnv* env) {
    if (IsWindowReady()) CloseWindow();
    free(env->candidates);
    env->candidates = NULL;
    env->candidate_count = 0;
    env->candidate_cap = 0;
    qm3_path_free(&env->candidate_path_points);
    qm3_path_segment_free(&env->candidate_path_segments);
    qm3_free_geodesic_state(env);
    qm3_path_free(&env->selected_path);
    free(env->sample_to_graph);
    env->sample_to_graph = NULL;
    free(env->sample_disabled);
    env->sample_disabled = NULL;
    qm3_loop_debug_free(&env->loop_debug);
    qm3_mesh_free(&env->mesh);
    qm3_surface_free(&env->surface);
}
