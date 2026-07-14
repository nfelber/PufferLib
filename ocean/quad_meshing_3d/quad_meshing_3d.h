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

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float episode_length_ratio;
    float num_quads;
    float num_quads_ratio;
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
    uint32_t boundary_tris;
    uint32_t flood_tris;
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
    int max_frontier;
    int max_candidates;
    int max_degree;
    float candidate_radius_min_ratio;
    float candidate_radius_max_ratio;
    float candidate_radius;
    float geodesic_steiner_spacing_ratio;
    float target_edge_length_ratio;
    float target_edge_length;
    float target_quad_area;
    float starting_frontier_edge_length;
    float episode_max_length_ratio;
    bool prevent_triangles;
    bool export_obj;
    const char* export_obj_path;

    float reward_invalid;
    float reward_incomplete;
    float reward_triangle;
    bool reward_cross_field;
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

static float qm3_candidate_radius_min(const QuadMeshing3DEnv* env) {
    return fmaxf(env->target_edge_length * env->candidate_radius_min_ratio, 0.0f);
}

static float qm3_candidate_radius_max(const QuadMeshing3DEnv* env) {
    return fmaxf(env->target_edge_length * env->candidate_radius_max_ratio, 1e-6f);
}

static void qm3_maybe_export_obj(QuadMeshing3DEnv* env) {
    if (env->export_obj) qm3_mesh_dump_obj(&env->mesh, env->export_obj_path);
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

static void qm3_trim_segment_at_point(Qm3Vec2* a, Qm3Vec2* b, Qm3Vec2 p, float tol) {
    Qm3Vec2 ab = qm3_v2_sub(*b, *a);
    float length_sq = qm3_v2_dot(ab, ab);
    if (length_sq <= tol * tol) return;
    float trim = fminf(4.0f * tol / sqrtf(length_sq), 1.0f);
    Qm3Vec2 da = qm3_v2_sub(*a, p);
    Qm3Vec2 db = qm3_v2_sub(*b, p);
    float endpoint_tol_sq = 4.0f * tol * tol;
    if (qm3_v2_dot(da, da) <= endpoint_tol_sq) *a = qm3_v2_lerp(*a, *b, trim);
    else if (qm3_v2_dot(db, db) <= endpoint_tol_sq) *b = qm3_v2_lerp(*b, *a, trim);
}

static bool qm3_segments_intersect_2d_except_endpoints(
    Qm3Vec2 a,
    Qm3Vec2 b,
    Qm3Vec2 c,
    Qm3Vec2 d,
    const Qm3Vec2* allowed,
    uint32_t allowed_count,
    float tol
) {
    if (!qm3_segments_intersect_2d(a, b, c, d, tol)) return false;
    for (uint32_t i = 0; i < allowed_count; ++i) {
        qm3_trim_segment_at_point(&a, &b, allowed[i], tol);
        qm3_trim_segment_at_point(&c, &d, allowed[i], tol);
    }
    Qm3Vec2 ab = qm3_v2_sub(b, a);
    Qm3Vec2 cd = qm3_v2_sub(d, c);
    if (qm3_v2_dot(ab, ab) <= tol * tol || qm3_v2_dot(cd, cd) <= tol * tol) return false;
    return qm3_segments_intersect_2d(a, b, c, d, tol);
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
                if (gv->graph_vertex != UINT32_MAX) {
                    if (gv->graph_vertex == source_vidx) continue;
                    if (target_vidx != UINT32_MAX && gv->graph_vertex == target_vidx) continue;
                    if (gv->graph_vertex >= env->mesh.vertex_count || env->mesh.vertices[gv->graph_vertex].disabled) continue;
                } else if (gv->incident_edge >= env->mesh.edge_count || env->mesh.edges[gv->incident_edge].disabled) {
                    continue;
                }
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
            if (e.disabled) continue;
            Qm3Vec2 allowed[2];
            uint32_t allowed_count = 0;
            if (e.a == source_vidx || e.b == source_vidx) {
                qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, seg->tri, env->mesh.vertices[source_vidx].pos, &allowed[allowed_count++]);
            }
            if (target_vidx != UINT32_MAX && target_vidx < env->mesh.vertex_count &&
                    (e.a == target_vidx || e.b == target_vidx)) {
                qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, seg->tri, env->mesh.vertices[target_vidx].pos, &allowed[allowed_count++]);
            }
            tests++;
            if (qm3_segments_intersect_2d_except_endpoints(a, b, graph_seg->a, graph_seg->b, allowed, allowed_count, tol)) {
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

typedef struct { double x, y; } Qm3LoopPoint;
typedef struct {
    Qm3LoopPoint a, b;
    double* split;
    int split_count, split_cap;
    int boundary_edge;
    bool cut;
} Qm3LoopPrimitive;
typedef struct { Qm3LoopPoint p; } Qm3LoopNode;
typedef struct { int a, b, boundary_edge; bool cut; } Qm3LoopEdge;
typedef struct { int origin, twin, next, edge, cell; double angle; } Qm3LoopHalfEdge;
typedef struct {
    Qm3LoopPoint* polygon;
    int polygon_count;
    int tri;
    double area;
} Qm3LoopCell;
typedef struct {
    int tri, edge, cell;
    uint32_t va, vb;
    double lo, hi;
    bool cut;
} Qm3LoopPortal;
typedef struct { int tri; Qm3LoopPoint a, b; double tol; } Qm3LoopCut;
typedef struct {
    Qm3LoopCell* cells;
    int cell_count, cell_cap;
    Qm3LoopPortal* portals;
    int portal_count, portal_cap;
    Qm3LoopCut* cuts;
    int cut_count, cut_cap;
    int* tri_ids;
    int tri_count;
    int* tri_cell_offsets;
    int* tri_portal_offsets;
    double* tri_tol;
} Qm3LoopSubdivision;

static Qm3LoopPoint qm3_loop_point(Qm3Vec2 p) { return (Qm3LoopPoint){(double)p.x, (double)p.y}; }
static Qm3LoopPoint qm3_loop_add(Qm3LoopPoint a, Qm3LoopPoint b) { return (Qm3LoopPoint){a.x + b.x, a.y + b.y}; }
static Qm3LoopPoint qm3_loop_sub(Qm3LoopPoint a, Qm3LoopPoint b) { return (Qm3LoopPoint){a.x - b.x, a.y - b.y}; }
static Qm3LoopPoint qm3_loop_scale(Qm3LoopPoint a, double s) { return (Qm3LoopPoint){a.x * s, a.y * s}; }
static double qm3_loop_dot(Qm3LoopPoint a, Qm3LoopPoint b) { return a.x * b.x + a.y * b.y; }
static double qm3_loop_cross(Qm3LoopPoint a, Qm3LoopPoint b) { return a.x * b.y - a.y * b.x; }
static double qm3_loop_len2(Qm3LoopPoint a) { return qm3_loop_dot(a, a); }

static void qm3_loop_subdivision_free(Qm3LoopSubdivision* sub) {
    for (int i = 0; i < sub->cell_count; ++i) free(sub->cells[i].polygon);
    free(sub->cells); free(sub->portals); free(sub->cuts); free(sub->tri_ids);
    free(sub->tri_cell_offsets); free(sub->tri_portal_offsets); free(sub->tri_tol);
    memset(sub, 0, sizeof(*sub));
}

static void qm3_loop_split_push(Qm3LoopPrimitive* p, double t, double tol) {
    t = fmin(fmax(t, 0.0), 1.0);
    for (int i = 0; i < p->split_count; ++i) if (fabs(p->split[i] - t) <= tol) return;
    if (p->split_count == p->split_cap) {
        p->split_cap = p->split_cap ? 2 * p->split_cap : 8;
        p->split = (double*)qm3_checked_realloc(p->split, (size_t)p->split_cap * sizeof(double));
    }
    p->split[p->split_count++] = t;
}

static void qm3_loop_split_sort(Qm3LoopPrimitive* p) {
    for (int i = 1; i < p->split_count; ++i) {
        double value = p->split[i];
        int j = i;
        while (j > 0 && p->split[j - 1] > value) { p->split[j] = p->split[j - 1]; --j; }
        p->split[j] = value;
    }
}

static double qm3_loop_point_segment_dist2(Qm3LoopPoint p, Qm3LoopPoint a, Qm3LoopPoint b) {
    Qm3LoopPoint ab = qm3_loop_sub(b, a);
    double den = qm3_loop_len2(ab);
    double t = den > 0.0 ? qm3_loop_dot(qm3_loop_sub(p, a), ab) / den : 0.0;
    t = fmin(fmax(t, 0.0), 1.0);
    return qm3_loop_len2(qm3_loop_sub(p, qm3_loop_add(a, qm3_loop_scale(ab, t))));
}

static bool qm3_loop_segment_on_triangle_edge(const QuadMeshing3DEnv* env, int tri, Qm3LoopPoint a, Qm3LoopPoint b, double tol) {
    for (int edge = 0; edge < 3; ++edge) {
        Qm3LoopPoint c = qm3_loop_point(env->continuous_ctx.tri2d_base[3 * tri + (edge + 1) % 3]);
        Qm3LoopPoint d = qm3_loop_point(env->continuous_ctx.tri2d_base[3 * tri + (edge + 2) % 3]);
        if (qm3_loop_point_segment_dist2(a, c, d) <= tol * tol &&
                qm3_loop_point_segment_dist2(b, c, d) <= tol * tol) return true;
    }
    return false;
}

static bool qm3_loop_snap_to_triangle(Qm3LoopPoint* p, const Qm3LoopPoint tri[3], double tol) {
    double best = tol * tol;
    Qm3LoopPoint snapped = *p;
    for (int i = 0; i < 3; ++i) {
        double d2 = qm3_loop_len2(qm3_loop_sub(*p, tri[i]));
        if (d2 <= best) { best = d2; snapped = tri[i]; }
    }
    for (int edge = 0; edge < 3; ++edge) {
        Qm3LoopPoint a = tri[(edge + 1) % 3], b = tri[(edge + 2) % 3], ab = qm3_loop_sub(b, a);
        double den = qm3_loop_len2(ab);
        if (den <= 0.0) continue;
        double t = fmin(fmax(qm3_loop_dot(qm3_loop_sub(*p, a), ab) / den, 0.0), 1.0);
        Qm3LoopPoint q = qm3_loop_add(a, qm3_loop_scale(ab, t));
        double d2 = qm3_loop_len2(qm3_loop_sub(*p, q));
        if (d2 <= best) { best = d2; snapped = q; }
    }
    *p = snapped;
    double orient = qm3_loop_cross(qm3_loop_sub(tri[1], tri[0]), qm3_loop_sub(tri[2], tri[0]));
    if (fabs(orient) <= tol * tol) return false;
    for (int edge = 0; edge < 3; ++edge) {
        Qm3LoopPoint a = tri[(edge + 1) % 3], b = tri[(edge + 2) % 3];
        double side = qm3_loop_cross(qm3_loop_sub(b, a), qm3_loop_sub(*p, a));
        if (side * orient < -tol * sqrt(qm3_loop_len2(qm3_loop_sub(b, a)))) return false;
    }
    return true;
}

static void qm3_loop_split_intersection(Qm3LoopPrimitive* a, Qm3LoopPrimitive* b, double tol) {
    Qm3LoopPoint r = qm3_loop_sub(a->b, a->a), s = qm3_loop_sub(b->b, b->a), ba = qm3_loop_sub(b->a, a->a);
    double rr = qm3_loop_len2(r), ss = qm3_loop_len2(s), den = qm3_loop_cross(r, s);
    if (rr <= tol * tol || ss <= tol * tol) return;
    if (fabs(den) > 1e-12 * sqrt(rr * ss)) {
        double ta = qm3_loop_cross(ba, s) / den, tb = qm3_loop_cross(ba, r) / den;
        double pa = tol / sqrt(rr), pb = tol / sqrt(ss);
        if (ta >= -pa && ta <= 1.0 + pa && tb >= -pb && tb <= 1.0 + pb) {
            qm3_loop_split_push(a, ta, pa); qm3_loop_split_push(b, tb, pb);
        }
        return;
    }
    if (fabs(qm3_loop_cross(ba, r)) > tol * sqrt(rr)) return;
    double ta0 = qm3_loop_dot(qm3_loop_sub(b->a, a->a), r) / rr;
    double ta1 = qm3_loop_dot(qm3_loop_sub(b->b, a->a), r) / rr;
    double tb0 = qm3_loop_dot(qm3_loop_sub(a->a, b->a), s) / ss;
    double tb1 = qm3_loop_dot(qm3_loop_sub(a->b, b->a), s) / ss;
    if (ta0 >= -tol / sqrt(rr) && ta0 <= 1.0 + tol / sqrt(rr)) qm3_loop_split_push(a, ta0, tol / sqrt(rr));
    if (ta1 >= -tol / sqrt(rr) && ta1 <= 1.0 + tol / sqrt(rr)) qm3_loop_split_push(a, ta1, tol / sqrt(rr));
    if (tb0 >= -tol / sqrt(ss) && tb0 <= 1.0 + tol / sqrt(ss)) qm3_loop_split_push(b, tb0, tol / sqrt(ss));
    if (tb1 >= -tol / sqrt(ss) && tb1 <= 1.0 + tol / sqrt(ss)) qm3_loop_split_push(b, tb1, tol / sqrt(ss));
}

static int qm3_loop_node_get(Qm3LoopNode** nodes, int* count, int* cap, Qm3LoopPoint p, double tol) {
    for (int i = 0; i < *count; ++i) if (qm3_loop_len2(qm3_loop_sub((*nodes)[i].p, p)) <= tol * tol) return i;
    if (*count == *cap) { *cap = *cap ? 2 * *cap : 16; *nodes = (Qm3LoopNode*)qm3_checked_realloc(*nodes, (size_t)*cap * sizeof(**nodes)); }
    (*nodes)[*count].p = p;
    return (*count)++;
}

static void qm3_loop_edge_add(Qm3LoopEdge** edges, int* count, int* cap, int a, int b, int boundary_edge, bool cut) {
    if (a == b) return;
    for (int i = 0; i < *count; ++i) {
        if (((*edges)[i].a == a && (*edges)[i].b == b) || ((*edges)[i].a == b && (*edges)[i].b == a)) {
            (*edges)[i].cut |= cut;
            if (boundary_edge >= 0) (*edges)[i].boundary_edge = boundary_edge;
            return;
        }
    }
    if (*count == *cap) { *cap = *cap ? 2 * *cap : 24; *edges = (Qm3LoopEdge*)qm3_checked_realloc(*edges, (size_t)*cap * sizeof(**edges)); }
    (*edges)[(*count)++] = (Qm3LoopEdge){a, b, boundary_edge, cut};
}

static int qm3_loop_local_root(int* parent, int node) {
    while (parent[node] != node) { parent[node] = parent[parent[node]]; node = parent[node]; }
    return node;
}

static bool qm3_loop_bridge_visible(const Qm3LoopNode* nodes, const Qm3LoopEdge* edges, int edge_count, int a, int b, double tol) {
    Qm3LoopPoint p = nodes[a].p, q = nodes[b].p, r = qm3_loop_sub(q, p);
    double rr = qm3_loop_len2(r);
    if (rr <= tol * tol) return false;
    for (int ni = 0; ni < edge_count; ++ni) {
        const Qm3LoopEdge* edge = &edges[ni];
        Qm3LoopPoint c = nodes[edge->a].p, d = nodes[edge->b].p, s = qm3_loop_sub(d, c);
        double ss = qm3_loop_len2(s), den = qm3_loop_cross(r, s);
        if (ss <= tol * tol) continue;
        Qm3LoopPoint cp = qm3_loop_sub(c, p);
        if (fabs(den) <= 1e-12 * sqrt(rr * ss)) {
            if (fabs(qm3_loop_cross(cp, r)) <= tol * sqrt(rr)) return false;
            continue;
        }
        double t = qm3_loop_cross(cp, s) / den;
        double u = qm3_loop_cross(cp, r) / den;
        double t_tol = tol / sqrt(rr), u_tol = tol / sqrt(ss);
        if (t > t_tol && t < 1.0 - t_tol && u > -u_tol && u < 1.0 + u_tol) return false;
    }
    return true;
}

static bool qm3_loop_connect_components(Qm3LoopNode* nodes, int node_count, Qm3LoopEdge** edges, int* edge_count, int* edge_cap, double tol) {
    int* parent = (int*)malloc((size_t)node_count * sizeof(int));
    if (!parent) return false;
    for (int i = 0; i < node_count; ++i) parent[i] = i;
    for (int i = 0; i < *edge_count; ++i) {
        int a = qm3_loop_local_root(parent, (*edges)[i].a), b = qm3_loop_local_root(parent, (*edges)[i].b);
        if (a != b) parent[b] = a;
    }
    int boundary_node = -1;
    for (int i = 0; i < *edge_count; ++i) if ((*edges)[i].boundary_edge >= 0) { boundary_node = (*edges)[i].a; break; }
    if (boundary_node < 0) { free(parent); return false; }
    for (;;) {
        int boundary_root = qm3_loop_local_root(parent, boundary_node);
        int other_root = -1;
        for (int i = 0; i < node_count; ++i) if (qm3_loop_local_root(parent, i) != boundary_root) { other_root = qm3_loop_local_root(parent, i); break; }
        if (other_root < 0) break;
        int best_a = -1, best_b = -1;
        double best_dist2 = INFINITY;
        for (int a = 0; a < node_count; ++a) {
            if (qm3_loop_local_root(parent, a) != boundary_root) continue;
            for (int b = 0; b < node_count; ++b) {
                if (qm3_loop_local_root(parent, b) != other_root) continue;
                double dist2 = qm3_loop_len2(qm3_loop_sub(nodes[b].p, nodes[a].p));
                if (dist2 >= best_dist2 || !qm3_loop_bridge_visible(nodes, *edges, *edge_count, a, b, tol)) continue;
                best_dist2 = dist2; best_a = a; best_b = b;
            }
        }
        if (best_a < 0) { free(parent); return false; }
        qm3_loop_edge_add(edges, edge_count, edge_cap, best_a, best_b, -1, false);
        int a = qm3_loop_local_root(parent, best_a), b = qm3_loop_local_root(parent, best_b);
        if (a != b) parent[b] = a;
    }
    free(parent);
    return true;
}

static void qm3_loop_cell_push(Qm3LoopSubdivision* sub, Qm3LoopCell cell) {
    if (sub->cell_count == sub->cell_cap) { sub->cell_cap = sub->cell_cap ? 2 * sub->cell_cap : 64; sub->cells = (Qm3LoopCell*)qm3_checked_realloc(sub->cells, (size_t)sub->cell_cap * sizeof(*sub->cells)); }
    sub->cells[sub->cell_count++] = cell;
}

static void qm3_loop_portal_push(Qm3LoopSubdivision* sub, Qm3LoopPortal portal) {
    if (sub->portal_count == sub->portal_cap) { sub->portal_cap = sub->portal_cap ? 2 * sub->portal_cap : 64; sub->portals = (Qm3LoopPortal*)qm3_checked_realloc(sub->portals, (size_t)sub->portal_cap * sizeof(*sub->portals)); }
    sub->portals[sub->portal_count++] = portal;
}

static void qm3_loop_cut_push(Qm3LoopSubdivision* sub, Qm3LoopCut cut) {
    if (sub->cut_count == sub->cut_cap) { sub->cut_cap = sub->cut_cap ? 2 * sub->cut_cap : 32; sub->cuts = (Qm3LoopCut*)qm3_checked_realloc(sub->cuts, (size_t)sub->cut_cap * sizeof(*sub->cuts)); }
    sub->cuts[sub->cut_count++] = cut;
}

static bool qm3_loop_build_triangle(const QuadMeshing3DEnv* env, const Qm3PathSegmentArray* segments, int tri_index, int tri_slot, Qm3LoopSubdivision* sub, unsigned char* flags, unsigned char* boundary_tri) {
    if (!env->continuous_ctx.tri2d_valid[tri_index]) return false;
    Qm3LoopPoint tri[3] = {
        qm3_loop_point(env->continuous_ctx.tri2d_base[3 * tri_index]),
        qm3_loop_point(env->continuous_ctx.tri2d_base[3 * tri_index + 1]),
        qm3_loop_point(env->continuous_ctx.tri2d_base[3 * tri_index + 2]),
    };
    double scale = 0.0;
    for (int i = 0; i < 3; ++i) scale = fmax(scale, sqrt(qm3_loop_len2(qm3_loop_sub(tri[(i + 1) % 3], tri[i]))));
    double tol = fmax(scale * 2e-6, 1e-10), area_tol = scale * scale * 1e-12;
    sub->tri_tol[tri_slot] = tol;
    Qm3LoopPrimitive* primitives = NULL;
    int primitive_count = 0, primitive_cap = 0;
#define QM3_LOOP_PRIMITIVE_PUSH(value) do { \
    if (primitive_count == primitive_cap) { primitive_cap = primitive_cap ? 2 * primitive_cap : 8; primitives = (Qm3LoopPrimitive*)qm3_checked_realloc(primitives, (size_t)primitive_cap * sizeof(*primitives)); } \
    primitives[primitive_count++] = (value); \
} while (0)
    for (int edge = 0; edge < 3; ++edge) QM3_LOOP_PRIMITIVE_PUSH(((Qm3LoopPrimitive){.a = tri[(edge + 1) % 3], .b = tri[(edge + 2) % 3], .boundary_edge = edge}));
    for (uint32_t si = 0; si < segments->count; ++si) {
        const Qm3PathSegment* segment = &segments->data[si];
        if (segment->tri != tri_index) continue;
        Qm3Vec2 af, bf;
        if (!qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, tri_index, segment->a, &af) || !qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, tri_index, segment->b, &bf)) goto fail;
        Qm3LoopPoint a = qm3_loop_point(af), b = qm3_loop_point(bf);
        if (!qm3_loop_snap_to_triangle(&a, tri, tol * 8.0) || !qm3_loop_snap_to_triangle(&b, tri, tol * 8.0)) goto fail;
        if (qm3_loop_len2(qm3_loop_sub(b, a)) <= tol * tol) continue;
        QM3_LOOP_PRIMITIVE_PUSH(((Qm3LoopPrimitive){.a = a, .b = b, .boundary_edge = -1, .cut = true}));
        qm3_loop_cut_push(sub, (Qm3LoopCut){tri_index, a, b, tol});
        boundary_tri[tri_index] = 1;
    }
    for (int i = 0; i < primitive_count; ++i) { qm3_loop_split_push(&primitives[i], 0.0, 0.0); qm3_loop_split_push(&primitives[i], 1.0, 0.0); }
    for (int i = 0; i < primitive_count; ++i) for (int j = i + 1; j < primitive_count; ++j) qm3_loop_split_intersection(&primitives[i], &primitives[j], tol);
    Qm3LoopNode* nodes = NULL; Qm3LoopEdge* edges = NULL;
    int node_count = 0, node_cap = 0, edge_count = 0, edge_cap = 0;
    for (int i = 0; i < primitive_count; ++i) {
        Qm3LoopPrimitive* p = &primitives[i];
        qm3_loop_split_sort(p);
        Qm3LoopPoint d = qm3_loop_sub(p->b, p->a);
        for (int j = 0; j + 1 < p->split_count; ++j) {
            if ((p->split[j + 1] - p->split[j]) * sqrt(qm3_loop_len2(d)) <= tol) continue;
            Qm3LoopPoint a = qm3_loop_add(p->a, qm3_loop_scale(d, p->split[j]));
            Qm3LoopPoint b = qm3_loop_add(p->a, qm3_loop_scale(d, p->split[j + 1]));
            int na = qm3_loop_node_get(&nodes, &node_count, &node_cap, a, tol);
            int nb = qm3_loop_node_get(&nodes, &node_count, &node_cap, b, tol);
            qm3_loop_edge_add(&edges, &edge_count, &edge_cap, na, nb, p->boundary_edge, p->cut);
        }
    }
    if (node_count < 3 || edge_count < 3) { free(nodes); free(edges); goto fail; }
    if (!qm3_loop_connect_components(nodes, node_count, &edges, &edge_count, &edge_cap, tol)) { free(nodes); free(edges); goto fail; }
    Qm3LoopHalfEdge* half = (Qm3LoopHalfEdge*)calloc((size_t)edge_count * 2, sizeof(*half));
    int* degree = (int*)calloc((size_t)node_count, sizeof(int));
    int* offset = (int*)calloc((size_t)node_count + 1, sizeof(int));
    int* cursor = (int*)calloc((size_t)node_count, sizeof(int));
    int* outgoing = (int*)malloc((size_t)edge_count * 2 * sizeof(int));
    if (!half || !degree || !offset || !cursor || !outgoing) { free(nodes); free(edges); free(half); free(degree); free(offset); free(cursor); free(outgoing); goto fail; }
    for (int i = 0; i < edge_count; ++i) {
        half[2 * i] = (Qm3LoopHalfEdge){edges[i].a, 2 * i + 1, -1, i, -1, 0.0};
        half[2 * i + 1] = (Qm3LoopHalfEdge){edges[i].b, 2 * i, -1, i, -1, 0.0};
        degree[edges[i].a]++; degree[edges[i].b]++;
    }
    for (int i = 0; i < node_count; ++i) offset[i + 1] = offset[i] + degree[i];
    memcpy(cursor, offset, (size_t)node_count * sizeof(int));
    for (int hi = 0; hi < edge_count * 2; ++hi) {
        int dest = half[half[hi].twin].origin;
        Qm3LoopPoint d = qm3_loop_sub(nodes[dest].p, nodes[half[hi].origin].p);
        half[hi].angle = atan2(d.y, d.x);
        outgoing[cursor[half[hi].origin]++] = hi;
    }
    for (int n = 0; n < node_count; ++n) for (int i = offset[n] + 1; i < offset[n + 1]; ++i) {
        int value = outgoing[i], j = i;
        while (j > offset[n] && half[outgoing[j - 1]].angle > half[value].angle) { outgoing[j] = outgoing[j - 1]; --j; }
        outgoing[j] = value;
    }
    for (int hi = 0; hi < edge_count * 2; ++hi) {
        int dest = half[half[hi].twin].origin, reverse = half[hi].twin, pos = -1;
        for (int i = offset[dest]; i < offset[dest + 1]; ++i) if (outgoing[i] == reverse) { pos = i; break; }
        if (pos < 0) { free(nodes); free(edges); free(half); free(degree); free(offset); free(cursor); free(outgoing); goto fail; }
        half[hi].next = outgoing[pos == offset[dest] ? offset[dest + 1] - 1 : pos - 1];
    }
    double tri_orient = qm3_loop_cross(qm3_loop_sub(tri[1], tri[0]), qm3_loop_sub(tri[2], tri[0]));
    unsigned char* visited = (unsigned char*)calloc((size_t)edge_count * 2, 1);
    double area_sum = 0.0;
    for (int start = 0; start < edge_count * 2; ++start) {
        if (visited[start]) continue;
        int h = start, count = 0;
        double twice_area = 0.0;
        do {
            if (h < 0 || h >= edge_count * 2 || visited[h] || count > edge_count * 2) { free(visited); free(nodes); free(edges); free(half); free(degree); free(offset); free(cursor); free(outgoing); goto fail; }
            visited[h] = 1;
            Qm3LoopPoint a = nodes[half[h].origin].p, b = nodes[half[half[h].next].origin].p;
            twice_area += qm3_loop_cross(a, b); h = half[h].next; count++;
        } while (h != start);
        double area = 0.5 * twice_area * (tri_orient > 0.0 ? 1.0 : -1.0);
        if (area <= area_tol) continue;
        Qm3LoopCell cell = {.polygon = (Qm3LoopPoint*)malloc((size_t)count * sizeof(Qm3LoopPoint)), .polygon_count = count, .tri = tri_index, .area = area};
        if (!cell.polygon) { free(visited); free(nodes); free(edges); free(half); free(degree); free(offset); free(cursor); free(outgoing); goto fail; }
        h = start;
        for (int i = 0; i < count; ++i) { cell.polygon[i] = nodes[half[h].origin].p; half[h].cell = sub->cell_count; h = half[h].next; }
        qm3_loop_cell_push(sub, cell); area_sum += area;
    }
    free(visited);
    if (fabs(area_sum - 0.5 * fabs(tri_orient)) > fmax(area_tol * 32.0, fabs(tri_orient) * 2e-5)) { free(nodes); free(edges); free(half); free(degree); free(offset); free(cursor); free(outgoing); goto fail; }
    Qm3Tri surface_tri = env->surface.triangles[tri_index];
    uint32_t vertices[3] = {surface_tri.a, surface_tri.b, surface_tri.c};
    for (int i = 0; i < edge_count; ++i) if (edges[i].boundary_edge >= 0) {
        int cell = half[2 * i].cell >= 0 ? half[2 * i].cell : half[2 * i + 1].cell;
        if (cell < 0 || (half[2 * i].cell >= 0 && half[2 * i + 1].cell >= 0)) { free(nodes); free(edges); free(half); free(degree); free(offset); free(cursor); free(outgoing); goto fail; }
        int edge = edges[i].boundary_edge, la = (edge + 1) % 3, lb = (edge + 2) % 3;
        uint32_t va = vertices[la], vb = vertices[lb]; Qm3LoopPoint origin = tri[la], d = qm3_loop_sub(tri[lb], origin);
        double den = qm3_loop_len2(d), t0 = qm3_loop_dot(qm3_loop_sub(nodes[edges[i].a].p, origin), d) / den, t1 = qm3_loop_dot(qm3_loop_sub(nodes[edges[i].b].p, origin), d) / den;
        if (va > vb) { uint32_t swap = va; va = vb; vb = swap; t0 = 1.0 - t0; t1 = 1.0 - t1; }
        qm3_loop_portal_push(sub, (Qm3LoopPortal){tri_index, edge, cell, va, vb, fmin(t0, t1), fmax(t0, t1), edges[i].cut});
        if (edges[i].cut) flags[3 * tri_index + edge] |= QM3_LOOP_EDGE_BLOCKED;
    }
    free(nodes); free(edges); free(half); free(degree); free(offset); free(cursor); free(outgoing);
    for (int i = 0; i < primitive_count; ++i) free(primitives[i].split);
    free(primitives);
#undef QM3_LOOP_PRIMITIVE_PUSH
    return true;
fail:
    for (int i = 0; i < primitive_count; ++i) free(primitives[i].split);
    free(primitives);
#undef QM3_LOOP_PRIMITIVE_PUSH
    return false;
}

static int qm3_loop_find_root(int* parent, int a) {
    while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
    return a;
}

static void qm3_loop_union(int* parent, unsigned char* rank, int a, int b) {
    a = qm3_loop_find_root(parent, a); b = qm3_loop_find_root(parent, b);
    if (a == b) return;
    if (rank[a] < rank[b]) { int swap = a; a = b; b = swap; }
    parent[b] = a;
    if (rank[a] == rank[b]) rank[a]++;
}

static bool qm3_loop_point_in_cell(Qm3LoopPoint p, const Qm3LoopCell* cell, double tol, bool* boundary) {
    int winding = 0; *boundary = false;
    for (int i = 0; i < cell->polygon_count; ++i) {
        Qm3LoopPoint a = cell->polygon[i], b = cell->polygon[(i + 1) % cell->polygon_count];
        if (qm3_loop_point_segment_dist2(p, a, b) <= tol * tol) { *boundary = true; return true; }
        double side = qm3_loop_cross(qm3_loop_sub(b, a), qm3_loop_sub(p, a));
        if (a.y <= p.y && b.y > p.y && side > 0.0) winding++;
        if (a.y > p.y && b.y <= p.y && side < 0.0) winding--;
    }
    return winding != 0;
}

static int qm3_loop_locate_cell(const Qm3LoopSubdivision* sub, int tri_slot, Qm3LoopPoint p, bool allow_boundary) {
    int found = -1;
    for (int i = sub->tri_cell_offsets[tri_slot]; i < sub->tri_cell_offsets[tri_slot + 1]; ++i) {
        bool boundary = false;
        if (!qm3_loop_point_in_cell(p, &sub->cells[i], sub->tri_tol[tri_slot], &boundary)) continue;
        if (boundary && !allow_boundary) continue;
        if (found >= 0 && found != i) return -1;
        found = i;
    }
    return found;
}

static int qm3_loop_tri_slot(const Qm3LoopSubdivision* sub, int tri) {
    int lo = 0, hi = sub->tri_count;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (sub->tri_ids[mid] < tri) lo = mid + 1;
        else hi = mid;
    }
    return lo < sub->tri_count && sub->tri_ids[lo] == tri ? lo : -1;
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

static void qm3_mark_vertex_frontier_dirty(QuadMeshing3DEnv* env, uint32_t vidx) {
    if (vidx < env->mesh.vertex_count) env->mesh.vertices[vidx].frontier_quality_dirty = true;
}

static void qm3_mark_edge_frontier_dirty(QuadMeshing3DEnv* env, uint32_t eidx) {
    if (eidx >= env->mesh.edge_count) return;
    Qm3MeshEdge* edge = &env->mesh.edges[eidx];
    qm3_mark_vertex_frontier_dirty(env, edge->a);
    qm3_mark_vertex_frontier_dirty(env, edge->b);
}

static void qm3_mark_face_frontier_dirty(QuadMeshing3DEnv* env, const Qm3MeshFace* face) {
    if (!face) return;
    for (uint32_t i = 0; i < face->n; ++i) qm3_mark_vertex_frontier_dirty(env, face->vertices[i]);
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

static int qm3_face_interior_side(const Qm3Mesh* mesh, const Qm3MeshFace* face) {
    if (!face || face->disabled || face->n < 3) return 0;
    Qm3Vec3 cycle_normal = {0};
    Qm3Vec3 surface_normal = {0};
    for (uint32_t i = 0; i < face->n; ++i) {
        const Qm3MeshVertex* a = &mesh->vertices[face->vertices[i]];
        const Qm3MeshVertex* b = &mesh->vertices[face->vertices[(i + 1) % face->n]];
        cycle_normal.x += a->pos.y * b->pos.z - a->pos.z * b->pos.y;
        cycle_normal.y += a->pos.z * b->pos.x - a->pos.x * b->pos.z;
        cycle_normal.z += a->pos.x * b->pos.y - a->pos.y * b->pos.x;
        surface_normal = qm3_add(surface_normal, a->normal);
    }
    if (qm3_dot(cycle_normal, cycle_normal) <= 1e-20f || qm3_dot(surface_normal, surface_normal) <= 1e-20f) return 0;
    return qm3_dot(cycle_normal, surface_normal) >= 0.0f ? 1 : 2;
}

static bool qm3_mesh_vertex_is_registered_face_boundary(const Qm3Mesh* mesh, uint32_t vidx) {
    for (uint32_t fi = 0; fi < mesh->face_count; ++fi) {
        if (qm3_mesh_vertex_is_face_boundary(&mesh->faces[fi], vidx)) return true;
    }
    return false;
}

static bool qm3_mesh_edge_is_registered_face_boundary(const Qm3Mesh* mesh, uint32_t eidx) {
    for (uint32_t fi = 0; fi < mesh->face_count; ++fi) {
        if (qm3_mesh_edge_is_face_boundary(&mesh->faces[fi], eidx)) return true;
    }
    return false;
}

static bool qm3_sample_is_registered_face_vertex(const QuadMeshing3DEnv* env, uint32_t sample_id) {
    if (sample_id >= env->surface.sample_count) return false;
    if (env->sample_to_graph && env->sample_to_graph[sample_id] >= 0) {
        return qm3_mesh_vertex_is_registered_face_boundary(&env->mesh, (uint32_t)env->sample_to_graph[sample_id]);
    }
    for (uint32_t vi = 0; vi < env->mesh.vertex_count; ++vi) {
        const Qm3MeshVertex* vertex = &env->mesh.vertices[vi];
        if (!vertex->disabled && vertex->sample_id == sample_id && qm3_mesh_vertex_is_registered_face_boundary(&env->mesh, vi)) return true;
    }
    return false;
}

static int qm3_loop_segment_tri_compare(const void* a, const void* b) {
    const Qm3PathSegment* sa = (const Qm3PathSegment*)a;
    const Qm3PathSegment* sb = (const Qm3PathSegment*)b;
    return (sa->tri > sb->tri) - (sa->tri < sb->tri);
}

static bool qm3_loop_label_cell(int* parent, unsigned char* root_side, int cell, int side) {
    if (cell < 0) return true;
    int root = qm3_loop_find_root(parent, cell);
    if (root_side[root] != 0 && root_side[root] != side) return false;
    root_side[root] = (unsigned char)side;
    return true;
}

static int qm3_loop_cut_root_seed_side(
    const QuadMeshing3DEnv* env,
    const Qm3LoopSubdivision* sub,
    const Qm3PathSegment* sorted_segments,
    const Qm3IntArray* segment_offsets,
    int* parent,
    int root
) {
    for (int tri_slot = 0; tri_slot < sub->tri_count; ++tri_slot) {
        int tri = sub->tri_ids[tri_slot];
        for (int pi = sub->tri_portal_offsets[tri_slot]; pi < sub->tri_portal_offsets[tri_slot + 1]; ++pi) {
            const Qm3LoopPortal* portal = &sub->portals[pi];
            if (!portal->cut || qm3_loop_find_root(parent, portal->cell) != root) continue;
            const Qm3LoopCell* cell = &sub->cells[portal->cell];
            Qm3LoopPoint center = {0};
            for (int p = 0; p < cell->polygon_count; ++p) center = qm3_loop_add(center, cell->polygon[p]);
            center = qm3_loop_scale(center, 1.0 / cell->polygon_count);

            Qm3Tri st = env->surface.triangles[tri];
            uint32_t vertices[3] = {st.a, st.b, st.c};
            int la = (portal->edge + 1) % 3, lb = (portal->edge + 2) % 3;
            Qm3LoopPoint edge_a = qm3_loop_point(env->continuous_ctx.tri2d_base[3 * tri + la]);
            Qm3LoopPoint edge_b = qm3_loop_point(env->continuous_ctx.tri2d_base[3 * tri + lb]);
            double edge_t = 0.5 * (portal->lo + portal->hi);
            if (vertices[la] > vertices[lb]) edge_t = 1.0 - edge_t;
            Qm3LoopPoint portal_mid = qm3_loop_add(edge_a, qm3_loop_scale(qm3_loop_sub(edge_b, edge_a), edge_t));

            for (int si = segment_offsets->data[tri_slot]; si < segment_offsets->data[tri_slot + 1]; ++si) {
                const Qm3PathSegment* segment = &sorted_segments[si];
                Qm3Vec2 af, bf;
                if (!qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, tri, segment->a, &af) ||
                        !qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, tri, segment->b, &bf)) continue;
                Qm3LoopPoint a = qm3_loop_point(af), b = qm3_loop_point(bf), d = qm3_loop_sub(b, a);
                double tol_sq = sub->tri_tol[tri_slot] * sub->tri_tol[tri_slot];
                if (qm3_loop_point_segment_dist2(portal_mid, a, b) > 4.0 * tol_sq) continue;
                if (qm3_loop_point_segment_dist2(center, a, b) <= tol_sq) continue;
                return qm3_loop_cross(d, qm3_loop_sub(center, a)) > 0.0 ? 1 : 2;
            }
        }
    }
    return 0;
}

static int qm3_loop_locate_side_cell(const Qm3LoopSubdivision* sub, int tri_slot, Qm3LoopPoint p,
    int* parent, const unsigned char* root_side, int side) {
    int found = -1;
    for (int i = sub->tri_cell_offsets[tri_slot]; i < sub->tri_cell_offsets[tri_slot + 1]; ++i) {
        if (root_side[qm3_loop_find_root(parent, i)] != side) continue;
        bool boundary = false;
        if (!qm3_loop_point_in_cell(p, &sub->cells[i], sub->tri_tol[tri_slot], &boundary)) continue;
        if (found >= 0 && found != i) return -1;
        found = i;
    }
    return found;
}

static float qm3_surface_triangle_area(const Qm3Surface* surface, int tri) {
    Qm3Tri t = surface->triangles[tri];
    Qm3Vec3 a = surface->vertices[t.a], b = surface->vertices[t.b], c = surface->vertices[t.c];
    float abx = b.x - a.x, aby = b.y - a.y, abz = b.z - a.z;
    float acx = c.x - a.x, acy = c.y - a.y, acz = c.z - a.z;
    float x = aby * acz - abz * acy;
    float y = abz * acx - abx * acz;
    float z = abx * acy - aby * acx;
    return 0.5f * sqrtf(x * x + y * y + z * z);
}

static bool qm3_loop_flood_enqueue(int tri, int side, const unsigned char* boundary_tri, unsigned char* side_mark,
    unsigned char* conflict_tri, Qm3IntArray* queue, uint32_t* conflicts) {
    if (tri < 0 || boundary_tri[tri]) return true;
    if (side_mark[tri] == side) return true;
    if (side_mark[tri] != 0) {
        conflict_tri[tri] = 1;
        (*conflicts)++;
        return false;
    }
    side_mark[tri] = (unsigned char)side;
    qm3_int_push(queue, tri);
    return true;
}

static bool qm3_loop_flood_pop(const QuadMeshing3DEnv* env, int side, const unsigned char* boundary_tri,
    unsigned char* side_mark, unsigned char* conflict_tri, Qm3IntArray* queue, int* head,
    Qm3IntArray* reached, float* area, uint32_t* tris, uint32_t* conflicts) {
    if (*head >= queue->size) return true;
    int tri = queue->data[(*head)++];
    *area += qm3_surface_triangle_area(&env->surface, tri);
    (*tris)++;
    qm3_int_push(reached, tri);
    for (int edge = 0; edge < 3; ++edge) {
        int neighbor = ((int*)&env->surface.triangle_neighbors[tri])[edge];
        if (!qm3_loop_flood_enqueue(neighbor, side, boundary_tri, side_mark, conflict_tri, queue, conflicts)) return false;
    }
    return true;
}

static Qm3LoopRemovalStats qm3_remove_samples_inside_loop(QuadMeshing3DEnv* env, const Qm3PathSegmentArray* loop_segments, int interior_side) {
    Qm3LoopRemovalStats stats = {0};
    if (loop_segments->count == 0 || !env->sample_disabled) return stats;
    uint32_t tri_count = env->surface.triangle_count;
    unsigned char* flags = (unsigned char*)calloc((size_t)tri_count * 3, 1);
    unsigned char* boundary_tri = (unsigned char*)calloc((size_t)tri_count, 1);
    unsigned char* side_mark = (unsigned char*)calloc((size_t)tri_count, 1);
    unsigned char* conflict_tri = (unsigned char*)calloc((size_t)tri_count, 1);
    Qm3LoopSubdivision sub = {0};
    sub.tri_ids = (int*)malloc((size_t)loop_segments->count * sizeof(int));
    sub.tri_cell_offsets = (int*)calloc((size_t)loop_segments->count + 1, sizeof(int));
    sub.tri_portal_offsets = (int*)calloc((size_t)loop_segments->count + 1, sizeof(int));
    sub.tri_tol = (double*)calloc((size_t)loop_segments->count, sizeof(double));
    Qm3PathSegment* sorted_segments = (Qm3PathSegment*)malloc((size_t)loop_segments->count * sizeof(*sorted_segments));
    Qm3IntArray boundary_list = {0}, segment_offsets = {0}, cut_offsets = {0}, cut_constraints = {0};
    bool valid = flags && boundary_tri && side_mark && conflict_tri && sub.tri_ids && sub.tri_cell_offsets &&
        sub.tri_portal_offsets && sub.tri_tol && sorted_segments && env->surface.triangle_neighbors;
    if (valid) {
        memcpy(sorted_segments, loop_segments->data, (size_t)loop_segments->count * sizeof(*sorted_segments));
        qsort(sorted_segments, loop_segments->count, sizeof(*sorted_segments), qm3_loop_segment_tri_compare);
    }
    for (uint32_t begin = 0; valid && begin < loop_segments->count;) {
        int tri = sorted_segments[begin].tri;
        if (tri < 0 || tri >= (int)tri_count) { valid = false; break; }
        uint32_t end = begin + 1;
        while (end < loop_segments->count && sorted_segments[end].tri == tri) end++;
        boundary_tri[tri] = 1;
        int tri_slot = sub.tri_count++;
        sub.tri_ids[tri_slot] = tri;
        qm3_int_push(&boundary_list, tri);
        qm3_int_push(&segment_offsets, (int)begin);
        qm3_int_push(&cut_offsets, sub.cut_count);
        sub.tri_cell_offsets[tri_slot] = sub.cell_count;
        sub.tri_portal_offsets[tri_slot] = sub.portal_count;
        Qm3PathSegmentArray local = {.data = &sorted_segments[begin], .count = end - begin, .cap = end - begin};
        valid = qm3_loop_build_triangle(env, &local, tri, tri_slot, &sub, flags, boundary_tri);
        sub.tri_cell_offsets[tri_slot + 1] = sub.cell_count;
        sub.tri_portal_offsets[tri_slot + 1] = sub.portal_count;
        begin = end;
    }
    qm3_int_push(&segment_offsets, (int)loop_segments->count);
    qm3_int_push(&cut_offsets, sub.cut_count);
    stats.boundary_tris = (uint32_t)boundary_list.size;
    int* parent = NULL;
    unsigned char* rank = NULL;
    unsigned char* root_side = NULL;
    double* breaks = NULL;
    int breaks_cap = 0;
    int failure_stage = 1;
    uint32_t ambiguous_classifications = 0;
    if (valid) {
        parent = (int*)malloc((size_t)sub.cell_count * sizeof(int));
        rank = (unsigned char*)calloc((size_t)sub.cell_count, 1);
        root_side = (unsigned char*)calloc((size_t)sub.cell_count, 1);
        valid = parent && rank && root_side && (env->surface.sample_count == 0 ||
            (env->surface_topo.tri_sample_offsets && env->surface_topo.tri_sample_ids));
    }
    for (int i = 0; valid && i < sub.cell_count; ++i) parent[i] = i;

    /* Stitch only boundary cells. The scratch buffer is reused for every edge. */
    if (valid) failure_stage = 2;
    for (int bi = 0; valid && bi < boundary_list.size; ++bi) {
        int tri = boundary_list.data[bi];
        int tri_slot = bi;
        for (int edge = 0; valid && edge < 3; ++edge) {
            int neighbor = ((int*)&env->surface.triangle_neighbors[tri])[edge];
            if (neighbor < 0 || neighbor < tri || !boundary_tri[neighbor]) continue;
            int neighbor_slot = qm3_loop_tri_slot(&sub, neighbor);
            if (neighbor_slot < 0) { valid = false; break; }
            Qm3Tri surface_tri = env->surface.triangles[tri];
            uint32_t vertices[3] = {surface_tri.a, surface_tri.b, surface_tri.c};
            uint32_t va = vertices[(edge + 1) % 3], vb = vertices[(edge + 2) % 3];
            if (va > vb) { uint32_t swap = va; va = vb; vb = swap; }
            int max_breaks = 2;
            for (int i = sub.tri_portal_offsets[tri_slot]; i < sub.tri_portal_offsets[tri_slot + 1]; ++i) if (sub.portals[i].edge == edge) max_breaks += 2;
            for (int i = sub.tri_portal_offsets[neighbor_slot]; i < sub.tri_portal_offsets[neighbor_slot + 1]; ++i) if (sub.portals[i].va == va && sub.portals[i].vb == vb) max_breaks += 2;
            if (max_breaks > breaks_cap) {
                breaks_cap = max_breaks;
                breaks = (double*)qm3_checked_realloc(breaks, (size_t)breaks_cap * sizeof(double));
            }
            int break_count = 0;
            breaks[break_count++] = 0.0;
            breaks[break_count++] = 1.0;
            for (int i = sub.tri_portal_offsets[tri_slot]; i < sub.tri_portal_offsets[tri_slot + 1]; ++i) {
                const Qm3LoopPortal* portal = &sub.portals[i];
                if (portal->edge != edge) continue;
                breaks[break_count++] = portal->lo; breaks[break_count++] = portal->hi;
            }
            for (int i = sub.tri_portal_offsets[neighbor_slot]; i < sub.tri_portal_offsets[neighbor_slot + 1]; ++i) {
                const Qm3LoopPortal* portal = &sub.portals[i];
                if (portal->va != va || portal->vb != vb) continue;
                breaks[break_count++] = portal->lo; breaks[break_count++] = portal->hi;
            }
            for (int i = 1; i < break_count; ++i) {
                double value = breaks[i]; int j = i;
                while (j > 0 && breaks[j - 1] > value) { breaks[j] = breaks[j - 1]; --j; }
                breaks[j] = value;
            }
            double edge_length = qm3_distance(env->surface.vertices[va], env->surface.vertices[vb]);
            double param_tol = edge_length > 0.0
                ? fmax(8.0 * fmax(sub.tri_tol[tri_slot], sub.tri_tol[neighbor_slot]) / edge_length, 1e-10)
                : 1e-10;
            int unique_count = 0;
            for (int i = 0; i < break_count; ++i) {
                double value = fmin(fmax(breaks[i], 0.0), 1.0);
                if (unique_count > 0 && value - breaks[unique_count - 1] <= param_tol) {
                    breaks[unique_count - 1] = 0.5 * (breaks[unique_count - 1] + value);
                } else {
                    breaks[unique_count++] = value;
                }
            }
            for (int bi = 0; valid && bi + 1 < unique_count; ++bi) {
                if (breaks[bi + 1] - breaks[bi] <= 2.0 * param_tol) continue;
                double midpoint = 0.5 * (breaks[bi] + breaks[bi + 1]);
                const Qm3LoopPortal* a = NULL;
                const Qm3LoopPortal* b = NULL;
                for (int i = sub.tri_portal_offsets[tri_slot]; i < sub.tri_portal_offsets[tri_slot + 1]; ++i) {
                    const Qm3LoopPortal* portal = &sub.portals[i];
                    if (portal->edge == edge && midpoint >= portal->lo - param_tol && midpoint <= portal->hi + param_tol) { a = portal; break; }
                }
                for (int i = sub.tri_portal_offsets[neighbor_slot]; i < sub.tri_portal_offsets[neighbor_slot + 1]; ++i) {
                    const Qm3LoopPortal* portal = &sub.portals[i];
                    if (portal->va == va && portal->vb == vb && midpoint >= portal->lo - param_tol && midpoint <= portal->hi + param_tol) { b = portal; break; }
                }
                if (!a || !b) { valid = false; break; }
                if (!a->cut && !b->cut) {
                    qm3_loop_union(parent, rank, a->cell, b->cell);
                } else {
                    qm3_int_push(&cut_constraints, a->cell);
                    qm3_int_push(&cut_constraints, b->cell);
                }
            }
        }
    }

    /* Directed probes label boundary-cell components without assuming one cell per triangle. */
    if (valid) failure_stage = 3;
    for (uint32_t si = 0; valid && si < loop_segments->count; ++si) {
        const Qm3PathSegment* segment = &sorted_segments[si];
        int tri_slot = qm3_loop_tri_slot(&sub, segment->tri);
        if (tri_slot < 0) { valid = false; break; }
        Qm3Vec2 af, bf;
        if (!qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, segment->tri, segment->a, &af) ||
            !qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, segment->tri, segment->b, &bf)) { valid = false; break; }
        Qm3LoopPoint a = qm3_loop_point(af), b = qm3_loop_point(bf), d = qm3_loop_sub(b, a);
        double len = sqrt(qm3_loop_len2(d));
        if (len <= sub.tri_tol[tri_slot]) continue;
        if (qm3_loop_segment_on_triangle_edge(env, segment->tri, a, b, sub.tri_tol[tri_slot] * 4.0)) continue;
        Qm3LoopPoint normal = {-d.y / len, d.x / len};
        const double positions[3] = {0.25, 0.5, 0.75};
        for (int pi = 0; pi < 3 && valid; ++pi) {
            Qm3LoopPoint mid = qm3_loop_add(a, qm3_loop_scale(d, positions[pi]));
            double eps = fmax(sub.tri_tol[tri_slot] * 4.0, len * 1e-6);
            for (int attempt = 0; attempt < 5; ++attempt) {
                int left = qm3_loop_locate_cell(&sub, tri_slot, qm3_loop_add(mid, qm3_loop_scale(normal, eps)), false);
                int right = qm3_loop_locate_cell(&sub, tri_slot, qm3_loop_add(mid, qm3_loop_scale(normal, -eps)), false);
                int left_probe_root = left >= 0 ? qm3_loop_find_root(parent, left) : -1;
                int right_probe_root = right >= 0 ? qm3_loop_find_root(parent, right) : -1;
                if (left_probe_root >= 0 && left_probe_root == right_probe_root) {
                    eps *= 4.0;
                    continue;
                }
                if (left_probe_root >= 0 && !qm3_loop_label_cell(parent, root_side, left, 1)) valid = false;
                if (right_probe_root >= 0 && !qm3_loop_label_cell(parent, root_side, right, 2)) valid = false;
                if (left_probe_root >= 0 || right_probe_root >= 0) break;
                eps *= 0.25;
            }
        }
    }

    /* Shared edge cuts are exact opposite-side constraints on the atomic
       intervals built above. Solve those constraints before handling cuts
       whose adjacent triangle is outside the local subdivision. */
    if (valid) failure_stage = 4;
    bool constraints_pending = cut_constraints.size > 0;
    while (valid && constraints_pending) {
        bool changed = false;
        constraints_pending = false;
        int first_unlabeled = -1;
        for (int i = 0; i + 1 < cut_constraints.size; i += 2) {
            int a = cut_constraints.data[i];
            int b = cut_constraints.data[i + 1];
            int root_a = qm3_loop_find_root(parent, a);
            int root_b = qm3_loop_find_root(parent, b);
            if (root_a == root_b) { valid = false; break; }
            int side_a = root_side[root_a];
            int side_b = root_side[root_b];
            if (side_a != 0 && side_b != 0) {
                if (side_a == side_b) { valid = false; break; }
            } else if (side_a != 0) {
                root_side[root_b] = (unsigned char)(3 - side_a);
                changed = true;
            } else if (side_b != 0) {
                root_side[root_a] = (unsigned char)(3 - side_b);
                changed = true;
            } else {
                constraints_pending = true;
                if (first_unlabeled < 0) first_unlabeled = root_a;
            }
        }
        if (valid && constraints_pending && !changed) {
            int seed_side = qm3_loop_cut_root_seed_side(
                env, &sub, sorted_segments, &segment_offsets, parent, first_unlabeled);
            if (seed_side == 0) valid = false;
            else root_side[first_unlabeled] = (unsigned char)seed_side;
        }
    }

    /* A cut on the edge of the local boundary set has no second cell
       constraint. Its local direction supplies the seed used by the flood. */
    for (int bi = 0; valid && bi < boundary_list.size; ++bi) {
        int tri_slot = bi;
        for (int i = sub.tri_portal_offsets[tri_slot]; valid && i < sub.tri_portal_offsets[tri_slot + 1]; ++i) {
            const Qm3LoopPortal* portal = &sub.portals[i];
            int root = qm3_loop_find_root(parent, portal->cell);
            if (!portal->cut || root_side[root] != 0) continue;
            int seed_side = qm3_loop_cut_root_seed_side(
                env, &sub, sorted_segments, &segment_offsets, parent, root);
            if (seed_side == 0) valid = false;
            else root_side[root] = (unsigned char)seed_side;
        }
    }

    if (valid) failure_stage = 5;
    for (int ci = 0; valid && ci < sub.cell_count; ++ci) {
        int side = root_side[qm3_loop_find_root(parent, ci)];
        if (side == 0) { valid = false; break; }
        if (side == 1) stats.left_area += (float)sub.cells[ci].area;
        else stats.right_area += (float)sub.cells[ci].area;
    }
    for (int bi = 0; valid && bi < boundary_list.size; ++bi) {
        int tri = boundary_list.data[bi];
        bool has_left = false, has_right = false;
        for (int ci = sub.tri_cell_offsets[bi]; ci < sub.tri_cell_offsets[bi + 1]; ++ci) {
            int side = root_side[qm3_loop_find_root(parent, ci)];
            has_left |= side == 1;
            has_right |= side == 2;
        }
        if (has_left) stats.left_tris++;
        if (has_right) stats.right_tris++;
        if (has_left != has_right) side_mark[tri] = has_left ? 1 : 2;
    }
    for (int i = 0; valid && i < sub.portal_count; ++i) {
        const Qm3LoopPortal* portal = &sub.portals[i];
        int side = root_side[qm3_loop_find_root(parent, portal->cell)];
        flags[3 * portal->tri + portal->edge] |= side == 1 ? QM3_LOOP_EDGE_LEFT : QM3_LOOP_EDGE_RIGHT;
    }

    Qm3IntArray queue_left = {0}, queue_right = {0}, reached_left = {0}, reached_right = {0};
    int left_head = 0, right_head = 0;
    if (valid) failure_stage = 6;
    for (int i = 0; valid && i < sub.portal_count; ++i) {
        const Qm3LoopPortal* portal = &sub.portals[i];
        int neighbor = ((int*)&env->surface.triangle_neighbors[portal->tri])[portal->edge];
        if (neighbor < 0 || boundary_tri[neighbor]) continue;
        int side = root_side[qm3_loop_find_root(parent, portal->cell)];
        if (portal->cut) side = 3 - side;
        Qm3IntArray* queue = side == 1 ? &queue_left : &queue_right;
        if (!qm3_loop_flood_enqueue(neighbor, side, boundary_tri, side_mark, conflict_tri, queue, &stats.flood_conflicts)) valid = false;
    }
    bool prefer_left = true;
    while (valid) {
        bool left_done = left_head >= queue_left.size;
        bool right_done = right_head >= queue_right.size;
        if ((left_done && stats.right_area >= stats.left_area) ||
            (right_done && stats.left_area >= stats.right_area)) break;
        if (left_done && right_done) break;
        if (prefer_left && !left_done) {
            valid = qm3_loop_flood_pop(env, 1, boundary_tri, side_mark, conflict_tri, &queue_left, &left_head,
                &reached_left, &stats.left_area, &stats.left_tris, &stats.flood_conflicts);
        } else if (!right_done) {
            valid = qm3_loop_flood_pop(env, 2, boundary_tri, side_mark, conflict_tri, &queue_right, &right_head,
                &reached_right, &stats.right_area, &stats.right_tris, &stats.flood_conflicts);
        } else {
            valid = qm3_loop_flood_pop(env, 1, boundary_tri, side_mark, conflict_tri, &queue_left, &left_head,
                &reached_left, &stats.left_area, &stats.left_tris, &stats.flood_conflicts);
        }
        prefer_left = !prefer_left;
    }
    stats.flood_tris = (uint32_t)(reached_left.size + reached_right.size);
    if (valid) {
        if (stats.left_area < stats.right_area) stats.chosen_side = 1;
        else if (stats.right_area < stats.left_area) stats.chosen_side = 2;
        else stats.chosen_side = interior_side == 2 ? 2 : 1;
    }

    /* Collect only samples in the completed smaller flood and boundary cells.
       Mutation is delayed until all selected samples classify unambiguously. */
    Qm3IntArray selected = {0};
    if (valid) failure_stage = 7;
    Qm3IntArray* chosen_reached = stats.chosen_side == 1 ? &reached_left : &reached_right;
    for (int ri = 0; valid && ri < chosen_reached->size; ++ri) {
        int tri = chosen_reached->data[ri];
        for (int si = env->surface_topo.tri_sample_offsets[tri]; si < env->surface_topo.tri_sample_offsets[tri + 1]; ++si) {
            uint32_t sample_id = (uint32_t)env->surface_topo.tri_sample_ids[si];
            if (sample_id >= env->surface.sample_count) { valid = false; break; }
            if (!env->sample_disabled[sample_id] && !qm3_sample_is_registered_face_vertex(env, sample_id)) qm3_int_push(&selected, (int)sample_id);
        }
    }
    for (int bi = 0; valid && bi < boundary_list.size; ++bi) {
        int tri = boundary_list.data[bi];
        for (int si = env->surface_topo.tri_sample_offsets[tri]; si < env->surface_topo.tri_sample_offsets[tri + 1]; ++si) {
            uint32_t sample_id = (uint32_t)env->surface_topo.tri_sample_ids[si];
            if (sample_id >= env->surface.sample_count) { valid = false; break; }
            if (env->sample_disabled[sample_id]) continue;
            Qm3Vec2 pf;
            if (!qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, tri, env->surface.samples[sample_id].p, &pf)) {
                ambiguous_classifications++;
                continue;
            }
            Qm3LoopPoint p = qm3_loop_point(pf);
            bool on_cut = false;
            for (int ci = cut_offsets.data[bi]; ci < cut_offsets.data[bi + 1]; ++ci) {
                const Qm3LoopCut* cut = &sub.cuts[ci];
                if (qm3_loop_point_segment_dist2(p, cut->a, cut->b) <= cut->tol * cut->tol) { on_cut = true; break; }
            }
            if (on_cut) { ambiguous_classifications++; continue; }
            int cell = qm3_loop_locate_side_cell(&sub, bi, p, parent, root_side, stats.chosen_side);
            if (cell < 0) continue;
            if (!qm3_sample_is_registered_face_vertex(env, sample_id)) qm3_int_push(&selected, (int)sample_id);
        }
    }
    for (int i = 0; valid && i < selected.size; ++i) {
        uint32_t sample_id = (uint32_t)selected.data[i];
        env->sample_disabled[sample_id] = 1;
        if (env->sample_to_graph) env->sample_to_graph[sample_id] = -1;
        env->disabled_sample_count++; stats.disabled_samples++;
    }
    if (!valid) {
        fprintf(stderr, "loop subdivision failed: stage=%d boundary=%d cells=%d cuts=%d\n", failure_stage, boundary_list.size, sub.cell_count, sub.cut_count);
        stats.chosen_side = 0;
        stats.disabled_samples = 0;
        stats.classification_conflicts++;
        for (int bi = 0; bi < boundary_list.size; ++bi) conflict_tri[boundary_list.data[bi]] = 1;
    }
    stats.classification_conflicts += ambiguous_classifications;
    qm3_loop_debug_capture(&env->loop_debug, flags, boundary_tri, side_mark, conflict_tri, tri_count);
    free(parent); free(rank); free(root_side); free(breaks); free(sorted_segments);
    qm3_int_free(&boundary_list); qm3_int_free(&segment_offsets); qm3_int_free(&cut_offsets); qm3_int_free(&cut_constraints);
    qm3_int_free(&queue_left); qm3_int_free(&queue_right); qm3_int_free(&reached_left); qm3_int_free(&reached_right); qm3_int_free(&selected);
    qm3_loop_subdivision_free(&sub);
    free(flags); free(boundary_tri); free(side_mark); free(conflict_tri);
    return stats;
}

static void qm3_prune_graph_after_loop_removal(QuadMeshing3DEnv* env) {
    if (!env->sample_disabled) return;
    for (uint32_t vi = 0; vi < env->mesh.vertex_count; ++vi) {
        Qm3MeshVertex* v = &env->mesh.vertices[vi];
        if (v->disabled || v->sample_id >= env->surface.sample_count) continue;
        if (!env->sample_disabled[v->sample_id]) continue;
        if (qm3_mesh_vertex_is_registered_face_boundary(&env->mesh, vi)) continue;
        v->disabled = true;
        if (v->frontier_index >= 0) qm3_mesh_update_frontier_vertex(&env->mesh, vi);
    }
    for (uint32_t ei = 0; ei < env->mesh.edge_count; ++ei) {
        Qm3MeshEdge* e = &env->mesh.edges[ei];
        if (e->disabled || qm3_mesh_edge_is_registered_face_boundary(&env->mesh, ei)) continue;
        if (env->mesh.vertices[e->a].disabled || env->mesh.vertices[e->b].disabled) {
            qm3_mark_edge_frontier_dirty(env, ei);
            qm3_mesh_disable_edge(&env->mesh, ei);
        }
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

static void qm3_insert_graph_tri_path_points(QuadMeshing3DEnv* env, uint32_t ei) {
    if (ei >= env->mesh.edge_count) return;
    qm3_ensure_graph_tri_buckets(env);
    const Qm3MeshEdge* edge = &env->mesh.edges[ei];
    if (edge->disabled || edge->path_count < 3 || edge->segment_count == 0) return;
    if ((uint64_t)edge->path_offset + edge->path_count > env->mesh.edge_path_points.count) return;
    if ((uint64_t)edge->segment_offset + edge->segment_count > env->mesh.edge_path_segments.count) return;

    float tol = fmaxf(qm3_surface_diag(&env->surface) * 1e-5f, 1e-6f);
    for (uint32_t pi = 1; pi + 1 < edge->path_count; ++pi) {
        Qm3Vec3 point = env->mesh.edge_path_points.points[edge->path_offset + pi];
        int base_tri = -1;
        for (uint32_t si = 0; si < edge->segment_count; ++si) {
            const Qm3PathSegment* segment = &env->mesh.edge_path_segments.data[edge->segment_offset + si];
            if (qm3_distance(point, segment->a) <= tol || qm3_distance(point, segment->b) <= tol) {
                base_tri = segment->tri;
                break;
            }
        }
        if (base_tri < 0) continue;

        Qm3IntArray support = {0};
        qm3_point_support_tris(&env->surface, point, base_tri, &support);
        for (int i = 0; i < support.size; ++i) {
            int tri = support.data[i];
            Qm3Vec2 p;
            if (!qm3_point_to_base_tri2d(&env->surface, &env->continuous_ctx, tri, point, &p)) continue;
            qm3_graph_tri_vertex_push(&env->graph_tri_vertices[tri], (Qm3GraphTriVertex){
                .graph_vertex = UINT32_MAX,
                .incident_edge = ei,
                .p = p,
            });
            env->graph_tri_vertex_count++;
        }
        qm3_int_free(&support);
    }
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
        qm3_insert_graph_tri_path_points(env, ei);
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

    if (target_dist < 0) return true;
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
        radius = qm3_candidate_radius_max(env);
        env->candidate_radius = radius;
    }
    float min_radius = fminf(qm3_candidate_radius_min(env), radius);

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
        float distance = INFINITY;
        qm3_best_sample_prop_node(&env->surface, &env->prop_graph, &env->fmm, sample_id, &distance);
        if (distance < min_radius) continue;
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
        qm3_mark_edge_frontier_dirty(env, edge_idx);
        qm3_insert_graph_tri_edge_segments(env, edge_idx);
        qm3_insert_graph_tri_path_points(env, edge_idx);
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
                qm3_mark_face_frontier_dirty(env, face);
                qm3_add_reward(env, 0.5f * qm3_compute_face_reward(env, face));
                qm3_build_face_loop_segments(&env->mesh, face, &loop_segments);
                env->loop_stats = qm3_remove_samples_inside_loop(env, &loop_segments, qm3_face_interior_side(&env->mesh, face));
                qm3_prune_graph_after_loop_removal(env);
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
                qm3_mark_face_frontier_dirty(env, face);
                qm3_add_reward(env, qm3_compute_face_reward(env, face));
                qm3_build_face_loop_segments(&env->mesh, face, &loop_segments);
                env->loop_stats = qm3_remove_samples_inside_loop(env, &loop_segments, qm3_face_interior_side(&env->mesh, face));
                qm3_prune_graph_after_loop_removal(env);
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
    if (env->max_frontier <= 0) env->max_frontier = 4096;
    if (env->max_candidates <= 0) env->max_candidates = 768;
    if (env->max_degree <= 0) env->max_degree = 8;
    if (env->candidate_radius_min_ratio <= 0.0f) env->candidate_radius_min_ratio = 0.5f;
    if (env->candidate_radius_max_ratio <= 0.0f) env->candidate_radius_max_ratio = 2.0f;
    if (env->geodesic_steiner_spacing_ratio <= 0.0f) env->geodesic_steiner_spacing_ratio = 0.005f;
    if (env->export_obj_path == NULL) env->export_obj_path = "quad_meshing_3d.obj";
    qm3_surface_init(&env->surface);
    qm3_mesh_init(&env->mesh, (uint32_t)env->max_degree);
    env->loaded_shape = -1;
    env->source_frontier_idx = -1;
    env->target_candidate_idx = -1;
    env->debug_candidate_idx = -1;
    env->phase = QM3_PHASE_SOURCE;
    env->last_action_result = QM3_ACTION_NONE;
    env->geodesic_ready = false;
}

static void qm3_update_episode_max_length(QuadMeshing3DEnv* env) {
    const float ideal_n_quads = env->surface.info.total_area / env->target_quad_area;
    const float ideal_episode_length = 2 * ideal_n_quads - env->mesh.frontier_count;
    env->episode_max_length = (int)ceilf(ideal_episode_length * env->episode_max_length_ratio);
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
    env->target_edge_length = avg * ratio;
    env->target_quad_area = env->target_edge_length * env->target_edge_length;
}

static void qm3_add_log(QuadMeshing3DEnv* env) {
    const float area_scale = env->target_quad_area / env->surface.info.total_area;
    env->log.perf += env->episode_return * area_scale;
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->episode_length;
    env->log.episode_length_ratio += (float)env->episode_length / env->episode_max_length;
    env->log.num_quads += (float)env->mesh.quad_count;
    env->log.num_quads_ratio += (float)env->mesh.quad_count * area_scale;
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

static float qm3_cross_field_alignment_from_dir_and_tri(const QuadMeshing3DEnv* env, Qm3Vec3 unit_dir, int tri) {
    if (tri < 0 || tri >= (int)env->surface.triangle_count) return 0.0f;
    float du = qm3_dot(unit_dir, qm3_normalize(env->surface.face_dir_u[tri]));
    float dv = qm3_dot(unit_dir, qm3_normalize(env->surface.face_dir_v[tri]));
    float s = fmaxf(du * du, dv * dv);
    return fminf(fmaxf(2.0f * s - 1.0f, 0.0f), 1.0f);
}

static float qm3_compute_face_cross_field_alignment(const QuadMeshing3DEnv* env, const Qm3MeshFace* face) {
    float alignment = 0.0f;
    for (uint32_t i = 0; i < 4; ++i) {
        const Qm3MeshVertex* a = &env->mesh.vertices[face->vertices[i]];
        const Qm3MeshVertex* b = &env->mesh.vertices[face->vertices[(i + 1) % 4]];
        Qm3Vec3 dir = qm3_sub(b->pos, a->pos);
        float len = qm3_len(dir);
        if (len > 1e-12f) dir = qm3_scale(dir, 1.0f / len);
        alignment += qm3_cross_field_alignment_from_dir_and_tri(env, dir, a->surface_tri);
        alignment += qm3_cross_field_alignment_from_dir_and_tri(env, dir, b->surface_tri);
    }
    return 0.125f * alignment;
}

static float qm3_compute_face_reward(QuadMeshing3DEnv* env, Qm3MeshFace* face) {
    if (face->n == 3) return env->reward_triangle;
    float q = qm3_compute_face_quality(env, face);
    if (env->reward_cross_field) q *= qm3_compute_face_cross_field_alignment(env, face);
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

typedef struct {
    Qm3Vec3 u;
    Qm3Vec3 v;
} Qm3CrossFieldQuery;

static Qm3CrossFieldQuery qm3_cross_field_query(const Qm3Surface* surface, int tri) {
    if (tri < 0 || tri >= (int)surface->triangle_count) {
        return (Qm3CrossFieldQuery){{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
    }
    return (Qm3CrossFieldQuery){
        qm3_normalize(surface->face_dir_u[tri]),
        qm3_normalize(surface->face_dir_v[tri]),
    };
}

static float qm3_cross_field_alignment(const Qm3CrossFieldQuery* field, Qm3Vec3 unit_dir) {
    float du = qm3_dot(unit_dir, field->u);
    float dv = qm3_dot(unit_dir, field->v);
    float s = fmaxf(du * du, dv * dv);
    return fminf(fmaxf(2.0f * s - 1.0f, 0.0f), 1.0f);
}

static int qm3_candidate_target_tri(const QuadMeshing3DEnv* env, const Qm3TargetCandidate* candidate) {
    if (candidate->kind == QM3_TARGET_EXISTING_FRONTIER && candidate->graph_vertex < env->mesh.vertex_count) {
        return env->mesh.vertices[candidate->graph_vertex].surface_tri;
    }
    if (candidate->sample_id < env->surface.sample_count) return (int)env->surface.samples[candidate->sample_id].tri;
    return -1;
}

static float qm3_target_cross_field_alignment(
    const QuadMeshing3DEnv* env,
    const Qm3CrossFieldQuery* source_field,
    Qm3Vec3 source_pos,
    const Qm3TargetCandidate* candidate
) {
    Qm3Vec3 dir = qm3_sub(qm3_candidate_target_pos3(env, candidate), source_pos);
    float len = qm3_len(dir);
    if (len <= 1e-12f) return 0.0f;
    dir = qm3_scale(dir, 1.0f / len);
    Qm3CrossFieldQuery target_field = qm3_cross_field_query(&env->surface, qm3_candidate_target_tri(env, candidate));
    return 0.5f * (
        qm3_cross_field_alignment(source_field, dir) +
        qm3_cross_field_alignment(&target_field, dir)
    );
}

static uint16_t qm3_u16_or_max(uint32_t value) {
    return value < UINT16_MAX ? (uint16_t)value : UINT16_MAX;
}

static uint32_t qm3_valid_target_count_capped(const QuadMeshing3DEnv* env) {
    uint32_t count = 0;
    uint32_t cap = env->max_candidates;
    for (uint32_t i = 0; i < env->candidate_count && count < cap; ++i) {
        if (env->candidates[i].path_ok) count++;
    }
    return count;
}

static int qm3_rand_range(unsigned int* rng, int max_exclusive) {
    return rand_r(rng) % max_exclusive;
}

static void qm3_compute_observations(QuadMeshing3DEnv* env) {
    if (!env->observations) return;
    uint32_t max_frontier = env->max_frontier;
    uint32_t max_degree = env->max_degree;
    uint32_t max_targets = env->max_candidates;
    QM3_ASSERT(env->mesh.frontier_count <= max_frontier);
    QM3_ASSERT(env->mesh.max_degree <= max_degree);

    Qm3SerialObsBuffer obs = {.sb = {.data = env->observations, .pos = 0}};
    SerialBuffer* sb = &obs.sb;

    serialize_u8(sb, (uint8_t)env->phase);
    serialize_float(sb, env->target_edge_length);
    serialize_u16(sb, qm3_u16_or_max(env->mesh.frontier_count));
    serialize_u16(sb, qm3_u16_or_max(env->mesh.max_degree));

    for (uint32_t i = 0; i < max_frontier; ++i) {
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

    for (uint32_t i = 0; i < max_frontier; ++i) {
        uint32_t written = 0;
        if (i < env->mesh.frontier_count) {
            uint32_t vidx = env->mesh.frontier[i];
            const Qm3MeshVertex* v = &env->mesh.vertices[vidx];
            for (uint32_t j = 0; j < v->degree && written < max_degree; ++j) {
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
        for (; written < max_degree; ++written) serialize_u16(sb, UINT16_MAX);
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
    QM3_ASSERT(target_count <= max_targets);
    serialize_u16(sb, qm3_u16_or_max(target_count));

    int* frontier_hop_distance = NULL;
    int* frontier_hop_queue = NULL;
    if (env->source_frontier_idx >= 0 && (uint32_t)env->source_frontier_idx < env->mesh.frontier_count) {
        frontier_hop_distance = (int*)malloc((size_t)env->mesh.frontier_count * sizeof(int));
        frontier_hop_queue = (int*)malloc((size_t)env->mesh.frontier_count * sizeof(int));
        QM3_ASSERT(frontier_hop_distance != NULL && frontier_hop_queue != NULL);
        qm3_compute_frontier_hop_distances(env, env->source_frontier_idx, frontier_hop_distance, frontier_hop_queue);
    }

    uint32_t written = 0;
    uint32_t source_vidx = env->mesh.frontier[env->source_frontier_idx];
    Qm3Vec3 source_pos = env->mesh.vertices[source_vidx].pos;
    Qm3CrossFieldQuery source_field = qm3_cross_field_query(
        &env->surface,
        env->mesh.vertices[source_vidx].surface_tri
    );
    for (uint32_t i = 0; i < env->candidate_count && written < target_count; ++i) {
        const Qm3TargetCandidate* candidate = &env->candidates[i];
        if (!candidate->path_ok) continue;
        qm3_serialize_vec3(sb, qm3_candidate_target_pos3(env, candidate));
        qm3_serialize_vec3(sb, qm3_candidate_target_normal(env, candidate));
        serialize_float(sb, candidate->path_length);
        serialize_u8(sb, (uint8_t)candidate->kind);
        uint8_t parity = 1;
        if (candidate->kind == QM3_TARGET_EXISTING_FRONTIER && candidate->graph_vertex < env->mesh.vertex_count) {
            int32_t fidx = env->mesh.vertices[candidate->graph_vertex].frontier_index;
            parity = (frontier_hop_distance && fidx >= 0 && (uint32_t)fidx < env->mesh.frontier_count)
                ? (uint8_t)(frontier_hop_distance[fidx] >= 0 ? (frontier_hop_distance[fidx] & 1) : 0)
                : 0;
        }
        serialize_u8(sb, parity);
        serialize_float(sb, qm3_target_cross_field_alignment(env, &source_field, source_pos, candidate));
        written++;
    }
    free(frontier_hop_distance);
    free(frontier_hop_queue);
}

static void qm3_update_vertex_frontier_cache(QuadMeshing3DEnv* env, uint32_t vidx) {
    if (vidx >= env->mesh.vertex_count) return;
    Qm3Mesh* mesh = &env->mesh;
    Qm3MeshVertex* vertex = &mesh->vertices[vidx];
    float edge_length_cost = 0.0f;
    float alignment_cost = 0.0f;
    for (uint32_t i = 0; i < vertex->degree; ++i) {
        size_t nidx = (size_t)vidx * mesh->max_degree + i;
        int32_t eidx = mesh->neighbor_edges[nidx];
        if (eidx < 0) continue;
        Qm3MeshEdge* edge = &mesh->edges[eidx];
        if (edge->disabled || edge->face_count >= 2) continue;
        float chord = qm3_edge_chord_length(mesh, edge);
        float path = edge->path_length > 0.0f ? edge->path_length : chord;
        edge_length_cost += qm3_open_edge_length_cost_from_chord(chord, env->target_edge_length);
        edge_length_cost += qm3_path_chord_cost(path, chord, env->target_edge_length);
        alignment_cost += qm3_open_edge_alignment_cost(env, edge);
    }
    vertex->frontier_edge_length_cost = edge_length_cost;
    vertex->frontier_alignment_cost = alignment_cost;
    vertex->frontier_angle_cost = qm3_vertex_open_angle_cost(env, vidx);
    vertex->frontier_quality_dirty = false;
}

static const Qm3MeshVertex* qm3_get_vertex_frontier_cache(QuadMeshing3DEnv* env, uint32_t vidx) {
    if (vidx >= env->mesh.vertex_count) return NULL;
    Qm3MeshVertex* vertex = &env->mesh.vertices[vidx];
    if (vertex->frontier_quality_dirty) qm3_update_vertex_frontier_cache(env, vidx);
    return vertex;
}

static float qm3_compute_frontier_potential(QuadMeshing3DEnv* env) {
    if (env->mesh.frontier_count == 0) return 0.0f;
    float frontier_cost = 0.0f;
    float weight_sum = env->frontier_edge_length_weight + env->frontier_alignment_weight + env->frontier_angle_weight;
    float max_degree = fmaxf((float)env->mesh.max_degree, 1.0f);
    for (uint32_t fi = 0; fi < env->mesh.frontier_count; ++fi) {
        const Qm3MeshVertex* vertex = qm3_get_vertex_frontier_cache(env, env->mesh.frontier[fi]);
        if (!vertex) continue;
        float edge_term = vertex->frontier_edge_length_cost / (2.0f * max_degree);
        float alignment_term = vertex->frontier_alignment_cost / max_degree;
        float angle_term = vertex->frontier_angle_cost;
        if (weight_sum > 0.0f) {
            frontier_cost += (
                env->frontier_edge_length_weight * edge_term
                + env->frontier_alignment_weight * alignment_term
                + env->frontier_angle_weight * angle_term
            ) / weight_sum;
        }
    }
    float max_frontier = fmaxf((float)env->max_frontier, 1.0f);
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
    float radius_ratio = env->candidate_radius_max_ratio > 0.0f ? env->candidate_radius_max_ratio : 2.0f;
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

    bool frontier_over_cap = env->mesh.frontier_count > env->max_frontier;
    bool terminal = env->episode_length >= env->episode_max_length || env->mesh.frontier_count == 0 || frontier_over_cap;
    if (terminal) {
        float reward = env->rewards ? env->rewards[0] : env->last_reward;
        if (env->mesh.frontier_count != 0) reward = env->reward_incomplete;
        env->episode_return += reward;
        qm3_add_log(env);
        qm3_maybe_export_obj(env);
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
        if (env->episode_length >= env->episode_max_length || env->mesh.frontier_count == 0 || env->mesh.frontier_count > env->max_frontier) {
            qm3_add_log(env);
            qm3_maybe_export_obj(env);
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
