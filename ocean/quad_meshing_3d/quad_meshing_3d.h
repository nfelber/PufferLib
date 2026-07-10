#pragma once

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
    uint32_t timing_intersection_tests;
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
    float target_quad_area;
    float episode_max_length_ratio;

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

static void qm3_draw_point_crosses(const Qm3Surface* surface, Camera3D camera, float size, Color color) {
    Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, camera.up));
    Vector3 up = Vector3Normalize(Vector3CrossProduct(right, forward));
    right = Vector3Scale(right, size);
    up = Vector3Scale(up, size);

    rlBegin(RL_LINES);
    rlColor4ub(color.r, color.g, color.b, color.a);
    for (uint32_t i = 0; i < surface->sample_count; ++i) {
        Vector3 p = qm3_v3(surface->samples[i].p);
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
        env->sample_to_graph[v->sample_id] = (int32_t)vi;
    }
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

static void qm3_compute_all_continuous_paths(QuadMeshing3DEnv* env) {
    qm3_path_clear(&env->candidate_path_points);
    qm3_path_segment_clear(&env->candidate_path_segments);
    env->continuous_paths_ok = 0;
    env->valid_candidate_count = 0;
    env->invalid_unreachable_count = 0;
    env->invalid_intersect_count = 0;
    env->timing_path_build_ms = 0.0;
    env->timing_intersection_ms = 0.0;
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

    for (uint32_t i = 0; i < query_count; ++i) {
        uint32_t sample_id = query_samples[i];
        if (sample_id >= env->surface.sample_count) continue;
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
        qm3_candidate_push(env, QM3_TARGET_EXISTING_FRONTIER, target_vidx, target_vertex->sample_id, validity);
    }
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

static void qm3_select_random_target(QuadMeshing3DEnv* env) {
    if (env->candidate_count == 0 || env->valid_candidate_count == 0) {
        qm3_select_target_candidate(env, -1);
        return;
    }
    int32_t start = (int32_t)(rand_r(&env->rng) % env->candidate_count);
    env->target_candidate_idx = qm3_next_valid_candidate(env, start, 1);
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
    }

    env->last_action_result = QM3_ACTION_TARGET_COMMITTED;
    env->phase = QM3_PHASE_SOURCE;
    qm3_clear_target_selection(env);
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

static void qm3_select_random_source(QuadMeshing3DEnv* env) {
    if (env->mesh.frontier_count == 0) {
        qm3_select_source_frontier(env, -1);
        return;
    }
    uint32_t source = (uint32_t)(rand_r(&env->rng) % env->mesh.frontier_count);
    qm3_select_source_frontier(env, (int32_t)source);
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
    float target_area = env->target_quad_area;
    if (target_area <= 1e-12f) {
        float sample_count = env->surface.sample_count > 0 ? (float)env->surface.sample_count : 1.0f;
        target_area = fmaxf(env->surface.info.total_area / sample_count, 1e-12f);
    }
    float ratio = env->episode_max_length_ratio > 0.0f ? env->episode_max_length_ratio : 1.5f;
    env->episode_max_length = (int)ceilf((env->surface.info.total_area / target_area) * ratio);
    if (env->episode_max_length < 1) env->episode_max_length = 1;
}

static void qm3_add_log(QuadMeshing3DEnv* env) {
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->episode_length;
    env->log.n += 1.0f;
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
    qm3_rebuild_sample_to_graph(env);
    qm3_build_geodesic_state(env);
    qm3_update_episode_max_length(env);
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
        qm3_rebuild_sample_to_graph(env);
        if (env->geodesic_ready) qm3_build_graph_tri_segments(env);
        qm3_update_episode_max_length(env);
    }
    env->candidate_radius = 0.0f;
    qm3_clear_target_selection(env);
    env->phase = QM3_PHASE_SOURCE;
    env->last_action_result = QM3_ACTION_NONE;
    if (env->rewards) env->rewards[0] = 0.0f;
    if (env->terminals) env->terminals[0] = 0.0f;
}

void c_step(QuadMeshing3DEnv* env) {
    if (env->rewards) env->rewards[0] = 0.0f;
    if (env->terminals) env->terminals[0] = 0.0f;
    env->episode_length++;

    if (env->phase == QM3_PHASE_SOURCE) {
        uint32_t source_slot = qm3_clamp_action_index(env->actions ? env->actions[0] : 0.0f);
        if (source_slot < env->mesh.frontier_count) qm3_select_source_frontier(env, (int32_t)source_slot);
        else env->last_action_result = QM3_ACTION_INVALID_SOURCE;
    } else {
        uint32_t target_slot = qm3_clamp_action_index(env->actions ? env->actions[0] : 0.0f);
        int32_t candidate_idx = qm3_nth_valid_candidate(env, target_slot);
        if (candidate_idx >= 0) qm3_commit_target_candidate(env, (uint32_t)candidate_idx);
        else env->last_action_result = QM3_ACTION_INVALID_TARGET;
    }

    bool terminal = env->episode_length >= env->episode_max_length || env->mesh.frontier_count == 0;
    if (terminal) {
        float reward = env->rewards ? env->rewards[0] : 0.0f;
        env->episode_return += reward;
        qm3_add_log(env);
        c_reset(env);
        if (env->rewards) env->rewards[0] = reward;
        if (env->terminals) env->terminals[0] = 1.0f;
    } else {
        env->episode_return += env->rewards ? env->rewards[0] : 0.0f;
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
    if (IsKeyPressed(KEY_PERIOD) && env->candidate_count > 0) qm3_select_debug_candidate(env, env->debug_candidate_idx + 1);
    if (IsKeyPressed(KEY_COMMA) && env->candidate_count > 0) qm3_select_debug_candidate(env, env->debug_candidate_idx - 1);
    if (IsKeyPressed(KEY_R) || IsKeyPressed(KEY_SPACE)) qm3_select_random_source(env);
    if (IsKeyPressed(KEY_T)) qm3_select_random_target(env);
    if (IsKeyPressed(KEY_RIGHT) && env->mesh.frontier_count > 0) qm3_select_source_frontier(env, env->source_frontier_idx + 1);
    if (IsKeyPressed(KEY_LEFT) && env->mesh.frontier_count > 0) qm3_select_source_frontier(env, env->source_frontier_idx - 1 + (int32_t)env->mesh.frontier_count);
    if (IsKeyPressed(KEY_D) && env->candidate_count > 0) qm3_select_target_candidate(env, env->target_candidate_idx + 1);
    if (IsKeyPressed(KEY_A) && env->candidate_count > 0) qm3_select_target_candidate(env, env->target_candidate_idx - 1);
    if (IsKeyPressed(KEY_ENTER) && env->phase == QM3_PHASE_TARGET && env->target_candidate_idx >= 0) {
        env->episode_length++;
        qm3_commit_target_candidate(env, (uint32_t)env->target_candidate_idx);
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
    if (env->render_show_vertices) qm3_draw_vertices(surface, env->camera, point_size, (Color){110, 130, 160, 170});
    if (env->render_show_samples) qm3_draw_point_crosses(surface, env->camera, point_size, (Color){130, 145, 155, 135});
    if (env->render_show_normals) qm3_draw_normals(surface, normal_length);
    if (env->render_show_cross_field) qm3_draw_cross_field(surface, cross_length);
    if (env->render_show_graph) {
        float offset = fmaxf(diag * 0.0008f, 1e-5f);
        qm3_draw_graph_edges(&env->mesh, offset);
        qm3_draw_mesh_vertex_crosses(&env->mesh, env->camera, point_size * 1.1f, offset * 1.5f);
    }
    if (env->render_show_graph_paths) qm3_draw_graph_path_segments(&env->mesh, 0.0f, (Color){80, 235, 255, 255});
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

    DrawRectangle(12, 12, 1160, 224, (Color){0, 0, 0, 170});
    DrawText(TextFormat("QMSURF3D | vertices: %u | triangles: %u | samples: %u | frontier_edges: %u | prop nodes: %d",
        surface->vertex_count, surface->triangle_count, surface->sample_count, surface->frontier_edge_count, env->prop_graph.node_count), 24, 24, 18, RAYWHITE);
    DrawText(TextFormat("area: %.6f | sample density: %.2f | sharp dihedral: %.2f deg",
        surface->info.total_area, surface->info.sample_density, surface->info.sharp_dihedral_radians * RAD2DEG), 24, 50, 18, RAYWHITE);
    DrawText(TextFormat("phase: %s | step: %d / %d | return: %.3f | frontier: %u | edges: %u | graph vertices: %u | last: %s",
        qm3_phase_str(env->phase),
        env->episode_length,
        env->episode_max_length,
        env->episode_return,
        env->mesh.frontier_count,
        env->mesh.edge_count,
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
    DrawText(TextFormat("invalid: unreachable %u | intersect %u | debug I:%s | draw: mesh M:%s | opaque O:%s | samples P:%s | vertices V:%s | normals N:%s | cross X:%s | graph G:%s | paths Y:%s | candidates C:%s",
        env->invalid_unreachable_count,
        env->invalid_intersect_count,
        env->render_debug_validity ? "on" : "off",
        env->render_show_mesh ? "on" : "off",
        env->render_mesh_opaque ? "on" : "off",
        env->render_show_samples ? "on" : "off",
        env->render_show_vertices ? "on" : "off",
        env->render_show_normals ? "on" : "off",
        env->render_show_cross_field ? "on" : "off",
        env->render_show_graph ? "on" : "off",
        env->render_show_graph_paths ? "on" : "off",
        env->render_show_candidates ? "on" : "off"), 24, 128, 18, (Color){180, 210, 255, 255});
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
            seg), 24, 154, 18, qm3_candidate_validity_color(reason));
    } else {
        DrawText("debug candidate: off (I toggles validity overlay)", 24, 154, 18, (Color){150, 160, 175, 255});
    }
    DrawText("controls: Ctrl+LMB source | Shift+LMB target | Enter commit | R/Space random source | Left/Right source | T random target | A/D valid target | ,/. debug target | Up/Down radius", 24, 180, 18, (Color){180, 190, 200, 255});

    int footer_y = GetScreenHeight() - 42;
    DrawRectangle(12, footer_y - 8, 920, 34, (Color){0, 0, 0, 170});
    DrawText(TextFormat("timing: targets %.3f ms | geodesic paths %.3f ms | intersections %.3f ms (%u tests) | total %.3f ms",
        env->timing_target_query_ms,
        env->timing_path_build_ms,
        env->timing_intersection_ms,
        env->timing_intersection_tests,
        env->timing_target_query_ms + env->timing_path_build_ms + env->timing_intersection_ms),
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
    qm3_mesh_free(&env->mesh);
    qm3_surface_free(&env->surface);
}
