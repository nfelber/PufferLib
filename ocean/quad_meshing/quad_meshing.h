#pragma once

#include "geometry.h"
#include "helpers.h"
#include "mesh.h"
#include "serialization.h"

#include "raylib.h"
#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define BENCHMARKING_ENABLED 0
#define BENCHMARKING_PRINT_EVERY 100000ULL
#include "benchmarking.h"

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    float n; // Required as the last field
} Log;

typedef struct {
    float starting_boundary_area;
    IntArray valid_boundary_idx;
    IntArray valid_candidate_idx;
    Vec2Array candidates_local;
} QuadMeshingCache;

// Required that you have some struct for your env
typedef struct {
    int num_agents;
    Log log;                     // Required field
    unsigned char* observations; // Required field. Ensure type matches in .py and .c
    float* actions;              // Required field. Ensure type matches in vecenv.h
    float* rewards;              // Required field
    float* terminals;            // Required field. Ensure type matches in vecenv.h
    unsigned int rng;

    // Env state
    QuadMesh mesh;
    Vec2Array boundary_poly;
    int episode_length;
    float episode_return;
    int source_frontier_idx;

    // Env settings
    int episode_max_length;
    int candidate_rings;
    int candidate_angles;
    float candidate_radius_min;
    float candidate_radius_max;
    float target_quad_area;
    const char** boundary_paths; // Non-owning
    int boundary_count;
    bool boundary_mode;

    // Perf
    int max_degree;
    unsigned int grid_res;
    float grid_cell_size;
    unsigned int grid_cell_cap;
    float intersection_tol;

    // Observation settings
    // ...

    // Action settings
    // ...

    // Reward settings
    float reward_invalid;

    // Rendering
    int render_target_fps;
    int render_width;
    int render_height;
    bool render_show_frontier;
    bool render_show_candidates;
    float render_line_thickness;
    float render_point_radius;
    float render_candidate_radius;
    Camera2D camera;
    float camera_zoom;
    int ui_pending_source;

    // Cache
    QuadMeshingCache cache;
} QuadMeshingEnv;

static void init_candidates(QuadMeshingEnv* env) {
    int rings = env->candidate_rings;
    int angles = env->candidate_angles;
    Vec2Array_reserve(&env->cache.candidates_local, rings*angles);
    for (int r = 0; r < rings; r++) {
        float frac = rings > 1 ? r / (float)(rings-1) : 0.5;
        float radius = env->candidate_radius_min +
            frac * (env->candidate_radius_max - env->candidate_radius_min);
        for (int a = 0; a < angles; a++) {
            float theta = (2.0f * PI * a) / angles;
            float rr = radius;
            Vec2Array_push(&env->cache.candidates_local, (Vec2){
                rr * cosf(theta),
                rr * sinf(theta),
            });
        }
    }
}

/** Allocates env buffers and computes observation layout. */
void quad_meshing_init(QuadMeshingEnv* env)
{
    mesh_init(&env->mesh, env->max_degree, env->grid_res, env->grid_cell_size, env->grid_cell_cap, env->intersection_tol);
    Vec2Array_init(&env->boundary_poly);
    Vec2Array_init(&env->cache.candidates_local);
    IntArray_init(&env->cache.valid_boundary_idx);
    IntArray_init(&env->cache.valid_candidate_idx);
    init_candidates(env);
    env->ui_pending_source = -1;
}

void add_log(QuadMeshingEnv* env) {
    env->log.perf += env->episode_return * env->target_quad_area / env->cache.starting_boundary_area;
    env->log.score += env->episode_return;
    env->log.episode_length += env->episode_length;
    env->log.episode_return += env->episode_return;
    env->log.n++;
}

static int parse_boundary_vertices(const char* json, Vec2Array* out) {
    const char* key = "\"vertices\"";
    const char* start = strstr(json, key);
    QM_ASSERT(start != NULL);
    start = strchr(start, '[');
    QM_ASSERT(start != NULL);

    int depth = 0;
    int value_count = 0;
    double current[2] = {0.0, 0.0};
    for (const char* p = start; *p != '\0'; ++p) {
        if (*p == '[') {
            depth++;
            continue;
        }
        if (*p == ']') {
            depth--;
            if (depth == 0) break;
            continue;
        }
        if (depth < 2) continue;
        char* end = NULL;
        double value = strtod(p, &end);
        if (end != p) {
            current[value_count % 2] = value;
            value_count++;
            p = end - 1;
            if (value_count % 2 == 0) {
                Vec2Array_push(out, (Vec2){(float)current[0], (float)current[1]});
            }
        }
    }
    QM_ASSERT(value_count % 2 == 0);
    int points = value_count / 2;
    return points;
}

static void normalize_boundary(Vec2Array* vertices) {
    float min_x =  FLT_MAX;
    float max_x = -FLT_MAX;
    float min_y =  FLT_MAX;
    float max_y = -FLT_MAX;

    for (int i=0; i<vertices->size; ++i) {
        Vec2 v = vertices->data[i];
        min_x = fmin(min_x, v.x);
        max_x = fmax(max_x, v.x);
        min_y = fmin(min_y, v.y);
        max_y = fmax(max_y, v.y);
    }

    float scale = 1 / fmax(max_x - min_x, max_y - min_y);

    for (int i=0; i<vertices->size; ++i) {
        Vec2 v = vertices->data[i];
        vertices->data[i].x = (v.x - min_x) * scale;
        vertices->data[i].y = (v.y - min_y) * scale;
    }
}

static void load_boundary(QuadMeshingEnv* env) {
    int choice = rand_range(&env->rng, env->boundary_count);
    const char* path = env->boundary_paths[choice];
    char* json = read_file_bytes(path, NULL);
    Vec2Array_resize(&env->boundary_poly, 0);
    parse_boundary_vertices(json, &env->boundary_poly);
    free(json);

    // Normalize in [0, 1] square
    normalize_boundary(&env->boundary_poly);

    // Cache area
    env->cache.starting_boundary_area = polygon_area(env->boundary_poly.data, env->boundary_poly.size);
}

typedef struct {
    SerialBuffer sb;
} SerialObsBuffer;

static void serialize_obs_substep(SerialObsBuffer* obs, uint8_t substep) {
    obs->sb.pos = 0;
    serialize_u8(&obs->sb, substep);
}

uint8_t deserialize_obs_substep(SerialObsBuffer* obs) {
    obs->sb.pos = 0;
    return deserialize_u8(&obs->sb);
}

static void serialize_obs_frontier(SerialObsBuffer* obs, const QuadMesh* mesh) {
    SerialBuffer* sb = &obs->sb;
    sb->pos = sizeof(uint8_t);

    // Frontier size
    serialize_u16(sb, mesh->frontier.size);
    serialize_u16(sb, mesh->max_degree);

    // Frontier vertices
    for (int i=0; i<mesh->frontier.size; ++i) {
        int vidx = mesh->frontier.data[i];
        Vec2 v = mesh->vertices.data[vidx].pos;
        serialize_float(sb, v.x);
        serialize_float(sb, v.y);
    }

    ByteArray face_incidence_buf;
    ByteArray_init(&face_incidence_buf);
    ByteArray_reserve(&face_incidence_buf, mesh->frontier.size * mesh->max_degree);

    // Frontier neighbors
    for (int i=0; i<mesh->frontier.size; ++i) {
        int vidx = mesh->frontier.data[i];
        int d = mesh->vertices.data[vidx].degree;
        int n_count = 0;
        for (int j=0; j<d; ++j) {
            int nidx = mesh_neighbor_idx(mesh, vidx, j);
            int nvidx = mesh->neighbors.data[nidx];
            int nfidx = mesh->vertices.data[nvidx].frontier_index;
            int eidx = mesh->neighbor_edges.data[nidx];
            int face_count = mesh->edges.data[eidx].face_count;
            if (nfidx < 0 || face_count == 2) continue;
            serialize_u16(sb, nfidx);

            // Serialize edge face incidence bitfield [incident ccw | incident cw]
            QM_ASSERT(face_count < 2);
            uint8_t face_incidence = 0b00;
            if (face_count == 1) {
                face_incidence = mesh_edge_face_orientation_from_vertex(mesh, eidx, vidx) ? 0b01 : 0b10;
            }
            ByteArray_push(&face_incidence_buf, face_incidence);
            ++n_count;
        }

        for (int j=0; j<mesh->max_degree - n_count; ++j) {
            serialize_u16(sb, UINT16_MAX);
            ByteArray_push(&face_incidence_buf, 0);
        }
    }

    for (int i=0; i<face_incidence_buf.size; ++i) {
        serialize_u8(sb, face_incidence_buf.data[i]);
    }
}

void deserialize_obs_frontier(SerialObsBuffer* obs, QuadMesh* mesh) {
    SerialBuffer* sb = &obs->sb;
    sb->pos = sizeof(uint8_t);

    // Frontier size
    uint16_t frontier_size = deserialize_u16(sb);
    mesh->max_degree = deserialize_u16(sb);

    // Frontier vertices
    for (int i=0; i<frontier_size; ++i) {
        float x = deserialize_float(sb);
        float y = deserialize_float(sb);
        int v = mesh_add_vertex(mesh, (Vec2){x, y});
        mesh_update_frontier_vertex(mesh, v);
    }

    // IntArray added_edges;
    // IntArray_init(&added_edges);

    const size_t face_incidence_offset = sizeof(uint16_t) * frontier_size * mesh->max_degree;

    // Frontier neighbors
    for (int i=0; i<frontier_size; ++i) {
        for (int j=0; j<mesh->max_degree; ++j) {
            size_t fi_pos = sb->pos + face_incidence_offset;
            uint16_t nidx = deserialize_u16(sb);
            size_t n_pos = sb->pos;
            sb->pos = fi_pos;
            uint8_t face_incidence = deserialize_u8(sb);
            sb->pos = n_pos;
            if (nidx == UINT16_MAX) continue;
            int eidx = mesh_add_edge(mesh, i, nidx);

            // Recover edge orientation from face incidence
            if (face_incidence != 0b11) {
                MeshEdge* e = &mesh->edges.data[eidx];
                e->face_orientation_cw = (i == e->a) == (face_incidence == 0b01);
                e->face_count = 1;
            }
        }
    }
}

static size_t obs_frontier_bytes(const QuadMesh* mesh) {
    return 2*sizeof(uint16_t) + mesh->frontier.size * (2*sizeof(float) + mesh->max_degree * (sizeof(uint16_t) + sizeof(uint8_t)));
}

static void serialize_obs_source(SerialObsBuffer* obs, const QuadMesh* mesh, uint16_t source) {
    obs->sb.pos = sizeof(uint8_t) + obs_frontier_bytes(mesh);
    serialize_u16(&obs->sb, source);
}

uint16_t deserialize_obs_source(SerialObsBuffer* obs, const QuadMesh* mesh, uint16_t source) {
    obs->sb.pos = sizeof(uint8_t) + obs_frontier_bytes(mesh);
    return deserialize_u16(&obs->sb);
}

static void serialize_obs_validity_mask(SerialObsBuffer* obs, QuadMeshingEnv* env, int source, bool boundary_mode) {
    SerialBuffer* sb = &obs->sb;
    sb->pos = sizeof(uint8_t) + obs_frontier_bytes(&env->mesh) + sizeof(uint16_t);

    // Empty valid index cache
    IntArray_resize(&env->cache.valid_boundary_idx, 0);

    int l3 = 0;
    int r3 = 0;
    if (boundary_mode) {
        l3 = mesh_ring_frontier_neighbor(&env->mesh, source, -3);
        r3 = mesh_ring_frontier_neighbor(&env->mesh, source,  3);
    }

    // Frontier validity mask
    BENCH_START(obs_validity_loop, "quad_meshing.obs_validity_loop");
    for (int i=0; i<env->mesh.frontier.size; ++i) {
        int target = env->mesh.frontier.data[i];
        uint8_t valid = !boundary_mode || target == l3 || target == r3;
        valid = valid && mesh_validate_existing_target(&env->mesh, source, target, false) == MESH_VALID_OK;
        if (valid) IntArray_push(&env->cache.valid_boundary_idx, i);
        serialize_u8(sb, valid);
    }
    BENCH_END(obs_validity_loop);
}

void deserialize_obs_validity_mask(SerialObsBuffer* obs, QuadMesh* mesh, BoolArray* mask) {
    SerialBuffer* sb = &obs->sb;
    sb->pos = sizeof(uint8_t) + obs_frontier_bytes(mesh) + sizeof(uint16_t);

    // Frontier validity mask
    BoolArray_reserve(mask, mesh->frontier.size);
    for (int i=0; i<mesh->frontier.size; ++i) {
        BoolArray_push(mask, deserialize_u8(sb));
    }
}

static size_t obs_validity_bytes(QuadMesh* mesh) {
    return mesh->frontier.size * sizeof(uint8_t);
}

static void serialize_obs_new_candidates(SerialObsBuffer* obs, QuadMeshingEnv* env, int source) {
    SerialBuffer* sb = &obs->sb;
    sb->pos = sizeof(uint8_t) + obs_frontier_bytes(&env->mesh) + sizeof(uint16_t) + obs_validity_bytes(&env->mesh);

    // Empty valid index cache
    IntArray_resize(&env->cache.valid_candidate_idx, 0);

    // New vertex candidates
    uint16_t valid_count = 0;
    size_t valid_count_pos = sb->pos;
    serialize_u16(sb, 0); // placeholder
    BENCH_START(obs_candidate_loop, "quad_meshing.obs_candidate_loop");
    for (int i=0; i<env->candidate_angles * env->candidate_rings; ++i) {
        Vec2 target_pos = add2(env->mesh.vertices.data[source].pos, env->cache.candidates_local.data[i]);
        if (mesh_validate_candidate_target(&env->mesh, source, target_pos, env->boundary_mode) != MESH_VALID_OK) continue;
        IntArray_push(&env->cache.valid_candidate_idx, i);
        serialize_float(sb, target_pos.x);
        serialize_float(sb, target_pos.y);
        ++valid_count;
    }
    BENCH_END(obs_candidate_loop);
    // Rewind to write the actual valid count
    sb->pos = valid_count_pos;
    serialize_u16(sb, valid_count);
}

void deserialize_obs_new_candidates(SerialObsBuffer* obs, QuadMesh* mesh, Vec2Array* candidates) {
    SerialBuffer* sb = &obs->sb;
    sb->pos = sizeof(uint8_t) + obs_frontier_bytes(mesh) + sizeof(uint16_t) + obs_validity_bytes(mesh);

    int valid_count = deserialize_u16(sb);

    // New vertex candidates
    Vec2Array_reserve(candidates, valid_count);
    for (int i=0; i<valid_count; ++i) {
        float x = deserialize_float(sb);
        float y = deserialize_float(sb);
        Vec2Array_push(candidates,(Vec2){x, y});
    }
}

static void compute_observations(QuadMeshingEnv* env) {
    QM_ASSERT(env->mesh.frontier.size > 0);

    int source_slot = env->source_frontier_idx;
    unsigned char substep = source_slot == -1 ? 0 : 1;
    int source = env->mesh.frontier.data[source_slot];

    SerialObsBuffer obs = {
        .sb = env->observations,
    };

    // Substep
    BENCH_START(obs_substep, "quad_meshing.obs_substep");
    serialize_obs_substep(&obs, substep);
    BENCH_END(obs_substep);

    // Assume substep 0 obs remains valid for substep 1
    if (substep == 0) {
        BENCH_START(obs_frontier, "quad_meshing.obs_frontier");
        serialize_obs_frontier(&obs, &env->mesh);
        BENCH_END(obs_frontier);
    } else {
        BENCH_START(obs_source, "quad_meshing.obs_source");
        serialize_obs_source(&obs, &env->mesh, (uint16_t)source_slot);
        BENCH_END(obs_source);
        BENCH_START(obs_validity_mask, "quad_meshing.obs_validity_mask");
        serialize_obs_validity_mask(&obs, env, source, env->boundary_mode);
        BENCH_END(obs_validity_mask);
        BENCH_START(obs_new_candidates, "quad_meshing.obs_new_candidates");
        serialize_obs_new_candidates(&obs, env, source);
        BENCH_END(obs_new_candidates);
    }
}

float compute_vertex_frontier_quality(QuadMeshingEnv* env, int vidx) {
    float max_cost = 0;
    const MeshVertex* v = &env->mesh.vertices.data[vidx];
    const Vec2 vp = env->mesh.vertices.data[vidx].pos;
    for (int i=0; i<v->degree; ++i) {
        const int nidx = mesh_neighbor_idx(&env->mesh, vidx, i);
        const int eidx = env->mesh.neighbor_edges.data[nidx];
        const MeshEdge* ei = &env->mesh.edges.data[eidx];

        const bool cw_face = mesh_edge_face_orientation_from_vertex(&env->mesh, eidx, vidx);
        if (ei->face_count == 2 || (ei->face_count == 1 && !cw_face)) continue;

        const int nvidx = env->mesh.neighbors.data[nidx];
        const Vec2 np = env->mesh.vertices.data[nvidx].pos;
        const Vec2 eivec = sub2(np, vp);

        float angle = FLT_MAX;
        for (int j=0; j<v->degree; ++j) {
            if (i==j) continue;

            const int nidx = mesh_neighbor_idx(&env->mesh, vidx, j);
            const int eidx = env->mesh.neighbor_edges.data[nidx];
            const MeshEdge* ej = &env->mesh.edges.data[eidx];

            const bool cw_face = mesh_edge_face_orientation_from_vertex(&env->mesh, eidx, vidx);
            if (ej->face_count == 2 || (ej->face_count == 1 && cw_face)) continue;

            const int nvidx = env->mesh.neighbors.data[nidx];
            const Vec2 np = env->mesh.vertices.data[nvidx].pos;
            const Vec2 ejvec = sub2(np, vp);

            float ang = atan2f(cross2(eivec, ejvec), dot2(eivec, ejvec));
            if (ang <= 0.0f) ang += 2.0f * M_PI;
            angle = fmin(angle, ang);
        }

        max_cost = fmax(
            max_cost,
            fabs(0.75 * M_PI - fabs(1.25 * M_PI - fabs(angle - 1.25 * M_PI) - 0.75 * M_PI) - 0.5 * M_PI)
        );
    }

    return 1.0 - max_cost / (0.5 * M_PI);
}

float compute_quad_quality(QuadMeshingEnv* env, const Vec2* quad) {
    // Element quality
    float Emin2 = FLT_MAX;
    float angleMin = FLT_MAX;
    float angleMax = FLT_MIN;
    bool isCCW = polygon_is_ccw(quad, 4);
    for (int i=3, j=0; j<4; i=j, ++j) {
        Vec2 edge = sub2(quad[j], quad[i]);
        Emin2 = fmin(Emin2, sqrd_norm2(edge));
        const float angle = polygonInteriorAngle(quad, 4, j, isCCW);
        angleMin = fmin(angleMin, angle);
        angleMax = fmax(angleMax, angle);
    }
    const float Emin = sqrtf(Emin2);
    const float Dmax = sqrtf(fmax(
      sqrd_norm2(sub2(quad[0], quad[2])),
      sqrd_norm2(sub2(quad[1], quad[3]))
    ));
    const float eq = sqrtf(sqrtf(2) * Emin * angleMin / (Dmax * angleMax));

    // Density quality
    const float A = polygon_area(quad, 4);

    return (1.0 - fabs(A / env->target_quad_area - 1.0)) * eq;
}

/** Resets the environment state and observation buffers. */
void c_reset(QuadMeshingEnv* env) {
    env->episode_length = 0;
    env->episode_return = 0.0;
    env->source_frontier_idx = -1;

    // Load new boundary
    load_boundary(env);

    // Reset mesh & init new frontier
    mesh_reset(&env->mesh);
    int boundary_ccw = polygon_is_ccw(env->boundary_poly.data, env->boundary_poly.size);
    for (int i = 0; i < env->boundary_poly.size; ++i) {
        mesh_add_vertex(&env->mesh, env->boundary_poly.data[i]);
    }
    for (int i = 0; i < env->boundary_poly.size; ++i) {
        int a = i;
        int b = (i + 1) % env->boundary_poly.size;
        int eidx = mesh_add_edge(&env->mesh, a, b);
        mesh_set_boundary_edge_face(&env->mesh, eidx, a, b, boundary_ccw);
    }

    compute_observations(env);
}

// void c_substep(QuadMeshingEnv* env, int substep) {
//     QM_ASSERT(substep < 1);
//     int source_slot = (int)env->actions[0];
//     compute_observations(env, 1, source_slot);
// }

/** Advances one environment step using the current action buffer. */
void c_step(QuadMeshingEnv* env) {
    BENCH_START(total, "quad_meshing.total");
    env->rewards[0] = 0.0;
    env->terminals[0] = 0;

    if (env->source_frontier_idx == -1) {
        BENCH_START(source_substep_obs, "quad_meshing.source_substep_obs");
        env->source_frontier_idx = (int)env->actions[0];
        QM_ASSERT(env->source_frontier_idx >= 0 && env->source_frontier_idx < env->mesh.frontier.size);
        compute_observations(env);
        BENCH_END(source_substep_obs);
        BENCH_END(total);
        BENCH_MAYBE_PRINT(total, "quad_meshing c_step");
        return;
    }

    env->episode_length++;

    BENCH_START(target_select_validate, "quad_meshing.target_select_validate");
    int target_slot = (int)env->actions[0];
    size_t valid_boundary_size = env->cache.valid_boundary_idx.size;
    if (target_slot < 0 || target_slot >= valid_boundary_size + env->cache.valid_candidate_idx.size) {
        target_slot = 0;
    }

    int source = env->mesh.frontier.data[env->source_frontier_idx];
    QM_ASSERT(source < env->mesh.vertices.size);
    env->source_frontier_idx = -1;

    int valid = 0;
    int target_vertex = -1;
    bool existing_target = target_slot < valid_boundary_size;

    if (existing_target) {
        int bidx = env->cache.valid_boundary_idx.data[target_slot];
        QM_ASSERT(bidx < env->mesh.frontier.size);
        target_vertex = env->mesh.frontier.data[bidx];
        valid = mesh_validate_existing_target(&env->mesh, source, target_vertex, env->boundary_mode) == MESH_VALID_OK;
    } else {
        int cidx = env->cache.valid_candidate_idx.data[target_slot - valid_boundary_size];
        QM_ASSERT(cidx < env->cache.candidates_local.size);
        Vec2 target_pos = add2(env->mesh.vertices.data[source].pos, env->cache.candidates_local.data[cidx]);
        valid = mesh_validate_candidate_target(&env->mesh, source, target_pos, env->boundary_mode) == MESH_VALID_OK;

        // Add vertex
        if (valid) target_vertex = mesh_add_vertex(&env->mesh, target_pos);
    }
    BENCH_END(target_select_validate);

    if (!valid) {
        BENCH_START(invalid_path, "quad_meshing.invalid_path");
        env->rewards[0] += env->reward_invalid;
        env->episode_return += env->reward_invalid;

        // Check episode termination
        if (env->episode_length >= env->episode_max_length) {
            env->terminals[0] = 1;
            add_log(env);
            c_reset(env);
        } else {
            compute_observations(env); // Reset to substep 0
        }
        BENCH_END(invalid_path);
        BENCH_END(total);
        BENCH_MAYBE_PRINT(total, "quad_meshing c_step");
        return;
    }

    // Add new edge
    if (env->boundary_mode && !existing_target) {
        BENCH_START(boundary_face_path, "quad_meshing.boundary_face_path");
        int l = mesh_ring_frontier_neighbor(&env->mesh, source, -1);
        int r = mesh_ring_frontier_neighbor(&env->mesh, source,  1);
        mesh_add_edge(&env->mesh, l, target_vertex);
        mesh_add_edge(&env->mesh, r, target_vertex);

        int verts[4] = {l, source, r, target_vertex};
        mesh_register_face(&env->mesh, verts, 4);
        Vec2 face[4];
        for (int j = 0; j < 4; ++j) face[j] = env->mesh.vertices.data[verts[j]].pos;
        const float vfq = fmin(
            fmin(
                compute_vertex_frontier_quality(env, l),
                compute_vertex_frontier_quality(env, r)
            ),
            compute_vertex_frontier_quality(env, target_vertex)
        );
        env->rewards[0] += 0.5 * compute_quad_quality(env, face) * vfq;
        BENCH_END(boundary_face_path);
    } else {
        BENCH_START(add_edge, "quad_meshing.add_edge");
        mesh_add_edge(&env->mesh, source, target_vertex);
        BENCH_END(add_edge);

        // Detect new potential quads/tris
        int cycles[64] = {0};
        BENCH_START(detect_quads, "quad_meshing.detect_quads");
        int cycle_count = mesh_detect_quads(&env->mesh, source, target_vertex, cycles, 16);
        BENCH_END(detect_quads);
        int new_face_count = 0; // make sure at most 2 new faces are added
        Vec2 face[4];
        BENCH_START(register_quads_reward, "quad_meshing.register_quads_reward");
        for (int i = 0; i < cycle_count; ++i) {
            int* verts = &cycles[i * 4];
            if (mesh_register_face(&env->mesh, verts, 4)) {
                for (int j = 0; j < 4; ++j) face[j] = env->mesh.vertices.data[verts[j]].pos;
                const float vfq = fmin(
                    compute_vertex_frontier_quality(env, source),
                    compute_vertex_frontier_quality(env, target_vertex)
                );
                env->rewards[0] += 0.5 * compute_quad_quality(env, face) * vfq;
                ++new_face_count;
            };
        }
        BENCH_END(register_quads_reward);
        BENCH_START(detect_tris, "quad_meshing.detect_tris");
        cycle_count = mesh_detect_triangles(&env->mesh, source, target_vertex, cycles, 16);
        BENCH_END(detect_tris);
        BENCH_START(register_tris_reward, "quad_meshing.register_tris_reward");
        for (int i = 0; i < cycle_count; ++i) {
            int* verts = &cycles[i * 3];
            if (mesh_register_face(&env->mesh, verts, 3)) {
                for (int j = 0; j < 3; ++j) face[j] = env->mesh.vertices.data[verts[j]].pos;
                face[3] = face[2]; // Duplicate last vertex to make (degenerate) quad
                const float vfq = fmin(
                    compute_vertex_frontier_quality(env, source),
                    compute_vertex_frontier_quality(env, target_vertex)
                );
                env->rewards[0] += 0.5 * compute_quad_quality(env, face) * vfq;
                ++new_face_count;
            };
        }
        BENCH_END(register_tris_reward);
        QM_ASSERT(new_face_count <= 2);
    }

    env->episode_return += env->rewards[0];

    // In boundary mode, mesh saturation breaks ring frontier constraint
    bool mesh_saturated = false;
    if (env->boundary_mode) {
        BENCH_START(saturation_check, "quad_meshing.saturation_check");
        for (int i=0; i<env->mesh.vertices.size; ++i) {
            if (env->mesh.vertices.data[i].degree >= env->mesh.max_degree) {
                mesh_saturated = true;
                break;
            }
        }
        BENCH_END(saturation_check);
    }

    // Check episode termination
    if ((env->mesh.frontier.size == 0) || (env->episode_length >= env->episode_max_length) || mesh_saturated) {
        BENCH_START(terminal_reset, "quad_meshing.terminal_reset");
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        BENCH_END(terminal_reset);
        BENCH_END(total);
        BENCH_MAYBE_PRINT(total, "quad_meshing c_step");
        return;
    }

    BENCH_START(final_observations, "quad_meshing.final_observations");
    compute_observations(env);
    BENCH_END(final_observations);
    BENCH_END(total);
    BENCH_MAYBE_PRINT(total, "quad_meshing c_step");
}

/** Renders the current environment state (raylib). */
void c_render(QuadMeshingEnv* env) {
    if (!IsWindowReady()) {
        InitWindow(env->render_width, env->render_height, "PufferLib Quad Meshing");
        SetTargetFPS(env->render_target_fps);
        env->camera.offset = (Vector2){env->render_width / 2.0f, env->render_height / 2.0f};
        env->camera.target = (Vector2){0.5f, -0.5f};
        env->camera.rotation = 0.0f;
        env->camera_zoom = fminf(env->render_width, env->render_height) * 0.9f;
        env->camera.zoom = env->camera_zoom;
    }

    if (IsKeyPressed(KEY_F)) env->render_show_frontier = !env->render_show_frontier;
    if (IsKeyPressed(KEY_C)) env->render_show_candidates = !env->render_show_candidates;

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

    Vector2 mouse = GetMousePosition();
    Vector2 vmouse_world = GetScreenToWorld2D(mouse, env->camera);
    Vec2 mouse_world = {vmouse_world.x, -vmouse_world.y};

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawText(TextFormat("step: %d | return: %f", env->episode_length, env->episode_return), 20, 20, 20, RAYWHITE);
    BeginMode2D(env->camera);

    for (int i = 0; i < env->boundary_poly.size; ++i) {
        Vec2 a = env->boundary_poly.data[i];
        Vec2 b = env->boundary_poly.data[(i + 1) % env->boundary_poly.size];
        Vector2 va = {a.x, -a.y};
        Vector2 vb = {b.x, -b.y};
        DrawLineEx(va, vb, line_thickness, (Color){60, 120, 120, 255});
    }

    for (int i = 0; i < env->mesh.edges.size; ++i) {
        MeshEdge e = env->mesh.edges.data[i];
        Vec2 a = env->mesh.vertices.data[e.a].pos;
        Vec2 b = env->mesh.vertices.data[e.b].pos;
        Vector2 va = {a.x, -a.y};
        Vector2 vb = {b.x, -b.y};
        Color c = (Color){220, 80, 80, 255};
        if (e.face_count == 1) c = (Color){255, 200, 80, 255};
        if (e.face_count >= 2) c = (Color){80, 200, 120, 255};
        DrawLineEx(va, vb, line_thickness, c);
    }

    UGridCellIterator it = ugrid_point_query(&env->mesh.edge_grid, mouse_world);
    for (int eidx; (eidx = ugrid_cell_it_next(&it)) != -1;) {
        MeshEdge e = env->mesh.edges.data[eidx];
        Vec2 a = env->mesh.vertices.data[e.a].pos;
        Vec2 b = env->mesh.vertices.data[e.b].pos;
        Vector2 va = {a.x, -a.y};
        Vector2 vb = {b.x, -b.y};
        Color c = (Color){255, 0, 0, 255};
        DrawLineEx(va, vb, line_thickness, c);
    }

    int selecting_target = env->ui_pending_source >= 0;
    int source = env->ui_pending_source >= 0 ? env->mesh.frontier.data[env->ui_pending_source] : -1;

    for (int i = 0; i < env->mesh.vertices.size; ++i) {
        Vector2 p = {env->mesh.vertices.data[i].pos.x, -env->mesh.vertices.data[i].pos.y};
        Color c = (Color){240, 240, 240, 255};
        int fidx = env->mesh.vertices.data[i].frontier_index;
        if (!selecting_target) {
            if (fidx >= 0) c = (Color){0, 140, 255, 255};
        } else if (source >= 0) {
            if (fidx >= 0) {
                MeshValidReason r = mesh_validate_existing_target(&env->mesh, source, i, env->boundary_mode);
                c = (r == MESH_VALID_OK) ? (Color){0, 220, 120, 255} : (Color){220, 80, 80, 255};
            } else {
                c = (Color){80, 80, 80, 255};
            }
        }
        DrawCircleV(p, point_radius, c);
    }

    if (selecting_target && env->render_show_candidates) {
        Vec2 s = env->mesh.vertices.data[source].pos;
        for (int i = 0; i < env->cache.candidates_local.size; ++i) {
            Vec2 cp = add2(s, env->cache.candidates_local.data[i]);
            Vector2 vcp = {cp.x, -cp.y};
            MeshValidReason r = mesh_validate_candidate_target(&env->mesh, source, cp, env->boundary_mode);
            Color c = (r == MESH_VALID_OK) ? (Color){0, 220, 120, 220} : (Color){220, 80, 80, 220};
            DrawCircleV(vcp, candidate_radius, c);
        }
    }

    EndMode2D();

    if (selecting_target && source >= 0) {
        float hover_radius = env->camera.zoom > 0 ? 10.0f / env->camera.zoom : 0.03f;
        float best = 1e9f;
        int best_target = -1;
        bool best_is_candidate = false;
        for (int i = 0; i < env->cache.candidates_local.size; ++i) {
            Vec2 cp = add2(env->mesh.vertices.data[source].pos, env->cache.candidates_local.data[i]);
            float d = sqrd_norm2(sub2(mouse_world, cp));
            if (d < best) {
                best = d;
                best_is_candidate = true;
                best_target = i;
            }
        }
        for (int i = 0; i < env->mesh.frontier.size; ++i) {
            int vidx = env->mesh.frontier.data[i];
            Vec2 p = env->mesh.vertices.data[vidx].pos;
            float d = sqrd_norm2(sub2(mouse_world, p));
            if (d < best) {
                best = d;
                best_is_candidate = false;
                best_target = i;
            }
        }
        if (best_target >= 0 && best <= hover_radius*hover_radius) {
            MeshValidReason r = MESH_VALID_OK;
            Vec2 tp;
            if (best_is_candidate) {
                tp = add2(
                    env->mesh.vertices.data[source].pos,
                    env->cache.candidates_local.data[best_target]
                );
                r = mesh_validate_candidate_target(&env->mesh, source, tp, env->boundary_mode);
            } else {
                int target_vertex = env->mesh.frontier.data[best_target];
                tp = env->mesh.vertices.data[target_vertex].pos;
                r = mesh_validate_existing_target(&env->mesh, source, target_vertex, env->boundary_mode);
            }
            DrawText(TextFormat("Target: %s (%f, %f)", mesh_valid_reason_str(r), tp.x, tp.y), 20, GetScreenHeight() - 40, 20, RAYWHITE);
        }
    }
    EndDrawing();
}

/** Releases any rendering resources. */
void c_close(QuadMeshingEnv* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    mesh_free(&env->mesh);
    Vec2Array_free(&env->boundary_poly);
    Vec2Array_free(&env->cache.candidates_local);
    IntArray_free(&env->cache.valid_boundary_idx);
    IntArray_free(&env->cache.valid_candidate_idx);
}
