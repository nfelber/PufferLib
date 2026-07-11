#undef NDEBUG

#include <float.h>
#include <math.h>
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

DEFINE_VECTOR(float, FloatArray)

typedef struct {
    float starting_boundary_area;
    float boundary_area;
    bool boundary_area_dirty;
    bool render_bounds_dirty;
    Vec2 render_min;
    Vec2 render_max;
    Vec2Array sdf_query_world;
    FloatArray sdf_query_values;
} QuadMeshingCache;

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
    float fixed_local_radius;
    int boundary_set_count;
    int boundary_set_index;
    float** boundary_sets;
    int* boundary_set_sizes;
    bool random_active_vertex;
    bool edge_mode;

    // Observations config
    bool observe_remaining_area;
    bool observe_local_radius;
    bool observe_boundary_cost;
    float observation_radius;
    int n_neighbors;
    int n_sdf_samples;

    // Actions config
    float action_radius;
    bool cartesian_actions;

    // Rewards config
    bool delayed_rewards;

    // Rendering
    bool render_enabled;
    int render_target_fps;
    bool export_meshes;
    bool mesh_enabled;
    char export_mesh_path[512];
    int export_mesh_counter;

    // Env state
    Polygon2D boundary;
    Polygon2D quad;
    Mesh2D mesh;
    SizeArray boundary_mesh_vertices;
    int active_vertex;
    int active_edge_start;
    float local_radius;
    int episode_length;
    float episode_return;
    int consecutive_invalid_actions;

    // Cache
    QuadMeshingCache cache;
} QuadMeshing;

static void build_export_mesh_path(QuadMeshing* env, char* out, size_t out_size) {
    const char* path_template = env->export_mesh_path;
    if (!path_template || path_template[0] == '\0') {
        path_template = "mesh.obj";
    }

    const char* token = strstr(path_template, "{episode}");
    if (token) {
        size_t prefix_len = (size_t)(token - path_template);
        const char* suffix = token + strlen("{episode}");
        if (prefix_len + strlen(suffix) + 32 >= out_size) {
            out[0] = '\0';
            return;
        }
        memcpy(out, path_template, prefix_len);
        snprintf(out + prefix_len, out_size - prefix_len, "%d%s", env->export_mesh_counter, suffix);
        return;
    }

    if (env->export_mesh_counter == 0) {
        snprintf(out, out_size, "%s", path_template);
        return;
    }

    const char* slash = strrchr(path_template, '/');
    const char* dot = strrchr(path_template, '.');
    if (dot && (!slash || dot > slash)) {
        int base_len = (int)(dot - path_template);
        snprintf(out, out_size, "%.*s_%d%s", base_len, path_template, env->export_mesh_counter, dot);
    } else {
        snprintf(out, out_size, "%s_%d", path_template, env->export_mesh_counter);
    }
}

static void export_mesh_obj(QuadMeshing* env) {
    if (!env->export_meshes) {
        return;
    }

    char path[1024];
    build_export_mesh_path(env, path, sizeof(path));
    if (path[0] == '\0') {
        fprintf(stderr, "Mesh export path too long\n");
        return;
    }

    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Failed to open mesh export path: %s\n", path);
        return;
    }

    fprintf(f, "# QuadMeshing OBJ export\n");
    for (size_t i = 0; i < env->mesh.vertices.size; i++) {
        Vec2 v = env->mesh.vertices.data[i];
        fprintf(f, "v %.8f %.8f 0\n", v.x, v.y);
    }
    for (size_t i = 0; i < env->mesh.edges.size; i++) {
        Edge e = env->mesh.edges.data[i];
        fprintf(f, "l %zu %zu\n", e.v1 + 1, e.v2 + 1);
    }

    fclose(f);
    env->export_mesh_counter++;
}

static float get_boundary_area(QuadMeshing* env) {
    if (env->cache.boundary_area_dirty) {
        env->cache.boundary_area = polygon2D_area(env->boundary);
        env->cache.boundary_area_dirty = false;
    }
    return env->cache.boundary_area;
}

static void mark_boundary_dirty(QuadMeshing* env) {
    env->cache.boundary_area_dirty = true;
    env->cache.render_bounds_dirty = true;
}

static void update_render_bounds(QuadMeshing* env) {
    if (!env->cache.render_bounds_dirty) {
        return;
    }

    Vec2 min = { FLT_MAX, FLT_MAX };
    Vec2 max = { -FLT_MAX, -FLT_MAX };

    for (int i = 0; i < env->boundary.vertices.size; i++) {
        Vec2 v = env->boundary.vertices.data[i];
        if (v.x < min.x) min.x = v.x;
        if (v.y < min.y) min.y = v.y;
        if (v.x > max.x) max.x = v.x;
        if (v.y > max.y) max.y = v.y;
    }

    for (size_t i = 0; i < env->mesh.vertices.size; i++) {
        Vec2 v = env->mesh.vertices.data[i];
        if (v.x < min.x) min.x = v.x;
        if (v.y < min.y) min.y = v.y;
        if (v.x > max.x) max.x = v.x;
        if (v.y > max.y) max.y = v.y;
    }

    env->cache.render_min = min;
    env->cache.render_max = max;
    env->cache.render_bounds_dirty = false;
}

static void load_starting_boundary(QuadMeshing* env, float* boundary_vertices, int num_vertices) {
    // Initialize starting boundary from provided vertices or default to square
    Vec2Array_resize(&env->starting_boundary.vertices, 0);

    if (boundary_vertices && num_vertices >= 3) {
        // Load from provided vertices
        Vec2Array_reserve(&env->starting_boundary.vertices, num_vertices);
        for (int i = 0; i < num_vertices; ++i) {
            Vec2Array_push(&env->starting_boundary.vertices, 
                          (Vec2){boundary_vertices[2*i], boundary_vertices[2*i+1]});
        }
    } else {
        // Default: square boundary
        const int n = 8;
        const int N = 4*n;
        Vec2Array_reserve(&env->starting_boundary.vertices, N);
        Vec2 dir = {1.0 / n, 0.0};
        Vec2 pos = {0.0, 0.0};
        for (int s=0; s<4; ++s) {
            for (int i=0; i<n; ++i) {
                Vec2Array_push(&env->starting_boundary.vertices, pos);
                pos = add2(pos, dir);
            }
            const float x = dir.x;
            dir.x = -dir.y;
            dir.y = x;
        }
    }
    
    // Normalize boundary so the longest axis has length 1
    if (env->starting_boundary.vertices.size > 0) {
        Vec2 min = { FLT_MAX, FLT_MAX };
        Vec2 max = { -FLT_MAX, -FLT_MAX };
        for (int i = 0; i < env->starting_boundary.vertices.size; ++i) {
            Vec2 v = env->starting_boundary.vertices.data[i];
            if (v.x < min.x) min.x = v.x;
            if (v.y < min.y) min.y = v.y;
            if (v.x > max.x) max.x = v.x;
            if (v.y > max.y) max.y = v.y;
        }
        const float width = max.x - min.x;
        const float height = max.y - min.y;
        const float max_dim = fmaxf(width, height);
        if (max_dim > 0.0f) {
            const float scale = 1.0f / max_dim;
            for (int i = 0; i < env->starting_boundary.vertices.size; ++i) {
                env->starting_boundary.vertices.data[i].x *= scale;
                env->starting_boundary.vertices.data[i].y *= scale;
            }
        }
    }

    env->starting_boundary.isCCW = is_polygon_ccw(env->starting_boundary);    

    env->cache.boundary_area_dirty = true;
    env->cache.render_bounds_dirty = true;
    env->cache.starting_boundary_area = polygon2D_area(env->starting_boundary);
    env->cache.boundary_area = env->cache.starting_boundary_area;

    if (env->starting_boundary.vertices.size == 0) {
        env->target_quad_area = 1.0f;
        env->episode_max_length = 1;
        env->active_vertex = 0;
        return;
    }

    // Calculate target_quad_area as square of average boundary segment length
    float perimeter = 0.0f;
    for (int i = env->starting_boundary.vertices.size - 1, j = 0; j < env->starting_boundary.vertices.size; i = j, ++j) {
        Vec2 v_curr = env->starting_boundary.vertices.data[i];
        Vec2 v_next = env->starting_boundary.vertices.data[j];
        float dx = v_next.x - v_curr.x;
        float dy = v_next.y - v_curr.y;
        perimeter += sqrtf(dx * dx + dy * dy);
    }
    float avg_segment_length = perimeter / env->starting_boundary.vertices.size;
    env->local_radius = env->fixed_local_radius > 0.0f ? env->fixed_local_radius : avg_segment_length;
    env->target_quad_area = avg_segment_length * avg_segment_length;

    // Calculate episode_max_length as 2 * (boundary area) / (target quad area)
    env->episode_max_length = (int)(2.0f * env->cache.starting_boundary_area / env->target_quad_area);

    env->active_vertex = 0;
}

void init(QuadMeshing* env, float* boundary_vertices, int num_vertices) {
    Vec2Array_init(&env->starting_boundary.vertices);
    load_starting_boundary(env, boundary_vertices, num_vertices);

    // Allocate boundary
    Vec2Array_init(&env->boundary.vertices);
    Vec2Array_reserve(&env->boundary.vertices, env->starting_boundary.vertices.size);
    env->boundary.isCCW = env->starting_boundary.isCCW;

    // Allocate quad
    Vec2Array_init(&env->quad.vertices);
    Vec2Array_resize(&env->quad.vertices, 4);
    env->quad.isCCW = env->starting_boundary.isCCW;

    mesh2D_init(&env->mesh);
    SizeArray_init(&env->boundary_mesh_vertices);
    Vec2Array_init(&env->cache.sdf_query_world);
    Vec2Array_resize(&env->cache.sdf_query_world, env->n_sdf_samples);
    FloatArray_init(&env->cache.sdf_query_values);
    FloatArray_resize(&env->cache.sdf_query_values, env->n_sdf_samples);

    env->active_vertex = 0;
    env->mesh_enabled = env->render_enabled || env->export_meshes;
}

void add_log(QuadMeshing* env) {
    // env->log.perf += (env->episode_length < env->episode_max_length) ?
    //     env->episode_return / polygon2D_area(env->starting_boundary) : 0;
    env->log.perf += env->episode_return * env->target_quad_area / env->cache.starting_boundary_area;
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
    if (dot2(bis, bis) < 1e-8) {
        bis = (Vec2){ -e1.y, e1.x };
    }

    bis = safe_normalize(bis);
    float cross = cross2(e1, e2);

    // Build orthonormal frame
    Frame2D f;
    f.origin = v;
    f.x = (env->boundary.isCCW && cross > 0) || (!env->boundary.isCCW && cross < 0) ? scalmul2(bis, -1.0) : bis;
    f.y = (Vec2){ -bis.y, bis.x };

    return f;
}

static int select_active_vertex_index(QuadMeshing* env) {
    if (env->random_active_vertex) {
        // int idx;
        // int ctr = 0;
        // do {
        //     idx = rand() % env->boundary.vertices.size;
        //     if (++ctr == 10) {
        //       break;
        //     }
        // } while (polygonInteriorAngle(env->boundary, idx) > M_PI);
        // return idx;

        int idx = rand() % env->boundary.vertices.size;
        int countdown = env->boundary.vertices.size;
        while (polygonInteriorAngle(env->boundary, env->active_vertex) > M_PI && --countdown != 0) {
            idx = polygon2D_neighbor_index(env->boundary, env->active_vertex, 1);
        }
        return idx;
    }

    int idx = 0;
    float angleMin = polygonInteriorAngle(env->boundary, 0);
    for (int i=1; i<env->boundary.vertices.size; ++i) {
        const float angle = polygonInteriorAngle(env->boundary, i);
        if (angle < angleMin) {
            angleMin = angle;
            idx = i;
        }
    }
    return idx;
}

static Frame2D compute_edge_local_frame(QuadMeshing* env) {
    Vec2 v0 = env->boundary.vertices.data[env->active_edge_start];
    Vec2 v1 = Polygon2D_neighbor(env->boundary, env->active_edge_start, 1);
    Vec2 tangent = safe_normalize(sub2(v1, v0));
    Vec2 inward = env->boundary.isCCW
        ? (Vec2){ -tangent.y, tangent.x }
        : (Vec2){ tangent.y, -tangent.x };
    Frame2D f;
    f.origin = scalmul2(add2(v0, v1), 0.5f);
    f.x = inward;
    f.y = tangent;
    return f;
}

static void write_sdf_observations(QuadMeshing* env, Frame2D frame, int* obs_idx) {
    float r = env->local_radius * env->observation_radius / env->n_sdf_samples;
    for (int i = 1; i < env->n_sdf_samples+1; ++i) {
        Vec2 query_local = { r * i, 0 };
        Vec2 query_world = local_to_world(frame, query_local);
        float d = clampf(eval_polygon2D_sdf(env->boundary, query_world), -1.0, 1.0);
        env->observations[(*obs_idx)++] = d;
        if (env->render_enabled) {
            env->cache.sdf_query_world.data[i - 1] = query_world;
            env->cache.sdf_query_values.data[i - 1] = d;
        }
    }
}

static float compute_boundary_edge_cost(QuadMeshing* env) {
    float cost = 0;
    Vec2 from = env->boundary.vertices.data[0];
    for (int i=0; i<env->boundary.vertices.size; ++i) {
        const Vec2 to = Polygon2D_neighbor(env->boundary, i, 1);
        const Vec2 edge = sub2(to, from);
        cost += fabs(dot2(edge, edge) - env->target_quad_area);
        from = to;
    }
    return cost / env->boundary.vertices.size;
}

static float compute_boundary_angle_cost(QuadMeshing* env) {
    float cost = 0;
    for (int i=0; i<env->boundary.vertices.size; ++i) {
        const float angle = polygonInteriorAngle(env->boundary, i);
        cost += fabs(0.75 * M_PI - fabs(M_PI - fabs(angle - M_PI) - 0.75 * M_PI) - 0.5 * M_PI);
    }
    return cost / (0.5 * M_PI * env->boundary.vertices.size);
}

static void compute_vertex_observations(QuadMeshing* env) {
    int obs_idx = 0;
    if (env->observe_remaining_area) {
        env->observations[obs_idx++] = get_boundary_area(env) / env->cache.starting_boundary_area;
    }

    if (env->observe_local_radius) {
        env->observations[obs_idx++] = env->local_radius;
    }

    if (env->observe_boundary_cost) {
        env->observations[obs_idx++] = compute_boundary_edge_cost(env);
        env->observations[obs_idx++] = compute_boundary_angle_cost(env);
    }

    Frame2D frame = compute_active_local_frame(env);
    for (int i=0; i<env->n_neighbors; ++i) {
        Vec2 ln = world_to_local(frame, Polygon2D_neighbor(env->boundary, env->active_vertex, -i-1));
        Vec2 rn = world_to_local(frame, Polygon2D_neighbor(env->boundary, env->active_vertex,  i+1));
        float langle = atan2f(ln.y, ln.x);
        float rangle = atan2f(rn.y, rn.x);
        float lr = norm2(ln);
        float rr = norm2(rn);
        env->observations[obs_idx++] = langle / M_PI;
        env->observations[obs_idx++] = lr;
        env->observations[obs_idx++] = rangle / M_PI;
        env->observations[obs_idx++] = rr;
    }

    write_sdf_observations(env, frame, &obs_idx);
}

static void compute_edge_observations(QuadMeshing* env) {
    int obs_idx = 0;
    if (env->observe_remaining_area) {
        env->observations[obs_idx++] = get_boundary_area(env) / env->cache.starting_boundary_area;
    }

    if (env->observe_local_radius) {
        env->observations[obs_idx++] = env->local_radius;
    }

    if (env->observe_boundary_cost) {
        env->observations[obs_idx++] = compute_boundary_edge_cost(env);
        env->observations[obs_idx++] = compute_boundary_angle_cost(env);
    }

    env->observations[obs_idx++] = env->local_radius;
    Frame2D frame = compute_edge_local_frame(env);
    for (int i=0; i<env->n_neighbors; ++i) {
        int left_idx = polygon2D_neighbor_index(env->boundary, env->active_edge_start, -i-1);
        int right_idx = polygon2D_neighbor_index(env->boundary, env->active_edge_start, i+2);
        Vec2 ln = world_to_local(frame, env->boundary.vertices.data[left_idx]);
        Vec2 rn = world_to_local(frame, env->boundary.vertices.data[right_idx]);
        env->observations[obs_idx++] = ln.x;
        env->observations[obs_idx++] = ln.y;
        env->observations[obs_idx++] = rn.x;
        env->observations[obs_idx++] = rn.y;
    }

    write_sdf_observations(env, frame, &obs_idx);
}

void compute_observations(QuadMeshing* env) {
    assert(env->boundary.vertices.size > 0);

    if (env->edge_mode) {
        env->active_edge_start = select_active_vertex_index(env);
        Vec2 v0 = env->boundary.vertices.data[env->active_edge_start];
        Vec2 v1 = Polygon2D_neighbor(env->boundary, env->active_edge_start, 1);
        float edge_length = norm2(sub2(v1, v0));
        // env->local_radius = env->fixed_local_radius > 0.0f ? env->fixed_local_radius : edge_length;
        compute_edge_observations(env);
        return;
    }

    env->active_vertex = select_active_vertex_index(env);
    float radius = 0.5 *
      norm2(sub2(Polygon2D_neighbor(env->boundary, env->active_vertex, -1), env->boundary.vertices.data[env->active_vertex])) +
      norm2(sub2(Polygon2D_neighbor(env->boundary, env->active_vertex,  1), env->boundary.vertices.data[env->active_vertex]));
    // env->local_radius = env->fixed_local_radius > 0.0f ? env->fixed_local_radius : radius;
    compute_vertex_observations(env);
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
    const float alpha = 10.0;
    const float A = polygon2D_area(quad);
    const float Ad = A / env->target_quad_area - 1.0;
    const float dq = 1.0 / (1.0 + alpha * Ad*Ad);

    // return A * eq * dq;
    // return fmin(A / env->target_quad_area, 1.0) * eq;
    return (1.0 - fabs(A / env->target_quad_area - 1.0)) * eq;
    // return eq * dq;
    // return 0.5 * (eq + dq);

    // float score = 0.0;
    // float min_alignment = 0.5 * sqrtf(2);
    // float target_length = sqrtf(env->target_quad_area);
    // for (int i=quad.vertices.size-1, j=0; j<quad.vertices.size; i=j, ++j) {
    //     Vec2 edge = sub2(quad.vertices.data[j], quad.vertices.data[i]);
    //     float edge_length = norm2(edge);
    //     Vec2 unit_edge = scalmul2(edge, 1 / edge_length);
    //     float h_alignment = fabs(dot2(unit_edge, (Vec2){1.0, 0.0}));
    //     float v_alignment = fabs(dot2(unit_edge, (Vec2){0.0, 1.0}));
    //     float alignment = (fmax(h_alignment, v_alignment) - min_alignment) / (1.0 - min_alignment);
    //     float length = 1.0 - fabs(edge_length / target_length - 1.0);
    //     score += alignment * length;
    // }
    //
    // return 0.25 * score;
}

static void remove_adjacent_pair(QuadMeshing* env, int start_idx) {
    const int size = env->boundary.vertices.size;
    if (start_idx == size - 1) {
        env->boundary.vertices.data[0] = env->boundary.vertices.data[size - 2];
        env->boundary.vertices.size -= 2;
        env->boundary_mesh_vertices.data[0] = env->boundary_mesh_vertices.data[size - 2];
        env->boundary_mesh_vertices.size -= 2;
    } else {
        Vec2Array_remove_range(&env->boundary.vertices, start_idx, start_idx + 1);
        SizeArray_remove_range(&env->boundary_mesh_vertices, start_idx, start_idx + 1);
    }
    mark_boundary_dirty(env);
}

static bool try_close_quad(
    QuadMeshing* env,
    int idx0,
    int idx1,
    int idx2,
    int idx3,
    int remove_start_idx,
    size_t quad_mesh_indices[4],
    bool* quad_mesh_indices_ready
) {
    env->quad.vertices.data[0] = env->boundary.vertices.data[idx0];
    env->quad.vertices.data[1] = env->boundary.vertices.data[idx1];
    env->quad.vertices.data[2] = env->boundary.vertices.data[idx2];
    env->quad.vertices.data[3] = env->boundary.vertices.data[idx3];
    const Segment2D new_edge = {env->quad.vertices.data[0], env->quad.vertices.data[3]};
    const float quad_area = polygon2D_area(env->quad);
    const float boundary_area = get_boundary_area(env);

    const bool action_valid = !(polygon2D_segment_intersect(env->boundary, new_edge, NULL, 1e-6) ||
                                boundary_area < quad_area ||
                                !(env->boundary.isCCW == is_polygon_ccw(env->quad)));
    if (action_valid) {
        quad_mesh_indices[0] = env->boundary_mesh_vertices.data[idx0];
        quad_mesh_indices[1] = env->boundary_mesh_vertices.data[idx1];
        quad_mesh_indices[2] = env->boundary_mesh_vertices.data[idx2];
        quad_mesh_indices[3] = env->boundary_mesh_vertices.data[idx3];
        *quad_mesh_indices_ready = true;
        remove_adjacent_pair(env, remove_start_idx);
    }
    return action_valid;
}

static Vec2 edge_action_local(QuadMeshing* env, float a, float b) {
    const float action_scale = env->local_radius * env->action_radius;
    if (env->cartesian_actions) {
        return (Vec2){ a * action_scale, b * action_scale };
    }
    const float theta = a * (float)M_PI;
    const float r = action_scale * b;
    return (Vec2){ cosf(theta) * r, sinf(theta) * r };
}

static void insert_two_boundary_vertices(
    QuadMeshing* env,
    int insert_after_idx,
    Vec2 v_left,
    Vec2 v_right,
    size_t mesh_left,
    size_t mesh_right
) {
    Vec2Array* verts = &env->boundary.vertices;
    SizeArray* meshes = &env->boundary_mesh_vertices;
    const size_t insert_pos = (size_t)insert_after_idx + 1;
    Vec2Array_reserve(verts, verts->size + 2);
    SizeArray_reserve(meshes, meshes->size + 2);
    memmove(&verts->data[insert_pos + 2], &verts->data[insert_pos],
            (verts->size - insert_pos) * sizeof(Vec2));
    memmove(&meshes->data[insert_pos + 2], &meshes->data[insert_pos],
            (meshes->size - insert_pos) * sizeof(size_t));
    verts->data[insert_pos] = v_left;
    verts->data[insert_pos + 1] = v_right;
    meshes->data[insert_pos] = mesh_left;
    meshes->data[insert_pos + 1] = mesh_right;
    verts->size += 2;
    meshes->size += 2;
    mark_boundary_dirty(env);
}

static bool step_edge_actions(
    QuadMeshing* env,
    int action_kind,
    float a1,
    float b1,
    float a2,
    float b2,
    size_t quad_mesh_indices[4],
    bool* quad_mesh_indices_ready
) {
    const int s = env->active_edge_start;
    const int s1 = polygon2D_neighbor_index(env->boundary, s, 1);
    const int s_m1 = polygon2D_neighbor_index(env->boundary, s, -1);
    const int s_m2 = polygon2D_neighbor_index(env->boundary, s, -2);
    const int s_p1 = polygon2D_neighbor_index(env->boundary, s, 2);
    const int s_p2 = polygon2D_neighbor_index(env->boundary, s, 3);

    if (action_kind == 0) {
        return try_close_quad(env, s_m2, s_m1, s, s1, s_m1, quad_mesh_indices, quad_mesh_indices_ready);
    }
    if (action_kind == 1) {
        return try_close_quad(env, s_m1, s, s1, s_p1, s, quad_mesh_indices, quad_mesh_indices_ready);
    }
    if (action_kind == 2) {
        return try_close_quad(env, s, s1, s_p1, s_p2, s1, quad_mesh_indices, quad_mesh_indices_ready);
    }

    Frame2D frame = compute_edge_local_frame(env);
    if (action_kind == 3) {
        Vec2 local = edge_action_local(env, a1, b1);
        Vec2 world = local_to_world(frame, local);
        env->quad.vertices.data[0] = env->boundary.vertices.data[s_m1];
        env->quad.vertices.data[1] = env->boundary.vertices.data[s];
        env->quad.vertices.data[2] = env->boundary.vertices.data[s1];
        env->quad.vertices.data[3] = world;
        const Segment2D new_edge1 = {env->quad.vertices.data[0], env->quad.vertices.data[3]};
        const Segment2D new_edge2 = {env->quad.vertices.data[2], env->quad.vertices.data[3]};
        if (!polygon2D_segment_intersect(env->boundary, new_edge1, NULL, 1e-6) &&
            !polygon2D_segment_intersect(env->boundary, new_edge2, NULL, 1e-6) &&
            eval_polygon2D_sdf(env->boundary, env->quad.vertices.data[3]) < 0.0)
        {
            const size_t new_idx = mesh2D_add_vertex(&env->mesh, env->quad.vertices.data[3]);
            quad_mesh_indices[0] = env->boundary_mesh_vertices.data[s_m1];
            quad_mesh_indices[1] = env->boundary_mesh_vertices.data[s];
            quad_mesh_indices[2] = env->boundary_mesh_vertices.data[s1];
            quad_mesh_indices[3] = new_idx;
            *quad_mesh_indices_ready = true;
            env->boundary.vertices.data[s] = env->quad.vertices.data[3];
            env->boundary_mesh_vertices.data[s] = new_idx;
            mark_boundary_dirty(env);
            return true;
        }
        return false;
    }

    if (action_kind == 4) {
        Vec2 local = edge_action_local(env, a1, b1);
        Vec2 world = local_to_world(frame, local);
        env->quad.vertices.data[0] = env->boundary.vertices.data[s];
        env->quad.vertices.data[1] = env->boundary.vertices.data[s1];
        env->quad.vertices.data[2] = env->boundary.vertices.data[s_p1];
        env->quad.vertices.data[3] = world;
        const Segment2D new_edge1 = {env->quad.vertices.data[0], env->quad.vertices.data[3]};
        const Segment2D new_edge2 = {env->quad.vertices.data[2], env->quad.vertices.data[3]};
        if (!polygon2D_segment_intersect(env->boundary, new_edge1, NULL, 1e-6) &&
            !polygon2D_segment_intersect(env->boundary, new_edge2, NULL, 1e-6) &&
            eval_polygon2D_sdf(env->boundary, env->quad.vertices.data[3]) < 0.0)
        {
            const size_t new_idx = mesh2D_add_vertex(&env->mesh, env->quad.vertices.data[3]);
            quad_mesh_indices[0] = env->boundary_mesh_vertices.data[s];
            quad_mesh_indices[1] = env->boundary_mesh_vertices.data[s1];
            quad_mesh_indices[2] = env->boundary_mesh_vertices.data[s_p1];
            quad_mesh_indices[3] = new_idx;
            *quad_mesh_indices_ready = true;
            env->boundary.vertices.data[s1] = env->quad.vertices.data[3];
            env->boundary_mesh_vertices.data[s1] = new_idx;
            mark_boundary_dirty(env);
            return true;
        }
        return false;
    }

    if (action_kind == 5) {
        Vec2 local1 = edge_action_local(env, a1, b1);
        Vec2 local2 = edge_action_local(env, a2, b2);
        Vec2 world1 = local_to_world(frame, local1);
        Vec2 world2 = local_to_world(frame, local2);

        Vec2 world_left = world1;
        Vec2 world_right = world2;
        if (local1.y > local2.y) {
            world_left = world2;
            world_right = world1;
        }

        env->quad.vertices.data[0] = env->boundary.vertices.data[s];
        env->quad.vertices.data[1] = env->boundary.vertices.data[s1];
        env->quad.vertices.data[2] = world_right;
        env->quad.vertices.data[3] = world_left;
        const Segment2D new_edge1 = {env->quad.vertices.data[0], env->quad.vertices.data[3]};
        const Segment2D new_edge2 = {env->quad.vertices.data[1], env->quad.vertices.data[2]};
        const Segment2D new_edge3 = {env->quad.vertices.data[3], env->quad.vertices.data[2]};
        if (!polygon2D_segment_intersect(env->boundary, new_edge1, NULL, 1e-6) &&
            !polygon2D_segment_intersect(env->boundary, new_edge2, NULL, 1e-6) &&
            !polygon2D_segment_intersect(env->boundary, new_edge3, NULL, 1e-6) &&
            eval_polygon2D_sdf(env->boundary, env->quad.vertices.data[2]) < 0.0 &&
            eval_polygon2D_sdf(env->boundary, env->quad.vertices.data[3]) < 0.0)
        {
            const size_t left_idx = mesh2D_add_vertex(&env->mesh, world_left);
            const size_t right_idx = mesh2D_add_vertex(&env->mesh, world_right);
            quad_mesh_indices[0] = env->boundary_mesh_vertices.data[s];
            quad_mesh_indices[1] = env->boundary_mesh_vertices.data[s1];
            quad_mesh_indices[2] = right_idx;
            quad_mesh_indices[3] = left_idx;
            *quad_mesh_indices_ready = true;
            insert_two_boundary_vertices(env, s, world_left, world_right, left_idx, right_idx);
            return true;
        }
        return false;
    }

    return false;
}

void c_reset(QuadMeshing* env) {
    // Reset episode metrics
    env->episode_length = 0;
    env->episode_return = 0.0;

    env->consecutive_invalid_actions = 0;

    if (env->boundary_set_count > 0) {
        // int index = env->boundary_set_index;
        // if (index < 0 || index >= env->boundary_set_count) {
        //     index = 0;
        // }
        int index = rand() % env->boundary_set_count;
        load_starting_boundary(env, env->boundary_sets[index], env->boundary_set_sizes[index]);
        env->boundary_set_index = (index + 1) % env->boundary_set_count;
        env->boundary.isCCW = env->starting_boundary.isCCW;
        env->quad.isCCW = env->starting_boundary.isCCW;
    }

    // Reset boundary
    Vec2Array_resize(&env->boundary.vertices, env->starting_boundary.vertices.size);
    memcpy(env->boundary.vertices.data, env->starting_boundary.vertices.data,
           env->starting_boundary.vertices.size * sizeof(Vec2));
    env->boundary.isCCW = env->starting_boundary.isCCW;
    env->cache.boundary_area = env->cache.starting_boundary_area;
    env->cache.boundary_area_dirty = false;
    env->cache.render_bounds_dirty = true;

    // Reset mesh
    mesh2D_free(&env->mesh);
    mesh2D_init(&env->mesh);
    SizeArray_resize(&env->boundary_mesh_vertices, env->boundary.vertices.size);
    for (int i = 0; i < env->boundary.vertices.size; ++i) {
        size_t vidx = mesh2D_add_vertex(&env->mesh, env->boundary.vertices.data[i]);
        env->boundary_mesh_vertices.data[i] = vidx;
    }

    // Starting observation 
    compute_observations(env);
}

void c_step(QuadMeshing* env) {
    env->episode_length++;
    env->rewards[0] = 0;
    env->terminals[0] = 0;

    // Check if episode is over
    if (env->boundary.vertices.size <= 4) {
        // Automatically close boundary and terminate episode
        env->quad.vertices.data[0] = env->boundary.vertices.data[0];
        env->quad.vertices.data[1] = Polygon2D_neighbor(env->boundary, 0, 1);
        env->quad.vertices.data[2] = Polygon2D_neighbor(env->boundary, 0, 2);
        env->quad.vertices.data[3] = Polygon2D_neighbor(env->boundary, 0, 3);

        // Add quad to mesh
        if (env->mesh_enabled) {
            size_t quad_mesh_indices[4] = {0};
            quad_mesh_indices[0] = env->boundary_mesh_vertices.data[0];
            quad_mesh_indices[1] = env->boundary_mesh_vertices.data[polygon2D_neighbor_index(env->boundary, 0, 1)];
            quad_mesh_indices[2] = env->boundary_mesh_vertices.data[polygon2D_neighbor_index(env->boundary, 0, 2)];
            quad_mesh_indices[3] = env->boundary_mesh_vertices.data[polygon2D_neighbor_index(env->boundary, 0, 3)];
            mesh2D_add_edge(&env->mesh, quad_mesh_indices[0], quad_mesh_indices[1]);
            mesh2D_add_edge(&env->mesh, quad_mesh_indices[1], quad_mesh_indices[2]);
            mesh2D_add_edge(&env->mesh, quad_mesh_indices[2], quad_mesh_indices[3]);
            mesh2D_add_edge(&env->mesh, quad_mesh_indices[3], quad_mesh_indices[0]);
        }
        
        // Compute last reward
        env->rewards[0] = compute_reward(env, env->quad);
        add_log(env);
        export_mesh_obj(env);
        c_reset(env);
        env->terminals[0] = 1;
        return;
    } else if (env->episode_length == env->episode_max_length) {
        // env->rewards[0] = -1.0;
        add_log(env);
        export_mesh_obj(env);
        c_reset(env);
        env->terminals[0] = 1;
        return;
    }

    bool action_valid = false;
    bool quad_mesh_indices_ready = false;
    size_t quad_mesh_indices[4] = {0};
    const int action_kind = roundf(env->actions[0]);

    if (env->edge_mode) {
        const float a1 = env->actions[1];
        const float b1 = env->actions[2];
        const float a2 = env->actions[3];
        const float b2 = env->actions[4];
        action_valid = step_edge_actions(
            env, action_kind, a1, b1, a2, b2, quad_mesh_indices, &quad_mesh_indices_ready);
    } else if (action_kind == 0) {
        // Close left
        int idx0 = polygon2D_neighbor_index(env->boundary, env->active_vertex, -2);
        int idx1 = polygon2D_neighbor_index(env->boundary, env->active_vertex, -1);
        int idx2 = env->active_vertex;
        int idx3 = polygon2D_neighbor_index(env->boundary, env->active_vertex, 1);
        action_valid = try_close_quad(
            env, idx0, idx1, idx2, idx3, idx1, quad_mesh_indices, &quad_mesh_indices_ready);
    } else if (action_kind == 1) {
        // Close right
        int idx0 = polygon2D_neighbor_index(env->boundary, env->active_vertex, -1);
        int idx1 = env->active_vertex;
        int idx2 = polygon2D_neighbor_index(env->boundary, env->active_vertex, 1);
        int idx3 = polygon2D_neighbor_index(env->boundary, env->active_vertex, 2);
        action_valid = try_close_quad(
            env, idx0, idx1, idx2, idx3, idx1, quad_mesh_indices, &quad_mesh_indices_ready);
    } else {
        const float action_angle = env->actions[1];
        const float action_radius = env->actions[2];
        const float action_x = env->actions[1];
        const float action_y = env->actions[2];
        env->quad.vertices.data[0] = Polygon2D_neighbor(env->boundary, env->active_vertex, -1);
        env->quad.vertices.data[1] = env->boundary.vertices.data[env->active_vertex];
        env->quad.vertices.data[2] = Polygon2D_neighbor(env->boundary, env->active_vertex,  1);
        const float action_scale = env->local_radius * env->action_radius;
        if (env->cartesian_actions) {
            Frame2D frame = compute_active_local_frame(env);
            Vec2 local = { action_x * action_scale, action_y * action_scale };
            env->quad.vertices.data[3] = local_to_world(frame, local);
        } else {
            const float t = 0.5 * (1.0 + action_angle);
            const float r = action_scale * action_radius;
            const Vec2 dir = slerp2(
                sub2(env->quad.vertices.data[0], env->quad.vertices.data[1]),
                sub2(env->quad.vertices.data[2], env->quad.vertices.data[1]),
                t, !env->boundary.isCCW
            );
            env->quad.vertices.data[3] = add2(env->quad.vertices.data[1], scalmul2(dir, r));
        }

        const Segment2D new_edge1 = {env->quad.vertices.data[0], env->quad.vertices.data[3]};
        const Segment2D new_edge2 = {env->quad.vertices.data[2], env->quad.vertices.data[3]};

        if (!polygon2D_segment_intersect(env->boundary, new_edge1, NULL, 1e-6) &&
            !polygon2D_segment_intersect(env->boundary, new_edge2, NULL, 1e-6) &&
            eval_polygon2D_sdf(env->boundary, env->quad.vertices.data[3]) < 0.0)
        {
            action_valid = true;
            quad_mesh_indices[0] = env->boundary_mesh_vertices.data[polygon2D_neighbor_index(env->boundary, env->active_vertex, -1)];
            quad_mesh_indices[1] = env->boundary_mesh_vertices.data[env->active_vertex];
            quad_mesh_indices[2] = env->boundary_mesh_vertices.data[polygon2D_neighbor_index(env->boundary, env->active_vertex, 1)];
            quad_mesh_indices[3] = mesh2D_add_vertex(&env->mesh, env->quad.vertices.data[3]);
            quad_mesh_indices_ready = true;
            env->boundary.vertices.data[env->active_vertex] = env->quad.vertices.data[3];
            env->boundary_mesh_vertices.data[env->active_vertex] = quad_mesh_indices[3];
            mark_boundary_dirty(env);
        }
    }

    if (action_valid) {
        float quad_reward = compute_reward(env, env->quad);
        assert(isfinite(quad_reward) && quad_reward <= 1.0f);
        if (env->delayed_rewards) {
            env->rewards[0] = 0.0f;
            env->episode_return += quad_reward / 128.0;
        } else {
            env->rewards[0] = quad_reward;
            env->episode_return += env->rewards[0];
        }
        env->consecutive_invalid_actions = 0;
    } else {
        env->rewards[0] = -0.1f;
        env->episode_return += env->rewards[0];
        ++env->consecutive_invalid_actions;
    }

    // Add latest quad to mesh
    if (action_valid && env->mesh_enabled) {
      assert(quad_mesh_indices_ready);
      mesh2D_add_edge(&env->mesh, quad_mesh_indices[0], quad_mesh_indices[1]);
      mesh2D_add_edge(&env->mesh, quad_mesh_indices[1], quad_mesh_indices[2]);
      mesh2D_add_edge(&env->mesh, quad_mesh_indices[2], quad_mesh_indices[3]);
      mesh2D_add_edge(&env->mesh, quad_mesh_indices[3], quad_mesh_indices[0]);
    }

    compute_observations(env);
}

void c_render(QuadMeshing* env) {
    if (!env->render_enabled) {
        return;
    }
    if (!IsWindowReady()) {
        InitWindow(1080, 720, "PufferLib QuadMeshing");
        SetTargetFPS(env->render_target_fps);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }

    BeginDrawing();
    ClearBackground(RAYWHITE);

    update_render_bounds(env);
    RenderContext ctx = compute_render_context(env->cache.render_min, env->cache.render_max);

    if (IsKeyDown(KEY_S)) {
        draw_sdf(&env->boundary, &ctx);
    }

    draw_mesh(&env->mesh, &ctx);
    draw_boundary(&env->boundary, &ctx);

    // Active vertex
    if (env->edge_mode) {
        Vec2 v0 = env->boundary.vertices.data[env->active_edge_start];
        Vec2 v1 = Polygon2D_neighbor(env->boundary, env->active_edge_start, 1);
        DrawLineV(world_to_screen(v0, &ctx), world_to_screen(v1, &ctx), RED);
    } else {
        DrawCircleV(world_to_screen(env->boundary.vertices.data[env->active_vertex], &ctx), 8.0, RED);
    }

    // Action radius
    const float r = env->local_radius * env->action_radius;
    if (env->cartesian_actions) {
        Frame2D frame = compute_edge_local_frame(env);
        Vector2 p0 = world_to_screen(local_to_world(frame, (Vec2){0., -r}), &ctx);
        Vector2 p1 = world_to_screen(local_to_world(frame, (Vec2){r,  -r}), &ctx);
        Vector2 p2 = world_to_screen(local_to_world(frame, (Vec2){r,   r}), &ctx);
        Vector2 p3 = world_to_screen(local_to_world(frame, (Vec2){0.,  r}), &ctx);
        DrawLineV(p0, p1, RED);
        DrawLineV(p1, p2, RED);
        DrawLineV(p2, p3, RED);
        DrawLineV(p3, p0, RED);
    } else {
        DrawCircleLinesV(
            world_to_screen(env->boundary.vertices.data[env->active_vertex], &ctx),
            world_to_screen_scale(r, &ctx), RED);
    }

    // SDF grid
    for (int i = 0; i < env->n_sdf_samples; ++i) {
        Vec2 query_world = env->cache.sdf_query_world.data[i];
        float d = env->cache.sdf_query_values.data[i];
        Color c = {
            (unsigned char)(255 * clamp01(0.5 - 5.0*d * 0.1)),
            (unsigned char)(255 * clamp01(0.5 + 5.0*d * 0.4)),
            (unsigned char)(255 * clamp01(0.5 - 5.0*d * 0.7)),
            255
        };
        DrawCircleV(world_to_screen(query_world, &ctx), 3.0, c);
    }

    const float boundary_cost = compute_boundary_angle_cost(env) + compute_boundary_edge_cost(env);
    DrawText(TextFormat("S: SDF | ESC: Quit | Boundary cost: %f | Episode return: %f", boundary_cost, env->episode_return), 10, 10, 20, DARKGRAY);

    EndDrawing();
}

void c_close(QuadMeshing* env) {
    if (env->boundary_sets) {
        for (int i = 0; i < env->boundary_set_count; ++i) {
            free(env->boundary_sets[i]);
        }
        free(env->boundary_sets);
        env->boundary_sets = NULL;
    }
    if (env->boundary_set_sizes) {
        free(env->boundary_set_sizes);
        env->boundary_set_sizes = NULL;
    }
    env->boundary_set_count = 0;
    Vec2Array_free(&env->starting_boundary.vertices);
    Vec2Array_free(&env->boundary.vertices);
    Vec2Array_free(&env->quad.vertices);
    mesh2D_free(&env->mesh);
    SizeArray_free(&env->boundary_mesh_vertices);
    Vec2Array_free(&env->cache.sdf_query_world);
    FloatArray_free(&env->cache.sdf_query_values);
    if (IsWindowReady()) {
        CloseWindow();
    }
}
