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
    bool random_active_vertex;

    // Observations config
    bool observe_remaining_area;
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
    float local_radius;
    int episode_length;
    float episode_return;

    // Cache
    QuadMeshingCache cache;
} QuadMeshing;

static void build_export_mesh_path(QuadMeshing* env, char* out, size_t out_size) {
    const char* template = env->export_mesh_path;
    if (!template || template[0] == '\0') {
        template = "mesh.obj";
    }

    const char* token = strstr(template, "{episode}");
    if (token) {
        size_t prefix_len = (size_t)(token - template);
        const char* suffix = token + strlen("{episode}");
        if (prefix_len + strlen(suffix) + 32 >= out_size) {
            out[0] = '\0';
            return;
        }
        memcpy(out, template, prefix_len);
        snprintf(out + prefix_len, out_size - prefix_len, "%d%s", env->export_mesh_counter, suffix);
        return;
    }

    if (env->export_mesh_counter == 0) {
        snprintf(out, out_size, "%s", template);
        return;
    }

    const char* slash = strrchr(template, '/');
    const char* dot = strrchr(template, '.');
    if (dot && (!slash || dot > slash)) {
        int base_len = (int)(dot - template);
        snprintf(out, out_size, "%.*s_%d%s", base_len, template, env->export_mesh_counter, dot);
    } else {
        snprintf(out, out_size, "%s_%d", template, env->export_mesh_counter);
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

void init(QuadMeshing* env, float* boundary_vertices, int num_vertices) {
    // Initialize starting boundary from provided vertices or default to square
    Vec2Array_init(&env->starting_boundary.vertices);
    
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

    // Calculate target_quad_area as square of average boundary segment length
    float perimeter = 0.0;
    for (int i = env->starting_boundary.vertices.size - 1, j = 0; j < env->starting_boundary.vertices.size; i = j, ++j) {
        Vec2 v_curr = env->starting_boundary.vertices.data[i];
        Vec2 v_next = env->starting_boundary.vertices.data[j];
        float dx = v_next.x - v_curr.x;
        float dy = v_next.y - v_curr.y;
        perimeter += sqrtf(dx * dx + dy * dy);
    }
    float avg_segment_length = perimeter / env->starting_boundary.vertices.size;
    env->target_quad_area = avg_segment_length * avg_segment_length;

    // Calculate episode_max_length as 2 * (boundary area) / (target quad area)
    env->episode_max_length = (int)(2.0 * env->cache.starting_boundary_area / env->target_quad_area);
    
    env->active_vertex = 0;    
    env->mesh_enabled = env->render_enabled || env->export_meshes;
}

void add_log(QuadMeshing* env) {
    // env->log.perf += (env->episode_length < env->episode_max_length) ?
    //     env->episode_return / polygon2D_area(env->starting_boundary) : 0;
    env->log.perf += env->episode_return / env->cache.starting_boundary_area;
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

void compute_observations(QuadMeshing* env) {
    assert(env->boundary.vertices.size > 0);

    // Determine active vertex
    if (env->random_active_vertex) {
        // Pick at random among interior angles <= pi
        do {
            env->active_vertex = rand() % env->boundary.vertices.size;
        } while (polygonInteriorAngle(env->boundary, env->active_vertex) > M_PI);
    } else {
        // Pick vertex with smallest interior angle
        env->active_vertex = 0;
        float angleMin = polygonInteriorAngle(env->boundary, 0);
        for (int i=1; i<env->boundary.vertices.size; ++i) {
            const float angle = polygonInteriorAngle(env->boundary, i);
            if (angle < angleMin) {
                angleMin = angle;
                env->active_vertex = i;
            }
        }
    }

    // Compute local radius
    if (env->fixed_local_radius > 0.0f) {
        env->local_radius = env->fixed_local_radius;
    } else {
        env->local_radius = 0.5 *
          norm2(sub2(Polygon2D_neighbor(env->boundary, env->active_vertex, -1), env->boundary.vertices.data[env->active_vertex])) +
          norm2(sub2(Polygon2D_neighbor(env->boundary, env->active_vertex,  1), env->boundary.vertices.data[env->active_vertex]));
    }

    // env->active_vertex = rand() % env->boundary.vertices.size;
    // int countdown = env->boundary.vertices.size;
    // while (polygonInteriorAngle(env->boundary, env->active_vertex) > M_PI) {
    //     env->active_vertex = polygon2D_neighbor_index(env->boundary, env->active_vertex, 1);
    //     if (--countdown == 0) {
    //         printf("============ BOUNDARY ============\n");
    //         for (int i=0; i<env->boundary.vertices.size; ++i) {
    //             const float angle = polygonInteriorAngle(env->boundary, i);
    //             printf("%d (%f, %f): %f\n", i, env->boundary.vertices.data[i].x, env->boundary.vertices.data[i].y, angle);
    //         }
    //         break;
    //     }
    // }

    int obs_idx = 0;

    if (env->observe_remaining_area) {
        env->observations[obs_idx++] = get_boundary_area(env) / env->cache.starting_boundary_area;
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

    // Make SDF observations
    float r = env->local_radius * env->observation_radius / env->n_sdf_samples;
    for (int i = 1; i < env->n_sdf_samples+1; ++i) {
        Vec2 query_local = { r * i, 0 };
        Vec2 query_world = local_to_world(frame, query_local);
        float d = clampf(eval_polygon2D_sdf(env->boundary, query_world), -1.0, 1.0);
        env->observations[obs_idx++] = d;
        if (env->render_enabled) {
            env->cache.sdf_query_world.data[i - 1] = query_world;
            env->cache.sdf_query_values.data[i - 1] = d;
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
    const float alpha = 10.0;
    const float A = polygon2D_area(quad);
    const float Ad = A / env->target_quad_area - 1.0;
    const float dq = 1.0 / (1.0 + alpha * Ad*Ad);

    // return A * eq * dq;
    // return fmin(A / env->target_quad_area, 1.0) * eq;
    return (1.0 - fabs(A / env->target_quad_area - 1.0)) * eq;
    // return eq * dq;
    // return 0.5 * (eq + dq);
}

void c_reset(QuadMeshing* env) {
    // Reset episode metrics
    env->episode_length = 0;
    env->episode_return = 0.0;

    // Reset boundary
    Vec2Array_resize(&env->boundary.vertices, env->starting_boundary.vertices.size);
    memcpy(env->boundary.vertices.data, env->starting_boundary.vertices.data,
           env->starting_boundary.vertices.size * sizeof(Vec2));
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
        env->rewards[0] = -1.0;
        add_log(env);
        export_mesh_obj(env);
        c_reset(env);
        env->terminals[0] = 1;
        return;
    }

    const int action_kind = roundf(env->actions[0]);
    const float action_angle = env->actions[1];
    const float action_radius = env->actions[2];
    const float action_x = env->actions[1];
    const float action_y = env->actions[2];

    bool action_valid = false;
    bool quad_mesh_indices_ready = false;
    size_t quad_mesh_indices[4] = {0};
    if (action_kind == 0) {
        // Close left
        env->quad.vertices.data[0] = Polygon2D_neighbor(env->boundary, env->active_vertex, -2);
        env->quad.vertices.data[1] = Polygon2D_neighbor(env->boundary, env->active_vertex, -1);
        env->quad.vertices.data[2] = env->boundary.vertices.data[env->active_vertex];
        env->quad.vertices.data[3] = Polygon2D_neighbor(env->boundary, env->active_vertex,  1);
        const Segment2D new_edge = {env->quad.vertices.data[0], env->quad.vertices.data[3]};
        const float quad_area = polygon2D_area(env->quad);
        const float boundary_area = get_boundary_area(env);

        action_valid = !(polygon2D_segment_intersect(env->boundary, new_edge, NULL, 1e-6) ||
                         boundary_area < quad_area ||
                         !(env->boundary.isCCW == is_polygon_ccw(env->quad)));
        if (action_valid) {
            quad_mesh_indices[0] = env->boundary_mesh_vertices.data[polygon2D_neighbor_index(env->boundary, env->active_vertex, -2)];
            quad_mesh_indices[1] = env->boundary_mesh_vertices.data[polygon2D_neighbor_index(env->boundary, env->active_vertex, -1)];
            quad_mesh_indices[2] = env->boundary_mesh_vertices.data[env->active_vertex];
            quad_mesh_indices[3] = env->boundary_mesh_vertices.data[polygon2D_neighbor_index(env->boundary, env->active_vertex, 1)];
            quad_mesh_indices_ready = true;
            if (env->active_vertex == 0) {
                env->boundary.vertices.data[0] = env->boundary.vertices.data[env->boundary.vertices.size-2];
                env->boundary.vertices.size -= 2;
                env->boundary_mesh_vertices.data[0] = env->boundary_mesh_vertices.data[env->boundary_mesh_vertices.size-2];
                env->boundary_mesh_vertices.size -= 2;
            } else {
                Vec2Array_remove_range(&env->boundary.vertices, env->active_vertex-1, env->active_vertex);
                SizeArray_remove_range(&env->boundary_mesh_vertices, env->active_vertex-1, env->active_vertex);
            }
            mark_boundary_dirty(env);
        }
    } else if (action_kind == 1) {
        // Close right
        env->quad.vertices.data[0] = Polygon2D_neighbor(env->boundary, env->active_vertex, -1);
        env->quad.vertices.data[1] = env->boundary.vertices.data[env->active_vertex];
        env->quad.vertices.data[2] = Polygon2D_neighbor(env->boundary, env->active_vertex,  1);
        env->quad.vertices.data[3] = Polygon2D_neighbor(env->boundary, env->active_vertex,  2);
        const Segment2D new_edge = {env->quad.vertices.data[0], env->quad.vertices.data[3]};
        const float quad_area = polygon2D_area(env->quad);
        const float boundary_area = get_boundary_area(env);

        action_valid = !(polygon2D_segment_intersect(env->boundary, new_edge, NULL, 1e-6) ||
                         boundary_area < quad_area ||
                         !(env->boundary.isCCW == is_polygon_ccw(env->quad)));
        if (action_valid) {
            quad_mesh_indices[0] = env->boundary_mesh_vertices.data[polygon2D_neighbor_index(env->boundary, env->active_vertex, -1)];
            quad_mesh_indices[1] = env->boundary_mesh_vertices.data[env->active_vertex];
            quad_mesh_indices[2] = env->boundary_mesh_vertices.data[polygon2D_neighbor_index(env->boundary, env->active_vertex, 1)];
            quad_mesh_indices[3] = env->boundary_mesh_vertices.data[polygon2D_neighbor_index(env->boundary, env->active_vertex, 2)];
            quad_mesh_indices_ready = true;
            if (env->active_vertex == env->boundary.vertices.size-1) {
                env->boundary.vertices.data[0] = env->boundary.vertices.data[env->boundary.vertices.size-2];
                env->boundary.vertices.size -= 2;
                env->boundary_mesh_vertices.data[0] = env->boundary_mesh_vertices.data[env->boundary_mesh_vertices.size-2];
                env->boundary_mesh_vertices.size -= 2;
            } else {
                Vec2Array_remove_range(&env->boundary.vertices, env->active_vertex, env->active_vertex+1);
                SizeArray_remove_range(&env->boundary_mesh_vertices, env->active_vertex, env->active_vertex+1);
            }
            mark_boundary_dirty(env);
        }
    } else {
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
    } else {
        env->rewards[0] = -0.1f;
        // env->rewards[0] = 0.0f;
        env->episode_return += env->rewards[0];
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
    DrawCircleV(world_to_screen(env->boundary.vertices.data[env->active_vertex], &ctx), 8.0, RED);

    if (!env->cartesian_actions) {
        // Action radius
        DrawCircleLinesV(
            world_to_screen(env->boundary.vertices.data[env->active_vertex], &ctx),
            world_to_screen_scale(env->local_radius * env->action_radius, &ctx), RED);
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

    DrawText(TextFormat("S: SDF | ESC: Quit | Episode return: %f", env->episode_return), 10, 10, 20, DARKGRAY);

    EndDrawing();
}

void c_close(QuadMeshing* env) {
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
