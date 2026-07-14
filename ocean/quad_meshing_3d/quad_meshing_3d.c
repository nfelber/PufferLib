#include "quad_meshing_3d.h"

#include <stdio.h>
#include <string.h>

static void usage(const char* exe) {
    fprintf(stderr, "Usage: %s surface.qmsurf [--check]\n", exe);
}

static bool qm3_check_face_registration_debug(void) {
    Qm3Mesh mesh;
    qm3_mesh_init(&mesh, 8);
    Qm3Vec3 n = {0.0f, 1.0f, 0.0f};
    uint32_t v0 = qm3_mesh_add_vertex(&mesh, (Qm3Vec3){0.0f, 0.0f, 0.0f}, n, UINT32_MAX, UINT32_MAX, -1);
    uint32_t v1 = qm3_mesh_add_vertex(&mesh, (Qm3Vec3){1.0f, 0.0f, 0.0f}, n, UINT32_MAX, UINT32_MAX, -1);
    uint32_t v2 = qm3_mesh_add_vertex(&mesh, (Qm3Vec3){1.0f, 0.0f, 1.0f}, n, UINT32_MAX, UINT32_MAX, -1);
    uint32_t v3 = qm3_mesh_add_vertex(&mesh, (Qm3Vec3){0.0f, 0.0f, 1.0f}, n, UINT32_MAX, UINT32_MAX, -1);
    qm3_mesh_add_edge_with_path(&mesh, v0, v1, 0, NULL, 0, NULL, 0, 0.0f);
    qm3_mesh_add_edge_with_path(&mesh, v1, v2, 0, NULL, 0, NULL, 0, 0.0f);
    qm3_mesh_add_edge_with_path(&mesh, v2, v3, 0, NULL, 0, NULL, 0, 0.0f);
    uint32_t new_quad_edge = qm3_mesh_add_edge_with_path(&mesh, v0, v3, 0, NULL, 0, NULL, 0, 0.0f);

    uint32_t cycles[64];
    uint32_t quads = qm3_mesh_detect_quads(&mesh, v0, v3, cycles, 16);
    bool quad_ok = quads > 0 && qm3_mesh_register_face(&mesh, cycles, 4);
    bool quad_counts_ok = quad_ok && mesh.face_count == 1 && mesh.quad_count == 1 && mesh.tri_count == 0 && mesh.edges[new_quad_edge].face_count == 1;

    Qm3Mesh tri_mesh;
    qm3_mesh_init(&tri_mesh, 8);
    uint32_t t0 = qm3_mesh_add_vertex(&tri_mesh, (Qm3Vec3){0.0f, 0.0f, 0.0f}, n, UINT32_MAX, UINT32_MAX, -1);
    uint32_t t1 = qm3_mesh_add_vertex(&tri_mesh, (Qm3Vec3){1.0f, 0.0f, 0.0f}, n, UINT32_MAX, UINT32_MAX, -1);
    uint32_t t2 = qm3_mesh_add_vertex(&tri_mesh, (Qm3Vec3){0.0f, 0.0f, 1.0f}, n, UINT32_MAX, UINT32_MAX, -1);
    qm3_mesh_add_edge_with_path(&tri_mesh, t0, t1, 0, NULL, 0, NULL, 0, 0.0f);
    qm3_mesh_add_edge_with_path(&tri_mesh, t1, t2, 0, NULL, 0, NULL, 0, 0.0f);
    uint32_t new_tri_edge = qm3_mesh_add_edge_with_path(&tri_mesh, t0, t2, 0, NULL, 0, NULL, 0, 0.0f);
    uint32_t tris = qm3_mesh_detect_triangles(&tri_mesh, t0, t2, cycles, 16);
    bool tri_ok = tris > 0 && qm3_mesh_register_face(&tri_mesh, cycles, 3);
    bool tri_counts_ok = tri_ok && tri_mesh.face_count == 1 && tri_mesh.quad_count == 0 && tri_mesh.tri_count == 1 && tri_mesh.edges[new_tri_edge].face_count == 1;

    printf("debug_face_registration quads_detected=%u quad_ok=%d tris_detected=%u tri_ok=%d\n", quads, quad_counts_ok ? 1 : 0, tris, tri_counts_ok ? 1 : 0);
    qm3_mesh_free(&mesh);
    qm3_mesh_free(&tri_mesh);
    return quad_counts_ok && tri_counts_ok;
}

static bool qm3_check_loop_subdivision_case(const Qm3PathSegment* segments, uint32_t segment_count, int expected_cells, double expected_min_area, bool expected_edge_cut) {
    Qm3Vec3 vertices[3] = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    Qm3Tri triangles[1] = {{0, 1, 2}};
    Qm3Vec2 tri2d[3] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
    unsigned char tri2d_valid[1] = {1};
    QuadMeshing3DEnv env = {0};
    env.surface.vertices = vertices;
    env.surface.vertex_count = 3;
    env.surface.triangles = triangles;
    env.surface.triangle_count = 1;
    env.continuous_ctx.tri2d_base = tri2d;
    env.continuous_ctx.tri2d_valid = tri2d_valid;
    Qm3PathSegmentArray path = {.data = (Qm3PathSegment*)segments, .count = segment_count, .cap = segment_count};
    Qm3LoopSubdivision sub = {0};
    sub.tri_ids = (int*)calloc(1, sizeof(int));
    sub.tri_count = 1;
    sub.tri_cell_offsets = (int*)calloc(2, sizeof(int));
    sub.tri_portal_offsets = (int*)calloc(2, sizeof(int));
    sub.tri_tol = (double*)calloc(1, sizeof(double));
    unsigned char flags[3] = {0};
    unsigned char boundary_tri[1] = {0};
    bool ok = sub.tri_ids && sub.tri_cell_offsets && sub.tri_portal_offsets && sub.tri_tol &&
        qm3_loop_build_triangle(&env, &path, 0, 0, &sub, flags, boundary_tri);
    double min_area = INFINITY;
    double total_area = 0.0;
    for (int i = 0; i < sub.cell_count; ++i) {
        min_area = fmin(min_area, sub.cells[i].area);
        total_area += sub.cells[i].area;
    }
    bool has_edge_cut = ((flags[0] | flags[1] | flags[2]) & QM3_LOOP_EDGE_BLOCKED) != 0;
    ok = ok && sub.cell_count == expected_cells && fabs(total_area - 0.5) <= 1e-6 &&
        fabs(min_area - expected_min_area) <= 1e-6 && has_edge_cut == expected_edge_cut;
    qm3_loop_subdivision_free(&sub);
    return ok;
}

static bool qm3_check_loop_subdivision_debug(void) {
    Qm3PathSegment chord[] = {{.tri = 0, .a = {0.5f, 0.0f, 0.0f}, .b = {0.0f, 0.5f, 0.0f}}};
    Qm3PathSegment vertex_cut[] = {{.tri = 0, .a = {0.0f, 0.0f, 0.0f}, .b = {0.5f, 0.5f, 0.0f}}};
    Qm3PathSegment interior_turn[] = {
        {.tri = 0, .a = {0.5f, 0.0f, 0.0f}, .b = {0.25f, 0.25f, 0.0f}},
        {.tri = 0, .a = {0.25f, 0.25f, 0.0f}, .b = {0.0f, 0.5f, 0.0f}},
    };
    Qm3PathSegment edge_cut[] = {{.tri = 0, .a = {0.0f, 0.0f, 0.0f}, .b = {1.0f, 0.0f, 0.0f}}};
    bool chord_ok = qm3_check_loop_subdivision_case(chord, 1, 2, 0.125, false);
    bool vertex_ok = qm3_check_loop_subdivision_case(vertex_cut, 1, 2, 0.25, false);
    bool turn_ok = qm3_check_loop_subdivision_case(interior_turn, 2, 2, 0.125, false);
    bool edge_ok = qm3_check_loop_subdivision_case(edge_cut, 1, 1, 0.5, true);
    printf("debug_loop_subdivision chord=%d vertex=%d interior_turn=%d edge=%d\n", chord_ok, vertex_ok, turn_ok, edge_ok);
    return chord_ok && vertex_ok && turn_ok && edge_ok;
}

static bool qm3_check_loop_removal_debug(void) {
    Qm3Vec3 vertices[3] = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    Qm3Tri triangles[1] = {{0, 1, 2}};
    Qm3TriNeighbors neighbors[1] = {{-1, -1, -1}};
    Qm3SurfaceSample samples[2] = {
        {.p = {0.3f, 0.3f, 0.0f}, .tri = 0},
        {.p = {0.8f, 0.1f, 0.0f}, .tri = 0},
    };
    Qm3Vec2 tri2d[3] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
    unsigned char tri2d_valid[1] = {1};
    unsigned char disabled[2] = {0};
    int sample_offsets[2] = {0, 2};
    int sample_ids[2] = {0, 1};
    Qm3PathSegment segments[] = {
        {.tri = 0, .a = {0.2f, 0.2f, 0.0f}, .b = {0.6f, 0.2f, 0.0f}},
        {.tri = 0, .a = {0.6f, 0.2f, 0.0f}, .b = {0.2f, 0.6f, 0.0f}},
        {.tri = 0, .a = {0.2f, 0.6f, 0.0f}, .b = {0.2f, 0.2f, 0.0f}},
    };
    QuadMeshing3DEnv env = {0};
    env.surface.vertices = vertices;
    env.surface.vertex_count = 3;
    env.surface.triangles = triangles;
    env.surface.triangle_neighbors = neighbors;
    env.surface.triangle_count = 1;
    env.surface.samples = samples;
    env.surface.sample_count = 2;
    env.continuous_ctx.tri2d_base = tri2d;
    env.continuous_ctx.tri2d_valid = tri2d_valid;
    env.surface_topo.tri_sample_offsets = sample_offsets;
    env.surface_topo.tri_sample_ids = sample_ids;
    env.sample_disabled = disabled;
    Qm3PathSegmentArray loop = {.data = segments, .count = 3, .cap = 3};
    Qm3LoopRemovalStats stats = qm3_remove_samples_inside_loop(&env, &loop, 1);
    bool ok = stats.chosen_side == 1 && stats.disabled_samples == 1 && disabled[0] && !disabled[1];
    printf("debug_loop_removal chosen=%d removed=%u inside=%d outside=%d\n",
        stats.chosen_side, stats.disabled_samples, disabled[0], disabled[1]);
    qm3_loop_debug_free(&env.loop_debug);
    return ok;
}

static bool qm3_check_loop_removal_across_triangles_debug(void) {
    Qm3Vec3 vertices[4] = {
        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
    };
    Qm3Tri triangles[2] = {{0, 1, 2}, {0, 2, 3}};
    Qm3TriNeighbors neighbors[2] = {{-1, 1, -1}, {-1, -1, 0}};
    Qm3SurfaceSample samples[2] = {
        {.p = {0.6f, 0.5f, 0.0f}, .tri = 0},
        {.p = {0.9f, 0.1f, 0.0f}, .tri = 0},
    };
    Qm3Vec2 tri2d[6] = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f},
        {0.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f},
    };
    unsigned char tri2d_valid[2] = {1, 1};
    unsigned char disabled[2] = {0};
    int sample_offsets[3] = {0, 2, 2};
    int sample_ids[2] = {0, 1};
    Qm3PathSegment segments[] = {
        {.tri = 1, .a = {0.1f, 0.5f, 0.0f}, .b = {0.300001f, 0.300001f, 0.0f}},
        {.tri = 0, .a = {0.299999f, 0.299999f, 0.0f}, .b = {0.5f, 0.1f, 0.0f}},
        {.tri = 0, .a = {0.5f, 0.1f, 0.0f}, .b = {0.9f, 0.5f, 0.0f}},
        {.tri = 0, .a = {0.9f, 0.5f, 0.0f}, .b = {0.699999f, 0.699999f, 0.0f}},
        {.tri = 1, .a = {0.700001f, 0.700001f, 0.0f}, .b = {0.5f, 0.9f, 0.0f}},
        {.tri = 1, .a = {0.5f, 0.9f, 0.0f}, .b = {0.1f, 0.5f, 0.0f}},
    };
    QuadMeshing3DEnv env = {0};
    env.surface.vertices = vertices;
    env.surface.vertex_count = 4;
    env.surface.triangles = triangles;
    env.surface.triangle_neighbors = neighbors;
    env.surface.triangle_count = 2;
    env.surface.samples = samples;
    env.surface.sample_count = 2;
    env.continuous_ctx.tri2d_base = tri2d;
    env.continuous_ctx.tri2d_valid = tri2d_valid;
    env.surface_topo.tri_sample_offsets = sample_offsets;
    env.surface_topo.tri_sample_ids = sample_ids;
    env.sample_disabled = disabled;
    Qm3PathSegmentArray loop = {.data = segments, .count = 6, .cap = 6};
    Qm3LoopRemovalStats stats = qm3_remove_samples_inside_loop(&env, &loop, 1);
    bool ok = stats.chosen_side == 1 && stats.disabled_samples == 1 && disabled[0] && !disabled[1];
    printf("debug_loop_removal_cross_tri chosen=%d removed=%u inside=%d outside=%d\n",
        stats.chosen_side, stats.disabled_samples, disabled[0], disabled[1]);
    qm3_loop_debug_free(&env.loop_debug);
    return ok;
}

static bool qm3_check_loop_removal_native_edge_debug(void) {
    Qm3Vec3 vertices[4] = {
        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
    };
    Qm3Tri triangles[2] = {{0, 1, 2}, {0, 2, 3}};
    Qm3TriNeighbors neighbors[2] = {{-1, 1, -1}, {-1, -1, 0}};
    Qm3SurfaceSample samples[2] = {
        {.p = {0.75f, 0.25f, 0.0f}, .tri = 0},
        {.p = {0.25f, 0.75f, 0.0f}, .tri = 1},
    };
    Qm3Vec2 tri2d[6] = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f},
        {0.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f},
    };
    unsigned char tri2d_valid[2] = {1, 1};
    unsigned char disabled[2] = {0};
    int sample_offsets[3] = {0, 1, 2};
    int sample_ids[2] = {0, 1};
    Qm3PathSegment segments[] = {
        {.tri = 0, .a = {0.0f, 0.0f, 0.0f}, .b = {1.0f, 0.0f, 0.0f}},
        {.tri = 0, .a = {1.0f, 0.0f, 0.0f}, .b = {1.0f, 1.0f, 0.0f}},
        {.tri = 0, .a = {1.0f, 1.0f, 0.0f}, .b = {0.0f, 0.0f, 0.0f}},
        {.tri = 1, .a = {1.0f, 1.0f, 0.0f}, .b = {0.0f, 0.0f, 0.0f}},
    };
    QuadMeshing3DEnv env = {0};
    env.surface.vertices = vertices;
    env.surface.vertex_count = 4;
    env.surface.triangles = triangles;
    env.surface.triangle_neighbors = neighbors;
    env.surface.triangle_count = 2;
    env.surface.samples = samples;
    env.surface.sample_count = 2;
    env.continuous_ctx.tri2d_base = tri2d;
    env.continuous_ctx.tri2d_valid = tri2d_valid;
    env.surface_topo.tri_sample_offsets = sample_offsets;
    env.surface_topo.tri_sample_ids = sample_ids;
    env.sample_disabled = disabled;
    Qm3PathSegmentArray loop = {.data = segments, .count = 4, .cap = 4};
    Qm3LoopRemovalStats stats = qm3_remove_samples_inside_loop(&env, &loop, 1);
    bool ok = stats.chosen_side == 1 && stats.disabled_samples == 1 && disabled[0] && !disabled[1];
    printf("debug_loop_removal_native_edge chosen=%d removed=%u inside=%d outside=%d conflicts=%u\n",
        stats.chosen_side, stats.disabled_samples, disabled[0], disabled[1], stats.classification_conflicts);
    qm3_loop_debug_free(&env.loop_debug);
    return ok;
}

static bool qm3_check_loop_removal_locality_debug(void) {
    const uint32_t triangle_count = 40000;
    Qm3Vec3 vertices[3] = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    Qm3Tri* triangles = (Qm3Tri*)malloc((size_t)triangle_count * sizeof(*triangles));
    Qm3TriNeighbors* neighbors = (Qm3TriNeighbors*)malloc((size_t)triangle_count * sizeof(*neighbors));
    Qm3Vec2* tri2d = (Qm3Vec2*)calloc((size_t)triangle_count * 3, sizeof(*tri2d));
    unsigned char* tri2d_valid = (unsigned char*)calloc(triangle_count, 1);
    Qm3SurfaceSample samples[2] = {
        {.p = {0.3f, 0.3f, 0.0f}, .tri = 0},
        {.p = {0.3f, 0.3f, 0.0f}, .tri = (int32_t)triangle_count - 1},
    };
    unsigned char disabled[2] = {0};
    Qm3PathSegment segments[] = {
        {.tri = 0, .a = {0.2f, 0.2f, 0.0f}, .b = {0.6f, 0.2f, 0.0f}},
        {.tri = 0, .a = {0.6f, 0.2f, 0.0f}, .b = {0.2f, 0.6f, 0.0f}},
        {.tri = 0, .a = {0.2f, 0.6f, 0.0f}, .b = {0.2f, 0.2f, 0.0f}},
    };
    bool allocated = triangles && neighbors && tri2d && tri2d_valid;
    if (!allocated) {
        free(triangles); free(neighbors); free(tri2d); free(tri2d_valid);
        return false;
    }
    for (uint32_t i = 0; i < triangle_count; ++i) {
        triangles[i] = (Qm3Tri){0, 1, 2};
        neighbors[i] = (Qm3TriNeighbors){-1, -1, -1};
    }
    tri2d[0] = (Qm3Vec2){0.0f, 0.0f};
    tri2d[1] = (Qm3Vec2){1.0f, 0.0f};
    tri2d[2] = (Qm3Vec2){0.0f, 1.0f};
    tri2d_valid[0] = 1;
    QuadMeshing3DEnv env = {0};
    env.surface.vertices = vertices;
    env.surface.vertex_count = 3;
    env.surface.triangles = triangles;
    env.surface.triangle_neighbors = neighbors;
    env.surface.triangle_count = triangle_count;
    env.surface.samples = samples;
    env.surface.sample_count = 2;
    env.surface_topo = qm3_surface_topo_build(&env.surface);
    env.continuous_ctx.tri2d_base = tri2d;
    env.continuous_ctx.tri2d_valid = tri2d_valid;
    env.sample_disabled = disabled;
    Qm3PathSegmentArray loop = {.data = segments, .count = 3, .cap = 3};
    double start_ms = qm3_time_ms();
    Qm3LoopRemovalStats stats = qm3_remove_samples_inside_loop(&env, &loop, 1);
    double elapsed_ms = qm3_time_ms() - start_ms;
    bool ok = stats.chosen_side == 1 && stats.boundary_tris == 1 && stats.flood_tris == 0 &&
        stats.disabled_samples == 1 && disabled[0] && !disabled[1];
    printf("debug_loop_removal_locality triangles=%u boundary=%u flood=%u removed=%u untouched=%d time_ms=%.3f\n",
        triangle_count, stats.boundary_tris, stats.flood_tris, stats.disabled_samples, !disabled[1], elapsed_ms);
    qm3_surface_topo_free(&env.surface_topo);
    qm3_loop_debug_free(&env.loop_debug);
    free(triangles); free(neighbors); free(tri2d); free(tri2d_valid);
    return ok;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char* path = argv[1];
    bool check_only = argc > 2 && strcmp(argv[2], "--check") == 0;
    const char* paths[] = {path};

    QuadMeshing3DEnv env = {0};
    env.shape_paths = paths;
    env.shape_count = 1;
    env.render_width = 1280;
    env.render_height = 900;
    env.render_target_fps = 144;
    env.max_frontier = 4096;
    env.max_candidates = 768;
    env.max_degree = 8;
    env.candidate_radius_min_ratio = 0.5f;
    env.candidate_radius_max_ratio = 1.5f;
    env.geodesic_steiner_spacing_ratio = 0.005f;
    env.target_edge_length_ratio = 1.0f;
    env.episode_max_length_ratio = 1.5f;
    env.prevent_triangles = true;
    env.export_obj = false;
    env.export_obj_path = "quad_meshing_3d.obj";
    env.reward_invalid = -0.05f;
    env.reward_incomplete = -1.0f;
    env.reward_triangle = -0.5f;
    env.reward_cross_field = true;
    env.base_quad_reward = 0.0f;
    env.potential_beta = 0.15f;
    env.potential_gamma = 1.0f;
    env.frontier_quality_weight = 1.0f;
    env.frontier_edge_length_weight = 1.0f;
    env.frontier_alignment_weight = 1.0f;
    env.frontier_angle_weight = 1.0f;
    env.frontier_size_pressure_weight = 1.0f;
    env.degree_pressure_weight = 1.0f;
    env.safe_frontier_size_ratio = 0.9f;
    env.safe_degree_ratio = 0.7f;
    env.render_show_mesh = true;
    env.render_mesh_opaque = false;
    env.render_show_samples = false;
    env.render_show_vertices = false;
    env.render_show_normals = false;
    env.render_show_cross_field = false;
    env.render_show_graph = true;
    env.render_show_candidates = true;

    quad_meshing_3d_init(&env);
    quad_meshing_3d_load_shape(&env, 0);

    if (check_only) {
        printf("loaded %s\n", path);
        printf("vertices=%u triangles=%u samples=%u frontier_edges=%u area=%.8f sample_density=%.3f sharp_dihedral_deg=%.3f\n",
            env.surface.vertex_count,
            env.surface.triangle_count,
            env.surface.sample_count,
            env.surface.frontier_edge_count,
            env.surface.info.total_area,
            env.surface.info.sample_density,
            env.surface.info.sharp_dihedral_radians * RAD2DEG);
        printf("graph_vertices=%u graph_edges=%u faces=%u quads=%u tris=%u graph_tri_segments=%u graph_tri_vertices=%u frontier_vertices=%u max_degree=%u\n",
            env.mesh.vertex_count,
            env.mesh.edge_count,
            env.mesh.face_count,
            env.mesh.quad_count,
            env.mesh.tri_count,
            env.graph_tri_segment_count,
            env.graph_tri_vertex_count,
            env.mesh.frontier_count,
            env.mesh.max_degree);
        printf("persistent_graph_path_points=%u persistent_graph_path_segments=%u\n",
            env.mesh.edge_path_points.count,
            env.mesh.edge_path_segments.count);
        printf("source_frontier=%d candidate_radius=%.8f candidate_count=%u candidate_samples=%u candidate_existing=%u prop_nodes=%d prop_edges=%d spacing=%.8f\n",
            env.source_frontier_idx,
            env.candidate_radius,
            env.candidate_count,
            env.candidate_sample_count,
            env.candidate_existing_count,
            env.prop_graph.node_count,
            env.prop_graph.edge_count,
            env.prop_graph.spacing);
        printf("continuous_paths_ok=%u valid_candidates=%u invalid_unreachable=%u invalid_intersect=%u target_candidate=%d path_ok=%d path_points=%u path_segments=%u cached_path_points=%u cached_path_segments=%u path_length=%.8f\n",
            env.continuous_paths_ok,
            env.valid_candidate_count,
            env.invalid_unreachable_count,
            env.invalid_intersect_count,
            env.target_candidate_idx,
            env.selected_path_ok ? 1 : 0,
            env.selected_path.count,
            env.target_candidate_idx >= 0 ? env.candidates[env.target_candidate_idx].segment_count : 0,
            env.candidate_path_points.count,
            env.candidate_path_segments.count,
            env.selected_path.length);
        printf("timing_ms targets=%.3f geodesic_paths=%.3f intersections=%.3f intersection_tests=%u total=%.3f\n",
            env.timing_target_query_ms,
            env.timing_path_build_ms,
            env.timing_intersection_ms,
            env.timing_intersection_tests,
            env.timing_target_query_ms + env.timing_path_build_ms + env.timing_intersection_ms);
        QM3_ASSERT(qm3_check_face_registration_debug());
        QM3_ASSERT(qm3_check_loop_subdivision_debug());
        QM3_ASSERT(qm3_check_loop_removal_debug());
        QM3_ASSERT(qm3_check_loop_removal_across_triangles_debug());
        QM3_ASSERT(qm3_check_loop_removal_native_edge_debug());
        QM3_ASSERT(qm3_check_loop_removal_locality_debug());
        c_close(&env);
        return 0;
    }

    c_render(&env);
    while (!WindowShouldClose()) {
        c_render(&env);
    }

    c_close(&env);
    return 0;
}
