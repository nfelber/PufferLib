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
    env.max_degree = 8;
    env.candidate_radius_ratio = 0.05f;
    env.geodesic_steiner_spacing_ratio = 0.005f;
    env.target_edge_length_ratio = 1.0f;
    env.reward_invalid = -0.05f;
    env.reward_incomplete = -1.0f;
    env.reward_triangle = -0.5f;
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
