#include "quad_meshing_3d.h"

#include <stdio.h>
#include <string.h>

static void usage(const char* exe) {
    fprintf(stderr, "Usage: %s surface.qmsurf [--check]\n", exe);
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
    env.max_degree = 16;
    env.candidate_radius_ratio = 0.05f;
    env.geodesic_steiner_spacing_ratio = 0.005f;
    env.render_show_mesh = true;
    env.render_mesh_opaque = false;
    env.render_show_samples = true;
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
        printf("graph_vertices=%u graph_edges=%u graph_tri_segments=%u graph_tri_vertices=%u frontier_vertices=%u max_degree=%u\n",
            env.mesh.vertex_count,
            env.mesh.edge_count,
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
