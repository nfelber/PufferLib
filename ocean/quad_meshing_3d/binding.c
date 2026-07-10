#include "quad_meshing_3d.h"

#include <dirent.h>
#include <string.h>

#define NUM_ATNS 1
#define ACT_SIZES {1}
#define OBS_TENSOR_T ByteTensor
#define OBS_SIZE 1

#define Env QuadMeshing3DEnv

#include "vecenv.h"

static bool qm3_str_has_suffix(const char* str, const char* suffix) {
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    return str_len >= suffix_len && strcmp(str + str_len - suffix_len, suffix) == 0;
}

static char* qm3_path_join(const char* folder, const char* name) {
    size_t folder_len = strlen(folder);
    size_t name_len = strlen(name);
    bool needs_sep = folder_len > 0 && folder[folder_len - 1] != '/';
    char* path = (char*)calloc(folder_len + needs_sep + name_len + 1, sizeof(char));
    QM3_ASSERT(path != NULL);
    memcpy(path, folder, folder_len);
    if (needs_sep) path[folder_len] = '/';
    memcpy(path + folder_len + needs_sep, name, name_len);
    return path;
}

static int qm3_compare_strings(const void* a, const void* b) {
    const char* sa = *(const char* const*)a;
    const char* sb = *(const char* const*)b;
    return strcmp(sa, sb);
}

static const char** qm3_list_files_with_suffix(const char* folder, const char* suffix, int* out_count) {
    DIR* dir = opendir(folder);
    QM3_ASSERT(dir != NULL);

    int count = 0;
    int capacity = 16;
    const char** paths = (const char**)calloc((size_t)capacity, sizeof(const char*));
    QM3_ASSERT(paths != NULL);

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (!qm3_str_has_suffix(entry->d_name, suffix)) continue;
        if (count == capacity) {
            capacity *= 2;
            paths = (const char**)realloc(paths, (size_t)capacity * sizeof(const char*));
            QM3_ASSERT(paths != NULL);
        }
        paths[count++] = qm3_path_join(folder, entry->d_name);
    }
    closedir(dir);

    qsort(paths, (size_t)count, sizeof(const char*), qm3_compare_strings);
    *out_count = count;
    return paths;
}

static void load_surface_paths_from_folder(Env* env, Dict* kwargs) {
    DictItem* folder_item = dict_get_unsafe(kwargs, "shape_folder");
    QM3_ASSERT(folder_item != NULL && folder_item->ptr != NULL);
    const char* folder = (const char*)folder_item->ptr;

    DictItem* names_item = dict_get_unsafe(kwargs, "shape_names");
    if (names_item != NULL && names_item->value > 0) {
        const char** names = (const char**)names_item->ptr;
        int count = (int)names_item->value;
        const char** paths = (const char**)calloc((size_t)count, sizeof(const char*));
        QM3_ASSERT(paths != NULL);
        for (int i = 0; i < count; ++i) paths[i] = qm3_path_join(folder, names[i]);
        env->shape_paths = paths;
        env->shape_count = count;
        return;
    }

    env->shape_paths = qm3_list_files_with_suffix(folder, ".qmsurf", &env->shape_count);
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->rng ^= (unsigned int)dict_get(kwargs, "seed")->value;
    env->max_degree = (int)dict_get(kwargs, "max_degree")->value;
    env->candidate_radius_ratio = (float)dict_get(kwargs, "candidate_radius_ratio")->value;
    env->geodesic_steiner_spacing_ratio = (float)dict_get(kwargs, "geodesic_steiner_spacing_ratio")->value;
    env->target_edge_length_ratio = (float)dict_get(kwargs, "target_edge_length_ratio")->value;
    env->episode_max_length_ratio = (float)dict_get(kwargs, "episode_max_length_ratio")->value;
    env->prevent_triangles = dict_get(kwargs, "prevent_triangles")->value > 0.5;
    env->reward_invalid = (float)dict_get(kwargs, "reward_invalid")->value;
    env->reward_incomplete = (float)dict_get(kwargs, "reward_incomplete")->value;
    env->reward_triangle = (float)dict_get(kwargs, "reward_triangle")->value;
    env->base_quad_reward = (float)dict_get(kwargs, "base_quad_reward")->value;
    env->potential_beta = (float)dict_get(kwargs, "potential_beta")->value;
    env->potential_gamma = (float)dict_get(kwargs, "potential_gamma")->value;
    env->frontier_quality_weight = (float)dict_get(kwargs, "frontier_quality_weight")->value;
    env->frontier_edge_length_weight = (float)dict_get(kwargs, "frontier_edge_length_weight")->value;
    env->frontier_alignment_weight = (float)dict_get(kwargs, "frontier_alignment_weight")->value;
    env->frontier_angle_weight = (float)dict_get(kwargs, "frontier_angle_weight")->value;
    env->frontier_size_pressure_weight = (float)dict_get(kwargs, "frontier_size_pressure_weight")->value;
    env->degree_pressure_weight = (float)dict_get(kwargs, "degree_pressure_weight")->value;
    env->safe_frontier_size_ratio = (float)dict_get(kwargs, "safe_frontier_size_ratio")->value;
    env->safe_degree_ratio = (float)dict_get(kwargs, "safe_degree_ratio")->value;

    load_surface_paths_from_folder(env, kwargs);
    QM3_ASSERT(env->shape_count > 0);

    env->render_target_fps = (int)dict_get(kwargs, "render_target_fps")->value;
    env->render_width = (int)dict_get(kwargs, "render_width")->value;
    env->render_height = (int)dict_get(kwargs, "render_height")->value;
    env->render_show_mesh = dict_get(kwargs, "render_show_mesh")->value > 0.5;
    env->render_mesh_opaque = dict_get(kwargs, "render_mesh_opaque")->value > 0.5;
    env->render_show_samples = dict_get(kwargs, "render_show_samples")->value > 0.5;
    env->render_show_vertices = dict_get(kwargs, "render_show_vertices")->value > 0.5;
    env->render_show_normals = dict_get(kwargs, "render_show_normals")->value > 0.5;
    env->render_show_cross_field = dict_get(kwargs, "render_show_cross_field")->value > 0.5;
    env->render_show_graph = dict_get(kwargs, "render_show_graph")->value > 0.5;
    env->render_show_graph_paths = dict_get(kwargs, "render_show_graph_paths")->value > 0.5;
    env->render_show_candidates = dict_get(kwargs, "render_show_candidates")->value > 0.5;

    quad_meshing_3d_init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}
