#include "quad_meshing.h"

#define Env QuadMeshing 
#include "../env_binding.h"

static void free_boundary_sets(Env* env) {
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
    env->boundary_set_index = 0;
}

static int load_boundary_sets_from_list(Env* env, PyObject* boundary_list) {
    if (!PyList_Check(boundary_list)) {
        PyErr_SetString(PyExc_ValueError, "boundary_vertices_list must be a list");
        return -1;
    }

    int list_count = PyList_Size(boundary_list);
    if (list_count <= 0) {
        return 0;
    }

    env->boundary_sets = (float**)calloc(list_count, sizeof(float*));
    env->boundary_set_sizes = (int*)calloc(list_count, sizeof(int));
    if (!env->boundary_sets || !env->boundary_set_sizes) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate boundary set storage");
        free_boundary_sets(env);
        return -1;
    }
    env->boundary_set_count = list_count;
    env->boundary_set_index = 0;

    for (int i = 0; i < list_count; ++i) {
        PyObject* boundary_obj = PyList_GetItem(boundary_list, i);
        if (!PyList_Check(boundary_obj)) {
            PyErr_SetString(PyExc_ValueError, "Each boundary must be a list of vertices");
            free_boundary_sets(env);
            return -1;
        }
        int boundary_vertices_count = PyList_Size(boundary_obj);
        if (boundary_vertices_count < 3) {
            PyErr_SetString(PyExc_ValueError, "Each boundary must have at least 3 vertices");
            free_boundary_sets(env);
            return -1;
        }
        float* vertices = (float*)malloc(boundary_vertices_count * 2 * sizeof(float));
        if (!vertices) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate boundary vertices");
            free_boundary_sets(env);
            return -1;
        }
        for (int j = 0; j < boundary_vertices_count; ++j) {
            PyObject* vertex = PyList_GetItem(boundary_obj, j);
            if (!PyList_Check(vertex) || PyList_Size(vertex) != 2) {
                PyErr_SetString(PyExc_ValueError, "Each vertex must be a [x, y] list");
                free(vertices);
                free_boundary_sets(env);
                return -1;
            }

            PyObject* x_obj = PyList_GetItem(vertex, 0);
            PyObject* y_obj = PyList_GetItem(vertex, 1);

            if ((!PyFloat_Check(x_obj) && !PyLong_Check(x_obj)) ||
                (!PyFloat_Check(y_obj) && !PyLong_Check(y_obj))) {
                PyErr_SetString(PyExc_ValueError, "Vertex coordinates must be numbers");
                free(vertices);
                free_boundary_sets(env);
                return -1;
            }

            vertices[2*j] = PyFloat_AsDouble(x_obj);
            vertices[2*j+1] = PyFloat_AsDouble(y_obj);
        }
        env->boundary_sets[i] = vertices;
        env->boundary_set_sizes[i] = boundary_vertices_count;
    }

    return 0;
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->random_active_vertex = unpack(kwargs, "random_active_vertex");

    env->observe_remaining_area = unpack(kwargs, "observe_remaining_area");
    env->observe_local_radius = unpack(kwargs, "observe_local_radius");
    env->observe_boundary_cost = unpack(kwargs, "observe_boundary_cost");
    env->observation_radius = unpack(kwargs, "observation_radius");
    env->n_neighbors = unpack(kwargs, "n_neighbors");
    env->n_sdf_samples = unpack(kwargs, "n_sdf_samples");

    env->action_radius = unpack(kwargs, "action_radius");
    env->cartesian_actions = unpack(kwargs, "cartesian_actions");
    env->fixed_local_radius = unpack(kwargs, "fixed_local_radius");
    env->edge_mode = unpack(kwargs, "edge_mode");

    env->delayed_rewards = unpack(kwargs, "delayed_rewards");

    env->render_enabled = unpack(kwargs, "render_enabled");
    env->render_target_fps = unpack(kwargs, "render_target_fps");
    env->export_meshes = unpack(kwargs, "export_meshes");

    const char* default_export_path = "mesh.obj";
    const char* export_path = default_export_path;
    PyObject* export_path_obj = PyDict_GetItemString(kwargs, "export_mesh_path");
    if (export_path_obj) {
        if (!PyUnicode_Check(export_path_obj)) {
            PyErr_SetString(PyExc_ValueError, "export_mesh_path must be a string");
            return -1;
        }
        export_path = PyUnicode_AsUTF8(export_path_obj);
    }
    if (!export_path || export_path[0] == '\0') {
        export_path = default_export_path;
    }
    strncpy(env->export_mesh_path, export_path, sizeof(env->export_mesh_path) - 1);
    env->export_mesh_path[sizeof(env->export_mesh_path) - 1] = '\0';
    
    // Extract boundary vertices list if provided
    float* boundary_vertices = NULL;
    int num_vertices = 0;

    PyObject* boundary_list_obj = PyDict_GetItemString(kwargs, "boundary_vertices_list");
    if (boundary_list_obj) {
        if (load_boundary_sets_from_list(env, boundary_list_obj) < 0) {
            return -1;
        }
        if (env->boundary_set_count > 0) {
            boundary_vertices = env->boundary_sets[0];
            num_vertices = env->boundary_set_sizes[0];
        }
    }

    init(env, boundary_vertices, num_vertices);
    
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
