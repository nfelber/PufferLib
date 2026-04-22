#include "quad_meshing.h"

#define Env QuadMeshing 
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->random_active_vertex = unpack(kwargs, "random_active_vertex");

    env->observe_remaining_area = unpack(kwargs, "observe_remaining_area");
    env->observation_radius = unpack(kwargs, "observation_radius");
    env->n_neighbors = unpack(kwargs, "n_neighbors");
    env->n_sdf_samples = unpack(kwargs, "n_sdf_samples");

    env->action_radius = unpack(kwargs, "action_radius");
    env->cartesian_actions = unpack(kwargs, "cartesian_actions");
    env->fixed_local_radius = unpack(kwargs, "fixed_local_radius");

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
    
    // Extract boundary vertices if provided
    float* boundary_vertices = NULL;
    int num_vertices = 0;
    
    PyObject* boundary_obj = PyDict_GetItemString(kwargs, "boundary_vertices");
    if (boundary_obj && PyList_Check(boundary_obj)) {
        num_vertices = PyList_Size(boundary_obj);
        if (num_vertices >= 3) {
            boundary_vertices = (float*)malloc(num_vertices * 2 * sizeof(float));
            if (!boundary_vertices) {
                PyErr_SetString(PyExc_MemoryError, "Failed to allocate boundary vertices");
                return -1;
            }
            
            for (int i = 0; i < num_vertices; ++i) {
                PyObject* vertex = PyList_GetItem(boundary_obj, i);
                if (!PyList_Check(vertex) || PyList_Size(vertex) != 2) {
                    PyErr_SetString(PyExc_ValueError, "Each vertex must be a [x, y] list");
                    free(boundary_vertices);
                    return -1;
                }
                
                PyObject* x_obj = PyList_GetItem(vertex, 0);
                PyObject* y_obj = PyList_GetItem(vertex, 1);
                
                if (!PyFloat_Check(x_obj) && !PyLong_Check(x_obj)) {
                    PyErr_SetString(PyExc_ValueError, "Vertex coordinates must be numbers");
                    free(boundary_vertices);
                    return -1;
                }
                
                boundary_vertices[2*i] = PyFloat_AsDouble(x_obj);
                boundary_vertices[2*i+1] = PyFloat_AsDouble(y_obj);
            }
        }
    }
    
    init(env, boundary_vertices, num_vertices);
    
    if (boundary_vertices) {
        free(boundary_vertices);
    }
    
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
