#include "quad_meshing.h"

#define Env QuadMeshing 
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->observation_density = unpack(kwargs, "observation_density");
    env->observation_radius = unpack(kwargs, "observation_radius");
    env->action_radius = unpack(kwargs, "action_radius");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
