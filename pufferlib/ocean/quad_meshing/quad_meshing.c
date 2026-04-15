#include "quad_meshing.h"
#include <stdlib.h>

int main() {
    QuadMeshing env;
    env.action_radius = 0.3;
    env.observation_radius = 1.0;
    env.observation_density = 32;
    env.sdf_accel_resolution = 64;

    init(&env, NULL, 0);

    env.observations = (float*)calloc(
        env.observation_density*env.observation_density, sizeof(float));
    env.actions = (float*)calloc(3, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_A)) {
                env.actions[0] = 0.0;
            } else if (IsKeyDown(KEY_D)) {
                env.actions[0] = 1.0;
            } else if (IsKeyDown(KEY_W)) {
                env.actions[0] = 2.0;
                env.actions[1] = 0.0;
                env.actions[2] = 0.58;
            }
        } else {
            env.actions[0] = ((float)rand() / (float)RAND_MAX) * 2.0 - 1.0;
            env.actions[1] = (float)rand() / (float)RAND_MAX;
            env.actions[2] = (float)rand() / (float)RAND_MAX;
        }
        c_step(&env);
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}
