'''An environment for learning optimal 2D quadrilateral mesh generation policies.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.quad_meshing import binding


class QuadMeshing(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, observation_density=8, observation_radius=0.3,
                 action_radius=0.3, render_mode=None, log_interval=128, buf=None, seed=0):
        '''
        Initialize the QuadMeshing environment.
        
        Args:
            num_envs: Number of parallel environments to run
            observation_density: NxN grid size for SDF observations (default 32 = 1024 floats)
            observation_radius: Spatial extent of observation region around active vertex
            action_radius: Maximum radius multiplier for placing new vertices
            render_mode: Rendering mode (not fully supported yet)
            log_interval: Number of steps between log reports
            buf: Optional pre-allocated observation buffer
            seed: Random seed
        '''
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.tick = 0
        
        # Observation space: observation_density x observation_density float32 SDF values
        num_obs = observation_density * observation_density
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0,
            shape=(num_obs,),
            dtype=np.float32
        )
        
        # Action space: 3D continuous
        # action[0]: kind in [-1, 1] (-1=close_left, 0=place_vertex, 1=close_right)
        # action[1]: angle in [-1, 1] for vertex placement
        # action[2]: radius multiplier in [0, 1]
        self.single_action_space = gymnasium.spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0,  1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        super().__init__(buf)
        
        # Initialize C environments with shared numpy buffers
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            observation_density=observation_density,
            observation_radius=observation_radius,
            action_radius=action_radius,
        )
    
    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed if seed is not None else 0)
        return self.observations, []
    
    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        
        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)
        
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )
    
    def render(self):
        binding.vec_render(self.c_envs, 0)
    
    def close(self):
        binding.vec_close(self.c_envs)


def test_performance(timeout=10, num_envs=4096):
    '''Benchmark environment performance.'''
    env = QuadMeshing(num_envs=num_envs)
    env.reset()
    
    import time
    start = time.time()
    steps = 0
    
    while time.time() - start < timeout:
        actions = np.random.uniform(
            low=env.single_action_space.low,
            high=env.single_action_space.high,
            size=(num_envs, 3)
        ).astype(np.float32)
        env.step(actions)
        steps += 1
    
    elapsed = time.time() - start
    sps = num_envs * steps / elapsed
    print(f'QuadMeshing SPS: {sps:,.0f}')
    env.close()


if __name__ == '__main__':
    test_performance()
