'''An environment for learning optimal 2D quadrilateral mesh generation policies.'''

import gymnasium
import numpy as np
import json
import os

import pufferlib
from pufferlib.ocean.quad_meshing import binding


class QuadMeshing(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, buf=None, seed=0,
                 observation_density=8, observation_radius=0.3, action_radius=0.3,
                 boundary_file=None, random_active_vertex=False, delayed_rewards=False,
                 render_enabled=False, render_target_fps=60,
                 export_meshes=False, export_mesh_path="mesh.obj"):
        '''
        Initialize the QuadMeshing environment.
        
        Args:
            num_envs: Number of parallel environments to run
            render_mode: Rendering mode (not fully supported yet)
            log_interval: Number of steps between log reports
            buf: Optional pre-allocated observation buffer
            seed: Random seed
            observation_density: NxN grid size for SDF observations (default 32 = 1024 floats)
            observation_radius: Spatial extent of observation region around active vertex
            action_radius: Maximum radius multiplier for placing new vertices
            boundary_file: Path to JSON file containing boundary vertices. If not provided,
                           defaults to a hard-coded square boundary. JSON format:
                           {"vertices": [[x1, y1], [x2, y2], ...]}
            random_active_vertex: If True, the active vertex is chosen randomly at each step.
            delayed_rewards: If True, only emit reward when the mesh is completed.
            render_enabled: If True, render-related work is enabled in the C env.
            render_target_fps: Target FPS used for rendering.
            export_meshes: If True, export OBJ mesh files when episodes end.
            export_mesh_path: Output path template for OBJ files. Use "{episode}"
                              to include the episode index in the filename.
         '''
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.tick = 0
        
        # Load boundary from file if provided
        boundary_vertices = None
        if boundary_file:
            boundary_file_path = boundary_file
            if not os.path.isabs(boundary_file_path):
                # Try to find relative to current working directory
                if not os.path.exists(boundary_file_path):
                    # Try relative to this file's directory
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    boundary_file_path = os.path.join(script_dir, boundary_file)
            
            if not os.path.exists(boundary_file_path):
                raise FileNotFoundError(f"Boundary file not found: {boundary_file}")
            
            try:
                with open(boundary_file_path, 'r') as f:
                    data = json.load(f)
                    if 'vertices' not in data:
                        raise ValueError("JSON must contain 'vertices' key with list of [x, y] coordinates")
                    boundary_vertices = data['vertices']
                    if not isinstance(boundary_vertices, list) or len(boundary_vertices) < 3:
                        raise ValueError("Boundary must have at least 3 vertices")
                    for vertex in boundary_vertices:
                        if not isinstance(vertex, list) or len(vertex) != 2:
                            raise ValueError("Each vertex must be a [x, y] list")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in boundary file: {e}")
        
        # Observation space: observation_density x observation_density float32 SDF values
        # num_obs = observation_density * observation_density
        num_obs = 13
        # num_obs = 12
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0,
            shape=(num_obs,),
            dtype=np.float32
        )
        
        # Action space: Hybrid (1 discrete + 2 continuous)
        # Stored as 3D continuous Box for buffer compatibility
        # action[0]: discrete action choice (0-2, stored as float 0.0, 1.0, or 2.0)
        #   - 0 = close_left
        #   - 1 = close_right
        #   - 2 = place_vertex
        # action[1]: angle in [-1, 1] for vertex placement (continuous)
        # action[2]: radius multiplier in [0, 1] (continuous)
        self.single_action_space = gymnasium.spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([2.0,  1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        # self.single_action_space = gymnasium.spaces.Box(
        #     low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
        #     high=np.array([1.0,  1.0, 1.0], dtype=np.float32),
        #     dtype=np.float32
        # )
        
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
            boundary_vertices=boundary_vertices,
            random_active_vertex=random_active_vertex,
            delayed_rewards=delayed_rewards,
            render_enabled=render_enabled,
            render_target_fps=render_target_fps,
            export_meshes=export_meshes,
            export_mesh_path=export_mesh_path,
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
