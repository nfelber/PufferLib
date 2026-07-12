"""Interactive PyGame visualizer for deserialize_frontier_obs.

Usage:
    python tests/test_quad_meshing_deserialize.py

Keys:
    Space/Enter  - Step the env with random actions
    R            - Reset environments
    Arrow Right  - Cycle to next agent
    Q            - Quit
"""

import ctypes
import random
import threading
import time

import pygame
import torch

from pufferlib import _C
from pufferlib.models import deserialize_substep0_obs, deserialize_candidates

# ============================================================================
# VecEnv creation
# ============================================================================


def create_vecenv():
    """Create quad_meshing VecEnv matching config/quad_meshing.ini."""
    args = {
        "vec": {"total_agents": 4, "num_buffers": 1, "num_threads": 1},
        "env": {
            "seed": 42,
            "episode_max_length_ratio": 2.0,
            "candidate_rings": 4,
            "candidate_angles": 16,
            "target_edge_length_ratio": 1.0,
            "candidate_radius_min_ratio": 0.16,
            "candidate_radius_max_ratio": 0.03,
            "shape_folder": "./resources/quad_meshing/shapes",
            "shape_names": ["dolphin.qmshape"],
            "boundary_mode": True,
            "prevent_triangles": False,
            "export_obj": False,
            "export_obj_path": "quad_meshing.obj",
            "reward_invalid": -0.1,
            "reward_triangle": -0.5,

            "render_target_fps": 60.0,
            "render_width": 1200.0,
            "render_height": 900.0,
            "render_show_frontier": True,
            "render_show_candidates": True,
            "render_show_cross_field": False,
            "render_show_indices": False,
            "render_line_thickness": 2.0,
            "render_point_radius": 4.0,
            "render_candidate_radius": 3.0,
            "max_degree": 16,
            "grid_res": 8,
            "grid_cell_size": 0.125,
            "grid_cell_cap": 8,
            "intersection_tol": 1e-3,
        },
    }
    vec = _C.create_vec(args)
    vec.reset()
    return vec


# ============================================================================
# Observation extraction (matches PuffeRL.vec_obs)
# ============================================================================

def get_obs(vec, device=None):
    """Extract uint8 obs buffer as torch tensor [total_agents, obs_size].

    Zero-copy via ctypes from the C observation pointer.
    Optionally copy to GPU device.
    """
    n = vec.total_agents * vec.obs_size
    buf = (ctypes.c_uint8 * n).from_address(vec.obs_ptr)
    obs = torch.frombuffer(buf, dtype=torch.uint8)
    obs = obs.reshape(vec.total_agents, vec.obs_size)
    if device is not None:
        obs = obs.to(device)
    return obs


# ============================================================================
# Action construction
# ============================================================================


def make_random_actions(vec):
    """Construct valid random actions by interleaving action selection with substep.

    For quad_meshing (NUM_ATNS=2):
        1. Pre-allocate action buffer, pick source_slot from substep 0 frontier
        2. Advance with gpu_substep to get substep 1 obs
        3. Fill target_slot from substep 1 validity mask + candidates

    Actions are stored as a flat float tensor of shape [total_agents * NUM_ATNS].
    Returns a contiguous torch float32 tensor.
    """
    NUM_ATNS = 2
    actions = torch.zeros(vec.total_agents * NUM_ATNS, dtype=torch.float32)

    # --- Step 1: pick source from substep 0 frontier ---
    obs = get_obs(vec)
    result0 = deserialize_substep0_obs(obs)

    # Use ptr offsets to get frontier sizes per agent
    ptr = result0["ptr"]
    frontier_sizes = []
    sources = []
    for b in range(vec.total_agents):
        fs = int(ptr[b + 1].item() - ptr[b].item())
        frontier_sizes.append(fs)
        if fs > 0:
            sources.append(float(random.randint(0, fs - 1)))
        else:
            sources.append(0.0)

    # Fill source actions and advance to substep 1
    actions[0::2] = torch.tensor(sources, dtype=torch.float32)
    vec.gpu_substep(actions.data_ptr(), 0)

    # --- Step 2: pick target from substep 1 obs ---
    obs1 = get_obs(vec)
    result1 = deserialize_candidates(obs1)

    counts = result1["counts"]
    validity = result1["validity"]

    targets = []
    for b in range(vec.total_agents):
        fs = frontier_sizes[b]
        vc = int(counts[b].item())

        # Count valid frontier nodes for this agent
        ptr_start = int(ptr[b].item())
        ptr_end = int(ptr[b + 1].item())
        agent_valid = validity[ptr_start:ptr_end]
        num_valid = int(agent_valid.sum().item())

        if num_valid == 0 and vc == 0:
            targets.append(float(random.randint(0, max(fs - 1, 0))) if fs > 0 else 0.0)
        elif num_valid > 0 and vc > 0:
            target_choice = random.random()
            if target_choice < num_valid / (num_valid + vc):
                valid_indices = torch.nonzero(agent_valid, as_tuple=True)[0]
                targets.append(float(valid_indices[random.randint(0, len(valid_indices) - 1)].item()))
            else:
                targets.append(float(fs + random.randint(0, vc - 1)))
        elif num_valid > 0:
            valid_indices = torch.nonzero(agent_valid, as_tuple=True)[0]
            targets.append(float(valid_indices[random.randint(0, len(valid_indices) - 1)].item()))
        else:
            targets.append(float(fs + random.randint(0, vc - 1)))

    # Fill target actions
    actions[1::2] = torch.tensor(targets, dtype=torch.float32)

    return actions.contiguous()


# ============================================================================
# PyGame rendering
# ============================================================================


AGENT_COLORS = [
    (255, 100, 100),
    (100, 100, 255),
    (100, 255, 100),
    (255, 200, 100),
]


def render_deserialized(result, agent_idx):
    """Draw one agent's deserialized graph in the PyGame window."""
    screen = pygame.display.get_surface()
    W, H = screen.get_size()

    # Isolate this agent's data
    agent_mask = result["batch"] == agent_idx
    x = result["x"][agent_mask]
    if len(x) == 0:
        return

    # Extract edges for this agent only using ptr offsets
    ptr_start = int(result["ptr"][agent_idx].item())
    ptr_end = int(result["ptr"][agent_idx + 1].item())
    edge_ptr_start = int(result["edge_ptr"][ptr_start].item())
    edge_ptr_end = int(result["edge_ptr"][ptr_end].item())
    src = result["edge_index"][0][edge_ptr_start:edge_ptr_end] - ptr_start
    dst = result["edge_index"][1][edge_ptr_start:edge_ptr_end] - ptr_start
    num_edges = len(src)

    # Transform coordinates to screen space
    x_min, y_min = x.min(dim=0)[0]
    x_max, y_max = x.max(dim=0)[0]

    # Auto-scale to fit window with 5% padding
    range_x = (x_max - x_min).item() if x_max != x_min else 1.0
    range_y = (y_max - y_min).item() if y_max != y_min else 1.0
    margin = max(range_x, range_y) * 0.05

    scale = W / max(range_x + 2 * margin, 1e-9)
    scale = min(scale, H / max(range_y + 2 * margin, 1e-9))

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    dx = (x[:, 0] - cx) * scale + W / 2
    dy = -(x[:, 1] - cy) * scale + H / 2  # flip y for screen coords

    # Draw edges
    for e in range(num_edges):
        s = int(src[e].item())
        d = int(dst[e].item())
        sx, sy = int(dx[s].item()), int(dy[s].item())
        ex, ey = int(dx[d].item()), int(dy[d].item())
        pygame.draw.line(screen, (128, 128, 128), (sx, sy), (ex, ey), 1)

    # Draw vertices
    color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
    radius = 4
    for i in range(len(x)):
        pygame.draw.circle(screen, color, (int(dx[i].item()), int(dy[i].item())), radius)

    # Draw label
    label = f"Agent {agent_idx}: {len(x)} nodes, {num_edges} edges"
    font = pygame.font.SysFont(None, 20)
    text = font.render(label, True, (255, 255, 255))
    screen.blit(text, (10, 10))


# ============================================================================
# Raylib render thread (runs in background with its own OpenGL context)
# ============================================================================


class RaylibRenderThread(threading.Thread):
    """Background thread that owns the raylib window and continuously renders."""

    def __init__(self, vec, target_fps=30):
        super().__init__(daemon=True)
        self.vec = vec
        self.target_fps = target_fps
        self._stop_event = threading.Event()
        self.agent_idx = 0

    def run(self):
        # Raylib context lives exclusively in this thread
        # Set FPS target
        frame_time = 1.0 / self.target_fps
        while not self._stop_event.is_set():
            self.vec.render(self.agent_idx)
            time.sleep(frame_time)

    def stop(self):
        self._stop_event.set()

    def update_agent_idx(self, idx):
        self.agent_idx = idx


# ============================================================================
# Main interactive loop
# ============================================================================


def test_deserialize_frontier_obs():
    """Interactive PyGame viewer for deserialize_frontier_obs.

    Press:
      Space/Enter  - Step the env with random actions
      R            - Reset environments
      Arrow Right  - Advance agent index (without stepping env)
      Q            - Quit
    """
    vec = create_vecenv()

    # Start raylib render thread (owns its own OpenGL context)
    raylib_thread = RaylibRenderThread(vec, target_fps=30)
    raylib_thread.start()

    pygame.init()
    screen = pygame.display.set_mode((1200, 900))
    pygame.mouse.set_visible(False)

    running = True
    step_count = 0
    agent_idx = 0

    while running:
        screen.fill((20, 20, 20))

        # Get observations and deserialize on GPU
        obs = get_obs(vec, device='cuda')
        result = deserialize_substep0_obs(obs)

        # Render one agent in pygame
        render_deserialized(result, agent_idx)

        # Draw status bar
        status = f"Step: {step_count}  |  Press Space: step  |  R: reset  |  Q: quit"
        font = pygame.font.SysFont(None, 18)
        text = font.render(status, True, (180, 180, 180))
        screen.blit(text, (10, H := screen.get_height() - 30))

        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    vec.reset()
                    step_count = 0
                elif event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                    actions = make_random_actions(vec)
                    vec.gpu_step(actions.data_ptr())
                    step_count += 1
                elif event.key == pygame.K_RIGHT:
                    agent_idx = (agent_idx + 1) % vec.total_agents
                    raylib_thread.update_agent_idx(agent_idx)

        pygame.time.Clock().tick(60)

    raylib_thread.stop()
    raylib_thread.join()
    pygame.quit()
    vec.close()


if __name__ == "__main__":
    test_deserialize_frontier_obs()
