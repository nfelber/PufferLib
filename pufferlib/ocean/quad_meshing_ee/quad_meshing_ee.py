import json

import pufferlib
import pufferlib.emulation


def _load_boundaries(boundary_file):
    from shapely.geometry import Polygon, MultiPolygon

    with open(boundary_file, 'r') as f:
        boundary_dict = json.load(f)
    return [MultiPolygon([Polygon(b["coords"])]) for b in boundary_dict]


def make_quad_meshing_ee(
    boundary_file=None,
    boundaries=None,
    neighbor_count=3,
    fov_point_count=5,
    boundary_idx=None,
    render_mode=None,
    buf=None,
    **kwargs,
):
    if boundaries is None:
        if boundary_file is None:
            raise ValueError("boundary_file or boundaries must be provided")
        boundaries = _load_boundaries(boundary_file)

    if boundary_idx is not None:
        boundaries = [boundaries[boundary_idx]]

    from .env.free_mesh_rl_env import FreeMeshRLEnv

    env = FreeMeshRLEnv(
        neighbor_count=neighbor_count,
        fov_point_count=fov_point_count,
        boundaries=boundaries,
        render_mode=render_mode,
    )
    env = pufferlib.ClipAction(env)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)
