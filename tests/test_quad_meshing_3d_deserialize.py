import ast
import configparser
import ctypes

import torch

from pufferlib import _C
from pufferlib.models import deserialize_observation_3d


def _parse_value(value):
    value = value.strip()
    if value in ("True", "False"):
        return value == "True"
    if value.startswith("[") or value.startswith("'") or value.startswith('"'):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_config(path="config/quad_meshing_3d.ini"):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(path)
    return {
        section: {key: _parse_value(value) for key, value in cfg[section].items()}
        for section in cfg.sections()
    }


def get_obs(vec):
    n = vec.total_agents * vec.obs_size
    buf = (ctypes.c_uint8 * n).from_address(vec.obs_ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).reshape(vec.total_agents, vec.obs_size)


def test_quad_meshing_3d_deserialize():
    if not torch.cuda.is_available():
        raise RuntimeError("quad_meshing_3d deserializer test requires CUDA")
    if _C.env_name != "quad_meshing_3d":
        raise RuntimeError("Build quad_meshing_3d first: ./build.sh quad_meshing_3d --float")

    args = load_config()
    max_degree = int(args["env"]["max_degree"])
    max_frontier = int(args["env"]["max_frontier"])
    max_candidates = int(args["env"]["max_candidates"])
    vec = _C.create_vec(args, 0)
    try:
        vec.reset()
        valid_batch_idx = torch.arange(vec.total_agents, device="cuda", dtype=torch.long)

        source_obs = get_obs(vec).to("cuda")
        source_graph = deserialize_observation_3d(
            source_obs,
            valid_batch_idx,
            D=max_degree,
            F_CAP=max_frontier,
            T_CAP=max_candidates,
            deserialize_targets=False,
            use_cuda_graph=False,
        )

        source_frontier_count = int(source_graph.batch_offsets[-1].item())
        assert int(source_obs[0, 0].item()) == 0
        assert source_frontier_count > 0
        assert source_graph.vertices.ndim == 2 and source_graph.vertices.shape[1] == 3
        assert source_graph.normals.shape == source_graph.vertices.shape
        assert torch.isfinite(source_graph.vertices).all()
        assert torch.isfinite(source_graph.normals).all()
        normal_lengths = torch.linalg.vector_norm(source_graph.normals, dim=-1)
        assert torch.allclose(normal_lengths, torch.ones_like(normal_lengths), atol=1e-3, rtol=1e-3)

        actions = torch.zeros(vec.total_agents * vec.num_atns, dtype=torch.float32)
        vec.cpu_step(actions.data_ptr())

        target_obs = get_obs(vec).to("cuda")
        target_graph, targets = deserialize_observation_3d(
            target_obs,
            valid_batch_idx,
            D=max_degree,
            F_CAP=max_frontier,
            T_CAP=max_candidates,
            deserialize_targets=True,
            use_cuda_graph=False,
        )

        target_count = int(targets.target_batch_offsets[-1].item())
        assert int(target_obs[0, 0].item()) == 1
        assert target_graph.vertices.shape[1] == 3
        assert targets.target_positions.ndim == 2 and targets.target_positions.shape[1] == 3
        assert targets.target_normals.shape == targets.target_positions.shape
        assert target_count >= 0
        assert torch.isfinite(targets.target_positions).all()
        assert torch.isfinite(targets.target_normals).all()
        assert torch.isfinite(targets.path_lengths).all()
        assert (targets.path_lengths >= 0).all()
        assert targets.target_frontier_parity.shape == targets.path_lengths.shape
        assert targets.target_frontier_parity.dtype == torch.bool

        print(
            "quad_meshing_3d deserialize ok:",
            f"source_frontier_count={source_frontier_count}",
            f"target_frontier_count={int(target_graph.batch_offsets[-1].item())}",
            f"target_count={target_count}",
        )
    finally:
        vec.close()


if __name__ == "__main__":
    test_quad_meshing_3d_deserialize()
