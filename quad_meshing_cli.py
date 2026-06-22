import argparse
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from pufferlib import pufferl
from pufferlib.ocean.quad_meshing.quad_meshing import QuadMeshing


def _help_text(cartesian_actions: bool, edge_mode: bool) -> str:
    if edge_mode:
        if cartesian_actions:
            add1 = "  3 <x> <y>          -> add_left (x in [0,1], y in [-1,1])\n"
            add2 = "  4 <x> <y>          -> add_right (x in [0,1], y in [-1,1])\n"
            add_two = "  5 <x1> <y1> <x2> <y2> -> add_two\n"
        else:
            add1 = "  3 <angle> <radius> -> add_left (angle in [-1,1], radius in [0,1])\n"
            add2 = "  4 <angle> <radius> -> add_right (angle in [-1,1], radius in [0,1])\n"
            add_two = "  5 <a1> <r1> <a2> <r2> -> add_two\n"
        return (
            "Controls (edge mode):\n"
            "  0                  -> close_left2\n"
            "  1                  -> close_left_right\n"
            "  2                  -> close_right2\n"
            f"{add1}"
            f"{add2}"
            f"{add_two}"
            "  r                  -> reset\n"
            "  h                  -> help\n"
            "  q                  -> quit\n"
        )
    if cartesian_actions:
        place = "  2 <x> <y>          -> place_vertex (local-frame x in [0,1], y in [-1,1])\n"
    else:
        place = "  2 <angle> <radius> -> place_vertex (angle in [-1,1], radius in [0,1])\n"
    return (
        "Controls:\n"
        "  0                  -> close_left\n"
        "  1                  -> close_right\n"
        f"{place}"
        "  r                  -> reset\n"
        "  h                  -> help\n"
        "  q                  -> quit\n"
    )


def _parse_command(line, cartesian_actions: bool, edge_mode: bool):
    if not line:
        return ("noop", None)
    text = line.strip()
    if not text:
        return ("noop", None)

    if text in {"q", "quit", "exit"}:
        return ("quit", None)
    if text in {"r", "reset"}:
        return ("reset", None)
    if text in {"h", "help", "?"}:
        return ("help", None)

    parts = text.split()
    if edge_mode:
        if parts[0] in {"0", "1", "2"} and len(parts) == 1:
            choice = float(parts[0])
            return ("action", np.array([[choice, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        if parts[0] in {"3", "4"}:
            if len(parts) != 3:
                if cartesian_actions:
                    return ("error", f"action {parts[0]} requires: {parts[0]} <x> <y>")
                return ("error", f"action {parts[0]} requires: {parts[0]} <angle> <radius>")
            try:
                v1 = float(parts[1])
                v2 = float(parts[2])
            except ValueError:
                if cartesian_actions:
                    return ("error", "x and y must be numbers")
                return ("error", "angle and radius must be numbers")
            choice = float(parts[0])
            return ("action", np.array([[choice, v1, v2, 0.0, 0.0]], dtype=np.float32))
        if parts[0] in {"5"}:
            if len(parts) != 5:
                if cartesian_actions:
                    return ("error", "action 5 requires: 5 <x1> <y1> <x2> <y2>")
                return ("error", "action 5 requires: 5 <a1> <r1> <a2> <r2>")
            try:
                v1 = float(parts[1])
                v2 = float(parts[2])
                v3 = float(parts[3])
                v4 = float(parts[4])
            except ValueError:
                if cartesian_actions:
                    return ("error", "x/y values must be numbers")
                return ("error", "angle/radius values must be numbers")
            return ("action", np.array([[5.0, v1, v2, v3, v4]], dtype=np.float32))
    if parts[0] in {"0", "1"} and len(parts) == 1:
        choice = float(parts[0])
        return ("action", np.array([[choice, 0.0, 0.0]], dtype=np.float32))

    if parts[0] in {"2", "place"}:
        if parts[0] == "place":
            parts = parts[1:]
        else:
            parts = parts[1:]
        if len(parts) != 2:
            if cartesian_actions:
                return ("error", "place_vertex requires: 2 <x> <y>")
            return ("error", "place_vertex requires: 2 <angle> <radius>")
        try:
            value1 = float(parts[0])
            value2 = float(parts[1])
        except ValueError:
            if cartesian_actions:
                return ("error", "x and y must be numbers")
            return ("error", "angle and radius must be numbers")
        return ("action", np.array([[2.0, value1, value2]], dtype=np.float32))

    return ("error", "unknown command; type 'h' for help")


def _render_loop(env, lock, stop_event, render_interval):
    while not stop_event.is_set():
        start = time.time()
        with lock:
            env.render()
        elapsed = time.time() - start
        if elapsed < render_interval:
            time.sleep(render_interval - elapsed)


@contextmanager
def _clean_argv():
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        yield
    finally:
        sys.argv = saved


def _load_quad_meshing_config():
    config_path = Path(pufferl.__file__).resolve().parent / "config/ocean/quad_meshing.ini"
    if not config_path.exists():
        raise FileNotFoundError(f"Quad meshing config not found: {config_path}")
    with _clean_argv():
        args = pufferl.load_config_file(str(config_path))
    return args


def main():
    args = _load_quad_meshing_config()
    env_cfg = args["env"]
    cartesian_actions = bool(env_cfg.get("cartesian_actions", False))
    edge_mode = bool(env_cfg.get("edge_mode", False))

    parser = argparse.ArgumentParser(
        description="Manual CLI control for the quad_meshing environment."
    )
    parser.add_argument("--boundary-file", type=str, default=None)
    cli_args = parser.parse_args()
    boundary_files = None
    if cli_args.boundary_file:
        boundary_files = [cli_args.boundary_file]

    env = QuadMeshing(
        num_envs=1,
        boundary_files=boundary_files or env_cfg.get("boundary_files"),
        random_active_vertex=env_cfg.get("random_active_vertex", False),
        observe_remaining_area=env_cfg.get("observe_remaining_area", False),
        observe_boundary_cost=env_cfg.get("observe_boundary_cost", False),
        observation_radius=env_cfg.get("observation_radius", 1.0),
        n_neighbors=env_cfg.get("n_neighbors", 0),
        n_sdf_samples=env_cfg.get("n_sdf_samples", 0),
        action_radius=env_cfg.get("action_radius", 1.0),
        cartesian_actions=cartesian_actions,
        fixed_local_radius=env_cfg.get("fixed_local_radius", 0.0),
        edge_mode=edge_mode,
        delayed_rewards=env_cfg.get("delayed_rewards", False),
        export_meshes=env_cfg.get("export_meshes", False),
        export_mesh_path=env_cfg.get("export_mesh_path", "mesh.obj"),
        render_enabled=True,
    )
    env.reset()

    stop_event = threading.Event()
    env_lock = threading.Lock()
    render_interval = 1.0 / 30.0
    render_thread = threading.Thread(
        target=_render_loop,
        args=(env, env_lock, stop_event, render_interval),
        daemon=True,
    )
    render_thread.start()

    try:
        print(_help_text(cartesian_actions, edge_mode))
        while not stop_event.is_set():
            try:
                line = input("action> ")
            except EOFError:
                stop_event.set()
                break

            cmd, data = _parse_command(line, cartesian_actions, edge_mode)
            if cmd == "noop":
                continue
            if cmd == "quit":
                stop_event.set()
                break
            if cmd == "help":
                print(_help_text(cartesian_actions, edge_mode))
                continue
            if cmd == "reset":
                with env_lock:
                    env.reset()
                print("Environment reset")
                continue
            if cmd == "error":
                print(f"Error: {data}")
                continue
            if cmd == "action":
                with env_lock:
                    obs, rewards, terminals, truncs, info = env.step(data)
                reward = float(rewards[0])
                done = bool(terminals[0])
                truncated = bool(truncs[0])
                print(f"reward={reward:.4f} done={done} truncated={truncated}")
                if info:
                    print(f"info: {info}")
                if done or truncated:
                    with env_lock:
                        env.reset()
                    print("Episode ended; environment reset")
    finally:
        stop_event.set()
        env.close()


if __name__ == "__main__":
    main()
