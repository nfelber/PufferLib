import argparse
import threading
import time

import numpy as np

from pufferlib.ocean.quad_meshing.quad_meshing import QuadMeshing


HELP_TEXT = (
    "Controls:\n"
    "  0                  -> close_left\n"
    "  1                  -> close_right\n"
    "  2 <angle> <radius> -> place_vertex (angle in [-1,1], radius in [0,1])\n"
    "  r                  -> reset\n"
    "  h                  -> help\n"
    "  q                  -> quit\n"
)


def _parse_command(line):
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
    if parts[0] in {"0", "1"} and len(parts) == 1:
        choice = float(parts[0])
        return ("action", np.array([[choice, 0.0, 0.0]], dtype=np.float32))

    if parts[0] in {"2", "place"}:
        if parts[0] == "place":
            parts = parts[1:]
        else:
            parts = parts[1:]
        if len(parts) != 2:
            return ("error", "place_vertex requires: 2 <angle> <radius>")
        try:
            angle = float(parts[0])
            radius = float(parts[1])
        except ValueError:
            return ("error", "angle and radius must be numbers")
        return ("action", np.array([[2.0, angle, radius]], dtype=np.float32))

    return ("error", "unknown command; type 'h' for help")


def _render_loop(env, lock, stop_event, render_interval):
    while not stop_event.is_set():
        start = time.time()
        with lock:
            env.render()
        elapsed = time.time() - start
        if elapsed < render_interval:
            time.sleep(render_interval - elapsed)


def main():
    parser = argparse.ArgumentParser(
        description="Manual CLI control for the quad_meshing environment."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observation-density", type=int, default=8)
    parser.add_argument("--observation-radius", type=float, default=0.3)
    parser.add_argument("--action-radius", type=float, default=0.06)
    parser.add_argument("--boundary-file", type=str, default=None)
    parser.add_argument("--delayed-rewards", action="store_true")
    parser.add_argument("--render-fps", type=float, default=60.0)
    parser.add_argument("--log-interval", type=int, default=128)
    args = parser.parse_args()

    env = QuadMeshing(
        num_envs=1,
        observation_density=args.observation_density,
        observation_radius=args.observation_radius,
        action_radius=args.action_radius,
        boundary_file=args.boundary_file,
        delayed_rewards=args.delayed_rewards,
        log_interval=args.log_interval,
        seed=args.seed,
        render_enabled=True,
    )
    env.reset(seed=args.seed)

    stop_event = threading.Event()
    env_lock = threading.Lock()
    render_interval = 1.0 / max(args.render_fps, 1.0)
    render_thread = threading.Thread(
        target=_render_loop,
        args=(env, env_lock, stop_event, render_interval),
        daemon=True,
    )
    render_thread.start()

    try:
        print(HELP_TEXT)
        while not stop_event.is_set():
            try:
                line = input("action> ")
            except EOFError:
                stop_event.set()
                break

            cmd, data = _parse_command(line)
            if cmd == "noop":
                continue
            if cmd == "quit":
                stop_event.set()
                break
            if cmd == "help":
                print(HELP_TEXT)
                continue
            if cmd == "reset":
                with env_lock:
                    env.reset(seed=args.seed)
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
                        env.reset(seed=args.seed)
                    print("Episode ended; environment reset")
    finally:
        stop_event.set()
        env.close()


if __name__ == "__main__":
    main()
