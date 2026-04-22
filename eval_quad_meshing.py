import argparse
import json
import subprocess
import sys
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import meshio
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import polygonize, triangulate

import pufferlib
from pufferlib import pufferl

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required to load eval configs. Install with `pip install PyYAML`."
    ) from exc


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def _resolve_path(path_value, base_dir: Path) -> Optional[Path]:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_model_path(value: str, base_dir: Path) -> str:
    if value == "latest":
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path.resolve())
    return str((base_dir / path).resolve())


def _load_eval_config(path: Path) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _load_boundary_vertices(path: Path) -> list[tuple[float, float]]:
    with open(path, "r") as handle:
        data = json.load(handle)
    vertices = data.get("vertices")
    if not isinstance(vertices, list) or len(vertices) < 3:
        raise ValueError("Boundary JSON must contain a 'vertices' list with at least 3 points")
    return [(float(x), float(y)) for x, y in vertices]


def _triangulate_boundary_to_obj(boundary_file: Path, out_path: Path) -> None:
    coords = _load_boundary_vertices(boundary_file)
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not poly.is_valid or poly.is_empty:
        raise ValueError(f"Boundary polygon is invalid after fixing: {boundary_file}")

    triangles = [tri for tri in triangulate(poly) if tri.centroid.within(poly)]
    vertex_index: dict[tuple[float, float], int] = {}
    vertices: list[tuple[float, float]] = []
    faces: list[tuple[int, int, int]] = []

    def get_index(pt: tuple[float, float]) -> int:
        if pt not in vertex_index:
            vertex_index[pt] = len(vertices) + 1
            vertices.append(pt)
        return vertex_index[pt]

    for tri in triangles:
        coords = list(tri.exterior.coords)[:-1]
        if len(coords) != 3:
            continue
        idx = [get_index((x, y)) for x, y in coords]
        faces.append((idx[0], idx[1], idx[2]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        handle.write("# Triangulated boundary OBJ\n")
        for x, y in vertices:
            handle.write(f"v {x:.8f} {y:.8f} 0\n")
        for i, j, k in faces:
            handle.write(f"f {i} {j} {k}\n")


@contextmanager
def _clean_argv():
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        yield
    finally:
        sys.argv = saved


def _select_action(logits, stochastic_policy: bool):
    if stochastic_policy:
        action, _, _ = pufferlib.pytorch.sample_logits(logits)
        return action

    if isinstance(logits, pufferlib.pytorch.HybridDistribution):
        parts = []
        if logits.has_discrete:
            discrete = torch.argmax(logits.probs, dim=-1)
            parts.append(discrete.T)
        if logits.has_continuous:
            cont = logits.cont_dist.mean
            parts.append(cont.view(cont.shape[0], -1))
        action = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return action.float()

    if isinstance(logits, torch.distributions.Normal):
        return logits.mean.float()

    if torch.is_tensor(logits):
        return torch.argmax(logits, dim=-1).float()

    action, _, _ = pufferlib.pytorch.sample_logits(logits)
    return action


def _run_puffer_eval(
    env_name: str,
    model_path: str,
    config_path: Optional[Path],
    boundary_file: Path,
    export_template: Path,
    seed: int,
    device: str,
    max_steps: int,
    stochastic_policy: bool,
) -> Path:
    with _clean_argv():
        if config_path is None:
            args = pufferl.load_config(env_name)
        else:
            args = pufferl.load_config_file(str(config_path))
    args["vec"]["backend"] = "Serial"
    args["vec"]["num_envs"] = 1
    args["vec"]["num_workers"] = 1
    args["env"]["num_envs"] = 1
    args["env"]["boundary_file"] = str(boundary_file)
    args["env"]["export_meshes"] = True
    args["env"]["export_mesh_path"] = str(export_template)
    args["env"]["render_enabled"] = False
    args["train"]["device"] = device
    args["load_model_path"] = model_path

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    resolved_env_name = env_name or args.get("env_name") or args.get("base", {}).get("env_name", "")
    vecenv = pufferl.load_env(resolved_env_name, args)
    policy = pufferl.load_policy(args, vecenv, resolved_env_name)
    policy.eval()

    obs, _ = vecenv.reset(seed=seed)
    state = {}
    if args["train"].get("use_rnn"):
        num_agents = vecenv.observation_space.shape[0]
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    steps = 0
    while True:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs).to(device)
            logits, _ = policy.forward_eval(obs_t, state)
            action_t = _select_action(logits, stochastic_policy)
            action = action_t.cpu().numpy().reshape(vecenv.action_space.shape)

        if isinstance(logits, (torch.distributions.Normal, pufferlib.pytorch.HybridDistribution)):
            action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

        obs, _, terminals, truncations, _ = vecenv.step(action)
        steps += 1
        if np.any(terminals) or np.any(truncations):
            break
        if steps >= max_steps:
            vecenv.close()
            raise RuntimeError(
                f"Episode did not terminate within {max_steps} steps for {boundary_file}"
            )

    vecenv.close()
    exported_path = Path(str(export_template).replace("{episode}", "0"))
    if not exported_path.exists():
        raise FileNotFoundError(f"Expected exported mesh at {exported_path}")
    return exported_path


def _run_instant_meshes(binary: Path, args: list[str], input_mesh: Path, output_mesh: Path) -> Path:
    cmd = [str(binary), *args, "-o", str(output_mesh), str(input_mesh)]
    subprocess.run(cmd, check=True)
    if not output_mesh.exists():
        raise FileNotFoundError(f"Instant-meshes did not create {output_mesh}")
    return output_mesh


def _discover_baselines(baseline_dir: Path) -> list[dict]:
    baselines = []
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")

    for entry in sorted(baseline_dir.iterdir()):
        if not entry.is_dir():
            continue
        model_files = sorted(entry.glob("*.pt"))
        config_files = sorted(entry.glob("*.ini"))
        if len(model_files) != 1:
            raise ValueError(
                f"Baseline {entry.name} must contain exactly one .pt file (found {len(model_files)})"
            )
        if len(config_files) != 1:
            raise ValueError(
                f"Baseline {entry.name} must contain exactly one .ini file (found {len(config_files)})"
            )
        baselines.append(
            {
                "name": entry.name,
                "path": str(model_files[0]),
                "config_path": config_files[0],
            }
        )
    return baselines


def _model_output_dirs(output_dir: Path, model_name: str) -> tuple[Path, Path, Path, Path]:
    model_root = output_dir / model_name
    meshes_dir = model_root / "meshes"
    figs_dir = model_root / "figs"
    scores_dir = model_root / "scores"
    return model_root, meshes_dir, figs_dir, scores_dir


def _scores_complete(scores_dir: Path, boundaries: list[dict], rl_runs: int) -> bool:
    for boundary in boundaries:
        boundary_name = _safe_name(boundary["name"])
        for run_idx in range(rl_runs):
            score_path = scores_dir / boundary_name / f"run_{run_idx}.json"
            if not score_path.exists():
                return False
    return True


def _load_scores(path: Path) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def _compare_scores(existing_dir: Path, candidate_dir: Path, boundaries: list[dict], rl_runs: int) -> list[str]:
    mismatches = []
    for boundary in boundaries:
        boundary_name = _safe_name(boundary["name"])
        for run_idx in range(rl_runs):
            existing_path = existing_dir / boundary_name / f"run_{run_idx}.json"
            candidate_path = candidate_dir / boundary_name / f"run_{run_idx}.json"
            if not existing_path.exists() or not candidate_path.exists():
                mismatches.append(f"{boundary_name}/run_{run_idx}: missing score file")
                continue

            existing = _load_scores(existing_path)
            candidate = _load_scores(candidate_path)
            for key in ("total", "mean", "median", "low", "high", "n_elements"):
                if key not in existing or key not in candidate:
                    mismatches.append(f"{boundary_name}/run_{run_idx}: missing key {key}")
                    continue
                a = existing[key]
                b = candidate[key]
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if not np.isclose(a, b, rtol=1e-6, atol=1e-8):
                        mismatches.append(
                            f"{boundary_name}/run_{run_idx}: {key} {a} != {b}"
                        )
                else:
                    if a != b:
                        mismatches.append(
                            f"{boundary_name}/run_{run_idx}: {key} {a} != {b}"
                        )
    return mismatches


def _obj_to_polygons(mesh_path: Path) -> list[np.ndarray]:
    vertices: list[list[float]] = []
    lines: list[tuple[int, int]] = []
    faces: list[list[int]] = []

    with open(mesh_path, "r") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 3:
                    vertices.append([float(parts[1]), float(parts[2])])
            elif line.startswith("l "):
                parts = line.split()
                if len(parts) >= 3:
                    a = int(parts[1].split("/")[0]) - 1
                    b = int(parts[2].split("/")[0]) - 1
                    lines.append((a, b))
            elif line.startswith("f "):
                parts = line.split()[1:]
                face = []
                for part in parts:
                    idx = part.split("/")[0]
                    if idx:
                        face.append(int(idx) - 1)
                if face:
                    faces.append(face)

    if not vertices:
        return []

    points = np.asarray(vertices, dtype=np.float32)

    polygons = []
    for face in faces:
        if len(face) == 4:
            polygons.append(points[face])
    if polygons:
        return polygons

    if lines:
        polygons = []
        for i in range(0, len(lines), 4):
            quad_edges = lines[i : i + 4]
            if len(quad_edges) < 4:
                break
            v0, v1 = quad_edges[0]
            v1a, v2 = quad_edges[1]
            if v1a != v1:
                v1a, v2 = v2, v1a
            if v1a != v1:
                continue
            v2a, v3 = quad_edges[2]
            if v2a != v2:
                v2a, v3 = v3, v2a
            if v2a != v2:
                continue
            v3a, v0a = quad_edges[3]
            if v3a != v3:
                v3a, v0a = v0a, v3a
            if v3a != v3 or v0a != v0:
                continue
            polygons.append(points[[v0, v1, v2, v3]])

        if polygons:
            return polygons

        segments = [LineString([points[a], points[b]]) for a, b in lines]
        polys = polygonize(MultiLineString(segments))
        for poly in polys:
            coords = np.asarray(poly.exterior.coords[:-1])
            if len(coords) == 4:
                polygons.append(coords)
        return polygons

    return []


def _mesh_to_polygons(mesh_path: Path) -> list[np.ndarray]:
    if mesh_path.suffix.lower() == ".obj":
        polygons = _obj_to_polygons(mesh_path)
        if polygons:
            return polygons

    mesh = meshio.read(mesh_path)
    points = mesh.points[:, :2]

    if "quad" in mesh.cells_dict:
        faces = mesh.cells_dict["quad"]
        return [points[face] for face in faces]

    if "polygon" in mesh.cells_dict:
        faces = [face for face in mesh.cells_dict["polygon"] if len(face) == 4]
        return [points[face] for face in faces]

    if "line" in mesh.cells_dict:
        lines = mesh.cells_dict["line"]
        segments = [LineString([points[a], points[b]]) for a, b in lines]
        polys = polygonize(MultiLineString(segments))
        polygons = []
        for poly in polys:
            coords = np.asarray(poly.exterior.coords[:-1])
            if len(coords) == 4:
                polygons.append(coords)
        return polygons

    raise ValueError(f"Unsupported mesh cells in {mesh_path}: {mesh.cells_dict.keys()}")


def _measure_quad(quad: Polygon) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (side_lengths, diagonal_lengths, internal_angles)
    """
    verts = np.array(quad.exterior.coords[:-1])
    assert(len(verts) == 4)

    sides = np.roll(verts, -1, axis=0) - verts
    side_lengths = np.linalg.norm(sides, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        unit_sides = sides / side_lengths[:, None]

    diagonal_lengths = np.linalg.norm(verts[:2] - verts[2:], axis=1)

    angles = np.arccos(np.clip(np.sum(-unit_sides * np.roll(unit_sides, -1, axis=0), axis=1), -1.0, 1.0))
    # Recompute largest angle from the 3 smaller ones to handle concavity
    max_idx = np.argmax(angles)
    angles[max_idx] = 2*np.pi - (np.sum(angles[:max_idx]) + np.sum(angles[max_idx+1:]))

    return side_lengths, diagonal_lengths, angles


def _compute_quad_score(quad: Polygon) -> float:
    side_lengths, diagonal_lengths, angles = _measure_quad(quad)

    q_edge = np.sqrt(2) * np.min(side_lengths) / np.max(diagonal_lengths)

    q_angle = np.min(angles) / np.max(angles)
    if np.isnan(q_angle):
        q_angle = 0

    return np.sqrt(q_edge * q_angle)


def _save_mesh_scores(polygons: list[np.ndarray], out_file: Path) -> dict:
    scores = []
    for poly in polygons:
        scores.append(_compute_quad_score(Polygon(poly)))

    scores_arr = np.array(scores, dtype=np.float32)
    summary = {
        "total": float(np.sum(scores_arr)) if len(scores_arr) else 0.0,
        "mean": float(np.mean(scores_arr)) if len(scores_arr) else 0.0,
        "median": float(np.median(scores_arr)) if len(scores_arr) else 0.0,
        "low": float(np.min(scores_arr)) if len(scores_arr) else 0.0,
        "high": float(np.max(scores_arr)) if len(scores_arr) else 0.0,
        "n_elements": int(len(scores_arr)),
    }

    with open(out_file, "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _plot_polygons(
    ax,
    polygons: list[np.ndarray],
    summary: dict,
    show_summary: bool,
    summary_fontsize: float = 9.0,
    summary_box_pad: float = 0.3,
) -> None:
    if polygons:
        scores = [_compute_quad_score(Polygon(poly)) for poly in polygons]
        collection = PolyCollection(
            polygons,
            array=np.array(scores),
            cmap="viridis",
            edgecolors="k",
            zorder=1,
            linewidth=0.5,
        )
        collection.set_clim(0.0, 1.0)
        ax.add_collection(collection)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if show_summary:
            summary_text = (
                f"n={summary['n_elements']}\n"
                f"score={summary['total']:.3f}\n"
                f"mean={summary['mean']:.3f}\n"
                f"median={summary['median']:.3f}\n"
                f"low={summary['low']:.3f}\n"
                f"high={summary['high']:.3f}"
            )
            ax.text(
                0.02,
                0.98,
                summary_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=summary_fontsize,
                color="black",
                bbox=dict(
                    boxstyle=f"round,pad={summary_box_pad}",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.5,
                ),
                clip_on=False,
                zorder=3,
            )
    else:
        ax.text(0.5, 0.5, "No quads found", ha="center", va="center")
        ax.set_axis_off()


def _plot_mesh_scores(polygons: list[np.ndarray], summary: dict, out_file: Path) -> None:
    fig, ax = plt.subplots()
    _plot_polygons(ax, polygons, summary, show_summary=True)
    if polygons:
        mappable = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="viridis")
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        fig.colorbar(mappable, cax=cax, label="Score")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Quad Element Scores")
    plt.savefig(out_file)
    plt.close(fig)


def _evaluate_mesh(mesh_path: Path, scores_path: Path, fig_path: Path) -> dict:
    polygons = _mesh_to_polygons(mesh_path)
    scores = _save_mesh_scores(polygons, scores_path)
    _plot_mesh_scores(polygons, scores, fig_path)
    return scores


def _best_run(meshes_dir: Path, scores_dir: Path, boundary_name: str, rl_runs: int) -> tuple[Path | None, dict | None]:
    best_score = None
    best_mesh = None
    best_summary = None
    for run_idx in range(rl_runs):
        score_path = scores_dir / boundary_name / f"run_{run_idx}.json"
        mesh_path = meshes_dir / boundary_name / f"run_{run_idx}_episode0.obj"
        if not score_path.exists() or not mesh_path.exists():
            continue
        summary = _load_scores(score_path)
        total = summary.get("total")
        if total is None:
            continue
        if best_score is None or total > best_score:
            best_score = total
            best_mesh = mesh_path
            best_summary = summary

    if best_mesh is None:
        single_score = scores_dir / f"{boundary_name}.json"
        single_mesh = meshes_dir / f"{boundary_name}.obj"
        if single_score.exists() and single_mesh.exists():
            summary = _load_scores(single_score)
            return single_mesh, summary

    return best_mesh, best_summary


def _render_summary_grid(
    output_dir: Path,
    model_entries: list[dict],
    boundaries: list[dict],
    rl_runs: int,
    include_instant: bool,
) -> None:
    model_names = [_safe_name(entry["name"]) for entry in model_entries]
    if include_instant and "instant_meshes" not in model_names:
        model_names.append("instant_meshes")
    boundary_names = [_safe_name(b["name"]) for b in boundaries]
    if not model_names or not boundary_names:
        return

    nrows = len(model_names)
    ncols = len(boundary_names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)

    for row_idx, model_name in enumerate(model_names):
        _, model_meshes_dir, _, model_scores_dir = _model_output_dirs(output_dir, model_name)
        for col_idx, boundary_name in enumerate(boundary_names):
            ax = axes[row_idx][col_idx]
            mesh_path, summary = _best_run(model_meshes_dir, model_scores_dir, boundary_name, rl_runs)
            if mesh_path is None or summary is None:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center")
                ax.set_axis_off()
            else:
                polygons = _mesh_to_polygons(mesh_path)
                _plot_polygons(
                    ax,
                    polygons,
                    summary,
                    show_summary=True,
                    summary_fontsize=5.0,
                )
            if row_idx == 0:
                ax.set_title(boundary_name, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(model_name, fontsize=10)

    mappable = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="viridis")
    fig.subplots_adjust(right=0.9, wspace=0.1, hspace=0.1)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(mappable, cax=cax, label="Score")
    out_path = output_dir / "summary_grid.svg"
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate quad meshing models and baselines.")
    parser.add_argument("--config", required=True, help="Path to eval config YAML.")
    parser.add_argument("--model", required=True, help="Model path or 'latest'.")
    parser.add_argument("--output", default="eval", help="Output directory (default: eval).")
    parser.add_argument("--env-name", default="puffer_quad_meshing", help="PufferLib env name.")
    parser.add_argument("--device", default="cpu", help="Torch device for evaluation.")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps per episode.")
    parser.add_argument(
        "--stochastic-policy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample policy actions while using seeded RNG (default: false).",
    )
    parser.add_argument(
        "--recheck-baselines",
        action="store_true",
        help="Re-evaluate all baselines in a temp folder and compare scores.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_eval_config(config_path)
    base_dir = config_path.parent

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    boundaries = []
    for entry in config.get("boundaries", []):
        if isinstance(entry, str):
            boundaries.append({"name": Path(entry).stem, "boundary_file": entry})
        elif isinstance(entry, dict):
            if "boundary_file" not in entry:
                raise ValueError("Each boundary entry must include 'boundary_file'.")
            name = entry.get("name") or Path(entry["boundary_file"]).stem
            normalized = dict(entry)
            normalized["name"] = name
            boundaries.append(normalized)
        else:
            raise ValueError("Boundary entries must be strings or dicts.")
    rl_runs = int(config.get("rl_evals_per_boundary", 1))
    seed_base = int(config.get("seed", 42))
    stochastic_policy = bool(config.get("stochastic_policy", args.stochastic_policy))

    baseline_dir = config.get("baseline_dir")
    if not baseline_dir:
        raise ValueError("eval config must define baseline_dir")
    baseline_dir_path = _resolve_path(baseline_dir, base_dir)
    if baseline_dir_path is None:
        raise ValueError("baseline_dir must be a valid path")

    baseline_entries = _discover_baselines(baseline_dir_path)
    model_entries = [{"name": "target", "path": args.model, "config_path": None}, *baseline_entries]

    instant_cfg = config.get("instant_meshes", {}) or {}
    instant_enabled = instant_cfg.get("enabled", True)
    instant_binary = _resolve_path(instant_cfg.get("binary", "./instant-meshes"), base_dir)
    instant_base_args = instant_cfg.get("args", ["-b"])
    instant_default_faces = int(instant_cfg.get("default_faces", 200))
    if instant_enabled:
        if instant_binary is None or not instant_binary.exists():
            raise FileNotFoundError(f"Instant-meshes binary not found: {instant_binary}")

    for model_entry in model_entries:
        model_name = _safe_name(model_entry["name"])
        if model_entry.get("config_path") is None:
            model_path = _resolve_model_path(model_entry["path"], base_dir)
        else:
            model_path = str(Path(model_entry["path"]).resolve())
        model_root, model_meshes_dir, model_figs_dir, model_scores_dir = _model_output_dirs(
            output_dir, model_name
        )

        is_baseline = model_entry.get("config_path") is not None
        if is_baseline and _scores_complete(model_scores_dir, boundaries, rl_runs):
            continue

        model_meshes_dir.mkdir(parents=True, exist_ok=True)
        model_figs_dir.mkdir(parents=True, exist_ok=True)
        model_scores_dir.mkdir(parents=True, exist_ok=True)

        for boundary in boundaries:
            boundary_name = _safe_name(boundary["name"])
            boundary_file = _resolve_path(boundary["boundary_file"], base_dir)
            if boundary_file is None or not boundary_file.exists():
                raise FileNotFoundError(f"Missing boundary_file for {boundary_name}")

            boundary_mesh_dir = model_meshes_dir / boundary_name
            boundary_mesh_dir.mkdir(parents=True, exist_ok=True)
            boundary_fig_dir = model_figs_dir / boundary_name
            boundary_fig_dir.mkdir(parents=True, exist_ok=True)
            boundary_scores_dir = model_scores_dir / boundary_name
            boundary_scores_dir.mkdir(parents=True, exist_ok=True)

            for run_idx in range(rl_runs):
                export_template = boundary_mesh_dir / f"run_{run_idx}_episode{{episode}}.obj"
                mesh_path = _run_puffer_eval(
                    env_name=args.env_name,
                    model_path=model_path,
                    config_path=model_entry.get("config_path"),
                    boundary_file=boundary_file,
                    export_template=export_template,
                    seed=seed_base + run_idx,
                    device=args.device,
                    max_steps=args.max_steps,
                    stochastic_policy=stochastic_policy,
                )

                score_path = boundary_scores_dir / f"run_{run_idx}.json"
                fig_path = boundary_fig_dir / f"run_{run_idx}.svg"
                _evaluate_mesh(mesh_path, score_path, fig_path)

    if instant_enabled:
        instant_root, instant_meshes_dir, instant_figs_dir, instant_scores_dir = _model_output_dirs(
            output_dir, "instant_meshes"
        )
        instant_inputs_dir = instant_root / "inputs"
        for boundary in boundaries:
            boundary_name = _safe_name(boundary["name"])
            boundary_file = _resolve_path(boundary["boundary_file"], base_dir)
            if boundary_file is None or not boundary_file.exists():
                raise FileNotFoundError(f"Missing boundary_file for {boundary_name}")

            mesh_out = instant_meshes_dir / f"{boundary_name}.obj"
            score_out = instant_scores_dir / f"{boundary_name}.json"
            fig_out = instant_figs_dir / f"{boundary_name}.svg"
            if mesh_out.exists() and score_out.exists() and fig_out.exists():
                continue

            faces = int(boundary.get("instant_faces", instant_default_faces))
            triangulated_input = instant_inputs_dir / f"{boundary_name}.obj"
            _triangulate_boundary_to_obj(boundary_file, triangulated_input)

            mesh_out.parent.mkdir(parents=True, exist_ok=True)
            instant_args = list(instant_base_args) + ["-f", str(faces)]
            _run_instant_meshes(instant_binary, instant_args, triangulated_input, mesh_out)

            score_out.parent.mkdir(parents=True, exist_ok=True)
            fig_out.parent.mkdir(parents=True, exist_ok=True)
            _evaluate_mesh(mesh_out, score_out, fig_out)

    if args.recheck_baselines:
        temp_root = output_dir / "_baseline_recheck"
        temp_root.mkdir(parents=True, exist_ok=True)
        mismatches = []
        for baseline in baseline_entries:
            model_name = _safe_name(baseline["name"])
            model_root, model_meshes_dir, model_figs_dir, model_scores_dir = _model_output_dirs(
                temp_root, model_name
            )
            model_meshes_dir.mkdir(parents=True, exist_ok=True)
            model_figs_dir.mkdir(parents=True, exist_ok=True)
            model_scores_dir.mkdir(parents=True, exist_ok=True)

            for boundary in boundaries:
                boundary_name = _safe_name(boundary["name"])
                boundary_file = _resolve_path(boundary["boundary_file"], base_dir)
                if boundary_file is None or not boundary_file.exists():
                    raise FileNotFoundError(f"Missing boundary_file for {boundary_name}")

                boundary_mesh_dir = model_meshes_dir / boundary_name
                boundary_fig_dir = model_figs_dir / boundary_name
                boundary_scores_dir = model_scores_dir / boundary_name
                boundary_mesh_dir.mkdir(parents=True, exist_ok=True)
                boundary_fig_dir.mkdir(parents=True, exist_ok=True)
                boundary_scores_dir.mkdir(parents=True, exist_ok=True)

                for run_idx in range(rl_runs):
                    export_template = boundary_mesh_dir / f"run_{run_idx}_episode{{episode}}.obj"
                    mesh_path = _run_puffer_eval(
                        env_name=args.env_name,
                        model_path=str(Path(baseline["path"]).resolve()),
                        config_path=baseline.get("config_path"),
                        boundary_file=boundary_file,
                        export_template=export_template,
                        seed=seed_base + run_idx,
                        device=args.device,
                        max_steps=args.max_steps,
                        stochastic_policy=stochastic_policy,
                    )
                    score_path = boundary_scores_dir / f"run_{run_idx}.json"
                    fig_path = boundary_fig_dir / f"run_{run_idx}.svg"
                    _evaluate_mesh(mesh_path, score_path, fig_path)

            existing_root, _, _, existing_scores_dir = _model_output_dirs(output_dir, model_name)
            if not existing_scores_dir.exists():
                mismatches.append(f"{model_name}: missing existing scores directory")
                continue
            mismatches.extend(
                _compare_scores(existing_scores_dir, model_scores_dir, boundaries, rl_runs)
            )

        if mismatches:
            print("Baseline regression mismatches detected:")
            for entry in mismatches:
                print(f"  - {entry}")
            print(f"Temp results preserved at: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)
            print("Baseline regression check passed; temp results removed.")

    _render_summary_grid(output_dir, model_entries, boundaries, rl_runs, include_instant=instant_enabled)


if __name__ == "__main__":
    main()
