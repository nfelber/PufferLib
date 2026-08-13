#!/usr/bin/env python3
"""Create report-ready quality plots for one or more planar OBJ meshes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


@dataclass(frozen=True)
class Mesh:
    path: Path
    vertices: np.ndarray
    edges: tuple[tuple[int, int], ...]
    faces: tuple[tuple[int, ...], ...]
    native_export: bool


@dataclass(frozen=True)
class QmShape:
    path: Path
    boundary: np.ndarray


@dataclass(frozen=True)
class RewardSettings:
    target_edge_length_ratio: float = 1.0
    base_quad_reward: float = 0.0
    triangle_reward: float = 0.0
    quad_reward_scale: float = 0.5


@dataclass(frozen=True)
class MeshEvaluation:
    mesh: Mesh
    quads: tuple[tuple[int, ...], ...]
    triangles: tuple[tuple[int, ...], ...]
    other_faces: tuple[tuple[int, ...], ...]
    qualities: np.ndarray
    singularities: tuple[int, ...]
    target_quad_area: float | None = None
    area_qualities: np.ndarray | None = None
    reward_qualities: np.ndarray | None = None
    face_reward_return: float | None = None


@dataclass(frozen=True)
class GridSpec:
    paths: tuple[tuple[Path, ...], ...]
    row_labels: tuple[str, ...]
    column_labels: tuple[str, ...]
    qmshape_paths: tuple[Path, ...] | None = None
    reward_settings: RewardSettings | None = None


@dataclass(frozen=True)
class AggregateStatistics:
    mesh_count: int
    triangles: int
    quads: int
    triangle_to_quad_percent: float | None
    singularities: int
    vertices: int
    singularity_to_vertex_percent: float | None
    mean_quality: float | None
    median_quality: float | None
    mean_minimum_quality: float | None
    total_face_reward_return: float | None
    mean_face_reward_return: float | None
    median_face_reward_return: float | None
    mean_reward_quality: float | None
    mean_area_quality: float | None
    mean_target_quad_area: float | None


def signed_polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def interior_angle(poly: np.ndarray, index: int, is_ccw: bool) -> float:
    previous = poly[(index - 1) % len(poly)] - poly[index]
    following = poly[(index + 1) % len(poly)] - poly[index]
    angle = math.atan2(
        previous[0] * following[1] - previous[1] * following[0],
        float(np.dot(previous, following)),
    )
    if is_ccw:
        angle = -angle
    if angle <= 0.0:
        angle += 2.0 * math.pi
    return angle


def compute_quad_quality(quad: np.ndarray) -> float:
    """Match the element-quality term `eq` in quad_meshing.h."""
    edge_lengths_sq = [
        float(np.dot(quad[j] - quad[j - 1], quad[j] - quad[j - 1]))
        for j in range(4)
    ]
    angles = [interior_angle(quad, j, signed_polygon_area(quad) > 0.0) for j in range(4)]
    diagonal_lengths_sq = (
        float(np.dot(quad[0] - quad[2], quad[0] - quad[2])),
        float(np.dot(quad[1] - quad[3], quad[1] - quad[3])),
    )
    denominator = math.sqrt(max(diagonal_lengths_sq)) * max(angles)
    if denominator <= 0.0 or min(edge_lengths_sq) <= 0.0:
        return 0.0
    quality = math.sqrt(
        math.sqrt(2.0) * math.sqrt(min(edge_lengths_sq)) * min(angles) / denominator
    )
    return float(np.clip(quality, 0.0, 1.0))


def compute_area_quality(area: float, target_area: float) -> float:
    """Match `compute_area_quality` in quad_meshing.h."""
    log_ratio = math.log(max(area / target_area, 1e-6))
    return math.exp(-2.0 * log_ratio * log_ratio)


def load_qmshape(path: Path) -> QmShape:
    data = path.read_bytes()
    if len(data) < 16 or data[:8] != b"QMSHAPE\0":
        raise ValueError(f"{path}: invalid qmshape magic")
    version, section_count = struct.unpack_from("=II", data, 8)
    if version != 1:
        raise ValueError(f"{path}: unsupported qmshape version {version}")

    offset = 16
    boundary: np.ndarray | None = None
    saw_cross_field = False
    for _ in range(section_count):
        if offset + 16 > len(data):
            raise ValueError(f"{path}: truncated qmshape section header")
        section_type, count, element_size, reserved = struct.unpack_from("=IIII", data, offset)
        offset += 16
        if reserved != 0:
            raise ValueError(f"{path}: nonzero qmshape section reserved field")
        section_size = count * element_size
        if offset + section_size > len(data):
            raise ValueError(f"{path}: truncated qmshape section data")
        if section_type == 1:
            if boundary is not None:
                raise ValueError(f"{path}: duplicate boundary section")
            if element_size != 8 or count < 3:
                raise ValueError(f"{path}: invalid boundary section")
            boundary = np.frombuffer(data, dtype="=f4", count=2 * count, offset=offset)
            boundary = boundary.reshape(count, 2).astype(np.float64)
        elif section_type == 2:
            if element_size != 40 or count == 0:
                raise ValueError(f"{path}: invalid cross-field section")
            saw_cross_field = True
        offset += section_size

    if boundary is None or not saw_cross_field:
        raise ValueError(f"{path}: missing required qmshape section")
    if not np.all(np.isfinite(boundary)):
        raise ValueError(f"{path}: boundary contains non-finite coordinates")
    boundary_edges = boundary - np.roll(boundary, 1, axis=0)
    perimeter = float(np.sum(np.linalg.norm(boundary_edges, axis=1)))
    if perimeter <= 0.0:
        raise ValueError(f"{path}: boundary has zero perimeter")
    return QmShape(path.resolve(), boundary)


def target_quad_area(shape: QmShape, settings: RewardSettings) -> float:
    if settings.target_edge_length_ratio <= 0.0:
        raise ValueError("target edge length ratio must be positive")
    edges = shape.boundary - np.roll(shape.boundary, 1, axis=0)
    mean_edge_length = float(np.sum(np.linalg.norm(edges, axis=1))) / len(shape.boundary)
    return (mean_edge_length * settings.target_edge_length_ratio) ** 2


def reward_coordinate_scale(mesh: Mesh, shape: QmShape) -> float:
    """Return the uniform OBJ-to-qmshape scale after provenance validation."""
    if mesh.path.stem != shape.path.stem:
        raise ValueError(
            f"{mesh.path}: shape stem '{shape.path.stem}' does not match mesh stem"
        )
    if mesh.native_export:
        boundary_size = len(shape.boundary)
        if len(mesh.vertices) < boundary_size:
            raise ValueError(f"{mesh.path}: fewer vertices than the source boundary")
        mesh_boundary = mesh.vertices[:boundary_size]
        translation = mesh_boundary[0] - shape.boundary[0]
        if not np.allclose(
            mesh_boundary - translation,
            shape.boundary,
            rtol=0.0,
            atol=2e-6,
        ):
            raise ValueError(
                f"{mesh.path}: native OBJ boundary shape does not match {shape.path}"
            )
        return 1.0

    mesh_span = np.ptp(mesh.vertices, axis=0)
    boundary_span = np.ptp(shape.boundary, axis=0)
    mesh_extent = float(np.max(mesh_span))
    boundary_extent = float(np.max(boundary_span))
    if mesh_extent <= 0.0 or boundary_extent <= 0.0:
        raise ValueError(f"{mesh.path}: degenerate mesh or source boundary extent")
    scale = boundary_extent / mesh_extent
    if not np.allclose(mesh_span * scale, boundary_span, rtol=2e-2, atol=2e-4):
        raise ValueError(
            f"{mesh.path}: normalized aspect ratio does not match boundary in {shape.path}"
        )
    return scale


def obj_index(token: str, vertex_count: int) -> int:
    index = int(token.split("/", 1)[0])
    return index - 1 if index > 0 else vertex_count + index


def reconstruct_planar_faces(
    vertices: np.ndarray, edges: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, ...], ...]:
    """Extract bounded faces from a straight-line planar graph."""
    adjacency: list[list[int]] = [[] for _ in range(len(vertices))]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    for vertex, neighbors in enumerate(adjacency):
        neighbors.sort(
            key=lambda neighbor: math.atan2(
                vertices[neighbor, 1] - vertices[vertex, 1],
                vertices[neighbor, 0] - vertices[vertex, 0],
            )
        )

    visited: set[tuple[int, int]] = set()
    faces: list[tuple[int, ...]] = []
    area_tolerance = np.finfo(float).eps * max(float(np.ptp(vertices, axis=0).prod()), 1.0)
    for a, b in edges:
        for start in ((a, b), (b, a)):
            if start in visited:
                continue
            face: list[int] = []
            directed = start
            while directed not in visited:
                visited.add(directed)
                previous, current = directed
                face.append(previous)
                neighbors = adjacency[current]
                if not neighbors:
                    break
                incoming_index = neighbors.index(previous)
                directed = (current, neighbors[(incoming_index - 1) % len(neighbors)])
            if directed == start and len(face) >= 3:
                area = signed_polygon_area(vertices[face])
                if area > area_tolerance:
                    faces.append(tuple(face))
    return tuple(faces)


def parse_obj(path: Path) -> Mesh:
    vertices: list[tuple[float, float]] = []
    edges: set[tuple[int, int]] = set()
    faces: list[tuple[int, ...]] = []
    text = path.read_text(encoding="utf-8")
    native_export = any(
        line.strip() == "# QuadMeshing OBJ export" for line in text.splitlines()[:5]
    )

    for line_number, raw in enumerate(text.splitlines(), 1):
        parts = raw.partition("#")[0].split()
        if not parts:
            continue
        if parts[0] == "v" and len(parts) >= 3:
            vertices.append((float(parts[1]), float(parts[2])))
        elif parts[0] in {"l", "f"}:
            minimum_size = 3 if parts[0] == "l" else 4
            if len(parts) < minimum_size:
                continue
            indices = tuple(obj_index(token, len(vertices)) for token in parts[1:])
            if any(index < 0 or index >= len(vertices) for index in indices):
                raise ValueError(f"{path}:{line_number}: vertex index out of range")
            pairs = zip(indices, indices[1:]) if parts[0] == "l" else zip(indices, indices[1:] + indices[:1])
            for a, b in pairs:
                if a != b:
                    edges.add(tuple(sorted((a, b))))
            if parts[0] == "f":
                faces.append(indices)

    vertex_array = np.asarray(vertices, dtype=np.float64)
    if len(vertex_array) == 0:
        raise ValueError(f"{path}: no vertices found")
    edge_tuple = tuple(sorted(edges))
    if not edge_tuple:
        raise ValueError(f"{path}: no mesh edges found")
    if not faces:
        faces = list(reconstruct_planar_faces(vertex_array, edge_tuple))
    if not faces:
        raise ValueError(f"{path}: no bounded faces found")
    return Mesh(path, vertex_array, edge_tuple, tuple(faces), native_export)


def evaluate_mesh(
    mesh: Mesh,
    shape: QmShape | None = None,
    reward_settings: RewardSettings | None = None,
) -> MeshEvaluation:
    quads = tuple(face for face in mesh.faces if len(face) == 4)
    triangles = tuple(face for face in mesh.faces if len(face) == 3)
    other_faces = tuple(face for face in mesh.faces if len(face) not in {3, 4})
    qualities = np.asarray(
        [compute_quad_quality(mesh.vertices[list(face)]) for face in quads], dtype=np.float64
    )

    neighbors: list[set[int]] = [set() for _ in range(len(mesh.vertices))]
    face_count: dict[tuple[int, int], int] = {edge: 0 for edge in mesh.edges}
    for a, b in mesh.edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    for face in mesh.faces:
        for a, b in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((a, b)))
            face_count[edge] = face_count.get(edge, 0) + 1
    boundary_vertices = {
        vertex
        for edge, count in face_count.items()
        if count != 2
        for vertex in edge
    }
    singularities = tuple(
        vertex
        for vertex, adjacent in enumerate(neighbors)
        if adjacent and vertex not in boundary_vertices and len(adjacent) != 4
    )
    if (shape is None) != (reward_settings is None):
        raise ValueError("shape and reward settings must be provided together")
    if shape is None or reward_settings is None:
        return MeshEvaluation(mesh, quads, triangles, other_faces, qualities, singularities)

    coordinate_scale = reward_coordinate_scale(mesh, shape)
    if not 0.0 <= reward_settings.base_quad_reward <= 1.0:
        raise ValueError("base quad reward must be between 0 and 1")
    if reward_settings.quad_reward_scale < 0.0:
        raise ValueError("quad reward scale must be non-negative")
    target_area = target_quad_area(shape, reward_settings)
    areas = np.asarray(
        [
            abs(signed_polygon_area(mesh.vertices[list(face)])) * coordinate_scale**2
            for face in quads
        ],
        dtype=np.float64,
    )
    area_qualities = np.asarray(
        [compute_area_quality(area, target_area) for area in areas], dtype=np.float64
    )
    reward_qualities = qualities * area_qualities
    quad_rewards = (
        reward_settings.base_quad_reward
        + (1.0 - reward_settings.base_quad_reward) * reward_qualities
    )
    face_reward_return = (
        reward_settings.quad_reward_scale * float(np.sum(quad_rewards))
        + reward_settings.triangle_reward * len(triangles)
    )
    return MeshEvaluation(
        mesh,
        quads,
        triangles,
        other_faces,
        qualities,
        singularities,
        target_area,
        area_qualities,
        reward_qualities,
        face_reward_return,
    )


def statistics_text(evaluation: MeshEvaluation, low_percentile: float) -> str:
    qualities = evaluation.qualities
    if len(qualities):
        quality_stats = [
            ("Mean quality", f"{np.mean(qualities):.3f}"),
            ("Median quality", f"{np.median(qualities):.3f}"),
            (f"P{low_percentile:g} quality", f"{np.percentile(qualities, low_percentile):.3f}"),
            ("Minimum quality", f"{np.min(qualities):.3f}"),
        ]
    else:
        quality_stats = [
            ("Mean quality", "n/a"),
            ("Median quality", "n/a"),
            ("Low quality", "n/a"),
            ("Minimum quality", "n/a"),
        ]
    mesh_stats = [
        ("Quads", str(len(evaluation.quads))),
        ("Triangles", str(len(evaluation.triangles))),
        ("Singularities", str(len(evaluation.singularities))),
        ("", ""),
    ]
    if evaluation.face_reward_return is not None:
        mean_area_quality = (
            f"{np.mean(evaluation.area_qualities):.3f}"
            if evaluation.area_qualities is not None and len(evaluation.area_qualities)
            else "n/a"
        )
        quality_stats.extend(
            [
                ("Mean area qual.", mean_area_quality),
                ("Face reward ret.", f"{evaluation.face_reward_return:.3f}"),
            ]
        )
    while len(mesh_stats) < len(quality_stats):
        mesh_stats.append(("", ""))
    return "\n".join(
        f"{mesh_label:<13} {mesh_value:>5}     {quality_label:<15} {quality_value:>5}"
        for (mesh_label, mesh_value), (quality_label, quality_value) in zip(
            mesh_stats, quality_stats
        )
    )


def plot_mesh(
    ax: plt.Axes,
    evaluation: MeshEvaluation,
    cmap: str,
    low_percentile: float,
    show_singularities: bool,
    title: str,
) -> PolyCollection:
    vertices = evaluation.mesh.vertices
    quad_collection = PolyCollection(
        [vertices[list(face)] for face in evaluation.quads],
        array=evaluation.qualities,
        cmap=cmap,
        norm=Normalize(0.0, 1.0),
        edgecolors="none",
        zorder=1,
    )
    ax.add_collection(quad_collection)
    if evaluation.triangles:
        ax.add_collection(
            PolyCollection(
                [vertices[list(face)] for face in evaluation.triangles],
                facecolors="#f28e2b",
                edgecolors="#9c4d08",
                linewidths=0.8,
                hatch="////",
                zorder=2,
            )
        )
    if evaluation.other_faces:
        ax.add_collection(
            PolyCollection(
                [vertices[list(face)] for face in evaluation.other_faces],
                facecolors="#d5d8dc",
                edgecolors="none",
                zorder=1,
            )
        )

    edge_segments = [(vertices[a], vertices[b]) for a, b in evaluation.mesh.edges]
    ax.add_collection(
        LineCollection(edge_segments, colors="#24272b", linewidths=0.45, alpha=0.9, zorder=3)
    )
    if show_singularities and evaluation.singularities:
        points = vertices[list(evaluation.singularities)]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=28,
            marker="o",
            facecolor="#cc2f7a",
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )

    ax.autoscale_view()
    x_min, y_min = np.min(vertices, axis=0)
    x_max, y_max = np.max(vertices, axis=0)
    padding = 0.035 * max(x_max - x_min, y_max - y_min, 1e-12)
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    title_padding = 94 if evaluation.face_reward_return is not None else 66
    ax.set_title(title or " ", fontsize=12, fontweight="semibold", pad=title_padding)
    ax.text(
        0.5,
        1.015,
        statistics_text(evaluation, low_percentile),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontfamily="monospace",
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.55,rounding_size=0.2",
            "facecolor": "white",
            "edgecolor": "#b6bbc2",
            "linewidth": 0.7,
            "alpha": 0.94,
        },
        zorder=10,
    )
    return quad_collection


def collect_obj_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for path in inputs:
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.obj")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(path)
    unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique_paths:
        raise ValueError("no OBJ files found")
    return unique_paths


def parse_labeled_entries(
    entries: object,
    value_key: str,
    default_label: Callable[[str], str],
) -> tuple[list[str], list[str]]:
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"grid config '{value_key}s' must be a non-empty array")
    values: list[str] = []
    labels: list[str] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            value = entry
            label = default_label(value)
        elif isinstance(entry, dict):
            value = entry.get(value_key)
            label = entry.get("label")
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"grid config '{value_key}s[{index}].{value_key}' must be a non-empty string"
                )
            if label is None:
                label = default_label(value)
            elif not isinstance(label, str):
                raise ValueError(
                    f"grid config '{value_key}s[{index}].label' must be a string"
                )
        else:
            raise ValueError(
                f"grid config '{value_key}s[{index}]' must be a string or object"
            )
        values.append(value)
        labels.append(label)
    return values, labels


def parse_reward_settings(config: object) -> RewardSettings:
    if not isinstance(config, dict):
        raise ValueError("grid config 'reward' must be an object")

    def number(name: str, default: float) -> float:
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"grid config 'reward.{name}' must be a number")
        return float(value)

    return RewardSettings(
        target_edge_length_ratio=number("target_edge_length_ratio", 1.0),
        base_quad_reward=number("base_quad_reward", 0.0),
        triangle_reward=number("triangle_reward", 0.0),
        quad_reward_scale=number("quad_reward_scale", 0.5),
    )


def load_grid_config(path: Path) -> GridSpec:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("grid config must contain a JSON object")
    folders, row_labels = parse_labeled_entries(
        config.get("folders"), "path", lambda value: Path(value).name
    )
    filenames, column_labels = parse_labeled_entries(
        config.get("files"), "name", lambda value: Path(value).stem
    )
    if any(Path(filename).is_absolute() for filename in filenames):
        raise ValueError("grid config file names must be relative to each folder")

    base = path.resolve().parent
    folder_paths = [Path(folder) if Path(folder).is_absolute() else base / folder for folder in folders]
    paths = tuple(
        tuple(folder / filename for filename in filenames)
        for folder in folder_paths
    )
    reward_config = config.get("reward")
    if reward_config is None:
        return GridSpec(paths, tuple(row_labels), tuple(column_labels))
    reward_settings = parse_reward_settings(reward_config)
    qmshape_folder_value = reward_config.get("qmshape_folder")
    if qmshape_folder_value is not None and (
        not isinstance(qmshape_folder_value, str) or not qmshape_folder_value
    ):
        raise ValueError("grid config 'reward.qmshape_folder' must be a non-empty string")
    qmshape_folder = None
    if qmshape_folder_value:
        candidate = Path(qmshape_folder_value)
        qmshape_folder = candidate if candidate.is_absolute() else base / candidate

    qmshape_paths: list[Path] = []
    file_entries = config["files"]
    for index, (entry, filename) in enumerate(zip(file_entries, filenames)):
        override = entry.get("qmshape") if isinstance(entry, dict) else None
        if override is not None:
            if not isinstance(override, str) or not override:
                raise ValueError(
                    f"grid config 'files[{index}].qmshape' must be a non-empty string"
                )
            candidate = Path(override)
            qmshape_paths.append(candidate if candidate.is_absolute() else base / candidate)
        elif qmshape_folder is not None:
            qmshape_paths.append(qmshape_folder / f"{Path(filename).stem}.qmshape")
        else:
            raise ValueError(
                "reward evaluation requires 'reward.qmshape_folder' or a qmshape per file"
            )
    return GridSpec(
        paths,
        tuple(row_labels),
        tuple(column_labels),
        tuple(qmshape_paths),
        reward_settings,
    )


def aggregate_statistics(evaluations: list[MeshEvaluation]) -> AggregateStatistics:
    triangles = sum(len(evaluation.triangles) for evaluation in evaluations)
    quads = sum(len(evaluation.quads) for evaluation in evaluations)
    singularities = sum(len(evaluation.singularities) for evaluation in evaluations)
    vertices = sum(len(evaluation.mesh.vertices) for evaluation in evaluations)
    quality_arrays = [evaluation.qualities for evaluation in evaluations if len(evaluation.qualities)]
    pooled_qualities = np.concatenate(quality_arrays) if quality_arrays else np.asarray([])
    mesh_minima = [float(np.min(qualities)) for qualities in quality_arrays]
    reward_evaluations = [
        evaluation for evaluation in evaluations if evaluation.face_reward_return is not None
    ]
    face_returns = [evaluation.face_reward_return for evaluation in reward_evaluations]
    reward_quality_arrays = [
        evaluation.reward_qualities
        for evaluation in reward_evaluations
        if evaluation.reward_qualities is not None and len(evaluation.reward_qualities)
    ]
    area_quality_arrays = [
        evaluation.area_qualities
        for evaluation in reward_evaluations
        if evaluation.area_qualities is not None and len(evaluation.area_qualities)
    ]
    pooled_reward_qualities = (
        np.concatenate(reward_quality_arrays) if reward_quality_arrays else np.asarray([])
    )
    pooled_area_qualities = (
        np.concatenate(area_quality_arrays) if area_quality_arrays else np.asarray([])
    )
    target_areas = [
        evaluation.target_quad_area
        for evaluation in reward_evaluations
        if evaluation.target_quad_area is not None
    ]
    return AggregateStatistics(
        mesh_count=len(evaluations),
        triangles=triangles,
        quads=quads,
        triangle_to_quad_percent=100.0 * triangles / quads if quads else None,
        singularities=singularities,
        vertices=vertices,
        singularity_to_vertex_percent=100.0 * singularities / vertices if vertices else None,
        mean_quality=float(np.mean(pooled_qualities)) if len(pooled_qualities) else None,
        median_quality=float(np.median(pooled_qualities)) if len(pooled_qualities) else None,
        mean_minimum_quality=float(np.mean(mesh_minima)) if mesh_minima else None,
        total_face_reward_return=float(np.sum(face_returns)) if face_returns else None,
        mean_face_reward_return=float(np.mean(face_returns)) if face_returns else None,
        median_face_reward_return=float(np.median(face_returns)) if face_returns else None,
        mean_reward_quality=(
            float(np.mean(pooled_reward_qualities)) if len(pooled_reward_qualities) else None
        ),
        mean_area_quality=(
            float(np.mean(pooled_area_qualities)) if len(pooled_area_qualities) else None
        ),
        mean_target_quad_area=float(np.mean(target_areas)) if target_areas else None,
    )


def format_csv_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.9g}"


def write_aggregate_csv(
    path: Path,
    evaluations: list[MeshEvaluation],
    grid_spec: GridSpec | None,
) -> None:
    groups: list[tuple[str, str, list[MeshEvaluation]]] = [
        ("overall", "All meshes", evaluations)
    ]
    if grid_spec:
        column_count = len(grid_spec.column_labels)
        for row, label in enumerate(grid_spec.row_labels):
            start = row * column_count
            groups.append(("folder", label, evaluations[start : start + column_count]))

    fieldnames = [
        "scope",
        "label",
        "mesh_count",
        "triangles",
        "quads",
        "triangle_to_quad_percent",
        "singularities",
        "vertices",
        "singularity_to_vertex_percent",
        "mean_quality",
        "median_quality",
        "mean_minimum_quality",
        "total_face_reward_return",
        "mean_face_reward_return",
        "median_face_reward_return",
        "mean_reward_quality",
        "mean_area_quality",
        "mean_target_quad_area",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for scope, label, group_evaluations in groups:
            stats = aggregate_statistics(group_evaluations)
            writer.writerow(
                {
                    "scope": scope,
                    "label": label,
                    "mesh_count": stats.mesh_count,
                    "triangles": stats.triangles,
                    "quads": stats.quads,
                    "triangle_to_quad_percent": format_csv_metric(
                        stats.triangle_to_quad_percent
                    ),
                    "singularities": stats.singularities,
                    "vertices": stats.vertices,
                    "singularity_to_vertex_percent": format_csv_metric(
                        stats.singularity_to_vertex_percent
                    ),
                    "mean_quality": format_csv_metric(stats.mean_quality),
                    "median_quality": format_csv_metric(stats.median_quality),
                    "mean_minimum_quality": format_csv_metric(stats.mean_minimum_quality),
                    "total_face_reward_return": format_csv_metric(
                        stats.total_face_reward_return
                    ),
                    "mean_face_reward_return": format_csv_metric(
                        stats.mean_face_reward_return
                    ),
                    "median_face_reward_return": format_csv_metric(
                        stats.median_face_reward_return
                    ),
                    "mean_reward_quality": format_csv_metric(stats.mean_reward_quality),
                    "mean_area_quality": format_csv_metric(stats.mean_area_quality),
                    "mean_target_quad_area": format_csv_metric(stats.mean_target_quad_area),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obj", nargs="*", type=Path, help="OBJ files or directories to evaluate")
    parser.add_argument(
        "--grid-config",
        type=Path,
        help="JSON configuration for a folder-by-filename evaluation matrix",
    )
    parser.add_argument("--output", "-o", type=Path, help="Save the figure (PDF, SVG, PNG, ...)")
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        help="Export overall and configured folder-row aggregate statistics as CSV",
    )
    parser.add_argument(
        "--qmshape-folder",
        type=Path,
        help="Enable face reward evaluation using stem-matched .qmshape files",
    )
    parser.add_argument(
        "--target-edge-length-ratio",
        type=float,
        default=1.0,
        help="Reward target edge length relative to mean boundary edge length",
    )
    parser.add_argument(
        "--base-quad-reward",
        type=float,
        default=0.0,
        help="Constant component of each quad reward (default: 0)",
    )
    parser.add_argument(
        "--triangle-reward",
        type=float,
        default=0.0,
        help="Reward contribution per triangle (default: 0)",
    )
    parser.add_argument(
        "--quad-reward-scale",
        type=float,
        default=0.5,
        help="Immediate reward multiplier per quad (default: 0.5)",
    )
    parser.add_argument("--columns", type=int, help="Number of columns in a multi-mesh grid")
    parser.add_argument("--cmap", default="RdYlGn", help="Matplotlib quality colormap")
    parser.add_argument(
        "--hide-singularities",
        action="store_true",
        help="Hide singularity markers while retaining their statistics",
    )
    parser.add_argument(
        "--low-percentile",
        type=float,
        default=10.0,
        help="Percentile reported as the low-quality statistic (default: 10)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output resolution (default: 300)")
    args = parser.parse_args()
    if args.columns is not None and args.columns < 1:
        parser.error("--columns must be at least 1")
    if not 0.0 <= args.low_percentile <= 100.0:
        parser.error("--low-percentile must be between 0 and 100")
    if bool(args.obj) == bool(args.grid_config):
        parser.error("provide either OBJ inputs or --grid-config, but not both")
    if args.grid_config and args.columns is not None:
        parser.error("--columns cannot be used with --grid-config")
    if args.grid_config and args.qmshape_folder is not None:
        parser.error("configure reward evaluation inside the grid JSON")

    try:
        grid_spec = load_grid_config(args.grid_config) if args.grid_config else None
        if grid_spec:
            paths = [path for row in grid_spec.paths for path in row]
            shape_paths = (
                [
                    grid_spec.qmshape_paths[index % len(grid_spec.column_labels)]
                    for index in range(len(paths))
                ]
                if grid_spec.qmshape_paths
                else None
            )
            reward_settings = grid_spec.reward_settings
        else:
            paths = collect_obj_paths(args.obj)
            if args.qmshape_folder:
                shape_folder = args.qmshape_folder.resolve()
                shape_paths = [shape_folder / f"{path.stem}.qmshape" for path in paths]
                reward_settings = RewardSettings(
                    target_edge_length_ratio=args.target_edge_length_ratio,
                    base_quad_reward=args.base_quad_reward,
                    triangle_reward=args.triangle_reward,
                    quad_reward_scale=args.quad_reward_scale,
                )
            else:
                shape_paths = None
                reward_settings = None

        shape_cache: dict[Path, QmShape] = {}
        evaluations: list[MeshEvaluation] = []
        for index, path in enumerate(paths):
            shape = None
            if shape_paths is not None:
                shape_path = shape_paths[index].resolve()
                if shape_path not in shape_cache:
                    shape_cache[shape_path] = load_qmshape(shape_path)
                shape = shape_cache[shape_path]
            evaluations.append(evaluate_mesh(parse_obj(path), shape, reward_settings))
        if args.aggregate_output:
            write_aggregate_csv(args.aggregate_output, evaluations, grid_spec)
    except (json.JSONDecodeError, OSError, ValueError) as error:
        parser.error(str(error))

    count = len(evaluations)
    if grid_spec:
        rows = len(grid_spec.row_labels)
        columns = len(grid_spec.column_labels)
    else:
        columns = args.columns or min(3, math.ceil(math.sqrt(count)))
        rows = math.ceil(count / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.1 * columns, 5.0 * rows),
        squeeze=False,
        layout="constrained",
    )
    for index, (ax, evaluation) in enumerate(zip(axes.flat, evaluations)):
        row, column = divmod(index, columns)
        if grid_spec:
            title = grid_spec.column_labels[column] if row == 0 else ""
        else:
            title = evaluation.mesh.path.stem
        plot_mesh(
            ax,
            evaluation,
            args.cmap,
            args.low_percentile,
            show_singularities=not args.hide_singularities,
            title=title,
        )
        if grid_spec and column == 0:
            ax.text(
                -0.075,
                0.5,
                grid_spec.row_labels[row],
                transform=ax.transAxes,
                ha="right",
                va="center",
                rotation=90,
                fontsize=11,
                fontweight="semibold",
                clip_on=False,
            )
    for ax in axes.flat[count:]:
        ax.set_visible(False)

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=args.cmap),
        ax=list(axes.flat[:count]),
        orientation="horizontal",
        fraction=0.025,
        pad=0.025,
        aspect=max(25, 12 * columns),
    )
    colorbar.set_label("Quadrilateral element quality", fontsize=10)
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    legend_handles = [
        Patch(facecolor="#f28e2b", edgecolor="#9c4d08", hatch="////", label="Triangle")
    ]
    if not args.hide_singularities:
        legend_handles.append(Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#cc2f7a",
            markeredgecolor="white",
            markersize=7,
            label="Interior singularity",
        ))
    if any(evaluation.other_faces for evaluation in evaluations):
        legend_handles.append(Patch(facecolor="#d5d8dc", label="Other polygon"))
    fig.legend(
        handles=legend_handles,
        loc="outside lower right",
        ncols=len(legend_handles),
        frameon=False,
        fontsize=9,
    )

    if args.output:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
