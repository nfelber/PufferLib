from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import gmsh
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from torch._C import dtype


# -----------------------------------------------------------------------------
# JSON boundary loading
# -----------------------------------------------------------------------------


def load_boundary_json(path: str | Path) -> np.ndarray:
    """
    Expected format:
        {
          "vertices": [
            [x0, y0],
            [x1, y1],
            ...
          ]
        }

    Returns:
        boundary: [B, 2], not explicitly closed.
    """
    with open(path, "r") as f:
        data = json.load(f)

    if "vertices" not in data:
        raise ValueError("JSON file must contain a 'vertices' field.")

    V = np.asarray(data["vertices"], dtype=np.float64)

    if V.ndim != 2 or V.shape[1] != 2:
        raise ValueError("'vertices' must have shape [num_vertices, 2].")

    if len(V) < 3:
        raise ValueError("Boundary must contain at least 3 vertices.")

    V = remove_duplicate_boundary_vertices(V)

    if len(V) < 3:
        raise ValueError("Boundary has fewer than 3 unique vertices after cleanup.")

    return V


def remove_duplicate_boundary_vertices(
    V: np.ndarray,
    eps: float = 1e-14,
) -> np.ndarray:
    """
    Removes consecutive duplicate points and a repeated final point if present.
    """
    cleaned = [V[0]]

    for p in V[1:]:
        if np.linalg.norm(p - cleaned[-1]) > eps:
            cleaned.append(p)

    V = np.asarray(cleaned, dtype=np.float64)

    if len(V) > 1 and np.linalg.norm(V[0] - V[-1]) <= eps:
        V = V[:-1]

    return V


# -----------------------------------------------------------------------------
# Gmsh polygon triangulation
# -----------------------------------------------------------------------------


def default_mesh_size(
    boundary: np.ndarray,
    points_per_bbox_diag: float = 80.0,
) -> float:
    bbox_min = boundary.min(axis=0)
    bbox_max = boundary.max(axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min)
    return float(diag / points_per_bbox_diag)


def triangulate_polygon_with_gmsh(
    boundary: np.ndarray,
    mesh_size: float | None = None,
    msh_debug_path: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate a simple polygon boundary with Gmsh.

    Args:
        boundary: [B, 2], not explicitly closed.
        mesh_size: target size. If None, inferred from bbox diagonal.
        msh_debug_path: optional .msh debug output.

    Returns:
        V: [N, 2]
        F: [T, 3]
    """
    if mesh_size is None:
        mesh_size = default_mesh_size(boundary)

    gmsh.initialize()

    try:
        gmsh.model.add("json_polygon")

        point_tags = []
        for x, y in boundary:
            point_tags.append(
                gmsh.model.geo.addPoint(
                    float(x),
                    float(y),
                    0.0,
                    float(mesh_size),
                )
            )

        line_tags = []
        n = len(point_tags)

        for i in range(n):
            a = point_tags[i]
            b = point_tags[(i + 1) % n]
            line_tags.append(gmsh.model.geo.addLine(a, b))

        loop = gmsh.model.geo.addCurveLoop(line_tags)
        gmsh.model.geo.addPlaneSurface([loop])

        gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        gmsh.model.mesh.generate(2)

        V, F = extract_triangle_mesh_from_gmsh()

        if msh_debug_path is not None:
            gmsh.write(str(msh_debug_path))

        return V, F

    finally:
        gmsh.finalize()


def extract_triangle_mesh_from_gmsh() -> Tuple[np.ndarray, np.ndarray]:
    node_tags, coords, _ = gmsh.model.mesh.getNodes()

    node_tags = np.asarray(node_tags, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)

    V = coords[:, :2].copy()

    node_tag_to_i = {int(tag): i for i, tag in enumerate(node_tags)}

    elem_types, elem_tags_by_type, elem_nodes_by_type = gmsh.model.mesh.getElements(dim=2)

    triangles = []

    for elem_type, _, elem_nodes in zip(
        elem_types,
        elem_tags_by_type,
        elem_nodes_by_type,
    ):
        # Gmsh type 2 = 3-node triangle.
        if elem_type != 2:
            continue

        elem_nodes = np.asarray(elem_nodes, dtype=np.int64).reshape(-1, 3)

        for nodes in elem_nodes:
            triangles.append([node_tag_to_i[int(n)] for n in nodes])

    if not triangles:
        raise RuntimeError("No linear triangles found in Gmsh mesh.")

    F = np.asarray(triangles, dtype=np.int64)

    return V, F


# -----------------------------------------------------------------------------
# OBJ export
# -----------------------------------------------------------------------------


def write_obj(
    obj_path: str | Path,
    V: np.ndarray,
    F: np.ndarray,
) -> None:
    """
    Write a triangular OBJ file.

    Args:
        V: [N, 2] or [N, 3]
        F: [T, 3], zero-indexed
    """
    obj_path = Path(obj_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)

    V = np.asarray(V)
    F = np.asarray(F)

    if V.ndim != 2 or V.shape[1] not in (2, 3):
        raise ValueError(f"V must have shape [N, 2] or [N, 3], got {V.shape}.")

    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"F must have shape [T, 3], got {F.shape}.")

    if V.shape[1] == 2:
        V3 = np.column_stack(
            [
                V[:, 0],
                V[:, 1],
                np.zeros(len(V), dtype=V.dtype),
            ]
        )
    else:
        V3 = V

    with open(obj_path, "w") as f:
        f.write("# Generated by cross_field_tools.py\n")

        for x, y, z in V3:
            f.write(f"v {float(x):.17g} {float(y):.17g} {float(z):.17g}\n")

        for a, b, c in F:
            # OBJ is 1-indexed.
            f.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")

    print(f"Saved OBJ: {obj_path}")
    print(f"Vertices: {V3.shape[0]}")
    print(f"Triangles: {F.shape[0]}")


def triangulate_boundary_to_obj(
    boundary_json: str | Path,
    obj_path: str | Path,
    mesh_size: float | None = None,
    msh_debug_path: str | Path | None = None,
) -> None:
    boundary = load_boundary_json(boundary_json)

    V, F = triangulate_polygon_with_gmsh(
        boundary=boundary,
        mesh_size=mesh_size,
        msh_debug_path=None if msh_debug_path is None else str(msh_debug_path),
    )

    write_obj(obj_path, V, F)


# -----------------------------------------------------------------------------
# Face-based cross-field query object
# -----------------------------------------------------------------------------


@dataclass
class CrossField2DFace:
    V: torch.Tensor  # [N, 2]
    F: torch.Tensor  # [T, 3]
    z: torch.Tensor  # [T, 2] face-based exp(4 i theta)
    boundary: torch.Tensor | None = None  # [B, 2] or [B, 3]
    face_barycenters: torch.Tensor | None = None  # [T, 2]
    face_normals: torch.Tensor | None = None
    dir_u: torch.Tensor | None = None
    dir_v: torch.Tensor | None = None

    @staticmethod
    def from_npz(
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "CrossField2DFace":
        data = np.load(path)

        V_np = np.asarray(data["V"], dtype=np.float32)
        F_np = np.asarray(data["F"], dtype=np.int64)
        z_np = np.asarray(data["z"], dtype=np.float32)

        if V_np.ndim != 2 or V_np.shape[1] not in (2, 3):
            raise ValueError(f"Expected V with shape [N, 2] or [N, 3], got {V_np.shape}.")

        if F_np.ndim != 2 or F_np.shape[1] != 3:
            raise ValueError(f"Expected F with shape [T, 3], got {F_np.shape}.")

        if z_np.ndim != 2 or z_np.shape[1] != 2:
            raise ValueError(f"Expected z with shape [T, 2], got {z_np.shape}.")

        if z_np.shape[0] != F_np.shape[0]:
            raise ValueError(
                "This visualizer expects the new face-based field format: "
                f"z.shape[0] == num_faces. Got z.shape={z_np.shape}, "
                f"num_faces={F_np.shape[0]}."
            )

        # Visualization is 2D; keep x/y only.
        V2_np = V_np[:, :2].copy()

        boundary = None
        if "boundary" in data:
            B_np = np.asarray(data["boundary"], dtype=np.float32)
            if B_np.ndim == 2 and B_np.shape[1] >= 2 and len(B_np) > 0:
                boundary = torch.tensor(B_np[:, :2], dtype=torch.float32, device=device)

        face_barycenters = None
        if "face_barycenters" in data:
            C_np = np.asarray(data["face_barycenters"], dtype=np.float32)
            if C_np.ndim == 2 and C_np.shape[1] >= 2:
                face_barycenters = torch.tensor(
                    C_np[:, :2],
                    dtype=torch.float32,
                    device=device,
                )

        face_normals = None
        if "face_normals" in data:
            N_np = np.asarray(data["face_normals"], dtype=np.float32)
            if N_np.ndim == 2 and N_np.shape[1] == 3:
                face_normals = torch.tensor(
                    N_np,
                    dtype=torch.float32,
                    device=device,
                )

        if "dir_u" in data and "dir_v" in data:
            dir_u = torch.tensor(np.asarray(data["dir_u"], dtype=np.float32)[:, :2], dtype=torch.float32, device=device)
            dir_v = torch.tensor(np.asarray(data["dir_v"], dtype=np.float32)[:, :2], dtype=torch.float32, device=device)
        else:
            dir_u = None
            dir_v = None

        return CrossField2DFace(
            V=torch.tensor(V2_np, dtype=torch.float32, device=device),
            F=torch.tensor(F_np, dtype=torch.long, device=device),
            z=torch.tensor(z_np, dtype=torch.float32, device=device),
            boundary=boundary,
            face_barycenters=face_barycenters,
            face_normals=face_normals,
            dir_u=dir_u,
            dir_v=dir_v,
        )

    def query(
        self,
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Query the face-based cross field at arbitrary 2D points.

        Args:
            points: [Q, 2]

        Returns:
            u:         [Q, 2], first direction
            v:         [Q, 2], second orthogonal direction
            tri_index: [Q], containing triangle index, or -1 outside

        Note:
            This assumes the cached z values are suitable for 2D global plotting.
            For the flat meshes produced by this script, this is the intended use.
        """
        points = points.to(self.V.device, dtype=torch.float32)

        tri_index, _bary = self._locate_points_bruteforce(points)

        u = torch.full_like(points, float("nan"))
        v = torch.full_like(points, float("nan"))

        inside = tri_index >= 0
        if not inside.any():
            return u, v, tri_index

        if self.dir_u is not None and self.dir_v is not None:
            u_inside = self.dir_u[tri_index[inside]]
            v_inside = self.dir_v[tri_index[inside]]

            u_inside = u_inside / u_inside.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            v_inside = v_inside / v_inside.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            # fallback old intrinsic z interpretation
            zq = self.z[tri_index[inside]]
            zq = zq / zq.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            theta = 0.25 * torch.atan2(zq[:, 1], zq[:, 0])
            u_inside = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
            v_inside = torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)

        u[inside] = u_inside
        v[inside] = v_inside

        return u, v, tri_index

    def _locate_points_bruteforce(
        self,
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Brute-force triangle lookup.

        This is fine for visualization and small tests. For training-scale
        sampling, replace this with a spatial index.
        """
        V = self.V
        F = self.F

        A = V[F[:, 0]]
        B = V[F[:, 1]]
        C = V[F[:, 2]]

        Q = points.shape[0]
        T = F.shape[0]

        tri_index = torch.full(
            (Q,),
            -1,
            dtype=torch.long,
            device=points.device,
        )

        bary = torch.full(
            (Q, 3),
            float("nan"),
            dtype=torch.float32,
            device=points.device,
        )

        eps = 1e-6

        for t in range(T):
            a = A[t]
            b = B[t]
            c = C[t]

            v0 = b - a
            v1 = c - a
            v2 = points - a

            den = v0[0] * v1[1] - v1[0] * v0[1]
            if torch.abs(den) < 1e-14:
                continue

            w1 = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den
            w2 = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den
            w0 = 1.0 - w1 - w2

            hit = (
                (tri_index < 0)
                & (w0 >= -eps)
                & (w1 >= -eps)
                & (w2 >= -eps)
            )

            tri_index[hit] = t
            bary[hit, 0] = w0[hit]
            bary[hit, 1] = w1[hit]
            bary[hit, 2] = w2[hit]

        return tri_index, bary


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def make_regular_grid(
    V: np.ndarray,
    resolution: int = 45,
    padding_ratio: float = 0.03,
) -> np.ndarray:
    bbox_min = V.min(axis=0)
    bbox_max = V.max(axis=0)

    span = bbox_max - bbox_min
    pad = padding_ratio * max(float(span[0]), float(span[1]))

    xmin, ymin = bbox_min - pad
    xmax, ymax = bbox_max + pad

    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)

    X, Y = np.meshgrid(xs, ys)

    return np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)


def estimate_segment_length(
    V_np: np.ndarray,
    segment_length: float | None,
) -> float:
    if segment_length is not None:
        return float(segment_length)

    bbox = V_np.max(axis=0) - V_np.min(axis=0)
    return 0.018 * float(max(bbox[0], bbox[1]))


def plot_cross_field_from_cache(
    cache_path: str | Path,
    plot_path: str | Path | None = "cross_field_grid.png",
    grid_res: int = 45,
    segment_length: float | None = None,
    show_mesh: bool = True,
    show_boundary: bool = True,
    show: bool = True,
    device: str = "cpu",
) -> None:
    field = CrossField2DFace.from_npz(cache_path, device=device)

    V_np = field.V.detach().cpu().numpy()
    F_np = field.F.detach().cpu().numpy()

    segment_length = estimate_segment_length(V_np, segment_length)

    P_np = make_regular_grid(V_np, resolution=grid_res)
    P = torch.tensor(P_np, dtype=torch.float32, device=field.V.device)

    with torch.no_grad():
        u, v, tri_index = field.query(P)

    u_np = u.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()
    tri_np = tri_index.detach().cpu().numpy()

    inside = tri_np >= 0

    P_in = P_np[inside]
    u_in = u_np[inside]
    v_in = v_np[inside]

    fig, ax = plt.subplots(figsize=(8, 8))

    if show_mesh:
        triangulation = mtri.Triangulation(V_np[:, 0], V_np[:, 1], F_np)
        ax.triplot(triangulation, linewidth=0.35, alpha=0.35)

    if show_boundary and field.boundary is not None:
        B = field.boundary.detach().cpu().numpy()

        if len(B) > 0:
            B_closed = np.vstack([B, B[0]])
            ax.plot(B_closed[:, 0], B_closed[:, 1], linewidth=1.5)

    # First direction family.
    ax.quiver(
        P_in[:, 0],
        P_in[:, 1],
        u_in[:, 0],
        u_in[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0 / segment_length,
        width=0.0012,
        headwidth=0,
        headlength=0,
        headaxislength=0,
        pivot="middle",
        alpha=0.9,
        color="r",
    )

    # Orthogonal direction family.
    ax.quiver(
        P_in[:, 0],
        P_in[:, 1],
        v_in[:, 0],
        v_in[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0 / segment_length,
        width=0.0012,
        headwidth=0,
        headlength=0,
        headaxislength=0,
        pivot="middle",
        alpha=0.9,
        color="b",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Face-based 4-RoSy cross field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    bbox_min = V_np.min(axis=0)
    bbox_max = V_np.max(axis=0)
    span = bbox_max - bbox_min
    pad = 0.05 * max(float(span[0]), float(span[1]))

    ax.set_xlim(bbox_min[0] - pad, bbox_max[0] + pad)
    ax.set_ylim(bbox_min[1] - pad, bbox_max[1] + pad)

    plt.tight_layout()

    if plot_path is not None:
        plt.savefig(plot_path, dpi=300)
        print(f"Saved plot: {plot_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_face_centers_from_cache(
    cache_path: str | Path,
    plot_path: str | Path | None = "cross_field_faces.png",
    segment_length: float | None = None,
    show_mesh: bool = True,
    show_boundary: bool = True,
    show: bool = True,
    device: str = "cpu",
) -> None:
    """
    Alternative visualization: one cross glyph at each face barycenter.

    This avoids point-location and shows the field exactly at its native
    face-based locations.
    """
    field = CrossField2DFace.from_npz(cache_path, device=device)

    V_np = field.V.detach().cpu().numpy()
    F_np = field.F.detach().cpu().numpy()
    z_np = field.z.detach().cpu().numpy()

    if field.face_barycenters is not None:
        P_np = field.face_barycenters.detach().cpu().numpy()
    else:
        P_np = V_np[F_np].mean(axis=1)

    segment_length = estimate_segment_length(V_np, segment_length)

    z_norm = z_np / np.maximum(np.linalg.norm(z_np, axis=1, keepdims=True), 1e-12)
    theta = 0.25 * np.arctan2(z_norm[:, 1], z_norm[:, 0])

    u_np = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    v_np = np.stack([-np.sin(theta), np.cos(theta)], axis=-1)

    fig, ax = plt.subplots(figsize=(8, 8))

    if show_mesh:
        triangulation = mtri.Triangulation(V_np[:, 0], V_np[:, 1], F_np)
        ax.triplot(triangulation, linewidth=0.35, alpha=0.35)

    if show_boundary and field.boundary is not None:
        B = field.boundary.detach().cpu().numpy()

        if len(B) > 0:
            B_closed = np.vstack([B, B[0]])
            ax.plot(B_closed[:, 0], B_closed[:, 1], linewidth=1.5)

    ax.quiver(
        P_np[:, 0],
        P_np[:, 1],
        u_np[:, 0],
        u_np[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0 / segment_length,
        width=0.0012,
        headwidth=0,
        headlength=0,
        headaxislength=0,
        pivot="middle",
        alpha=0.9,
        color="r",
    )

    ax.quiver(
        P_np[:, 0],
        P_np[:, 1],
        v_np[:, 0],
        v_np[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0 / segment_length,
        width=0.0012,
        headwidth=0,
        headlength=0,
        headaxislength=0,
        pivot="middle",
        alpha=0.9,
        color="b",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Face-centered 4-RoSy cross field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    bbox_min = V_np.min(axis=0)
    bbox_max = V_np.max(axis=0)
    span = bbox_max - bbox_min
    pad = 0.05 * max(float(span[0]), float(span[1]))

    ax.set_xlim(bbox_min[0] - pad, bbox_max[0] + pad)
    ax.set_ylim(bbox_min[1] - pad, bbox_max[1] + pad)

    plt.tight_layout()

    if plot_path is not None:
        plt.savefig(plot_path, dpi=300)
        print(f"Saved plot: {plot_path}")

    if show:
        plt.show()

    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tools for 2D boundary triangulation and face-based cross-field visualization.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # -------------------------------------------------------------------------
    # triangulate
    # -------------------------------------------------------------------------
    tri = subparsers.add_parser(
        "triangulate",
        help="Triangulate a 2D boundary JSON and export a triangular OBJ.",
    )

    tri.add_argument(
        "--boundary",
        type=str,
        required=True,
        help="Path to JSON boundary file with {'vertices': [[x, y], ...]}.",
    )

    tri.add_argument(
        "--obj",
        type=str,
        required=True,
        help="Path to output OBJ file.",
    )

    tri.add_argument(
        "--mesh-size",
        type=float,
        default=None,
        help="Target Gmsh mesh size. If omitted, inferred from bbox diagonal.",
    )

    tri.add_argument(
        "--msh-debug",
        type=str,
        default=None,
        help="Optional path to save a debug .msh file.",
    )

    # -------------------------------------------------------------------------
    # visualize
    # -------------------------------------------------------------------------
    vis = subparsers.add_parser(
        "visualize",
        help="Visualize a face-based cross-field cache exported by the C++ program.",
    )

    vis.add_argument(
        "--cache",
        type=str,
        required=True,
        help="Path to .npz cross-field cache.",
    )

    vis.add_argument(
        "--plot",
        type=str,
        default="cross_field_grid.png",
        help="Path to save the plot. Use --plot none to disable saving.",
    )

    vis.add_argument(
        "--grid-res",
        type=int,
        default=45,
        help="Regular grid resolution per axis for visualization.",
    )

    vis.add_argument(
        "--segment-length",
        type=float,
        default=None,
        help="Length of plotted direction segments. If omitted, chosen from bbox size.",
    )

    vis.add_argument(
        "--face-centers",
        action="store_true",
        help="Plot one cross at each face barycenter instead of querying a regular grid.",
    )

    vis.add_argument(
        "--no-mesh",
        action="store_true",
        help="Do not draw the triangle mesh.",
    )

    vis.add_argument(
        "--no-boundary",
        action="store_true",
        help="Do not draw the boundary.",
    )

    vis.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot but do not open a matplotlib window.",
    )

    vis.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used for field queries.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "triangulate":
        triangulate_boundary_to_obj(
            boundary_json=args.boundary,
            obj_path=args.obj,
            mesh_size=args.mesh_size,
            msh_debug_path=args.msh_debug,
        )

    elif args.command == "visualize":
        plot_path = None if args.plot.lower() == "none" else args.plot

        if args.face_centers:
            plot_face_centers_from_cache(
                cache_path=args.cache,
                plot_path=plot_path,
                segment_length=args.segment_length,
                show_mesh=not args.no_mesh,
                show_boundary=not args.no_boundary,
                show=not args.no_show,
                device=args.device,
            )
        else:
            plot_cross_field_from_cache(
                cache_path=args.cache,
                plot_path=plot_path,
                grid_res=args.grid_res,
                segment_length=args.segment_length,
                show_mesh=not args.no_mesh,
                show_boundary=not args.no_boundary,
                show=not args.no_show,
                device=args.device,
            )

    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
