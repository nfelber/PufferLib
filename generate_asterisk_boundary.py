#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def _intersect_lines(p1, d1, p2, d2):
    det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
    if abs(det) < 1e-12:
        raise ValueError("Adjacent arm side lines are parallel; cannot build boundary.")
    rhs_x = p2[0] - p1[0]
    rhs_y = p2[1] - p1[1]
    t = (rhs_x * (-d2[1]) - rhs_y * (-d2[0])) / det
    return [p1[0] + t * d1[0], p1[1] + t * d1[1]]


def _build_control_polygon(num_arms, arm_length, arm_width):
    half_w = arm_width / 2.0
    valley_radius = half_w / math.tan(math.pi / num_arms)
    tip_radius = valley_radius + arm_length

    angles = [2.0 * math.pi * i / num_arms for i in range(num_arms)]
    u = [[math.cos(a), math.sin(a)] for a in angles]
    v = [[-math.sin(a), math.cos(a)] for a in angles]

    vertices = []
    for i in range(num_arms):
        j = (i + 1) % num_arms

        side_i_plus_point = [half_w * v[i][0], half_w * v[i][1]]
        side_j_minus_point = [-half_w * v[j][0], -half_w * v[j][1]]
        valley = _intersect_lines(side_i_plus_point, u[i], side_j_minus_point, u[j])

        tip_j_minus = [tip_radius * u[j][0] - half_w * v[j][0], tip_radius * u[j][1] - half_w * v[j][1]]
        tip_j_plus = [tip_radius * u[j][0] + half_w * v[j][0], tip_radius * u[j][1] + half_w * v[j][1]]

        vertices.extend([valley, tip_j_minus, tip_j_plus])

    return vertices


def _sample_edges_preserve_corners(vertices, spacing=1.0):
    if spacing <= 0:
        raise ValueError("Spacing must be positive.")

    sampled = [vertices[0]]
    for i in range(len(vertices)):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % len(vertices)]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        length = math.hypot(dx, dy)
        if length <= 1e-12:
            continue

        steps = int(length // spacing)
        for k in range(1, steps + 1):
            dist = k * spacing
            if dist >= length - 1e-12:
                break
            t = dist / length
            sampled.append([p0[0] + t * dx, p0[1] + t * dy])
        sampled.append(p1)

    if len(sampled) >= 2:
        start = sampled[0]
        end = sampled[-1]
        if abs(start[0] - end[0]) <= 1e-12 and abs(start[1] - end[1]) <= 1e-12:
            sampled.pop()

    if len(sampled) < 3:
        raise ValueError("Degenerate boundary: fewer than 3 vertices.")

    return sampled


def _normalize_to_unit_box(vertices):
    xs = [p[0] for p in vertices]
    ys = [p[1] for p in vertices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y)
    if span <= 1e-12:
        raise ValueError("Degenerate boundary: zero span.")

    return [[(x - min_x) / span, (y - min_y) / span] for x, y in vertices]


def generate_asterisk_boundary(num_arms, arm_length, arm_width):
    if num_arms < 2:
        raise ValueError("Number of arms must be at least 2.")
    if arm_length <= 0 or arm_width <= 0:
        raise ValueError("Arm length and arm width must be positive integers.")

    control = _build_control_polygon(num_arms, arm_length, arm_width)

    sampled = _sample_edges_preserve_corners(control, spacing=1.0)
    normalized = _normalize_to_unit_box(sampled)
    return {"vertices": normalized}


def main():
    parser = argparse.ArgumentParser(
        description="Generate an asterisk-shaped boundary JSON with evenly distributed vertices."
    )
    parser.add_argument("num_arms", type=int, help="Number of rectangular arms (>= 2).")
    parser.add_argument(
        "arm_length",
        type=int,
        help="Arm rectangle side length in integer units (from valley corner to tip corner).",
    )
    parser.add_argument("arm_width", type=int, help="Arm width in integer units.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to boundaries/asterisk_<arms>_<length>_<width>.json",
    )
    args = parser.parse_args()

    boundary = generate_asterisk_boundary(args.num_arms, args.arm_length, args.arm_width)

    output = args.output
    if output is None:
        output = Path(__file__).resolve().parent / f"asterisk_{args.num_arms}_{args.arm_length}_{args.arm_width}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        json.dump(boundary, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(boundary['vertices'])} vertices to {output}")


if __name__ == "__main__":
    main()
