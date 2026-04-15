import numpy as np
from shapely import LinearRing
from shapely.geometry import Polygon, MultiPolygon
from typing import Tuple


def get_nearest_points_in_cone(points: np.ndarray, cone_origin: np.ndarray,
                               cone_angle_from: float, cone_angle_to: float,
                               cone_radius: float, subdivisions: int=1) -> list[np.ndarray]:
    full_angle = (cone_angle_to - cone_angle_from) % (2 * np.pi)
    sub_angles = np.linspace(-0.5, 0.5, subdivisions + 1) * full_angle
    thresholds = list(np.cos(sub_angles[:len(sub_angles)//2])) + [1]

    half_angle = cone_angle_from + 0.5 * full_angle
    cone_dir = np.array([np.cos(half_angle), np.sin(half_angle)])

    points_relative = points - cone_origin
    dists = np.linalg.norm(points_relative, axis=1)
    projections = points_relative @ cone_dir.T

    cos_angles = np.divide(projections, dists, where=dists != 0, out=None)

    perp_projections = points_relative @ np.array([cone_dir[1], -cone_dir[0]]).T

    inside_subcone = [(cos_angles > thresholds[i]) & (cos_angles <= thresholds[i+1]) & (dists < cone_radius) for i in range(len(thresholds) - 1)]
    for i in range(subdivisions//2 - 1, -1, -1):
        inside_subcone += [inside_subcone[i] & (perp_projections < 0)]
        inside_subcone[i] &= perp_projections >= 0

    nearest_points = []

    for mask in inside_subcone:
        if not mask.any():
            nearest_points += [None]
        else:
            nearest_idx = np.argmin(dists[mask])
            nearest_points += [points[mask][nearest_idx]]

    return nearest_points


def compute_3_point_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    v1 = p2 - p1
    v2 = p3 - p2
    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)
    u1 = v1 / l1
    u2 = v2 / l2

    angle = np.arccos(np.clip(np.dot(-u1, u2), -1.0, 1.0))

    return angle if np.cross(u1, u2) > 0 else 2*np.pi - angle


def compute_vector_angle(v: np.ndarray) -> float:
    u = v / np.linalg.norm(v)
    return np.arctan2(u[1], u[0])


def measure_quad(quad: Polygon) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def make_ccw(mp: MultiPolygon) -> MultiPolygon:
    def make_ring_ccw(ring):
        if not LinearRing(ring).is_ccw:
            return ring[::-1]
        return ring


    new_polygons = []
    for polygon in mp.geoms:
        # Ensure exterior is CCW
        exterior = make_ring_ccw(polygon.exterior.coords[:])

        # Ensure all interiors are also CCW
        interiors = [make_ring_ccw(interior.coords[:]) for interior in polygon.interiors]

        new_polygons.append(Polygon(exterior, interiors))

    return MultiPolygon(new_polygons)
