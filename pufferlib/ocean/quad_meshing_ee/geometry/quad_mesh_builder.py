import numpy as np
import shapely
from shapely import LineString, LinearRing, MultiLineString, MultiPolygon, Polygon, Point
from collections.abc import Sequence

from .util import compute_3_point_angle, make_ccw

from typing import Sequence, Optional, Tuple 

class BoundaryPoint:
    def __init__(self, x: float, y: float, polygon_idx: int, point_idx: int, is_interior: bool=False, interior_idx=None) -> None:
        self.xy = np.array([x, y])
        self.polygon_idx = polygon_idx
        self.point_idx = point_idx
        self.is_interior = is_interior
        self.interior_idx = interior_idx


class QuadMeshBuilder:
    def __init__(self, boundary: MultiPolygon) -> None:
        self.initial_boundary: MultiPolygon = make_ccw(boundary)
        self.boundary: MultiPolygon = make_ccw(boundary)

        # Mesh data
        self.V = [] # list of tuples: Shape (N, 2)
        self.F = [] # list of tuples: Shape (F, 4)


    def compute_unmeshed_ratio(self) -> float:
        return self.boundary.area / self.initial_boundary.area

    def make_quad(self, points: Sequence[np.ndarray]) -> Optional[Polygon]:
        """
        Creates a new quad from the given sequence of 4 points.
        Returns True if the quad was successfully created and False otherwise.
        """
        assert(len(points) == 4)

        quad = Polygon(points)
        if not quad.is_valid:
            return None

        if len(quad.exterior.coords) - 1 != 4:
            return None

        if not self.boundary.contains(quad):
            return None

        diff = shapely.difference(self.boundary, quad)
        if not isinstance(diff, MultiPolygon):
            if isinstance(diff, Polygon):
                diff = MultiPolygon([diff])
            else:
                return None

        self.boundary = make_ccw(diff)

        quad_idx = []
        for p in quad.exterior.coords[:4]:
            # Is exact comparison ok?
            idx = next((i for i, v in enumerate(self.V) if p[0] == v[0] and p[1] == v[1]), None)
            if idx is None:
                idx = len(self.V)
                self.V += [p]
            quad_idx += [idx]

        self.F += [quad_idx]

        return quad


    def get_neighbor(self, point: BoundaryPoint, offset: int, initial=False) -> BoundaryPoint:
        ring = self.get_boundary_point_ring(point, initial)

        # Exclude last redundant coordinate 
        idx = (point.point_idx + offset) % (len(ring.coords) - 1)
        x, y = ring.coords[idx]
        return BoundaryPoint(x, y, point.polygon_idx, idx, point.is_interior, point.interior_idx)

    
    def get_boundary_vertices(self, initial=False) -> np.ndarray:
        all_coords = []

        for poly in self.initial_boundary.geoms if initial else self.boundary.geoms:
            all_coords += [np.array(poly.exterior.coords[:-1])]
            for interior in poly.interiors:
                all_coords += [np.array(interior.coords[:-1])]

        return np.vstack(all_coords)


    def get_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.V), np.array(self.F)

    
    def iter_boundary_points(self, initial=False):
        for poly_idx, polygon in enumerate(self.initial_boundary.geoms if initial else self.boundary.geoms):
            # Exterior ring
            for point_idx, coord in enumerate(polygon.exterior.coords[:-1]):
                yield BoundaryPoint(coord[0], coord[1], poly_idx, point_idx)

            # Interior rings (holes)
            for interior_idx, interior in enumerate(polygon.interiors):
                for point_idx, coord in enumerate(interior.coords[:-1]):
                    yield BoundaryPoint(coord[0], coord[1], poly_idx, point_idx, True, interior_idx)


    def get_boundary_point_at(self, x: float, y: float, initial=False, tol=1e-6) -> Optional[BoundaryPoint]:
        sq_tol = tol**2
        for point in self.iter_boundary_points(initial):
            if np.sum((point.xy - np.array([x, y]))**2) < sq_tol:
                return point
        return None


    def get_boundary_point_ring(self, point: BoundaryPoint, initial=False) -> LinearRing:
        polygon = (self.initial_boundary.geoms if initial else self.boundary.geoms)[point.polygon_idx]

        if point.is_interior:
            assert(point.interior_idx is not None)
            return polygon.interiors[point.interior_idx]
        else:
            return polygon.exterior


    def intersect_ray(self,
                  ray_origin: np.ndarray,
                  ray_direction: np.ndarray,
                  max_length: float) -> np.ndarray:

        # 1) build the ray segment
        line = LineString([
            tuple(ray_origin),
            tuple(ray_origin + ray_direction * max_length)
        ])

        # 2) intersect with your polygon boundary
        intersection = shapely.intersection(line, self.boundary)

        # 3) collect *all* intersection coordinates
        pts = []
        if intersection.is_empty:
            # nothing hit within max_length → return the ray endpoint
            return ray_origin + ray_direction * max_length

        if isinstance(intersection, LineString):
            pts.extend(intersection.coords)
        elif isinstance(intersection, MultiLineString):
            for seg in intersection.geoms:
                pts.extend(seg.coords)
        elif isinstance(intersection, Point):
            pts.append(intersection.coords[0])
        else:
            # (if you ever get MultiPoint or GeometryCollection)
            for geom in getattr(intersection, "geoms", []):
                if hasattr(geom, "coords"):
                    pts.extend(geom.coords)

        # 4) make into an (N,2) array
        all_coords = np.array(pts, dtype=float)     # shape (N,2)
        if all_coords.size == 0:
            # still nothing usable
            return ray_origin + ray_direction * max_length

        # 5) find the intersection *closest* to the origin,
        #    but skip anything that is essentially the origin itself:
        sqd = np.sum((all_coords - ray_origin)**2, axis=1)
        mask = sqd > 1e-8
        if not np.any(mask):
            return ray_origin + ray_direction * max_length

        closest = all_coords[mask][ np.argmin(sqd[mask]) ]
        return closest


    def to_vertex_local(self, vertex: BoundaryPoint, points: np.ndarray) -> np.ndarray:
        angle = compute_3_point_angle(self.get_neighbor(vertex, -1).xy, vertex.xy, self.get_neighbor(vertex, 1).xy)
        v2 = self.get_neighbor(vertex, 1).xy - vertex.xy
        u2 = v2 / np.linalg.norm(v2)

        translation = -vertex.xy
        rotation = -(np.arctan2(u2[1], u2[0]) + 0.5 * angle)
        rotation_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)],
        ])

        return (rotation_matrix @ (points + translation).T).T


    def from_vertex_local(self, vertex: BoundaryPoint, points: np.ndarray) -> np.ndarray:
        angle = compute_3_point_angle(self.get_neighbor(vertex, -1).xy, vertex.xy, self.get_neighbor(vertex, 1).xy)
        v2 = self.get_neighbor(vertex, 1).xy - vertex.xy
        u2 = v2 / np.linalg.norm(v2)

        translation = vertex.xy
        rotation = (np.arctan2(u2[1], u2[0]) + 0.5 * angle)
        rotation_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)],
        ])

        return (rotation_matrix @ points.T).T + translation
