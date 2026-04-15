import numpy as np
import pygame

from ..geometry.quad_mesh_builder import QuadMeshBuilder


class BasicRenderer:
    WINDOW_MARGIN = 0.1 # Fraction
    BG_COLOR = (30, 30, 30)

    def __init__(self, max_window_size: int, bounds) -> None:
        self.max_window_size = max_window_size

        self.window = None
        self.clock = None

        self.set_bounds(bounds)


    def _to_screen(self, coords: np.ndarray):
        return ((coords - self.origin) * self.scaling + 0.5 * self.window_size).tolist()


    def _from_screen(self, coords: np.ndarray):
        return (coords - 0.5 * self.window_size) / self.scaling + self.origin


    def create_window(self, caption: str, *args, **kwargs):
        pygame.init()
        pygame.display.set_caption(caption)
        self.window = pygame.display.set_mode(self.window_size.tolist(), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()


    def render_frame(self, framerate, *args, **kwargs):
        assert(self.clock is not None)
        self._render_frame(*args, **kwargs)
        pygame.display.flip()
        self.clock.tick(framerate)


    def _render_frame(self, *args, **kwargs):
        assert(self.window is not None)
        self.window.fill(self.BG_COLOR)


    def close(self, *args, **kwargs):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


    def set_bounds(self, bounds) -> None:
        minx, miny, maxx, maxy = bounds

        if self.window is None:
            width = maxx - minx
            height = maxy - miny
            window_width = self.max_window_size * min(1, width / height)
            window_height = self.max_window_size * min(1, height / width)

            self.window_size = np.array([window_width, window_height])

        self.origin = 0.5 * np.array([minx + maxx, miny + maxy])
        self.scaling = self.window_size * (1 - 2 * self.WINDOW_MARGIN) / np.array([maxx - minx, miny - maxy])



class QuadMeshBuilderRenderer(BasicRenderer):
    BOUNDARY_LINE_COLOR = (0, 200, 200)
    BOUNDARY_LINE_WIDTH = 2
    MESH_LINE_COLOR = (200, 200, 200)
    MESH_LINE_WIDTH = 1
    VERTEX_COLOR = (100, 200, 255)
    VERTEX_RADIUS = 5

    def __init__(self, window_size: int, bounds) -> None:
        super().__init__(window_size, bounds)


    def _render_frame(self, mesh_builder: QuadMeshBuilder, *args, **kwargs):
        super()._render_frame()
        assert(self.window is not None)

        # Draw mesh
        V, F = mesh_builder.get_mesh()
        for face in F:
            pygame.draw.polygon(self.window, self.MESH_LINE_COLOR, [self._to_screen(V[i]) for i in face], width=self.MESH_LINE_WIDTH)

        # Draw boundary
        for point in mesh_builder.iter_boundary_points():
            neighbor = mesh_builder.get_neighbor(point, 1)
            pygame.draw.line(self.window, self.BOUNDARY_LINE_COLOR, self._to_screen(point.xy), self._to_screen(neighbor.xy), width=self.BOUNDARY_LINE_WIDTH)
            pygame.draw.circle(self.window, self.VERTEX_COLOR, self._to_screen(point.xy), self.VERTEX_RADIUS)
