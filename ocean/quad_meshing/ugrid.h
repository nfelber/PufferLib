#pragma once

#include "geometry.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>

typedef struct {
    unsigned int resolution;
    float inv_cell_size;

    IntArray cell_data;
    IntArray cell_item_count;
    IntArray cell_next;
    unsigned int cell_capacity;
} UGrid;

void ugrid_reset(UGrid* grid) {
    int cell_count = grid->resolution * grid->resolution;
    IntArray_resize(&grid->cell_data, cell_count * grid->cell_capacity);
    IntArray_resize(&grid->cell_item_count, cell_count);
    IntArray_resize(&grid->cell_next, cell_count);
    for (int i=0; i<cell_count; ++i) {
        grid->cell_item_count.data[i] = 0;
        grid->cell_next.data[i] = -1;
    }
}

void ugrid_init(
    UGrid* grid,
    unsigned int resolution,
    float cell_size,
    unsigned int cell_capacity
) {
    grid->resolution = resolution;
    grid->inv_cell_size = 1 / cell_size;
    grid->cell_capacity = cell_capacity;

    IntArray_init(&grid->cell_data);
    IntArray_init(&grid->cell_item_count);
    IntArray_init(&grid->cell_next);

    ugrid_reset(grid);
}

void ugrid_free(UGrid* grid) {
    IntArray_free(&grid->cell_data);
    IntArray_free(&grid->cell_item_count);
    IntArray_free(&grid->cell_next);
}

static int ugrid_wrap_index(int i, unsigned int resolution) {
    i %= resolution;
    return i < 0 ? i + resolution : i;
}

static size_t ugrid_cell_idx(const UGrid* grid, int cell_x, int cell_y) {
    int x = ugrid_wrap_index(cell_x, grid->resolution);
    int y = ugrid_wrap_index(cell_y, grid->resolution);

    return (size_t)y * (size_t)grid->resolution + (size_t)x;
}

size_t ugrid_position_to_cell(const UGrid* grid, Vec2 p) {
    int cell_x = (int)floorf(grid->inv_cell_size * p.x);
    int cell_y = (int)floorf(grid->inv_cell_size * p.y);
    int x = ugrid_wrap_index(cell_x, grid->resolution);
    int y = ugrid_wrap_index(cell_y, grid->resolution);

    return (size_t)y * (size_t)grid->resolution + (size_t)x;
}

void ugrid_place(UGrid* grid, int cell, int v) {
    int next_cell = grid->cell_next.data[cell];
    int item_count = grid->cell_item_count.data[cell];
    if (item_count < grid->cell_capacity) {
        grid->cell_data.data[cell * grid->cell_capacity + item_count] = v;
        grid->cell_item_count.data[cell]++;
    } else if (next_cell != -1) {
        // Follow to next cell
        ugrid_place(grid, next_cell, v);
    } else {
        // Create new cell
        IntArray_push(&grid->cell_data, v);
        IntArray_push(&grid->cell_item_count, 1);
        for (int i=1; i<grid->cell_capacity; ++i) {
            IntArray_push(&grid->cell_data, 0);
        }
        grid->cell_next.data[cell] = grid->cell_next.size;
        IntArray_push(&grid->cell_next, -1);
    }
}

typedef struct {
    UGrid* grid; // Non-owning
    size_t cell;
    int i;
} UGridCellIterator;

DEFINE_VECTOR(UGridCellIterator, UGridCellIteratorArray)

int ugrid_cell_it_next(UGridCellIterator* it) {
    const UGrid* grid = it->grid;

    int item_count = grid->cell_item_count.data[it->cell];
    if (it->i >= item_count) {
        int next_cell = grid->cell_next.data[it->cell];
        if (next_cell != -1) {
            it->cell = next_cell;
            it->i = 0;
            return ugrid_cell_it_next(it);
        } else {
            return -1;
        }
    }

    return grid->cell_data.data[it->cell * grid->cell_capacity + it->i++];
}

UGridCellIterator ugrid_point_query(const UGrid* grid, Vec2 p) {
    return (UGridCellIterator) {
        .grid = grid,
        .cell = ugrid_position_to_cell(grid, p),
        .i = 0,
    };
}

void ugrid_segment_query_dda(const UGrid* grid, Vec2 from, Vec2 to, UGridCellIteratorArray* it) {
    const float eps_t = 1e-7f;

    float x0f = grid->inv_cell_size * from.x;
    float y0f = grid->inv_cell_size * from.y;
    float x1f = grid->inv_cell_size * to.x;
    float y1f = grid->inv_cell_size * to.y;

    int x =     (int)floorf(x0f);
    int y =     (int)floorf(y0f);
    int end_x = (int)floorf(x1f);
    int end_y = (int)floorf(y1f);

    float dx = x1f - x0f;
    float dy = y1f - y0f;

    int step_x = dx > 0.0f ? 1 : dx < 0.0f ? -1 : 0;
    int step_y = dy > 0.0f ? 1 : dy < 0.0f ? -1 : 0;

    float t_max_x;
    float t_max_y;
    float t_delta_x;
    float t_delta_y;

    if (step_x != 0) {
        float next_boundary_x = (float)x + (step_x > 0 ? 1.0f : 0.0f);
        t_max_x = (next_boundary_x - x0f) / dx;
        t_delta_x = 1.0f / fabsf(dx);
    } else {
        t_max_x = HUGE_VALF;
        t_delta_x = HUGE_VALF;
    }

    if (step_y != 0) {
        float next_boundary_y = (float)y + (step_y > 0 ? 1.0f : 0.0f);
        t_max_y = (next_boundary_y - y0f) / dy;
        t_delta_y = 1.0f / fabsf(dy);
    } else {
        t_max_y = HUGE_VALF;
        t_delta_y = HUGE_VALF;
    }

    UGridCellIteratorArray_push(it, (UGridCellIterator) {
        .grid = grid,
        .cell = ugrid_cell_idx(grid, x, y),
        .i = 0,
    });

    while (x != end_x || y != end_y) {
        if (x == end_x) {
            y += step_y;
            t_max_y += t_delta_y;
        } else if (y == end_y) {
            x += step_x;
            t_max_x += t_delta_x;
        } else if (t_max_x + eps_t < t_max_y) {
            x += step_x;
            t_max_x += t_delta_x;
        } else if (t_max_y + eps_t < t_max_x) {
            y += step_y;
            t_max_y += t_delta_y;
        } else {
            // The segment crosses very close to a grid corner
            int next_x = x + step_x;
            int next_y = y + step_y;

            UGridCellIteratorArray_push(it, (UGridCellIterator) {
                .grid = grid,
                .cell = ugrid_cell_idx(grid, next_x, y),
                .i = 0,
            });

            UGridCellIteratorArray_push(it, (UGridCellIterator) {
                .grid = grid,
                .cell = ugrid_cell_idx(grid, x, next_y),
                .i = 0,
            });

            x = next_x;
            y = next_y;

            t_max_x += t_delta_x;
            t_max_y += t_delta_y;
        }

        UGridCellIteratorArray_push(it, (UGridCellIterator) {
            .grid = grid,
            .cell = ugrid_cell_idx(grid, x, y),
            .i = 0,
        });
    }
}

static float ugrid_clampf(float v, float lo, float hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

static float ugrid_point_segment_dist2(
    float px, float py,
    float ax, float ay,
    float bx, float by
) {
    float vx = bx - ax;
    float vy = by - ay;

    float wx = px - ax;
    float wy = py - ay;

    float len2 = vx * vx + vy * vy;

    if (len2 <= 0.0f) {
        float dx = px - ax;
        float dy = py - ay;
        return dx * dx + dy * dy;
    }

    float t = (wx * vx + wy * vy) / len2;
    t = ugrid_clampf(t, 0.0f, 1.0f);

    float cx = ax + t * vx;
    float cy = ay + t * vy;

    float dx = px - cx;
    float dy = py - cy;

    return dx * dx + dy * dy;
}

static int ugrid_segment_aabb_intersects(
    float x0, float y0,
    float x1, float y1,
    float min_x, float min_y,
    float max_x, float max_y
) {
    const float eps = 1e-8f;

    float t0 = 0.0f;
    float t1 = 1.0f;

    float dx = x1 - x0;
    float dy = y1 - y0;

    if (fabsf(dx) < eps) {
        if (x0 < min_x || x0 > max_x) {
            return 0;
        }
    } else {
        float inv_dx = 1.0f / dx;
        float tx0 = (min_x - x0) * inv_dx;
        float tx1 = (max_x - x0) * inv_dx;

        if (tx0 > tx1) {
            float tmp = tx0;
            tx0 = tx1;
            tx1 = tmp;
        }

        if (tx0 > t0) t0 = tx0;
        if (tx1 < t1) t1 = tx1;

        if (t0 > t1) {
            return 0;
        }
    }

    if (fabsf(dy) < eps) {
        if (y0 < min_y || y0 > max_y) {
            return 0;
        }
    } else {
        float inv_dy = 1.0f / dy;
        float ty0 = (min_y - y0) * inv_dy;
        float ty1 = (max_y - y0) * inv_dy;

        if (ty0 > ty1) {
            float tmp = ty0;
            ty0 = ty1;
            ty1 = tmp;
        }

        if (ty0 > t0) t0 = ty0;
        if (ty1 < t1) t1 = ty1;

        if (t0 > t1) {
            return 0;
        }
    }

    return 1;
}

static float ugrid_point_aabb_dist2(
    float px, float py,
    float min_x, float min_y,
    float max_x, float max_y
) {
    float cx = ugrid_clampf(px, min_x, max_x);
    float cy = ugrid_clampf(py, min_y, max_y);

    float dx = px - cx;
    float dy = py - cy;

    return dx * dx + dy * dy;
}

static float ugrid_segment_segment_dist2(
    float ax, float ay,
    float bx, float by,
    float cx, float cy,
    float dx, float dy
) {
    float d0 = ugrid_point_segment_dist2(ax, ay, cx, cy, dx, dy);
    float d1 = ugrid_point_segment_dist2(bx, by, cx, cy, dx, dy);
    float d2 = ugrid_point_segment_dist2(cx, cy, ax, ay, bx, by);
    float d3 = ugrid_point_segment_dist2(dx, dy, ax, ay, bx, by);

    float d = d0;
    if (d1 < d) d = d1;
    if (d2 < d) d = d2;
    if (d3 < d) d = d3;

    return d;
}

static float ugrid_segment_aabb_dist2(
    float x0, float y0,
    float x1, float y1,
    float min_x, float min_y,
    float max_x, float max_y
) {
    if (ugrid_segment_aabb_intersects(
            x0, y0,
            x1, y1,
            min_x, min_y,
            max_x, max_y
        )) {
        return 0.0f;
    }

    float d = ugrid_point_aabb_dist2(x0, y0, min_x, min_y, max_x, max_y);

    float d1 = ugrid_point_aabb_dist2(x1, y1, min_x, min_y, max_x, max_y);
    if (d1 < d) d = d1;

    // Bottom edge
    d1 = ugrid_segment_segment_dist2(
        x0, y0, x1, y1,
        min_x, min_y,
        max_x, min_y
    );
    if (d1 < d) d = d1;

    // Top edge
    d1 = ugrid_segment_segment_dist2(
        x0, y0, x1, y1,
        min_x, max_y,
        max_x, max_y
    );
    if (d1 < d) d = d1;

    // Left edge
    d1 = ugrid_segment_segment_dist2(
        x0, y0, x1, y1,
        min_x, min_y,
        min_x, max_y
    );
    if (d1 < d) d = d1;

    // Right edge
    d1 = ugrid_segment_segment_dist2(
        x0, y0, x1, y1,
        max_x, min_y,
        max_x, max_y
    );
    if (d1 < d) d = d1;

    return d;
}

void ugrid_segment_query(
    const UGrid* grid,
    Vec2 from,
    Vec2 to,
    float tolerance,
    UGridCellIteratorArray* it
) {
    const float eps = 1e-6f;

    if (tolerance <= 0.0f) {
        ugrid_segment_query_dda(grid, from, to, it);
        return;
    }

    float x0f = grid->inv_cell_size * from.x;
    float y0f = grid->inv_cell_size * from.y;
    float x1f = grid->inv_cell_size * to.x;
    float y1f = grid->inv_cell_size * to.y;

    // tolerance is in world units.
    // Convert to cell units.
    float r = tolerance * grid->inv_cell_size;
    float r2 = r * r;

    int min_x = (int)floorf(fminf(x0f, x1f) - r);
    int min_y = (int)floorf(fminf(y0f, y1f) - r);
    int max_x = (int)floorf(fmaxf(x0f, x1f) + r);
    int max_y = (int)floorf(fmaxf(y0f, y1f) + r);

    for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
            float cell_min_x = (float)x;
            float cell_min_y = (float)y;
            float cell_max_x = (float)x + 1.0f;
            float cell_max_y = (float)y + 1.0f;

            float d2 = ugrid_segment_aabb_dist2(
                x0f, y0f,
                x1f, y1f,
                cell_min_x, cell_min_y,
                cell_max_x, cell_max_y
            );

            if (d2 <= r2 + eps) {
                UGridCellIteratorArray_push(it, (UGridCellIterator) {
                    .grid = grid,
                    .cell = ugrid_cell_idx(grid, x, y),
                    .i = 0,
                });
            }
        }
    }
}

