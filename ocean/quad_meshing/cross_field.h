#pragma once

#include "shape.h"
#include "ugrid.h"

typedef struct {
    Vec2 u;
    Vec2 v;
} CrossFieldQuery;

typedef struct {
    const QmShape* shape;
    UGrid grid;
    UGridCellIteratorArray grid_it;
    bool loaded;
} CrossField;

void cross_field_init(CrossField* field, unsigned int grid_res, float grid_cell_size, unsigned int grid_cell_cap) {
    field->shape = NULL;
    ugrid_init(&field->grid, grid_res, grid_cell_size, grid_cell_cap);
    UGridCellIteratorArray_init(&field->grid_it);
    field->loaded = false;
}

void cross_field_free(CrossField* field) {
    ugrid_free(&field->grid);
    UGridCellIteratorArray_free(&field->grid_it);
    field->shape = NULL;
    field->loaded = false;
}

static void cross_field_validate_face(const QmShapeCrossFieldFace* face) {
    QM_ASSERT(fabsf(triangle_area2(face->tri)) > 1e-12f);
    QM_ASSERT(isfinite(face->tri.a.x) && isfinite(face->tri.a.y));
    QM_ASSERT(isfinite(face->tri.b.x) && isfinite(face->tri.b.y));
    QM_ASSERT(isfinite(face->tri.c.x) && isfinite(face->tri.c.y));
    QM_ASSERT(isfinite(face->u.x) && isfinite(face->u.y));
    QM_ASSERT(isfinite(face->v.x) && isfinite(face->v.y));
    QM_ASSERT(fabsf(norm2(face->u) - 1.0f) < 1e-3f);
    QM_ASSERT(fabsf(norm2(face->v) - 1.0f) < 1e-3f);
}

static void cross_field_place_face(CrossField* field, int face_idx) {
    const QmShapeCrossFieldFace* face = &field->shape->cross_field_faces.data[face_idx];
    float min_x = fminf(face->tri.a.x, fminf(face->tri.b.x, face->tri.c.x));
    float max_x = fmaxf(face->tri.a.x, fmaxf(face->tri.b.x, face->tri.c.x));
    float min_y = fminf(face->tri.a.y, fminf(face->tri.b.y, face->tri.c.y));
    float max_y = fmaxf(face->tri.a.y, fmaxf(face->tri.b.y, face->tri.c.y));

    int cell_min_x = (int)floorf(field->grid.inv_cell_size * min_x);
    int cell_max_x = (int)floorf(field->grid.inv_cell_size * max_x);
    int cell_min_y = (int)floorf(field->grid.inv_cell_size * min_y);
    int cell_max_y = (int)floorf(field->grid.inv_cell_size * max_y);

    for (int y = cell_min_y; y <= cell_max_y; ++y) {
        for (int x = cell_min_x; x <= cell_max_x; ++x) {
            size_t cell = ugrid_cell_idx(&field->grid, x, y);
            ugrid_place(&field->grid, (int)cell, face_idx);
        }
    }
}

void cross_field_build(CrossField* field, const QmShape* shape) {
    QM_ASSERT(shape->cross_field_faces.size > 0);
    field->shape = shape;
    ugrid_reset(&field->grid);
    UGridCellIteratorArray_resize(&field->grid_it, 0);

    for (int i = 0; i < shape->cross_field_faces.size; ++i) {
        cross_field_validate_face(&shape->cross_field_faces.data[i]);
        cross_field_place_face(field, i);
    }

    field->loaded = true;
}

CrossFieldQuery cross_field_query(const CrossField* field, Vec2 p) {
    QM_ASSERT(field->loaded);
    QM_ASSERT(field->shape != NULL);

    UGridCellIterator it = ugrid_point_query(&field->grid, p);
    for (int face_idx; (face_idx = ugrid_cell_it_next(&it)) != -1;) {
        QM_ASSERT(face_idx >= 0 && face_idx < field->shape->cross_field_faces.size);
        const QmShapeCrossFieldFace* face = &field->shape->cross_field_faces.data[face_idx];
        if (point_in_triangle(p, face->tri, 1e-5f)) {
            return (CrossFieldQuery){face->u, face->v};
        }
    }

    fprintf(stderr, "cross_field_query miss at (%f, %f)\n", p.x, p.y);
    QM_ASSERT(false);
    return (CrossFieldQuery){{0.0f, 0.0f}, {0.0f, 0.0f}};
}

// [0, 1]
float cross_field_alignment(const CrossFieldQuery* field, Vec2 unit_dir) {
    const float du = dot2(unit_dir, field->u);
    const float dv = dot2(unit_dir, field->v);
    const float s = fmaxf(du * du, dv * dv); // [0.5, 1]
    return 2.0f * s - 1.0f;
}
