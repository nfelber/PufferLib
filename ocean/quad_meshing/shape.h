#pragma once

#include "geometry.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define QMSHAPE_VERSION 1u
#define QMSHAPE_SECTION_BOUNDARY_VERTICES 1u
#define QMSHAPE_SECTION_CROSS_FIELD_FACES 2u

typedef struct {
    Triangle2 tri;
    Vec2 u;
    Vec2 v;
} QmShapeCrossFieldFace;

DEFINE_VECTOR(QmShapeCrossFieldFace, QmShapeCrossFieldFaceArray)

typedef struct {
    Vec2Array boundary;
    QmShapeCrossFieldFaceArray cross_field_faces;
} QmShape;

typedef struct {
    uint32_t type;
    uint32_t count;
    uint32_t elem_size;
    uint32_t reserved;
} QmShapeSectionHeader;

static void qmshape_read_exact(FILE* f, void* dst, size_t size) {
    QM_ASSERT(fread(dst, 1, size, f) == size);
}

void qmshape_init(QmShape* shape) {
    Vec2Array_init(&shape->boundary);
    QmShapeCrossFieldFaceArray_init(&shape->cross_field_faces);
}

void qmshape_free(QmShape* shape) {
    Vec2Array_free(&shape->boundary);
    QmShapeCrossFieldFaceArray_free(&shape->cross_field_faces);
}

void qmshape_reset(QmShape* shape) {
    Vec2Array_resize(&shape->boundary, 0);
    QmShapeCrossFieldFaceArray_resize(&shape->cross_field_faces, 0);
}

void qmshape_load(QmShape* shape, const char* path) {
    FILE* f = fopen(path, "rb");
    QM_ASSERT(f != NULL);

    qmshape_reset(shape);

    char magic[8];
    qmshape_read_exact(f, magic, sizeof(magic));
    QM_ASSERT(memcmp(magic, "QMSHAPE\0", sizeof(magic)) == 0);

    uint32_t version = 0;
    uint32_t section_count = 0;
    qmshape_read_exact(f, &version, sizeof(version));
    qmshape_read_exact(f, &section_count, sizeof(section_count));
    QM_ASSERT(version == QMSHAPE_VERSION);

    bool saw_boundary = false;
    bool saw_cross_field = false;

    for (uint32_t i = 0; i < section_count; ++i) {
        QmShapeSectionHeader header;
        qmshape_read_exact(f, &header, sizeof(header));
        QM_ASSERT(header.reserved == 0);

        size_t bytes = (size_t)header.count * (size_t)header.elem_size;
        if (header.type == QMSHAPE_SECTION_BOUNDARY_VERTICES) {
            QM_ASSERT(header.elem_size == sizeof(Vec2));
            Vec2Array_resize(&shape->boundary, header.count);
            qmshape_read_exact(f, shape->boundary.data, bytes);
            QM_ASSERT(shape->boundary.size >= 3);
            saw_boundary = true;
        } else if (header.type == QMSHAPE_SECTION_CROSS_FIELD_FACES) {
            QM_ASSERT(header.elem_size == sizeof(QmShapeCrossFieldFace));
            QmShapeCrossFieldFaceArray_resize(&shape->cross_field_faces, header.count);
            qmshape_read_exact(f, shape->cross_field_faces.data, bytes);
            QM_ASSERT(shape->cross_field_faces.size > 0);
            saw_cross_field = true;
        } else {
            QM_ASSERT(fseek(f, (long)bytes, SEEK_CUR) == 0);
        }
    }

    fclose(f);
    QM_ASSERT(saw_boundary && saw_cross_field);
}
