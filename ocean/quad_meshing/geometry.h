#pragma once

#include "helpers.h"
#include "memory.h"

#include <math.h>
#include <stdbool.h>

typedef struct {
    float x, y;
} Vec2;

DEFINE_VECTOR(Vec2, Vec2Array);

static inline Vec2 add2(Vec2 a, Vec2 b) { 
    return (Vec2){a.x + b.x, a.y + b.y}; 
}

static inline Vec2 sub2(Vec2 a, Vec2 b) { 
    return (Vec2){a.x - b.x, a.y - b.y}; 
}

static inline Vec2 scalmul2(Vec2 a, float b) { 
    return (Vec2){a.x * b, a.y * b}; 
}

static inline float dot2(Vec2 a, Vec2 b) { 
    return a.x * b.x + a.y * b.y; 
}

static inline float cross2(Vec2 a, Vec2 b) {
    return a.x * b.y - a.y * b.x;
}

static inline float sqrd_norm2(Vec2 a) { 
    return dot2(a, a);
}

static inline float norm2(Vec2 a) { 
    return sqrtf(dot2(a, a)); 
}

static inline Vec2 safe_normalize(Vec2 v) {
    float n = norm2(v);
    if (n < 1e-8f) return (Vec2){0, 0};
    return scalmul2(v, 1.0f / n);
}

Vec2 slerp2(Vec2 a, Vec2 b, float t, bool ccw) {
    float na = norm2(a);
    float nb = norm2(b);

    if (na < 1e-8f || nb < 1e-8f) return (Vec2){0, 0};

    a = scalmul2(a, 1.0f / na);
    b = scalmul2(b, 1.0f / nb);

    float dot = clampf(dot2(a, b), -1.0f, 1.0f);
    float cross = cross2(a, b);

    float angle = atan2f(cross, dot);

    // Enforce rotation direction
    if (ccw && angle < 0.0f) {
        angle += 2.0f * M_PI;
    } else if (!ccw && angle > 0.0f) {
        angle -= 2.0f * M_PI;
    }

    float theta = angle * t;

    float c = cosf(theta);
    float s = sinf(theta);

    return (Vec2){
        a.x * c - a.y * s,
        a.x * s + a.y * c
    };
}

/** Signed polygon area (shoelace). */
float polygon_signed_area(const Vec2* poly, int size) {
    float area = 0.0f;
    for (int i = 0; i < size; i++) {
        Vec2 a = poly[i];
        Vec2 b = poly[(i + 1) % size];
        area += a.x * b.y - b.x * a.y;
    }
    return 0.5f * area;
}

/** Polygon area. */
float polygon_area(const Vec2* poly, int size) {;
    return fabs(polygon_signed_area(poly, size));
}

/** Returns 1 if polygon is CCW. */
bool polygon_is_ccw(const Vec2* poly, int size) {
    return polygon_signed_area(poly, size) > 0.0f;
}

float polygonInteriorAngle(const Vec2* poly, int size, int i, bool isCCW) {
    Vec2 p0 = poly[((i - 1) % size + size) % size];
    Vec2 p1 = poly[i];
    Vec2 p2 = poly[(i + 1) % size];

    Vec2 v1 = sub2(p0, p1);
    Vec2 v2 = sub2(p2, p1);

    float len1 = norm2(v1);
    float len2 = norm2(v2);

    if (len1 == 0.0 || len2 == 0.0) return 0.0;

    Vec2 u1 = scalmul2(v1, 1.0 / len1);
    Vec2 u2 = scalmul2(v2, 1.0 / len2);

    float angle = acosf(clampf(dot2(u1, u2), -1.0, 1.0));

    float s = isCCW ? 1.0 : -1.0;
    if (cross2(u1, u2) * s > 0.0) {
        angle = 2.0 * (float)M_PI - angle;
    }

    return angle;
}

/** Strict segment-segment intersection test. */
bool segments_intersect(Vec2 p1, Vec2 p2, Vec2 q1, Vec2 q2) {
    Vec2 r = sub2(p2, p1);
    Vec2 s = sub2(q2, q1);
    float rxs = cross2(r, s);
    float q_p_r = cross2(sub2(q1, p1), r);
    if (fabsf(rxs) < 1e-8f && fabsf(q_p_r) < 1e-8f) {
        return 0;
    }
    if (fabsf(rxs) < 1e-8f && fabsf(q_p_r) >= 1e-8f) {
        return 0;
    }
    float t = cross2(sub2(q1, p1), s) / rxs;
    float u = cross2(sub2(q1, p1), r) / rxs;
    return (t >= 0.0f && t <= 1.0f && u >= 0.0f && u <= 1.0f);
}
