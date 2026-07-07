#pragma once

#include "helpers.h"
#include "memory.h"

#include <math.h>
#include <stdbool.h>

typedef struct {
    float x, y;
} Vec2;

DEFINE_VECTOR(Vec2, Vec2Array);

typedef struct {
    Vec2 a;
    Vec2 b;
    Vec2 c;
} Triangle2;

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

float triangle_area2(Triangle2 tri) {
    return cross2(sub2(tri.b, tri.a), sub2(tri.c, tri.a));
}

bool point_in_triangle(Vec2 p, Triangle2 tri, float eps) {
    Vec2 v0 = sub2(tri.b, tri.a);
    Vec2 v1 = sub2(tri.c, tri.a);
    Vec2 v2 = sub2(p, tri.a);
    float den = cross2(v0, v1);
    QM_ASSERT(fabsf(den) > 1e-12f);

    float w1 = cross2(v2, v1) / den;
    float w2 = cross2(v0, v2) / den;
    float w0 = 1.0f - w1 - w2;

    return w0 >= -eps && w1 >= -eps && w2 >= -eps;
}

float point_segment_dist_sq(Vec2 p, Vec2 a, Vec2 b);

bool point_in_polygon(Vec2 p, const Vec2* poly, int size, float eps) {
    bool inside = false;
    for (int i = 0, j = size - 1; i < size; j = i++) {
        Vec2 a = poly[i];
        Vec2 b = poly[j];

        if (point_segment_dist_sq(p, a, b) <= eps * eps) return true;

        bool crosses = (a.y > p.y) != (b.y > p.y);
        if (crosses) {
            float x = (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x;
            if (p.x < x + eps) inside = !inside;
        }
    }
    return inside;
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

float point_segment_dist_sq(Vec2 p, Vec2 a, Vec2 b) {
    const float eps = 1e-12f;

    Vec2 ab = sub2(b, a);
    float ab2 = dot2(ab, ab);

    // Degenerate segment: a == b
    if (ab2 < eps) {
        Vec2 d = sub2(p, a);
        return dot2(d, d);
    }

    float t = dot2(sub2(p, a), ab) / ab2;
    t = clampf(t, 0.0f, 1.0f);

    Vec2 closest = add2(a, scalmul2(ab, t));
    Vec2 d = sub2(p, closest);

    return dot2(d, d);
}

/** Strict segment-segment intersection test. */
static bool segments_intersect_strict(Vec2 p1, Vec2 p2, Vec2 q1, Vec2 q2) {
    const float eps = 1e-8;

    Vec2 r = sub2(p2, p1);
    Vec2 s = sub2(q2, q1);
    Vec2 qmp = sub2(q1, p1);

    float rxs = cross2(r, s);
    float qmpxr = cross2(qmp, r);

    // Parallel
    if (fabsf(rxs) < eps) {
        // Parallel but not colinear
        if (fabsf(qmpxr) >= eps) return false;

        // Colinear: check for overlap along p1 -> p2
        float rr = dot2(r, r);

        // Degenerate p segment: p1 == p2
        if (rr < eps) {
            // Treat as intersecting if p1 lies on q segment
            float ss = dot2(s, s);

            // Both are points
            if (ss < eps) {
                Vec2 d = sub2(q1, p1);
                return dot2(d, d) < eps;
            }

            float tq = dot2(sub2(p1, q1), s) / ss;
            return tq >= -eps && tq <= 1.0f + eps;
        }

        float t0 = dot2(sub2(q1, p1), r) / rr;
        float t1 = dot2(sub2(q2, p1), r) / rr;

        if (t0 > t1) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        return t0 <= 1.0f + eps && t1 >= -eps;
    }

    // Non-parallel: solve p1 + t*r = q1 + u*s
    float t = cross2(qmp, s) / rxs;
    float u = cross2(qmp, r) / rxs;

    return t >= -eps && t <= 1.0f + eps &&
           u >= -eps && u <= 1.0f + eps;
}

bool segments_intersect(Vec2 p1, Vec2 p2, Vec2 q1, Vec2 q2, float tol) {
    if (tol < 0.0f) tol = 0.0f;

    // First check true intersection.
    if (segments_intersect_strict(p1, p2, q1, q2)) {
        return true;
    }

    float tol2 = tol * tol;

    // If they do not intersect, the closest points involve at least one endpoint.
    return point_segment_dist_sq(p1, q1, q2) <= tol2 ||
           point_segment_dist_sq(p2, q1, q2) <= tol2 ||
           point_segment_dist_sq(q1, p1, p2) <= tol2 ||
           point_segment_dist_sq(q2, p1, p2) <= tol2;
}
