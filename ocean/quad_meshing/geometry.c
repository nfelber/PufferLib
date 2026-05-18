#include "geometry.h"

#include <math.h>

Vector2 v2_add(Vector2 a, Vector2 b) {
    return (Vector2){a.x + b.x, a.y + b.y};
}

Vector2 v2_sub(Vector2 a, Vector2 b) {
    return (Vector2){a.x - b.x, a.y - b.y};
}

float v2_dot(Vector2 a, Vector2 b) {
    return a.x * b.x + a.y * b.y;
}

float v2_cross(Vector2 a, Vector2 b) {
    return a.x * b.y - a.y * b.x;
}

float v2_len(Vector2 a) {
    return sqrtf(a.x * a.x + a.y * a.y);
}

float polygon_signed_area(const Vector2* poly, int count) {
    float area = 0.0f;
    for (int i = 0; i < count; i++) {
        Vector2 a = poly[i];
        Vector2 b = poly[(i + 1) % count];
        area += a.x * b.y - b.x * a.y;
    }
    return 0.5f * area;
}

float polygon_area(const Vector2* poly, int count) {
    return fabs(polygon_signed_area(poly, count));
}

bool polygon_is_ccw(const Vector2* poly, int count) {
    return polygon_signed_area(poly, count) > 0.0f;
}

bool segments_intersect(Vector2 p1, Vector2 p2, Vector2 q1, Vector2 q2) {
    Vector2 r = v2_sub(p2, p1);
    Vector2 s = v2_sub(q2, q1);
    float rxs = v2_cross(r, s);
    float q_p_r = v2_cross(v2_sub(q1, p1), r);
    if (fabsf(rxs) < 1e-8f && fabsf(q_p_r) < 1e-8f) {
        return 0;
    }
    if (fabsf(rxs) < 1e-8f && fabsf(q_p_r) >= 1e-8f) {
        return 0;
    }
    float t = v2_cross(v2_sub(q1, p1), s) / rxs;
    float u = v2_cross(v2_sub(q1, p1), r) / rxs;
    return (t > 0.0f && t < 1.0f && u > 0.0f && u < 1.0f);
}
