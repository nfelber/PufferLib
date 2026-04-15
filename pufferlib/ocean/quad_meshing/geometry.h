#pragma once

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include "memory.h"

typedef struct {
    float x, y;
} Vec2;

static inline float clampf(float v, float min, float max) {
  const float t = v < min ? min : v;
  return t > max ? max : t;
}

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

typedef struct {
    Vec2 origin;
    Vec2 x;
    Vec2 y;
} Frame2D;

static inline Vec2 local_to_world(Frame2D f, Vec2 p) {
    return add2(
        f.origin,
        add2(
            scalmul2(f.x, p.x),
            scalmul2(f.y, p.y)
        )
    );
}

static inline Vec2 world_to_local(Frame2D f, Vec2 p) {
    Vec2 d = sub2(p, f.origin);

    return (Vec2){
        dot2(d, f.x),
        dot2(d, f.y)
    };
}

typedef struct {
    Vec2 a, b;
} Segment2D;

static inline float segment2D_sqrd_length(Segment2D seg) {
    Vec2 ab = sub2(seg.b, seg.a);
    return dot2(ab, ab);
}

static inline float segment2D_length(Segment2D seg) {
    return sqrtf(segment2D_sqrd_length(seg));
}

bool segment2D_intersect(Segment2D s1, Segment2D s2, Vec2 *intersection, float tol) {
    Vec2 p = s1.a;
    Vec2 r = sub2(s1.b, s1.a);

    Vec2 q = s2.a;
    Vec2 s = sub2(s2.b, s2.a);

    float rxs = cross2(r, s);
    float q_pxs = cross2(sub2(q, p), r);

    // Parallel
    if (fabsf(rxs) < 1e-8f) {
        // Collinear
        if (fabsf(q_pxs) < 1e-8f) {
            float t0 = dot2(sub2(q, p), r) / dot2(r, r);
            float t1 = t0 + dot2(s, r) / dot2(r, r);

            if ((t0 >= tol && t0 <= 1-tol) || (t1 >= tol && t1 <= 1-tol)) {
                if (intersection) *intersection = q; // arbitrary overlap point
                return true;
            }
        }
        return false;
    }

    float t = cross2(sub2(q, p), s) / rxs;
    float u = cross2(sub2(q, p), r) / rxs;

    if (t >= tol && t <= 1-tol && u >= tol && u <= 1-tol) {
        if (intersection) *intersection = add2(p, scalmul2(r, t));
        return true;
    }

    return false;
}

Vec2 closest_point_on_segment2D(Vec2 p, Segment2D s) {
    Vec2 ab = sub2(s.b, s.a);
    Vec2 ap = sub2(p, s.a);

    float ab_len2 = dot2(ab, ab);
    if (ab_len2 == 0.0) return s.a;

    float t = clampf(dot2(ap, ab) / ab_len2, 0.0, 1.0);
    return add2(s.a, scalmul2(ab, t));
}

float point_segment2D_sqrd_distance(Vec2 p, Segment2D s) {
    Vec2 ab = sub2(s.b, s.a);
    Vec2 ap = sub2(p, s.a);

    float ab_len2 = dot2(ab, ab);
    if (ab_len2 == 0.0) return dot2(ap, ap);

    float t = clampf(dot2(ap, ab) / ab_len2, 0.0, 1.0);
    Vec2 cp = sub2(ap, scalmul2(ab, t));
    return dot2(cp, cp);
}

float point_segment2D_distance(Vec2 p, Segment2D s) {
    float distance2 = point_segment2D_sqrd_distance(p, s);
    return sqrtf(distance2);
}

DEFINE_VECTOR(Vec2, Vec2Array)
DEFINE_VECTOR(size_t, SizeArray)

typedef struct {
    Vec2Array vertices;
    bool isCCW;
} Polygon2D;

static inline int polygon2D_neighbor_index(Polygon2D poly, int i, int offset) {
    const int n = poly.vertices.size;
    return ((i + offset) % n + n) % n;
}

Vec2 Polygon2D_neighbor(Polygon2D poly, int i, int offset) {
    return poly.vertices.data[polygon2D_neighbor_index(poly, i, offset)];
}

// Shoelace formula, positive if vertices are ccw and negative if cw
float polygon2D_signed_area(Polygon2D poly) {
    float A = 0.0;
    for (int i=poly.vertices.size-1, j=0; j<poly.vertices.size; i=j, ++j) {
        A += poly.vertices.data[i].x * poly.vertices.data[j].y -
             poly.vertices.data[j].x * poly.vertices.data[i].y;
    }
    A = 0.5 * A;
    return A;
}

float polygon2D_area(Polygon2D poly) {
    return fabs(polygon2D_signed_area(poly));
}

bool is_polygon_ccw(Polygon2D poly) {
    return polygon2D_signed_area(poly) >= 0;
}

float polygonInteriorAngle(Polygon2D poly, int i) {
    Vec2 p0 = Polygon2D_neighbor(poly, i, -1);
    Vec2 p1 = poly.vertices.data[i];
    Vec2 p2 = Polygon2D_neighbor(poly, i, 1);

    Vec2 v1 = sub2(p0, p1);
    Vec2 v2 = sub2(p2, p1);

    float len1 = norm2(v1);
    float len2 = norm2(v2);

    if (len1 == 0.0 || len2 == 0.0) return 0.0;

    Vec2 u1 = scalmul2(v1, 1.0 / len1);
    Vec2 u2 = scalmul2(v2, 1.0 / len2);

    float angle = acosf(clampf(dot2(u1, u2), -1.0, 1.0));

    float s = poly.isCCW ? 1.0 : -1.0;
    if (cross2(u1, u2) * s > 0.0) {
        angle = 2.0 * (float)M_PI - angle;
    }

    return angle;
}

float eval_polygon2D_sdf(Polygon2D poly, Vec2 p) {
    float d = dot2(
      sub2(p, poly.vertices.data[0]),
      sub2(p, poly.vertices.data[0])
    );
    float s = 1.0;
    
    for (int i=poly.vertices.size-1, j=0; j<poly.vertices.size; i=j, ++j) {
        Segment2D edge = {poly.vertices.data[i], poly.vertices.data[j]};

        // Distance to edge
        d = fmin(d, point_segment2D_sqrd_distance(p, edge));
        
        // Sign calculation (winding number)
        Vec2 ab = sub2(edge.b, edge.a);
        Vec2 ap = sub2(p, edge.a);
        bool c1 = p.y >= edge.a.y;
        bool c2 = p.y <  edge.b.y;
        bool c3 = ab.x*ap.y > ab.y*ap.x;
        if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) s*=-1.0;  
    }
    
    return s*sqrtf(d);
}

bool polygon2D_segment_intersect(Polygon2D poly, Segment2D seg, Vec2 *intersection, float tol) {
    for (int i=poly.vertices.size-1, j=0; j<poly.vertices.size; i=j, ++j) {
        Segment2D edge = {poly.vertices.data[i], poly.vertices.data[j]};

        if (segment2D_intersect(edge, seg, intersection, tol)) {
            return true;
        }
    }
    
    return false;
}

typedef struct {
    size_t v1, v2;
} Edge;

DEFINE_VECTOR(Edge, EdgeArray)

typedef struct {
    Vec2Array vertices;
    EdgeArray edges;
} Mesh2D;

void mesh2D_init(Mesh2D *m) {
    Vec2Array_init(&m->vertices);
    EdgeArray_init(&m->edges);
}

void mesh2D_free(Mesh2D *m) {
    Vec2Array_free(&m->vertices);
    EdgeArray_free(&m->edges);
}

size_t mesh2D_add_vertex(Mesh2D *m, Vec2 v) {
    Vec2Array_push(&m->vertices, v);
    return m->vertices.size - 1;
}

void mesh2D_add_edge(Mesh2D *m, size_t v1, size_t v2) {
    assert(v1 < m->vertices.size);
    assert(v2 < m->vertices.size);

    Edge e = { v1, v2 };
    EdgeArray_push(&m->edges, e);
}
