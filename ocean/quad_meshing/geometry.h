#pragma once

#include "raylib.h"

/** Returns a + b. */
Vector2 v2_add(Vector2 a, Vector2 b);

/** Returns a - b. */
Vector2 v2_sub(Vector2 a, Vector2 b);

/** Dot product of a and b. */
float v2_dot(Vector2 a, Vector2 b);

/** 2D cross product (scalar). */
float v2_cross(Vector2 a, Vector2 b);

/** Vector length. */
float v2_len(Vector2 a);

/** Signed polygon area (shoelace). */
float polygon_signed_area(const Vector2* poly, int count);

/** Polygon area. */
float polygon_area(const Vector2* poly, int count);

/** Returns 1 if polygon is CCW. */
bool polygon_is_ccw(const Vector2* poly, int count);

/** Strict segment-segment intersection test. */
bool segments_intersect(Vector2 p1, Vector2 p2, Vector2 q1, Vector2 q2);
