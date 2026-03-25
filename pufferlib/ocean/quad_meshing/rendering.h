#pragma once

#include "geometry.h"
#include "raylib.h"

typedef struct {
    Vec2 min;
    Vec2 max;
    float scale;
    float offsetX;
    float offsetY;
    int screenW;
    int screenH;
} RenderContext;

static inline Vector2 world_to_screen(Vec2 p, const RenderContext* ctx) {
    return (Vector2){
        ctx->offsetX + (p.x - ctx->min.x) * ctx->scale,
        ctx->offsetY + (p.y - ctx->min.y) * ctx->scale
    };
}

static inline Vec2 screen_to_world(int x, int y, const RenderContext* ctx) {
    return (Vec2){
        ctx->min.x + (x - ctx->offsetX) / ctx->scale,
        ctx->min.y + (y - ctx->offsetY) / ctx->scale
    };
}

RenderContext compute_render_context(Vec2 min, Vec2 max) {
    RenderContext ctx;

    ctx.min = min;
    ctx.max = max;

    float width = ctx.max.x - ctx.min.x;
    float height = ctx.max.y - ctx.min.y;

    if (width < 1e-6f) width = 1.0f;
    if (height < 1e-6f) height = 1.0f;

    ctx.screenW = GetScreenWidth();
    ctx.screenH = GetScreenHeight();

    ctx.scale = 0.6f * fminf(ctx.screenW / width, ctx.screenH / height);

    ctx.offsetX = (ctx.screenW - width * ctx.scale) * 0.5f;
    ctx.offsetY = (ctx.screenH - height * ctx.scale) * 0.5f;

    return ctx;
}

void draw_mesh(const Mesh2D* mesh, const RenderContext* ctx) {
    // Edges
    for (size_t i = 0; i < mesh->edges.size; i++) {
        Edge e = mesh->edges.data[i];

        Vec2 v1 = mesh->vertices.data[e.v1];
        Vec2 v2 = mesh->vertices.data[e.v2];

        Vector2 p1 = world_to_screen(v1, ctx);
        Vector2 p2 = world_to_screen(v2, ctx);

        DrawLineEx(p1, p2, 2.0f, GRAY);
    }
}

void draw_boundary(const Polygon2D* poly, const RenderContext* ctx) {
    // Edges
    for (int i = 0; i < poly->vertices.size; i++) {
        Vec2 v1 = poly->vertices.data[i];
        Vec2 v2 = Polygon2D_neighbor(*poly, i, 1);

        Vector2 p1 = world_to_screen(v1, ctx);
        Vector2 p2 = world_to_screen(v2, ctx);

        DrawLineEx(p1, p2, 3.0f, BLUE);
    }

    // Vertices
    for (int i = 0; i < poly->vertices.size; i++) {
        Vector2 p = world_to_screen(poly->vertices.data[i], ctx);
        DrawCircleV(p, 5.0f, BLUE);
    }
}

static inline float clamp01(float x) {
    return x < 0 ? 0 : (x > 1 ? 1 : x);
}

static inline float smoothstep(float a, float b, float x) {
    float t = clamp01((x - a) / (b - a));
    return t * t * (3.0f - 2.0f * t);
}

static inline float mixf(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

static inline Vector3 mix3(Vector3 a, Vector3 b, float t) {
    return (Vector3){
        mixf(a.x, b.x, t),
        mixf(a.y, b.y, t),
        mixf(a.z, b.z, t)
    };
}

void draw_sdf(const Polygon2D* poly, const RenderContext* ctx) {
    const int step = 4;

    for (int y = 0; y < ctx->screenH; y += step) {
        for (int x = 0; x < ctx->screenW; x += step) {

            Vec2 p = screen_to_world(x, y, ctx);
            float d = eval_polygon2D_sdf(*poly, p);

            float s = (d > 0.0f) ? 1.0f : -1.0f;

            Vector3 col = {
                1.0f - s * 0.1f,
                1.0f - s * 0.4f,
                1.0f - s * 0.7f
            };

            float absd = fabsf(d);

            float k1 = 0.8f - expf(-8.0f * absd);
            col.x *= k1; col.y *= k1; col.z *= k1;

            float k2 = 0.8f + 0.2f * cosf(120.0f * d);
            col.x *= k2; col.y *= k2; col.z *= k2;

            float edge = 1.0f - smoothstep(0.0f, 0.01f, absd);
            col = mix3(col, (Vector3){1,1,1}, edge);

            Color c = {
                (unsigned char)(255 * clamp01(col.x)),
                (unsigned char)(255 * clamp01(col.y)),
                (unsigned char)(255 * clamp01(col.z)),
                255
            };

            DrawRectangle(x, y, step, step, c);
        }
    }
}

