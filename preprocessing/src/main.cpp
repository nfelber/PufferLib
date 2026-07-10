#include <Eigen/Core>

#include <cnpy.h>

#include <directional/readOBJ.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/power_field.h>
#include <directional/power_to_raw.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kRoSyDegree = 4;

constexpr char kShapeMagic[8] = {'Q', 'M', 'S', 'H', 'A', 'P', 'E', '\0'};
constexpr std::uint32_t kShapeVersion = 1;
constexpr std::uint32_t kSectionBoundaryVertices = 1;
constexpr std::uint32_t kSectionCrossFieldFaces = 2;

constexpr char kSurfaceMagic[8] = {'Q', 'M', 'S', 'U', 'R', 'F', '3', 'D'};
constexpr std::uint32_t kSurfaceVersion = 1;
constexpr std::uint32_t kSurfSectionInfo = 1;
constexpr std::uint32_t kSurfSectionVertices = 2;
constexpr std::uint32_t kSurfSectionTriangles = 3;
constexpr std::uint32_t kSurfSectionVertexNormals = 4;
constexpr std::uint32_t kSurfSectionFaceNormals = 5;
constexpr std::uint32_t kSurfSectionTriangleNeighbors = 6;
constexpr std::uint32_t kSurfSectionFaceDirU = 7;
constexpr std::uint32_t kSurfSectionFaceDirV = 8;
constexpr std::uint32_t kSurfSectionSamples = 9;
constexpr std::uint32_t kSurfSectionFrontierEdges = 10;

struct Vec2f {
    float x;
    float y;
};

struct Vec3f {
    float x;
    float y;
    float z;
};

struct Tri3u32 {
    std::uint32_t a;
    std::uint32_t b;
    std::uint32_t c;
};

struct Tri3i32 {
    std::int32_t a;
    std::int32_t b;
    std::int32_t c;
};

struct SurfaceInfoRecord {
    float total_area;
    float sample_density;
    float sharp_dihedral_radians;
    std::uint32_t sample_count;
};

struct SurfaceSampleRecord {
    Vec3f p;
    Vec3f n;
    std::uint32_t tri;
    std::uint32_t reserved;
};

struct FrontierEdgeRecord {
    std::uint32_t a;
    std::uint32_t b;
};

struct CrossFieldFaceRecord {
    Vec2f a;
    Vec2f b;
    Vec2f c;
    Vec2f u;
    Vec2f v;
};

struct Normalize2D {
    double min_x;
    double min_y;
    double scale;
};

struct Normalize3D {
    Eigen::RowVector3d center;
    double scale;
};

struct SectionHeader {
    std::uint32_t type;
    std::uint32_t count;
    std::uint32_t elem_size;
    std::uint32_t reserved;
};

double sqr(double x) {
    return x * x;
}

double dist2(Vec2f a, Vec2f b) {
    return sqr(static_cast<double>(a.x) - static_cast<double>(b.x)) +
           sqr(static_cast<double>(a.y) - static_cast<double>(b.y));
}

double point_segment_dist2(Vec2f p, Vec2f a, Vec2f b) {
    const double ax = a.x;
    const double ay = a.y;
    const double bx = b.x;
    const double by = b.y;
    const double px = p.x;
    const double py = p.y;

    const double vx = bx - ax;
    const double vy = by - ay;
    const double wx = px - ax;
    const double wy = py - ay;
    const double len2 = vx * vx + vy * vy;

    if (len2 <= 1e-30) {
        return sqr(px - ax) + sqr(py - ay);
    }

    double t = (wx * vx + wy * vy) / len2;
    t = std::clamp(t, 0.0, 1.0);
    const double cx = ax + t * vx;
    const double cy = ay + t * vy;
    return sqr(px - cx) + sqr(py - cy);
}

template <typename T>
void write_pod(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error("Failed while writing binary shape cache.");
    }
}

template <typename T>
void write_array(std::ofstream& out, const std::vector<T>& values) {
    if (!values.empty()) {
        out.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T))
        );
        if (!out) {
            throw std::runtime_error("Failed while writing binary shape cache array.");
        }
    }
}

template <typename Scalar, typename Derived>
std::vector<Scalar> eigen_to_row_major_vector(const Eigen::MatrixBase<Derived>& M) {
    std::vector<Scalar> out(static_cast<size_t>(M.rows() * M.cols()));

    for (Eigen::Index r = 0; r < M.rows(); ++r) {
        for (Eigen::Index c = 0; c < M.cols(); ++c) {
            out[static_cast<size_t>(r * M.cols() + c)] =
                static_cast<Scalar>(M(r, c));
        }
    }

    return out;
}

template <typename Scalar>
void npz_save_matrix(
    const std::string& filename,
    const std::string& name,
    const std::vector<Scalar>& data,
    size_t rows,
    size_t cols,
    const std::string& mode
) {
    cnpy::npz_save(
        filename,
        name,
        data.data(),
        {rows, cols},
        mode
    );
}

template <typename Scalar>
void npz_save_vector(
    const std::string& filename,
    const std::string& name,
    const std::vector<Scalar>& data,
    const std::string& mode
) {
    cnpy::npz_save(
        filename,
        name,
        data.data(),
        {data.size()},
        mode
    );
}

Eigen::MatrixXi boundary_edges_matrix(const directional::TriMesh& mesh) {
    Eigen::MatrixXi E(mesh.boundEdges.size(), 2);

    for (int i = 0; i < mesh.boundEdges.size(); ++i) {
        const int e = mesh.boundEdges(i);
        E(i, 0) = mesh.EV(e, 0);
        E(i, 1) = mesh.EV(e, 1);
    }

    return E;
}

Eigen::VectorXi boundary_vertices_vector(const directional::TriMesh& mesh) {
    std::set<int> unique_vertices;

    for (int i = 0; i < mesh.boundEdges.size(); ++i) {
        const int e = mesh.boundEdges(i);
        unique_vertices.insert(mesh.EV(e, 0));
        unique_vertices.insert(mesh.EV(e, 1));
    }

    Eigen::VectorXi B(static_cast<int>(unique_vertices.size()));

    int k = 0;
    for (int v : unique_vertices) {
        B(k++) = v;
    }

    return B;
}

Eigen::MatrixXd boundary_vertex_positions(
    const directional::TriMesh& mesh,
    const Eigen::VectorXi& boundary_vertices
) {
    Eigen::MatrixXd P(boundary_vertices.size(), 3);

    for (int i = 0; i < boundary_vertices.size(); ++i) {
        P.row(i) = mesh.V.row(boundary_vertices(i));
    }

    return P;
}

std::vector<int> ordered_boundary_loop(const directional::TriMesh& mesh) {
    if (mesh.boundEdges.size() == 0) {
        return {};
    }

    std::unordered_map<int, std::vector<int>> adj;
    adj.reserve(static_cast<size_t>(mesh.boundEdges.size() * 2));

    for (int i = 0; i < mesh.boundEdges.size(); ++i) {
        const int e = mesh.boundEdges(i);
        const int a = mesh.EV(e, 0);
        const int b = mesh.EV(e, 1);
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    for (const auto& item : adj) {
        if (item.second.size() != 2) {
            throw std::runtime_error(
                "Expected a single manifold boundary loop for shape preprocessing."
            );
        }
    }

    const int start = mesh.EV(mesh.boundEdges(0), 0);
    std::vector<int> loop;
    loop.reserve(adj.size());

    int prev = -1;
    int curr = start;
    for (;;) {
        loop.push_back(curr);

        const std::vector<int>& nbrs = adj.at(curr);
        const int next = (nbrs[0] != prev) ? nbrs[0] : nbrs[1];
        prev = curr;
        curr = next;

        if (curr == start) {
            break;
        }

        if (loop.size() > adj.size()) {
            throw std::runtime_error("Boundary walk did not close.");
        }
    }

    if (loop.size() != adj.size()) {
        throw std::runtime_error(
            "Expected one boundary loop; found disconnected boundary components."
        );
    }

    return loop;
}

std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open boundary JSON: " + path);
    }

    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size <= 0) {
        throw std::runtime_error("Boundary JSON is empty: " + path);
    }
    in.seekg(0, std::ios::beg);

    std::string text(static_cast<size_t>(size), '\0');
    in.read(text.data(), size);
    if (!in) {
        throw std::runtime_error("Failed to read boundary JSON: " + path);
    }

    return text;
}

std::vector<Vec2f> load_boundary_json(const std::string& path) {
    const std::string json = read_text_file(path);
    const std::string key = "\"vertices\"";
    size_t pos = json.find(key);
    if (pos == std::string::npos) {
        throw std::runtime_error("Boundary JSON missing vertices key: " + path);
    }

    pos = json.find('[', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("Boundary JSON vertices value is not an array: " + path);
    }

    std::vector<Vec2f> out;
    int depth = 0;
    int value_count = 0;
    double current[2] = {0.0, 0.0};

    const char* start = json.c_str() + pos;
    for (const char* p = start; *p != '\0'; ++p) {
        if (*p == '[') {
            ++depth;
            continue;
        }
        if (*p == ']') {
            --depth;
            if (depth == 0) {
                break;
            }
            continue;
        }
        if (depth < 2) {
            continue;
        }
        if (!std::isdigit(static_cast<unsigned char>(*p)) &&
            *p != '-' && *p != '+' && *p != '.') {
            continue;
        }

        char* end = nullptr;
        const double value = std::strtod(p, &end);
        if (end != p) {
            current[value_count % 2] = value;
            ++value_count;
            p = end - 1;
            if (value_count % 2 == 0) {
                out.push_back(Vec2f{
                    static_cast<float>(current[0]),
                    static_cast<float>(current[1]),
                });
            }
        }
    }

    if (value_count % 2 != 0 || out.size() < 3) {
        throw std::runtime_error("Boundary JSON has invalid vertices: " + path);
    }

    return out;
}

double signed_area(const std::vector<Vec2f>& poly) {
    double area = 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        const Vec2f a = poly[i];
        const Vec2f b = poly[(i + 1) % poly.size()];
        area += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
    }
    return 0.5 * area;
}

Normalize2D make_normalize_2d(const Eigen::MatrixXd& V) {
    if (V.rows() == 0 || V.cols() < 2) {
        throw std::runtime_error("Expected non-empty 2D/3D vertex matrix.");
    }

    double min_x = V(0, 0);
    double max_x = V(0, 0);
    double min_y = V(0, 1);
    double max_y = V(0, 1);

    for (int i = 1; i < V.rows(); ++i) {
        min_x = std::min(min_x, V(i, 0));
        max_x = std::max(max_x, V(i, 0));
        min_y = std::min(min_y, V(i, 1));
        max_y = std::max(max_y, V(i, 1));
    }

    const double span = std::max(max_x - min_x, max_y - min_y);
    if (span <= 1e-14) {
        throw std::runtime_error("Cannot normalize degenerate shape bbox.");
    }

    return Normalize2D{min_x, min_y, 1.0 / span};
}

Normalize2D make_normalize_2d(const std::vector<Vec2f>& V) {
    if (V.empty()) {
        throw std::runtime_error("Expected non-empty boundary vertices.");
    }

    double min_x = V[0].x;
    double max_x = V[0].x;
    double min_y = V[0].y;
    double max_y = V[0].y;

    for (size_t i = 1; i < V.size(); ++i) {
        min_x = std::min(min_x, static_cast<double>(V[i].x));
        max_x = std::max(max_x, static_cast<double>(V[i].x));
        min_y = std::min(min_y, static_cast<double>(V[i].y));
        max_y = std::max(max_y, static_cast<double>(V[i].y));
    }

    const double span = std::max(max_x - min_x, max_y - min_y);
    if (span <= 1e-14) {
        throw std::runtime_error("Cannot normalize degenerate boundary bbox.");
    }

    return Normalize2D{min_x, min_y, 1.0 / span};
}

Vec2f normalize_point_2d(Vec2f p, const Normalize2D& norm) {
    return Vec2f{
        static_cast<float>((static_cast<double>(p.x) - norm.min_x) * norm.scale),
        static_cast<float>((static_cast<double>(p.y) - norm.min_y) * norm.scale),
    };
}

std::vector<Vec2f> normalize_points_2d(const std::vector<Vec2f>& points, const Normalize2D& norm) {
    std::vector<Vec2f> out;
    out.reserve(points.size());
    for (Vec2f p : points) {
        out.push_back(normalize_point_2d(p, norm));
    }
    return out;
}

Vec2f normalize_point_2d(const Eigen::RowVector3d& p, const Normalize2D& norm) {
    return Vec2f{
        static_cast<float>((p.x() - norm.min_x) * norm.scale),
        static_cast<float>((p.y() - norm.min_y) * norm.scale),
    };
}

std::vector<Vec2f> normalized_mesh_boundary(
    const directional::TriMesh& mesh,
    const std::vector<int>& boundary_loop,
    const Normalize2D& norm
) {
    std::vector<Vec2f> out;
    out.reserve(boundary_loop.size());
    for (int vidx : boundary_loop) {
        out.push_back(normalize_point_2d(mesh.V.row(vidx), norm));
    }
    return out;
}

Normalize3D make_normalize_3d(const Eigen::MatrixXd& V) {
    if (V.rows() == 0 || V.cols() < 3) {
        throw std::runtime_error("Expected non-empty 3D vertex matrix.");
    }

    Eigen::RowVector3d min_v = V.row(0);
    Eigen::RowVector3d max_v = V.row(0);
    for (int i = 1; i < V.rows(); ++i) {
        min_v = min_v.cwiseMin(V.row(i));
        max_v = max_v.cwiseMax(V.row(i));
    }

    const Eigen::RowVector3d size = max_v - min_v;
    const double span = std::max(size.x(), std::max(size.y(), size.z()));
    if (span <= 1e-14) {
        throw std::runtime_error("Cannot normalize degenerate 3D surface bbox.");
    }

    return Normalize3D{0.5 * (min_v + max_v), 1.0 / span};
}

Vec3f normalize_point_3d(const Eigen::RowVector3d& p, const Normalize3D& norm) {
    const Eigen::RowVector3d q = (p - norm.center) * norm.scale;
    return Vec3f{
        static_cast<float>(q.x()),
        static_cast<float>(q.y()),
        static_cast<float>(q.z()),
    };
}

Vec3f row_to_vec3f(const Eigen::RowVector3d& p) {
    return Vec3f{
        static_cast<float>(p.x()),
        static_cast<float>(p.y()),
        static_cast<float>(p.z()),
    };
}

Eigen::RowVector3d vec3_to_row(Vec3f v) {
    return Eigen::RowVector3d(v.x, v.y, v.z);
}

Vec3f normalize_vec3f(Vec3f v) {
    const double len = std::sqrt(
        static_cast<double>(v.x) * v.x +
        static_cast<double>(v.y) * v.y +
        static_cast<double>(v.z) * v.z
    );
    if (len <= 1e-20) return Vec3f{0.0f, 1.0f, 0.0f};
    return Vec3f{
        static_cast<float>(v.x / len),
        static_cast<float>(v.y / len),
        static_cast<float>(v.z / len),
    };
}

double triangle_area_3d(Vec3f a, Vec3f b, Vec3f c) {
    const Eigen::RowVector3d ab = vec3_to_row(b) - vec3_to_row(a);
    const Eigen::RowVector3d ac = vec3_to_row(c) - vec3_to_row(a);
    return 0.5 * ab.cross(ac).norm();
}

Vec3f triangle_normal_3d(Vec3f a, Vec3f b, Vec3f c) {
    const Eigen::RowVector3d ab = vec3_to_row(b) - vec3_to_row(a);
    const Eigen::RowVector3d ac = vec3_to_row(c) - vec3_to_row(a);
    const Eigen::RowVector3d n = ab.cross(ac);
    const double len = n.norm();
    if (len <= 1e-20) return Vec3f{0.0f, 1.0f, 0.0f};
    return row_to_vec3f(n / len);
}

Vec3f mix3(Vec3f a, Vec3f b, Vec3f c, double wa, double wb, double wc) {
    return Vec3f{
        static_cast<float>(wa * a.x + wb * b.x + wc * c.x),
        static_cast<float>(wa * a.y + wb * b.y + wc * c.y),
        static_cast<float>(wa * a.z + wb * b.z + wc * c.z),
    };
}

std::uint64_t edge_key(std::uint32_t a, std::uint32_t b) {
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
}

struct EdgeAdjacency {
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    std::int32_t f0 = -1;
    std::int32_t f1 = -1;
    std::int32_t e0 = -1;
    std::int32_t e1 = -1;
};

std::unordered_map<std::uint64_t, EdgeAdjacency> build_edge_adjacency(
    const std::vector<Tri3u32>& triangles
) {
    std::unordered_map<std::uint64_t, EdgeAdjacency> edges;
    edges.reserve(triangles.size() * 3);

    for (std::uint32_t f = 0; f < triangles.size(); ++f) {
        const std::uint32_t v[3] = {triangles[f].a, triangles[f].b, triangles[f].c};
        for (std::int32_t e = 0; e < 3; ++e) {
            std::uint32_t a = v[(e + 1) % 3];
            std::uint32_t b = v[(e + 2) % 3];
            const std::uint64_t key = edge_key(a, b);
            auto& adj = edges[key];
            adj.a = std::min(a, b);
            adj.b = std::max(a, b);
            if (adj.f0 < 0) {
                adj.f0 = static_cast<std::int32_t>(f);
                adj.e0 = e;
            } else if (adj.f1 < 0) {
                adj.f1 = static_cast<std::int32_t>(f);
                adj.e1 = e;
            }
        }
    }

    return edges;
}

std::vector<Tri3i32> build_surface_triangle_neighbors(
    const std::vector<Tri3u32>& triangles,
    const std::unordered_map<std::uint64_t, EdgeAdjacency>& edges
) {
    std::vector<Tri3i32> neighbors(triangles.size(), Tri3i32{-1, -1, -1});
    for (const auto& item : edges) {
        const EdgeAdjacency& adj = item.second;
        if (adj.f0 < 0 || adj.f1 < 0) continue;
        std::int32_t* n0 = &neighbors[static_cast<size_t>(adj.f0)].a;
        std::int32_t* n1 = &neighbors[static_cast<size_t>(adj.f1)].a;
        n0[adj.e0] = adj.f1;
        n1[adj.e1] = adj.f0;
    }
    return neighbors;
}

double min_dist2_to_polyline(Vec2f p, const std::vector<Vec2f>& poly) {
    if (poly.size() < 2) {
        throw std::runtime_error("Cannot validate against a degenerate polygon.");
    }

    double best = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < poly.size(); ++i) {
        const Vec2f a = poly[i];
        const Vec2f b = poly[(i + 1) % poly.size()];
        best = std::min(best, point_segment_dist2(p, a, b));
    }
    return best;
}

void validate_boundary_match(
    const std::vector<Vec2f>& json_boundary,
    const std::vector<Vec2f>& mesh_boundary
) {
    const double tol = 1e-4;
    const double tol2 = tol * tol;

    if (std::abs(signed_area(json_boundary)) <= 1e-12) {
        throw std::runtime_error("JSON boundary has near-zero area.");
    }

    if (std::abs(signed_area(mesh_boundary)) <= 1e-12) {
        throw std::runtime_error("OBJ boundary has near-zero area.");
    }

    for (Vec2f p : json_boundary) {
        if (min_dist2_to_polyline(p, mesh_boundary) > tol2) {
            throw std::runtime_error(
                "JSON boundary vertex does not lie on the OBJ boundary."
            );
        }
    }

    for (Vec2f p : mesh_boundary) {
        if (min_dist2_to_polyline(p, json_boundary) > tol2) {
            throw std::runtime_error(
                "OBJ boundary vertex does not lie on the JSON boundary."
            );
        }
    }
}

Vec2f normalize_direction_2d(const Eigen::RowVector3d& d) {
    const double x = d.x();
    const double y = d.y();
    const double len = std::sqrt(x * x + y * y);

    if (len <= 1e-14) {
        throw std::runtime_error("Cross-field direction has zero xy length.");
    }

    return Vec2f{
        static_cast<float>(x / len),
        static_cast<float>(y / len),
    };
}

/**
 * Build one hard boundary-alignment constraint per boundary-adjacent face.
 *
 * For every boundary edge, we project its tangent into the local basis of the
 * adjacent face, accumulate the 4th power, then take a representative root.
 *
 * This mirrors the Python logic:
 *
 *     z_boundary += (tx + i ty)^4
 *
 * but does it per face instead of per vertex.
 */
void build_boundary_face_constraints(
    const directional::TriMesh& mesh,
    Eigen::VectorXi& const_spaces,
    Eigen::MatrixXd& const_vectors_extrinsic,
    Eigen::VectorXd& align_weights
) {
    using Complex = std::complex<double>;

    std::unordered_map<int, Complex> accum;

    for (int i = 0; i < mesh.boundEdges.size(); ++i) {
        const int e = mesh.boundEdges(i);

        const int f0 = mesh.EF(e, 0);
        const int f1 = mesh.EF(e, 1);

        const int f = (f0 >= 0) ? f0 : f1;

        if (f < 0) {
            continue;
        }

        const int a = mesh.EV(e, 0);
        const int b = mesh.EV(e, 1);

        Eigen::RowVector3d edge = mesh.V.row(b) - mesh.V.row(a);
        const double len = edge.norm();

        if (len < 1e-14) {
            continue;
        }

        Eigen::RowVector3d t = edge / len;

        const double tx = t.dot(mesh.FBx.row(f));
        const double ty = t.dot(mesh.FBy.row(f));

        Complex u(tx, ty);

        const double unorm = std::abs(u);
        if (unorm < 1e-14) {
            continue;
        }

        u /= unorm;

        Complex z = std::pow(u, kRoSyDegree);
        accum[f] += z;
    }

    std::vector<int> faces;
    std::vector<Eigen::RowVector3d> vectors;

    faces.reserve(accum.size());
    vectors.reserve(accum.size());

    for (const auto& item : accum) {
        const int f = item.first;
        Complex z = item.second;

        if (std::abs(z) < 1e-14) {
            continue;
        }

        z /= std::abs(z);

        const double theta = std::arg(z) / static_cast<double>(kRoSyDegree);

        const double cx = std::cos(theta);
        const double cy = std::sin(theta);

        Eigen::RowVector3d v =
            cx * mesh.FBx.row(f) +
            cy * mesh.FBy.row(f);

        const double vnorm = v.norm();
        if (vnorm < 1e-14) {
            continue;
        }

        v /= vnorm;

        faces.push_back(f);
        vectors.push_back(v);
    }

    const int m = static_cast<int>(faces.size());

    const_spaces.resize(m);
    const_vectors_extrinsic.resize(m, 3);
    align_weights = Eigen::VectorXd::Constant(m, -1.0);

    for (int i = 0; i < m; ++i) {
        const_spaces(i) = faces[static_cast<size_t>(i)];
        const_vectors_extrinsic.row(i) = vectors[static_cast<size_t>(i)];
    }
}

int append_face_constraint_from_edge(
    const directional::TriMesh& mesh,
    int face,
    int edge,
    std::unordered_map<int, std::complex<double>>& accum
) {
    if (face < 0) return 0;

    const int a = mesh.EV(edge, 0);
    const int b = mesh.EV(edge, 1);
    Eigen::RowVector3d edge_vec = mesh.V.row(b) - mesh.V.row(a);
    const double len = edge_vec.norm();
    if (len < 1e-14) return 0;

    const Eigen::RowVector3d t = edge_vec / len;
    const double tx = t.dot(mesh.FBx.row(face));
    const double ty = t.dot(mesh.FBy.row(face));

    std::complex<double> u(tx, ty);
    const double unorm = std::abs(u);
    if (unorm < 1e-14) return 0;

    u /= unorm;
    accum[face] += std::pow(u, kRoSyDegree);
    return 1;
}

void finalize_face_constraints(
    const directional::TriMesh& mesh,
    const std::unordered_map<int, std::complex<double>>& accum,
    Eigen::VectorXi& const_spaces,
    Eigen::MatrixXd& const_vectors_extrinsic,
    Eigen::VectorXd& align_weights
) {
    std::vector<int> faces;
    std::vector<Eigen::RowVector3d> vectors;

    faces.reserve(accum.size());
    vectors.reserve(accum.size());

    for (const auto& item : accum) {
        const int f = item.first;
        std::complex<double> z = item.second;

        if (std::abs(z) < 1e-14) continue;
        z /= std::abs(z);

        const double theta = std::arg(z) / static_cast<double>(kRoSyDegree);
        const double cx = std::cos(theta);
        const double cy = std::sin(theta);

        Eigen::RowVector3d v = cx * mesh.FBx.row(f) + cy * mesh.FBy.row(f);
        const double vnorm = v.norm();
        if (vnorm < 1e-14) continue;

        faces.push_back(f);
        vectors.push_back(v / vnorm);
    }

    const int m = static_cast<int>(faces.size());
    const_spaces.resize(m);
    const_vectors_extrinsic.resize(m, 3);
    align_weights = Eigen::VectorXd::Constant(m, -1.0);

    for (int i = 0; i < m; ++i) {
        const_spaces(i) = faces[static_cast<size_t>(i)];
        const_vectors_extrinsic.row(i) = vectors[static_cast<size_t>(i)];
    }
}

void build_sharp_feature_face_constraints(
    const directional::TriMesh& mesh,
    double sharp_dihedral_radians,
    Eigen::VectorXi& const_spaces,
    Eigen::MatrixXd& const_vectors_extrinsic,
    Eigen::VectorXd& align_weights
) {
    std::unordered_map<int, std::complex<double>> accum;

    for (int e = 0; e < mesh.EV.rows(); ++e) {
        const int f0 = mesh.EF(e, 0);
        const int f1 = mesh.EF(e, 1);

        bool sharp = f0 < 0 || f1 < 0;
        if (f0 >= 0 && f1 >= 0) {
            const double dot = std::clamp(mesh.faceNormals.row(f0).dot(mesh.faceNormals.row(f1)), -1.0, 1.0);
            sharp = std::acos(dot) >= sharp_dihedral_radians;
        }

        if (!sharp) continue;
        append_face_constraint_from_edge(mesh, f0, e, accum);
        append_face_constraint_from_edge(mesh, f1, e, accum);
    }

    finalize_face_constraints(mesh, accum, const_spaces, const_vectors_extrinsic, align_weights);
}

void normalize_power_field(Eigen::MatrixXd& z) {
    if (z.cols() != 2) {
        throw std::runtime_error("Expected power field with shape [num_faces, 2].");
    }

    for (int i = 0; i < z.rows(); ++i) {
        const double x = z(i, 0);
        const double y = z(i, 1);
        const double n = std::sqrt(x * x + y * y);

        if (n < 1e-12) {
            z(i, 0) = 1.0;
            z(i, 1) = 0.0;
        } else {
            z(i, 0) /= n;
            z(i, 1) /= n;
        }
    }
}

void save_npz(
    const std::string& output_npz,
    const directional::TriMesh& mesh,
    const Eigen::MatrixXd& z_face_power,
    const Eigen::VectorXi& boundary_vertices,
    const Eigen::MatrixXi& boundary_edges,
    const Eigen::MatrixXd& dir_u,
    const Eigen::MatrixXd& dir_v
) {
    const Eigen::MatrixXd boundary_positions =
        boundary_vertex_positions(mesh, boundary_vertices);

    const Eigen::MatrixXd face_barycenters = mesh.barycenters;
    const Eigen::MatrixXd face_normals = mesh.faceNormals;

    const std::vector<float> V_data =
        eigen_to_row_major_vector<float>(mesh.V);

    const std::vector<std::int64_t> F_data =
        eigen_to_row_major_vector<std::int64_t>(mesh.F);

    const std::vector<float> z_data =
        eigen_to_row_major_vector<float>(z_face_power);

    const std::vector<std::int64_t> boundary_vertices_data =
        eigen_to_row_major_vector<std::int64_t>(boundary_vertices);

    const std::vector<std::int64_t> boundary_edges_data =
        eigen_to_row_major_vector<std::int64_t>(boundary_edges);

    const std::vector<float> boundary_positions_data =
        eigen_to_row_major_vector<float>(boundary_positions);

    const std::vector<float> barycenters_data =
        eigen_to_row_major_vector<float>(face_barycenters);

    const std::vector<float> normals_data =
        eigen_to_row_major_vector<float>(face_normals);

    const std::vector<std::int32_t> N_data = {kRoSyDegree};

    npz_save_matrix<float>(
        output_npz,
        "V",
        V_data,
        static_cast<size_t>(mesh.V.rows()),
        static_cast<size_t>(mesh.V.cols()),
        "w"
    );

    npz_save_matrix<std::int64_t>(
        output_npz,
        "F",
        F_data,
        static_cast<size_t>(mesh.F.rows()),
        static_cast<size_t>(mesh.F.cols()),
        "a"
    );

    npz_save_matrix<float>(
        output_npz,
        "z",
        z_data,
        static_cast<size_t>(z_face_power.rows()),
        static_cast<size_t>(z_face_power.cols()),
        "a"
    );

    npz_save_vector<std::int32_t>(
        output_npz,
        "N",
        N_data,
        "a"
    );

    npz_save_vector<std::int64_t>(
        output_npz,
        "boundary_vertices",
        boundary_vertices_data,
        "a"
    );

    npz_save_matrix<std::int64_t>(
        output_npz,
        "boundary_edges",
        boundary_edges_data,
        static_cast<size_t>(boundary_edges.rows()),
        static_cast<size_t>(boundary_edges.cols()),
        "a"
    );

    // Kept for compatibility with your old key name.
    // For OBJ input this is not the original polygon, but boundary vertex positions.
    npz_save_matrix<float>(
        output_npz,
        "boundary",
        boundary_positions_data,
        static_cast<size_t>(boundary_positions.rows()),
        static_cast<size_t>(boundary_positions.cols()),
        "a"
    );

    // Useful for rendering a face-based field.
    npz_save_matrix<float>(
        output_npz,
        "face_barycenters",
        barycenters_data,
        static_cast<size_t>(face_barycenters.rows()),
        static_cast<size_t>(face_barycenters.cols()),
        "a"
    );

    npz_save_matrix<float>(
        output_npz,
        "face_normals",
        normals_data,
        static_cast<size_t>(face_normals.rows()),
        static_cast<size_t>(face_normals.cols()),
        "a"
    );

    const std::vector<float> dir_u_data =
      eigen_to_row_major_vector<float>(dir_u);

    const std::vector<float> dir_v_data =
        eigen_to_row_major_vector<float>(dir_v);

    npz_save_matrix<float>(
        output_npz,
        "dir_u",
        dir_u_data,
        static_cast<size_t>(dir_u.rows()),
        static_cast<size_t>(dir_u.cols()),
        "a"
    );

    npz_save_matrix<float>(
        output_npz,
        "dir_v",
        dir_v_data,
        static_cast<size_t>(dir_v.rows()),
        static_cast<size_t>(dir_v.cols()),
        "a"
    );
}

void save_shape_cache(
    const std::string& output_path,
    const directional::TriMesh& mesh,
    const Eigen::MatrixXd& dir_u,
    const Eigen::MatrixXd& dir_v,
    const std::vector<Vec2f>* input_boundary
) {
    if (dir_u.rows() != mesh.F.rows() || dir_v.rows() != mesh.F.rows()) {
        throw std::runtime_error("Cross-field direction arrays must be face-sized.");
    }

    const std::vector<int> boundary_loop = ordered_boundary_loop(mesh);
    const Normalize2D norm = input_boundary != nullptr
        ? make_normalize_2d(*input_boundary)
        : make_normalize_2d(mesh.V);

    const std::vector<Vec2f> mesh_boundary = normalized_mesh_boundary(mesh, boundary_loop, norm);

    std::vector<Vec2f> boundary;
    if (input_boundary != nullptr) {
        boundary = normalize_points_2d(*input_boundary, norm);
        validate_boundary_match(boundary, mesh_boundary);
    } else {
        boundary = mesh_boundary;
    }

    std::vector<CrossFieldFaceRecord> faces;
    faces.reserve(static_cast<size_t>(mesh.F.rows()));
    for (int f = 0; f < mesh.F.rows(); ++f) {
        const int ia = mesh.F(f, 0);
        const int ib = mesh.F(f, 1);
        const int ic = mesh.F(f, 2);

        CrossFieldFaceRecord rec;
        rec.a = normalize_point_2d(mesh.V.row(ia), norm);
        rec.b = normalize_point_2d(mesh.V.row(ib), norm);
        rec.c = normalize_point_2d(mesh.V.row(ic), norm);
        rec.u = normalize_direction_2d(dir_u.row(f));
        rec.v = normalize_direction_2d(dir_v.row(f));
        faces.push_back(rec);
    }

    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open output shape cache: " + output_path);
    }

    out.write(kShapeMagic, sizeof(kShapeMagic));
    if (!out) {
        throw std::runtime_error("Failed to write shape cache magic.");
    }

    const std::uint32_t section_count = 2;
    write_pod(out, kShapeVersion);
    write_pod(out, section_count);

    const SectionHeader boundary_header{
        kSectionBoundaryVertices,
        static_cast<std::uint32_t>(boundary.size()),
        static_cast<std::uint32_t>(sizeof(Vec2f)),
        0,
    };
    write_pod(out, boundary_header);
    write_array(out, boundary);

    const SectionHeader faces_header{
        kSectionCrossFieldFaces,
        static_cast<std::uint32_t>(faces.size()),
        static_cast<std::uint32_t>(sizeof(CrossFieldFaceRecord)),
        0,
    };
    write_pod(out, faces_header);
    write_array(out, faces);
}

void save_surface_cache_3d(
    const std::string& output_path,
    const directional::TriMesh& mesh,
    const Eigen::MatrixXd& dir_u,
    const Eigen::MatrixXd& dir_v,
    double sample_density,
    double sharp_dihedral_radians,
    std::uint32_t sample_seed
) {
    if (dir_u.rows() != mesh.F.rows() || dir_v.rows() != mesh.F.rows()) {
        throw std::runtime_error("Cross-field direction arrays must be face-sized.");
    }
    if (sample_density < 0.0) {
        throw std::runtime_error("3D sample density must be non-negative.");
    }

    const Normalize3D norm = make_normalize_3d(mesh.V);

    std::vector<Vec3f> vertices;
    vertices.reserve(static_cast<size_t>(mesh.V.rows()));
    for (int i = 0; i < mesh.V.rows(); ++i) {
        vertices.push_back(normalize_point_3d(mesh.V.row(i), norm));
    }

    std::vector<Tri3u32> triangles;
    triangles.reserve(static_cast<size_t>(mesh.F.rows()));
    for (int f = 0; f < mesh.F.rows(); ++f) {
        triangles.push_back(Tri3u32{
            static_cast<std::uint32_t>(mesh.F(f, 0)),
            static_cast<std::uint32_t>(mesh.F(f, 1)),
            static_cast<std::uint32_t>(mesh.F(f, 2)),
        });
    }

    const auto edges = build_edge_adjacency(triangles);
    const std::vector<Tri3i32> triangle_neighbors = build_surface_triangle_neighbors(triangles, edges);

    std::vector<Vec3f> face_normals;
    std::vector<double> face_areas;
    std::vector<double> face_cdf;
    face_normals.reserve(triangles.size());
    face_areas.reserve(triangles.size());
    face_cdf.reserve(triangles.size());
    double total_area = 0.0;
    for (const Tri3u32& tri : triangles) {
        Vec3f a = vertices[tri.a];
        Vec3f b = vertices[tri.b];
        Vec3f c = vertices[tri.c];
        const double area = triangle_area_3d(a, b, c);
        total_area += area;
        face_areas.push_back(area);
        face_cdf.push_back(total_area);
        face_normals.push_back(triangle_normal_3d(a, b, c));
    }
    if (total_area <= 0.0) {
        throw std::runtime_error("3D surface has zero area after normalization.");
    }

    std::vector<Eigen::RowVector3d> vertex_normal_acc(vertices.size(), Eigen::RowVector3d::Zero());
    for (size_t f = 0; f < triangles.size(); ++f) {
        const Tri3u32& tri = triangles[f];
        const Eigen::RowVector3d n = vec3_to_row(face_normals[f]);
        const Eigen::RowVector3d weighted = n * face_areas[f];
        vertex_normal_acc[tri.a] += weighted;
        vertex_normal_acc[tri.b] += weighted;
        vertex_normal_acc[tri.c] += weighted;
    }

    std::vector<Vec3f> vertex_normals;
    vertex_normals.reserve(vertices.size());
    for (Eigen::RowVector3d n : vertex_normal_acc) {
        const double len = n.norm();
        if (len <= 1e-20) vertex_normals.push_back(Vec3f{0.0f, 1.0f, 0.0f});
        else vertex_normals.push_back(row_to_vec3f(n / len));
    }

    std::vector<Vec3f> face_dir_u;
    std::vector<Vec3f> face_dir_v;
    face_dir_u.reserve(triangles.size());
    face_dir_v.reserve(triangles.size());
    for (int f = 0; f < mesh.F.rows(); ++f) {
        face_dir_u.push_back(normalize_vec3f(row_to_vec3f(dir_u.row(f))));
        face_dir_v.push_back(normalize_vec3f(row_to_vec3f(dir_v.row(f))));
    }

    const std::uint32_t random_sample_count = static_cast<std::uint32_t>(std::ceil(total_area * sample_density));
    std::vector<SurfaceSampleRecord> samples;
    samples.reserve(random_sample_count);
    std::mt19937 rng(sample_seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (std::uint32_t i = 0; i < random_sample_count; ++i) {
        const double pick = unit(rng) * total_area;
        const size_t tri_idx = static_cast<size_t>(
            std::lower_bound(face_cdf.begin(), face_cdf.end(), pick) - face_cdf.begin()
        );
        const Tri3u32& tri = triangles[std::min(tri_idx, triangles.size() - 1)];

        double r1 = unit(rng);
        double r2 = unit(rng);
        if (r1 + r2 > 1.0) {
            r1 = 1.0 - r1;
            r2 = 1.0 - r2;
        }
        const double wa = 1.0 - r1 - r2;
        const double wb = r1;
        const double wc = r2;
        Vec3f p = mix3(vertices[tri.a], vertices[tri.b], vertices[tri.c], wa, wb, wc);
        Vec3f n = normalize_vec3f(mix3(vertex_normals[tri.a], vertex_normals[tri.b], vertex_normals[tri.c], wa, wb, wc));
        samples.push_back(SurfaceSampleRecord{
            p,
            n,
            static_cast<std::uint32_t>(std::min(tri_idx, triangles.size() - 1)),
            0,
        });
    }

    std::vector<FrontierEdgeRecord> frontier_edges;
    frontier_edges.reserve(edges.size());
    const std::uint32_t invalid_sample = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> frontier_vertex_samples(vertices.size(), invalid_sample);
    const auto sample_for_frontier_vertex = [&](std::uint32_t vertex, std::int32_t incident_tri) -> std::uint32_t {
        std::uint32_t& sample_id = frontier_vertex_samples[vertex];
        if (sample_id != invalid_sample) return sample_id;
        if (vertex >= vertices.size()) {
            throw std::runtime_error("Frontier vertex index is out of range.");
        }
        if (incident_tri < 0) {
            throw std::runtime_error("Sharp frontier vertex has no incident triangle.");
        }
        sample_id = static_cast<std::uint32_t>(samples.size());
        samples.push_back(SurfaceSampleRecord{
            vertices[vertex],
            vertex_normals[vertex],
            static_cast<std::uint32_t>(incident_tri),
            0,
        });
        return sample_id;
    };

    for (const auto& item : edges) {
        const EdgeAdjacency& adj = item.second;
        float angle = static_cast<float>(M_PI);
        bool sharp = adj.f1 < 0;
        if (adj.f0 >= 0 && adj.f1 >= 0) {
            const Eigen::RowVector3d n0 = vec3_to_row(face_normals[static_cast<size_t>(adj.f0)]);
            const Eigen::RowVector3d n1 = vec3_to_row(face_normals[static_cast<size_t>(adj.f1)]);
            const double dot = std::clamp(n0.dot(n1), -1.0, 1.0);
            angle = static_cast<float>(std::acos(dot));
            sharp = angle >= sharp_dihedral_radians;
        }
        if (sharp) {
            const std::int32_t incident_tri = adj.f0 >= 0 ? adj.f0 : adj.f1;
            frontier_edges.push_back(FrontierEdgeRecord{
                sample_for_frontier_vertex(adj.a, incident_tri),
                sample_for_frontier_vertex(adj.b, incident_tri),
            });
        }
    }

    const SurfaceInfoRecord info{
        static_cast<float>(total_area),
        static_cast<float>(sample_density),
        static_cast<float>(sharp_dihedral_radians),
        static_cast<std::uint32_t>(samples.size()),
    };

    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open output 3D surface cache: " + output_path);
    }

    out.write(kSurfaceMagic, sizeof(kSurfaceMagic));
    if (!out) throw std::runtime_error("Failed to write 3D surface cache magic.");

    const std::uint32_t section_count = 10;
    write_pod(out, kSurfaceVersion);
    write_pod(out, section_count);

    const auto write_section = [&out](std::uint32_t type, std::uint32_t count, std::uint32_t elem_size, const void* data) {
        const SectionHeader header{type, count, elem_size, 0};
        write_pod(out, header);
        const size_t bytes = static_cast<size_t>(count) * static_cast<size_t>(elem_size);
        if (bytes > 0) {
            out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
            if (!out) throw std::runtime_error("Failed while writing 3D surface cache section.");
        }
    };

    write_section(kSurfSectionInfo, 1, sizeof(SurfaceInfoRecord), &info);
    write_section(kSurfSectionVertices, static_cast<std::uint32_t>(vertices.size()), sizeof(Vec3f), vertices.data());
    write_section(kSurfSectionTriangles, static_cast<std::uint32_t>(triangles.size()), sizeof(Tri3u32), triangles.data());
    write_section(kSurfSectionVertexNormals, static_cast<std::uint32_t>(vertex_normals.size()), sizeof(Vec3f), vertex_normals.data());
    write_section(kSurfSectionFaceNormals, static_cast<std::uint32_t>(face_normals.size()), sizeof(Vec3f), face_normals.data());
    write_section(kSurfSectionTriangleNeighbors, static_cast<std::uint32_t>(triangle_neighbors.size()), sizeof(Tri3i32), triangle_neighbors.data());
    write_section(kSurfSectionFaceDirU, static_cast<std::uint32_t>(face_dir_u.size()), sizeof(Vec3f), face_dir_u.data());
    write_section(kSurfSectionFaceDirV, static_cast<std::uint32_t>(face_dir_v.size()), sizeof(Vec3f), face_dir_v.data());
    write_section(kSurfSectionSamples, static_cast<std::uint32_t>(samples.size()), sizeof(SurfaceSampleRecord), samples.data());
    write_section(kSurfSectionFrontierEdges, static_cast<std::uint32_t>(frontier_edges.size()), sizeof(FrontierEdgeRecord), frontier_edges.data());
}

void print_usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " input.obj output.qmshape [--boundary boundary.json] [--npz debug.npz] [--no-boundary-constraints]\n"
        << "  " << argv0 << " input.obj output.qmsurf [--3d] [--sample-density N] [--sharp-dihedral-deg D] [--sample-seed S] [--npz debug.npz]\n\n"
        << "Primary output:\n"
        << "  .qmshape          binary normalized boundary + face cross-field cache\n\n"
        << "  .qmsurf           binary normalized 3D surface + normals + cross-field + samples\n\n"
        << "Options:\n"
        << "  --boundary PATH   store this JSON boundary and use it for normalization; validates it against input.obj\n"
        << "  --npz PATH        additionally write a debug .npz cache\n\n"
        << "  --3d             write QMSURF3D cache instead of the 2D boundary cache\n"
        << "  --sample-density N       uniform candidate samples per normalized surface-area unit; default 1000\n"
        << "  --sharp-dihedral-deg D   starting frontier feature threshold; default 45\n"
        << "  --sample-seed S          deterministic candidate sampling seed; default 1\n\n"
        << "Output arrays:\n"
        << "  V                 float32 [num_vertices, 3]\n"
        << "  F                 int64   [num_faces, 3]\n"
        << "  z                 float32 [num_faces, 2]   real/imag of exp(4 i theta)\n"
        << "  N                 int32   [1]\n"
        << "  boundary_vertices int64   [num_boundary_vertices]\n"
        << "  boundary_edges    int64   [num_boundary_edges, 2]\n"
        << "  boundary          float32 [num_boundary_vertices, 3]\n"
        << "  face_barycenters  float32 [num_faces, 3]\n"
        << "  face_normals      float32 [num_faces, 3]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }

        const std::string input_obj = argv[1];
        const std::string output_shape = argv[2];

        bool use_boundary_constraints = true;
        bool output_3d_surface = false;
        std::string output_npz;
        std::string input_boundary_json;
        double sample_density = 1000.0;
        double sharp_dihedral_deg = 45.0;
        std::uint32_t sample_seed = 1;

        for (int i = 3; i < argc; ++i) {
            const std::string arg = argv[i];

            if (arg == "--no-boundary-constraints") {
                use_boundary_constraints = false;
            } else if (arg == "--3d") {
                output_3d_surface = true;
            } else if (arg == "--npz") {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value after --npz\n";
                    print_usage(argv[0]);
                    return EXIT_FAILURE;
                }
                output_npz = argv[++i];
            } else if (arg == "--boundary") {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value after --boundary\n";
                    print_usage(argv[0]);
                    return EXIT_FAILURE;
                }
                input_boundary_json = argv[++i];
            } else if (arg == "--sample-density") {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value after --sample-density\n";
                    print_usage(argv[0]);
                    return EXIT_FAILURE;
                }
                sample_density = std::stod(argv[++i]);
            } else if (arg == "--sharp-dihedral-deg") {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value after --sharp-dihedral-deg\n";
                    print_usage(argv[0]);
                    return EXIT_FAILURE;
                }
                sharp_dihedral_deg = std::stod(argv[++i]);
            } else if (arg == "--sample-seed") {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value after --sample-seed\n";
                    print_usage(argv[0]);
                    return EXIT_FAILURE;
                }
                sample_seed = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            } else {
                std::cerr << "Unknown argument: " << arg << "\n";
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        }

        directional::TriMesh mesh;

        if (!directional::readOBJ(input_obj, mesh)) {
            throw std::runtime_error("Failed to read OBJ: " + input_obj);
        }

        if (mesh.F.cols() != 3) {
            throw std::runtime_error("Expected a triangular OBJ mesh.");
        }

        if (mesh.F.rows() == 0 || mesh.V.rows() == 0) {
            throw std::runtime_error("Mesh is empty.");
        }

        std::cout << "Loaded mesh\n";
        std::cout << "  vertices: " << mesh.V.rows() << "\n";
        std::cout << "  faces:    " << mesh.F.rows() << "\n";
        std::cout << "  boundary edges: " << mesh.boundEdges.size() << "\n";

        std::vector<Vec2f> input_boundary;
        const std::vector<Vec2f>* input_boundary_ptr = nullptr;
        if (!input_boundary_json.empty()) {
            input_boundary = load_boundary_json(input_boundary_json);
            input_boundary_ptr = &input_boundary;
            std::cout << "Loaded boundary JSON\n";
            std::cout << "  vertices: " << input_boundary.size() << "\n";
        }

        directional::PCFaceTangentBundle ftb;
        ftb.init(mesh);

        Eigen::VectorXi const_spaces;
        Eigen::MatrixXd const_vectors;
        Eigen::VectorXd align_weights;

        if (output_3d_surface) {
            const double sharp_dihedral_radians = sharp_dihedral_deg * M_PI / 180.0;
            build_sharp_feature_face_constraints(
                mesh,
                sharp_dihedral_radians,
                const_spaces,
                const_vectors,
                align_weights
            );

            std::cout << "Sharp feature constraints: " << const_spaces.size()
                      << " faces at threshold " << sharp_dihedral_deg << " deg\n";
            if (const_spaces.size() == 0) {
                std::cout
                    << "No sharp feature constraints found. Computing unconstrained field; "
                    << "global orientation is arbitrary.\n";
            }
        } else if (use_boundary_constraints && mesh.boundEdges.size() > 0) {
            build_boundary_face_constraints(
                mesh,
                const_spaces,
                const_vectors,
                align_weights
            );

            std::cout << "Boundary constraints: " << const_spaces.size() << "\n";
        } else {
            const_spaces.resize(0);
            const_vectors.resize(0, 3);
            align_weights.resize(0);

            if (mesh.boundEdges.size() == 0) {
                std::cout
                    << "Mesh has no boundary. Computing unconstrained field; "
                    << "global orientation is arbitrary.\n";
            } else {
                std::cout
                    << "Boundary constraints disabled. Computing unconstrained field; "
                    << "global orientation is arbitrary.\n";
            }
        }

        directional::CartesianField field;

        const bool normalize_field = true;

        directional::power_field(
            ftb,
            const_spaces,
            const_vectors,
            align_weights,
            kRoSyDegree,
            field,
            normalize_field
        );

        if (field.intField.cols() != 2) {
            throw std::runtime_error(
                "Directional did not return a POWER_FIELD with two intrinsic columns."
            );
        }

        Eigen::MatrixXd z_face_power = field.intField;
        normalize_power_field(z_face_power);

        directional::CartesianField rawField;

        directional::power_to_raw(
            field,
            kRoSyDegree,
            rawField,
            true
        );

        Eigen::MatrixXd dir_u(mesh.F.rows(), 3);
        Eigen::MatrixXd dir_v(mesh.F.rows(), 3);

        for (int f = 0; f < mesh.F.rows(); ++f) {
            // rawField.extField is [num_faces, 3 * N].
            // For N = 4, columns are:
            //   u0.xyz, u1.xyz, u2.xyz, u3.xyz
            dir_u.row(f) = rawField.extField.block(f, 0, 1, 3);
            dir_v.row(f) = rawField.extField.block(f, 3, 1, 3);
        }

        const Eigen::VectorXi boundary_vertices = boundary_vertices_vector(mesh);
        const Eigen::MatrixXi boundary_edges = boundary_edges_matrix(mesh);

        if (output_3d_surface) {
            if (input_boundary_ptr != nullptr) {
                throw std::runtime_error("--boundary is only supported for 2D .qmshape output.");
            }
            const double sharp_dihedral_radians = sharp_dihedral_deg * M_PI / 180.0;
            save_surface_cache_3d(
                output_shape,
                mesh,
                dir_u,
                dir_v,
                sample_density,
                sharp_dihedral_radians,
                sample_seed
            );
        } else {
            save_shape_cache(
                output_shape,
                mesh,
                dir_u,
                dir_v,
                input_boundary_ptr
            );
        }

        if (!output_npz.empty()) {
            save_npz(
                output_npz,
                mesh,
                z_face_power,
                boundary_vertices,
                boundary_edges,
                dir_u,
                dir_v
            );
            std::cout << "Saved debug npz: " << output_npz << "\n";
        }

        std::cout << "Saved " << (output_3d_surface ? "3D surface" : "shape") << " cache: " << output_shape << "\n";
        std::cout << "faces: " << z_face_power.rows() << "\n";

        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
