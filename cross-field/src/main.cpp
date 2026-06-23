#include <Eigen/Core>

#include <cnpy.h>

#include <directional/readOBJ.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/power_field.h>
#include <directional/power_to_raw.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kRoSyDegree = 4;

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

void print_usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " input.obj output.npz [--no-boundary-constraints]\n\n"
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
        const std::string output_npz = argv[2];

        bool use_boundary_constraints = true;

        for (int i = 3; i < argc; ++i) {
            const std::string arg = argv[i];

            if (arg == "--no-boundary-constraints") {
                use_boundary_constraints = false;
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

        directional::PCFaceTangentBundle ftb;
        ftb.init(mesh);

        Eigen::VectorXi const_spaces;
        Eigen::MatrixXd const_vectors;
        Eigen::VectorXd align_weights;

        if (use_boundary_constraints && mesh.boundEdges.size() > 0) {
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

        save_npz(
            output_npz,
            mesh,
            z_face_power,
            boundary_vertices,
            boundary_edges,
            dir_u,
            dir_v
        );

        std::cout << "Saved cache: " << output_npz << "\n";
        std::cout << "z shape: [" << z_face_power.rows() << ", 2]\n";

        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
