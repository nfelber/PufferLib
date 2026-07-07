#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./create_qmshape_dataset.sh BOUNDARY_DIR OUTPUT_DIR [options]

Creates one .qmshape file per .json boundary by running:
  1. preprocessing/cf_util.py triangulate
  2. preprocessing/build/shape_preprocess --boundary

Options:
  --mesh-size VALUE      Pass a fixed target mesh size to cf_util.py.
  --obj-dir DIR         Store intermediate triangulated OBJ files in DIR.
  --build-dir DIR       CMake build dir for shape_preprocess. Default: preprocessing/build
  --force               Rebuild .qmshape files even if outputs already exist.
  --keep-objs           Keep intermediate OBJ files when --obj-dir is not provided.
  -h, --help            Show this help.

Example:
  ./create_qmshape_dataset.sh \
    resources/quad_meshing/boundaries \
    resources/quad_meshing/shapes
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if [ "$#" -lt 2 ]; then
    usage
    exit 1
fi

boundary_dir="$1"
output_dir="$2"
shift 2

mesh_size=""
obj_dir=""
build_dir="preprocessing/build"
force=0
keep_objs=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --mesh-size)
            if [ "$#" -lt 2 ]; then
                echo "Missing value after --mesh-size" >&2
                exit 1
            fi
            mesh_size="$2"
            shift 2
            ;;
        --obj-dir)
            if [ "$#" -lt 2 ]; then
                echo "Missing value after --obj-dir" >&2
                exit 1
            fi
            obj_dir="$2"
            shift 2
            ;;
        --build-dir)
            if [ "$#" -lt 2 ]; then
                echo "Missing value after --build-dir" >&2
                exit 1
            fi
            build_dir="$2"
            shift 2
            ;;
        --force)
            force=1
            shift
            ;;
        --keep-objs)
            keep_objs=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [ ! -d "$boundary_dir" ]; then
    echo "Boundary directory does not exist: $boundary_dir" >&2
    exit 1
fi

mkdir -p "$output_dir"

shape_preprocess="$build_dir/shape_preprocess"
if [ ! -x "$shape_preprocess" ]; then
    cmake -S preprocessing -B "$build_dir"
    cmake --build "$build_dir" --target shape_preprocess -j "$(nproc)"
fi

cleanup_obj_dir=0
if [ -z "$obj_dir" ]; then
    if [ "$keep_objs" -eq 1 ]; then
        obj_dir="$output_dir/objs"
    else
        obj_dir="$(mktemp -d)"
        cleanup_obj_dir=1
    fi
fi
mkdir -p "$obj_dir"

if [ "$cleanup_obj_dir" -eq 1 ]; then
    trap 'rm -rf "$obj_dir"' EXIT
fi

shopt -s nullglob
boundary_paths=("$boundary_dir"/*.json)
if [ "${#boundary_paths[@]}" -eq 0 ]; then
    echo "No .json boundary files found in: $boundary_dir" >&2
    exit 1
fi

created=0
skipped=0
failed=0

for boundary_path in "${boundary_paths[@]}"; do
    name="$(basename "$boundary_path" .json)"
    obj_path="$obj_dir/$name.obj"
    shape_path="$output_dir/$name.qmshape"

    if [ "$force" -eq 0 ] && [ -f "$shape_path" ]; then
        echo "Skipping existing: $shape_path"
        skipped=$((skipped + 1))
        continue
    fi

    echo "Processing: $boundary_path"

    tri_cmd=(python preprocessing/cf_util.py triangulate --boundary "$boundary_path" --obj "$obj_path")
    if [ -n "$mesh_size" ]; then
        tri_cmd+=(--mesh-size "$mesh_size")
    fi

    if ! "${tri_cmd[@]}"; then
        echo "Failed to triangulate: $boundary_path" >&2
        failed=$((failed + 1))
        continue
    fi

    if ! "$shape_preprocess" "$obj_path" "$shape_path" --boundary "$boundary_path"; then
        echo "Failed to create qmshape: $shape_path" >&2
        failed=$((failed + 1))
        continue
    fi

    created=$((created + 1))
done

echo "Done. created=$created skipped=$skipped failed=$failed"

if [ "$failed" -ne 0 ]; then
    exit 1
fi
