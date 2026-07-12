#!/bin/bash

set -euo pipefail

usage() {
    printf '%s\n' \
        "Usage: $0 MANIFEST [--max-concurrent N] [--image PATH] [--output-dir PATH] [--build]" \
        '' \
        'The mounted source must contain a backend built for the manifest environment.' \
        'Use --build to submit one backend build job before the experiment array.'
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

MANIFEST=$1
shift
MAX_CONCURRENT=1
IMAGE=${IMAGE:-$HOME/docker-images/pufferlib_quad_meshing_latest.sif}
BUILD_BACKEND=0
OUTPUT_DIR=${OUTPUT_DIR:-}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --max-concurrent) MAX_CONCURRENT=${2:?Missing value for --max-concurrent}; shift 2 ;;
        --image) IMAGE=${2:?Missing value for --image}; shift 2 ;;
        --output-dir) OUTPUT_DIR=${2:?Missing value for --output-dir}; shift 2 ;;
        --build) BUILD_BACKEND=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 1 ;;
    esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
MANIFEST=$(realpath "$MANIFEST")
IMAGE=$(realpath "$IMAGE")
if [ -z "$OUTPUT_DIR" ]; then
    MANIFEST_NAME=$(basename "$MANIFEST" .toml)
    OUTPUT_DIR="$SOURCE_DIR/cluster_runs/experiments/$MANIFEST_NAME"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

TASK_COUNT=$(python -m pufferlib.experiments count "$MANIFEST")
ENV_NAME=$(python -m pufferlib.experiments env "$MANIFEST")
if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Invalid --max-concurrent value: %s\n' "$MAX_CONCURRENT" >&2
    exit 1
fi
ARRAY_END=$((TASK_COUNT - 1))
DEPENDENCY=()

if [ "$BUILD_BACKEND" = 1 ]; then
    BUILD_JOB=$(sbatch --parsable \
        --export="ALL,IMAGE=$IMAGE,SOURCE_DIR=$SOURCE_DIR,OUTPUT_DIR=$OUTPUT_DIR,ENV_NAME=$ENV_NAME" \
        "$SCRIPT_DIR/slurm/dev_build_quad_meshing.run")
    DEPENDENCY=(--dependency="afterok:$BUILD_JOB")
    printf 'Submitted backend build job %s for %s\n' "$BUILD_JOB" "$ENV_NAME"
fi

ARRAY_JOB=$(sbatch --parsable \
    --array="0-$ARRAY_END%$MAX_CONCURRENT" \
    --output="$OUTPUT_DIR/slurm-%x-%A_%a.out" \
    "${DEPENDENCY[@]}" \
    "$SCRIPT_DIR/slurm/dev_experiments.run" \
    "$MANIFEST" "$SOURCE_DIR" "$OUTPUT_DIR" "$IMAGE")

printf 'Submitted experiment array %s: %s tasks, up to %s concurrent\n' \
    "$ARRAY_JOB" "$TASK_COUNT" "$MAX_CONCURRENT"
printf 'Environment: %s\nOutputs: %s\n' "$ENV_NAME" "$OUTPUT_DIR"
