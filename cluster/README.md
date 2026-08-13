# PufferLib Quad Meshing on Apptainer/Singularity

This setup builds a Docker image with the current source code and compiled
`quad_meshing` CUDA backend baked in. On the cluster, Apptainer converts that
Docker image to a `.sif` file and Slurm runs experiments from it.

The image is built for NVIDIA V100 GPUs with `NVCC_ARCH=sm_70` and uses
PyTorch CUDA 12.1 wheels, which are compatible with the cluster's 535 driver
(`nvidia-smi` reports CUDA 12.2).

## 1. Build and Push the Docker Image

Run this from the repository root on a machine with Docker:

```bash
docker build -f cluster/Dockerfile -t <dockerhub-user>/pufferlib-quad-meshing:latest .
docker push <dockerhub-user>/pufferlib-quad-meshing:latest
```

Use a versioned tag for real experiments, for example:

```bash
docker build -f cluster/Dockerfile -t <dockerhub-user>/pufferlib-quad-meshing:$(git rev-parse --short HEAD) .
docker push <dockerhub-user>/pufferlib-quad-meshing:$(git rev-parse --short HEAD)
```

## 2. Pull the Image on the Cluster

On the cluster login node:

```bash
mkdir -p ~/myimages
cd ~/myimages
apptainer pull pufferlib_quad_meshing_latest.sif docker://docker.io/<dockerhub-user>/pufferlib-quad-meshing:latest
```

For a versioned image:

```bash
apptainer pull pufferlib_quad_meshing_<tag>.sif docker://docker.io/<dockerhub-user>/pufferlib-quad-meshing:<tag>
```

## 3. Smoke Test

Submit a short GPU job before starting a long run:

```bash
IMAGE=$HOME/myimages/pufferlib_quad_meshing_latest.sif sbatch cluster/slurm/smoke_test_quad_meshing.run
```

Check the Slurm output file. It should print PyTorch/CUDA information,
`compiled_env quad_meshing`, and complete a tiny training run.

If Triton/CUDA JIT compilation fails, run the diagnostic job:

```bash
IMAGE=$HOME/myimages/pufferlib_quad_meshing_cu121.sif sbatch cluster/slurm/diagnose_cuda_jit.run
```

This prints the effective compiler, CUDA library paths, `libcuda` visibility,
manual `gcc -lcuda` link tests, and a minimal Triton kernel compile.

## 4. Train One Agent

```bash
IMAGE=$HOME/myimages/pufferlib_quad_meshing_latest.sif \
WANDB_GROUP=quad_meshing_baseline \
sbatch cluster/slurm/train_quad_meshing.run
```

Extra PufferLib arguments can be appended through `PUFFER_ARGS`:

```bash
IMAGE=$HOME/myimages/pufferlib_quad_meshing_latest.sif \
WANDB_GROUP=quad_meshing_dolphin \
PUFFER_ARGS="--train.total-timesteps 10000000 --tag v100" \
sbatch cluster/slurm/train_quad_meshing.run
```

Outputs default to `./cluster_runs` relative to the directory where you submit
the job. Override with `OUTPUT_DIR=/path/to/runs` if needed.

## 5. Run a Sweep

The sweep code runs multiple independent trials across the GPUs visible on one
node. `--sweep.gpus N` should match the number of GPUs requested from Slurm.

Edit `cluster/slurm/sweep_quad_meshing.run` and change:

```bash
#SBATCH --gres=gpu:1
```

to the number of GPUs you want, for example:

```bash
#SBATCH --gres=gpu:4
```

Then submit with the same number in `GPUS`:

```bash
IMAGE=$HOME/myimages/pufferlib_quad_meshing_latest.sif \
GPUS=4 \
WANDB_GROUP=quad_meshing_sweep_001 \
sbatch cluster/slurm/sweep_quad_meshing.run
```

With the current defaults, `train.gpus = 1`, so `GPUS=4` means four sweep trials
run concurrently on GPU ids `0`, `1`, `2`, and `3`. This is a single-node setup,
not a multi-node sweep coordinator.

## 6. W&B Login

If W&B is not already configured on the cluster, run once in an interactive
container shell:

```bash
srun --pty --qos normal --partition gpu --gres=gpu:1 apptainer shell --nv $HOME/myimages/pufferlib_quad_meshing_latest.sif
wandb login
exit
```

Alternatively, submit jobs with `WANDB_API_KEY` exported in your environment if
that is allowed by your cluster policy.

## 7. Interactive Debug Shell

```bash
srun --pty --qos normal --partition gpu --gres=gpu:1 \
  apptainer shell --nv $HOME/myimages/pufferlib_quad_meshing_latest.sif
```

Inside the container:

```bash
cd /workspace/PufferLib
puffer train quad_meshing --slowly --train.total-timesteps 16384
```

## 8. Optional Source Bind Mount for Debugging

The default image bakes in the source code and compiled backend. That is best
for reproducible experiments.

For fast iteration, keep using the same `.sif` and mount a cluster checkout over
the baked source. The scripts below assume you submit from the checkout root, or
set `SOURCE_DIR=/path/to/PufferLib`.

Build only the mounted checkout's quad backend:

```bash
IMAGE=$HOME/myimages/pufferlib_quad_meshing_cu121.sif \
sbatch cluster/slurm/dev_build_quad_meshing.run
```

Run training from the mounted checkout:

```bash
IMAGE=$HOME/myimages/pufferlib_quad_meshing_cu121.sif \
WANDB_GROUP=quad_meshing_dev \
sbatch cluster/slurm/dev_train_quad_meshing.run
```

Rebuild the backend at the start of the training job:

```bash
IMAGE=$HOME/myimages/pufferlib_quad_meshing_cu121.sif \
BUILD_BACKEND=1 \
WANDB_GROUP=quad_meshing_dev \
sbatch cluster/slurm/dev_train_quad_meshing.run
```

Run a dev sweep. Request the same GPU count from Slurm and PufferLib:

```bash
IMAGE=$HOME/myimages/pufferlib_quad_meshing_cu121.sif \
GPUS=4 \
WANDB_GROUP=quad_meshing_dev_sweep \
sbatch --gres=gpu:4 cluster/slurm/dev_sweep_quad_meshing.run
```

For a coordinated Protein sweep across multiple two-GPU nodes, use the
distributed sweep worker. Put `SWEEP_DIR` on a shared filesystem such as
`/scratch` for active runs or `/work` for longer retention. Do not use
`/tmp/${SLURM_JOB_ID}` because it is local to one node.

```bash
IMAGE=$HOME/myimages/pufferlib_quad_meshing_cu121.sif \
SOURCE_DIR=$PWD \
SWEEP_DIR=/scratch/$USER/puffer_sweeps/quad_holiday_001 \
WANDB_GROUP=quad_holiday_001 \
PUFFER_ARGS="--dist-sweep-max-worker-trials 4" \
sbatch --array=0-31%8 cluster/slurm/dev_dist_sweep_quad_meshing.run
```

Each array task requests one GPU and repeatedly reserves trials from the shared
`state.json` under `SWEEP_DIR`. Slurm can place two one-GPU workers on each
two-GPU node. `%8` caps concurrent workers; start with a modest value so Protein
gets observations before too many future trials are reserved. Increase it once
the sweep is running smoothly.

Useful files in `SWEEP_DIR`:

```text
state.json      # shared sweep state: running/completed/failed trials
checkpoints/    # checkpoints from completed non-sweep uploads are suppressed
logs/           # per-run JSON logs
wandb/          # W&B local files
cache/triton/   # Triton JIT cache
```

If a worker dies while a trial is running, the trial remains in `running` until
`--dist-sweep-stale-seconds` expires. The default is 48 hours. Set a larger value
if your jobs can legitimately run longer.

Equivalent raw Apptainer command:

```bash
apptainer exec --nv \
  --bind /path/to/PufferLib:/workspace/PufferLib \
  $HOME/myimages/pufferlib_quad_meshing_latest.sif \
  bash -lc "cd /workspace/PufferLib && NVCC_ARCH=sm_70 bash build.sh quad_meshing --float && puffer train quad_meshing --slowly"
```

This is convenient for iteration, but less reproducible. The mounted checkout
must have a freshly built `pufferlib/_C*.so` matching the container environment.
For final experiments, rebuild and pull a baked image tagged with the git commit.

## Notes

- Always use `apptainer exec --nv` for GPU jobs.
- `NVCC_ARCH=sm_70` targets V100 GPUs.
- Checkpoints, JSON logs, W&B files, and caches are written under `OUTPUT_DIR`.
- The Slurm scripts use `--gres=gpu:N`, matching the cluster examples.
- The scripts pin `CC`, `CXX`, and `CUDAHOSTCXX` inside the container so host
  Spack compiler paths do not leak into Triton or CUDA JIT compilation.
- The scripts set `LIBRARY_PATH=/usr/local/cuda/lib64/stubs` inside the
  container so Triton can link JIT helper modules against `-lcuda`; runtime CUDA
  calls still use the host driver libraries exposed by `apptainer --nv`.
- The scripts set `CPATH=/usr/lib/gcc/x86_64-linux-gnu/11/include` to work
  around clusters where Apptainer/GCC does not find GCC's internal headers such
  as `stddef.h` during Triton JIT compilation.

## Explicit Experiment Arrays

Use an experiment manifest when you want a fixed set of configurations rather
than an automatic sweep. The precedence for every run is:

```text
config/default.ini
config/<environment>.ini
manifest [defaults]
manifest [[experiments]] overrides
selected seed
```

Thus `config/default.ini` continues to provide every setting not replaced by
the selected environment config. See `experiments/quad_meshing.toml` and
`experiments/quad_meshing_3d.toml` for complete examples.

Submit all experiment and seed combinations from the cluster with:

```bash
cluster/submit_dev_experiments.sh experiments/quad_meshing.toml \
  --image "$HOME/myimages/pufferlib_quad_meshing_latest.sif" \
  --max-concurrent 4 \
  --build
```

`--build` submits one build job for the manifest environment and makes the
array depend on it. Omit it when the mounted checkout already contains the
correct float backend. Array tasks never build, so tasks from one array can run
concurrently without racing on `pufferlib/_C*.so`.

The submission helper validates the TOML on the host and writes a JSON runtime
manifest into the output directory. Consequently, the container does not need
Python 3.11 or the optional `tomli` package.

The default output directory is
`cluster_runs/experiments/<manifest-name>`. Override it with `--output-dir`.
Each array task requests one GPU, and `--max-concurrent` controls the Slurm
array concurrency cap.

> [!Warning]
> Do not overlap 2D and 3D arrays that mount the same checkout. Both
> environment builds write the same `pufferlib/_C*.so`; build and finish one
> environment's batch before building the other.

Each W&B run uses the manifest's project and group, with names such as
`deeper-painn-seed-2`. The experiment name is also recorded as W&B `job_type`
and as `config.experiment`, while the seed is recorded as
`config.experiment_seed`. This makes repeated seeds easy to group and compare
without creating a W&B sweep.

## Evaluate Explicit Experiments

Evaluate every experiment and seed in a manifest against every shape directly
inside a test dataset folder:

```bash
python -m pufferlib.evaluate_experiments \
  experiments/quad_meshing.toml \
  resources/quad_meshing/shapes/test \
  --download \
  --remote-root 'username@cluster:~/PufferLib/cluster_runs/experiments/quad_meshing'
```

`--download` first uses `rsync` over SSH to fetch the completed run logs. It
maps each manifest `(experiment, seed)` pair to its run ID, downloads that run's
checkpoint directory, and stores the highest numeric-step checkpoint as:

```text
eval/<wandb-group>/models/<experiment>/<seed>/model.pt
```

Omit `--download` to reuse models already present in that layout. Generated
meshes are written to:

```text
eval/<wandb-group>/meshes/<experiment>/<seed>/<shape>.obj
```

Use `--output-dir PATH` to replace the top-level `eval` directory. The command
requires the float CUDA backend for the manifest environment and evaluates all
test shapes as one vectorized environment batch without opening a render
window. Each environment receives one shape and a distinct staging output, and
the evaluator preserves its first terminal mesh export.

By default, the whole dataset is evaluated concurrently for each model. Limit
GPU memory and environment concurrency with, for example:

```bash
python -m pufferlib.evaluate_experiments MANIFEST DATASET --batch-size 16
```

The relevant backend must be rebuilt after changing the indexed shape/export
binding implementation.

The downloader only considers valid completed JSON logs with matching
environment, W&B group, experiment name, and seed. If a task was retried, it
prefers the candidate with the highest final agent step and reports discarded
run IDs. All expected models must be available before evaluation starts.

## Plot W&B Training Curves

Download full metric histories from W&B and generate seed-aggregated training
curves directly from an experiment manifest:

```bash
python -m pufferlib.plot_experiments \
  experiments/quad_meshing.toml \
  --entity YOUR_WANDB_ENTITY \
  --metric env/score
```

The command writes an individual figure for every experiment and one figure
containing every experiment:

```text
eval/<wandb-group>/training_curves/<metric>/
  baseline.pdf
  baseline.png
  other_experiment.pdf
  other_experiment.png
  all_experiments.pdf
  all_experiments.png
  baseline.csv
  other_experiment.csv
```

Use `--experiment NAME` one or more times to limit which individual figures are
written. The `all_experiments` figure always contains every experiment in the
manifest. Use `--format pdf png svg` to select output formats and `--ylabel` to
override the automatically generated metric label.
