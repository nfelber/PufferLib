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
