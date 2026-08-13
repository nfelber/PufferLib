![figure](https://pufferai.github.io/source/resource/header.png)

[![Discord](https://dcbadge.vercel.app/api/server/spT4huaGYV?style=plastic)](https://discord.gg/spT4huaGYV)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez)](https://twitter.com/jsuarez)

PufferLib is a fast and sane reinforcement learning library that can train tiny, super-human models in seconds. The included learning algorithm, hyperparameter tuning, and simulation methods are the product of our own research. All our tools are free and open source. Need a high performance environment for your application? We build them professionally and offer training + extended support. Contact jsuarez🐡puffer🐡ai.

All of our documentation is hosted at [puffer.ai](https://puffer.ai "PufferLib Documentation"). @jsuarez5341 on [Discord](https://discord.gg/puffer) for support. Post there before opening issues. We're always looking for new contributors!

## Star to puff up the project!

<a href="https://star-history.com/#pufferai/pufferlib&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
 </picture>
</a>

## The quad meshing environment

Here are some instructions to get started with the quad meshing environment. As
a prerequisite, after cloning this repository, you should start by installing
PufferLib as an editable python project. For example, with `uv`:

```bash
uv pip install -e . --no-build-isolation
```

### Compile the environment code

To compile the environment C code in `./ocean/quad_meshing` and CUDA training
backend, use:

```bash
bash build.sh quad_meshing --float
```

To play around with the environment manually, you can also build the standalone
executable with:

```bash
# Creates the ./quad_meshing executable
bash build.sh quad_meshing --fast
```

> [!Note]
> For the 3D environment (`./ocean/quad_meshing_3d`), simply replace
> `quad_meshing` with `quad_meshing_3d` in the above commands.

### Preprocess the dataset

The preprocessor depends on
[Directional](https://github.com/avaxman/Directional) and is therefore written
in C++ in `./preprocessing`. It can be built with CMake:

```bash
cmake -S preprocessing -B preprocessing/build -DCMAKE_BUILD_TYPE=Release
cmake --build preprocessing/build --target shape_preprocess
```

To automatically convert the whole train and test splits from JSON boundaries
to preprocessed `qmshape` files, use the `create_qmshape_dataset.sh` script:

```bash
bash create_qmshape_dataset.sh resources/quad_meshing/boundaries/train resources/quad_meshing/shapes/train
bash create_qmshape_dataset.sh resources/quad_meshing/boundaries/test resources/quad_meshing/shapes/test
```

### Train an agent

Start by editing the environment's config in `./config/quad_meshing.ini`
to your liking, then run `puffer train`:

```bash
# Optionally add --wandb to track the experiment
puffer train quad_meshing --slowly
```

> [!Note]
> Adding `--slowly` is required to use PyTorch models (`./pufferlib/models.py`)
> instead of models written directly in CUDA.

### Evaluate the agent

When training is complete, you can evaluate the agent with `puffer eval`:

```bash
# You can replace latest with specific weights
puffer eval quad_meshing --slowly --vec.total-agents 1 --load-model-path latest
```

You can also use the inspector tool to step through the environment manually
and observe the logits assigned by the model to each evaluated source / target
vertex:

```bash
python quad_meshing_inspector.py --total-agents 1 --checkpoint latest 
```

### Plot 2D meshes

The `eval_obj.py` script takes a JSON grid configuration enumerating a set of
OBJ meshes from different policies (or algorithms) to plot in a grid, and
produces a figure with the rendered meshes colored by quadrilateral quality:

```bash
python eval_obj.py --grid-config eval_grids/instant_mesh_eval_grid.json --output instant_mesh_eval.pdf --hide-singularities --cmap cividis --aggregate-output statistics.csv
```

> [!Note] The OBJ files must already have been created, for example using
> `puffer eval` or the experiments pipeline described below.

### Experiments pipeline

The experiments pipeline is designed to be run on a remote cluster using Slurm
jobs inside a Docker container (via Apptainer), and might lack the flexibility
to be used in a different setup in its current form. The jobs and relevant
documentation can be found in `cluster/`. The experiments themselves are
configured in `experiments/`.
