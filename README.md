![figure](https://pufferai.github.io/source/resource/header.png)

[![PyPI version](https://badge.fury.io/py/pufferlib.svg)](https://badge.fury.io/py/pufferlib)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pufferlib)
![Github Actions](https://github.com/PufferAI/PufferLib/actions/workflows/install.yml/badge.svg)
[![](https://dcbadge.vercel.app/api/server/spT4huaGYV?style=plastic)](https://discord.gg/spT4huaGYV)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez5341)](https://twitter.com/jsuarez5341)

PufferLib is the reinforcement learning library I wish existed during my PhD. It started as a compatibility layer to make working with complex environments a breeze. Now, it's a high-performance toolkit for research and industry with optimized parallel simulation, environments that run and train at 1M+ steps/second, and tons of quality of life improvements for practitioners. All our tools are free and open source. We also offer priority service for companies, startups, and labs!

![Trailer](https://github.com/PufferAI/puffer.ai/blob/main/docs/assets/puffer_2.gif?raw=true)

All of our documentation is hosted at [puffer.ai](https://puffer.ai "PufferLib Documentation"). @jsuarez5341 on [Discord](https://discord.gg/puffer) for support -- post here before opening issues. We're always looking for new contributors, too!

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

To compile the environment C code in `./pufferlib/ocean/quad_meshing`, use:

```bash
ONLY_QUAD=1 python setup.py build_ext --inplace --force
```

### Train an agent

Start by editing the environment's config in `./config/ocean/quad_meshing.ini`
to your liking, then run `puffer train`:

```bash
# Optionally add --wandb to track the experiment
puffer train puffer_quad_meshing
```

### Evaluate the agent

When training is complete, you can evaluate the agent with `puffer eval`:

```bash
# You can replace latest with specific weights, e.g., baselines/single_vertex_1/model.pt
puffer eval puffer_quad_meshing --load-model-path latest --env.render-enabled True
```

To generate meshes and plots automatically for a set of models, you can add the
models and configs in `./baselines/`, edit the eval configuration in
`./eval_config.yaml` and use the evaluation script `./eval_quad_meshing.py`:

```bash
python eval_quad_meshing.py --config eval_config.yaml --model latest --output eval
```

The results will be found in the `eval/` folder.

> [!WARNING]
> If [Instant Meshes](https://github.com/wjakob/instant-meshes) is enabled, an
> Instant Meshes executable must exist at the path set for
> `instant_meshes.binary` in `./eval_config.yaml`.
