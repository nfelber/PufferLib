import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pufferlib.experiments import load_manifest


def select_runs(runs, manifest):
    expected = {
        (experiment['name'], seed)
        for experiment in manifest['experiments']
        for seed in manifest['batch']['seeds']
    }
    names = {
        f'{experiment}-seed-{seed}': (experiment, seed)
        for experiment, seed in expected
    }
    candidates = {key: [] for key in expected}
    for run in runs:
        config = run.config
        key = (config.get('experiment'), config.get('experiment_seed'))
        if key not in expected:
            # api.runs() returns lightweight lazy objects whose config can be
            # empty until loaded. Explicit experiment runs have stable names.
            key = names.get(run.name)
        if key in expected and run.state != 'running':
            candidates[key].append(run)

    selected = {}
    for key, matches in candidates.items():
        if not matches:
            continue
        matches.sort(key=lambda run: (
            run.state == 'finished', str(run.created_at or ''), run.id
        ))
        selected[key] = matches[-1]
        if len(matches) > 1:
            discarded = ', '.join(run.id for run in matches[:-1])
            print(f'Using W&B run {matches[-1].id} for {key[0]} seed {key[1]}; '
                  f'discarded older runs: {discarded}')
        if selected[key].state != 'finished':
            print(f'WARNING: Using {selected[key].state} W&B run '
                  f'{selected[key].id} for {key[0]} seed {key[1]} because no '
                  'finished retry exists. The aggregate will stop at the '
                  'shared history range.')
    return selected


def hydrate_run(run):
    if not run.config or not isinstance(run.config.get('train'), dict):
        run.load(force=True)
    return run


def _metric_slug(metric):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', metric).strip('_')


def load_history(run, metric, cache_dir, refresh=False,
        full_history=False, history_samples=10_000):
    cache_dir.mkdir(parents=True, exist_ok=True)
    mode = 'full' if full_history else f'sampled-{history_samples}'
    path = cache_dir / f'{run.id}-{_metric_slug(metric)}-{mode}.json'
    if path.is_file() and not refresh:
        with path.open() as f:
            return json.load(f)

    history = []
    if full_history:
        rows = run.scan_history(
            keys=['agent_steps', metric], page_size=10_000,
        )
    else:
        rows = run.history(
            samples=history_samples, keys=['agent_steps', metric], pandas=False,
        )
    for row in rows:
        step = row.get('agent_steps')
        value = row.get(metric)
        if (isinstance(step, (int, float)) and isinstance(value, (int, float))
                and math.isfinite(step) and math.isfinite(value)):
            history.append([float(step), float(value)])
    if not history:
        raise RuntimeError(f'W&B run {run.id} has no finite values for {metric}')
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(history, f)
    temporary.replace(path)
    return history


def prepare_history(history, total_timesteps):
    # Keep the latest value when W&B contains duplicate logs at one step.
    values = {
        float(step): float(value)
        for step, value in history
        if step <= total_timesteps and math.isfinite(step) and math.isfinite(value)
    }
    if len(values) < 2:
        raise ValueError('Need at least two metric values within total_timesteps')
    steps = np.array(sorted(values), dtype=np.float64)
    metric = np.array([values[step] for step in steps], dtype=np.float64)
    return steps, metric


def aggregate_histories(histories, points=400):
    if not histories:
        raise ValueError('No histories to aggregate')
    start = max(steps[0] for steps, _ in histories)
    stop = min(steps[-1] for steps, _ in histories)
    if stop <= start:
        raise ValueError('Seed histories do not overlap in agent_steps')
    grid = np.linspace(start, stop, points)
    aligned = np.stack([
        np.interp(grid, steps, values) for steps, values in histories
    ])
    return {
        'steps': grid,
        'mean': aligned.mean(axis=0),
        'std': aligned.std(axis=0, ddof=1) if aligned.shape[0] > 1 else np.zeros_like(grid),
        'num_seeds': aligned.shape[0],
    }


def _display_name(name):
    return name.replace('_', ' ').replace('-', ' ').title()


def _metric_label(metric):
    labels = {
        'env/score': 'Score',
        'env/episode_return': 'Episode return',
        'env/episode_length': 'Episode length',
        'env/num_quads': 'Number of quadrilaterals',
        'loss/policy_loss': 'Policy loss',
        'loss/value_loss': 'Value loss',
        'loss/entropy': 'Policy entropy',
    }
    return labels.get(metric, metric.replace('/', ' ').replace('_', ' ').title())


def _configure_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIX Two Text', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'savefig.dpi': 300,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    return plt


def _style_axis(axis, ylabel):
    axis.set_xlabel('Environment steps (millions)')
    axis.set_ylabel(ylabel)
    axis.grid(axis='y', color='#d9d9d9', linewidth=0.6, alpha=0.7)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.tick_params(direction='out', length=3, width=0.7)


def _draw_curve(axis, aggregate, label, color):
    steps = aggregate['steps'] / 1_000_000
    lower = aggregate['mean'] - aggregate['std']
    upper = aggregate['mean'] + aggregate['std']
    axis.fill_between(
        steps, lower, upper, color=color,
        alpha=0.16, linewidth=0,
    )
    axis.plot(steps, aggregate['mean'], color=color, linewidth=1.8, label=label)


def save_figure(figure, stem, formats):
    for extension in formats:
        figure.savefig(stem.with_suffix(f'.{extension}'), bbox_inches='tight')


def save_aggregate_csv(path, aggregate):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['agent_steps', 'mean', 'std', 'num_seeds'])
        for values in zip(
                aggregate['steps'], aggregate['mean'], aggregate['std']):
            writer.writerow([*values, aggregate['num_seeds']])


def plot_results(aggregates, metric, output_dir, individual_names, formats, ylabel=None):
    plt = _configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    ylabel = ylabel or _metric_label(metric)
    tab20 = plt.get_cmap('tab20').colors
    colors = tab20[::2] + tab20[1::2]

    for name in individual_names:
        aggregate = aggregates[name]
        figure, axis = plt.subplots(figsize=(5.5, 5.5 * 4.1 / 7.0))
        _draw_curve(axis, aggregate, None, colors[0])
        _style_axis(axis, ylabel)
        axis.set_title(_display_name(name))
        # figure.text(
        #     0.5, 0.025,
        #     rf'Mean $\pm$ standard deviation across $n = {aggregate["num_seeds"]}$ seeds.',
        #     ha='center', va='bottom', fontsize=7.5,
        # )
        figure.subplots_adjust(bottom=0.21, left=0.13, right=0.97, top=0.89)
        save_figure(figure, output_dir / name, formats)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.1))
    for index, (name, aggregate) in enumerate(aggregates.items()):
        _draw_curve(axis, aggregate, _display_name(name), colors[index % len(colors)])
    _style_axis(axis, ylabel)
    handles, labels = axis.get_legend_handles_labels()
    legend_columns = min(3, len(labels))
    legend_rows = math.ceil(len(labels) / legend_columns)
    figure.legend(
        handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.06),
        ncol=legend_columns,
        frameon=False, columnspacing=1.4, handlelength=2.2,
    )
    # seed_counts = {aggregate['num_seeds'] for aggregate in aggregates.values()}
    # count_text = str(seed_counts.pop()) if len(seed_counts) == 1 else 'varying'
    # figure.text(
    #     0.5, 0.015,
    #     rf'Mean $\pm$ standard deviation across $n = {count_text}$ seeds.',
    #     ha='center', va='bottom', fontsize=7.5,
    # )
    bottom_margin = 0.23 + 0.04 * (legend_rows - 1)
    figure.subplots_adjust(bottom=bottom_margin, left=0.11, right=0.98, top=0.97)
    save_figure(figure, output_dir / 'all_experiments', formats)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Plot seed-aggregated W&B training curves from an experiment manifest'
    )
    parser.add_argument('manifest')
    parser.add_argument('--metric', required=True)
    parser.add_argument('--entity', help='W&B entity; defaults to WANDB_ENTITY or API default')
    parser.add_argument('--output-dir', default='eval')
    parser.add_argument(
        '--experiment', action='append', default=[],
        help='Experiment to plot individually; repeatable. Defaults to every experiment.',
    )
    parser.add_argument('--points', type=int, default=400)
    parser.add_argument('--refresh', action='store_true', help='Redownload cached histories')
    parser.add_argument(
        '--full-history', action='store_true',
        help='Use W&B scan_history instead of the faster sampled-history API',
    )
    parser.add_argument(
        '--history-samples', type=int, default=10_000,
        help='Maximum points requested per run in the default fast mode',
    )
    parser.add_argument('--format', nargs='+', choices=['pdf', 'png', 'svg'], default=['pdf', 'png'])
    parser.add_argument('--ylabel')
    cli = parser.parse_args(argv)
    if cli.points < 2:
        parser.error('--points must be at least 2')
    if cli.history_samples < 2:
        parser.error('--history-samples must be at least 2')

    _, manifest = load_manifest(cli.manifest)
    experiment_names = [experiment['name'] for experiment in manifest['experiments']]
    unsafe = [name for name in experiment_names if _metric_slug(name) != name]
    if unsafe:
        parser.error(f'Experiment names are not safe output filenames: {unsafe}')
    unknown = set(cli.experiment) - set(experiment_names)
    if unknown:
        parser.error(f'Unknown experiments: {sorted(unknown)}')
    individual_names = cli.experiment or experiment_names

    import os
    import wandb

    api = wandb.Api()
    entity = cli.entity or os.environ.get('WANDB_ENTITY') or api.default_entity
    if not entity:
        parser.error('Could not determine W&B entity; provide --entity')
    project = manifest['batch'].get('project', 'puffer4')
    group = manifest['batch'].get('group', 'debug')
    if _metric_slug(group) != group:
        parser.error(f'W&B group is not a safe output directory name: {group!r}')
    print(f'Fetching W&B runs from {entity}/{project}, group={group}')
    runs = api.runs(f'{entity}/{project}', filters={'group': group})
    selected = select_runs(runs, manifest)
    expected = {
        (name, seed) for name in experiment_names for seed in manifest['batch']['seeds']
    }
    missing = sorted(expected - set(selected))
    if missing:
        formatted = ', '.join(f'{name}/seed-{seed}' for name, seed in missing)
        raise RuntimeError(f'Missing W&B runs for: {formatted}')

    root = Path(cli.output_dir).expanduser().resolve() / group / 'training_curves'
    metric_dir = root / _metric_slug(cli.metric)
    cache_dir = root / 'wandb_cache'
    aggregates = {}
    total_runs = len(experiment_names) * len(manifest['batch']['seeds'])
    with tqdm(total=total_runs, unit='run', desc='Fetching W&B histories') as progress:
        for name in experiment_names:
            histories = []
            for seed in manifest['batch']['seeds']:
                run = hydrate_run(selected[(name, seed)])
                progress.set_postfix_str(f'{name} seed {seed}')
                total_timesteps = run.config.get('train', {}).get('total_timesteps')
                if not isinstance(total_timesteps, (int, float)):
                    raise RuntimeError(f'Run {run.id} has no train.total_timesteps config')
                history = load_history(
                    run, cli.metric, cache_dir, cli.refresh,
                    cli.full_history, cli.history_samples,
                )
                histories.append(prepare_history(history, total_timesteps))
                progress.update()
            aggregates[name] = aggregate_histories(histories, cli.points)
            metric_dir.mkdir(parents=True, exist_ok=True)
            save_aggregate_csv(metric_dir / f'{name}.csv', aggregates[name])

    plot_results(
        aggregates, cli.metric, metric_dir, individual_names,
        cli.format, cli.ylabel,
    )
    print(f'Wrote plots and aggregate CSV files to {metric_dir}')


if __name__ == '__main__':
    main()
