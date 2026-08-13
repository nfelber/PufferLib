import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm

from pufferlib.experiments import load_manifest, resolve_task


_SAFE_COMPONENT = re.compile(r'^[A-Za-z0-9._-]+$')
_CHECKPOINT = re.compile(r'^(\d+)\.bin$')
_SHAPE_SUFFIX = {
    'quad_meshing': '.qmshape',
    'quad_meshing_3d': '.qmsurf',
}


def _safe_component(value, label):
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f'{label} must contain only letters, digits, dots, underscores, or hyphens: {value!r}'
        )
    return value


def expected_runs(manifest):
    runs = []
    for experiment in manifest['experiments']:
        name = _safe_component(experiment['name'], 'Experiment name')
        for seed in manifest['batch']['seeds']:
            runs.append((name, seed))
    return runs


def _final_agent_step(log):
    steps = log.get('metrics', {}).get('agent_steps', [])
    return max((step for step in steps if isinstance(step, (int, float))), default=-1)


def discover_completed_runs(log_dir, manifest):
    expected = set(expected_runs(manifest))
    group = manifest['batch'].get('group', 'debug')
    candidates = {key: [] for key in expected}
    invalid = []
    for path in sorted(Path(log_dir).glob('*.json')):
        try:
            with path.open() as f:
                log = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            invalid.append(f'{path.name}: {e}')
            continue

        key = (log.get('experiment'), log.get('experiment_seed'))
        if (key not in expected or log.get('env_name') != manifest['batch']['env']
                or log.get('wandb_group', 'debug') != group or not log.get('metrics')):
            continue
        candidates[key].append((path, log))

    selected = {}
    for key, matches in candidates.items():
        if not matches:
            continue
        matches.sort(key=lambda item: (
            _final_agent_step(item[1]), item[0].stat().st_mtime_ns, item[0].stem
        ))
        selected[key] = matches[-1][0].stem
        if len(matches) > 1:
            discarded = ', '.join(path.stem for path, _ in matches[:-1])
            print(f'Using run {selected[key]} for {key[0]} seed {key[1]}; '
                  f'discarded older runs: {discarded}')
    if invalid:
        print('Ignored invalid log files: ' + '; '.join(invalid))
    return selected


def latest_checkpoint(directory):
    candidates = []
    for path in Path(directory).iterdir():
        match = _CHECKPOINT.fullmatch(path.name)
        if match and path.is_file() and path.stat().st_size > 0:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f'No non-empty numeric .bin checkpoints in {directory}')
    return max(candidates, key=lambda item: item[0])[1]


def _remote_path(root, relative):
    return f'{root.rstrip("/")}/{relative.as_posix()}/'


def _rsync(source, destination, filters=()):
    command = ['rsync', '-a', *filters, source, str(destination)]
    print(' '.join(command))
    subprocess.run(command, check=True)


def download_models(manifest, remote_root, models_dir):
    env_name = manifest['batch']['env']
    with tempfile.TemporaryDirectory(prefix='puffer-eval-') as tmp:
        tmp = Path(tmp)
        logs_dir = tmp / 'logs'
        logs_dir.mkdir()
        _rsync(
            _remote_path(remote_root, Path('logs') / env_name),
            logs_dir,
            filters=('--include=*.json', '--exclude=*'),
        )
        selected = discover_completed_runs(logs_dir, manifest)
        missing = [key for key in expected_runs(manifest) if key not in selected]
        if missing:
            formatted = ', '.join(f'{name}/seed-{seed}' for name, seed in missing)
            raise RuntimeError(f'No completed remote run found for: {formatted}')

        for name, seed in expected_runs(manifest):
            run_id = selected[(name, seed)]
            checkpoint_dir = tmp / 'checkpoints' / run_id
            checkpoint_dir.mkdir(parents=True)
            _rsync(
                _remote_path(remote_root, Path('checkpoints') / env_name / run_id),
                checkpoint_dir,
            )
            source = latest_checkpoint(checkpoint_dir)
            destination = models_dir / name / str(seed) / 'model.pt'
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            print(f'Downloaded {name} seed {seed}: {source.name} -> {destination}')


def evaluate_checkpoint(manifest_path, manifest, task_index, checkpoint,
        shapes, mesh_dir, max_steps, batch_size, progress):
    from pufferlib import pufferl

    args = resolve_task(manifest_path, manifest, task_index, mesh_dir)
    args['slowly'] = True
    args['wandb'] = False
    args['load_model_path'] = None
    args['reset_state'] = False
    args['vec']['num_buffers'] = 1
    args['train']['horizon'] = 1

    backend = pufferl._resolve_backend(args)
    for start in range(0, len(shapes), batch_size):
        batch = shapes[start:start + batch_size]
        args['vec']['total_agents'] = len(batch)
        args['train']['minibatch_size'] = len(batch)
        args['env']['shape_folder'] = str(batch[0].parent.resolve())
        args['env']['shape_names'] = [shape.name for shape in batch]
        args['env']['export_obj'] = True

        mesh_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.staging-', dir=mesh_dir) as staging:
            staging = Path(staging)
            staged = [staging / f'{index}.obj' for index in range(len(batch))]
            destinations = [mesh_dir / f'{shape.stem}.obj' for shape in batch]
            for destination in destinations:
                destination.unlink(missing_ok=True)
            args['env']['export_obj_paths'] = [str(path.resolve()) for path in staged]

            runner = backend.create_pufferl(args)
            try:
                backend.load_weights(runner, str(checkpoint.resolve()))
                runner.policy.eval()
                pending = set(range(len(batch)))
                for _ in range(max_steps):
                    backend.rollouts(runner)
                    completed = [index for index in pending if staged[index].exists()]
                    for index in completed:
                        staged[index].replace(destinations[index])
                        pending.remove(index)
                        progress.update()
                    if not pending:
                        break
                    progress.set_postfix_str(
                        f'{args["experiment"]} seed {args["experiment_seed"]}: '
                        f'{len(batch) - len(pending)}/{len(batch)} in batch'
                    )
                else:
                    names = ', '.join(batch[index].name for index in sorted(pending))
                    raise RuntimeError(
                        f'{len(pending)} shapes did not export within {max_steps} steps: {names}'
                    )
            finally:
                backend.close(runner)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Download and evaluate all policies in an experiment manifest'
    )
    parser.add_argument('manifest')
    parser.add_argument('dataset', help='Folder containing .qmshape or .qmsurf test shapes')
    parser.add_argument('--output-dir', default='eval')
    parser.add_argument('--download', action='store_true')
    parser.add_argument(
        '--remote-root',
        help='Rsync source root, e.g. user@host:~/PufferLib/cluster_runs/experiments/quad_meshing',
    )
    parser.add_argument('--max-steps', type=int, default=1_000_000)
    parser.add_argument(
        '--batch-size', type=int, default=0,
        help='Shapes evaluated concurrently per model; 0 uses the full dataset',
    )
    cli = parser.parse_args(argv)

    manifest_path, manifest = load_manifest(cli.manifest)
    env_name = manifest['batch']['env']
    if env_name not in _SHAPE_SUFFIX:
        raise ValueError(f'Unsupported evaluation environment: {env_name}')
    if cli.download and not cli.remote_root:
        parser.error('--download requires --remote-root')
    if cli.max_steps <= 0:
        parser.error('--max-steps must be positive')
    if cli.batch_size < 0:
        parser.error('--batch-size must be non-negative')

    dataset = Path(cli.dataset).expanduser().resolve()
    if not dataset.is_dir():
        raise NotADirectoryError(dataset)
    shapes = sorted(dataset.glob(f'*{_SHAPE_SUFFIX[env_name]}'))
    if not shapes:
        raise FileNotFoundError(
            f'No {_SHAPE_SUFFIX[env_name]} shapes found directly in {dataset}'
        )

    group = _safe_component(manifest['batch'].get('group', 'debug'), 'Experiment group')
    root = Path(cli.output_dir).expanduser().resolve() / group
    models_dir = root / 'models'
    meshes_dir = root / 'meshes'
    if cli.download:
        download_models(manifest, cli.remote_root, models_dir)

    runs = expected_runs(manifest)
    missing_models = [
        models_dir / name / str(seed) / 'model.pt'
        for name, seed in runs
        if not (models_dir / name / str(seed) / 'model.pt').is_file()
    ]
    if missing_models:
        formatted = '\n'.join(f'  {path}' for path in missing_models)
        raise FileNotFoundError(
            f'Missing {len(missing_models)} expected models. Provide --download '
            f'or place checkpoints at these paths:\n{formatted}'
        )

    total = len(runs) * len(shapes)
    batch_size = cli.batch_size or len(shapes)
    with tqdm(total=total, unit='mesh', desc='Evaluating policies') as progress:
        for task_index, (name, seed) in enumerate(runs):
            checkpoint = models_dir / name / str(seed) / 'model.pt'
            mesh_dir = meshes_dir / name / str(seed)
            evaluate_checkpoint(
                manifest_path, manifest, task_index, checkpoint, shapes,
                mesh_dir, cli.max_steps, batch_size, progress,
            )


if __name__ == '__main__':
    main()
