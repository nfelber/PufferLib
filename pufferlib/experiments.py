import argparse
from copy import deepcopy
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


_BATCH_KEYS = {'env', 'group', 'project', 'seeds', 'slowly', 'tag', 'wandb'}


def load_manifest(path):
    path = Path(path).expanduser().resolve()
    with path.open('rb') as f:
        manifest = tomllib.load(f)

    unknown = set(manifest) - {'batch', 'defaults', 'experiments'}
    if unknown:
        raise ValueError(f'Unknown top-level manifest keys: {sorted(unknown)}')

    batch = manifest.get('batch')
    if not isinstance(batch, dict):
        raise ValueError('Manifest must contain a [batch] table')
    unknown = set(batch) - _BATCH_KEYS
    if unknown:
        raise ValueError(f'Unknown [batch] keys: {sorted(unknown)}')
    if not isinstance(batch.get('env'), str) or not batch['env']:
        raise ValueError('[batch].env must be a non-empty string')

    seeds = batch.get('seeds')
    if not isinstance(seeds, list) or not seeds:
        raise ValueError('[batch].seeds must be a non-empty list')
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError('[batch].seeds must contain only integers')

    experiments = manifest.get('experiments')
    if not isinstance(experiments, list) or not experiments:
        raise ValueError('Manifest must contain at least one [[experiments]] table')
    names = []
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError('Each [[experiments]] entry must be a table')
        name = experiment.get('name')
        if not isinstance(name, str) or not name:
            raise ValueError('Each [[experiments]] entry needs a non-empty name')
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError('Experiment names must be unique')

    defaults = manifest.get('defaults', {})
    if not isinstance(defaults, dict):
        raise ValueError('[defaults] must be a table')
    return path, manifest


def task_count(manifest):
    return len(manifest['experiments']) * len(manifest['batch']['seeds'])


def select_task(manifest, index):
    count = task_count(manifest)
    if index < 0 or index >= count:
        raise IndexError(f'Experiment task index {index} is outside 0..{count - 1}')
    seeds = manifest['batch']['seeds']
    experiment = manifest['experiments'][index // len(seeds)]
    return experiment, seeds[index % len(seeds)]


def _compatible_type(default, override):
    if default is None:
        return True
    if isinstance(default, bool):
        return isinstance(override, bool)
    if isinstance(default, int):
        return isinstance(override, int) and not isinstance(override, bool)
    if isinstance(default, float):
        return isinstance(override, (int, float)) and not isinstance(override, bool)
    return isinstance(override, type(default))


def merge_overrides(config, overrides, prefix=''):
    for key, value in overrides.items():
        path = f'{prefix}.{key}' if prefix else key
        if key not in config:
            raise ValueError(f'Unknown configuration key: {path}')
        if isinstance(config[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f'Configuration section {path} must be a table')
            merge_overrides(config[key], value, path)
        elif isinstance(value, dict) or not _compatible_type(config[key], value):
            expected = type(config[key]).__name__
            raise ValueError(f'Invalid type for {path}: expected {expected}')
        else:
            config[key] = value


def resolve_task(path, manifest, index, output_dir):
    # Importing pufferl loads the environment-specific native backend, so keep it
    # out of manifest-only operations used on the login node.
    from pufferlib import pufferl

    batch = manifest['batch']
    experiment, seed = select_task(manifest, index)
    args = pufferl.load_config(batch['env'], argv=[])
    merge_overrides(args, manifest.get('defaults', {}))
    merge_overrides(args, {k: v for k, v in experiment.items() if k != 'name'})

    args['seed'] = seed
    args['env']['seed'] = seed
    if 'seed' in args['train']:
        args['train']['seed'] = seed
    args['slowly'] = batch.get('slowly', True)
    args['wandb'] = batch.get('wandb', True)
    args['wandb_project'] = batch.get('project', 'puffer4')
    args['wandb_group'] = batch.get('group', 'debug')
    args['wandb_name'] = f'{experiment["name"]}-seed-{seed}'
    args['wandb_job_type'] = experiment['name']
    args['tag'] = batch.get('tag')
    args['experiment'] = experiment['name']
    args['experiment_seed'] = seed
    args['experiment_manifest'] = str(path)
    args['checkpoint_dir'] = str(Path(output_dir) / 'checkpoints')
    args['log_dir'] = str(Path(output_dir) / 'logs')
    return args


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run explicit PufferLib experiments')
    subparsers = parser.add_subparsers(dest='command', required=True)
    count_parser = subparsers.add_parser('count', help='Print expanded task count')
    count_parser.add_argument('manifest')
    env_parser = subparsers.add_parser('env', help='Print the manifest environment')
    env_parser.add_argument('manifest')
    run_parser = subparsers.add_parser('run', help='Run one expanded task')
    run_parser.add_argument('manifest')
    run_parser.add_argument('--index', type=int, required=True)
    run_parser.add_argument('--output-dir', required=True)
    cli = parser.parse_args(argv)

    path, manifest = load_manifest(cli.manifest)
    if cli.command == 'count':
        print(task_count(manifest))
        return
    if cli.command == 'env':
        print(manifest['batch']['env'])
        return

    experiment, seed = select_task(manifest, cli.index)
    print(f'Running experiment={experiment["name"]} seed={seed} '
          f'index={cli.index}/{task_count(manifest) - 1}')
    args = resolve_task(path, manifest, cli.index, cli.output_dir)
    from pufferlib import pufferl
    pufferl.train(env_name=manifest['batch']['env'], args=args)


if __name__ == '__main__':
    main()
