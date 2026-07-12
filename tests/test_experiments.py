import textwrap
import json

import pytest

from pufferlib.experiments import (
    load_manifest,
    merge_overrides,
    select_task,
    task_count,
)


def write_manifest(tmp_path, extra=''):
    path = tmp_path / 'experiments.toml'
    path.write_text(textwrap.dedent(f'''\
        [batch]
        env = "quad_meshing"
        group = "test"
        seeds = [7, 11]

        [defaults.train]
        total_timesteps = 1000

        [[experiments]]
        name = "baseline"

        [[experiments]]
        name = "deeper"
        [experiments.policy]
        network_layers = 4
        {extra}
    '''))
    return path


def test_manifest_expands_experiments_by_seed(tmp_path):
    _, manifest = load_manifest(write_manifest(tmp_path))
    assert task_count(manifest) == 4
    assert select_task(manifest, 0)[0]['name'] == 'baseline'
    assert select_task(manifest, 0)[1] == 7
    assert select_task(manifest, 2)[0]['name'] == 'deeper'
    assert select_task(manifest, 3)[1] == 11


def test_manifest_rejects_duplicate_names(tmp_path):
    path = write_manifest(tmp_path).read_text().replace('deeper', 'baseline')
    manifest = tmp_path / 'duplicate.toml'
    manifest.write_text(path)
    with pytest.raises(ValueError, match='unique'):
        load_manifest(manifest)


def test_manifest_explains_toml_boolean_syntax(tmp_path):
    path = write_manifest(tmp_path)
    path.write_text(path.read_text() + '\ninvalid = False\n')
    with pytest.raises(ValueError, match='booleans must be lowercase'):
        load_manifest(path)


def test_json_manifest_does_not_require_toml_parser(tmp_path):
    _, manifest = load_manifest(write_manifest(tmp_path))
    path = tmp_path / 'experiments.json'
    path.write_text(json.dumps(manifest))
    _, loaded = load_manifest(path)
    assert loaded == manifest


def test_merge_overrides_is_recursive_and_validated():
    config = {'train': {'learning_rate': 0.01}, 'policy': {'layers': 2}}
    merge_overrides(config, {'train': {'learning_rate': 0.001}})
    assert config['train']['learning_rate'] == 0.001
    assert config['policy']['layers'] == 2

    with pytest.raises(ValueError, match='train.unknown'):
        merge_overrides(config, {'train': {'unknown': 1}})
    with pytest.raises(ValueError, match='Invalid type'):
        merge_overrides(config, {'policy': {'layers': 2.5}})
