"""One validated sampling plan and shared representation for selected networks.

This module does not import parameters, launch SUMO, or construct a model.
Native movement/phase sizes come from the same static Tls representation used
by environments; no constants for a particular six-network training set.
"""
from functools import lru_cache
import json
from pathlib import Path
from statistics import mean, pstdev


def parse_training(config, root):
    if not isinstance(config, dict):
        raise ValueError('training must be a mapping')
    if set(config) == {'network', 'seed'}:
        # Compatibility with the previous single-network YAML/Ray snapshots.
        config = {'mode': 'single', 'seed': config['seed'],
                  'networks': [{'name': config['network'], 'workers': 6}]}
    if set(config) != {'mode', 'seed', 'networks'}:
        raise ValueError('training must contain mode, seed and networks; do not mix network/networks')
    mode, seed, networks = (config[k] for k in ('mode', 'seed', 'networks'))
    if mode not in ('single', 'joint'):
        raise ValueError('training.mode must be single or joint')
    if type(seed) is not int or not 0 <= seed < 2**32 - 10000:
        raise ValueError('training.seed must be a nonnegative integer below 2**32 - 10000')
    if not isinstance(networks, list) or not networks:
        raise ValueError('training.networks must be a nonempty list')
    result = []
    for item in networks:
        if not isinstance(item, dict) or set(item) != {'name', 'workers'}:
            raise ValueError('Each training.networks entry must contain name and workers')
        name, workers = item['name'], item['workers']
        if (not isinstance(name, str) or Path(name).name != name
                or not (Path(root)/'maps'/name/f'{name}_config.json').is_file()):
            raise ValueError('training network must name an existing maps directory')
        if type(workers) is not int or workers < 1:
            raise ValueError('training network workers must be a positive integer')
        result.append({'name': name, 'workers': workers})
    if len({r['name'] for r in result}) != len(result):
        raise ValueError('Duplicate training network; set its workers instead')
    if (mode == 'single' and len(result) != 1) or (mode == 'joint' and len(result) < 2):
        raise ValueError('single needs exactly one network; joint needs at least two distinct networks')
    return {'mode': mode, 'seed': seed, 'networks': result}


def worker_networks(plan):
    return [item['name'] for item in plan['networks'] for _ in range(item['workers'])]


@lru_cache(maxsize=None)
def native_layout(root, name):
    from env.tls import Tls
    config = json.loads((Path(root)/'maps'/name/f'{name}_config.json').read_text())
    if not config:
        raise ValueError(f'Empty network configuration: {name}')
    signals = [Tls(k, v, None) for k, v in config.items()]
    return {'movement_dim': max(len(t.traffic_movement_list) for t in signals),
            'max_phases': max(len(t.action_space) for t in signals), 'junctions': len(signals)}


def joint_layout(plan, root):
    native = {r['name']: native_layout(str(root), r['name']) for r in plan['networks']}
    return {'movement_dim': max(s['movement_dim'] for s in native.values()),
            'max_phases': max(s['max_phases'] for s in native.values()),
            'total_agents': sum(native[r['name']]['junctions'] * r['workers'] for r in plan['networks']),
            'native': native}


def apply_schema(native_movement, native_phases, schema):
    """Use the saved representation for evaluation, never truncate a target."""
    if not isinstance(schema, dict):
        raise ValueError('Invalid checkpoint representation schema')
    dims, phases = schema.get('input_dim'), schema.get('max_phases')
    if (not isinstance(dims, list) or len(dims) != 2
            or any(type(v) is not int or v <= 0 for v in dims)
            or dims[1] != 8 or schema.get('int_vec_dim') != 55
            or type(phases) is not int or phases < 1
            or dims[0] < native_movement or phases < native_phases):
        raise ValueError(f'Target network exceeds/incompatible with checkpoint schema {schema}; '
                         f'native movements={native_movement}, phases={native_phases}')
    return dims[0], phases


def merge_validation(reports):
    """Macro over network means; retain every episode for ATT validity checks."""
    aggregate = {}
    metrics = next(iter(reports.values()))['aggregate']
    for metric in metrics:
        values = [r['aggregate'][metric]['mean'] for r in reports.values()
                  if r['aggregate'][metric]['mean'] is not None]
        aggregate[metric] = {'mean': mean(values) if values else None,
                             'std': pstdev(values) if values else None,
                             'valid_networks': len(values)}
    return {'mode': 'joint', 'stage': 'validation', 'networks': reports,
            'episodes': [dict(row, network=name) for name, report in reports.items() for row in report['episodes']],
            'aggregate': aggregate,
            'aggregation': {'mean': 'equal weight per training-network mean',
                            'std': 'population standard deviation of network means'}}
