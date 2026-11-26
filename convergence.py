"""Retrospective learning-curve analysis inspired by T-REX, section 3.4.

Uses frozen-policy validation ATT, not best-so-far scores or test performance.
This is an empirical adaptation, not a mathematical convergence guarantee.
The CLI only needs learning_data.json and matplotlib; it imports no simulator,
training settings, TensorBoard reader, checkpoint, or test helper.
"""
import argparse
import json
import math
import os
from pathlib import Path


VERSION = 'trex_validation_convergence_v1'
REFERENCE = 'https://doi.org/10.1186/s12544-026-00841-1'
DEFAULT_CONFIG = {
    'smoothing_window': 3,
    'final_window': 5,
    'relative_tolerance': 0.05,
    'minimum_stable_updates': 500,
}
METRICS = ('ATT', 'Queue Length', 'Trip Completion Rate')


def validate_config(config):
    if not isinstance(config, dict) or set(config) != set(DEFAULT_CONFIG):
        raise ValueError('convergence must contain ' + ', '.join(DEFAULT_CONFIG))
    for key in ('smoothing_window', 'final_window', 'minimum_stable_updates'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'convergence.{key} must be a positive integer')
    tolerance = config['relative_tolerance']
    if type(tolerance) not in (float, int) or not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('convergence.relative_tolerance must be finite and in (0, 1)')
    return dict(config)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def number(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def prepare_reward_log(path, completed):
    """Discard uncommitted appends after interruption, including a partial line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = row.get('completed_updates')
            if type(step) is int and 0 < step <= completed:
                if rows and step <= rows[-1]['completed_updates']:
                    raise ValueError('Convergence training reward updates must be increasing')
                rows.append(row)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.writelines(json.dumps(row, allow_nan=False) + '\n' for row in rows)
    os.replace(temporary, path)


def append_reward(path, completed, reward, network_rewards):
    row = {'completed_updates': completed, 'reward': number(float(reward)),
           'networks': {name: number(float(value)) for name, value in network_rewards.items()}}
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')


def compact_validation(history, networks, seeds):
    """Preserve per-seed measurements; derive means with fixed equal weighting."""
    records = []
    for item in history:
        report = item['report']
        leaves = report.get('networks', {report.get('network'): report})
        if set(leaves) != set(networks):
            raise ValueError('Convergence validation network coverage differs from training')
        parts = {}
        for name in networks:
            episodes = leaves[name]['episodes']
            if [row['seed'] for row in episodes] != seeds:
                raise ValueError('Convergence validation seed coverage differs from fixed seeds')
            rows = [{key: number(row.get(key)) for key in METRICS} |
                    {'seed': row['seed'], 'completed_vehicles': row.get('completed_vehicles')}
                    for row in episodes]
            valid = all(type(row['completed_vehicles']) is int and row['completed_vehicles'] > 0
                        and row['ATT'] is not None and row['ATT'] > 0 for row in rows)
            means = {}
            for metric in METRICS:
                values = [row[metric] for row in rows]
                means[metric] = (sum(values) / len(values)
                                 if values and all(v is not None for v in values) else None)
            if not valid:
                means['ATT'] = None  # Zero completed trips cannot masquerade as zero ATT.
            parts[name] = {'valid': valid, 'metrics': means, 'episodes': rows}
        overall = {}
        for metric in METRICS:
            values = [parts[name]['metrics'][metric] for name in networks]
            overall[metric] = sum(values) / len(values) if all(v is not None for v in values) else None
        records.append({'completed_updates': item['completed_updates'],
                        'valid': all(part['valid'] for part in parts.values()),
                        'reason': item.get('reason'), 'metrics': overall, 'networks': parts})
    return records


def moving_average(values, window):
    """Trailing FULL windows only; retain nulls/gaps instead of bridging them."""
    result = []
    for i in range(len(values)):
        chunk = values[max(0, i-window+1):i+1]
        result.append(sum(chunk) / window if len(chunk) == window
                      and all(value is not None for value in chunk) else None)
    return result


def analyze_curve(updates, values, config):
    config = validate_config(config)
    if len(updates) != len(values) or any(type(u) is not int or u < 0 for u in updates):
        raise ValueError('Curve updates and values must align, with nonnegative integer updates')
    if any(b <= a for a, b in zip(updates, updates[1:])):
        raise ValueError('Curve updates must be strictly increasing')
    valid = [number(value) for value in values]
    valid = [value if value is not None and value > 0 else None for value in valid]
    smooth = moving_average(valid, config['smoothing_window'])
    result = {'status': 'insufficient_data', 'reason': None,
              'estimated_convergence_update': None, 'confirmation_update': None,
              'analysis_end_update': updates[-1] if updates else None,
              'validation_points': len(updates), 'invalid_points': valid.count(None),
              'initial_reference_update': None, 'initial_att': None,
              'final_reference_att': None, 'final_smoothed_att': None,
              'best_smoothed_att': None, 'best_smoothed_update': None,
              'tolerance_band': None, 'observed_stable_updates': None,
              'CR': None, 'FPD': None, 'smoothed_att': smooth}
    if not updates:
        result['reason'] = 'No fixed-policy validation observations'
        return result
    if None in valid:
        result.update(status='invalid_measurements',
                      reason='Missing/invalid ATT or empty trips; no gap-spanning convergence claim')
        return result
    usable = [(u, value) for u, value in zip(updates, smooth) if value is not None]
    if len(usable) < config['final_window']:
        result['reason'] = 'Too few full smoothing windows for the final reference'
        return result
    initial = usable[0][1]
    final = usable[-1][1]
    reference = sum(value for _, value in usable[-config['final_window']:]) / config['final_window']
    best_update, best = min(usable, key=lambda pair: pair[1])
    lower, upper = reference * (1-config['relative_tolerance']), reference * (1+config['relative_tolerance'])
    result.update(initial_reference_update=usable[0][0], initial_att=initial,
                  final_reference_att=reference, final_smoothed_att=final,
                  best_smoothed_att=best, best_smoothed_update=best_update,
                  tolerance_band=[lower, upper], FPD=abs(final-best)/best)
    # T-REX's final/best *smoothed* curve values define FPD and the CR sign.
    # A trailing-window reference and minimum real update span make the
    # validation-grid stability test explicit; these are our declared adaptation.
    candidates = [(i, u) for i, (u, _) in enumerate(usable)
                  if updates[-1]-u >= config['minimum_stable_updates']]
    if not candidates:
        result['reason'] = 'Insufficient observation span to confirm sustained stability'
        return result
    for i, step in candidates:
        if all(lower <= value <= upper for _, value in usable[i:]):
            confirmation = next(u for u, _ in usable[i:]
                                if u-step >= config['minimum_stable_updates'])
            sign = 1 if final < initial else -1 if final > initial else 0
            result.update(status='converged', reason='Tolerance band holds through the end of observation',
                          estimated_convergence_update=step, confirmation_update=confirmation,
                          observed_stable_updates=updates[-1]-step,
                          CR=sign/step if step else None)
            return result
    result.update(status='not_converged', reason='No sufficiently long terminal segment stays in the tolerance band')
    return result


def build_data(run, selection, config, metadata):
    root = Path(run)
    rewards = root / 'convergence/training_rewards.jsonl'
    completed = metadata['completed_updates']
    training = [json.loads(line) for line in rewards.read_text().splitlines()] if rewards.exists() else []
    training = [row for row in training if row['completed_updates'] <= completed]
    return {'format_version': VERSION, 'reference': REFERENCE,
            'adaptation': 'Frozen-policy validation ATT; trailing full windows, terminal reference mean and minimum stable update span',
            'analysis_config': validate_config(config), 'metadata': metadata,
            'definitions': {
                'x_axis': 'completed PPO updates, not SUMO seconds, gradient epochs or individual Worker episodes',
                'ATT': 'Mean completed-trip travel time (s), equal weight over fixed validation traffic seeds',
                'overall': 'Equal weight per training-network validation mean',
                'reward': 'Mean sampled episode reward across Workers; per-network means stored separately',
                'FPD': 'abs(last smoothed ATT - minimum smoothed ATT) / minimum smoothed ATT',
                'CR': 'sign(initial smoothed ATT - last smoothed ATT) / estimated convergence update; null if unconfirmed',
                'uncertainty': 'One training seed; no cross-training-seed confidence band is inferred',
                'scope': 'Retrospective empirical stability within this observed budget, not proof of convergence or guaranteed future budget'},
            'training_rewards': training,
            'validation': compact_validation(selection['history'], metadata['networks'], metadata['validation_seeds'])}


def _series(data, scope, metric):
    return [row['metrics'][metric] if scope == 'overall' else row['networks'][scope]['metrics'][metric]
            for row in data['validation']]


def analyze_data(data):
    if data['format_version'] != VERSION:
        raise ValueError('Unsupported convergence data format')
    config = validate_config(data['analysis_config'])
    updates = [row['completed_updates'] for row in data['validation']]
    scopes = ['overall', *data['metadata']['networks']]
    results = {name: analyze_curve(updates, _series(data, name, 'ATT'), config) for name in scopes}
    if data['metadata']['training_status'] == 'running':
        for result in results.values():
            result.update(status='incomplete_training', reason='Training has not finished; no final convergence claim',
                          estimated_convergence_update=None, confirmation_update=None, CR=None)
    return {'format_version': VERSION, 'reference': REFERENCE,
            'method': data['adaptation'], 'analysis_config': config, 'metadata': data['metadata'],
            'resolution_updates': data['metadata']['validation_interval'],
            'overall': results['overall'], 'networks': {name: results[name] for name in data['metadata']['networks']},
            'interpretation': 'converged means empirical stability, which can also be suboptimal or degraded; inspect final ATT, CR sign and completion flow. Null is not zero. No automatic change to max_episodes or early stopping.'}


def plot_data(data, summary, output):
    # Lazy import: plotting never runs in the PPO update/sampling path.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    output = Path(output)
    updates = [row['completed_updates'] for row in data['validation']]
    scopes = ['overall'] if len(data['metadata']['networks']) == 1 else ['overall', *data['metadata']['networks']]
    clean = lambda values: [float('nan') if value is None else value for value in values]
    for scope in scopes:
        result = summary['overall'] if scope == 'overall' else summary['networks'][scope]
        fig = Figure(figsize=(12, 8), constrained_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 2)
        ax = axes[0, 0]
        ax.plot(updates, clean(_series(data, scope, 'ATT')), 'o-', color='#8aa4c8', lw=1, ms=3,
                label='Observed validation ATT')
        ax.plot(updates, clean(result['smoothed_att']), color='#2166ac', lw=2, label='Trailing mean')
        if result['tolerance_band'] is not None:
            ax.axhspan(*result['tolerance_band'], color='#1b9e77', alpha=.15, label='Stability tolerance band')
            ax.axhline(result['final_reference_att'], color='#1b9e77', ls=':', lw=1)
        if result['estimated_convergence_update'] is not None:
            ax.axvline(result['estimated_convergence_update'], color='#d95f02', ls='--',
                       label=f"Estimated Tc = {result['estimated_convergence_update']}")
        ax.set(title='Fixed-policy validation travel time', ylabel='ATT (s)')
        ax.legend(fontsize=8)
        rewards = data['training_rewards']
        ax = axes[0, 1]
        reward_values = [row['reward'] if scope == 'overall' else row['networks'].get(scope) for row in rewards]
        ax.plot([row['completed_updates'] for row in rewards], clean(reward_values), color='#7570b3', lw=1)
        ax.set(title='Sampled training reward', ylabel='Mean Worker episode reward')
        if not rewards:
            ax.text(.5, .5, 'No reward records available', transform=ax.transAxes, ha='center')
        for ax, metric, label in [(axes[1, 0], 'Queue Length', 'Queue length (veh/lane)'),
                                  (axes[1, 1], 'Trip Completion Rate', 'Completed vehicles / s')]:
            ax.plot(updates, clean(_series(data, scope, metric)), 'o-', color='#2166ac', lw=1.5, ms=3)
            ax.set(title='Fixed-policy validation '+metric, ylabel=label)
        for ax in axes.flat:
            ax.set_xlabel('Completed PPO updates')
            ax.grid(alpha=.2)
        info = data['metadata']
        title = f"{info['algorithm']} | seed={info['training_seed']} | {scope} | {result['status']}"
        fig.suptitle(title+'\nSingle training run; empirical validation stability', fontsize=12)
        filename = 'learning_curve.png' if scope == 'overall' else f'learning_curve_{scope}.png'
        temporary = output / (filename+'.tmp')
        try:
            fig.savefig(temporary, format='png', dpi=160, metadata={'Software': VERSION})
            os.replace(temporary, output / filename)
        finally:
            temporary.unlink(missing_ok=True)
            fig.clear()


def regenerate(input_path, output=None):
    """Rebuild JSON analysis and PNG using only the self-contained saved data."""
    input_path = Path(input_path)
    data = json.loads(input_path.read_text())
    output = Path(output) if output is not None else input_path.parent
    output.mkdir(parents=True, exist_ok=True)
    summary = analyze_data(data)
    write_json(output / 'summary.json', summary)
    plot_data(data, summary, output)
    return summary


def finish(run, selection, config, metadata):
    output = Path(run) / 'convergence'
    data = build_data(run, selection, config, metadata)
    # Commit base data before plotting; images can always be recreated from it.
    write_json(output / 'learning_data.json', data)
    return regenerate(output / 'learning_data.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Saved convergence/learning_data.json')
    parser.add_argument('--output', type=Path, help='Output directory; defaults to the input directory')
    args = parser.parse_args()
    report = regenerate(args.input, args.output)
    print(json.dumps({'status': report['overall']['status'],
                      'estimated_convergence_update': report['overall']['estimated_convergence_update']}, indent=2))


if __name__ == '__main__':
    main()
