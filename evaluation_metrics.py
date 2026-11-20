"""SUMO evaluation metrics shared by learning and rule-based controllers.

ATT/AQL follow LibSignal commit 127af9f93902 (world/world_sumo.py and
common/metrics.py). AWT is the completed trips' mean SUMO waitingTime.
"""
import json
from math import fsum
from pathlib import Path
from statistics import fmean, pstdev
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml


PAPER_METRICS = ('Queue Length', 'Speed', 'Intersection Delay',
                 'Trip Completion Rate', 'Trip Time', 'Trip Delay')
METRICS = ('ATT', 'AWT', 'AQL', *PAPER_METRICS)
TRIP_METRICS = {'ATT', 'AWT', 'Trip Time', 'Trip Delay'}
TRIP_COLUMNS = ('episode', 'id', 'arrival_sec', 'duration_sec', 'wait_sec', 'time_loss_sec')


def summarize_episode(traffic, trips, episode, seed):
    """One calculation path for live evaluation data and historical CSV files."""
    traffic = pd.DataFrame(traffic)
    trips = pd.DataFrame(trips)
    traffic_columns = ('episode', 'time_sec', 'avg_queue', 'avg_speed_mps',
                       'avg_wait_sec', 'number_arrived_car')
    missing = set(traffic_columns) - set(traffic.columns)
    if missing:
        raise ValueError(f'Traffic data is missing columns: {sorted(missing)}')
    traffic = traffic.loc[traffic.episode == episode].sort_values('time_sec')
    if traffic.empty:
        raise ValueError(f'No traffic samples for episode {episode}')
    values = traffic[list(traffic_columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite traffic data in episode {episode}')
    times = traffic.time_sec.to_numpy(dtype=float)
    # Traffic records are written every simulation step; account for clock
    # offsets (e.g. Monaco) rather than dividing by the absolute end timestamp.
    differences = np.diff(times)
    step = float(differences[0]) if len(differences) else 1.0
    if step <= 0 or not np.allclose(differences, step):
        raise ValueError('Traffic timestamps must be unique and uniformly spaced')
    seconds = float(times[-1] - times[0] + step)
    arrived = traffic.number_arrived_car.to_numpy(dtype=float)
    if (arrived < 0).any() or not np.equal(arrived, np.floor(arrived)).all():
        raise ValueError('number_arrived_car must contain nonnegative per-step integer counts')

    if trips.empty:
        finished = pd.DataFrame(columns=TRIP_COLUMNS)
    else:
        missing = set(TRIP_COLUMNS) - set(trips.columns)
        if missing:
            raise ValueError(f'Trip data is missing columns: {sorted(missing)}')
        finished = trips.loc[(trips.episode == episode) & (trips.arrival_sec.astype(float) >= 0)]
        if finished.id.duplicated().any():
            raise ValueError(f'Duplicate completed vehicle IDs in episode {episode}')
        measurements = finished[['duration_sec', 'wait_sec', 'time_loss_sec']].to_numpy(dtype=float)
        if not np.isfinite(measurements).all() or (measurements < 0).any():
            raise ValueError('Completed trip measurements must be finite and nonnegative')
    count = len(finished)
    if count != int(arrived.sum()):
        raise ValueError(f'Completed SUMO arrivals and TripInfo records do not match in episode {episode}')
    # Keep LibSignal's ATT=0 fallback, but exclude empty-trip episodes from
    # cross-seed travel/waiting aggregates. Trip Time reuses the ATT result.
    att = fsum(finished.duration_sec.astype(float)) / count if count else 0.0
    awt = fsum(finished.wait_sec.astype(float)) / count if count else None
    loss = fsum(finished.time_loss_sec.astype(float)) / count if count else None
    queue = pd.to_numeric(traffic['AQL'], errors='raise').dropna() if 'AQL' in traffic else pd.Series(dtype=float)
    if not np.isfinite(queue.to_numpy()).all() or (queue < 0).any():
        raise ValueError('AQL samples must be finite and nonnegative')
    return {
        'episode': int(episode), 'seed': int(seed), 'completed_vehicles': count,
        'simulation_seconds': seconds, 'traffic_samples': len(traffic),
        'queue_samples': len(queue),
        'ATT': att, 'AWT': awt, 'AQL': float(queue.mean()) if len(queue) else None,
        'Queue Length': float(traffic.avg_queue.mean()),
        'Speed': float(traffic.avg_speed_mps.mean()),
        'Intersection Delay': float(traffic.avg_wait_sec.mean()),
        'Trip Completion Rate': float(arrived.sum() / seconds),
        'Trip Time': att if count else None,
        'Trip Delay': loss,
    }


class SumoEvaluationMetrics:
    DETECTION_DISTANCE = 200

    def __init__(self, incoming_lanes, outgoing_lanes):
        self.incoming = {tls: tuple(lanes) for tls, lanes in incoming_lanes.items()}
        self.lanes = {tls: tuple(dict.fromkeys((*lanes, *outgoing_lanes[tls])))
                      for tls, lanes in self.incoming.items()}
        if not self.incoming:
            raise ValueError('Evaluation metrics require at least one intersection')
        self.reset()

    def reset(self):
        self.waiting_seen = {tls: set() for tls in self.incoming}

    def observe_step(self, simulator, sample_queue):
        # Read-only queries; this must not change actions, simulation or RNG state.
        nearby = {}
        waiting = {}
        for lane in dict.fromkeys(lane for lanes in self.lanes.values() for lane in lanes):
            nearby[lane] = []
            for vehicle in simulator.lane.getLastStepVehicleIDs(lane):
                if vehicle not in waiting:
                    next_tls = simulator.vehicle.getNextTLS(vehicle)
                    waiting[vehicle] = (simulator.vehicle.getWaitingTime(vehicle)
                                        if next_tls and next_tls[0][2] <= self.DETECTION_DISTANCE else None)
                if waiting[vehicle] is not None:
                    nearby[lane].append(vehicle)

        total_queue = 0
        for tls, lanes in self.lanes.items():
            seen = self.waiting_seen[tls]
            for lane in lanes:
                seen.update(vehicle for vehicle in nearby[lane] if waiting[vehicle] > 0)
            # LibSignal retains a vehicle's positive waiting state after it moves.
            # Preserve that behavior for reference parity, rather than substituting
            # instantaneous halting counts from Unicorn's original avg_queue.
            total_queue += sum(vehicle in seen for lane in self.incoming[tls]
                               for vehicle in nearby[lane])

        if not sample_queue:
            return None
        return total_queue / len(self.incoming)


class EvaluationReport:
    def __init__(self, run, network, decision_seconds):
        self.run = run
        self.network = network
        self.decision_seconds = decision_seconds
        self.episodes = []

    def add_episode(self, traffic, trips, episode, seed):
        result = summarize_episode(traffic, trips, episode, seed)
        self.episodes.append(result)
        att = f"{result['ATT']:.3f}s" if result['completed_vehicles'] else 'N/A (no completed trips)'
        awt = f"{result['AWT']:.3f}s" if result['AWT'] is not None else 'N/A'
        aql = f"{result['AQL']:.3f}veh/intersection" if result['AQL'] is not None else 'N/A (not recorded)'
        print(f"[Metrics] seed={seed} ATT={att} AWT={awt} "
              f"AQL={aql} completed={result['completed_vehicles']}")
        paper = ', '.join(f'{name}={result[name]:.3f}' if result[name] is not None else f'{name}=N/A'
                          for name in PAPER_METRICS)
        print(f'[Paper metrics] seed={seed} {paper}')

    def save(self):
        aggregate = {}
        for metric in METRICS:
            values = [row[metric] for row in self.episodes if row[metric] is not None
                      and (metric not in TRIP_METRICS or row['completed_vehicles'] > 0)]
            aggregate[metric] = {'mean': fmean(values) if values else None,
                                 'std': pstdev(values) if values else None,
                                 'valid_episodes': len(values)}
        report = {
            'algorithm': self.run.algorithm, 'network': self.network,
            'definitions': {
                'reference': {'repository': 'DaRL-LibSignal/LibSignal',
                              'commit': '127af9f93902', 'metrics': ['ATT', 'AQL']},
                'ATT': {'unit': 's', 'formula': 'mean(duration_sec) of completed trips; 0 when none',
                        'scope': 'Actual network departure to arrival; excludes unfinished trips and insertion delay.'},
                'AWT': {'unit': 's', 'formula': 'mean(wait_sec) of completed trips',
                        'scope': 'SUMO TripInfo waitingTime: cumulative stopped time over each completed trip.'},
                'AQL': {'unit': 'veh/intersection', 'formula': 'mean(nonempty AQL samples in traffic CSV)',
                        'scope': 'LibSignal incoming-lane waiting counts averaged over intersections. Positive waiting state is retained after moving, as in the reference. Missing historical AQL is null, not reconstructed from avg_queue.'},
                'Queue Length': {'unit': 'veh/lane', 'formula': 'mean(avg_queue) over simulation steps',
                                 'scope': 'Original Unicorn incoming-lane halting-count average; not LibSignal AQL.'},
                'Speed': {'unit': 'm/s', 'formula': 'mean(avg_speed_mps) over simulation steps',
                          'scope': 'Original network vehicle-speed average at each step; empty-network samples remain zero.'},
                'Intersection Delay': {'unit': 's', 'formula': 'mean(avg_wait_sec) over simulation steps',
                                       'scope': 'Original network average of current continuous waiting time, reset by vehicle movement.'},
                'Trip Completion Rate': {'unit': 'veh/s', 'formula': 'sum(number_arrived_car) / simulation_seconds',
                                         'scope': 'Arrival flow, not the percentage of requested trips completed.'},
                'Trip Time': {'unit': 's', 'formula': 'mean(duration_sec) of completed trips; reuses ATT',
                              'scope': 'Completed vehicles only; null when none.'},
                'Trip Delay': {'unit': 's', 'formula': 'mean(time_loss_sec) of completed trips',
                               'scope': 'SUMO timeLoss relative to ideal speed, including low-speed driving losses; not cumulative waitingTime.'},
                'detection_distance_m': SumoEvaluationMetrics.DETECTION_DISTANCE,
                'queue_sampling_seconds': self.decision_seconds,
                'queue_series': 'The traffic CSV AQL column contains decision-boundary samples; other rows are empty.',
                'aggregate': 'Equal weight per valid seed/episode; std is population standard deviation (ddof=0).',
                'paper_alignment': 'The six fields use the original Unicorn CSV measurements. Trip Delay is operationalized as SUMO timeLoss. The public paper/code do not specify the complete table-aggregation script; cross-seed std is not asserted to reproduce Table II parentheses.',
            },
            'episodes': self.episodes, 'aggregate': aggregate,
        }
        path = self.run.data / 'metrics.json'
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('Evaluation metrics:', path)
        return report


def summarize_run(run_dir):
    """Read an evaluation run's CSVs and replace only data/metrics.json."""
    run_dir = Path(run_dir).expanduser().resolve()
    config = yaml.safe_load((run_dir / 'config.yaml').read_text())
    if config['stage'] != 'evaluate':
        raise ValueError('Expected an evaluation run directory')
    settings = config['evaluation']
    network = settings['NETWORK']
    seeds = settings['SEEDS']
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Evaluation snapshot must contain distinct seeds')
    data = run_dir / 'data'
    traffic = pd.read_csv(data / f'{network}_traffic.csv')
    try:
        trips = pd.read_csv(data / f'{network}_trip.csv')
    except pd.errors.EmptyDataError:
        trips = pd.DataFrame(columns=TRIP_COLUMNS)
    expected = set(range(len(seeds)))
    if set(traffic.episode) != expected or (not trips.empty and not set(trips.episode) <= expected):
        raise ValueError('CSV episodes do not match the evaluation seed list')
    run = SimpleNamespace(data=data, algorithm=config['algorithm'])
    report = EvaluationReport(run, network, config['sumo']['GREEN_DURATION'])
    for episode, seed in enumerate(seeds):
        report.add_episode(traffic, trips, episode, seed)
    return report.save()
