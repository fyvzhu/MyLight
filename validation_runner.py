"""Isolated source-network validation; reuses the normal evaluation algorithms."""
import argparse
import hashlib
import json
from pathlib import Path

from evaluation import seed_policy
from parameters import VALIDATION_PARAMS, SUMO_PARAMS, SCENARIO_PARAMS
from run_paths import RunPaths
from utils import set_env
from model_io import evaluation_model, checkpoint_schema
from training_scenarios import merge_validation


def validate_network(root, algorithm, checkpoint=None, network=None):
    root = Path(root)
    run = RunPaths(root, 'evaluate', algorithm)
    run.data.mkdir(parents=True, exist_ok=True)
    run.simulation.mkdir(parents=True, exist_ok=True)
    seed_policy(VALIDATION_PARAMS.POLICY_SEED)
    env = set_env(0, run, test=True, network=network or SUMO_PARAMS.NET_NAME,
                  schema=checkpoint_schema(checkpoint, algorithm) if checkpoint else None)
    env.max_test_step = VALIDATION_PARAMS.MAX_STEPS
    try:
        if checkpoint is None:
            from evaluator_non_rl import Evaluator
            evaluator = Evaluator(env, run)
        else:
            from evaluator_rl import Evaluator
            model = evaluation_model(env, algorithm, checkpoint)
            evaluator = Evaluator(env, model, run)
            evaluator.load_model(checkpoint, validation=True)
        for index, seed in enumerate(VALIDATION_PARAMS.SEEDS):
            evaluator.evaluate(index, seed)
        evaluator.output_eval_data()
        path = run.data / 'metrics.json'
        report = json.loads(path.read_text())
        report['stage'] = 'validation'
        report['seeds'] = VALIDATION_PARAMS.SEEDS
        report['policy_seed'] = VALIDATION_PARAMS.POLICY_SEED
        report['max_steps'] = VALIDATION_PARAMS.MAX_STEPS
        if checkpoint is not None:
            report['checkpoint_sha256'] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        return report
    finally:
        if env.simulator.isLoaded():
            env.simulator.close()


def validate(root, algorithm, checkpoint=None):
    if SCENARIO_PARAMS.MODE == 'single':
        return validate_network(root, algorithm, checkpoint)
    root = Path(root)
    reports = {item['name']: validate_network(root/item['name'], algorithm, checkpoint, item['name'])
               for item in SCENARIO_PARAMS.NETWORKS}
    report = merge_validation(reports)
    report.update(seeds=VALIDATION_PARAMS.SEEDS, policy_seed=VALIDATION_PARAMS.POLICY_SEED,
                  max_steps=VALIDATION_PARAMS.MAX_STEPS)
    if checkpoint:
        report['checkpoint_sha256'] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    (root/'data').mkdir(exist_ok=True)
    (root/'data/metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--algorithm', choices=['Unicorn', 'HeteroLight', 'MaxPressure'], required=True)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    if (args.algorithm == 'MaxPressure') != (args.checkpoint is None):
        parser.error('MaxPressure is the completion reference; learning algorithms require --checkpoint')
    validate(args.output, args.algorithm, args.checkpoint)
