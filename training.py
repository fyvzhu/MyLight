"""Shared lifecycle: sample -> PPO update -> source validation -> atomic commit.

Drivers retain their network-specific PPO updates with configured policy clipping.
A checkpoint is committed only at
completed update boundaries, with root/worker RNGs and both optimizer states.
Resume uses --resume-run on the same driver and writes into the same run.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import ray
import torch
from torch.utils.tensorboard import SummaryWriter
import parameters as settings
from parameters import INPUT_PARAMS, TRAIN_PARAMS, NETWORK_PARAMS, SUMO_PARAMS, VALIDATION_PARAMS, EVALUATION_PARAMS, AES_PARAMS, ADVANCED_VAE_PARAMS
from evaluation import seed_policy
from early_stopping import initial_state, observe_validation
from convergence import prepare_reward_log, append_reward, finish as finish_convergence
from model_io import atomic_save, capture_rng, restore_rng, weights_snapshot, load_weights, scenario_metadata
from run_paths import RunPaths, PROJECT_ROOT, project_path, _public
from utils import set_env


SELECTION_RULE = 'trex_inspired_validation_att_v1'


def training_contract(algorithm):
    contract = {'algorithm': algorithm, 'max_episodes': INPUT_PARAMS.MAX_EPISODES,
            'selection_rule': SELECTION_RULE,
            'input': _public(INPUT_PARAMS), 'training': _public(TRAIN_PARAMS),
            'network': _public(NETWORK_PARAMS), 'sumo': _public(SUMO_PARAMS),
            'validation': _public(VALIDATION_PARAMS),
            'early_stopping': _public(settings.EARLY_STOPPING_PARAMS), 'test_seeds': EVALUATION_PARAMS.SEEDS}
    # Keep disabled contracts byte-for-byte equivalent as data, including for
    # existing pre-AES checkpoints. Enabling/changing AES requires a new run.
    if AES_PARAMS.ENABLED:
        from aes import VERSION
        contract['aes'] = {'version': VERSION, 'config': dict(AES_PARAMS.CONFIG)}
    if INPUT_PARAMS.CO_TRAIN:
        contract['scenarios'] = scenario_metadata()
    if ADVANCED_VAE_PARAMS.ENABLED:
        from advanced_vae import VERSION
        contract['advanced_vae']={'version':VERSION,'config':dict(ADVANCED_VAE_PARAMS.CONFIG)}
    return contract


def prepare_aes_log(path, completed):
    """Drop diagnostics beyond the latest committed checkpoint on recovery."""
    if not path.exists():
        return
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A crash may truncate the last uncommitted append.
        if row['completed_updates'] <= completed:
            rows.append(row)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
    os.replace(temporary, path)


def validation_score(report, seeds, networks):
    """Mean ATT for one frozen policy, with equal seed/network weights.

    Inspired by T-REX section 3.5's minimum mean travel-time criterion;
    fixed independent validation replaces its consecutive training window.
    Empty trips are invalid measurements, never a zero-ATT winner.
    """
    if 'networks' in report:
        current = report['networks']
        if not isinstance(current, dict) or not current or list(current) != networks:
            raise ValueError('Validation report training networks do not match the configured plan')
        scores, failures = [], []
        for name in current:
            score, reason = validation_score(current[name], seeds, [name])
            if score is None:
                failures.append(f'{name}: {reason}')
            else:
                scores.append(score)
        if failures:
            return None, '; '.join(failures)
        return sum(scores)/len(scores), None
    if len(networks) != 1 or report.get('network') != networks[0]:
        raise ValueError('Validation report does not cover the configured training networks')
    rows = report['episodes']
    if [r['seed'] for r in rows] != seeds or not rows:
        raise ValueError('Validation report seeds do not match the fixed validation seeds')
    failures = []
    for row in rows:
        count, att = row.get('completed_vehicles'), row.get('ATT')
        if (type(count) is not int or count <= 0 or type(att) not in (int, float)
                or not math.isfinite(att) or att <= 0):
            failures.append(row['seed'])
    if failures:
        return None, f'No completed trips or invalid ATT for validation seeds {failures}'
    return sum(r['ATT'] for r in rows) / len(rows), None


def run_validation(run, name, algorithm, checkpoint):
    output = run.root / 'validation' / name
    if output.exists():
        shutil.rmtree(output)  # Retry of this uncommitted validation only.
    args = [sys.executable, '-B', '-u', str(PROJECT_ROOT / 'validation_runner.py'),
            '--output', str(output), '--algorithm', algorithm, '--checkpoint', str(checkpoint)]
    # Separate process and CPU policy isolate all RNGs, SUMO and CUDA resources.
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'UNICORN_SETTINGS_JSON': json.dumps(settings._config)}
    subprocess.run(args, cwd=PROJECT_ROOT, env=env, check=True)
    return json.loads((output / 'data/metrics.json').read_text())


def write_selection(run, selection):
    # All authoritative history lives in the committed training checkpoint.
    # Materialized reports and best weights are rebuilt from it on recovery.
    directory = run.root / 'validation'
    directory.mkdir(exist_ok=True)
    report = {k: v for k, v in selection.items() if k != 'best_weights'}
    report['status'] = 'selected' if selection['best_weights'] else 'no_valid_model'
    temporary = directory / 'selection.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, directory / 'selection.json')
    best = run.model / 'best_model.pkl'
    if selection['best_weights'] is not None:
        atomic_save(selection['best_weights'], best)
    else:
        best.unlink(missing_ok=True)


def commit(run, payload):
    folder = run.model / 'checkpoints'
    path = folder / f"checkpoint_{payload['completed_updates']:08d}.pkl"
    atomic_save(payload, path)
    # Delete only after the new complete checkpoint is atomically visible.
    for old in sorted(folder.glob('checkpoint_????????.pkl'))[:-TRAIN_PARAMS.KEEP_CHECKPOINTS]:
        old.unlink()
    write_selection(run, payload['training_state']['selection'])
    print(f"[Checkpoint] committed update={payload['completed_updates']} -> {path}", flush=True)


def output_convergence(run, algorithm, completed, selection):
    metadata = {'algorithm': algorithm, 'run_name': run.root.name,
                'training_seed': TRAIN_PARAMS.SEED, 'training_mode': settings.SCENARIO_PARAMS.MODE,
                'networks': [item['name'] for item in settings.SCENARIO_PARAMS.NETWORKS],
                'worker_plan': [dict(item) for item in settings.SCENARIO_PARAMS.NETWORKS],
                'completed_updates': completed, 'budget_updates': INPUT_PARAMS.MAX_EPISODES,
                'validation_interval': VALIDATION_PARAMS.INTERVAL,
                'validation_seeds': list(VALIDATION_PARAMS.SEEDS),
                'validation_policy_seed': VALIDATION_PARAMS.POLICY_SEED,
                'validation_max_steps': VALIDATION_PARAMS.MAX_STEPS,
                'training_status': selection['training_status'], 'stop_reason': selection['stop_reason'],
                'ppo_variant': NETWORK_PARAMS.PPO_VARIANT if algorithm == 'Unicorn' else None,
                'clipping': NETWORK_PARAMS.CLIPPING, 'aes': AES_PARAMS.ENABLED,
                'advanced_vae': ADVANCED_VAE_PARAMS.ENABLED}
    report = finish_convergence(run.root, selection, settings.CONVERGENCE_PARAMS.CONFIG, metadata)
    print(f'[Convergence] {report["overall"]["status"]}; '
          f'estimated update={report["overall"]["estimated_convergence_update"]}; '
          f'see {run.root / "convergence/summary.json"}', flush=True)


def train(algorithm, Model, Runner, combine, update, log_metrics, resume_run=None):
    if ADVANCED_VAE_PARAMS.ENABLED and algorithm != 'Unicorn':
        raise ValueError('advanced_vae applies only to Unicorn')
    if AES_PARAMS.ENABLED and algorithm != 'Unicorn':
        raise ValueError('aes=true currently applies to the Unicorn driver only')
    if settings.JOB_TYPE != settings.JOB_OPTIONS.GET_EXPERIENCE or NETWORK_PARAMS.UPDATE_TYPE != 'PPO_MAC':
        raise ValueError('This training pipeline requires PPO_MAC experience sampling')
    seed_policy(TRAIN_PARAMS.SEED)
    contract = training_contract(algorithm)
    entropy_scheduler = None
    if AES_PARAMS.ENABLED:
        from aes import AdaptiveEntropy
        entropy_scheduler = AdaptiveEntropy(AES_PARAMS.CONFIG)
    checkpoint = None
    if resume_run:
        run = RunPaths(project_path(resume_run), 'train', algorithm)
        candidates = sorted((run.model / 'checkpoints').glob('checkpoint_????????.pkl'))
        if not candidates:
            raise FileNotFoundError('No complete training checkpoint in this run/models/checkpoints')
        checkpoint = torch.load(candidates[-1], map_location='cpu')
        if checkpoint.get('training_state', {}).get('contract') != contract:
            raise ValueError('Training/validation settings differ from the frozen checkpoint configuration')
        if entropy_scheduler is not None:
            entropy_scheduler.load_state_dict(checkpoint['training_state'].get('aes'))
            if entropy_scheduler.step != checkpoint['completed_updates'] * NETWORK_PARAMS.K_EPOCH:
                raise ValueError('AES clock differs from the committed Actor update count')
    else:
        run = RunPaths.create('train', algorithm)
        run.save_config()
    print('Training output:', run.root, flush=True)
    print('PPO clipping:', NETWORK_PARAMS.CLIPPING, flush=True)
    if INPUT_PARAMS.CO_TRAIN:
        print('Joint training:', scenario_metadata(), flush=True)
    if algorithm == 'Unicorn':
        print('PPO variant:', NETWORK_PARAMS.PPO_VARIANT, flush=True)
        if entropy_scheduler is not None:
            print('Adaptive entropy:', dict(AES_PARAMS.CONFIG), flush=True)
    device = torch.device('cuda') if TRAIN_PARAMS.USE_GPU else torch.device('cpu')
    env = set_env(None, run)
    model = Model(input_dim=env.tls_obs_space, agent_dim=env.tls_all_agent_space,
                  int_vec_dim=env.tls_int_attr_space, actor_lr=NETWORK_PARAMS.A_LR_Q,
                  critic_lr=NETWORK_PARAMS.C_LR_Q).to(device)
    advanced_vae = None
    if ADVANCED_VAE_PARAMS.ENABLED:
        from advanced_vae import AdvancedVAE
        advanced_vae=AdvancedVAE(model,ADVANCED_VAE_PARAMS.CONFIG)
        print('Advanced VAE:',dict(ADVANCED_VAE_PARAMS.CONFIG),flush=True)
    completed, window = 0, []
    if checkpoint is not None:
        load_weights(candidates[-1], model, env, algorithm, device)
        state = checkpoint['training_state']
        for optimizer, saved in zip((model.actor_optimizer, model.critic_optimizer), state['optimizers']):
            optimizer.load_state_dict(saved)
        completed, window = checkpoint['completed_updates'], state['summary_window']
        if advanced_vae is not None:
            advanced_vae.load_state_dict(state.get('advanced_vae'),NETWORK_PARAMS.K_EPOCH)
            if advanced_vae.completed_updates != completed:
                raise ValueError('Advanced VAE clock differs from committed updates')
        selection = state['selection']
        # Ignore/remove validation artifacts from work after the last commit.
        for child in (run.root / 'validation').glob('update_*'):
            suffix = child.name.removeprefix('update_')
            if child.is_dir() and suffix.isdigit() and int(suffix) > completed:
                shutil.rmtree(child)
        write_selection(run, selection)
        print(f'[Resume] completed updates={completed}; continuing in the same directory', flush=True)
    else:
        selection = {'rule': SELECTION_RULE, 'metric': VALIDATION_PARAMS.METRIC,
                     'best_score': None, 'best_update': None, 'best_weights': None, 'history': [],
                     'early_stopping': initial_state(), 'training_status': 'running',
                     'stop_reason': None, 'stopped_update': None}
    reward_path = run.root / 'convergence/training_rewards.jsonl'
    prepare_reward_log(reward_path, completed)
    if selection['training_status'] != 'running' or completed >= INPUT_PARAMS.MAX_EPISODES:
        output_convergence(run, algorithm, completed, selection)
        print(f'[Training] already finished ({selection["stop_reason"]}); '
              'no updates or validation repeated', flush=True)
        return run
    if entropy_scheduler is not None:
        prepare_aes_log(run.root / 'aes.jsonl', completed)
    if advanced_vae is not None:
        prepare_aes_log(run.root / 'advanced_vae.jsonl', completed)
    initial_rng = capture_rng()
    workers = []
    summary = None
    try:
        print('[Sampling runtime]', settings.SAMPLING_RUNTIME_ENV, flush=True)
        # Persistent actors each reserve one logical CPU. Declare the configured
        # capacity so an explicit Worker plan cannot wait forever for the last
        # actors when the container CPU quota is smaller than that plan.
        ray.init(num_cpus=TRAIN_PARAMS.NUM_META_AGENTS,
                 num_gpus=TRAIN_PARAMS.NUM_GPU if TRAIN_PARAMS.USE_GPU else 0,
                 runtime_env={'env_vars': {
                     'UNICORN_SETTINGS_JSON': json.dumps(settings._config),
                     **settings.SAMPLING_RUNTIME_ENV}})
        summary = SummaryWriter(str(run.tensorboard), purge_step=completed + 1 if checkpoint else None)
        workers = [Runner.remote(i, run) for i in range(TRAIN_PARAMS.NUM_META_AGENTS)]
        if checkpoint is not None:
            ray.get([worker.restore_sampling_state.remote(saved)
                     for worker, saved in zip(workers, state['workers'])])
            restore_rng(state['rng'])
        else:
            # Force all constructors to finish before capturing the initial states.
            ray.get([worker.sampling_state.remote() for worker in workers])
            # Ray service-port allocation consumes Python random numbers. Keep
            # framework startup outside the training RNG stream.
            restore_rng(initial_rng)
        sampling_path = run.model / 'sampling_weights.pth'

        def save_boundary():
            payload = weights_snapshot(model, env, algorithm, completed)
            payload['training_state'] = {
                'contract': contract, 'optimizers': [model.actor_optimizer.state_dict(), model.critic_optimizer.state_dict()],
                'rng': capture_rng(), 'workers': ray.get([w.sampling_state.remote() for w in workers]),
                'summary_window': window, 'selection': selection}
            if entropy_scheduler is not None:
                payload['training_state']['aes'] = entropy_scheduler.state_dict()
            if advanced_vae is not None:
                payload['training_state']['advanced_vae']=advanced_vae.state_dict()
            summary.flush()
            commit(run, payload)

        if checkpoint is None:
            save_boundary()
        while completed < INPUT_PARAMS.MAX_EPISODES:
            atomic_save(weights_snapshot(model, env, algorithm, completed), sampling_path)
            # Fixed worker order is important for reproducible shuffling on resume.
            jobs = ray.get([worker.job.remote(completed + 1) for worker in workers])
            buffers, metrics = combine(jobs)
            if INPUT_PARAMS.CO_TRAIN:
                # Keep network identities for diagnostics without changing the
                # original junction-weighted PPO/auxiliary objective.
                for item in settings.SCENARIO_PARAMS.NETWORKS:
                    name = item['name']
                    group = [j for j in jobs if j[2]['network'] == name]
                    if len(group) != item['workers']:
                        raise ValueError('Collected Worker/network mapping differs from the frozen plan')
                    summary.add_scalar(f'Training/{name}/Reward', sum(j[1][0] for j in group)/len(group), completed+1)
                    summary.add_scalar(f'Training/{name}/JunctionTransitions',
                                       sum(j[0][0][0].shape[0]*j[2]['agents'] for j in group), completed+1)
            if advanced_vae is not None:
                if entropy_scheduler is not None:
                    entropy_scheduler.begin_batch(torch.cat([job[2]['aes_td'] for job in jobs], dim=1))
                train_metrics=update(network=model,device=device,experience_buffers=buffers,
                    entropy_scheduler=entropy_scheduler,advanced_vae=advanced_vae)
            elif entropy_scheduler is not None:
                entropy_scheduler.begin_batch(torch.cat([job[2]['aes_td'] for job in jobs], dim=1))
                train_metrics = update(network=model, device=device, experience_buffers=buffers,
                                       entropy_scheduler=entropy_scheduler)
            else:
                train_metrics = update(network=model, device=device, experience_buffers=buffers)
            completed += 1
            network_rewards = {}
            for item in settings.SCENARIO_PARAMS.NETWORKS:
                group = ([job for job in jobs if job[2]['network'] == item['name']]
                         if INPUT_PARAMS.CO_TRAIN else jobs)
                network_rewards[item['name']] = sum(float(job[1][0]) for job in group) / len(group)
            append_reward(reward_path, completed, metrics[0], network_rewards)
            if advanced_vae is not None:
                diagnostics=advanced_vae.diagnostics()
                for name,value in diagnostics.items():
                    summary.add_scalar('AdvancedVAE/'+name,value,completed)
                with (run.root/'advanced_vae.jsonl').open('a') as stream:
                    stream.write(json.dumps({'completed_updates':completed,**diagnostics},allow_nan=False)+'\n')
            if entropy_scheduler is not None:
                diagnostics = entropy_scheduler.diagnostics(NETWORK_PARAMS.K_EPOCH)
                for name, value in diagnostics.items():
                    summary.add_scalar('AES/' + name, value, completed)
                with (run.root / 'aes.jsonl').open('a') as stream:
                    stream.write(json.dumps({'completed_updates': completed, **diagnostics}, allow_nan=False) + '\n')
            window.append(list(train_metrics) + list(metrics))
            print(f'[Training] completed update {completed}/{INPUT_PARAMS.MAX_EPISODES}', flush=True)
            if len(window) >= TRAIN_PARAMS.SUMMARY_WINDOW or completed == INPUT_PARAMS.MAX_EPISODES:
                log_metrics(summary, window, completed)
                window = []
            if TRAIN_PARAMS.RESET_OPTIM and completed % TRAIN_PARAMS.RESET_OPTIM_STEP == 0:
                model.reset_optimizer()
                if advanced_vae is not None:advanced_vae.attach_optimizers()
            validate_now = completed % VALIDATION_PARAMS.INTERVAL == 0 or completed == INPUT_PARAMS.MAX_EPISODES
            if validate_now:
                # Validate exactly the weights produced by this update.
                candidate = weights_snapshot(model, env, algorithm, completed)
                atomic_save(candidate, sampling_path)
                report = run_validation(run, f'update_{completed:08d}', algorithm, sampling_path)
                score, reason = validation_score(report, VALIDATION_PARAMS.SEEDS,
                    [item['name'] for item in settings.SCENARIO_PARAMS.NETWORKS])
                improved = score is not None and (selection['best_score'] is None or score < selection['best_score'])
                if improved:
                    selection['best_score'], selection['best_update'] = score, completed
                    candidate['selection'] = {'rule': SELECTION_RULE,
                                              'metric': VALIDATION_PARAMS.METRIC, 'score': score,
                                              'validation_seeds': VALIDATION_PARAMS.SEEDS,
                                              'aggregation': 'equal weight per network mean over validation seeds'}
                    selection['best_weights'] = candidate
                stop_now = observe_validation(selection['early_stopping'], completed, score,
                                              selection['best_score'], settings._config['early_stopping'])
                selection['history'].append({'completed_updates': completed, 'valid': score is not None,
                                             'score': score, 'reason': reason, 'new_best': improved, 'report': report,
                                             'early_stopping': dict(selection['early_stopping'])})
                if stop_now:
                    selection.update(training_status='early_stopped',
                                     stop_reason='no_significant_validation_improvement', stopped_update=completed)
                summary.add_scalar('EarlyStopping/Bad Validations', selection['early_stopping']['bad_validations'], completed)
                for metric in ('ATT', 'Queue Length', 'Trip Completion Rate'):
                    value = report['aggregate'][metric]['mean']
                    if value is not None and math.isfinite(value):
                        summary.add_scalar('Validation/' + metric, value, completed)
                for name, part in report.get('networks', {}).items():
                    for metric in ('ATT', 'Queue Length', 'Trip Completion Rate'):
                        value = part['aggregate'][metric]['mean']
                        if value is not None and math.isfinite(value):
                            summary.add_scalar(f'Validation/{name}/{metric}', value, completed)
                summary.add_scalar('Validation/Valid', int(score is not None), completed)
                print(f'[Validation] update={completed} valid={score is not None} ATT={score} '
                      f'best_update={selection["best_update"]} reason={reason}', flush=True)
            if completed >= INPUT_PARAMS.MAX_EPISODES and selection['training_status'] == 'running':
                selection.update(training_status='budget_exhausted', stopped_update=completed,
                                 stop_reason='max_episodes' if selection['best_score'] is not None else 'no_valid_model')
            if selection['training_status'] == 'early_stopped' and window:
                # An early stop may fall between TensorBoard summary windows.
                # Flush the final partial window before committing the finished state.
                log_metrics(summary, window, completed)
                window = []
            if validate_now or completed % TRAIN_PARAMS.CHECKPOINT_INTERVAL == 0:
                save_boundary()
            del buffers, jobs
            if selection['training_status'] == 'early_stopped':
                print(f'[Early stopping] update={completed}; '
                      f'consecutive validations without significant improvement={selection["early_stopping"]["bad_validations"]}; '
                      f'best_update={selection["best_update"]}', flush=True)
                break
        output_convergence(run, algorithm, completed, selection)
        print(f'[Training] finished ({selection["stop_reason"]}); best update={selection["best_update"]}; '
              f'see {run.root / "validation/selection.json"}', flush=True)
        return run
    except KeyboardInterrupt:
        print('[Training] interrupted; resume from the latest committed checkpoint with --resume-run', flush=True)
        raise
    finally:
        for worker in workers:
            ray.kill(worker)
        if summary is not None:
            summary.close()
        if ray.is_initialized():
            ray.shutdown()


def entrypoint(algorithm, Model, Runner, combine, update, log_metrics):
    parser = argparse.ArgumentParser(description=f'{algorithm}: training, source validation and best-model selection')
    parser.add_argument('--resume-run', help='Continue the same run from its latest complete training checkpoint')
    args = parser.parse_args()
    train(algorithm, Model, Runner, combine, update, log_metrics, args.resume_run)
