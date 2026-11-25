"""The checkpoint format shared by training, sampling and evaluation."""
import os
from pathlib import Path
import random
import numpy as np
import torch
from parameters import INPUT_PARAMS, VALIDATION_PARAMS, NETWORK_PARAMS, AES_PARAMS, SCENARIO_PARAMS, ADVANCED_VAE_PARAMS

FORMAT_VERSION = 1


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def model_schema(env, algorithm):
    # Intersection count is deliberately absent: shared weights can be applied
    # to a different number of intersections with the same input/action schema.
    return {'algorithm': algorithm, 'observation': INPUT_PARAMS.OBS_TYPE,
            'input_dim': [int(v) for v in env.tls_obs_space],
            'int_vec_dim': int(env.tls_int_attr_space),
            'max_phases': int(env.tls_max_phase_dim)}


def weights_snapshot(model, env, algorithm, completed_updates):
    payload = {'format_version': FORMAT_VERSION, 'schema': model_schema(env, algorithm),
            'clipping': NETWORK_PARAMS.CLIPPING,
            'advantage_normalization': NETWORK_PARAMS.ADVANTAGE_NORMALIZATION,
            'source_network': env.net_file_name, 'completed_updates': completed_updates,
            'validation_seeds': list(VALIDATION_PARAMS.SEEDS),
            'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    if INPUT_PARAMS.CO_TRAIN:
        payload['source_network'] = None
        payload['scenarios'] = scenario_metadata()
    if algorithm == 'Unicorn':
        payload['ppo'] = ppo_metadata(model)
        if AES_PARAMS.ENABLED:
            from aes import VERSION
            payload['aes'] = {'version': VERSION, 'config': dict(AES_PARAMS.CONFIG)}
        if ADVANCED_VAE_PARAMS.ENABLED:
            from advanced_vae import VERSION
            payload['advanced_vae']={'version':VERSION,'config':dict(ADVANCED_VAE_PARAMS.CONFIG)}
    return payload


def scenario_metadata():
    return {'mode': SCENARIO_PARAMS.MODE,
            'networks': [dict(r) for r in SCENARIO_PARAMS.NETWORKS],
            'worker_networks': list(SCENARIO_PARAMS.WORKER_NETWORKS),
            'layout': SCENARIO_PARAMS.LAYOUT}


def checkpoint_schema(path, algorithm):
    """Obtain saved input padding before constructing an evaluation environment."""
    saved = torch.load(path, map_location='cpu')
    schema = saved.get('schema')
    if (saved.get('format_version') != FORMAT_VERSION or not isinstance(schema, dict)
            or schema.get('algorithm') != algorithm or schema.get('observation') != INPUT_PARAMS.OBS_TYPE):
        raise ValueError('Unsupported/incompatible checkpoint schema')
    return schema


def ppo_metadata(model):
    spec = {'variant': model.ppo_variant, 'disppo': model.disppo_config}
    if model.ppo_variant == 'disppo':
        spec['value_head'] = 'phase_sum_zero_mean_offsets'
    return spec


def evaluation_model(env, algorithm, checkpoint):
    """Create the saved architecture, independently of the next training config."""
    options = {}
    if algorithm == 'Unicorn':
        from models.Unicorn import Unicorn as Model
        saved = torch.load(checkpoint, map_location='cpu')
        # Original Unicorn checkpoints predate the optional distributional module.
        spec = saved.get('ppo', {'variant': 'unicorn', 'disppo': None})
        options = {'ppo_variant': spec['variant'], 'disppo_config': spec['disppo']}
        if spec['variant'] == 'disppo' and spec['disppo'] is None:
            raise ValueError('DisPPO checkpoint is missing its distributional configuration')
    elif algorithm == 'HeteroLight':
        from models.HeteroLight import HeteroLight as Model
    else:
        raise ValueError(f'No learning model for {algorithm}')
    return Model(input_dim=env.tls_obs_space, agent_dim=env.tls_agent_space,
                 int_vec_dim=env.tls_int_attr_space, actor_lr=1e-4, critic_lr=1e-4, **options)


def load_weights(path, model, env, algorithm, device):
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported checkpoint format; use a checkpoint from the current training pipeline')
    expected = model_schema(env, algorithm)
    if checkpoint.get('schema') != expected:
        raise ValueError(f'Checkpoint input/action schema {checkpoint.get("schema")} differs from target {expected}; '
                         'changing intersection count is supported, changing movement/phase schema is not')
    if algorithm == 'Unicorn':
        saved_ppo = checkpoint.get('ppo', {'variant': 'unicorn', 'disppo': None})
        if saved_ppo != ppo_metadata(model):
            raise ValueError(f'Checkpoint PPO module {saved_ppo} differs from model {ppo_metadata(model)}')
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


class SamplingState:
    """Only called at completed rollout boundaries, when SUMO has closed."""
    def sampling_state(self):
        state = {'rng': capture_rng(), 'sumo_seed': self.env.seed,
                 'completed_episode': self.curr_episode}
        if INPUT_PARAMS.CO_TRAIN:
            state['network'] = self.env.net_file_name
        return state

    def restore_sampling_state(self, state):
        if INPUT_PARAMS.CO_TRAIN and state.get('network') != self.env.net_file_name:
            raise ValueError('Restored Worker belongs to a different training network')
        self.env.seed = state['sumo_seed']
        self.curr_episode = state['completed_episode']
        restore_rng(state['rng'])
