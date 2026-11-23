"""The checkpoint format shared by training, sampling and evaluation."""
import os
from pathlib import Path
import random
import numpy as np
import torch
from parameters import INPUT_PARAMS, VALIDATION_PARAMS, NETWORK_PARAMS

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
    return {'format_version': FORMAT_VERSION, 'schema': model_schema(env, algorithm),
            'clipping': NETWORK_PARAMS.CLIPPING,
            'source_network': env.net_file_name, 'completed_updates': completed_updates,
            'validation_seeds': list(VALIDATION_PARAMS.SEEDS),
            'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


def load_weights(path, model, env, algorithm, device):
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported checkpoint format; use a checkpoint from the current training pipeline')
    expected = model_schema(env, algorithm)
    if checkpoint.get('schema') != expected:
        raise ValueError(f'Checkpoint input/action schema {checkpoint.get("schema")} differs from target {expected}; '
                         'changing intersection count is supported, changing movement/phase schema is not')
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
        return {'rng': capture_rng(), 'sumo_seed': self.env.seed,
                'completed_episode': self.curr_episode}

    def restore_sampling_state(self, state):
        self.env.seed = state['sumo_seed']
        self.curr_episode = state['completed_episode']
        restore_rng(state['rng'])
