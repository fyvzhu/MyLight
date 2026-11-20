"""Shared settings and RNG initialization for the two evaluation entry points."""
import random
import numpy as np
import torch
from parameters import EVALUATION_PARAMS
from run_paths import LEARNING_ALGORITHMS, RULE_ALGORITHMS, project_path


def evaluation_settings(learning):
    algorithm = EVALUATION_PARAMS.ALGORITHM
    allowed = LEARNING_ALGORITHMS if learning else RULE_ALGORITHMS
    if algorithm not in allowed:
        raise ValueError(f'This entry point evaluates {allowed}; set evaluation.algorithm accordingly')
    checkpoint = EVALUATION_PARAMS.CHECKPOINT
    if learning:
        if not checkpoint:
            raise ValueError('Set evaluation.checkpoint to a saved training checkpoint file')
        checkpoint = project_path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Evaluation checkpoint not found: {checkpoint}')
    elif checkpoint is not None:
        raise ValueError('Set evaluation.checkpoint to null for a rule-based algorithm')
    return algorithm, checkpoint


def seed_policy(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
