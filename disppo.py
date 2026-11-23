"""DisPPO's distributional value module for Unicorn's existing PPO pipeline.

Adapted from DisPPO's compute_distributional_gae_advantage_return and
compute_quantile_value_loss (verl/trainer/ppo/core_algos.py). We use junction
returns, unbounded negative traffic rewards and an explicit fragment bootstrap,
rather than LLM token masks or bounded [0, 1] values. The Actor, entropy, VAE,
contrastive objectives and policy clipping stay in the shared Unicorn driver.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def validate_config(config):
    keys = {'num_quantiles', 'gae_lambda', 'huber_kappa',
            'initial_quantile_spacing', 'value_clip'}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError(f'disppo must contain {sorted(keys)}')
    n = config['num_quantiles']
    if type(n) is not int or n < 3 or n % 2 != 1:
        raise ValueError('disppo.num_quantiles must be an odd integer >= 3')
    lam = config['gae_lambda']
    if type(lam) not in (int, float) or not 0 <= lam <= 1:
        raise ValueError('disppo.gae_lambda must be in [0, 1]')
    for key in ('huber_kappa', 'initial_quantile_spacing', 'value_clip'):
        value = config[key]
        if key == 'value_clip' and value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'disppo.{key} must be a finite positive number'
                             + (' or null' if key == 'value_clip' else ''))
    return dict(config)


class OrderedQuantileOffsets(nn.Module):
    """Ordered, zero-mean junction offsets around Unicorn's summed value."""
    def __init__(self, features, num_quantiles, initial_spacing):
        super().__init__()
        self.delta = nn.Linear(features, num_quantiles - 1)
        self.half = num_quantiles // 2
        nn.init.zeros_(self.delta.weight)
        # Stable inverse softplus, including large configured spacings.
        nn.init.constant_(self.delta.bias,
                          initial_spacing + math.log(-math.expm1(-initial_spacing)))

    def forward(self, features):
        delta = F.softplus(self.delta(features))
        left = delta[..., :self.half].cumsum(-1).flip(-1)
        right = delta[..., self.half:].cumsum(-1)
        offsets = torch.cat((-left, torch.zeros_like(delta[..., :1]), right), dim=-1)
        # A common translation preserves ordering. The original scalar branch
        # remains the mean for asymmetric learned spacings, not only at init.
        return offsets - offsets.mean(-1, keepdim=True)


def distributional_gae(rewards, quantiles, bootstrap, gamma, lam, terminated=None):
    """T x agents x quantiles TD(lambda), with a separate terminal mask.

    A fragment/time limit bootstraps from the next observed state. Only true
    terminals zero continuation. Whitening is done once after merging workers,
    so it uses the same global batch as the policy update.
    """
    if quantiles.ndim != 3 or rewards.shape != quantiles.shape[:-1]:
        raise ValueError('D-GAE expects rewards [time, agents], quantiles [time, agents, N]')
    if bootstrap.shape != quantiles.shape[1:] or not len(rewards):
        raise ValueError('D-GAE needs a nonempty fragment and bootstrap [agents, N]')
    if terminated is None:
        terminated = torch.zeros_like(rewards, dtype=torch.bool)
    if terminated.shape != rewards.shape:
        raise ValueError('D-GAE terminal mask must match rewards')
    with torch.no_grad():
        tail = torch.zeros_like(bootstrap)
        next_q = bootstrap
        targets = []
        for t in reversed(range(len(rewards))):
            continuation = (~terminated[t].bool()).to(quantiles.dtype).unsqueeze(-1)
            delta = rewards[t].unsqueeze(-1) + gamma * continuation * next_q - quantiles[t]
            tail = delta + gamma * lam * continuation * tail
            targets.append(quantiles[t] + tail)
            next_q = quantiles[t]
        targets = torch.stack(targets[::-1])
        advantages = targets.mean(-1) - quantiles.mean(-1)
    return advantages, targets


def quantile_huber_loss(predictions, targets, old_quantiles, kappa=1.0, value_clip=None):
    """Author pairwise quantile loss (including its outer 0.5 factor).

    Average all predicted/target quantile pairs, then time and junctions. Optional
    PPO value clipping takes the maximum of unclipped/clipped pair losses.
    """
    if predictions.shape != targets.shape or predictions.shape != old_quantiles.shape:
        raise ValueError('Quantile predictions, targets and old values must have the same shape')
    n = predictions.shape[-1]
    tau = (torch.arange(n, device=predictions.device, dtype=predictions.dtype) + .5) / n
    tau = tau.reshape(*([1] * (predictions.ndim - 1)), n, 1)
    targets = targets.detach().unsqueeze(-2)

    def pairs(values):
        error = targets - values.unsqueeze(-1)
        huber = torch.where(error.abs() <= kappa, .5 * error.square(),
                            kappa * (error.abs() - .5 * kappa))
        weight = (tau - (error.detach() < 0).to(predictions.dtype)).abs()
        return weight * huber

    loss = pairs(predictions)
    if value_clip is not None:
        old = old_quantiles.detach()
        clipped = torch.maximum(torch.minimum(predictions, old + value_clip), old - value_clip)
        loss = torch.maximum(loss, pairs(clipped))
    return .5 * loss.mean()
