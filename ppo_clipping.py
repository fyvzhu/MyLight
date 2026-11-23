"""Policy surrogates for the shared PPO training flow.

KL2Clip uses per-sampled-action extrema under KL(old || new); the categorical
constraint reduces to Bernoulli KL at its extrema. DPPO-Full-TV is the tested
Unicorn adaptation of the directional gate in sail-sg/Stable-RL: measure TV
over every legal phase and block only movement further from the old policy.
Neither surrogate imposes a hard bound on the final network's divergence.

DPPO reference: https://github.com/sail-sg/Stable-RL, commit
c6af1d119c0fe30d16dd9a837d35fdf695acd226, verl/trainer/ppo/core_algos.py.
The full categorical TV gate is the small-action-space adaptation tested here,
rather than the author's selected-token binary-distribution approximation.
"""
import math

import torch


# Same budgets as the paired SUMO comparison. At p=0.25 the KL upper ratio is 1.2.
KL_BUDGET = .25 * math.log(.25 / .30) + .75 * math.log(.75 / .70)
TV_RADIUS = .05


def categorical_bounds(probabilities, budget=KL_BUDGET):
    """Per-action lower/upper q(a)/p(a) under categorical KL(p || q)."""
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('Positive finite KL budget required')
    p = probabilities.detach().double()
    if not torch.isfinite(p).all() or (p <= 0).any() or (p > 1).any():
        raise ValueError('Expected probabilities of sampled legal actions in (0,1]')
    deterministic = p >= 1 - 1e-10
    p = p.clamp(1e-12, 1 - 1e-12)

    def divergence(q):
        return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))

    left, right = torch.full_like(p, 1e-15), p.clone()
    for _ in range(60):
        middle = (left + right) / 2
        outside = divergence(middle) > budget
        left, right = torch.where(outside, middle, left), torch.where(outside, right, middle)
    lower = (left + right) / (2 * p)
    left, right = p.clone(), torch.full_like(p, 1 - 1e-15)
    for _ in range(60):
        middle = (left + right) / 2
        outside = divergence(middle) > budget
        left, right = torch.where(outside, left, middle), torch.where(outside, middle, right)
    upper = (left + right) / (2 * p)
    lower = torch.where(deterministic, torch.ones_like(lower), lower)
    upper = torch.where(deterministic, torch.ones_like(upper), upper)
    return lower.to(probabilities.dtype), upper.to(probabilities.dtype)


class PolicyClipping:
    """Prepare one rollout batch; reuse its frozen old policy across PPO epochs."""

    def __init__(self, method, old_logp, old_policy=None, eps_clip=.2):
        self.method = method
        self.old_policy = None
        if method == 'fixed_clipping':
            self.lower, self.upper = 1.0 - eps_clip, 1.0 + eps_clip
        elif method == 'kl2clip':
            self.lower, self.upper = categorical_bounds(old_logp.exp())
        elif method == 'dppo_full_tv':
            if old_policy is None:
                raise ValueError('dppo_full_tv requires the full policy recorded during sampling')
            self.old_policy = old_policy.detach().to(old_logp.device)
            if (self.old_policy.shape[:-1] != old_logp.shape
                    or not torch.isfinite(self.old_policy).all()
                    or (self.old_policy < 0).any()
                    or not torch.allclose(self.old_policy.sum(-1), torch.ones_like(old_logp), atol=1e-5)):
                raise ValueError('Invalid sampled full-policy probabilities')
        else:
            raise ValueError(f'Unknown PPO clipping method: {method}')

    def loss(self, policy, ratio, advantages):
        """Return the actor surrogate and its clipping/gating diagnostic mask."""
        if self.method == 'dppo_full_tv':
            if policy.shape != self.old_policy.shape:
                raise ValueError('Current and sampled full policies must have the same shape')
            # Masked phases have zero mass in both stored and current policies.
            tv = .5 * (policy.detach() - self.old_policy).abs().sum(-1)
            outward = ((advantages > 0) & (ratio.detach() > 1)) | ((advantages < 0) & (ratio.detach() < 1))
            blocked = (tv > TV_RADIUS) & outward
            loss = -((~blocked).to(ratio.dtype) * ratio * advantages).mean()
        else:
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, self.lower, self.upper) * advantages
            loss = -1 * torch.min(surr1, surr2).mean()
            blocked = ratio.detach().gt(self.upper) | ratio.detach().lt(self.lower)
        return loss, blocked
