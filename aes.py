"""Global Adaptive Entropy Scheduling for the shared Unicorn PPO update.

Formula reference: local AES-main/cleanrl/cleanrl/aes.py, Tracking Drift,
https://proceedings.mlr.press/v306/wang26jc.html (MIT, 2026 Tracking Drift contributors).
Q_q(|one-step TD|) -> EMA -> sqrt(prefix mean) -> bounded coefficient.
As in the paired SUMO study, each real Actor optimizer step advances the
scheduler; all Actor epochs in a batch share its sampled rollout TD statistic.
No historical TD normalization, local entropy weights or extra Critic forward.

Reference scheduler license notice:
MIT License, Copyright (c) 2026 Tracking Drift contributors
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math
from statistics import mean

import torch


# Scale calibrated on the independent 40-update traffic pilot, not test scores.
DEFAULT_CONFIG = {'q': 0.9, 'ema_decay': 0.95, 'kappa': 0.0016828572303074895,
                  'minimum': 0.0004, 'maximum': 0.008}
VERSION = 'global_rollout_td_actor_step_v1'


def validate_config(config):
    if not isinstance(config, dict) or set(config) != set(DEFAULT_CONFIG):
        raise ValueError(f'aes_config must contain {sorted(DEFAULT_CONFIG)}')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in config.values()):
        raise ValueError('aes_config values must be finite numbers')
    if (not 0 <= config['q'] <= 1 or not 0 <= config['ema_decay'] < 1
            or config['kappa'] <= 0 or not 0 < config['minimum'] <= config['maximum']):
        raise ValueError('Invalid AES quantile, EMA decay, scale or coefficient bounds')
    return dict(config)


def rollout_td(rewards, values, gamma):
    """Sampled [T, junctions, 1/N] values -> [T-1, junctions] mean-value TD.

    Uses the original Tth observed-state bootstrap and T-1 training fragment.
    Does not use GAE, normalized advantages, quantile spread or value-loss error.
    """
    with torch.no_grad():
        value = values.detach().mean(-1)
        rewards = torch.as_tensor(rewards, dtype=value.dtype, device=value.device)
        if rewards.shape != value.shape or value.ndim != 2 or len(value) < 2:
            raise ValueError('AES requires matching sampled rewards/values [T>=2, junctions]')
        return rewards[:-1] + gamma * value[1:] - value[:-1]


class AdaptiveEntropy:
    def __init__(self, config):
        self.config = validate_config(config)
        self.ema = None
        self.accumulated = 0.0
        self.step = 0
        self.value = self.config['minimum']
        self.batch = None

    def begin_batch(self, td):
        td = td.detach().float().cpu()
        if td.ndim != 2 or not td.numel() or not torch.isfinite(td).all():
            raise ValueError('AES needs finite nonempty rollout TD [time, junctions]')
        self.batch = {'td_quantile': float(torch.quantile(td.abs().flatten(), self.config['q'])),
                      'td_abs_mean': float(td.abs().mean()),
                      'td_rms': float(td.square().mean().sqrt()),
                      'coefficients': [], 'policy_entropies': []}

    def coefficient(self, entropy):
        if self.batch is None:
            raise RuntimeError('AES begin_batch() must precede the Actor update')
        raw = self.batch['td_quantile']
        decay = self.config['ema_decay']
        self.ema = raw if self.ema is None else decay * self.ema + (1 - decay) * raw
        self.step += 1
        self.accumulated += self.ema
        scheduled = self.config['kappa'] * math.sqrt(self.accumulated / self.step)
        self.value = min(max(scheduled, self.config['minimum']), self.config['maximum'])
        self.batch['coefficients'].append(self.value)
        self.batch['policy_entropies'].append(float(entropy.detach()))
        return self.value

    def diagnostics(self, expected_steps):
        if self.batch is None or len(self.batch['coefficients']) != expected_steps:
            raise RuntimeError('AES must advance exactly once per Actor optimizer step')
        coefficients = self.batch['coefficients']
        return {'TD_Quantile': self.batch['td_quantile'], 'TD_QuantileLevel': self.config['q'],
                'TD_AbsMean': self.batch['td_abs_mean'],
                'TD_RMS': self.batch['td_rms'], 'TD_EMA': self.ema,
                'EntropyCoef': mean(coefficients), 'EntropyCoefLast': self.value,
                'EntropyCoefMin': min(coefficients), 'EntropyCoefMax': max(coefficients),
                'PolicyEntropy': mean(self.batch['policy_entropies']),
                'LowerBoundFraction': mean(float(v == self.config['minimum']) for v in coefficients),
                'UpperBoundFraction': mean(float(v == self.config['maximum']) for v in coefficients),
                'ActorSteps': self.step}

    def state_dict(self):
        return {'version': VERSION, 'config': dict(self.config), 'ema': self.ema,
                'accumulated': self.accumulated, 'step': self.step, 'value': self.value}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or set(state) != set(self.state_dict())
                or state['version'] != VERSION or state['config'] != self.config):
            raise ValueError('AES checkpoint configuration differs or state is incomplete')
        step, ema, total, value = (state[k] for k in ('step', 'ema', 'accumulated', 'value'))
        if (type(step) is not int or step < 0
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in (total, value))
                or total < 0 or not self.config['minimum'] <= value <= self.config['maximum']
                or (step == 0 and (ema is not None or total != 0 or value != self.config['minimum']))
                or (step > 0 and (type(ema) not in (float, int) or not math.isfinite(ema) or ema < 0))):
            raise ValueError('Invalid AES checkpoint counters/statistics')
        self.ema, self.accumulated, self.step, self.value = ema, total, step, value
        self.batch = None
