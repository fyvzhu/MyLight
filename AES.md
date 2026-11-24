AES training switch
===================

`config.yaml` selects one shared Unicorn training pipeline:

```yaml
ppo_variant: disppo   # unicorn or disppo
clipping: kl2clip     # fixed_clipping, kl2clip or dppo_full_tv
aes: true
aes_config:
  q: 0.9
  ema_decay: 0.95
  kappa: 0.0016828572303074895
  minimum: 0.0004
  maximum: 0.008
```

The default is `aes: false`. It retains the fixed `EL_FACTOR=0.002`, the
selected PPO variant/clipping, original buffers and original numerical update.
No TD diagnostics or scheduler state are computed/saved on the disabled path.
Missing AES fields in old settings are interpreted as disabled/default settings.
The AES-enabled training entry is Unicorn; HeteroLight training rejects `aes: true`
explicitly. Evaluation of saved models does not require AES to be enabled.

| Parameter | Meaning | Allowed values |
|---|---|---|
| `aes` | Enable the training entropy scheduler | YAML `true` or `false` |
| `q` | Quantile of absolute one-step sampled TD | Number in `[0, 1]` |
| `ema_decay` | Weight on the previous TD EMA | Number in `[0, 1)` |
| `kappa` | Scale multiplying the square root of the historical EMA mean | Positive finite number |
| `minimum` | Lower entropy coefficient bound | Positive finite number, at most `maximum` |
| `maximum` | Upper entropy coefficient bound | Finite number, at least `minimum` |

The default scale/bounds match the independent 40-update Grid5x5 pilot and
subsequent A1 comparison, not the author's original task-specific hyperparameters.
They are configurable; no pilot or automatic scale search runs during formal training.
The earlier 200-update comparison did not establish a traffic performance benefit.
Compatibility with DisPPO and KL2Clip is not evidence of additive improvements.

The sampler reuses its existing Critic forward. With DisPPO, it first averages
the quantiles to obtain the scalar value. AES uses
`r[t] + gamma * V[t+1] - V[t]` over the exact original T-1 fragment, using the
last observed value as bootstrap. It does not change D-GAE or quantile targets.
No new VAE random draws are introduced. Named worker metadata carries TD
separately from the existing positional buffers.

The scheduler computes `Q_q(abs(TD))` once per fresh merged rollout batch.
Every actual Actor optimizer step advances its EMA and cumulative mean using
that batch statistic; the existing six Actor epochs thus produce six scheduler
steps. The resulting bounded coefficient replaces only the Actor's entropy
coefficient. PPO policy clipping, Critic updates, VAE and contrastive losses keep
their existing definitions. There is no TD scale normalization, local entropy
weight or change to the entropy definition/mask.

For enabled runs, `train_outputs/Unicorn/<run>/aes.jsonl` records each update's
TD quantile/EMA, coefficient mean/min/max, entropy, bound-hit fractions and
Actor step count. The same values appear under `AES/` in TensorBoard;
`TD_Quantile` corresponds to Q90 with the default `q=0.9`.

Checkpoints save the scheduler's config/version, EMA, accumulated proxy,
step counter and last coefficient alongside the existing optimizers and RNGs.
Resume freezes the AES flag and active config. Existing pre-AES checkpoints
remain resumable with AES disabled, provided the other original settings match.
Enabling/disabling AES or changing its active settings requires a new training
run; it cannot silently reset or replace the scheduler during a resume.
Saved weights retain their original architecture and can be evaluated with either
AES flag. AES adapts the training loss; it does not update frozen policies online.

Implementation lives in `aes.py`, `runner_unicorn.py`, `driver_unicorn.py` and
the shared `training.py`; formal modules never import `tests` or the downloaded
reference repository. Existing torch/numpy/PyYAML dependencies are sufficient.

Reference: [Tracking Drift](https://proceedings.mlr.press/v306/wang26jc.html),
local `参考代码/AES改进/AES-main/cleanrl/cleanrl/aes.py` (MIT).
The traffic adaptation retains the existing separate Actor/Critic training order.
It is not a claim to reproduce all of the reference paper's experiments.
