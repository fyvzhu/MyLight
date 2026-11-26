# Training convergence analysis

Both `driver_unicorn.py` and `driver_heterolight.py` use the shared lifecycle in
`training.py`. At normal completion or early stopping they export:

```
train_outputs/<algorithm>/<run>/convergence/
  training_rewards.jsonl      # per-update overall/per-network training rewards
  learning_data.json          # self-contained source data, settings and provenance
  summary.json                # overall and per-network empirical convergence
  learning_curve.png          # ATT, sampled Reward, Queue Length and arrival flow
  learning_curve_<network>.png # additional per-network figures in joint mode
```

The reference is Nguyen et al., *A framework for benchmarking traffic signal
control robustness under incidents: comparative study of reinforcement
learning-based methods*, European Transport Research Review (2026), section 3.4,
<https://doi.org/10.1186/s12544-026-00841-1>.
This is an adaptation: T-REX analyzes training-episode traffic measurements;
this project analyzes each frozen candidate on independent, fixed validation
traffic seeds. Historical best-so-far scores and final test scores are not used.
Figures 3 and 4 of T-REX include an environment transfer; these plots do not add
an unrequested transfer/incident experiment.

## Default protocol

`validation.interval: 100` gives about 25 observations in a 2500-update run.
Changing from 200 to 100 increases validation work; it does not change the PPO
gradient rule. A PPO update includes all sampling Workers and is not an
individual Worker episode, gradient epoch, or SUMO second.

```yaml
convergence:
  smoothing_window: 3
  final_window: 5
  relative_tolerance: 0.05
  minimum_stable_updates: 500
```

For each network, compute ATT by equally averaging its fixed validation seeds.
The overall curve gives each network mean equal weight, irrespective of its
Worker/intersection count. Training Reward retains the existing mean over
Workers, with separate per-network means; it is an auxiliary diagnostic and is
not substituted for validation ATT.

1. Smooth ATT using trailing **full** windows of three validation observations.
   The first two points have no smoothed value. Invalid observations are never
   interpolated or removed to connect a seemingly stable curve.
2. Define a terminal reference as the mean of the last five smoothed observations.
   Its tolerance band is the reference multiplied by `[0.95, 1.05]`.
3. Find the earliest observed update whose subsequent smoothed observations all
   stay inside this band **through the end of the recorded run**, with at least
   500 real PPO updates between that point and the end. `confirmation_update`
   is the first observation at least 500 updates after the candidate. The full
   suffix must still pass, even after that confirmation point.
4. Report `estimated_convergence_update` as that observed update. Do not
   interpolate an exact step between validation points. If the series is too
   short, contains invalid ATT/empty trips, or has no sustained stable suffix,
   report `insufficient_data`, `invalid_measurements`, or `not_converged`, with
   a null convergence update. An unfinished run is `incomplete_training`.

The three/five-observation windows, 5% band and 500-update confirmation span are
declared project defaults, **not T-REX's original numeric settings**. Fix them
before comparing methods and inspect sensitivity if conclusions depend on them.
With default spacing, at least eight validation observations are needed even
for a perfectly flat series to pass both full windows and the 500-update span.
Sparse historical runs must not be interpreted at 100-update resolution.

Following T-REX's formulas, `FPD = abs(final - best) / best`, using the last and
minimum **smoothed** ATT, and `CR = sign(initial - final) / Tc`. Here `initial`
is the first complete smoothed validation observation; its actual update is
recorded and it is **not an untrained step-zero measurement**. The terminal
reference mean determines the band, while the final smoothed endpoint determines
FPD and the sign of CR. FPD is a ratio (0.1 means 10%). CR is null without a
confirmed Tc; it is not replaced by zero. Stable degraded performance may have a
negative CR. Stability alone does not imply good traffic control or an optimum.

Only one training seed is available inside a run. No cross-training-seed standard
deviation is invented from the validation traffic seeds. The green shaded area
is the **tolerance band**, not a confidence interval. For a thesis, compare
multiple independent training seeds under the same budget and validation rules.
Arrival flow is shown because ATT covers completed trips only.

## Regenerate without checkpoints or simulation

Keep `learning_data.json`. It includes the actual validation update indices,
per-network/per-seed ATT, queue and flow measurements, completed-vehicle counts,
per-update training rewards, run metadata and the complete analysis settings.
This one file is sufficient to recreate `summary.json` and all PNGs:

```bash
python convergence.py --input train_outputs/Unicorn/<run>/convergence/learning_data.json
```

An optional `--output /path/to/figures` writes recreated results elsewhere.
The command does not import the simulator, current `config.yaml`, TensorBoard
events, model checkpoints, or `tests`. It needs only the standard library and
the already installed Matplotlib. PNG output is headless and deterministic for
the same input/code/plotting environment.

The reward journal is truncated to the latest committed update when resuming,
so interrupted appends do not create duplicate/future observations. The final
base data is atomically saved before rendering. If rendering fails after the
last training checkpoint, fix the plotting issue and use the regeneration
command; training does not need to be repeated. Resuming an already finished
compatible run also recreates missing convergence exports without new training.

This analysis never changes `max_episodes`, best-model selection, PPO losses or
early stopping. `Tc` is retrospective and depends on the observed budget and
chosen thresholds. Use several pilot seeds to inform a common future budget;
do not assume `max_episodes = 1.25 * Tc` is a general rule. Joint-mode conclusions
must inspect each network as well as the macro-average.
