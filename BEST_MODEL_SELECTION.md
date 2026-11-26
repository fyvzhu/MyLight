# Best-model selection

The sole selection rule is `trex_inspired_validation_att_v1`: minimize fixed
validation mean travel time (ATT). It is shared by Unicorn and HeteroLight,
single/joint training, and all Unicorn training modules.

## Reference and adaptation

Nguyen et al. (2026), *A framework for benchmarking traffic signal control
robustness under incidents: comparative study of reinforcement learning-based
methods*, European Transport Research Review, 18:85.
https://doi.org/10.1186/s12544-026-00841-1

Section 3.5, Experiment 2 selects the best policy using the lowest mean travel
time over ten consecutive training episodes. Here we borrow its travel-time
selection objective, but evaluate **one frozen checkpoint** on fixed independent
validation traffic seeds. We do not implement the changing-policy ten-episode
training window or claim an exact reproduction of T-REX. The downloaded T-REX
implementation is a reference only; production imports none of its code.

Suggested methods description:

> Inspired by T-REX's travel-time-based policy selection, we periodically evaluate
> frozen policies on fixed validation demand seeds and retain the checkpoint
> with the lowest mean validation travel time. In joint training, validation
> scores are averaged equally across training networks. Held-out test seeds and
> networks are excluded from model selection.

## Configuration and score

```yaml
validation:
  interval: 100
  seeds: [21001, 21002, 21003, 21004, 21005]
  policy_seed: 2026
  max_steps: 3600
  metric: ATT
```

`metric` accepts only `ATT`; it is not a switch between selection mechanisms.
The seeds remain the existing five independent validation seeds, not ten
changing training episodes. Validation also runs at the final training update.
Changing seed count or horizon changes the validation budget and should be fixed
before comparing methods.

For each seed, ATT is the arithmetic mean of completed vehicles' trip durations
in seconds, from actual network departure to arrival. Seed scores receive equal
weight even when completed-vehicle counts differ. Joint training then assigns
equal weight to each network's seed mean, independently of Worker/junction count.
Every configured network and seed must be present. Empty trips, non-finite ATT,
or nonpositive ATT invalidate that candidate rather than create a zero-score win.
Ties retain the earlier best. Each strict decrease updates best_model, independent
of whether early stopping considers it a significant improvement.

Only ATT ranks valid candidates. Queue Length and Trip Completion Rate are
recorded alongside it; there is no MaxPressure throughput reference or threshold.
ATT covers completed trips only, so report completion flow and unfinished traffic
when interpreting results; a lower selected ATT does not guarantee better flow.

## Weights, logs and recovery

The lifecycle is: finish PPO update -> clone weights -> validate that snapshot ->
retain the same snapshot if ATT improves -> atomically commit training state.
Validation uses a separate process and fixed policy seed. The snapshot is not
modified by later PPO updates. `models/best_model.pkl` records its selected update,
score, aggregation and selection rule. `validation/selection.json` stores the full
history; individual reports include the validated snapshot's SHA256. Adding
selection/training metadata changes a file's serialized checksum, so tensor
equality, rather than whole-file checksum equality, identifies the same policy.

TensorBoard records `Validation/ATT`, `Validation/Queue Length`,
`Validation/Trip Completion Rate` and `Validation/Valid`, plus per-network curves
in joint mode. Early stopping, if enabled, monitors this same ATT score. Initial
invalid candidates cannot trigger false convergence before any valid model exists.

The old Queue Length/MaxPressure rule and its YAML fields have been removed.
Old training states fail the frozen-contract check; there is no conversion or
fallback. Existing frozen models can still be evaluated. Historical output
directories are never rewritten. New-rule resumes restore the selected weights,
validation history, early-stop state, optimizers, auxiliary state and RNGs.
