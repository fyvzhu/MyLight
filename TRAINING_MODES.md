Single-network and joint-network training
=======================================

Both modes use `python driver_unicorn.py` and the same sampling, PPO,
validation, checkpoint and resume lifecycle. The default remains single Grid5x5
with six Workers. The current `ppo_variant`, `clipping` and `aes` choices are
preserved independently of the training mode.

Single network
--------------

```yaml
training:
  mode: single
  seed: 0
  networks:
    - name: grid_network_5_5
      workers: 6
```

Exactly one network is required. All six Workers sample separate traffic/policy
trajectories on that network, as in the previous single-network pipeline.
Old `training: {network: grid_network_5_5, seed: 0}` settings remain accepted
and mean single mode with six Workers. Do not mix `network` with `networks`.

Joint networks
--------------

```yaml
training:
  mode: joint
  seed: 0
  networks:
    - {name: cologne_network_8, workers: 1}
    - {name: ingolstadt_network_21, workers: 1}
    - {name: arterial_network_4_4, workers: 1}
    - {name: grid_network_4_4, workers: 1}
    - {name: shaoxing_network_7, workers: 1}
    - {name: shenzhen_network_29, workers: 1}
```

This is the original Unicorn six-network assignment: one Worker per network,
97 junctions in total, with 36 movement slots and 8 phase slots. Those numbers
are calculated from the selected maps, not constants in the environment.

Joint mode requires at least two distinct network names. Each `workers` must
be a positive YAML integer. Worker IDs are assigned in list order; for example,
two Grid5x5 Workers followed by one Grid4x4 Worker yields IDs 0/1/2, 66 actual
junction instances, a 36-movement/8-phase representation, and three Workers.
The total Worker count is computed automatically, with no additional
`NUM_META_AGENTS` or `CO_TRAIN` setting to maintain in Python.
Ray's logical CPU capacity is declared from that count so all persistent
Workers can start. This does not increase the container's physical CPU quota;
the configured Workers still share its CPU/GPU resources.

All selected Workers contribute to one shared Actor/Critic. Movement/phase
features and neighbor action vectors use common padding; invalid phases remain
masked. Actual junctions are concatenated without adding dummy junctions.
Neighbors stay local to their original network. Network/Worker identities are
kept in runtime metadata, diagnostics and checkpoints.

The existing sampling horizon and signal execution cadence stay unchanged
across networks, so merged fragments have compatible time dimensions. A
`max_episodes` update still means one full collection from every Worker followed
by the existing PPO epochs. Increasing Workers also increases simulation and
junction-sample budgets. This is not task-balanced gradient aggregation: PPO
continues to use the existing junction-sample objective, with no new reward,
value or advantage normalization.

Validation and best-model selection
-----------------------------------

The existing `validation` block is sufficient. Every unique training network
is validated using the configured seeds and duration, regardless of its number
of Workers. The fixed MaxPressure reference is obtained for every such network.
Every network/seed must satisfy the existing completion/validity constraint;
an average completion rate cannot conceal one failed network or seed.

Once all constraints pass, the score is the equal-weight mean of the networks'
mean Queue Length. In single mode this is exactly the previous score. A joint
reference/candidate is written to:

```
validation/reference/<network>/data/metrics.json
validation/update_XXXXXXXX/<network>/data/metrics.json
validation/update_XXXXXXXX/data/metrics.json   # macro summary with all subreports
validation/selection.json
```

The macro JSON's mean gives each network equal weight; its std describes the
network means. Each leaf report retains its original per-seed statistics.
No validation seed, duration or completion threshold is silently reduced.
Joint validation therefore costs more simulation time than validating one map.
TensorBoard includes per-network training reward/sample counts and validation
Queue Length, alongside the existing global training and AES diagnostics.

Evaluation and checkpoint compatibility
---------------------------------------

`evaluation.network` continues to choose one target map and
`evaluation.checkpoint` chooses the saved model. The target need not be in the
training list. Evaluation reads the model's saved input schema before creating
the target environment; smaller native movement/phase dimensions are padded,
and dimensions exceeding the saved schema fail explicitly rather than being
truncated. The current `training.mode` and AES flag do not override the saved
model architecture or representation.

The six-network example excludes Grid5x5. Evaluating its model on Grid5x5 is
unseen-network transfer. To train on Grid5x5, add it explicitly to the training
list. Held-out test networks never participate in validation/model selection.

Single-mode checkpoints retain their previous schema and contract. Old runs
can be resumed when their original settings and Worker count match. Joint
checkpoints additionally freeze the complete network/Worker plan and shared
layout, and check each Worker's network identity on restoration. Continue with:

```
python driver_unicorn.py --resume-run train_outputs/Unicorn/<run>
```

Changing the mode, network list, Worker allocation, active AES configuration,
PPO variant or clipping during a resume is rejected. Start a new run for those
experiments. Joint training remains compatible with scalar Unicorn, DisPPO,
all existing policy clipping modes, and the AES training switch. AES is still
supported by the Unicorn driver only; HeteroLight requires `aes: false`.

No runtime module imports `tests` or the downloaded reference repositories.
The feature uses the existing Python/PyTorch/NumPy/PyYAML/Ray/SUMO environment;
no new dependency installation is required.
