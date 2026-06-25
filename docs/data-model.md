# Data Model

1. Each batch contains a number of groups.
2. Each group can contain multiple trajectories.
3. Each trajectory can contain multiple chat completion requests.
4. Each trajectory can contain multiple sub-trajectories (when a harness is supported).
5. A reward is computed for each top-level trajectory.
6. Advantages are computed for every trajectory within a group only when all trajectories within that group have received a reward.
7. Once enough groups are collected, a batch is considered complete, and a training step (`train_step`) is triggered.
