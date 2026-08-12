# MPC for 6-DOF Universal Robots UR5

Closed-loop model predictive control using Pinocchio dynamics (optionally augmented with a learned K-NODE residual).

## Quick Start

```bash
# Nominal dynamics benchmark
python mpc/mpc_benchmark_harness.py --config ur5_basic --no-knode

# With learned K-NODE model
python mpc/mpc_benchmark_harness.py \
    --config ur5_basic \
    --knode-path runs/<run_name>
```

## State / Control Dimensions (UR5)

| Symbol | Size | Contents |
|--------|------|----------|
| MPC state `x` | 12 | `[q(6), q̇(6)]` |
| Control `u` | 6 | joint torques |
| K-NODE state `z` | 18 | `[q(6), q̇(6), τ(6)]` |

## Configuration

- Robot parameters: `config/robot/ur5.json`
- MPC trajectories: `config/mpc/ur5_basic.json`

Trajectory specs support optional `"cz"` for the z coordinate of position-only tracking.

## Migration Notes

See [docs/ur6-migration.md](../docs/ur6-migration.md) for the full planar → UR5 migration guide.
