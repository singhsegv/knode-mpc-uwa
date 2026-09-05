# UR5 Full-Arm Dataset Configs

JSON trajectory specs for 6-DOF UR5 data collection and K-NODE training.

All trajectories are **position-only** in 3D task space `(x, y, z)`. Workspace parameters are tuned for the UR5 reachable volume near the default home configuration.

## Configs

| File | Purpose |
|------|---------|
| `ur5_demo.json` | Smoke test — 1 train + 1 test circle |
| `ur5_basic.json` | Small training set — circle, line, rectangle |
| `ur5_generalization.json` | Varied shapes, speeds, and a helix trajectory |

## Trajectory spec fields

Common fields:
- `type`: `circle`, `line`, `rectangle`, `lemniscate`, or `helix`
- `duration`: rollout length [s]
- `omega`: angular speed parameter
- `cz`: constant end-effector height [m] (default 0.35)

Type-specific:
- **circle**: `cx`, `cy`, `r`
- **line**: `lx0`, `ly0`, `lx1`, `ly1`, optional `lz0`, `lz1`
- **rectangle**: `lx0`, `ly0`, `lx1`, `ly1`, optional `cz`
- **lemniscate**: `cx`, `cy`, `a`
- **helix**: `cx`, `cy`, `r`, `cz0`, `cz1`, `z_omega`

## Usage

```bash
# Collect training data only
python utils/training_data_generator.py --config ur5_demo --split train

# Collect both train and test splits
python utils/training_data_generator.py --config ur5_basic --split both

# Full experiment (auto-generates data if missing)
python run_experiment.py --dataset ur5_demo --model basic1 --robot ur5
```

Data is saved under `data/<name>/train/` and `data/<name>/test/`.
