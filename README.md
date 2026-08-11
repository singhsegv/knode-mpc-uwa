# Learning 2D Planar Arm Dynamics

## About
Spawned from `main` branch but now with a full arm but same infra setup.

## Running Experiments

```bash
# UR5 full-arm pipeline
python utils/training_data_generator.py --config ur5_demo --split both
python run_experiment.py --dataset ur5_demo --model basic1 --robot ur5

# Planar arm (legacy)
python run_experiment.py --dataset evaluation1 --model basic1 --robot planar

# Comparing rollouts
python utils/compare_collected_rollouts.py data/standalone data/other_run --save

# Performance and generalization
python evaluate/benchmark_model.py --run <run_name> --epoch 2100 --dataset benchmark_basic --plot
```


## Docs

- [UR5 6-DOF Migration Guide](docs/ur6-migration.md)
- [Learning on Manifolds — References & Reading Guide](docs/learning-on-manifolds.md)
- [Training Performance — CUDA, GPU, and JAX](docs/training-performance-gpu-jax.md)
- [Pinocchio Alternatives — Batched Dynamics for Training](docs/pinocchio-alternatives-batched-dynamics.md)
- [frax — JAX Robot Dynamics for KNODE Training](docs/frax-jax-dynamics.md)

## About
1. Data Collection Harness: `simulate.py`
2. Robot Dynmics Simulator: `dynamics.py`
3. Conda Env Issues: KMP_DUPLICATE_LIB_OK=TRUE python knode-learn-dynamics.py
    - Torch (from pip) and Pinocchio (from conda) doesn't work together
    - Fix: `conda install -c conda-forge pinocchio pytorch`

## Todo
- Sanity checks for gravity and hold etc like I did for mujoco
- Check urdf: Origins for links and Other stuff for joints
- EE orientation tracking (theta is hardcoded to 0)
- Missing J_dot term in qddot_des
