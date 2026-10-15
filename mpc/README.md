# MPC for 6-DOF Universal Robots UR5

Closed-loop model predictive control. The **plant** is simulated with MuJoCo (`robots.simulator.Manipulator`). The **MPC internal model** is built in `mpc/casadi_dynamics.py`.

## Dynamics backends

| Component | Engine |
|-----------|--------|
| Plant rollouts (data gen, MPC closed loop) | MuJoCo (`plant_step`, `step_mujoco`) |
| K-NODE training nominal | MuJoCo (`Manipulator.forward_dynamics`) |
| MPC optimizer model | CasADi — see `mpc_dynamics_backend` below |

Set in `config/robot/*.json`:

```json
"mpc_dynamics_backend": "pinocchio"
```

- **`pinocchio`** (default): symbolic ABA via `pinocchio.casadi`. Required for **acados** and **fatrop** (they codegen smooth SX graphs).
- **`mujoco`**: CasADi Callback over MuJoCo forward dynamics + Euler. Use with **`--solver ipopt`** when you want the MPC prediction model to match the MuJoCo plant.

## Quick Start

```bash
export ACADOS_SOURCE_DIR=/Users/singhsegv/all-things-robotics/acados
export DYLD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$DYLD_LIBRARY_PATH

# Nominal dynamics with IPOPT (CasADi Opti)
python mpc/mpc_benchmark_harness.py --config ur5_basic --no-knode --mpc-horizon 10 --solver ipopt --output-dir imperfect-non-knode-ipopt

# Nominal dynamics with Acados (needs ACADOS_SOURCE_DIR + acados_template)
python mpc/mpc_benchmark_harness.py --config ur5_basic --no-knode --mpc-horizon 10 --solver acados --output-dir imperfect-non-knode-acados

# With learned K-NODE model (ipopt / fatrop only for now)
python mpc/mpc_benchmark_harness.py \
    --config ur5_basic \
    --knode-path runs/<run_name>
```

`--solver` choices: `ipopt` (default), `fatrop`, `acados`.
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
