# CaDeLaC Panda + payload evaluation (knode-mpc)

Adapt knode-MPC to the [CaDeLaC](https://github.com/Schulze18/cadelac) MuJoCo setup:
Franka Panda, random EE payloads (0–4 kg), joint Fourier (EXC) tracking at 50 Hz.

Paper: https://arxiv.org/pdf/2506.15249

## What changed

| Piece | Role |
|-------|------|
| `robots/panda` | Symlink to CaDeLaC Panda MJCF |
| `robots/panda_payload.py` | MjSpec EE box payloads (plant w/ inertia, nominal w/o) |
| `config/robot/panda.json` | 7-DoF, torque limits, Pinocchio `panda.xml` for MPC |
| `config/models/panda_q_qdot.json` | Residual net input `[q, qdot]` → `Δq̈` |
| `utils/exc_trajectory.py` | CaDeLaC-style Fourier EXC references |
| `utils/collect_panda_payload_data.py` | Multi-env CTC data collection |
| `utils/import_cadelac_dataset.py` | Convert HF pickle → `.npz` once LFS is pulled |
| `mpc/eval_cadelac_payload.py` | Multi-env joint-tracking eval (nominal or knode) |

Nominal MPC model = Pinocchio from `robots/panda/panda.xml` (**no** payload).  
Plant = MuJoCo scene + EE box mass (mismatch to learn / evaluate against).

## Quick start

### 1. Collect a small multi-payload dataset

```bash
python -m utils.collect_panda_payload_data --dataset panda_payload_exc
```

Or import their HuggingFace set (after downloading the real `.pkl`, not the LFS pointer):

```bash
# from cadelac repo / HF:
# git clone https://huggingface.co/datasets/schulze18/cadelac ...
python -m utils.import_cadelac_dataset \
  --pkl /path/to/panda_mj_101_rand_envs_20_runs_50Hz_lqr.pkl \
  --out data/panda_cadelac_hf
```

### 2. Train KNODE residual

With the imported HuggingFace rollouts:

```bash
python run_experiment.py \
  --dataset panda_cadelac_hf \
  --model panda_q_qdot \
  --robot panda \
  --lookahead 2 \
  --epochs 3000
```

Or with locally collected data (`panda_payload_exc`).

### 3. Evaluate on held-out payload envs

```bash
# Nominal MPC baseline
python -m mpc.eval_cadelac_payload --config panda_exc --robot panda --solver acados

# With trained residual
python -m mpc.eval_cadelac_payload --config panda_exc --robot panda --solver acados \
  --knode-run runs/<your_run_dir>
```

Default eval is intentionally smaller than the paper (5 envs × 2 runs). Paper protocol is 30 envs × 20 runs — raise `n_eval_envs` / `runs_per_env` in `config/mpc/panda_exc.json` or via CLI flags.

## Dataset fields (CaDeLaC)

Their pickle provides `qp, qv, qa, tau, diff_tau, ...`. KNODE only needs `q=qp`, `qdot=qv`, `tau` for residual ODE training. Residual torque / LSTM context are CaDeLaC-specific and not required for this residual-`Δq̈` stack.

## Notes

- Control: 50 Hz (`dt=0.02`), horizon `N=12`, costs match CaDeLaC `Q`/`R` (state order here is `[q; qd]`).
- Physics uses 10 MuJoCo substeps per control step (`physics_substeps=10`).
- Their trained DeLaN/LSTM is **not** wired in; this path evaluates **your** KNODE residual MPC on **their** plant/task.
