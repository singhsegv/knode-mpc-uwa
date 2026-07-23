# Training Performance: CUDA, GPU, and JAX

A performance guide for KNODE training in this PyTorch codebase — covering whether NVIDIA CUDA helps, whether JAX can deliver 10–50× speedups, and what to fix first.

**Short answer:** CUDA will **not** help much with this code as written, and may hurt. A **10–50× JAX speedup is plausible versus the current implementation**, but not from “JAX instead of PyTorch” alone — it comes from fixing structural bottlenecks that JAX makes easier to address.

---

## Table of Contents

1. [Where Time Actually Goes](#1-where-time-actually-goes)
2. [Will NVIDIA CUDA Help?](#2-will-nvidia-cuda-help)
3. [Is JAX 10–50× Realistic?](#3-is-jax-1050-realistic)
4. [What Scales with Lookahead?](#4-what-scales-with-lookahead)
5. [What to Do (Practical Order)](#5-what-to-do-practical-order)
6. [Adjoint-Specific Notes](#6-adjoint-specific-notes)
7. [Recommended Path](#7-recommended-path)
8. [Quick Decision Guide](#8-quick-decision-guide)

---

## 1. Where Time Actually Goes

The hot path each epoch is in `knode/learn.py`:

```python
y_hat = model(training_x, training_timesteps)[1:].reshape(-1, 6)[:, :4]
labels = torch.stack(training_labels, dim=1)[1:].reshape(-1, 6)[:, :4]

loss = loss_fn(y_hat, labels)
loss.backward()
optimizer.step()
```

That calls `odeint_adjoint` → `PlanarArmODEFunc.forward` → **`compute_nominal`**, which is the main bottleneck in `knode/model.py`:

```python
def compute_nominal(self, x, xdot, u):
    rows = x.shape[0]
    output = np.zeros((rows, 6))

    for idx in range(rows):
        x_np = x[idx].detach().cpu().numpy()
        xdot_np = xdot[idx].detach().cpu().numpy()
        u_np = u[idx].detach().cpu().numpy()
        ...
        qdot, qddot = self.robot.get_imperfect_dynamics(x_np, xdot_np, u_np)
        ...

    output = torch.from_numpy(output).to(x.device, dtype=x.dtype)
    return output
```

### Problems stacked together

| Issue | Effect |
|-------|--------|
| **Everything on CPU** | No `.cuda()` anywhere; model is `float64` on CPU (`run_experiment.py`) |
| **Pinocchio in a Python `for` loop** | Every RK4 substep evaluates the vector field; adjoint re-evaluates it in backward |
| **CPU↔Torch copies per row** | `.detach().cpu().numpy()` inside the loop |
| **Huge full-batch training** | ~10 trajectories × 7500 steps ≈ **~75k windows/epoch** at once |
| **`float64`** | Fine for accuracy; bad for GPU throughput even if you moved to CUDA |

So the bottleneck is **not** “adjoint is slow on CPU.” It is **“Python + Pinocchio per sample, inside every ODE stage, for a giant batch.”**

The model even notes this:

```text
TODO: Batching and all is a shit show right now  (PlanarArmKnodeModel, knode/model.py)
```

---

## 2. Will NVIDIA CUDA Help?

### Lookahead = 2 (essentially one RK4 step)

- **As-is: no, probably not.** Pinocchio stays on CPU; you’d pay sync/copy overhead if the MLP were on GPU.
- The residual net is tiny (`6 → 128 → 128 → 2`). GPU only wins when large batched matmuls dominate. Here they do not.
- `odeint_adjoint` on GPU with a CPU nominal model is a worst-case hybrid.

### Longer lookahead (e.g. 10, 50, 100)

- Cost scales roughly **linearly with number of integration steps**.
- Adjoint cost also grows (more forward/backward vector-field evaluations).
- GPU helps **only if the whole vector field is GPU-native and batched**:
  - residual MLP on GPU
  - nominal dynamics vectorized in Torch/JAX on GPU
  - no Python loop, no per-sample Pinocchio calls

### Rule of thumb for this repo today

```text
CUDA speedup ≈ 0× to maybe 1.5× as-is
CUDA speedup ≈ 2–8× after fully GPU-vectorized dynamics + float32 + real batching
```

Not worth buying a GPU just to add `.cuda()` to this code.

---

## 3. Is JAX 10–50× Realistic?

**Versus your current code: yes, that range is believable.**  
**Versus a well-optimized PyTorch version: more like 1.5–5×, not 50×.**

JAX gets big wins when you can:

1. **`jit` the entire train step** (forward + adjoint + optimizer)
2. **`vmap` over batch** instead of Python loops
3. Keep the ODE solver in-array (**Diffrax** instead of torchdiffeq + Pinocchio callbacks) -> FRAX + DIFFRAX SHIZZ
4. Use **`float32`**
5. Avoid host↔device sync in the vector field

Your current structure is exactly what JAX/Diffrax examples compare against when they show huge speedups: Python loops, external C++ called from inside `forward`, no JIT, full-batch adjoint ODE.

A straight port “same architecture, still call Pinocchio in `forward`” would **not** be 10–50×.

---

## 4. What Scales with Lookahead?

For fixed `step_size=0.002` (from `config/models/basic1.json`):

| Lookahead | Integration horizon | Rough relative cost |
|-----------|---------------------|---------------------|
| 2 | 0.002 s (1 step) | 1× |
| 10 | 0.018 s (~9 steps) | ~5–10× |
| 50 | 0.098 s (~49 steps) | ~25–50× |
| 100 | 0.198 s (~99 steps) | ~50–100× |

Adjoint memory stays manageable, but **compute grows with horizon**. Longer lookahead makes vector-field efficiency matter more — but only after you fix the Pinocchio loop.

Lookahead is configured via `run_experiment.py`:

```bash
python run_experiment.py --dataset focussed_line --model basic1 --lookahead 50
```

---

## 5. What to Do (Practical Order)

### Step 1: Biggest win in PyTorch first (no JAX, no GPU)

**Replace the per-row Pinocchio loop with batched nominal dynamics in Torch.**

For a 2R planar arm you can write \(M(q), C(q,\dot q), g(q)\) analytically and do:

```python
qdd = torch.linalg.solve(M, tau - C @ qdot - g)  # batched
```

Or batch Pinocchio if you must keep URDF fidelity — but not one `aba()` call per row in Python.

**Expected speedup: 10–100×** on training alone.

### Step 2: Mini-batch instead of all ~75k windows at once

Use mini-batches (512–4096). Helps memory, cache, and optimizer noise. Often faster per epoch even on CPU.

### Step 3: Consider `float32` for training

The codebase uses `float64` everywhere (`knode/learn.py`, `utils/model_utils.py`, `run_experiment.py`). For learning a small residual, `float32` is usually enough and much faster on GPU. Keep `float64` for eval/MPC if needed.

### Step 4: Then try CUDA

After steps 1–3:

- Put model + batched dynamics on GPU
- Keep data on GPU between steps
- Use mini-batches large enough to saturate GPU

Expect **2–5×** over optimized CPU PyTorch for longer lookahead.

### Step 5: JAX migration path (if you still need more)

Only worth it after you know where time goes.

**Stack:**

- **JAX + Flax/Equinox** for the residual net
- **Diffrax** for ODE integration + adjoint (`diffrax.BacksolveAdjoint` or `RecursiveCheckpointAdjoint`)
- **`jax.vmap`** over batch
- **Analytical 2R dynamics in JAX** for nominal (best option)
- Optional: **`jax.pmap`** on multi-GPU

**Do not:**

- Call Pinocchio inside `jax.pure_callback` in the inner RK4 loop — you'll lose most of the benefit

**Migration strategy:**

1. Profile current PyTorch after batched nominal (establish baseline)
2. Reimplement only `PlanarArmODEFunc` + train loop in JAX
3. Match losses/checkpoints numerically on lookahead=2
4. Scale lookahead and batch size

---

## 6. Adjoint-Specific Notes

`odeint_adjoint` (from `torchdiffeq`, used in `PlanarArmKnodeModel.forward`) saves memory (good for long horizons) but **does not reduce compute** — backward re-runs the vector field.

For **lookahead=2**, you integrate almost one step; adjoint overhead can be a large fraction of total time. Alternatives:

- **Direct backprop through fixed RK4** for small horizons (sometimes faster than adjoint for 1–5 steps)
- **Checkpointed adjoint** for very long horizons
- In JAX/Diffrax: pick adjoint vs backprop through solver based on horizon length

So adjoint is not the reason GPU/JAX will magically help; the **vector-field implementation** is.

---

## 7. Recommended Path

```text
Phase A (PyTorch, CPU)     → batch nominal dynamics, mini-batch, float32
Phase B (PyTorch, CUDA)    → if still slow after A
Phase C (JAX/Diffrax)      → only if you need more after A+B, or want cleaner vmap/jit for MPC later
```

**Do not jump straight to JAX expecting 50×** without fixing the Pinocchio Python loop — that loop is the main problem in both frameworks.

---

## 8. Quick Decision Guide

| Question | Answer for this codebase |
|----------|--------------------------|
| Will GPU help now? | **No meaningful help** |
| Will GPU help after refactor? | **Yes, especially for longer lookahead** |
| Is JAX 10–50× real? | **Yes vs current code; no vs optimized PyTorch** |
| Best first step? | **Vectorize nominal dynamics in Torch, mini-batch** |
| Does adjoint need GPU? | **No — adjoint needs a fast, batched vector field** |

---

## Relevant Files

| File | Role in performance |
|------|---------------------|
| `knode/model.py` | `compute_nominal` loop, `odeint_adjoint`, residual MLP |
| `knode/learn.py` | Full-batch training loop, `float64`, no mini-batching |
| `run_experiment.py` | `--lookahead`, model init, no device placement |
| `robots/simulator.py` | Pinocchio `aba()` called from nominal path |
| `utils/model_utils.py` | Eval rollouts (sequential, one step at a time) |

---

## Summary

| Approach | Expected speedup vs today | Effort |
|----------|---------------------------|--------|
| Add `.cuda()` as-is | ~0× | Low |
| Batched Torch nominal + mini-batch + float32 | **10–100×** | Medium |
| Above + CUDA | **+2–5×** on top | Medium |
| Full JAX/Diffrax rewrite with JAX-native dynamics | **10–50× vs today; ~1.5–5× vs optimized PyTorch** | High |

The highest-ROI change is **vectorized nominal dynamics in PyTorch** before considering JAX or CUDA hardware.
