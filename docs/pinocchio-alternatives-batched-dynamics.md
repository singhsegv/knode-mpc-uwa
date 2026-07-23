# Pinocchio Alternatives for Batched Dynamics Training

A guide to replacing per-sample Pinocchio calls in the KNODE training loop without hand-derived analytical dynamics. Options are grouped by how well they fit the current PyTorch + URDF stack.

**Core problem:** You don't need to hand-derive \(M, C, g\). You need a **batched, autodiff-friendly** rigid-body backend instead of **Pinocchio called in a Python loop** in `knode/model.py` → `compute_nominal`.

---

## Table of Contents

1. [Best Fit for Current PyTorch KNODE](#1-best-fit-for-current-pytorch-knode)
2. [Hybrid: Pinocchio for Data/MPC, BARD for Training](#2-hybrid-pinocchio-for-datampc-bard-for-training)
3. [Best Fit if Moving to JAX](#3-best-fit-if-moving-to-jax) — includes [frax](frax-jax-dynamics.md) and MJX
4. [Stay in Pinocchio Ecosystem (No New Sim)](#4-stay-in-pinocchio-ecosystem-no-new-sim)
5. [Symbolic Auto-Generation from URDF](#5-symbolic-auto-generation-from-urdf)
6. [Other Libraries (Usually Not the Speed Fix)](#6-other-libraries-usually-not-the-speed-fix)
7. [Recommendation for This Repo](#7-recommendation-for-this-repo)
8. [Practical Migration Path (PyTorch + BARD)](#8-practical-migration-path-pytorch--bard)
9. [Important Constraint: Sim Mismatch](#9-important-constraint-sim-mismatch)

---

## 1. Best Fit for Current PyTorch KNODE

### BARD — `bard-pytorch-dynamics` (top PyTorch pick)

- **Repo:** https://github.com/YueWang996/bard-pytorch-dynamics
- **Paper:** [Batched Differentiable Rigid Body Dynamics in PyTorch](https://arxiv.org/abs/2605.31481)

**What it is:** Featherstone algorithms (FK, RNEA, CRBA, **ABA**) implemented in **pure PyTorch**, with **batch + GPU + autograd**.

**Why it fits this codebase:**

- Loads **URDF** (e.g. `robots/planar-2R.urdf`)
- Drop-in conceptual replacement for `robot.forward_dynamics()` / `aba()` in `compute_nominal`
- Designed exactly for the problem here: Pinocchio in a GPU training loop
- Claims large speedups at batch 4096; even on CPU, batched tensor ops beat the per-row loop

**Caveats:**

- Newer / less battle-tested than Pinocchio
- Validate numerically against Pinocchio on the 2R model before trusting training
- You'd still implement `get_imperfect_dynamics` logic (extra gravity torque) on top of batched ABA

**Rough integration idea:**

```python
# Instead of for idx in range(rows): pin.aba(...)
qdd = bard.aba(model, data, q, qdot, tau)  # (B, nv), fully torch
```

This is the closest thing to “keep PyTorch + URDF + no hand math + fix perf.”

**Feature comparison (from BARD repo):**

| Feature | BARD | Pinocchio | RBDL | pytorch_kinematics |
|---------|:----:|:---------:|:----:|:------------------:|
| PyTorch Native | ✅ | ❌ | ❌ | ✅ |
| Batch Processing | ✅ | ❌ | ❌ | ✅ |
| GPU Acceleration | ✅ | ❌ | ❌ | ✅ |
| Auto-Differentiation | ✅ | ❌ | ❌ | ✅ |
| Forward Dynamics (ABA) | ✅ | ✅ | ✅ | ❌ |
| URDF Parsing | ✅ | ✅ | ✅ | ✅ |

---

## 2. Hybrid: Pinocchio for Data/MPC, BARD for Training

Hybrid is very reasonable:

| Component | Library |
|-----------|---------|
| Data collection (`utils/training_data_generator.py`) | Pinocchio (unchanged) |
| MPC / CasADi (`mpc/casadi_models.py`) | `pinocchio.casadi` (unchanged) |
| KNODE nominal in `knode/model.py` | **BARD** (batched) |

You only need train/sim mismatch to match numerically, not the same library everywhere.

---

## 3. Best Fit if Moving to JAX

### frax — Fast Robot Kinematics and Dynamics in JAX (top JAX + URDF pick)

- **Repo:** https://github.com/StanfordASL/frax
- **Install:** `pip install frax`
- **Paper:** [arXiv:2604.04310](https://arxiv.org/abs/2604.04310)
- **Full write-up:** [frax — JAX Robot Dynamics for KNODE Training](frax-jax-dynamics.md)

**What it is:** Pure JAX rigid-body kin/dynamics from **URDF** — `jit`, `vmap`, autodiff. Lighter than MJX; no MJCF conversion.

**Why it fits this repo:** 2R planar arm, imperfect nominal via `tau + g(q)`, batched KNODE training. Arguably **better than MJX** when you only need manipulator dynamics, not full contact sim.

**Caveats:** JAX training stack required (Diffrax, not `torchdiffeq`); beta API; validate vs Pinocchio.

### MuJoCo MJX (`mujoco.mjx`)

- **Docs:** https://mujoco.readthedocs.io/en/stable/mjx.html
- **Install:** `pip install mujoco` (includes MJX-JAX)

**What it is:** JAX reimplementation of MuJoCo dynamics; **`jax.vmap` + `jit`** over thousands of parallel rollouts.

**Why people cite 10–50×:** The whole vector field stays in XLA — no Python loop, no CPU callback.

**Example pattern:**

```python
from mujoco import mjx
import jax

model = mjx.put_model(mujoco_model)
data = mjx.make_data(model)
step_fn = jax.jit(mjx.step)
new_data = jax.vmap(step_fn, in_axes=(None, 0))(model, batched_data)
```

**Caveats for this repo:**

- Model format is **MJCF**, not URDF (there is `archives/3R-stuff/planar-3R.xml` as a starting point)
- You must rebuild the robot in MuJoCo and match inertias/gravity to Pinocchio/URDF
- Imperfect nominal (the `tau + g(q)` hack) must be replicated in the MuJoCo model or added in JAX
- Full training stack becomes JAX + Diffrax/MJX, not “drop into current `torchdiffeq` code”

**When to choose:** You're willing to migrate training to JAX and re-validate sim vs data pipeline.

**Also:** **MJX-Warp** (NVIDIA) if you want max GPU throughput and don't need differentiability through the sim.

| | MJX-JAX | MJX-Warp |
|---|---------|----------|
| Differentiability | ✅ | ✗ |
| Hardware | GPU, TPU, Apple Silicon | NVIDIA GPU optimized |

### Brax generalized pipeline

- **Repo:** https://github.com/google/brax

Brax maintainers now say: use **MJX** for physics, not Brax's old generalized sim. Treat Brax as RL training infra, not the dynamics engine.

> Only `brax/training` is actively maintained as of 0.13.0. For physics simulation, use MJX (`pip install mujoco_mjx`) or MuJoCo Warp rather than Brax as a wrapper.

---

## 4. Stay in Pinocchio Ecosystem (No New Sim)

These don't replace Pinocchio with another runtime library, but avoid **hand** analytical derivations.

### `pinocchio.casadi` + CasADi `map` (already used in MPC)

Existing code in `mpc/casadi_models.py`:

```python
nominal_qdd = cpin.aba(cmodel, cdata, q, v, tau)
```

**Idea:** Build the symbolic CasADi graph once, then evaluate with **CasADi's `map`** over batch dimension. Faster than Python `for`, still URDF-accurate.

**Limits:**

- Still CPU CasADi in most setups
- Awkward inside PyTorch autograd (external op / callback)
- Better for MPC export than for `odeint_adjoint` training unless you wrap carefully

Good for **MPC path**, not the cleanest fix for **KNODE training**.

### Pinocchio + CppADCodeGen (auto-generated C, not hand math)

Pinocchio supports [CppADCodeGen](https://github.com/joaoleal/CppADCodeGen) — generate **C code from URDF** once, compile, call from Python/Torch.

**Pros:** Same physics as Pinocchio, no manual equations  
**Cons:** Build/toolchain pain; still need a Torch wrapper for batched GPU

### L4CasADi — *not* a Pinocchio replacement

There is a comment about L4CasADi in `mpc/dafuq.py`. That tool embeds **PyTorch NNs in CasADi** for MPC — opposite direction. Useful for deploying learned models in CasADi, **not** for speeding up nominal Pinocchio in training.

- **Repo:** https://github.com/Tim-Salzmann/l4casadi
- **Paper:** [Learning for CasADi (L4CasADi)](https://arxiv.org/abs/2312.05873)

---

## 5. Symbolic Auto-Generation from URDF

These **derive** \(M, C, g\) from URDF/SymPy; you compile/lambdify once, then evaluate fast. Not “hand analytical”, but one-time setup.

| Library | URL | Notes |
|---------|-----|-------|
| **Odysseus** | https://github.com/adrian-kriegel/odysseus | URDF → symbolic Euler–Lagrange; good for NMPC/control codegen |
| **ManipulaPy** | https://manipulapy.readthedocs.io/ | URDF → analytical dynamics objects (`URDFToSerialManipulator`) |
| **SymPyBotics** | https://github.com/cdsousa/sympybotics | MDH-based; unmaintained; fine for 2R |
| **rigmech** | https://github.com/DarrenLevine/rigmech | URDF → SymPy → lambdify; save `.bin` after first slow compile |
| **URDFly** | https://github.com/Democratizing-Dexterous/URDFly | URDF parsing, MDH, dynamics regressor codegen |

For a **2R arm**, one of these can codegen closed-form dynamics in minutes. That's not pen-and-paper math — it's automated symbolic derivation.

**Downside:** Usually **NumPy/SymPy**, not native GPU batching unless you re-export to Torch/JAX.

**Example (rigmech pattern):**

```python
Robot = rigmech("your_robot_name")
if not Robot.load():
    Robot.loadURDF("your_robot.urdf")
    Robot.generateEqns(Simplify=True)
    Robot.save()
q, dq, ddq = Robot.ForwardDynamics(dt, q, dq, qForceJoints, xyzFictitiousAccels)
```

---

## 6. Other Libraries (Usually Not the Speed Fix)

| Library | Verdict |
|---------|---------|
| **Drake (`pydrake`)** | Excellent correctness; CPU; sequential. Good sim, bad for ~75k batched ODE steps. |
| **RBDL** | Same class as Pinocchio — fast single config, not batched GPU training. |
| **pytorch_kinematics / torchkin** | FK/Jacobian only — **no ABA/forward dynamics**. |
| **PyBullet** | Wrong tool for differentiable batched training. |
| **Isaac Sim / Isaac Lab** | Heavy RL stack; overkill unless you're all-in on NVIDIA sim. |

---

## 7. Recommendation for This Repo

```text
Want to stay PyTorch + URDF + minimal migration?
  → BARD first. Validate vs Pinocchio on planar-2R. Replace compute_nominal loop.

Want JAX + URDF manipulator dynamics (no MJCF)?
  → frax. See frax-jax-dynamics.md. Pair with Diffrax for training.

Want max training throughput + full MuJoCo sim in JAX?
  → MuJoCo MJX (convert URDF/XML → MJCF, match params).

Want to keep Pinocchio everywhere but avoid hand math?
  → Symbolic codegen (Odysseus/ManipulaPy) for 2R only, OR CasADi map for MPC only.

Don't bother for training speed:
  → Raw Drake/RBDL/PyBullet, or .cuda() without batched dynamics.
```

See also: [Training Performance — CUDA, GPU, and JAX](training-performance-gpu-jax.md) for why batched dynamics matter more than hardware alone.

---

## 8. Practical Migration Path (PyTorch + BARD)

1. Load same URDF in BARD and Pinocchio
2. Compare `aba(q, qdot, tau)` on random batches (including training `q` range)
3. Replace `compute_nominal` in `knode/model.py` with batched BARD call
4. Reimplement imperfect nominal as batched `tau_eff = tau + g(q)` using BARD's gravity or RNEA
5. Add mini-batching in `knode/learn.py`
6. Then try GPU + `float32`

**Target replacement** — current bottleneck in `knode/model.py`:

```python
def compute_nominal(self, x, xdot, u):
    rows = x.shape[0]
    output = np.zeros((rows, 6))

    for idx in range(rows):
        x_np = x[idx].detach().cpu().numpy()
        ...
        qdot, qddot = self.robot.get_imperfect_dynamics(x_np, xdot_np, u_np)
        ...

    output = torch.from_numpy(output).to(x.device, dtype=x.dtype)
    return output
```

---

## 9. Important Constraint: Sim Mismatch

Ground-truth sim uses Pinocchio with imperfect dynamics in `robots/simulator.py`:

```python
def get_imperfect_dynamics(self, q, qdot, tau):
    tau_eff = tau + self.gravity_forces(q)
    qdd = self.forward_dynamics(q, qdot, tau_eff)
    return qdot, qdd
```

Whatever backend you train with must **match this numerically** (or you re-generate all training data in the new sim).

| Backend | Matching strategy |
|---------|---------------------|
| **BARD** | Can match Pinocchio ABA + explicit gravity offset |
| **MJX** | Careful MJCF param alignment + same `tau_eff` logic in JAX |
| **Symbolic codegen** | Encode same gravity compensation in generated equations |

Training data is collected via `utils/training_data_generator.py` using Pinocchio `step_torchdiffeq`. The KNODE nominal only needs to be consistent with what the residual is trying to correct — but rollout evaluation must still align with data generation physics.

---

## Summary

| Approach | Expected benefit | Effort | Stays PyTorch? |
|----------|-------------------|--------|----------------|
| **BARD** (batched URDF dynamics) | Large training speedup | Medium | ✅ |
| **frax** (JAX URDF kin/dyn) | Large with JAX + Diffrax stack | High | ❌ |
| **MJX** (JAX MuJoCo) | Very large with full JAX stack | High | ❌ |
| **CasADi map / CppADCodeGen** | Moderate; better for MPC | Medium–High | Partial |
| **Symbolic URDF codegen** | Fast eval, not native GPU batch | Medium (one-time) | Export-dependent |
| **Drake / RBDL / PyBullet** | Little for batched training | Low | ✅ |

**Bottom line:** If you refuse hand-derived equations but want speed in PyTorch, **`bard-pytorch-dynamics`** is the most direct Pinocchio replacement. If you're open to JAX with **URDF**, **[frax](https://github.com/StanfordASL/frax)** is the best fit for manipulator nominal dynamics — see [frax-jax-dynamics.md](frax-jax-dynamics.md). For full sim in JAX, use **`mujoco.mjx`**. Symbolic URDF tools (Odysseus, ManipulaPy) are the middle ground: automated, not hand-written, but not GPU-native without extra export work.

---

## Relevant Files in This Repo

| File | Role |
|------|------|
| `knode/model.py` | `compute_nominal` Pinocchio loop to replace |
| `robots/simulator.py` | Pinocchio ground-truth sim + imperfect dynamics |
| `mpc/casadi_models.py` | `pinocchio.casadi` for MPC (can stay) |
| `utils/training_data_generator.py` | Data collection with Pinocchio |
| `robots/planar-2R.urdf` | URDF shared across backends |
| `archives/3R-stuff/planar-3R.xml` | Possible starting point for MJX/MJCF |
