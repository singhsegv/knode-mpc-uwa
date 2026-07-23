# frax — JAX Robot Dynamics for KNODE Training

An evaluation of [StanfordASL/frax](https://github.com/StanfordASL/frax) (`frax`: Fast Robot Kinematics and Dynamics in JAX) as a Pinocchio replacement for batched nominal dynamics in this codebase.

**Short answer:** **Yes, you can use `frax`** — it is one of the best JAX options for “URDF manipulator dynamics in a learning loop.” It is **not** a drop-in for the current PyTorch `odeint_adjoint` stack without migrating training to JAX (+ Diffrax).

**Paper:** [frax: Fast Robot Kinematics and Dynamics in JAX](https://arxiv.org/abs/2604.04310) (Morton & Pavone, 2026)

---

## Table of Contents

1. [Why frax Is Relevant](#1-why-frax-is-relevant)
2. [What Blocks a Direct Plug-In](#2-what-blocks-a-direct-plug-in)
3. [frax vs Other Options](#3-frax-vs-other-options)
4. [Sketch of a frax Architecture](#4-sketch-of-a-frax-architecture)
5. [Practical Recommendation](#5-practical-recommendation)
6. [Installation & Usage Notes](#6-installation--usage-notes)
7. [Bottom Line](#7-bottom-line)

See also:

- [Pinocchio Alternatives — Batched Dynamics for Training](pinocchio-alternatives-batched-dynamics.md)
- [Training Performance — CUDA, GPU, and JAX](training-performance-gpu-jax.md)

---

## 1. Why frax Is Relevant

The training bottleneck is **batched rigid-body dynamics + autodiff** from **URDF**, without a Python loop over Pinocchio (see `knode/model.py` → `compute_nominal`).

`frax` is built for that:

| What you need | frax |
|---------------|------|
| Load URDF | ✅ `frax.Robot("planar-2R.urdf")` |
| Revolute joints (2R arm) | ✅ revolute / prismatic / fixed supported |
| Dynamics terms | ✅ `mass_matrix`, gravity, Coriolis, Jacobians, etc. |
| Batched + fast | ✅ JAX `jit` / `vmap`; very high throughput on GPU at large batch |
| Autodiff through dynamics | ✅ native JAX |
| Full contact sim | ❌ not the focus (lighter than MJX) |

For **“replace Pinocchio nominal in learning”**, `frax` is closer to the right tool than MJX because:

- **URDF directly** — no MJCF conversion
- **Manipulator dynamics**, not full physics sim
- **JAX-native**, like the 10–50× training path described in the performance doc

Official positioning ([frax README](https://github.com/StanfordASL/frax)): Pinocchio speed + Python ease + MJX-style differentiability and parallelization.

### Hello-world example

```python
import frax
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # recommended for accuracy

robot = frax.Robot("robots/planar-2R.urdf")
q = np.zeros(robot.num_joints)
M = robot.mass_matrix(q)
print(M)
```

Wrap hot paths in JIT:

```python
@jax.jit
def jit_mass_matrix(q_):
    return robot.mass_matrix(q_)
```

---

## 2. What Blocks a Direct Plug-In

### PyTorch vs JAX

Current training uses PyTorch + `torchdiffeq`:

```python
# knode/model.py
def forward(self, y0, timesteps):
    z = odeint_adjoint(self.func, y0, timesteps, method="rk4", options={"step_size": self.step_size})
    return z
```

`frax` is **JAX-only**. Options:

| Approach | Verdict |
|----------|---------|
| Call `frax` from PyTorch via `jax.pure_callback` each RK4 stage | Works, but kills most speedup |
| Move train loop to **JAX + Diffrax**, nominal via `frax` | ✅ right way |
| Keep PyTorch, use **BARD** for nominal instead | ✅ if you want to stay in Torch |

**Can use `frax`, but plan on a JAX training stack** — not `.cuda()` on the current code.

### Beta / changing API

The [frax README](https://github.com/StanfordASL/frax) warns the library is **actively under development** and internal operations may change between beta versions. Fine for research; pin a commit/version and add Pinocchio parity tests.

### Forward dynamics explicitly

`frax` exposes `mass_matrix`, gravity, Coriolis, etc. Forward dynamics is typically:

\[
\ddot q = M(q)^{-1}\big(\tau_{\text{eff}} - C(q,\dot q)\dot q - g(q)\big)
\]

or via any forward-dynamics helper in the API. That matches imperfect nominal in `robots/simulator.py`:

```python
def get_imperfect_dynamics(self, q, qdot, tau):
    tau_eff = tau + self.gravity_forces(q)
    qdd = self.forward_dynamics(q, qdot, tau_eff)
    return qdot, qdd
```

Express as **`tau_eff = tau + g(q)` in JAX with `frax`**, then solve for `qdd`. Verify numerically against Pinocchio before training.

### GPU isn’t always faster for one robot

`frax` notes that for **a single robot instance**, CPU can beat GPU. Training uses **~75k windows per epoch** — there `vmap` over batch should favor GPU. Profile batch size vs device.

### Precision

`frax` recommends **`jax_enable_x64`** for accuracy (this repo uses `float64` today). On GPU that is often 2–6× slower than `float32` — same tradeoff as PyTorch.

---

## 3. frax vs Other Options

| Library | Stays PyTorch? | URDF? | Batched train nominal? | Full sim / contacts? |
|---------|----------------|-------|------------------------|----------------------|
| **BARD** | ✅ | ✅ | ✅ | ❌ |
| **frax** | ❌ (JAX) | ✅ | ✅ | ❌ |
| **MJX** | ❌ (JAX) | ❌ (MJCF) | ✅ | ✅ |
| **Pinocchio loop** | ✅ | ✅ | ❌ | ❌ |

- **Willing to go JAX:** `frax` is arguably **better than MJX** for this 2R URDF KNODE (no contacts, dynamics-only).
- **Stay PyTorch:** use **BARD**, not `frax`.

### frax vs MJX (JAX path)

| | frax | MJX |
|---|------|-----|
| Model format | URDF | MJCF |
| Scope | Manipulator kin/dyn | Full MuJoCo sim |
| Contacts | Limited / not focus | Full |
| Fit for KNODE nominal | **Strong** | Overkill unless you need sim parity in MJCF |

### frax vs BARD (PyTorch path)

| | frax | BARD |
|---|------|------|
| Framework | JAX | PyTorch |
| Migration cost | High (new train stack) | Medium (swap `compute_nominal`) |
| URDF | ✅ | ✅ |
| `torchdiffeq` | ❌ | ✅ |

---

## 4. Sketch of a frax Architecture

```text
Data generation / MPC     → Pinocchio (unchanged)
KNODE training            → JAX + Diffrax + frax nominal + Flax/Equinox residual MLP
```

Conceptual vector field:

```python
@jax.jit
@jax.vmap
def vector_field(z):
    q, qd, tau = split(z)
    tau_eff = tau + robot.gravity(q)   # imperfect nominal (match simulator.py)
    qdd = forward_dynamics(q, qd, tau_eff)  # M^{-1}(...) via frax terms
    residual = mlp(normalize(z))
    return pack(qd, qdd + residual_qdd, 0)
```

Integration with **Diffrax** instead of `odeint_adjoint`; adjoint/backprop through the JAX vector field.

**Validation checklist:**

1. Random `(q, qdot, tau)` batches: `frax` vs Pinocchio `aba` / `forward_dynamics`
2. Imperfect nominal: `tau + g(q)` path matches `get_imperfect_dynamics`
3. Short rollout: Diffrax RK4 vs `torchdiffeq` RK4 at `step_size=0.002`
4. Then train with lookahead=2 before scaling horizon

---

## 5. Practical Recommendation

**Use `frax` if:**

- You are OK migrating **training** to JAX (+ Diffrax)
- You want URDF without MJCF conversion
- You care about autodiff through kinematics/dynamics (useful for MPC/control later)

**Don’t use `frax` if:**

- You want minimal change to current PyTorch KNODE → **BARD** instead
- You need production-stable APIs today → Pinocchio + batching/codegen first
- You need full contact-rich sim in the training loop → **MJX**

---

## 6. Installation & Usage Notes

From [frax README](https://github.com/StanfordASL/frax):

```bash
pip install frax
```

Optional tags: `[cuda12]`, `[cuda13]`, `[tpu]` for JAX backends.

**Performance tips (from upstream):**

- CPU: JAX `0.4.30` recommended for best CPU perf (see [jax#26021](https://github.com/jax-ml/jax/issues/26021))
- Enable x64: `jax.config.update("jax_enable_x64", True)` or `JAX_ENABLE_X64=True`
- Single-thread CPU often better for small matrices: `scripts/setup_cpu_env.sh` in frax repo
- **JIT the top-most function** — frax does not `@jit` every method by default
- Inside JIT: use `jax.numpy`; outside: use NumPy
- MJCF support: **TODO** in frax (URDF is the supported path today)
- Non-kinematic-chain joints (e.g. gripper): set as `fixed` in URDF for now

**frax also recommends** (same ecosystem):

- [google/mujoco/mjx](https://mujoco.readthedocs.io/en/stable/mjx.html) — parallelized sim (JAX)
- [stack-of-tasks/pinocchio](https://github.com/stack-of-tasks/pinocchio) — reference dynamics
- [pyroki](https://github.com/chungmin99/pyroki) — global IK (JAX)

---

## 7. Bottom Line

| Question | Answer |
|----------|--------|
| Can you use `frax`? | **Yes** |
| Drop into current PyTorch code? | **No** (needs JAX training path) |
| Better than MJX for 2R URDF KNODE? | **Probably yes** |
| Better than BARD if staying PyTorch? | **No** |

**Summary:** `frax` is the **top JAX + URDF** option for batched manipulator dynamics in this project. Pair it with Diffrax for ODE training; keep Pinocchio for data collection and CasADi MPC unless you migrate those too.

---

## Relevant Files in This Repo

| File | Role |
|------|------|
| `knode/model.py` | Current PyTorch ODE + Pinocchio loop to replace in JAX path |
| `robots/planar-2R.urdf` | URDF load target for `frax.Robot(...)` |
| `robots/simulator.py` | Pinocchio ground truth + imperfect dynamics to match |
| `knode/learn.py` | Training loop to reimplement in JAX |
| `mpc/casadi_models.py` | Can stay on Pinocchio/CasADi for MPC |
