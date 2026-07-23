# Learning on Manifolds (Non-Euclidean Spaces) in Robotics

A reading guide for understanding and improving KNODE-style dynamics learning when joint configuration lives on curved spaces (e.g. \(S^1 \times S^1\) for a 2R planar arm) rather than treating everything as flat \(\mathbb{R}^n\).

This document collects foundational references, structured learning papers, and neural-ODE-on-manifold work — organized as a reading path from geometry basics to methods closest to this codebase.

---

## Table of Contents

1. [Foundations: Configuration Spaces in Robotics](#1-foundations-configuration-spaces-in-robotics)
2. [Manifolds & Optimization (Math Toolkit)](#2-manifolds--optimization-math-toolkit)
3. [Geometric Deep Learning (General ML on Non-Euclidean Domains)](#3-geometric-deep-learning-general-ml-on-non-euclidean-domains)
4. [Learning Robot Dynamics with Structure](#4-learning-robot-dynamics-with-structure)
5. [Neural ODEs on Manifolds / Lie Groups](#5-neural-odes-on-manifolds--lie-groups)
6. [Representations for Revolute Joints (\(S^1\))](#6-representations-for-revolute-joints-s1)
7. [Suggested Reading Order](#7-suggested-reading-order)
8. [Mapping to This Codebase](#8-mapping-to-this-codebase)
9. [If You Only Read Three Things](#9-if-you-only-read-three-things)

---

## 1. Foundations: Configuration Spaces in Robotics

Start here to make \(Q\), \(TQ\), \(S^1\), \(SO(2)\), and \(SE(3)\) concrete.

### Lynch & Park — *Modern Robotics* (free online)

- **URL:** https://modernrobotics.northwestern.edu/
- **Read:** Ch. 2–3 (configuration space, degrees of freedom, \(SE(3)\))
- **Why:** Very accessible; best first read for “why angles are not just \(\mathbb{R}^n\)”.
- **For this repo:** The 2R planar arm has \(Q = S^1 \times S^1\); mobile bases / end-effector poses live on \(SE(3)\).

### Murray, Li & Sastry — *A Mathematical Introduction to Robotic Manipulation*

- **URL:** http://www.cds.caltech.edu/~murray/mlswiki
- **Read:** Ch. 2 (configuration space, rigid-body kinematics)
- **Why:** More rigorous than Lynch & Park; good if you want proofs and clean definitions of \(Q\).

### Spong, Hutchinson & Vidyasagar — *Robot Modeling and Control*

- **Why:** Standard undergrad/grad text on Lagrangian dynamics in generalized coordinates.
- **For this repo:** Connects \(q, \dot q, \ddot q\) to \(M(q), C(q,\dot q), g(q)\) — exactly what Pinocchio implements in `robots/simulator.py`.

### Bullo & Lewis — *Geometric Control of Mechanical Systems*

- **Why:** The “serious” book on mechanics on manifolds for control.
- **Read for:** Why states live on \(TQ\), and why energy/conservation structure matters.
- **Note:** Heavier, but the reference if you want the full geometric picture.

---

## 2. Manifolds & Optimization (Math Toolkit)

These explain *why* Euclidean SGD on angles/rotations is awkward, and what “respecting the manifold” means during learning.

### Boumal — *An Introduction to Optimization on Smooth Manifolds* (free PDF)

- **URL:** https://www.nicolasboumal.net/book
- **Why:** Best modern intro to retractions, tangent spaces, Riemannian gradients.
- **Note:** Not robotics-specific, but the clearest explanation of “learn/update on a manifold”.

### Absil, Mahony & Sepulchre — *Optimization Algorithms on Matrix Manifolds*

- **Why:** Classic reference for \(SO(n)\), Stiefel manifolds, etc.
- **Use when:** You work with rotations as matrices instead of Euler angles.

### Lee — *Introduction to Smooth Manifolds* (optional, very math-heavy)

- **Why:** Full differential-geometry definitions of tangent bundle, charts, etc.
- **Note:** Only if you want the formal math underneath everything else.

---

## 3. Geometric Deep Learning (General ML on Non-Euclidean Domains)

### Bronstein, Bruna, Cohen & Veličković — *Geometric Deep Learning* (free book)

- **URL:** https://geometricdeeplearning.com/
- **Why:** Explains the blueprint: symmetries, invariance/equivariance, learning on graphs/manifolds/groups.
- **Note:** Not robotics-only, but gives the conceptual framework behind “don’t treat everything as \(\mathbb{R}^n\)”.

### Cohen & Welling — “Group Equivariant Convolutional Networks” (ICML 2016)

- **Why:** Foundational equivariance paper; useful mindset for “network should respect \(SO(2)\) periodicity”.

---

## 4. Learning Robot Dynamics with Structure

These are the most directly relevant to **learning dynamics mismatch** (rather than just representing poses).

### Lutter, Ritter & Peters — “Deep Lagrangian Networks: Using Physics as Model Prior” (DeLaN, ICLR 2019)

- **URL:** https://proceedings.mlr.press/v97/lutter19a.html
- **Why:** Learns \(L(q,\dot q)\) instead of black-box \(\ddot q = f_\theta(q,\dot q, u)\).
- **For this repo:** Generalized coordinates, but structure-preserving; very close in spirit to nominal + residual KNODE.

### Cranmer et al. — “Lagrangian Neural Networks” (2020)

- **URL:** https://arxiv.org/abs/2003.04630
- **Why:** Simpler entry point than DeLaN.

### Greydanus, Dzamba & Yosinski — “Hamiltonian Neural Networks” (NeurIPS 2019)

- **URL:** https://arxiv.org/abs/1906.01563
- **Why:** Learns energy-conserving flows; good for understanding state on \((q,p) \in T^*Q\).

### Finzi et al. — “Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints” (2020)

- **Why:** Handles constraints / better coordinate choices; relevant if you think in terms of \(Q\) constraints.

### Zhong, Dey & Chakraborty — “Extending Lagrangian and Hamiltonian Neural Networks with Differentiable Contact Models” (2021)

- **Why:** More robotics-flavored extensions of Hamiltonian/Lagrangian neural networks.

### Sanchez-Gonzalez et al. — “Learning to Simulate Complex Physics with Graph Networks” (2020)

- **Why:** Not manifold-theoretic, but important baseline for learned simulators in robotics.

### Gray-box / hybrid modeling (search terms)

For the specific setup in this repo (nominal Pinocchio + learned residual), also search:

- `physics-informed neural network robot dynamics residual learning`
- `gray-box robot dynamics learning`

---

## 5. Neural ODEs on Manifolds / Lie Groups

This codebase uses `odeint_adjoint` on a vector field (`knode/model.py`). These papers ask: *what if the state doesn’t live in \(\mathbb{R}^n\)?*

### Lou et al. — “Neural Manifold Ordinary Differential Equations” (NeurIPS 2020)

- **Why:** Good first paper on NODEs constrained to stay on a manifold.

### Mathieu et al. — “Riemannian Continuous-time Graph Neural Networks” (2020)

- **Why:** Riemannian ODE viewpoint; useful for integration on curved spaces.

### Wotte, Sacks & Manchester — “Optimal Potential Shaping on SE(3) via Neural ODEs on Lie Groups” (IJRR 2024)

- **URL:** https://doi.org/10.1177/02783649241256044
- **Code:** https://github.com/YPWotte/Lie_nODEs
- **Why:** Robotics + neural ODE on a Lie group; working Lie-group NODE pipeline.

### Duong et al. — “Port-Hamiltonian Neural ODE Networks on Lie Groups for Robot Dynamics Learning and Control” (2024)

- **URL:** https://arxiv.org/abs/2401.09520
- **Why:** Probably the closest recent paper to “KNODE but geometry-aware for robots”.
- **Combines:** Neural ODE + energy structure + Lie group state.

### Duruisseaux et al. — “Lie Group Forced Variational Integrator Networks (LieFVIN)” (L4DC 2023)

- **URL:** https://proceedings.mlr.press/v211/duruisseaux23a.html
- **Extended version:** https://arxiv.org/pdf/2211.16006.pdf
- **Why:** Alternative to neural ODE: learn discrete flow maps that preserve symplectic + Lie group structure.
- **Use when:** Open-loop drift / integration cost bothers you.

### Wotte et al. — “Geometric Neural Ordinary Differential Equations: From Manifolds to Lie Groups” (*Entropy*, 2025)

- **URL:** https://www.mdpi.com/1099-4300/27/8/878
- **Why:** Tutorial-style unification of manifold/Lie-group NODE adjoint methods.
- **Use as:** Survey + how-to after reading one or two specific papers.

### Stable Robot Motions on Manifolds — sNMODE (2025)

- **URL:** https://arxiv.org/html/2510.05707v1
- **Why:** Stable dynamical systems on Riemannian manifolds via Lyapunov-constrained neural manifold ODEs.
- **Use when:** You care about MPC/stability on \(S^3\), \(\mathbb{R}^3 \times S^3\), etc.

---

## 6. Representations for Revolute Joints (\(S^1\))

The 2R planar arm is simpler than \(SE(3)\): each joint is \(SO(2) \cong S^1\). These ideas show up everywhere:

| Idea | Where it appears |
|------|------------------|
| \((\sin q, \cos q)\) embedding | Classical in FK/dynamics; standard NN input trick |
| Wrapped angle loss | Robotics tracking papers, orientation estimation |
| Unit complex numbers \(e^{iq} \in S^1\) | Lie group view of revolute joints |
| Fourier features on angles | Periodic function approximation theory |

There is less one canonical “\(S^1 \times S^1\) dynamics learning” paper than for \(SE(3)\), but the **DeLaN / Hamiltonian / port-Hamiltonian** papers in Section 4 all apply — you just have a flat torus instead of a matrix group.

### Orientation on \(SO(3)\) / quaternions (next step up from the 2R arm)

**Zhou et al. — “On the Continuity of Rotation Representations in Neural Networks” (CVPR 2019)**

- **Why:** Famous “6D rotation representation” paper; teaches why naive Euler angles are bad for learning.

**Lezcano-Casado — “Trivializations for Gradient-Based Learning on Manifolds” (NeurIPS 2019)**

- **Why:** Practical trick: learn in tangent space, map back via exponential map.
- **Note:** Very implementable idea for a residual network.

---

## 7. Suggested Reading Order

### Week 1 — Geometry basics

1. *Modern Robotics* Ch. 2–3
2. Boumal Ch. 1–3 (manifolds, tangent spaces, retractions)

### Week 2 — Structured dynamics learning

1. Greydanus et al., Hamiltonian Neural Networks (short, intuitive)
2. Lutter et al., DeLaN (closest to manipulator dynamics learning)
3. This repo’s Pinocchio + KNODE code side-by-side (`knode/model.py`, `robots/simulator.py`)

### Week 3 — Manifold / Lie-group neural ODEs

1. Lou et al., Neural Manifold ODEs
2. Duong et al., port-Hamiltonian Lie-group NODE (robotics)
3. Wotte et al., Geometric NODE tutorial paper

### Optional depth

- Bullo & Lewis (*Geometric Control of Mechanical Systems*)
- Duruisseaux et al., LieFVIN (discrete structure-preserving alternative to `odeint`)
- *Geometric Deep Learning* book (big picture)

---

## 8. Mapping to This Codebase

| Topic | Current code | What papers suggest |
|-------|--------------|---------------------|
| State | \([q, \dot q, \tau] \in \mathbb{R}^6\) in `PlanarArmODEFunc` | State on \(TQ\); \(q \in T^2 = S^1 \times S^1\) |
| Nominal | Pinocchio ABA in `robots/simulator.py` | Analytic Lagrangian on \(Q\) |
| Residual | MLP → \(\Delta \ddot q\) in `knode/model.py` | \(\Delta \tau\) or \(\Delta L\); or tangent-space correction |
| Integration | Euclidean RK4 via `torchdiffeq` | Variational integrator / Lie-group integrator |
| Loss | Euclidean MSE on \(q\) in `knode/learn.py` | Wrapped angle loss; Riemannian metric on \(Q\) |

### Practical improvements suggested by this literature (not yet implemented)

1. **Sin/cos encoding** for \(q\) in the residual network input (keep raw \(q\) for Pinocchio).
2. **Wrapped angle loss** for \(q_1, q_2\); keep Euclidean MSE on \(\dot q\).
3. **Wrap after integration** if rollouts can cross \(\pm\pi\).
4. **Learn \(\Delta\tau\)** instead of \(\Delta \ddot q\) given the gravity-mismatch setup in `get_imperfect_dynamics`.
5. **Structured residual** (e.g. low-rank gravity correction) instead of a fully free MLP.

---

## 9. If You Only Read Three Things

1. **Modern Robotics Ch. 2** — understand \(Q\) and \(TQ\) in robotics language.
2. **Lutter et al., DeLaN** — learn dynamics with physics structure instead of raw \(\mathbb{R}^n\) black boxes.
3. **Duong et al., port-Hamiltonian Lie-group NODE** — neural ODE + robot dynamics + geometry in one paper.

---

## Quick Concept Recap

For a 2R revolute arm:

- **Configuration space:** \(Q = S^1 \times S^1 \cong T^2\) (joint angles mod \(2\pi\)).
- **Dynamical state:** \((q, \dot q) \in TQ\) — pose on the torus plus velocity in the tangent space \(\mathbb{R}^2\).
- **This codebase today:** Uses a local Euclidean chart \([q_1, q_2, \dot q_1, \dot q_2]\), which is fine for trajectories that stay in one patch (e.g. `focussed_line` dataset) but does not explicitly enforce periodicity or manifold constraints in the residual net or loss.
