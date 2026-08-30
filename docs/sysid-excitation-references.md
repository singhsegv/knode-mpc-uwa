# System Identification & Excitation Trajectory References

References for designing excitation trajectories to learn manipulator dynamics (used when building datasets such as `config/datasets/ur5_sysid_excitation.json`).

---

## Foundational / Classical

### 1. Swevers et al. (1997)

**Optimal robot excitation and identification**  
*IEEE Transactions on Robotics and Automation*, 13(5), 730–740.  
DOI: [10.1109/70.631234](https://doi.org/10.1109/70.631234)

Key idea: **Fourier-series joint trajectories**, periodic excitation, optimize for parameter uncertainty rather than only regressor conditioning.

### 2. Gautier & Khalil (1992)

**Exciting trajectories for the identification of base inertial parameters of robots**  
*The International Journal of Robotics Research*, 11(4), 362–375.  
DOI: [10.1177/027836499201100408](https://doi.org/10.1177/027836499201100408)

Key idea: what makes a trajectory “exciting” for dynamic parameter identification.

### 3. Armstrong (1989)

**On finding exciting trajectories for identification experiments involving systems with nonlinear dynamics**  
*The International Journal of Robotics Research*, 8(6), 28–48.  
DOI: [10.1177/027836498900800603](https://doi.org/10.1177/027836498900800603)

Key idea: nonlinear systems need rich motion, not just slow back-and-forth paths.

### 4. Park (2006)

**Fourier-based optimal excitation trajectories for the dynamic identification of robots**  
*Robotica*, 24(5), 625–633.  
DOI: [10.1017/S0263574706002712](https://doi.org/10.1017/S0263574706002712)

Key idea: practical Fourier parameterization and optimization for robot ID.

---

## Trajectory Optimization / Regressor Conditioning

### 5. Presse & Gautier (1993)

**New criteria of exciting trajectories for robot identification**  
*IEEE International Conference on Robotics and Automation (ICRA)*.  
DOI: [10.1109/ROBOT.1993.292259](https://doi.org/10.1109/ROBOT.1993.292259)

Key idea: alternative excitation criteria beyond simple condition-number minimization.

### 6. Generation of Periodic Trajectories for Optimal Robot Excitation

DOI: [10.1115/1.2831194](https://doi.org/10.1115/1.2831194)

Key idea: finite Fourier series with motion constraints; control over frequency content.

### 7. Condition-number-based excitation optimization (ICRA 2017)

**Generating persistently exciting trajectory based on condition number optimization**  
DOI: [10.1109/ICRA.2017.7989770](https://doi.org/10.1109/ICRA.2017.7989770)

Key idea: optimize trajectories by directly differentiating condition number of the regressor.

---

## More Recent / Practical Extensions

### 8. Rackl, Lampariello & Hirzinger (2012)

**Robot excitation trajectories for dynamic parameter estimation using optimized B-splines**  
*ICRA*.  
DOI: [10.1109/ICRA.2012.6225279](https://doi.org/10.1109/ICRA.2012.6225279)

Key idea: B-spline alternative to Fourier trajectories.

### 9. Analytical constrained excitation optimization (2025)

**An Analytical Approach for Dealing With Explicit Physical Constraints in Excitation Optimization Problems of Dynamic Identification**  
*IEEE Transactions on Robotics*.  
DOI: [10.1109/TRO.2025.3543296](https://doi.org/10.1109/TRO.2025.3543296)

Key idea: feasible Fourier trajectories under joint/velocity/acceleration limits.

### 10. Hands-on / URDF-based excitation (2024)

**Excitation Trajectory Optimization for Dynamic Parameter Identification Using Virtual Constraints in Hands-on Robotic System**  
arXiv: [2401.16566](https://arxiv.org/abs/2401.16566)

Key idea: modern Fourier-series formulation with explicit constraints; useful for URDF-based setups.

---

## What to Read First

| If you want… | Start with |
|---|---|
| The standard recipe | **Swevers (1997)** + **Park (2006)** |
| Why excitation matters | **Gautier & Khalil (1992)** + **Armstrong (1989)** |
| Optimization details | **ICRA 2017 condition-number paper** |
| Modern implementation | **arXiv:2401.16566** or **TRO 2025** |

---

## Survey / Textbook Context

- **Siciliano, Sciavicco, Villani, Oriolo** — *Robotics: Modelling, Planning and Control*  
  Chapter on dynamic identification / experimental identification of manipulators.

- **Khalil & Dombre** — *Modeling, Identification and Control of Robots*  
  Strong on the identification formulation and regressor structure.

---

## Mapping to This Repo

This codebase collects data in **task space** (`utils/improved_trajectory_generator.py` → IK in `utils/training_data_generator.py`), not with optimized joint-space Fourier trajectories. The practical excitation set in `config/datasets/ur5_sysid_excitation.json` applies ID principles indirectly:

- **Frequency diversity** — mix slow and fast `omega`
- **Amplitude diversity** — large slow paths (gravity) vs small fast paths (inertia/Coriolis)
- **Plane diversity** — `xy` / `xz` / `yz` circles and lemniscates
- **Direction reversals** — lemniscate figure-8s
- **3D coupling** — spatial helix with `helix_axis`

For optimal excitation in the classical sense, the next step would be a **joint-space Fourier trajectory** type optimized on regressor condition number.
