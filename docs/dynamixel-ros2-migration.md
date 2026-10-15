# Moving the MPC and KNODE stack to a Dynamixel arm on ROS 2

The hardware robot replaces only the MuJoCo plant step. The Acados solver and the KNODE residual stay as they are: each tick they read `[q, q̇]`, solve for a torque, and apply the first torque for one control period.

In `mpc/evaluate_mpc.py` that loop is `mpc.solve` then `plant.plant_step`. On the Dynamixel arm, `plant_step` becomes: write an effort command through ros2_control, wait one period, read joint state back.

## What has to match

| Piece | Stays | Must agree with the arm |
|---|---|---|
| Acados model | Pinocchio ABA + tanh residual, forward Euler, SQP-RTI | URDF joint order, gravity, torque limits |
| KNODE | Residual added to `q̈`, input `q` or `[q, q̇]` | Same `dt` as the command period, tanh MLP with the layer layout in `get_knode_dynamics` |
| Plant | MuJoCo goes away at runtime | Commanded generalized torque `τ` in the Pinocchio joint frame |

The solver outputs torque. Dynamixel servos accept that only in **current control mode** (operating mode 0 on X-series: XM/XH/YM, and some XL330s). Position mode keeps the servo’s internal PID in the loop, so the torque the MPC computed is not the torque the joint applies. Confirm the SKU has current mode before building the stack around this MPC.

## Steps

**1. Fix the model to the mechanism.** Build a URDF whose joint order, signs, and zeros are the ones ros2_control will use. Put inertias and center-of-mass in that file; Pinocchio ABA in `mpc/casadi_dynamics.py` is the nominal model inside the MPC. Point the robot config at that URDF (`pinocchio_mjcf` today, or `buildModelFromUrdf`). Set `model_gravity` to real gravity in the base frame. Set `torque_limits` to continuous torque, not stall torque.

**2. Bring up ros2_control in current mode.** A `SystemInterface` (ROBOTIS `dynamixel_hardware` or equivalent) with Protocol 2.0 sync read/write, baud high enough for the joint count (4 Mbps on the U2D2). Expose state interfaces `position`, `velocity`, and `effort` (present current converted to torque), and a command interface `effort`. Use `joint_state_broadcaster` plus `forward_command_controller` on effort. Add a controller-manager timeout that drops torque enable if commands stop.

**3. Calibrate one joint at a time.** Encoder zero and direction must match the URDF. Fit a torque constant so `τ = Kt · gear · current` in the Pinocchio sign convention: a positive `τ` must accelerate positive `q`. Check gravity compensation at a few postures (`τ ≈ g(q)` from Pinocchio) before any tracking. Log present current against commanded current so you know the current loop’s lag and bias.

**4. Treat velocity as estimated, not as a sensor.** Dynamixel `present_velocity` is coarse. Filter position (or fuse it) into `q̇` before it enters ABA or a `q_qdot` residual. Coriolis and the residual are both sensitive to that noise.

**5. Close the loop outside the realtime hardware cycle.** Keep the Python Acados node off the controller-manager thread. Hardware interface runs faster (200–500 Hz) and holds the last torque. The MPC node runs at one fixed `dt` and must finish `solve()` inside that period. The sim loop uses 10 ms; a 20 ms period is the safer first target on a 4–7 motor bus. `step_size` in the KNODE config, the Acados `dt`, and the ROS command period have to be the same number. The generated C solver can move into a ros2_control controller later if that jitter matters.

**6. Add limits the current OCP does not have.** `AcadosManipulatorControl` bounds only `u`. Before hardware, add stage bounds on `q` and `q̇`, plus a torque slew limit. Start from a position hold, enable current mode in a small box, and command `g(q)` only. Then nominal MPC, then KNODE-MPC.

**7. Collect training data in the same layout the learner already expects.** KNODE integrates `[q, q̇, τ]` with zero-order hold on `τ` (`knode/model.py`). Log commanded generalized torque, not a reconstructed torque from current, at the control `dt`. Excite with a safe current-mode trajectory (gravity compensation plus a low-amplitude Fourier or chirp) over the workspace you will track. Use `residual_input: "q_qdot"` so gearbox friction and cable drag can depend on velocity. Keep the tanh MLP shape that `get_knode_dynamics` hardcodes (layers at indices 0, 2, 4, 6) unless you change that function too.

**8. Validate in that order.** Gravity compensation holds a pose. Nominal Pinocchio MPC tracks a slow joint trajectory with the residual disabled. Replay a logged torque sequence open-loop and compare measured `q̈` to ABA; that gap is what the residual should learn. Train on hardware logs, then enable `enable_knode` on the same slow trajectory. Only then raise speed and leave the calibration box.

## Checklist

- Servos support current control mode, and ros2_control commands effort in that mode.
- URDF joint order, sign, and zero match the hardware interface and the Pinocchio model inside Acados.
- `τ = Kt · current` checked per joint; `torque_limits` are continuous ratings.
- `model_gravity` is real gravity in the base frame.
- Velocity used by MPC and KNODE is filtered.
- Command period = KNODE `step_size` = Acados `dt`, and RTI solve time is below that period.
- Watchdog disables torque if the MPC node stalls.
- Position, velocity, and torque-rate limits exist before the first tracking run.
- Dataset rows are `(q, q̇, τ_commanded)` in the Pinocchio frame.
- Residual net is tanh and matches the layer indexing in `get_knode_dynamics`.
- Bring-up order: hold, gravity compensation, nominal MPC, open-loop residual check, KNODE-MPC.
