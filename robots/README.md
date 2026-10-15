# Robot Models

## CaDeLaC Panda + payload

`robots/panda` is a symlink to the CaDeLaC Franka MJCF. Config `panda.json` uses:

- **Plant:** MuJoCo `scene_no_collision` (+ optional EE box via `robots/panda_payload.py`)
- **MPC nominal:** Pinocchio from `robots/panda/panda.xml` (`pinocchio_mjcf`)

See `CADELAC.md` for training / multi-env EXC eval.

## MuJoCo UR5e scenes

Simulation uses Menagerie UR5e via [`robot_descriptions`](https://github.com/robot-descriptions/robot_descriptions.py) (caches `mujoco_menagerie/universal_robots_ur5e/`), wrapped by project scenes in `robots/scenes/`:

| Scene | Config | Medium |
|---|---|---|
| `ur5e_default.xml` | `mujoco-perfect-ur5` | air (`density=0`) |
| `ur5e_underwater.xml` | `mujoco-underwater-ur5` | water (`density=1000`, `viscosity=0.0009`) |

```python
from robots.simulator import make_robot

robot, robot_cfg, _ = make_robot("mujoco-perfect-ur5")
# or: make_robot("mujoco-underwater-ur5")
```

- **Floor + light:** in the scene XML (floor has normal contact)
- **Gravity:** scene `<option gravity="...">`, overridden by config `plant_gravity` at load
- **Fluids:** MuJoCo global medium (not a finite tank); translucent box in underwater scene is visual only
- **Actuation:** Menagerie PD actuators are disabled; torques go through `qfrc_applied`
- **Deps:** `pip install mujoco robot_descriptions`

## Pinocchio / URDF configs

`ur5.json`, `perfect-ur5.json`, `ur5-mpc.json` still use `"description": "ur5e_description"` for Pinocchio / CasADi paths.
