# Robot Models

## UR5 (default)

The UR5 uses [`robot_descriptions`](https://github.com/robot-descriptions/robot_descriptions.py):

```python
from robot_descriptions.loaders.pinocchio import load_robot_description

robot = load_robot_description("ur5_official_description")
```

Configured in `config/robot/ur5.json` with `"description": "ur5_official_description"`.

- **End-effector frame:** `tool0` (official ROS-Industrial convention)
- **First run:** downloads and caches the official UR xacro/URDF under `~/.cache/robot_descriptions/`
- **Dependencies:** `pip install robot_descriptions xacrodoc`

## Planar 2R (legacy)

Still loaded from the local URDF at `robots/planar-2R.urdf` via `config/robot/planar.json`.

## Switching robots

Set `"description": "<name>_official_description"` in `config/robot/*.json` for other Universal Robots models supported by `robot_descriptions` (e.g. `ur5e_description`, `ur10_official_description`).
