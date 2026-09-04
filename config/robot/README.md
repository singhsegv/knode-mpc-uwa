# Robot configs

- `mujoco-*.json` — MuJoCo plant via `scene` → `robots/scenes/*.xml` (Menagerie UR5e)
- other `*.json` — Pinocchio / URDF via `description` (`robot_descriptions`)

Scenes are not pinocchio-compatible; keep separate configs per backend.
