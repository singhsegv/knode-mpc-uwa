"""
CaDeLaC-style Panda EE payload environments via MuJoCo MjSpec.

Plant models attach a box under `end_effector` with mass U[0, 4] kg and COM
offset U[-0.3, 0.3]^3 m (paper / Schulze et al. IROS 2025). Nominal models keep
the same visual box but zero inertia so residuals match CaDeLaC's split.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import mujoco
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PANDA_DIR = os.path.join(SCRIPT_DIR, "panda")
DEFAULT_SCENE = "scene_no_collision.xml"


@dataclass(frozen=True)
class PayloadSpec:
    mass: float
    pos: Tuple[float, float, float]
    diaginertia: Tuple[float, float, float] = (0.001, 0.001, 0.001)

    @classmethod
    def sample(cls, rng: np.random.Generator, mass_range=(0.0, 4.0), pos_range=(-0.3, 0.3)):
        mass = float(rng.uniform(*mass_range))
        pos = tuple(float(x) for x in rng.uniform(pos_range[0], pos_range[1], size=3))
        return cls(mass=mass, pos=pos)


def panda_scene_path(scene_name: str = DEFAULT_SCENE) -> str:
    path = os.path.join(PANDA_DIR, scene_name if scene_name.endswith(".xml") else f"{scene_name}.xml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Panda scene not found: {path}")
    return path


def panda_mjcf_path(name: str = "panda.xml") -> str:
    path = os.path.join(PANDA_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Panda MJCF not found: {path}")
    return path


def _attach_ee_box(spec: mujoco.MjSpec, payload: Optional[PayloadSpec], with_inertia: bool) -> None:
    ee = next((b for b in spec.bodies if b.name == "end_effector"), None)
    if ee is None:
        raise ValueError("Panda model is missing body 'end_effector'")

    if payload is None:
        return

    box = ee.add_body(name="ee_box", pos=list(payload.pos))
    box.add_geom(
        name="ee_box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.05, 0.05, 0.05],
        contype=0,
        conaffinity=0,
        group=2,
        rgba=[0.5, 0.5, 0.5, 0.5],
    )
    if with_inertia and payload.mass > 0.0:
        box.explicitinertial = True
        box.mass = float(payload.mass)
        box.ipos = np.zeros(3)
        box.inertia = np.asarray(payload.diaginertia, dtype=float)


def compile_panda_model(
    payload: Optional[PayloadSpec] = None,
    with_inertia: bool = True,
    scene_name: str = DEFAULT_SCENE,
) -> mujoco.MjModel:
    """Compile a Panda scene, optionally with an EE payload box."""
    spec = mujoco.MjSpec.from_file(panda_scene_path(scene_name))
    _attach_ee_box(spec, payload, with_inertia=with_inertia)
    return spec.compile()


def build_payload_models(
    num_envs: int,
    seed: int = 0,
    include_nominal: bool = True,
    scene_name: str = DEFAULT_SCENE,
    masses: Optional[Sequence[float]] = None,
    positions: Optional[Sequence[Sequence[float]]] = None,
):
    """
    Build plant (with inertia) and nominal (no inertia) model pairs.

    Returns
    -------
    plant_models : list[MjModel]
    nominal_models : list[MjModel]
    payloads : list[PayloadSpec | None]
    """
    rng = np.random.default_rng(seed)
    payloads = []

    n_rand = num_envs - 1 if include_nominal else num_envs
    if include_nominal:
        payloads.append(None)

    for i in range(n_rand):
        if masses is not None and positions is not None:
            payloads.append(
                PayloadSpec(
                    mass=float(masses[i]),
                    pos=tuple(float(x) for x in positions[i]),
                )
            )
        else:
            payloads.append(PayloadSpec.sample(rng))

    plant_models = [compile_panda_model(p, with_inertia=True, scene_name=scene_name) for p in payloads]
    nominal_models = [compile_panda_model(p, with_inertia=False, scene_name=scene_name) for p in payloads]
    return plant_models, nominal_models, payloads
