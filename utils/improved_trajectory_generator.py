"""
Similar to trajectory_generator.py but richer trajectories covering full R3 space in better ways.
"""
import numpy as np

def circle_trajectory(t, cx=0.5, cy=0.5, cz=0.0, omega=0.3, axis="xy", r=0.2):
    """Circular trajectory in the XY, XZ, or YZ plane."""
    axis = axis.lower()

    if axis not in ("xy", "xz", "yz"):
        raise ValueError(f"circle axis must be 'xy', 'xz', or 'yz', got {axis!r}")

    phi = omega * t
    c, s = np.cos(phi), np.sin(phi)

    x, y, z = cx, cy, cz
    xdot, ydot, zdot = 0.0, 0.0, 0.0
    xddot, yddot, zddot = 0.0, 0.0, 0.0

    if axis == "xy":
        x = cx + r * c
        y = cy + r * s
        xdot = -r * omega * s
        ydot = r * omega * c
        xddot = -r * omega ** 2 * c
        yddot = -r * omega ** 2 * s
    elif axis == "xz":
        x = cx + r * c
        z = cz + r * s
        xdot = -r * omega * s
        zdot = r * omega * c
        xddot = -r * omega ** 2 * c
        zddot = -r * omega ** 2 * s
    else:  # yz
        y = cy + r * c
        z = cz + r * s
        ydot = -r * omega * s
        zdot = r * omega * c
        yddot = -r * omega ** 2 * c
        zddot = -r * omega ** 2 * s

    return x, y, z, xdot, ydot, zdot, xddot, yddot, zddot


def lemniscate_trajectory(t, cx=0.5, cy=0.5, cz=0.0, omega=0.3, axis="xy", a=0.2):
    """Lemniscate of Bernoulli (figure-8) in the XY, XZ, or YZ plane."""
    axis = axis.lower()

    if axis not in ("xy", "xz", "yz"):
        raise ValueError(f"lemniscate axis must be 'xy', 'xz', or 'yz', got {axis!r}")

    phi = omega * t
    c, s = np.cos(phi), np.sin(phi)
    d = 1.0 + s * s
    d2 = d * d
    d3 = d2 * d

    u = a * c / d
    v = a * s * c / d

    u_term = s * (d + 2.0 * c * c)
    v_term = (c * c - s * s) * d - 2.0 * s * s * c * c
    u_phi = -a * u_term / d2
    v_phi = a * v_term / d2

    u_phiphi = -a * c * (d * (d + 2.0 * c * c - 2.0 * s * s) - 4.0 * s * s * (d + 2.0 * c * c)) / d3
    v_phiphi = a * (-2.0 * s * c) * (d * (2.0 * d + c * c - s * s) + 2.0 * v_term) / d3

    udot = u_phi * omega
    vdot = v_phi * omega
    uddot = u_phiphi * omega ** 2
    vddot = v_phiphi * omega ** 2

    x, y, z = cx, cy, cz
    xdot, ydot, zdot = 0.0, 0.0, 0.0
    xddot, yddot, zddot = 0.0, 0.0, 0.0

    if axis == "xy":
        x = cx + u
        y = cy + v
        xdot, ydot = udot, vdot
        xddot, yddot = uddot, vddot
    elif axis == "xz":
        x = cx + u
        z = cz + v
        xdot, zdot = udot, vdot
        xddot, zddot = uddot, vddot
    else:  # yz
        y = cy + u
        z = cz + v
        ydot, zdot = udot, vdot
        yddot, zddot = uddot, vddot

    return x, y, z, xdot, ydot, zdot, xddot, yddot, zddot


def _helix_frame(helix_axis):
    """Orthonormal (u, v, d): circle lies in span{u,v}, advance along unit axis d."""
    d = np.asarray(helix_axis, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        raise ValueError("helix_axis must be a non-zero vector")
    d = d / norm

    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(ref, d)
    u = u / np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v, d


def helix_trajectory(
    t, cx=0.35, cy=0.15, cz=0.375, omega=0.5, r=0.08, pitch=0.1,
    helix_axis=(0.0, 0.0, 1.0),
):
    """
    3D helix through R^3: circular motion in the plane perpendicular to helix_axis,
    advancing `pitch` meters along that axis per full revolution.
    """
    center = np.array([cx, cy, cz], dtype=float)
    u, v, d = _helix_frame(helix_axis)

    phi = omega * t
    c, s = np.cos(phi), np.sin(phi)
    radial = r * (c * u + s * v)
    advance = (pitch * phi / (2.0 * np.pi)) * d

    pos = center + radial + advance
    vel = r * omega * (-s * u + c * v) + (pitch * omega / (2.0 * np.pi)) * d
    acc = -r * omega ** 2 * radial

    return (*pos, *vel, *acc)


def _parse_helix_axis(spec):
    if "helix_axis" not in spec:
        return np.array([0.0, 0.0, 1.0])
    return np.asarray(spec["helix_axis"], dtype=float)


def sample_trajectory(t, spec):
    """
    Sample a 3D task-space trajectory from a JSON spec.
    Returns (x, y, z, xdot, ydot, zdot, xddot, yddot, zddot).
    """
    trajectory = spec["type"]
    omega = spec.get("omega", 0.3)

    if trajectory == "circle":
        return circle_trajectory(
            t, cx=spec["cx"], cy=spec["cy"], cz=spec["cz"],
            omega=omega, axis=spec["axis"], r=spec["r"],
        )
    if trajectory == "lemniscate":
        return lemniscate_trajectory(
            t, cx=spec["cx"], cy=spec["cy"], cz=spec["cz"],
            omega=omega, axis=spec["axis"], a=spec["a"],
        )
    if trajectory == "helix":
        return helix_trajectory(
            t,
            cx=spec.get("cx", 0.35),
            cy=spec.get("cy", 0.15),
            cz=spec.get("cz", 0.375),
            omega=omega,
            r=spec.get("r", 0.08),
            pitch=spec.get("pitch", 0.1),
            helix_axis=_parse_helix_axis(spec),
        )

    raise ValueError(f"Unknown trajectory type: {trajectory}")


def get_trajectory_start(spec):
    """
    Return task-space start position (x, y, z) for a trajectory spec.
    """
    trajectory = spec["type"]
    omega = spec.get("omega", 0.3)
    cz = spec.get("cz", 0.35)

    if trajectory == "line":
        x = 0.5 * (spec["lx0"] + spec["lx1"])
        y = 0.5 * (spec["ly0"] + spec["ly1"])
        z = 0.5 * (spec.get("lz0", cz) + spec.get("lz1", cz))
        return x, y, z

    if trajectory == "circle":
        axis = spec.get("axis", "xy").lower()
        if axis == "xy":
            return spec["cx"] + spec["r"], spec["cy"], cz
        if axis == "xz":
            return spec["cx"] + spec["r"], spec["cy"], cz
        if axis == "yz":
            return spec["cx"], spec["cy"] + spec["r"], cz
        raise ValueError(f"circle axis must be 'xy', 'xz', or 'yz', got {axis!r}")

    if trajectory == "lemniscate":
        axis = spec.get("axis", "xy").lower()
        if axis == "xy":
            return spec["cx"] + spec["a"], spec["cy"], cz
        if axis == "xz":
            return spec["cx"] + spec["a"], spec["cy"], cz
        if axis == "yz":
            return spec["cx"], spec["cy"] + spec["a"], cz
        raise ValueError(f"lemniscate axis must be 'xy', 'xz', or 'yz', got {axis!r}")

    if trajectory == "helix":
        cx = spec.get("cx", 0.35)
        cy = spec.get("cy", 0.15)
        cz = spec.get("cz", 0.375)
        r = spec.get("r", 0.08)
        u, _, _ = _helix_frame(_parse_helix_axis(spec))
        start = np.array([cx, cy, cz]) + r * u
        return float(start[0]), float(start[1]), float(start[2])

    raise ValueError(f"Unknown trajectory type: {trajectory}")
