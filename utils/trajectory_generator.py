import numpy as np


def _with_z(x, y, xdot, ydot, xddot, yddot, cz=0.0, czdot=0.0, czddot=0.0):
    """Attach constant (or specified) z motion to a planar XY trajectory."""
    return x, y, cz, xdot, ydot, czdot, xddot, yddot, czddot


def circle_trajectory(t, r=0.2, cx=0.5, cy=0.5, omega=0.3, cz=0.0, czdot=0.0, czddot=0.0):
    """Circular trajectory in XY with optional constant Z."""
    phi = omega * t
    c, s = np.cos(phi), np.sin(phi)
    x = cx + r * c
    y = cy + r * s
    xdot = -r * omega * s
    ydot = r * omega * c
    xddot = -r * omega ** 2 * c
    yddot = -r * omega ** 2 * s
    return _with_z(x, y, xdot, ydot, xddot, yddot, cz, czdot, czddot)


def lemniscate_trajectory(t, a=0.2, cx=0.5, cy=0.5, omega=0.3, cz=0.0, czdot=0.0, czddot=0.0):
    """Lemniscate of Bernoulli (figure-8) in XY with analytic derivatives."""
    phi = omega * t
    c, s = np.cos(phi), np.sin(phi)
    d = 1.0 + s * s
    d2 = d * d
    d3 = d2 * d

    x = cx + a * c / d
    y = cy + a * s * c / d

    u = s * (d + 2.0 * c * c)
    v = (c * c - s * s) * d - 2.0 * s * s * c * c
    x_phi = -a * u / d2
    y_phi = a * v / d2

    x_phiphi = -a * c * (d * (d + 2.0 * c * c - 2.0 * s * s) - 4.0 * s * s * (d + 2.0 * c * c)) / d3
    y_phiphi = a * (-2.0 * s * c) * (d * (2.0 * d + c * c - s * s) + 2.0 * v) / d3

    xdot = x_phi * omega
    ydot = y_phi * omega
    xddot = x_phiphi * omega ** 2
    yddot = y_phiphi * omega ** 2
    return _with_z(x, y, xdot, ydot, xddot, yddot, cz, czdot, czddot)


def line_trajectory(t, x0, y0, x1, y1, omega=0.3, z0=0.0, z1=0.0):
    """Straight-line segment oscillating between endpoints in XY; linear Z blend."""
    phi = omega * t
    s, c = np.sin(phi), np.cos(phi)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    cz = 0.5 * (z0 + z1)
    dx = 0.5 * (x1 - x0)
    dy = 0.5 * (y1 - y0)
    dz = 0.5 * (z1 - z0)
    x = cx + dx * s
    y = cy + dy * s
    z = cz + dz * s
    xdot = dx * omega * c
    ydot = dy * omega * c
    zdot = dz * omega * c
    xddot = -dx * omega ** 2 * s
    yddot = -dy * omega ** 2 * s
    zddot = -dz * omega ** 2 * s
    return x, y, z, xdot, ydot, zdot, xddot, yddot, zddot


def rectangle_trajectory(t, x0, y0, x1, y1, omega=0.3, cz=0.0, czdot=0.0, czddot=0.0):
    """Trace the perimeter of an axis-aligned rectangle in XY at constant Z."""
    x_min = min(x0, x1)
    x_max = max(x0, x1)
    y_min = min(y0, y1)
    y_max = max(y0, y1)
    w = x_max - x_min
    h = y_max - y_min

    phi = (omega * t) % 4.0

    if phi < 1.0:
        u = phi
        x = x_min
        y = y_min + u * h
        x_phi, y_phi = 0.0, h
    elif phi < 2.0:
        u = phi - 1.0
        x = x_min + u * w
        y = y_max
        x_phi, y_phi = w, 0.0
    elif phi < 3.0:
        u = phi - 2.0
        x = x_max
        y = y_max - u * h
        x_phi, y_phi = 0.0, -h
    else:
        u = phi - 3.0
        x = x_max - u * w
        y = y_min
        x_phi, y_phi = -w, 0.0

    xdot = x_phi * omega
    ydot = y_phi * omega
    xddot = 0.0
    yddot = 0.0
    return _with_z(x, y, xdot, ydot, xddot, yddot, cz, czdot, czddot)


def helix_trajectory(t, r=0.08, cx=0.35, cy=0.15, cz0=0.30, cz1=0.45, omega=0.5, z_omega=0.3):
    """Helical trajectory: circle in XY with sinusoidal Z motion."""
    phi = omega * t
    c, s = np.cos(phi), np.sin(phi)
    x = cx + r * c
    y = cy + r * s
    z = 0.5 * (cz0 + cz1) + 0.5 * (cz1 - cz0) * np.sin(z_omega * t)
    xdot = -r * omega * s
    ydot = r * omega * c
    zdot = 0.5 * (cz1 - cz0) * z_omega * np.cos(z_omega * t)
    xddot = -r * omega ** 2 * c
    yddot = -r * omega ** 2 * s
    zddot = -0.5 * (cz1 - cz0) * z_omega ** 2 * np.sin(z_omega * t)
    return x, y, z, xdot, ydot, zdot, xddot, yddot, zddot


def sample_trajectory(t, spec):
    """
    Sample a 3D task-space trajectory from a JSON spec.
    Returns (x, y, z, xdot, ydot, zdot, xddot, yddot, zddot).
    """
    trajectory = spec["type"]
    omega = spec.get("omega", 0.3)
    cz = spec.get("cz", 0.35)
    czdot = spec.get("czdot", 0.0)
    czddot = spec.get("czddot", 0.0)

    if trajectory == "circle":
        return circle_trajectory(
            t, r=spec["r"], cx=spec["cx"], cy=spec["cy"],
            omega=omega, cz=cz, czdot=czdot, czddot=czddot,
        )
    if trajectory == "lemniscate":
        return lemniscate_trajectory(
            t, a=spec["a"], cx=spec["cx"], cy=spec["cy"],
            omega=omega, cz=cz, czdot=czdot, czddot=czddot,
        )
    if trajectory == "line":
        return line_trajectory(
            t, spec["lx0"], spec["ly0"], spec["lx1"], spec["ly1"],
            omega=omega, z0=spec.get("lz0", cz), z1=spec.get("lz1", cz),
        )
    if trajectory == "rectangle":
        return rectangle_trajectory(
            t, spec["lx0"], spec["ly0"], spec["lx1"], spec["ly1"],
            omega=omega, cz=cz, czdot=czdot, czddot=czddot,
        )
    if trajectory == "helix":
        return helix_trajectory(
            t,
            r=spec.get("r", 0.08),
            cx=spec.get("cx", 0.35),
            cy=spec.get("cy", 0.15),
            cz0=spec.get("cz0", 0.30),
            cz1=spec.get("cz1", 0.45),
            omega=omega,
            z_omega=spec.get("z_omega", 0.3),
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
        return spec["cx"] + spec["r"], spec["cy"], cz

    if trajectory == "lemniscate":
        return spec["cx"] + spec["a"], spec["cy"], cz

    if trajectory == "rectangle":
        x, y, _, _, _, _ = rectangle_trajectory(
            0, spec["lx0"], spec["ly0"], spec["lx1"], spec["ly1"], omega=omega, cz=cz,
        )
        return x, y, cz

    if trajectory == "helix":
        x, y, z, _, _, _, _, _, _ = helix_trajectory(
            0,
            r=spec.get("r", 0.08),
            cx=spec.get("cx", 0.35),
            cy=spec.get("cy", 0.15),
            cz0=spec.get("cz0", 0.30),
            cz1=spec.get("cz1", 0.45),
            omega=omega,
            z_omega=spec.get("z_omega", 0.3),
        )
        return x, y, z

    raise ValueError(f"Unknown trajectory type: {trajectory}")
