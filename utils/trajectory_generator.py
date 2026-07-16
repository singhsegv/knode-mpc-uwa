import numpy as np


def circle_trajectory(t, r=0.2, cx=0.5, cy=0.5, omega=0.3):
    """
    Circular trajectories of multiple radii
        - Should capture varying effects of gravity.
    """
    phi = omega * t
    c, s = np.cos(phi), np.sin(phi)
    x = cx + r * c
    y = cy + r * s
    xdot = -r * omega * s
    ydot = r * omega * c
    xddot = -r * omega**2 * c
    yddot = -r * omega**2 * s
    return x, y, xdot, ydot, xddot, yddot


def lemniscate_trajectory(t, a=0.2, cx=0.5, cy=0.5, omega=0.3):
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
    xddot = x_phiphi * omega**2
    yddot = y_phiphi * omega**2
    return x, y, xdot, ydot, xddot, yddot


def line_trajectory(t, x0, y0, x1, y1, omega=0.3):
    """
    Straight-line segment oscillating between endpoints (x0, y0) and (x1, y1).
    """
    phi = omega * t
    s, c = np.sin(phi), np.cos(phi)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    dx = 0.5 * (x1 - x0)
    dy = 0.5 * (y1 - y0)
    x = cx + dx * s
    y = cy + dy * s
    xdot = dx * omega * c
    ydot = dy * omega * c
    xddot = -dx * omega**2 * s
    yddot = -dy * omega**2 * s
    return x, y, xdot, ydot, xddot, yddot
    