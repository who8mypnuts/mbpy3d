"""Small 3D vector / matrix helpers (ported from Math3D.pas conventions).

A camera matrix (``vgrads``) is a 3x3 numpy array whose *rows* are the world
space directions of screen X, screen Y and the view axis.
"""
from __future__ import annotations

import math

import numpy as np

Vec3 = np.ndarray
Mat3 = np.ndarray


def normalize(v: Vec3) -> Vec3:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v.copy()


def normalise_matrix_to(length: float, m: Mat3) -> Mat3:
    """`NormaliseMatrixTo`: scale every row of ``m`` to ``length``."""
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m * (length / norms)


def rotate_vector_reverse(v: Vec3, m: Mat3) -> Vec3:
    """`RotateVectorReverse`: v' = v * M (sum of v[i] * row i)."""
    return v @ m


def vgrads_from_angles(ax: float, ay: float, az: float) -> Mat3:
    """View matrix from three rotation angles in radians (rotate around X, then
    Y, then Z), as `CalcVGradsFromHeader8rots`."""
    s1, c1 = math.sin(ax), math.cos(ax)
    s2, c2 = math.sin(ay), math.cos(ay)
    s3, c3 = math.sin(az), math.cos(az)
    m = np.zeros((3, 3))
    for i in range(3):
        x1 = y1 = z1 = 0.0
        if i == 0:
            x1 = 1.0
        elif i == 1:
            y1 = 1.0
        else:
            z1 = 1.0
        x2 = x1
        y2 = c1 * y1 + s1 * z1
        z2 = c1 * z1 - s1 * y1
        x1 = c2 * x2 + s2 * z2
        y1 = y2
        z1 = c2 * z2 - s2 * x2
        m[i] = (c3 * x1 + s3 * y1, c3 * y1 - s3 * x1, z1)
    return m


def angles_from_vgrads(m: Mat3) -> tuple[float, float, float]:
    """Inverse of :func:`vgrads_from_angles` for an orthonormal matrix.

    With c/s the cosines/sines of (ax, ay, az) the rows are
    ``row0 = (c3 c2, -s3 c2, -s2)``, ``row1 = (.., .., -c2 s1)``,
    ``row2 = (.., .., c2 c1)``.
    """
    m = normalise_matrix_to(1.0, np.asarray(m, dtype=float))
    ay = math.asin(max(-1.0, min(1.0, -m[0, 2])))
    c2 = math.cos(ay)
    if abs(c2) > 1e-9:
        az = math.atan2(-m[0, 1], m[0, 0])
        ax = math.atan2(-m[1, 2], m[2, 2])
    else:  # gimbal lock: az = 0, row1 = (-s2 s1, c1, 0)
        az = 0.0
        s2 = -m[0, 2]
        ax = math.atan2(-m[1, 0] * s2, m[1, 1])
    return ax, ay, az


def rotation_about_axis(axis: Vec3, angle: float) -> Mat3:
    """Rodrigues rotation matrix (applied as M @ v)."""
    a = normalize(np.asarray(axis, dtype=float))
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = a
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def build_view_vector_dfov(xa: float, ya: float) -> Vec3:
    """`BuildViewVectorDFOV`: normalize(-sin ya, sin xa, cos xa * cos ya)."""
    v = np.array([-math.sin(ya), math.sin(xa), math.cos(xa) * math.cos(ya)])
    return normalize(v)
