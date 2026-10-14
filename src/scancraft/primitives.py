"""Analytic surface primitives and robust least-squares fitting.

Every primitive exposes the same small interface used by segmentation,
vertex snapping and B-rep construction:

* ``signed_distance(points)`` - signed distance of points to the surface
* ``gradient(points)`` - unit surface normal direction at the closest point
* ``project(points)`` - closest points on the surface
* ``to_dict()`` - JSON friendly description (used by the feature report)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

EPS = 1e-12


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, EPS)


def _orthonormal_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to ``axis`` and to each other."""
    axis = _unit(axis)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(axis[0]) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(axis, helper))
    e2 = np.cross(axis, e1)
    return e1, e2


def _tilt(axis: np.ndarray, e1: np.ndarray, e2: np.ndarray, ab: np.ndarray) -> np.ndarray:
    return _unit(axis + ab[0] * e1 + ab[1] * e2)


class Primitive:
    kind = "primitive"

    def signed_distance(self, points: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def gradient(self, points: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def distance(self, points: np.ndarray) -> np.ndarray:
        return np.abs(self.signed_distance(np.atleast_2d(points)))

    def project(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return points - self.signed_distance(points)[:, None] * self.gradient(points)

    def to_dict(self) -> dict:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Plane(Primitive):
    origin: np.ndarray
    normal: np.ndarray
    kind = "plane"

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=float)
        self.normal = _unit(self.normal)

    def signed_distance(self, points):
        return (np.atleast_2d(points) - self.origin) @ self.normal

    def gradient(self, points):
        return np.broadcast_to(self.normal, np.atleast_2d(points).shape).copy()

    def to_dict(self):
        return {"type": self.kind, "origin": self.origin.tolist(), "normal": self.normal.tolist()}


@dataclass
class Sphere(Primitive):
    center: np.ndarray
    radius: float
    kind = "sphere"

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.radius = float(self.radius)

    def signed_distance(self, points):
        return np.linalg.norm(np.atleast_2d(points) - self.center, axis=1) - self.radius

    def gradient(self, points):
        return _unit(np.atleast_2d(points) - self.center)

    def to_dict(self):
        return {"type": self.kind, "center": self.center.tolist(), "radius": self.radius}


@dataclass
class Cylinder(Primitive):
    point: np.ndarray
    axis: np.ndarray
    radius: float
    kind = "cylinder"

    def __post_init__(self):
        self.axis = _unit(self.axis)
        point = np.asarray(self.point, dtype=float)
        # keep the axis point canonical (closest to the origin)
        self.point = point - (point @ self.axis) * self.axis
        self.radius = float(self.radius)

    def _radial(self, points):
        d = np.atleast_2d(points) - self.point
        return d - np.outer(d @ self.axis, self.axis)

    def signed_distance(self, points):
        return np.linalg.norm(self._radial(points), axis=1) - self.radius

    def gradient(self, points):
        return _unit(self._radial(points))

    def to_dict(self):
        return {
            "type": self.kind,
            "point": self.point.tolist(),
            "axis": self.axis.tolist(),
            "radius": self.radius,
        }


@dataclass
class Cone(Primitive):
    apex: np.ndarray
    axis: np.ndarray  # points from the apex into the cone
    half_angle: float
    kind = "cone"

    def __post_init__(self):
        self.apex = np.asarray(self.apex, dtype=float)
        self.axis = _unit(self.axis)
        self.half_angle = float(self.half_angle)

    def _hr(self, points):
        d = np.atleast_2d(points) - self.apex
        h = d @ self.axis
        radial = d - np.outer(h, self.axis)
        return h, radial

    def signed_distance(self, points):
        h, radial = self._hr(points)
        rho = np.linalg.norm(radial, axis=1)
        return rho * np.cos(self.half_angle) - h * np.sin(self.half_angle)

    def gradient(self, points):
        _, radial = self._hr(points)
        return _unit(
            np.cos(self.half_angle) * _unit(radial) - np.sin(self.half_angle) * self.axis
        )

    def to_dict(self):
        return {
            "type": self.kind,
            "apex": self.apex.tolist(),
            "axis": self.axis.tolist(),
            "half_angle_deg": float(np.degrees(self.half_angle)),
        }


@dataclass
class Torus(Primitive):
    center: np.ndarray
    axis: np.ndarray
    major_radius: float  # radius of the tube's centre circle
    minor_radius: float  # tube radius
    kind = "torus"

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.axis = _unit(self.axis)
        self.major_radius = float(self.major_radius)
        self.minor_radius = float(self.minor_radius)

    def _tube(self, points):
        """Vector from the nearest point of the centre circle to ``points``."""
        d = np.atleast_2d(points) - self.center
        h = d @ self.axis
        radial = d - np.outer(h, self.axis)
        rho = np.linalg.norm(radial, axis=1)
        return np.outer(rho - self.major_radius, 1.0) * _unit(radial) + np.outer(h, self.axis)

    def signed_distance(self, points):
        return np.linalg.norm(self._tube(points), axis=1) - self.minor_radius

    def gradient(self, points):
        return _unit(self._tube(points))

    def to_dict(self):
        return {
            "type": self.kind,
            "center": self.center.tolist(),
            "axis": self.axis.tolist(),
            "major_radius": self.major_radius,
            "minor_radius": self.minor_radius,
        }


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def fit_plane(points: np.ndarray, normals: Optional[np.ndarray] = None, normal_points=None) -> Optional[Plane]:
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return None
    centroid = points.mean(axis=0)
    _, s, vt = np.linalg.svd(points - centroid, full_matrices=False)
    if len(s) < 2 or s[1] < EPS:
        return None
    normal = vt[-1]
    if normals is not None and len(normals) and (np.asarray(normals) @ normal).sum() < 0:
        normal = -normal
    return Plane(centroid, normal)


def fit_sphere(points: np.ndarray, normals: Optional[np.ndarray] = None, normal_points=None) -> Optional[Sphere]:
    points = np.asarray(points, dtype=float)
    if len(points) < 4:
        return None
    a = np.hstack([2.0 * points, np.ones((len(points), 1))])
    b = (points**2).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    center = sol[:3]
    r2 = sol[3] + center @ center
    if not np.isfinite(r2) or r2 <= 0:
        return None
    sphere = Sphere(center, np.sqrt(r2))

    def residual(x):
        return np.linalg.norm(points - x[:3], axis=1) - x[3]

    try:
        res = least_squares(residual, np.r_[sphere.center, sphere.radius], method="lm")
        if res.success and res.x[3] > 0:
            sphere = Sphere(res.x[:3], res.x[3])
    except (ValueError, np.linalg.LinAlgError):
        pass
    return sphere


def _fit_circle_2d(xy: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
    if len(xy) < 3:
        return None
    a = np.hstack([2.0 * xy, np.ones((len(xy), 1))])
    b = (xy**2).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    c = sol[:2]
    r2 = sol[2] + c @ c
    if not np.isfinite(r2) or r2 <= 0:
        return None
    return c, float(np.sqrt(r2))


def fit_cylinder(points: np.ndarray, normals: Optional[np.ndarray], normal_points=None) -> Optional[Cylinder]:
    points = np.asarray(points, dtype=float)
    if normals is None or len(points) < 5 or len(normals) < 2:
        return None
    normals = _unit(normals)
    w, v = np.linalg.eigh(normals.T @ normals)
    # normals of a cylinder span a plane: one ~zero and two significant eigenvalues
    if w[1] < 1e-3 * max(w[2], EPS):
        return None
    axis = v[:, 0]
    e1, e2 = _orthonormal_frame(axis)
    xy = np.c_[points @ e1, points @ e2]
    circle = _fit_circle_2d(xy)
    if circle is None:
        return None
    c2, r = circle
    point = c2[0] * e1 + c2[1] * e2

    def residual(x):
        a = _tilt(axis, e1, e2, x[2:4])
        p = point + x[0] * e1 + x[1] * e2
        d = points - p
        rad = d - np.outer(d @ a, a)
        return np.linalg.norm(rad, axis=1) - x[4]

    try:
        res = least_squares(residual, np.array([0.0, 0.0, 0.0, 0.0, r]), method="lm")
        if res.success and res.x[4] > 0:
            x = res.x
            return Cylinder(point + x[0] * e1 + x[1] * e2, _tilt(axis, e1, e2, x[2:4]), x[4])
    except (ValueError, np.linalg.LinAlgError):
        pass
    return Cylinder(point, axis, r)


def fit_cone(points: np.ndarray, normals: Optional[np.ndarray], normal_points=None) -> Optional[Cone]:
    """Fit a cone.  ``normal_points`` are the points the ``normals`` belong to
    (e.g. face centroids); they default to ``points`` when lengths match."""
    points = np.asarray(points, dtype=float)
    if normals is None or len(points) < 6 or len(normals) < 3:
        return None
    normals = _unit(normals)
    # all normals make the same angle with the axis: n . a = c
    m = np.hstack([normals, -np.ones((len(normals), 1))])
    _, _, vt = np.linalg.svd(m, full_matrices=False)
    sol = vt[-1]
    na = np.linalg.norm(sol[:3])
    if na < EPS:
        return None
    axis = sol[:3] / na
    c = sol[3] / na
    if abs(c) < 0.02 or abs(c) > 0.98:  # cylinder- or plane-like
        return None
    # every tangent plane passes through the apex
    if normal_points is None:
        if len(normals) != len(points):
            return None
        normal_points = points
    rhs = (normals * np.asarray(normal_points, dtype=float)).sum(axis=1)
    try:
        apex, *_ = np.linalg.lstsq(normals, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if ((points - apex) @ axis).mean() < 0:
        axis = -axis
    d = points - apex
    h = d @ axis
    rho = np.linalg.norm(d - np.outer(h, axis), axis=1)
    alpha = float(np.arctan2(rho, h).mean())
    if not (0.01 < alpha < np.pi / 2 - 0.01):
        return None
    e1, e2 = _orthonormal_frame(axis)

    def residual(x):
        a = _tilt(axis, e1, e2, x[3:5])
        dd = points - x[:3]
        hh = dd @ a
        rr = np.linalg.norm(dd - np.outer(hh, a), axis=1)
        return rr * np.cos(x[5]) - hh * np.sin(x[5])

    try:
        res = least_squares(residual, np.r_[apex, 0.0, 0.0, alpha], method="lm")
        if res.success and 0.01 < res.x[5] < np.pi / 2 - 0.01:
            return Cone(res.x[:3], _tilt(axis, e1, e2, res.x[3:5]), res.x[5])
    except (ValueError, np.linalg.LinAlgError):
        pass
    return Cone(apex, axis, alpha)


def _fit_circle_3d(q: np.ndarray):
    """Circle through 3D points: ``(center, axis, radius, max_residual)``."""
    centroid = q.mean(axis=0)
    _, s, vt = np.linalg.svd(q - centroid, full_matrices=False)
    if len(s) < 2 or s[1] < 1e-3 * max(s[0], EPS):
        return None  # (nearly) collinear points do not define a plane
    axis = vt[-1]
    e1, e2 = _orthonormal_frame(axis)
    circle = _fit_circle_2d(np.c_[(q - centroid) @ e1, (q - centroid) @ e2])
    if circle is None:
        return None
    c2, r = circle
    center = centroid + c2[0] * e1 + c2[1] * e2
    d = q - center
    h = d @ axis
    rho = np.linalg.norm(d - np.outer(h, axis), axis=1)
    return center, axis, r, float(np.sqrt((rho - r) ** 2 + h**2).max())


def fit_torus(points: np.ndarray, normals: Optional[np.ndarray], normal_points=None) -> Optional[Torus]:
    """Fit a torus (e.g. a fillet rolled around a round edge).

    Moving every surface point by the tube radius against its normal lands on
    the torus' centre circle; the tube radius is found by a 1D search that
    makes those points most circular, then everything is refined with
    least squares.  Degenerate fits (sphere- or cylinder-like) are rejected.
    """
    points = np.asarray(points, dtype=float)
    if normals is None or len(points) < 8 or len(normals) < 6:
        return None
    if normal_points is None:
        if len(normals) != len(points):
            return None
        normal_points = points
    normal_points = np.asarray(normal_points, dtype=float)
    normals = _unit(normals)
    size = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if size < EPS:
        return None
    best = None
    for r in np.geomspace(1e-3 * size, 2.0 * size, 48):
        for sign in (1.0, -1.0):
            circ = _fit_circle_3d(normal_points - sign * r * normals)
            if circ is None:
                continue
            if best is None or circ[3] < best[0]:
                best = (circ[3], r, circ)
    if best is None:
        return None
    _, r0, (c0, a0, big_r0, _) = best
    e1, e2 = _orthonormal_frame(a0)

    def residual(x):
        a = _tilt(a0, e1, e2, x[3:5])
        d = points - x[:3]
        h = d @ a
        rho = np.linalg.norm(d - np.outer(h, a), axis=1)
        return np.sqrt((rho - x[5]) ** 2 + h**2) - x[6]

    x = np.r_[c0, 0.0, 0.0, big_r0, r0]
    try:
        res = least_squares(residual, x, method="lm", max_nfev=80)
        if res.success:
            x = res.x
    except (ValueError, np.linalg.LinAlgError):
        pass
    big_r, r = float(x[5]), float(x[6])
    if not (r > 1e-6 * size and big_r > 1.01 * r and big_r < 1e3 * size and r < 1e3 * size):
        return None
    return Torus(x[:3], _tilt(a0, e1, e2, x[3:5]), big_r, r)


FITTERS = {
    "plane": fit_plane,
    "cylinder": fit_cylinder,
    "sphere": fit_sphere,
    "cone": fit_cone,
    "torus": fit_torus,
}


def fit_primitive(kind: str, points: np.ndarray, normals: Optional[np.ndarray] = None, normal_points=None):
    try:
        return FITTERS[kind](points, normals, normal_points)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None
