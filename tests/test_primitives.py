import numpy as np
import trimesh

from scancraft.primitives import fit_cylinder, fit_plane, fit_sphere, fit_torus


def test_fit_plane_and_sphere():
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(-1, 1, (50, 2)), np.full(50, 2.0)]
    plane = fit_plane(pts)
    assert abs(abs(plane.normal[2]) - 1) < 1e-9
    d = rng.normal(size=(200, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    sphere = fit_sphere(d * 3 + [1, 2, 3])
    assert np.allclose(sphere.center, [1, 2, 3], atol=1e-6) and abs(sphere.radius - 3) < 1e-6


def test_fit_cylinder():
    m = trimesh.creation.cylinder(1.5, 4, sections=64)
    side = np.abs(m.face_normals[:, 2]) < 0.1
    cyl = fit_cylinder(m.triangles_center[side], m.face_normals[side])
    assert abs(abs(cyl.axis[2]) - 1) < 1e-3


def test_fit_torus():
    m = trimesh.creation.torus(2.0, 0.5, major_sections=48, minor_sections=24)
    t = fit_torus(m.vertices, m.vertex_normals)
    assert t is not None
    assert abs(t.major_radius - 2.0) < 0.02 and abs(t.minor_radius - 0.5) < 0.02
