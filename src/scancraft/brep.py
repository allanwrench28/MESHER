"""OpenCASCADE B-rep construction: surfaces, edges, faces, sewing and STEP export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import RBFInterpolator, RectBivariateSpline

from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepLib import BRepLib
from OCP.GC import GC_MakeSegment
from OCP.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_ProjectPointOnSurf
from OCP.Geom import (
    Geom_BSplineSurface,
    Geom_Circle,
    Geom_ConicalSurface,
    Geom_CylindricalSurface,
    Geom_Plane,
    Geom_RectangularTrimmedSurface,
    Geom_SphericalSurface,
    Geom_ToroidalSurface,
)
from OCP.GeomAbs import GeomAbs_C2
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Ax2, gp_Ax3, gp_Dir, gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.ShapeFix import ShapeFix_Face, ShapeFix_Shape, ShapeFix_Solid
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_ON, TopAbs_REVERSED, TopAbs_SHELL, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape, TopoDS_Shell, TopoDS_Vertex

from .occ_compat import array1_of_int, array1_of_pnt, array1_of_real, array2_of_pnt, cast_edge, cast_face, cast_shell, cast_solid, set_mode, static
from .primitives import Cone, Cylinder, Plane, Primitive, Sphere, Torus, _orthonormal_frame, _unit
from .topology import OPEN, Topology


def _pnt(p) -> gp_Pnt:
    return gp_Pnt(float(p[0]), float(p[1]), float(p[2]))


def _dir(d) -> gp_Dir:
    d = _unit(np.asarray(d, dtype=float))
    return gp_Dir(float(d[0]), float(d[1]), float(d[2]))


def _ax3(origin, z) -> gp_Ax3:
    z = _unit(np.asarray(z, dtype=float))
    x, _ = _orthonormal_frame(z)
    return gp_Ax3(_pnt(origin), _dir(z), _dir(x))


# ---------------------------------------------------------------------------
# surfaces
# ---------------------------------------------------------------------------


def analytic_surface(primitive: Primitive, region_points: np.ndarray):
    """Create the OCC surface for a fitted primitive."""
    if isinstance(primitive, Plane):
        c = region_points.mean(axis=0)
        origin = c - ((c - primitive.origin) @ primitive.normal) * primitive.normal
        return Geom_Plane(_ax3(origin, primitive.normal))
    if isinstance(primitive, Cylinder):
        h = ((region_points - primitive.point) @ primitive.axis).mean()
        return Geom_CylindricalSurface(_ax3(primitive.point + h * primitive.axis, primitive.axis), primitive.radius)
    if isinstance(primitive, Sphere):
        return Geom_SphericalSurface(_ax3(primitive.center, [0.0, 0.0, 1.0]), primitive.radius)
    if isinstance(primitive, Cone):
        h0 = float(((region_points - primitive.apex) @ primitive.axis).mean())
        h0 = max(h0, 1e-6)
        loc = primitive.apex + h0 * primitive.axis
        return Geom_ConicalSurface(_ax3(loc, primitive.axis), primitive.half_angle, h0 * np.tan(primitive.half_angle))
    if isinstance(primitive, Torus):
        return _torus_surface(primitive, region_points)
    raise TypeError(f"unsupported primitive {primitive!r}")


def _torus_surface(torus: Torus, region_points: np.ndarray):
    """Toroidal surface, trimmed in V around the region when it is a band.

    A torus is periodic in both directions and its V seam (the outer equator)
    is exactly where fillets usually start, so boundary circles on the seam
    make the face side ambiguous.  Restricting V to the region's angular range
    removes that ambiguity.
    """
    surf = Geom_ToroidalSurface(_ax3(torus.center, torus.axis), torus.major_radius, torus.minor_radius)
    d = np.asarray(region_points, dtype=float) - torus.center
    h = d @ torus.axis
    rho = np.linalg.norm(d - np.outer(h, torus.axis), axis=1)
    v = np.sort(np.mod(np.arctan2(h, rho - torus.major_radius), 2 * np.pi))
    if len(v) < 2:
        return surf
    gaps = np.diff(np.r_[v, v[0] + 2 * np.pi])
    k = int(np.argmax(gaps))
    if gaps[k] < np.radians(20):
        return surf  # the region wraps around the tube
    v1 = v[(k + 1) % len(v)]
    v2 = v1 + 2 * np.pi - gaps[k]
    pad = min(0.25 * gaps[k], np.radians(5))
    return Geom_RectangularTrimmedSurface(surf, float(v1 - pad), float(v2 + pad), False, True)


def facet_surface(tri: np.ndarray):
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    if np.linalg.norm(n) < 1e-300:
        return None
    return Geom_Plane(_ax3(tri.mean(axis=0), n))


def surface_normal_sign(surf, points: np.ndarray, normals: np.ndarray, weights: np.ndarray) -> float:
    """+1 when the surface normal agrees with the mesh normals, -1 otherwise."""
    from OCP.GeomLProp import GeomLProp_SLProps

    total = 0.0
    step = max(1, len(points) // 200)
    for p, n, w in zip(points[::step], normals[::step], weights[::step]):
        proj = GeomAPI_ProjectPointOnSurf(_pnt(p), surf)
        if proj.NbPoints() == 0:
            continue
        u, v = proj.LowerDistanceParameters()
        props = GeomLProp_SLProps(surf, u, v, 1, 1e-9)
        if not props.IsNormalDefined():
            continue
        sn = props.Normal()
        total += w * (sn.X() * n[0] + sn.Y() * n[1] + sn.Z() * n[2])
    return 1.0 if total >= 0 else -1.0


def _height_spline_surface(spline, origin, e1, e2, n):
    """Exact OCC B-spline surface of a bicubic height spline ``h(u, v)``.

    ``x = u`` and ``y = v`` are reproduced exactly by placing the control
    points at the Greville abscissae of the knot vectors.
    """
    tx, ty, coef = spline.tck
    nx, ny = len(tx) - 4, len(ty) - 4
    coef = np.asarray(coef).reshape(nx, ny)
    gx = np.array([tx[i + 1 : i + 4].mean() for i in range(nx)])
    gy = np.array([ty[j + 1 : j + 4].mean() for j in range(ny)])
    poles = (
        origin[None, None, :]
        + gx[:, None, None] * e1[None, None, :]
        + gy[None, :, None] * e2[None, None, :]
        + coef[:, :, None] * n[None, None, :]
    )
    ukn, umult = np.unique(tx, return_counts=True)
    vkn, vmult = np.unique(ty, return_counts=True)
    return Geom_BSplineSurface(
        array2_of_pnt(poles),
        array1_of_real(ukn),
        array1_of_real(vkn),
        array1_of_int(umult),
        array1_of_int(vmult),
        3,
        3,
    )


def heightfield_surface(
    points: np.ndarray,
    face_normals: np.ndarray,
    tol: float,
    grid: int = 16,
    max_error: Optional[float] = None,
    max_grid: int = 96,
):
    """Approximate a freeform patch with a bicubic B-spline surface.

    The patch is parameterised over its best-fit plane, which works for any
    patch whose normals stay within a cone (a height field).  Heights are
    interpolated with a thin-plate spline, sampled on a grid and converted
    to an exact B-spline.  The grid is refined until the surface passes within
    ``max_error`` of every sample.  Returns ``(surface, max_error)`` or ``None``.
    """
    if len(points) < 4:
        return None
    max_error = tol if max_error is None else max_error
    n = _unit(face_normals.mean(axis=0))
    if not np.isfinite(n).all() or (face_normals @ n).min() < 0.15:
        return None
    e1, e2 = _orthonormal_frame(n)
    c = points.mean(axis=0)
    d = points - c
    uv = np.c_[d @ e1, d @ e2]
    h = d @ n
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    span = hi - lo
    if (span <= 0).any():
        return None
    lo, hi = lo - 0.05 * span, hi + 0.05 * span
    # thin-plate spline: exact at the samples, smooth (and smoothly
    # extrapolated) everywhere else, also for non-convex patch outlines
    _, keep = np.unique(np.round(uv / max(span.max(), 1e-300), 9), axis=0, return_index=True)
    uv_k, h_k = uv[keep], h[keep]
    try:
        rbf = RBFInterpolator(
            uv_k,
            h_k,
            kernel="thin_plate_spline",
            neighbors=None if len(uv_k) <= 400 else 64,
            smoothing=0.0,
        )
    except (np.linalg.LinAlgError, ValueError):
        return None
    # keep the grid spacing roughly isotropic for elongated patches
    aspect = float(np.sqrt(span[0] / span[1]))
    best = None
    g = grid
    while True:
        nu = int(np.clip(round(g * aspect), 6, max_grid))
        nv = int(np.clip(round(g / aspect), 6, max_grid))
        us = np.linspace(lo[0], hi[0], nu)
        vs = np.linspace(lo[1], hi[1], nv)
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        try:
            hh = rbf(np.c_[uu.ravel(), vv.ravel()]).reshape(nu, nv)
            if not np.isfinite(hh).all():
                return None
            spline = RectBivariateSpline(us, vs, hh, kx=3, ky=3, s=0)
        except (np.linalg.LinAlgError, ValueError):
            return None
        # vertical deviation is an upper bound of the true distance
        err = float(np.abs(spline.ev(uv[:, 0], uv[:, 1]) - h).max())
        if best is None or err < best[1]:
            best = (spline, err)
        if err <= max_error or (nu >= max_grid and nv >= max_grid):
            break
        g *= 2
    try:
        surf = _height_spline_surface(best[0], c, e1, e2, n)
    except Exception:
        return None
    return surf, best[1]


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------


def _make_vertex(p, tol) -> TopoDS_Vertex:
    v = TopoDS_Vertex()
    BRep_Builder().MakeVertex(v, _pnt(p), float(tol))
    return v


def _line_dev(points: np.ndarray) -> float:
    a, b = points[0], points[-1]
    ab = b - a
    L = np.linalg.norm(ab)
    if L < 1e-300:
        return np.inf
    t = ab / L
    d = points - a
    return float(np.linalg.norm(d - np.outer(d @ t, t), axis=1).max())


def _fit_circle_3d(points: np.ndarray):
    """Return (center, normal, radius, max_dev) of a circle through ``points``."""
    c = points.mean(axis=0)
    _, s, vt = np.linalg.svd(points - c, full_matrices=False)
    normal = vt[-1]
    planar_dev = np.abs((points - c) @ normal).max()
    e1, e2 = vt[0], vt[1]
    xy = np.c_[(points - c) @ e1, (points - c) @ e2]
    a = np.hstack([2.0 * xy, np.ones((len(xy), 1))])
    b = (xy**2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cc = sol[:2]
    r2 = sol[2] + cc @ cc
    if r2 <= 0:
        return None
    r = np.sqrt(r2)
    center = c + cc[0] * e1 + cc[1] * e2
    dev = np.abs(np.linalg.norm(xy - cc, axis=1) - r).max()
    # orient the normal so the parameter increases along the chain
    if np.dot(np.cross(points[0] - center, points[1] - center), normal) < 0:
        normal = -normal
    return center, normal, float(r), float(max(dev, planar_dev))


def _curve_on_surfaces(curve, p0: float, p1: float, surfaces, tol: float, n: int) -> bool:
    """True when ``curve`` between ``p0`` and ``p1`` lies within ``tol`` of every surface."""
    if not surfaces:
        return False
    for t in np.linspace(p0, p1, max(n, 5)):
        p = curve.Value(float(t))
        for surf in surfaces:
            proj = GeomAPI_ProjectPointOnSurf(p, surf)
            if proj.NbPoints() == 0 or proj.LowerDistance() > tol:
                return False
    return True


def _arc(points: np.ndarray, closed: bool, center, normal, r):
    """Circle / arc ``(curve, p0, p1)`` through the chain ``points`` or ``None``."""
    normal = _unit(np.asarray(normal, dtype=float))
    x = points[0] - center
    x = _unit(x - (x @ normal) * normal)
    y = np.cross(normal, x)
    rel = points - center
    ang = np.unwrap(np.arctan2(rel @ y, rel @ x))
    total = float(ang[-1] - ang[0])
    if total < 0:
        normal, total = -normal, -total  # the parameter must increase along the chain
    curve = Geom_Circle(gp_Ax2(_pnt(center), _dir(normal), _dir(x)), r)
    if closed:
        return curve, 0.0, 2.0 * np.pi
    if not (1e-9 < total < 2 * np.pi - 1e-9):
        return None
    return curve, 0.0, total


def _circle_curve(points: np.ndarray, closed: bool, max_dev: float):
    """Best-fit circle / arc through a chain: ``(curve, p0, p1)`` or ``None``."""
    circ = _fit_circle_3d(points[:-1] if closed else points)
    if circ is None:
        return None
    center, normal, r, dev = circ
    if dev > max_dev or r > 1e6 * max(max_dev, 1e-12):
        return None
    return _arc(points, closed, center, normal, r)


def _revolution_axis(surf):
    """``(point, direction)`` of a surface of revolution; direction ``None`` for spheres."""
    if isinstance(surf, Geom_RectangularTrimmedSurface):
        surf = surf.BasisSurface()
    if isinstance(surf, Geom_SphericalSurface):
        return np.array(_xyz(surf.Location())), None
    if isinstance(surf, (Geom_Plane, Geom_CylindricalSurface, Geom_ConicalSurface, Geom_ToroidalSurface)):
        ax = surf.Axis()
        return np.array(_xyz(ax.Location())), np.array(_xyz(ax.Direction()))
    return None


def _xyz(p):
    return (p.X(), p.Y(), p.Z())


def _coaxial_circle(points: np.ndarray, closed: bool, surfaces, tol: float, max_dev: float):
    """Exact circle shared by two coaxial surfaces of revolution.

    Planes, spheres, cylinders, cones and tori with a common axis (e.g. a
    fillet between a shaft and a shoulder, or a corner ball and a fillet)
    always meet in circles perpendicular to that axis - also where they are
    tangent and a numerical intersection is ill-conditioned.
    """
    if len(surfaces) != 2:
        return None
    axes = [_revolution_axis(s) for s in surfaces]
    if any(a is None for a in axes):
        return None
    (p0, d0), (p1, d1) = axes
    planes = [isinstance(s, Geom_Plane) for s in surfaces]
    if d0 is None and d1 is None:
        if np.linalg.norm(p1 - p0) < tol:
            return None
        c, a = p0, _unit(p1 - p0)
    elif d0 is None or d1 is None:
        c, a, other = (p0, d1, p1) if d0 is None else (p1, d0, p0)
        is_plane = planes[1] if d0 is None else planes[0]
        if not is_plane:
            off = c - other
            if np.linalg.norm(off - (off @ a) * a) > tol:
                return None
    else:
        if np.linalg.norm(np.cross(d0, d1)) > 1e-3:
            return None
        if all(planes):
            return None
        a = d0
        c = p1 if planes[0] else p0
        if not any(planes):
            off = p1 - p0
            if np.linalg.norm(off - (off @ a) * a) > tol:
                return None
    pts = points[:-1] if closed else points
    d = pts - c
    h = d @ a
    radial = d - np.outer(h, a)
    rho = np.linalg.norm(radial, axis=1)
    hm, r = float(h.mean()), float(rho.mean())
    if r < tol:
        return None
    dev = float(np.sqrt((h - hm) ** 2 + (rho - r) ** 2).max())
    if dev > max_dev:
        return None
    return _arc(points, closed, c + hm * a, a, r)


def chain_edges(
    points: np.ndarray, v_first, v_last, straight: bool, tol: float, surfaces=(), relax: float = 6.0
) -> list:
    """Build the B-rep edge(s) for a chain of points.

    Tries, in order: straight line, circular arc / circle, B-spline.  Falls back
    to a polyline when everything else fails.

    The mesh boundary between two regions zig-zags along triangle edges, most
    visibly between tangent surfaces (fillets).  When ``surfaces`` (the OCC
    surfaces of the regions on both sides) are given, lines, arcs and smooth
    B-splines that deviate up to ``relax * tol`` from the zig-zag are accepted
    as long as they stay within ``tol`` of all those surfaces.
    """
    points = np.asarray(points, dtype=float)
    closed = len(points) > 2 and v_first.IsSame(v_last)
    surfaces = [s for s in surfaces if s is not None]
    n_check = 2 * len(points) + 3

    def edge(curve, p0, p1):
        # a fitted curve may end slightly off the (zig-zag) chain end points:
        # grow the vertex tolerance to cover the gap instead of failing
        for v, t in ((v_first, p0), (v_last, p1)):
            gap = curve.Value(float(t)).Distance(static(BRep_Tool, "Pnt")(v))
            if gap > static(BRep_Tool, "Tolerance")(v):
                if gap > relax * tol:
                    return None
                BRep_Builder().UpdateVertex(v, 1.01 * gap)
        try:
            mk = BRepBuilderAPI_MakeEdge(curve, v_first, v_last, p0, p1)
            if mk.IsDone():
                return mk.Edge()
        except Exception:
            pass
        return None

    if not closed:
        dev = _line_dev(points) if len(points) > 2 else 0.0
        ok = straight and len(points) == 2 or dev <= tol
        if not ok and surfaces and dev <= relax * tol:
            a, b = points[0], points[-1]
            line = GC_MakeSegment(_pnt(a), _pnt(b)).Value()
            ok = _curve_on_surfaces(line, line.FirstParameter(), line.LastParameter(), surfaces, tol, n_check)
        if ok or straight:
            try:
                mk = BRepBuilderAPI_MakeEdge(v_first, v_last)
                if mk.IsDone() and (ok or dev <= tol):
                    return [mk.Edge()]
            except Exception:
                pass
    if len(points) >= 3 and surfaces:
        arc = _coaxial_circle(points, closed, surfaces, tol, relax * tol)
        if arc is not None and _curve_on_surfaces(*arc, surfaces, tol, n_check):
            e = edge(*arc)
            if e is not None:
                return [e]
    if len(points) >= 4:
        arc = _circle_curve(points, closed, tol)
        if arc is None and surfaces:
            arc = _circle_curve(points, closed, relax * tol)
            if arc is not None and not _curve_on_surfaces(*arc, surfaces, tol, n_check):
                arc = None
        if arc is not None:
            e = edge(*arc)
            if e is not None:
                return [e]
    if len(points) >= 3:
        fits = [(relax * tol * 0.5, True)] if surfaces and len(points) >= 5 else []
        fits.append((max(tol * 0.5, 1e-7), False))
        for fit_tol, check in fits:
            try:
                fit = GeomAPI_PointsToBSpline(array1_of_pnt(points), 2 if len(points) < 5 else 3, 8, GeomAbs_C2, fit_tol)
                if not fit.IsDone():
                    continue
                curve = fit.Curve()
                p0, p1 = curve.FirstParameter(), curve.LastParameter()
                if check and not _curve_on_surfaces(curve, p0, p1, surfaces, tol, n_check):
                    continue
                e = edge(curve, p0, p1)
                if e is not None:
                    return [e]
            except Exception:
                pass
    # polyline fallback
    edges = []
    verts = [v_first] + [_make_vertex(p, tol) for p in points[1:-1]] + [v_last]
    for a, b in zip(verts[:-1], verts[1:]):
        mk = BRepBuilderAPI_MakeEdge(a, b)
        if mk.IsDone():
            edges.append(mk.Edge())
    return edges


# ---------------------------------------------------------------------------
# faces and solids
# ---------------------------------------------------------------------------


@dataclass
class BuildResult:
    shape: TopoDS_Shape
    closed: bool
    valid: bool
    n_faces: int
    n_edges: int
    volume: float
    failed_regions: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def count(shape, kind) -> int:
    n = 0
    ex = TopExp_Explorer(shape, kind)
    while ex.More():
        n += 1
        ex.Next()
    return n


def volume(shape) -> float:
    props = GProp_GProps()
    static(BRepGProp, "VolumeProperties")(shape, props)
    return float(props.Mass())


def _wire(edge_lists: list[list]) -> Optional[object]:
    mk = BRepBuilderAPI_MakeWire()
    for edges in edge_lists:
        for e in edges:
            mk.Add(e)
            if not mk.IsDone():
                return None
    return mk.Wire() if mk.IsDone() else None


def _bbox_diag(points: np.ndarray) -> float:
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) if len(points) else 0.0


def build_brep(
    vertices: np.ndarray,
    topo: Topology,
    surfaces: dict,
    tol: float,
    straight_regions: set[int],
    samples: Optional[dict] = None,
) -> BuildResult:
    """Assemble faces into a sewn shell / solid.

    ``samples`` maps region ids to ``(points, normals)`` taken from the
    interior of the region; they are used to pick the correct side of closed
    surfaces and to orient faces along the mesh normals.
    """
    samples = samples or {}
    notes: list[str] = list(topo.notes)
    vtol = max(tol * 0.5, 1e-7)
    occ_vertices: dict[int, TopoDS_Vertex] = {}

    def vertex(i: int) -> TopoDS_Vertex:
        if i not in occ_vertices:
            occ_vertices[i] = _make_vertex(vertices[i], vtol)
        return occ_vertices[i]

    chain_geom: dict[int, list] = {}
    for ch in topo.chains:
        pts = vertices[ch.vertices]
        straight = any(r in straight_regions for r in ch.regions)
        chain_geom[ch.id] = chain_edges(
            pts,
            vertex(ch.vertices[0]),
            vertex(ch.vertices[-1]),
            straight,
            tol,
            surfaces=[] if straight else [surfaces.get(r) for r in ch.regions if r != OPEN],
        )

    faces = []
    failed: list[int] = []
    for rid, surf in surfaces.items():
        loops = topo.loops.get(rid, [])
        try:
            face = _make_face(surf, loops, topo, chain_geom, vertices, tol, samples.get(rid))
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"region {rid}: face construction raised {exc!r}")
            face = None
        if face is None:
            failed.append(rid)
            continue
        faces.append(face)

    sew = BRepBuilderAPI_Sewing(max(tol, 1e-6))
    for f in faces:
        sew.Add(f)
    sew.Perform()
    sewed = sew.SewedShape()

    solids = []
    shells = []
    ex = TopExp_Explorer(sewed, TopAbs_SHELL)
    while ex.More():
        shells.append(cast_shell(ex.Current()))
        ex.Next()
    if not shells and not sewed.IsNull() and sewed.ShapeType() == TopAbs_FACE:
        # a single closed face (e.g. a full sphere) is not wrapped in a shell by sewing
        shell = TopoDS_Shell()
        builder = BRep_Builder()
        builder.MakeShell(shell)
        builder.Add(shell, sewed)
        shells.append(shell)
    closed = bool(shells) and sew.NbFreeEdges() == 0 and not failed
    if closed:
        for sh in shells:
            mk = BRepBuilderAPI_MakeSolid(sh)
            if not mk.IsDone():
                closed = False
                break
            solid = mk.Solid()
            fix = ShapeFix_Solid(solid)
            fix.Perform()
            solid = cast_solid(fix.Solid()) if fix.Solid().ShapeType() == TopAbs_SOLID else solid
            static(BRepLib, "OrientClosedSolid")(solid)
            solids.append(solid)
    parts = solids if closed else (shells or [sewed])
    if len(parts) == 1:
        shape = parts[0]
    else:
        comp = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(comp)
        for p in parts:
            builder.Add(comp, p)
        shape = comp

    if shape.IsNull():
        return BuildResult(shape, False, False, 0, 0, 0.0, failed, notes + ["no faces could be built"])
    shape = refine(shape)
    valid = BRepCheck_Analyzer(shape).IsValid()
    if not valid:
        fixer = ShapeFix_Shape(shape)
        fixer.Perform()
        fixed = fixer.Shape()
        if BRepCheck_Analyzer(fixed).IsValid():
            shape, valid = fixed, True
    vol = volume(shape) if closed else 0.0
    if sew.NbFreeEdges():
        notes.append(f"{sew.NbFreeEdges()} free edges after sewing")
    return BuildResult(
        shape=shape,
        closed=closed,
        valid=bool(valid),
        n_faces=count(shape, TopAbs_FACE),
        n_edges=count(shape, TopAbs_EDGE),
        volume=vol,
        failed_regions=failed,
        notes=notes,
    )


def _trimmed_face(surf, loops, topo, chain_geom, vertices, tol, flip):
    """Face on ``surf`` bounded by ``loops`` (traversed backwards when ``flip``)."""
    if not loops:
        mk = BRepBuilderAPI_MakeFace(surf, max(tol, 1e-7))
        return mk.Face() if mk.IsDone() else None
    wires = []
    sizes = []
    for loop in loops:
        edge_lists = []
        vids = []
        items = [(cid, not fwd) for cid, fwd in reversed(loop)] if flip else loop
        for cid, fwd in items:
            edges = chain_geom[cid]
            if not fwd:
                edges = [cast_edge(e.Reversed()) for e in reversed(edges)]
            edge_lists.append(edges)
            vids.extend(topo.chains[cid].vertices)
        w = _wire(edge_lists)
        if w is None:
            return None
        wires.append(w)
        sizes.append(_bbox_diag(vertices[vids]))
    order = np.argsort(sizes)[::-1]
    mk = BRepBuilderAPI_MakeFace(surf, wires[order[0]], True)
    if not mk.IsDone():
        return None
    for i in order[1:]:
        mk.Add(wires[i])
    fix = ShapeFix_Face(mk.Face())
    fix.SetPrecision(max(tol * 0.1, 1e-7))
    fix.SetMaxTolerance(max(tol * 2, 1e-6))
    # wire orientation is known from the mesh winding: never let ShapeFix flip it
    set_mode(fix, "FixOrientationMode", 0)
    set_mode(fix, "FixAddNaturalBoundMode", 0)
    set_mode(fix, "FixMissingSeamMode", 1)
    set_mode(fix, "FixWireMode", 1)
    fix.Perform()
    return fix.Face()


def _inside_fraction(face, points: np.ndarray, tol: float) -> float:
    """Fraction of ``points`` (on or near the face's surface) inside ``face``."""
    if not len(points):
        return 1.0
    from OCP.BRepTools import BRepTools
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.gp import gp_Pnt2d

    surf = static(BRep_Tool, "Surface")(face)
    umin, umax, vmin, vmax = static(BRepTools, "UVBounds")(face)
    classifier = BRepClass_FaceClassifier()
    inside = 0
    for p in points:
        proj = GeomAPI_ProjectPointOnSurf(_pnt(p), surf)
        if proj.NbPoints() == 0:
            continue
        u, v = proj.LowerDistanceParameters()
        if surf.IsUPeriodic():
            per = surf.UPeriod()
            u = umin + (u - umin) % per
        if surf.IsVPeriodic():
            per = surf.VPeriod()
            v = vmin + (v - vmin) % per
        classifier.Perform(face, gp_Pnt2d(u, v), 1e-9)
        if classifier.State() in (TopAbs_IN, TopAbs_ON):
            inside += 1
    return inside / len(points)


def _make_face(surf, loops, topo, chain_geom, vertices, tol, samples=None):
    """Build a trimmed face on ``surf`` bounded by the region's boundary loops.

    Loops run counter-clockwise around the region seen from the mesh normal
    side.  Closed surfaces (spheres, cylinders...) are ambiguous - a circle
    splits a sphere in two caps - so the face is verified against interior
    sample points of the region and rebuilt from the reversed loops when it
    covers the wrong side.  Finally the face is oriented along the mesh normals.
    """
    pts, nrm = samples if samples is not None else (np.zeros((0, 3)), np.zeros((0, 3)))
    guess = surface_normal_sign(surf, pts, nrm, np.ones(len(pts))) < 0 if len(pts) else False
    best = None
    for flip in (guess, not guess) if loops else (False,):
        face = _trimmed_face(surf, loops, topo, chain_geom, vertices, tol, flip)
        if face is None or face.IsNull():
            continue
        frac = _inside_fraction(face, pts, tol)
        if best is None or frac > best[0]:
            best = (frac, face)
        if frac >= 0.999:
            break
    if best is None:
        return None
    face = best[1]
    if len(pts):
        sign = surface_normal_sign(surf, pts, nrm, np.ones(len(pts)))
        if face.Orientation() == TopAbs_REVERSED:
            sign = -sign
        if sign < 0:
            face = cast_face(face.Reversed())
    return face


def refine(shape):
    """Merge faces lying on the same surface (like FreeCAD's "Refine shape")."""
    try:
        unify = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
        unify.SetLinearTolerance(1e-6)
        unify.Build()
        out = unify.Shape()
        if BRepCheck_Analyzer(out).IsValid() or not BRepCheck_Analyzer(shape).IsValid():
            return out
    except Exception:
        pass
    return shape


def write_step(shape, path: str, schema: str = "AP214") -> None:
    from OCP.Interface import Interface_Static

    static(Interface_Static, "SetCVal")("write.step.schema", schema)
    static(Interface_Static, "SetCVal")("write.step.unit", "MM")
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(str(path))
    if status != IFSelect_RetDone:
        raise IOError(f"failed to write STEP file {path}")
