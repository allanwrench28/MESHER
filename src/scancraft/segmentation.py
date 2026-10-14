"""Mesh segmentation into surface features.

The mesh is partitioned into connected *regions* of triangles, each of which
is explained by a single analytic surface (plane, cylinder, sphere, cone,
torus).
Triangles that cannot be explained by any primitive are grouped into
*freeform* regions, which are later approximated with B-spline surfaces.

Algorithm (greedy primitive region growing, in the spirit of
Efficient-RANSAC / Variational Shape Approximation):

1. Seeds are visited from the flattest to the most curved triangle.
2. For every seed a plane is grown, and cylinders / spheres / cones / tori are
   estimated from the seed's unlabelled neighbourhood and grown as well.
   A triangle joins a growing region when all its vertices lie within
   ``tolerance`` of the surface and its normal agrees with the surface
   normal within ``angle_tolerance``.
3. The largest candidate wins (curved surfaces must clearly beat planes).
4. Adjacent regions of the same type that fit a single surface are merged,
   left-over triangles are attached to a neighbour if they fit, and the
   rest become freeform regions.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import trimesh

from .primitives import Plane, Primitive, fit_primitive

CURVED_KINDS = ("cylinder", "sphere", "cone", "torus")


@dataclass
class SegmentationParams:
    tolerance: float
    angle_tolerance_deg: float = 25.0
    kinds: tuple[str, ...] = ("plane", "cylinder", "sphere", "cone", "torus")
    neighbourhood_faces: int = 40
    min_curved_faces: int = 6
    curved_advantage: float = 1.5
    complex_advantage: float = 1.1
    sharp_angle_deg: float = 50.0
    max_smooth_facet_faces: int = 4
    design_plane_ratio: float = 4.0
    boundary_iterations: int = 6


@dataclass
class Region:
    id: int
    kind: str
    faces: np.ndarray
    primitive: Optional[Primitive] = None
    max_error: float = 0.0
    rms_error: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "kind": self.kind,
            "triangles": int(len(self.faces)),
            "max_error": float(self.max_error),
            "rms_error": float(self.rms_error),
        }
        if self.primitive is not None:
            d["surface"] = self.primitive.to_dict()
        d.update(self.extra)
        return d


class _MeshData:
    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.V = np.asarray(mesh.vertices, dtype=float)
        self.F = np.asarray(mesh.faces, dtype=np.int64)
        self.FN = np.asarray(mesh.face_normals, dtype=float)
        self.FC = np.asarray(mesh.triangles_center, dtype=float)
        self.area = np.asarray(mesh.area_faces, dtype=float)
        adj = np.asarray(mesh.face_adjacency, dtype=np.int64).reshape(-1, 2)
        angles = np.asarray(mesh.face_adjacency_angles, dtype=float).reshape(-1)
        nf = len(self.F)
        nbrs: list[list[int]] = [[] for _ in range(nf)]
        nbr_angles: list[list[float]] = [[] for _ in range(nf)]
        for (a, b), ang in zip(adj, angles):
            nbrs[a].append(b)
            nbrs[b].append(a)
            nbr_angles[a].append(ang)
            nbr_angles[b].append(ang)
        self.nbrs = [np.array(n, dtype=np.int64) for n in nbrs]
        self.nbr_angles = [np.array(a, dtype=float) for a in nbr_angles]
        edges = np.linalg.norm(self.V[self.F] - self.V[np.roll(self.F, 1, axis=1)], axis=2)
        self.longest_edge = edges.max(axis=1)
        self.roughness = np.array(
            [a.mean() if len(a) else np.pi for a in self.nbr_angles], dtype=float
        )

    def region_samples(self, faces: np.ndarray):
        faces = np.asarray(faces, dtype=np.int64)
        vids = np.unique(self.F[faces].ravel())
        return self.V[vids], self.FN[faces], self.FC[faces]


def _face_ok(data: _MeshData, model: Primitive, faces: np.ndarray, sign: float, tol: float, cos_tol: float) -> np.ndarray:
    faces = np.atleast_1d(faces)
    pts = data.V[data.F[faces]].reshape(-1, 3)
    dist = model.distance(pts).reshape(-1, 3).max(axis=1)
    agree = sign * (model.gradient(data.FC[faces]) * data.FN[faces]).sum(axis=1)
    return (dist <= tol) & (agree >= cos_tol)


def _orientation(data: _MeshData, model: Primitive, faces: np.ndarray) -> float:
    s = (model.gradient(data.FC[faces]) * data.FN[faces]).sum(axis=1)
    return 1.0 if (s * data.area[faces]).sum() >= 0 else -1.0


def _is_flat(data: _MeshData, faces: np.ndarray, tol: float) -> bool:
    """True when ``faces`` lie within ``tol`` of a plane (a curved fit would be degenerate)."""
    pts = data.region_samples(faces)[0]
    plane = fit_primitive("plane", pts)
    return plane is None or plane.distance(pts).max() <= tol


def _is_design_plane(data: _MeshData, faces: np.ndarray, labels: np.ndarray, ratio: float) -> bool:
    """True for exactly flat regions much wider than the triangles around them.

    A flat CAD face is often tessellated with a couple of large triangles next
    to finely tessellated fillets.  Such a region is a genuine planar feature,
    unlike a single facet strip of a coarse cylinder whose neighbours are as
    wide as the strip itself.
    """
    outside = np.unique(np.concatenate([data.nbrs[f] for f in faces]))
    outside = outside[(labels[outside] < 0) & ~np.isin(outside, faces)]
    if not len(outside):
        return True
    pts = data.region_samples(faces)[0]
    d = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    minor = float(np.ptp(d @ vt[1]))
    # mean triangle "width" (height over the longest edge), weighted by size
    width = 2.0 * data.area[outside].sum() / max(data.longest_edge[outside].sum(), 1e-300)
    return minor >= ratio * width


def _swallows_plane(data: _MeshData, plane_best, faces: np.ndarray, model: Primitive, tol: float) -> bool:
    """True when a curved candidate swallows a planar region it fits worse.

    Coarse CAD tessellations put few, large triangles on flat faces; their
    vertices (e.g. the corners of a rectangle) may all lie on a large sphere or
    torus that also reaches into the neighbouring fillets.  The triangle
    centroids reveal the difference: they lie on the plane, not on the curved
    surface.  Only plane regions that make up a large part of the candidate are
    considered, so single facet strips of coarse cylinders are not affected.
    """
    _, _, plane_faces, plane = plane_best
    common = np.intersect1d(plane_faces, faces)
    plane_area = data.area[plane_faces].sum()
    if data.area[common].sum() < 0.5 * plane_area or plane_area < 0.25 * data.area[faces].sum():
        return False
    pts = np.vstack([data.region_samples(common)[0], data.FC[common]])
    curved_err = float(model.distance(pts).max())
    plane_err = float(plane.distance(pts).max())
    return curved_err > 0.25 * tol and plane_err < 0.1 * curved_err


def _grow(data, model, seed, labels, kind, tol, cos_tol, refit=True):
    sign = _orientation(data, model, np.array([seed]))
    if not _face_ok(data, model, np.array([seed]), sign, tol, cos_tol)[0]:
        return None, None
    region = [seed]
    in_region = {seed}
    queue = deque([seed])
    next_refit = 8
    while queue:
        f = queue.popleft()
        cand = [n for n in data.nbrs[f] if labels[n] < 0 and n not in in_region]
        if not cand:
            continue
        cand = np.array(cand, dtype=np.int64)
        ok = _face_ok(data, model, cand, sign, tol, cos_tol)
        for n in cand[ok]:
            in_region.add(int(n))
            region.append(int(n))
            queue.append(int(n))
        if refit and len(region) >= next_refit:
            next_refit = int(len(region) * 1.5) + 1
            pts, nrm, ctr = data.region_samples(np.array(region))
            new = fit_primitive(kind, pts, nrm * sign, ctr)
            if new is not None and new.distance(pts).max() <= max(model.distance(pts).max(), tol):
                model = new
                sign = _orientation(data, model, np.array(region))
    return np.array(sorted(region), dtype=np.int64), model


def _neighbourhood(data: _MeshData, seed: int, labels: np.ndarray, count: int, sharp: float) -> np.ndarray:
    seen = {seed}
    order = [seed]
    queue = deque([seed])
    while queue and len(order) < count:
        f = queue.popleft()
        for n, ang in zip(data.nbrs[f], data.nbr_angles[f]):
            n = int(n)
            if n in seen or labels[n] >= 0 or ang > sharp:
                continue
            seen.add(n)
            order.append(n)
            queue.append(n)
            if len(order) >= count:
                break
    return np.array(order, dtype=np.int64)


def _region_error(data: _MeshData, model: Primitive, faces: np.ndarray) -> tuple[float, float]:
    pts = data.V[np.unique(data.F[faces].ravel())]
    d = model.distance(pts)
    return float(d.max()), float(np.sqrt((d**2).mean()))


def _best_candidate(data: _MeshData, seed: int, labels, params: SegmentationParams, tol, cos_tol, sharp):
    """Best region ``(score, kind, faces, model)`` grown from ``seed`` or ``None``."""
    best = None
    if "plane" in params.kinds:
        faces, model = _grow(data, Plane(data.FC[seed], data.FN[seed]), seed, labels, "plane", tol, cos_tol)
        if faces is not None:
            best = (float(data.area[faces].sum()), "plane", faces, model)
    plane_best = best
    hood = None
    for kind in CURVED_KINDS:
        if kind not in params.kinds:
            continue
        if hood is None:
            hood = _neighbourhood(data, seed, labels, params.neighbourhood_faces, sharp)
        if len(hood) < params.min_curved_faces:
            break
        pts, nrm, ctr = data.region_samples(hood)
        model = fit_primitive(kind, pts, nrm, ctr)
        if model is None:
            continue
        faces, model = _grow(data, model, seed, labels, kind, tol, cos_tol)
        if faces is None or len(faces) < params.min_curved_faces or _is_flat(data, faces, tol):
            continue
        if plane_best is not None and _swallows_plane(data, plane_best, faces, model, tol):
            continue
        score = float(data.area[faces].sum())
        if best is None:
            threshold = 0.0
        elif best[1] == "plane":
            threshold = best[0] * params.curved_advantage
        else:
            # later kinds are more general: they must be clearly better
            threshold = best[0] * params.complex_advantage
        if score > threshold:
            best = (score, kind, faces, model)
    return best


def segment_mesh(mesh: trimesh.Trimesh, params: SegmentationParams) -> tuple[np.ndarray, list[Region]]:
    """Partition ``mesh`` into regions. Returns per-face labels and the regions."""
    data = _MeshData(mesh)
    nf = len(data.F)
    tol = float(params.tolerance)
    cos_tol = float(np.cos(np.radians(params.angle_tolerance_deg)))
    sharp = float(np.radians(params.sharp_angle_deg))
    labels = -np.ones(nf, dtype=np.int64)
    models: dict[int, tuple[str, Primitive]] = {}

    order = np.lexsort((-data.area, np.round(data.roughness, 6)))
    next_id = 0
    flat_cos = float(np.cos(np.radians(0.5)))
    design: set[int] = set()
    if "plane" in params.kinds and params.design_plane_ratio > 0:
        for seed in order:
            seed = int(seed)
            if labels[seed] >= 0:
                continue
            # only exactly parallel triangles: a band of fine fillet triangles
            # may lie within tolerance of a plane, but its normals turn
            faces, model = _grow(data, Plane(data.FC[seed], data.FN[seed]), seed, labels, "plane", tol, flat_cos)
            if faces is not None and _is_design_plane(data, faces, labels, params.design_plane_ratio):
                labels[faces] = next_id
                models[next_id] = ("plane", model)
                design.add(next_id)
                next_id += 1
    # Lazy greedy "largest region first" (as in Efficient RANSAC): candidates
    # are grown from seeds spread over the unlabelled triangles; the largest is
    # re-grown against the current labels and accepted if it still beats every
    # other candidate.  Growing from seeds in mesh order instead would let a
    # small region claim triangles that belong to a large feature.
    dead = np.zeros(nf, dtype=bool)
    while True:
        heap: list[tuple[float, int, int]] = []
        covered = np.zeros(nf, dtype=bool)
        for seed in order:
            seed = int(seed)
            if labels[seed] >= 0 or dead[seed] or covered[seed]:
                continue
            cand = _best_candidate(data, seed, labels, params, tol, cos_tol, sharp)
            if cand is None:
                dead[seed] = True
                continue
            covered[cand[2]] = True
            heapq.heappush(heap, (-cand[0], len(heap), seed))
        accepted = 0
        while heap:
            _, _, seed = heapq.heappop(heap)
            if labels[seed] >= 0:
                continue
            cand = _best_candidate(data, seed, labels, params, tol, cos_tol, sharp)
            if cand is None:
                dead[seed] = True
                continue
            if heap and cand[0] < -heap[0][0]:
                heapq.heappush(heap, (-cand[0], len(heap) + nf, seed))
                continue
            _, kind, faces, model = cand
            labels[faces] = next_id
            models[next_id] = (kind, model)
            next_id += 1
            accepted += 1
        if not accepted:
            break

    labels, models = _merge_regions(data, labels, models, tol, cos_tol)
    labels, models = _release_smooth_facets(data, labels, models, params, keep=design)
    if params.boundary_iterations > 0:
        labels = _refine_boundaries(data, labels, models, tol, cos_tol, params.boundary_iterations)
    labels = _absorb_leftovers(data, labels, models, tol, cos_tol)

    # remaining triangles: connected components become freeform regions
    free = np.flatnonzero(labels < 0)
    if len(free):
        free_set = set(int(f) for f in free)
        while free_set:
            start = free_set.pop()
            comp = [start]
            queue = deque([start])
            while queue:
                f = queue.popleft()
                for n in data.nbrs[f]:
                    n = int(n)
                    if n in free_set:
                        free_set.remove(n)
                        comp.append(n)
                        queue.append(n)
            labels[comp] = next_id
            models[next_id] = ("freeform", None)
            next_id += 1

    # compact ids and build Region objects (refit with all region vertices)
    regions: list[Region] = []
    new_labels = -np.ones(nf, dtype=np.int64)
    for old in sorted(models):
        faces = np.flatnonzero(labels == old)
        if not len(faces):
            continue
        kind, model = models[old]
        rid = len(regions)
        new_labels[faces] = rid
        region = Region(rid, kind, faces, model)
        if model is not None:
            pts, nrm, ctr = data.region_samples(faces)
            sign = _orientation(data, model, faces)
            refit = fit_primitive(kind, pts, nrm * sign, ctr)
            if refit is not None and refit.distance(pts).max() <= model.distance(pts).max():
                region.primitive = refit
            region.max_error, region.rms_error = _region_error(data, region.primitive, faces)
        regions.append(region)
    return new_labels, regions


def _region_adjacency(data: _MeshData, labels: np.ndarray) -> set[tuple[int, int]]:
    adj = np.asarray(data.mesh.face_adjacency).reshape(-1, 2)
    la, lb = labels[adj[:, 0]], labels[adj[:, 1]]
    mask = (la != lb) & (la >= 0) & (lb >= 0)
    pairs = np.sort(np.c_[la[mask], lb[mask]], axis=1)
    return set(map(tuple, np.unique(pairs, axis=0).tolist())) if len(pairs) else set()


def _merge_regions(data, labels, models, tol, cos_tol):
    changed = True
    while changed:
        changed = False
        for a, b in sorted(_region_adjacency(data, labels)):
            if a not in models or b not in models:
                continue
            ka, ma = models[a]
            kb, mb = models[b]
            if ka != kb:
                continue
            faces = np.flatnonzero((labels == a) | (labels == b))
            if ka != "plane" and _is_flat(data, faces, tol):
                continue
            pts, nrm, ctr = data.region_samples(faces)
            sa = _orientation(data, ma, np.flatnonzero(labels == a))
            model = fit_primitive(ka, pts, nrm * sa, ctr)
            if model is None:
                continue
            sign = _orientation(data, model, faces)
            if _face_ok(data, model, faces, sign, tol, cos_tol).all():
                labels[labels == b] = a
                models[a] = (ka, model)
                del models[b]
                changed = True
                break
    return labels, models


def _release_smooth_facets(data, labels, models, params, keep=frozenset()):
    """Drop tiny planar regions that blend smoothly into all their neighbours.

    On a coarsely tessellated freeform surface (e.g. a torus) every quad is
    planar; those quads are tessellation artefacts, not design features, so
    they are handed back to the freeform stage.
    """
    smooth = np.radians(params.angle_tolerance_deg)
    adj = np.asarray(data.mesh.face_adjacency).reshape(-1, 2)
    angles = np.asarray(data.mesh.face_adjacency_angles).reshape(-1)
    la, lb = labels[adj[:, 0]], labels[adj[:, 1]]
    boundary = la != lb
    max_angle: dict[int, float] = {}
    for lab, ang in zip(np.r_[la[boundary], lb[boundary]], np.r_[angles[boundary], angles[boundary]]):
        lab = int(lab)
        max_angle[lab] = max(max_angle.get(lab, 0.0), float(ang))
    counts = np.bincount(labels[labels >= 0], minlength=max(models) + 1 if models else 0)
    for rid, (kind, _) in list(models.items()):
        if kind != "plane" or rid in keep or counts[rid] > params.max_smooth_facet_faces:
            continue
        if max_angle.get(rid, np.pi) < smooth:
            labels[labels == rid] = -1
            del models[rid]
    return labels, models


def _refine_boundaries(data, labels, models, tol, cos_tol, iterations):
    """Move boundary triangles to the neighbouring surface that fits them best.

    Greedy growing assigns the triangles of a tangent blend zone (where a
    fillet meets a flat face or a corner patch) to whichever region reached
    them first, producing ragged boundaries and spurious contacts between
    regions.  Distances cannot tell tangent surfaces apart there, but normals
    can: each boundary triangle goes to the feasible region whose surface
    normal agrees best with the triangle normal.  Region pieces cut off by the
    moves are released and re-absorbed afterwards.
    """
    adj = np.asarray(data.mesh.face_adjacency, dtype=np.int64).reshape(-1, 2)
    signs = {
        rid: _orientation(data, model, np.flatnonzero(labels == rid))
        for rid, (_, model) in models.items()
        if model is not None and (labels == rid).any()
    }
    margin = np.radians(0.5)

    def angle(rid, faces):
        _, model = models[rid]
        c = signs[rid] * (model.gradient(data.FC[faces]) * data.FN[faces]).sum(axis=1)
        return np.arccos(np.clip(c, -1.0, 1.0))

    for _ in range(iterations):
        la, lb = labels[adj[:, 0]], labels[adj[:, 1]]
        mask = (la != lb) & (la >= 0) & (lb >= 0)
        if not mask.any():
            break
        moves: dict[int, int] = {}
        for f, other in np.r_[np.c_[adj[mask, 0], lb[mask]], np.c_[adj[mask, 1], la[mask]]]:
            f, other = int(f), int(other)
            own = int(labels[f])
            if own not in signs or other not in signs:
                continue
            fa = np.array([f])
            if not _face_ok(data, models[other][1], fa, signs[other], tol, cos_tol)[0]:
                continue
            # centroids must fit too: corner planes of coarse tessellations
            if models[other][1].distance(data.FC[fa])[0] > tol:
                continue
            gain = angle(own, fa)[0] - angle(other, fa)[0]
            if gain > margin and gain > moves.get(f, (None, 0.0))[1]:
                moves[f] = (other, gain)
        if not moves:
            break
        for f, (other, _) in moves.items():
            labels[f] = other

    # keep only the largest connected piece of every region
    for rid in list(signs):
        faces = np.flatnonzero(labels == rid)
        if len(faces) < 2:
            continue
        sub = adj[(labels[adj[:, 0]] == rid) & (labels[adj[:, 1]] == rid)]
        graph = trimesh.graph.connected_components(sub, nodes=faces, min_len=1)
        if len(graph) > 1:
            biggest = max(graph, key=lambda c: data.area[c].sum())
            for comp in graph:
                if comp is not biggest:
                    labels[comp] = -1
    return labels


def _absorb_leftovers(data, labels, models, tol, cos_tol):
    signs: dict[int, float] = {}
    for _ in range(4):
        free = np.flatnonzero(labels < 0)
        if not len(free):
            break
        moved = False
        for f in free:
            cands = {int(labels[n]) for n in data.nbrs[f] if labels[n] >= 0}
            best, best_d = None, None
            for c in cands:
                kind, model = models[c]
                if c not in signs:
                    signs[c] = _orientation(data, model, np.flatnonzero(labels == c))
                if _face_ok(data, model, np.array([f]), signs[c], tol * 2.0, cos_tol)[0]:
                    d = model.distance(data.V[data.F[f]]).max()
                    if best is None or d < best_d:
                        best, best_d = c, d
            if best is not None:
                labels[f] = best
                moved = True
        if not moved:
            break
    return labels


def split_by_normal_cone(mesh: trimesh.Trimesh, faces: np.ndarray, max_angle_deg: float = 60.0) -> list[np.ndarray]:
    """Split a set of faces into connected patches whose normals stay within a cone.

    Such patches are height fields over their mean plane and can be
    approximated with a single B-spline surface.
    """
    faces = np.asarray(faces, dtype=np.int64)
    fn = np.asarray(mesh.face_normals)
    area = np.asarray(mesh.area_faces)
    member = set(int(f) for f in faces)
    nbrs: dict[int, list[int]] = {int(f): [] for f in faces}
    for a, b in np.asarray(mesh.face_adjacency).reshape(-1, 2):
        a, b = int(a), int(b)
        if a in member and b in member:
            nbrs[a].append(b)
            nbrs[b].append(a)
    cos_max = np.cos(np.radians(max_angle_deg))
    unassigned = set(member)
    patches = []
    for seed in faces[np.argsort(-area[faces])]:
        seed = int(seed)
        if seed not in unassigned:
            continue
        unassigned.remove(seed)
        axis = fn[seed]
        patch = [seed]
        queue = deque([seed])
        while queue:
            f = queue.popleft()
            for n in nbrs[f]:
                if n in unassigned and fn[n] @ axis >= cos_max:
                    unassigned.remove(n)
                    patch.append(n)
                    queue.append(n)
        patches.append(np.array(sorted(patch), dtype=np.int64))
    return patches
