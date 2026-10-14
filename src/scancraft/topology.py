"""Region topology: boundary chains, corner vertices, loops and vertex snapping."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import trimesh

from .primitives import Primitive

OPEN = -1  # pseudo region on the far side of an open mesh boundary


@dataclass
class Chain:
    """A maximal run of mesh edges separating the same two regions.

    ``vertices`` are mesh vertex indices.  A closed chain repeats its first
    vertex at the end.
    """

    id: int
    regions: tuple[int, ...]
    vertices: list[int]

    @property
    def closed(self) -> bool:
        return len(self.vertices) > 2 and self.vertices[0] == self.vertices[-1]


@dataclass
class Topology:
    chains: list[Chain]
    loops: dict[int, list[list[tuple[int, bool]]]]  # region -> loops of (chain id, forward)
    corners: set[int]
    open_edges: int = 0
    notes: list[str] = field(default_factory=list)


def build_topology(mesh: trimesh.Trimesh, labels: np.ndarray, force_corners: Iterable[int] = ()) -> Topology:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    fue = np.asarray(mesh.faces_unique_edges, dtype=np.int64)

    edge_faces: list[list[int]] = [[] for _ in range(len(edges))]
    for f, es in enumerate(fue):
        for e in es:
            edge_faces[e].append(f)

    edge_key: dict[int, tuple[int, ...]] = {}
    open_edges = 0
    for e, fs in enumerate(edge_faces):
        labs = {int(labels[f]) for f in fs}
        if len(fs) == 1:
            labs.add(OPEN)
            open_edges += 1
        if len(labs) > 1:
            edge_key[e] = tuple(sorted(labs))

    vert_bedges: dict[int, list[int]] = defaultdict(list)
    for e in edge_key:
        a, b = edges[e]
        vert_bedges[int(a)].append(e)
        vert_bedges[int(b)].append(e)

    corners: set[int] = set(int(v) for v in force_corners if int(v) in vert_bedges)
    for v, es in vert_bedges.items():
        labs = set()
        for e in es:
            labs.update(edge_key[e])
        if len(labs) >= 3 or len(es) != 2:
            corners.add(v)

    def other(e: int, v: int) -> int:
        a, b = edges[e]
        return int(b) if int(a) == v else int(a)

    visited: set[int] = set()
    chains: list[Chain] = []

    def walk(start_v: int, start_e: int) -> list[int]:
        verts = [start_v]
        e, v = start_e, start_v
        key = edge_key[start_e]
        while True:
            visited.add(e)
            v = other(e, v)
            verts.append(v)
            if v in corners or v == start_v:
                break
            nxt = [x for x in vert_bedges[v] if x not in visited and edge_key[x] == key]
            if not nxt:
                break
            e = nxt[0]
        return verts

    for c in sorted(corners):
        for e in vert_bedges[c]:
            if e in visited:
                continue
            verts = walk(c, e)
            chains.append(Chain(len(chains), edge_key[e], verts))
    for e in edge_key:
        if e in visited:
            continue
        start = int(edges[e][0])
        verts = walk(start, e)
        chains.append(Chain(len(chains), edge_key[e], verts))

    # map oriented vertex pairs to chains
    pair_map: dict[tuple[int, int], tuple[int, bool]] = {}
    for ch in chains:
        for a, b in zip(ch.vertices[:-1], ch.vertices[1:]):
            pair_map[(a, b)] = (ch.id, True)
            pair_map[(b, a)] = (ch.id, False)

    # per region: oriented chains (region on the left of the half-edges)
    region_chains: dict[int, dict[int, bool]] = defaultdict(dict)
    for f, (a, b, c) in enumerate(faces):
        r = int(labels[f])
        for u, v in ((a, b), (b, c), (c, a)):
            hit = pair_map.get((int(u), int(v)))
            if hit is not None and r in chains[hit[0]].regions:
                region_chains[r].setdefault(hit[0], hit[1])

    loops: dict[int, list[list[tuple[int, bool]]]] = {}
    notes: list[str] = []
    for r, oriented in region_chains.items():
        starts: dict[int, list[int]] = defaultdict(list)
        ends: dict[int, int] = {}
        for cid, fwd in oriented.items():
            vs = chains[cid].vertices if fwd else chains[cid].vertices[::-1]
            starts[vs[0]].append(cid)
            ends[cid] = vs[-1]
        unused = set(oriented)
        region_loops = []
        while unused:
            cid = min(unused)
            first_start = (chains[cid].vertices if oriented[cid] else chains[cid].vertices[::-1])[0]
            loop = []
            while cid is not None:
                unused.discard(cid)
                loop.append((cid, oriented[cid]))
                end = ends[cid]
                if end == first_start:
                    break
                cand = [c for c in starts[end] if c in unused]
                cid = cand[0] if cand else None
            if cid is None:
                notes.append(f"region {r}: boundary loop could not be closed")
            region_loops.append(loop)
        loops[r] = region_loops
    return Topology(chains, loops, corners, open_edges, notes)


def snap_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    primitives: dict[int, Optional[Primitive]],
    max_shift: float,
    iterations: int = 12,
) -> np.ndarray:
    """Move vertices onto the fitted surfaces of all regions they belong to.

    Vertices shared by several regions converge to the surfaces'
    intersection (an edge or a corner), which makes edges crisp.  The
    shift of any vertex is limited to ``max_shift``.
    """
    vertices = np.asarray(vertices, dtype=float)
    out = vertices.copy()
    vert_regions: list[set[int]] = [set() for _ in range(len(vertices))]
    for f, tri in enumerate(faces):
        r = int(labels[f])
        if primitives.get(r) is not None:
            for v in tri:
                vert_regions[int(v)].add(r)

    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for v, regs in enumerate(vert_regions):
        if regs:
            groups[tuple(sorted(regs))].append(v)

    for regs, vids in groups.items():
        vids = np.array(vids, dtype=np.int64)
        x0 = vertices[vids]
        prims = [primitives[r] for r in regs]
        if len(prims) == 1:
            x = prims[0].project(x0)
        else:
            x = x0.copy()
            lam = 1e-6
            for _ in range(iterations):
                d = np.stack([p.signed_distance(x) for p in prims], axis=1)  # (n, k)
                g = np.stack([p.gradient(x) for p in prims], axis=1)  # (n, k, 3)
                jtj = np.einsum("nki,nkj->nij", g, g) + lam * np.eye(3)
                jtr = np.einsum("nki,nk->ni", g, d) + lam * (x - x0)
                step = np.linalg.solve(jtj, jtr[..., None])[..., 0]
                x = x - step
                if np.abs(step).max() < 1e-12:
                    break
        shift = np.linalg.norm(x - x0, axis=1)
        ok = shift <= max_shift
        out[vids[ok]] = x[ok]
        # vertices that could not be snapped to all surfaces: project onto the nearest one
        if (~ok).any():
            bad = vids[~ok]
            xb = vertices[bad]
            dists = np.stack([p.distance(xb) for p in prims], axis=1)
            nearest = dists.argmin(axis=1)
            for i, v in enumerate(bad):
                proj = prims[nearest[i]].project(xb[i : i + 1])[0]
                if np.linalg.norm(proj - xb[i]) <= max_shift:
                    out[v] = proj
    return out


def contract_short_chains(
    topo: Topology, vertices: np.ndarray, min_length: float, max_segments: int = 4, protect: Iterable[int] = ()
):
    """Collapse tiny chains into a single corner vertex.

    Where four (or more) faces meet in one point of the design - e.g. the
    corner of a filleted box, where two fillets, a flat face and the corner
    sphere touch - the mesh boundary splits that point into several triple
    junctions joined by very short chains.  Those chains are removed and their
    end points merged, so the B-rep gets a single vertex there.

    Chains touching a region in ``protect`` are kept.  Returns
    ``(topology, vertices)``; ``vertices`` is a modified copy.
    """
    protect = set(protect)
    vertices = np.array(vertices, dtype=float, copy=True)
    chains = [Chain(c.id, c.regions, list(c.vertices)) for c in topo.chains]
    loops = {r: [list(lp) for lp in lps] for r, lps in topo.loops.items()}
    removed: set[int] = set()
    corners = set(topo.corners)

    def length(ch: Chain) -> float:
        p = vertices[ch.vertices]
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    changed = True
    while changed:
        changed = False
        cands = sorted(
            (length(ch), ch.id)
            for ch in chains
            if ch.id not in removed
            and not ch.closed
            and len(ch.vertices) - 1 <= max_segments
            and ch.vertices[0] != ch.vertices[-1]
            and OPEN not in ch.regions
            and not protect.intersection(ch.regions)
        )
        for ln, cid in cands:
            if ln >= min_length:
                break
            ch = chains[cid]
            keep, drop = ch.vertices[0], ch.vertices[-1]
            ends = {keep, drop}
            # another chain joining the same two corners would degenerate
            if any(
                o.id != cid and o.id not in removed and {o.vertices[0], o.vertices[-1]} == ends
                for o in chains
            ):
                continue
            if any(len(lp) == 1 and lp[0][0] == cid for r in ch.regions for lp in loops.get(r, [])):
                continue
            for r in ch.regions:
                loops[r] = [[it for it in lp if it[0] != cid] for lp in loops.get(r, [])]
            removed.add(cid)
            for o in chains:
                if o.id in removed:
                    continue
                if o.vertices[0] == drop:
                    o.vertices[0] = keep
                if o.vertices[-1] == drop:
                    o.vertices[-1] = keep
            vertices[keep] = vertices[[keep, drop]].mean(axis=0)
            corners.discard(drop)
            changed = True
            break

    if not removed:
        return topo, vertices
    remap: dict[int, int] = {}
    kept: list[Chain] = []
    for ch in chains:
        if ch.id in removed:
            continue
        remap[ch.id] = len(kept)
        kept.append(Chain(len(kept), ch.regions, ch.vertices))
    new_loops = {r: [[(remap[c], f) for c, f in lp] for lp in lps if lp] for r, lps in loops.items()}
    notes = list(topo.notes) + [f"merged {len(removed)} tiny boundary chain(s) into corners"]
    return Topology(kept, new_loops, corners, topo.open_edges, notes), vertices
