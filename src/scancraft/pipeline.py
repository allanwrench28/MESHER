"""End-to-end mesh -> B-rep -> STEP conversion pipeline."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from .brep import (
    BuildResult,
    analytic_surface,
    build_brep,
    facet_surface,
    heightfield_surface,
    write_step,
)
from .mesh_io import RepairReport, load_mesh, repair_mesh
from .segmentation import Region, SegmentationParams, segment_mesh, split_by_normal_cone
from .topology import build_topology, contract_short_chains, snap_vertices

MODES = ("smooth", "faceted")


@dataclass
class ConvertOptions:
    """Conversion settings.

    ``tolerance`` is the maximum allowed distance between mesh vertices and
    the fitted surfaces (model units).  When omitted it is
    ``rel_tolerance * bounding-box diagonal``.  Scanned / noisy meshes need a
    larger tolerance than meshes exported from CAD.
    """

    mode: str = "smooth"
    tolerance: Optional[float] = None
    rel_tolerance: float = 1e-3
    angle_tolerance_deg: float = 25.0
    kinds: tuple[str, ...] = ("plane", "cylinder", "sphere", "cone", "torus")
    freeform: bool = True
    freeform_tolerance_factor: float = 3.0
    freeform_max_angle_deg: float = 45.0
    max_snap_factor: float = 4.0
    fallback_to_faceted: bool = True
    repair: bool = True

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")


@dataclass
class ConversionResult:
    shape: object
    regions: list[Region]
    build: BuildResult
    report: dict = field(default_factory=dict)

    @property
    def is_solid(self) -> bool:
        return self.build.closed and self.build.valid

    def write_step(self, path: str | Path) -> None:
        write_step(self.shape, str(path))

    def write_report(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.report, indent=2))


def _resolve_tolerance(mesh: trimesh.Trimesh, options: ConvertOptions) -> float:
    if options.tolerance is not None:
        if options.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        return float(options.tolerance)
    diag = float(np.linalg.norm(mesh.extents))
    return max(options.rel_tolerance * diag, 1e-9)


def _facet_regions(faces: np.ndarray, start_id: int) -> list[Region]:
    return [Region(start_id + i, "facet", np.array([f], dtype=np.int64)) for i, f in enumerate(faces)]


def _surfaces_for(
    mesh: trimesh.Trimesh,
    vertices: np.ndarray,
    regions: list[Region],
    tol: float,
    options: ConvertOptions,
) -> tuple[list[Region], dict]:
    """Create OCC surfaces; freeform regions are approximated or exploded to facets."""
    F = np.asarray(mesh.faces)
    FN = np.asarray(mesh.face_normals)
    out: list[Region] = []
    surfaces: dict[int, object] = {}

    def add(region: Region, surf) -> None:
        region.id = len(out)
        out.append(region)
        surfaces[region.id] = surf

    def add_facets(faces) -> None:
        for f in faces:
            surf = facet_surface(vertices[F[f]])
            if surf is not None:
                add(Region(0, "facet", np.array([f], dtype=np.int64)), surf)

    for region in regions:
        if region.primitive is not None:
            pts = vertices[np.unique(F[region.faces].ravel())]
            add(region, analytic_surface(region.primitive, pts))
        elif region.kind == "freeform" and options.freeform and len(region.faces) >= 4:
            for patch in split_by_normal_cone(mesh, region.faces, options.freeform_max_angle_deg):
                vids = np.unique(F[patch].ravel())
                fit = None
                if len(patch) >= 4:
                    fit = heightfield_surface(
                        vertices[vids], FN[patch], tol, max_error=tol * options.freeform_tolerance_factor
                    )
                if fit is not None and fit[1] <= tol * options.freeform_tolerance_factor:
                    surf, err = fit
                    add(Region(0, "freeform", patch, None, err, 0.0, {"surface": {"type": "bspline"}}), surf)
                else:
                    add_facets(patch)
        else:
            add_facets(region.faces)
    return out, surfaces


def _build(mesh, vertices, regions, surfaces, tol):
    labels = -np.ones(len(mesh.faces), dtype=np.int64)
    straight = set()
    force = set()
    centers = np.asarray(mesh.triangles_center)
    normals = np.asarray(mesh.face_normals)
    areas = np.asarray(mesh.area_faces)
    for r in regions:
        labels[r.faces] = r.id
        if r.kind == "facet":
            straight.add(r.id)
            force.update(int(v) for v in np.asarray(mesh.faces)[r.faces].ravel())
    topo = build_topology(mesh, labels, force_corners=force)
    seg = [
        np.linalg.norm(np.diff(vertices[ch.vertices], axis=0), axis=1)
        for ch in topo.chains
        if not straight.intersection(ch.regions)
    ]
    if seg:
        seg = np.concatenate(seg)
        diag = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        min_len = min(max(3.0 * tol, 2.5 * float(np.median(seg))), 0.05 * diag)
        topo, vertices = contract_short_chains(topo, vertices, min_len, protect=straight)
    on_boundary = np.zeros(len(vertices), dtype=bool)
    for ch in topo.chains:
        on_boundary[ch.vertices] = True
    F = np.asarray(mesh.faces)
    samples = {}
    for r in regions:
        if r.kind == "facet":
            continue
        faces = r.faces[~on_boundary[F[r.faces]].any(axis=1)]
        if not len(faces):
            faces = r.faces
        faces = faces[np.argsort(-areas[faces])][:12]
        # face centroids moved onto the snapped vertices' triangle
        pts = vertices[F[faces]].mean(axis=1)
        samples[r.id] = (pts, normals[faces])
    return build_brep(vertices, topo, surfaces, tol, straight, samples), topo


def convert_mesh(mesh: trimesh.Trimesh, options: ConvertOptions | None = None) -> ConversionResult:
    """Convert a triangle mesh into an OpenCASCADE B-rep shape."""
    options = options or ConvertOptions()
    t0 = time.time()
    if options.repair:
        mesh, repair = repair_mesh(mesh)
    else:
        repair = RepairReport(len(mesh.faces), len(mesh.faces), watertight=bool(mesh.is_watertight))
    tol = _resolve_tolerance(mesh, options)
    vertices = np.asarray(mesh.vertices, dtype=float)

    mode = options.mode
    attempts = [mode] + (["faceted"] if mode != "faceted" and options.fallback_to_faceted else [])
    result = None
    notes: list[str] = []
    for attempt in attempts:
        if attempt == "faceted":
            regions = _facet_regions(np.arange(len(mesh.faces)), 0)
            snapped = vertices
        else:
            _, regions = segment_mesh(
                mesh,
                SegmentationParams(
                    tolerance=tol,
                    angle_tolerance_deg=options.angle_tolerance_deg,
                    kinds=tuple(options.kinds),
                ),
            )
            labels = -np.ones(len(mesh.faces), dtype=np.int64)
            for r in regions:
                labels[r.faces] = r.id
            prims = {r.id: r.primitive for r in regions}
            snapped = snap_vertices(vertices, np.asarray(mesh.faces), labels, prims, tol * options.max_snap_factor)
        regions, surfaces = _surfaces_for(mesh, snapped, regions, tol, options)
        build, topo = _build(mesh, snapped, regions, surfaces, tol)
        result = (attempt, regions, build, topo)
        if build.closed and build.valid:
            break
        if not repair.watertight:
            break  # an open mesh cannot become a closed solid in any mode
        notes.append(f"{attempt} conversion did not produce a valid closed solid")

    mode_used, regions, build, topo = result
    kinds = Counter(r.kind for r in regions)
    mesh_volume = float(mesh.volume) if mesh.is_watertight else None
    report = {
        "mode_requested": mode,
        "mode_used": mode_used,
        "tolerance": tol,
        "repair": repair.to_dict(),
        "mesh": {
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.faces)),
            "volume": mesh_volume,
        },
        "regions": {
            "count": len(regions),
            "by_kind": dict(kinds),
            # facets are reported in aggregate only, they can be numerous
            "features": [r.to_dict() for r in regions if r.kind != "facet"],
        },
        "brep": {
            "solid": bool(build.closed),
            "valid": bool(build.valid),
            "faces": build.n_faces,
            "edges": build.n_edges,
            "volume": build.volume if build.closed else None,
            "volume_deviation": (
                abs(build.volume - mesh_volume) / mesh_volume
                if build.closed and mesh_volume
                else None
            ),
        },
        "notes": notes + build.notes + repair.notes,
        "seconds": round(time.time() - t0, 3),
    }
    return ConversionResult(build.shape, regions, build, report)


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    options: ConvertOptions | None = None,
    report_path: str | Path | None = None,
) -> ConversionResult:
    """Load ``input_path`` (STL/OBJ/PLY/...), convert it and write a STEP file."""
    mesh = load_mesh(input_path)
    result = convert_mesh(mesh, options)
    result.write_step(output_path)
    result.report["input"] = str(input_path)
    result.report["output"] = str(output_path)
    if report_path is not None:
        result.write_report(report_path)
    return result
