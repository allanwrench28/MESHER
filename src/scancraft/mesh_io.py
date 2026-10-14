"""Mesh loading and repair."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class RepairReport:
    input_faces: int
    output_faces: int
    removed_degenerate: int = 0
    removed_duplicate: int = 0
    holes_filled: bool = False
    watertight: bool = False
    winding_consistent: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load any mesh format supported by trimesh (STL, OBJ, PLY, OFF, 3MF, GLB...)."""
    mesh = trimesh.load(str(path), force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"{path}: no triangles found")
    return mesh


def repair_mesh(mesh: trimesh.Trimesh, merge_digits: int | None = None) -> tuple[trimesh.Trimesh, RepairReport]:
    """Return a cleaned copy of ``mesh`` suitable for conversion.

    Merges coincident vertices, removes degenerate and duplicate triangles,
    makes the winding consistent with outward normals and fills small holes.
    """
    mesh = mesh.copy()
    report = RepairReport(input_faces=len(mesh.faces), output_faces=len(mesh.faces))

    if merge_digits is None:
        scale = float(np.linalg.norm(mesh.extents)) or 1.0
        merge_digits = max(0, int(-np.floor(np.log10(scale * 1e-7))))
    mesh.merge_vertices(digits_vertex=merge_digits)

    n = len(mesh.faces)
    mesh.update_faces(mesh.nondegenerate_faces())
    report.removed_degenerate = n - len(mesh.faces)
    n = len(mesh.faces)
    mesh.update_faces(mesh.unique_faces())
    report.removed_duplicate = n - len(mesh.faces)
    mesh.remove_unreferenced_vertices()

    trimesh.repair.fix_winding(mesh)
    if not mesh.is_watertight:
        report.holes_filled = bool(trimesh.repair.fill_holes(mesh))
    trimesh.repair.fix_inversion(mesh)
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()

    report.watertight = bool(mesh.is_watertight)
    report.winding_consistent = bool(mesh.is_winding_consistent)
    if not report.watertight:
        report.notes.append(
            "mesh is not watertight after repair; the result will be an open shell"
        )
    report.output_faces = len(mesh.faces)
    return mesh, report
