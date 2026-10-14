import json

import pytest
import trimesh
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

from scancraft import ConvertOptions, convert_mesh
from scancraft.cli import main
from scancraft.occ_compat import cast_edge


def _fillet_all(shape, r):
    mk = BRepFilletAPI_MakeFillet(shape)
    ex = TopExp_Explorer(shape, TopAbs_EDGE)
    while ex.More():
        mk.Add(r, cast_edge(ex.Current()))
        ex.Next()
    return mk.Shape()


def _check(result, kinds, faces, max_dev=0.02):
    rep = result.report
    assert rep["mode_used"] == "smooth"
    assert rep["brep"]["solid"] and rep["brep"]["valid"]
    assert rep["regions"]["by_kind"] == kinds
    assert rep["brep"]["faces"] == faces
    assert rep["brep"]["volume_deviation"] < max_dev


@pytest.mark.parametrize(
    "mesh, kinds, faces",
    [
        (trimesh.creation.box((2, 3, 4)), {"plane": 6}, 6),
        (trimesh.creation.cylinder(1, 3, sections=48), {"plane": 2, "cylinder": 1}, 3),
        (trimesh.creation.icosphere(3, 2.0), {"sphere": 1}, 1),
        (trimesh.creation.cone(1, 2, sections=48), {"plane": 1, "cone": 1}, 2),
        (trimesh.creation.torus(2, 0.5, major_sections=48, minor_sections=24), {"torus": 1}, 1),
    ],
)
def test_primitive_solids(mesh, kinds, faces):
    _check(convert_mesh(mesh), kinds, faces)


def test_plate_with_hole(occ_mesh):
    shape = BRepAlgoAPI_Cut(
        BRepPrimAPI_MakeBox(4, 4, 1).Shape(),
        BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(2, 2, -1), gp_Dir(0, 0, 1)), 0.8, 3).Shape(),
    ).Shape()
    _check(convert_mesh(occ_mesh(shape)), {"plane": 6, "cylinder": 1}, 7)


def test_filleted_box(occ_mesh):
    mesh = occ_mesh(_fillet_all(BRepPrimAPI_MakeBox(4, 3, 2).Shape(), 0.4), 0.005, 0.15)
    _check(convert_mesh(mesh), {"plane": 6, "cylinder": 12, "sphere": 8}, 26)


def test_filleted_cylinder(occ_mesh):
    mesh = occ_mesh(_fillet_all(BRepPrimAPI_MakeCylinder(1.5, 2).Shape(), 0.3), 0.005, 0.15)
    _check(convert_mesh(mesh), {"plane": 2, "cylinder": 1, "torus": 2}, 5)


def test_faceted_mode():
    res = convert_mesh(trimesh.creation.box((2, 3, 4)), ConvertOptions(mode="faceted"))
    assert res.is_solid and res.report["brep"]["faces"] == 6


def test_cli_writes_step_and_report(tmp_path):
    src = tmp_path / "cyl.stl"
    trimesh.creation.cylinder(1, 3, sections=48).export(src)
    out, rep = tmp_path / "cyl.step", tmp_path / "cyl.json"
    assert main([str(src), str(out), "--report", str(rep)]) == 0
    text = out.read_text()
    assert "ISO-10303-21" in text and "CYLINDRICAL_SURFACE" in text
    assert json.loads(rep.read_text())["brep"]["solid"]
