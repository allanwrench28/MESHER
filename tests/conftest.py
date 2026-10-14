import os
import tempfile

import pytest
import trimesh
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer


def shape_to_mesh(shape, linear=0.01, angular=0.2) -> trimesh.Trimesh:
    """Tessellate an OCC shape into a trimesh (used to build CAD-like test meshes)."""
    BRepMesh_IncrementalMesh(shape, linear, False, angular, True)
    fd, path = tempfile.mkstemp(suffix=".stl")
    os.close(fd)
    try:
        writer = StlAPI_Writer()
        writer.ASCIIMode = False
        writer.Write(shape, path)
        return trimesh.load(path, force="mesh")
    finally:
        os.remove(path)


@pytest.fixture
def occ_mesh():
    return shape_to_mesh
