"""Small compatibility layer over the OCP (OpenCASCADE) Python bindings.

OCP 7.x exposes static C++ methods with an ``_s`` suffix (``TopoDS.Face_s``)
while some newer builds expose them without it.  ``static`` hides that
difference so the rest of the code base works with either.
"""

from __future__ import annotations

from typing import Any, Callable


def static(cls: Any, name: str) -> Callable[..., Any]:
    """Return the static method ``name`` of ``cls`` regardless of OCP naming."""
    fn = getattr(cls, name + "_s", None)
    if fn is None:
        fn = getattr(cls, name)
    return fn


def set_mode(obj, name: str, value: int) -> bool:
    """Set a ShapeFix ``*Mode`` flag (a property in some OCP builds)."""
    try:
        attr = getattr(type(obj), name, None)
        if isinstance(attr, property) or not callable(getattr(obj, name, None)):
            setattr(obj, name, value)
            return True
    except (AttributeError, TypeError):
        pass
    return False


def cast_face(shape):
    from OCP.TopoDS import TopoDS

    return static(TopoDS, "Face")(shape)


def cast_edge(shape):
    from OCP.TopoDS import TopoDS

    return static(TopoDS, "Edge")(shape)


def cast_shell(shape):
    from OCP.TopoDS import TopoDS

    return static(TopoDS, "Shell")(shape)


def cast_solid(shape):
    from OCP.TopoDS import TopoDS

    return static(TopoDS, "Solid")(shape)


def cast_vertex(shape):
    from OCP.TopoDS import TopoDS

    return static(TopoDS, "Vertex")(shape)


def array1_of_pnt(points):
    """Build a 1-based OCC array of ``gp_Pnt`` from an (n, 3) sequence."""
    from OCP.gp import gp_Pnt

    try:
        from OCP.TColgp import TColgp_Array1OfPnt as Array1
    except ImportError:  # OCP >= 8
        from OCP.collections import Array1_gp_Pnt as Array1
    arr = Array1(1, len(points))
    for i, p in enumerate(points, start=1):
        arr.SetValue(i, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    return arr


def array2_of_pnt(grid):
    """Build a 1-based OCC 2D array of ``gp_Pnt`` from an (nu, nv, 3) array."""
    from OCP.gp import gp_Pnt

    try:
        from OCP.TColgp import TColgp_Array2OfPnt as Array2
    except ImportError:  # OCP >= 8
        from OCP.collections import Array2_gp_Pnt as Array2
    nu, nv = grid.shape[:2]
    arr = Array2(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            p = grid[i, j]
            arr.SetValue(i + 1, j + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    return arr


def array1_of_real(values):
    try:
        from OCP.TColStd import TColStd_Array1OfReal as Array1
    except ImportError:  # OCP >= 8
        from OCP.collections import Array1_double as Array1
    arr = Array1(1, len(values))
    for i, v in enumerate(values, start=1):
        arr.SetValue(i, float(v))
    return arr


def array1_of_int(values):
    try:
        from OCP.TColStd import TColStd_Array1OfInteger as Array1
    except ImportError:  # OCP >= 8
        from OCP.collections import Array1_int as Array1
    arr = Array1(1, len(values))
    for i, v in enumerate(values, start=1):
        arr.SetValue(i, int(v))
    return arr
