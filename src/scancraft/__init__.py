"""Scancraft: convert triangle meshes into clean, watertight B-rep solids (STEP)."""

from .mesh_io import load_mesh, repair_mesh
from .pipeline import ConversionResult, ConvertOptions, convert_file, convert_mesh

__all__ = ["ConversionResult", "ConvertOptions", "convert_file", "convert_mesh", "load_mesh", "repair_mesh"]
__version__ = "0.1.0"
