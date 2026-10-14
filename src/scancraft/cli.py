"""Command line interface: ``scancraft input.stl output.step``."""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import ConvertOptions, convert_file


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scancraft", description="Convert a triangle mesh into a solid STEP model.")
    ap.add_argument("input", help="mesh file (STL, OBJ, PLY, OFF, 3MF, GLB ...)")
    ap.add_argument("output", help="STEP file to write")
    ap.add_argument("--mode", choices=["smooth", "faceted"], default="smooth",
                    help="smooth: fit planes/cylinders/cones/spheres/tori/B-splines (default); faceted: planar faces only")
    ap.add_argument("--tolerance", type=float, default=None, help="absolute fitting tolerance (model units)")
    ap.add_argument("--rel-tolerance", type=float, default=1e-3, help="tolerance relative to the bounding-box diagonal")
    ap.add_argument("--angle-tolerance", type=float, default=25.0, help="max normal deviation in degrees")
    ap.add_argument("--no-freeform", action="store_true", help="do not fit B-spline surfaces; use facets instead")
    ap.add_argument("--no-fallback", action="store_true", help="do not fall back to faceted mode on failure")
    ap.add_argument("--report", help="write a JSON conversion report to this path")
    args = ap.parse_args(argv)

    options = ConvertOptions(
        mode=args.mode,
        tolerance=args.tolerance,
        rel_tolerance=args.rel_tolerance,
        angle_tolerance_deg=args.angle_tolerance,
        freeform=not args.no_freeform,
        fallback_to_faceted=not args.no_fallback,
    )
    result = convert_file(args.input, args.output, options, args.report)
    rep = result.report
    summary = {
        "mode_used": rep["mode_used"],
        "faces": rep["brep"]["faces"],
        "solid": rep["brep"]["solid"],
        "valid": rep["brep"]["valid"],
        "regions": rep["regions"]["by_kind"],
        "volume_deviation": rep["brep"]["volume_deviation"],
        "notes": rep["notes"],
    }
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0 if result.is_solid else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
