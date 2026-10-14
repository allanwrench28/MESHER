# Scancraft

Mesh-to-CAD inspired by Rhino, Geomagic Design X and Fusion 360's *Convert Mesh*. Scancraft turns a triangle
mesh (STL, OBJ, PLY, OFF, 3MF, GLB…) into a **closed, watertight B-rep solid** built from real CAD surfaces
(planes, cylinders, cones, spheres, tori and B-spline patches) instead of thousands of triangles. It then writes
the solid as a **STEP** file. The output opens in any CAD package as a solid body, for example a filleted box
comes out as 26 faces: 6 planes, 12 cylinders and 8 spheres.

## Desktop app

The desktop app lets you convert meshes without remembering command-line options.

1. Download the ZIP for your computer from
   [Releases](https://github.com/allanwrench28/Scancraft/releases):
   **windows-x64**, **linux-x64** (Ubuntu 22.04 or newer), or **macos-arm64** (Apple Silicon).
2. Extract the **entire** archive. Keep the supporting files beside the application.
3. Open `Scancraft/Scancraft.exe` on Windows, `Scancraft.app` on macOS, or `Scancraft/Scancraft` on Linux.
   On Linux you may need to mark the executable as runnable (`chmod +x Scancraft/Scancraft`).
4. Click **Browse…** to choose a mesh, choose where to save the STEP file, and click **Convert to STEP**.

Use **smooth** for most parts; **faceted** preserves triangle faces. The app uses the same repair,
fitting and fallback defaults as the CLI. It stays responsive during conversion, confirms before
replacing files, and warns if the result is not a valid closed solid. Optionally save a JSON report
beside the STEP file. Large meshes can take several minutes; wait for conversion to finish before closing.

The downloadable applications are unsigned and may trigger Windows/macOS security warnings.
Only run downloads you trust and follow your organization's security policy. Intel Macs and other
platforms can use the Python installation below. Linux requires a desktop session and system
OpenGL libraries (on Ubuntu: `sudo apt install libgl1 libxrender1 libxext6`).

### Run from Python

Install Python 3.10 or newer, then install the downloaded wheel with
`python -m pip install /path/to/scancraft-0.2.0-py3-none-any.whl`, or install from this source tree:

```
python -m pip install .
scancraft-gui
```

The GUI uses Tk, which is included in standard Windows/macOS Python installers. On Ubuntu/Debian,
install it with `sudo apt install python3-tk`. The CLI remains available for servers without a desktop.

## Command line and Python API

```
pip install -e .            # needs Python >= 3.10; installs OpenCASCADE via cadquery-ocp
scancraft part.stl part.step --report part.json
```

```python
from scancraft import convert_file, ConvertOptions
result = convert_file("part.stl", "part.step", ConvertOptions(mode="smooth"))
print(result.is_solid, result.report["regions"]["by_kind"])
```

## CLI options

| option | meaning |
| --- | --- |
| `--mode smooth` (default) | fits analytic and B-spline surfaces (similar to Fusion's *prismatic* / organic conversion) |
| `--mode faceted` | one planar face per triangle, with coplanar faces merged (similar to Fusion's *faceted* conversion) |
| `--tolerance` / `--rel-tolerance` | fitting tolerance; the default is 0.1 % of the bounding-box diagonal |
| `--angle-tolerance` | maximum normal deviation in degrees (default 25) |
| `--no-freeform` | uses facets instead of B-spline patches where no primitive fits |
| `--no-fallback` | does not fall back to faceted mode when smooth conversion fails |
| `--report file.json` | writes the detected features, their fit errors, the B-rep statistics and the volume deviation |

If the result is not a valid closed solid, the CLI exits with code 1.

## How it works

1. **Repair** (`mesh_io.py`): merges vertices, removes degenerate and duplicate triangles, fixes winding,
   fills holes and fixes inverted meshes.
2. **Segmentation** (`segmentation.py`). Scancraft finds the large regions first, in the style of Efficient RANSAC:
   * Flat faces of a CAD tessellation are detected first.
   * Next, candidate planes, cylinders, cones, spheres and tori (`primitives.py`) are grown from seeds. The
     largest candidate is accepted first.
   * Boundaries in tangent blend zones are then refined using normal agreement. Leftover triangles are
     absorbed into neighbouring regions.
   * Anything still unexplained becomes a freeform region.
3. **Topology** (`topology.py`): extracts boundary chains, corners and loops between regions. Vertices are snapped
   onto the intersections of the fitted surfaces. When a corner of the design (for example where a fillet meets a
   corner ball) was split into several tiny chains, those chains are merged into one vertex.
4. **B-rep** (`brep.py`): builds an exact OCC surface for each region. Freeform regions are split into
   height-field patches and approximated with bicubic B-splines (a thin-plate spline sampled on an adaptive grid).
   Edges become lines or arcs where possible: exact coaxial circles, then fitted arcs, then B-splines,
   each checked against the surfaces on both sides. Faces are trimmed by the boundary loops, verified against
   interior samples, then sewn into a shell. A solid is made from the shell, and coplanar or co-surface faces
   are merged (`UnifySameDomain`).
5. **Fallback**: when smooth mode cannot produce a valid closed solid, faceted mode is used. Faceted mode always
   gives a valid solid for a watertight input.

## About REA

[REA](https://github.com/morluto/rea) is an agent toolkit for inspecting binaries and applications. You can use it
to study how commercial tools behave, for example their feature recognition, the face counts they produce and
how they handle fillets and freeform surfaces. Scancraft does **not** contain or decompile any proprietary code.
It is a clean-room implementation of the publicly documented approach (primitive segmentation, then surface
fitting, trimming and sewing) on top of OpenCASCADE. To compare results, run REA against a reference tool and
compare the face counts and kinds in the `--report` output.

## Limitations

* Smooth conversion of organic, heavily freeform meshes often fails to sew. In that case Scancraft falls back to
  a faceted solid.
* Large meshes (more than about 100k triangles) are slow. Decimate them first.
* Open meshes produce a shell, not a solid.

## Development

```
pip install -e .[test]
pytest
```

On Linux, exercise the desktop tests with `xvfb-run -a pytest` (requires `python3-tk` and `xvfb`);
without a display, GUI widget tests are skipped.

### Build and publish a release

The **Release** GitHub Actions workflow tests and builds standalone desktop bundles on Windows,
Linux and macOS, and builds a Python wheel. Once the workflow is on the default branch, a maintainer
can select **Actions → Release → Run workflow**, choosing the commit's branch to publish version
`0.2.0`. Alternatively, push a `v0.2.0` tag matching the version in `pyproject.toml`.
Publication only runs after all platform builds succeed and includes SHA-256 checksums.
For subsequent releases, first increase the package version; existing releases are never overwritten.

To build a desktop bundle locally:

```
python -m pip install ".[desktop-build]"
python -m PyInstaller --noconfirm --clean --windowed --name Scancraft --collect-all OCP --collect-all scipy --collect-all trimesh --hidden-import networkx src/scancraft/desktop.py
```

The resulting application is in `dist/Scancraft` (`dist/Scancraft.app` on macOS).
Build on each target platform; PyInstaller does not cross-compile.
