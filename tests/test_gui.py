import time
from types import SimpleNamespace

import pytest
import trimesh

from scancraft import gui


@pytest.fixture
def app(monkeypatch):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("A display is required for desktop tests (use xvfb-run on Linux).")
    root.withdraw()
    for name in ("showerror", "showinfo", "showwarning"):
        monkeypatch.setattr(f"tkinter.messagebox.{name}", lambda *args, **kwargs: None)
    monkeypatch.setattr("tkinter.messagebox.askyesno", lambda *args, **kwargs: True)
    instance = gui.ScancraftApp(root)
    yield instance
    root.destroy()


def wait_for_conversion(app):
    deadline = time.monotonic() + 30
    while app.busy and time.monotonic() < deadline:
        app.root.update()
        time.sleep(0.01)
    assert not app.busy


def test_validate_paths(tmp_path):
    src = tmp_path / "mesh.stl"
    src.touch()
    out = tmp_path / "model.step"
    assert gui.validate_paths(str(src), str(out), True) == (src, out, out.with_suffix(".json"))
    assert gui.validate_paths(str(src), str(out), False)[2] is None


@pytest.mark.parametrize("source,destination", [
    ("", "model.step"), ("missing.stl", "model.step"),
    ("mesh.stl", "model.stl"), ("mesh.stl", "missing/model.step"),
])
def test_invalid_paths(tmp_path, source, destination):
    (tmp_path / "mesh.stl").touch()
    with pytest.raises(ValueError):
        gui.validate_paths(str(tmp_path / source) if source else "", str(tmp_path / destination), False)


@pytest.mark.parametrize("filename", ["model.step", "model.json"])
def test_cannot_overwrite_source(tmp_path, filename):
    src = tmp_path / filename
    src.touch()
    with pytest.raises(ValueError, match="overwrite"):
        gui.validate_paths(str(src), str(tmp_path / "model.step"), True)


def test_cannot_overwrite_source_via_symlink(tmp_path):
    src = tmp_path / "mesh.stl"
    src.touch()
    out = tmp_path / "model.step"
    try:
        out.symlink_to(src)
    except OSError:
        pytest.skip("Symlinks are unavailable.")
    with pytest.raises(ValueError):
        gui.validate_paths(str(src), str(out), False)


def test_choose_source(app, monkeypatch, tmp_path):
    src = tmp_path / "part.stl"
    monkeypatch.setattr("tkinter.filedialog.askopenfilename", lambda **kwargs: str(src))
    app.choose_source()
    assert app.source.get() == str(src)
    assert app.destination.get() == str(src.with_suffix(".step"))


def test_desktop_conversion(app, tmp_path):
    src, out = tmp_path / "box.stl", tmp_path / "box.step"
    trimesh.creation.box().export(src)
    app.source.set(str(src))
    app.destination.set(str(out))
    app.save_report.set(True)
    app.start_conversion()
    assert app.busy
    assert str(app.mode_control.cget("state")) == "disabled"
    wait_for_conversion(app)
    assert "ISO-10303-21" in out.read_text()
    assert out.with_suffix(".json").exists()
    assert "Saved" in app.status.get()


def test_conversion_error_restores_controls(app, monkeypatch, tmp_path):
    src = tmp_path / "bad.stl"
    src.touch()
    app.source.set(str(src))
    app.destination.set(str(tmp_path / "bad.step"))

    def fail(*args):
        raise ValueError("No triangles found")

    monkeypatch.setattr(gui, "convert_file", fail)
    app.start_conversion()
    wait_for_conversion(app)
    assert "failed" in app.status.get()
    assert str(app.mode_control.cget("state")) == "readonly"


def test_overwrite_cancelled(app, monkeypatch, tmp_path):
    src, out = tmp_path / "mesh.stl", tmp_path / "model.step"
    src.touch()
    out.write_text("original")
    app.source.set(str(src))
    app.destination.set(str(out))
    monkeypatch.setattr("tkinter.messagebox.askyesno", lambda *args, **kwargs: False)
    app.start_conversion()
    assert not app.busy
    assert out.read_text() == "original"


def test_invalid_solid_is_warning(app, monkeypatch, tmp_path):
    warnings = []
    monkeypatch.setattr("tkinter.messagebox.showwarning", lambda *args, **kwargs: warnings.append(args))
    app.set_busy(True)
    app.events.put(("done", (tmp_path / "open.step", False, {"mode_used": "smooth", "brep": {"faces": 1}})))
    app.poll()
    assert warnings
    assert "not a valid closed solid" in app.status.get()


def test_worker_keeps_tk_on_main_thread(app, monkeypatch, tmp_path):
    result = SimpleNamespace(is_solid=True, report={"mode_used": "faceted", "brep": {"faces": 6}})
    monkeypatch.setattr(gui, "convert_file", lambda *args: result)
    app.convert(tmp_path / "box.stl", tmp_path / "box.step", None, None)
    assert app.events.get_nowait() == ("done", (tmp_path / "box.step", True, result.report))
