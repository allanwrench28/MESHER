"""A small desktop interface to the existing conversion pipeline."""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path

from .pipeline import ConvertOptions, convert_file


def validate_paths(source: str, destination: str, save_report: bool) -> tuple[Path, Path, Path | None]:
    if not source.strip() or not destination.strip():
        raise ValueError("Choose a mesh file and a STEP destination first.")
    src = Path(source).expanduser().resolve()
    out = Path(destination).expanduser().resolve()
    if not src.is_file():
        raise ValueError("The selected mesh file does not exist.")
    if out.suffix.lower() not in (".step", ".stp"):
        raise ValueError("The destination must end in .step or .stp.")
    report = out.with_suffix(".json") if save_report else None
    for target in (out, report):
        if target is None:
            continue
        if target == src or (target.exists() and target.samefile(src)):
            raise ValueError("Choose a destination that will not overwrite your input mesh.")
        if not target.parent.is_dir() or target.is_dir():
            raise ValueError("Choose an existing folder and a file name for the destination.")
    return src, out, report


class ScancraftApp:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.busy = False
        self.events = queue.Queue()
        self.source = tk.StringVar(root)
        self.destination = tk.StringVar(root)
        self.mode = tk.StringVar(root, value="smooth")
        self.save_report = tk.BooleanVar(root, value=False)
        self.status = tk.StringVar(root, value="Choose a mesh to get started.")
        root.title("Scancraft — Mesh to CAD")
        root.minsize(600, 340)
        root.protocol("WM_DELETE_WINDOW", self.close)
        frame = ttk.Frame(root, padding=20)
        frame.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text="Turn your mesh into a CAD model", font=("", 16)).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        ttk.Label(frame, text="1. Choose a mesh   2. Choose where to save   3. Convert").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 16)
        )
        self.controls = []
        for row, label, variable, command in (
            (2, "Mesh file (STL, OBJ, PLY…)", self.source, self.choose_source),
            (4, "Save STEP file", self.destination, self.choose_destination),
        ):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            entry = ttk.Entry(frame, textvariable=variable, width=55)
            entry.grid(row=row + 1, column=0, sticky="ew", pady=(4, 12))
            button = ttk.Button(frame, text="Browse…", command=command)
            button.grid(row=row + 1, column=1, padx=(10, 0))
            self.controls.extend((entry, button))
        ttk.Label(frame, text="Conversion mode").grid(row=6, column=0, sticky="w")
        mode = ttk.Combobox(frame, textvariable=self.mode, values=("smooth", "faceted"), state="readonly")
        mode.grid(row=7, column=0, sticky="w", pady=4)
        self.mode_control = mode
        ttk.Label(frame, text="Smooth is recommended; faceted preserves the mesh triangles.").grid(
            row=8, column=0, columnspan=2, sticky="w"
        )
        report = ttk.Checkbutton(frame, text="Also save a JSON conversion report", variable=self.save_report)
        report.grid(row=9, column=0, columnspan=2, sticky="w", pady=12)
        convert = ttk.Button(frame, text="Convert to STEP", command=self.start_conversion)
        convert.grid(row=10, column=0, sticky="w")
        self.controls.extend((report, convert))
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=11, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Label(frame, textvariable=self.status, wraplength=560).grid(
            row=12, column=0, columnspan=2, sticky="w"
        )

    def choose_source(self):
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            parent=self.root, title="Choose a mesh",
            filetypes=[("Mesh files", "*.stl *.obj *.ply *.off *.3mf *.glb *.gltf"), ("All files", "*")],
        )
        if path:
            self.source.set(path)
            self.destination.set(str(Path(path).with_suffix(".step")))
            self.status.set("Ready to convert. Your original mesh will not be changed.")

    def choose_destination(self):
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save CAD model", defaultextension=".step",
            initialfile=Path(self.destination.get()).name or "model.step",
            filetypes=[("STEP model", "*.step *.stp")],
        )
        if path:
            self.destination.set(path)

    def set_busy(self, busy):
        self.busy = busy
        for control in self.controls:
            control.configure(state="disabled" if busy else "normal")
        self.mode_control.configure(state="disabled" if busy else "readonly")
        if busy:
            self.progress.start()
        else:
            self.progress.stop()

    def start_conversion(self):
        from tkinter import messagebox

        if self.busy:
            return
        try:
            src, out, report = validate_paths(
                self.source.get(), self.destination.get(), self.save_report.get()
            )
            options = ConvertOptions(mode=self.mode.get())
        except (ValueError, OSError) as exc:
            messagebox.showerror("Check your files", str(exc), parent=self.root)
            return
        existing = [str(path) for path in (out, report) if path is not None and path.exists()]
        if existing and not messagebox.askyesno(
            "Replace existing files?", "These files will be replaced:\n\n" + "\n".join(existing),
            parent=self.root,
        ):
            return
        self.set_busy(True)
        self.status.set("Converting… Large meshes can take several minutes.")
        threading.Thread(target=self.convert, args=(src, out, options, report), daemon=True).start()
        self.root.after(100, self.poll)

    def convert(self, src, out, options, report):
        try:
            result = convert_file(src, out, options, report)
            self.events.put(("done", (out, result.is_solid, result.report)))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def poll(self):
        from tkinter import messagebox

        try:
            kind, payload = self.events.get_nowait()
        except queue.Empty:
            self.root.after(100, self.poll)
            return
        self.set_busy(False)
        if kind == "error":
            self.status.set("Conversion failed. Check your mesh and destination, then try again.")
            messagebox.showerror("Conversion failed", payload, parent=self.root)
            return
        out, solid, report = payload
        summary = f"Saved {out}\nMode: {report['mode_used']}. Faces: {report['brep']['faces']}."
        if solid:
            self.status.set(summary)
            messagebox.showinfo("Your CAD model is ready", summary, parent=self.root)
        else:
            self.status.set("Saved a STEP shape, but it is not a valid closed solid.")
            messagebox.showwarning(
                "Mesh needs attention",
                summary + "\nThe result is not a valid closed solid. Repair holes in the mesh and try again.",
                parent=self.root,
            )

    def close(self):
        from tkinter import messagebox

        if self.busy:
            messagebox.showinfo("Conversion in progress", "Please wait for conversion to finish.", parent=self.root)
        else:
            self.root.destroy()


def main() -> int:
    try:
        import tkinter as tk
    except ImportError:
        print("Scancraft needs Tk. On Ubuntu/Debian, install python3-tk.", file=sys.stderr)
        return 1
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Cannot open the desktop app: {exc}. Use scancraft for command-line conversion.", file=sys.stderr)
        return 1
    ScancraftApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
