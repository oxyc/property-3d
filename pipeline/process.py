"""One command from walk folder (or AirDropped zip) to viewable output.

usage: process.py <walk_dir_or_zip> [out_root] [--jobs 3] [--no-visual]   (out_root defaults to ../out)

Runs optimize (drift correction) -> fuse -> georef -> preview, copies viewer.html next to the results, and prints
how to open it. A walk with stretches of bad tracking (ARKit sliding, then relocalising) goes through merge.py
instead: the bad stretches are cut out and the clean segments registered back together (see merge.json).
--jobs N: cores to use (default 3, leaving a performance core free); see cores.py.
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # python -I drops the script dir from sys.path

import cores  # noqa: E402

if __name__ == "__main__":
    cores.setup()  # caps the threads Open3D uses; must happen before it is imported

import fuse  # noqa: E402
import georef  # noqa: E402
import merge  # noqa: E402
import optimize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("walk")
    ap.add_argument("out_root", nargs="?", default=str(HERE.parent / "out"))
    ap.add_argument("--jobs", type=int, default=cores.DEFAULT_JOBS, help="cores to use (default %(default)s)")
    ap.add_argument("--no-visual", action="store_true", help="no photo-feature loop constraints")
    a = ap.parse_args()
    src, out_root = Path(a.walk).expanduser().resolve(), Path(a.out_root)

    if src.suffix == ".zip":
        walks = HERE.parent / "walks"
        with zipfile.ZipFile(src) as z:
            z.extractall(walks)  # the app zips the walk_* folder itself
        src = walks / src.stem

    out = out_root / src.name
    segments, removed, _ = optimize.split_walk(fuse.read_walk(src)[1])
    if removed:  # bad tracking: cut it out and register the clean pieces to each other (merge.py with one walk)
        merge.merge(out, [src], jobs=a.jobs, visual=not a.no_visual)
    else:
        optimize.optimize(src, out, visual=not a.no_visual, jobs=a.jobs)
        fuse.fuse(src, out, poses=out / "poses.npz", jobs=a.jobs)
        georef.georef(out, src, poses=out / "poses.npz")
        subprocess.run([sys.executable, "-I", str(HERE / "preview.py"), str(out)], check=True)
        shutil.copy(HERE / "viewer.html", out / "viewer.html")

    print(f"\nDone: {out}")
    print(f"View in 3D:  python3 -m http.server -d {out} 8000   then open http://localhost:8000/viewer.html")


if __name__ == "__main__":
    main()
