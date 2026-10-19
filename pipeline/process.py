"""One command from walk folder (or AirDropped zip) to viewable output.

usage: process.py <walk_dir_or_zip> [out_root]   (out_root defaults to ../out)

Runs optimize (drift correction) -> fuse -> georef -> preview, copies viewer.html next to the results, and prints
how to open it.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # python -I drops the script dir from sys.path

import fuse  # noqa: E402
import georef  # noqa: E402
import optimize  # noqa: E402

src = Path(sys.argv[1]).expanduser().resolve()
out_root = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "out"

if src.suffix == ".zip":
    walks = HERE.parent / "walks"
    with zipfile.ZipFile(src) as z:
        z.extractall(walks)  # the app zips the walk_* folder itself
    src = walks / src.stem

out = out_root / src.name
optimize.optimize(src, out)
fuse.fuse(src, out, poses=out / "poses.npz")
georef.georef(out, src, poses=out / "poses.npz")
subprocess.run([sys.executable, "-I", str(HERE / "preview.py"), str(out)], check=True)
shutil.copy(HERE / "viewer.html", out / "viewer.html")

print(f"\nDone: {out}")
print(f"View in 3D:  python3 -m http.server -d {out} 8000   then open http://localhost:8000/viewer.html")
