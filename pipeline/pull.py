#!/usr/bin/env python3
"""Pull a walk off the phone over Wi-Fi (stdlib only).

In the WalkScan app: Walks > (...) > Send to Mac. It shows an address like http://192.168.1.23:51234.

    python3 pull.py http://192.168.1.23:51234 [dest]        # dest defaults to ../walks
    python3 pull.py http://192.168.1.23:51234 --tar         # one streamed tar instead of per-file requests

Files already complete (same size) are skipped and partial ones resume, so it is safe to re-run after a
dropped connection. Sizes are verified at the end; only then is the phone told the pull was good, which
lets the app offer "Delete from phone". Keep the phone awake and on the same Wi-Fi.
"""

import argparse
import http.client
import json
import os
import sys
import tarfile
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 256 * 1024


class Progress:
    def __init__(self, total_files, total_bytes):
        self.total_files, self.total_bytes = total_files, total_bytes
        self.files = 0
        self.bytes = 0  # complete + resumed + received
        self.received = 0  # only what came over the wire in this run (for the speed)
        self.start = time.time()
        self.lock = threading.Lock()
        self.window = [(self.start, 0)]

    def add(self, n, wire=True):
        with self.lock:
            self.bytes += n
            if wire:
                self.received += n

    def file_done(self):
        with self.lock:
            self.files += 1

    def line(self):
        now = time.time()
        with self.lock:
            self.window.append((now, self.received))
            self.window = [w for w in self.window if now - w[0] <= 5]
            t0, b0 = self.window[0]
            speed = (self.received - b0) / (now - t0) if now > t0 else 0.0
            done, files = self.bytes, self.files
        left = max(self.total_bytes - done, 0) / speed if speed > 1 else 0
        eta = f"{int(left // 60)}m{int(left % 60):02d}s" if speed > 1 else "?"
        return (f"{files}/{self.total_files} files  {done / 1e9:.2f}/{self.total_bytes / 1e9:.2f} GB  "
                f"{speed / 1e6:.2f} MB/s  eta {eta}")


def safe_join(root: Path, rel: str) -> Path:
    parts = rel.split("/")
    if rel.startswith("/") or "\\" in rel or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"unsafe path from phone: {rel!r}")
    return root.joinpath(*parts)


class Client:
    """One persistent HTTP connection per thread."""

    def __init__(self, host, port):
        self.host, self.port = host, port
        self.local = threading.local()

    def conn(self):
        if getattr(self.local, "c", None) is None:
            self.local.c = http.client.HTTPConnection(self.host, self.port, timeout=30)
        return self.local.c

    def reset(self):
        c = getattr(self.local, "c", None)
        if c is not None:
            c.close()
        self.local.c = None

    def get(self, path, headers=None):
        c = self.conn()
        c.request("GET", path, headers=headers or {})
        return c.getresponse()


def download(client, walk_dir: Path, rel: str, size: int, base: str, progress: Progress):
    final = safe_join(walk_dir, rel)
    part = final.with_name(final.name + ".part")
    if final.exists() and final.stat().st_size == size:
        progress.add(size, wire=False)
        progress.file_done()
        return
    final.parent.mkdir(parents=True, exist_ok=True)
    counted = 0
    for attempt in range(8):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have > size:
                part.unlink()
                have = 0
            if have > counted:
                progress.add(have - counted, wire=False)
                counted = have
            if have < size:
                headers = {"Range": f"bytes={have}-"} if have else {}
                r = client.get(f"{base}/files/{urllib.parse.quote(rel)}", headers)
                if r.status not in (200, 206):
                    r.read()
                    raise OSError(f"HTTP {r.status} for {rel}")
                if r.status == 200 and have:  # server ignored the range
                    progress.add(-have, wire=False)
                    counted = have = 0
                with open(part, "ab" if r.status == 206 else "wb") as f:
                    while chunk := r.read(CHUNK):
                        f.write(chunk)
                        progress.add(len(chunk))
                        counted += len(chunk)
            else:
                part.touch()
            if part.stat().st_size != size:
                raise OSError(f"{rel}: got {part.stat().st_size} of {size} bytes")
            os.replace(part, final)
            progress.file_done()
            return
        except (OSError, http.client.HTTPException) as e:
            client.reset()
            if attempt == 7:
                raise
            print(f"\n  retrying {rel}: {e}", file=sys.stderr)
            time.sleep(min(2 ** attempt, 10))


def pull_tar(client, base, walk_dir: Path, name: str, progress: Progress):
    """Stream /walk.tar and unpack it as it arrives. Not resumable, but one request for everything."""
    r = client.get(f"{base}/walk.tar")
    if r.status != 200:
        raise OSError(f"HTTP {r.status} for /walk.tar")

    class Counting:
        def read(self, n=-1):
            data = r.read(n)
            progress.add(len(data))
            return data

    root = walk_dir.parent
    with tarfile.open(fileobj=Counting(), mode="r|") as tf:
        for m in tf:
            if not m.name.startswith(name + "/") or not (m.isfile() or m.isdir()):
                continue
            target = safe_join(root, m.name.rstrip("/"))
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as out:
                src = tf.extractfile(m)
                while chunk := src.read(CHUNK):
                    out.write(chunk)
            progress.file_done()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="address shown by the app, e.g. http://192.168.1.23:51234")
    ap.add_argument("dest", nargs="?", help="folder that will contain the walk (default: <repo>/walks)")
    ap.add_argument("--tar", action="store_true", help="stream one tar instead of downloading files individually")
    ap.add_argument("--workers", type=int, default=6, help="parallel downloads (default 6)")
    args = ap.parse_args()

    url = urllib.parse.urlsplit(args.url if "//" in args.url else "http://" + args.url)
    host, port = url.hostname, url.port or 80
    base = url.path.rstrip("/")
    dest = Path(args.dest) if args.dest else Path(__file__).resolve().parent.parent / "walks"
    client = Client(host, port)

    try:
        r = client.get(f"{base}/index.json")
        if r.status != 200:
            raise OSError(f"HTTP {r.status}")
        index = json.loads(r.read())
    except (OSError, http.client.HTTPException, ValueError) as e:
        sys.exit(f"cannot reach the phone at {host}:{port} ({e}). Is Send to Mac open, and are you on the same Wi-Fi?")

    name = index["name"]
    files = index["files"]
    walk_dir = safe_join(dest, name)
    total = sum(f["size"] for f in files)
    print(f"{name}: {len(files)} files, {total / 1e9:.2f} GB -> {walk_dir}"
          + ("" if index.get("ended") else "   (walk was never finished)"))
    walk_dir.mkdir(parents=True, exist_ok=True)

    progress = Progress(len(files), total)
    stop = threading.Event()

    def report():
        while not stop.wait(2):
            end = "\r" if sys.stderr.isatty() else "\n"
            print("  " + progress.line(), end=end, file=sys.stderr, flush=True)

    threading.Thread(target=report, daemon=True).start()
    try:
        if args.tar:
            pull_tar(client, base, walk_dir, name, progress)
        else:
            with ThreadPoolExecutor(args.workers) as pool:
                futures = [pool.submit(download, client, walk_dir, f["path"], f["size"], base, progress)
                           for f in files]
                for fut in futures:
                    fut.result()
    except (OSError, http.client.HTTPException, ValueError, tarfile.TarError) as e:
        stop.set()
        sys.exit(f"\ntransfer failed: {e}\nRun the same command again to resume.")
    stop.set()
    print("  " + progress.line())

    bad = []
    for f in files:
        p = safe_join(walk_dir, f["path"])
        if not p.exists() or p.stat().st_size != f["size"]:
            bad.append(f["path"])
    if bad:
        print(f"VERIFY FAILED: {len(bad)} files missing or wrong size, e.g. {bad[:3]}. Re-run to fetch them.")
        sys.exit(1)
    print(f"verified: {len(files)} files, sizes match")

    try:
        c = http.client.HTTPConnection(host, port, timeout=10)
        c.request("POST", f"{base}/verified", body=b"")
        c.getresponse().read()
        print("told the phone: the app now offers 'Delete from phone'")
    except (OSError, http.client.HTTPException):
        print("(could not tell the phone; the walk is safe on the Mac either way)")


if __name__ == "__main__":
    main()
