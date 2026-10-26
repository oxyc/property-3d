"""Generates the four printed markers m1..m4 (deterministic; needs only Pillow and numpy).

    python3 ios/markers/make_markers.py [count]     # default 12

Writes ios/WalkScan/Markers/m1.png ... m<count>.png (bundled in the app; the app treats each as 0.18 m wide) and
ios/markers/walkscan_markers.pdf (A4, one marker per page, the square printed exactly 18.0 cm wide).
Each marker is random black/white blocks of varied sizes plus circles: high contrast, non-repeating and
non-symmetric, so ARKit rates it as a good reference image. Re-running gives identical files.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PNG_DIR = HERE.parent / "WalkScan" / "Markers"
SIDE = 1800  # px for 18 cm: 10 px/mm
BORDER = 100  # 1 cm grey border inside the square
GREY = 128
WANTED = 0.25  # markers are re-rolled until no pair, rotation or mirror correlates above this
DPI = 300
A4 = (round(21.0 / 2.54 * DPI), round(29.7 / 2.54 * DPI))


def pattern(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    n = SIDE - 2 * BORDER
    img = Image.new("L", (n, n), 255)
    d = ImageDraw.Draw(img)

    def split(x, y, w, h, depth):
        if w < 90 or h < 90 or (depth > 2 and rng.random() < 0.3):
            d.rectangle([x, y, x + w - 1, y + h - 1], fill=int(rng.choice([0, 255])))
            return
        ratio = rng.uniform(0.3, 0.7)
        if w > h * 1.3 or (h <= w * 1.3 and rng.random() < 0.5):
            cut = int(w * ratio)
            split(x, y, cut, h, depth + 1)
            split(x + cut, y, w - cut, h, depth + 1)
        else:
            cut = int(h * ratio)
            split(x, y, w, cut, depth + 1)
            split(x, y + cut, w, h - cut, depth + 1)

    split(0, 0, n, n, 0)
    for _ in range(int(rng.integers(9, 14))):  # circles, sometimes with a contrasting dot inside
        r = int(rng.integers(40, 170))
        cx, cy = (int(v) for v in rng.integers(r, n - r, size=2))
        fill = int(rng.choice([0, 255]))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=255 - fill, width=6)
        if rng.random() < 0.5:
            q = r // 3
            d.ellipse([cx - q, cy - q, cx + q, cy + q], fill=255 - fill)
    for _ in range(int(rng.integers(25, 40))):  # small blocks: fine detail for feature points
        s = int(rng.integers(18, 55))
        x, y = (int(v) for v in rng.integers(0, n - s, size=2))
        d.rectangle([x, y, x + s, y + s], fill=int(rng.choice([0, 255])))
    return img


def marker(index: int, others: list[Image.Image]) -> Image.Image:
    """Rerolls (new seed) until black and white are balanced and the marker is dissimilar to `others`."""
    for attempt in range(200):
        inner = pattern(1000 * (index + 1) + attempt)
        out = Image.new("L", (SIDE, SIDE), GREY)
        out.paste(inner, (BORDER, BORDER))
        if 0.38 < float(np.mean(np.asarray(inner) < 128)) < 0.62 and max_correlation(others + [out]) < WANTED:
            return out
    raise SystemExit(f"no dissimilar pattern found for m{index + 1}")


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def page(id_: str, img: Image.Image) -> Image.Image:
    p = Image.new("L", A4, 255)
    side = round(18.0 / 2.54 * DPI)  # exactly 18 cm at DPI
    left, top = (A4[0] - side) // 2, round(3.0 / 2.54 * DPI)
    p.paste(img.resize((side, side), Image.LANCZOS), (left, top))
    d = ImageDraw.Draw(p)
    d.text((left, top + side + 60), f"Marker {id_}", fill=0, font=font(48))
    d.text((left, top + side + 140), "Print at 100%, no scaling; the square must measure 18.0 cm.", fill=0, font=font(34))
    return p


def max_correlation(images: list[Image.Image]) -> float:
    """Highest normalized (zero-mean) correlation between any two markers, or between a marker and its own
    rotations (90/180/270) and mirrors. Random patterns sit near 0; identical or symmetric ones near 1."""
    small = [np.asarray(im.resize((128, 128), Image.BOX), dtype=np.float64) for im in images]

    def variants(a):
        for k in range(4):
            r = np.rot90(a, k)
            yield k, r
            yield k + 4, np.fliplr(r)

    def ncc(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    worst = 0.0
    for i, a in enumerate(small):
        for j, b in enumerate(small):
            for v, other in variants(b):
                if i == j and v == 0:
                    continue  # the marker against itself, unchanged
                if i > j and i != j:
                    continue  # pair already covered (the variant set is closed under inversion)
                worst = max(worst, abs(ncc(a, other)))
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("count", nargs="?", type=int, default=12, help="number of markers (default 12)")
    count = parser.parse_args().count
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    images: list[Image.Image] = []
    for i in range(count):
        images.append(marker(i, images))
    print(f"{count} markers; max normalized correlation (pairs, rotations, mirrors): {max_correlation(images):.3f}")
    for i, img in enumerate(images):
        img.save(PNG_DIR / f"m{i + 1}.png")
    pages = [page(f"m{i + 1}", img) for i, img in enumerate(images)]
    pages[0].save(HERE / "walkscan_markers.pdf", save_all=True, append_images=pages[1:], resolution=DPI)


if __name__ == "__main__":
    main()
