"""Crop the IDE ortho/DSM/DTM tiles to a square around a lat/lon point.

usage: crop_ide.py <tile_dir> <out_dir> <lat> <lon> <sheet> <suffix> [half_size_m]

Writes cropped GeoTIFFs plus PNG previews (ortho, and canopy height = DSM - DTM).
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
lat, lon = float(sys.argv[3]), float(sys.argv[4])
SHEET, SUFFIX = sys.argv[5], sys.argv[6]  # e.g. sheet name from find_tiles.py, "Remesa_02_MDO"
HALF = float(sys.argv[7]) if len(sys.argv) > 7 else 60.0  # metres

OUT.mkdir(parents=True, exist_ok=True)
files = {
    "ortho": f"{SHEET}_RGBI_8_{SUFFIX}.tif",
    "dsm": f"{SHEET}_MDS_{SUFFIX}.tif",
    "dtm": f"{SHEET}_MDT_{SUFFIX}.tif",
}

with rasterio.open(SRC / files["dsm"]) as ds:
    crs = ds.crs
x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
bounds = (x - HALF, y - HALF, x + HALF, y + HALF)
print("CRS:", crs, "centre:", round(x, 2), round(y, 2))

arrays = {}
for key, name in files.items():
    with rasterio.open(SRC / name) as ds:
        win = from_bounds(*bounds, transform=ds.transform)
        data = ds.read(window=win, boundless=True, fill_value=ds.nodata or 0)
        profile = ds.profile | {
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": ds.window_transform(win),
        }
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile["tiled"] = False
        with rasterio.open(OUT / f"{key}.tif", "w", **profile) as dst:
            dst.write(data)
        print(key, data.shape, "res", ds.res, "nodata", ds.nodata)
        arrays[key] = data

# Ortho preview (first three bands are RGB).
rgb = np.moveaxis(arrays["ortho"][:3], 0, -1).astype(np.uint8)
Image.fromarray(rgb).save(OUT / "ortho.png")

# Canopy/building height: upsample the 1 m DTM onto the 10 cm DSM grid.
dsm = arrays["dsm"][0].astype(np.float32)
with rasterio.open(OUT / "dtm.tif") as ds:
    dtm = ds.read(
        1, out_shape=dsm.shape, resampling=Resampling.bilinear
    ).astype(np.float32)
chm = np.clip(dsm - dtm, 0, None)
np.save(OUT / "chm.npy", chm)
print("ground elev min/max:", float(dtm.min()), float(dtm.max()))
print("object height max:", float(chm.max()), "p95:", float(np.percentile(chm, 95)))

norm = np.clip(chm / max(np.percentile(chm, 99.5), 1e-6), 0, 1)
ramp = np.stack([norm, np.sqrt(norm), 1 - norm], axis=-1)  # blue (low) to yellow-white (high)
Image.fromarray((ramp * 255).astype(np.uint8)).save(OUT / "height.png")
