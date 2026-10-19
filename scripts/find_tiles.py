"""Find the IDE Uruguay ortho/elevation sheets that contain a lat/lon point."""
import sys

import shapefile
from shapely.geometry import Point, shape

GRIDS = sys.argv[1]
lat, lon = float(sys.argv[2]), float(sys.argv[3])
pt = Point(lon, lat)

for name in ("Nacional", "Urbana"):
    sf = shapefile.Reader(f"{GRIDS}/Articulacion_Ortoimagenes_{name}")
    fields = [f[0] for f in sf.fields[1:]]
    for sr in sf.iterShapeRecords():
        if shape(sr.shape.__geo_interface__).contains(pt):
            print(name, dict(zip(fields, sr.record)))
