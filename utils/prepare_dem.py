"""Prepare the TINITALY DEM for the Marsicano case study.

The utility locates an available GeoTIFF tile, reads its declared coordinate
reference system, reprojects it to EPSG:32633, clips the Marsicano domain,
resamples the raster to a 5 m grid, and writes GeoTIFF and ESRI ASCII outputs.

Required packages are listed in the repository-level requirements.txt file.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# The first name matches a standard TINITALY tile. The second name supports
# the GeoTIFF already distributed with this repository.
INPUT_CANDIDATES = [
    DATA_DIR / "w46090_s10.tif",
    DATA_DIR / "w46090_s10_UTM33_5m.tif",
]

OUTPUT_TIF = DATA_DIR / "w46090_s10_Marsicano_UTM33_5m.tif"
OUTPUT_ASC = DATA_DIR / "w46090_s10_Marsicano_UTM33_5m.asc"

TARGET_CRS = "EPSG:32633"
TARGET_DX = 5.0

# Marsicano processing extent in WGS 84 / UTM zone 33N.
XMIN = 403000.0
XMAX = 409000.0
YMIN = 4625500.0
YMAX = 4629500.0

NODATA = -9999.0

# Approximate geographic point used only to confirm the projected location.
CHECK_LON = 13.87
CHECK_LAT = 41.79


def find_input() -> Path:
    """Return the first available source GeoTIFF."""
    for candidate in INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    names = ", ".join(str(path) for path in INPUT_CANDIDATES)
    raise FileNotFoundError(
        "No source GeoTIFF was found. Place one of these files in the data "
        f"directory: {names}"
    )


def main() -> int:
    """Create the Marsicano DEM files and report basic raster diagnostics."""
    try:
        input_tif = find_input()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    width = int(round((XMAX - XMIN) / TARGET_DX))
    height = int(round((YMAX - YMIN) / TARGET_DX))

    if width <= 0 or height <= 0:
        print("[ERROR] The requested processing extent is invalid.")
        return 2

    # Define a north-up target grid from its upper-left corner.
    dst_transform = from_origin(XMIN, YMAX, TARGET_DX, TARGET_DX)
    destination = np.full((height, width), NODATA, dtype=np.float32)

    with rasterio.open(input_tif) as src:
        if src.crs is None:
            print("[ERROR] The source GeoTIFF does not declare a CRS.")
            return 3

        print(f"[INPUT]       {input_tif.resolve()}")
        print(f"[SRC CRS]     {src.crs}")
        print(f"[SRC RES]     {src.res}")
        print(f"[SRC BOUNDS]  {src.bounds}")
        print(f"[DST CRS]     {TARGET_CRS}")
        print(f"[DST RES]     ({TARGET_DX}, {TARGET_DX})")
        print(f"[DST BOUNDS]  ({XMIN}, {YMIN}) - ({XMAX}, {YMAX})")
        print(f"[DST SHAPE]   {width} columns x {height} rows")

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            dst_nodata=NODATA,
            resampling=Resampling.average,
            init_dest_nodata=True,
            num_threads=2,
        )

    valid = np.isfinite(destination) & (destination != NODATA)
    n_valid = int(valid.sum())

    if n_valid == 0:
        print("[ERROR] The projected source raster does not intersect the Marsicano extent.")
        return 4

    valid_fraction = n_valid / destination.size
    zmin = float(destination[valid].min())
    zmax = float(destination[valid].max())

    common_profile = {
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": dst_transform,
        "nodata": NODATA,
    }

    tif_profile = {
        **common_profile,
        "driver": "GTiff",
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
    }
    with rasterio.open(OUTPUT_TIF, "w", **tif_profile) as dst:
        dst.write(destination, 1)

    asc_profile = {
        **common_profile,
        "driver": "AAIGrid",
        "DECIMAL_PRECISION": 6,
    }
    with rasterio.open(OUTPUT_ASC, "w", **asc_profile) as dst:
        dst.write(destination, 1)

    transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    check_x, check_y = transformer.transform(CHECK_LON, CHECK_LAT)
    check_inside = XMIN <= check_x <= XMAX and YMIN <= check_y <= YMAX

    print(f"[VALID]       {valid_fraction:.2%}")
    print(f"[ELEVATION]   min={zmin:.3f} m, max={zmax:.3f} m")
    print(f"[CHECK]       Marsicano approximate point: x={check_x:.3f}, y={check_y:.3f}")
    print(f"[CHECK]       Point inside processing extent: {check_inside}")
    print(f"[SAVED TIF]   {OUTPUT_TIF.resolve()}")
    print(f"[SAVED ASC]   {OUTPUT_ASC.resolve()}")
    print("The Marsicano configuration already points to the generated ASCII file.")
    print("Delete the DEM cache only when the source raster or processing settings change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
