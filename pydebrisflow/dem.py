from __future__ import annotations

"""DEM reading, clipping, resampling, and cache management."""

import json
import math
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .config import GridConfig

# -----------------------------------------------------------------------------
# DEM loading: crop while streaming the large ASCII grid from ZIP.
# -----------------------------------------------------------------------------

def _read_ascii_header(stream: Any) -> Tuple[Dict[str, float], List[bytes]]:
    header: Dict[str, float] = {}
    buffered: List[bytes] = []
    required = {"ncols", "nrows", "cellsize"}
    while True:
        line = stream.readline()
        if not line:
            break
        parts = line.decode("utf-8", errors="ignore").strip().split()
        if len(parts) >= 2 and parts[0].lower() in {
            "ncols", "nrows", "xllcorner", "xllcenter", "yllcorner", "yllcenter", "cellsize", "nodata_value"
        }:
            header[parts[0].lower()] = float(parts[1])
            continue
        buffered.append(line)
        break
    if not required.issubset(header):
        raise RuntimeError(f"Incomplete ESRI ASCII header: {header}")
    return header, buffered


def _open_dem_stream(path: str, member: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".zip":
        zf = zipfile.ZipFile(path, "r")
        names = zf.namelist()
        use = member if member in names else next((n for n in names if n.lower().endswith((".asc", ".ascii"))), None)
        if use is None:
            zf.close()
            raise RuntimeError("No ESRI ASCII DEM found inside ZIP")
        return zf, zf.open(use, "r")
    return None, open(path, "rb")


def load_esri_ascii_cropped(cfg: GridConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    owner, stream = _open_dem_stream(cfg.dem_path, cfg.zip_member)
    try:
        header, buffered = _read_ascii_header(stream)
        ncols = int(header["ncols"])
        nrows = int(header["nrows"])
        cell = float(header["cellsize"])
        nodata = float(header.get("nodata_value", -9999.0))
        xll = float(header.get("xllcorner", header.get("xllcenter", 0.0)))
        yll = float(header.get("yllcorner", header.get("yllcenter", 0.0)))
        center_header = "xllcenter" in header or "yllcenter" in header
        x0 = xll if center_header else xll + 0.5 * cell
        y0 = yll if center_header else yll + 0.5 * cell

        if cfg.clip_enabled:
            i0 = max(0, int(math.floor((cfg.xmin - x0) / cell)))
            i1 = min(ncols - 1, int(math.ceil((cfg.xmax - x0) / cell)))
            # File row 0 is north. Convert requested y range to inclusive file rows.
            r0 = max(0, int(math.floor((y0 + (nrows - 1) * cell - cfg.ymax) / cell)))
            r1 = min(nrows - 1, int(math.ceil((y0 + (nrows - 1) * cell - cfg.ymin) / cell)))
        else:
            i0, i1, r0, r1 = 0, ncols - 1, 0, nrows - 1

        rows: List[np.ndarray] = []
        pending = buffered[:]
        for r in range(nrows):
            line = pending.pop(0) if pending else stream.readline()
            if not line:
                raise RuntimeError(f"DEM ended at row {r}, expected {nrows}")
            if r < r0 or r > r1:
                continue
            arr = np.fromstring(line.decode("ascii", errors="ignore"), sep=" ", dtype=np.float64)
            if arr.size != ncols:
                raise RuntimeError(f"DEM row {r} has {arr.size} values, expected {ncols}")
            rows.append(arr[i0:i1 + 1])
        z_north_to_south = np.vstack(rows)
        z = np.flipud(z_north_to_south)
        invalid = (~np.isfinite(z)) | (np.abs(z) > 1.0e20) | np.isclose(z, nodata, rtol=0.0, atol=max(1e-8, abs(nodata) * 1e-7))
        z[invalid] = np.nan
        x = x0 + cell * np.arange(i0, i1 + 1, dtype=np.float64)
        file_rows = np.arange(r0, r1 + 1)
        y_north = y0 + (nrows - 1 - file_rows) * cell
        y = np.flip(y_north).astype(np.float64)
        valid = np.isfinite(z)
        return x, y, z.astype(np.float32), valid
    finally:
        try:
            stream.close()
        finally:
            if owner is not None:
                owner.close()


def resample_dem(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    target_dx: float,
    min_valid_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if x.size < 2 or y.size < 2 or target_dx <= 0.0:
        return x, y, z, np.isfinite(z)
    dx0 = float(abs(x[1] - x[0]))
    dy0 = float(abs(y[1] - y[0]))
    if abs(target_dx - 0.5 * (dx0 + dy0)) <= 0.02 * target_dx:
        return x, y, z, np.isfinite(z)
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    valid0 = np.isfinite(z)
    if not np.any(valid0):
        raise RuntimeError("DEM crop contains no valid cells")
    if np.all(valid0):
        zfill = z.astype(np.float64)
    else:
        idx = distance_transform_edt(~valid0, return_distances=False, return_indices=True)
        zfill = z[tuple(idx)].astype(np.float64)
    if target_dx > max(dx0, dy0):
        zfill = gaussian_filter(zfill, sigma=(0.35 * target_dx / dy0, 0.35 * target_dx / dx0), mode="nearest")
    x2 = np.arange(float(x[0]), float(x[-1]) + 0.25 * target_dx, target_dx)
    y2 = np.arange(float(y[0]), float(y[-1]) + 0.25 * target_dx, target_dx)
    zi = RegularGridInterpolator((y, x), zfill, method="linear", bounds_error=False, fill_value=np.nan)
    vi = RegularGridInterpolator((y, x), valid0.astype(np.float64), method="linear", bounds_error=False, fill_value=0.0)
    Y, X = np.meshgrid(y2, x2, indexing="ij")
    pts = np.column_stack((Y.ravel(), X.ravel()))
    z2 = zi(pts).reshape(Y.shape)
    vf = vi(pts).reshape(Y.shape)
    valid2 = np.isfinite(z2) & (vf >= min_valid_fraction)
    z2[~valid2] = np.nan
    return x2.astype(np.float64), y2.astype(np.float64), z2.astype(np.float32), valid2


def _dem_cache_signature(cfg: GridConfig) -> str:
    st = os.stat(cfg.dem_path)
    payload = {
        "version": 1,
        "source": os.path.abspath(cfg.dem_path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "zip_member": cfg.zip_member,
        "target_dx": float(cfg.target_dx),
        "clip_enabled": bool(cfg.clip_enabled),
        "bbox": [float(cfg.xmin), float(cfg.xmax), float(cfg.ymin), float(cfg.ymax)],
        "min_valid_fraction": float(cfg.min_valid_fraction),
    }
    return json.dumps(payload, sort_keys=True)


def load_or_build_dem(cfg: GridConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = Path(cfg.cache_path)
    signature = _dem_cache_signature(cfg)
    if cache.is_file():
        try:
            with np.load(cache) as d:
                cached_sig = str(d["signature"].item()) if "signature" in d else ""
                if cached_sig == signature:
                    return d["x"], d["y"], d["zb"], d["valid"].astype(bool)
        except Exception:
            pass
    x, y, z, valid = load_esri_ascii_cropped(cfg)
    x, y, z, valid = resample_dem(x, y, z, cfg.target_dx, cfg.min_valid_fraction)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(cache.suffix + ".tmp.npz")
    np.savez_compressed(tmp, x=x, y=y, zb=z, valid=valid.astype(np.uint8), signature=np.array(signature))
    os.replace(tmp, cache)
    return x, y, z, valid
