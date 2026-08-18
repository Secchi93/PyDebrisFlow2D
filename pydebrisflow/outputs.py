from __future__ import annotations

"""Snapshots, maps, and animated depth output."""

import os
from typing import List, Optional, Sequence

import numpy as np

from .config import SolverConfig
from .geometry import _map_extent, _show_base_map
from .physics import composition_fields, terrain_geometry, terrain_velocity_components


def _apply_publication_plot_style() -> None:
    """Use manuscript-scale typography for solver maps, animations, and colorbars."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 13.0,
        "axes.titlesize": 14.0,
        "axes.labelsize": 13.5,
        "xtick.labelsize": 12.0,
        "ytick.labelsize": 12.0,
        "legend.fontsize": 11.5,
        "lines.markersize": 6.5,
    })


def save_snapshot(out_dir: str, index: int, t: float, Uc: np.ndarray, zb: np.ndarray, x: np.ndarray, y: np.ndarray, cfg: SolverConfig) -> str:
    f = composition_fields(Uc, cfg)
    dx = float(abs(x[1] - x[0])); dy = float(abs(y[1] - y[0]))
    dzdx, dzdy, _ = terrain_geometry(zb, dx, dy)
    v_down, v_cross = terrain_velocity_components(f["u"], f["v"], dzdx, dzdy)
    phi_u = np.where(f["hsu"] > 0.0, f["hcu"] / np.maximum(f["hsu"], 1.0e-30), 0.0)
    phi_l = np.where(f["hsl"] > 0.0, f["hcl"] / np.maximum(f["hsl"], 1.0e-30), 0.0)
    path = os.path.join(out_dir, f"state_{index:05d}.npz")
    np.savez_compressed(
        path, t=np.array(t), x=x, y=y, zb=zb.astype(np.float32),
        h=f["h"].astype(np.float32), u=f["u"].astype(np.float32), v=f["v"].astype(np.float32),
        velocity_downslope=v_down.astype(np.float32), velocity_crossslope=v_cross.astype(np.float32),
        rho=f["rho"].astype(np.float32), solid_fraction=f["cs"].astype(np.float32),
        coarse_fraction=f["fc"].astype(np.float32), coarse_upper=phi_u.astype(np.float32), coarse_lower=phi_l.astype(np.float32),
        U=Uc.astype(np.float32),
    )
    return path


def save_maps(out_dir: str, t: float, Uc: np.ndarray, zb: np.ndarray, x: np.ndarray, y: np.ndarray, cfg: SolverConfig, suffix: str) -> List[str]:
    import matplotlib.pyplot as plt
    _apply_publication_plot_style()

    f = composition_fields(Uc, cfg)
    dx = float(abs(x[1] - x[0])); dy = float(abs(y[1] - y[0]))
    dzdx, dzdy, _ = terrain_geometry(zb, dx, dy)
    v_down, v_cross = terrain_velocity_components(f["u"], f["v"], dzdx, dzdy)
    fields = [
        ("h", f["h"], "Depth h [m]", "turbo"),
        ("speed", f["speed"], "Speed [m/s]", "turbo"),
        ("velocity_x", f["u"], "Cartesian velocity u [m/s]", "coolwarm"),
        ("velocity_y", f["v"], "Cartesian velocity v [m/s]", "coolwarm"),
        ("velocity_downslope", v_down, "Local downslope velocity [m/s]", "coolwarm"),
        ("velocity_crossslope", v_cross, "Local cross-slope velocity [m/s]", "coolwarm"),
        ("rho", f["rho"], "Mixture density [kg/m3]", "viridis"),
        ("solid_fraction", f["cs"], "Solid volume fraction", "viridis"),
        ("coarse_fraction", f["fc"], "Coarse fraction of solids", "viridis"),
        ("segregation", np.where(f["hsu"] > 0.0, f["hcu"] / np.maximum(f["hsu"], 1e-30), 0.0)
         - np.where(f["hsl"] > 0.0, f["hcl"] / np.maximum(f["hsl"], 1e-30), 0.0),
         "Upper minus lower coarse fraction", "coolwarm"),
    ]
    extent = _map_extent(x, y, dx, dy)
    paths: List[str] = []
    for name, arr, label, cmap in fields:
        if cfg.output.use_hillshade:
            fig, ax, _ = _show_base_map(
                x, y, zb, dx, dy, f"{label}, t={t:.2f} s",
                hillshade_alpha=cfg.output.hillshade_alpha,
                elevation_colorbar=False,
            )
            plotted = np.ma.masked_where(~f["wet"], arr)
            im = ax.imshow(plotted, origin="lower", extent=extent, interpolation="bilinear",
                           cmap=cmap, alpha=0.84)
        else:
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(arr, origin="lower", extent=extent, interpolation="bilinear", cmap=cmap)
            ax.contour(zb, levels=15, origin="lower", extent=extent, linewidths=0.35)
            ax.set_aspect("equal")
            ax.set_title(f"{label}, t={t:.2f} s")
            ax.set_xlabel("Easting [m]")
            ax.set_ylabel("Northing [m]")
        fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{name}_{suffix}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def make_depth_gif(snapshot_paths: Sequence[str], out_path: str, cfg: SolverConfig) -> Optional[str]:
    if not snapshot_paths:
        return None
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    _apply_publication_plot_style()

    frames: List[np.ndarray] = []
    vmax = 0.0
    records = []
    for path in snapshot_paths:
        with np.load(path) as d:
            rec = (float(d["t"]), d["x"].copy(), d["y"].copy(), d["zb"].copy(), d["h"].copy())
            records.append(rec)
            vmax = max(vmax, float(np.percentile(rec[4][rec[4] > cfg.numerics.h_dry], 99.0))
                       if np.any(rec[4] > cfg.numerics.h_dry) else 0.0)
    vmax = max(vmax, 1.0e-6)
    for t, x, y, zb, h in records:
        dx = float(abs(x[1] - x[0])); dy = float(abs(y[1] - y[0]))
        extent = _map_extent(x, y, dx, dy)
        if cfg.output.use_hillshade:
            fig, ax, _ = _show_base_map(
                x, y, zb, dx, dy, f"Debris-flow depth, t={t:.1f} s",
                hillshade_alpha=cfg.output.hillshade_alpha,
                elevation_colorbar=False,
            )
            hplot = np.ma.masked_where(h <= cfg.numerics.h_dry, h)
            im = ax.imshow(hplot, origin="lower", extent=extent, vmin=0.0, vmax=vmax,
                           interpolation="bilinear", cmap="turbo", alpha=0.86)
        else:
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(h, origin="lower", extent=extent, vmin=0.0, vmax=vmax,
                           interpolation="bilinear", cmap="turbo")
            ax.set_aspect("equal")
            ax.set_title(f"Debris-flow depth, t={t:.1f} s")
            ax.set_xlabel("Easting [m]")
            ax.set_ylabel("Northing [m]")
        fig.colorbar(im, ax=ax, label="h [m]", fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out_path, frames, fps=max(1, int(cfg.output.gif_fps)))
    return out_path
