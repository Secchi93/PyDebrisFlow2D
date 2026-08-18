from __future__ import annotations

"""Interactive polygons, masks, hillshade, and setup previews."""

import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml

from .config import SolverConfig

# -----------------------------------------------------------------------------
# Interactive geometry and hillshade visualization.
# -----------------------------------------------------------------------------


def _apply_publication_plot_style() -> None:
    """Use manuscript-scale typography for every solver-generated map/preview."""
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
def _hillshade(zb: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Return a north-up analytical hillshade in the range [0, 255]."""
    z = np.asarray(zb, dtype=np.float64)
    finite = np.isfinite(z)
    if not np.any(finite):
        return np.zeros_like(z, dtype=np.uint8)
    fill = float(np.nanmedian(z[finite]))
    z = np.where(finite, z, fill)
    dz_dy, dz_dx = np.gradient(z, dy, dx)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(dz_dy, -dz_dx)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(45.0)
    shade = (np.sin(altitude) * np.cos(slope)
             + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect))
    lo, hi = np.percentile(shade[finite], [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip(255.0 * (shade - lo) / (hi - lo), 0.0, 255.0)
    return out.astype(np.uint8)


def _map_extent(x: np.ndarray, y: np.ndarray, dx: float, dy: float) -> List[float]:
    return [
        float(x.min() - 0.5 * dx), float(x.max() + 0.5 * dx),
        float(y.min() - 0.5 * dy), float(y.max() + 0.5 * dy),
    ]


def _show_base_map(
    x: np.ndarray,
    y: np.ndarray,
    zb: np.ndarray,
    dx: float,
    dy: float,
    title: str,
    hillshade_alpha: float = 0.28,
    elevation_colorbar: bool = True,
):
    import matplotlib.pyplot as plt
    _apply_publication_plot_style()

    extent = _map_extent(x, y, dx, dy)
    valid = np.isfinite(zb)
    z_plot = np.ma.masked_where(~valid, zb)
    fig, ax = plt.subplots(figsize=(10, 8))
    elev = ax.imshow(
        z_plot, cmap="terrain", extent=extent, origin="lower",
        interpolation="bilinear",
    )
    if hillshade_alpha > 0.0:
        hs = _hillshade(zb, dx, dy)
        ax.imshow(
            np.ma.masked_where(~valid, hs), cmap="gray", extent=extent,
            origin="lower", alpha=float(hillshade_alpha), interpolation="bilinear",
            vmin=0.0, vmax=255.0,
        )
    if elevation_colorbar:
        fig.colorbar(elev, ax=ax, label="Quota [m]", fraction=0.046, pad=0.04)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.annotate(
        "N", xy=(0.965, 0.965), xytext=(0.965, 0.855),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=13, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black"),
    )
    ax.text(
        0.01, 0.99, "Nord in alto", transform=ax.transAxes,
        ha="left", va="top", fontsize=11,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="0.6"),
    )
    fig.tight_layout()
    return fig, ax, extent


def click_polygon(
    x: np.ndarray,
    y: np.ndarray,
    zb: np.ndarray,
    dx: float,
    dy: float,
    title: str,
    hillshade_alpha: float = 0.28,
) -> List[List[float]]:
    """Draw a polygon with live feedback: left-click, right-click undo, Enter finish."""
    import matplotlib.pyplot as plt

    fig, ax, _ = _show_base_map(
        x, y, zb, dx, dy,
        title + "\nLeft click: add vertex | right click: undo | Enter: confirm | Esc: cancel",
        hillshade_alpha=hillshade_alpha,
    )
    line, = ax.plot([], [], "o-", linewidth=2.0, markersize=6.5, color="tab:red")
    status = ax.text(
        0.01, 0.01, "Add at least 3 vertices.", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=11.5,
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="0.5"),
    )
    state: Dict[str, Any] = {"points": [], "confirmed": False, "cancelled": False}

    def redraw() -> None:
        pts = state["points"]
        if not pts:
            line.set_data([], [])
        else:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if len(pts) >= 3:
                xs = xs + [xs[0]]
                ys = ys + [ys[0]]
            line.set_data(xs, ys)
        status.set_text(f"Vertices: {len(pts)}" + (" | press Enter to confirm" if len(pts) >= 3 else " | at least 3 are required"))
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes is not ax:
            return
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            state["points"].append((float(event.xdata), float(event.ydata)))
            redraw()
            if bool(getattr(event, "dblclick", False)) and len(state["points"]) >= 3:
                state["confirmed"] = True
                plt.close(fig)
        elif event.button == 3 and state["points"]:
            state["points"].pop()
            redraw()

    def on_key(event) -> None:
        key = (event.key or "").lower()
        if key in ("enter", "return"):
            if len(state["points"]) < 3:
                status.set_text("Invalid polygon: add at least 3 vertices.")
                fig.canvas.draw_idle()
                return
            state["confirmed"] = True
            plt.close(fig)
        elif key in ("escape", "esc", "q"):
            state["cancelled"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=True)
    if state["cancelled"]:
        raise RuntimeError(f"Selection cancelled: {title}")
    if not state["confirmed"] or len(state["points"]) < 3:
        raise RuntimeError(f"Polygon was not confirmed: {title}")
    return [[float(px), float(py)] for px, py in state["points"]]


def polygon_mask(x: np.ndarray, y: np.ndarray, polygon: Sequence[Sequence[float]]) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.ravel(), Y.ravel()))
    return MplPath(np.asarray(polygon, dtype=float)).contains_points(points).reshape(X.shape)


def prepare_interactive_geometry(
    cfg: SolverConfig,
    x: np.ndarray,
    y: np.ndarray,
    zb: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Acquire friction and release polygons, then build spatial override maps."""
    fr = cfg.friction
    regions = list(fr.regions or [])
    if fr.interactive and len(regions) < 2:
        print("[UI] Draw friction region 1 and press Enter when the polygon is complete.")
        poly1 = click_polygon(x, y, zb, dx, dy, "Friction region 1", cfg.output.hillshade_alpha)
        print("[UI] Draw friction region 2 and press Enter when the polygon is complete.")
        poly2 = click_polygon(x, y, zb, dx, dy, "Friction region 2", cfg.output.hillshade_alpha)
        regions = [
            {"name": "friction_region_1", "mu": float(fr.region_1_mu), "xi": float(fr.region_1_xi), "polygon": poly1},
            {"name": "friction_region_2", "mu": float(fr.region_2_mu), "xi": float(fr.region_2_xi), "polygon": poly2},
        ]
        fr.regions = regions

    mu_override = np.full(zb.shape, np.nan, dtype=np.float64)
    xi_override = np.full(zb.shape, np.nan, dtype=np.float64)
    for index, region in enumerate(regions, start=1):
        if not isinstance(region, dict):
            raise RuntimeError(f"friction.regions[{index-1}] must be a mapping")
        poly = region.get("polygon", [])
        if len(poly) < 3:
            raise RuntimeError(f"Friction region {index} requires at least 3 polygon vertices")
        mu = float(region.get("mu", fr.region_1_mu if index == 1 else fr.region_2_mu))
        xi = float(region.get("xi", fr.region_1_xi if index == 1 else fr.region_2_xi))
        if mu < 0.0 or xi <= 0.0:
            raise RuntimeError(f"Invalid mu/xi in friction region {index}")
        mask = polygon_mask(x, y, poly) & valid
        if not np.any(mask):
            raise RuntimeError(f"Friction region {index} contains no valid DEM cells")
        mu_override[mask] = mu
        xi_override[mask] = xi
        print(f"[FRICTION] region={index} cells={int(mask.sum())} mu={mu:g} xi={xi:g}")

    if cfg.release.enabled and cfg.release.interactive_polygon:
        print("[UI] Draw the release polygon and press Enter when it is complete.")
        cfg.release.mode = "polygon"
        cfg.release.polygon = click_polygon(
            x, y, zb, dx, dy, "Release polygon", cfg.output.hillshade_alpha,
        )
    return mu_override, xi_override


def save_setup_preview(
    out_dir: str,
    cfg: SolverConfig,
    x: np.ndarray,
    y: np.ndarray,
    zb: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
    release_mask: np.ndarray,
) -> str:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch

    fig, ax, extent = _show_base_map(
        x, y, zb, dx, dy, "Confirmed interactive geometry",
        hillshade_alpha=cfg.output.hillshade_alpha,
    )
    colors = ["tab:orange", "tab:purple"]
    for i, region in enumerate(cfg.friction.regions or []):
        poly = np.asarray(region.get("polygon", []), dtype=float)
        if poly.shape[0] < 3:
            continue
        patch = PolygonPatch(poly, closed=True, fill=True, facecolor=colors[i % len(colors)],
                             edgecolor=colors[i % len(colors)], alpha=0.24, linewidth=2.0,
                             label=f"Friction region {i+1}: μ={float(region.get('mu')):g}, ξ={float(region.get('xi')):g}")
        ax.add_patch(patch)
    release_overlay = np.ma.masked_where(~release_mask, release_mask.astype(float))
    ax.imshow(release_overlay, cmap="Reds", alpha=0.58, extent=extent, origin="lower", vmin=0.0, vmax=1.0)
    if str(cfg.release.mode).lower() == "polygon" and len(cfg.release.polygon) >= 3:
        rp = np.asarray(cfg.release.polygon, dtype=float)
        ax.add_patch(PolygonPatch(rp, closed=True, fill=False, edgecolor="red", linewidth=2.2,
                                  label=f"Release: {cfg.release.volume_m3:g} m³"))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", framealpha=0.9)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "interactive_setup.png")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    geometry = {
        "friction": {"regions": cfg.friction.regions},
        "release": {"mode": cfg.release.mode, "polygon": cfg.release.polygon, "volume_m3": cfg.release.volume_m3},
    }
    with open(os.path.join(out_dir, "runtime_geometry.yaml"), "w", encoding="utf-8") as stream:
        yaml.safe_dump(geometry, stream, sort_keys=False)
    print("[UI] Geometry preview saved:", path)
    return path
