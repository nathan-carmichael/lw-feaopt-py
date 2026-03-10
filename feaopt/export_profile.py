"""
exportProfile — Rasterise optimised structure, extract boundaries, export CAD profiles.

Port of exportProfile.m from ShepherdLab LW FEA-OPT.

Dependencies: numpy, scipy, scikit-image (skimage)
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import os


DEFAULT_OPTIONS = {
    'num_spline_points': 1000,
    'smoothing': 1.0,
    'scale_to_mm': 1000,
    'resolution': 25000,
    'min_thickness_frac': 0,
    'export_filename': None,
    'plot': True,
    'verbose': True,
}


def export_profile(geometry, options=None):
    """
    Rasterise an optimised structure, extract boundary loops, fit splines,
    and export SolidWorks-ready point files.

    Parameters
    ----------
    geometry : StructureGeometry
    options : dict, optional

    Returns
    -------
    profile : dict
    """
    opt = {**DEFAULT_OPTIONS, **(options or {})}
    scale = opt['scale_to_mm']

    # ── Filter thin elements ──────────────────────────────────────────
    thickness = geometry.thickness
    is_rigid = geometry.is_rigid
    active = ~is_rigid

    if opt['min_thickness_frac'] > 0:
        t_max = np.max(thickness[active])
        t_thresh = t_max * opt['min_thickness_frac']
        active = active & (thickness >= t_thresh)
        if opt['verbose']:
            n_removed = int(np.sum(~is_rigid)) - int(np.sum(active))
            if n_removed > 0:
                print(f"  Filtered {n_removed} thin elements (t < {t_thresh*scale:.3f} mm)")

    active_ids = np.where(active)[0]
    if len(active_ids) == 0:
        raise ValueError("No elements remain after filtering.")

    # ── Build element rectangles ──────────────────────────────────────
    rects = []
    for i in active_ids:
        n1, n2 = geometry.get_member_nodes(i)
        x1, y1 = geometry.nodes[n1]
        x2, y2 = geometry.nodes[n2]
        theta = np.arctan2(y2 - y1, x2 - x1)
        ht = geometry.thickness[i] / 2
        px = -np.sin(theta) * ht
        py = np.cos(theta) * ht
        rects.append(np.array([
            [x1+px, y1+py], [x2+px, y2+py],
            [x2-px, y2-py], [x1-px, y1-py],
        ]))

    # ── Image bounds ──────────────────────────────────────────────────
    all_corners = np.vstack(rects)
    x_min, x_max = all_corners[:, 0].min(), all_corners[:, 0].max()
    y_min, y_max = all_corners[:, 1].min(), all_corners[:, 1].max()
    pad = max((x_max - x_min), (y_max - y_min)) * 0.05
    x_min -= pad; x_max += pad
    y_min -= pad; y_max += pad

    # ── Rasterise ─────────────────────────────────────────────────────
    res = opt['resolution']
    img_w = max(2, round((x_max - x_min) * res))
    img_h = max(2, round((y_max - y_min) * res))

    max_dim = 8000
    if img_w > max_dim or img_h > max_dim:
        shrink = max_dim / max(img_w, img_h)
        img_w = round(img_w * shrink)
        img_h = round(img_h * shrink)
        res = img_w / (x_max - x_min)
        if opt['verbose']:
            print(f"  Image capped at {img_w}x{img_h} pixels")

    bw = np.zeros((img_h, img_w), dtype=bool)

    try:
        from skimage.draw import polygon as sk_polygon
    except ImportError:
        raise ImportError("scikit-image is required for profile export. Install with: pip install scikit-image")

    for corners in rects:
        px_x = np.round((corners[:, 0] - x_min) / (x_max - x_min) * (img_w - 1)).astype(int)
        px_y = np.round((corners[:, 1] - y_min) / (y_max - y_min) * (img_h - 1)).astype(int)
        px_x = np.clip(px_x, 0, img_w - 1)
        px_y = np.clip(px_y, 0, img_h - 1)
        px_r = img_h - 1 - px_y  # flip y for image coords

        rr, cc = sk_polygon(px_r, px_x, shape=(img_h, img_w))
        bw[rr, cc] = True

    if opt['verbose']:
        print(f"  Rasterised {len(active_ids)} elements onto {img_w} x {img_h} image")

    # ── Morphological cleanup ─────────────────────────────────────────
    try:
        from skimage.morphology import disk, closing, remove_small_objects
        se_r = max(1, min(3, round(res * np.max(thickness[active]) * 0.1)))
        bw = closing(bw, disk(se_r))
        min_size = round(0.001 * img_w * img_h)
        if min_size > 0:
            bw = remove_small_objects(bw, min_size=max(1, min_size))
    except ImportError:
        pass  # skip morphological cleanup if skimage not fully available

    # ── Extract boundary contours ─────────────────────────────────────
    try:
        from skimage.measure import find_contours
    except ImportError:
        raise ImportError("scikit-image is required for boundary extraction.")

    contours = find_contours(bw.astype(float), 0.5)

    if opt['verbose']:
        print(f"  Found {len(contours)} boundary loops")

    # ── Fit splines to each loop ──────────────────────────────────────
    loops = []
    for bd in contours:
        # bd is [row, col] — convert to world coords
        world_x = (bd[:, 1]) / (img_w - 1) * (x_max - x_min) + x_min
        world_y = (img_h - 1 - bd[:, 0]) / (img_h - 1) * (y_max - y_min) + y_min

        # Remove duplicate consecutive points
        dx = np.diff(world_x)
        dy = np.diff(world_y)
        ds = np.sqrt(dx**2 + dy**2)
        keep = np.concatenate([[True], ds > 1e-12])
        world_x = world_x[keep]
        world_y = world_y[keep]

        # Ensure closed
        if np.linalg.norm([world_x[-1] - world_x[0], world_y[-1] - world_y[0]]) > 1e-10:
            world_x = np.append(world_x, world_x[0])
            world_y = np.append(world_y, world_y[0])

        # Arc-length parameterisation
        ds_seg = np.sqrt(np.diff(world_x)**2 + np.diff(world_y)**2)
        s = np.concatenate([[0], np.cumsum(ds_seg)])
        total_perim = s[-1]

        # Spline fit
        if len(s) >= 4 and total_perim > 0:
            s_fine = np.linspace(0, total_perim, opt['num_spline_points'] + 1)
            s_fine = s_fine[:-1]

            try:
                xs = interp1d(s, world_x, kind='cubic', fill_value='extrapolate')(s_fine)
                ys = interp1d(s, world_y, kind='cubic', fill_value='extrapolate')(s_fine)
            except Exception:
                xs = np.interp(s_fine, s, world_x)
                ys = np.interp(s_fine, s, world_y)

            # Smooth the interpolated curve (periodic-aware)
            sigma = max(1, int(opt['num_spline_points'] * (1 - opt['smoothing'])))
            xs = gaussian_filter1d(xs, sigma=sigma, mode='wrap')
            ys = gaussian_filter1d(ys, sigma=sigma, mode='wrap')

            xs = np.append(xs, xs[0])
            ys = np.append(ys, ys[0])
        else:
            xs, ys = world_x, world_y

        loop = {
            'x': world_x * scale,
            'y': world_y * scale,
            'xs': xs * scale,
            'ys': ys * scale,
            'is_hole': False,  # simplified — full hole detection requires more logic
            'arc_length': total_perim * scale,
            'num_raw_points': len(world_x),
        }
        loops.append(loop)

    profile = {
        'loops': loops,
        'num_loops': len(loops),
        'image': bw,
        'x_range': [x_min, x_max],
        'y_range': [y_min, y_max],
        'scale': scale,
        'export_files': [],
    }

    # ── Export ─────────────────────────────────────────────────────────
    if opt['export_filename'] is not None:
        profile['export_files'] = _write_files(profile, opt)

    # ── Plot ──────────────────────────────────────────────────────────
    if opt['plot']:
        _plot_profile(profile)

    return profile


def _write_files(profile, opt):
    base = opt['export_filename']
    files = []

    for b, loop in enumerate(profile['loops']):
        tag = f"hole_{b}" if loop['is_hole'] else f"loop_{b}"
        f = f"{base}_{tag}.txt"
        _write_xyz(f, loop['xs'], loop['ys'])
        files.append(f)

    # Save as numpy archive
    f = f"{base}.npz"
    np.savez(f, **{f"loop_{i}_xs": l['xs'] for i, l in enumerate(profile['loops'])},
                **{f"loop_{i}_ys": l['ys'] for i, l in enumerate(profile['loops'])})
    files.append(f)

    if opt['verbose']:
        print(f"\n  Exported files:")
        for ff in files:
            print(f"    {ff}")
        print(f"\n  SolidWorks import:")
        print(f"    Insert > Curve > Curve Through XYZ Points")
        print(f"    Import each loop file as a separate spline.")

    return files


def _write_xyz(filename, x, y):
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with open(filename, 'w') as f:
        for xi, yi in zip(x, y):
            f.write(f"{xi:.6f} {yi:.6f} 0.000000\n")


def _plot_profile(profile):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    blue = (0.3, 0.55, 0.85)

    for loop in profile['loops']:
        if loop['is_hole']:
            ax.fill(loop['xs'], loop['ys'], facecolor='white',
                    edgecolor=blue, linewidth=1.5, linestyle='--')
        else:
            ax.fill(loop['xs'], loop['ys'], facecolor=(0.75, 0.85, 0.95),
                    edgecolor=blue, linewidth=2, alpha=0.5)

    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title('Optimized Geometry Profile')
    plt.tight_layout()
    plt.show()