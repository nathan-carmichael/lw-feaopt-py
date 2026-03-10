"""
StructurePlotter — Visualization utilities for 2-D frame structures.

Port of StructurePlotter.m — uses PatchCollection for fast batch rendering.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.collections import PolyCollection
    from matplotlib import cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _make_beam_polygon(x1, y1, x2, y2, thickness):
    """Return 4 corner points for a beam rectangle."""
    theta = np.pi / 2 - np.arctan2(y2 - y1, x2 - x1)
    ht = thickness / 2
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [x1 + ht*ct, y1 - ht*st],
        [x1 - ht*ct, y1 + ht*st],
        [x2 - ht*ct, y2 + ht*st],
        [x2 + ht*ct, y2 - ht*st],
    ])


def draw_beams_batch(ax, nodes, connectivity, thickness, colors):
    M = connectivity.shape[0]
    colors = np.asarray(colors)
    if colors.ndim == 1:
        colors = np.tile(colors, (M, 1))

    # Vectorized vertex computation — no Python loop
    n1 = connectivity[:, 0]
    n2 = connectivity[:, 1]
    x1, y1 = nodes[n1, 0], nodes[n1, 1]
    x2, y2 = nodes[n2, 0], nodes[n2, 1]
    theta = np.pi / 2 - np.arctan2(y2 - y1, x2 - x1)
    ht = thickness / 2
    ct, st = np.cos(theta), np.sin(theta)

    # Build (M, 4, 2) array of vertices
    verts = np.zeros((M, 4, 2))
    verts[:, 0, 0] = x1 + ht * ct;  verts[:, 0, 1] = y1 - ht * st
    verts[:, 1, 0] = x1 - ht * ct;  verts[:, 1, 1] = y1 + ht * st
    verts[:, 2, 0] = x2 - ht * ct;  verts[:, 2, 1] = y2 + ht * st
    verts[:, 3, 0] = x2 + ht * ct;  verts[:, 3, 1] = y2 - ht * st

    pc = PolyCollection(verts, edgecolors='none')
    pc.set_facecolor(colors[:, :3])
    if colors.shape[1] == 4:
        pc.set_alpha(colors[:, 3])
    ax.add_collection(pc)

    pad = np.max(thickness) * 2
    ax.set_xlim(nodes[connectivity.ravel(), 0].min() - pad,
                nodes[connectivity.ravel(), 0].max() + pad)
    ax.set_ylim(nodes[connectivity.ravel(), 1].min() - pad,
                nodes[connectivity.ravel(), 1].max() + pad)


def draw_beams_stress(ax, nodes, connectivity, thickness, stress_values, cmap_name='jet', vmin=0, vmax=None):
    """Draw beams coloured by stress. Returns the ScalarMappable for colorbar.

    Parameters
    ----------
    stress_values : (M,) array — one value per element (e.g. max von Mises in MPa)
    """
    M = connectivity.shape[0]
    if vmax is None:
        vmax = np.nanmax(stress_values)
    if vmax < 1e-12:
        vmax = 1.0

    cmap_obj = cm.get_cmap(cmap_name)
    norm = plt.Normalize(vmin, vmax)
    colors = cmap_obj(norm(np.nan_to_num(stress_values, nan=0.0)))[:, :3]

    draw_beams_batch(ax, nodes, connectivity, thickness, colors)

    sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    return sm


class StructurePlotter:
    """Visualization for 2-D frame structures using matplotlib."""

    def __init__(self, geometry, analysis=None):
        if not HAS_MPL:
            raise ImportError("matplotlib is required for StructurePlotter.")
        self.geometry = geometry
        self.analysis = analysis
        self.colors = {
            'member_original': (0.2, 0.4, 0.8),
            'member_deformed': (0.8, 0.2, 0.2),
            'member_ghost': (0.85, 0.85, 0.85),
            'rigid': (0.3, 0.3, 0.3),
            'node': (0.1, 0.1, 0.1),
            'load': (0, 0.6, 0),
        }

    # ── Single beam drawing (for backward compat) ─────────────────────

    def plot_beam(self, ax, x, y, thickness, color):
        """Draw a single beam segment. For batch drawing, use draw_beams_batch."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        for k in range(len(y) - 1):
            c = color[k] if np.ndim(color) > 1 else color
            t = thickness if np.isscalar(thickness) else thickness[k]
            verts = _make_beam_polygon(x[k], y[k], x[k+1], y[k+1], t)
            ax.add_patch(Polygon(verts, closed=True, facecolor=c, edgecolor='none'))

    # ── Batch structure drawing ───────────────────────────────────────

    def draw_structure(self, ax, nodes=None, thickness=None, color=None):
        """Draw all elements at once (fast)."""
        g = self.geometry
        if nodes is None:
            nodes = g.nodes
        if thickness is None:
            thickness = g.thickness
        if color is None:
            color = self.colors['member_original']
        draw_beams_batch(ax, nodes, g.connectivity, thickness, color)

    def draw_stress(self, ax, nodes=None, thickness=None, stress_values=None, vmax=None):
        """Draw all elements coloured by stress (fast). Returns ScalarMappable."""
        g = self.geometry
        if nodes is None:
            nodes = g.nodes
        if thickness is None:
            thickness = g.thickness
        return draw_beams_stress(ax, nodes, g.connectivity, thickness,
                                 stress_values, vmax=vmax)

    # ── Support symbols ───────────────────────────────────────────────

    def plot_supports(self, ax):
        geom = self.geometry
        sz = 0.02 * max(np.ptp(geom.nodes[:, 0]), np.ptp(geom.nodes[:, 1]))

        node_con = np.zeros((geom.num_nodes, 3), dtype=bool)
        for c in geom.constraints:
            node_con[c.node_id, c.dof] = True

        for n in range(geom.num_nodes):
            if not np.any(node_con[n]):
                continue
            xy = geom.get_node_coords(n)
            if all(node_con[n]):
                tx = xy[0] + np.array([-1, 1, 0, -1]) * sz
                ty = xy[1] + np.array([-1, -1, 0.5, -1]) * sz
                ax.fill(tx, ty, 'k')
            elif node_con[n, 0] and node_con[n, 1]:
                circ = plt.Circle((xy[0], xy[1]), sz/2, fill=False,
                                  edgecolor='k', linewidth=2)
                ax.add_patch(circ)

    # ── Load arrows ───────────────────────────────────────────────────

    def plot_loads(self, ax, load_case, scale=None):
        if scale is None:
            span = max(np.ptp(self.geometry.nodes[:, 0]), np.ptp(self.geometry.nodes[:, 1]))
            load_mag = load_case.get_total_load()['force_magnitude']
            scale = 0.1 * span / max(load_mag, 1.0)

        for rec in load_case.load_records:
            xy = self.geometry.get_node_coords(rec.node_id)
            fx = rec.fx * load_case.lambda_
            fy = rec.fy * load_case.lambda_
            if abs(fx) > np.finfo(float).eps:
                ax.quiver(xy[0], xy[1], fx*scale, 0, color=self.colors['load'],
                          scale=1, scale_units='xy', width=0.003)
            if abs(fy) > np.finfo(float).eps:
                ax.quiver(xy[0], xy[1], 0, fy*scale, color=self.colors['load'],
                          scale=1, scale_units='xy', width=0.003)