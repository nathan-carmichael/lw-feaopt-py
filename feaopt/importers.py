"""
Geometry importers — CSV and IGES curve import with arc-length resampling.

Port of importCSVCurve.m and importIGSCurve.m from ShepherdLab LW FEA-OPT.

Fixed to match MATLAB behaviour: reads X,Y from control points, skips Z
and transformation matrices (entity 124), matching SolidWorks 2D sketch export.
"""

import numpy as np
import os


# ═══════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def _resample_by_arc_length(pts, num_el):
    """Resample a polyline to num_el+1 nodes with uniform arc-length spacing."""
    dxy = np.diff(pts, axis=0)
    ds = np.sqrt(dxy[:, 0]**2 + dxy[:, 1]**2)
    cum_s = np.concatenate([[0], np.cumsum(ds)])
    total_arc = cum_s[-1]

    s_target = np.linspace(0, total_arc, num_el + 1)
    nodes = np.zeros((num_el + 1, 2))
    nodes[0] = pts[0]
    nodes[-1] = pts[-1]

    for i in range(1, num_el):
        lo = np.searchsorted(cum_s, s_target[i], side='right') - 1
        lo = max(0, min(lo, len(pts) - 2))
        hi = lo + 1
        if lo == hi or cum_s[hi] == cum_s[lo]:
            nodes[i] = pts[lo]
        else:
            frac = (s_target[i] - cum_s[lo]) / (cum_s[hi] - cum_s[lo])
            nodes[i] = pts[lo] + frac * (pts[hi] - pts[lo])

    return nodes, total_arc


def _edge_tangent(p1, p2):
    d = p2 - p1
    tang = d / np.linalg.norm(d)
    angle = np.arctan2(tang[1], tang[0])
    return tang, angle


# ═══════════════════════════════════════════════════════════════════════
#  CSV Importer
# ═══════════════════════════════════════════════════════════════════════

def import_csv_curve(filename, num_elements=250, scale=1.0, flip=False,
                     plot=False, verbose=True):
    """
    Import a 2-D curve from a CSV of (x, y) coordinates.

    Returns dict with: nodes, connectivity, num_nodes, num_elements,
    arc_length, raw_points, start_tangent, end_tangent, start_angle, end_angle
    """
    raw = np.genfromtxt(filename, delimiter=',', skip_header=0)

    # Auto-detect header
    if np.any(np.isnan(raw[0])):
        raw = raw[1:]

    if raw.shape[1] > 2:
        raw = raw[:, :2]
    raw = raw[~np.any(np.isnan(raw), axis=1)]

    if raw.shape[0] < 2:
        raise ValueError(f"Need at least 2 points, found {raw.shape[0]}.")

    raw_points = raw * scale
    if flip:
        raw_points = raw_points[::-1]

    if verbose:
        print(f"\n=== CSV Import: {filename} ===")
        print(f"  Raw points: {raw_points.shape[0]}")

    # Remove duplicate consecutive points
    seg_len = np.sqrt(np.sum(np.diff(raw_points, axis=0)**2, axis=1))
    keep = np.concatenate([[True], seg_len > 1e-12])
    raw_points = raw_points[keep]

    nodes, total_arc = _resample_by_arc_length(raw_points, num_elements)

    num_nodes = nodes.shape[0]
    connectivity = np.column_stack([np.arange(num_elements), np.arange(1, num_elements + 1)])

    start_tangent, start_angle = _edge_tangent(nodes[0], nodes[1])
    end_tangent, end_angle = _edge_tangent(nodes[-2], nodes[-1])

    result = {
        'nodes': nodes,
        'connectivity': connectivity,
        'num_nodes': num_nodes,
        'num_elements': num_elements,
        'arc_length': total_arc,
        'raw_points': raw_points,
        'start_tangent': start_tangent,
        'end_tangent': end_tangent,
        'start_angle': start_angle,
        'end_angle': end_angle,
    }

    if verbose:
        print(f"  Arc length:    {total_arc:.4f} (scaled units)")
        print(f"  Nodes: {num_nodes}  |  Elements: {num_elements}")
        print(f"  Start: [{nodes[0,0]:.4f}, {nodes[0,1]:.4f}]  tang: {np.degrees(start_angle):.1f} deg")
        print(f"  End:   [{nodes[-1,0]:.4f}, {nodes[-1,1]:.4f}]  tang: {np.degrees(end_angle):.1f} deg")

    if plot:
        _plot_import_result(raw_points, nodes, total_arc, num_elements,
                            start_tangent, end_tangent, 'CSV Import')

    return result


# ═══════════════════════════════════════════════════════════════════════
#  IGES Importer
# ═══════════════════════════════════════════════════════════════════════

_SUPPORTED_TYPES = {100, 104, 106, 110, 112, 126}

_TYPE_NAMES = {
    100: 'Circular Arc', 104: 'Conic Arc', 106: 'Copious Data',
    110: 'Line', 112: 'Parametric Spline', 126: 'B-Spline',
}


def import_igs_curve(filename, num_elements=250, scale=1.0,
                     samples_per_entity=300, flip=False,
                     plot=False, verbose=True):
    """
    Import a 2-D curve from an IGES/IGS file.

    Supported entities: 110 (Line), 100 (Arc), 104 (Conic), 112 (Param Spline),
    126 (B-Spline/NURBS), 106 (Copious Data).

    Returns dict with same layout as import_csv_curve, plus raw_entities.
    """
    with open(filename, 'r') as f:
        all_lines = f.read().splitlines()

    d_lines, p_lines = [], []
    for ln in all_lines:
        if len(ln) < 73:
            continue
        sec = ln[72]
        if sec == 'D':
            d_lines.append(ln)
        elif sec == 'P':
            p_lines.append(ln)

    # Parse directory entries
    num_de = len(d_lines) // 2
    dir_entries = []
    for i in range(num_de):
        l1 = d_lines[2*i]
        l2 = d_lines[2*i + 1]
        etype = int(l1[0:8].strip())
        pstart = int(l1[8:16].strip())
        try:
            pc = int(l2[24:32].strip())
        except (ValueError, IndexError):
            pc = 1
        dir_entries.append({'entity_type': etype, 'param_start': pstart, 'param_count': pc})

    # Collect parameter data
    p_map = {}
    for ln in p_lines:
        seq = int(ln[73:80].strip())
        dat = ln[0:64].strip()
        p_map[seq] = p_map.get(seq, '') + dat

    # Build entity list
    entities = []
    for de in dir_entries:
        if de['entity_type'] not in _SUPPORTED_TYPES:
            continue
        pstr = ''
        for p in range(de['param_start'], de['param_start'] + de['param_count']):
            if p in p_map:
                pstr += p_map[p]
        pstr = pstr.replace(';', '')
        parts = pstr.split(',')
        vals = []
        for part in parts:
            try:
                vals.append(float(part.strip()))
            except ValueError:
                vals.append(0.0)
        entities.append({'type': de['entity_type'], 'params': np.array(vals)})

    if verbose:
        print(f"\n=== IGES Import: {filename} ===")
        print(f"  Directory entries: {len(dir_entries)}")
        print(f"  Curve entities:   {len(entities)}")
        for i, e in enumerate(entities):
            print(f"    {i}: type {e['type']} ({_TYPE_NAMES.get(e['type'], 'Unknown')})")

    if not entities:
        raise ValueError(f"No supported curve entities found in {filename}.")

    # Sample each entity
    polys = [_sample_entity(e, samples_per_entity) for e in entities]
    # Remove empties
    keep = [i for i, p in enumerate(polys) if p is not None and len(p) > 0]
    polys = [polys[i] for i in keep]
    entities = [entities[i] for i in keep]

    if not polys:
        raise ValueError("No valid curve data extracted.")

    # Chain entities
    chained = _chain_entities(polys, verbose)
    if chained is None or len(chained) == 0:
        raise ValueError("Could not chain entities into a connected curve.")

    chained = chained * scale
    if flip:
        chained = chained[::-1]

    nodes, total_arc = _resample_by_arc_length(chained, num_elements)

    num_nodes = nodes.shape[0]
    connectivity = np.column_stack([np.arange(num_elements), np.arange(1, num_elements + 1)])

    start_tangent, start_angle = _edge_tangent(nodes[0], nodes[1])
    end_tangent, end_angle = _edge_tangent(nodes[-2], nodes[-1])

    result = {
        'nodes': nodes,
        'connectivity': connectivity,
        'num_nodes': num_nodes,
        'num_elements': num_elements,
        'arc_length': total_arc,
        'raw_entities': entities,
        'raw_points': chained,
        'start_tangent': start_tangent,
        'end_tangent': end_tangent,
        'start_angle': start_angle,
        'end_angle': end_angle,
    }

    if verbose:
        print(f"\n  Arc length:    {total_arc:.4f} (scaled units)")
        print(f"  Nodes: {num_nodes}  |  Elements: {num_elements}")
        print(f"  Start: [{nodes[0,0]:.4f}, {nodes[0,1]:.4f}]  tang: {np.degrees(start_angle):.1f} deg")
        print(f"  End:   [{nodes[-1,0]:.4f}, {nodes[-1,1]:.4f}]  tang: {np.degrees(end_angle):.1f} deg")

    if plot:
        _plot_igs_import_result(polys, scale, chained, nodes, total_arc,
                                num_elements, start_tangent, end_tangent)

    return result


# ═══════════════════════════════════════════════════════════════════════
#  IGES Entity Sampling — 2D (X,Y), matching MATLAB implementation
# ═══════════════════════════════════════════════════════════════════════

def _sample_entity(ent, n):
    t = ent['type']
    p = ent['params']
    if t == 110:
        return _sample_line_110(p, n)
    elif t == 100:
        return _sample_arc_100(p, n)
    elif t == 126:
        return _sample_bspline_126(p, n)
    elif t == 112:
        return _sample_param_spline_112(p, n)
    elif t == 106:
        return _sample_copious_106(p)
    elif t == 104:
        return _sample_conic_104(p, n)
    return None


def _sample_line_110(p, n):
    # Entity 110 — Line segment
    # p: [110, x1, y1, z1, x2, y2, z2]
    i = 1  # 0-based (MATLAB i=2)
    t = np.linspace(0, 1, n)
    x = p[i] + t * (p[i+3] - p[i])
    y = p[i+1] + t * (p[i+4] - p[i+1])
    return np.column_stack([x, y])


def _sample_arc_100(p, n):
    # Entity 100 — Circular arc (CCW)
    # p: [100, ZT, cx, cy, xs, ys, xe, ye]
    i = 1
    cx, cy = p[i+1], p[i+2]
    xs, ys = p[i+3], p[i+4]
    xe, ye = p[i+5], p[i+6]
    r = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    a0 = np.arctan2(ys - cy, xs - cx)
    a1 = np.arctan2(ye - cy, xe - cx)
    if a1 <= a0:
        a1 += 2 * np.pi
    ang = np.linspace(a0, a1, n)
    return np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])


def _sample_bspline_126(p, n):
    # Entity 126 — Rational B-Spline (NURBS)
    # Matches MATLAB sampleBSpline126 exactly
    i = 1  # 0-based (MATLAB i=2)
    K = int(round(p[i]))        # num_ctrl_pts - 1
    M_deg = int(round(p[i+1]))  # degree
    i += 6                       # skip K, M, PROP1..PROP4 (MATLAB: i = i + 6)

    N = K + 1              # number of control points
    A = N + M_deg + 1      # number of knots

    knots = p[i:i+A]
    i += A
    weights = p[i:i+N]
    i += N

    # Read control points as (X, Y), skip Z — matches MATLAB exactly
    cp = np.zeros((N, 2))
    for k in range(N):
        cp[k] = [p[i], p[i+1]]
        i += 3  # skip z

    # Parameter range
    if i + 1 < len(p):
        v0, v1 = p[i], p[i+1]
    else:
        v0 = knots[M_deg]
        v1 = knots[-M_deg-1] if M_deg > 0 else knots[-1]

    t_vals = np.linspace(v0, v1 - 1e-10, n)
    pts = np.zeros((n, 2))
    for k in range(n):
        pts[k] = _eval_nurbs(t_vals[k], cp, knots, weights, M_deg)
    return pts


def _eval_nurbs(t, cp, knots, weights, deg):
    """Evaluate NURBS curve at parameter t using de Boor's algorithm.

    Matches MATLAB evalNURBS exactly: homogeneous coords [w*x, w*y, w].
    """
    n_cp = len(cp)

    # Find knot span (MATLAB: k = deg+1, then while k < n_cp && knots(k+1) <= t)
    k = deg
    while k < n_cp - 1 and knots[k + 1] <= t:
        k += 1

    # De Boor in homogeneous coordinates [w*x, w*y, w]
    d = np.zeros((deg + 1, 3))
    for j in range(deg + 1):
        ci = k - deg + j
        if 0 <= ci < n_cp:
            w = weights[ci]
            d[j] = [cp[ci, 0] * w, cp[ci, 1] * w, w]

    for r in range(1, deg + 1):
        for j in range(deg, r - 1, -1):
            ci = k - deg + j
            denom = knots[ci + deg - r + 1] - knots[ci]
            if abs(denom) < 1e-14:
                alpha = 0.0
            else:
                alpha = (t - knots[ci]) / denom
            d[j] = (1 - alpha) * d[j-1] + alpha * d[j]

    hw = d[deg]
    if abs(hw[2]) > 1e-14:
        return np.array([hw[0] / hw[2], hw[1] / hw[2]])
    return np.array([hw[0], hw[1]])


def _sample_param_spline_112(p, n):
    # Entity 112 — Parametric spline (cubic polynomial segments)
    # Matches MATLAB sampleParamSpline112
    i = 1
    N_seg = int(round(p[i+3]))
    i += 4
    breaks = p[i:i+N_seg+1]
    i += N_seg + 1

    pts_list = []
    for s in range(N_seg):
        AX, BX, CX, DX = p[i], p[i+1], p[i+2], p[i+3]; i += 4
        AY, BY, CY, DY = p[i], p[i+1], p[i+2], p[i+3]; i += 4
        i += 4  # skip Z coefficients

        dt = breaks[s+1] - breaks[s]
        n_seg_pts = max(2, round(n / N_seg))
        tl = np.linspace(0, dt, n_seg_pts)

        xseg = AX + BX*tl + CX*tl**2 + DX*tl**3
        yseg = AY + BY*tl + CY*tl**2 + DY*tl**3

        seg = np.column_stack([xseg, yseg])
        if pts_list:
            seg = seg[1:]  # skip duplicate join point
        pts_list.append(seg)

    return np.vstack(pts_list) if pts_list else None


def _sample_copious_106(p):
    # Entity 106 — Copious data (point list)
    i = 1
    flag = int(round(p[i]))
    np_pts = int(round(p[i+1]))
    i += 2

    pts = np.zeros((np_pts, 2))
    if flag == 1:
        for k in range(np_pts):
            pts[k] = [p[i], p[i+1]]; i += 2
    elif flag == 2:
        for k in range(np_pts):
            pts[k] = [p[i], p[i+1]]; i += 3
    elif flag == 3:
        for k in range(np_pts):
            pts[k] = [p[i], p[i+1]]; i += 6
    else:
        return None
    return pts


def _sample_conic_104(p, n):
    # Entity 104 — Conic arc (simplified: straight line between endpoints)
    i = 1 + 6  # skip coefficients + ZT (0-based adjustment)
    x1, y1 = p[i], p[i+1]
    x2, y2 = p[i+2], p[i+3]
    t = np.linspace(0, 1, n)
    return np.column_stack([x1 + t*(x2-x1), y1 + t*(y2-y1)])


# ═══════════════════════════════════════════════════════════════════════
#  Entity chaining
# ═══════════════════════════════════════════════════════════════════════

def _chain_entities(polys, verbose=False):
    n_ent = len(polys)
    if n_ent == 0:
        return None
    if n_ent == 1:
        return polys[0]

    starts = np.array([p[0] for p in polys])
    ends = np.array([p[-1] for p in polys])

    used = [False] * n_ent
    order = [0]
    flipped = [False] * n_ent
    used[0] = True
    cur_end = ends[0]

    for step in range(1, n_ent):
        best_d, best_j, best_flip = np.inf, -1, False
        for j in range(n_ent):
            if used[j]:
                continue
            ds = np.linalg.norm(starts[j] - cur_end)
            de = np.linalg.norm(ends[j] - cur_end)
            if ds < best_d:
                best_d, best_j, best_flip = ds, j, False
            if de < best_d:
                best_d, best_j, best_flip = de, j, True

        if best_j < 0:
            import warnings
            warnings.warn(f"Could not chain all entities. Using {step} of {n_ent}.")
            break

        order.append(best_j)
        used[best_j] = True
        flipped[best_j] = best_flip
        cur_end = starts[best_j] if best_flip else ends[best_j]

        if verbose and best_d > 1e-4:
            print(f"  Warning: gap of {best_d:.6f} between entity {order[-2]} and {best_j}")

    chained = None
    for idx in order:
        pts = polys[idx]
        if flipped[idx]:
            pts = pts[::-1]
        if chained is None:
            chained = pts
        else:
            chained = np.vstack([chained, pts[1:]])

    if verbose:
        chain_str = ' '.join(f"{idx}{'(flip)' if flipped[idx] else ''}" for idx in order)
        print(f"  Chain order: {chain_str}")

    return chained


# ═══════════════════════════════════════════════════════════════════════
#  Plotting (optional — uses matplotlib)
# ═══════════════════════════════════════════════════════════════════════

def _plot_import_result(raw, nodes, arc_len, num_el, t_start, t_end, title):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)
    arrow_len = arc_len * 0.08

    ax1.plot(raw[:, 0], raw[:, 1], 'k.-', markersize=3)
    ax1.plot(raw[0, 0], raw[0, 1], 'go', markersize=8)
    ax1.plot(raw[-1, 0], raw[-1, 1], 'rs', markersize=8)
    ax1.set_aspect('equal'); ax1.grid(True)
    ax1.set_title(f'Raw Points ({raw.shape[0]})')

    ax2.plot(nodes[:, 0], nodes[:, 1], 'b.-', markersize=2)
    ax2.plot(nodes[0, 0], nodes[0, 1], 'go', markersize=8)
    ax2.plot(nodes[-1, 0], nodes[-1, 1], 'rs', markersize=8)
    ax2.quiver(nodes[0, 0], nodes[0, 1], t_start[0]*arrow_len, t_start[1]*arrow_len,
               color='g', scale=1, scale_units='xy')
    ax2.quiver(nodes[-1, 0], nodes[-1, 1], t_end[0]*arrow_len, t_end[1]*arrow_len,
               color='r', scale=1, scale_units='xy')
    ax2.set_aspect('equal'); ax2.grid(True)
    ax2.set_title(f'Resampled ({num_el} elements)')
    plt.tight_layout()
    plt.show()


def _plot_igs_import_result(polys, scale, chained, nodes, arc_len,
                            num_el, t_start, t_end):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('IGES Import')
    arrow_len = arc_len * 0.08

    colors = cm.hsv(np.linspace(0, 1, len(polys)))
    for i, pts in enumerate(polys):
        pts_s = pts * scale
        ax1.plot(pts_s[:, 0], pts_s[:, 1], '-', linewidth=2, color=colors[i])
    ax1.plot(chained[0, 0], chained[0, 1], 'go', markersize=8)
    ax1.plot(chained[-1, 0], chained[-1, 1], 'rs', markersize=8)
    ax1.set_aspect('equal'); ax1.grid(True)
    ax1.set_title(f'Raw Entities ({len(polys)})')

    ax2.plot(nodes[:, 0], nodes[:, 1], 'b.-', markersize=2)
    ax2.plot(nodes[0, 0], nodes[0, 1], 'go', markersize=8)
    ax2.plot(nodes[-1, 0], nodes[-1, 1], 'rs', markersize=8)
    ax2.quiver(nodes[0, 0], nodes[0, 1], t_start[0]*arrow_len, t_start[1]*arrow_len,
               color='g', scale=1, scale_units='xy')
    ax2.quiver(nodes[-1, 0], nodes[-1, 1], t_end[0]*arrow_len, t_end[1]*arrow_len,
               color='r', scale=1, scale_units='xy')
    ax2.set_aspect('equal'); ax2.grid(True)
    ax2.set_title(f'Resampled ({num_el} elements)')
    plt.tight_layout()
    plt.show()