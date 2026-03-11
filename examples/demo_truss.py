"""
demo_truss_optimization.py — Simple overhanging truss with end moment.

Port of demo_truss_optimization.m — with full MATLAB-style plotting.
"""

import numpy as np
import os
import time

from feaopt import (StructureGeometry, FEAAnalysis, LoadCase,
                    optimize_thickness, export_profile,
                    subdivide_truss)

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

if HAS_MPL:
    from feaopt import StructurePlotter


def main():
    t_start = time.time()

    # ==================== PARAMETERS ====================
    bay_width = 0.060
    height = 0.050
    n_bays = 4
    support_bay = 2

    E = 200e9
    ultimate = 400e6
    rho = 7850

    beam_width = 0.008
    initial_thickness = 0.003

    M_tip = 10
    Fy_tip = -3000

    num_sub = 50

    # ==================== BUILD TRUSS ====================
    n_bot = n_bays + 1
    bot_x = np.arange(n_bays + 1) * bay_width
    bot_y = np.zeros(n_bot)

    n_top_warren = n_bays
    warren_x = (np.arange(n_top_warren) + 0.5) * bay_width
    warren_y = height * np.ones(n_top_warren)

    top_x = np.concatenate([[0], warren_x])
    top_y = np.concatenate([[height], warren_y])
    n_top = len(top_x)

    truss_nodes = np.vstack([
        np.column_stack([bot_x, bot_y]),
        np.column_stack([top_x, top_y]),
    ])

    top_offset = n_bot

    conn = []
    for i in range(n_bot - 1):
        conn.append([i, i + 1])
    conn.append([0, top_offset])
    for i in range(n_top - 1):
        conn.append([top_offset + i, top_offset + i + 1])
    for i in range(n_top_warren):
        top_idx = top_offset + 1 + i
        conn.append([i, top_idx])
        conn.append([top_idx, i + 1])
    conn = np.array(conn, dtype=int)

    pin_node = 0
    roller_node = top_offset
    tip_node = n_bot - 1

    print(f"=== Simple Overhanging Truss ===")
    print(f"  {n_bays} bays ({bay_width*1e3:.0f} mm each),  height: {height*1e3:.0f} mm")
    print(f"  Nodes: {truss_nodes.shape[0]},  Members: {conn.shape[0]}")
    print(f"  Pin: node {pin_node}  |  Fixed: node {roller_node}")
    print(f"  Tip: node {tip_node}")

    # ==================== SUBDIVIDE ====================
    print(f"\n=== Subdividing ({num_sub} sub-elements per member) ===")
    sub_nodes, sub_conn, member_map = subdivide_truss(truss_nodes, conn, num_sub)
    num_nodes = sub_nodes.shape[0]
    num_elements = sub_conn.shape[0]
    print(f"  Nodes: {num_nodes},  Elements: {num_elements}")

    # ==================== FEA SETUP ====================
    props = {
        'E': E * np.ones(num_elements),
        'width': beam_width * np.ones(num_elements),
        'thickness': initial_thickness * np.ones(num_elements),
        'ultimate': ultimate * np.ones(num_elements),
        'is_rigid': np.zeros(num_elements, dtype=bool),
        'rho': rho,
    }

    constraints = np.array([
        [pin_node,    0, 0], [pin_node,    1, 0], [pin_node,    2, 0],
        [roller_node, 0, 0], [roller_node, 1, 0], [roller_node, 2, 0],
    ])

    geometry = StructureGeometry(sub_nodes, sub_conn, props, constraints)
    analysis = FEAAnalysis(geometry, verbose=False)

    load_case = LoadCase(geometry)
    load_case.add_nodal_load(tip_node, 0, Fy_tip, M_tip, 'Tip force + moment')
    load_case.set_load_factor(1.0)

    print(f"\n=== Loading ===")
    print(f"  Fy = {Fy_tip:.0f} N,  Mz = {M_tip:.0f} Nm  at node {tip_node}")

    # ==================== INITIAL ANALYSIS ====================
    print(f"\n=== Initial Analysis ===")
    lin_results = analysis.run_linear_analysis(load_case)
    lin_stress = analysis.perform_stress_analysis()
    print(f"  Max disp: {lin_results['max_displacement']*1e3:.4f} mm  |  "
          f"FOS: {lin_stress['FOS']['min_FOS']:.2f}  |  "
          f"Max stress: {lin_stress['stresses']['max_von_mises']/1e6:.1f} MPa")
    print(f"  Mass:     {geometry.get_total_mass()*1e3:.4f} g")

    # Save initial state
    initial_nodes = geometry.nodes.copy()
    initial_thickness = geometry.thickness.copy()
    initial_conn = geometry.connectivity.copy()
    initial_mass = geometry.get_total_mass()
    nl_deformed = analysis.get_deformed_coordinates(1)
    nl_stress = lin_stress

    # ==================== OPTIMISE ====================
    print(f"\n=== Thickness Optimisation ===")

    opt_options = {
        'alpha': 0.35, 'safety_factor': 2, 'max_iterations': 300,
        't_min': 0.002, 't_max': 0.02, 'convergence_tol': 0.001,
        'move_limit': 0.15, 'use_nonlinear': False, 'verbose': True,
    }

    optimized_geom, opt_results = optimize_thickness(
        geometry, analysis, load_case, opt_options)

    # ==================== POST-OPTIMISATION ====================
    analysis_opt = FEAAnalysis(optimized_geom, verbose=False)
    opt_final = analysis_opt.run_linear_analysis(load_case)
    opt_stress = analysis_opt.perform_stress_analysis()
    opt_deformed = analysis_opt.get_deformed_coordinates(1)

    print(f"\n  Post-optimisation:")
    print(f"    Max disp: {opt_final['max_displacement']*1e3:.4f} mm  |  "
          f"FOS: {opt_stress['FOS']['min_FOS']:.2f}  |  "
          f"Max stress: {opt_stress['stresses']['max_von_mises']/1e6:.1f} MPa")

    # ==================== EXPORT ====================
    print(f"\n=== Exporting Profile ===")
    profile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'profiles')

    export_opts = {
        'num_spline_points': 1000, 'smoothing': 0.999999999,
        'scale_to_mm': 1000,
        'export_filename': os.path.join(profile_dir, 'simple_truss'),
        'plot': False, 'verbose': True,
    }

    profile = export_profile(optimized_geom, export_opts)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.2f} s")

    # ==================== PLOTTING (MATLAB-style) ====================
    if not HAS_MPL:
        return

    cmap = cm.jet

    plotter_init = StructurePlotter(geometry, analysis)
    plotter_opt = StructurePlotter(optimized_geom, analysis_opt)

    from feaopt.structure_plotter import draw_beams_batch, draw_beams_stress

    # ===== FIGURE: Initial + Optimised side by side =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Simple Overhanging Truss — Before & After',
                 fontsize=13, fontweight='bold')

    # -- Top-left: Initial deflection --
    ax = axes[0, 0]
    draw_beams_batch(ax, initial_nodes, initial_conn, initial_thickness,
                     (0.85, 0.85, 0.85))
    draw_beams_batch(ax, nl_deformed, initial_conn, initial_thickness,
                     (0.2, 0.4, 0.8))
    plotter_init.plot_supports(ax)
    ax.set_aspect('equal'); ax.grid(True)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title(f'Initial Deflection ({lin_results["max_displacement"]*1e3:.3f} mm)')

    # -- Top-right: Initial stress --
    ax = axes[0, 1]
    init_vm = nl_stress['stresses']['combined']
    init_stress_mpa = np.nanmax(np.abs(init_vm), axis=1) / 1e6
    sm = draw_beams_stress(ax, initial_nodes, initial_conn, initial_thickness,
                           init_stress_mpa)
    plotter_init.plot_supports(ax)
    cb = plt.colorbar(sm, ax=ax); cb.set_label('Stress [MPa]')
    ax.set_aspect('equal'); ax.grid(True)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    max_init = np.nanmax(init_stress_mpa)
    ax.set_title(f'Initial Stress ({max_init:.1f} MPa, FOS: {nl_stress["FOS"]["min_FOS"]:.2f})')

    # -- Bottom-left: Optimised deflection --
    ax = axes[1, 0]
    draw_beams_batch(ax, optimized_geom.nodes, optimized_geom.connectivity,
                     optimized_geom.thickness, (0.85, 0.85, 0.85))
    draw_beams_batch(ax, opt_deformed, optimized_geom.connectivity,
                     optimized_geom.thickness, (0.8, 0.2, 0.2))
    plotter_opt.plot_supports(ax)
    ax.set_aspect('equal'); ax.grid(True)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title(f'Optimised Deflection ({opt_final["max_displacement"]*1e3:.3f} mm)')

    # -- Bottom-right: Optimised stress --
    ax = axes[1, 1]
    opt_vm = opt_stress['stresses']['combined']
    opt_stress_mpa = np.nanmax(np.abs(opt_vm), axis=1) / 1e6
    sm = draw_beams_stress(ax, optimized_geom.nodes, optimized_geom.connectivity,
                           optimized_geom.thickness, opt_stress_mpa)
    plotter_opt.plot_supports(ax)
    cb = plt.colorbar(sm, ax=ax); cb.set_label('Stress [MPa]')
    ax.set_aspect('equal'); ax.grid(True)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    max_opt = np.nanmax(opt_stress_mpa)
    ax.set_title(f'Optimised Stress ({max_opt:.1f} MPa, FOS: {opt_stress["FOS"]["min_FOS"]:.2f})')

    plt.tight_layout()

    # ===== FIGURE: Exported profile =====
    if profile['num_loops'] > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        blue = (0.3, 0.55, 0.85)
        for loop in profile['loops']:
            if loop['is_hole']:
                ax2.fill(loop['xs'], loop['ys'], facecolor='white',
                         edgecolor=blue, linewidth=1.5, linestyle='--')
            else:
                ax2.fill(loop['xs'], loop['ys'], facecolor=(0.75, 0.85, 0.95),
                         edgecolor=blue, linewidth=2, alpha=0.5)
        ax2.set_aspect('equal'); ax2.grid(True)
        ax2.set_xlabel('X [mm]'); ax2.set_ylabel('Y [mm]')
        ax2.set_title('Exported Profile — Smoothed Spline Loops')
        plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()