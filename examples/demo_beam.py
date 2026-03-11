"""
demo_beam_optimization.py — Beam thickness optimisation demo.

Port of demo_beam_optimization.m from ShepherdLab LW FEA-OPT.
"""

import numpy as np
import os
import time

from feaopt import (StructureGeometry, FEAAnalysis, LoadCase,
                    optimize_thickness, export_profile,
                    import_csv_curve, import_igs_curve)

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not found. Plotting disabled.")

if HAS_MPL:
    from feaopt import StructurePlotter


def main():
    t_start = time.time()

    # ==================== PARAMETERS ====================

    # Material (Ti-6Al-4V example)
    E = 69e9
    ultimate = 900e6
    rho = 4820

    # Cross-section
    beam_width = 0.008
    initial_thickness = 0.003

    # Loading
    Fx, Fy = 0, -333

    # Mesh
    num_elements = 250

    # ==================== IMPORT GEOMETRY ====================

    filename = 's_test_2.igs'
    import_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    ext = os.path.splitext(import_file)[1].lower()
    if ext == '.csv':
        result = import_csv_curve(import_file, num_elements=num_elements,
                                  scale=0.001, flip=False, plot=False, verbose=True)
    elif ext in ('.igs', '.iges'):
        result = import_igs_curve(import_file, num_elements=num_elements,
                                  scale=0.001, flip=False, plot=False, verbose=True)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    all_nodes = result['nodes']
    connectivity = result['connectivity']
    num_nodes = result['num_nodes']
    num_elements = result['num_elements']

    print(f"\n=== Imported Geometry ===")
    print(f"  Arc length:  {result['arc_length']*1e3:.4f} mm")
    print(f"  Start:       [{all_nodes[0,0]*1e3:.4f}, {all_nodes[0,1]*1e3:.4f}] mm")
    print(f"  End:         [{all_nodes[-1,0]*1e3:.4f}, {all_nodes[-1,1]*1e3:.4f}] mm")

    # ==================== FEA SETUP ====================
    # NOTE: 0-based indexing in Python
    fixed_node = 0
    free_node = num_nodes - 1

    props = {
        'E': E * np.ones(num_elements),
        'width': beam_width * np.ones(num_elements),
        'thickness': initial_thickness * np.ones(num_elements),
        'ultimate': ultimate * np.ones(num_elements),
        'is_rigid': np.zeros(num_elements, dtype=bool),
        'rho': rho,
    }

    # Constraints: [node_id (0-based), dof (0-2), value]
    constraints = np.array([
        [fixed_node, 0, 0],   # ux = 0
        [fixed_node, 1, 0],   # uy = 0
        [fixed_node, 2, 0],   # theta = 0
        [free_node,  0, 0],   # ux = 0 (guided)
        [free_node,  2, 0],   # theta = 0
    ])

    geometry = StructureGeometry(all_nodes, connectivity, props, constraints)
    analysis = FEAAnalysis(geometry, verbose=False)

    print(f"\n=== Loading ===")
    print(f"  Force: [{Fx:.1f}, {Fy:.1f}] N at node {free_node}")

    load_case = LoadCase(geometry)
    load_case.add_nodal_load(free_node, Fx, Fy, 0)
    load_case.set_load_factor(1.0)

    # ==================== INITIAL ANALYSIS ====================
    print(f"\n=== Initial Analysis ===")

    print("  Linear...")
    lin_results = analysis.run_linear_analysis(load_case)
    lin_stress = analysis.perform_stress_analysis()
    print(f"    Max disp:  {lin_results['max_displacement']*1e3:.4f} mm  |  "
          f"FOS: {lin_stress['FOS']['min_FOS']:.2f}  |  "
          f"Max stress: {lin_stress['stresses']['max_von_mises']/1e6:.1f} MPa")

    print("  Nonlinear...")
    nl_results = analysis.solve_large_deflection(load_case, 2)
    nl_stress = analysis.perform_stress_analysis()
    nl_deformed = analysis.get_deformed_coordinates(1)
    print(f"    Max disp:  {nl_results['max_displacement']*1e3:.4f} mm  |  "
          f"FOS: {nl_stress['FOS']['min_FOS']:.2f}  |  "
          f"Max stress: {nl_stress['stresses']['max_von_mises']/1e6:.1f} MPa")
    print(f"    Mass:      {geometry.get_total_mass()*1e3:.4f} g")

    # Save initial state
    initial_nodes = geometry.nodes.copy()
    initial_thickness = geometry.thickness.copy()
    initial_conn = geometry.connectivity.copy()
    initial_mass = geometry.get_total_mass()

    # ==================== THICKNESS OPTIMISATION ====================
    print(f"\n=== Thickness Optimisation ===")

    opt_options = {
        'alpha': 0.4,
        'safety_factor': 2,
        'max_iterations': 200,
        't_min': 0.0005,
        't_max': 0.020,
        'convergence_tol': 0.0005,
        'move_limit': 0.2,
        'use_nonlinear': True,
        'verbose': True,
    }

    optimized_geom, opt_results = optimize_thickness(geometry, analysis, load_case, opt_options)

    # ==================== POST-OPTIMISATION ====================
    analysis_opt = FEAAnalysis(optimized_geom, verbose=False)
    opt_final = analysis_opt.solve_large_deflection(load_case, 2)
    opt_stress = analysis_opt.perform_stress_analysis()
    opt_deformed = analysis_opt.get_deformed_coordinates(1)

    print(f"\n  Post-optimisation:")
    print(f"    Max disp:  {opt_final['max_displacement']*1e3:.4f} mm  |  "
          f"FOS: {opt_stress['FOS']['min_FOS']:.2f}  |  "
          f"Max stress: {opt_stress['stresses']['max_von_mises']/1e6:.1f} MPa")

    # ==================== EXPORT ====================
    print(f"\n=== Exporting Profile ===")

    profile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'profiles')

    export_opts = {
        'num_spline_points': 1000,
        'scale_to_mm': 1000,
        'export_filename': os.path.join(profile_dir, 'optimised_beam'),
        'plot': False,
        'verbose': True,
    }

    profile = export_profile(optimized_geom, export_opts)

    # ==================== SUMMARY ====================
    opt_t = optimized_geom.thickness[:num_elements]
    print(f"\n============================")
    print(f"  OPTIMISATION SUMMARY")
    print(f"============================\n")
    print(f"Mass")
    print(f"  Initial:       {initial_mass*1e3:.4f} g")
    print(f"  Optimised:     {opt_results['mass_final']*1e3:.4f} g")
    print(f"  Reduction:     {opt_results['mass_reduction']:.1f}%")
    print(f"\nStress & Safety")
    print(f"  Optimised FOS: {opt_stress['FOS']['min_FOS']:.2f}")
    print(f"  Max stress:    {opt_stress['stresses']['max_von_mises']/1e6:.1f} MPa")
    print(f"\nThickness")
    print(f"  Min:           {np.min(opt_t)*1e3:.2f} mm")
    print(f"  Max:           {np.max(opt_t)*1e3:.2f} mm")
    print(f"  Ratio:         {np.max(opt_t)/np.min(opt_t):.2f} : 1")

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
    fig.suptitle('Beam Thickness Optimisation — Before & After',
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
    ax.set_title(f'Initial Deflection ({nl_results["max_displacement"]*1e3:.3f} mm)')

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