"""
optimizeThickness — Stress-ratio thickness optimization for 2-D frames.

Port of optimizeThickness.m from ShepherdLab LW FEA-OPT.
"""

import numpy as np


DEFAULT_OPTIONS = {
    'alpha': 0.4,
    'max_iterations': 100,
    'convergence_tol': 0.01,
    'move_limit': 0.2,
    't_min': 0.001,
    't_max': 0.020,
    'safety_factor': 1.5,
    'use_nonlinear': False,
    'num_load_steps': 2,
    'verbose': True,
}


def optimize_thickness(geometry, analysis, load_case, options=None):
    """
    Iteratively adjust element thicknesses for uniform stress.

    Parameters
    ----------
    geometry : StructureGeometry (modified in-place)
    analysis : FEAAnalysis
    load_case : LoadCase
    options : dict, optional

    Returns
    -------
    geometry : same (mutated) object
    results : dict
    """
    opt = {**DEFAULT_OPTIONS, **(options or {})}

    M = geometry.num_members
    optimizable = ~geometry.is_rigid
    num_opt = int(np.sum(optimizable))

    if num_opt == 0:
        raise ValueError("No non-rigid elements to optimise.")

    sigma_allow = geometry.ultimate / opt['safety_factor']
    t_current = geometry.thickness.copy()

    t_history = np.zeros((M, opt['max_iterations']))
    mass_hist = np.zeros(opt['max_iterations'])
    stress_hist = np.zeros(opt['max_iterations'])
    conv_hist = np.zeros(opt['max_iterations'])

    saved_verbose = analysis.verbose
    analysis.verbose = False

    if opt['verbose']:
        print(f"\n=== Thickness Optimisation ===")
        print(f"  Optimisable elements: {num_opt} / {M}")
        print(f"  alpha = {opt['alpha']:.2f},  FOS target = {opt['safety_factor']:.2f},  "
              f"move limit = {opt['move_limit']*100:.0f}%")
        print(f"  Thickness bounds: [{opt['t_min']*1e3:.2f}, {opt['t_max']*1e3:.2f}] mm")
        print(f"\n  {'Iter':<5}  {'Mass [kg]':<11}  {'Max Stress':<12}  {'Max dT [%]':<10}  Status")
        print(f"  {'-'*58}")

    converged = False

    for it in range(opt['max_iterations']):
        t_history[:, it] = t_current

        geometry.update_thickness(t_current)
        analysis.element_matrices.reset_to_initial_configuration()

        if opt['use_nonlinear'] and it > 0:
            analysis.solve_large_deflection(load_case, opt['num_load_steps'])
        else:
            analysis.run_linear_analysis(load_case)

        stress_res = analysis.perform_stress_analysis()
        vm = stress_res['stresses']['von_mises']
        max_vm = np.max(np.abs(vm), axis=1)

        stress_ratio = np.zeros(M)
        for i in range(M):
            if optimizable[i]:
                if max_vm[i] > np.finfo(float).eps:
                    stress_ratio[i] = max_vm[i] / sigma_allow[i]
                else:
                    stress_ratio[i] = 0.1

        t_new = t_current.copy()
        for i in range(M):
            if not optimizable[i]:
                continue
            t_proposed = t_current[i] * stress_ratio[i]**opt['alpha']
            dt_max = t_current[i] * opt['move_limit']
            t_proposed = max(t_proposed, t_current[i] - dt_max)
            t_proposed = min(t_proposed, t_current[i] + dt_max)
            t_new[i] = max(opt['t_min'], min(opt['t_max'], t_proposed))

        rel_change = np.abs(t_new - t_current) / t_current
        max_change = float(np.max(rel_change[optimizable]))

        current_mass = float(np.sum(geometry.rho * geometry.width * t_current * geometry.L))
        mass_hist[it] = current_mass
        stress_hist[it] = float(np.max(max_vm[optimizable]))
        conv_hist[it] = max_change

        if opt['verbose']:
            if max_change < opt['convergence_tol']:
                tag = 'CONVERGED'
            elif np.any(stress_ratio[optimizable] > 1):
                tag = f"{int(np.sum(stress_ratio[optimizable] > 1))} overstressed"
            else:
                tag = 'Optimising'
            print(f"  {it+1:<5}  {current_mass:<11.4f}  {stress_hist[it]:<12.2e}  "
                  f"{max_change*100:<10.2f}  {tag}")

        t_current = t_new

        if max_change < opt['convergence_tol']:
            converged = True
            if opt['verbose']:
                print(f"\n  Converged after {it+1} iterations.")
            break

    num_iters = it + 1

    # Final state
    geometry.update_thickness(t_current)
    analysis.element_matrices.reset_to_initial_configuration()

    if opt['use_nonlinear']:
        analysis.solve_large_deflection(load_case, opt['num_load_steps'])
    else:
        analysis.run_linear_analysis(load_case)

    final_stress = analysis.perform_stress_analysis()
    final_mass = float(np.sum(geometry.rho * geometry.width * t_current * geometry.L))

    analysis.verbose = saved_verbose

    # Stress uniformity
    opt_stresses = max_vm[optimizable]
    opt_stresses = opt_stresses[np.isfinite(opt_stresses)]
    if len(opt_stresses) > 0 and np.mean(opt_stresses) > 0:
        stress_uniformity = float(np.std(opt_stresses) / np.mean(opt_stresses))
    else:
        stress_uniformity = np.inf

    results = {
        'converged': converged,
        'iterations': num_iters,
        'thickness_initial': t_history[:, 0],
        'thickness_final': t_current,
        'thickness_history': t_history[:, :num_iters],
        'mass_initial': mass_hist[0],
        'mass_final': final_mass,
        'mass_reduction': (mass_hist[0] - final_mass) / mass_hist[0] * 100,
        'mass_history': mass_hist[:num_iters],
        'max_stress_history': stress_hist[:num_iters],
        'convergence_history': conv_hist[:num_iters],
        'final_stress_results': final_stress,
        'stress_uniformity': stress_uniformity,
    }

    if opt['verbose']:
        print(f"\n=== Optimisation Summary ===")
        print(f"  Initial mass:   {results['mass_initial']:.4f} kg")
        print(f"  Final mass:     {results['mass_final']:.4f} kg")
        print(f"  Mass reduction: {results['mass_reduction']:.1f}%")
        print(f"  Min FOS:        {final_stress['FOS']['min_FOS']:.2f}")
        print(f"  Max stress:     {final_stress['stresses']['max_von_mises']/1e6:.1f} MPa")
        print(f"  Stress CoV:     {stress_uniformity*100:.1f}%")
        if not converged:
            print(f"\n  WARNING: did not converge within {opt['max_iterations']} iterations.")

    return geometry, results
