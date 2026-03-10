"""
FEAAnalysis — Main analysis controller for linear and nonlinear FEA.

Port of FEAAnalysis.m — vectorized for performance.
"""

import numpy as np
import time
import warnings
from scipy.sparse.linalg import spsolve

from feaopt.structure_geometry import StructureGeometry
from feaopt.element_matrices import ElementMatrices
from feaopt.analysis_state import AnalysisState


class FEAAnalysis:
    """Linear and Newton-Raphson nonlinear FEA for 2-D frames."""

    def __init__(self, geometry, **kwargs):
        self.geometry = geometry
        self.element_matrices = ElementMatrices(geometry)
        self.current_state = AnalysisState(geometry)

        self.cumulative_displacements = np.zeros(geometry.num_dof_total)
        self.deformed_nodes = geometry.nodes.copy()

        self.verbose = kwargs.get('verbose', True)
        self.max_iterations = kwargs.get('max_iterations', 1000)
        self.force_tolerance = kwargs.get('force_tolerance', 1e-2)
        self.energy_tolerance = kwargs.get('energy_tolerance', 1e-6)
        self.displacement_tolerance = kwargs.get('displacement_tolerance', 1e-6)

        self.converged_states = []
        self.load_factors = []
        self.iteration_history = []
        self.analysis_time = 0.0

        self.member_stresses = None
        self.stress_results = None

    # ── Linear analysis ───────────────────────────────────────────────

    def run_linear_analysis(self, load_case):
        t0 = time.time()
        if self.verbose:
            print("\n=== Linear Analysis ===")

        self.reset_analysis()

        K = self.element_matrices.assemble_global_stiffness_matrix(self.geometry)
        P = load_case.get_scaled_load()
        free = self.geometry.dof_map.free_dof

        K_ff = K[np.ix_(free, free)]
        P_f = P[free]

        u_f = spsolve(K_ff, P_f)

        self.current_state.update_displacements(u_f)
        self.current_state.f_external = P.copy()

        self.current_state.extract_member_displacements(self.element_matrices)
        self.current_state.compute_member_forces(self.element_matrices)

        results = self._package_linear_results()

        elapsed = time.time() - t0
        self.analysis_time += elapsed
        if self.verbose:
            print(f"  Solved in {elapsed:.3f} s  |  Max disp: {results['max_displacement']:.3e} m")

        return results

    # ── Large-deflection (Newton-Raphson) ─────────────────────────────

    def solve_large_deflection(self, load_case, num_steps):
        t0 = time.time()

        num_rigid = int(np.sum(self.geometry.is_rigid))
        if self.verbose:
            print(f"\n=== Large Deflection Analysis (Newton-Raphson) ===")
            print(f"  Load steps: {num_steps}  |  Max iter: {self.max_iterations}  |  Tol: {self.force_tolerance:.1e}")
            if num_rigid > 0:
                print(f"  Rigid elements: {num_rigid} (penalty method)")
            print(f"  {'-'*50}")

        self.reset_analysis()

        free = self.geometry.dof_map.free_dof
        P_ref = load_case.P_ref
        disp_hist = np.zeros(num_steps)

        for step in range(num_steps):
            lf = (step + 1) / num_steps
            self.load_factors.append(lf)

            if self.verbose:
                print(f"\n  Step {step+1}/{num_steps}  (lambda = {lf:.3f})")

            self.current_state.f_external = P_ref * lf
            self.current_state.f_free_ext = self.current_state.f_external[free].copy()

            converged = self._newton_raphson_step(step)

            if not converged:
                warnings.warn(
                    f"Step {step+1} did not converge after {self.max_iterations} iterations "
                    f"(|R| = {self.current_state.residual_norm:.2e})."
                )

            self.cumulative_displacements = self.current_state.u.copy()

            self.current_state.extract_member_displacements(self.element_matrices)
            self.current_state.compute_member_forces(self.element_matrices)

            disp_hist[step] = self.current_state.get_max_displacement()

            snap = AnalysisState(self.geometry)
            snap.copy_from(self.current_state)
            self.converged_states.append(snap)

            if self.verbose:
                print(f"    -> Max disp = {disp_hist[step]:.4e} m")

        self._update_deformed_nodes()

        results = self._package_nonlinear_results(disp_hist)

        elapsed = time.time() - t0
        self.analysis_time += elapsed
        if self.verbose:
            print(f"\n  Analysis complete in {elapsed:.3f} s")
            print(f"  Final max disp: {results['max_displacement']:.4e} m")

        return results

    # ── Stress post-processing ────────────────────────────────────────

    def perform_stress_analysis(self):
        stresses = self._calculate_member_stresses()
        fos = self._calculate_factor_of_safety(stresses)
        results = {'stresses': stresses, 'FOS': fos}
        self.stress_results = results
        return results

    # ── Utility ───────────────────────────────────────────────────────

    def reset_analysis(self):
        self.current_state.reset()
        self.element_matrices.reset_to_initial_configuration()
        self.cumulative_displacements = np.zeros(self.geometry.num_dof_total)
        self.deformed_nodes = self.geometry.nodes.copy()
        self.converged_states = []
        self.load_factors = []
        self.iteration_history = []
        self.member_stresses = None
        self.stress_results = None

    def get_deformed_coordinates(self, scale=1.0):
        """Vectorized deformed coordinate computation."""
        coords = self.geometry.nodes.copy()
        u = self.current_state.u
        coords[:, 0] += scale * u[0::3][:self.geometry.num_nodes]
        coords[:, 1] += scale * u[1::3][:self.geometry.num_nodes]
        return coords

    # ── Newton-Raphson internals ──────────────────────────────────────

    def _newton_raphson_step(self, step):
        free = self.geometry.dof_map.free_dof
        converged = False

        for it in range(self.max_iterations):
            self._update_geometry_from_displacements()

            self.current_state.compute_internal_forces(self.element_matrices)
            self.current_state.compute_residual()

            K_t = self.element_matrices.assemble_global_stiffness_matrix(self.geometry)
            K_ff = K_t[np.ix_(free, free)]

            du = spsolve(K_ff, self.current_state.residual)

            if not np.all(np.isfinite(du)):
                warnings.warn(f"NaN/Inf in du at step {step+1}, iteration {it+1}.")
                break

            converged, _ = self._check_convergence(it, du)

            energy = abs(float(self.current_state.residual @ du))
            self.iteration_history.append(
                [step+1, it+1, self.current_state.residual_norm, float(np.linalg.norm(du)), energy]
            )

            if self.verbose and (it == 0 or (it+1) % 5 == 0 or converged):
                tag = "  [CONVERGED]" if converged else ""
                print(f"    Iter {it+1:3d}:  |R|={self.current_state.residual_norm:.2e}  "
                      f"|du|={np.linalg.norm(du):.2e}  E={energy:.2e}{tag}")

            if converged:
                break

            self.current_state.apply_increment(du)

        return converged

    def _update_geometry_from_displacements(self):
        """Vectorized geometry update from current displacements."""
        geom = self.geometry
        u = self.current_state.u

        # Vectorized deformed node positions
        self.deformed_nodes[:, 0] = geom.nodes[:, 0] + u[0::3][:geom.num_nodes]
        self.deformed_nodes[:, 1] = geom.nodes[:, 1] + u[1::3][:geom.num_nodes]

        # Vectorized element geometry
        n1 = geom.connectivity[:, 0]
        n2 = geom.connectivity[:, 1]
        dx = self.deformed_nodes[n2, 0] - self.deformed_nodes[n1, 0]
        dy = self.deformed_nodes[n2, 1] - self.deformed_nodes[n1, 1]
        new_theta = np.arctan2(dy, dx)
        new_L = np.sqrt(dx**2 + dy**2)

        self.element_matrices.update_all_member_geometry(new_L, new_theta)

    def _check_convergence(self, it, du):
        R_norm = self.current_state.residual_norm
        f_ref = max(np.linalg.norm(self.current_state.f_free_ext), 1.0)
        u_ref = max(self.current_state.get_max_displacement(), 1e-6)

        force_err = R_norm / f_ref
        disp_err = np.linalg.norm(du) / u_ref
        energy = abs(float(self.current_state.residual @ du))

        force_ok = force_err < self.force_tolerance
        disp_ok = disp_err < self.displacement_tolerance
        energy_ok = energy < self.energy_tolerance

        if it == 0:
            converged = force_ok
        else:
            converged = force_ok and (disp_ok or energy_ok)

        info = {'force_error': force_err, 'disp_error': disp_err, 'energy': energy}
        return converged, info

    def _update_deformed_nodes(self):
        u = self.current_state.u
        self.deformed_nodes[:, 0] = self.geometry.nodes[:, 0] + u[0::3][:self.geometry.num_nodes]
        self.deformed_nodes[:, 1] = self.geometry.nodes[:, 1] + u[1::3][:self.geometry.num_nodes]

    # ── Stress internals (vectorized) ─────────────────────────────────

    def _calculate_member_stresses(self):
        if np.all(self.current_state.member_forces_local == 0):
            self.current_state.extract_member_displacements(self.element_matrices)
            self.current_state.compute_member_forces(self.element_matrices)

        M = self.geometry.num_members
        g = self.geometry
        f = self.current_state.member_forces_local  # (M, 6)

        A = g.A
        I = g.I
        c = g.thickness / 2.0

        # Vectorized stress computation
        axial = np.column_stack([f[:, 0] / A, f[:, 3] / A])
        bending = np.column_stack([f[:, 2] * c / I, f[:, 5] * c / I])
        shear = np.column_stack([1.5 * f[:, 1] / A, 1.5 * f[:, 4] / A])

        sig = axial + bending
        von_mises = np.sqrt(sig**2 + 3 * shear**2)
        max_stress = np.max(von_mises, axis=1)

        # Mask rigid elements
        rigid = g.is_rigid
        for arr in (axial, bending, shear, von_mises):
            arr[rigid] = np.nan
        max_stress[rigid] = np.nan

        s = {
            'axial': axial,
            'bending': bending,
            'shear': shear,
            'von_mises': von_mises,
            'max_stress': max_stress,
            'is_rigid': rigid.copy(),
        }

        nr = ~rigid
        self._aggregate_stress_maxima(s, nr)
        s['combined'] = s['von_mises']
        s['max_combined'] = s.get('max_von_mises', 0)

        self.member_stresses = s
        return s

    def _calculate_factor_of_safety(self, stresses):
        M = self.geometry.num_members
        g = self.geometry

        s_ult = g.ultimate
        vm = np.abs(stresses['combined'])  # (M, 2)

        # Vectorized FOS
        stress_based = np.full((M, 2), np.nan)
        nr = ~g.is_rigid

        with np.errstate(divide='ignore', invalid='ignore'):
            stress_based[nr] = s_ult[nr, None] / np.maximum(vm[nr], np.finfo(float).eps)

        # Inf where stress is essentially zero
        tiny = vm < np.finfo(float).eps  # (M, 2)
        nr2 = nr[:, None] & tiny         # broadcast to (M, 2)
        stress_based[nr2] = np.inf

        member_FOS = np.full(M, np.nan)
        member_FOS[nr] = np.min(stress_based[nr], axis=1)

        vals = member_FOS[nr]
        vals = vals[np.isfinite(vals)]

        fos = {
            'stress_based': stress_based,
            'member_FOS': member_FOS,
            'is_rigid': g.is_rigid.copy(),
        }

        if len(vals) > 0:
            fos['min_FOS'] = float(np.min(vals))
            idx = int(np.argmin(vals))
            nr_ids = np.where(nr)[0]
            fos['critical_member'] = int(nr_ids[idx])
        else:
            fos['min_FOS'] = np.inf
            fos['critical_member'] = -1

        return fos

    @staticmethod
    def _aggregate_stress_maxima(s, nr):
        ms = s['max_stress'][nr]
        ms = ms[np.isfinite(ms)]
        if len(ms) > 0:
            s['max_von_mises'] = float(np.max(ms))
            idx = int(np.argmax(s['max_stress'][nr]))
            nr_ids = np.where(nr)[0]
            s['critical_member'] = int(nr_ids[idx])
        else:
            s['max_von_mises'] = 0.0
            s['critical_member'] = -1

        for key in ('axial', 'bending', 'shear'):
            vals = np.abs(s[key][nr])
            vals = vals[np.isfinite(vals)]
            s[f'max_{key}'] = float(np.max(vals)) if len(vals) > 0 else 0.0

    # ── Results packaging ─────────────────────────────────────────────

    def _package_linear_results(self):
        return {
            'displacements': self.current_state.u.copy(),
            'displacements_free': self.current_state.u_free.copy(),
            'max_displacement': self.current_state.get_max_displacement(),
            'external_forces': self.current_state.f_external.copy(),
            'member_displ_local': self.current_state.member_displ_local.copy(),
            'member_displ_global': self.current_state.member_displ_global.copy(),
            'nodes_deformed': self.get_deformed_coordinates(1),
        }

    def _package_nonlinear_results(self, disp_hist):
        return {
            'displacements': self.current_state.u.copy(),
            'displacements_free': self.current_state.u_free.copy(),
            'max_displacement': self.current_state.get_max_displacement(),
            'external_forces': self.current_state.f_external.copy(),
            'member_displ_local': self.current_state.member_displ_local.copy(),
            'member_displ_global': self.current_state.member_displ_global.copy(),
            'nodes_deformed': self.deformed_nodes.copy(),
            'member_rotations': self.element_matrices.theta_current - self.geometry.theta0,
            'load_history': list(self.load_factors),
            'displacement_history': disp_hist,
            'num_steps': len(self.load_factors),
            'state_history': self.converged_states,
        }