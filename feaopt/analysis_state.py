"""
AnalysisState — Mutable state vector container for FEA analysis.

Port of AnalysisState.m — uses vectorized element matrix operations.
"""

import numpy as np


class AnalysisState:
    """Holds displacement, force, and residual vectors for N-R workflow."""

    def __init__(self, geometry):
        self.geometry = geometry
        self.num_nodes = geometry.num_nodes
        self.num_members = geometry.num_members
        self.num_dof_total = geometry.num_dof_total
        self.num_dof_free = geometry.num_dof_free
        self.reset()

    def reset(self):
        D = self.num_dof_total
        F = self.num_dof_free
        M = self.num_members

        self.u = np.zeros(D)
        self.f_external = np.zeros(D)
        self.f_internal = np.zeros(D)

        self.u_free = np.zeros(F)
        self.f_free_ext = np.zeros(F)
        self.f_free_int = np.zeros(F)

        self.residual = np.zeros(F)
        self.residual_norm = 0.0
        self.last_delta_u = None

        self.member_forces_local = np.zeros((M, 6))
        self.member_forces_global = np.zeros((M, 6))
        self.member_displ_local = np.zeros((M, 6))
        self.member_displ_global = np.zeros((M, 6))

    def copy_from(self, src):
        self.u = src.u.copy()
        self.f_external = src.f_external.copy()
        self.f_internal = src.f_internal.copy()
        self.u_free = src.u_free.copy()
        self.f_free_ext = src.f_free_ext.copy()
        self.f_free_int = src.f_free_int.copy()
        self.residual = src.residual.copy()
        self.residual_norm = src.residual_norm
        self.last_delta_u = src.last_delta_u.copy() if src.last_delta_u is not None else None
        self.member_forces_local = src.member_forces_local.copy()
        self.member_forces_global = src.member_forces_global.copy()
        self.member_displ_local = src.member_displ_local.copy()
        self.member_displ_global = src.member_displ_global.copy()

    # ── DOF mapping ───────────────────────────────────────────────────

    def update_displacements(self, u_free_new):
        if u_free_new.size != self.num_dof_free:
            raise ValueError(f"Expected length {self.num_dof_free}, got {u_free_new.size}.")
        self.u_free = u_free_new.ravel().copy()
        self.u[:] = 0.0
        self.u[self.geometry.dof_map.free_dof] = self.u_free

    def map_full_to_free(self):
        idx = self.geometry.dof_map.free_dof
        self.u_free = self.u[idx].copy()
        self.f_free_ext = self.f_external[idx].copy()
        self.f_free_int = self.f_internal[idx].copy()

    def map_free_to_full(self):
        idx = self.geometry.dof_map.free_dof
        self.u[idx] = self.u_free
        self.f_external[idx] = self.f_free_ext
        self.f_internal[idx] = self.f_free_int

    # ── Force computation (vectorized) ────────────────────────────────

    def compute_internal_forces(self, elem_matrices):
        """Vectorized internal force assembly."""
        geom = self.geometry
        self.f_internal = elem_matrices.compute_internal_forces_vectorized(self.u, geom)
        self.f_internal[geom.dof_map.fixed_dof] = 0.0
        self.f_free_int = self.f_internal[geom.dof_map.free_dof].copy()

    def compute_residual(self):
        self.residual = self.f_free_ext - self.f_free_int
        self.residual_norm = float(np.linalg.norm(self.residual))

    # ── Element-level results (vectorized) ────────────────────────────

    def extract_member_displacements(self, elem_matrices):
        """Vectorized extraction of member displacements."""
        dof_all = self.geometry.assembly_map.member_dof
        u_elem = self.u[dof_all]  # (M, 6)
        self.member_displ_global = u_elem
        self.member_displ_local = np.einsum('mij,mj->mi', elem_matrices.T, u_elem)

    def compute_member_forces(self, elem_matrices):
        """Vectorized member force computation."""
        f_local, f_global = elem_matrices.compute_member_forces_vectorized(
            self.u, self.geometry)
        self.member_forces_local = f_local
        self.member_forces_global = f_global

    # ── N-R increment helpers ─────────────────────────────────────────

    def apply_increment(self, delta_u_free):
        self.last_delta_u = delta_u_free.copy()
        self.u_free += delta_u_free
        self.u[self.geometry.dof_map.free_dof] = self.u_free

    def scale_state(self, factor):
        self.u *= factor
        self.u_free *= factor

    # ── Queries ───────────────────────────────────────────────────────

    def get_nodal_displacements(self, node_id):
        dof = self.geometry.get_node_dof(node_id)
        return self.u[dof]

    def get_nodal_rotations(self):
        return self.u[2::3]

    def get_member_forces(self, i):
        self.geometry._assert_member(i)
        f = self.member_forces_local[i]
        axial = np.array([f[0], f[3]])
        shear = np.array([f[1], f[4]])
        moment = np.array([f[2], f[5]])
        return axial, shear, moment

    def get_max_displacement(self):
        ux = self.u[0::3]
        uy = self.u[1::3]
        return float(np.max(np.sqrt(ux**2 + uy**2)))

    def display_summary(self):
        print(f"\n=== Analysis State Summary ===")
        print(f"Nodes: {self.num_nodes}, Elements: {self.num_members}")
        print(f"Total DOFs: {self.num_dof_total}, Free DOFs: {self.num_dof_free}")
        print(f"Max displacement: {self.get_max_displacement():.3e} m")
        print(f"Residual norm:    {self.residual_norm:.3e}")