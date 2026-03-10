"""
ElementMatrices — Pre-computed element stiffness and transformation matrices.

Port of ElementMatrices.m — vectorized for performance.
"""

import numpy as np
from scipy import sparse


class ElementMatrices:
    """Manages local/global stiffness and transformation matrices for all elements."""

    def __init__(self, geometry):
        self.geometry = geometry
        self.num_members = geometry.num_members
        self.rigid_stiffness_factor = 1e2

        M = self.num_members
        self.k_local = np.zeros((M, 6, 6))
        self.T = np.zeros((M, 6, 6))
        self.k_global = np.zeros((M, 6, 6))

        self.theta_current = geometry.theta0.copy()
        self.L_current = geometry.L.copy()

        self._dirty_local = np.zeros(M, dtype=bool)
        self._dirty_global = np.zeros(M, dtype=bool)

        self.rebuild_all()

    # ── Full rebuild (vectorized) ─────────────────────────────────────

    def rebuild_all(self):
        self._build_all_local_matrices()
        self._build_all_transformation_matrices()
        self._build_all_global_matrices()
        self._dirty_local[:] = False
        self._dirty_global[:] = False

    def reset_to_initial_configuration(self):
        self.theta_current = self.geometry.theta0.copy()
        self.L_current = self.geometry.L.copy()
        self.rebuild_all()

    # ── Vectorized matrix builders ────────────────────────────────────

    def _build_all_local_matrices(self):
        """Build all local stiffness matrices at once using vectorized ops."""
        g = self.geometry
        M = self.num_members

        E = g.E.copy()
        rigid = g.is_rigid
        E[rigid] *= self.rigid_stiffness_factor

        A = g.A
        I = g.I
        L = self.L_current

        ea_L = E * A / L
        ei_L3 = E * I / L**3
        ei_L2 = ei_L3 * L      # EI/L^2
        ei_L1 = ei_L3 * L**2   # EI/L

        # Build all 6x6 matrices at once
        k = np.zeros((M, 6, 6))

        k[:, 0, 0] =  ea_L;    k[:, 0, 3] = -ea_L
        k[:, 3, 0] = -ea_L;    k[:, 3, 3] =  ea_L

        k[:, 1, 1] =  12*ei_L3;   k[:, 1, 2] =  6*ei_L2
        k[:, 1, 4] = -12*ei_L3;   k[:, 1, 5] =  6*ei_L2

        k[:, 2, 1] =  6*ei_L2;    k[:, 2, 2] =  4*ei_L1
        k[:, 2, 4] = -6*ei_L2;    k[:, 2, 5] =  2*ei_L1

        k[:, 4, 1] = -12*ei_L3;   k[:, 4, 2] = -6*ei_L2
        k[:, 4, 4] =  12*ei_L3;   k[:, 4, 5] = -6*ei_L2

        k[:, 5, 1] =  6*ei_L2;    k[:, 5, 2] =  2*ei_L1
        k[:, 5, 4] = -6*ei_L2;    k[:, 5, 5] =  4*ei_L1

        self.k_local = k

    def _build_all_transformation_matrices(self):
        """Build all transformation matrices at once."""
        c = np.cos(self.theta_current)
        s = np.sin(self.theta_current)
        M = self.num_members
        T = np.zeros((M, 6, 6))

        T[:, 0, 0] =  c;  T[:, 0, 1] = s
        T[:, 1, 0] = -s;  T[:, 1, 1] = c
        T[:, 2, 2] = 1.0
        T[:, 3, 3] =  c;  T[:, 3, 4] = s
        T[:, 4, 3] = -s;  T[:, 4, 4] = c
        T[:, 5, 5] = 1.0

        self.T = T

    def _build_all_global_matrices(self):
        """k_global = T^T @ k_local @ T for all elements (batched)."""
        # (M,6,6) batched matrix multiply
        Tt = np.transpose(self.T, (0, 2, 1))  # T^T
        self.k_global = Tt @ self.k_local @ self.T

    # ── Geometry updates (Newton-Raphson) ─────────────────────────────

    def update_member_geometry(self, i, new_L, new_theta):
        L_changed = abs(new_L - self.L_current[i]) / self.L_current[i] > 1e-8
        theta_changed = abs(new_theta - self.theta_current[i]) > 1e-10

        if not L_changed and not theta_changed:
            return

        self.L_current[i] = new_L
        self.theta_current[i] = new_theta

        if L_changed:
            self._rebuild_single_local(i)
        if theta_changed:
            self._rebuild_single_transform(i)

        Ti = self.T[i]
        Kg = Ti.T @ self.k_local[i] @ Ti
        Kg[np.abs(Kg) < 1e-12] = 0.0
        self.k_global[i] = Kg

    def update_all_member_geometry(self, lengths, thetas):
        """Batch update — rebuilds everything vectorized."""
        self.L_current = lengths.copy()
        self.theta_current = thetas.copy()
        self._build_all_local_matrices()
        self._build_all_transformation_matrices()
        self._build_all_global_matrices()

    # ── Matrix retrieval ──────────────────────────────────────────────

    def get_local_matrix(self, i):
        return self.k_local[i]

    def get_global_matrix(self, i):
        return self.k_global[i]

    def get_transformation(self, i):
        return self.T[i]

    def get_current_angle(self, i):
        return self.theta_current[i]

    def get_current_length(self, i):
        return self.L_current[i]

    # ── Global assembly (vectorized) ──────────────────────────────────

    def assemble_global_stiffness_matrix(self, geometry):
        """Assemble full sparse [D x D] stiffness matrix — vectorized."""
        D = geometry.num_dof_total
        M = self.num_members
        dof_all = geometry.assembly_map.member_dof  # (M, 6)

        # Build row/col index arrays for all elements at once
        row_idx = np.repeat(dof_all, 6, axis=1)           # (M, 36)
        col_idx = np.tile(dof_all, (1, 6))                 # (M, 36) — wrong pattern

        # Correct: for each element, row[i]*6+j gives row dof_all[i], col dof_all[j]
        row_idx = dof_all[:, :, None] * np.ones((1, 1, 6), dtype=int)  # (M, 6, 6)
        col_idx = dof_all[:, None, :] * np.ones((1, 6, 1), dtype=int)  # (M, 6, 6)

        rows = row_idx.ravel()
        cols = col_idx.ravel()
        vals = self.k_global.ravel()

        K = sparse.csr_matrix((vals, (rows, cols)), shape=(D, D))
        return K

    # ── Vectorized internal force computation ─────────────────────────

    def compute_internal_forces_vectorized(self, u, geometry):
        """Compute f_internal = sum of T^T @ k_local @ T @ u_elem for all elements.

        Returns the full f_internal vector.
        """
        D = geometry.num_dof_total
        dof_all = geometry.assembly_map.member_dof  # (M, 6)

        # Gather element displacements: (M, 6)
        u_elem = u[dof_all]

        # Batched: f_local = k_local @ (T @ u_elem)
        # T @ u_elem: (M,6,6) @ (M,6,1) -> (M,6,1)
        u_local = np.einsum('mij,mj->mi', self.T, u_elem)
        f_local = np.einsum('mij,mj->mi', self.k_local, u_local)
        f_global = np.einsum('mji,mj->mi', self.T, f_local)  # T^T @ f_local

        # Scatter-add into global vector
        f_int = np.zeros(D)
        np.add.at(f_int, dof_all, f_global)

        return f_int

    # ── Vectorized member forces ──────────────────────────────────────

    def compute_member_forces_vectorized(self, u, geometry):
        """Compute local and global member forces for all elements at once.

        Returns (forces_local, forces_global) each (M, 6).
        """
        dof_all = geometry.assembly_map.member_dof
        u_elem = u[dof_all]  # (M, 6)

        u_local = np.einsum('mij,mj->mi', self.T, u_elem)
        f_local = np.einsum('mij,mj->mi', self.k_local, u_local)
        f_global = np.einsum('mji,mj->mi', self.T, f_local)

        return f_local, f_global

    # ── Rigid stiffness ───────────────────────────────────────────────

    def set_rigid_stiffness_factor(self, factor):
        self.rigid_stiffness_factor = factor
        self.rebuild_all()

    # ── Private single-element rebuilders (for N-R per-element updates) ──

    def _rebuild_single_local(self, i):
        g = self.geometry
        Ev = g.E[i]
        if g.is_rigid[i]:
            Ev *= self.rigid_stiffness_factor
        Av, Iv, Lv = g.A[i], g.I[i], self.L_current[i]

        ea_L = Ev * Av / Lv
        ei_L3 = Ev * Iv / Lv**3

        k = np.array([
            [ ea_L,         0,              0,             -ea_L,         0,              0            ],
            [ 0,            12*ei_L3,       6*ei_L3*Lv,     0,          -12*ei_L3,       6*ei_L3*Lv   ],
            [ 0,            6*ei_L3*Lv,     4*ei_L3*Lv**2,  0,          -6*ei_L3*Lv,     2*ei_L3*Lv**2],
            [-ea_L,         0,              0,              ea_L,         0,              0            ],
            [ 0,           -12*ei_L3,      -6*ei_L3*Lv,     0,           12*ei_L3,      -6*ei_L3*Lv   ],
            [ 0,            6*ei_L3*Lv,     2*ei_L3*Lv**2,  0,          -6*ei_L3*Lv,     4*ei_L3*Lv**2],
        ])
        self.k_local[i] = k

    def _rebuild_single_transform(self, i):
        c = np.cos(self.theta_current[i])
        s = np.sin(self.theta_current[i])
        self.T[i] = np.array([
            [ c,  s,  0,  0,  0,  0],
            [-s,  c,  0,  0,  0,  0],
            [ 0,  0,  1,  0,  0,  0],
            [ 0,  0,  0,  c,  s,  0],
            [ 0,  0,  0, -s,  c,  0],
            [ 0,  0,  0,  0,  0,  1],
        ])