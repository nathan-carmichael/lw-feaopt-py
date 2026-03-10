"""
StructureGeometry — Mutable structure topology, geometry, and material data.

Port of StructureGeometry.m from ShepherdLab LW FEA-OPT.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Constraint:
    node_id: int  # 0-based
    dof: int      # 0, 1, or 2 (ux, uy, theta)
    value: float


@dataclass
class DOFMap:
    node_to_dof: np.ndarray   # [N x 3]
    dof_to_node: np.ndarray   # [D x 2] (node_id, local_dof)
    fixed_dof: np.ndarray     # sorted indices
    free_dof: np.ndarray      # sorted indices


@dataclass
class AssemblyMap:
    member_dof: np.ndarray    # [M x 6]
    member_nodes: np.ndarray  # [M x 2]


class StructureGeometry:
    """Mutable structure topology, geometry, and material data for 2-D frames."""

    def __init__(self, nodes, connectivity, props, constraints_raw):
        """
        Parameters
        ----------
        nodes : (N, 2) array — node coordinates [x, y] in metres
        connectivity : (M, 2) array — element connectivity [start, end], 0-based
        props : dict with keys:
            Required: 'E', 'thickness', 'width'
            Optional: 'ultimate', 'rho', 'is_rigid'
            Each can be scalar or (M,) array.
        constraints_raw : (C, 3) array — [node_id (0-based), dof (0-2), value]
                          or list of Constraint objects
        """
        self.nodes = np.asarray(nodes, dtype=float)
        self.connectivity = np.asarray(connectivity, dtype=int)
        self.num_nodes = self.nodes.shape[0]
        self.num_members = self.connectivity.shape[0]
        self.num_dof_total = 3 * self.num_nodes

        M = self.num_members

        # Required properties
        self.E = self._expand(props['E'], M, 'E')
        self.thickness = self._expand(props['thickness'], M, 'thickness')
        self.width = self._expand(props['width'], M, 'width')

        # Optional properties
        self.ultimate = self._expand(props.get('ultimate', 0.0), M, 'ultimate')
        self.rho = self._expand(props.get('rho', 7850.0), M, 'rho')

        is_rigid = props.get('is_rigid', False)
        if np.isscalar(is_rigid):
            self.is_rigid = np.full(M, bool(is_rigid))
        else:
            self.is_rigid = np.asarray(is_rigid, dtype=bool).reshape(M)

        # Derived section properties
        self.A = self.width * self.thickness
        self.I = self.width * self.thickness**3 / 12.0

        # Constraints
        self.constraints = self._parse_constraints(constraints_raw)
        self.num_constraints = len(self.constraints)

        # Geometry
        self.L = self._compute_lengths()
        self.theta0 = self._compute_orientations()

        # DOF maps
        self.dof_map = self._build_dof_map()
        self.num_dof_free = len(self.dof_map.free_dof)
        self.assembly_map = self._build_assembly_map()

        # Validation
        self._validate()

    # ── Mutators ──────────────────────────────────────────────────────

    def update_node_positions(self, new_nodes):
        if new_nodes.shape != self.nodes.shape:
            raise ValueError(f"new_nodes must be ({self.num_nodes}, 2)")
        self.nodes = np.asarray(new_nodes, dtype=float)
        self.L = self._compute_lengths()
        self.theta0 = self._compute_orientations()

    def update_thickness(self, new_t):
        new_t = self._expand(new_t, self.num_members, 'thickness')
        if np.any(new_t <= 0):
            raise ValueError("Thickness must be positive.")
        self.thickness = new_t
        self.A = self.width * self.thickness
        self.I = self.width * self.thickness**3 / 12.0

    # ── Element queries ───────────────────────────────────────────────

    def get_member_nodes(self, i):
        """Return (n1, n2) node indices for element i."""
        self._assert_member(i)
        return int(self.connectivity[i, 0]), int(self.connectivity[i, 1])

    def get_member_dof(self, i):
        """Return 6 global DOF indices for element i."""
        self._assert_member(i)
        return self.assembly_map.member_dof[i]

    # ── Node queries ──────────────────────────────────────────────────

    def get_node_coords(self, i):
        self._assert_node(i)
        return self.nodes[i]

    def get_node_dof(self, i):
        """Return 3 global DOF indices for node i."""
        self._assert_node(i)
        return self.dof_map.node_to_dof[i]

    def is_dof_constrained(self, dof_id):
        return dof_id in self.dof_map.fixed_dof

    # ── Aggregate queries ─────────────────────────────────────────────

    def get_member_mass(self, i):
        self._assert_member(i)
        return self.rho[i] * self.A[i] * self.L[i]

    def get_total_mass(self):
        return float(np.sum(self.rho * self.A * self.L))

    # ── Private helpers ───────────────────────────────────────────────

    def _compute_lengths(self):
        n1 = self.connectivity[:, 0]
        n2 = self.connectivity[:, 1]
        dx = self.nodes[n2, 0] - self.nodes[n1, 0]
        dy = self.nodes[n2, 1] - self.nodes[n1, 1]
        return np.sqrt(dx**2 + dy**2)

    def _compute_orientations(self):
        n1 = self.connectivity[:, 0]
        n2 = self.connectivity[:, 1]
        dx = self.nodes[n2, 0] - self.nodes[n1, 0]
        dy = self.nodes[n2, 1] - self.nodes[n1, 1]
        return np.arctan2(dy, dx)

    def _build_dof_map(self):
        N, D = self.num_nodes, self.num_dof_total

        node_to_dof = np.zeros((N, 3), dtype=int)
        for i in range(N):
            node_to_dof[i] = [3*i, 3*i+1, 3*i+2]

        dof_to_node = np.zeros((D, 2), dtype=int)
        for n in range(N):
            for ld in range(3):
                gd = 3*n + ld
                dof_to_node[gd] = [n, ld]

        fixed = []
        for c in self.constraints:
            fixed.append(3 * c.node_id + c.dof)
        fixed = np.unique(np.array(fixed, dtype=int))

        free = np.setdiff1d(np.arange(D), fixed)

        return DOFMap(node_to_dof, dof_to_node, fixed, free)

    def _build_assembly_map(self):
        M = self.num_members
        n1 = self.connectivity[:, 0]
        n2 = self.connectivity[:, 1]
        mdof = np.zeros((M, 6), dtype=int)
        for i in range(M):
            mdof[i] = [3*n1[i], 3*n1[i]+1, 3*n1[i]+2,
                       3*n2[i], 3*n2[i]+1, 3*n2[i]+2]
        return AssemblyMap(mdof, self.connectivity.copy())

    def _validate(self):
        if np.any(self.connectivity < 0) or np.any(self.connectivity >= self.num_nodes):
            raise ValueError(f"Connectivity references nodes outside [0, {self.num_nodes-1}].")
        zero_L = np.where(self.L < np.finfo(float).eps)[0]
        if len(zero_L) > 0:
            raise ValueError(f"Zero-length elements: {zero_L}")
        for i, c in enumerate(self.constraints):
            if c.node_id < 0 or c.node_id >= self.num_nodes:
                raise ValueError(f"Constraint {i} references invalid node {c.node_id}.")
            if c.dof < 0 or c.dof > 2:
                raise ValueError(f"Constraint {i} has invalid DOF {c.dof}.")
        if self.num_dof_free == self.num_dof_total:
            import warnings
            warnings.warn("No constraints — rigid-body motion possible.")
        if np.any(self.E <= 0):
            raise ValueError("Elastic modulus must be positive.")
        if np.any(self.A <= 0):
            raise ValueError("Cross-sectional area must be positive.")

    def _assert_member(self, i):
        if i < 0 or i >= self.num_members:
            raise IndexError(f"Element {i} out of range [0, {self.num_members-1}].")

    def _assert_node(self, i):
        if i < 0 or i >= self.num_nodes:
            raise IndexError(f"Node {i} out of range [0, {self.num_nodes-1}].")

    @staticmethod
    def _expand(val, M, name):
        val = np.atleast_1d(np.asarray(val, dtype=float))
        if val.size == 1:
            return np.full(M, val.item())
        elif val.size == M:
            return val.reshape(M)
        else:
            raise ValueError(f"'{name}' must be scalar or length {M}.")

    @staticmethod
    def _parse_constraints(raw):
        if isinstance(raw, np.ndarray) or (isinstance(raw, list) and len(raw) > 0
                                            and not isinstance(raw[0], Constraint)):
            raw = np.asarray(raw)
            return [Constraint(int(r[0]), int(r[1]), float(r[2])) for r in raw]
        elif isinstance(raw, list):
            return raw
        else:
            raise ValueError("Constraints must be (C,3) array or list of Constraint.")
