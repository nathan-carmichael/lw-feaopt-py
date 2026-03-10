"""
LoadCase — Nodal load definition and load-factor management for FEA.

Port of LoadCase.m from ShepherdLab LW FEA-OPT.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class LoadRecord:
    node_id: int
    fx: float
    fy: float
    mz: float
    description: str


class LoadCase:
    """Nodal load definition with load-factor scaling for nonlinear stepping."""

    def __init__(self, geometry):
        self.geometry = geometry
        self.num_nodes = geometry.num_nodes
        self.num_dof_total = geometry.num_dof_total
        self.num_dof_free = geometry.num_dof_free

        self.P_ref = np.zeros(self.num_dof_total)
        self.P_current = np.zeros(self.num_dof_total)

        self.lambda_ = 0.0
        self.lambda_history: List[float] = []
        self.load_records: List[LoadRecord] = []

    # ── Load definition ───────────────────────────────────────────────

    def add_nodal_load(self, node_id, fx=0.0, fy=0.0, mz=0.0, description=None):
        if description is None:
            description = f"Load at node {node_id}"
        if node_id < 0 or node_id >= self.num_nodes:
            raise ValueError(f"Node {node_id} out of range [0, {self.num_nodes-1}].")

        dof = self.geometry.get_node_dof(node_id)
        self.P_ref[dof[0]] += fx
        self.P_ref[dof[1]] += fy
        self.P_ref[dof[2]] += mz

        self.load_records.append(LoadRecord(node_id, fx, fy, mz, description))
        self._refresh_current()

    def clear_loads(self):
        self.P_ref[:] = 0.0
        self.P_current[:] = 0.0
        self.lambda_ = 0.0
        self.lambda_history = []
        self.load_records = []

    # ── Load factor control ───────────────────────────────────────────

    def set_load_factor(self, lf):
        if lf < 0:
            import warnings
            warnings.warn(f"Negative load factor ({lf:.3f}) applied.")
        self.lambda_ = lf
        self.lambda_history.append(lf)
        self._refresh_current()

    def increment_load_factor(self, delta):
        self.set_load_factor(self.lambda_ + delta)

    def reset_load_factor(self):
        self.lambda_ = 0.0
        self.lambda_history = []
        self._refresh_current()

    # ── Load retrieval ────────────────────────────────────────────────

    def get_scaled_load(self):
        return self.P_current.copy()

    def get_free_dof_loads(self):
        return self.P_current[self.geometry.dof_map.free_dof].copy()

    def get_reference_free_dof_loads(self):
        return self.P_ref[self.geometry.dof_map.free_dof].copy()

    def apply_to_state(self, state):
        state.f_external = self.P_current.copy()
        state.f_free_ext = self.get_free_dof_loads()

    # ── Load steps ────────────────────────────────────────────────────

    def generate_load_steps(self, num_steps, step_type='linear'):
        if step_type == 'linear':
            factors = np.linspace(0, 1, num_steps + 1)[1:]
        elif step_type == 'quadratic':
            t = np.linspace(0, 1, num_steps + 1)
            factors = t[1:]**2
        else:
            raise ValueError(f"Unknown step type '{step_type}'.")
        return factors

    # ── Queries ───────────────────────────────────────────────────────

    def get_total_load(self):
        fx_all = self.P_current[0::3]
        fy_all = self.P_current[1::3]
        mz_all = self.P_current[2::3]
        return {
            'force_magnitude': float(np.linalg.norm(np.concatenate([fx_all, fy_all]))),
            'moment_magnitude': float(np.linalg.norm(mz_all)),
            'force_sum': np.array([fx_all.sum(), fy_all.sum()]),
            'moment_sum': float(mz_all.sum()),
        }

    def get_nodal_load(self, node_id):
        dof = self.geometry.get_node_dof(node_id)
        return self.P_current[dof[0]], self.P_current[dof[1]], self.P_current[dof[2]]

    def has_loads(self):
        return np.any(self.P_ref != 0)

    def display_loads(self):
        print(f"\n=== Load Case Summary ===")
        print(f"  Loads defined: {len(self.load_records)}")
        print(f"  Load factor:   {self.lambda_:.3f}\n")
        if self.load_records:
            print(f"  {'#':<4}  {'Node':<6}  {'Fx [N]':>12}  {'Fy [N]':>12}  {'Mz [Nm]':>12}  Description")
            print(f"  {'-'*68}")
            for i, r in enumerate(self.load_records):
                print(f"  {i:<4}  {r.node_id:<6}  {r.fx*self.lambda_:>12.2f}  "
                      f"{r.fy*self.lambda_:>12.2f}  {r.mz*self.lambda_:>12.2f}  {r.description}")
        s = self.get_total_load()
        print(f"\n  Totals:  |F| = {s['force_magnitude']:.3e} N,  |M| = {s['moment_magnitude']:.3e} Nm")

    # ── Private ───────────────────────────────────────────────────────

    def _refresh_current(self):
        self.P_current = self.P_ref * self.lambda_
