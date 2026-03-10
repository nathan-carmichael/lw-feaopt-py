"""
ShepherdLab LW FEA-OPT — Lightweight 2-D Frame Topology Optimisation

A pure-Python toolkit for finite element analysis and stress-ratio
thickness optimisation of 2-D frame and truss structures.

Usage:
    from feaopt import StructureGeometry, FEAAnalysis, LoadCase
    from feaopt import optimize_thickness, subdivide_truss
    from feaopt import import_csv_curve, import_igs_curve
    from feaopt import export_profile
"""

from feaopt.structure_geometry import StructureGeometry, Constraint
from feaopt.element_matrices import ElementMatrices
from feaopt.analysis_state import AnalysisState
from feaopt.fea_analysis import FEAAnalysis
from feaopt.load_case import LoadCase
from feaopt.optimize_thickness import optimize_thickness
from feaopt.subdivide_truss import subdivide_truss
from feaopt.importers import import_csv_curve, import_igs_curve
from feaopt.export_profile import export_profile

try:
    from feaopt.structure_plotter import StructurePlotter
except ImportError:
    # matplotlib not installed — plotting unavailable
    pass

__version__ = "1.0.0"

__all__ = [
    "StructureGeometry", "Constraint",
    "ElementMatrices",
    "AnalysisState",
    "FEAAnalysis",
    "LoadCase",
    "optimize_thickness",
    "subdivide_truss",
    "import_csv_curve", "import_igs_curve",
    "export_profile",
    "StructurePlotter",
]
