# ShepherdLab LW FEA-OPT (Python)

A lightweight, pure-Python toolkit for **finite element analysis** and **stress-ratio thickness optimisation** of 2-D frame and truss structures. Ported from the original MATLAB implementation, designed to be readable, hackable, and easy to extend.

## Features

- **2-D Euler–Bernoulli frame elements** with axial, shear, and bending DOFs
- **Linear and nonlinear (Newton-Raphson)** large-deflection analysis
- **Stress-ratio thickness optimisation** — iteratively redistributes material for uniform stress
- **Truss subdivision** — refine coarse truss meshes into fine frame elements for organic topology optimisation
- **Geometry import** from CSV and IGES/IGS files (e.g. SolidWorks spline exports)
- **Profile export** — trace optimised outlines and write SolidWorks-ready spline files
- **Visualisation** with physically-scaled beam thickness rendering
- **Rigid element support** via stiffness penalty method
- **Stress post-processing** — axial, bending, shear, von Mises, and factor of safety
- **Vectorised internals** — NumPy/SciPy throughout for performance

## Quick Start

```bash
# 1. Install dependencies
pip install numpy scipy matplotlib scikit-image

# 2. Run an included demo
python examples/demo_beam.py       # curved beam from IGS file
python examples/demo_truss.py      # Warren truss with subdivision
```

## Requirements

- Python 3.9 or later
- **NumPy** and **SciPy** — core numerical computation and sparse linear algebra
- **matplotlib** — visualisation (optional but recommended)
- **scikit-image** — required for profile export (rasterisation, morphological operations, contour tracing)

## Project Layout

```
lw-feaopt-py/
├── README.md
├── LICENSE
│
├── feaopt/
│   ├── __init__.py
│   ├── structure_geometry.py
│   ├── element_matrices.py
│   ├── analysis_state.py
│   ├── fea_analysis.py
│   ├── load_case.py
│   ├── structure_plotter.py
│   ├── optimize_thickness.py
│   ├── subdivide_truss.py
│   ├── importers.py
│   └── export_profile.py
│
├── examples/
│   ├── demo_beam.py
│   └── demo_truss.py
│
├── profiles/
│   └── About profiles.txt
│
└── docs/
    ├── main.tex
    └── chapters/
```

## Usage

### Define a structure

```python
import numpy as np
from feaopt import StructureGeometry, FEAAnalysis, LoadCase

nodes = np.array([[0, 0], [0.1, 0], [0.2, 0], [0.2, 0.1]])
conn = np.array([[0, 1], [1, 2], [2, 3]])  # 0-based indexing

props = {
    'E':         200e9,       # Young's modulus (Pa)
    'thickness': 0.005,       # element thickness (m)
    'width':     0.010,       # out-of-plane width (m)
    'ultimate':  400e6,       # UTS (Pa)
    'rho':       7850,        # density (kg/m³)
}

# Constraints: [node_id (0-based), dof (0=ux, 1=uy, 2=theta), value]
constraints = np.array([
    [0, 0, 0],   # node 0: ux = 0
    [0, 1, 0],   #         uy = 0
    [0, 2, 0],   #         theta = 0
])

geom = StructureGeometry(nodes, conn, props, constraints)
```

### Run analysis

```python
analysis = FEAAnalysis(geom)

lc = LoadCase(geom)
lc.add_nodal_load(3, 0, -5000, 0, 'Tip load')  # node 3, Fy = -5 kN
lc.set_load_factor(1.0)

# Linear
results = analysis.run_linear_analysis(lc)

# Nonlinear (5 load steps)
results = analysis.solve_large_deflection(lc, 5)

# Post-processing
stress = analysis.perform_stress_analysis()
print(f"Min FOS: {stress['FOS']['min_FOS']:.2f}")
```

### Optimise thickness

```python
from feaopt import optimize_thickness

opt_options = {
    'safety_factor': 2,
    'alpha': 0.4,
    'use_nonlinear': True,
}
geom, res = optimize_thickness(geom, analysis, lc, opt_options)
print(f"Mass reduction: {res['mass_reduction']:.1f}%")
```

### Truss subdivision & optimisation

```python
from feaopt import subdivide_truss, optimize_thickness, export_profile

# Define a truss
nodes = np.array([[0, 0], [0.1, 0], [0.2, 0], [0.05, 0.08], [0.15, 0.08]])
conn = np.array([[0, 1], [1, 2], [0, 3], [3, 1], [1, 4], [4, 2], [3, 4]])

# Subdivide each member into 20 small frame elements
fine_nodes, fine_conn, member_map = subdivide_truss(nodes, conn, 20)

# Build geometry, analyse, and optimise as usual
geom = StructureGeometry(fine_nodes, fine_conn, props, constraints)
analysis = FEAAnalysis(geom)
results = analysis.run_linear_analysis(lc)
geom, res = optimize_thickness(geom, analysis, lc, opt_options)

# Export with thin-element filtering (removes vanished material)
export_opts = {
    'min_thickness_frac': 0.15,
    'export_filename': 'profiles/truss',
}
profile = export_profile(geom, export_opts)
```

The `min_thickness_frac` option excludes elements thinner than a fraction of the maximum, leaving only the structural load paths in the exported outline.

### Visualise

```python
from feaopt import StructurePlotter

plotter = StructurePlotter(geom, analysis)
plotter.draw_structure(ax)
plotter.draw_stress(ax, stress_values=stress_mpa)
plotter.plot_supports(ax)
plotter.plot_loads(ax, lc)
```

### Import geometry from file

```python
from feaopt import import_csv_curve, import_igs_curve

# From CSV (x,y columns)
result = import_csv_curve('curve.csv', scale=0.001, num_elements=200)

# From IGES (e.g. SolidWorks sketch export)
result = import_igs_curve('sketch.igs', scale=0.001, num_elements=200)

# Use the imported mesh
geom = StructureGeometry(result['nodes'], result['connectivity'], props, constraints)
```

### Export optimised profile for CAD

```python
from feaopt import export_profile

opts = {
    'scale_to_mm': 1000,
    'smoothing': 0.999999999,
    'export_filename': 'profiles/my_part',
}
profile = export_profile(geom, opts)

# For trusses: filter out near-zero-thickness elements first
opts['min_thickness_frac'] = 0.15
profile = export_profile(geom, opts)
```

Files created in `profiles/` (one per boundary loop):
- `my_part_loop_0.txt` — outer boundary spline (X Y Z)
- `my_part_hole_1.txt` — inner hole spline (if any)
- `my_part.npz` — full NumPy data archive

The exporter works by rasterising every element as a filled rectangle onto a binary image, then using scikit-image's `find_contours` to trace all closed contours. Each contour is spline-fitted and exported as a separate file. This handles any topology — beams, trusses, branching, and structures with holes.

> **Performance note:** For small geometries, the default `resolution` of 50,000 pixels/m may produce a very small image. Increase it (e.g. to 500,000) for finer detail. For large geometries, the morphological closing disk radius can become expensive; it is automatically capped for performance.

## Analysis Methods

### Linear

Standard direct stiffness method: assembles the global stiffness matrix as a SciPy sparse matrix, partitions by free/fixed DOFs, and solves with `scipy.sparse.linalg.spsolve`.

### Nonlinear (Newton-Raphson)

Incremental load stepping with Newton-Raphson iterations at each step. At every iteration the code:

1. Updates element geometry (lengths, orientations) from current displacements
2. Assembles the tangent stiffness matrix
3. Computes internal forces and the residual R = F_ext − F_int
4. Solves for a displacement increment K_t · du = R
5. Checks force, displacement, and energy convergence criteria

### Thickness Optimisation

Fully stressed design via the stress-ratio method. At each iteration, every non-rigid element's thickness is updated as:

```
t_new = t_old * (sigma / sigma_allow)^alpha
```

where `alpha` controls aggressiveness (default 0.4). Move limits and absolute bounds prevent oscillation and ensure manufacturability.

## Differences from MATLAB Version

- **0-based indexing** — node IDs, element IDs, DOF indices, and constraint definitions all use 0-based indexing.
- **Dicts instead of structs** — results, options, and imported data are returned as Python dicts rather than MATLAB structs.
- **Sparse assembly** — the global stiffness matrix is assembled as a `scipy.sparse.csr_matrix` using vectorised index operations.
- **Vectorised element operations** — stiffness matrices, transformations, internal forces, and stress recovery are computed for all elements simultaneously using NumPy broadcasting and `einsum`.
- **scikit-image for export** — profile export uses `skimage.draw.polygon`, `skimage.morphology.closing`, and `skimage.measure.find_contours` in place of MATLAB's Image Processing Toolbox functions.
- **NumPy archive for export** — exported data is saved as `.npz` files rather than `.mat` files.

## License

MIT — see [LICENSE](LICENSE).
