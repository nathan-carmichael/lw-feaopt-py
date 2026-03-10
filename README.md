# ShepherdLab LW FEA-OPT

A lightweight, pure-MATLAB toolkit for **finite element analysis** and **stress-ratio thickness optimisation** of 2-D frame and truss structures. Designed to be readable, hackable, and easy to extend.

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

## Quick Start

```matlab
% 1. Add LW FEA-OPT to the MATLAB path
run('setup.m')

% 2. Run an included demo
demo_beam_optimization      % curved beam from IGS file
demo_truss_optimization     % Warren truss with subdivision
```

## Project Layout

```
lw-feaopt/
├── .gitignore
├── README.md
├── LICENSE
├── setup.m
│
├── src/
│   ├── StructureGeometry.m
│   ├── ElementMatrices.m
│   ├── AnalysisState.m
│   ├── FEAAnalysis.m
│   ├── LoadCase.m
│   ├── StructurePlotter.m
│   ├── optimizeThickness.m
│   └── subdivideTruss.m
│
├── io/
│   ├── importCSVCurve.m
│   ├── importIGSCurve.m
│   └── exportProfile.m
│
├── examples/
│   ├── demo_beam_optimization.m
│   └── demo_truss_optimization.m
│
├── profiles/
│   └── .gitkeep          
│
└── docs/
    ├── main.tex
    └── chapters/
        ├── ch1_introduction.tex
        ├── ch2_theory.tex
        ├── ch3_api_reference.tex
        ├── ch4_examples.tex
        ├── ch5_export.tex
        └── app_a_file_listing.tex
```

## Usage

### Define a structure

```matlab
nodes = [0 0; 1 0; 2 0; 2 1];
conn  = [1 2; 2 3; 3 4];

props = struct( ...
    'E',         200e9, ...       % Young's modulus (Pa)
    'thickness', 0.005, ...       % element thickness (m)
    'width',     0.010, ...       % out-of-plane width (m)
    'ultimate',  400e6, ...       % UTS (Pa)
    'rho',       7850);           % density (kg/m^3)

constraints = [1 1 0;   % node 1: ux = 0
               1 2 0;   %         uy = 0
               1 3 0];  %         theta = 0

geom = StructureGeometry(nodes, conn, props, constraints);
```

### Run analysis

```matlab
analysis = FEAAnalysis(geom);

lc = LoadCase(geom);
lc.addNodalLoad(4, 0, -5000, 0, 'Tip load');
lc.setLoadFactor(1.0);

% Linear
results = analysis.runLinearAnalysis(lc);

% Nonlinear (5 load steps)
results = analysis.solveLargeDeflection(lc, 5);

% Post-processing
stress = analysis.performStressAnalysis();
fprintf('Min FOS: %.2f\n', stress.FOS.min_FOS);
```

### Optimise thickness

```matlab
opt = struct('safety_factor', 2, 'alpha', 0.4, 'use_nonlinear', true);
[geom, res] = optimizeThickness(geom, analysis, lc, opt);
fprintf('Mass reduction: %.1f%%\n', res.mass_reduction);
```

### Truss subdivision & optimisation

```matlab
% Define a truss
nodes = [0 0; 1 0; 2 0; 0.5 0.8; 1.5 0.8];
conn  = [1 2; 2 3; 1 4; 4 2; 2 5; 5 3; 4 5];

% Subdivide each member into 20 small frame elements
[fine_nodes, fine_conn, member_map] = subdivideTruss(nodes, conn, 20);

% Build geometry, analyse, and optimise as usual
geom = StructureGeometry(fine_nodes, fine_conn, props, constraints);
analysis = FEAAnalysis(geom);
results = analysis.runLinearAnalysis(lc);
[geom, res] = optimizeThickness(geom, analysis, lc, opt);

% Export with thin-element filtering (removes vanished material)
profile = exportProfile(geom, struct('min_thickness_frac', 0.15, ...
    'export_filename', 'profiles/truss'));
```

The `min_thickness_frac` option excludes elements thinner than a fraction of the
maximum, leaving only the structural load paths in the exported outline.

### Visualise

```matlab
plotter = StructurePlotter(geom, analysis);
plotter.plotStructure();
plotter.plotDeformedShape();
plotter.plotStressDistribution('combined');
```

### Import geometry from file

```matlab
% From CSV (x,y columns)
result = importCSVCurve('curve.csv', struct('scale', 0.001, 'num_elements', 200));

% From IGES (e.g. SolidWorks sketch export)
result = importIGSCurve('sketch.igs', struct('scale', 0.001, 'num_elements', 200));

% Use the imported mesh
geom = StructureGeometry(result.nodes, result.connectivity, props, constraints);
```

### Export optimised profile for CAD

```matlab
% Rasterise the optimised structure, extract boundary loops, and export
opts = struct('scale_to_mm', 1000, 'smoothing', 0.999999999, ...
              'export_filename', 'profiles/my_part');
profile = exportProfile(geom, opts);

% For trusses: filter out near-zero-thickness elements first
opts.min_thickness_frac = 0.15;
profile = exportProfile(geom, opts);

% Files created in profiles/ (one per boundary loop):
%   my_part_loop_1.txt    — outer boundary spline (X Y Z)
%   my_part_hole_2.txt    — inner hole spline (if any)
%   my_part.mat           — full MATLAB data
```

The exporter works by rasterising every element as a filled rectangle onto
a binary image, then using `bwboundaries` to trace all closed contours.
Each contour is spline-fitted and exported as a separate file. This handles
any topology — beams, trusses, branching, and structures with holes.

## Analysis Methods

### Linear

Standard direct stiffness method: assembles the global stiffness matrix, partitions by free/fixed DOFs, and solves `K_ff * u_f = P_f`.

### Nonlinear (Newton-Raphson)

Incremental load stepping with Newton-Raphson iterations at each step. At every iteration the code:

1. Updates element geometry (lengths, orientations) from current displacements
2. Assembles the tangent stiffness matrix
3. Computes internal forces and the residual `R = F_ext - F_int`
4. Solves for a displacement increment `K_t * du = R`
5. Checks force, displacement, and energy convergence criteria

### Thickness Optimisation

Fully stressed design via the stress-ratio method. At each iteration, every non-rigid element's thickness is updated as:

```
t_new = t_old * (sigma / sigma_allow)^alpha
```

where `alpha` controls aggressiveness (default 0.4). Move limits and absolute bounds prevent oscillation and ensure manufacturability.

## Requirements

- MATLAB R2019b or later
- **Image Processing Toolbox** — required for profile export (`bwboundaries`, `poly2mask`, `imclose`)
- Curve Fitting Toolbox — optional, enables `csaps` smoothing (falls back to `pchip` interpolation)

## License

MIT — see [LICENSE](LICENSE).
