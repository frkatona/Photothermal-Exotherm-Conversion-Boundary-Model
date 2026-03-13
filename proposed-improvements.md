# Proposed Improvements

This document lists practical ways to move the project toward a more professional-grade simulation tool.

The ideas are grouped by goal:

- accuracy and physical realism
- numerical robustness and efficiency
- user experience and workflow
- software quality and maintainability

Each item is written as an approach rather than a vague wish, so it can be turned into actual work later.

## 1. Accuracy And Physical Realism

### Temperature-dependent material properties

Current material properties appear to be constant.

Professional tools often make these depend on temperature and sometimes conversion:

- `k(T, alpha)`
- `cp(T, alpha)`
- `rho(T, alpha)`

Why it matters:

- Laser heating can be highly nonlinear.
- Constant properties can mispredict peak temperature and cooling rate.

Approach:

- Start with tabulated curves or piecewise-linear fits.
- Interpolate values inside each timestep.

### Better reaction kinetics

The current reaction model is a simple Arrhenius-style conversion rate.

More realistic curing or decomposition models may require:

- nth-order kinetics
- autocatalytic kinetics
- diffusion-limited conversion at high alpha
- multiple competing reactions

Why it matters:

- Conversion behavior often controls heat release and final material state.

Approach:

- Add a kinetics model selector in the backend.
- Validate each model against published data or experiments.

### More realistic laser absorption

The current model uses a depth-averaged volumetric source based on film thickness and absorption.

More accurate approaches may include:

- true depth-dependent absorption through the thickness
- reflection losses at the surface
- wavelength-dependent absorption
- beam truncation and optics effects

Approach:

- For thin films, keep the current model as a fast option.
- Add an optional depth-resolved model for higher-fidelity runs.

### 3D geometry or layered stacks

A 2D in-plane model is useful, but professional prediction often needs:

- through-thickness resolution
- multilayer stacks
- substrate coupling
- buried interfaces

Approach:

- First extend to 2.5D or layered 1D-through-thickness coupling.
- Move to full 3D only if the use case truly requires it.

### Better boundary conditions

Current boundary conditions are simple and practical, but professional accuracy often needs:

- radiative losses
- substrate conduction
- insulated or symmetry boundaries
- mixed boundary conditions by edge or region

Approach:

- Expose boundary type per side or per region.
- Add radiative loss as an optional nonlinear term.

### Verification against experiments

A professional simulator needs calibration and validation, not just plausible output.

Approach:

- import experimental temperature traces
- compare measured and simulated peak temperature, cooling rate, and cure/conversion
- fit uncertain parameters such as absorption, convection, and kinetics

## 2. Numerical Robustness And Efficiency

### Mesh-convergence and timestep-convergence tools

Users should be able to answer:

- Is the mesh fine enough?
- Is the timestep small enough?
- Are the results stable and converged?

Approach:

- Add one-click mesh refinement and timestep refinement studies.
- Report change in peak temperature and final conversion.

### Adaptive timestepping

Fixed timesteps are simple, but they waste work when the solution is changing slowly and can miss sharp transients when it changes quickly.

Approach:

- Reduce `dt` near pulse peaks and steep temperature changes.
- Increase `dt` during slow cooling periods.

Benefits:

- higher accuracy where needed
- shorter runs overall

### CFL number and Runge-Kutta: useful here or not?

These ideas are common in simulation discussions, but they are not equally relevant to this specific model.

#### CFL number

The CFL number is most useful for transport or wave problems, where information moves across the mesh with a physical speed.

Examples:

- fluid advection
- shock propagation
- wave equations
- moving fronts dominated by transport

This project is currently much more of a diffusion-reaction problem than an advection problem. For that reason, the more relevant simple stability check is usually the Fourier number, not the CFL number.

Practical takeaway:

- if the model remains heat diffusion plus reaction plus source terms, Fourier-style timestep checks are the better beginner-facing metric
- if the model later adds true advection, moving material, or carrier transport, then a CFL check becomes useful

So the utility of CFL right now is limited. It is not wrong to mention it, but it is not the main control knob for the current equations.

#### Runge-Kutta methods

Runge-Kutta methods are a family of time-integration schemes. They can improve accuracy, but they are not automatically the right answer for a heat solver.

Why the benefit is limited in the current code:

- diffusion terms are often stiff, which makes explicit Runge-Kutta require very small timesteps
- the current solver already uses an implicit solve for the thermal step, which is usually the more practical approach for diffusion
- adding radiative loss and strongly temperature-dependent reaction terms can increase stiffness further

Where Runge-Kutta could still help:

- for a reaction-only substep in an operator-splitting approach
- for adaptive embedded error estimates, if the timestepper is redesigned
- for non-stiff auxiliary equations added later

Practical takeaway:

- replacing the current thermal solve with a plain explicit RK2 or RK4 method would probably hurt robustness and may even be slower overall because `dt` would need to shrink
- a more professional next step is usually adaptive implicit or semi-implicit timestepping, not plain explicit Runge-Kutta
- if higher temporal accuracy is needed, look first at implicit higher-order schemes, IMEX methods, or better timestep control

So Runge-Kutta is not useless, but for this kind of stiff heat-transfer problem it is not the first upgrade I would recommend.

### Stronger linear solver and preconditioning

A more professional solver stack would likely outperform the current plain CG approach.

Approach:

- add diagonal or incomplete-factorization preconditioning
- benchmark solver iteration count and wall time
- optionally support alternative solvers for different matrix types

### Matrix-free or structured-grid operators

The domain is structured, which opens the door to more efficient operators than generic sparse assembly.

Approach:

- keep FEM as the reference path
- add a faster structured-grid path where the assumptions are acceptable

### Binary result storage

Professional runs should not depend on keeping everything in JS memory.

Approach:

- save results in a compact binary format
- stream frames to the frontend on demand
- keep summaries and metadata separate from full fields

### Pause, cancel, and resume

Long runs should be controllable.

Approach:

- add cancel support first
- then pause/resume if state serialization is practical

## 3. User Experience And Workflow

### Rich parameter guidance

Warnings are already useful, but professional UX benefits from:

- inline help text
- recommended ranges
- explanations of failure modes
- links between suspicious parameter combinations and suggested fixes

### Compare-run workflow

Users often want to compare two or more runs quickly.

Approach:

- store run snapshots
- compare peak temperature, conversion, runtime, and settings
- overlay time-history curves

### Better result summaries

Professional users want a short answer before inspecting fields.

Examples:

- peak temperature and when it occurred
- hottest location
- final average conversion
- absorbed energy estimate
- fraction of domain above threshold temperature

### Improved plot interaction

Approach:

- hover readouts
- click-to-probe point histories
- zoom and pan
- fixed versus autoscaled legends

### Playback and animation controls

Further improvements beyond the current playback controls could include:

- reverse playback
- skip to peak temperature
- loop modes
- export GIF in addition to MP4

### Project files and run history

A more professional desktop tool should let users reopen work cleanly.

Approach:

- save complete project state to a file
- include run metadata, notes, plots, and exports
- show recent projects and recent runs

## 4. Software Quality And Maintainability

### Verification test suite

Beyond unit tests, professional simulation software benefits from:

- manufactured-solution tests
- energy-balance checks
- mesh-refinement regression tests
- solver-consistency tests

### Golden-reference cases

Keep a small set of trusted benchmark runs and compare new code against them automatically.

Approach:

- define acceptable tolerances for temperature and conversion outputs
- run them in CI

### Versioned parameter schemas

As the app grows, saved presets and project files will need migration support.

Approach:

- version project files and presets
- add migration code when parameters change

### Reproducibility metadata

Professional outputs should record:

- app version
- simulation settings
- solver mode
- timestamp
- runtime

This makes later review much easier.

## 5. Suggested Priority Order

If the goal is to improve the tool steadily without losing momentum, this is a practical order:

### Near-term

- better solver/preconditioning
- convergence-study tools
- improved summaries and compare-run workflow
- richer parameter help and validation
- cancel support

### Mid-term

- temperature-dependent properties
- better kinetics options
- binary result storage
- point probes and richer plot interaction
- project-file format

### Long-term

- layered or 3D modeling
- calibration against experiments
- optional high-fidelity absorption model
- alternative structured-grid or GPU-oriented backend

## 6. Decision Rule

When choosing the next improvement, ask:

1. Will this improve physical trustworthiness?
2. Will this reduce runtime or memory enough to matter?
3. Will this make real user workflows easier?
4. Can it be validated?

If the answer is "yes" to at least two of those, it is usually a strong candidate.
