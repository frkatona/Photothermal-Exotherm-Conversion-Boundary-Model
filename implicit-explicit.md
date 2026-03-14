# Implicit and Explicit Methods

This note explains what "explicit" and "implicit" time-integration methods mean in general, what the current code is doing, and why those choices make sense here.

## Short version

- Explicit methods compute the next state directly from information already known at the current step.
- Implicit methods define the next state through an equation that contains the unknown future state, so each step requires solving a system.
- This project uses a mixed approach:
  - reaction/conversion is updated explicitly
  - the heat equation is advanced implicitly
  - laser and cooling source terms are treated mostly explicitly

That makes the current scheme an IMEX-style or semi-implicit operator-split method rather than a fully explicit or fully implicit one.

## The general difference

Suppose a variable `u` follows

`du/dt = f(u, t)`

### Explicit step

Forward Euler is the simplest explicit method:

`u^(n+1) = u^n + dt * f(u^n, t^n)`

Everything on the right-hand side is already known. This makes the step cheap and simple.

Advantages:

- easy to implement
- cheap per step
- natural for source terms or weakly coupled effects

Disadvantages:

- often limited by stability
- may require very small `dt`
- can struggle with stiff physics

### Implicit step

Backward Euler is the simplest implicit method:

`u^(n+1) = u^n + dt * f(u^(n+1), t^(n+1))`

Now the unknown future state appears on both sides, so you must solve for `u^(n+1)`.

Advantages:

- much more stable for stiff problems
- can take larger timesteps for diffusion-dominated systems

Disadvantages:

- each step is more expensive
- implementation is more complex
- nonlinear problems may require Newton iterations or other nonlinear solves

## Why this matters in thermal problems

Diffusion is the classic reason to prefer implicit methods. If heat conduction is handled explicitly, the timestep is usually restricted by a Fourier-type condition of the form

`dt <= C * dx^2 / alpha`

where `alpha = k / (rho cp)` is thermal diffusivity.

As the mesh gets finer, `dx` gets smaller and the stable explicit timestep shrinks like `dx^2`. That becomes painful quickly.

Reaction kinetics can also become stiff, especially near a thermal runaway or sharp Arrhenius threshold. If handled explicitly, they often need either:

- very small `dt`, or
- adaptive stepping, or
- a more advanced IMEX / fully implicit coupling

## What this project does now

The main timestep loop in `fem_heat_reaction_rust/src/lib.rs` follows this pattern:

1. Compute the reaction rate from the current state.
2. Update conversion `alpha` explicitly.
3. Build source terms from laser heating, reaction enthalpy, and radiation.
4. Solve the temperature update implicitly through a sparse linear system.

In simplified form, the code is doing something like:

`alpha^(n+1) = alpha^n + dt * R(T^n, alpha^n)`

and then

`(M rho cp + dt K) T^(n+1) = M rho cp T^n + dt * F(...)`

where:

- `M` is the mass matrix
- `K` is the conduction/boundary matrix
- `F(...)` contains the assembled source terms

So:

- `alpha` is explicit
- the thermal diffusion solve is implicit
- the overall scheme is split, not fully monolithic

## Why those choices are reasonable here

### 1. Implicit heat solve

This is the most important choice numerically.

Heat diffusion on a fine mesh is exactly the kind of problem where implicit stepping pays off. A fully explicit thermal solve would force very small timesteps just to stay stable, even before worrying about pulse resolution or reaction accuracy.

Using an implicit heat solve lets the code:

- avoid the worst diffusion stability restriction
- work on finer meshes without `dt` collapsing as badly
- reuse sparse linear algebra each step

That is why the thermal update is solved through Jacobi-preconditioned CG rather than a direct explicit formula.

### 2. Explicit reaction update

The reaction update is currently cheap and simple:

- no nonlinear coupled solve is needed
- no Newton method is needed
- no extra Jacobians or line searches are needed

This is a practical choice for a custom codebase. The downside is that it introduces splitting and explicit-timestep error, especially when temperature and conversion are changing rapidly together.

That is one reason the code also uses adaptive timestepping: the explicit part needs protection near sharp pulse heating and fast kinetics.

### 3. Explicit source treatment

The laser source is prescribed rather than solved from another PDE, so treating it explicitly is natural.

Radiation and reaction heat are also assembled as source terms instead of being fully coupled into a nonlinear solve. That keeps the implementation manageable, but it also means the scheme is not fully implicit in all physics.

## What the current scheme is good at

- Diffusion-dominated heating on a 2D mesh
- Repeated laser pulses with adaptive step control
- Moderate reaction coupling without building a full nonlinear multiphysics solver
- A pragmatic balance between speed, robustness, and code complexity

## What the current scheme gives up

Because the method is split and only partly implicit, it has some real limitations:

- reaction and temperature are not solved as one monolithic nonlinear system
- reaction heat uses a lagged rate rather than a fully converged coupled future-state rate
- radiation is not treated in a fully nonlinear implicit way
- explicit conversion updates can overshoot if `dt` is too large, which is why the code clamps `alpha` into `[0, 1]`
- accuracy still depends strongly on timestep choice even though stability is better than a fully explicit heat solver

So the current design improves stability, but it does not eliminate timestep sensitivity.

## Why not make everything explicit?

Because the heat equation would become much harder to run efficiently at useful mesh sizes.

If the thermal part were explicit:

- the mesh refinement study would get much more expensive
- the timestep would be controlled by conduction stability even in calm parts of the run
- pulse resolution and diffusion stability would both force `dt` down

That is usually a bad trade in this kind of diffusion-reaction problem.

## Why not make everything implicit?

A fully implicit method is attractive in principle, but it is a much bigger solver project.

To do that well, you would typically need:

- a coupled nonlinear residual for `T` and `alpha`
- Jacobians or Jacobian-free nonlinear iterations
- Newton damping / line search
- stronger preconditioning
- more careful convergence controls

That would likely improve robustness in stiff regimes, but it is much more engineering than the current split formulation.

## Practical rationale for this repository

The current choice is a good engineering compromise:

- implicit where the stability payoff is large: diffusion
- explicit where implementation simplicity is valuable: reaction/source updates
- adaptive `dt` to keep the explicit pieces from getting too inaccurate

That is a reasonable place to be for a custom simulation tool that needs to stay understandable and modifiable.

## If accuracy needs to improve later

The most natural upgrade path is not "make everything implicit immediately." A more realistic sequence would be:

1. tighten adaptive timestep criteria
2. benchmark timestep sensitivity more systematically
3. improve preconditioning / solver robustness
4. move toward a more formal IMEX scheme
5. only then consider a fully coupled nonlinear implicit solve

## Bottom line

In general:

- explicit = simpler, cheaper per step, more timestep-limited
- implicit = more stable, more expensive per step, more solver-heavy

Here:

- heat diffusion is implicit because it is the stiff, stability-sensitive part
- reaction/conversion is explicit because it keeps the code much simpler
- the overall method is a pragmatic semi-implicit split designed to avoid the worst cost of explicit diffusion without taking on the full complexity of a monolithic nonlinear solver
