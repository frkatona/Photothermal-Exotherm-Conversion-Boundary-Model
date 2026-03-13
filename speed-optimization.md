# Speed Optimization Notes

This project already has a workable CPU solver. The fastest path to better runtime is to improve the current CPU implementation first, then consider larger numerical changes, and only then look at GPU work.

## The Main Idea

The simulation is **not** easy to parallelize across time steps.

Why:

- Step `n + 1` depends on the temperature and conversion produced by step `n`.
- That means the time loop itself is mostly serial.

What *is* parallel-friendly:

- Per-node calculations like laser heating and reaction rate
- Sparse matrix-vector products
- Dot products inside the conjugate-gradient solver

That is why the first optimization pass focuses on the work **inside each step**, not on running many steps at once.

## What Was Implemented In This Pass

### 1. Parallel sparse matrix-vector multiply

The solver performs many sparse matrix-vector multiplies. Each output row is independent, so rows can be computed in parallel on the CPU.

Why this helps:

- Sparse matvec is a core building block of the solve.
- It appears both in the timestep assembly and inside each CG iteration.

Risk:

- Low. It does not change the math, only how rows are scheduled across CPU threads.

### 2. Parallel per-node source and reaction calculations

The code now parallelizes:

- laser source evaluation over all nodes
- reaction-rate evaluation over all nodes
- large vector dot products

Why this helps:

- These loops are embarrassingly parallel.
- They scale well on a multi-core CPU.

Risk:

- Low. Each node is still computed with the same formula as before.

### 3. Fewer temporary arrays in the hot loop

The original code created many intermediate arrays during each timestep and during each CG iteration.

This pass replaces several of those with in-place operations.

Why this helps:

- Fewer heap allocations
- Less memory traffic
- Better cache behavior

Risk:

- Low to medium. The math is unchanged, but in-place updates require more care when coding.

### 4. Precompute constants outside the timestep loop

Values that do not change during the run should not be rebuilt every step.

This pass moves constant terms like:

- the boundary forcing vector
- the reaction heat scale factor

outside the timestep loop.

Why this helps:

- It removes repeated work from every step.

Risk:

- Very low.

## Beginner-Friendly Ranking Of Future Work

### Best next steps

These are the best return on effort if more speed is needed.

1. Benchmark with a realistic case
   - Use a representative `nxy`, `dt`, and `t_final`.
   - Measure release builds only.
   - Optimize based on measured hotspots, not guesses.

2. Tune save frequency
   - Saving fewer frames does not reduce the physics cost much, but it does reduce memory pressure and frontend payload size.
   - If playback does not need dense output, increase `save_interval`.

3. Use a better preconditioner for CG
   - The current CG solver is simple and functional.
   - A stronger preconditioner can reduce iteration count substantially.
   - This can be one of the highest-value numerical improvements.

4. Avoid rebuilding laser normalization work if the model allows it
   - Right now the laser moves, so the source must still be recomputed.
   - But if parts of the normalization or beam footprint can be reused safely, that may save time.

### Medium-effort improvements

1. Matrix-free or stencil-style operator
   - The mesh is structured and rectangular.
   - That is a good fit for a more direct grid operator.
   - This could reduce sparse-matrix overhead significantly.

2. Structured-grid reformulation
   - The current code uses FEM on a structured triangular mesh.
   - If the problem can tolerate a finite-difference or finite-volume form, the implementation could become much faster and simpler to optimize.

3. Single-precision study
   - Consumer GPUs and many CPUs handle `f32` more efficiently than `f64`.
   - This requires accuracy validation.
   - It is worth testing only after a CPU baseline is established.

## Why GPU Work Was Not The First Step

A GPU port is not just a switch you flip.

It would likely require:

- new array and sparse-matrix data layouts
- a new linear algebra backend
- GPU kernels for sparse matvec and vector ops
- careful handling of precision, especially if `f64` is kept
- validation to ensure the solver still behaves correctly

That is a larger project than the CPU optimizations above.

## Practical Advice For Running Faster Right Now

- Use release builds for any serious run.
- Keep `nxy`, `dt`, and `t_final` physically justified rather than oversized.
- Increase `save_interval` if you do not need many saved frames.
- Watch the warning panel for under-resolved or unstable settings.

## Suggested Next Benchmark Procedure

1. Pick one realistic simulation case.
2. Run it in release mode three times.
3. Record:
   - total runtime
   - number of steps
   - mesh size
   - saved frame count
4. Change one thing at a time.
5. Keep the result if speed improves and output stays acceptable.

That process is much more reliable than chasing isolated micro-optimizations.
