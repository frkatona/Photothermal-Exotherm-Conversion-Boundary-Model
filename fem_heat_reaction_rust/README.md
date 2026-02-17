# FEM Heat Reaction Simulation - Rust Implementation

High-performance finite element simulation of photothermal exothermic conversion using Rust.

## Features

- **Fast execution**: Rust's zero-cost abstractions and efficient sparse matrix operations
- **Parallel computation**: Rayon for multi-threaded operations  
- **Raster scanning**: Bidirectional laser beam path with continuous motion
- **Validation warnings**: Same stability checks as Python version
- **CSV export**: Results compatible with Python visualization tools

## Building

```bash
cd fem_heat_reaction_rust
cargo build --release
```

## Running

```bash
cargo run --release
```

## Performance

Rust implementation is typically **10-100x faster** than Python depending on grid size, due to:
- Compiled native code
- Efficient sparse matrix operations
- No interpreter overhead
- Potential for parallelization

## Output

Results are saved to `output_final.csv` containing:
- Node coordinates (x, y)
- Final temperature field
- Final conversion field

Use Python for visualization:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output_final.csv')
plt.tricontourf(df['x']*1e6, df['y']*1e6, df['temperature'])
plt.xlabel('X [µm]')
plt.ylabel('Y [µm]')
plt.colorbar(label='Temperature [K]')
plt.show()
```

## Architecture

- `mesh.rs`: Mesh generation and FEM assembly
- `physics.rs`: Material properties, laser source, reaction kinetics
- `main.rs`: Time-stepping algorithm and simulation driver

## Current Limitations

The sparse linear solver in `main.rs` uses a simplified approach. For production use, implement proper conjugate gradient iteration using `nalgebra-sparse` or similar library.

## Comparison with Python

| Feature | Python | Rust |
|---------|--------|------|
| Speed | ⭐ | ⭐⭐⭐⭐⭐ |
| Development | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Visualization | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Memory efficiency | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**Recommendation**: Use Rust for large-scale parameter sweeps or production runs. Use Python for exploration and visualization.
