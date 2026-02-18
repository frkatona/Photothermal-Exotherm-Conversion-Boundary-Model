"""Quick test to demonstrate updated progress bar with temperature display."""

from fem_heat_reaction import FEMHeatSimulation
import numpy as np

# Create small simulation for quick test
sim = FEMHeatSimulation(Lx=200e-6, Ly=200e-6, Nx=50, Ny=50)

# Run very short simulation to see progress bar
print("\nRunning quick test to demonstrate progress bar...")
times, Ts, Alphas = sim.run_simulation(t_final=1e-7, dt=1e-10)

print(f"\n\nTest complete!")
print(f"Final Max Temperature: {np.max(Ts[-1]):.2f} K")
