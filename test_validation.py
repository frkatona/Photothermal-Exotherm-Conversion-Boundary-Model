"""Test script to demonstrate the validation warnings."""

from fem_heat_reaction import FEMHeatSimulation
import numpy as np

# Create simulation with realistic parameters
sim = FEMHeatSimulation(Lx=0.02, Ly=0.02, Nx=100, Ny=100)

# This will print all validation warnings without actually running the full simulation
# We'll just call the validation part by creating a small simulation
print("\n" + "="*60)
print("TESTING VALIDATION WITH DIFFERENT PARAMETERS")
print("="*60)

# Test 1: Current parameters (should show some warnings)
print("\nTest 1: Current parameters (100x100 grid, 2cm domain)")
times, Ts, Alphas = sim.run_simulation(t_final=0.001, dt=1e-6)
print(f"\nSimulation Results:")
print(f"  Max T: {np.max(Ts):.2f} K (ΔT = {np.max(Ts)-300:.2f} K)")
print(f"  Max Conv: {np.max(Alphas):.4f}")
