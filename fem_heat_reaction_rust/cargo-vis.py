"""
Visualization script for Rust FEM heat reaction simulation output.
Reads multi-frame CSV data from the output/ directory and produces
a 4-panel MP4 animation matching the Python reference.

Usage:
    python cargo-vis.py [output_dir]

Requires: matplotlib, pandas, numpy, ffmpeg (for MP4 encoding)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation

def load_results(output_dir='output'):
    """Load all simulation results from CSV files."""
    print(f"Loading results from {output_dir}/...")
    
    # Mesh
    mesh = pd.read_csv(os.path.join(output_dir, 'mesh.csv'))
    x = mesh['x'].values
    y = mesh['y'].values
    
    # Elements
    elements = pd.read_csv(os.path.join(output_dir, 'elements.csv'))
    triangles = elements[['n1', 'n2', 'n3']].values
    
    # Frame index
    frames = pd.read_csv(os.path.join(output_dir, 'frames.csv'))
    frame_times = frames['time'].values
    num_frames = len(frame_times)
    
    # Metrics (every timestep)
    metrics = pd.read_csv(os.path.join(output_dir, 'metrics.csv'))
    
    # Per-frame data
    temps = []
    alphas = []
    lasers = []
    for i in range(num_frames):
        t = pd.read_csv(os.path.join(output_dir, f'temperature_{i:04d}.csv'))
        a = pd.read_csv(os.path.join(output_dir, f'alpha_{i:04d}.csv'))
        l = pd.read_csv(os.path.join(output_dir, f'laser_{i:04d}.csv'))
        temps.append(t['temperature'].values)
        alphas.append(a['alpha'].values)
        lasers.append(l['laser'].values)
    
    print(f"  Loaded {num_frames} frames, {len(metrics)} metric steps")
    print(f"  Mesh: {len(x)} nodes, {len(triangles)} triangles")
    
    return x, y, triangles, frame_times, temps, alphas, lasers, metrics


def create_animation(x, y, triangles, frame_times, temps, alphas, lasers, metrics,
                     filename='simulation.mp4'):
    """Create 4-panel animation matching the Python reference."""
    print("Creating animation...")
    
    # Convert coordinates to µm for display
    x_um = x * 1e6
    y_um = y * 1e6
    
    # Triangulation for matplotlib
    triang = mtri.Triangulation(x_um, y_um, triangles)
    
    # Pre-compute metrics for line plots
    max_temps = [np.max(t) for t in temps]
    avg_convs = [np.mean(a) for a in alphas]
    
    # ---- Compute fixed color scale ranges ----
    
    # Temperature
    T_min = min(np.min(t) for t in temps)
    T_max = max(np.max(t) for t in temps)
    T_range = T_max - T_min
    if T_range < 1.0:
        T_center = (T_min + T_max) / 2
        T_min = T_center - 0.5
        T_max = T_center + 0.5
        print(f"  [Note] Small temperature variation ({T_range:.3f} K), using centered range")
    T_levels = np.linspace(T_min, T_max, 25)
    
    # Conversion
    A_min = min(np.min(a) for a in alphas)
    A_max = max(np.max(a) for a in alphas)
    if A_max - A_min < 0.01:
        A_max = max(A_min + 0.01, 0.1)
        print(f"  [Note] Minimal conversion, using 0-{A_max:.2%} range")
    A_levels = np.linspace(A_min, A_max, 21)
    
    # Laser
    S_max = max(np.max(l) for l in lasers)
    if S_max < 1e-12:
        S_max = 1.0
    S_levels = np.linspace(0, S_max, 21)
    
    print(f"\nColor Scale Ranges:")
    print(f"  Temperature: {T_min:.2f} - {T_max:.2f} K (ΔT = {T_max-T_min:.3f} K)")
    print(f"  Conversion:  {A_min:.4f} - {A_max:.4f}")
    print(f"  Laser Power: 0 - {S_max:.2e}")
    
    # ---- Setup figure ----
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    plt.subplots_adjust(wspace=0.3, bottom=0.15)
    
    # Panel 1: Laser
    c0 = axes[0].tricontourf(triang, lasers[0], levels=S_levels, cmap='hot', extend='max')
    axes[0].set_title('Laser Power Density [W/area]')
    axes[0].set_xlabel('X [µm]')
    axes[0].set_ylabel('Y [µm]')
    fig.colorbar(c0, ax=axes[0], label='Power [W]')
    
    # Panel 2: Temperature
    c1 = axes[1].tricontourf(triang, temps[0], levels=T_levels, cmap='inferno', extend='both')
    axes[1].set_title('Temperature Field')
    axes[1].set_xlabel('X [µm]')
    axes[1].set_ylabel('Y [µm]')
    fig.colorbar(c1, ax=axes[1], label='Temperature [K]')
    
    # Panel 3: Conversion
    c2 = axes[2].tricontourf(triang, alphas[0], levels=A_levels, cmap='viridis', extend='max')
    axes[2].set_title('Reaction Conversion')
    axes[2].set_xlabel('X [µm]')
    axes[2].set_ylabel('Y [µm]')
    fig.colorbar(c2, ax=axes[2], label='Fraction [-]')
    
    # Panel 4: Line plots
    ax3 = axes[3]
    ax3_right = ax3.twinx()
    
    line_T, = ax3.plot([], [], 'r-', linewidth=2, label='Max Temp')
    line_conv, = ax3_right.plot([], [], 'g-', linewidth=2, label='Avg Conversion')
    
    # Use full metrics for line plot x-axis range
    times_us = metrics['time'].values * 1e6
    ax3.set_xlim(0, times_us[-1])
    ax3.set_ylim(T_min, T_max * 1.05)
    ax3_right.set_ylim(0, 1.05)
    
    ax3.set_xlabel('Time [µs]')
    ax3.set_ylabel('Max Temperature [K]', color='r')
    ax3_right.set_ylabel('Avg Conversion [-]', color='g')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    lines = [line_T, line_conv]
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    num_frames = len(frame_times)
    
    def update(frame):
        t = frame_times[frame]
        t_us = t * 1e6
        
        # Progress
        progress = (frame + 1) / num_frames
        bar_width = 40
        filled = int(bar_width * progress)
        bar = '=' * filled + '-' * (bar_width - filled)
        print(f'\rEncoding: [{bar}] {progress*100:.1f}%', end='', flush=True)
        
        # Panel 1: Laser
        axes[0].clear()
        axes[0].set_title(f'Laser Power (t={t_us:.2f} µs)')
        axes[0].set_xlabel('X [µm]')
        axes[0].set_ylabel('Y [µm]')
        axes[0].tricontourf(triang, lasers[frame], levels=S_levels, cmap='hot', extend='max')
        
        # Panel 2: Temperature
        axes[1].clear()
        axes[1].set_title(f'Temperature (t={t_us:.2f} µs)')
        axes[1].set_xlabel('X [µm]')
        axes[1].set_ylabel('Y [µm]')
        axes[1].tricontourf(triang, temps[frame], levels=T_levels, cmap='inferno', extend='both')
        
        # Panel 3: Conversion
        axes[2].clear()
        axes[2].set_title(f'Conversion (t={t_us:.2f} µs)')
        axes[2].set_xlabel('X [µm]')
        axes[2].set_ylabel('Y [µm]')
        axes[2].tricontourf(triang, alphas[frame], levels=A_levels, cmap='viridis', extend='max')
        
        # Panel 4: Line plots - use metrics up to current frame time
        mask = metrics['time'].values <= t + 1e-15
        current_times_us = metrics['time'].values[mask] * 1e6
        line_T.set_data(current_times_us, metrics['max_temp'].values[mask])
        line_conv.set_data(current_times_us, metrics['avg_conversion'].values[mask])
        
        return line_T, line_conv
    
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
    
    try:
        ani.save(filename, writer='ffmpeg')
        print(f"\nAnimation saved to {filename}")
    except Exception as e:
        print(f"\nCould not save video: {e}")
        print("Trying to save as GIF instead...")
        try:
            gif_name = filename.replace('.mp4', '.gif')
            ani.save(gif_name, writer='pillow', fps=20)
            print(f"Animation saved to {gif_name}")
        except Exception as e2:
            print(f"Could not save GIF either: {e2}")
            print("Showing animation interactively instead...")
            plt.show()


def plot_final_state(x, y, triangles, temps, alphas, metrics):
    """Quick static plot of the final state."""
    x_um = x * 1e6
    y_um = y * 1e6
    triang = mtri.Triangulation(x_um, y_um, triangles)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(wspace=0.3, bottom=0.15)
    
    # Temperature
    c0 = axes[0].tricontourf(triang, temps[-1], levels=25, cmap='inferno')
    axes[0].set_title('Final Temperature')
    axes[0].set_xlabel('X [µm]')
    axes[0].set_ylabel('Y [µm]')
    fig.colorbar(c0, ax=axes[0], label='Temperature [K]')
    
    # Conversion
    c1 = axes[1].tricontourf(triang, alphas[-1], levels=21, cmap='viridis')
    axes[1].set_title('Final Conversion')
    axes[1].set_xlabel('X [µm]')
    axes[1].set_ylabel('Y [µm]')
    fig.colorbar(c1, ax=axes[1], label='Fraction [-]')
    
    # Metrics over time
    times_us = metrics['time'].values * 1e6
    ax2 = axes[2]
    ax2_r = ax2.twinx()
    ax2.plot(times_us, metrics['max_temp'].values, 'r-', linewidth=2, label='Max Temp')
    ax2_r.plot(times_us, metrics['avg_conversion'].values, 'g-', linewidth=2, label='Avg Conv')
    ax2.set_xlabel('Time [µs]')
    ax2.set_ylabel('Max Temperature [K]', color='r')
    ax2_r.set_ylabel('Avg Conversion [-]', color='g')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title('Time History')
    
    plt.tight_layout()
    plt.savefig('final_state.png', dpi=150)
    print("Saved final_state.png")
    plt.show()


if __name__ == '__main__':
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'output'
    
    x, y, triangles, frame_times, temps, alphas, lasers, metrics = load_results(output_dir)
    
    print(f"\nSimulation Results:")
    print(f"  Max T: {max(np.max(t) for t in temps):.2f} K "
          f"(ΔT = {max(np.max(t) for t in temps) - 300:.2f} K)")
    print(f"  Max Conv: {max(np.max(a) for a in alphas):.4f}")
    
    # Check if we should just plot final state or make animation
    if '--static' in sys.argv:
        plot_final_state(x, y, triangles, temps, alphas, metrics)
    else:
        create_animation(x, y, triangles, frame_times, temps, alphas, lasers, metrics)
        # Also save a static plot for quick reference
        plot_final_state(x, y, triangles, temps, alphas, metrics)