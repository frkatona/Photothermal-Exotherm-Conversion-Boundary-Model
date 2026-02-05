
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as mtri
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time

def generate_mesh(Lx, Ly, Nx, Ny):
    """
    Generates a structured triangular mesh on a rectangular domain [0, Lx] x [0, Ly].
    
    Args:
        Lx, Ly: Domain dimensions.
        Nx, Ny: Number of elements along X and Y axes.
        
    Returns:
        nodes: (NumNodes, 2) array of node coordinates.
        elements: (NumElements, 3) array of node indices for each triangle.
    """
    x = np.linspace(0, Lx, Nx + 1)
    y = np.linspace(0, Ly, Ny + 1)
    X, Y = np.meshgrid(x, y)
    nodes = np.column_stack((X.flatten(), Y.flatten()))
    
    elements = []
    for j in range(Ny):
        for i in range(Nx):
            # Node indices for the current rectangular cell
            n1 = j * (Nx + 1) + i
            n2 = n1 + 1
            n3 = (j + 1) * (Nx + 1) + i
            n4 = n3 + 1
            
            # Split rectangle into two triangles: (n1, n2, n4) and (n1, n4, n3)
            # This diagonal choice (/ or \) doesn't strictly matter for structured uniform grid,
            # but consistency is good.
            elements.append([n1, n2, n4])
            elements.append([n1, n4, n3])
            
    return nodes, np.array(elements)

def element_matrices(coords):
    """
    Computes local mass and stiffness matrices for a linear triangular element.
    
    Args:
        coords: (3, 2) array of node coordinates for the triangle [(x1,y1), (x2,y2), (x3,y3)]
        
    Returns:
        Me: (3,3) local mass matrix.
        Ke: (3,3) local stiffness matrix.
        Area: Area of the element.
    """
    # Basis functions are linear: N_i(x,y) = (a_i + b_i*x + c_i*y) / (2*Area)
    # Area = 0.5 * det([[1, x1, y1], [1, x2, y2], [1, x3, y3]])
    
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    
    Area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
    
    # Gradient of basis functions (constant over element)
    # b_1 = y2 - y3, c_1 = x3 - x2
    # b_2 = y3 - y1, c_2 = x1 - x3
    # b_3 = y1 - y2, c_3 = x2 - x1
    
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    
    # Stiffness Matrix K_ij = int(grad(Ni) . grad(Nj)) dA
    # Since gradients are constant: K_ij = (bi*bj + ci*cj) / (4*Area^2) * Area
    Ke = (np.outer(b, b) + np.outer(c, c)) / (4.0 * Area)
    
    # Mass Matrix M_ij = int(Ni * Nj) dA
    # For linear triangles, M = (Area/12) * [[2,1,1], [1,2,1], [1,1,2]]
    Me = (Area / 12.0) * np.array([[2, 1, 1],
                                   [1, 2, 1],
                                   [1, 1, 2]])
                                   
    return Me, Ke, Area

def generate_boundary_edges(Nx, Ny):
    """
    Identifies boundary edges for a structured mesh.
    
    Returns:
        edges: list of (n1, n2) tuples for boundary segments.
    """
    edges = []
    # Bottom (j=0)
    for i in range(Nx):
        n1 = i
        n2 = i + 1
        edges.append((n1, n2))
    
    # Top (j=Ny)
    offset = Ny * (Nx + 1)
    for i in range(Nx):
        n1 = offset + i
        n2 = offset + i + 1
        edges.append((n1, n2))
        
    # Left (i=0)
    for j in range(Ny):
        n1 = j * (Nx + 1)
        n2 = (j + 1) * (Nx + 1)
        edges.append((n1, n2))
        
    # Right (i=Nx)
    for j in range(Ny):
        n1 = j * (Nx + 1) + Nx
        n2 = (j + 1) * (Nx + 1) + Nx
        edges.append((n1, n2))
        
    return edges

def compute_boundary_matrices(nodes, edges, h_coeff):
    """
    Computes global boundary stiffness and load contributions for Robin BC.
    q_out = h * (T - T_inf)
    Weak form surface term: int(h * (T - T_inf) * v) dGamma
    = int(h*T*v) - int(h*T_inf*v)
    
    LHS contribution (K_bound): int(h*T*v)
    RHS contribution (F_bound): int(h*T_inf*v) -> This is a source term, so +F means heat IN.
    Actually, heat equation: C dT/dt - k lap(T) = S
    Boundary: -k grad(T).n = q_out = h(T - T_inf)
    Weak form: .. + int(q_out * v) dGamma = ...
    .. + int(h*T*v) - int(h*T_inf*v) = ...
    So LHS adds +int(h*N_i*N_j), RHS adds +int(h*T_inf*N_i)
    
    Returns:
        K_bound: (num_nodes, num_nodes) sparse matrix
        F_bound_coef: (num_nodes,) vector (representing integral of h * v). 
                      Multiply by T_inf later to get force.
    """
    num_nodes = len(nodes)
    K_b = lil_matrix((num_nodes, num_nodes))
    F_b = np.zeros(num_nodes)
    
    for n1, n2 in edges:
        # Edge length
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        L = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # 1D Boundary Mass Matrix for linear elements
        # int(Ni Nj) = L/6 * [[2,1],[1,2]]
        Me_1d = (L / 6.0) * np.array([[2, 1], [1, 2]])
        
        # 1D Load Vector
        # int(Ni) = L/2
        Fe_1d = (L / 2.0) * np.array([1, 1])
        
        # Assemble
        indices = [n1, n2]
        for i in range(2):
            F_b[indices[i]] += Fe_1d[i] * h_coeff
            for j in range(2):
                K_b[indices[i], indices[j]] += Me_1d[i, j] * h_coeff
                
    return K_b.tocsr(), F_b

def assemble_system(nodes, elements):
    """
    Assembles global Mass and Stiffness matrices.
    """
    num_nodes = len(nodes)
    M = lil_matrix((num_nodes, num_nodes))
    K = lil_matrix((num_nodes, num_nodes))
    
    for tri in elements:
        # Get coordinates of the triangle nodes
        coords = nodes[tri]
        
        # Compute local matrices
        Me, Ke, _ = element_matrices(coords)
        
        # Add to global matrices
        # We can implement this more efficiently with COO format construction eventually,
        # but lil_matrix is fine for moderate mesh sizes.
        for i in range(3):
            for j in range(3):
                row = tri[i]
                col = tri[j]
                M[row, col] += Me[i, j]
                K[row, col] += Ke[i, j]
                
    return M.tocsr(), K.tocsr()

class FEMHeatSimulation:
    def __init__(self, Lx=1.0, Ly=1.0, Nx=200, Ny=200):
        # Geometry [m]
        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.nodes, self.elements = generate_mesh(Lx, Ly, Nx, Ny)
        self.M, self.K = assemble_system(self.nodes, self.elements)
        self.num_nodes = len(self.nodes)
        
        # Physical Parameters
        self.rho : float = 10.0          # Density [kg/m^3]
        self.Cp : float = 10.0           # Heat Capacity [J/(kg K)]
        self.k : float = 0.1             # Thermal Conductivity [W/(m K)]
        self.h_coeff : float = 100.0     # Convection Coefficient [W/(m^2 K)] ("Elastic" BC)
        
        # Reaction Parameters
        # rate constant from Arrhenius equation: k = A exp(-Ea/(R T))
        # assume first order rate law for now
        self.A : float = 1     # Pre-exponential factor [1/s] (assume small for now to try to fix spatial reaction runaway)
        self.Ea : float = 1e4      # Activation Energy [J/mol]
        self.R_gas : float = 8.0    # Gas constant [J/(mol K)]
        self.dH : float = 1e-2      # Heat of reaction [J/kg] (exothermic > 0) (assume small for now)
        self.initial_T : float = 300.0 # Initial Temperature [K]
        self.T_inf : float = 300.0     # Ambient Temperature [K]
        
        # Laser Parameters (Gaussian Pulse)
        self.laser_power : float = 1e4       # Peak Power [W]
        self.laser_initial_center : np.ndarray = np.array([Lx/2, Ly/2]) 
        self.laser_sigma : float = 0.05          # beam diameter [m]
        self.laser_pulse_rate : float = 0.05      # time between pulses [s]
        self.laser_pulse_width : float = 0.05    # width of pulse [s]
        self.laser_move_step : float = 0.01       # distance between pulse centers [m]
        
        # State Arrays
        self.T = np.ones(self.num_nodes) * self.initial_T
        self.alpha = np.zeros(self.num_nodes) # Conversion fraction [0-1]
        
        # Boundary Matrices
        self.boundary_edges = generate_boundary_edges(Nx, Ny)
        self.K_bound, self.F_bound_coef = compute_boundary_matrices(self.nodes, self.boundary_edges, self.h_coeff)

    def laser_source_profile(self, t):
        """Spatial-temporal profile of the laser."""
        # 1. Pulse logic
        # Pulse index
        pulse_idx = int(t / self.laser_pulse_rate)
        
        # Time within current pulse cycle
        t_local = t % self.laser_pulse_rate
        
        # Center position: Moves up (y) by 0.1m per pulse
        center_y = self.laser_initial_center[1] + pulse_idx * self.laser_move_step
        # RETURN HERE: wrap around when its boundary is near the edge 
        current_center = np.array([self.laser_initial_center[0], center_y])
        
        # Temporal Profile: Gaussian peak at center of pulse window
        # Center the pulse in the 0.1s window, e.g. at 0.05s
        t_peak = self.laser_pulse_rate / 2.0
        # Width sigma approx pulse_width / 4 ? 
        # pulse the laser at a rate of 0.1 seconds
        temporal = np.exp(-((t_local - t_peak)**2) / (2 * (self.laser_pulse_width/2.5)**2))
        
        # Spatial Norm Calculation
        dist_sq = np.sum((self.nodes - current_center)**2, axis=1)
        spatial_raw = np.exp(-dist_sq / (2 * self.laser_sigma**2))
        
        # Normalize: Integral(spatial) = 1
        # Integral = 1^T * M * spatial_raw
        # Use simple dot product if M is matrix
        spatial_integral = np.sum(self.M.dot(spatial_raw))
        
        # Avoid zero division
        if spatial_integral < 1e-12:
            spatial_norm = spatial_raw # Should not happen unless mesh huge or sigma tiny
        else:
            spatial_norm = spatial_raw / spatial_integral
            
        return self.laser_power * temporal * spatial_norm

    def reaction_rate(self, T, alpha):
        """Computes d(alpha)/dt = (1-alpha) * A * exp(-Ea/RT)"""
        # Avoid division by zero T
        T_clipped = np.maximum(T, 1e-6)
        k_rate = self.A * np.exp(-self.Ea / (self.R_gas * T_clipped))
        return (1.0 - alpha) * k_rate
        

    def run_simulation(self, t_final=2.0, dt=0.01):
        num_steps = int(t_final / dt)
        times = np.linspace(0, t_final, num_steps)
        
        print(f"Simulation Analysis:")
        print(f"  - Grid Points: {self.num_nodes} (Nx={self.Nx}, Ny={self.Ny})")
        print(f"  - Time Steps: {num_steps}")
        print(f"  - dt: {dt:.4f} s")
        
        # Storage for animation
        results_T = [self.T.copy()]
        results_alpha = [self.alpha.copy()]
        
        # System Matrices for Implicit Euler Diffusion
        # (M + dt * theta * K) T^{n+1} = ...
        # Using theta=1.0 (Fully Implicit/Backward Euler for stability)
        # Equation: M * (T_new - T_old)/dt + K * T_new = F_source
        # (M + dt * K) T_new = M * T_old + dt * F_source
        
        # Note: M and K are global matrices. 
        # But we need to account for density/capacity/conductivity.
        # Heat Eq: rho*Cp dT/dt = k * laplacian(T) + Sources
        # Weak form: int(rho*Cp*dT/dt * v) + int(k * grad(T) * grad(v)) = int(Source * v)
        # Matrix form: rho*Cp * M * dT/dt + k * K * T = F_source_vec
        
        M_eff = self.rho * self.Cp * self.M
        K_eff = self.k * self.K
        
        # Pre-factor LHS matrix for speed
        # LHS = M_eff + dt * (K_eff + K_bound)
        LHS = M_eff + dt * (K_eff + self.K_bound)
        
        print("Starting time stepping...")
        sim_start_time = time.time()
        total_steps = len(times) - 1
        
        for step_idx, t in enumerate(times[1:]):
            
            # Progress Bar
            if step_idx % 5 == 0 or step_idx == total_steps - 1:
                progress = (step_idx + 1) / total_steps
                bar_width = 40
                filled_len = int(bar_width * progress)
                bar = '=' * filled_len + '-' * (bar_width - filled_len)
                print(f'\r[{bar}] {progress*100:.1f}%', end='', flush=True)
            
            # 1. Update Reaction (Explicit step for simplicity)
            # alpha_{n+1} = alpha_n + dt * rate(T_n, alpha_n)
            d_alpha = self.reaction_rate(self.T, self.alpha)
            self.alpha += dt * d_alpha
            self.alpha = np.clip(self.alpha, 0.0, 1.0)
            
            # 2. Compute Heat Source Terms
            # Laser Source
            S_laser_nodes = self.laser_source_profile(t)
            
            # Reaction Heat Source: Q = dH * d(alpha)/dt
            # Note: d_alpha is rate per unit time
            S_reaction_nodes = self.dH * d_alpha * self.rho # Volumetric source approx
            
            S_total_nodes = S_laser_nodes + S_reaction_nodes
            
            # Convert nodal source values to Force Vector F
            # F = int(S * N_i) dA approx M * S_nodal for linear basis
            # Consistent mass matrix multiplication
            F_source = self.M.dot(S_total_nodes)
            
            # 3. Solve Heat Equation
            # RHS = M_eff * T_n + dt * (F_source + F_bound)
            # F_bound_total = F_bound_coef * T_inf
            F_bound_total = self.F_bound_coef * self.T_inf
            
            RHS = M_eff.dot(self.T) + dt * (F_source + F_bound_total)
            
            # Solve linear system
            self.T = spsolve(LHS, RHS)
            
            results_T.append(self.T.copy())
            results_alpha.append(self.alpha.copy())
            
        elapsed = time.time() - sim_start_time
        print(f"\nSimulation complete in {elapsed:.2f} s")
        return times, results_T, results_alpha

    def animate_results(self, times, results_T, results_alpha, filename='simulation.mp4'):
        print("Creating animation...")
        anim_start_time = time.time()
        
        # Pre-compute metrics for line plots
        max_temps = [np.max(T) for T in results_T]
        conv_fractions = [np.mean(alpha) for alpha in results_alpha] 
        
        # Setup Figure: 4 subplots (Laser, Temp, Conv, Lines)
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        plt.subplots_adjust(wspace=0.3, bottom=0.15)
        
        # Grid for interpolation/contouring
        triang = mtri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.elements)
        
        # Value ranges for fixed consistency
        T_min, T_max = np.min(results_T), np.max(results_T)
        # Ensure T_max > T_min to avoid zero division in levels
        if T_max <= T_min: T_max = T_min + 1.0
        
        # Define fixed contour levels to prevent "flickering" or auto-scaling issues
        T_levels = np.linspace(T_min, T_max, 25)
        Alpha_levels = np.linspace(0, 1.0, 21)
        
        # Laser source max
        # Peak at half the pulse duration
        t_peak_sample = self.laser_pulse_rate / 2.0
        S_peak_profile = self.laser_source_profile(t_peak_sample)
        S_max_val = np.max(S_peak_profile)
        S_levels = np.linspace(0, S_max_val, 21)
        
        # 1. Laser Source
        c0 = axes[0].tricontourf(triang, self.laser_source_profile(0), levels=S_levels, cmap='hot', extend='max')
        axes[0].set_title('Laser Power Density [W/area]')
        axes[0].set_xlabel('X [m]')
        axes[0].set_ylabel('Y [m]')
        fig.colorbar(c0, ax=axes[0], label='Power [W]')
        
        # 2. Temperature Map
        c1 = axes[1].tricontourf(triang, results_T[0], levels=T_levels, cmap='inferno', extend='both')
        axes[1].set_title('Temperature Field')
        axes[1].set_xlabel('X [m]')
        axes[1].set_ylabel('Y [m]')
        fig.colorbar(c1, ax=axes[1], label='Temperature [K]')
        
        # 3. Conversion Map
        c2 = axes[2].tricontourf(triang, results_alpha[0], levels=Alpha_levels, cmap='viridis', extend='max')
        axes[2].set_title('Reaction Conversion')
        axes[2].set_xlabel('X [m]')
        axes[2].set_ylabel('Y [m]')
        fig.colorbar(c2, ax=axes[2], label='Fraction [-]')
        
        # 4. Line Plots
        ax3 = axes[3]
        ax3_right = ax3.twinx()
        
        line_T, = ax3.plot([], [], 'r-', linewidth=2, label='Max Temp')
        line_conv, = ax3_right.plot([], [], 'g-', linewidth=2, label='Avg Conversion')
        
        ax3.set_xlim(0, times[-1])
        ax3.set_ylim(T_min, T_max * 1.1)
        ax3_right.set_ylim(0, 1.05)
        
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Max Temperature [K]', color='r')
        ax3_right.set_ylabel('Avg Conversion [-]', color='g')
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        # Add legend
        lines = [line_T, line_conv]
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')

        def update(frame):
            t = times[frame]
            
            # Progress Bar for Animation
            # frame goes from 0 to len(times)-1
            total_frames = len(times)
            progress = (frame + 1) / total_frames
            bar_width = 40
            filled_len = int(bar_width * progress)
            bar = '=' * filled_len + '-' * (bar_width - filled_len)
            print(f'\rEncoding: [{bar}] {progress*100:.1f}%', end='', flush=True)

            # Update Laser
            axes[0].clear()
            axes[0].set_title(f'Power Flux (t={t:.2f} s)')
            axes[0].set_xlabel('X [m]')
            axes[0].set_ylabel('Y [m]')
            S_current = self.laser_source_profile(t)
            axes[0].tricontourf(triang, S_current, levels=S_levels, cmap='hot', extend='max')
            
            # Update Temp
            axes[1].clear()
            axes[1].set_title(f'Temperature (t={t:.2f} s)')
            axes[1].set_xlabel('X [m]')
            axes[1].set_ylabel('Y [m]')
            axes[1].tricontourf(triang, results_T[frame], levels=T_levels, cmap='inferno', extend='both')
            
            # Update Conversion
            axes[2].clear()
            axes[2].set_title(f'Conversion (t={t:.2f} s)')
            axes[2].set_xlabel('X [m]')
            axes[2].set_ylabel('Y [m]')
            axes[2].tricontourf(triang, results_alpha[frame], levels=Alpha_levels, cmap='viridis', extend='max')
            
            # Update Lines
            current_times = times[:frame+1]
            line_T.set_data(current_times, max_temps[:frame+1])
            line_conv.set_data(current_times, conv_fractions[:frame+1])
            
            return line_T, line_conv
            
        ani = animation.FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)
        
        try:
            # We can use progress_callback if available, but let's stick to our update loop print
            # standard save doesn't expose frame index easily to callback in older versions
            ani.save(filename, writer='ffmpeg')
            elapsed = time.time() - anim_start_time
            print(f"\nAnimation saved to {filename} in {elapsed:.2f} s")
        except Exception as e:
            print(f"\nCould not save video: {e}")



if __name__ == "__main__":
    # Lower resolution for faster debug run
    sim = FEMHeatSimulation(Lx=1.0, Ly=1.0, Nx=200, Ny=200)
    times, Ts, Alphas = sim.run_simulation(t_final=1.0, dt=0.01)
    
    print(f"Max T: {np.max(Ts)}")
    print(f"Max Conv: {np.max(Alphas)}")
    
    sim.animate_results(times, Ts, Alphas)
