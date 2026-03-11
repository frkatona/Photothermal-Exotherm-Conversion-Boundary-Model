pub mod mesh;
pub mod physics;

use mesh::{Mesh, assemble_system, compute_boundary_matrices};
use physics::{PhysicalParams, LaserParams, reaction_rate, spmv, scale_csmat};
use ndarray::Array1;
use sprs::CsMat;
use serde::Serialize;

/// All simulation results, separated into per-frame (subsampled) and per-step (metrics).
#[derive(Serialize, Clone)]
pub struct SimResults {
    /// Time values for each saved frame
    pub frame_times: Vec<f64>,
    /// Temperature field at each saved frame
    pub frame_temps: Vec<Vec<f64>>,
    /// Conversion field at each saved frame
    pub frame_alphas: Vec<Vec<f64>>,
    /// Laser source profile at each saved frame
    pub frame_lasers: Vec<Vec<f64>>,
    /// Per-timestep time values (every step)
    pub step_times: Vec<f64>,
    /// Per-timestep max temperature
    pub step_max_temp: Vec<f64>,
    /// Per-timestep mean conversion
    pub step_avg_conv: Vec<f64>,
}

/// A single frame of simulation data, emitted during streaming.
#[derive(Serialize, Clone)]
pub struct SimFrame {
    pub frame_index: usize,
    pub time: f64,
    pub temperature: Vec<f64>,
    pub alpha: Vec<f64>,
    pub laser: Vec<f64>,
    pub max_temp: f64,
    pub avg_conversion: f64,
    pub progress: f64,
}

/// Mesh data for the frontend, serialized once at the start.
#[derive(Serialize, Clone)]
pub struct MeshData {
    pub nodes_x: Vec<f64>,
    pub nodes_y: Vec<f64>,
    pub elements: Vec<[usize; 3]>,
    pub num_nodes: usize,
    pub num_elements: usize,
}

/// Serializable simulation parameters for the frontend.
#[derive(Serialize, serde::Deserialize, Clone)]
pub struct SimParams {
    // Domain
    pub lx: f64,
    pub ly: f64,
    pub nx: usize,
    pub ny: usize,
    // Time
    pub t_final: f64,
    pub dt: f64,
    pub save_interval: usize,
    // Material
    pub rho: f64,
    pub cp: f64,
    pub k: f64,
    pub h_coeff: f64,
    // Reaction
    pub a_pre: f64,
    pub ea: f64,
    pub delta_h: f64,
    // Laser
    pub laser_power: f64,
    pub sigma: f64,
    pub pulse_width: f64,
    // Environment
    pub t_init: f64,
    pub t_inf: f64,
}

impl Default for SimParams {
    fn default() -> Self {
        SimParams {
            lx: 200e-6,
            ly: 200e-6,
            nx: 200,
            ny: 200,
            t_final: 1e-8,
            dt: 1e-10,
            save_interval: 100,
            rho: 1100.0,
            cp: 1500.0,
            k: 0.3,
            h_coeff: 10.0,
            a_pre: 1e8,
            ea: 8e4,
            delta_h: 5e5,
            laser_power: 1000.0,
            sigma: 50e-6 / 2.35,
            pulse_width: 10e-8,
            t_init: 300.0,
            t_inf: 300.0,
        }
    }
}

pub struct FEMSimulation {
    pub mesh: Mesh,
    pub params: PhysicalParams,
    pub laser: LaserParams,
    pub m_matrix: CsMat<f64>,
    pub k_matrix: CsMat<f64>,
    pub k_bound: CsMat<f64>,
    pub f_bound_coef: Array1<f64>,
    pub temperature: Array1<f64>,
    pub alpha: Array1<f64>,
}

impl FEMSimulation {
    pub fn new(lx: f64, ly: f64, nx: usize, ny: usize) -> Self {
        let mesh = Mesh::new(lx, ly, nx, ny);
        let (m_matrix, k_matrix) = assemble_system(&mesh);
        
        let params = PhysicalParams::default();
        let laser = LaserParams::new(lx, ly);
        
        let edges = mesh.boundary_edges(nx, ny);
        let (k_bound, f_bound_coef) = compute_boundary_matrices(&mesh, &edges, params.h_coeff);
        
        let temperature = Array1::from_elem(mesh.num_nodes, params.t_init);
        let alpha = Array1::zeros(mesh.num_nodes);
        
        FEMSimulation {
            mesh,
            params,
            laser,
            m_matrix,
            k_matrix,
            k_bound,
            f_bound_coef,
            temperature,
            alpha,
        }
    }

    pub fn new_with_params(p: &SimParams) -> Self {
        let mesh = Mesh::new(p.lx, p.ly, p.nx, p.ny);
        let (m_matrix, k_matrix) = assemble_system(&mesh);
        
        let params = PhysicalParams {
            rho: p.rho,
            cp: p.cp,
            k: p.k,
            h_coeff: p.h_coeff,
            a_pre: p.a_pre,
            ea: p.ea,
            r_gas: 8.314,
            delta_h: p.delta_h,
            t_init: p.t_init,
            t_inf: p.t_inf,
        };
        
        let mut laser = LaserParams::new(p.lx, p.ly);
        laser.power = p.laser_power;
        laser.sigma = p.sigma;
        laser.pulse_width = p.pulse_width;
        
        let edges = mesh.boundary_edges(p.nx, p.ny);
        let (k_bound, f_bound_coef) = compute_boundary_matrices(&mesh, &edges, params.h_coeff);
        
        let temperature = Array1::from_elem(mesh.num_nodes, params.t_init);
        let alpha = Array1::zeros(mesh.num_nodes);
        
        FEMSimulation {
            mesh,
            params,
            laser,
            m_matrix,
            k_matrix,
            k_bound,
            f_bound_coef,
            temperature,
            alpha,
        }
    }

    /// Get mesh data serializable for the frontend.
    pub fn mesh_data(&self) -> MeshData {
        let mut nodes_x = Vec::with_capacity(self.mesh.num_nodes);
        let mut nodes_y = Vec::with_capacity(self.mesh.num_nodes);
        for i in 0..self.mesh.num_nodes {
            nodes_x.push(self.mesh.nodes[[i, 0]]);
            nodes_y.push(self.mesh.nodes[[i, 1]]);
        }
        
        let mut elements = Vec::with_capacity(self.mesh.num_elements);
        for e in 0..self.mesh.num_elements {
            elements.push([
                self.mesh.elements[[e, 0]],
                self.mesh.elements[[e, 1]],
                self.mesh.elements[[e, 2]],
            ]);
        }
        
        MeshData {
            nodes_x,
            nodes_y,
            elements,
            num_nodes: self.mesh.num_nodes,
            num_elements: self.mesh.num_elements,
        }
    }
    
    pub fn validate(&self, t_final: f64, dt: f64) -> Vec<String> {
        let mut warnings = Vec::new();
        
        let steps_ratio = t_final / dt;
        if (steps_ratio - steps_ratio.round()).abs() > 1e-9 {
            warnings.push(format!("final_t/dt = {:.6} is non-integer!", steps_ratio));
        }
        
        let dx = self.mesh.nodes[[1, 0]] - self.mesh.nodes[[0, 0]];
        let dy = self.mesh.nodes[[self.mesh.num_nodes / 2, 1]] - self.mesh.nodes[[0, 1]];
        let aspect_ratio = dx / dy;
        if (aspect_ratio - 1.0).abs() > 0.01 {
            warnings.push(format!("dx/dy = {:.4} is non-uniform!", aspect_ratio));
        }
        
        let fwhm = 2.35 * self.laser.sigma;
        let points_per_fwhm = fwhm / dx;
        if points_per_fwhm < 12.0 {
            warnings.push(format!("FWHM resolution insufficient: {:.1} points (need >12)", points_per_fwhm));
        }
        
        let alpha_thermal = self.params.k / (self.params.rho * self.params.cp);
        let dt_critical = dx.powi(2) / (2.0 * alpha_thermal);
        let fourier_number = alpha_thermal * dt / dx.powi(2);
        
        if dt > dt_critical {
            warnings.push(format!("Fourier stability condition violated! Fo = {:.4}", fourier_number));
        }
        
        warnings
    }
    
    /// Run the simulation, calling `on_frame` for each saved frame.
    pub fn run_streaming<F>(
        &mut self,
        t_final: f64,
        dt: f64,
        save_interval: usize,
        mut on_frame: F,
    ) where
        F: FnMut(SimFrame),
    {
        let num_steps = (t_final / dt) as usize;
        
        // Emit initial frame
        let initial_laser = self.laser.source_profile(&self.mesh.nodes, 0.0, &self.m_matrix);
        on_frame(SimFrame {
            frame_index: 0,
            time: 0.0,
            temperature: self.temperature.to_vec(),
            alpha: self.alpha.to_vec(),
            laser: initial_laser.to_vec(),
            max_temp: self.temperature.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            avg_conversion: self.alpha.mean().unwrap_or(0.0),
            progress: 0.0,
        });
        
        // Precompute effective matrices
        let m_eff = scale_csmat(&self.m_matrix, self.params.rho * self.params.cp);
        let k_eff = scale_csmat(&self.k_matrix, self.params.k);
        let k_sum = &k_eff + &self.k_bound;
        let k_dt = scale_csmat(&k_sum, dt);
        let lhs = &m_eff + &k_dt;
        
        let mut frame_index = 1;
        
        for step in 0..num_steps {
            let t = (step + 1) as f64 * dt;
            
            // Update reaction (explicit)
            let d_alpha = reaction_rate(&self.temperature, &self.alpha, &self.params);
            self.alpha = &self.alpha + &(&d_alpha * dt);
            self.alpha.mapv_inplace(|a| a.clamp(0.0, 1.0));
            
            // Compute source terms
            let s_laser = self.laser.source_profile(&self.mesh.nodes, t, &self.m_matrix);
            let s_reaction = &d_alpha * (self.params.delta_h * self.params.rho);
            let s_total = &s_laser + &s_reaction;
            
            // Force vector
            let f_source = spmv(&self.m_matrix, &s_total);
            let f_bound = &self.f_bound_coef * self.params.t_inf;
            
            // RHS = M_eff * T_old + dt * (F_source + F_bound)
            let f_total = &f_source + &f_bound;
            let rhs = spmv(&m_eff, &self.temperature) + &(&f_total * dt);
            
            // Solve linear system
            self.temperature = cg_solve(&lhs, &rhs, &self.temperature, 1e-10, 1000);
            
            // Save frame if on interval
            if (step + 1) % save_interval == 0 || step == num_steps - 1 {
                let max_t = self.temperature.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let avg_a = self.alpha.mean().unwrap_or(0.0);
                let progress = (step + 1) as f64 / num_steps as f64;
                
                on_frame(SimFrame {
                    frame_index,
                    time: t,
                    temperature: self.temperature.to_vec(),
                    alpha: self.alpha.to_vec(),
                    laser: s_laser.to_vec(),
                    max_temp: max_t,
                    avg_conversion: avg_a,
                    progress,
                });
                
                frame_index += 1;
            }
        }
    }
}

/// Conjugate Gradient solver for symmetric positive-definite sparse systems.
pub fn cg_solve(
    a: &CsMat<f64>,
    b: &Array1<f64>,
    x0: &Array1<f64>,
    tol: f64,
    max_iter: usize,
) -> Array1<f64> {
    let mut x = x0.clone();
    let mut r = b - &spmv(a, &x);
    let mut p = r.clone();
    let mut rs_old = r.dot(&r);
    
    if rs_old.sqrt() < tol {
        return x;
    }
    
    for _iter in 0..max_iter {
        let ap = spmv(a, &p);
        let p_dot_ap = p.dot(&ap);
        
        if p_dot_ap.abs() < 1e-30 {
            break;
        }
        
        let alpha = rs_old / p_dot_ap;
        x = &x + &(&p * alpha);
        r = &r - &(&ap * alpha);
        
        let rs_new = r.dot(&r);
        if rs_new.sqrt() < tol {
            break;
        }
        
        let beta = rs_new / rs_old;
        p = &r + &(&p * beta);
        rs_old = rs_new;
    }
    
    x
}
