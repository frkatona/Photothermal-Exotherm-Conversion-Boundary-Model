mod mesh;
mod physics;

use mesh::{Mesh, assemble_system, compute_boundary_matrices};
use physics::{PhysicalParams, LaserParams, reaction_rate, spmv, scale_csmat};
use ndarray::Array1;
use sprs::CsMat;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;
use std::fs::{self, File};
use std::io::{Write, BufWriter};

fn main() {
    println!("\n{}", "=".repeat(60));
    println!("FEM Heat Reaction Simulation (Rust)");
    println!("{}\n", "=".repeat(60));
    
    // Domain parameters
    let lx = 200e-6;  // 200 µm
    let ly = 200e-6;  // 200 µm
    let nx = 200;
    let ny = 200;
    
    // Simulation parameters
    let t_final = 1e-8;  // 200 µs (covers ~4 pulse cycles, visible beam movement)
    let dt = 1e-10;         // 1 ns (implicit scheme allows larger dt, still fine resolution)
    let save_interval = 100; 

    // Create simulation
    let mut sim = FEMSimulation::new(lx, ly, nx, ny);
    
    // Validate and run
    sim.validate(t_final, dt);
    let results = sim.run(t_final, dt, save_interval);
    
    // Save results
    let output_dir = "output";
    save_results(&sim.mesh, &results, output_dir);
    
    println!("\n{}", "=".repeat(60));
    println!("Simulation complete!");
    println!("{}", "=".repeat(60));
}

/// All simulation results, separated into per-frame (subsampled) and per-step (metrics).
pub struct SimResults {
    /// Time values for each saved frame
    pub frame_times: Vec<f64>,
    /// Temperature field at each saved frame
    pub frame_temps: Vec<Array1<f64>>,
    /// Conversion field at each saved frame
    pub frame_alphas: Vec<Array1<f64>>,
    /// Laser source profile at each saved frame
    pub frame_lasers: Vec<Array1<f64>>,
    /// Per-timestep time values (every step)
    pub step_times: Vec<f64>,
    /// Per-timestep max temperature
    pub step_max_temp: Vec<f64>,
    /// Per-timestep mean conversion
    pub step_avg_conv: Vec<f64>,
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
        println!("Generating mesh...");
        let mesh = Mesh::new(lx, ly, nx, ny);
        
        println!("Assembling system matrices...");
        let (m_matrix, k_matrix) = assemble_system(&mesh);
        
        let params = PhysicalParams::default();
        let laser = LaserParams::new(lx, ly);
        
        // Boundary conditions
        let edges = mesh.boundary_edges(nx, ny);
        let (k_bound, f_bound_coef) = compute_boundary_matrices(&mesh, &edges, params.h_coeff);
        
        // Initial conditions
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
    
    pub fn validate(&self, t_final: f64, dt: f64) {
        println!("\n{}", "=".repeat(60));
        println!("VALIDATION CHECKS");
        println!("{}", "=".repeat(60));
        
        // 1. Check t_final/dt is integer
        let steps_ratio = t_final / dt;
        if (steps_ratio - steps_ratio.round()).abs() > 1e-9 {
            println!("[WARNING] final_t/dt = {:.6} is non-integer!", steps_ratio);
        }
        
        // 2. Check dx/dy uniformity
        let dx = self.mesh.nodes[[1, 0]] - self.mesh.nodes[[0, 0]];
        let dy = self.mesh.nodes[[self.mesh.num_nodes / 2, 1]] - self.mesh.nodes[[0, 1]];
        let aspect_ratio = dx / dy;
        if (aspect_ratio - 1.0).abs() > 0.01 {
            println!("[WARNING] dx/dy = {:.4} is non-uniform!", aspect_ratio);
        }
        
        // 3. Check FWHM resolution
        let fwhm = 2.35 * self.laser.sigma;
        let points_per_fwhm = fwhm / dx;
        if points_per_fwhm < 12.0 {
            println!("[WARNING] FWHM resolution insufficient!");
            println!("  Points across FWHM = {:.1} (need >12)", points_per_fwhm);
        } else {
            println!("[OK] FWHM resolution: {:.1} points (>12 required)", points_per_fwhm);
        }
        
        // 4. Check Fourier stability
        let alpha_thermal = self.params.k / (self.params.rho * self.params.cp);
        let dt_critical = dx.powi(2) / (2.0 * alpha_thermal);
        let fourier_number = alpha_thermal * dt / dx.powi(2);
        
        if dt > dt_critical {
            println!("[WARNING] Fourier stability condition violated!");
            println!("  Fo = {:.4}", fourier_number);
        } else {
            println!("[OK] Fourier stability: Fo = {:.4}", fourier_number);
        }
        
        println!("\nSimulation Parameters:");
        println!("  Domain: {:.1} µm x {:.1} µm", self.mesh.nodes[[self.mesh.num_nodes-1, 0]]*1e6, 
                                                     self.mesh.nodes[[self.mesh.num_nodes-1, 1]]*1e6);
        println!("  Grid: {} nodes", self.mesh.num_nodes);
        println!("  Timesteps: {}", (t_final / dt) as usize);
        println!("{}\n", "=".repeat(60));
    }
    
    pub fn run(&mut self, t_final: f64, dt: f64, save_interval: usize) -> SimResults {
        let num_steps = (t_final / dt) as usize;
        
        // Subsampled frame storage
        let mut frame_times = vec![0.0];
        let mut frame_temps = vec![self.temperature.clone()];
        let mut frame_alphas = vec![self.alpha.clone()];
        let mut frame_lasers = vec![self.laser.source_profile(&self.mesh.nodes, 0.0, &self.m_matrix)];
        
        // Per-step metrics (lightweight)
        let mut step_times = vec![0.0];
        let mut step_max_temp = vec![self.temperature.iter().cloned().fold(f64::NEG_INFINITY, f64::max)];
        let mut step_avg_conv = vec![self.alpha.mean().unwrap_or(0.0)];
        
        println!("Starting time stepping ({} steps, saving every {}th → ~{} frames)...",
            num_steps, save_interval, num_steps / save_interval + 1);
        let pb = ProgressBar::new(num_steps as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{bar:40}] {percent:>3}% | {msg}")
            .unwrap());
        
        // Precompute effective matrices
        let m_eff = scale_csmat(&self.m_matrix, self.params.rho * self.params.cp);
        let k_eff = scale_csmat(&self.k_matrix, self.params.k);
        let k_sum = &k_eff + &self.k_bound;
        let k_dt = scale_csmat(&k_sum, dt);
        let lhs = &m_eff + &k_dt;
        
        let start_time = Instant::now();
        
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
            
            // Force vector: F_source = M * S_total
            let f_source = spmv(&self.m_matrix, &s_total);
            let f_bound = &self.f_bound_coef * self.params.t_inf;
            
            // RHS = M_eff * T_old + dt * (F_source + F_bound)
            let f_total = &f_source + &f_bound;
            let rhs = spmv(&m_eff, &self.temperature) + &(&f_total * dt);
            
            // Solve linear system: LHS * T_new = RHS
            self.temperature = cg_solve(&lhs, &rhs, &self.temperature, 1e-10, 1000);
            
            // Per-step metrics
            let max_t = self.temperature.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let avg_a = self.alpha.mean().unwrap_or(0.0);
            step_times.push(t);
            step_max_temp.push(max_t);
            step_avg_conv.push(avg_a);
            
            // Save frame if on interval
            if (step + 1) % save_interval == 0 || step == num_steps - 1 {
                frame_times.push(t);
                frame_temps.push(self.temperature.clone());
                frame_alphas.push(self.alpha.clone());
                frame_lasers.push(s_laser);
            }
            
            if step % 100 == 0 {
                pb.set_position(step as u64);
                pb.set_message(format!("Max T: {:.2} K", max_t));
            }
        }
        
        pb.finish_with_message(format!("Done! Max T: {:.2} K", step_max_temp.last().unwrap()));
        let elapsed = start_time.elapsed();
        println!("Simulation complete in {:.2} s", elapsed.as_secs_f64());
        println!("  Saved {} animation frames", frame_times.len());
        
        SimResults {
            frame_times,
            frame_temps,
            frame_alphas,
            frame_lasers,
            step_times,
            step_max_temp,
            step_avg_conv,
        }
    }
}

/// Conjugate Gradient solver for symmetric positive-definite sparse systems.
/// Solves A * x = b given initial guess x0.
fn cg_solve(
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
    
    // Early exit if already converged
    if rs_old.sqrt() < tol {
        return x;
    }
    
    for _iter in 0..max_iter {
        let ap = spmv(a, &p);
        let p_dot_ap = p.dot(&ap);
        
        if p_dot_ap.abs() < 1e-30 {
            break; // Prevent division by zero
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

/// Save all simulation results to CSV files in the output directory.
fn save_results(mesh: &Mesh, results: &SimResults, output_dir: &str) {
    println!("\nSaving results to {}/ ...", output_dir);
    fs::create_dir_all(output_dir).unwrap();
    
    // 1. Mesh nodes
    {
        let mut f = BufWriter::new(File::create(format!("{}/mesh.csv", output_dir)).unwrap());
        writeln!(f, "x,y").unwrap();
        for i in 0..mesh.num_nodes {
            writeln!(f, "{},{}", mesh.nodes[[i, 0]], mesh.nodes[[i, 1]]).unwrap();
        }
    }
    
    // 2. Elements (triangle connectivity)
    {
        let mut f = BufWriter::new(File::create(format!("{}/elements.csv", output_dir)).unwrap());
        writeln!(f, "n1,n2,n3").unwrap();
        for e in 0..mesh.num_elements {
            writeln!(f, "{},{},{}", 
                mesh.elements[[e, 0]], 
                mesh.elements[[e, 1]], 
                mesh.elements[[e, 2]]
            ).unwrap();
        }
    }
    
    // 3. Frame index → time mapping
    {
        let mut f = BufWriter::new(File::create(format!("{}/frames.csv", output_dir)).unwrap());
        writeln!(f, "frame,time").unwrap();
        for (i, t) in results.frame_times.iter().enumerate() {
            writeln!(f, "{},{}", i, t).unwrap();
        }
    }
    
    // 4. Per-frame temperature, alpha, laser
    for (i, ((temp, alpha), laser)) in results.frame_temps.iter()
        .zip(results.frame_alphas.iter())
        .zip(results.frame_lasers.iter())
        .enumerate()
    {
        // Temperature
        {
            let mut f = BufWriter::new(
                File::create(format!("{}/temperature_{:04}.csv", output_dir, i)).unwrap()
            );
            writeln!(f, "temperature").unwrap();
            for &v in temp.iter() {
                writeln!(f, "{}", v).unwrap();
            }
        }
        // Alpha
        {
            let mut f = BufWriter::new(
                File::create(format!("{}/alpha_{:04}.csv", output_dir, i)).unwrap()
            );
            writeln!(f, "alpha").unwrap();
            for &v in alpha.iter() {
                writeln!(f, "{}", v).unwrap();
            }
        }
        // Laser
        {
            let mut f = BufWriter::new(
                File::create(format!("{}/laser_{:04}.csv", output_dir, i)).unwrap()
            );
            writeln!(f, "laser").unwrap();
            for &v in laser.iter() {
                writeln!(f, "{}", v).unwrap();
            }
        }
    }
    
    // 5. Metrics (every timestep)
    {
        let mut f = BufWriter::new(File::create(format!("{}/metrics.csv", output_dir)).unwrap());
        writeln!(f, "time,max_temp,avg_conversion").unwrap();
        for i in 0..results.step_times.len() {
            writeln!(f, "{},{},{}", 
                results.step_times[i], 
                results.step_max_temp[i], 
                results.step_avg_conv[i]
            ).unwrap();
        }
    }
    
    println!("Saved {} frames + metrics to {}/", results.frame_times.len(), output_dir);
    
    // Also save a combined final-state CSV for quick inspection
    {
        let mut f = BufWriter::new(File::create(format!("{}/output_final.csv", output_dir)).unwrap());
        writeln!(f, "x,y,temperature,alpha").unwrap();
        let final_t = results.frame_temps.last().unwrap();
        let final_a = results.frame_alphas.last().unwrap();
        for i in 0..mesh.num_nodes {
            writeln!(f, "{},{},{},{}", 
                mesh.nodes[[i, 0]], 
                mesh.nodes[[i, 1]], 
                final_t[i], 
                final_a[i]
            ).unwrap();
        }
    }
}
