pub mod mesh;
pub mod physics;

use mesh::{Mesh, assemble_system, compute_boundary_matrices};
use physics::{PhysicalParams, LaserParams, dot_product, radiative_cooling_source, reaction_rate, spmv, scale_csmat};
use ndarray::{Array1, Array2};
use sprs::{CsMat, TriMat};
use serde::{Deserialize, Serialize};
use std::f64::consts::LN_2;
use std::time::Instant;

fn default_transformed_absorption_coeff() -> f64 {
    1e5
}

fn default_transformed_emissivity() -> f64 {
    0.85
}

fn default_adaptive_time_stepping() -> bool {
    true
}

fn default_adaptive_subdomains() -> bool {
    true
}

fn default_subdomain_edge_temp_trigger() -> bool {
    true
}

fn default_subdomain_edge_temp_rise() -> f64 {
    3.0
}

fn default_gaussian_spatial_profile() -> bool {
    true
}

fn default_gaussian_temporal_profile() -> bool {
    true
}

/// All simulation results, separated into per-frame (subsampled) and per-step (metrics).
#[derive(Serialize, Deserialize, Clone)]
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
#[derive(Serialize, Deserialize, Clone)]
pub struct SimActiveBounds {
    pub i_min: usize,
    pub i_max: usize,
    pub j_min: usize,
    pub j_max: usize,
}

/// A single frame of simulation data, emitted during streaming.
#[derive(Serialize, Deserialize, Clone)]
pub struct SimFrame {
    pub frame_index: usize,
    pub time: f64,
    pub temperature: Vec<f64>,
    pub alpha: Vec<f64>,
    pub laser: Vec<f64>,
    pub active_bounds: Option<SimActiveBounds>,
    pub max_temp: f64,
    pub avg_conversion: f64,
    pub progress: f64,
}

/// Lightweight progress metadata that can be emitted without frame payloads.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SimProgress {
    pub step: usize,
    pub num_steps: usize,
    pub time: f64,
    pub progress: f64,
    pub max_temp: f64,
    pub active_nodes: usize,
    pub total_nodes: usize,
}

/// Saved-frame diagnostics for time-history plotting.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct SimTimeSeriesPoint {
    pub time: f64,
    pub max_temp: f64,
    pub avg_conversion: f64,
    pub laser_power: f64,
    pub enthalpy_power: f64,
    pub convection_power: f64,
    pub radiation_power: f64,
}

/// Mesh data for the frontend, serialized once at the start.
#[derive(Serialize, Deserialize, Clone)]
pub struct MeshData {
    pub nodes_x: Vec<f64>,
    pub nodes_y: Vec<f64>,
    pub elements: Vec<[usize; 3]>,
    pub num_nodes: usize,
    pub num_elements: usize,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct SolverStats {
    pub preconditioner: String,
    pub solve_calls: usize,
    pub total_iterations: usize,
    pub max_iterations: usize,
    pub avg_iterations: f64,
    pub total_solve_secs: f64,
    pub avg_solve_secs: f64,
    pub max_residual_norm: f64,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct RunSummary {
    pub num_steps: usize,
    pub completed_steps: usize,
    pub cancelled: bool,
    pub solver: SolverStats,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ActiveBounds {
    i_min: usize,
    i_max: usize,
    j_min: usize,
    j_max: usize,
}

impl ActiveBounds {
    fn node_count(&self) -> usize {
        (self.i_max - self.i_min + 1) * (self.j_max - self.j_min + 1)
    }

    fn is_full(&self, nx: usize, ny: usize) -> bool {
        self.i_min == 0 && self.i_max == nx && self.j_min == 0 && self.j_max == ny
    }

    fn merge(self, other: ActiveBounds) -> ActiveBounds {
        ActiveBounds {
            i_min: self.i_min.min(other.i_min),
            i_max: self.i_max.max(other.i_max),
            j_min: self.j_min.min(other.j_min),
            j_max: self.j_max.max(other.j_max),
        }
    }

    fn to_public(self) -> SimActiveBounds {
        SimActiveBounds {
            i_min: self.i_min,
            i_max: self.i_max,
            j_min: self.j_min,
            j_max: self.j_max,
        }
    }
}

struct ActiveSubdomain {
    bounds: ActiveBounds,
    node_ids: Vec<usize>,
    nodes: Array2<f64>,
    m_matrix: CsMat<f64>,
    m_eff: CsMat<f64>,
    k_bound: CsMat<f64>,
    k_sum: CsMat<f64>,
    f_bound: Array1<f64>,
    interface_coupling: Array1<f64>,
}

pub struct CgSolveReport {
    pub solution: Array1<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub residual_norm: f64,
    pub solve_secs: f64,
}

/// Serializable simulation parameters for the frontend.
#[derive(Serialize, serde::Deserialize, Clone)]
#[serde(default)]
pub struct SimParams {
    // Domain (symmetric)
    pub lxy: f64,
    pub nxy: usize,
    // Time
    pub t_final: f64,
    pub dt: f64,
    #[serde(default = "default_adaptive_time_stepping")]
    pub adaptive_time_stepping: bool,
    #[serde(default = "default_adaptive_subdomains")]
    pub adaptive_subdomains: bool,
    #[serde(default = "default_subdomain_edge_temp_trigger")]
    pub subdomain_edge_temp_trigger: bool,
    #[serde(default = "default_subdomain_edge_temp_rise")]
    pub subdomain_edge_temp_rise: f64,
    pub save_interval: usize,
    // Material
    pub rho: f64,
    pub cp: f64,
    pub k: f64,
    pub h_coeff: f64,
    pub emissivity: f64,
    #[serde(default = "default_transformed_emissivity")]
    pub emissivity_transformed: f64,
    // Reaction
    pub a_pre: f64,
    pub ea: f64,
    pub delta_h: f64,
    // Laser
    pub pulse_energy: f64,
    pub sigma: f64,
    pub pulse_width: f64,
    #[serde(default = "default_gaussian_spatial_profile")]
    pub gaussian_spatial: bool,
    #[serde(default = "default_gaussian_temporal_profile")]
    pub gaussian_temporal: bool,
    pub pulse_rate: f64,
    pub scan_speed: f64,
    pub line_spacing: f64,
    pub scan_margin: f64,
    pub film_thickness: f64,
    pub absorption_coeff: f64,
    #[serde(default = "default_transformed_absorption_coeff")]
    pub absorption_coeff_transformed: f64,
    // Environment
    pub t_init: f64,
    pub t_inf: f64,
}

impl Default for SimParams {
    fn default() -> Self {
        SimParams {
            lxy: 200e-6,
            nxy: 200,
            t_final: 1e-8,
            dt: 1e-10,
            adaptive_time_stepping: true,
            adaptive_subdomains: true,
            subdomain_edge_temp_trigger: true,
            subdomain_edge_temp_rise: 3.0,
            save_interval: 100,
            rho: 1100.0,
            cp: 1500.0,
            k: 0.3,
            h_coeff: 10.0,
            emissivity: 0.85,
            emissivity_transformed: 0.85,
            a_pre: 1e8,
            ea: 8e4,
            delta_h: 5e5,
            pulse_energy: 1e-4,
            sigma: 50e-6 / 2.35,
            pulse_width: 10e-8,
            gaussian_spatial: true,
            gaussian_temporal: true,
            pulse_rate: 20e3,
            scan_speed: 0.1,
            line_spacing: 40e-6,
            scan_margin: 20e-6,
            film_thickness: 10e-6,
            absorption_coeff: 1e5,
            absorption_coeff_transformed: 1e5,
            t_init: 300.0,
            t_inf: 300.0,
        }
    }
}

pub struct FEMSimulation {
    pub mesh: Mesh,
    pub nx: usize,
    pub ny: usize,
    pub params: PhysicalParams,
    pub adaptive_time_stepping: bool,
    pub adaptive_subdomains: bool,
    pub subdomain_edge_temp_trigger: bool,
    pub subdomain_edge_temp_rise: f64,
    pub laser: LaserParams,
    pub m_matrix: CsMat<f64>,
    pub k_matrix: CsMat<f64>,
    pub k_bound: CsMat<f64>,
    pub f_bound_coef: Array1<f64>,
    pub temperature: Array1<f64>,
    pub alpha: Array1<f64>,
}

impl FEMSimulation {
    fn integrated_volume_power_on(&self, m_matrix: &CsMat<f64>, source: &Array1<f64>) -> f64 {
        if self.laser.film_thickness <= 0.0 {
            return 0.0;
        }
        spmv(m_matrix, source).sum() * self.laser.film_thickness
    }

    fn estimated_step_count(&self, t_final: f64, dt: f64) -> usize {
        if !(t_final > 0.0 && dt > 0.0) {
            return 0;
        }
        ((t_final / dt).ceil() as usize).max(1)
    }

    fn dx(&self) -> f64 {
        self.laser.lx / (self.nx.max(1) as f64)
    }

    fn dy(&self) -> f64 {
        self.laser.ly / (self.ny.max(1) as f64)
    }

    fn should_force_full_domain(&self, t_final: f64) -> bool {
        if self.laser.pulse_energy <= 0.0 {
            return true;
        }

        let background_reaction = if self.params.a_pre > 0.0 {
            let t_ref = self.params.t_init.max(1e-6);
            self.params.a_pre * (-self.params.ea / (self.params.r_gas * t_ref)).exp() * t_final
        } else {
            0.0
        };
        if background_reaction > 1e-4 {
            return true;
        }

        let radiative_or_convective =
            self.params.h_coeff.abs() > 0.0
                || self.params.emissivity.abs() > 0.0
                || self.params.emissivity_transformed.abs() > 0.0;
        if radiative_or_convective && (self.params.t_init - self.params.t_inf).abs() > 1e-9 {
            return true;
        }

        false
    }

    fn coord_bounds_to_active_bounds(
        &self,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
    ) -> ActiveBounds {
        let dx = self.dx();
        let dy = self.dy();

        let i_min = ((x_min / dx).floor() as isize).clamp(0, self.nx as isize) as usize;
        let i_max = ((x_max / dx).ceil() as isize).clamp(0, self.nx as isize) as usize;
        let j_min = ((y_min / dy).floor() as isize).clamp(0, self.ny as isize) as usize;
        let j_max = ((y_max / dy).ceil() as isize).clamp(0, self.ny as isize) as usize;

        ActiveBounds {
            i_min: i_min.min(i_max),
            i_max: i_max.max(i_min),
            j_min: j_min.min(j_max),
            j_max: j_max.max(j_min),
        }
    }

    fn clamp_or_promote_bounds(&self, bounds: ActiveBounds) -> ActiveBounds {
        let full_nodes = (self.nx + 1) * (self.ny + 1);
        if bounds.node_count() * 100 >= full_nodes * 85 {
            ActiveBounds {
                i_min: 0,
                i_max: self.nx,
                j_min: 0,
                j_max: self.ny,
            }
        } else {
            bounds
        }
    }

    fn desired_active_bounds(
        &self,
        time: f64,
        t_final: f64,
        dt_max: f64,
        current_bounds: Option<ActiveBounds>,
        boundary_hot: bool,
    ) -> ActiveBounds {
        if !self.adaptive_subdomains {
            return ActiveBounds {
                i_min: 0,
                i_max: self.nx,
                j_min: 0,
                j_max: self.ny,
            };
        }

        if self.should_force_full_domain(t_final) {
            return ActiveBounds {
                i_min: 0,
                i_max: self.nx,
                j_min: 0,
                j_max: self.ny,
            };
        }

        let remaining = (t_final - time).max(0.0);
        let nominal_lookahead = (20.0 * dt_max)
            .max(2.0 * self.laser.pulse_width)
            .max(if self.laser.pulse_period.is_finite() {
                2.0 * self.laser.pulse_period
            } else {
                0.0
            });
        let lookahead = nominal_lookahead.min(remaining);

        let diffusivity = if self.params.rho > 0.0 && self.params.cp > 0.0 {
            (self.params.k / (self.params.rho * self.params.cp)).max(0.0)
        } else {
            0.0
        };
        let diffusion_halo = 4.0 * (diffusivity * nominal_lookahead.max(dt_max)).sqrt();
        let beam_halo = self.laser.support_radius();
        let base_halo = beam_halo + diffusion_halo + 3.0 * self.dx().max(self.dy());

        let (x_now, y_now) = self.laser.beam_position(time);
        let (x_future, y_future) = self.laser.beam_position((time + lookahead).min(t_final));

        let mut desired = self.coord_bounds_to_active_bounds(
            x_now.min(x_future) - base_halo,
            x_now.max(x_future) + base_halo,
            y_now.min(y_future) - base_halo,
            y_now.max(y_future) + base_halo,
        );

        if let Some(current) = current_bounds {
            desired = desired.merge(current);
            if boundary_hot && !current.is_full(self.nx, self.ny) {
                let growth_cells = ((base_halo / self.dx().max(self.dy())).ceil() as usize).max(2);
                desired = desired.merge(ActiveBounds {
                    i_min: current.i_min.saturating_sub(growth_cells),
                    i_max: (current.i_max + growth_cells).min(self.nx),
                    j_min: current.j_min.saturating_sub(growth_cells),
                    j_max: (current.j_max + growth_cells).min(self.ny),
                });
            }
        }

        self.clamp_or_promote_bounds(desired)
    }

    fn build_active_subdomain(&self, bounds: ActiveBounds) -> ActiveSubdomain {
        let mut node_ids = Vec::with_capacity(bounds.node_count());
        let mut global_to_local = vec![usize::MAX; self.mesh.num_nodes];

        for j in bounds.j_min..=bounds.j_max {
            for i in bounds.i_min..=bounds.i_max {
                let global = j * (self.nx + 1) + i;
                global_to_local[global] = node_ids.len();
                node_ids.push(global);
            }
        }

        let local_n = node_ids.len();
        let mut nodes = Array2::zeros((local_n, 2));
        for (local_idx, &global_idx) in node_ids.iter().enumerate() {
            nodes[[local_idx, 0]] = self.mesh.nodes[[global_idx, 0]];
            nodes[[local_idx, 1]] = self.mesh.nodes[[global_idx, 1]];
        }

        let mut m_tri = TriMat::new((local_n, local_n));
        let mut k_tri = TriMat::new((local_n, local_n));
        let mut k_bound_tri = TriMat::new((local_n, local_n));
        let mut f_bound_coef = Array1::<f64>::zeros(local_n);
        let mut inactive_k_sum = Array1::<f64>::zeros(local_n);
        let mut inactive_k_bound_sum = Array1::<f64>::zeros(local_n);

        for (local_row, &global_row) in node_ids.iter().enumerate() {
            if let Some(row) = self.m_matrix.outer_view(global_row) {
                for (global_col, &value) in row.iter() {
                    let local_col = global_to_local[global_col];
                    if local_col != usize::MAX {
                        m_tri.add_triplet(local_row, local_col, value);
                    }
                }
            }

            if let Some(row) = self.k_matrix.outer_view(global_row) {
                for (global_col, &value) in row.iter() {
                    let local_col = global_to_local[global_col];
                    if local_col != usize::MAX {
                        k_tri.add_triplet(local_row, local_col, value);
                    } else {
                        inactive_k_sum[local_row] += value;
                    }
                }
            }

            if let Some(row) = self.k_bound.outer_view(global_row) {
                for (global_col, &value) in row.iter() {
                    let local_col = global_to_local[global_col];
                    if local_col != usize::MAX {
                        k_bound_tri.add_triplet(local_row, local_col, value);
                    } else {
                        inactive_k_bound_sum[local_row] += value;
                    }
                }
            }

            f_bound_coef[local_row] = self.f_bound_coef[global_row];
        }

        let m_matrix = m_tri.to_csr();
        let k_matrix = k_tri.to_csr();
        let k_bound = k_bound_tri.to_csr();
        let m_eff = scale_csmat(&m_matrix, self.params.rho * self.params.cp);
        let k_sum = &scale_csmat(&k_matrix, self.params.k) + &k_bound;
        let f_bound = &f_bound_coef * self.params.t_inf;
        let mut interface_coupling = inactive_k_sum;
        interface_coupling *= self.params.k;
        interface_coupling += &inactive_k_bound_sum;
        interface_coupling *= self.params.t_init;

        ActiveSubdomain {
            bounds,
            node_ids,
            nodes,
            m_matrix,
            m_eff,
            k_bound,
            k_sum,
            f_bound,
            interface_coupling,
        }
    }

    fn gather_active_field(&self, active: &ActiveSubdomain, field: &Array1<f64>) -> Array1<f64> {
        Array1::from_iter(active.node_ids.iter().map(|&global| field[global]))
    }

    fn scatter_active_field(
        field: &mut Array1<f64>,
        active: &ActiveSubdomain,
        local_values: &Array1<f64>,
    ) {
        for (local_idx, &global_idx) in active.node_ids.iter().enumerate() {
            field[global_idx] = local_values[local_idx];
        }
    }

    fn full_field_from_active(
        &self,
        active: &ActiveSubdomain,
        local_values: &Array1<f64>,
    ) -> Vec<f64> {
        let mut values = vec![0.0; self.mesh.num_nodes];
        for (local_idx, &global_idx) in active.node_ids.iter().enumerate() {
            values[global_idx] = local_values[local_idx];
        }
        values
    }

    fn active_boundary_is_hot(
        &self,
        active: &ActiveSubdomain,
        local_temp: &Array1<f64>,
        local_alpha: &Array1<f64>,
    ) -> bool {
        let legacy_temp_tol = 0.5;
        let temp_tol = self.subdomain_edge_temp_rise.max(0.0);
        let alpha_tol = 1e-4;
        let nx_local = active.bounds.i_max - active.bounds.i_min + 1;
        let ny_local = active.bounds.j_max - active.bounds.j_min + 1;

        for j_local in 0..ny_local {
            for i_local in 0..nx_local {
                let on_edge = i_local == 0
                    || j_local == 0
                    || i_local + 1 == nx_local
                    || j_local + 1 == ny_local;
                if !on_edge {
                    continue;
                }

                let local_idx = j_local * nx_local + i_local;
                let temperature_triggers_growth = if self.subdomain_edge_temp_trigger {
                    local_temp[local_idx] - self.params.t_init > temp_tol
                } else {
                    (local_temp[local_idx] - self.params.t_init).abs() > legacy_temp_tol
                };
                if temperature_triggers_growth || local_alpha[local_idx] > alpha_tol {
                    return true;
                }
            }
        }

        false
    }

    fn adaptive_step_size(
        &self,
        time: f64,
        t_final: f64,
        dt_max: f64,
        prev_dt: f64,
        d_alpha: &Array1<f64>,
        s_laser_current: &Array1<f64>,
        s_radiation_current: &Array1<f64>,
    ) -> f64 {
        let remaining = (t_final - time).max(0.0);
        if remaining <= 0.0 {
            return 0.0;
        }

        let dt_cap = dt_max.min(remaining).max(f64::EPSILON);
        if !self.adaptive_time_stepping {
            return dt_cap;
        }

        let mut dt = dt_cap;
        let pulse_resolution_dt = if self.laser.gaussian_temporal && self.laser.pulse_width > 0.0 {
            (self.laser.pulse_width / 6.0).max(f64::EPSILON)
        } else {
            dt_cap
        };
        let repetition_period = self
            .laser
            .pulse_period
            .max(self.laser.pulse_width)
            .max(f64::EPSILON);
        let pulse_sigma = if self.laser.gaussian_temporal && self.laser.pulse_width > 0.0 {
            (self.laser.pulse_width / (2.0 * (2.0 * LN_2).sqrt())).max(f64::EPSILON)
        } else {
            f64::EPSILON
        };
        let t_local = time.rem_euclid(repetition_period);
        let t_peak = if self.laser.gaussian_temporal {
            0.5 * self.laser.pulse_width.min(repetition_period)
        } else {
            0.0
        };
        if self.laser.gaussian_temporal && (t_local - t_peak).abs() <= 3.0 * pulse_sigma {
            dt = dt.min(pulse_resolution_dt);
        } else if !self.laser.gaussian_temporal && self.laser.pulse_period.is_finite() {
            dt = dt.min(self.laser.pulse_period.max(f64::EPSILON));
        }

        let max_alpha_rate = d_alpha.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        if max_alpha_rate > 1e-12 {
            dt = dt.min(0.02 / max_alpha_rate);
        }

        let max_laser_source = s_laser_current
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let max_radiation_source = s_radiation_current
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let reaction_source_scale = self.params.delta_h * self.params.rho * max_alpha_rate;
        let rho_cp = (self.params.rho * self.params.cp).max(f64::EPSILON);
        let max_temp_rate = (max_laser_source + max_radiation_source + reaction_source_scale) / rho_cp;
        if max_temp_rate > 1e-9 {
            dt = dt.min(2.0 / max_temp_rate);
        }

        let mut min_dt = (dt_max * 1e-4).max(f64::EPSILON);
        if self.laser.gaussian_temporal && self.laser.pulse_width > 0.0 {
            min_dt = min_dt.min((self.laser.pulse_width / 80.0).max(f64::EPSILON));
        } else if !self.laser.gaussian_temporal && self.laser.pulse_period.is_finite() {
            min_dt = min_dt.min(self.laser.pulse_period.max(f64::EPSILON));
        }

        if prev_dt > 0.0 {
            dt = dt.min(prev_dt * 1.6);
        }

        dt.clamp(min_dt.min(remaining), remaining.max(f64::EPSILON))
    }

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
            nx,
            ny,
            params,
            adaptive_time_stepping: true,
            adaptive_subdomains: true,
            subdomain_edge_temp_trigger: true,
            subdomain_edge_temp_rise: 3.0,
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
        let mesh = Mesh::new(p.lxy, p.lxy, p.nxy, p.nxy);
        let (m_matrix, k_matrix) = assemble_system(&mesh);
        
        let params = PhysicalParams {
            rho: p.rho,
            cp: p.cp,
            k: p.k,
            h_coeff: p.h_coeff,
            emissivity: p.emissivity,
            emissivity_transformed: p.emissivity_transformed,
            a_pre: p.a_pre,
            ea: p.ea,
            r_gas: 8.314,
            delta_h: p.delta_h,
            t_init: p.t_init,
            t_inf: p.t_inf,
        };
        
        // Compute peak power from pulse energy
        let peak_power = if p.pulse_width > 0.0 {
            p.pulse_energy / p.pulse_width
        } else {
            0.0
        };

        let mut laser = LaserParams::new(p.lxy, p.lxy);
        laser.power = peak_power;
        laser.pulse_energy = p.pulse_energy;
        laser.sigma = p.sigma;
        laser.pulse_width = p.pulse_width;
        laser.gaussian_spatial = p.gaussian_spatial;
        laser.gaussian_temporal = p.gaussian_temporal;
        laser.pulse_period = if p.pulse_rate > 0.0 {
            1.0 / p.pulse_rate
        } else {
            f64::INFINITY
        };
        laser.scan_speed = p.scan_speed;
        laser.line_spacing = p.line_spacing;
        laser.scan_margin = p.scan_margin;
        laser.film_thickness = p.film_thickness;
        laser.absorption_coeff = p.absorption_coeff;
        laser.absorption_coeff_transformed = p.absorption_coeff_transformed;
        
        let edges = mesh.boundary_edges(p.nxy, p.nxy);
        let (k_bound, f_bound_coef) = compute_boundary_matrices(&mesh, &edges, params.h_coeff);
        
        let temperature = Array1::from_elem(mesh.num_nodes, params.t_init);
        let alpha = Array1::zeros(mesh.num_nodes);
        
        FEMSimulation {
            mesh,
            nx: p.nxy,
            ny: p.nxy,
            params,
            adaptive_time_stepping: p.adaptive_time_stepping,
            adaptive_subdomains: p.adaptive_subdomains,
            subdomain_edge_temp_trigger: p.subdomain_edge_temp_trigger,
            subdomain_edge_temp_rise: p.subdomain_edge_temp_rise,
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
        
        let beam_width = if self.laser.gaussian_spatial {
            2.35 * self.laser.sigma
        } else {
            2.0 * (2.0 * LN_2).sqrt() * self.laser.sigma
        };
        let points_per_beam_width = beam_width / dx;
        if points_per_beam_width < 12.0 {
            warnings.push(format!(
                "{} resolution insufficient: {:.1} points (need >12)",
                if self.laser.gaussian_spatial { "beam FWHM" } else { "top-hat diameter" },
                points_per_beam_width
            ));
        }

        if !self.laser.gaussian_temporal
            && self.laser.pulse_period.is_finite()
            && dt > self.laser.pulse_period
        {
            warnings.push(
                "dt exceeds the pulse period while temporal spreading is disabled, so multiple pulses can collapse into one step"
                    .to_string(),
            );
        }

        if self.laser.film_thickness <= 0.0 {
            warnings.push("film thickness must be > 0 to generate volumetric laser heating and emissive cooling".to_string());
        }

        if self.laser.line_spacing <= 0.0 {
            warnings.push("line spacing must be > 0 for raster scanning".to_string());
        }

        if self.laser.scan_speed <= 0.0 {
            warnings.push("scan speed must be > 0 for raster scanning".to_string());
        }

        if self.laser.scan_margin * 2.0 >= self.laser.lx.min(self.laser.ly) {
            warnings.push("scan margin leaves no interior area for the raster path".to_string());
        }

        if self.subdomain_edge_temp_rise < 0.0 {
            warnings.push("subdomain edge temperature-rise trigger must be >= 0".to_string());
        }

        if self.laser.absorption_coeff < 0.0 {
            warnings.push("base absorption coefficient must be >= 0".to_string());
        }

        if self.laser.absorption_coeff_transformed < 0.0 {
            warnings.push("transformed-material absorption coefficient must be >= 0".to_string());
        }

        if !(0.0..=1.0).contains(&self.params.emissivity) {
            warnings.push("base emissivity must lie between 0 and 1".to_string());
        }

        if !(0.0..=1.0).contains(&self.params.emissivity_transformed) {
            warnings.push("transformed-material emissivity must lie between 0 and 1".to_string());
        }

        let absorbed_fraction_base = self.laser.absorbed_fraction_untransformed();
        let absorbed_fraction_transformed = self.laser.absorbed_fraction_transformed();
        if absorbed_fraction_base < 1e-6 && absorbed_fraction_transformed < 1e-6 {
            warnings.push("laser absorption is effectively zero across both untransformed and transformed material states".to_string());
        } else {
            if absorbed_fraction_base < 1e-6 {
                warnings.push("untransformed material absorbs effectively zero laser energy".to_string());
            }
            if absorbed_fraction_transformed < 1e-6 {
                warnings.push("transformed material absorbs effectively zero laser energy".to_string());
            }
        }
        
        let alpha_thermal = self.params.k / (self.params.rho * self.params.cp);
        let dt_critical = dx.powi(2) / (2.0 * alpha_thermal);
        let fourier_number = alpha_thermal * dt / dx.powi(2);
        
        if dt > dt_critical {
            if self.adaptive_time_stepping {
                warnings.push(format!(
                    "maximum dt gives Fo = {:.4}; adaptive stepping is enabled, but a smaller max dt may still improve accuracy",
                    fourier_number
                ));
            } else {
                warnings.push(format!("Fourier stability condition violated! Fo = {:.4}", fourier_number));
            }
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
    ) -> RunSummary
    where
        F: FnMut(SimFrame),
    {
        self.run_streaming_with_progress(
            t_final,
            dt,
            save_interval,
            |frame| on_frame(frame),
            |_| {},
        )
    }

    /// Run the simulation, calling `on_frame` for saved frames and `on_progress`
    /// for throttled scalar progress updates.
    pub fn run_streaming_with_progress<F, P>(
        &mut self,
        t_final: f64,
        dt: f64,
        save_interval: usize,
        mut on_frame: F,
        mut on_progress: P,
    ) -> RunSummary
    where
        F: FnMut(SimFrame),
        P: FnMut(SimProgress),
    {
        self.run_streaming_with_control(
            t_final,
            dt,
            save_interval,
            |frame| on_frame(frame),
            |progress| on_progress(progress),
            |_| {},
            |_, _| true,
        )
    }

    /// Run the simulation, calling `on_frame` for saved frames, `on_progress`
    /// for throttled scalar progress updates, `on_series_point` for saved-frame
    /// time-history diagnostics, and `control` once per timestep.
    pub fn run_streaming_with_control<F, P, S, C>(
        &mut self,
        t_final: f64,
        dt: f64,
        save_interval: usize,
        mut on_frame: F,
        mut on_progress: P,
        mut on_series_point: S,
        mut control: C,
    ) -> RunSummary
    where
        F: FnMut(SimFrame),
        P: FnMut(SimProgress),
        S: FnMut(SimTimeSeriesPoint),
        C: FnMut(usize, f64) -> bool,
    {
        let estimated_num_steps = self.estimated_step_count(t_final, dt);
        let progress_interval = (estimated_num_steps / 400).max(1);

        let mut active = self.build_active_subdomain(self.desired_active_bounds(0.0, t_final, dt, None, false));
        let initial_temp_local = self.gather_active_field(&active, &self.temperature);
        let initial_alpha_local = self.gather_active_field(&active, &self.alpha);
        let initial_laser_local = self
            .laser
            .source_profile(&active.nodes, &initial_alpha_local, 0.0, 0.0, &active.m_matrix);
        let initial_laser = self.full_field_from_active(&active, &initial_laser_local);

        on_frame(SimFrame {
            frame_index: 0,
            time: 0.0,
            temperature: self.temperature.to_vec(),
            alpha: self.alpha.to_vec(),
            laser: initial_laser,
            active_bounds: Some(active.bounds.to_public()),
            max_temp: self.temperature.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            avg_conversion: self.alpha.mean().unwrap_or(0.0),
            progress: 0.0,
        });
        on_progress(SimProgress {
            step: 0,
            num_steps: estimated_num_steps,
            time: 0.0,
            progress: if t_final <= 0.0 { 1.0 } else { 0.0 },
            max_temp: self.temperature.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            active_nodes: active.node_ids.len(),
            total_nodes: self.mesh.num_nodes,
        });

        if !(t_final > 0.0 && dt > 0.0) {
            return RunSummary {
                num_steps: 0,
                completed_steps: 0,
                cancelled: false,
                solver: SolverStats {
                    preconditioner: "Jacobi PCG".to_string(),
                    ..SolverStats::default()
                },
            };
        }

        let reaction_heat_scale = self.params.delta_h * self.params.rho;
        let initial_reaction_rate = reaction_rate(&initial_temp_local, &initial_alpha_local, &self.params);
        let mut initial_enthalpy_source = initial_reaction_rate.clone();
        initial_enthalpy_source *= reaction_heat_scale;
        let initial_radiation = radiative_cooling_source(
            &initial_temp_local,
            &initial_alpha_local,
            &self.params,
            self.laser.film_thickness,
        );
        let initial_convection_power =
            (&active.f_bound - &spmv(&active.k_bound, &initial_temp_local)).sum();
        on_series_point(SimTimeSeriesPoint {
            time: 0.0,
            max_temp: self.temperature.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            avg_conversion: self.alpha.mean().unwrap_or(0.0),
            laser_power: self.integrated_volume_power_on(&active.m_matrix, &initial_laser_local),
            enthalpy_power: self.integrated_volume_power_on(&active.m_matrix, &initial_enthalpy_source),
            convection_power: initial_convection_power,
            radiation_power: self.integrated_volume_power_on(&active.m_matrix, &initial_radiation),
        });

        let mut frame_index = 1;
        let mut cancelled = false;
        let mut completed_steps = 0_usize;
        let mut total_iterations = 0_usize;
        let mut max_iterations = 0_usize;
        let mut total_solve_secs = 0.0_f64;
        let mut max_residual_norm = 0.0_f64;
        let mut current_time = 0.0_f64;
        let mut previous_dt = dt.max(f64::EPSILON);
        let mut boundary_hot = false;

        while current_time < t_final - f64::EPSILON {
            let step = completed_steps;
            if !control(step, current_time) {
                cancelled = true;
                break;
            }

            let desired_bounds = self.desired_active_bounds(
                current_time,
                t_final,
                dt,
                Some(active.bounds),
                boundary_hot,
            );
            if desired_bounds != active.bounds {
                active = self.build_active_subdomain(desired_bounds);
            }

            let temp_local = self.gather_active_field(&active, &self.temperature);
            let alpha_local = self.gather_active_field(&active, &self.alpha);

            // Update reaction (explicit) on the currently active region only.
            let d_alpha = reaction_rate(&temp_local, &alpha_local, &self.params);
            let s_laser_current = self
                .laser
                .source_profile(&active.nodes, &alpha_local, current_time, previous_dt, &active.m_matrix);
            let s_radiation_current = radiative_cooling_source(
                &temp_local,
                &alpha_local,
                &self.params,
                self.laser.film_thickness,
            );
            let step_dt = self.adaptive_step_size(
                current_time,
                t_final,
                dt,
                previous_dt,
                &d_alpha,
                &s_laser_current,
                &s_radiation_current,
            );
            let mut next_step_dt = step_dt;
            let pulse_keyframe_time = self
                .laser
                .first_pulse_event_in_window(current_time, (current_time + step_dt).min(t_final));
            if let Some(event_time) = pulse_keyframe_time {
                next_step_dt = (event_time - current_time).max(f64::EPSILON);
            }
            let t = (current_time + next_step_dt).min(t_final);

            let mut alpha_local_next = alpha_local.clone();
            alpha_local_next.scaled_add(next_step_dt, &d_alpha);
            alpha_local_next.mapv_inplace(|a| a.clamp(0.0, 1.0));
            Self::scatter_active_field(&mut self.alpha, &active, &alpha_local_next);

            // Compute source terms on the active region.
            let s_laser = self
                .laser
                .source_profile(&active.nodes, &alpha_local_next, t, next_step_dt, &active.m_matrix);
            let s_radiation = radiative_cooling_source(
                &temp_local,
                &alpha_local_next,
                &self.params,
                self.laser.film_thickness,
            );
            let mut s_total = s_laser.clone();
            s_total.scaled_add(reaction_heat_scale, &d_alpha);
            s_total += &s_radiation;

            let f_source = spmv(&active.m_matrix, &s_total);
            let mut f_total = f_source;
            f_total += &active.f_bound;
            f_total -= &active.interface_coupling;
            let mut rhs = spmv(&active.m_eff, &temp_local);
            rhs.scaled_add(next_step_dt, &f_total);

            let k_dt = scale_csmat(&active.k_sum, next_step_dt);
            let lhs = &active.m_eff + &k_dt;

            let solve_report = pcg_solve_jacobi(&lhs, &rhs, &temp_local, 1e-10, 1000);
            total_iterations += solve_report.iterations;
            max_iterations = max_iterations.max(solve_report.iterations);
            total_solve_secs += solve_report.solve_secs;
            max_residual_norm = max_residual_norm.max(solve_report.residual_norm);
            let temperature_local_next = solve_report.solution;
            Self::scatter_active_field(&mut self.temperature, &active, &temperature_local_next);
            completed_steps += 1;
            current_time = t;
            previous_dt = next_step_dt;
            boundary_hot = self.active_boundary_is_hot(&active, &temperature_local_next, &alpha_local_next);

            let max_t = self.temperature.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let avg_a = self.alpha.mean().unwrap_or(0.0);
            let progress = if t_final > 0.0 {
                (current_time / t_final).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let reaction_rate_current = reaction_rate(&temperature_local_next, &alpha_local_next, &self.params);
            let mut enthalpy_source_current = reaction_rate_current.clone();
            enthalpy_source_current *= reaction_heat_scale;
            let radiation_current = radiative_cooling_source(
                &temperature_local_next,
                &alpha_local_next,
                &self.params,
                self.laser.film_thickness,
            );
            let convection_power = (&active.f_bound - &spmv(&active.k_bound, &temperature_local_next)).sum();

            on_series_point(SimTimeSeriesPoint {
                time: current_time,
                max_temp: max_t,
                avg_conversion: avg_a,
                laser_power: self.integrated_volume_power_on(&active.m_matrix, &s_laser),
                enthalpy_power: self.integrated_volume_power_on(&active.m_matrix, &enthalpy_source_current),
                convection_power,
                radiation_power: self.integrated_volume_power_on(&active.m_matrix, &radiation_current),
            });

            // Save frame if on interval, at the run end, or when the step lands on a pulse event.
            if completed_steps % save_interval == 0
                || current_time >= t_final - f64::EPSILON
                || pulse_keyframe_time.is_some()
            {
                on_frame(SimFrame {
                    frame_index,
                    time: current_time,
                    temperature: self.temperature.to_vec(),
                    alpha: self.alpha.to_vec(),
                    laser: self.full_field_from_active(&active, &s_laser),
                    active_bounds: Some(active.bounds.to_public()),
                    max_temp: max_t,
                    avg_conversion: avg_a,
                    progress,
                });

                frame_index += 1;
            }

            let progress = if t_final > 0.0 {
                (current_time / t_final).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let dynamic_total_steps = if completed_steps > 0 && current_time > 0.0 {
                (((t_final / (current_time / completed_steps as f64)).ceil()) as usize)
                    .max(completed_steps)
                    .max(estimated_num_steps)
            } else {
                estimated_num_steps
            };
            if completed_steps % progress_interval == 0 || progress >= 1.0 - 1e-12 {
                on_progress(SimProgress {
                    step: completed_steps,
                    num_steps: dynamic_total_steps.max(1),
                    time: current_time,
                    progress,
                    max_temp: max_t,
                    active_nodes: active.node_ids.len(),
                    total_nodes: self.mesh.num_nodes,
                });
            }
        }

        let solve_calls = completed_steps;
        RunSummary {
            num_steps: if self.adaptive_time_stepping {
                completed_steps
            } else {
                estimated_num_steps
            },
            completed_steps,
            cancelled,
            solver: SolverStats {
                preconditioner: "Jacobi PCG".to_string(),
                solve_calls,
                total_iterations,
                max_iterations,
                avg_iterations: if solve_calls > 0 {
                    total_iterations as f64 / solve_calls as f64
                } else {
                    0.0
                },
                total_solve_secs,
                avg_solve_secs: if solve_calls > 0 {
                    total_solve_secs / solve_calls as f64
                } else {
                    0.0
                },
                max_residual_norm,
            },
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
    pcg_solve_jacobi(a, b, x0, tol, max_iter).solution
}

fn jacobi_preconditioner(a: &CsMat<f64>) -> Array1<f64> {
    let mut inv_diag = Array1::ones(a.rows());
    for (row_idx, row_vec) in a.outer_iterator().enumerate() {
        let mut diagonal = 0.0;
        for (col_idx, &value) in row_vec.iter() {
            if col_idx == row_idx {
                diagonal = value;
                break;
            }
        }
        inv_diag[row_idx] = if diagonal.abs() > 1e-30 { 1.0 / diagonal } else { 1.0 };
    }
    inv_diag
}

fn apply_diagonal_preconditioner(inv_diag: &Array1<f64>, residual: &Array1<f64>) -> Array1<f64> {
    let mut result = residual.clone();
    if let (Some(result_slice), Some(inv_diag_slice)) = (
        result.as_slice_memory_order_mut(),
        inv_diag.as_slice_memory_order(),
    ) {
        for (value, inv) in result_slice.iter_mut().zip(inv_diag_slice.iter()) {
            *value *= *inv;
        }
        return result;
    }

    for i in 0..result.len() {
        result[i] *= inv_diag[i];
    }
    result
}

pub fn pcg_solve_jacobi(
    a: &CsMat<f64>,
    b: &Array1<f64>,
    x0: &Array1<f64>,
    tol: f64,
    max_iter: usize,
) -> CgSolveReport {
    let solve_start = Instant::now();
    let preconditioner = jacobi_preconditioner(a);
    let mut x = x0.clone();
    let mut r = b - &spmv(a, &x);
    let mut residual_norm = dot_product(&r, &r).sqrt();
    
    if residual_norm < tol {
        return CgSolveReport {
            solution: x,
            iterations: 0,
            converged: true,
            residual_norm,
            solve_secs: solve_start.elapsed().as_secs_f64(),
        };
    }

    let mut z = apply_diagonal_preconditioner(&preconditioner, &r);
    let mut p = z.clone();
    let mut rz_old = dot_product(&r, &z);
    let mut iterations = 0_usize;

    for iter in 0..max_iter {
        iterations = iter + 1;
        let ap = spmv(a, &p);
        let p_dot_ap = dot_product(&p, &ap);
        
        if p_dot_ap.abs() < 1e-30 {
            break;
        }
        
        let alpha = rz_old / p_dot_ap;
        x.scaled_add(alpha, &p);
        r.scaled_add(-alpha, &ap);
        
        residual_norm = dot_product(&r, &r).sqrt();
        if residual_norm < tol {
            break;
        }

        z = apply_diagonal_preconditioner(&preconditioner, &r);
        let rz_new = dot_product(&r, &z);
        if rz_old.abs() < 1e-30 {
            break;
        }

        let beta = rz_new / rz_old;
        p.mapv_inplace(|value| value * beta);
        p += &z;
        rz_old = rz_new;
    }

    CgSolveReport {
        solution: x,
        iterations,
        converged: residual_norm < tol,
        residual_norm,
        solve_secs: solve_start.elapsed().as_secs_f64(),
    }
}

#[cfg(test)]
mod tests {
    use super::{ActiveBounds, FEMSimulation, SimParams};
    use ndarray::Array1;

    #[test]
    fn short_single_pulse_run_heats_above_initial_temperature() {
        let mut params = SimParams::default();
        params.adaptive_time_stepping = false;
        params.lxy = 200e-6;
        params.nxy = 16;
        params.t_final = 10e-9;
        params.dt = 1e-9;
        params.save_interval = 1;
        params.rho = 100.0;
        params.cp = 100.0;
        params.k = 0.0;
        params.h_coeff = 0.0;
        params.a_pre = 0.0;
        params.delta_h = 0.0;
        params.pulse_energy = 1e-3;
        params.sigma = 20e-6;
        params.pulse_rate = 20e3;
        params.scan_speed = 0.1;
        params.line_spacing = 40e-6;
        params.scan_margin = 20e-6;
        params.film_thickness = 10e-6;
        params.absorption_coeff = 1e5;

        let mut sim = FEMSimulation::new_with_params(&params);
        let mut max_temp = params.t_init;

        sim.run_streaming(params.t_final, params.dt, params.save_interval, |frame| {
            max_temp = max_temp.max(frame.max_temp);
        });

        assert!(
            max_temp > params.t_init + 1e-6,
            "expected laser heating above T_init, got {max_temp}"
        );
    }

    #[test]
    fn streaming_progress_reaches_completion() {
        let mut params = SimParams::default();
        params.adaptive_time_stepping = false;
        params.nxy = 8;
        params.t_final = 4e-9;
        params.dt = 1e-9;
        params.save_interval = 2;
        params.a_pre = 0.0;
        params.delta_h = 0.0;
        params.h_coeff = 0.0;

        let mut sim = FEMSimulation::new_with_params(&params);
        let mut final_progress = 0.0;
        let mut final_step = 0;
        let mut final_max_temp = params.t_init;
        let mut progress_events = 0;

        let summary = sim.run_streaming_with_progress(
            params.t_final,
            params.dt,
            params.save_interval,
            |_frame| {},
            |progress| {
                progress_events += 1;
                final_progress = progress.progress;
                final_step = progress.step;
                final_max_temp = progress.max_temp;
            },
        );

        assert!(progress_events >= 2, "expected initial and final progress events");
        assert_eq!(final_step, (params.t_final / params.dt) as usize);
        assert_eq!(summary.solver.solve_calls, final_step);
        assert_eq!(summary.solver.preconditioner, "Jacobi PCG");
        assert!(final_max_temp.is_finite(), "expected finite progress max temperature");
        assert!(
            (final_progress - 1.0).abs() < 1e-12,
            "expected final progress to be 1.0, got {final_progress}"
        );
    }

    #[test]
    fn radiative_cooling_lowers_temperature_without_other_sources() {
        let mut params = SimParams::default();
        params.adaptive_time_stepping = false;
        params.nxy = 8;
        params.t_init = 800.0;
        params.t_inf = 300.0;
        params.t_final = 10e-6;
        params.dt = 1e-6;
        params.save_interval = 1;
        params.rho = 100.0;
        params.cp = 100.0;
        params.k = 0.0;
        params.h_coeff = 0.0;
        params.a_pre = 0.0;
        params.delta_h = 0.0;
        params.pulse_energy = 0.0;
        params.emissivity = 1.0;
        params.emissivity_transformed = 1.0;

        let mut sim = FEMSimulation::new_with_params(&params);
        let mut last_max_temp = params.t_init;

        sim.run_streaming(params.t_final, params.dt, params.save_interval, |frame| {
            last_max_temp = frame.max_temp;
        });

        assert!(
            last_max_temp < params.t_init,
            "expected radiative cooling below T_init, got {last_max_temp}"
        );
    }

    #[test]
    fn adaptive_time_stepping_refines_large_max_dt() {
        let mut params = SimParams::default();
        params.adaptive_time_stepping = true;
        params.nxy = 8;
        params.t_final = 40e-9;
        params.dt = 20e-9;
        params.save_interval = 1;
        params.rho = 100.0;
        params.cp = 100.0;
        params.k = 0.0;
        params.h_coeff = 0.0;
        params.a_pre = 0.0;
        params.delta_h = 0.0;
        params.pulse_energy = 1e-3;
        params.pulse_width = 8e-9;
        params.sigma = 20e-6;

        let mut sim = FEMSimulation::new_with_params(&params);
        let summary = sim.run_streaming(params.t_final, params.dt, params.save_interval, |_frame| {});

        assert!(
            summary.completed_steps > 2,
            "expected adaptive stepping to refine beyond the coarse fixed-step count, got {}",
            summary.completed_steps
        );
    }

    #[test]
    fn time_series_can_sample_every_step_even_with_sparse_frame_saves() {
        let mut params = SimParams::default();
        params.adaptive_time_stepping = false;
        params.nxy = 8;
        params.t_final = 6e-9;
        params.dt = 1e-9;
        params.save_interval = 3;
        params.a_pre = 0.0;
        params.delta_h = 0.0;
        params.h_coeff = 0.0;

        let mut sim = FEMSimulation::new_with_params(&params);
        let mut frame_count = 0_usize;
        let mut series_count = 0_usize;

        sim.run_streaming_with_control(
            params.t_final,
            params.dt,
            params.save_interval,
            |_frame| {
                frame_count += 1;
            },
            |_progress| {},
            |_point| {
                series_count += 1;
            },
            |_, _| true,
        );

        assert!(series_count > frame_count, "expected denser scalar history than saved frames");
    }

    #[test]
    fn localized_laser_can_start_with_a_subdomain_smaller_than_the_full_mesh() {
        let params = SimParams::default();
        let sim = FEMSimulation::new_with_params(&params);

        let bounds = sim.desired_active_bounds(0.0, params.t_final, params.dt, None, false);

        assert!(
            !bounds.is_full(sim.nx, sim.ny),
            "expected localized default laser heating to start on a reduced active box"
        );
    }

    #[test]
    fn uniform_initial_cooling_forces_full_domain_solves() {
        let mut params = SimParams::default();
        params.pulse_energy = 0.0;
        params.t_init = 500.0;
        params.t_inf = 300.0;

        let sim = FEMSimulation::new_with_params(&params);
        let bounds = sim.desired_active_bounds(0.0, params.t_final, params.dt, None, false);

        assert!(
            bounds.is_full(sim.nx, sim.ny),
            "expected globally hot initial conditions with cooling to force a full-domain solve"
        );
    }

    #[test]
    fn subdomain_edge_temperature_trigger_respects_configured_rise() {
        let mut params = SimParams::default();
        params.nxy = 6;
        params.subdomain_edge_temp_trigger = true;
        params.subdomain_edge_temp_rise = 10.0;

        let sim = FEMSimulation::new_with_params(&params);
        let bounds = ActiveBounds {
            i_min: 2,
            i_max: 3,
            j_min: 2,
            j_max: 3,
        };
        let active = sim.build_active_subdomain(bounds);
        let mut local_temp = Array1::from_elem(active.node_ids.len(), params.t_init + 9.0);
        let local_alpha = Array1::zeros(active.node_ids.len());

        assert!(
            !sim.active_boundary_is_hot(&active, &local_temp, &local_alpha),
            "expected a 9 K edge rise to stay below the configured 10 K growth threshold"
        );

        local_temp[0] = params.t_init + 11.0;
        assert!(
            sim.active_boundary_is_hot(&active, &local_temp, &local_alpha),
            "expected an 11 K edge rise to trigger boundary-driven subdomain growth"
        );
    }

    #[test]
    fn reported_laser_energy_matches_absorbed_pulse_energy_scale() {
        let mut params = SimParams::default();
        params.adaptive_time_stepping = false;
        params.adaptive_subdomains = true;
        params.nxy = 8;
        params.t_final = 2e-6;
        params.dt = 1e-6;
        params.save_interval = 1;
        params.k = 0.0;
        params.h_coeff = 0.0;
        params.a_pre = 0.0;
        params.delta_h = 0.0;
        params.pulse_energy = 30e-6;
        params.pulse_width = 100e-9;
        params.gaussian_temporal = false;
        params.pulse_rate = 1e6;
        params.film_thickness = 10e-6;
        params.absorption_coeff = 1e4;
        params.absorption_coeff_transformed = 1e4;

        let absorbed_fraction = 1.0 - (-params.absorption_coeff * params.film_thickness).exp();
        let expected_energy = params.pulse_energy * absorbed_fraction;

        let mut sim = FEMSimulation::new_with_params(&params);
        let mut first_nonzero_laser_energy = 0.0_f64;

        sim.run_streaming_with_control(
            params.t_final,
            params.dt,
            params.save_interval,
            |_frame| {},
            |_progress| {},
            |point| {
                if first_nonzero_laser_energy <= 0.0 && point.laser_power > 0.0 {
                    first_nonzero_laser_energy = point.laser_power * params.dt;
                }
            },
            |_, _| true,
        );

        let relative_error = if expected_energy > 0.0 {
            ((first_nonzero_laser_energy - expected_energy) / expected_energy).abs()
        } else {
            0.0
        };

        assert!(
            relative_error < 0.05,
            "expected absorbed pulse energy near {expected_energy:e}, got {first_nonzero_laser_energy:e}"
        );
    }
}
