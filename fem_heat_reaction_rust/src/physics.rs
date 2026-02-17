use ndarray::Array1;
use sprs::CsMat;
use std::f64::consts::E;

/// Sparse matrix-vector product: result = A * x
pub fn spmv(a: &CsMat<f64>, x: &Array1<f64>) -> Array1<f64> {
    let mut result = Array1::zeros(a.rows());
    for (row_idx, row_vec) in a.outer_iterator().enumerate() {
        let mut sum = 0.0;
        for (col_idx, &val) in row_vec.iter() {
            sum += val * x[col_idx];
        }
        result[row_idx] = sum;
    }
    result
}

/// Scale a sparse matrix by a scalar: result = scalar * A
pub fn scale_csmat(a: &CsMat<f64>, scalar: f64) -> CsMat<f64> {
    a.map(|&v| v * scalar)
}

/// Physical parameters for the simulation
pub struct PhysicalParams {
    // Material properties
    pub rho: f64,              // Density [kg/m^3]
    pub cp: f64,               // Heat capacity [J/(kg K)]
    pub k: f64,                // Thermal conductivity [W/(m K)]
    pub h_coeff: f64,          // Convection coefficient [W/(m^2 K)]
    
    // Reaction parameters
    pub a_pre: f64,            // Pre-exponential factor [1/s]
    pub ea: f64,               // Activation energy [J/mol]
    pub r_gas: f64,            // Gas constant [J/(mol K)]
    pub delta_h: f64,          // Heat of reaction [J/kg]
    
    // Environmental
    pub t_init: f64,           // Initial temperature [K]
    pub t_inf: f64,            // Ambient temperature [K]
}

impl Default for PhysicalParams {
    fn default() -> Self {
        PhysicalParams {
            rho: 1100.0,
            cp: 1500.0,
            k: 0.3,
            h_coeff: 10.0,
            a_pre: 1e8,
            ea: 8e4,
            r_gas: 8.314,
            delta_h: 5e5,
            t_init: 300.0,
            t_inf: 300.0,
        }
    }
}

/// Laser parameters
pub struct LaserParams {
    pub power: f64,            // Peak power [W]
    pub x_start: f64,          // Starting X position [m]
    pub y_start: f64,          // Starting Y position [m]
    pub y_margin: f64,         // Margin from top [m]
    pub x_step: f64,           // Horizontal step [m]
    pub sigma: f64,            // Beam 1/e^2 radius [m]
    pub pulse_rate: f64,       // Time between pulses [s]
    pub pulse_width: f64,      // Pulse duration [s]
    pub move_speed: f64,       // Beam velocity [m/s]
    pub ly: f64,               // Domain height [m] (needed for raster logic)
}

impl LaserParams {
    pub fn new(_lx: f64, ly: f64) -> Self {
        LaserParams {
            power: 1000.0,
            x_start: 100e-6,
            y_start: 50e-6,
            y_margin: 50e-6,
            x_step: 100e-6,
            sigma: 50e-6 / 2.35,
            pulse_rate: 1.0 / 20000.0,
            pulse_width: 10e-5,
            move_speed: 0.1,
            ly,
        }
    }
    
    /// Calculate laser source profile at given time for all nodes
    pub fn source_profile(&self, nodes: &ndarray::Array2<f64>, t: f64, m_matrix: &sprs::CsMat<f64>) -> Array1<f64> {
        let num_nodes = nodes.nrows();
        
        // Calculate beam position based on raster scanning
        let scan_height = self.ly - self.y_start - self.y_margin;
        let time_per_line = scan_height / self.move_speed;
        
        let line_number = (t / time_per_line) as usize;
        let t_in_line = t % time_per_line;
        
        // X position: steps right with each line
        let x_pos = self.x_start + (line_number as f64) * self.x_step;
        
        // Wrap around if exceeding domain (simplified, would need lx parameter)
        // For now, just let it continue
        
        // Y position: alternates up/down (bidirectional raster)
        let y_pos = if line_number % 2 == 0 {
            // Even lines: scan upward
            self.y_start + (t_in_line / time_per_line) * scan_height
        } else {
            // Odd lines: scan downward
            self.ly - self.y_margin - (t_in_line / time_per_line) * scan_height
        };
        
        // Temporal profile: Gaussian pulse
        let _pulse_idx = (t / self.pulse_rate) as usize;
        let t_local = t % self.pulse_rate;
        let t_peak = self.pulse_rate / 2.0;
        let temporal = E.powf(-((t_local - t_peak).powi(2)) / (2.0 * (self.pulse_width / 2.5).powi(2)));
        
        // Spatial profile: Gaussian beam
        let mut spatial_raw = Array1::zeros(num_nodes);
        for i in 0..num_nodes {
            let dx = nodes[[i, 0]] - x_pos;
            let dy = nodes[[i, 1]] - y_pos;
            let dist_sq = dx * dx + dy * dy;
            spatial_raw[i] = E.powf(-dist_sq / (2.0 * self.sigma.powi(2)));
        }
        
        // Normalize using mass matrix (sparse mat-vec product)
        let spatial_integral: f64 = spmv(m_matrix, &spatial_raw).sum();
        let spatial_norm = if spatial_integral > 1e-12 {
            &spatial_raw / spatial_integral
        } else {
            spatial_raw
        };
        
        &spatial_norm * (self.power * temporal)
    }
}

/// Compute reaction rate: d(alpha)/dt = (1-alpha) * A * exp(-Ea/RT)
pub fn reaction_rate(temp: &Array1<f64>, alpha: &Array1<f64>, params: &PhysicalParams) -> Array1<f64> {
    let mut rate = Array1::zeros(temp.len());
    
    for i in 0..temp.len() {
        let t_clipped = temp[i].max(1e-6);
        let k_rate = params.a_pre * E.powf(-params.ea / (params.r_gas * t_clipped));
        rate[i] = (1.0 - alpha[i]) * k_rate;
    }
    
    rate
}
