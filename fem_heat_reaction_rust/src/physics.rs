use ndarray::Array1;
use rayon::prelude::*;
use sprs::CsMat;
use std::f64::consts::LN_2;

const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// Sparse matrix-vector product: result = A * x
pub fn spmv(a: &CsMat<f64>, x: &Array1<f64>) -> Array1<f64> {
    let indptr_binding = a.indptr();
    let indptr = indptr_binding.raw_storage();
    let indices = a.indices();
    let data = a.data();

    let compute_row = |row_idx: usize| {
        let start = indptr[row_idx];
        let end = indptr[row_idx + 1];
        let mut sum = 0.0;
        for entry_idx in start..end {
            sum += data[entry_idx] * x[indices[entry_idx]];
        }
        sum
    };

    let result = if a.rows() >= 512 && a.nnz() >= 4096 {
        (0..a.rows())
            .into_par_iter()
            .map(compute_row)
            .collect::<Vec<_>>()
    } else {
        (0..a.rows()).map(compute_row).collect::<Vec<_>>()
    };

    Array1::from_vec(result)
}

/// Dot product helper with Rayon fallback for large vectors.
pub fn dot_product(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    if let (Some(a_slice), Some(b_slice)) = (a.as_slice_memory_order(), b.as_slice_memory_order()) {
        if a_slice.len() >= 4096 {
            return a_slice
                .par_iter()
                .zip(b_slice.par_iter())
                .map(|(lhs, rhs)| lhs * rhs)
                .sum();
        }
    }

    a.iter().zip(b.iter()).map(|(lhs, rhs)| lhs * rhs).sum()
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
    pub emissivity: f64,       // Emissivity before transformation [-]
    pub emissivity_transformed: f64, // Emissivity after transformation [-]
    
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
            emissivity: 0.85,
            emissivity_transformed: 0.85,
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
    pub pulse_energy: f64,     // Energy per pulse [J]
    pub sigma: f64,            // Beam 1/e^2 radius [m]
    pub pulse_period: f64,     // Time between pulses [s]
    pub pulse_width: f64,      // Pulse duration [s]
    pub gaussian_spatial: bool, // Use a Gaussian spot profile instead of a top-hat
    pub gaussian_temporal: bool, // Use a Gaussian pulse profile instead of a single-step pulse
    pub scan_speed: f64,       // Beam velocity [m/s]
    pub line_spacing: f64,     // Raster line spacing [m]
    pub scan_margin: f64,      // Margin from all domain edges [m]
    pub film_thickness: f64,   // Film thickness [m]
    pub absorption_coeff: f64, // Base-material Beer-Lambert absorption coefficient [1/m]
    pub absorption_coeff_transformed: f64, // Transformed-material absorption coefficient [1/m]
    pub lx: f64,               // Domain width [m]
    pub ly: f64,               // Domain height [m] (needed for raster logic)
}

impl LaserParams {
    pub fn new(lx: f64, ly: f64) -> Self {
        LaserParams {
            power: 1000.0,
            pulse_energy: 1000.0 * 10e-8,
            sigma: 50e-6 / 2.35,
            pulse_period: 1.0 / 20000.0,
            pulse_width: 10e-8,        // Pulse duration [s] (100 ns FWHM)
            gaussian_spatial: true,
            gaussian_temporal: true,
            scan_speed: 0.1,
            line_spacing: 40e-6,
            scan_margin: 20e-6,
            film_thickness: 10e-6,
            absorption_coeff: 1e5,
            absorption_coeff_transformed: 1e5,
            lx,
            ly,
        }
    }

    fn repetition_period(&self) -> f64 {
        self.pulse_period.max(self.pulse_width).max(f64::EPSILON)
    }

    fn pulse_center_offset(&self) -> f64 {
        if self.gaussian_temporal {
            0.5 * self.pulse_width.min(self.repetition_period())
        } else {
            0.0
        }
    }

    fn pulse_sigma(&self) -> f64 {
        // The UI treats pulse width as a Gaussian FWHM.
        (self.pulse_width / (2.0 * (2.0 * LN_2).sqrt())).max(f64::EPSILON)
    }

    fn temporal_profile(&self, t: f64) -> f64 {
        let period = self.repetition_period();
        let t_local = t.rem_euclid(period);
        let t_peak = self.pulse_center_offset();
        let sigma_t = self.pulse_sigma();
        (-((t_local - t_peak).powi(2)) / (2.0 * sigma_t.powi(2))).exp()
    }

    fn instantaneous_pulses_in_window(&self, start: f64, end: f64) -> usize {
        if !(end > start) {
            return 0;
        }

        let pulse_time = self.pulse_center_offset();
        if !self.pulse_period.is_finite() {
            return usize::from(start <= pulse_time && pulse_time < end);
        }

        let period = self.repetition_period();
        let min_index = (((start - pulse_time) / period).ceil() as isize).max(0);
        let max_index = (((end - pulse_time) / period).ceil() as isize - 1).max(-1);
        if max_index < min_index {
            return 0;
        }
        (max_index - min_index + 1) as usize
    }

    fn temporal_power(&self, t: f64, step_dt: f64) -> f64 {
        if self.gaussian_temporal {
            return self.power * self.temporal_profile(t);
        }

        if !(step_dt > 0.0) {
            return 0.0;
        }

        let pulse_count = self.instantaneous_pulses_in_window((t - step_dt).max(0.0), t);
        pulse_count as f64 * self.pulse_energy / step_dt.max(f64::EPSILON)
    }

    fn top_hat_radius(&self) -> f64 {
        ((2.0 * LN_2).sqrt() * self.sigma).max(f64::EPSILON)
    }

    pub fn first_pulse_event_in_window(&self, start: f64, end: f64) -> Option<f64> {
        if !(end > start) {
            return None;
        }

        let pulse_time = self.pulse_center_offset();
        if !self.pulse_period.is_finite() {
            return ((pulse_time > start) && (pulse_time <= end)).then_some(pulse_time);
        }

        let period = self.repetition_period();
        let pulse_index = (((start - pulse_time) / period).floor() as isize + 1).max(0) as usize;
        let candidate = pulse_time + pulse_index as f64 * period;
        ((candidate > start) && (candidate <= end)).then_some(candidate)
    }

    fn beam_position(&self, t: f64) -> (f64, f64) {
        let x_min = self.scan_margin.clamp(0.0, self.lx);
        let x_max = (self.lx - self.scan_margin).max(x_min);
        let y_min = self.scan_margin.clamp(0.0, self.ly);
        let y_max = (self.ly - self.scan_margin).max(y_min);
        let scan_height = (y_max - y_min).max(0.0);

        let spacing = self.line_spacing.max(f64::EPSILON);
        let scan_width = (x_max - x_min).max(0.0);
        let line_count = (scan_width / spacing).floor() as usize + 1;

        if self.scan_speed <= 0.0 || scan_height <= 0.0 {
            return (x_min, y_min);
        }

        let time_per_line = scan_height / self.scan_speed.max(f64::EPSILON);
        let raw_line = (t / time_per_line).floor().max(0.0) as usize;
        let line_index = raw_line % line_count.max(1);
        let t_in_line = t.rem_euclid(time_per_line);
        let progress = (t_in_line / time_per_line).clamp(0.0, 1.0);

        let x_pos = (x_min + line_index as f64 * spacing).min(x_max);
        let y_pos = if raw_line % 2 == 0 {
            y_min + progress * scan_height
        } else {
            y_max - progress * scan_height
        };

        (x_pos, y_pos)
    }

    fn absorbed_fraction_for_coeff(&self, absorption_coeff: f64) -> f64 {
        if self.film_thickness <= 0.0 || absorption_coeff <= 0.0 {
            return 0.0;
        }
        1.0 - (-absorption_coeff * self.film_thickness).exp()
    }

    pub fn absorbed_fraction_untransformed(&self) -> f64 {
        self.absorbed_fraction_for_coeff(self.absorption_coeff)
    }

    pub fn absorbed_fraction_transformed(&self) -> f64 {
        self.absorbed_fraction_for_coeff(self.absorption_coeff_transformed)
    }

    pub fn blended_absorbed_fraction(&self, conversion: f64) -> f64 {
        let alpha = conversion.clamp(0.0, 1.0);
        (1.0 - alpha) * self.absorbed_fraction_untransformed()
            + alpha * self.absorbed_fraction_transformed()
    }
    
    /// Calculate laser source profile at given time for all nodes
    pub fn source_profile(
        &self,
        nodes: &ndarray::Array2<f64>,
        alpha: &Array1<f64>,
        t: f64,
        step_dt: f64,
        m_matrix: &sprs::CsMat<f64>,
    ) -> Array1<f64> {
        let num_nodes = nodes.nrows();

        let (x_pos, y_pos) = self.beam_position(t);
        
        // Temporal profile: either a Gaussian pulse or a single-step pulse-energy deposit.
        let temporal_power = self.temporal_power(t, step_dt);
        
        // Spatial profile: Gaussian beam or top-hat beam
        let inv_two_sigma_sq = 1.0 / (2.0 * self.sigma.powi(2));
        let top_hat_radius_sq = self.top_hat_radius().powi(2);
        let mut spatial_raw = if let Some(node_slice) = nodes.as_slice_memory_order() {
            let values = if num_nodes >= 2048 {
                node_slice
                    .par_chunks_exact(2)
                    .map(|coords| {
                        let dx = coords[0] - x_pos;
                        let dy = coords[1] - y_pos;
                        if self.gaussian_spatial {
                            (-(dx * dx + dy * dy) * inv_two_sigma_sq).exp()
                        } else {
                            usize::from(dx * dx + dy * dy <= top_hat_radius_sq) as f64
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                node_slice
                    .chunks_exact(2)
                    .map(|coords| {
                        let dx = coords[0] - x_pos;
                        let dy = coords[1] - y_pos;
                        if self.gaussian_spatial {
                            (-(dx * dx + dy * dy) * inv_two_sigma_sq).exp()
                        } else {
                            usize::from(dx * dx + dy * dy <= top_hat_radius_sq) as f64
                        }
                    })
                    .collect::<Vec<_>>()
            };
            Array1::from_vec(values)
        } else {
            let mut spatial_raw = Array1::zeros(num_nodes);
            for i in 0..num_nodes {
                let dx = nodes[[i, 0]] - x_pos;
                let dy = nodes[[i, 1]] - y_pos;
                spatial_raw[i] = if self.gaussian_spatial {
                    (-(dx * dx + dy * dy) * inv_two_sigma_sq).exp()
                } else {
                    usize::from(dx * dx + dy * dy <= top_hat_radius_sq) as f64
                };
            }
            spatial_raw
        };

        if !self.gaussian_spatial && spatial_raw.iter().all(|value| *value <= 0.0) {
            let nearest_index = (0..num_nodes)
                .min_by(|lhs, rhs| {
                    let lhs_dx = nodes[[*lhs, 0]] - x_pos;
                    let lhs_dy = nodes[[*lhs, 1]] - y_pos;
                    let rhs_dx = nodes[[*rhs, 0]] - x_pos;
                    let rhs_dy = nodes[[*rhs, 1]] - y_pos;
                    let lhs_r2 = lhs_dx * lhs_dx + lhs_dy * lhs_dy;
                    let rhs_r2 = rhs_dx * rhs_dx + rhs_dy * rhs_dy;
                    lhs_r2
                        .partial_cmp(&rhs_r2)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(index) = nearest_index {
                spatial_raw[index] = 1.0;
            }
        }
        
        // Normalize using mass matrix (sparse mat-vec product)
        let spatial_integral: f64 = spmv(m_matrix, &spatial_raw).sum();
        let spatial_norm = if spatial_integral > 1e-12 {
            &spatial_raw / spatial_integral
        } else {
            spatial_raw
        };

        // Blend absorbed energy between untransformed and transformed material
        // using the local conversion fraction.
        let thickness = self.film_thickness.max(f64::EPSILON);
        let absorbed_fraction_base = self.absorbed_fraction_untransformed();
        let absorbed_fraction_transformed = self.absorbed_fraction_transformed();
        let source_scale = temporal_power / thickness;

        if let (Some(spatial_slice), Some(alpha_slice)) = (
            spatial_norm.as_slice_memory_order(),
            alpha.as_slice_memory_order(),
        ) {
            let values = if num_nodes >= 2048 {
                spatial_slice
                    .par_iter()
                    .zip(alpha_slice.par_iter())
                    .map(|(spatial_value, alpha_value)| {
                        let conversion = alpha_value.clamp(0.0, 1.0);
                        let absorbed_fraction = (1.0 - conversion) * absorbed_fraction_base
                            + conversion * absorbed_fraction_transformed;
                        spatial_value * absorbed_fraction * source_scale
                    })
                    .collect::<Vec<_>>()
            } else {
                spatial_slice
                    .iter()
                    .zip(alpha_slice.iter())
                    .map(|(spatial_value, alpha_value)| {
                        let conversion = alpha_value.clamp(0.0, 1.0);
                        let absorbed_fraction = (1.0 - conversion) * absorbed_fraction_base
                            + conversion * absorbed_fraction_transformed;
                        spatial_value * absorbed_fraction * source_scale
                    })
                    .collect::<Vec<_>>()
            };
            return Array1::from_vec(values);
        }

        let mut volumetric_source = spatial_norm;
        for i in 0..num_nodes {
            let conversion = alpha[i].clamp(0.0, 1.0);
            let absorbed_fraction = (1.0 - conversion) * absorbed_fraction_base
                + conversion * absorbed_fraction_transformed;
            volumetric_source[i] *= absorbed_fraction * source_scale;
        }
        volumetric_source
    }
}

/// Compute reaction rate: d(alpha)/dt = (1-alpha) * A * exp(-Ea/RT)
pub fn reaction_rate(temp: &Array1<f64>, alpha: &Array1<f64>, params: &PhysicalParams) -> Array1<f64> {
    if let (Some(temp_slice), Some(alpha_slice)) = (temp.as_slice_memory_order(), alpha.as_slice_memory_order()) {
        let values = if temp.len() >= 2048 {
            temp_slice
                .par_iter()
                .zip(alpha_slice.par_iter())
                .map(|(temp_i, alpha_i)| {
                    let t_clipped = temp_i.max(1e-6);
                    let k_rate = params.a_pre * (-params.ea / (params.r_gas * t_clipped)).exp();
                    (1.0 - alpha_i) * k_rate
                })
                .collect::<Vec<_>>()
        } else {
            temp_slice
                .iter()
                .zip(alpha_slice.iter())
                .map(|(temp_i, alpha_i)| {
                    let t_clipped = temp_i.max(1e-6);
                    let k_rate = params.a_pre * (-params.ea / (params.r_gas * t_clipped)).exp();
                    (1.0 - alpha_i) * k_rate
                })
                .collect::<Vec<_>>()
        };
        return Array1::from_vec(values);
    }

    let mut rate = Array1::zeros(temp.len());
    for i in 0..temp.len() {
        let t_clipped = temp[i].max(1e-6);
        let k_rate = params.a_pre * (-params.ea / (params.r_gas * t_clipped)).exp();
        rate[i] = (1.0 - alpha[i]) * k_rate;
    }
    rate
}

/// Depth-averaged radiative cooling for one exposed film surface.
pub fn radiative_cooling_source(
    temp: &Array1<f64>,
    alpha: &Array1<f64>,
    params: &PhysicalParams,
    film_thickness: f64,
) -> Array1<f64> {
    if film_thickness <= 0.0 {
        return Array1::zeros(temp.len());
    }

    let thickness = film_thickness.max(f64::EPSILON);
    let ambient_t4 = params.t_inf.powi(4);
    let source_scale = STEFAN_BOLTZMANN / thickness;

    if let (Some(temp_slice), Some(alpha_slice)) = (temp.as_slice_memory_order(), alpha.as_slice_memory_order()) {
        let values = if temp.len() >= 2048 {
            temp_slice
                .par_iter()
                .zip(alpha_slice.par_iter())
                .map(|(temp_i, alpha_i)| {
                    let temperature = temp_i.max(1e-6);
                    let conversion = alpha_i.clamp(0.0, 1.0);
                    let emissivity = ((1.0 - conversion) * params.emissivity
                        + conversion * params.emissivity_transformed)
                        .clamp(0.0, 1.0);
                    -(emissivity * source_scale) * (temperature.powi(4) - ambient_t4)
                })
                .collect::<Vec<_>>()
        } else {
            temp_slice
                .iter()
                .zip(alpha_slice.iter())
                .map(|(temp_i, alpha_i)| {
                    let temperature = temp_i.max(1e-6);
                    let conversion = alpha_i.clamp(0.0, 1.0);
                    let emissivity = ((1.0 - conversion) * params.emissivity
                        + conversion * params.emissivity_transformed)
                        .clamp(0.0, 1.0);
                    -(emissivity * source_scale) * (temperature.powi(4) - ambient_t4)
                })
                .collect::<Vec<_>>()
        };
        return Array1::from_vec(values);
    }

    let mut source = Array1::zeros(temp.len());
    for i in 0..temp.len() {
        let temperature = temp[i].max(1e-6);
        let conversion = alpha[i].clamp(0.0, 1.0);
        let emissivity = ((1.0 - conversion) * params.emissivity
            + conversion * params.emissivity_transformed)
            .clamp(0.0, 1.0);
        source[i] = -(emissivity * source_scale) * (temperature.powi(4) - ambient_t4);
    }
    source
}

#[cfg(test)]
mod tests {
    use super::{LaserParams, PhysicalParams, radiative_cooling_source};
    use ndarray::Array1;

    #[test]
    fn default_pulse_overlaps_short_simulation_window() {
        let laser = LaserParams::new(200e-6, 200e-6);

        assert!(laser.temporal_profile(0.0) > 0.49);
        assert!(laser.temporal_profile(10e-9) > 0.6);
        assert!(laser.temporal_profile(laser.pulse_period * 0.5) < 1e-12);
    }

    #[test]
    fn single_step_pulse_deposits_one_pulse_energy_over_the_step() {
        let mut laser = LaserParams::new(200e-6, 200e-6);
        laser.gaussian_temporal = false;
        laser.pulse_energy = 2.5e-6;
        laser.pulse_period = 1.0 / 20_000.0;

        let power = laser.temporal_power(1e-9, 2e-9);
        assert!((power * 2e-9 - laser.pulse_energy).abs() < 1e-18);
    }

    #[test]
    fn top_hat_beam_marks_inside_and_outside_nodes() {
        let mut laser = LaserParams::new(200e-6, 200e-6);
        laser.gaussian_spatial = false;
        laser.sigma = 10e-6;
        let radius = laser.top_hat_radius();

        let inside = (radius * 0.5).powi(2);
        let outside = (radius * 1.5).powi(2);

        assert_eq!(usize::from(inside <= radius.powi(2)), 1);
        assert_eq!(usize::from(outside <= radius.powi(2)), 0);
    }

    #[test]
    fn pulse_event_window_finds_first_pulse_center() {
        let laser = LaserParams::new(200e-6, 200e-6);
        let event = laser.first_pulse_event_in_window(0.0, laser.pulse_width);

        assert!(event.is_some());
        assert!(event.unwrap() > 0.0);
    }

    #[test]
    fn absorbed_fraction_matches_beer_lambert_law() {
        let mut laser = LaserParams::new(200e-6, 200e-6);
        laser.film_thickness = 10e-6;
        laser.absorption_coeff = 1e5;

        let expected = 1.0 - (-1.0_f64).exp();
        assert!((laser.absorbed_fraction_untransformed() - expected).abs() < 1e-12);
    }

    #[test]
    fn blended_absorption_interpolates_between_material_states() {
        let mut laser = LaserParams::new(200e-6, 200e-6);
        laser.film_thickness = 10e-6;
        laser.absorption_coeff = 1e5;
        laser.absorption_coeff_transformed = 2e5;

        let base = laser.absorbed_fraction_untransformed();
        let transformed = laser.absorbed_fraction_transformed();
        let blended = laser.blended_absorbed_fraction(0.25);

        assert!((blended - (0.75 * base + 0.25 * transformed)).abs() < 1e-12);
    }

    #[test]
    fn raster_path_wraps_across_multiple_lines() {
        let laser = LaserParams::new(200e-6, 200e-6);
        let time_per_line = (laser.ly - 2.0 * laser.scan_margin) / laser.scan_speed;
        let (x_pos, _y_pos) = laser.beam_position(time_per_line * 6.1);

        assert!((x_pos - 60e-6).abs() < 1e-9);
    }

    #[test]
    fn radiative_cooling_is_negative_above_ambient() {
        let params = PhysicalParams {
            emissivity: 0.9,
            emissivity_transformed: 0.9,
            t_inf: 300.0,
            ..PhysicalParams::default()
        };
        let temp = Array1::from_vec(vec![400.0]);
        let alpha = Array1::from_vec(vec![0.0]);
        let source = radiative_cooling_source(&temp, &alpha, &params, 10e-6);

        assert!(source[0] < 0.0, "expected radiative cooling sink, got {}", source[0]);
    }
}
