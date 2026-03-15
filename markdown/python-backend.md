# Python Backend Sketch

This file rewrites the main Rust simulation backend in a more Python-shaped form so the control flow is easier to read.

It is intentionally "code-like" rather than fully runnable:

- It focuses on the core FEM solver in `fem_heat_reaction_rust/src`.
- It leaves out most Tauri command wiring, binary file storage, and frontend event plumbing.
- It keeps the same major pieces: mesh generation, matrix assembly, laser heating, reaction, radiation, adaptive timestepping, and the semi-implicit temperature solve.

## High-level mapping

- `mesh.rs` -> mesh generation and FEM assembly
- `physics.rs` -> laser/source/reaction/radiation helpers
- `lib.rs` -> simulation object, adaptive stepping, solver loop, and PCG solve

## Python-style translation

```python
from __future__ import annotations

from dataclasses import dataclass
from math import exp, floor, inf, log, sqrt
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


LN2 = log(2.0)
STEFAN_BOLTZMANN = 5.670374419e-8


def spmv(a: csr_matrix, x: np.ndarray) -> np.ndarray:
    """Sparse matrix-vector multiply."""
    return a @ x


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def scale_csmat(a: csr_matrix, scalar: float) -> csr_matrix:
    return a * scalar


@dataclass
class Mesh:
    nodes: np.ndarray         # shape = (num_nodes, 2)
    elements: np.ndarray      # shape = (num_elements, 3)
    num_nodes: int
    num_elements: int

    @classmethod
    def new(cls, lx: float, ly: float, nx: int, ny: int) -> "Mesh":
        num_nodes = (nx + 1) * (ny + 1)
        num_elements = 2 * nx * ny

        dx = lx / nx
        dy = ly / ny

        nodes = np.zeros((num_nodes, 2), dtype=float)
        for j in range(ny + 1):
            for i in range(nx + 1):
                node_id = j * (nx + 1) + i
                nodes[node_id, 0] = i * dx
                nodes[node_id, 1] = j * dy

        elements = np.zeros((num_elements, 3), dtype=int)
        elem_id = 0
        for j in range(ny):
            for i in range(nx):
                n1 = j * (nx + 1) + i
                n2 = n1 + 1
                n3 = (j + 1) * (nx + 1) + i
                n4 = n3 + 1

                elements[elem_id] = [n1, n2, n4]
                elem_id += 1
                elements[elem_id] = [n1, n4, n3]
                elem_id += 1

        return cls(
            nodes=nodes,
            elements=elements,
            num_nodes=num_nodes,
            num_elements=num_elements,
        )

    def boundary_edges(self, nx: int, ny: int) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []

        for i in range(nx):
            edges.append((i, i + 1))

        offset = ny * (nx + 1)
        for i in range(nx):
            edges.append((offset + i, offset + i + 1))

        for j in range(ny):
            edges.append((j * (nx + 1), (j + 1) * (nx + 1)))

        for j in range(ny):
            edges.append((j * (nx + 1) + nx, (j + 1) * (nx + 1) + nx))

        return edges


def element_matrices(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    coords: shape (3, 2), one triangle's node coordinates
    returns: (me, ke, area)
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=float)
    c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=float)

    ke = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            ke[i, j] = (b[i] * b[j] + c[i] * c[j]) / (4.0 * area)

    me = (area / 12.0) * np.array(
        [
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ],
        dtype=float,
    )

    return me, ke, area


def assemble_system(mesh: Mesh) -> tuple[csr_matrix, csr_matrix]:
    n = mesh.num_nodes
    m = lil_matrix((n, n), dtype=float)
    k = lil_matrix((n, n), dtype=float)

    for elem_idx in range(mesh.num_elements):
        node_ids = mesh.elements[elem_idx]
        coords = mesh.nodes[node_ids]
        me, ke, _ = element_matrices(coords)

        for i_local, i_global in enumerate(node_ids):
            for j_local, j_global in enumerate(node_ids):
                m[i_global, j_global] += me[i_local, j_local]
                k[i_global, j_global] += ke[i_local, j_local]

    return m.tocsr(), k.tocsr()


def compute_boundary_matrices(
    mesh: Mesh,
    edges: list[tuple[int, int]],
    h_coeff: float,
) -> tuple[csr_matrix, np.ndarray]:
    n = mesh.num_nodes
    k_bound = lil_matrix((n, n), dtype=float)
    f_bound = np.zeros(n, dtype=float)

    for n1, n2 in edges:
        x1, y1 = mesh.nodes[n1]
        x2, y2 = mesh.nodes[n2]
        length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        me_1d = (length / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]], dtype=float)
        fe_1d = (length / 2.0) * np.array([1.0, 1.0], dtype=float)
        idx = [n1, n2]

        for i in range(2):
            f_bound[idx[i]] += fe_1d[i] * h_coeff
            for j in range(2):
                k_bound[idx[i], idx[j]] += me_1d[i, j] * h_coeff

    return k_bound.tocsr(), f_bound


@dataclass
class PhysicalParams:
    rho: float = 1100.0
    cp: float = 1500.0
    k: float = 0.3
    h_coeff: float = 10.0
    emissivity: float = 0.85
    emissivity_transformed: float = 0.85
    a_pre: float = 1e8
    ea: float = 8e4
    r_gas: float = 8.314
    delta_h: float = 5e5
    t_init: float = 300.0
    t_inf: float = 300.0


@dataclass
class LaserParams:
    power: float
    pulse_energy: float
    sigma: float
    pulse_period: float
    pulse_width: float
    gaussian_spatial: bool
    gaussian_temporal: bool
    scan_speed: float
    line_spacing: float
    scan_margin: float
    film_thickness: float
    absorption_coeff: float
    absorption_coeff_transformed: float
    lx: float
    ly: float

    @classmethod
    def new(cls, lx: float, ly: float) -> "LaserParams":
        pulse_width = 10e-8
        power = 1000.0
        return cls(
            power=power,
            pulse_energy=power * pulse_width,
            sigma=50e-6 / 2.35,
            pulse_period=1.0 / 20_000.0,
            pulse_width=pulse_width,
            gaussian_spatial=True,
            gaussian_temporal=True,
            scan_speed=0.1,
            line_spacing=40e-6,
            scan_margin=20e-6,
            film_thickness=10e-6,
            absorption_coeff=1e5,
            absorption_coeff_transformed=1e5,
            lx=lx,
            ly=ly,
        )

    def repetition_period(self) -> float:
        return max(self.pulse_period, self.pulse_width, np.finfo(float).eps)

    def pulse_center_offset(self) -> float:
        if self.gaussian_temporal:
            return 0.5 * min(self.pulse_width, self.repetition_period())
        return 0.0

    def pulse_sigma(self) -> float:
        return max(self.pulse_width / (2.0 * sqrt(2.0 * LN2)), np.finfo(float).eps)

    def temporal_profile(self, t: float) -> float:
        period = self.repetition_period()
        t_local = t % period
        t_peak = self.pulse_center_offset()
        sigma_t = self.pulse_sigma()
        return exp(-((t_local - t_peak) ** 2) / (2.0 * sigma_t ** 2))

    def instantaneous_pulses_in_window(self, start: float, end: float) -> int:
        if not (end > start):
            return 0

        pulse_time = self.pulse_center_offset()
        if not np.isfinite(self.pulse_period):
            return int(start <= pulse_time < end)

        period = self.repetition_period()
        min_idx = max(int(np.ceil((start - pulse_time) / period)), 0)
        max_idx = max(int(np.ceil((end - pulse_time) / period)) - 1, -1)
        if max_idx < min_idx:
            return 0
        return max_idx - min_idx + 1

    def temporal_power(self, t: float, step_dt: float) -> float:
        if self.gaussian_temporal:
            return self.power * self.temporal_profile(t)

        if not (step_dt > 0.0):
            return 0.0

        pulse_count = self.instantaneous_pulses_in_window(max(t - step_dt, 0.0), t)
        return pulse_count * self.pulse_energy / max(step_dt, np.finfo(float).eps)

    def top_hat_radius(self) -> float:
        return max(sqrt(2.0 * LN2) * self.sigma, np.finfo(float).eps)

    def first_pulse_event_in_window(self, start: float, end: float) -> float | None:
        if not (end > start):
            return None

        pulse_time = self.pulse_center_offset()
        if not np.isfinite(self.pulse_period):
            return pulse_time if start < pulse_time <= end else None

        period = self.repetition_period()
        pulse_index = max(int(floor((start - pulse_time) / period)) + 1, 0)
        candidate = pulse_time + pulse_index * period
        return candidate if start < candidate <= end else None

    def beam_position(self, t: float) -> tuple[float, float]:
        """
        The current code uses pulse-index-based raster placement:
        each pulse belongs to a column, and each new column begins
        at the scan margin from the edge.
        """
        x_min = np.clip(self.scan_margin, 0.0, self.lx)
        x_max = max(self.lx - self.scan_margin, x_min)
        y_min = np.clip(self.scan_margin, 0.0, self.ly)
        y_max = max(self.ly - self.scan_margin, y_min)
        scan_height = max(y_max - y_min, 0.0)

        spacing = max(self.line_spacing, np.finfo(float).eps)
        scan_width = max(x_max - x_min, 0.0)
        line_count = int(floor(scan_width / spacing)) + 1

        if self.scan_speed <= 0.0 or scan_height <= 0.0:
            return x_min, y_min

        period = self.pulse_period
        if not (np.isfinite(period) and period > 0.0):
            return x_min, y_min

        pulse_pitch = self.scan_speed * period
        if not (pulse_pitch > 0.0):
            return x_min, y_min

        pulses_per_column = int(floor(scan_height / pulse_pitch)) + 1
        pulse_offset = self.pulse_center_offset()

        if self.gaussian_temporal:
            pulse_index = max(int(round((t - pulse_offset) / period)), 0)
        else:
            pulse_index = max(int(floor((t - pulse_offset) / period)), 0)

        column_index = (pulse_index // max(pulses_per_column, 1)) % max(line_count, 1)
        pulse_in_column = pulse_index % max(pulses_per_column, 1)

        x_pos = min(x_min + column_index * spacing, x_max)
        travel = min(pulse_in_column * pulse_pitch, scan_height)
        y_pos = y_min + travel if column_index % 2 == 0 else y_max - travel

        return x_pos, y_pos

    def absorbed_fraction_for_coeff(self, absorption_coeff: float) -> float:
        if self.film_thickness <= 0.0 or absorption_coeff <= 0.0:
            return 0.0
        return 1.0 - exp(-absorption_coeff * self.film_thickness)

    def absorbed_fraction_untransformed(self) -> float:
        return self.absorbed_fraction_for_coeff(self.absorption_coeff)

    def absorbed_fraction_transformed(self) -> float:
        return self.absorbed_fraction_for_coeff(self.absorption_coeff_transformed)

    def blended_absorbed_fraction(self, conversion: float) -> float:
        a = np.clip(conversion, 0.0, 1.0)
        return (
            (1.0 - a) * self.absorbed_fraction_untransformed()
            + a * self.absorbed_fraction_transformed()
        )

    def source_profile(
        self,
        nodes: np.ndarray,
        alpha: np.ndarray,
        t: float,
        step_dt: float,
        m_matrix: csr_matrix,
    ) -> np.ndarray:
        """
        Returns a depth-averaged volumetric heat source [W/m^3].
        """
        x_pos, y_pos = self.beam_position(t)
        temporal_power = self.temporal_power(t, step_dt)

        dx = nodes[:, 0] - x_pos
        dy = nodes[:, 1] - y_pos
        r2 = dx * dx + dy * dy

        if self.gaussian_spatial:
            spatial_raw = np.exp(-r2 / (2.0 * self.sigma ** 2))
        else:
            radius2 = self.top_hat_radius() ** 2
            spatial_raw = (r2 <= radius2).astype(float)

            if np.all(spatial_raw <= 0.0):
                nearest = int(np.argmin(r2))
                spatial_raw[nearest] = 1.0

        spatial_integral = float(spmv(m_matrix, spatial_raw).sum())
        if spatial_integral > 1e-12:
            spatial_norm = spatial_raw / spatial_integral
        else:
            spatial_norm = spatial_raw

        thickness = max(self.film_thickness, np.finfo(float).eps)
        absorbed_base = self.absorbed_fraction_untransformed()
        absorbed_tx = self.absorbed_fraction_transformed()
        source_scale = temporal_power / thickness

        conversion = np.clip(alpha, 0.0, 1.0)
        absorbed_fraction = (1.0 - conversion) * absorbed_base + conversion * absorbed_tx
        return spatial_norm * absorbed_fraction * source_scale


def reaction_rate(temp: np.ndarray, alpha: np.ndarray, params: PhysicalParams) -> np.ndarray:
    """
    d(alpha)/dt = (1 - alpha) * A * exp(-Ea / (R T))
    """
    t_clipped = np.maximum(temp, 1e-6)
    k_rate = params.a_pre * np.exp(-params.ea / (params.r_gas * t_clipped))
    return (1.0 - alpha) * k_rate


def radiative_cooling_source(
    temp: np.ndarray,
    alpha: np.ndarray,
    params: PhysicalParams,
    film_thickness: float,
) -> np.ndarray:
    """
    Depth-averaged radiative cooling source [W/m^3] for one exposed surface.
    Negative means cooling.
    """
    if film_thickness <= 0.0:
        return np.zeros_like(temp)

    thickness = max(film_thickness, np.finfo(float).eps)
    ambient_t4 = params.t_inf ** 4
    source_scale = STEFAN_BOLTZMANN / thickness

    temperature = np.maximum(temp, 1e-6)
    conversion = np.clip(alpha, 0.0, 1.0)
    emissivity = np.clip(
        (1.0 - conversion) * params.emissivity
        + conversion * params.emissivity_transformed,
        0.0,
        1.0,
    )

    return -(emissivity * source_scale) * (temperature ** 4 - ambient_t4)


@dataclass
class SimParams:
    lxy: float = 200e-6
    nxy: int = 200
    t_final: float = 1e-8
    dt: float = 1e-10
    adaptive_time_stepping: bool = True
    save_interval: int = 100

    rho: float = 1100.0
    cp: float = 1500.0
    k: float = 0.3
    h_coeff: float = 10.0
    emissivity: float = 0.85
    emissivity_transformed: float = 0.85

    a_pre: float = 1e8
    ea: float = 8e4
    delta_h: float = 5e5

    pulse_energy: float = 1e-4
    sigma: float = 50e-6 / 2.35
    pulse_width: float = 1e-7
    gaussian_spatial: bool = True
    gaussian_temporal: bool = True
    pulse_rate: float = 20e3
    scan_speed: float = 0.1
    line_spacing: float = 40e-6
    scan_margin: float = 20e-6
    film_thickness: float = 10e-6
    absorption_coeff: float = 1e5
    absorption_coeff_transformed: float = 1e5

    t_init: float = 300.0
    t_inf: float = 300.0


@dataclass
class CgSolveReport:
    solution: np.ndarray
    iterations: int
    converged: bool
    residual_norm: float
    solve_secs: float


class FEMSimulation:
    def __init__(
        self,
        mesh: Mesh,
        params: PhysicalParams,
        adaptive_time_stepping: bool,
        laser: LaserParams,
        m_matrix: csr_matrix,
        k_matrix: csr_matrix,
        k_bound: csr_matrix,
        f_bound_coef: np.ndarray,
        temperature: np.ndarray,
        alpha: np.ndarray,
    ) -> None:
        self.mesh = mesh
        self.params = params
        self.adaptive_time_stepping = adaptive_time_stepping
        self.laser = laser
        self.m_matrix = m_matrix
        self.k_matrix = k_matrix
        self.k_bound = k_bound
        self.f_bound_coef = f_bound_coef
        self.temperature = temperature
        self.alpha = alpha

    @classmethod
    def new_with_params(cls, p: SimParams) -> "FEMSimulation":
        mesh = Mesh.new(p.lxy, p.lxy, p.nxy, p.nxy)
        m_matrix, k_matrix = assemble_system(mesh)

        params = PhysicalParams(
            rho=p.rho,
            cp=p.cp,
            k=p.k,
            h_coeff=p.h_coeff,
            emissivity=p.emissivity,
            emissivity_transformed=p.emissivity_transformed,
            a_pre=p.a_pre,
            ea=p.ea,
            r_gas=8.314,
            delta_h=p.delta_h,
            t_init=p.t_init,
            t_inf=p.t_inf,
        )

        peak_power = p.pulse_energy / p.pulse_width if p.pulse_width > 0.0 else 0.0
        laser = LaserParams.new(p.lxy, p.lxy)
        laser.power = peak_power
        laser.pulse_energy = p.pulse_energy
        laser.sigma = p.sigma
        laser.pulse_width = p.pulse_width
        laser.gaussian_spatial = p.gaussian_spatial
        laser.gaussian_temporal = p.gaussian_temporal
        laser.pulse_period = 1.0 / p.pulse_rate if p.pulse_rate > 0.0 else inf
        laser.scan_speed = p.scan_speed
        laser.line_spacing = p.line_spacing
        laser.scan_margin = p.scan_margin
        laser.film_thickness = p.film_thickness
        laser.absorption_coeff = p.absorption_coeff
        laser.absorption_coeff_transformed = p.absorption_coeff_transformed

        edges = mesh.boundary_edges(p.nxy, p.nxy)
        k_bound, f_bound_coef = compute_boundary_matrices(mesh, edges, params.h_coeff)

        temperature = np.full(mesh.num_nodes, params.t_init, dtype=float)
        alpha = np.zeros(mesh.num_nodes, dtype=float)

        return cls(
            mesh=mesh,
            params=params,
            adaptive_time_stepping=p.adaptive_time_stepping,
            laser=laser,
            m_matrix=m_matrix,
            k_matrix=k_matrix,
            k_bound=k_bound,
            f_bound_coef=f_bound_coef,
            temperature=temperature,
            alpha=alpha,
        )

    def integrated_volume_power(self, source: np.ndarray) -> float:
        return float(spmv(self.m_matrix, source).sum())

    def estimated_step_count(self, t_final: float, dt: float) -> int:
        if not (t_final > 0.0 and dt > 0.0):
            return 0
        return max(int(np.ceil(t_final / dt)), 1)

    def adaptive_step_size(
        self,
        time: float,
        t_final: float,
        dt_max: float,
        prev_dt: float,
        d_alpha: np.ndarray,
        s_laser_current: np.ndarray,
        s_radiation_current: np.ndarray,
    ) -> float:
        """
        This is heuristic adaptive stepping, not embedded RK error control.

        It reduces dt when:
        - the solver is near a laser pulse
        - reaction is changing quickly
        - the temperature source terms are large
        """
        remaining = max(t_final - time, 0.0)
        if remaining <= 0.0:
            return 0.0

        dt_cap = max(min(dt_max, remaining), np.finfo(float).eps)
        if not self.adaptive_time_stepping:
            return dt_cap

        dt = dt_cap

        if self.laser.gaussian_temporal and self.laser.pulse_width > 0.0:
            pulse_resolution_dt = max(self.laser.pulse_width / 6.0, np.finfo(float).eps)
        else:
            pulse_resolution_dt = dt_cap

        repetition_period = max(
            self.laser.pulse_period,
            self.laser.pulse_width,
            np.finfo(float).eps,
        )

        if self.laser.gaussian_temporal and self.laser.pulse_width > 0.0:
            pulse_sigma = max(
                self.laser.pulse_width / (2.0 * sqrt(2.0 * LN2)),
                np.finfo(float).eps,
            )
        else:
            pulse_sigma = np.finfo(float).eps

        t_local = time % repetition_period
        t_peak = 0.5 * min(self.laser.pulse_width, repetition_period) if self.laser.gaussian_temporal else 0.0

        if self.laser.gaussian_temporal and abs(t_local - t_peak) <= 3.0 * pulse_sigma:
            dt = min(dt, pulse_resolution_dt)
        elif (not self.laser.gaussian_temporal) and np.isfinite(self.laser.pulse_period):
            dt = min(dt, max(self.laser.pulse_period, np.finfo(float).eps))

        max_alpha_rate = float(np.max(np.abs(d_alpha)))
        if max_alpha_rate > 1e-12:
            dt = min(dt, 0.02 / max_alpha_rate)

        max_laser_source = float(np.max(np.abs(s_laser_current)))
        max_radiation_source = float(np.max(np.abs(s_radiation_current)))
        reaction_source_scale = self.params.delta_h * self.params.rho * max_alpha_rate
        rho_cp = max(self.params.rho * self.params.cp, np.finfo(float).eps)
        max_temp_rate = (max_laser_source + max_radiation_source + reaction_source_scale) / rho_cp
        if max_temp_rate > 1e-9:
            dt = min(dt, 2.0 / max_temp_rate)

        min_dt = max(dt_max * 1e-4, np.finfo(float).eps)
        if self.laser.gaussian_temporal and self.laser.pulse_width > 0.0:
            min_dt = min(min_dt, max(self.laser.pulse_width / 80.0, np.finfo(float).eps))
        elif (not self.laser.gaussian_temporal) and np.isfinite(self.laser.pulse_period):
            min_dt = min(min_dt, max(self.laser.pulse_period, np.finfo(float).eps))

        if prev_dt > 0.0:
            dt = min(dt, prev_dt * 1.6)

        return float(np.clip(dt, min(min_dt, remaining), max(remaining, np.finfo(float).eps)))

    def run_streaming_with_control(
        self,
        t_final: float,
        dt: float,
        save_interval: int,
        on_frame,
        on_progress,
        on_series_point,
        control,
    ) -> dict:
        """
        Main solver loop.

        Important pattern:
        - alpha is updated explicitly
        - temperature is solved implicitly
        """
        estimated_num_steps = self.estimated_step_count(t_final, dt)
        progress_interval = max(estimated_num_steps // 400, 1)

        initial_laser = self.laser.source_profile(
            self.mesh.nodes,
            self.alpha,
            t=0.0,
            step_dt=0.0,
            m_matrix=self.m_matrix,
        )

        on_frame(
            frame_index=0,
            time=0.0,
            temperature=self.temperature.copy(),
            alpha=self.alpha.copy(),
            laser=initial_laser.copy(),
            max_temp=float(self.temperature.max()),
            avg_conversion=float(self.alpha.mean()),
            progress=0.0,
        )

        on_progress(
            step=0,
            num_steps=estimated_num_steps,
            time=0.0,
            progress=0.0 if t_final > 0.0 else 1.0,
            max_temp=float(self.temperature.max()),
        )

        if not (t_final > 0.0 and dt > 0.0):
            return {
                "num_steps": 0,
                "completed_steps": 0,
                "cancelled": False,
                "solver": {"preconditioner": "Jacobi PCG"},
            }

        m_eff = scale_csmat(self.m_matrix, self.params.rho * self.params.cp)
        k_eff = scale_csmat(self.k_matrix, self.params.k)
        k_sum = k_eff + self.k_bound
        f_bound = self.f_bound_coef * self.params.t_inf
        reaction_heat_scale = self.params.delta_h * self.params.rho

        initial_reaction_rate = reaction_rate(self.temperature, self.alpha, self.params)
        initial_enthalpy_source = reaction_heat_scale * initial_reaction_rate
        initial_radiation = radiative_cooling_source(
            self.temperature,
            self.alpha,
            self.params,
            self.laser.film_thickness,
        )
        initial_convection_power = float((f_bound - spmv(self.k_bound, self.temperature)).sum())

        on_series_point(
            time=0.0,
            max_temp=float(self.temperature.max()),
            avg_conversion=float(self.alpha.mean()),
            laser_power=self.integrated_volume_power(initial_laser),
            enthalpy_power=self.integrated_volume_power(initial_enthalpy_source),
            convection_power=initial_convection_power,
            radiation_power=self.integrated_volume_power(initial_radiation),
        )

        frame_index = 1
        cancelled = False
        completed_steps = 0
        total_iterations = 0
        max_iterations = 0
        total_solve_secs = 0.0
        max_residual_norm = 0.0
        current_time = 0.0
        previous_dt = max(dt, np.finfo(float).eps)

        while current_time < t_final - np.finfo(float).eps:
            if not control(completed_steps, current_time):
                cancelled = True
                break

            # 1) Explicit reaction update
            d_alpha = reaction_rate(self.temperature, self.alpha, self.params)

            # 2) Estimate current forcing for timestep control
            s_laser_current = self.laser.source_profile(
                self.mesh.nodes,
                self.alpha,
                t=current_time,
                step_dt=previous_dt,
                m_matrix=self.m_matrix,
            )
            s_radiation_current = radiative_cooling_source(
                self.temperature,
                self.alpha,
                self.params,
                self.laser.film_thickness,
            )

            step_dt = self.adaptive_step_size(
                current_time,
                t_final,
                dt,
                previous_dt,
                d_alpha,
                s_laser_current,
                s_radiation_current,
            )

            next_step_dt = step_dt
            pulse_keyframe_time = self.laser.first_pulse_event_in_window(
                current_time,
                min(current_time + step_dt, t_final),
            )
            if pulse_keyframe_time is not None:
                next_step_dt = max(pulse_keyframe_time - current_time, np.finfo(float).eps)

            t = min(current_time + next_step_dt, t_final)

            self.alpha = self.alpha + next_step_dt * d_alpha
            self.alpha = np.clip(self.alpha, 0.0, 1.0)

            # 3) Build source terms at the new time
            s_laser = self.laser.source_profile(
                self.mesh.nodes,
                self.alpha,
                t=t,
                step_dt=next_step_dt,
                m_matrix=self.m_matrix,
            )
            s_radiation = radiative_cooling_source(
                self.temperature,
                self.alpha,
                self.params,
                self.laser.film_thickness,
            )

            s_total = s_laser + reaction_heat_scale * d_alpha + s_radiation
            f_source = spmv(self.m_matrix, s_total)

            # 4) Semi-implicit heat solve:
            #    (M rho cp + dt K_sum) T_new = M rho cp T_old + dt (F_source + F_bound)
            rhs = spmv(m_eff, self.temperature) + next_step_dt * (f_source + f_bound)
            lhs = m_eff + scale_csmat(k_sum, next_step_dt)

            solve_report = pcg_solve_jacobi(
                lhs,
                rhs,
                x0=self.temperature,
                tol=1e-10,
                max_iter=1000,
            )

            self.temperature = solve_report.solution
            completed_steps += 1
            current_time = t
            previous_dt = next_step_dt

            total_iterations += solve_report.iterations
            max_iterations = max(max_iterations, solve_report.iterations)
            total_solve_secs += solve_report.solve_secs
            max_residual_norm = max(max_residual_norm, solve_report.residual_norm)

            max_t = float(self.temperature.max())
            avg_a = float(self.alpha.mean())
            progress = np.clip(current_time / t_final, 0.0, 1.0) if t_final > 0.0 else 1.0

            reaction_rate_current = reaction_rate(self.temperature, self.alpha, self.params)
            enthalpy_source_current = reaction_heat_scale * reaction_rate_current
            radiation_current = radiative_cooling_source(
                self.temperature,
                self.alpha,
                self.params,
                self.laser.film_thickness,
            )
            convection_power = float((f_bound - spmv(self.k_bound, self.temperature)).sum())

            on_series_point(
                time=current_time,
                max_temp=max_t,
                avg_conversion=avg_a,
                laser_power=self.integrated_volume_power(s_laser),
                enthalpy_power=self.integrated_volume_power(enthalpy_source_current),
                convection_power=convection_power,
                radiation_power=self.integrated_volume_power(radiation_current),
            )

            should_save_frame = (
                completed_steps % save_interval == 0
                or current_time >= t_final - np.finfo(float).eps
                or pulse_keyframe_time is not None
            )
            if should_save_frame:
                on_frame(
                    frame_index=frame_index,
                    time=current_time,
                    temperature=self.temperature.copy(),
                    alpha=self.alpha.copy(),
                    laser=s_laser.copy(),
                    max_temp=max_t,
                    avg_conversion=avg_a,
                    progress=progress,
                )
                frame_index += 1

            if completed_steps > 0 and current_time > 0.0:
                avg_step = current_time / completed_steps
                dynamic_total_steps = max(
                    int(np.ceil(t_final / avg_step)),
                    completed_steps,
                    estimated_num_steps,
                )
            else:
                dynamic_total_steps = estimated_num_steps

            if completed_steps % progress_interval == 0 or progress >= 1.0 - 1e-12:
                on_progress(
                    step=completed_steps,
                    num_steps=max(dynamic_total_steps, 1),
                    time=current_time,
                    progress=progress,
                    max_temp=max_t,
                )

        solve_calls = completed_steps
        return {
            "num_steps": completed_steps if self.adaptive_time_stepping else estimated_num_steps,
            "completed_steps": completed_steps,
            "cancelled": cancelled,
            "solver": {
                "preconditioner": "Jacobi PCG",
                "solve_calls": solve_calls,
                "total_iterations": total_iterations,
                "max_iterations": max_iterations,
                "avg_iterations": total_iterations / solve_calls if solve_calls else 0.0,
                "total_solve_secs": total_solve_secs,
                "avg_solve_secs": total_solve_secs / solve_calls if solve_calls else 0.0,
                "max_residual_norm": max_residual_norm,
            },
        }


def jacobi_preconditioner(a: csr_matrix) -> np.ndarray:
    diag = a.diagonal()
    inv_diag = np.ones_like(diag)
    mask = np.abs(diag) > 1e-30
    inv_diag[mask] = 1.0 / diag[mask]
    return inv_diag


def apply_diagonal_preconditioner(inv_diag: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return inv_diag * residual


def pcg_solve_jacobi(
    a: csr_matrix,
    b: np.ndarray,
    x0: np.ndarray,
    tol: float,
    max_iter: int,
) -> CgSolveReport:
    """
    Jacobi-preconditioned conjugate gradient.
    This is the main temperature linear solver in the current backend.
    """
    solve_start = perf_counter()

    inv_diag = jacobi_preconditioner(a)
    x = x0.copy()
    r = b - spmv(a, x)
    residual_norm = sqrt(dot_product(r, r))

    if residual_norm < tol:
        return CgSolveReport(
            solution=x,
            iterations=0,
            converged=True,
            residual_norm=residual_norm,
            solve_secs=perf_counter() - solve_start,
        )

    z = apply_diagonal_preconditioner(inv_diag, r)
    p = z.copy()
    rz_old = dot_product(r, z)
    iterations = 0

    for k in range(max_iter):
        iterations = k + 1
        ap = spmv(a, p)
        p_dot_ap = dot_product(p, ap)
        if abs(p_dot_ap) < 1e-30:
            break

        alpha = rz_old / p_dot_ap
        x = x + alpha * p
        r = r - alpha * ap

        residual_norm = sqrt(dot_product(r, r))
        if residual_norm < tol:
            break

        z = apply_diagonal_preconditioner(inv_diag, r)
        rz_new = dot_product(r, z)
        if abs(rz_old) < 1e-30:
            break

        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return CgSolveReport(
        solution=x,
        iterations=iterations,
        converged=(residual_norm < tol),
        residual_norm=residual_norm,
        solve_secs=perf_counter() - solve_start,
    )
```

## What this leaves out

This translation skips a few backend-adjacent pieces on purpose:

- Tauri command handlers in `tauri-app/src-tauri/src/main.rs`
- binary run storage and frame streaming
- snapshot management
- convergence-study orchestration
- UI-specific serialization details

If you want, I can also make a second document that rewrites the Tauri-side Rust backend in the same Python-like style.
