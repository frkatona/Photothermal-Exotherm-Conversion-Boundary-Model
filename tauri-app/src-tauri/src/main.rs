use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_ENGINE};
use fem_heat_reaction::{FEMSimulation, MeshData, RunSummary, SimFrame, SimParams, SimProgress, SimTimeSeriesPoint, SolverStats};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fs::{self, File};
use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager};

struct SimRuntimeState {
    running: bool,
    paused: bool,
    cancel_requested: bool,
}

struct SimControl {
    state: Mutex<SimRuntimeState>,
    condvar: Condvar,
}

#[derive(Serialize, Clone)]
struct SimProgressEvent {
    step: usize,
    num_steps: usize,
    progress: f64,
    sim_time: f64,
    elapsed_secs: f64,
    eta_secs: Option<f64>,
}

#[derive(Serialize, Clone)]
struct SimRunReadyEvent {
    run_id: String,
    frame_count: usize,
    elapsed_secs: f64,
    storage_bytes: u64,
    solver: SolverStats,
}

#[derive(Serialize, Clone)]
struct SimpleMessageEvent {
    message: String,
}

#[derive(Serialize, Clone)]
struct ProjectStats {
    folder_size_bytes: u64,
}

#[derive(Serialize, Deserialize, Default)]
struct ParamStore {
    last_used: Option<SimParams>,
    presets: Vec<NamedPreset>,
}

#[derive(Serialize, Deserialize, Clone)]
struct NamedPreset {
    name: String,
    params: SimParams,
}

#[derive(Serialize, Deserialize, Clone)]
struct StoredFrameMeta {
    frame_index: usize,
    time: f64,
    max_temp: f64,
    avg_conversion: f64,
    progress: f64,
}

#[derive(Serialize, Deserialize, Clone)]
struct StoredRunMeta {
    run_id: String,
    created_at_ms: u64,
    params: SimParams,
    mesh: MeshData,
    frames: Vec<StoredFrameMeta>,
    temp_range: [f64; 2],
    conv_range: [f64; 2],
    laser_range: [f64; 2],
    elapsed_secs: f64,
    storage_bytes: u64,
    solver: SolverStats,
}

#[derive(Serialize, Deserialize, Clone, Default)]
struct StoredRunTimeSeries {
    times: Vec<f64>,
    max_temps: Vec<f64>,
    avg_conversions: Vec<f64>,
    laser_powers: Vec<f64>,
    enthalpy_powers: Vec<f64>,
    convection_powers: Vec<f64>,
    radiation_powers: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone)]
struct SnapshotManifest {
    id: String,
    created_at_ms: u64,
    params: SimParams,
    run_id: Option<String>,
}

#[derive(Serialize, Clone)]
struct SnapshotListItem {
    id: String,
    created_at_ms: u64,
}

#[derive(Serialize, Clone)]
struct SnapshotDetail {
    id: String,
    created_at_ms: u64,
    params: SimParams,
    run_id: Option<String>,
    laser_png_base64: String,
    temp_png_base64: String,
    conv_png_base64: String,
    metrics_png_base64: String,
    sources_png_base64: String,
}

#[derive(Deserialize)]
struct SnapshotPayload {
    params: SimParams,
    run_id: Option<String>,
    laser_png_base64: String,
    temp_png_base64: String,
    conv_png_base64: String,
    metrics_png_base64: String,
    sources_png_base64: String,
}

#[derive(Serialize, Clone)]
struct ConvergenceCaseResult {
    label: String,
    nxy: usize,
    dt: f64,
    elapsed_secs: f64,
    final_max_temp: f64,
    final_avg_conversion: f64,
    avg_cg_iterations: f64,
    max_cg_iterations: usize,
}

#[derive(Serialize, Clone)]
struct ConvergenceStudyResult {
    mesh_cases: Vec<ConvergenceCaseResult>,
    dt_cases: Vec<ConvergenceCaseResult>,
}

fn current_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn get_store_path(app: &AppHandle) -> PathBuf {
    let app_data = app.path().app_data_dir().unwrap_or_else(|_| PathBuf::from("."));
    let _ = fs::create_dir_all(&app_data);
    app_data.join("param_store.json")
}

fn read_store(app: &AppHandle) -> ParamStore {
    let path = get_store_path(app);
    if let Ok(data) = fs::read_to_string(&path) {
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        ParamStore::default()
    }
}

fn write_store(app: &AppHandle, store: &ParamStore) -> Result<(), String> {
    let path = get_store_path(app);
    let json = serde_json::to_string_pretty(store).map_err(|e| e.to_string())?;
    fs::write(&path, json).map_err(|e| e.to_string())
}

fn project_root_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .unwrap_or(manifest_dir)
}

fn simulation_files_dir(app: &AppHandle) -> PathBuf {
    let app_data = app.path().app_data_dir().unwrap_or_else(|_| PathBuf::from("."));
    let root = app_data.join("simulation_files");
    let _ = fs::create_dir_all(&root);
    root
}

fn runs_dir(app: &AppHandle) -> PathBuf {
    let dir = simulation_files_dir(app).join("runs");
    let _ = fs::create_dir_all(&dir);
    dir
}

fn snapshots_dir(app: &AppHandle) -> PathBuf {
    let dir = simulation_files_dir(app).join("snapshots");
    let _ = fs::create_dir_all(&dir);
    dir
}

fn videos_dir(app: &AppHandle) -> PathBuf {
    let dir = simulation_files_dir(app).join("videos");
    let _ = fs::create_dir_all(&dir);
    dir
}

fn folder_size_bytes(path: &Path) -> io::Result<u64> {
    let mut total = 0_u64;
    let mut stack = vec![path.to_path_buf()];

    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                stack.push(entry.path());
            } else if file_type.is_file() {
                total = total.saturating_add(entry.metadata()?.len());
            }
        }
    }

    Ok(total)
}

fn write_bin<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let file = File::create(path).map_err(|e| e.to_string())?;
    bincode::serialize_into(file, value).map_err(|e| e.to_string())
}

fn read_bin<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    bincode::deserialize_from(file).map_err(|e| e.to_string())
}

fn frame_path(run_dir: &Path, frame_index: usize) -> PathBuf {
    run_dir.join("frames").join(format!("frame_{frame_index:06}.bin"))
}

fn run_meta_path(run_dir: &Path) -> PathBuf {
    run_dir.join("run_meta.bin")
}

fn run_timeseries_path(run_dir: &Path) -> PathBuf {
    run_dir.join("run_timeseries.bin")
}

fn snapshot_manifest_path(snapshot_dir: &Path) -> PathBuf {
    snapshot_dir.join("snapshot.bin")
}

fn open_in_file_manager(path: &Path) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        Command::new("explorer")
            .arg(path)
            .spawn()
            .map_err(|e| e.to_string())?;
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(path)
            .spawn()
            .map_err(|e| e.to_string())?;
        return Ok(());
    }

    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        Command::new("xdg-open")
            .arg(path)
            .spawn()
            .map_err(|e| e.to_string())?;
        Ok(())
    }
}

fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    let b64 = if let Some(pos) = input.find(',') {
        &input[pos + 1..]
    } else {
        input
    };
    BASE64_ENGINE.decode(b64).map_err(|e| e.to_string())
}

fn png_file_to_data_uri(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    Ok(format!("data:image/png;base64,{}", BASE64_ENGINE.encode(bytes)))
}

fn with_control_state<T, F>(control: &Arc<SimControl>, mutator: F) -> Result<T, String>
where
    F: FnOnce(&mut SimRuntimeState) -> Result<T, String>,
{
    let mut state = control.state.lock().map_err(|e| e.to_string())?;
    mutator(&mut state)
}

fn reset_control_state(control: &Arc<SimControl>) {
    if let Ok(mut state) = control.state.lock() {
        state.running = false;
        state.paused = false;
        state.cancel_requested = false;
        control.condvar.notify_all();
    }
}

fn run_case(mut params: SimParams, label: &str) -> Result<ConvergenceCaseResult, String> {
    let mut sim = FEMSimulation::new_with_params(&params);
    let start = Instant::now();
    let last_frame = RefCell::new(None::<SimFrame>);
    params.save_interval = 1;
    let summary = sim.run_streaming(params.t_final, params.dt, params.save_interval, |frame| {
        *last_frame.borrow_mut() = Some(frame);
    });
    let elapsed_secs = start.elapsed().as_secs_f64();
    let final_frame = last_frame
        .into_inner()
        .ok_or_else(|| "Convergence study produced no frames".to_string())?;

    Ok(ConvergenceCaseResult {
        label: label.to_string(),
        nxy: params.nxy,
        dt: params.dt,
        elapsed_secs,
        final_max_temp: final_frame.max_temp,
        final_avg_conversion: final_frame.avg_conversion,
        avg_cg_iterations: summary.solver.avg_iterations,
        max_cg_iterations: summary.solver.max_iterations,
    })
}

// === Commands ===

#[tauri::command]
fn get_default_params() -> SimParams {
    SimParams::default()
}

#[tauri::command]
fn load_last_params(app: AppHandle) -> Option<SimParams> {
    read_store(&app).last_used
}

#[tauri::command]
fn save_last_params(app: AppHandle, params: SimParams) -> Result<(), String> {
    let mut store = read_store(&app);
    store.last_used = Some(params);
    write_store(&app, &store)
}

#[tauri::command]
fn save_preset(app: AppHandle, name: String, params: SimParams) -> Result<(), String> {
    let mut store = read_store(&app);
    if let Some(existing) = store.presets.iter_mut().find(|preset| preset.name == name) {
        existing.params = params;
    } else {
        store.presets.push(NamedPreset { name, params });
    }
    write_store(&app, &store)
}

#[tauri::command]
fn list_presets(app: AppHandle) -> Vec<NamedPreset> {
    read_store(&app).presets
}

#[tauri::command]
fn delete_preset(app: AppHandle, name: String) -> Result<(), String> {
    let mut store = read_store(&app);
    store.presets.retain(|preset| preset.name != name);
    write_store(&app, &store)
}

#[tauri::command]
fn get_project_stats() -> Result<ProjectStats, String> {
    let root = project_root_dir();
    let folder_size_bytes = folder_size_bytes(&root).map_err(|e| e.to_string())?;
    Ok(ProjectStats { folder_size_bytes })
}

#[tauri::command]
fn open_simulation_files_folder(app: AppHandle) -> Result<String, String> {
    let folder = simulation_files_dir(&app);
    open_in_file_manager(&folder)?;
    Ok(folder.to_string_lossy().to_string())
}

#[tauri::command]
fn load_run_metadata(app: AppHandle, run_id: String) -> Result<StoredRunMeta, String> {
    let run_dir = runs_dir(&app).join(run_id);
    read_bin(&run_meta_path(&run_dir))
}

#[tauri::command]
fn load_run_time_series(app: AppHandle, run_id: String) -> Result<StoredRunTimeSeries, String> {
    let run_dir = runs_dir(&app).join(&run_id);
    let path = run_timeseries_path(&run_dir);
    if path.exists() {
        return read_bin(&path);
    }

    let meta: StoredRunMeta = read_bin(&run_meta_path(&run_dir))?;
    Ok(StoredRunTimeSeries {
        times: meta.frames.iter().map(|frame| frame.time).collect(),
        max_temps: meta.frames.iter().map(|frame| frame.max_temp).collect(),
        avg_conversions: meta.frames.iter().map(|frame| frame.avg_conversion).collect(),
        laser_powers: Vec::new(),
        enthalpy_powers: Vec::new(),
        convection_powers: Vec::new(),
        radiation_powers: Vec::new(),
    })
}

#[tauri::command]
fn load_run_frame(app: AppHandle, run_id: String, frame_index: usize) -> Result<SimFrame, String> {
    let run_dir = runs_dir(&app).join(run_id);
    read_bin(&frame_path(&run_dir, frame_index))
}

#[tauri::command]
fn pause_simulation(control: tauri::State<'_, Arc<SimControl>>) -> Result<(), String> {
    with_control_state(&control, |state| {
        if !state.running {
            return Err("No simulation is currently running".to_string());
        }
        state.paused = true;
        Ok(())
    })
}

#[tauri::command]
fn resume_simulation(control: tauri::State<'_, Arc<SimControl>>) -> Result<(), String> {
    let result = with_control_state(&control, |state| {
        if !state.running {
            return Err("No simulation is currently running".to_string());
        }
        state.paused = false;
        Ok(())
    });
    control.condvar.notify_all();
    result
}

#[tauri::command]
fn cancel_simulation(control: tauri::State<'_, Arc<SimControl>>) -> Result<(), String> {
    let result = with_control_state(&control, |state| {
        if !state.running {
            return Err("No simulation is currently running".to_string());
        }
        state.cancel_requested = true;
        state.paused = false;
        Ok(())
    });
    control.condvar.notify_all();
    result
}

#[tauri::command]
fn run_simulation(
    app: AppHandle,
    control: tauri::State<'_, Arc<SimControl>>,
    params: SimParams,
) -> Result<(), String> {
    with_control_state(&control, |state| {
        if state.running {
            return Err("Simulation already running".to_string());
        }
        state.running = true;
        state.paused = false;
        state.cancel_requested = false;
        Ok(())
    })?;

    let app_handle = app.clone();
    let control_arc = Arc::clone(&control.inner().clone());

    thread::spawn(move || {
        let created_at_ms = current_unix_ms();
        let run_id = format!("run_{created_at_ms}");
        let run_dir = runs_dir(&app_handle).join(&run_id);
        let frames_dir = run_dir.join("frames");

        let thread_result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
            fs::create_dir_all(&frames_dir).map_err(|e| e.to_string())?;

            let start = Instant::now();
            let mut sim = FEMSimulation::new_with_params(&params);
            let mesh = sim.mesh_data();

            let frame_metas = RefCell::new(Vec::<StoredFrameMeta>::new());
            let time_series = RefCell::new(StoredRunTimeSeries::default());
            let write_error = RefCell::new(None::<String>);
            let temp_min = RefCell::new(f64::INFINITY);
            let temp_max = RefCell::new(f64::NEG_INFINITY);
            let conv_max = RefCell::new(0.0_f64);
            let laser_max = RefCell::new(0.0_f64);

            let progress_handle = app_handle.clone();
            let control_for_loop = Arc::clone(&control_arc);
            let mut last_progress_emit = start.checked_sub(Duration::from_secs(1)).unwrap_or(start);
            let mut last_progress_value = 0.0_f64;

            let summary: RunSummary = sim.run_streaming_with_control(
                params.t_final,
                params.dt,
                params.save_interval,
                |frame: SimFrame| {
                    if write_error.borrow().is_some() {
                        return;
                    }

                    let path = frame_path(&run_dir, frame.frame_index);
                    if let Err(err) = write_bin(&path, &frame) {
                        *write_error.borrow_mut() = Some(err);
                        return;
                    }

                    if let Some(local_min) = frame.temperature.iter().copied().reduce(f64::min) {
                        let mut min_ref = temp_min.borrow_mut();
                        *min_ref = (*min_ref).min(local_min);
                    }
                    if let Some(local_max) = frame.temperature.iter().copied().reduce(f64::max) {
                        let mut max_ref = temp_max.borrow_mut();
                        *max_ref = (*max_ref).max(local_max);
                    }
                    if let Some(local_conv_max) = frame.alpha.iter().copied().reduce(f64::max) {
                        let mut conv_ref = conv_max.borrow_mut();
                        *conv_ref = (*conv_ref).max(local_conv_max);
                    }
                    if let Some(local_laser_max) = frame.laser.iter().copied().reduce(f64::max) {
                        let mut laser_ref = laser_max.borrow_mut();
                        *laser_ref = (*laser_ref).max(local_laser_max);
                    }

                    frame_metas.borrow_mut().push(StoredFrameMeta {
                        frame_index: frame.frame_index,
                        time: frame.time,
                        max_temp: frame.max_temp,
                        avg_conversion: frame.avg_conversion,
                        progress: frame.progress,
                    });
                },
                |progress: SimProgress| {
                    let now = Instant::now();
                    let elapsed_secs = start.elapsed().as_secs_f64();
                    let progress_delta = progress.progress - last_progress_value;
                    let should_emit = progress.step == 0
                        || progress.step == progress.num_steps
                        || now.duration_since(last_progress_emit) >= Duration::from_millis(120)
                        || progress_delta >= 0.01;

                    if !should_emit {
                        return;
                    }

                    let eta_secs = if progress.progress > 0.0 {
                        Some(elapsed_secs * (1.0 - progress.progress) / progress.progress.max(f64::EPSILON))
                    } else {
                        None
                    };

                    let _ = progress_handle.emit(
                        "sim-progress",
                        SimProgressEvent {
                            step: progress.step,
                            num_steps: progress.num_steps,
                            progress: progress.progress,
                            sim_time: progress.time,
                            elapsed_secs,
                            eta_secs,
                        },
                    );

                    last_progress_emit = now;
                    last_progress_value = progress.progress;
                },
                |point: SimTimeSeriesPoint| {
                    let mut series = time_series.borrow_mut();
                    series.times.push(point.time);
                    series.max_temps.push(point.max_temp);
                    series.avg_conversions.push(point.avg_conversion);
                    series.laser_powers.push(point.laser_power);
                    series.enthalpy_powers.push(point.enthalpy_power);
                    series.convection_powers.push(point.convection_power);
                    series.radiation_powers.push(point.radiation_power);
                },
                |step, time| {
                    if write_error.borrow().is_some() {
                        return false;
                    }

                    let mut state = match control_for_loop.state.lock() {
                        Ok(guard) => guard,
                        Err(_) => return false,
                    };

                    loop {
                        if state.cancel_requested {
                            return false;
                        }
                        if !state.paused {
                            break;
                        }
                        let _ = progress_handle.emit(
                            "sim-progress",
                            SimProgressEvent {
                                step,
                                num_steps: (params.t_final / params.dt) as usize,
                                progress: step as f64 / ((params.t_final / params.dt) as usize).max(1) as f64,
                                sim_time: time,
                                elapsed_secs: start.elapsed().as_secs_f64(),
                                eta_secs: None,
                            },
                        );
                        state = match control_for_loop.condvar.wait(state) {
                            Ok(guard) => guard,
                            Err(_) => return false,
                        };
                    }

                    true
                },
            );

            if let Some(err) = write_error.into_inner() {
                return Err(err);
            }

            if summary.cancelled {
                let _ = fs::remove_dir_all(&run_dir);
                let _ = app_handle.emit(
                    "sim-run-cancelled",
                    SimpleMessageEvent {
                        message: "Simulation cancelled".to_string(),
                    },
                );
                return Ok(());
            }

            let elapsed_secs = start.elapsed().as_secs_f64();
            let mut meta = StoredRunMeta {
                run_id: run_id.clone(),
                created_at_ms,
                params: params.clone(),
                mesh,
                frames: frame_metas.into_inner(),
                temp_range: [*temp_min.borrow(), *temp_max.borrow()],
                conv_range: [0.0, *conv_max.borrow()],
                laser_range: [0.0, *laser_max.borrow()],
                elapsed_secs,
                storage_bytes: 0,
                solver: summary.solver.clone(),
            };

            if !meta.temp_range[0].is_finite() {
                meta.temp_range[0] = params.t_init;
            }
            if !meta.temp_range[1].is_finite() {
                meta.temp_range[1] = params.t_init;
            }
            meta.conv_range[1] = meta.conv_range[1].max(0.01);
            meta.laser_range[1] = meta.laser_range[1].max(1.0);

            write_bin(&run_meta_path(&run_dir), &meta)?;
            write_bin(&run_timeseries_path(&run_dir), &time_series.into_inner())?;
            meta.storage_bytes = folder_size_bytes(&run_dir).map_err(|e| e.to_string())?;
            write_bin(&run_meta_path(&run_dir), &meta)?;

            let _ = app_handle.emit(
                "sim-run-ready",
                SimRunReadyEvent {
                    run_id,
                    frame_count: meta.frames.len(),
                    elapsed_secs,
                    storage_bytes: meta.storage_bytes,
                    solver: meta.solver,
                },
            );

            Ok(())
        }));

        match thread_result {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                let _ = app_handle.emit("sim-run-error", SimpleMessageEvent { message: err });
            }
            Err(_) => {
                let _ = app_handle.emit(
                    "sim-run-error",
                    SimpleMessageEvent {
                        message: "Simulation worker panicked before completing".to_string(),
                    },
                );
            }
        }

        reset_control_state(&control_arc);
    });

    Ok(())
}

#[tauri::command]
fn save_video_frames(app: AppHandle, frames_png_base64: Vec<String>, output_name: String) -> Result<String, String> {
    let videos_dir = videos_dir(&app);
    let frames_dir = videos_dir.join(format!("{}_frames", &output_name));
    let _ = fs::create_dir_all(&frames_dir);

    for (i, b64) in frames_png_base64.iter().enumerate() {
        let bytes = base64_decode(b64)?;
        let frame_path = frames_dir.join(format!("frame_{i:04}.png"));
        fs::write(&frame_path, bytes).map_err(|e| e.to_string())?;
    }

    let output_path = videos_dir.join(format!("{output_name}.mp4"));
    let result = Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate",
            "30",
            "-i",
            &frames_dir.join("frame_%04d.png").to_string_lossy(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            &output_path.to_string_lossy(),
        ])
        .output();

    let _ = fs::remove_dir_all(&frames_dir);

    match result {
        Ok(output) if output.status.success() => Ok(output_path.to_string_lossy().to_string()),
        Ok(output) => Err(format!("ffmpeg failed: {}", String::from_utf8_lossy(&output.stderr))),
        Err(e) => Err(format!("Could not run ffmpeg (is it installed?): {e}")),
    }
}

#[tauri::command]
fn save_snapshot(app: AppHandle, payload: SnapshotPayload) -> Result<SnapshotListItem, String> {
    let created_at_ms = current_unix_ms();
    let snapshot_id = format!("snapshot_{created_at_ms}");
    let snapshot_dir = snapshots_dir(&app).join(&snapshot_id);
    fs::create_dir_all(&snapshot_dir).map_err(|e| e.to_string())?;

    fs::write(snapshot_dir.join("laser.png"), base64_decode(&payload.laser_png_base64)?).map_err(|e| e.to_string())?;
    fs::write(snapshot_dir.join("temp.png"), base64_decode(&payload.temp_png_base64)?).map_err(|e| e.to_string())?;
    fs::write(snapshot_dir.join("conv.png"), base64_decode(&payload.conv_png_base64)?).map_err(|e| e.to_string())?;
    fs::write(snapshot_dir.join("metrics.png"), base64_decode(&payload.metrics_png_base64)?).map_err(|e| e.to_string())?;
    fs::write(snapshot_dir.join("sources.png"), base64_decode(&payload.sources_png_base64)?).map_err(|e| e.to_string())?;

    let manifest = SnapshotManifest {
        id: snapshot_id.clone(),
        created_at_ms,
        params: payload.params,
        run_id: payload.run_id,
    };
    write_bin(&snapshot_manifest_path(&snapshot_dir), &manifest)?;

    Ok(SnapshotListItem {
        id: snapshot_id,
        created_at_ms,
    })
}

#[tauri::command]
fn list_snapshots(app: AppHandle) -> Result<Vec<SnapshotListItem>, String> {
    let mut snapshots = Vec::new();
    for entry in fs::read_dir(snapshots_dir(&app)).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        if !entry.file_type().map_err(|e| e.to_string())?.is_dir() {
            continue;
        }
        let manifest_path = snapshot_manifest_path(&entry.path());
        if !manifest_path.exists() {
            continue;
        }
        let manifest: SnapshotManifest = read_bin(&manifest_path)?;
        snapshots.push(SnapshotListItem {
            id: manifest.id,
            created_at_ms: manifest.created_at_ms,
        });
    }
    snapshots.sort_by(|a, b| b.created_at_ms.cmp(&a.created_at_ms));
    Ok(snapshots)
}

#[tauri::command]
fn load_snapshot(app: AppHandle, snapshot_id: String) -> Result<SnapshotDetail, String> {
    let snapshot_dir = snapshots_dir(&app).join(&snapshot_id);
    let manifest: SnapshotManifest = read_bin(&snapshot_manifest_path(&snapshot_dir))?;
    let sources_path = snapshot_dir.join("sources.png");
    Ok(SnapshotDetail {
        id: manifest.id,
        created_at_ms: manifest.created_at_ms,
        params: manifest.params,
        run_id: manifest.run_id,
        laser_png_base64: png_file_to_data_uri(&snapshot_dir.join("laser.png"))?,
        temp_png_base64: png_file_to_data_uri(&snapshot_dir.join("temp.png"))?,
        conv_png_base64: png_file_to_data_uri(&snapshot_dir.join("conv.png"))?,
        metrics_png_base64: png_file_to_data_uri(&snapshot_dir.join("metrics.png"))?,
        sources_png_base64: if sources_path.exists() {
            png_file_to_data_uri(&sources_path)?
        } else {
            png_file_to_data_uri(&snapshot_dir.join("metrics.png"))?
        },
    })
}

#[tauri::command]
fn delete_snapshot(app: AppHandle, snapshot_id: String) -> Result<(), String> {
    let snapshot_dir = snapshots_dir(&app).join(snapshot_id);
    fs::remove_dir_all(snapshot_dir).map_err(|e| e.to_string())
}

#[tauri::command]
fn run_convergence_study(
    control: tauri::State<'_, Arc<SimControl>>,
    params: SimParams,
) -> Result<ConvergenceStudyResult, String> {
    with_control_state(&control, |state| {
        if state.running {
            return Err("A simulation is already running".to_string());
        }
        state.running = true;
        state.paused = false;
        state.cancel_requested = false;
        Ok(())
    })?;

    let result = (|| {
        let coarse_nxy = (params.nxy / 2).max(8);
        let fine_nxy = (params.nxy.saturating_mul(2)).max(params.nxy + 1);

        let mut mesh_cases = Vec::new();
        let mut mesh_params = params.clone();
        mesh_params.nxy = coarse_nxy;
        mesh_cases.push(run_case(mesh_params, "Coarse mesh (0.5x)")?);
        mesh_cases.push(run_case(params.clone(), "Base mesh (1x)")?);
        let mut fine_mesh_params = params.clone();
        fine_mesh_params.nxy = fine_nxy;
        mesh_cases.push(run_case(fine_mesh_params, "Fine mesh (2x)")?);

        let mut dt_cases = Vec::new();
        let mut coarse_dt_params = params.clone();
        coarse_dt_params.dt *= 2.0;
        dt_cases.push(run_case(coarse_dt_params, "Coarse dt (2x)")?);
        dt_cases.push(run_case(params.clone(), "Base dt (1x)")?);
        let mut fine_dt_params = params.clone();
        fine_dt_params.dt *= 0.5;
        dt_cases.push(run_case(fine_dt_params, "Fine dt (0.5x)")?);

        Ok(ConvergenceStudyResult { mesh_cases, dt_cases })
    })();

    reset_control_state(&control);
    result
}

fn main() {
    tauri::Builder::default()
        .manage(Arc::new(SimControl {
            state: Mutex::new(SimRuntimeState {
                running: false,
                paused: false,
                cancel_requested: false,
            }),
            condvar: Condvar::new(),
        }))
        .invoke_handler(tauri::generate_handler![
            get_default_params,
            load_last_params,
            save_last_params,
            save_preset,
            list_presets,
            delete_preset,
            get_project_stats,
            open_simulation_files_folder,
            load_run_metadata,
            load_run_time_series,
            load_run_frame,
            pause_simulation,
            resume_simulation,
            cancel_simulation,
            run_simulation,
            save_video_frames,
            save_snapshot,
            list_snapshots,
            load_snapshot,
            delete_snapshot,
            run_convergence_study,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
