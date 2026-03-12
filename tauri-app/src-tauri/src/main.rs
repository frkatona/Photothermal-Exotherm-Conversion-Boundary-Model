use tauri::{AppHandle, Emitter, Manager};
use serde::{Serialize, Deserialize};
use std::sync::{Arc, Mutex};
use std::thread;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use fem_heat_reaction::{FEMSimulation, SimParams, SimFrame, SimProgress, MeshData};

/// State to track whether a simulation is running
struct SimState {
    running: bool,
}

/// Batch result: mesh + all frames at once
#[derive(Serialize, Clone)]
struct SimBatchResult {
    mesh: MeshData,
    frames: Vec<SimFrame>,
    elapsed_secs: f64,
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
struct ProjectStats {
    folder_size_bytes: u64,
}

/// Saved parameter presets file structure
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

fn folder_size_bytes(path: &PathBuf) -> io::Result<u64> {
    let mut total = 0_u64;
    let mut stack = vec![path.clone()];

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
    // Replace existing preset with same name or add new
    if let Some(existing) = store.presets.iter_mut().find(|p| p.name == name) {
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
    store.presets.retain(|p| p.name != name);
    write_store(&app, &store)
}

#[tauri::command]
fn get_project_stats() -> Result<ProjectStats, String> {
    let root = project_root_dir();
    let folder_size_bytes = folder_size_bytes(&root).map_err(|e| e.to_string())?;
    Ok(ProjectStats { folder_size_bytes })
}

#[tauri::command]
fn run_simulation(app: AppHandle, state: tauri::State<'_, Arc<Mutex<SimState>>>, params: SimParams) -> Result<(), String> {
    {
        let mut sim_state = state.lock().map_err(|e| e.to_string())?;
        if sim_state.running {
            return Err("Simulation already running".to_string());
        }
        sim_state.running = true;
    }
    
    let app_handle = app.clone();
    let state_arc = (*state).clone();
    
    thread::spawn(move || {
        let start = Instant::now();
        let mut sim = FEMSimulation::new_with_params(&params);
        let mesh = sim.mesh_data();
        
        let mut frames: Vec<SimFrame> = Vec::new();
        let progress_handle = app_handle.clone();
        let mut last_progress_emit = start.checked_sub(Duration::from_secs(1)).unwrap_or(start);
        let mut last_progress_value = 0.0_f64;

        sim.run_streaming_with_progress(
            params.t_final,
            params.dt,
            params.save_interval,
            |frame: SimFrame| {
                frames.push(frame);
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

                let _ = progress_handle.emit("sim-progress", SimProgressEvent {
                    step: progress.step,
                    num_steps: progress.num_steps,
                    progress: progress.progress,
                    sim_time: progress.time,
                    elapsed_secs,
                    eta_secs,
                });

                last_progress_emit = now;
                last_progress_value = progress.progress;
            },
        );
        
        let elapsed = start.elapsed().as_secs_f64();
        let _ = app_handle.emit("sim-batch-result", SimBatchResult {
            mesh,
            frames,
            elapsed_secs: elapsed,
        });
        
        if let Ok(mut s) = state_arc.lock() {
            s.running = false;
        }
    });
    
    Ok(())
}

#[tauri::command]
fn save_video_frames(app: AppHandle, frames_png_base64: Vec<String>, output_name: String) -> Result<String, String> {
    let app_data = app.path().app_data_dir().unwrap_or_else(|_| PathBuf::from("."));
    let videos_dir = app_data.join("videos");
    let _ = fs::create_dir_all(&videos_dir);
    
    // Save individual frame PNGs for ffmpeg
    let frames_dir = videos_dir.join(format!("{}_frames", &output_name));
    let _ = fs::create_dir_all(&frames_dir);
    
    for (i, b64) in frames_png_base64.iter().enumerate() {
        let bytes = base64_decode(b64)?;
        let frame_path = frames_dir.join(format!("frame_{:04}.png", i));
        fs::write(&frame_path, bytes).map_err(|e| e.to_string())?;
    }
    
    let output_path = videos_dir.join(format!("{}.mp4", &output_name));
    
    // Try ffmpeg
    let result = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate", "30",
            "-i", &frames_dir.join("frame_%04d.png").to_string_lossy(),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            &output_path.to_string_lossy(),
        ])
        .output();
    
    // Clean up frames directory
    let _ = fs::remove_dir_all(&frames_dir);
    
    match result {
        Ok(output) if output.status.success() => {
            Ok(output_path.to_string_lossy().to_string())
        }
        Ok(output) => {
            Err(format!("ffmpeg failed: {}", String::from_utf8_lossy(&output.stderr)))
        }
        Err(e) => Err(format!("Could not run ffmpeg (is it installed?): {}", e))
    }
}

fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    // Strip data URI prefix if present
    let b64 = if let Some(pos) = input.find(",") {
        &input[pos + 1..]
    } else {
        input
    };
    
    // Simple base64 decode
    let mut bytes = Vec::new();
    let chars: Vec<u8> = b64.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    let lut = |c: u8| -> Result<u8, String> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            b'=' => Ok(0),
            _ => Err(format!("Invalid base64 char: {}", c as char)),
        }
    };
    
    for chunk in chars.chunks(4) {
        if chunk.len() < 4 { break; }
        let a = lut(chunk[0])?;
        let b = lut(chunk[1])?;
        let c = lut(chunk[2])?;
        let d = lut(chunk[3])?;
        bytes.push((a << 2) | (b >> 4));
        if chunk[2] != b'=' { bytes.push((b << 4) | (c >> 2)); }
        if chunk[3] != b'=' { bytes.push((c << 6) | d); }
    }
    
    Ok(bytes)
}

fn main() {
    tauri::Builder::default()
        .manage(Arc::new(Mutex::new(SimState { running: false })))
        .invoke_handler(tauri::generate_handler![
            get_default_params,
            load_last_params,
            save_last_params,
            save_preset,
            list_presets,
            delete_preset,
            get_project_stats,
            run_simulation,
            save_video_frames,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
