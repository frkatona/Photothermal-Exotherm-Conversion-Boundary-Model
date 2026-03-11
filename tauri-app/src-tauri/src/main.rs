use tauri::{AppHandle, Emitter};
use serde::Serialize;
use std::sync::{Arc, Mutex};
use std::thread;
use fem_heat_reaction::{FEMSimulation, SimParams, SimFrame, MeshData};

/// State to track whether a simulation is running
struct SimState {
    running: bool,
}

#[derive(Serialize, Clone)]
struct SimStartPayload {
    mesh: MeshData,
    total_frames_estimate: usize,
    params: SimParams,
}

#[derive(Serialize, Clone)]
struct SimCompletePayload {
    total_frames: usize,
    elapsed_secs: f64,
}

#[tauri::command]
fn get_default_params() -> SimParams {
    SimParams::default()
}

#[tauri::command]
fn run_simulation(app: AppHandle, state: tauri::State<'_, Arc<Mutex<SimState>>>, params: SimParams) -> Result<(), String> {
    // Check if already running
    {
        let mut sim_state = state.lock().map_err(|e| e.to_string())?;
        if sim_state.running {
            return Err("Simulation already running".to_string());
        }
        sim_state.running = true;
    }
    
    let app_handle = app.clone();
    let state_arc = (*state).clone(); // Clone the Arc
    
    thread::spawn(move || {
        let start = std::time::Instant::now();
        
        // Create simulation from params
        let mut sim = FEMSimulation::new_with_params(&params);
        
        // Validate
        let warnings = sim.validate(params.t_final, params.dt);
        for w in &warnings {
            eprintln!("[WARN] {}", w);
        }
        
        // Get mesh data
        let mesh = sim.mesh_data();
        let total_frames_estimate = (params.t_final / params.dt) as usize / params.save_interval + 2;
        
        // Emit start event with mesh
        let _ = app_handle.emit("sim-start", SimStartPayload {
            mesh,
            total_frames_estimate,
            params: params.clone(),
        });
        
        // Run simulation streaming frames
        sim.run_streaming(params.t_final, params.dt, params.save_interval, |frame: SimFrame| {
            let _ = app_handle.emit("sim-frame", frame);
        });
        
        let elapsed = start.elapsed().as_secs_f64();
        let _ = app_handle.emit("sim-complete", SimCompletePayload {
            total_frames: total_frames_estimate,
            elapsed_secs: elapsed,
        });
        
        // Mark as no longer running
        if let Ok(mut s) = state_arc.lock() {
            s.running = false;
        }
    });
    
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .manage(Arc::new(Mutex::new(SimState { running: false })))
        .invoke_handler(tauri::generate_handler![get_default_params, run_simulation])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
