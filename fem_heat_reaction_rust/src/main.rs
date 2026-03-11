use fem_heat_reaction::{FEMSimulation, SimFrame};
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
    let t_final = 1e-8;
    let dt = 1e-10;
    let save_interval = 100;

    println!("Generating mesh...");
    println!("Assembling system matrices...");
    let mut sim = FEMSimulation::new(lx, ly, nx, ny);
    
    // Validate
    let warnings = sim.validate(t_final, dt);
    println!("\n{}", "=".repeat(60));
    println!("VALIDATION CHECKS");
    println!("{}", "=".repeat(60));
    for w in &warnings {
        println!("[WARNING] {}", w);
    }
    if warnings.is_empty() {
        println!("[OK] All checks passed");
    }
    println!("\nSimulation Parameters:");
    println!("  Domain: {:.1} µm x {:.1} µm", lx * 1e6, ly * 1e6);
    println!("  Grid: {} nodes", sim.mesh.num_nodes);
    println!("  Timesteps: {}", (t_final / dt) as usize);
    println!("{}\n", "=".repeat(60));
    
    // Run with progress bar
    let num_steps = (t_final / dt) as usize;
    let pb = ProgressBar::new(num_steps as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{bar:40}] {percent:>3}% | {msg}")
        .unwrap());
    
    let output_dir = "output";
    fs::create_dir_all(output_dir).unwrap();
    
    // Save mesh data
    {
        let mesh_data = sim.mesh_data();
        let mut f = BufWriter::new(File::create(format!("{}/mesh.csv", output_dir)).unwrap());
        writeln!(f, "x,y").unwrap();
        for i in 0..mesh_data.num_nodes {
            writeln!(f, "{},{}", mesh_data.nodes_x[i], mesh_data.nodes_y[i]).unwrap();
        }
        
        let mut f = BufWriter::new(File::create(format!("{}/elements.csv", output_dir)).unwrap());
        writeln!(f, "n1,n2,n3").unwrap();
        for e in &mesh_data.elements {
            writeln!(f, "{},{},{}", e[0], e[1], e[2]).unwrap();
        }
    }
    
    let mut frame_times = Vec::new();
    let mut all_max_temps = Vec::new();
    let mut all_avg_convs = Vec::new();
    
    let start_time = Instant::now();
    
    sim.run_streaming(t_final, dt, save_interval, |frame: SimFrame| {
        frame_times.push(frame.time);
        all_max_temps.push(frame.max_temp);
        all_avg_convs.push(frame.avg_conversion);
        
        // Save frame CSVs
        {
            let mut f = BufWriter::new(
                File::create(format!("{}/temperature_{:04}.csv", output_dir, frame.frame_index)).unwrap()
            );
            writeln!(f, "temperature").unwrap();
            for &v in &frame.temperature {
                writeln!(f, "{}", v).unwrap();
            }
        }
        {
            let mut f = BufWriter::new(
                File::create(format!("{}/alpha_{:04}.csv", output_dir, frame.frame_index)).unwrap()
            );
            writeln!(f, "alpha").unwrap();
            for &v in &frame.alpha {
                writeln!(f, "{}", v).unwrap();
            }
        }
        {
            let mut f = BufWriter::new(
                File::create(format!("{}/laser_{:04}.csv", output_dir, frame.frame_index)).unwrap()
            );
            writeln!(f, "laser").unwrap();
            for &v in &frame.laser {
                writeln!(f, "{}", v).unwrap();
            }
        }
        
        let step_approx = ((frame.progress) * num_steps as f64) as u64;
        pb.set_position(step_approx);
        pb.set_message(format!("Max T: {:.2} K", frame.max_temp));
    });
    
    pb.finish_with_message(format!("Done! Max T: {:.2} K", all_max_temps.last().unwrap_or(&0.0)));
    let elapsed = start_time.elapsed();
    println!("Simulation complete in {:.2} s", elapsed.as_secs_f64());
    println!("  Saved {} animation frames", frame_times.len());
    
    // Save frames index
    {
        let mut f = BufWriter::new(File::create(format!("{}/frames.csv", output_dir)).unwrap());
        writeln!(f, "frame,time").unwrap();
        for (i, t) in frame_times.iter().enumerate() {
            writeln!(f, "{},{}", i, t).unwrap();
        }
    }
    
    // Save metrics
    {
        let mut f = BufWriter::new(File::create(format!("{}/metrics.csv", output_dir)).unwrap());
        writeln!(f, "time,max_temp,avg_conversion").unwrap();
        for i in 0..frame_times.len() {
            writeln!(f, "{},{},{}", frame_times[i], all_max_temps[i], all_avg_convs[i]).unwrap();
        }
    }
    
    println!("\n{}", "=".repeat(60));
    println!("Simulation complete!");
    println!("{}", "=".repeat(60));
}
