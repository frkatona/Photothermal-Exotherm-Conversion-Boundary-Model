- add a button to the interface which opens the folder where the simulation files are located
- a better solver stack to improve on the current plain CG approach (diagonal or incomplete factorization preconditioning; benchmark solver iteration count and wall time). matrix-free or structured-grid operators 
- save results in compact binary format
- stream frames to the frontend on demand
keep summaries and metadata separate from full fields
- pause, resume, and cancel controls for long simulation runs
- inline help text
- add button to store run snapshots (just the parameters and the final frame of each graph), browse snapshots by completion date, and delete if unwanted


- dynamic time stepping

- time and mesh refinement

- fix crashing at the end of a simulation (at least implement a crash report and allow for pulling the last run into the graph view)