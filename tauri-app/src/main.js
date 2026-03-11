/**
 * main.js — Application orchestration for the FEM Photothermal Simulation viewer.
 */
import { renderTriMesh, renderColorbar, renderMetricsChart } from './renderer.js';

// ============================================================
// Tauri API imports
// ============================================================
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

// ============================================================
// STATE
// ============================================================
let meshData = null;      // { nodes_x, nodes_y, elements, num_nodes, num_elements }
let frameHistory = [];    // Array of { time, max_temp, avg_conversion }
let currentFrame = null;  // Latest SimFrame
let isRunning = false;

// Auto-range tracking
let tempRange = [Infinity, -Infinity];
let convRange = [0, 0.01];
let laserRange = [0, 1];

// DOM references
const canvasLaser = document.getElementById('canvas-laser');
const canvasTemp = document.getElementById('canvas-temp');
const canvasConv = document.getElementById('canvas-conv');
const canvasMetrics = document.getElementById('canvas-metrics');
const colorbarLaser = document.getElementById('colorbar-laser');
const colorbarTemp = document.getElementById('colorbar-temp');
const colorbarConv = document.getElementById('colorbar-conv');
const btnRun = document.getElementById('btn-run');
const btnReset = document.getElementById('btn-reset');
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const progressContainer = document.getElementById('progress-container');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const progressDetail = document.getElementById('progress-detail');

// ============================================================
// PARAMETER COLLECTION
// ============================================================

function collectParams() {
    return {
        lx: parseFloat(document.getElementById('param-lx').value) * 1e-6,
        ly: parseFloat(document.getElementById('param-ly').value) * 1e-6,
        nx: parseInt(document.getElementById('param-nx').value),
        ny: parseInt(document.getElementById('param-ny').value),
        t_final: parseFloat(document.getElementById('param-tfinal').value) * 1e-9,
        dt: parseFloat(document.getElementById('param-dt').value) * 1e-12,
        save_interval: parseInt(document.getElementById('param-saveint').value),
        rho: parseFloat(document.getElementById('param-rho').value),
        cp: parseFloat(document.getElementById('param-cp').value),
        k: parseFloat(document.getElementById('param-k').value),
        h_coeff: parseFloat(document.getElementById('param-hcoeff').value),
        a_pre: parseFloat(document.getElementById('param-apre').value),
        ea: parseFloat(document.getElementById('param-ea').value),
        delta_h: parseFloat(document.getElementById('param-deltah').value),
        laser_power: parseFloat(document.getElementById('param-power').value),
        sigma: parseFloat(document.getElementById('param-sigma').value) * 1e-6,
        pulse_width: parseFloat(document.getElementById('param-pulsewidth').value) * 1e-9,
        t_init: parseFloat(document.getElementById('param-tinit').value),
        t_inf: parseFloat(document.getElementById('param-tinf').value),
    };
}

function populateParams(params) {
    document.getElementById('param-lx').value = (params.lx * 1e6).toFixed(1);
    document.getElementById('param-ly').value = (params.ly * 1e6).toFixed(1);
    document.getElementById('param-nx').value = params.nx;
    document.getElementById('param-ny').value = params.ny;
    document.getElementById('param-tfinal').value = (params.t_final * 1e9).toFixed(1);
    document.getElementById('param-dt').value = (params.dt * 1e12).toFixed(0);
    document.getElementById('param-saveint').value = params.save_interval;
    document.getElementById('param-rho').value = params.rho;
    document.getElementById('param-cp').value = params.cp;
    document.getElementById('param-k').value = params.k;
    document.getElementById('param-hcoeff').value = params.h_coeff;
    document.getElementById('param-apre').value = params.a_pre;
    document.getElementById('param-ea').value = params.ea;
    document.getElementById('param-deltah').value = params.delta_h;
    document.getElementById('param-power').value = params.laser_power;
    document.getElementById('param-sigma').value = (params.sigma * 1e6).toFixed(1);
    document.getElementById('param-pulsewidth').value = (params.pulse_width * 1e9).toFixed(0);
    document.getElementById('param-tinit').value = params.t_init;
    document.getElementById('param-tinf').value = params.t_inf;
}

// ============================================================
// STATUS HELPERS
// ============================================================

function setStatus(state, text) {
    statusIndicator.className = `status ${state}`;
    statusText.textContent = text;
}

function setProgress(fraction, detail = '') {
    progressContainer.classList.remove('hidden');
    progressFill.style.width = `${(fraction * 100).toFixed(1)}%`;
    progressText.textContent = `${(fraction * 100).toFixed(1)}%`;
    progressDetail.textContent = detail;
}

// ============================================================
// CANVAS SIZING
// ============================================================

function resizeCanvases() {
    const canvases = [canvasLaser, canvasTemp, canvasConv, canvasMetrics];
    for (const c of canvases) {
        const rect = c.parentElement.getBoundingClientRect();
        const titleEl = c.parentElement.querySelector('.viz-title');
        const colorbarEl = c.parentElement.querySelector('.colorbar');
        const titleH = titleEl ? titleEl.offsetHeight : 0;
        const colorbarH = colorbarEl ? colorbarEl.offsetHeight + 24 : 0;
        const availH = rect.height - titleH - colorbarH;
        const dpr = window.devicePixelRatio || 1;

        c.width = Math.floor(rect.width * dpr);
        c.height = Math.floor(Math.max(100, availH) * dpr);
        c.style.width = `${rect.width}px`;
        c.style.height = `${Math.max(100, availH)}px`;
        c.getContext('2d').scale(dpr, dpr);
        // Reset the logical dimensions for rendering
        c.width = Math.floor(rect.width);
        c.height = Math.floor(Math.max(100, availH));
    }
    // Re-render current frame if we have one
    if (currentFrame && meshData) {
        renderCurrentFrame();
    }
}

// ============================================================
// RENDERING
// ============================================================

function renderCurrentFrame() {
    if (!meshData || !currentFrame) return;

    // Temperature
    renderTriMesh(
        canvasTemp,
        meshData.nodes_x, meshData.nodes_y,
        meshData.elements,
        currentFrame.temperature,
        'inferno',
        tempRange
    );
    renderColorbar(colorbarTemp, 'inferno', tempRange[0], tempRange[1]);

    // Conversion
    renderTriMesh(
        canvasConv,
        meshData.nodes_x, meshData.nodes_y,
        meshData.elements,
        currentFrame.alpha,
        'viridis',
        convRange
    );
    renderColorbar(colorbarConv, 'viridis', convRange[0], convRange[1]);

    // Laser
    renderTriMesh(
        canvasLaser,
        meshData.nodes_x, meshData.nodes_y,
        meshData.elements,
        currentFrame.laser,
        'hot',
        laserRange
    );
    renderColorbar(colorbarLaser, 'hot', laserRange[0], laserRange[1]);

    // Metrics chart
    if (frameHistory.length > 1) {
        renderMetricsChart(
            canvasMetrics,
            frameHistory.map(f => f.time),
            frameHistory.map(f => f.max_temp),
            frameHistory.map(f => f.avg_conversion)
        );
    }
}

// ============================================================
// EVENT LISTENERS
// ============================================================

async function setupEventListeners() {
    await listen('sim-start', (event) => {
        const payload = event.payload;
        meshData = payload.mesh;
        frameHistory = [];
        currentFrame = null;
        tempRange = [Infinity, -Infinity];
        convRange = [0, 0.01];
        laserRange = [0, 1];

        setStatus('running', 'Simulating...');
        setProgress(0, `0/${payload.total_frames_estimate} frames`);
    });

    await listen('sim-frame', (event) => {
        const frame = event.payload;
        currentFrame = frame;

        // Update ranges
        if (frame.max_temp > tempRange[1]) tempRange[1] = frame.max_temp;
        if (frame.max_temp < tempRange[0]) tempRange[0] = Math.min(...frame.temperature);

        const maxConv = Math.max(...frame.alpha);
        if (maxConv > convRange[1]) convRange[1] = Math.max(maxConv, 0.01);

        const maxLaser = Math.max(...frame.laser);
        if (maxLaser > laserRange[1]) laserRange[1] = maxLaser;

        // Track metrics
        frameHistory.push({
            time: frame.time,
            max_temp: frame.max_temp,
            avg_conversion: frame.avg_conversion,
        });

        setProgress(frame.progress, `Frame ${frame.frame_index} | T_max: ${frame.max_temp.toFixed(2)} K`);
        renderCurrentFrame();
    });

    await listen('sim-complete', (event) => {
        const payload = event.payload;
        isRunning = false;
        btnRun.disabled = false;
        btnRun.querySelector('.btn-icon-text').textContent = '▶';
        setStatus('complete', `Done in ${payload.elapsed_secs.toFixed(2)}s`);
        setProgress(1, `Complete: ${frameHistory.length} frames`);
    });
}

// ============================================================
// RUN SIMULATION
// ============================================================

async function runSimulation() {
    if (isRunning) return;

    isRunning = true;
    btnRun.disabled = true;
    btnRun.querySelector('.btn-icon-text').textContent = '⏳';
    setStatus('running', 'Initializing...');
    progressContainer.classList.remove('hidden');

    try {
        const params = collectParams();
        await invoke('run_simulation', { params });
    } catch (err) {
        setStatus('error', `Error: ${err}`);
        isRunning = false;
        btnRun.disabled = false;
        btnRun.querySelector('.btn-icon-text').textContent = '▶';
    }
}

// ============================================================
// INIT
// ============================================================

async function init() {
    // Load defaults
    try {
        const defaults = await invoke('get_default_params');
        populateParams(defaults);
    } catch (e) {
        console.warn('Could not load defaults:', e);
    }

    // Setup event listeners
    await setupEventListeners();

    // Button handlers
    btnRun.addEventListener('click', runSimulation);
    btnReset.addEventListener('click', async () => {
        try {
            const defaults = await invoke('get_default_params');
            populateParams(defaults);
        } catch (e) {
            console.warn('Reset failed:', e);
        }
    });

    // Canvas resizing
    window.addEventListener('resize', resizeCanvases);
    // Initial resize after layout settles
    setTimeout(resizeCanvases, 100);
}

init();
