/**
 * main.js — Batch simulation with scrubber playback, parameter persistence,
 * laser preview, and MP4 export.
 */
import { renderTriMesh, renderColorbar, renderMetricsChart } from './renderer.js';

// ============================================================
// DEBUG LOGGING
// ============================================================
const debugLog = document.getElementById('debug-log');
const debugPanel = document.getElementById('debug-panel');
let debugVisible = false;

function log(level, msg, data = null) {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false, fractionalSecondDigits: 3 });
    const colors = { info: '#58a6ff', warn: '#d29922', error: '#f85149', ok: '#3fb950', event: '#a371f7' };
    const color = colors[level] || '#8b949e';
    const prefix = `[${timestamp}] [${level.toUpperCase()}]`;
    const consoleFn = level === 'error' ? console.error : level === 'warn' ? console.warn : console.log;
    consoleFn(prefix, msg, data !== null ? data : '');
    if (debugLog) {
        const entry = document.createElement('div');
        entry.className = `debug-entry debug-${level}`;
        entry.style.color = color;
        let text = `${prefix} ${msg}`;
        if (data !== null) {
            try {
                const s = typeof data === 'object' ? JSON.stringify(data, null, 0) : String(data);
                text += ` → ${s.length > 200 ? s.substring(0, 200) + '…' : s}`;
            } catch { text += ` → [unstringifiable]`; }
        }
        entry.textContent = text;
        debugLog.appendChild(entry);
        debugLog.scrollTop = debugLog.scrollHeight;
    }
}

// ============================================================
// Tauri API
// ============================================================
let invoke, listen;
function initTauriAPI() {
    if (!window.__TAURI__?.core?.invoke || !window.__TAURI__?.event?.listen) {
        log('error', 'Tauri API not available');
        return false;
    }
    invoke = window.__TAURI__.core.invoke;
    listen = window.__TAURI__.event.listen;
    log('ok', 'Tauri API ready');
    return true;
}

// ============================================================
// STATE
// ============================================================
let meshData = null;
let allFrames = [];
let currentFrameIdx = 0;
let isRunning = false;
let isPlaying = false;
let playTimer = null;
let tempRange = [300, 301], convRange = [0, 0.01], laserRange = [0, 1];
let autoSaveTimer = null;

// DOM
const canvasLaser = document.getElementById('canvas-laser');
const canvasTemp = document.getElementById('canvas-temp');
const canvasConv = document.getElementById('canvas-conv');
const canvasMetrics = document.getElementById('canvas-metrics');
const vizGrid = document.getElementById('viz-grid');
const vizCells = Array.from(document.querySelectorAll('.viz-cell'));
const colorbarLaser = document.getElementById('colorbar-laser');
const colorbarTemp = document.getElementById('colorbar-temp');
const colorbarConv = document.getElementById('colorbar-conv');
const btnRun = document.getElementById('btn-run');
const btnReset = document.getElementById('btn-reset');
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const statusDetail = document.getElementById('status-detail');
const statusFolderSize = document.getElementById('status-folder-size');
const progressContainer = document.getElementById('progress-container');
const progressFill = document.getElementById('progress-fill');
const progressTextEl = document.getElementById('progress-text');
const progressDetail = document.getElementById('progress-detail');
const scrubberContainer = document.getElementById('scrubber-container');
const scrubber = document.getElementById('scrubber');
const scrubberInfo = document.getElementById('scrubber-info');
const btnPlay = document.getElementById('btn-play');
const playbackSpeedSelect = document.getElementById('playback-speed');
const paramPanel = document.getElementById('param-panel');
const paramScroll = paramPanel?.querySelector('.param-scroll');
const canvasScanPreview = document.getElementById('canvas-scan-preview');
const infoScan = document.getElementById('info-scan');
const warningPanel = document.getElementById('warning-panel');
const warningList = document.getElementById('warning-list');
const warningStatusBadge = document.getElementById('warning-status-badge');
const warningStatusIcon = document.getElementById('warning-status-icon');
const warningTooltipPassed = document.getElementById('warning-tooltip-passed');
const warningTooltipFailed = document.getElementById('warning-tooltip-failed');
const runEstimateEl = document.getElementById('run-estimate');
const displayColormapTemp = document.getElementById('display-colormap-temp');
const displayColormapConv = document.getElementById('display-colormap-conv');
const displayColormapLaser = document.getElementById('display-colormap-laser');
const displayContourTempEnabled = document.getElementById('display-contour-temp-enabled');
const displayContourTempThreshold = document.getElementById('display-contour-temp-threshold');
const displayContourConvEnabled = document.getElementById('display-contour-conv-enabled');
const displayContourConvThreshold = document.getElementById('display-contour-conv-threshold');
const displayContourLaserEnabled = document.getElementById('display-contour-laser-enabled');
const displayContourLaserThreshold = document.getElementById('display-contour-laser-threshold');

let displaySettings = {
    tempColormap: displayColormapTemp?.value || 'inferno',
    convColormap: displayColormapConv?.value || 'viridis',
    laserColormap: displayColormapLaser?.value || 'hot',
    tempContourEnabled: Boolean(displayContourTempEnabled?.checked),
    tempContourThreshold: getNumberInput('display-contour-temp-threshold', 300.5),
    convContourEnabled: Boolean(displayContourConvEnabled?.checked),
    convContourThreshold: getNumberInput('display-contour-conv-threshold', 0.5),
    laserContourEnabled: Boolean(displayContourLaserEnabled?.checked),
    laserContourThreshold: getNumberInput('display-contour-laser-threshold', 0),
};
let projectFolderSizeBytes = null;
let lastRunElapsedSecs = null;
let currentRunOps = null;
let focusedVizId = null;
let runtimeModelSecsPerOp = (() => {
    const saved = Number(window.localStorage?.getItem('runtime_model_secs_per_op'));
    return Number.isFinite(saved) && saved > 0 ? saved : 1.2e-7;
})();
let playbackSpeedMultiplier = (() => {
    const saved = Number(window.localStorage?.getItem('playback_speed_multiplier'));
    return Number.isFinite(saved) && saved > 0 ? saved : 1;
})();
let completionAudioContext = null;

const PARAM_FIELD_FORMATS = {
    'param-lxy': { decimals: 1 },
    'param-nxy': { integer: true },
    'param-tfinal': { decimals: 3 },
    'param-dt': { decimals: 3 },
    'param-saveint': { integer: true },
    'param-rho': { decimals: 2 },
    'param-cp': { decimals: 2 },
    'param-k': { decimals: 4 },
    'param-hcoeff': { decimals: 3 },
    'param-apre': { decimals: 3 },
    'param-ea': { decimals: 3 },
    'param-deltah': { decimals: 3 },
    'param-pulseenergy': { decimals: 3 },
    'param-sigma': { decimals: 3 },
    'param-pulsewidth': { decimals: 3 },
    'param-pulserate': { decimals: 3 },
    'param-scanspeed': { decimals: 3 },
    'param-linespacing': { decimals: 3 },
    'param-scanmargin': { decimals: 3 },
    'param-filmthickness': { decimals: 3 },
    'param-absorption': { decimals: 3 },
    'param-absorption-transformed': { decimals: 3 },
    'param-tinit': { decimals: 3 },
    'param-tinf': { decimals: 3 },
};

function getNumberInput(id, fallback = 0) {
    const value = parseFloat(document.getElementById(id)?.value);
    return Number.isFinite(value) ? value : fallback;
}

function getIntInput(id, fallback = 0) {
    const value = parseFloat(document.getElementById(id)?.value);
    return Number.isFinite(value) ? Math.max(0, Math.round(value)) : fallback;
}

function formatInputValue(value, options = {}) {
    if (!Number.isFinite(value)) return '';

    const {
        decimals = 3,
        integer = false,
        scientificAbove = 1000,
        scientificBelow = 1e-3,
    } = options;
    const abs = Math.abs(value);

    if (abs >= scientificAbove || (abs > 0 && abs < scientificBelow)) {
        const exponentDigits = integer ? 0 : Math.max(0, Math.min(decimals, 6));
        return value
            .toExponential(exponentDigits)
            .replace(/\.?0+e/, 'e')
            .replace('e+', 'e');
    }

    if (integer) {
        return `${Math.round(value)}`;
    }

    return value.toFixed(decimals).replace(/\.?0+$/, '');
}

function setFormattedFieldValue(id, value) {
    const input = document.getElementById(id);
    if (!input) return;
    input.value = formatInputValue(value, PARAM_FIELD_FORMATS[id]);
    updateScientificClass(input);
}

function normalizeParameterInput(input) {
    if (!input || !(input.id in PARAM_FIELD_FORMATS)) return;
    const value = parseFloat(input.value);
    if (!Number.isFinite(value)) return;
    input.value = formatInputValue(value, PARAM_FIELD_FORMATS[input.id]);
    updateScientificClass(input);
}

function normalizeParameterInputs() {
    Object.keys(PARAM_FIELD_FORMATS).forEach((id) => {
        normalizeParameterInput(document.getElementById(id));
    });
}

function updateScientificClass(input) {
    if (!input) return;
    input.classList.toggle('scientific', /e/i.test(input.value));
}

function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes < 0) return '--';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let value = bytes;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024;
        unitIndex += 1;
    }
    return `${value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2)} ${units[unitIndex]}`;
}

function formatDuration(value) {
    if (!Number.isFinite(value) || value < 0) return '--';
    if (value < 1) return `${(value * 1000).toFixed(0)} ms`;
    if (value < 60) return `${value.toFixed(value < 10 ? 2 : 1)} s`;
    const minutes = Math.floor(value / 60);
    const seconds = value % 60;
    if (minutes < 60) return `${minutes}m ${seconds.toFixed(0)}s`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ${minutes % 60}m`;
}

function formatPercent(value) {
    if (!Number.isFinite(value)) return '--';
    return `${(value * 100).toFixed(value >= 0.1 ? 1 : 2)}%`;
}

async function ensureAudioContext() {
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) return null;
    if (!completionAudioContext) {
        completionAudioContext = new AudioContextCtor();
    }
    if (completionAudioContext.state === 'suspended') {
        try {
            await completionAudioContext.resume();
        } catch {
            return completionAudioContext;
        }
    }
    return completionAudioContext;
}

function scheduleTone(context, startTime, frequency, duration, gain, type = 'triangle') {
    const oscillator = context.createOscillator();
    const gainNode = context.createGain();

    oscillator.type = type;
    oscillator.frequency.setValueAtTime(frequency, startTime);
    gainNode.gain.setValueAtTime(0.0001, startTime);
    gainNode.gain.exponentialRampToValueAtTime(gain, startTime + 0.012);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, startTime + duration);

    oscillator.connect(gainNode);
    gainNode.connect(context.destination);
    oscillator.start(startTime);
    oscillator.stop(startTime + duration + 0.03);
}

async function playCompletionChime() {
    const context = await ensureAudioContext();
    if (!context || context.state !== 'running') return;

    const start = context.currentTime + 0.02;
    scheduleTone(context, start, 659.25, 0.12, 0.025, 'triangle');
    scheduleTone(context, start + 0.11, 987.77, 0.18, 0.03, 'sine');
}

function applyPlaybackSpeedSetting(rawValue = playbackSpeedSelect?.value) {
    const parsed = Number(rawValue);
    playbackSpeedMultiplier = Number.isFinite(parsed) && parsed > 0 ? parsed : 1;
    if (playbackSpeedSelect && playbackSpeedSelect.value !== String(playbackSpeedMultiplier)) {
        playbackSpeedSelect.value = String(playbackSpeedMultiplier);
    }
    window.localStorage?.setItem('playback_speed_multiplier', String(playbackSpeedMultiplier));

    if (isPlaying) {
        stopPlayback();
        startPlayback();
    }
}

function setStatusMeta(detailText = '', folderText = null) {
    if (statusDetail) statusDetail.textContent = detailText || 'Project folder: --';
    if (statusFolderSize) {
        statusFolderSize.textContent = folderText ?? `Size: ${formatBytes(projectFolderSizeBytes)}`;
    }
}

function updateRunEstimate(summary = null) {
    if (!runEstimateEl) return;
    const currentSummary = summary ?? computeSimulationSummary();
    if (currentSummary.totalSteps <= 0 || currentSummary.nodesTotal <= 0) {
        runEstimateEl.textContent = 'Est. --';
        return;
    }
    const estimatedSeconds = currentSummary.totalOps * runtimeModelSecsPerOp;
    runEstimateEl.textContent = `Est. ${formatDuration(Math.max(0.05, estimatedSeconds))}`;
}

async function refreshProjectStats() {
    if (!invoke) return;
    try {
        const stats = await invoke('get_project_stats');
        projectFolderSizeBytes = stats.folder_size_bytes;
        setStatusMeta(
            lastRunElapsedSecs !== null ? `Done in ${formatDuration(lastRunElapsedSecs)}` : 'Project folder',
            `Size: ${formatBytes(projectFolderSizeBytes)}`,
        );
    } catch (e) {
        log('warn', 'Project size scan failed', String(e));
        setStatusMeta(
            lastRunElapsedSecs !== null ? `Done in ${formatDuration(lastRunElapsedSecs)}` : 'Project folder',
            'Size: unavailable',
        );
    }
}

function collectDisplaySettings() {
    const midpoint = ([min, max]) => (Number.isFinite(min) && Number.isFinite(max) ? (min + max) * 0.5 : 0);
    return {
        tempColormap: displayColormapTemp?.value || 'inferno',
        convColormap: displayColormapConv?.value || 'viridis',
        laserColormap: displayColormapLaser?.value || 'hot',
        tempContourEnabled: Boolean(displayContourTempEnabled?.checked),
        tempContourThreshold: getNumberInput('display-contour-temp-threshold', midpoint(tempRange)),
        convContourEnabled: Boolean(displayContourConvEnabled?.checked),
        convContourThreshold: getNumberInput('display-contour-conv-threshold', midpoint(convRange)),
        laserContourEnabled: Boolean(displayContourLaserEnabled?.checked),
        laserContourThreshold: getNumberInput('display-contour-laser-threshold', midpoint(laserRange)),
    };
}

function renderCurrentColorbars() {
    renderColorbar(colorbarTemp, displaySettings.tempColormap, tempRange[0], tempRange[1]);
    renderColorbar(colorbarConv, displaySettings.convColormap, convRange[0], convRange[1]);
    renderColorbar(colorbarLaser, displaySettings.laserColormap, laserRange[0], laserRange[1]);
}

function applyDisplaySettings() {
    displaySettings = collectDisplaySettings();
    if (allFrames.length > 0 && meshData) {
        renderFrame(currentFrameIdx);
    } else {
        renderCurrentColorbars();
    }
}

// ============================================================
// LASER PREVIEW (renders Gaussian beam on laser canvas before simulation)
// ============================================================

function getScanParams() {
    return {
        lxy: getNumberInput('param-lxy', 200) * 1e-6,
        tFinal: getNumberInput('param-tfinal', 1e-8),
        pulseRate: getNumberInput('param-pulserate', 20) * 1e3,
        scanSpeed: getNumberInput('param-scanspeed', 100) * 1e-3,
        lineSpacing: getNumberInput('param-linespacing', 40) * 1e-6,
        scanMargin: Math.max(getNumberInput('param-scanmargin', 20) * 1e-6, 0),
        filmThickness: Math.max(getNumberInput('param-filmthickness', 10) * 1e-6, 0),
    };
}

function getScanWindow(params) {
    const xMin = Math.min(params.scanMargin, params.lxy);
    const xLimit = Math.max(xMin, params.lxy - params.scanMargin);
    const yMin = Math.min(params.scanMargin, params.lxy);
    const yMax = Math.max(yMin, params.lxy - params.scanMargin);
    const spacing = Math.max(params.lineSpacing, 1e-12);
    const lineCount = Math.max(1, Math.floor(Math.max(xLimit - xMin, 0) / spacing) + 1);
    const xMax = lineCount > 1 ? Math.min(xLimit, xMin + (lineCount - 1) * spacing) : xMin;

    return {
        xMin,
        xMax,
        xLimit,
        yMin,
        yMax,
        lineCount,
        scanWidth: Math.max(xMax - xMin, 0),
        scanHeight: Math.max(yMax - yMin, 0),
    };
}

function beamPositionAtTime(t, params) {
    const window = getScanWindow(params);

    if (params.scanSpeed <= 0 || window.scanHeight <= 0) {
        return { x: window.xMin, y: window.yMin, lineCount: window.lineCount };
    }

    const timePerLine = window.scanHeight / params.scanSpeed;
    const rawLine = Math.max(0, Math.floor(t / timePerLine));
    const lineIndex = rawLine % window.lineCount;
    const tInLine = ((t % timePerLine) + timePerLine) % timePerLine;
    const progress = Math.min(1, Math.max(0, tInLine / timePerLine));
    const x = Math.min(window.xMax, window.xMin + lineIndex * Math.max(params.lineSpacing, 1e-12));
    const y = rawLine % 2 === 0
        ? window.yMin + progress * window.scanHeight
        : window.yMax - progress * window.scanHeight;

    return { x, y, lineCount: window.lineCount };
}

function formatLengthMeters(value) {
    const microns = value * 1e6;
    if (Math.abs(microns) >= 1000) {
        return `${(microns / 1000).toFixed(2)} mm`;
    }
    return `${microns.toFixed(microns >= 100 ? 0 : 1)} um`;
}

function formatSeconds(value) {
    if (!Number.isFinite(value)) return `${value} s`;
    const abs = Math.abs(value);
    if (abs === 0) return '0 s';
    if (abs < 1e-3 || abs >= 1e3) return `${value.toExponential(2)} s`;
    if (abs < 1) return `${value.toFixed(6).replace(/0+$/, '').replace(/\.$/, '')} s`;
    return `${value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')} s`;
}

function formatScaledValue(value, units) {
    const abs = Math.abs(value);
    if (abs === 0) {
        return `0 ${units[units.length - 1][1]}`;
    }
    for (const [scale, label] of units) {
        if (abs >= scale) {
            return `${(value / scale).toFixed(2)} ${label}`;
        }
    }
    return `${value.toExponential(2)} ${units[units.length - 1][1]}`;
}

function drawDimensionLine(ctx, x1, y1, x2, y2, label, vertical = false) {
    const arrow = 4;
    const drawArrow = (fromX, fromY, toX, toY) => {
        const angle = Math.atan2(toY - fromY, toX - fromX);
        ctx.beginPath();
        ctx.moveTo(toX, toY);
        ctx.lineTo(
            toX - arrow * Math.cos(angle - Math.PI / 6),
            toY - arrow * Math.sin(angle - Math.PI / 6),
        );
        ctx.lineTo(
            toX - arrow * Math.cos(angle + Math.PI / 6),
            toY - arrow * Math.sin(angle + Math.PI / 6),
        );
        ctx.closePath();
        ctx.fill();
    };

    ctx.save();
    ctx.strokeStyle = 'rgba(230, 237, 243, 0.30)';
    ctx.fillStyle = 'rgba(230, 237, 243, 0.55)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    drawArrow(x2, y2, x1, y1);
    drawArrow(x1, y1, x2, y2);

    ctx.fillStyle = '#8b949e';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    if (vertical) {
        ctx.save();
        ctx.translate(x1 - 8, (y1 + y2) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(label, 0, 0);
        ctx.restore();
    } else {
        ctx.fillText(label, (x1 + x2) / 2, y1 - 8);
    }
    ctx.restore();
}

function renderLaserPreview() {
    const canvas = canvasScanPreview;
    if (!canvas || canvas.width === 0 || canvas.height === 0) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const params = getScanParams();
    const window = getScanWindow(params);
    const outerPad = 14;
    const topBand = 18;
    const bottomBand = 10;
    const sideGap = 14;
    const sideWidth = Math.max(42, Math.min(70, w * 0.24));
    const topRect = {
        x: outerPad,
        y: topBand,
        w: Math.max(64, w - sideWidth - sideGap - outerPad * 2),
        h: Math.max(48, h - topBand - bottomBand),
    };
    const sideRect = {
        x: topRect.x + topRect.w + sideGap,
        y: topBand + 8,
        w: Math.max(30, w - (topRect.x + topRect.w + sideGap) - outerPad),
        h: Math.max(38, topRect.h - 16),
    };
    const mapTopX = (x) => topRect.x + (x / Math.max(params.lxy, 1e-12)) * topRect.w;
    const mapTopY = (y) => topRect.y + topRect.h - (y / Math.max(params.lxy, 1e-12)) * topRect.h;

    ctx.fillStyle = '#0b1016';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('top view', topRect.x, 11);
    ctx.fillText('side view', sideRect.x, 11);

    ctx.strokeStyle = 'rgba(88, 166, 255, 0.28)';
    ctx.lineWidth = 1;
    ctx.strokeRect(topRect.x, topRect.y, topRect.w, topRect.h);

    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = 'rgba(248, 81, 73, 0.22)';
    ctx.strokeRect(
        mapTopX(window.xMin),
        mapTopY(window.yMax),
        Math.max(1, mapTopX(window.xLimit) - mapTopX(window.xMin)),
        Math.max(1, mapTopY(window.yMin) - mapTopY(window.yMax)),
    );
    ctx.setLineDash([]);

    const sampleCount = Math.max(
        2,
        Math.min(1400, Math.max(48, window.lineCount * 24, Math.ceil(params.tFinal * Math.max(params.pulseRate, 1)) * 2)),
    );
    let first = null;
    let last = null;
    ctx.beginPath();
    for (let i = 0; i < sampleCount; i++) {
        const t = sampleCount === 1 ? 0 : (i / (sampleCount - 1)) * params.tFinal;
        const pos = beamPositionAtTime(t, params);
        const px = mapTopX(pos.x);
        const py = mapTopY(pos.y);
        if (i === 0) {
            ctx.moveTo(px, py);
            first = { x: px, y: py };
        } else {
            ctx.lineTo(px, py);
        }
        last = { x: px, y: py };
    }
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 2;
    ctx.stroke();

    if (first) {
        ctx.fillStyle = '#3fb950';
        ctx.beginPath();
        ctx.arc(first.x, first.y, 3, 0, Math.PI * 2);
        ctx.fill();
    }

    if (last) {
        ctx.fillStyle = '#f85149';
        ctx.beginPath();
        ctx.arc(last.x, last.y, 3, 0, Math.PI * 2);
        ctx.fill();
    }

    drawDimensionLine(
        ctx,
        topRect.x,
        h - 4,
        topRect.x + topRect.w,
        h - 4,
        formatLengthMeters(params.lxy),
    );
    drawDimensionLine(
        ctx,
        topRect.x - 6,
        topRect.y,
        topRect.x - 6,
        topRect.y + topRect.h,
        formatLengthMeters(params.lxy),
        true,
    );

    const slabHeight = Math.max(10, Math.min(sideRect.h * 0.42, 18));
    const slabX = sideRect.x + 5;
    const slabY = sideRect.y + (sideRect.h - slabHeight) / 2;
    const slabW = Math.max(12, sideRect.w - 10);
    ctx.fillStyle = 'rgba(88, 166, 255, 0.14)';
    ctx.strokeStyle = 'rgba(88, 166, 255, 0.32)';
    ctx.lineWidth = 1;
    ctx.fillRect(slabX, slabY, slabW, slabHeight);
    ctx.strokeRect(slabX, slabY, slabW, slabHeight);

    drawDimensionLine(
        ctx,
        slabX - 6,
        slabY,
        slabX - 6,
        slabY + slabHeight,
        formatLengthMeters(params.filmThickness),
        true,
    );

    if (infoScan) {
        const pulsesPerRun = params.tFinal * params.pulseRate;
        infoScan.textContent =
            `Domain ${formatLengthMeters(params.lxy)} x ${formatLengthMeters(params.lxy)} x ${formatLengthMeters(params.filmThickness)} | ` +
            `Scan ${formatLengthMeters(window.scanWidth)} x ${formatLengthMeters(window.scanHeight)} | ` +
            `${window.lineCount} lines | ${pulsesPerRun.toFixed(2)} pulses/run`;
    }

    return;

    // Render a 2D Gaussian on the canvas using the 'hot' colormap colors
    const imgData = ctx.createImageData(w, h);
    const cx = w / 2, cy = h / 2;
    const sigPx = (sigma_um / lxy_um) * w;

    for (let py = 0; py < h; py++) {
        for (let px = 0; px < w; px++) {
            const dx = px - cx, dy = py - cy;
            const r2 = dx * dx + dy * dy;
            const val = Math.exp(-r2 / (2 * sigPx * sigPx));
            // Hot colormap: black → red → yellow → white
            let r, g, b;
            if (val < 0.33) {
                const t = val / 0.33;
                r = Math.floor(t * 255); g = 0; b = 0;
            } else if (val < 0.66) {
                const t = (val - 0.33) / 0.33;
                r = 255; g = Math.floor(t * 255); b = 0;
            } else {
                const t = (val - 0.66) / 0.34;
                r = 255; g = 255; b = Math.floor(t * 255);
            }
            const i = (py * w + px) * 4;
            imgData.data[i] = r;
            imgData.data[i + 1] = g;
            imgData.data[i + 2] = b;
            imgData.data[i + 3] = 255;
        }
    }
    ctx.putImageData(imgData, 0, 0);

    // Draw crosshair showing beam center
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
    ctx.moveTo(0, cy); ctx.lineTo(w, cy);
    ctx.stroke();

    // Draw circle at 1/e² radius (2σ)
    ctx.beginPath();
    ctx.arc(cx, cy, 2 * sigPx, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.setLineDash([6, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '11px monospace';
    ctx.fillText(`σ = ${sigma_um.toFixed(1)} µm`, 8, h - 8);
}

function computeSimulationSummary() {
    const lxy_um = getNumberInput('param-lxy', 200);
    const nxy = Math.max(1, getIntInput('param-nxy', 200));
    const tFinal_s = getNumberInput('param-tfinal', 1e-8);
    const dt_s = getNumberInput('param-dt', 1e-10);
    const saveInt = Math.max(1, getIntInput('param-saveint', 100));
    const rho = getNumberInput('param-rho', 1100);
    const cp = getNumberInput('param-cp', 1500);
    const k = getNumberInput('param-k', 0.3);
    const pulseEnergy_uJ = getNumberInput('param-pulseenergy', 100);
    const sigma_um = getNumberInput('param-sigma', 21.3);
    const pulseWidth_ns = getNumberInput('param-pulsewidth', 100);
    const pulseRate_kHz = getNumberInput('param-pulserate', 20);
    const scanSpeed_mm_s = getNumberInput('param-scanspeed', 100);
    const lineSpacing_um = getNumberInput('param-linespacing', 40);
    const scanMargin_um = getNumberInput('param-scanmargin', 20);
    const filmThickness_um = getNumberInput('param-filmthickness', 10);
    const absorptionCoeffBase = getNumberInput('param-absorption', 1e5);
    const absorptionCoeffTransformed = getNumberInput('param-absorption-transformed', absorptionCoeffBase);

    const dx_um = lxy_um / nxy;
    const dx_m = dx_um * 1e-6;
    const totalSteps = dt_s > 0 ? Math.max(0, Math.floor(tFinal_s / dt_s)) : 0;
    const savedFrames = totalSteps > 0 ? Math.floor(totalSteps / saveInt) + 1 : 1;
    const pulseRate_hz = pulseRate_kHz * 1e3;
    const pulseCount = tFinal_s * pulseRate_hz;
    const pulseWidth_s = pulseWidth_ns * 1e-9;
    const pulsePeriod_s = pulseRate_hz > 0 ? 1 / pulseRate_hz : Infinity;
    const scanSpeed_m_s = scanSpeed_mm_s * 1e-3;
    const pulseTravel_um = pulseRate_hz > 0 ? (scanSpeed_m_s / pulseRate_hz) * 1e6 : 0;
    const fwhm_um = 2.35 * sigma_um;
    const pointsPerFwhm = dx_um > 0 ? fwhm_um / dx_um : 0;
    const nodesTotal = nxy * nxy;
    const totalOps = nodesTotal * totalSteps;
    const lineCount = Math.max(1, Math.floor(Math.max(lxy_um - 2 * scanMargin_um, 0) / Math.max(lineSpacing_um, 1e-9)) + 1);
    const scanWidth_um = lineCount > 1 ? Math.min(Math.max(lxy_um - 2 * scanMargin_um, 0), (lineCount - 1) * lineSpacing_um) : 0;
    const scanHeight_um = Math.max(lxy_um - 2 * scanMargin_um, 0);
    const filmThickness_m = Math.max(filmThickness_um * 1e-6, 0);
    const sigma_m = sigma_um * 1e-6;
    const absorbedFractionBase = filmThickness_m > 0 && absorptionCoeffBase > 0
        ? 1 - Math.exp(-absorptionCoeffBase * filmThickness_m)
        : 0;
    const absorbedFractionTransformed = filmThickness_m > 0 && absorptionCoeffTransformed > 0
        ? 1 - Math.exp(-absorptionCoeffTransformed * filmThickness_m)
        : 0;
    const alphaThermal = rho > 0 && cp > 0 ? k / (rho * cp) : Infinity;
    const fourierNumber = Number.isFinite(alphaThermal) && dx_m > 0
        ? alphaThermal * dt_s / (dx_m * dx_m)
        : Infinity;
    const warnings = [];
    const checks = [];
    const addCheck = (passLabel, passed, failLabel) => {
        checks.push({ label: passed ? passLabel : failLabel, passed });
        if (!passed) warnings.push(failLabel);
    };

    addCheck('t_final is greater than 0.', tFinal_s > 0, 't_final must be greater than 0.');
    addCheck('dt is greater than 0.', dt_s > 0, 'dt must be greater than 0.');
    addCheck(
        'dt is smaller than t_final.',
        !(dt_s > 0 && tFinal_s > 0) || dt_s < tFinal_s,
        'dt is greater than or equal to t_final, so the run will have at most one timestep.',
    );
    addCheck(
        'The run contains at least 10 timesteps.',
        totalSteps <= 0 || totalSteps >= 10,
        'The run has fewer than 10 timesteps, so the time history and heating response will be poorly resolved.',
    );
    addCheck(
        'Save interval does not exceed total timesteps.',
        totalSteps <= 0 || saveInt <= totalSteps,
        'Save interval exceeds the total timestep count, so playback will contain only the endpoints.',
    );
    addCheck(
        'dt resolves the laser pulse width.',
        !(pulseWidth_s > 0) || dt_s <= pulseWidth_s / 5,
        'dt is too large relative to the laser pulse width; the pulse shape will be temporally under-resolved.',
    );
    addCheck(
        'Pulse width does not exceed the repetition period.',
        !(pulseRate_hz > 0) || pulseWidth_s <= pulsePeriod_s,
        'Pulse width exceeds the pulse period, so pulses overlap continuously.',
    );
    addCheck(
        'Beam resolution is at least 12 points per FWHM.',
        pointsPerFwhm >= 12,
        `Beam resolution is low at ${pointsPerFwhm.toFixed(1)} points/FWHM; the laser spot will be spatially smeared.`,
    );
    addCheck(
        'Scan margin leaves interior raster area.',
        scanMargin_um * 2 < lxy_um,
        'Scan margin leaves no interior scan area.',
    );
    addCheck(
        'Beam travel per pulse stays within one FWHM.',
        !(pulseRate_hz > 0) || pulseTravel_um <= Math.max(fwhm_um, 1e-6),
        'The beam moves farther than one beam FWHM between pulses, so the raster path will be sparsely sampled.',
    );
    addCheck(
        'Line spacing does not exceed the beam FWHM.',
        lineCount <= 1 || lineSpacing_um <= Math.max(fwhm_um, 1e-6),
        'Line spacing exceeds the beam FWHM, so adjacent raster lines will have gaps.',
    );
    addCheck(
        'Base absorption coefficient is non-negative.',
        absorptionCoeffBase >= 0,
        'Base absorption coefficient must be greater than or equal to 0.',
    );
    addCheck(
        'Transformed-material absorption coefficient is non-negative.',
        absorptionCoeffTransformed >= 0,
        'Transformed-material absorption coefficient must be greater than or equal to 0.',
    );
    addCheck(
        'Untransformed material absorbs measurable laser energy.',
        absorbedFractionBase >= 1e-6,
        'Untransformed material absorbs effectively zero laser energy with the current settings.',
    );
    addCheck(
        'Transformed material absorbs measurable laser energy.',
        absorbedFractionTransformed >= 1e-6,
        'Transformed material absorbs effectively zero laser energy with the current settings.',
    );
    addCheck(
        'Fourier number stays at or below 0.5.',
        !Number.isFinite(fourierNumber) || fourierNumber <= 0.5,
        `Fourier number is ${fourierNumber.toFixed(3)}, which exceeds the stability limit.`,
    );
    addCheck(
        'The domain is not heavily clipping the beam footprint.',
        sigma_um * 6 <= lxy_um,
        'The domain is too small relative to the beam width, so the laser footprint is heavily clipped by the boundaries.',
    );
    addCheck(
        'The simulated time span covers at least one pulse period.',
        !(pulseRate_hz > 0) || tFinal_s >= pulsePeriod_s,
        'The simulated time span is shorter than one pulse period, so the run does not cover a full repetition cycle.',
    );

    return {
        lxy_um,
        nxy,
        tFinal_s,
        dt_s,
        saveInt,
        rho,
        cp,
        pulseEnergy_uJ,
        sigma_um,
        pulseWidth_ns,
        pulseWidth_s,
        pulseRate_kHz,
        pulseRate_hz,
        scanSpeed_mm_s,
        scanSpeed_m_s,
        lineSpacing_um,
        scanMargin_um,
        filmThickness_um,
        dx_um,
        totalSteps,
        savedFrames,
        pulseCount,
        nodesTotal,
        totalOps,
        lineCount,
        scanWidth_um,
        scanHeight_um,
        absorptionCoeffBase,
        absorptionCoeffTransformed,
        absorbedFractionBase,
        absorbedFractionTransformed,
        sigma_m,
        filmThickness_m,
        fourierNumber,
        pointsPerFwhm,
        pulseTravel_um,
        fwhm_um,
        checks,
        warnings,
    };
}

function renderWarnings(summary) {
    if (!warningList) return;
    const warnings = summary?.warnings || [];
    const checks = summary?.checks || [];
    warningList.innerHTML = '';
    if (!warnings.length) {
        const empty = document.createElement('div');
        empty.className = 'warning-empty';
        empty.textContent = 'No warnings';
        warningList.appendChild(empty);
        return;
    }

    warnings.forEach((warning) => {
        const item = document.createElement('div');
        item.className = 'warning-item';
        item.textContent = warning;
        warningList.appendChild(item);
    });

    if (!warningStatusBadge || !warningStatusIcon || !warningTooltipPassed || !warningTooltipFailed) return;

    const passedChecks = checks.filter((check) => check.passed);
    const failedChecks = checks.filter((check) => !check.passed);
    const populateTooltipList = (container, entries, className) => {
        container.innerHTML = '';
        if (!entries.length) {
            const empty = document.createElement('div');
            empty.className = 'warning-tooltip-item empty';
            empty.textContent = className === 'pass' ? 'No passed checks recorded.' : 'No failed checks.';
            container.appendChild(empty);
            return;
        }
        entries.forEach((entry) => {
            const item = document.createElement('div');
            item.className = `warning-tooltip-item ${className}`;
            item.textContent = entry.label;
            container.appendChild(item);
        });
    };

    const allChecksPassed = failedChecks.length === 0;
    warningStatusBadge.classList.toggle('warning-status-pass', allChecksPassed);
    warningStatusBadge.classList.toggle('warning-status-fail', !allChecksPassed);
    warningStatusBadge.setAttribute(
        'aria-label',
        allChecksPassed
            ? `All ${passedChecks.length} validation checks passed.`
            : `${failedChecks.length} validation checks flagged; ${passedChecks.length} passed.`,
    );
    warningStatusIcon.innerHTML = allChecksPassed ? '&#10003;' : '&#9888;';

    populateTooltipList(warningTooltipPassed, passedChecks, 'pass');
    populateTooltipList(warningTooltipFailed, failedChecks, 'fail');
}

// ============================================================
// COMPUTED INFO DISPLAYS
// ============================================================

function updateComputedInfo() {
    const summary = computeSimulationSummary();
    const lxy_um = getNumberInput('param-lxy', 200);
    const nxy = getIntInput('param-nxy', 200);
    const tFinal_s = getNumberInput('param-tfinal', 1e-8);
    const dt_s = getNumberInput('param-dt', 1e-10);
    const saveInt = Math.max(1, getIntInput('param-saveint', 100));
    const rho = getNumberInput('param-rho', 1100);
    const cp = getNumberInput('param-cp', 1500);
    const pulseEnergy_uJ = getNumberInput('param-pulseenergy', 100);
    const sigma_um = getNumberInput('param-sigma', 21.3);
    const pulseWidth_ns = getNumberInput('param-pulsewidth', 100);
    const pulseRate_kHz = getNumberInput('param-pulserate', 20);
    const filmThickness_um = getNumberInput('param-filmthickness', 10);
    const absorptionCoeffBase = getNumberInput('param-absorption', 1e5);
    const absorptionCoeffTransformed = getNumberInput('param-absorption-transformed', absorptionCoeffBase);

    const dx_um = lxy_um / nxy;
    document.getElementById('info-spatial').textContent = `dx = ${dx_um.toFixed(2)} µm`;

    const totalSteps = dt_s > 0 ? Math.max(0, Math.floor(tFinal_s / dt_s)) : 0;
    const savedFrames = totalSteps > 0 ? Math.floor(totalSteps / saveInt) + 1 : 1;
    const pulseCount = tFinal_s * (pulseRate_kHz * 1e3);
    document.getElementById('info-timesteps').textContent =
        `${totalSteps.toLocaleString()} steps -> ${savedFrames} frames -> ${pulseCount.toFixed(2)} pulses`;

    const rhoCp = rho * cp;
    document.getElementById('info-rhocp').innerHTML =
        `&rho;C<sub>p</sub> = ${formatScaledValue(rhoCp, [[1e9, 'GJ/m^3.K'], [1e6, 'MJ/m^3.K'], [1e3, 'kJ/m^3.K'], [1, 'J/m^3.K']])}`;

    const pulseWidth_s = pulseWidth_ns * 1e-9;
    const peakPower = pulseWidth_s > 0 ? (pulseEnergy_uJ * 1e-6) / pulseWidth_s : 0;
    const sigma_m = sigma_um * 1e-6;
    const filmThickness_m = Math.max(filmThickness_um * 1e-6, 0);
    const peakIrr_Wcm2 = sigma_m > 0 ? (peakPower / (2 * Math.PI * sigma_m * sigma_m)) * 1e-4 : 0;
    const absorbedFractionBase = filmThickness_m > 0 && absorptionCoeffBase > 0
        ? 1 - Math.exp(-absorptionCoeffBase * filmThickness_m)
        : 0;
    const absorbedFractionTransformed = filmThickness_m > 0 && absorptionCoeffTransformed > 0
        ? 1 - Math.exp(-absorptionCoeffTransformed * filmThickness_m)
        : 0;
    const peakVolSourceBase_Wm3 = filmThickness_m > 0 && sigma_m > 0
        ? (peakPower * absorbedFractionBase) / (2 * Math.PI * sigma_m * sigma_m * filmThickness_m)
        : 0;
    const peakVolSourceTransformed_Wm3 = filmThickness_m > 0 && sigma_m > 0
        ? (peakPower * absorbedFractionTransformed) / (2 * Math.PI * sigma_m * sigma_m * filmThickness_m)
        : 0;

    const powerStr = formatScaledValue(peakPower, [[1e6, 'MW'], [1e3, 'kW'], [1, 'W']]);
    const irrStr = formatScaledValue(peakIrr_Wcm2, [[1e9, 'GW/cm^2'], [1e6, 'MW/cm^2'], [1e3, 'kW/cm^2'], [1, 'W/cm^2']]);
    const volSourceBaseStr = formatScaledValue(peakVolSourceBase_Wm3, [[1e12, 'TW/m^3'], [1e9, 'GW/m^3'], [1e6, 'MW/m^3'], [1e3, 'kW/m^3'], [1, 'W/m^3']]);
    const volSourceTransformedStr = formatScaledValue(peakVolSourceTransformed_Wm3, [[1e12, 'TW/m^3'], [1e9, 'GW/m^3'], [1e6, 'MW/m^3'], [1e3, 'kW/m^3'], [1, 'W/m^3']]);
    document.getElementById('info-laser').innerHTML =
        `Peak: ${powerStr} | ${irrStr}<br>Abs base/trans: ${(absorbedFractionBase * 100).toFixed(1)}% / ${(absorbedFractionTransformed * 100).toFixed(1)}%<br>` +
        `Q base/trans: ${volSourceBaseStr} / ${volSourceTransformedStr}`;

    const nodesTotal = nxy * nxy;
    const totalOps = nodesTotal * totalSteps;
    const opsStr = totalOps >= 1e9 ? `${(totalOps / 1e9).toFixed(1)}G` : totalOps >= 1e6 ? `${(totalOps / 1e6).toFixed(1)}M` : `${totalOps.toLocaleString()}`;
    document.getElementById('info-ops').textContent = `${nodesTotal.toLocaleString()} nodes × ${totalSteps.toLocaleString()} steps = ${opsStr} ops`;

    document.getElementById('info-spatial').textContent = `dx = ${summary.dx_um.toFixed(2)} um`;
    document.getElementById('info-timesteps').textContent =
        `${summary.totalSteps.toLocaleString()} steps -> ${summary.savedFrames} frames -> ${summary.pulseCount.toFixed(2)} pulses`;
    const summaryOpsStr = summary.totalOps >= 1e9
        ? `${(summary.totalOps / 1e9).toFixed(1)}G`
        : summary.totalOps >= 1e6
            ? `${(summary.totalOps / 1e6).toFixed(1)}M`
            : `${summary.totalOps.toLocaleString()}`;
    document.getElementById('info-ops').textContent =
        `${summary.nodesTotal.toLocaleString()} nodes x ${summary.totalSteps.toLocaleString()} steps = ${summaryOpsStr} ops`;
    renderWarnings(summary);
    updateRunEstimate(summary);

    // Update laser preview when laser params change
    renderLaserPreview();
}

// ============================================================
// PARAMETER COLLECTION & PERSISTENCE
// ============================================================

function collectParams() {
    return {
        lxy: getNumberInput('param-lxy') * 1e-6,
        nxy: getIntInput('param-nxy'),
        t_final: getNumberInput('param-tfinal'),
        dt: getNumberInput('param-dt'),
        save_interval: Math.max(1, getIntInput('param-saveint')),
        rho: getNumberInput('param-rho'),
        cp: getNumberInput('param-cp'),
        k: getNumberInput('param-k'),
        h_coeff: getNumberInput('param-hcoeff'),
        a_pre: getNumberInput('param-apre'),
        ea: getNumberInput('param-ea'),
        delta_h: getNumberInput('param-deltah'),
        pulse_energy: getNumberInput('param-pulseenergy') * 1e-6,
        sigma: getNumberInput('param-sigma') * 1e-6,
        pulse_width: getNumberInput('param-pulsewidth') * 1e-9,
        pulse_rate: getNumberInput('param-pulserate') * 1e3,
        scan_speed: getNumberInput('param-scanspeed') * 1e-3,
        line_spacing: getNumberInput('param-linespacing') * 1e-6,
        scan_margin: getNumberInput('param-scanmargin') * 1e-6,
        film_thickness: getNumberInput('param-filmthickness') * 1e-6,
        absorption_coeff: getNumberInput('param-absorption'),
        absorption_coeff_transformed: getNumberInput('param-absorption-transformed'),
        t_init: getNumberInput('param-tinit'),
        t_inf: getNumberInput('param-tinf'),
    };
}

function populateParams(params) {
    setFormattedFieldValue('param-lxy', params.lxy * 1e6);
    setFormattedFieldValue('param-nxy', params.nxy);
    setFormattedFieldValue('param-tfinal', params.t_final);
    setFormattedFieldValue('param-dt', params.dt);
    setFormattedFieldValue('param-saveint', params.save_interval);
    setFormattedFieldValue('param-rho', params.rho);
    setFormattedFieldValue('param-cp', params.cp);
    setFormattedFieldValue('param-k', params.k);
    setFormattedFieldValue('param-hcoeff', params.h_coeff);
    setFormattedFieldValue('param-apre', params.a_pre);
    setFormattedFieldValue('param-ea', params.ea);
    setFormattedFieldValue('param-deltah', params.delta_h);
    setFormattedFieldValue('param-pulseenergy', params.pulse_energy * 1e6);
    setFormattedFieldValue('param-sigma', params.sigma * 1e6);
    setFormattedFieldValue('param-pulsewidth', params.pulse_width * 1e9);
    setFormattedFieldValue('param-pulserate', params.pulse_rate * 1e-3);
    setFormattedFieldValue('param-scanspeed', params.scan_speed * 1e3);
    setFormattedFieldValue('param-linespacing', params.line_spacing * 1e6);
    setFormattedFieldValue('param-scanmargin', params.scan_margin * 1e6);
    setFormattedFieldValue('param-filmthickness', params.film_thickness * 1e6);
    setFormattedFieldValue('param-absorption', params.absorption_coeff);
    setFormattedFieldValue('param-absorption-transformed', params.absorption_coeff_transformed ?? params.absorption_coeff);
    setFormattedFieldValue('param-tinit', params.t_init);
    setFormattedFieldValue('param-tinf', params.t_inf);
    updateComputedInfo();
}

function scheduleAutoSave() {
    if (autoSaveTimer) clearTimeout(autoSaveTimer);
    autoSaveTimer = setTimeout(async () => {
        try {
            const params = collectParams();
            await invoke('save_last_params', { params });
            log('info', 'Auto-saved params');
        } catch (e) { log('warn', 'Auto-save failed', String(e)); }
    }, 500);
}

// ============================================================
// PRESET MANAGEMENT
// ============================================================

const presetDropdown = document.getElementById('preset-dropdown');
const presetList = document.getElementById('preset-list');

async function showPresetDropdown() {
    try {
        const presets = await invoke('list_presets');
        presetList.innerHTML = '';
        if (presets.length === 0) {
            presetList.innerHTML = '<div class="preset-item preset-empty">No saved presets</div>';
        } else {
            for (const p of presets) {
                const item = document.createElement('div');
                item.className = 'preset-item';
                const nameSpan = document.createElement('span');
                nameSpan.textContent = p.name;
                nameSpan.className = 'preset-name';
                nameSpan.addEventListener('click', () => {
                    populateParams(p.params);
                    scheduleAutoSave();
                    presetDropdown.classList.add('hidden');
                    log('ok', `Loaded preset: ${p.name}`);
                });
                const delBtn = document.createElement('button');
                delBtn.textContent = '✕';
                delBtn.className = 'btn-icon preset-delete';
                delBtn.title = 'Delete preset';
                delBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    await invoke('delete_preset', { name: p.name });
                    log('ok', `Deleted preset: ${p.name}`);
                    showPresetDropdown();
                });
                item.appendChild(nameSpan);
                item.appendChild(delBtn);
                presetList.appendChild(item);
            }
        }
        presetDropdown.classList.remove('hidden');
    } catch (e) { log('error', 'List presets failed', String(e)); }
}

async function savePresetPrompt() {
    const name = prompt('Enter a name for this parameter preset:');
    if (!name || !name.trim()) return;
    try {
        const params = collectParams();
        await invoke('save_preset', { name: name.trim(), params });
        log('ok', `Saved preset: ${name.trim()}`);
    } catch (e) { log('error', 'Save preset failed', String(e)); }
}

// ============================================================
// STATUS / CANVAS
// ============================================================

function setStatus(state, text) {
    statusIndicator.className = `status ${state}`;
    statusText.textContent = text;
}

function updateParamLayout() {
    if (!paramPanel) return;
    paramPanel.classList.add('two-column');
}

function resizeCanvases() {
    updateParamLayout();

    for (const c of [canvasLaser, canvasTemp, canvasConv, canvasMetrics]) {
        const rect = c.getBoundingClientRect();
        c.width = Math.max(1, Math.floor(rect.width));
        c.height = Math.max(100, Math.floor(rect.height));
    }

    if (canvasScanPreview) {
        const rect = canvasScanPreview.getBoundingClientRect();
        canvasScanPreview.width = Math.max(1, Math.floor(rect.width));
        canvasScanPreview.height = Math.max(1, Math.floor(rect.height));
    }

    if (allFrames.length > 0 && meshData) {
        renderFrame(currentFrameIdx);
    } else {
        renderCurrentColorbars();
        renderLaserPreview();
    }
}

function formatTemperature(value) {
    const span = Math.abs(tempRange[1] - tempRange[0]);
    if (span < 0.01) return value.toFixed(5);
    if (span < 0.1) return value.toFixed(4);
    if (span < 1.0) return value.toFixed(3);
    return value.toFixed(1);
}

function contourOptionsFor(kind) {
    if (kind === 'temp') {
        return {
            contour: {
                enabled: displaySettings.tempContourEnabled,
                threshold: displaySettings.tempContourThreshold,
                color: 'rgba(230, 237, 243, 0.94)',
            },
        };
    }
    if (kind === 'conv') {
        return {
            contour: {
                enabled: displaySettings.convContourEnabled,
                threshold: displaySettings.convContourThreshold,
                color: 'rgba(255, 213, 128, 0.92)',
            },
        };
    }
    return {
        contour: {
            enabled: displaySettings.laserContourEnabled,
            threshold: displaySettings.laserContourThreshold,
            color: 'rgba(138, 216, 255, 0.94)',
        },
    };
}

function setFocusedVizCell(vizId = null) {
    focusedVizId = vizId;
    vizGrid?.classList.toggle('single-focus', Boolean(vizId));
    vizCells.forEach((cell) => {
        cell.classList.toggle('is-focused', cell.id === vizId);
    });
    requestAnimationFrame(() => resizeCanvases());
}

// ============================================================
// RENDERING
// ============================================================

function renderFrame(idx) {
    if (!meshData || idx < 0 || idx >= allFrames.length) return;
    const frame = allFrames[idx];

    renderTriMesh(
        canvasTemp,
        meshData.nodes_x,
        meshData.nodes_y,
        meshData.elements,
        frame.temperature,
        displaySettings.tempColormap,
        tempRange,
        contourOptionsFor('temp'),
    );

    renderTriMesh(
        canvasConv,
        meshData.nodes_x,
        meshData.nodes_y,
        meshData.elements,
        frame.alpha,
        displaySettings.convColormap,
        convRange,
        contourOptionsFor('conv'),
    );

    renderTriMesh(
        canvasLaser,
        meshData.nodes_x,
        meshData.nodes_y,
        meshData.elements,
        frame.laser,
        displaySettings.laserColormap,
        laserRange,
        contourOptionsFor('laser'),
    );
    renderCurrentColorbars();

    if (idx >= 0) {
        renderMetricsChart(canvasMetrics,
            allFrames.slice(0, idx + 1).map(f => f.time),
            allFrames.slice(0, idx + 1).map(f => f.max_temp),
            allFrames.slice(0, idx + 1).map(f => f.avg_conversion),
            allFrames[allFrames.length - 1]?.time ?? null);
    }

    scrubberInfo.textContent = `Frame ${idx}/${allFrames.length - 1} | t=${formatSeconds(frame.time)} | Tmax=${formatTemperature(frame.max_temp)}K`;
}

// ============================================================
// BATCH RESULT
// ============================================================

function computeGlobalRanges(frames) {
    let tMin = Infinity, tMax = -Infinity, cMax = 0, lMax = 0;
    for (const f of frames) {
        if (f.max_temp > tMax) tMax = f.max_temp;
        const fMin = Math.min(...f.temperature);
        if (fMin < tMin) tMin = fMin;
        const cM = Math.max(...f.alpha);
        if (cM > cMax) cMax = cM;
        const lM = Math.max(...f.laser);
        if (lM > lMax) lMax = lM;
    }
    tempRange = [tMin, tMax];
    convRange = [0, Math.max(cMax, 0.01)];
    laserRange = [0, Math.max(lMax, 1)];
}

function handleBatchResult(result) {
    log('ok', `Batch: ${result.frames.length} frames, ${result.elapsed_secs.toFixed(2)}s`);
    meshData = result.mesh;
    allFrames = result.frames;
    currentFrameIdx = 0;
    lastRunElapsedSecs = result.elapsed_secs;
    if (currentRunOps && currentRunOps > 0) {
        const measuredSecsPerOp = result.elapsed_secs / currentRunOps;
        runtimeModelSecsPerOp = 0.65 * runtimeModelSecsPerOp + 0.35 * measuredSecsPerOp;
        window.localStorage?.setItem('runtime_model_secs_per_op', String(runtimeModelSecsPerOp));
    }
    currentRunOps = null;
    computeGlobalRanges(allFrames);
    scrubber.max = allFrames.length - 1;
    scrubber.value = 0;
    scrubberContainer.classList.remove('hidden');
    progressContainer.classList.add('hidden');
    isRunning = false;
    btnRun.disabled = false;
    btnRun.querySelector('.btn-icon-text').textContent = '\u25B6';
    setStatus('complete', 'Done');
    setStatusMeta(`Done in ${formatDuration(result.elapsed_secs)}`, `Size: ${formatBytes(projectFolderSizeBytes)}`);
    resizeCanvases();
    renderFrame(0);
    updateRunEstimate();
    void playCompletionChime();
    refreshProjectStats();
}

function handleSimulationProgress(progress) {
    if (!isRunning) return;

    const clampedProgress = Math.max(0, Math.min(1, progress.progress ?? 0));
    const percentText = formatPercent(clampedProgress);
    const stepText = `${(progress.step ?? 0).toLocaleString()} / ${(progress.num_steps ?? 0).toLocaleString()} steps`;
    const simTimeText = `t = ${formatSeconds(progress.sim_time ?? 0)}`;
    const etaText = Number.isFinite(progress.eta_secs) ? `ETA ${formatDuration(progress.eta_secs)}` : 'ETA --';

    progressContainer.classList.remove('hidden');
    progressFill.style.width = `${(clampedProgress * 100).toFixed(2)}%`;
    progressTextEl.textContent = `Computing ${percentText}`;
    progressDetail.textContent = `${stepText} | ${simTimeText} | ${etaText}`;
    setStatusMeta(`${percentText} complete | ${etaText}`, `Size: ${formatBytes(projectFolderSizeBytes)}`);
}

// ============================================================
// PLAYBACK
// ============================================================

function startPlayback() {
    if (allFrames.length < 2) return;
    isPlaying = true;
    btnPlay.textContent = '\u23F8';
    const baseFps = Math.min(30, Math.max(5, allFrames.length / 3));
    const fps = Math.min(120, Math.max(2, baseFps * playbackSpeedMultiplier));
    playTimer = setInterval(() => {
        currentFrameIdx = (currentFrameIdx + 1) % allFrames.length;
        scrubber.value = currentFrameIdx;
        renderFrame(currentFrameIdx);
    }, 1000 / fps);
}

function stopPlayback() {
    isPlaying = false;
    btnPlay.textContent = '\u25B6';
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
}

// ============================================================
// MP4 EXPORT
// ============================================================

async function exportMP4() {
    if (allFrames.length === 0 || !meshData) {
        log('warn', 'No frames to export');
        return;
    }

    log('info', `Exporting ${allFrames.length} frames as MP4...`);
    setStatus('running', 'Exporting MP4...');
    btnRun.disabled = true;

    // Create an offscreen canvas to compose all 4 panels
    const pw = 800, ph = 400;
    const offscreen = document.createElement('canvas');
    offscreen.width = pw * 2;
    offscreen.height = ph * 2;
    const octx = offscreen.getContext('2d');

    const framesPng = [];
    for (let i = 0; i < allFrames.length; i++) {
        // Render each panel on temp canvases (reuse the existing ones)
        renderFrame(i);
        // Compose 2x2 grid
        octx.fillStyle = '#0d1117';
        octx.fillRect(0, 0, offscreen.width, offscreen.height);
        octx.drawImage(canvasLaser, 0, 0, pw, ph);
        octx.drawImage(canvasTemp, pw, 0, pw, ph);
        octx.drawImage(canvasConv, 0, ph, pw, ph);
        octx.drawImage(canvasMetrics, pw, ph, pw, ph);
        // Labels
        octx.fillStyle = '#e6edf3';
        octx.font = 'bold 14px monospace';
        octx.fillText('Laser Source [W/m^3]', 10, 20);
        octx.fillText('Temperature Field [K]', pw + 10, 20);
        octx.fillText('Reaction Conversion', 10, ph + 20);
        octx.fillText('Time History', pw + 10, ph + 20);
        framesPng.push(offscreen.toDataURL('image/png'));

        if (i % 10 === 0) {
            progressTextEl.textContent = `Encoding frame ${i + 1}/${allFrames.length}`;
        }
    }

    try {
        const outputName = `sim_${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}`;
        const result = await invoke('save_video_frames', { framesPngBase64: framesPng, outputName });
        log('ok', `MP4 saved: ${result}`);
        setStatus('complete', `MP4 saved!`);
    } catch (e) {
        log('error', 'MP4 export failed', String(e));
        setStatus('error', `Export failed: ${e}`);
    }

    btnRun.disabled = false;
    // Restore current frame view
    renderFrame(currentFrameIdx);
}

// ============================================================
// RUN SIMULATION
// ============================================================

async function runSimulation() {
    if (isRunning) return;
    stopPlayback();
    log('info', '--- Starting simulation ---');
    void ensureAudioContext();
    isRunning = true;
    const summary = computeSimulationSummary();
    currentRunOps = summary.totalOps;
    btnRun.disabled = true;
    btnRun.querySelector('.btn-icon-text').textContent = '\u23F3';
    setStatus('running', 'Running');
    setStatusMeta(`Estimated ${formatDuration(Math.max(0.05, summary.totalOps * runtimeModelSecsPerOp))}`, `Size: ${formatBytes(projectFolderSizeBytes)}`);
    progressContainer.classList.remove('hidden');
    progressFill.style.width = '0%';
    progressTextEl.textContent = 'Starting simulation...';
    progressDetail.textContent = `${summary.totalSteps.toLocaleString()} steps scheduled`;
    scrubberContainer.classList.add('hidden');

    try {
        const params = collectParams();
        await invoke('run_simulation', { params });
        log('ok', 'Sim invoked (running in background)');
    } catch (err) {
        log('error', 'Sim invoke failed', String(err));
        setStatus('error', `Error: ${err}`);
        setStatusMeta('Run failed', `Size: ${formatBytes(projectFolderSizeBytes)}`);
        isRunning = false;
        currentRunOps = null;
        btnRun.disabled = false;
        btnRun.querySelector('.btn-icon-text').textContent = '\u25B6';
        progressContainer.classList.add('hidden');
    }
}

// ============================================================
// DEBUG PANEL
// ============================================================

function setupDebugPanel() {
    document.getElementById('btn-debug-toggle').addEventListener('click', () => {
        debugVisible = !debugVisible;
        debugPanel.classList.toggle('hidden', !debugVisible);
    });
    document.getElementById('btn-debug-clear').addEventListener('click', () => {
        debugLog.innerHTML = '';
        log('info', 'Log cleared');
    });
}

// ============================================================
// INIT
// ============================================================

async function init() {
    log('info', 'App initializing...');
    setupDebugPanel();
    if (!initTauriAPI()) {
        setStatus('error', 'Tauri API unavailable');
        debugVisible = true;
        debugPanel.classList.remove('hidden');
        return;
    }

    // Load last-used params, fallback to defaults
    try {
        const last = await invoke('load_last_params');
        if (last) {
            log('ok', 'Restored last-used params');
            populateParams(last);
        } else {
            const defaults = await invoke('get_default_params');
            populateParams(defaults);
        }
    } catch (e) {
        log('warn', 'Param load failed', String(e));
        try { populateParams(await invoke('get_default_params')); } catch { }
        normalizeParameterInputs();
        updateComputedInfo();
    }

    // Event listeners
    await listen('sim-batch-result', (event) => {
        log('event', 'sim-batch-result');
        handleBatchResult(event.payload);
    });
    await listen('sim-progress', (event) => {
        handleSimulationProgress(event.payload);
    });

    // Button handlers
    btnRun.addEventListener('click', runSimulation);
    btnReset.addEventListener('click', async () => {
        try { populateParams(await invoke('get_default_params')); scheduleAutoSave(); } catch { }
    });
    document.getElementById('btn-save-preset').addEventListener('click', savePresetPrompt);
    document.getElementById('btn-load-preset').addEventListener('click', () => {
        if (presetDropdown.classList.contains('hidden')) showPresetDropdown();
        else presetDropdown.classList.add('hidden');
    });
    document.getElementById('btn-export-mp4').addEventListener('click', exportMP4);
    playbackSpeedSelect?.addEventListener('change', () => applyPlaybackSpeedSetting(playbackSpeedSelect.value));

    // Scrubber
    scrubber.addEventListener('input', () => { stopPlayback(); currentFrameIdx = parseInt(scrubber.value); renderFrame(currentFrameIdx); });
    btnPlay.addEventListener('click', () => { if (isPlaying) stopPlayback(); else startPlayback(); });

    // Auto-save on param change + update computed info + laser preview
    document.querySelectorAll('#param-panel input').forEach(input => {
        input.addEventListener('input', () => {
            updateScientificClass(input);
            updateComputedInfo();
            scheduleAutoSave();
        });
        input.addEventListener('blur', () => {
            normalizeParameterInput(input);
            updateComputedInfo();
            scheduleAutoSave();
        });
    });

    [
        displayColormapTemp,
        displayColormapConv,
        displayColormapLaser,
        displayContourTempEnabled,
        displayContourTempThreshold,
        displayContourConvEnabled,
        displayContourConvThreshold,
        displayContourLaserEnabled,
        displayContourLaserThreshold,
    ].forEach((control) => {
        if (!control) return;
        const eventName = control.type === 'number' ? 'input' : 'change';
        control.addEventListener(eventName, applyDisplaySettings);
    });

    vizCells.forEach((cell) => {
        cell.addEventListener('click', () => {
            setFocusedVizCell(focusedVizId === cell.id ? null : cell.id);
        });
    });

    // Close preset dropdown when clicking elsewhere
    document.addEventListener('click', (e) => {
        if (!presetDropdown.contains(e.target) &&
            e.target.id !== 'btn-load-preset') {
            presetDropdown.classList.add('hidden');
        }
    });

    applyDisplaySettings();
    applyPlaybackSpeedSetting(playbackSpeedSelect?.value || playbackSpeedMultiplier);
    updateComputedInfo();
    refreshProjectStats();
    window.addEventListener('resize', resizeCanvases);
    requestAnimationFrame(() => {
        resizeCanvases();
        renderLaserPreview();
    });

    log('ok', 'App initialized');
    setStatus('idle', 'Ready');
    setStatusMeta('Project folder', `Size: ${formatBytes(projectFolderSizeBytes)}`);
}

init().catch(err => {
    log('error', 'Init failed', String(err));
    debugVisible = true;
    if (debugPanel) debugPanel.classList.remove('hidden');
});
