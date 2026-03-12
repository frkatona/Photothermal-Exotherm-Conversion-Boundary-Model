/**
 * renderer.js — Canvas-based triangulated mesh rendering with colormaps.
 */

// ============================================================
// COLORMAPS
// ============================================================

/** Generate the 'inferno' colormap LUT (256 entries). */
function generateInfernoLUT() {
    // Key stops from matplotlib's inferno
    const stops = [
        [0.0, [0, 0, 4]],
        [0.05, [10, 7, 46]],
        [0.1, [31, 12, 72]],
        [0.15, [55, 12, 90]],
        [0.2, [79, 13, 96]],
        [0.25, [105, 15, 97]],
        [0.3, [127, 24, 89]],
        [0.35, [149, 38, 76]],
        [0.4, [168, 55, 60]],
        [0.45, [186, 73, 44]],
        [0.5, [201, 93, 32]],
        [0.55, [213, 115, 22]],
        [0.6, [224, 137, 14]],
        [0.65, [231, 161, 11]],
        [0.7, [237, 185, 19]],
        [0.75, [240, 210, 41]],
        [0.8, [241, 229, 71]],
        [0.85, [242, 242, 105]],
        [0.9, [247, 250, 139]],
        [0.95, [251, 253, 172]],
        [1.0, [252, 255, 164]],
    ];
    return buildLUT(stops);
}

/** Generate the 'viridis' colormap LUT. */
function generateViridisLUT() {
    const stops = [
        [0.0, [68, 1, 84]],
        [0.05, [72, 20, 103]],
        [0.1, [72, 37, 118]],
        [0.15, [67, 55, 129]],
        [0.2, [59, 72, 133]],
        [0.25, [49, 89, 137]],
        [0.3, [40, 106, 137]],
        [0.35, [33, 122, 133]],
        [0.4, [29, 137, 126]],
        [0.45, [32, 152, 116]],
        [0.5, [44, 166, 104]],
        [0.55, [68, 179, 88]],
        [0.6, [94, 190, 70]],
        [0.65, [122, 200, 53]],
        [0.7, [155, 207, 35]],
        [0.75, [189, 213, 25]],
        [0.8, [217, 220, 37]],
        [0.85, [237, 226, 57]],
        [0.9, [249, 234, 82]],
        [0.95, [253, 245, 117]],
        [1.0, [253, 231, 37]],
    ];
    return buildLUT(stops);
}

/** Generate the 'hot' colormap LUT. */
function generateHotLUT() {
    const stops = [
        [0.0, [10, 0, 0]],
        [0.125, [85, 0, 0]],
        [0.25, [170, 0, 0]],
        [0.375, [255, 42, 0]],
        [0.5, [255, 128, 0]],
        [0.625, [255, 213, 0]],
        [0.75, [255, 255, 63]],
        [0.875, [255, 255, 170]],
        [1.0, [255, 255, 255]],
    ];
    return buildLUT(stops);
}

function buildLUT(stops) {
    const lut = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) {
        const t = i / 255;
        // Find surrounding stops
        let lo = 0, hi = stops.length - 1;
        for (let s = 0; s < stops.length - 1; s++) {
            if (t >= stops[s][0] && t <= stops[s + 1][0]) {
                lo = s; hi = s + 1; break;
            }
        }
        const range = stops[hi][0] - stops[lo][0];
        const f = range > 0 ? (t - stops[lo][0]) / range : 0;
        lut[i * 3 + 0] = Math.round(stops[lo][1][0] + f * (stops[hi][1][0] - stops[lo][1][0]));
        lut[i * 3 + 1] = Math.round(stops[lo][1][1] + f * (stops[hi][1][1] - stops[lo][1][1]));
        lut[i * 3 + 2] = Math.round(stops[lo][1][2] + f * (stops[hi][1][2] - stops[lo][1][2]));
    }
    return lut;
}

const COLORMAPS = {
    inferno: generateInfernoLUT(),
    viridis: generateViridisLUT(),
    hot: generateHotLUT(),
};

function lookupColor(lut, t) {
    const idx = Math.max(0, Math.min(255, Math.round(t * 255)));
    return [lut[idx * 3], lut[idx * 3 + 1], lut[idx * 3 + 2]];
}

function formatScalar(n, span = Math.abs(n)) {
    if (!Number.isFinite(n)) return String(n);
    const abs = Math.abs(n);
    if (abs > 0 && (abs < 1e-4 || abs > 1e5)) return n.toExponential(2);
    if (span < 0.01) return n.toFixed(5);
    if (span < 0.1) return n.toFixed(4);
    if (span < 1.0) return n.toFixed(3);
    return n.toFixed(2);
}

// ============================================================
// TRIANGLE MESH RENDERING
// ============================================================

/**
 * Render a triangulated mesh with values mapped through a colormap.
 * @param {HTMLCanvasElement} canvas
 * @param {Float64Array|number[]} nodesX - X coords of mesh nodes 
 * @param {Float64Array|number[]} nodesY - Y coords of mesh nodes
 * @param {Array<[number,number,number]>} elements - Triangle connectivity
 * @param {Float64Array|number[]} values - Scalar value at each node
 * @param {string} colormapName - 'inferno', 'viridis', or 'hot'
 * @param {[number, number]} range - [min, max] for color mapping
 */
export function renderTriMesh(canvas, nodesX, nodesY, elements, values, colormapName, range, options = {}) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (!elements || elements.length === 0) return;

    const lut = COLORMAPS[colormapName] || COLORMAPS.inferno;
    const [vMin, vMax] = range;
    const vRange = vMax - vMin || 1;
    const contour = options.contour || null;
    const contourEnabled = contour?.enabled && Number.isFinite(contour?.threshold);
    const contourThreshold = contour?.threshold ?? null;
    const contourColor = contour?.color || 'rgba(230, 237, 243, 0.92)';

    // Compute domain bounds
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (let i = 0; i < nodesX.length; i++) {
        if (nodesX[i] < xMin) xMin = nodesX[i];
        if (nodesX[i] > xMax) xMax = nodesX[i];
        if (nodesY[i] < yMin) yMin = nodesY[i];
        if (nodesY[i] > yMax) yMax = nodesY[i];
    }

    const margin = 8;
    const plotW = w - 2 * margin;
    const plotH = h - 2 * margin;
    const scaleX = plotW / (xMax - xMin || 1);
    const scaleY = plotH / (yMax - yMin || 1);
    const scale = Math.min(scaleX, scaleY);
    const offsetX = margin + (plotW - scale * (xMax - xMin)) / 2;
    const offsetY = margin + (plotH - scale * (yMax - yMin)) / 2;

    // Draw each triangle with per-triangle average color
    for (let e = 0; e < elements.length; e++) {
        const [n0, n1, n2] = elements[e];

        // Average value for the triangle
        const avgVal = (values[n0] + values[n1] + values[n2]) / 3;
        const t = Math.max(0, Math.min(1, (avgVal - vMin) / vRange));
        const [r, g, b] = lookupColor(lut, t);

        // Map node coords to canvas
        const x0 = offsetX + (nodesX[n0] - xMin) * scale;
        const y0 = offsetY + (yMax - nodesY[n0]) * scale; // Flip Y
        const x1 = offsetX + (nodesX[n1] - xMin) * scale;
        const y1 = offsetY + (yMax - nodesY[n1]) * scale;
        const x2 = offsetX + (nodesX[n2] - xMin) * scale;
        const y2 = offsetY + (yMax - nodesY[n2]) * scale;

        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.closePath();
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fill();
    }

    if (!contourEnabled) return;

    const intersectEdge = (xa, ya, va, xb, yb, vb, level) => {
        const da = va - level;
        const db = vb - level;

        if (Math.abs(da) < 1e-12 && Math.abs(db) < 1e-12) return null;
        if (Math.abs(da) < 1e-12) return { x: xa, y: ya };
        if (Math.abs(db) < 1e-12) return { x: xb, y: yb };
        if (da * db > 0) return null;

        const t = (level - va) / (vb - va);
        return {
            x: xa + (xb - xa) * t,
            y: ya + (yb - ya) * t,
        };
    };

    const dedupePoints = (points) => {
        const unique = [];
        for (const point of points) {
            if (!point) continue;
            const exists = unique.some((candidate) =>
                Math.abs(candidate.x - point.x) < 0.5 && Math.abs(candidate.y - point.y) < 0.5
            );
            if (!exists) unique.push(point);
        }
        return unique;
    };

    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = contourColor;
    ctx.lineWidth = 1.1;
    ctx.shadowColor = 'rgba(0, 0, 0, 0.4)';
    ctx.shadowBlur = 2;

    for (let e = 0; e < elements.length; e++) {
        const [n0, n1, n2] = elements[e];
        const x0 = offsetX + (nodesX[n0] - xMin) * scale;
        const y0 = offsetY + (yMax - nodesY[n0]) * scale;
        const x1 = offsetX + (nodesX[n1] - xMin) * scale;
        const y1 = offsetY + (yMax - nodesY[n1]) * scale;
        const x2 = offsetX + (nodesX[n2] - xMin) * scale;
        const y2 = offsetY + (yMax - nodesY[n2]) * scale;

        const points = dedupePoints([
            intersectEdge(x0, y0, values[n0], x1, y1, values[n1], contourThreshold),
            intersectEdge(x1, y1, values[n1], x2, y2, values[n2], contourThreshold),
            intersectEdge(x2, y2, values[n2], x0, y0, values[n0], contourThreshold),
        ]);

        if (points.length >= 2) {
            ctx.moveTo(points[0].x, points[0].y);
            ctx.lineTo(points[1].x, points[1].y);
        }
    }

    ctx.stroke();
    ctx.restore();
}

/**
 * Render a colorbar gradient on a div element.
 */
export function renderColorbar(element, colormapName, min, max) {
    const lut = COLORMAPS[colormapName] || COLORMAPS.inferno;
    const stops = [];
    for (let i = 0; i <= 10; i++) {
        const t = i / 10;
        const [r, g, b] = lookupColor(lut, t);
        stops.push(`rgb(${r},${g},${b}) ${t * 100}%`);
    }
    element.style.background = `linear-gradient(0deg, ${stops.join(', ')})`;

    // Format range for display
    const fmtNum = (n) => {
        if (Math.abs(n) < 0.01 || Math.abs(n) > 1e5) return n.toExponential(2);
        const span = Math.abs(max - min);
        if (span < 0.01) return n.toFixed(5);
        if (span < 0.1) return n.toFixed(4);
        if (span < 1.0) return n.toFixed(3);
        return n.toFixed(2);
    };
    element.setAttribute('data-min', fmtNum(min));
    element.setAttribute('data-max', fmtNum(max));
    element.setAttribute('data-range', `${fmtNum(min)}  —  ${fmtNum(max)}`);
}

// ============================================================
// METRICS LINE CHART
// ============================================================

/**
 * Render a dual-axis line chart.
 * @param {HTMLCanvasElement} canvas
 * @param {number[]} times - X axis (will be displayed in µs)
 * @param {number[]} maxTemps - Left Y axis data
 * @param {number[]} avgConvs - Right Y axis data
 */
export function renderMetricsChart(canvas, times, maxTemps, avgConvs, xMaxFixed = null) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (times.length < 1) return;

    const margin = { top: 20, right: 60, bottom: 36, left: 60 };
    const plotW = w - margin.left - margin.right;
    const plotH = h - margin.top - margin.bottom;

    // Convert times to µs
    const xMin = 0;
    const xMax = Math.max(xMaxFixed ?? Math.max(...times), 1e-12);

    const tMin = Math.min(...maxTemps);
    const tMax = Math.max(...maxTemps);
    const tRange = tMax - tMin || 1;

    const cMax = Math.max(...avgConvs, 0.01);

    const mapX = (x) => margin.left + ((x - xMin) / (xMax - xMin || 1)) * plotW;
    const mapTY = (y) => margin.top + plotH - ((y - tMin) / tRange) * plotH;
    const mapCY = (y) => margin.top + plotH - (y / cMax) * plotH;

    // Grid
    ctx.strokeStyle = 'rgba(48, 54, 61, 0.6)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = margin.top + (i / 4) * plotH;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(margin.left + plotW, y);
        ctx.stroke();
    }

    const drawSeries = (values, mapY, strokeStyle) => {
        ctx.strokeStyle = strokeStyle;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < times.length; i++) {
            const x = mapX(times[i]);
            const y = mapY(values[i]);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        if (times.length === 1) {
            ctx.fillStyle = strokeStyle;
            ctx.beginPath();
            ctx.arc(mapX(times[0]), mapY(values[0]), 3, 0, Math.PI * 2);
            ctx.fill();
        }
    };

    drawSeries(maxTemps, mapTY, '#f85149');
    drawSeries(avgConvs, mapCY, '#3fb950');

    // Axes
    ctx.strokeStyle = '#484f58';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + plotH);
    ctx.lineTo(margin.left + plotW, margin.top + plotH);
    ctx.stroke();

    // Labels
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#8b949e';
    ctx.fillText('Time [µs]', margin.left + plotW / 2, h - 4);

    ctx.clearRect(margin.left + plotW / 2 - 40, h - 16, 80, 14);
    ctx.fillText('Time [s]', margin.left + plotW / 2, h - 4);

    // X axis ticks
    const numXTicks = 5;
    for (let i = 0; i <= numXTicks; i++) {
        const xVal = xMin + (i / numXTicks) * (xMax - xMin);
        const x = mapX(xVal);
        ctx.fillText(formatScalar(xVal, xMax - xMin), x, margin.top + plotH + 14);
    }

    // Left Y axis (Temperature)
    ctx.save();
    ctx.translate(10, margin.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#f85149';
    ctx.fillText('Max T [K]', 0, 0);
    ctx.restore();

    ctx.textAlign = 'right';
    ctx.fillStyle = '#f85149';
    for (let i = 0; i <= 4; i++) {
        const val = tMin + (i / 4) * tRange;
        const y = margin.top + plotH - (i / 4) * plotH;
        ctx.fillText(formatScalar(val, tRange), margin.left - 4, y + 3);
    }

    // Right Y axis (Conversion)
    ctx.save();
    ctx.translate(w - 6, margin.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#3fb950';
    ctx.textAlign = 'center';
    ctx.fillText('Avg Conv [-]', 0, 0);
    ctx.restore();

    ctx.textAlign = 'left';
    ctx.fillStyle = '#3fb950';
    for (let i = 0; i <= 4; i++) {
        const val = (i / 4) * cMax;
        const y = margin.top + plotH - (i / 4) * plotH;
        ctx.fillText(val.toFixed(4), margin.left + plotW + 4, y + 3);
    }

    // Legend
    const legendX = margin.left + 8;
    const legendY = margin.top + 10;
    ctx.font = '11px Inter, sans-serif';

    ctx.fillStyle = '#f85149';
    ctx.fillRect(legendX, legendY - 4, 12, 3);
    ctx.fillText('Max Temp', legendX + 16, legendY);

    ctx.fillStyle = '#3fb950';
    ctx.fillRect(legendX, legendY + 12, 12, 3);
    ctx.fillText('Avg Conv', legendX + 16, legendY + 16);
}
