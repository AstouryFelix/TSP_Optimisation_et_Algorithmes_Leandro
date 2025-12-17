// ==================== ÉTAT DE L'APPLICATION ====================
const state = {
    tspData: null,      // Données du fichier .tsp
    outData: null,      // Données du fichier .out
    jsonData: null,     // Données du fichier .json

    // Viewport (zoom et pan)
    offsetX: 0,
    offsetY: 0,
    scale: 1,

    // Interaction
    isDragging: false,
    lastX: 0,
    lastY: 0,
};

// ==================== ÉLÉMENTS DOM ====================
const elements = {
    tspFile: document.getElementById('tsp-file'),
    outFile: document.getElementById('out-file'),
    jsonFile: document.getElementById('json-file'),
    tspStatus: document.getElementById('tsp-status'),
    outStatus: document.getElementById('out-status'),
    jsonStatus: document.getElementById('json-status'),
    visualizeBtn: document.getElementById('visualize-btn'),
    stats: document.getElementById('stats'),
    canvas: document.getElementById('tsp-canvas'),
    showCities: document.getElementById('show-cities'),
    showPath: document.getElementById('show-path'),
    showInitial: document.getElementById('show-initial'),
    showLabels: document.getElementById('show-labels'),
    resetZoom: document.getElementById('reset-zoom'),
};

const ctx = elements.canvas.getContext('2d');

// ==================== INITIALISATION ====================
function init() {
    setupCanvas();
    setupEventListeners();
    console.log('TSP Visualizer initialisé');
}

function setupCanvas() {
    const container = elements.canvas.parentElement;
    elements.canvas.width = container.clientWidth;
    elements.canvas.height = container.clientHeight;
}

function setupEventListeners() {
    // Upload de fichiers
    elements.tspFile.addEventListener('change', handleTSPFile);
    elements.outFile.addEventListener('change', handleOUTFile);
    elements.jsonFile.addEventListener('change', handleJSONFile);

    // Bouton de visualisation
    elements.visualizeBtn.addEventListener('click', visualize);

    // Options d'affichage
    elements.showCities.addEventListener('change', visualize);
    elements.showPath.addEventListener('change', visualize);
    elements.showInitial.addEventListener('change', visualize);
    elements.showLabels.addEventListener('change', visualize);

    // Reset zoom
    elements.resetZoom.addEventListener('click', resetZoom);

    // Interaction canvas (zoom et pan)
    elements.canvas.addEventListener('mousedown', handleMouseDown);
    elements.canvas.addEventListener('mousemove', handleMouseMove);
    elements.canvas.addEventListener('mouseup', handleMouseUp);
    elements.canvas.addEventListener('wheel', handleWheel);

    // Resize
    window.addEventListener('resize', () => {
        setupCanvas();
        visualize();
    });
}

// ==================== GESTION DES FICHIERS ====================
function handleTSPFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        state.tspData = parseTSPFile(e.target.result);
        elements.tspStatus.textContent = `✅ ${file.name} chargé`;
        elements.tspStatus.style.color = '#10b981';
        checkReadyToVisualize();
        updateStats();
    };
    reader.readAsText(file);
}

function handleOUTFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        state.outData = parseOUTFile(e.target.result);
        elements.outStatus.textContent = `✅ ${file.name} chargé`;
        elements.outStatus.style.color = '#10b981';
        checkReadyToVisualize();
        updateStats();
    };
    reader.readAsText(file);
}

function handleJSONFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        state.jsonData = JSON.parse(e.target.result);
        elements.jsonStatus.textContent = `✅ ${file.name} chargé`;
        elements.jsonStatus.style.color = '#10b981';
        checkReadyToVisualize();
        updateStats();
    };
    reader.readAsText(file);
}

// ==================== PARSING DES FICHIERS ====================
function parseTSPFile(content) {
    const lines = content.split('\n');
    let n = 0;
    let edgeWeightType = 'EUC_2D';
    const coords = [];
    let inCoordSection = false;

    for (const line of lines) {
        const trimmed = line.trim();

        if (trimmed.startsWith('DIMENSION')) {
            n = parseInt(trimmed.split(':')[1].trim());
        } else if (trimmed.startsWith('EDGE_WEIGHT_TYPE')) {
            edgeWeightType = trimmed.split(':')[1].trim();
        } else if (trimmed === 'NODE_COORD_SECTION') {
            inCoordSection = true;
        } else if (trimmed === 'EOF') {
            break;
        } else if (inCoordSection && trimmed) {
            const parts = trimmed.split(/\s+/);
            if (parts.length >= 3) {
                coords.push({
                    id: parseInt(parts[0]),
                    x: parseFloat(parts[1]),
                    y: parseFloat(parts[2])
                });
            }
        }
    }

    return { n, edgeWeightType, coords };
}

function parseOUTFile(content) {
    const lines = content.trim().split('\n');
    const path = lines[0].split(/\s+/).map(Number);
    const cost = lines[1] ? parseInt(lines[1]) : null;

    return { path, cost };
}

// ==================== VÉRIFICATION ET STATS ====================
function checkReadyToVisualize() {
    // On peut visualiser si on a le .tsp
    const ready = state.tspData !== null;
    elements.visualizeBtn.disabled = !ready;
}

function updateStats() {
    let html = '<h3>📊 Statistiques</h3>';

    if (state.jsonData) {
        // Affichage des stats depuis le JSON
        html += `
            <div class="stat-item">
                <span class="stat-label">Instance :</span>
                <span class="stat-value">${state.jsonData.instance}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Nombre de villes :</span>
                <span class="stat-value">${state.jsonData.n_cities}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Coût initial :</span>
                <span class="stat-value">${state.jsonData.initial_cost}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Coût optimisé :</span>
                <span class="stat-value">${state.jsonData.optimized_cost}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Amélioration :</span>
                <span class="stat-value" style="color: #10b981">-${state.jsonData.improvement} (-${((state.jsonData.improvement / state.jsonData.initial_cost) * 100).toFixed(1)}%)</span>
            </div>
        `;
    } else if (state.tspData) {
        html += `
            <div class="stat-item">
                <span class="stat-label">Nombre de villes :</span>
                <span class="stat-value">${state.tspData.n}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Type :</span>
                <span class="stat-value">${state.tspData.edgeWeightType}</span>
            </div>
        `;

        if (state.outData) {
            html += `
                <div class="stat-item">
                    <span class="stat-label">Coût de la solution :</span>
                    <span class="stat-value">${state.outData.cost || 'N/A'}</span>
                </div>
            `;
        }
    } else {
        html += '<p>Chargez un fichier pour voir les statistiques</p>';
    }

    elements.stats.innerHTML = html;
}

// ==================== VISUALISATION ====================
function visualize() {
    if (!state.tspData && !state.jsonData) return;

    // Clear canvas
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

    // Obtenir les coordonnées et le chemin
    let coords, path, initialPath;

    if (state.jsonData) {
        // Mode JSON complet
        coords = state.jsonData.coordinates.map((c, i) => ({ id: i + 1, x: c[0], y: c[1] }));
        path = state.jsonData.optimized_path;
        initialPath = state.jsonData.initial_path;
    } else if (state.tspData) {
        // Mode .tsp + .out
        coords = state.tspData.coords;
        path = state.outData ? state.outData.path : null;
    }

    if (!coords || coords.length === 0) return;

    // Normaliser les coordonnées pour le canvas
    const normalized = normalizeCoordinates(coords);

    // Dessiner la solution initiale si demandée
    if (elements.showInitial.checked && initialPath) {
        drawPath(normalized, initialPath, '#ef4444', 1, 0.3);
    }

    // Dessiner le chemin optimisé
    if (elements.showPath.checked && path) {
        drawPath(normalized, path, '#3b82f6', 2, 1);
    }

    // Dessiner les villes
    if (elements.showCities.checked) {
        drawCities(normalized);
    }

    // Dessiner les labels si demandés
    if (elements.showLabels.checked) {
        drawLabels(normalized);
    }
}

function normalizeCoordinates(coords) {
    // Trouver les limites
    const xs = coords.map(c => c.x);
    const ys = coords.map(c => c.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const rangeX = maxX - minX;
    const rangeY = maxY - minY;

    // Padding
    const padding = 50;
    const width = elements.canvas.width - 2 * padding;
    const height = elements.canvas.height - 2 * padding;

    // Échelle pour garder le ratio d'aspect
    const scaleX = width / rangeX;
    const scaleY = height / rangeY;
    const scale = Math.min(scaleX, scaleY);

    // Normaliser
    return coords.map(c => ({
        id: c.id,
        x: padding + (c.x - minX) * scale + state.offsetX,
        y: padding + (c.y - minY) * scale + state.offsetY
    }));
}

function drawCities(coords) {
    ctx.fillStyle = '#f1f5f9';
    coords.forEach(c => {
        ctx.beginPath();
        ctx.arc(c.x * state.scale, c.y * state.scale, 4, 0, Math.PI * 2);
        ctx.fill();
    });
}

function drawPath(coords, path, color, lineWidth, opacity) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.globalAlpha = opacity;

    ctx.beginPath();
    for (let i = 0; i < path.length; i++) {
        const cityIndex = path[i];
        const coord = coords[cityIndex];

        if (i === 0) {
            ctx.moveTo(coord.x * state.scale, coord.y * state.scale);
        } else {
            ctx.lineTo(coord.x * state.scale, coord.y * state.scale);
        }
    }

    // Fermer le cycle
    const firstCoord = coords[path[0]];
    ctx.lineTo(firstCoord.x * state.scale, firstCoord.y * state.scale);
    ctx.stroke();

    ctx.globalAlpha = 1;
}

function drawLabels(coords) {
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Inter';
    coords.forEach(c => {
        ctx.fillText(c.id.toString(), c.x * state.scale + 6, c.y * state.scale - 6);
    });
}

// ==================== INTERACTION (ZOOM ET PAN) ====================
function handleMouseDown(e) {
    state.isDragging = true;
    state.lastX = e.offsetX;
    state.lastY = e.offsetY;
}

function handleMouseMove(e) {
    if (!state.isDragging) return;

    const dx = e.offsetX - state.lastX;
    const dy = e.offsetY - state.lastY;

    state.offsetX += dx / state.scale;
    state.offsetY += dy / state.scale;

    state.lastX = e.offsetX;
    state.lastY = e.offsetY;

    visualize();
}

function handleMouseUp() {
    state.isDragging = false;
}

function handleWheel(e) {
    e.preventDefault();

    const zoomIntensity = 0.1;
    const delta = e.deltaY > 0 ? -zoomIntensity : zoomIntensity;

    state.scale += delta;
    state.scale = Math.max(0.5, Math.min(5, state.scale));

    visualize();
}

function resetZoom() {
    state.offsetX = 0;
    state.offsetY = 0;
    state.scale = 1;
    visualize();
}

// ==================== DÉMARRAGE ====================
init();
