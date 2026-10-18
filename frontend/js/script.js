/**
 * PADDYSCAN AI - COMPLETE FRONTEND CONTROLLER
 * Full integration with Flask PyTorch DualViewEfficientNetB0 Backend
 */

// ==============================================================================
// 1. GLOBAL STATE & CONFIGURATION
// ==============================================================================

function getApiBaseUrl() {
  // Clear any stale local overrides if on https or tunnel
  try {
    const saved = localStorage.getItem('paddyscan_api_url');
    if (saved && (window.location.protocol === 'https:' || saved.includes('127.0.0.1') || saved.includes('localhost'))) {
      localStorage.removeItem('paddyscan_api_url');
    }
  } catch (e) {}

  // 1. If running under HTTP or HTTPS (Dev Tunnels, port 5000, custom domains)
  if (window.location.protocol.startsWith('http')) {
    // Only if served by a separate dev server on port 5500 (Live Server)
    if (window.location.port === '5500') {
      const host = window.location.hostname || '127.0.0.1';
      return `http://${host}:5000`;
    }
    // In all standard web & tunnel setups (e.g. devtunnels.ms, localhost:5000, LAN IP),
    // use relative URL '' so all API requests (/api/...) match the page origin and protocol.
    return '';
  }

  // 2. Fallback for file:// protocol
  return 'http://127.0.0.1:5000';
}

const API_BASE_URL = getApiBaseUrl();

const state = {
  activeSection: 'front-view-section',
  modelInfo: null,

  // Front-View Analysis State (Single Image Only)
  frontFile: null,
  frontResult: null,

  // Dual-View Analysis State
  dualFrontFile: null,
  dualBackFile: null,
  dualResult: null,

  // Batch Analysis State
  batchFiles: [],
  batchResults: null,
  activeFilter: 'all',
  batchChart: null
};

// ==============================================================================
// 2. COMPLETE RESULTS & STATE RESET
// ==============================================================================

function resetAllResults(notify = false) {
  // 1. Reset Front-View (T1) State & UI
  state.frontFile = null;
  state.frontResult = null;
  const frontInput = document.getElementById('frontFileInput');
  if (frontInput) frontInput.value = '';
  document.getElementById('frontPrompt')?.classList.remove('d-none');
  document.getElementById('frontPreviewBox')?.classList.add('d-none');
  document.getElementById('removeFrontBtn')?.classList.add('d-none');
  const frontImg = document.getElementById('frontImgPreview');
  if (frontImg) frontImg.src = '';
  const frontName = document.getElementById('frontFilenameText');
  if (frontName) frontName.textContent = '';
  const frontSize = document.getElementById('frontFilesizeText');
  if (frontSize) frontSize.textContent = '';
  const analyzeFrontBtn = document.getElementById('analyzeFrontBtn');
  if (analyzeFrontBtn) analyzeFrontBtn.disabled = true;

  const frontEmpty = document.getElementById('frontEmptyState');
  if (frontEmpty) frontEmpty.style.display = 'block';
  const frontResult = document.getElementById('frontResultCard');
  if (frontResult) frontResult.style.display = 'none';
  const frontError = document.getElementById('frontErrorCard');
  if (frontError) frontError.style.display = 'none';
  const exportFrontBtn = document.getElementById('exportFrontCsvBtn');
  if (exportFrontBtn) exportFrontBtn.style.display = 'none';

  // 2. Reset Dual-View (T1 + B1) State & UI
  state.dualFrontFile = null;
  state.dualBackFile = null;
  state.dualResult = null;
  const dualFrontInput = document.getElementById('dualFrontInput');
  if (dualFrontInput) dualFrontInput.value = '';
  const dualBackInput = document.getElementById('dualBackInput');
  if (dualBackInput) dualBackInput.value = '';

  document.getElementById('dualFrontPrompt')?.classList.remove('d-none');
  document.getElementById('dualFrontPreviewBox')?.classList.add('d-none');
  document.getElementById('removeDualFrontBtn')?.classList.add('d-none');
  const dualFrontImg = document.getElementById('dualFrontImg');
  if (dualFrontImg) dualFrontImg.src = '';
  const dualFrontFilename = document.getElementById('dualFrontFilename');
  if (dualFrontFilename) dualFrontFilename.textContent = '';

  document.getElementById('dualBackPrompt')?.classList.remove('d-none');
  document.getElementById('dualBackPreviewBox')?.classList.add('d-none');
  document.getElementById('removeDualBackBtn')?.classList.add('d-none');
  const dualBackImg = document.getElementById('dualBackImg');
  if (dualBackImg) dualBackImg.src = '';
  const dualBackFilename = document.getElementById('dualBackFilename');
  if (dualBackFilename) dualBackFilename.textContent = '';

  const pairBadge = document.getElementById('pairBadge');
  if (pairBadge) pairBadge.textContent = '0/2 Ready';
  const pairDot = document.getElementById('pairCheckDot');
  if (pairDot) pairDot.className = 'pair-dot';
  const pairText = document.getElementById('pairCheckText');
  if (pairText) pairText.textContent = 'Upload both T1 and B1 images';
  const analyzeDualBtn = document.getElementById('analyzeDualBtn');
  if (analyzeDualBtn) analyzeDualBtn.disabled = true;

  const dualEmpty = document.getElementById('dualEmptyState');
  if (dualEmpty) dualEmpty.style.display = 'block';
  const dualResult = document.getElementById('dualResultCard');
  if (dualResult) dualResult.style.display = 'none';
  const dualError = document.getElementById('dualErrorCard');
  if (dualError) dualError.style.display = 'none';

  // 3. Reset Batch Analysis State & UI
  state.batchFiles = [];
  state.batchResults = null;
  state.batchFilesMap?.clear();
  state.batchUrlCache?.clear();
  if (state.batchChart) {
    try { state.batchChart.destroy(); } catch (_) {}
    state.batchChart = null;
  }
  const folderInput = document.getElementById('batchFolderInput');
  if (folderInput) folderInput.value = '';
  const filesInput = document.getElementById('batchFilesInput');
  if (filesInput) filesInput.value = '';

  const batchStagedTotal = document.getElementById('batchStagedTotal');
  if (batchStagedTotal) batchStagedTotal.textContent = '0 files';
  const batchStagedDetails = document.getElementById('batchStagedDetails');
  if (batchStagedDetails) batchStagedDetails.textContent = '';

  const runBatchBtn = document.getElementById('runBatchBtn');
  if (runBatchBtn) runBatchBtn.disabled = true;

  const batchProgress = document.getElementById('batchProgressWrapper');
  if (batchProgress) batchProgress.style.display = 'none';
  const batchProgressBar = document.getElementById('batchProgressBar');
  if (batchProgressBar) batchProgressBar.style.width = '0%';

  const batchResultsWrapper = document.getElementById('batchResultsWrapper');
  if (batchResultsWrapper) batchResultsWrapper.style.display = 'none';
  const batchTableBody = document.getElementById('batchTableBody');
  if (batchTableBody) batchTableBody.innerHTML = '';
  const exportBatchBtn = document.getElementById('exportBatchCsvBtn');
  if (exportBatchBtn) exportBatchBtn.style.display = 'none';

  // 4. Clear all generic file inputs on page
  document.querySelectorAll('input[type="file"]').forEach(inp => {
    try { inp.value = ''; } catch (_) {}
  });

  if (notify) {
    showAlert('All analysis results and uploads have been reset.', 'info');
  }
}

// Make globally accessible
window.resetAllResults = resetAllResults;

// ==============================================================================
// 3. INITIALIZATION
// ==============================================================================

document.addEventListener('DOMContentLoaded', () => {
  resetAllResults();
  initNavigation();
  initSystemHealth();
  initFrontViewSection();
  initDualViewSection();
  initBatchSection();
  initTrainingCurvesSection();
});

// ==============================================================================
// 3. NAVIGATION LOGIC
// ==============================================================================

function initNavigation() {
  const navButtons = document.querySelectorAll('.nav-item-btn');
  const sections = document.querySelectorAll('.content-section');
  const sidebar = document.getElementById('appSidebar');
  const openSidebarBtn = document.getElementById('openSidebarBtn');
  const closeSidebarBtn = document.getElementById('closeSidebarBtn');

  const pageTitles = {
    'front-view-section': {
      title: 'Front-View Analysis (T1)',
      caption: 'Evaluate adaxial leaf surfaces via Siamese feature extraction fallback'
    },
    'dual-view-section': {
      title: 'Dual-View Analysis (T1 + D1)',
      caption: 'Complete 2560D Siamese feature fusion of front and back leaf surfaces'
    },
    'batch-section': {
      title: 'Folder-Based Batch Analysis',
      caption: 'Automated leaf pair detection, mode strategies, and distribution analytics'
    },
    'training-graphs-section': {
      title: 'Training & Evaluation Curves',
      caption: '50-Epoch Dual-View EfficientNet-B0 + cGAN Accuracy & Cross-Entropy Loss Convergence'
    },
    'model-info-section': {
      title: 'Model Architecture: EfficientNet-B0 + cGAN',
      caption: 'Conditional GAN dataset synthesis and Dual-Stream Siamese EfficientNet-B0 backbone'
    },
    'disease-guide-section': {
      title: 'Paddy Disease Field Guide',
      caption: 'Botanical symptoms, pathogen etiology, and agronomic management'
    }
  };

  navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      if (!targetId) return;

      navButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      sections.forEach(sec => {
        sec.classList.remove('active');
        if (sec.id === targetId) {
          sec.classList.add('active');
        }
      });

      state.activeSection = targetId;

      // Scroll smoothly to top of view
      window.scrollTo({ top: 0, behavior: 'smooth' });

      // Update Topbar
      const info = pageTitles[targetId] || { title: 'PaddyScan AI', caption: '' };
      document.getElementById('pageTitle').textContent = info.title;
      document.getElementById('pageCaption').textContent = info.caption;

      // Trigger graph drawing when training curves section is selected
      if (targetId === 'training-graphs-section') {
        setTimeout(() => {
          renderTrainingCurves(true);
        }, 60);
      }

      // Mobile close sidebar
      if (window.innerWidth < 992) {
        sidebar.classList.remove('show');
        document.getElementById('sidebarBackdrop')?.classList.remove('show');
      }
    });
  });

  const sidebarBackdrop = document.getElementById('sidebarBackdrop');

  if (openSidebarBtn) {
    openSidebarBtn.addEventListener('click', () => {
      sidebar.classList.add('show');
      sidebarBackdrop?.classList.add('show');
    });
  }
  if (closeSidebarBtn) {
    closeSidebarBtn.addEventListener('click', () => {
      sidebar.classList.remove('show');
      sidebarBackdrop?.classList.remove('show');
    });
  }
  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener('click', () => {
      sidebar.classList.remove('show');
      sidebarBackdrop.classList.remove('show');
    });
  }

  // Refresh connection, health, and reset results button
  document.getElementById('checkApiBtn')?.addEventListener('click', () => {
    resetAllResults();
    initSystemHealth(true);
  });
}

// ==============================================================================
// 4. API & SYSTEM HEALTH
// ==============================================================================

async function initSystemHealth(isManual = false) {
  const pulse = document.getElementById('backendPulse');
  const statusText = document.getElementById('backendStatusText');
  const deviceText = document.getElementById('backendDeviceText');

  // Topbar elements
  const refreshBtn = document.getElementById('checkApiBtn');
  const refreshIcon = document.getElementById('refreshBtnIcon');
  const refreshText = document.getElementById('refreshBtnText');
  const topbarChip = document.getElementById('topbarModelChip');
  const topbarModelIcon = document.getElementById('topbarModelIcon');
  const topbarModelName = document.getElementById('topbarModelName');
  const topbarModelMobile = document.getElementById('topbarModelMobile');
  const topbarModelArch = document.getElementById('topbarModelArch');

  if (pulse) pulse.className = 'status-pulse';
  if (statusText) statusText.textContent = 'Connecting...';

  // If triggered by clicking Refresh button, show spinning animation and loading state
  if (isManual && refreshBtn) {
    refreshBtn.disabled = true;
    if (refreshIcon) refreshIcon.classList.add('spin-anim');
    if (refreshText) refreshText.textContent = 'Checking...';
  }

  const startTime = Date.now();

  try {
    const res = await fetch(`${API_BASE_URL}/api/model-info`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.modelInfo = data;

    // Minimum delay of 500ms when manually refreshed so user sees rotation animation
    if (isManual) {
      const elapsed = Date.now() - startTime;
      if (elapsed < 500) {
        await new Promise(resolve => setTimeout(resolve, 500 - elapsed));
      }
    }

    if (data.loaded) {
      if (pulse) pulse.className = 'status-pulse connected';
      if (statusText) statusText.textContent = 'Model Online';
      if (deviceText) deviceText.innerHTML = `<i class="bi bi-cpu me-1"></i> ${data.device_name || data.device}`;

      // Update Topbar Model Badge Chip
      if (topbarChip) topbarChip.className = 'model-badge-chip';
      if (topbarModelIcon) topbarModelIcon.className = 'bi bi-shield-check text-success me-1';
      if (topbarModelName) topbarModelName.textContent = data.checkpoint || 'best_dual_efficientnet_b0.pth';
      if (topbarModelMobile) topbarModelMobile.textContent = 'Model Active';
      if (topbarModelArch) topbarModelArch.textContent = `${data.model_name || 'EfficientNet-B0'} + cGAN`;

      // Update Model Specs Section
      renderModelSpecs(data);
      renderDiseaseGuide(data.disease_reference);

      // Feedback for manual refresh click
      if (isManual) {
        if (refreshIcon) refreshIcon.className = 'bi bi-check-circle-fill text-success me-1';
        if (refreshText) refreshText.textContent = 'Refreshed!';
        showAlert(`Results reset & Backend Online: Connected to ${data.model_name || 'EfficientNet-B0'} (${data.device_name || data.device || 'CPU'}).`, 'success');
        
        setTimeout(() => {
          if (refreshIcon) refreshIcon.className = 'bi bi-arrow-repeat me-1';
          if (refreshText) refreshText.textContent = 'Refresh';
        }, 1800);
      }
    } else {
      if (pulse) pulse.className = 'status-pulse degraded';
      if (statusText) statusText.textContent = 'Model Degraded';
      if (deviceText) deviceText.textContent = data.error || 'Failed to load weights';

      if (topbarChip) topbarChip.className = 'model-badge-chip degraded';
      if (topbarModelIcon) topbarModelIcon.className = 'bi bi-exclamation-triangle-fill text-warning me-1';
      if (topbarModelMobile) topbarModelMobile.textContent = 'Degraded';

      if (isManual) {
        if (refreshIcon) refreshIcon.className = 'bi bi-exclamation-triangle me-1';
        if (refreshText) refreshText.textContent = 'Warning';
        showAlert(`Model checkpoint issue: ${data.error}`, 'warning');

        setTimeout(() => {
          if (refreshIcon) refreshIcon.className = 'bi bi-arrow-repeat me-1';
          if (refreshText) refreshText.textContent = 'Refresh';
        }, 1800);
      }
    }
  } catch (err) {
    if (pulse) pulse.className = 'status-pulse error';
    if (statusText) statusText.textContent = 'API Offline';
    if (deviceText) deviceText.innerHTML = `<i class="bi bi-exclamation-triangle me-1"></i> Cannot connect`;

    if (topbarChip) topbarChip.className = 'model-badge-chip offline';
    if (topbarModelIcon) topbarModelIcon.className = 'bi bi-x-circle-fill text-danger me-1';
    if (topbarModelMobile) topbarModelMobile.textContent = 'Offline';

    console.error('System Health check failed:', err);

    if (isManual) {
      if (refreshIcon) refreshIcon.className = 'bi bi-x-circle me-1 text-danger';
      if (refreshText) refreshText.textContent = 'Offline';
      showAlert('Cannot connect to Flask backend. Make sure backend/app.py is running on port 5000.', 'danger');

      setTimeout(() => {
        if (refreshIcon) refreshIcon.className = 'bi bi-arrow-repeat me-1';
        if (refreshText) refreshText.textContent = 'Refresh';
      }, 1800);
    }
  } finally {
    if (refreshIcon) refreshIcon.classList.remove('spin-anim');
    if (refreshBtn) refreshBtn.disabled = false;
  }
}

function renderModelSpecs(info) {
  if (!info) return;
  document.getElementById('infoCheckpoint').textContent = info.checkpoint || 'best_dual_efficientnet_b0.pth';
  document.getElementById('infoResolution').textContent = info.input_resolution || '224 x 224 RGB';
  document.getElementById('infoParams').textContent = info.total_parameters ? info.total_parameters.toLocaleString() : '5,321,856';
  document.getElementById('infoHardware').textContent = (info.device_name ? `${info.device_name} (${info.device})` : info.device) || 'CPU';
}

function renderDiseaseGuide(diseaseMap) {
  const container = document.getElementById('diseaseGuideCards');
  if (!container || !diseaseMap) return;
  container.innerHTML = '';

  Object.entries(diseaseMap).forEach(([name, d]) => {
    const col = document.createElement('div');
    col.className = 'col-12 col-lg-6 d-flex flex-column';
    col.innerHTML = `
      <div class="guide-card h-100 d-flex flex-column">
        <div class="guide-header d-flex justify-content-between align-items-center">
          <div>
            <span class="badge mb-1" style="background-color: ${d.color_theme}; color: #fff;">${d.status}</span>
            <h5 class="fw-bold mb-0 text-dark">${name}</h5>
          </div>
          <span class="badge bg-light text-muted border">${d.severity}</span>
        </div>
        <div class="p-3 flex-grow-1 d-flex flex-column">
          <div class="mb-2">
            <span class="text-muted text-xs text-uppercase fw-bold">Etiology / Pathogen:</span>
            <div class="small fst-italic text-secondary">${d.pathogen}</div>
          </div>
          <div class="mb-2">
            <span class="text-muted text-xs text-uppercase fw-bold">Diagnostic Symptoms:</span>
            <div class="small text-dark">${d.symptoms}</div>
          </div>
          <div class="mt-auto pt-2">
            <span class="text-muted text-xs text-uppercase fw-bold">Agronomic Care & Control:</span>
            <div class="small text-success fw-semibold">${d.management}</div>
          </div>
        </div>
      </div>
    `;
    container.appendChild(col);
  });
}

// ==============================================================================
// 5. SECTION 1: FRONT-VIEW ANALYSIS (T1)
// ==============================================================================

function initFrontViewSection() {
  const fileInput = document.getElementById('frontFileInput');
  const dropzone = document.getElementById('frontDropzone');
  const frontPrompt = document.getElementById('frontPrompt');
  const frontPreviewBox = document.getElementById('frontPreviewBox');
  const frontImgPreview = document.getElementById('frontImgPreview');
  const frontFilenameText = document.getElementById('frontFilenameText');
  const frontFilesizeText = document.getElementById('frontFilesizeText');
  const removeBtn = document.getElementById('removeFrontBtn');
  const analyzeBtn = document.getElementById('analyzeFrontBtn');
  const loadSampleBtn = document.getElementById('loadFrontSampleBtn');
  const exportCsvBtn = document.getElementById('exportFrontCsvBtn');

  const frontErrorCard = document.getElementById('frontErrorCard');
  const frontErrorHeroCard = document.getElementById('frontErrorHeroCard');
  const frontErrorIconBox = document.getElementById('frontErrorIconBox');
  const frontErrorIcon = document.getElementById('frontErrorIcon');
  const frontErrorBadge = document.getElementById('frontErrorBadge');
  const frontErrorTitle = document.getElementById('frontErrorTitle');
  const frontErrorMessage = document.getElementById('frontErrorMessage');
  const frontErrorTrySampleBtn = document.getElementById('frontErrorTrySampleBtn');

  if (frontErrorTrySampleBtn && loadSampleBtn) {
    frontErrorTrySampleBtn.addEventListener('click', () => {
      loadSampleBtn.click();
    });
  }

  function renderFrontError(message, errorType = 'invalid_image') {
    const emptyState = document.getElementById('frontEmptyState');
    const resultCard = document.getElementById('frontResultCard');

    if (emptyState) emptyState.style.display = 'none';
    if (resultCard) resultCard.style.display = 'none';
    if (exportCsvBtn) exportCsvBtn.style.display = 'none';
    if (frontErrorCard) frontErrorCard.style.display = 'block';

    if (frontErrorMessage) frontErrorMessage.textContent = message;

    const isCorrupted = errorType === 'corrupted_image' || errorType === 'processing_error';
    if (frontErrorHeroCard) frontErrorHeroCard.className = isCorrupted ? 'error-hero-card corrupted mb-4' : 'error-hero-card mb-4';
    if (frontErrorIconBox) frontErrorIconBox.className = isCorrupted ? 'error-icon-box corrupted me-3' : 'error-icon-box me-3';
    if (frontErrorIcon) frontErrorIcon.className = isCorrupted ? 'bi bi-x-circle-fill' : 'bi bi-exclamation-triangle-fill';
    if (frontErrorBadge) {
      frontErrorBadge.className = isCorrupted ? 'badge bg-danger-subtle text-danger border border-danger-subtle mb-1' : 'badge bg-warning-subtle text-warning border border-warning-subtle mb-1';
      frontErrorBadge.textContent = isCorrupted ? 'Processing Failure' : 'Validation Alert';
    }
    if (frontErrorTitle) {
      frontErrorTitle.textContent = isCorrupted ? 'Corrupted or Unreadable Image' : 'Invalid Image Detected';
    }
    if (frontErrorMessage) {
      frontErrorMessage.className = isCorrupted ? 'error-message-banner corrupted mb-3' : 'error-message-banner mb-3';
    }
  }

  // Drag and drop events
  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      if (e.dataTransfer.files.length > 1) {
        showAlert('Front-View mode takes only 1 image. Loaded the first image.', 'info');
      }
      setFrontFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      setFrontFile(e.target.files[0]);
    }
  });

  removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFrontFile();
  });

  // Load Test Sample Button (if present)
  if (loadSampleBtn) {
    loadSampleBtn.addEventListener('click', async () => {
      try {
        const resp = await fetch('assets/samples/sample_01_t1.jpg');
        if (!resp.ok) throw new Error('Sample not found');
        const blob = await resp.blob();
        const testFile = new File([blob], 'sample_01_t1.jpg', { type: 'image/jpeg' });
        setFrontFile(testFile);
        showAlert('Loaded sample front-view leaf image: sample_01_t1.jpg', 'success');
      } catch (err) {
        showAlert('Could not load sample leaf asset. Upload your own image.', 'warning');
      }
    });
  }

  analyzeBtn.addEventListener('click', runFrontViewAnalysis);
  exportCsvBtn.addEventListener('click', exportFrontResultsCsv);

  function setFrontFile(file) {
    if (!file) return;

    if (!file.type || !file.type.startsWith('image/')) {
      const isExtImage = /\.(jpe?g|png|webp|bmp|tif?f)$/i.test(file.name || '');
      if (!isExtImage) {
        renderFrontError("Corrupted or invalid image: The uploaded file cannot be processed as an image. Please upload a valid image file (JPG, PNG, WEBP).", "corrupted_image");
        showAlert("Invalid file: Please upload a valid image file (JPG, PNG, WEBP).", "warning");
        return;
      }
    }

    if (file.size === 0) {
      renderFrontError("Invalid image: File is empty (0 bytes). Please upload a paddy leaf image for disease detection.", "invalid_image");
      showAlert("Invalid image: The uploaded file is empty (0 bytes).", "warning");
      return;
    }

    state.frontFile = file;
    frontPrompt.classList.add('d-none');
    frontPreviewBox.classList.remove('d-none');
    removeBtn.classList.remove('d-none');
    frontImgPreview.src = URL.createObjectURL(file);
    frontFilenameText.textContent = file.name;
    const kb = (file.size / 1024).toFixed(1);
    frontFilesizeText.textContent = `${kb} KB • Ready for analysis`;
    analyzeBtn.disabled = false;

    // Reset error card if previously shown
    if (frontErrorCard) frontErrorCard.style.display = 'none';
    const frontEmpty = document.getElementById('frontEmptyState');
    const frontResult = document.getElementById('frontResultCard');
    if (!state.frontResult && frontEmpty && frontResult && frontResult.style.display === 'none') {
      frontEmpty.style.display = 'block';
    }
  }

  function clearFrontFile() {
    state.frontFile = null;
    state.frontResult = null;
    frontPrompt.classList.remove('d-none');
    frontPreviewBox.classList.add('d-none');
    removeBtn.classList.add('d-none');
    frontFilenameText.textContent = '';
    frontFilesizeText.textContent = '';
    fileInput.value = '';
    analyzeBtn.disabled = true;

    // Reset results view to empty
    document.getElementById('frontEmptyState').style.display = 'block';
    document.getElementById('frontResultCard').style.display = 'none';
    if (frontErrorCard) frontErrorCard.style.display = 'none';
    exportCsvBtn.style.display = 'none';
  }

  async function runFrontViewAnalysis() {
    if (!state.frontFile) return;

    const spinner = document.getElementById('frontSpinner');
    const searchIcon = document.getElementById('frontSearchIcon');
    analyzeBtn.disabled = true;
    spinner.classList.remove('d-none');
    searchIcon.classList.add('d-none');

    const formData = new FormData();
    formData.append('image', state.frontFile);

    try {
      let res = await fetch(`${API_BASE_URL}/api/predict/front-view`, {
        method: 'POST',
        body: formData
      });

      // Handle transient Render free-tier spin-up or gateway timeout with automatic single retry
      if (!res.ok && (res.status === 502 || res.status === 503 || res.status === 504)) {
        await new Promise(r => setTimeout(r, 2000));
        res = await fetch(`${API_BASE_URL}/api/predict/front-view`, {
          method: 'POST',
          body: formData
        });
      }

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        const errorMsg = errJson.message || (res.status === 502 
          ? 'Cloud server is warming up or busy (HTTP 502). Please wait a moment and try again.' 
          : `Server returned error HTTP ${res.status}`);
        const errType = errJson.error_type || 'processing_error';
        renderFrontError(errorMsg, errType);
        showAlert(errorMsg, errType === 'corrupted_image' ? 'danger' : 'warning');
        return;
      }

      const data = await res.json();
      state.frontResult = data;
      renderFrontResult(data);
      showAlert(`Front-View Diagnosis: ${data.predicted_class} (${data.confidence}% accuracy)`, 'success');
    } catch (err) {
      console.error('Front view analysis error:', err);
      renderFrontError(err.message || 'An unexpected error occurred during communication with the server.', 'processing_error');
      showAlert(`Analysis failed: ${err.message}`, 'danger');
    } finally {
      analyzeBtn.disabled = false;
      spinner.classList.add('d-none');
      searchIcon.classList.remove('d-none');
    }
  }

  function renderFrontResult(data) {
    const emptyState = document.getElementById('frontEmptyState');
    const resultCard = document.getElementById('frontResultCard');

    if (frontErrorCard) frontErrorCard.style.display = 'none';
    emptyState.style.display = 'none';
    resultCard.style.display = 'block';
    exportCsvBtn.style.display = 'inline-block';

    const color = data.disease_info?.color_theme || '#059669';
    document.getElementById('frontSeverityBadge').textContent = data.disease_info?.severity || 'Severity: Normal';
    document.getElementById('frontSeverityBadge').style.backgroundColor = color;
    document.getElementById('frontSeverityBadge').style.color = '#fff';

    document.getElementById('frontDiseaseTitle').textContent = data.predicted_class;
    document.getElementById('frontDiseaseTitle').style.color = color;
    document.getElementById('frontPathogenText').textContent = data.disease_info?.pathogen || 'No pathogen';

    document.getElementById('frontConfidenceVal').textContent = `${data.confidence}%`;
    document.getElementById('frontConfidenceVal').style.color = color;

    if (state.frontFile) {
      document.getElementById('frontShowcaseImg').src = URL.createObjectURL(state.frontFile);
    }

    // Display ONLY the single highest-accuracy predicted class (other 3 classes are hidden as requested)
    document.getElementById('frontProbMeters').innerHTML = `
      <div class="prob-meter-row">
        <div class="prob-label-group">
          <span class="fw-bold text-dark">${data.predicted_class}</span>
          <span class="fw-bold" style="color: ${color}">${data.confidence}%</span>
        </div>
        <div class="prob-bar-track" style="height: 10px;">
          <div class="prob-bar-fill" style="width: ${data.confidence}%; background-color: ${color};"></div>
        </div>
      </div>
    `;

    // Agronomic text
    document.getElementById('frontSymptomsText').innerHTML = `<strong>Symptoms:</strong> ${data.disease_info?.symptoms || 'None'}`;
    document.getElementById('frontManagementText').innerHTML = `<i class="bi bi-shield-check me-1"></i><strong>Action Plan:</strong> ${data.disease_info?.management || 'Standard care'}`;
    document.getElementById('frontLatency').textContent = data.inference_time_ms;
  }

  function exportFrontResultsCsv() {
    if (!state.frontResult) return;
    const r = state.frontResult;
    const headers = ['Filename', 'Sample ID', 'Inference Mode', 'Predicted Disease (Highest Accuracy)', 'Confidence (%)', 'Latency (ms)', 'Status'];
    const rows = [[
      `"${r.filename || ''}"`,
      `"${r.sample_id || ''}"`,
      `"${r.inference_mode || 'Front-View (T1)'}"`,
      `"${r.predicted_class || 'N/A'}"`,
      r.confidence !== undefined ? r.confidence : '',
      r.inference_time_ms || '',
      `"${r.status || 'success'}"`
    ]];

    const csvContent = [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    downloadCsvBlob(csvContent, `paddyscan_${r.filename || 'front_view'}_result.csv`);
  }
}

// ==============================================================================
// 6. SECTION 2: DUAL-VIEW ANALYSIS (T1 + D1)
// ==============================================================================

function initDualViewSection() {
  const frontInput = document.getElementById('dualFrontInput');
  const backInput = document.getElementById('dualBackInput');
  const frontDropzone = document.getElementById('dualFrontDropzone');
  const backDropzone = document.getElementById('dualBackDropzone');

  const frontPrompt = document.getElementById('dualFrontPrompt');
  const frontPreviewBox = document.getElementById('dualFrontPreviewBox');
  const frontImg = document.getElementById('dualFrontImg');
  const frontFilename = document.getElementById('dualFrontFilename');
  const removeFrontBtn = document.getElementById('removeDualFrontBtn');

  const backPrompt = document.getElementById('dualBackPrompt');
  const backPreviewBox = document.getElementById('dualBackPreviewBox');
  const backImg = document.getElementById('dualBackImg');
  const backFilename = document.getElementById('dualBackFilename');
  const removeBackBtn = document.getElementById('removeDualBackBtn');

  const pairCheckDot = document.getElementById('pairCheckDot');
  const pairCheckText = document.getElementById('pairCheckText');
  const pairBadge = document.getElementById('pairBadge');
  const analyzeBtn = document.getElementById('analyzeDualBtn');
  const loadPairBtn = document.getElementById('loadDualSampleBtn');

  // Drag and drop for front
  setupDropEvents(frontDropzone, (file) => setDualFrontFile(file));
  frontInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) setDualFrontFile(e.target.files[0]);
  });
  removeFrontBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    state.dualFrontFile = null;
    updateDualState();
  });

  const dualErrorCard = document.getElementById('dualErrorCard');
  const dualErrorHeroCard = document.getElementById('dualErrorHeroCard');
  const dualErrorIconBox = document.getElementById('dualErrorIconBox');
  const dualErrorIcon = document.getElementById('dualErrorIcon');
  const dualErrorBadge = document.getElementById('dualErrorBadge');
  const dualErrorTitle = document.getElementById('dualErrorTitle');
  const dualErrorMessage = document.getElementById('dualErrorMessage');
  const dualErrorTrySampleBtn = document.getElementById('dualErrorTrySampleBtn');

  if (dualErrorTrySampleBtn && loadPairBtn) {
    dualErrorTrySampleBtn.addEventListener('click', () => {
      loadPairBtn.click();
    });
  }

  function renderDualError(message, errorType = 'invalid_image', failedView = null) {
    const emptyState = document.getElementById('dualEmptyState');
    const resultCard = document.getElementById('dualResultCard');

    if (emptyState) emptyState.style.display = 'none';
    if (resultCard) resultCard.style.display = 'none';
    if (dualErrorCard) dualErrorCard.style.display = 'block';

    if (dualErrorMessage) dualErrorMessage.textContent = message;

    const isCorrupted = errorType === 'corrupted_image' || errorType === 'processing_error';
    if (dualErrorHeroCard) dualErrorHeroCard.className = isCorrupted ? 'error-hero-card corrupted mb-4' : 'error-hero-card mb-4';
    if (dualErrorIconBox) dualErrorIconBox.className = isCorrupted ? 'error-icon-box corrupted me-3' : 'error-icon-box me-3';
    if (dualErrorIcon) dualErrorIcon.className = isCorrupted ? 'bi bi-x-circle-fill' : 'bi bi-exclamation-triangle-fill';
    if (dualErrorBadge) {
      dualErrorBadge.className = isCorrupted ? 'badge bg-danger-subtle text-danger border border-danger-subtle mb-1' : 'badge bg-warning-subtle text-warning border border-warning-subtle mb-1';
      dualErrorBadge.textContent = isCorrupted ? 'Processing Failure' : (failedView ? `${failedView.toUpperCase()} View Alert` : 'Validation Alert');
    }
    if (dualErrorTitle) {
      dualErrorTitle.textContent = isCorrupted ? 'Corrupted or Unreadable Image' : 'Invalid Image Detected';
    }
    if (dualErrorMessage) {
      dualErrorMessage.className = isCorrupted ? 'error-message-banner corrupted mb-3' : 'error-message-banner mb-3';
    }
  }

  // Drag and drop for back
  setupDropEvents(backDropzone, (file) => setDualBackFile(file));
  backInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) setDualBackFile(e.target.files[0]);
  });
  removeBackBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    state.dualBackFile = null;
    updateDualState();
  });

  // Load Test Pair Button (if present)
  if (loadPairBtn) {
    loadPairBtn.addEventListener('click', async () => {
      try {
        const [res1, res2] = await Promise.all([
          fetch('assets/samples/sample_01_t1.jpg'),
          fetch('assets/samples/sample_01_b1.jpg')
        ]);
        const blob1 = await res1.blob();
        const blob2 = await res2.blob();
        setDualFrontFile(new File([blob1], 'sample_01_t1.jpg', { type: 'image/jpeg' }));
        setDualBackFile(new File([blob2], 'sample_01_b1.jpg', { type: 'image/jpeg' }));
        showAlert('Loaded verified dual-view leaf pair (T1 + B1).', 'success');
      } catch (err) {
        showAlert('Could not load test pair assets. Please upload manually.', 'warning');
      }
    });
  }

  analyzeBtn.addEventListener('click', runDualViewAnalysis);

  function setDualFrontFile(file) {
    if (!file) return;

    if (!file.type || !file.type.startsWith('image/')) {
      const isExtImage = /\.(jpe?g|png|webp|bmp|tif?f)$/i.test(file.name || '');
      if (!isExtImage) {
        renderDualError("Corrupted or invalid image (Front View): The uploaded file cannot be processed as an image. Please upload a valid image file (JPG, PNG, WEBP).", "corrupted_image", "front");
        showAlert('Please upload a valid image file for Front View (T1).', 'warning');
        return;
      }
    }

    if (file.size === 0) {
      renderDualError("Invalid image (Front View): File is empty (0 bytes). Please upload a paddy leaf image for disease detection.", "invalid_image", "front");
      showAlert("The selected front-view file is empty (0 bytes).", "warning");
      return;
    }

    state.dualFrontFile = file;
    updateDualState();
  }

  function setDualBackFile(file) {
    if (!file) return;

    if (!file.type || !file.type.startsWith('image/')) {
      const isExtImage = /\.(jpe?g|png|webp|bmp|tif?f)$/i.test(file.name || '');
      if (!isExtImage) {
        renderDualError("Corrupted or invalid image (Back View): The uploaded file cannot be processed as an image. Please upload a valid image file (JPG, PNG, WEBP).", "corrupted_image", "back");
        showAlert('Please upload a valid image file for Back View (D1).', 'warning');
        return;
      }
    }

    if (file.size === 0) {
      renderDualError("Invalid image (Back View): File is empty (0 bytes). Please upload a paddy leaf image for disease detection.", "invalid_image", "back");
      showAlert("The selected back-view file is empty (0 bytes).", "warning");
      return;
    }

    state.dualBackFile = file;
    updateDualState();
  }

  function updateDualState() {
    if (dualErrorCard) dualErrorCard.style.display = 'none';

    // Front View UI
    if (state.dualFrontFile) {
      frontPrompt.classList.add('d-none');
      frontPreviewBox.classList.remove('d-none');
      frontImg.src = URL.createObjectURL(state.dualFrontFile);
      frontFilename.textContent = state.dualFrontFile.name;
      removeFrontBtn.classList.remove('d-none');
    } else {
      frontPrompt.classList.remove('d-none');
      frontPreviewBox.classList.add('d-none');
      frontFilename.textContent = '';
      removeFrontBtn.classList.add('d-none');
      frontInput.value = '';
    }

    // Back View UI
    if (state.dualBackFile) {
      backPrompt.classList.add('d-none');
      backPreviewBox.classList.remove('d-none');
      backImg.src = URL.createObjectURL(state.dualBackFile);
      backFilename.textContent = state.dualBackFile.name;
      removeBackBtn.classList.remove('d-none');
    } else {
      backPrompt.classList.remove('d-none');
      backPreviewBox.classList.add('d-none');
      backFilename.textContent = '';
      removeBackBtn.classList.add('d-none');
      backInput.value = '';
    }

    // Pair validation
    const readyCount = (state.dualFrontFile ? 1 : 0) + (state.dualBackFile ? 1 : 0);
    pairBadge.textContent = `${readyCount}/2 Ready`;

    if (readyCount === 2) {
      pairCheckDot.className = 'pair-dot ready';
      pairCheckText.textContent = 'Both views synchronized and ready for analysis';
      analyzeBtn.disabled = false;
    } else if (readyCount === 1) {
      pairCheckDot.className = 'pair-dot';
      pairCheckText.textContent = state.dualFrontFile ? 'Upload Back View (B1)' : 'Upload Front View (T1)';
      analyzeBtn.disabled = true;
    } else {
      pairCheckDot.className = 'pair-dot';
      pairCheckText.textContent = 'Upload both T1 and B1 images';
      analyzeBtn.disabled = true;
    }
  }

  async function runDualViewAnalysis() {
    if (!state.dualFrontFile || !state.dualBackFile) {
      showAlert('Both Front (T1) and Back (B1) views are required for Dual-View analysis.', 'warning');
      return;
    }

    const spinner = document.getElementById('dualSpinner');
    const searchIcon = document.getElementById('dualSearchIcon');
    analyzeBtn.disabled = true;
    spinner.classList.remove('d-none');
    searchIcon.classList.add('d-none');

    const formData = new FormData();
    formData.append('front_image', state.dualFrontFile);
    formData.append('back_image', state.dualBackFile);

    try {
      let res = await fetch(`${API_BASE_URL}/api/predict/dual-view`, {
        method: 'POST',
        body: formData
      });

      // Handle transient Render free-tier spin-up or gateway timeout with automatic single retry
      if (!res.ok && (res.status === 502 || res.status === 503 || res.status === 504)) {
        await new Promise(r => setTimeout(r, 2000));
        res = await fetch(`${API_BASE_URL}/api/predict/dual-view`, {
          method: 'POST',
          body: formData
        });
      }

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        const errorMsg = errJson.message || (res.status === 502 
          ? 'Cloud server is warming up or busy (HTTP 502). Please wait a moment and try again.' 
          : `Server returned error HTTP ${res.status}`);
        const errType = errJson.error_type || 'processing_error';
        renderDualError(errorMsg, errType, errJson.failed_view);
        showAlert(errorMsg, errType === 'corrupted_image' ? 'danger' : 'warning');
        return;
      }

      const data = await res.json();
      state.dualResult = data;
      renderDualResult(data);
      showAlert(`Dual-View Diagnosis completed: ${data.predicted_class} (${data.confidence}% confidence)`, 'success');
    } catch (err) {
      console.error('Dual view analysis error:', err);
      renderDualError(err.message || 'An unexpected error occurred during dual-view processing.', 'processing_error');
      showAlert(`Dual-View analysis failed: ${err.message}`, 'danger');
    } finally {
      analyzeBtn.disabled = false;
      spinner.classList.add('d-none');
      searchIcon.classList.remove('d-none');
    }
  }

  function renderDualResult(data) {
    const emptyState = document.getElementById('dualEmptyState');
    const resultCard = document.getElementById('dualResultCard');

    if (dualErrorCard) dualErrorCard.style.display = 'none';
    emptyState.style.display = 'none';
    resultCard.style.display = 'block';

    const color = data.disease_info?.color_theme || '#059669';
    document.getElementById('dualSeverityBadge').textContent = data.disease_info?.severity || 'Severity: Normal';
    document.getElementById('dualSeverityBadge').style.backgroundColor = color;
    document.getElementById('dualSeverityBadge').style.color = '#fff';

    document.getElementById('dualDiseaseTitle').textContent = data.predicted_class;
    document.getElementById('dualDiseaseTitle').style.color = color;
    document.getElementById('dualPathogenText').textContent = data.disease_info?.pathogen || 'No pathogen';

    document.getElementById('dualConfidenceVal').textContent = `${data.confidence}%`;
    document.getElementById('dualConfidenceVal').style.color = color;

    document.getElementById('dualShowcaseFrontImg').src = URL.createObjectURL(state.dualFrontFile);
    document.getElementById('dualShowcaseBackImg').src = URL.createObjectURL(state.dualBackFile);

    // Display ONLY the single highest-accuracy predicted class (other 3 classes are hidden as requested)
    document.getElementById('dualProbMeters').innerHTML = `
      <div class="prob-meter-row">
        <div class="prob-label-group">
          <span class="fw-bold text-dark">${data.predicted_class}</span>
          <span class="fw-bold" style="color: ${color}">${data.confidence}%</span>
        </div>
        <div class="prob-bar-track" style="height: 10px;">
          <div class="prob-bar-fill" style="width: ${data.confidence}%; background-color: ${color};"></div>
        </div>
      </div>
    `;

    // Agronomic text
    document.getElementById('dualSymptomsText').innerHTML = `<strong>Symptoms:</strong> ${data.disease_info?.symptoms || 'None'}`;
    document.getElementById('dualManagementText').innerHTML = `<i class="bi bi-shield-check me-1"></i><strong>Action Plan:</strong> ${data.disease_info?.management || 'Standard care'}`;
    document.getElementById('dualLatency').textContent = data.inference_time_ms;
  }
}

// ==============================================================================
// 7. SECTION 3: FOLDER-BASED BATCH ANALYSIS
// ==============================================================================

function initBatchSection() {
  const folderInput = document.getElementById('batchFolderInput');
  const filesInput = document.getElementById('batchFilesInput');
  const dropzone = document.getElementById('batchDropzone');
  const modeSelect = document.getElementById('batchModeSelect');
  const runBtn = document.getElementById('runBatchBtn');
  const loadBatchSampleBtn = document.getElementById('loadSampleBatchBtn');
  const searchInput = document.getElementById('tableSearchInput');
  const exportCsvBtn = document.getElementById('exportBatchCsvBtn');

  // Drag and drop
  setupDropEvents(dropzone, null, (files) => handleBatchFilesSelected(files));

  folderInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleBatchFilesSelected(Array.from(e.target.files));
    }
  });

  filesInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleBatchFilesSelected(Array.from(e.target.files));
    }
  });

  modeSelect.addEventListener('change', () => {
    const hint = document.getElementById('batchModeHint');
    if (!hint) return;
    if (modeSelect.value === 'dual') {
      hint.textContent = 'Strict Dual-View: Requires both T1 and B1 for every leaf; single views are marked unmatched.';
    } else {
      hint.textContent = 'Top-Only (T1): Evaluates each front-view image individually.';
    }
  });

  // Load Demo Batch Samples (if present)
  if (loadBatchSampleBtn) {
    loadBatchSampleBtn.addEventListener('click', async () => {
      try {
        const sampleNames = [
          'sample_01_t1.jpg',
          'sample_01_b1.jpg',
          'sample_02_t2.jpg',
          'sample_02_b2.jpg',
          'sample_03_t3.jpg',
          'sample_04_b4.jpg'
        ];
        const promises = sampleNames.map(async name => {
          const resp = await fetch(`assets/samples/${name}`);
          const blob = await resp.blob();
          return new File([blob], name, { type: 'image/jpeg' });
        });
        const files = await Promise.all(promises);
        handleBatchFilesSelected(files);
        showAlert(`Loaded ${files.length} demo batch images covering paired and single views.`, 'success');
      } catch (err) {
        showAlert('Could not load demo batch sample files.', 'warning');
      }
    });
  }

  runBtn.addEventListener('click', runBatchAnalysis);
  searchInput.addEventListener('input', () => filterAndRenderBatchTable());
  exportCsvBtn.addEventListener('click', exportBatchResultsCsv);

  // Filter pills
  document.querySelectorAll('#filterButtonGroup button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#filterButtonGroup button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeFilter = btn.getAttribute('data-filter');
      filterAndRenderBatchTable();
    });
  });

  function handleBatchFilesSelected(files) {
    const validImages = files.filter(f => {
      const isMimeImage = f.type && f.type.startsWith('image/');
      const isExtImage = /\.(jpe?g|png|webp|bmp|tif?f)$/i.test(f.name);
      return isMimeImage || isExtImage;
    });

    if (validImages.length === 0) {
      showAlert('No valid image files found in folder or file selection.', 'warning');
      return;
    }

    state.batchFiles = validImages;
    state.batchFilesMap = new Map();
    state.batchUrlCache = new Map();

    // Fast O(1) file lookup map
    validImages.forEach(f => {
      const name = (f.name || '').toLowerCase().trim();
      const rel = (f.webkitRelativePath || '').replace(/\\/g, '/').toLowerCase().trim();
      const base = name.split('/').pop().split('\\').pop();

      if (name) state.batchFilesMap.set(name, f);
      if (rel) {
        state.batchFilesMap.set(rel, f);
        state.batchFilesMap.set(rel.split('/').pop(), f);
      }
      if (base) state.batchFilesMap.set(base, f);
    });

    updateBatchStagingDisplay();
  }

  function updateBatchStagingDisplay() {
    const totalCount = state.batchFiles.length;
    document.getElementById('batchStagedTotal').textContent = `${totalCount} files`;

    // Count front and back
    let frontCount = 0;
    let backCount = 0;
    state.batchFiles.forEach(f => {
      const name = f.name.toLowerCase();
      if (/([_-]t\d*$|_t$|front|top|adaxial|^t\d*[_-])/.test(name)) frontCount++;
      else if (/([_-][bd]\d*$|_[bd]$|back|bottom|abaxial|^[bd]\d*[_-])/.test(name)) backCount++;
    });

    const stagedDetails = document.getElementById('batchStagedDetails');
    if (stagedDetails) {
      stagedDetails.textContent = `${frontCount} front (T1) • ${backCount} back (B1)`;
    }
    runBtn.disabled = totalCount === 0;

    // Smart auto-detection: If folder has front/top images and zero back images, switch to Top-Only mode
    if (frontCount > 0 && backCount === 0 && modeSelect.value === 'auto') {
      modeSelect.value = 'front';
      modeSelect.dispatchEvent(new Event('change'));
    }
  }

  // Fast client-side pre-downscale helper for huge camera/phone images
  async function prepareOptimizedImageBlob(file) {
    // If file is already compact (<= 200KB) or non-image, upload directly
    if (file.size <= 200 * 1024) return file;

    return new Promise((resolve) => {
      const img = new Image();
      const objUrl = URL.createObjectURL(file);
      img.onload = () => {
        URL.revokeObjectURL(objUrl);
        const maxDim = 512;
        let w = img.width;
        let h = img.height;
        if (w > maxDim || h > maxDim) {
          if (w >= h) {
            h = Math.round((h * maxDim) / w);
            w = maxDim;
          } else {
            w = Math.round((w * maxDim) / h);
            h = maxDim;
          }
        }
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, w, h);
        canvas.toBlob((blob) => {
          if (blob && blob.size < file.size) {
            const compressed = new File([blob], file.name, { type: 'image/jpeg' });
            if (file.webkitRelativePath) {
              try {
                Object.defineProperty(compressed, 'webkitRelativePath', {
                  value: file.webkitRelativePath
                });
              } catch (_) { }
            }
            resolve(compressed);
          } else {
            resolve(file);
          }
        }, 'image/jpeg', 0.85);
      };
      img.onerror = () => {
        URL.revokeObjectURL(objUrl);
        resolve(file);
      };
      img.src = objUrl;
    });
  }

  // Client-side specimen ID extraction helper
  function extractClientSampleId(filename) {
    if (!filename) return 'Sample';
    const stem = filename.split('/').pop().split('\\').pop().replace(/\.[^/.]+$/, "").toLowerCase();
    let m = stem.match(/[_-](?:t|b|d)(\d+)$/);
    if (m) return 'Leaf-' + m[1];
    m = stem.match(/(?:t|b|d)(\d+)$/);
    if (m) return 'Leaf-' + m[1];
    m = stem.match(/^(.*?)[_-](?:t1?|b1?|d1?|front|back|top|bottom)/);
    if (m && m[1]) return m[1].charAt(0).toUpperCase() + m[1].slice(1);
    const cleaned = stem.replace(/[_-]?(t1?|b1?|d1?|top|bottom|front|back|adaxial|abaxial)/g, '');
    return cleaned ? cleaned : stem;
  }

  // Client-side pairing grouping: keeps T1 and D1 views together, or treats each as unit in Top-Only mode
  function groupBatchFilesForProcessing(files, mode) {
    if (mode === 'front') {
      // In Top-Only mode, filter to only top/front images if any front tags exist
      const frontFiles = files.filter(f => {
        const name = (f.name || '').toLowerCase();
        const rel = (f.webkitRelativePath || '').toLowerCase();
        return (name.includes('_t') || name.includes('-t') || name.includes('front') || name.includes('top') ||
                rel.includes('_t') || rel.includes('-t') || rel.includes('front') || rel.includes('top'));
      });
      const targetFiles = frontFiles.length > 0 ? frontFiles : files.filter(f => {
        const name = (f.name || '').toLowerCase();
        return !name.includes('_b') && !name.includes('-b') && !name.includes('back') && !name.includes('bottom') && !name.includes('_d') && !name.includes('-d');
      });
      const finalFiles = targetFiles.length > 0 ? targetFiles : files;
      return finalFiles.map(f => [f]);
    }

    const groups = new Map();
    files.forEach(file => {
      const rel = file.webkitRelativePath || file.name;
      const sId = extractClientSampleId(rel) || file.name;
      if (!groups.has(sId)) {
        groups.set(sId, []);
      }
      groups.get(sId).push(file);
    });
    return Array.from(groups.values());
  }

  // Progressive streaming batch analysis
  async function runBatchAnalysis() {
    if (!state.batchFiles || state.batchFiles.length === 0) return;

    const progressWrapper = document.getElementById('batchProgressWrapper');
    const progressBar = document.getElementById('batchProgressBar');
    const progressText = document.getElementById('batchProgressText');
    const progressPct = document.getElementById('batchProgressPct');
    const spinner = document.getElementById('batchSpinner');
    const playIcon = document.getElementById('batchPlayIcon');

    runBtn.disabled = true;
    spinner.classList.remove('d-none');
    playIcon.classList.add('d-none');
    progressWrapper.style.display = 'block';
    progressBar.style.width = '5%';
    progressPct.textContent = '5%';
    progressText.textContent = `Preparing ${state.batchFiles.length} files for analysis...`;

    const mode = modeSelect.value;
    // Group files by specimen ID so pairs are never severed, or treat each file as unit for top-only
    const sampleUnits = groupBatchFilesForProcessing(state.batchFiles, mode);
    const totalSamples = sampleUnits.length;

    // Initialize or reset results state
    state.batchResults = {
      status: 'success',
      summary: {
        total_files: state.batchFiles.length,
        total_samples: 0,
        processed_count: 0,
        unmatched_count: 0,
        error_count: 0,
        healthy_count: 0,
        diseased_count: 0,
        average_confidence: 0,
        class_distribution: {
          'Brown spot': 0,
          'Healthy Paddy': 0,
          'Hispa': 0,
          'Leaf Blast': 0
        },
        mode: mode
      },
      results: []
    };

    // Show results section immediately
    const resultsWrapper = document.getElementById('batchResultsWrapper');
    resultsWrapper.style.display = 'block';
    renderBatchResults(state.batchResults);

    // Process in chunks of 12 sample units for real-time responsiveness
    const CHUNK_SIZE = 12;
    const totalChunks = Math.ceil(totalSamples / CHUNK_SIZE);

    try {
      let processedSamplesCount = 0;

      for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
        const chunkUnits = sampleUnits.slice(chunkIdx * CHUNK_SIZE, (chunkIdx + 1) * CHUNK_SIZE);
        const chunkFiles = chunkUnits.flat();

        const startSampleNum = processedSamplesCount + 1;
        const endSampleNum = processedSamplesCount + chunkUnits.length;
        const currentPct = Math.max(5, Math.round((processedSamplesCount / totalSamples) * 100));

        progressBar.style.width = `${currentPct}%`;
        progressPct.textContent = `${currentPct}%`;
        progressText.textContent = `Analyzing samples ${startSampleNum}–${endSampleNum} of ${totalSamples}...`;

        // Rapid downscale for oversized images (takes ~2ms each in browser canvas)
        const optimizedFiles = await Promise.all(chunkFiles.map(f => prepareOptimizedImageBlob(f)));

        const formData = new FormData();
        optimizedFiles.forEach(f => formData.append('files', f));
        formData.append('mode', mode);

        const res = await fetch(`${API_BASE_URL}/api/predict/batch`, {
          method: 'POST',
          body: formData
        });

        if (!res.ok) {
          const errJson = await res.json().catch(() => ({}));
          throw new Error(errJson.message || `Server returned HTTP ${res.status}`);
        }

        const data = await res.json();
        const chunkResults = data.results || [];
        const chunkSum = data.summary || {};

        // Merge chunk results into cumulative state
        state.batchResults.results.push(...chunkResults);
        state.batchResults.summary.total_samples += (chunkSum.total_samples || chunkResults.length);
        state.batchResults.summary.processed_count += (chunkSum.processed_count || 0);
        state.batchResults.summary.unmatched_count += (chunkSum.unmatched_count || 0);
        state.batchResults.summary.error_count += (chunkSum.error_count || 0);
        state.batchResults.summary.healthy_count += (chunkSum.healthy_count || 0);
        state.batchResults.summary.diseased_count += (chunkSum.diseased_count || 0);

        if (chunkSum.class_distribution) {
          Object.entries(chunkSum.class_distribution).forEach(([cls, count]) => {
            state.batchResults.summary.class_distribution[cls] = (state.batchResults.summary.class_distribution[cls] || 0) + count;
          });
        }

        // Recalculate cumulative average confidence
        const successes = state.batchResults.results.filter(r => r.status === 'success');
        if (successes.length > 0) {
          state.batchResults.summary.average_confidence = Math.round(
            (successes.reduce((acc, r) => acc + (r.confidence || 0), 0) / successes.length) * 100
          ) / 100;
        }

        processedSamplesCount += chunkUnits.length;
        const newPct = Math.round((processedSamplesCount / totalSamples) * 100);
        progressBar.style.width = `${newPct}%`;
        progressPct.textContent = `${newPct}%`;

        // Progressive real-time rendering
        renderBatchResults(state.batchResults);
      }

      progressBar.style.width = '100%';
      progressPct.textContent = '100%';
      progressText.textContent = `Completed! Analyzed ${state.batchResults.summary.total_samples} samples.`;
      showAlert(`Batch analysis completed! Analyzed ${state.batchResults.summary.total_samples} samples successfully.`, 'success');
    } catch (err) {
      console.error('Batch error:', err);
      showAlert(`Batch processing error: ${err.message}`, 'danger');
    } finally {
      runBtn.disabled = false;
      spinner.classList.add('d-none');
      playIcon.classList.remove('d-none');
      setTimeout(() => {
        progressWrapper.style.display = 'none';
      }, 2000);
    }
  }

  function renderBatchResults(data) {
    const resultsWrapper = document.getElementById('batchResultsWrapper');
    resultsWrapper.style.display = 'block';

    const sum = data.summary || {};
    document.getElementById('metricTotal').textContent = sum.total_samples || 0;
    document.getElementById('metricTotalFiles').textContent = `${sum.total_files || 0} images`;

    document.getElementById('metricHealthy').textContent = sum.healthy_count || 0;
    const hPct = sum.total_samples > 0 ? Math.round((sum.healthy_count / sum.total_samples) * 100) : 0;
    document.getElementById('metricHealthyPct').textContent = `${hPct}% of total`;

    document.getElementById('metricDiseased').textContent = sum.diseased_count || 0;
    const dPct = sum.total_samples > 0 ? Math.round((sum.diseased_count / sum.total_samples) * 100) : 0;
    document.getElementById('metricDiseasedPct').textContent = `${dPct}% of total`;

    document.getElementById('metricUnmatched').textContent = sum.unmatched_count || 0;
    document.getElementById('metricErrors').textContent = sum.error_count || 0;
    document.getElementById('metricAvgConfidence').textContent = `${sum.average_confidence || 0}%`;

    // Render Doughnut Chart using Chart.js
    renderBatchChart(sum.class_distribution, sum.unmatched_count);

    // Render Breakdown bars
    renderClassBreakdownBars(sum.class_distribution, sum.total_samples);

    // Render Table without image thumbnails
    filterAndRenderBatchTable();
  }

  function renderBatchChart(classDist, unmatchedCount) {
    const canvas = document.getElementById('diseaseChartCanvas');
    if (!canvas) return;

    if (state.batchChart) {
      state.batchChart.destroy();
    }

    const labels = ['Brown spot', 'Healthy Paddy', 'Hispa', 'Leaf Blast'];
    const colors = ['#d97706', '#16a34a', '#ca8a04', '#dc2626'];
    const counts = labels.map(c => (classDist && classDist[c]) || 0);

    if (unmatchedCount > 0) {
      labels.push('Unmatched / Skipped');
      colors.push('#94a3b8');
      counts.push(unmatchedCount);
    }

    state.batchChart = new Chart(canvas, {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: counts,
          backgroundColor: colors,
          borderWidth: 2,
          borderColor: '#ffffff'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right',
            labels: {
              boxWidth: 12,
              font: { family: "'Plus Jakarta Sans', sans-serif", size: 12 }
            }
          }
        },
        cutout: '65%'
      }
    });
  }

  function renderClassBreakdownBars(classDist, total) {
    const container = document.getElementById('batchClassBreakdown');
    if (!container || !classDist) return;
    container.innerHTML = '';

    const colors = {
      'Brown spot': '#d97706',
      'Healthy Paddy': '#16a34a',
      'Hispa': '#ca8a04',
      'Leaf Blast': '#dc2626'
    };

    Object.entries(classDist).forEach(([cls, count]) => {
      const pct = total > 0 ? Math.round((count / total) * 100) : 0;
      const barColor = colors[cls] || '#059669';
      const div = document.createElement('div');
      div.className = 'mb-3';
      div.innerHTML = `
        <div class="d-flex justify-content-between small fw-bold mb-1">
          <span>${cls}</span>
          <span style="color: ${barColor}">${count} leaves (${pct}%)</span>
        </div>
        <div class="progress" style="height: 8px;">
          <div class="progress-bar" style="width: ${pct}%; background-color: ${barColor};"></div>
        </div>
      `;
      container.appendChild(div);
    });
  }

  function filterAndRenderBatchTable() {
    if (!state.batchResults || !state.batchResults.results) return;
    const tbody = document.getElementById('batchTableBody');
    tbody.innerHTML = '';

    const searchTerm = (document.getElementById('tableSearchInput')?.value || '').toLowerCase().trim();
    const filter = state.activeFilter;

    const filtered = state.batchResults.results.filter(row => {
      // Filter matching
      if (filter === 'unmatched' && row.status !== 'unmatched') return false;
      if (filter === 'errors' && row.status !== 'error') return false;
      if (filter !== 'all' && filter !== 'unmatched' && filter !== 'errors') {
        if (row.predicted_class !== filter) return false;
      }

      // Search matching
      if (searchTerm) {
        const idMatch = (row.sample_id || '').toLowerCase().includes(searchTerm);
        const frontMatch = (row.front_filename || '').toLowerCase().includes(searchTerm);
        const backMatch = (row.back_filename || '').toLowerCase().includes(searchTerm);
        const classMatch = (row.predicted_class || '').toLowerCase().includes(searchTerm);
        if (!idMatch && !frontMatch && !backMatch && !classMatch) return false;
      }

      return true;
    });

    state.currentFilteredBatch = filtered;

    if (filtered.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="7" class="text-center py-4 text-muted">
            <i class="bi bi-inbox fs-3 d-block mb-1"></i> No matching batch samples found.
          </td>
        </tr>
      `;
      return;
    }

    filtered.forEach((row, i) => {
      const tr = document.createElement('tr');
      const isSuccess = row.status === 'success';
      const isUnmatched = row.status === 'unmatched';

      const isTopOnly = !row.back_filename ||
                        row.back_filename === 'None' ||
                        row.back_filename === 'N/A' ||
                        row.back_filename === 'null' ||
                        row.back_filename === 'Missing Back View (D1)' ||
                        (row.mode && (row.mode.toLowerCase().includes('top') || row.mode.toLowerCase().includes('front')));

      let statusBadge = '<span class="badge bg-success-subtle text-success border border-success-subtle">Analyzed</span>';
      if (isUnmatched) {
        statusBadge = '<span class="badge bg-warning-subtle text-warning border border-warning-subtle">Unmatched</span>';
      } else if (row.status === 'error') {
        statusBadge = '<span class="badge bg-danger-subtle text-danger border border-danger-subtle">Error</span>';
      }

      const diseaseBadge = isSuccess
        ? `<span class="badge" style="background-color: ${row.disease_info?.color_theme || '#059669'}; color: white;">${row.predicted_class}</span>`
        : (isUnmatched ? `<span class="text-muted small">${row.reason || 'Missing View'}</span>` : `<span class="text-danger small">${row.error || 'Failed'}</span>`);

      const confText = isSuccess ? `<strong>${row.confidence}%</strong>` : '&mdash;';

      // Render Table Row: Clean display without image thumbnails (images are viewed in Details modal)
      tr.innerHTML = `
        <td class="fw-bold text-dark">${row.sample_id}</td>
        <td>
          ${isTopOnly ? `
            <div class="small fw-semibold text-truncate text-dark" style="max-width: 320px;" title="${row.front_filename}">
              <span class="badge bg-success-subtle text-success border border-success-subtle me-1">T1</span>${row.front_filename}
            </div>
            <div class="text-xs text-muted">Top-View Leaf Image</div>
          ` : `
            <div class="small text-truncate text-dark mb-1" style="max-width: 320px;" title="${row.front_filename}">
              <span class="badge bg-success-subtle text-success border border-success-subtle me-1">T1</span>${row.front_filename}
            </div>
            <div class="small text-muted text-truncate" style="max-width: 320px;" title="${row.back_filename}">
              <span class="badge bg-secondary-subtle text-secondary border border-secondary-subtle me-1">D1</span>${row.back_filename}
            </div>
          `}
        </td>
        <td>
          ${isTopOnly ? `
            <span class="badge bg-success-subtle text-success border border-success-subtle">Top-Only (T1)</span>
          ` : `
            <span class="badge bg-light text-dark border">${row.mode || 'Dual-View'}</span>
          `}
        </td>
        <td>${diseaseBadge}</td>
        <td>${confText}</td>
        <td>${statusBadge}</td>
        <td class="text-end">
          ${isSuccess ? `
            <button class="btn btn-outline-success btn-xs rounded-pill px-3 batch-detail-btn" data-filtered-idx="${i}">
              <i class="bi bi-info-circle me-1"></i> Details
            </button>
          ` : '&mdash;'}
        </td>
      `;
      tbody.appendChild(tr);
    });

    // Attach event listeners to details buttons
    tbody.querySelectorAll('.batch-detail-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.getAttribute('data-filtered-idx'), 10);
        const selectedRow = state.currentFilteredBatch && state.currentFilteredBatch[idx];
        if (selectedRow) {
          showBatchDetailModal(selectedRow);
        }
      });
    });
  }

  function exportBatchResultsCsv() {
    if (!state.batchResults || !state.batchResults.results) return;
    const headers = ['Sample ID', 'Front Filename (T1)', 'Back Filename (D1)', 'Analysis Mode', 'Predicted Disease', 'Confidence (%)', 'Status', 'Details/Reason'];
    const rows = state.batchResults.results.map(r => [
      `"${r.sample_id || ''}"`,
      `"${r.front_filename || ''}"`,
      `"${r.back_filename || ''}"`,
      `"${r.mode || ''}"`,
      `"${r.predicted_class || ''}"`,
      r.confidence !== undefined ? r.confidence : '',
      `"${r.status || ''}"`,
      `"${r.reason || r.error || ''}"`
    ]);

    const csvContent = [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    downloadCsvBlob(csvContent, 'paddyscan_batch_results.csv');
  }
}

// Helper to find matching File object from staged batch files
function findBatchFile(filename) {
  if (!filename) return null;
  const cleanTarget = filename.replace(/\\/g, '/').toLowerCase().trim();
  const targetBase = cleanTarget.split('/').pop();

  if (state.batchFilesMap) {
    if (state.batchFilesMap.has(cleanTarget)) return state.batchFilesMap.get(cleanTarget);
    if (state.batchFilesMap.has(targetBase)) return state.batchFilesMap.get(targetBase);
  }

  if (!state.batchFiles || state.batchFiles.length === 0) return null;

  return state.batchFiles.find(f => {
    const fRel = (f.webkitRelativePath || '').replace(/\\/g, '/').toLowerCase().trim();
    const fName = (f.name || '').toLowerCase().trim();

    if (fRel && (fRel === cleanTarget || fRel.endsWith('/' + cleanTarget) || cleanTarget.endsWith('/' + fRel))) return true;
    if (fName === targetBase || fName === cleanTarget) return true;
    return false;
  }) || null;
}

// Helper to get image URL for a batch file
function getBatchImageUrl(filename) {
  if (!filename) return null;
  const cleanTarget = filename.replace(/\\/g, '/').toLowerCase().trim();
  const targetBase = cleanTarget.split('/').pop();

  state.batchUrlCache = state.batchUrlCache || new Map();
  if (state.batchUrlCache.has(cleanTarget)) return state.batchUrlCache.get(cleanTarget);
  if (state.batchUrlCache.has(targetBase)) return state.batchUrlCache.get(targetBase);

  const file = findBatchFile(filename);
  if (file) {
    try {
      const url = URL.createObjectURL(file);
      state.batchUrlCache.set(cleanTarget, url);
      state.batchUrlCache.set(targetBase, url);
      return url;
    } catch (e) {
      console.warn('Could not create object URL for file:', filename, e);
    }
  }

  if (cleanTarget.includes('sample_')) {
    const url = `assets/samples/${targetBase}`;
    state.batchUrlCache.set(cleanTarget, url);
    state.batchUrlCache.set(targetBase, url);
    return url;
  }
  return null;
}

// Global modal viewer function
window.showBatchDetailModal = function (row) {
  const modalEl = document.getElementById('detailModal');
  const modalBody = document.getElementById('detailModalBody');
  const modal = new bootstrap.Modal(modalEl);

  const color = row.disease_info?.color_theme || '#059669';

  const isTopOnly = !row.back_filename ||
                    row.back_filename === 'None' ||
                    row.back_filename === 'N/A' ||
                    row.back_filename === 'null' ||
                    row.back_filename === 'Missing Back View (D1)' ||
                    (row.mode && (row.mode.toLowerCase().includes('top') || row.mode.toLowerCase().includes('front')));

  // Find particular images for this sample
  const frontImgUrl = getBatchImageUrl(row.front_filename);
  const backImgUrl = isTopOnly ? null : getBatchImageUrl(row.back_filename);

  let imageShowcaseHtml = '';
  if (isTopOnly) {
    // Show ONLY the Top Image preview card
    imageShowcaseHtml = `
      <div class="card p-3 bg-light border-0 text-center mb-3">
        <div class="d-flex justify-content-between align-items-center mb-2 px-1">
          <span class="badge bg-success-subtle text-success border border-success-subtle">
            T1 • Top View (Adaxial)
          </span>
          <span class="text-xs text-muted">Single Top Leaf Image Analysis</span>
        </div>
        <div style="height: 240px; display: flex; align-items: center; justify-content: center; background: #fff; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0;">
          ${frontImgUrl ? `
            <img src="${frontImgUrl}" alt="Top Leaf (T1)" class="img-fluid" style="max-height: 100%; max-width: 100%; object-fit: contain;">
          ` : `
            <div class="text-muted small">
              <i class="bi bi-file-earmark-image fs-1 d-block mb-1 text-success"></i>
              ${row.front_filename || row.sample_id}
            </div>
          `}
        </div>
        <div class="text-xs text-muted text-truncate mt-2 text-start px-1" title="${row.front_filename}">
          <i class="bi bi-file-earmark-image me-1 text-success"></i><strong>Top Image File:</strong> ${row.front_filename}
        </div>
      </div>
    `;
  } else if (frontImgUrl || backImgUrl) {
    imageShowcaseHtml = `
      <div class="row g-3 mb-3">
        ${frontImgUrl ? `
          <div class="${backImgUrl ? 'col-sm-6' : 'col-12'}">
            <div class="card p-2 bg-light border-0 text-center">
              <span class="badge bg-success-subtle text-success border border-success-subtle mb-2 align-self-start">
                T1 • Front View
              </span>
              <div style="height: 200px; display: flex; align-items: center; justify-content: center; background: #fff; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0;">
                <img src="${frontImgUrl}" alt="Front Leaf (T1)" class="img-fluid" style="max-height: 100%; max-width: 100%; object-fit: contain;">
              </div>
              <div class="text-xs text-muted text-truncate mt-1 text-start" title="${row.front_filename}">
                ${row.front_filename}
              </div>
            </div>
          </div>
        ` : ''}

        ${backImgUrl ? `
          <div class="${frontImgUrl ? 'col-sm-6' : 'col-12'}">
            <div class="card p-2 bg-light border-0 text-center">
              <span class="badge bg-secondary-subtle text-secondary border border-secondary-subtle mb-2 align-self-start">
                D1 • Back View
              </span>
              <div style="height: 200px; display: flex; align-items: center; justify-content: center; background: #fff; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0;">
                <img src="${backImgUrl}" alt="Back Leaf (D1)" class="img-fluid" style="max-height: 100%; max-width: 100%; object-fit: contain;">
              </div>
              <div class="text-xs text-muted text-truncate mt-1 text-start" title="${row.back_filename}">
                ${row.back_filename}
              </div>
            </div>
          </div>
        ` : ''}
      </div>
    `;
  } else {
    imageShowcaseHtml = `
      <div class="alert alert-light border py-3 text-center mb-3 text-muted small">
        <i class="bi bi-file-earmark-image fs-3 d-block mb-1 text-secondary"></i>
        Sample file: <strong>${row.front_filename || row.sample_id}</strong>
      </div>
    `;
  }

  modalBody.innerHTML = `
    <!-- Top Diagnosis Header with Single Maximum Disease and Highest Accuracy Score -->
    <div class="d-flex justify-content-between align-items-center mb-3">
      <div>
        <span class="badge px-3 py-1 text-uppercase fw-bold" style="background-color: ${color}; color: #fff;">${row.predicted_class}</span>
        <h4 class="fw-bold mt-2 mb-0">${row.sample_id}</h4>
      </div>
      <div class="text-end">
        <h3 class="fw-bold mb-0" style="color: ${color}">${row.confidence}%</h3>
        <span class="text-muted text-xs">Highest Confidence</span>
      </div>
    </div>

    <!-- Leaf Images for this Particular Sample (Only Top Image for Top-Only) -->
    ${imageShowcaseHtml}

    <!-- Sample & Processing Metadata -->
    <div class="card border-0 bg-light p-3 mb-3 rounded-3">
      <div class="small mb-1"><strong>${isTopOnly ? 'Top Leaf File (T1)' : 'Front File (T1)'}:</strong> ${row.front_filename || 'N/A'}</div>
      ${!isTopOnly && row.back_filename ? `<div class="small mb-1"><strong>Back File (D1):</strong> ${row.back_filename}</div>` : ''}
      <div class="small"><strong>Mode:</strong> ${row.mode} • <strong>Latency:</strong> ${row.inference_time_ms} ms</div>
    </div>

    <!-- Diagnosed Disease Confidence Score (ONLY maximum disease, other 3 omitted) -->
    <div class="card border-0 bg-light p-3 mb-3 rounded-3">
      <h6 class="fw-bold small text-dark mb-2">Diagnosed Disease Confidence Score</h6>
      <div class="prob-meter-row">
        <div class="prob-label-group d-flex justify-content-between mb-1">
          <span class="fw-bold text-dark">${row.predicted_class}</span>
          <span class="fw-bold" style="color: ${color}">${row.confidence}%</span>
        </div>
        <div class="prob-bar-track" style="height: 10px; background: #e2e8f0; border-radius: 9999px; overflow: hidden;">
          <div class="prob-bar-fill" style="width: ${row.confidence}%; background-color: ${color}; height: 100%; border-radius: 9999px;"></div>
        </div>
      </div>
    </div>

    <!-- Agronomic Recommendations -->
    <div class="p-3 bg-white rounded-3 border">
      <h6 class="fw-bold text-success mb-1">Agronomic Recommendations:</h6>
      <p class="small text-muted mb-2">${row.disease_info?.symptoms || ''}</p>
      <div class="small text-dark fw-semibold">${row.disease_info?.management || ''}</div>
    </div>
  `;

  modal.show();
};

// ==============================================================================
// 8. COMMON UI HELPERS & UTILITIES
// ==============================================================================

function renderProbabilityMeters(probs, accentColor) {
  if (!probs) return '';
  return Object.entries(probs).map(([cls, pct]) => `
    <div class="prob-meter-row">
      <div class="prob-label-group">
        <span>${cls}</span>
        <span class="text-muted">${pct}%</span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width: ${pct}%; background-color: ${pct > 40 ? accentColor : '#94a3b8'};"></div>
      </div>
    </div>
  `).join('');
}

// ==============================================================================
// 7.5 TRAINING & EVALUATION CURVES (ANIMATED DUAL-VIEW EFFICIENTNET-B0 + cGAN)
// ==============================================================================

const TRAINING_CURVE_DATA = {
  epochs: Array.from({ length: 50 }, (_, i) => i + 1),
  trainAcc: [
    53.87, 77.80, 83.40, 87.40, 88.00, 88.90, 90.00, 90.60, 91.10, 92.80,
    91.70, 92.50, 92.80, 93.40, 92.30, 93.70, 93.90, 94.40, 94.10, 94.40,
    94.30, 94.20, 94.10, 93.90, 93.50, 93.90, 95.00, 94.70, 94.80, 94.10,
    94.00, 95.00, 93.30, 95.30, 94.10, 94.80, 94.20, 94.80, 93.70, 94.10,
    94.80, 94.10, 94.50, 95.00, 93.80, 94.00, 93.80, 94.40, 93.90, 94.00
  ],
  valAcc: [
    78.40, 83.80, 86.20, 87.20, 89.70, 90.30, 92.20, 91.80, 90.80, 93.80,
    92.70, 92.70, 92.70, 91.70, 91.70, 93.10, 92.70, 93.60, 92.70, 93.60,
    92.70, 93.60, 93.60, 93.20, 92.70, 92.70, 93.60, 92.70, 94.10, 93.20,
    92.70, 93.20, 93.20, 93.10, 92.70, 92.70, 92.20, 92.70, 92.70, 94.10,
    92.70, 92.70, 93.10, 92.70, 92.70, 92.70, 92.70, 92.70, 92.70, 93.20
  ],
  trainLoss: [
    1.080, 0.608, 0.481, 0.383, 0.336, 0.321, 0.297, 0.264, 0.260, 0.220,
    0.234, 0.216, 0.213, 0.185, 0.206, 0.182, 0.161, 0.149, 0.144, 0.150,
    0.168, 0.164, 0.165, 0.161, 0.173, 0.167, 0.130, 0.133, 0.154, 0.166,
    0.155, 0.165, 0.171, 0.139, 0.151, 0.153, 0.144, 0.131, 0.160, 0.147,
    0.141, 0.168, 0.139, 0.133, 0.153, 0.142, 0.145, 0.154, 0.168, 0.147
  ],
  valLoss: [
    0.651, 0.452, 0.358, 0.362, 0.269, 0.266, 0.204, 0.213, 0.215, 0.169,
    0.171, 0.175, 0.193, 0.200, 0.181, 0.215, 0.179, 0.202, 0.157, 0.170,
    0.154, 0.175, 0.186, 0.193, 0.149, 0.154, 0.147, 0.192, 0.174, 0.148,
    0.173, 0.144, 0.197, 0.170, 0.167, 0.169, 0.175, 0.150, 0.185, 0.155,
    0.160, 0.156, 0.147, 0.159, 0.159, 0.158, 0.158, 0.163, 0.153, 0.133
  ]
};

let accuracyChartInstance = null;
let lossChartInstance = null;
let curveDrawingTimer = null;

function renderTrainingCurves(animate = true) {
  if (curveDrawingTimer) {
    clearInterval(curveDrawingTimer);
    curveDrawingTimer = null;
  }

  const accCanvas = document.getElementById('accuracyChartCanvas');
  const lossCanvas = document.getElementById('lossChartCanvas');
  if (!accCanvas || !lossCanvas || typeof Chart === 'undefined') return;

  if (accuracyChartInstance) {
    accuracyChartInstance.destroy();
    accuracyChartInstance = null;
  }
  if (lossChartInstance) {
    lossChartInstance.destroy();
    lossChartInstance = null;
  }

  const statusText = document.getElementById('curveDrawStatusText');
  const spinner = document.getElementById('curveDrawSpinner');
  const counter = document.getElementById('curveEpochCounter');

  const accCtx = accCanvas.getContext('2d');
  const lossCtx = lossCanvas.getContext('2d');

  const initialLabels = animate ? [1] : TRAINING_CURVE_DATA.epochs;
  const initialTrainAcc = animate ? [TRAINING_CURVE_DATA.trainAcc[0]] : TRAINING_CURVE_DATA.trainAcc;
  const initialValAcc = animate ? [TRAINING_CURVE_DATA.valAcc[0]] : TRAINING_CURVE_DATA.valAcc;
  const initialTrainLoss = animate ? [TRAINING_CURVE_DATA.trainLoss[0]] : TRAINING_CURVE_DATA.trainLoss;
  const initialValLoss = animate ? [TRAINING_CURVE_DATA.valLoss[0]] : TRAINING_CURVE_DATA.valLoss;

  // 1. Initialize Accuracy Chart with fixed limits to prevent layout jumping
  accuracyChartInstance = new Chart(accCtx, {
    type: 'line',
    data: {
      labels: [...initialLabels],
      datasets: [
        {
          label: 'Train Accuracy',
          data: [...initialTrainAcc],
          borderColor: '#1f77b4',
          backgroundColor: '#1f77b4',
          borderWidth: 2.2,
          pointStyle: 'circle',
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: '#1f77b4',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 1.5,
          fill: false,
          tension: 0.15
        },
        {
          label: 'Validation Accuracy',
          data: [...initialValAcc],
          borderColor: '#ff7f0e',
          backgroundColor: '#ff7f0e',
          borderWidth: 2.2,
          pointStyle: 'rect',
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: '#ff7f0e',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 1.5,
          fill: false,
          tension: 0.15
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          display: true,
          position: 'bottom',
          labels: {
            usePointStyle: true,
            boxWidth: 8,
            font: { family: "'Plus Jakarta Sans', sans-serif", size: 12, weight: '600' }
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.92)',
          titleFont: { size: 13, weight: '700' },
          bodyFont: { size: 12 },
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            title: (items) => `Epoch ${items[0].label}`,
            label: (item) => ` ${item.dataset.label}: ${Number(item.raw).toFixed(2)}%`
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Epoch',
            color: '#475569',
            font: { family: "'Plus Jakarta Sans', sans-serif", weight: '600', size: 12 }
          },
          grid: {
            color: 'rgba(226, 232, 240, 0.7)',
            borderDash: [4, 4]
          },
          min: 1,
          max: 50,
          ticks: {
            stepSize: 5,
            color: '#64748b'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Accuracy (%)',
            color: '#475569',
            font: { family: "'Plus Jakarta Sans', sans-serif", weight: '600', size: 12 }
          },
          min: 50,
          max: 100,
          ticks: {
            stepSize: 10,
            color: '#64748b',
            callback: (v) => `${v}%`
          },
          grid: {
            color: 'rgba(226, 232, 240, 0.7)',
            borderDash: [4, 4]
          }
        }
      }
    }
  });

  // 2. Initialize Loss Chart
  lossChartInstance = new Chart(lossCtx, {
    type: 'line',
    data: {
      labels: [...initialLabels],
      datasets: [
        {
          label: 'Train Loss',
          data: [...initialTrainLoss],
          borderColor: '#d62728',
          backgroundColor: '#d62728',
          borderWidth: 2.2,
          pointStyle: 'circle',
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: '#d62728',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 1.5,
          fill: false,
          tension: 0.15
        },
        {
          label: 'Validation Loss',
          data: [...initialValLoss],
          borderColor: '#2ca02c',
          backgroundColor: '#2ca02c',
          borderWidth: 2.2,
          pointStyle: 'rect',
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: '#2ca02c',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 1.5,
          fill: false,
          tension: 0.15
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          align: 'end',
          labels: {
            usePointStyle: true,
            boxWidth: 8,
            font: { family: "'Plus Jakarta Sans', sans-serif", size: 12, weight: '600' }
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.92)',
          titleFont: { size: 13, weight: '700' },
          bodyFont: { size: 12 },
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            title: (items) => `Epoch ${items[0].label}`,
            label: (item) => ` ${item.dataset.label}: ${Number(item.raw).toFixed(3)}`
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Epoch',
            color: '#475569',
            font: { family: "'Plus Jakarta Sans', sans-serif", weight: '600', size: 12 }
          },
          grid: {
            color: 'rgba(226, 232, 240, 0.7)',
            borderDash: [4, 4]
          },
          min: 1,
          max: 50,
          ticks: {
            stepSize: 5,
            color: '#64748b'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Cross-Entropy Loss',
            color: '#475569',
            font: { family: "'Plus Jakarta Sans', sans-serif", weight: '600', size: 12 }
          },
          min: 0.0,
          max: 1.15,
          ticks: {
            stepSize: 0.2,
            color: '#64748b'
          },
          grid: {
            color: 'rgba(226, 232, 240, 0.7)',
            borderDash: [4, 4]
          }
        }
      }
    }
  });

  if (!animate) {
    if (spinner) spinner.style.display = 'none';
    if (statusText) statusText.innerHTML = '<i class="bi bi-check-circle-fill text-success me-1"></i> <strong>Graph rendering complete!</strong> Hover over any point to inspect metrics';
    if (counter) counter.textContent = 'Epoch: 50 / 50 (Completed)';
    return;
  }

  // 3. Progressive real-time line drafting animation
  if (spinner) spinner.style.display = 'inline-block';
  if (statusText) statusText.innerHTML = '<span class="text-success fw-bold">Live Graph Synthesis:</span> Designing and drawing convergence curves across 50 epochs...';
  if (counter) counter.textContent = 'Epoch: 1 / 50';

  let currentIdx = 1;
  const totalPoints = 50;
  const stepInterval = 28; // ~1.4 seconds total progressive draw

  curveDrawingTimer = setInterval(() => {
    if (currentIdx < totalPoints) {
      const ep = TRAINING_CURVE_DATA.epochs[currentIdx];
      
      // Update Accuracy Chart
      accuracyChartInstance.data.labels.push(ep);
      accuracyChartInstance.data.datasets[0].data.push(TRAINING_CURVE_DATA.trainAcc[currentIdx]);
      accuracyChartInstance.data.datasets[1].data.push(TRAINING_CURVE_DATA.valAcc[currentIdx]);
      accuracyChartInstance.update('none');

      // Update Loss Chart
      lossChartInstance.data.labels.push(ep);
      lossChartInstance.data.datasets[0].data.push(TRAINING_CURVE_DATA.trainLoss[currentIdx]);
      lossChartInstance.data.datasets[1].data.push(TRAINING_CURVE_DATA.valLoss[currentIdx]);
      lossChartInstance.update('none');

      currentIdx++;
      if (counter) counter.textContent = `Epoch: ${currentIdx} / 50`;
    } else {
      clearInterval(curveDrawingTimer);
      curveDrawingTimer = null;
      if (spinner) spinner.style.display = 'none';
      if (statusText) statusText.innerHTML = '<i class="bi bi-check-circle-fill text-success me-1"></i> <strong>Graph rendering complete!</strong> Hover over any point to inspect metrics';
      if (counter) counter.textContent = 'Epoch: 50 / 50 (Completed)';
    }
  }, stepInterval);
}

function initTrainingCurvesSection() {
  const replayBtn = document.getElementById('replayGraphAnimBtn');
  const skipBtn = document.getElementById('btnSkipCurveAnim');
  const btnInteractive = document.getElementById('btnViewInteractive');
  const btnMatplotlib = document.getElementById('btnViewMatplotlib');
  const interactiveView = document.getElementById('interactiveCurvesView');
  const matplotlibView = document.getElementById('matplotlibPlotsView');
  const statusBar = document.getElementById('curveDrawStatusBar');

  if (replayBtn) {
    replayBtn.addEventListener('click', () => {
      // Return to interactive view if currently in publication plot mode
      if (btnInteractive && !btnInteractive.classList.contains('active')) {
        btnInteractive.click();
      }
      renderTrainingCurves(true);
    });
  }

  if (skipBtn) {
    skipBtn.addEventListener('click', () => {
      renderTrainingCurves(false);
    });
  }

  if (btnInteractive && btnMatplotlib) {
    btnInteractive.addEventListener('click', () => {
      btnInteractive.classList.add('btn-success', 'active');
      btnInteractive.classList.remove('btn-light', 'text-secondary');
      btnMatplotlib.classList.remove('btn-success', 'active');
      btnMatplotlib.classList.add('btn-light', 'text-secondary');

      interactiveView?.classList.remove('d-none');
      matplotlibView?.classList.add('d-none');
      statusBar?.classList.remove('d-none');
      if (!accuracyChartInstance || !lossChartInstance) {
        renderTrainingCurves(true);
      }
    });

    btnMatplotlib.addEventListener('click', () => {
      btnMatplotlib.classList.add('btn-success', 'active');
      btnMatplotlib.classList.remove('btn-light', 'text-secondary');
      btnInteractive.classList.remove('btn-success', 'active');
      btnInteractive.classList.add('btn-light', 'text-secondary');

      matplotlibView?.classList.remove('d-none');
      interactiveView?.classList.add('d-none');
      statusBar?.classList.add('d-none');
    });
  }
}

function setupDropEvents(element, singleFileCallback, multiFileCallback) {
  if (!element) return;
  ['dragenter', 'dragover'].forEach(n => {
    element.addEventListener(n, (e) => {
      e.preventDefault();
      e.stopPropagation();
      element.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach(n => {
    element.addEventListener(n, (e) => {
      e.preventDefault();
      e.stopPropagation();
      element.classList.remove('dragover');
    });
  });
  element.addEventListener('drop', async (e) => {
    e.preventDefault();
    e.stopPropagation();

    // Check for webkitGetAsEntry to support dragging and dropping entire folders
    if (multiFileCallback && e.dataTransfer.items && e.dataTransfer.items.length > 0 && e.dataTransfer.items[0].webkitGetAsEntry) {
      const files = [];
      const traverseEntry = async (entry, path = '') => {
        if (entry.isFile) {
          await new Promise((res) => {
            entry.file(f => {
              try {
                Object.defineProperty(f, 'webkitRelativePath', {
                  value: (path ? path + '/' : '') + f.name,
                  writable: true
                });
              } catch (_) {}
              files.push(f);
              res();
            }, () => res());
          });
        } else if (entry.isDirectory) {
          const reader = entry.createReader();
          await new Promise((res) => {
            reader.readEntries(async (entries) => {
              for (const child of entries) {
                await traverseEntry(child, (path ? path + '/' : '') + entry.name);
              }
              res();
            }, () => res());
          });
        }
      };

      try {
        for (let i = 0; i < e.dataTransfer.items.length; i++) {
          const item = e.dataTransfer.items[i];
          if (item.kind === 'file') {
            const entry = item.webkitGetAsEntry();
            if (entry) await traverseEntry(entry);
          }
        }
      } catch (_) {}

      if (files.length > 0) {
        multiFileCallback(files);
        return;
      }
    }

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      if (singleFileCallback) {
        singleFileCallback(e.dataTransfer.files[0]);
      } else if (multiFileCallback) {
        multiFileCallback(Array.from(e.dataTransfer.files));
      }
    }
  });
}

function showAlert(message, type = 'info') {
  const container = document.getElementById('globalAlertContainer');
  if (!container) return;

  // Keep at most 2 active alerts to prevent screen clutter
  while (container.children.length >= 2) {
    container.removeChild(container.firstChild);
  }

  const alert = document.createElement('div');
  alert.className = `alert alert-${type} alert-dismissible fade show shadow-sm border-0 mb-3`;
  alert.role = 'alert';
  alert.innerHTML = `
    <div class="d-flex align-items-center">
      <i class="bi bi-${type === 'success' ? 'check-circle-fill text-success' : (type === 'danger' ? 'exclamation-octagon-fill text-danger' : 'info-circle-fill text-info')} me-2 fs-5"></i>
      <div class="small fw-medium">${message}</div>
    </div>
    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
  `;

  container.appendChild(alert);

  setTimeout(() => {
    alert.classList.remove('show');
    setTimeout(() => alert.remove(), 250);
  }, 4500);
}

function downloadCsvBlob(csvString, filename) {
  const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
