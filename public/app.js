/**
 * MedSeg-XAI Live Vercel Application Engine
 * =========================================
 * Client-side interactive engine supporting multi-design heatmaps,
 * interactive wipe comparator, cohort analytics, and case switching.
 */

// Global State
let appData = null;
let currentCaseId = "case_01";
let currentDesign = "turbo";
let currentOpacity = 0.55;
let currentWipeMode = "clean"; // "clean", "scrambled", "segmentation"

// Case Catalog In-Memory Fallback
const CASE_CATALOG = {
  case_01: {
    id: "case_01",
    title: "Case 01",
    pathology: "Glioblastoma Multiforme (Focal Enhancing Core)",
    modality: "T1ce (Contrast-Enhanced T1-weighted MRI)",
    slice_dim: "256 x 256",
    description: "Circumscribed contrast-enhancing lesion in the left fronto-parietal region. High contrast-to-noise ratio relative to background white matter.",
    box: [124, 124, 173, 173],
    metrics: {
      dice_clean: 0.962,
      dice_after_decoder_rand: 0.021,
      spearman_decoder_full: 0.124,
      ssim_decoder_full: 0.682,
      pass_v1_ssim: false,
      pass_v2_spearman: true
    }
  },
  case_02: {
    id: "case_02",
    title: "Case 02",
    pathology: "Diffuse High-Grade Glioma",
    modality: "T1ce (Contrast-Enhanced T1-weighted MRI)",
    slice_dim: "512 x 512 x 3",
    description: "Right temporal mass with prominent central necrotic cavity and thick peripheral rim enhancement.",
    box: [180, 210, 310, 350],
    metrics: {
      dice_clean: 0.941,
      dice_after_decoder_rand: 0.015,
      spearman_decoder_full: 0.098,
      ssim_decoder_full: 0.645,
      pass_v1_ssim: false,
      pass_v2_spearman: true
    }
  },
  case_03: {
    id: "case_03",
    title: "Case 03",
    pathology: "Anaplastic Astrocytoma",
    modality: "T1ce (Contrast-Enhanced T1-weighted MRI)",
    slice_dim: "512 x 512 x 3",
    description: "Left temporal lobe intra-axial mass with heterogeneous contrast enhancement and surrounding vasogenic edema.",
    box: [140, 160, 290, 310],
    metrics: {
      dice_clean: 0.972,
      dice_after_decoder_rand: 0.033,
      spearman_decoder_full: 0.141,
      ssim_decoder_full: 0.710,
      pass_v1_ssim: false,
      pass_v2_spearman: true
    }
  },
  case_04: {
    id: "case_04",
    title: "Case 04",
    pathology: "Deep Temporal Lesion",
    modality: "T1ce (Contrast-Enhanced T1-weighted MRI)",
    slice_dim: "512 x 512 x 3",
    description: "Right deep temporal lesion adjacent to the lateral ventricle. Distinct hypointense necrotic core with sharp peripheral borders.",
    box: [200, 190, 330, 320],
    metrics: {
      dice_clean: 0.970,
      dice_after_decoder_rand: 0.018,
      spearman_decoder_full: 0.112,
      ssim_decoder_full: 0.674,
      pass_v1_ssim: false,
      pass_v2_spearman: true
    }
  },
  case_05: {
    id: "case_05",
    title: "Case 05",
    pathology: "Infiltrative Frontal Glioma (Diffuse Margin)",
    modality: "T1ce (Contrast-Enhanced T1-weighted MRI)",
    slice_dim: "512 x 512 x 3",
    description: "Left frontal lobe infiltrative mass presenting with faint peripheral contrast enhancement and diffuse non-enhancing margins. Real-world challenging boundary.",
    box: [160, 140, 320, 290],
    metrics: {
      dice_clean: 0.641,
      dice_after_decoder_rand: 0.012,
      spearman_decoder_full: 0.153,
      ssim_decoder_full: 0.690,
      pass_v1_ssim: false,
      pass_v2_spearman: true
    }
  }
};

const HEATMAP_DESIGNS = [
  { id: "turbo", name: "Thermal Turbo", desc: "Standard clinical high-dynamic-range spectrum" },
  { id: "plasma", name: "Medical Plasma", desc: "High-contrast perceptually uniform saliency gradient" },
  { id: "inferno", name: "Solar Inferno", desc: "Deep black-purple-red-gold margin emphasis" },
  { id: "viridis", name: "Diagnostic Viridis", desc: "Colorblind-friendly green-yellow medical baseline" },
  { id: "contour", name: "Topographic Iso-Contour", desc: "Heatmap with vector attention field isolines" },
  { id: "residual", name: "Residual Attention Delta", desc: "Absolute difference highlighting destroyed attention" }
];

let dataBaseUrl = "data";

function resolveAsset(subpath) {
  return `${dataBaseUrl}/${subpath}`;
}

// Initialize
document.addEventListener("DOMContentLoaded", async () => {
  try {
    const res = await fetch("data/manifest.json");
    if (res.ok) {
      dataBaseUrl = "data";
      appData = await res.json();
    } else {
      const res2 = await fetch("public/data/manifest.json");
      if (res2.ok) {
        dataBaseUrl = "public/data";
        appData = await res2.json();
      }
    }
  } catch (err) {
    try {
      const res2 = await fetch("public/data/manifest.json");
      if (res2.ok) {
        dataBaseUrl = "public/data";
        appData = await res2.json();
      }
    } catch (err2) {
      console.warn("Using offline fallback catalog:", err2);
    }
  }

  setupEventListeners();
  renderCohortTable();
  renderDesignCatalog();
  setupSplitComparator();
  updateView();
});

// Event Listeners Setup
function setupEventListeners() {
  // Case Selector
  const caseSelector = document.getElementById("caseSelector");
  caseSelector.addEventListener("change", (e) => {
    currentCaseId = e.target.value;
    updateView();
  });

  // Tab Navigation
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => (p.style.display = "none"));

      btn.classList.add("active");
      const targetTab = document.getElementById(`tab-${btn.dataset.tab}`);
      if (targetTab) {
        targetTab.style.display = "block";
      }
    });
  });

  // Heatmap Design Switcher
  document.querySelectorAll(".design-btn[data-design]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".design-btn[data-design]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentDesign = btn.dataset.design;
      updateHeatmapImages();
      updateSplitComparatorImages();
    });
  });

  // Opacity Slider
  const opacitySlider = document.getElementById("opacitySlider");
  const opacityValue = document.getElementById("opacityValue");
  opacitySlider.addEventListener("input", (e) => {
    currentOpacity = e.target.value / 100;
    opacityValue.textContent = `${e.target.value}%`;
    document.getElementById("p3-image").style.opacity = currentOpacity;
    document.querySelectorAll(".catalog-heatmap-img").forEach((img) => {
      img.style.opacity = currentOpacity;
    });
  });

  // Wipe Modes
  document.getElementById("wipeModeClean").addEventListener("click", (e) => {
    setWipeMode("clean", e.target);
  });
  document.getElementById("wipeModeScrambled").addEventListener("click", (e) => {
    setWipeMode("scrambled", e.target);
  });
  document.getElementById("wipeModeSegmentation").addEventListener("click", (e) => {
    setWipeMode("segmentation", e.target);
  });
}

function setWipeMode(mode, targetBtn) {
  document.querySelectorAll("#tab-design-lab .design-btn").forEach((b) => b.classList.remove("active"));
  targetBtn.classList.add("active");
  currentWipeMode = mode;
  updateSplitComparatorImages();
}

// Update View Data
function updateView() {
  const caseData = CASE_CATALOG[currentCaseId];
  if (!caseData) return;

  const box = caseData.box;
  const metrics = caseData.metrics;

  // Header
  document.getElementById("caseHeaderTitle").textContent = `${caseData.title}: ${caseData.pathology}`;
  document.getElementById("caseHeaderDesc").textContent = caseData.description;

  // Dossier
  const boxW = box[2] - box[0];
  const boxH = box[3] - box[1];
  document.getElementById("caseDossier").innerHTML = `
    <div><strong>Pathology:</strong> ${caseData.pathology}</div>
    <div><strong>Modality:</strong> ${caseData.modality}</div>
    <div><strong>Resolution:</strong> <span class="meta-badge">${caseData.slice_dim}</span></div>
    <div><strong>Prompt Box:</strong> <span class="meta-badge">[${box.join(", ")}]</span></div>
    <div><strong>Prompt Size:</strong> <span class="meta-badge">${boxW} &times; ${boxH} px</span></div>
    <div><strong>Model:</strong> MedSAM ViT-Base (Decoder Prompt-Gated)</div>
  `;

  // Panel 1: Raw Input with Bounding Box
  document.getElementById("p1-image").src = resolveAsset(`${currentCaseId}/raw_box.png`);
  document.getElementById("p1-coords").textContent = `Prompt Box: [${box.join(", ")}]`;

  // Panel 2: Segmentation
  document.getElementById("p2-image").src = resolveAsset(`${currentCaseId}/segmentation.png`);
  const diceBadge = document.getElementById("p2-dice-badge");
  diceBadge.textContent = `Dice: ${metrics.dice_clean.toFixed(3)}`;
  if (metrics.dice_clean >= 0.85) {
    diceBadge.className = "badge-tag badge-green";
  } else {
    diceBadge.className = "badge-tag badge-red";
  }

  // Case 05 Under-segmentation notice
  const case05Alert = document.getElementById("p2-case05-alert");
  if (currentCaseId === "case_05") {
    case05Alert.style.display = "block";
  } else {
    case05Alert.style.display = "none";
  }

  // Panel 3: Heatmap
  updateHeatmapImages();

  // Verification Module
  document.getElementById("mprt-intact-img").src = resolveAsset(`${currentCaseId}/heatmap_turbo.png`);
  document.getElementById("mprt-scrambled-img").src = resolveAsset(`${currentCaseId}/scrambled_turbo.png`);
  document.getElementById("mprt-spearman-val").textContent = metrics.spearman_decoder_full.toFixed(3);
  document.getElementById("mprt-rand-dice-val").textContent = metrics.dice_after_decoder_rand.toFixed(3);
  document.getElementById("mprt-ssim-val").textContent = metrics.ssim_decoder_full.toFixed(3);

  // Transparency Curve
  document.getElementById("transparency-curve-img").src = resolveAsset(`${currentCaseId}/curve.png`);

  // Design Lab
  renderDesignCatalog();
  updateSplitComparatorImages();
}

function updateHeatmapImages() {
  const p3Img = document.getElementById("p3-image");
  if (currentDesign === "residual") {
    p3Img.src = resolveAsset(`${currentCaseId}/residual_turbo.png`);
  } else {
    p3Img.src = resolveAsset(`${currentCaseId}/heatmap_${currentDesign}.png`);
  }
  p3Img.style.opacity = currentOpacity;
}

// Interactive Wipe Comparator
function setupSplitComparator() {
  const comp = document.getElementById("splitComparator");
  const overlay = document.getElementById("splitOverlay");
  let isDown = false;

  function move(e) {
    if (!isDown) return;
    const rect = comp.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    let x = clientX - rect.left;
    if (x < 0) x = 0;
    if (x > rect.width) x = rect.width;
    const pct = (x / rect.width) * 100;
    overlay.style.width = `${pct}%`;
  }

  comp.addEventListener("mousedown", () => (isDown = true));
  window.addEventListener("mouseup", () => (isDown = false));
  comp.addEventListener("mousemove", move);

  comp.addEventListener("touchstart", () => (isDown = true));
  window.addEventListener("touchend", () => (isDown = false));
  comp.addEventListener("touchmove", move);

  // Set initial position to 50%
  overlay.style.width = "50%";
}

function updateSplitComparatorImages() {
  const underlay = document.getElementById("splitUnderlay");
  const overlayInner = document.getElementById("splitOverlayInner");

  const heatSrc = currentDesign === "residual"
    ? resolveAsset(`${currentCaseId}/residual_turbo.png`)
    : resolveAsset(`${currentCaseId}/heatmap_${currentDesign}.png`);

  if (currentWipeMode === "clean") {
    // Left: Clean Heatmap | Right: Raw MRI
    underlay.style.backgroundImage = `url('${resolveAsset(currentCaseId + "/raw.png")}')`;
    overlayInner.style.backgroundImage = `url('${heatSrc}')`;
  } else if (currentWipeMode === "scrambled") {
    // Left: Clean Heatmap | Right: Scrambled Heatmap
    underlay.style.backgroundImage = `url('${resolveAsset(currentCaseId + "/scrambled_turbo.png")}')`;
    overlayInner.style.backgroundImage = `url('${heatSrc}')`;
  } else {
    // Left: Clean Heatmap | Right: Segmentation contours
    underlay.style.backgroundImage = `url('${resolveAsset(currentCaseId + "/segmentation.png")}')`;
    overlayInner.style.backgroundImage = `url('${heatSrc}')`;
  }
}

// Render Design Catalog Grid in Tab 2
function renderDesignCatalog() {
  const grid = document.getElementById("designCatalogGrid");
  grid.innerHTML = "";

  HEATMAP_DESIGNS.forEach((d) => {
    const card = document.createElement("div");
    card.className = "panel-card";
    card.style.cursor = "pointer";

    let src = resolveAsset(`${currentCaseId}/heatmap_${d.id}.png`);
    if (d.id === "residual") src = resolveAsset(`${currentCaseId}/residual_turbo.png`);

    card.innerHTML = `
      <div class="glass-header" style="padding-bottom: 6px;">
        <span style="font-size: 0.9rem;">${d.name}</span>
        ${currentDesign === d.id ? '<span class="badge-tag">Active</span>' : ""}
      </div>
      <div style="font-size: 0.74rem; color: #64748b; margin-bottom: 8px;">${d.desc}</div>
      <div class="image-viewport">
        <img class="catalog-heatmap-img" src="${src}" alt="${d.name}" style="opacity: ${currentOpacity};">
      </div>
    `;

    card.addEventListener("click", () => {
      currentDesign = d.id;
      document.querySelectorAll(".design-btn[data-design]").forEach((btn) => {
        if (btn.dataset.design === d.id) {
          btn.classList.add("active");
        } else {
          btn.classList.remove("active");
        }
      });
      updateHeatmapImages();
      updateSplitComparatorImages();
      renderDesignCatalog();
    });

    grid.appendChild(card);
  });
}

// Render Cohort Table in Tab 3
function renderCohortTable() {
  const tbody = document.getElementById("cohortTableBody");
  tbody.innerHTML = "";

  Object.values(CASE_CATALOG).forEach((c) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${c.id}</strong></td>
      <td>${c.pathology}</td>
      <td><span class="meta-badge">${c.slice_dim}</span></td>
      <td><strong>${c.metrics.dice_clean.toFixed(3)}</strong></td>
      <td>${c.metrics.dice_after_decoder_rand.toFixed(3)}</td>
      <td><strong>${c.metrics.spearman_decoder_full.toFixed(3)}</strong></td>
      <td>${c.metrics.ssim_decoder_full.toFixed(3)}</td>
      <td><span class="badge-tag badge-green">PASS (&le; 0.3)</span></td>
      <td><span class="badge-tag badge-red">FAIL (&le; 0.5)</span></td>
    `;
    tbody.appendChild(tr);
  });
}
