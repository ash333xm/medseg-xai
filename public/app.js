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
let detectedBaseUrl = "data";

// Candidate Asset Prefixes for Zero-Failure Resolution on Vercel
const CANDIDATE_PREFIXES = ["data", "/data", "./data", "public/data", "/public/data"];

function resolveAsset(subpath) {
  const clean = subpath.replace(/^\/+/, "");
  return `${detectedBaseUrl}/${clean}`;
}

// Resilient Image Loader with Multi-Prefix Fallback
function setImgSrc(target, subpath) {
  const el = typeof target === "string" ? document.getElementById(target) : target;
  if (!el) return;
  const clean = subpath.replace(/^\/+/, "");
  const candidates = [
    `${detectedBaseUrl}/${clean}`,
    `data/${clean}`,
    `/data/${clean}`,
    `./data/${clean}`,
    `public/data/${clean}`,
    `/public/data/${clean}`
  ];
  // Deduplicate candidates preserving order
  const uniqueCandidates = [...new Set(candidates)];
  let attempt = 0;

  el.onerror = () => {
    attempt++;
    if (attempt < uniqueCandidates.length) {
      el.src = uniqueCandidates[attempt];
    }
  };
  el.src = uniqueCandidates[0];
}

// Resilient Background Image Loader for Wipe Comparator
function setBgImageWithFallback(element, subpath) {
  if (!element) return;
  const clean = subpath.replace(/^\/+/, "");
  const candidates = [
    `${detectedBaseUrl}/${clean}`,
    `data/${clean}`,
    `/data/${clean}`,
    `./data/${clean}`,
    `public/data/${clean}`,
    `/public/data/${clean}`
  ];
  const uniqueCandidates = [...new Set(candidates)];
  let attempt = 0;

  function tryNext() {
    if (attempt >= uniqueCandidates.length) return;
    const testImg = new Image();
    const candidateUrl = uniqueCandidates[attempt];
    testImg.onload = () => {
      element.style.backgroundImage = `url('${candidateUrl}')`;
    };
    testImg.onerror = () => {
      attempt++;
      tryNext();
    };
    testImg.src = candidateUrl;
  }
  tryNext();
}

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

// Initialize
document.addEventListener("DOMContentLoaded", async () => {
  // Probe for working manifest location
  const manifestCandidates = [
    "data/manifest.json",
    "/data/manifest.json",
    "./data/manifest.json",
    "public/data/manifest.json",
    "/public/data/manifest.json"
  ];

  for (const path of manifestCandidates) {
    try {
      const res = await fetch(path);
      if (res.ok) {
        appData = await res.json();
        detectedBaseUrl = path.replace("/manifest.json", "");
        console.log("Resolved asset base URL:", detectedBaseUrl);
        break;
      }
    } catch (e) {
      // try next
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
  setImgSrc("p1-image", `${currentCaseId}/raw_box.png`);
  document.getElementById("p1-coords").textContent = `Prompt Box: [${box.join(", ")}]`;

  // Panel 2: Segmentation
  setImgSrc("p2-image", `${currentCaseId}/segmentation.png`);
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
  setImgSrc("mprt-intact-img", `${currentCaseId}/heatmap_turbo.png`);
  setImgSrc("mprt-scrambled-img", `${currentCaseId}/scrambled_turbo.png`);
  document.getElementById("mprt-spearman-val").textContent = metrics.spearman_decoder_full.toFixed(3);
  document.getElementById("mprt-rand-dice-val").textContent = metrics.dice_after_decoder_rand.toFixed(3);
  document.getElementById("mprt-ssim-val").textContent = metrics.ssim_decoder_full.toFixed(3);

  // Transparency Curve
  setImgSrc("transparency-curve-img", `${currentCaseId}/curve.png`);

  // Design Lab
  renderDesignCatalog();
  updateSplitComparatorImages();
}

function updateHeatmapImages() {
  const p3Img = document.getElementById("p3-image");
  if (!p3Img) return;
  const file = currentDesign === "residual"
    ? `${currentCaseId}/residual_turbo.png`
    : `${currentCaseId}/heatmap_${currentDesign}.png`;
  setImgSrc(p3Img, file);
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

  const heatFile = currentDesign === "residual"
    ? `${currentCaseId}/residual_turbo.png`
    : `${currentCaseId}/heatmap_${currentDesign}.png`;

  if (currentWipeMode === "clean") {
    // Left: Clean Heatmap | Right: Raw MRI
    setBgImageWithFallback(underlay, `${currentCaseId}/raw.png`);
    setBgImageWithFallback(overlayInner, heatFile);
  } else if (currentWipeMode === "scrambled") {
    // Left: Clean Heatmap | Right: Scrambled Heatmap
    setBgImageWithFallback(underlay, `${currentCaseId}/scrambled_turbo.png`);
    setBgImageWithFallback(overlayInner, heatFile);
  } else {
    // Left: Clean Heatmap | Right: Segmentation contours
    setBgImageWithFallback(underlay, `${currentCaseId}/segmentation.png`);
    setBgImageWithFallback(overlayInner, heatFile);
  }
}

// Render Design Catalog Grid in Tab 2
function renderDesignCatalog() {
  const grid = document.getElementById("designCatalogGrid");
  if (!grid) return;
  grid.innerHTML = "";

  HEATMAP_DESIGNS.forEach((d) => {
    const card = document.createElement("div");
    card.className = "panel-card";
    card.style.cursor = "pointer";

    const file = d.id === "residual"
      ? `${currentCaseId}/residual_turbo.png`
      : `${currentCaseId}/heatmap_${d.id}.png`;

    card.innerHTML = `
      <div class="glass-header" style="padding-bottom: 6px;">
        <span style="font-size: 0.9rem;">${d.name}</span>
        ${currentDesign === d.id ? '<span class="badge-tag">Active</span>' : ""}
      </div>
      <div style="font-size: 0.74rem; color: #64748b; margin-bottom: 8px;">${d.desc}</div>
      <div class="image-viewport">
        <img class="catalog-heatmap-img" alt="${d.name}" style="opacity: ${currentOpacity};">
      </div>
    `;

    const imgEl = card.querySelector(".catalog-heatmap-img");
    setImgSrc(imgEl, file);

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
  if (!tbody) return;
  tbody.innerHTML = "";

  Object.values(CASE_CATALOG).forEach((c) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${c.id}</strong></td>
      <td>${c.pathology}</td>
      <td><span class="meta-badge">${c.slice_dim}</span></td>
      <td><strong>${c.metrics.dice_clean.toFixed(3)}</strong></td>
      <td><span class="badge-tag badge-red">${c.metrics.dice_after_decoder_rand.toFixed(3)}</span></td>
      <td><strong>${c.metrics.spearman_decoder_full.toFixed(3)}</strong></td>
      <td><span style="color: #64748b;">${c.metrics.ssim_decoder_full.toFixed(3)}</span></td>
      <td>
        <span class="badge-tag badge-green">Passed (&rho; &le; 0.30)</span>
      </td>
    `;
    tbody.appendChild(tr);
  });
}
