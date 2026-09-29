/**
 * MedSeg-XAI Clinical Studio Application Engine
 * =============================================
 * Real clinical dataset and XAI audit visualization engine.
 * Synchronized with Google Drive cases and outputs.
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
    `/public/data/${clean}`,
    clean,
    `/${clean}`
  ];
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

// Authentic Clinical Case Catalog (Synchronized with Google Drive outputs/results/results.json)
const CASE_CATALOG = {
  case_01: {
    id: "case_01",
    title: "Case 01",
    pathology: "Glioblastoma Multiforme (Focal Enhancing Core)",
    modality: "T1ce / FLAIR (Axial z=63, MSD Task01)",
    slice_dim: "512 x 512",
    src: "BRATS_001",
    z: 63,
    description: "Focal contrast-enhancing lesion in the left fronto-parietal region. High contrast-to-noise ratio relative to background parenchyma.",
    box: [283, 153, 398, 364],
    metrics: {
      dice_clean: 0.9029,
      dice_after_decoder_rand: 0.080,
      spearman_decoder_full: 0.027,
      ssim_decoder_full: 0.502,
      pass_v1_ssim: false,
      pass_v2_spearman: true
    }
  },
  case_02: {
    id: "case_02",
    title: "Case 02",
    pathology: "Diffuse High-Grade Glioma",
    modality: "T1ce / FLAIR (Axial z=104, MSD Task01)",
    slice_dim: "512 x 512",
    src: "BRATS_002",
    z: 104,
    description: "Right temporal mass with prominent central necrosis and peripheral rim enhancement.",
    box: [153, 289, 270, 432],
    metrics: {
      dice_clean: 0.9033,
      dice_after_decoder_rand: 0.055,
      spearman_decoder_full: 0.013,
      ssim_decoder_full: 0.609,
      pass_v1_ssim: false,
      pass_v2_spearman: true
    }
  },
  case_03: {
    id: "case_03",
    title: "Case 03",
    pathology: "Anaplastic Astrocytoma",
    modality: "T1ce / FLAIR (Axial z=85, MSD Task01)",
    slice_dim: "512 x 512",
    src: "BRATS_003",
    z: 85,
    description: "Left temporal lobe intra-axial mass with heterogeneous contrast enhancement and surrounding vasogenic edema.",
    box: [165, 244, 391, 436],
    metrics: {
      dice_clean: 0.8359,
      dice_after_decoder_rand: 0.132,
      spearman_decoder_full: 0.062,
      ssim_decoder_full: 0.529,
      pass_v1_ssim: false,
      pass_v2_spearman: true
    }
  },
  case_04: {
    id: "case_04",
    title: "Case 04",
    pathology: "Deep Temporal Lesion",
    modality: "T1ce / FLAIR (Axial z=95, MSD Task01)",
    slice_dim: "512 x 512",
    src: "BRATS_004",
    z: 95,
    description: "Right deep temporal lesion adjacent to lateral ventricle. Sharp margins and distinct contrast enhancement.",
    box: [121, 191, 259, 347],
    metrics: {
      dice_clean: 0.9702,
      dice_after_decoder_rand: 0.087,
      spearman_decoder_full: -0.091,
      ssim_decoder_full: 0.509,
      pass_v1_ssim: false,
      pass_v2_spearman: true
    }
  },
  case_05: {
    id: "case_05",
    title: "Case 05",
    pathology: "Infiltrative Frontal Glioma (Diffuse Margin)",
    modality: "T1ce / FLAIR (Axial z=100, MSD Task01)",
    slice_dim: "512 x 512",
    src: "BRATS_005",
    z: 100,
    description: "Left frontal infiltrative glioma with faint peripheral enhancement. Model isolates hyperintense core while under-segmenting infiltrative border (Dice: 0.641).",
    box: [266, 272, 400, 443],
    metrics: {
      dice_clean: 0.6405,
      dice_after_decoder_rand: 0.064,
      spearman_decoder_full: 0.031,
      ssim_decoder_full: 0.600,
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
        if (appData && appData.cases) {
          Object.keys(appData.cases).forEach((cid) => {
            if (CASE_CATALOG[cid]) {
              CASE_CATALOG[cid] = { ...CASE_CATALOG[cid], ...appData.cases[cid] };
            }
          });
        }
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
  if (caseSelector) {
    caseSelector.addEventListener("change", (e) => {
      currentCaseId = e.target.value;
      updateView();
    });
  }

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
  if (opacitySlider && opacityValue) {
    opacitySlider.addEventListener("input", (e) => {
      currentOpacity = e.target.value / 100;
      opacityValue.textContent = `${e.target.value}%`;
      const p3 = document.getElementById("p3-image");
      if (p3) p3.style.opacity = currentOpacity;
      document.querySelectorAll(".catalog-heatmap-img").forEach((img) => {
        img.style.opacity = currentOpacity;
      });
    });
  }

  // Wipe Modes
  const wClean = document.getElementById("wipeModeClean");
  const wScram = document.getElementById("wipeModeScrambled");
  const wSeg = document.getElementById("wipeModeSegmentation");
  if (wClean) wClean.addEventListener("click", (e) => setWipeMode("clean", e.target));
  if (wScram) wScram.addEventListener("click", (e) => setWipeMode("scrambled", e.target));
  if (wSeg) wSeg.addEventListener("click", (e) => setWipeMode("segmentation", e.target));
}

function setWipeMode(mode, targetBtn) {
  document.querySelectorAll("#tab-design-lab .design-btn").forEach((b) => b.classList.remove("active"));
  if (targetBtn) targetBtn.classList.add("active");
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
  const titleEl = document.getElementById("caseHeaderTitle");
  if (titleEl) titleEl.textContent = `${caseData.title}: ${caseData.pathology}`;
  
  const descEl = document.getElementById("caseHeaderDesc");
  if (descEl) descEl.textContent = caseData.description;

  // Dossier
  const boxW = box[2] - box[0];
  const boxH = box[3] - box[1];
  const dossierEl = document.getElementById("caseDossier");
  if (dossierEl) {
    dossierEl.innerHTML = `
      <div><strong>Source Volume:</strong> <span class="meta-badge">${caseData.src || "BraTS"} (Slice z=${caseData.z || 63})</span></div>
      <div><strong>Pathology:</strong> ${caseData.pathology}</div>
      <div><strong>Modality:</strong> ${caseData.modality}</div>
      <div><strong>Resolution:</strong> <span class="meta-badge">${caseData.slice_dim}</span></div>
      <div><strong>Prompt Box:</strong> <span class="meta-badge">[${box.join(", ")}]</span></div>
      <div><strong>Prompt Size:</strong> <span class="meta-badge">${boxW} &times; ${boxH} px</span></div>
      <div><strong>Model:</strong> MedSAM ViT-Base (Prompt-Gated Decoder)</div>
    `;
  }

  // Panel 1: Raw Input with Bounding Box
  setImgSrc("p1-image", `${currentCaseId}/raw_box.png`);
  const coordsEl = document.getElementById("p1-coords");
  if (coordsEl) coordsEl.textContent = `Prompt Box: [${box.join(", ")}] (${boxW}×${boxH} px)`;

  // Panel 2: Segmentation
  setImgSrc("p2-image", `${currentCaseId}/segmentation.png`);
  const diceBadge = document.getElementById("p2-dice-badge");
  if (diceBadge) {
    diceBadge.textContent = `Dice: ${metrics.dice_clean.toFixed(4)}`;
    if (metrics.dice_clean >= 0.85) {
      diceBadge.className = "badge-tag badge-green";
    } else {
      diceBadge.className = "badge-tag badge-red";
    }
  }

  // Case 05 Under-segmentation notice
  const case05Alert = document.getElementById("p2-case05-alert");
  if (case05Alert) {
    if (currentCaseId === "case_05") {
      case05Alert.style.display = "block";
    } else {
      case05Alert.style.display = "none";
    }
  }

  // Panel 3: Heatmap
  updateHeatmapImages();

  // Verification Module
  setImgSrc("mprt-intact-img", `${currentCaseId}/heatmap_turbo.png`);
  setImgSrc("mprt-scrambled-img", `${currentCaseId}/scrambled_turbo.png`);
  
  const spVal = document.getElementById("mprt-spearman-val");
  if (spVal) spVal.textContent = metrics.spearman_decoder_full.toFixed(3);
  
  const rdVal = document.getElementById("mprt-rand-dice-val");
  if (rdVal) rdVal.textContent = metrics.dice_after_decoder_rand.toFixed(3);
  
  const smVal = document.getElementById("mprt-ssim-val");
  if (smVal) smVal.textContent = metrics.ssim_decoder_full.toFixed(3);

  // Transparency Curve
  setImgSrc("transparency-curve-img", `${currentCaseId}/curve.png`);

  // Full Publication Figure & Montage
  setImgSrc("full-figure-img", `${currentCaseId}/figure.png`);
  setImgSrc("clinician-figure-img", `${currentCaseId}/figure.png`);
  setImgSrc("audit-montage-img", `${currentCaseId}/montage.png`);
  setImgSrc("clinician-montage-img", `${currentCaseId}/montage.png`);

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
  if (!comp || !overlay) return;
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

  // Initial position 50%
  overlay.style.width = "50%";
}

function updateSplitComparatorImages() {
  const underlay = document.getElementById("splitUnderlay");
  const overlayInner = document.getElementById("splitOverlayInner");
  if (!underlay || !overlayInner) return;

  const heatFile = currentDesign === "residual"
    ? `${currentCaseId}/residual_turbo.png`
    : `${currentCaseId}/heatmap_${currentDesign}.png`;

  if (currentWipeMode === "clean") {
    setBgImageWithFallback(underlay, `${currentCaseId}/raw.png`);
    setBgImageWithFallback(overlayInner, heatFile);
  } else if (currentWipeMode === "scrambled") {
    setBgImageWithFallback(underlay, `${currentCaseId}/scrambled_turbo.png`);
    setBgImageWithFallback(overlayInner, heatFile);
  } else {
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
      <td><span class="meta-badge">${c.src || "BraTS"} (z=${c.z})</span></td>
      <td><strong>${c.metrics.dice_clean.toFixed(4)}</strong></td>
      <td><span class="badge-tag badge-red">${c.metrics.dice_after_decoder_rand.toFixed(3)}</span></td>
      <td><strong>${c.metrics.spearman_decoder_full.toFixed(3)}</strong></td>
      <td><span style="color: #64748b;">${c.metrics.ssim_decoder_full.toFixed(3)}</span></td>
      <td>
        <span class="badge-tag badge-green">Passed (&rho; &le; 0.30)</span>
      </td>
      <td>
        <span class="badge-tag badge-red">Failed (SSIM &gt; 0.50)</span>
      </td>
    `;
    tbody.appendChild(tr);
  });
}
