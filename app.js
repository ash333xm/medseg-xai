/**
 * MedSeg-XAI Application Engine
 * =============================
 * Enterprise-grade medical segmentation and explainability workstation.
 * Features:
 * - 16 Full Clinical Cases (BraTS MRI, BTCV Abdominal CT, DeepLesion, Stress Corpus)
 * - Cache-Busting with ASSET_REV to guarantee 100% authentic scans without stale browser caching
 * - Full-Width Spacious Clinician PACS Workstation
 * - Quick Modality Filtering (All, Brain, Abdomen, Thorax, Stress)
 * - Human-in-the-Loop Cyclical Pipeline Interactive Inspector
 */

const ASSET_REV = "auth_rev_20261110_v3";

// Global State
let currentCaseId = "case_01";
let currentDesign = "turbo";
let currentOpacity = 0.55;
let currentThreshold = 15;
let overrideAuditGate = false;
let isPlayingSlices = false;
let slicePlayInterval = null;
let currentFilter = "all";

// Complete 16 Clinical Cases Catalog
const CASE_CATALOG = {
  case_01: {
    id: "case_01",
    title: "Case 01",
    pathology: "Glioblastoma Multiforme (Focal Enhancing Core)",
    modality: "T1ce / FLAIR (Axial z=63, MSD Task01)",
    category: "Brain MRI",
    organ: "Brain (Frontoparietal)",
    slice_dim: "512 x 512",
    z: 63,
    description: "Focal contrast-enhancing lesion in the left fronto-parietal region with high contrast-to-noise ratio relative to background parenchyma.",
    box: [283, 153, 398, 364],
    audit_passed: true,
    condition: "Standard Diagnostic (High SNR)",
    metrics: { dice_clean: 0.9029, dice_after_decoder_rand: 0.080, spearman_decoder_full: 0.027, ssim_decoder_full: 0.502 }
  },
  case_02: {
    id: "case_02",
    title: "Case 02",
    pathology: "Diffuse High-Grade Glioma",
    modality: "T1ce / FLAIR (Axial z=71, MSD Task01)",
    category: "Brain MRI",
    organ: "Brain (Bilateral)",
    slice_dim: "512 x 512",
    z: 71,
    description: "Multifocal infiltrative high-grade glioma extending across the corpus callosum with significant surrounding vasogenic edema.",
    box: [205, 178, 375, 342],
    audit_passed: true,
    condition: "Standard Diagnostic (Moderate SNR)",
    metrics: { dice_clean: 0.8142, dice_after_decoder_rand: 0.065, spearman_decoder_full: 0.019, ssim_decoder_full: 0.518 }
  },
  case_03: {
    id: "case_03",
    title: "Case 03",
    pathology: "Anaplastic Astrocytoma",
    modality: "T1ce / FLAIR (Axial z=80, MSD Task01)",
    category: "Brain MRI",
    organ: "Brain (Periventricular)",
    slice_dim: "512 x 512",
    z: 80,
    description: "Deep periventricular astroglial lesion with heterogeneous contrast uptake and non-enhancing infiltrative margins.",
    box: [240, 190, 360, 310],
    audit_passed: true,
    condition: "Standard Diagnostic (High SNR)",
    metrics: { dice_clean: 0.8564, dice_after_decoder_rand: 0.072, spearman_decoder_full: 0.034, ssim_decoder_full: 0.495 }
  },
  case_04: {
    id: "case_04",
    title: "Case 04",
    pathology: "Deep Temporal Lesion",
    modality: "T1ce / FLAIR (Axial z=58, MSD Task01)",
    category: "Brain MRI",
    organ: "Brain (Temporal)",
    slice_dim: "512 x 512",
    z: 58,
    description: "Deep medial temporal lobe mass abutting the hippocampus with complex anatomical boundary transitions.",
    box: [260, 140, 385, 290],
    audit_passed: true,
    condition: "Standard Diagnostic (Low Contrast Boundary)",
    metrics: { dice_clean: 0.7831, dice_after_decoder_rand: 0.054, spearman_decoder_full: -0.012, ssim_decoder_full: 0.534 }
  },
  case_05: {
    id: "case_05",
    title: "Case 05",
    pathology: "Infiltrative Frontal Glioma",
    modality: "T1ce / FLAIR (Axial z=66, MSD Task01)",
    category: "Brain MRI",
    organ: "Brain (Frontal)",
    slice_dim: "512 x 512",
    z: 66,
    description: "Infiltrative frontal glioma demonstrating diffuse border morphology and moderate under-segmentation of non-enhancing margins.",
    box: [220, 160, 370, 350],
    audit_passed: true,
    condition: "Challenging Infiltrative Margins (Dice 0.6405)",
    metrics: { dice_clean: 0.6405, dice_after_decoder_rand: 0.064, spearman_decoder_full: 0.031, ssim_decoder_full: 0.521 }
  },
  case_06: {
    id: "case_06",
    title: "Case 06",
    pathology: "Oligodendroglioma (1p/19q Co-deleted)",
    modality: "T2-FLAIR Mismatch (Axial z=76, MSD Task01)",
    category: "Brain MRI",
    organ: "Brain (Frontoparietal)",
    slice_dim: "512 x 512",
    z: 76,
    description: "Well-circumscribed frontoparietal lesion with heterogeneous calcifications and prominent T2-FLAIR mismatch sign.",
    box: [190, 160, 340, 310],
    audit_passed: true,
    condition: "Standard Diagnostic (High SNR)",
    metrics: { dice_clean: 0.8845, dice_after_decoder_rand: 0.076, spearman_decoder_full: 0.022, ssim_decoder_full: 0.498 }
  },
  case_07: {
    id: "case_07",
    title: "Case 07",
    pathology: "Parasagittal Meningioma (Dural Tail Sign)",
    modality: "T1ce Contrast MRI (Axial z=84, MSD Task01)",
    category: "Brain MRI",
    organ: "Brain (Extra-axial Convexity)",
    slice_dim: "512 x 512",
    z: 84,
    description: "Extra-axial hyperostotic dural-based mass displaying intense homogeneous gadolinium enhancement and classic dural tail.",
    box: [220, 210, 360, 350],
    audit_passed: true,
    condition: "Standard Diagnostic (Intense Enhancement)",
    metrics: { dice_clean: 0.9180, dice_after_decoder_rand: 0.088, spearman_decoder_full: 0.015, ssim_decoder_full: 0.485 }
  },
  case_08: {
    id: "case_08",
    title: "Case 08",
    pathology: "Posterior Fossa Brain Metastasis",
    modality: "T1ce / FLAIR (Axial z=42, MSD Task01)",
    category: "Brain MRI",
    organ: "Brain (Cerebellar Hemisphere)",
    slice_dim: "512 x 512",
    z: 42,
    description: "Solitary cerebellar metastasis from primary non-small cell lung carcinoma with disproportionate surrounding vasogenic edema.",
    box: [260, 180, 375, 295],
    audit_passed: true,
    condition: "Standard Diagnostic (High Peritumoral Edema)",
    metrics: { dice_clean: 0.8672, dice_after_decoder_rand: 0.062, spearman_decoder_full: 0.028, ssim_decoder_full: 0.512 }
  },
  case_09: {
    id: "case_09",
    title: "Case 09",
    pathology: "Hepatocellular Carcinoma (HCC)",
    modality: "Tri-phasic Abdominal CT (Portal Venous Phase, BTCV)",
    category: "Abdomen CT",
    organ: "Liver (Segments VII/VIII)",
    slice_dim: "512 x 512",
    z: 64,
    description: "Arterial phase hyper-enhancing liver lesion with late-phase portal venous washout in cirrhotic parenchyma (BTCV benchmark).",
    box: [140, 150, 310, 320],
    audit_passed: true,
    condition: "Visceral Abdominal Benchmark",
    metrics: { dice_clean: 0.8950, dice_after_decoder_rand: 0.071, spearman_decoder_full: 0.018, ssim_decoder_full: 0.490 }
  },
  case_10: {
    id: "case_10",
    title: "Case 10",
    pathology: "Renal Cell Carcinoma (Clear Cell)",
    modality: "Abdominal Contrast-Enhanced CT (Corticomedullary, BTCV)",
    category: "Abdomen CT",
    organ: "Kidney (Right Cortex)",
    slice_dim: "512 x 512",
    z: 55,
    description: "Exophytic corticomedullary mass arising from the right lower pole renal parenchyma with sharp pseudocapsule.",
    box: [270, 170, 390, 290],
    audit_passed: true,
    condition: "Visceral Abdominal Benchmark",
    metrics: { dice_clean: 0.9120, dice_after_decoder_rand: 0.079, spearman_decoder_full: 0.024, ssim_decoder_full: 0.505 }
  },
  case_11: {
    id: "case_11",
    title: "Case 11",
    pathology: "Pancreatic Ductal Adenocarcinoma",
    modality: "Pancreatic Protocol Thin-Slice CT (BTCV)",
    category: "Abdomen CT",
    organ: "Pancreas (Head / Uncinate)",
    slice_dim: "512 x 512",
    z: 68,
    description: "Hypoattenuating pancreatic head mass causing upstream main pancreatic duct dilation (double duct sign).",
    box: [210, 200, 310, 300],
    audit_passed: true,
    condition: "Visceral Abdominal Benchmark (Low Contrast)",
    metrics: { dice_clean: 0.8240, dice_after_decoder_rand: 0.058, spearman_decoder_full: 0.035, ssim_decoder_full: 0.528 }
  },
  case_12: {
    id: "case_12",
    title: "Case 12",
    pathology: "Splenomegaly & Acute Focal Infarction",
    modality: "Abdominal Contrast-Enhanced CT (BTCV)",
    category: "Abdomen CT",
    organ: "Spleen",
    slice_dim: "512 x 512",
    z: 60,
    description: "Peripherally-based wedge-shaped hypoattenuation in the splenic parenchyma representing acute focal ischemia.",
    box: [200, 290, 330, 420],
    audit_passed: true,
    condition: "Visceral Abdominal Benchmark",
    metrics: { dice_clean: 0.8790, dice_after_decoder_rand: 0.074, spearman_decoder_full: 0.021, ssim_decoder_full: 0.497 }
  },
  case_13: {
    id: "case_13",
    title: "Case 13",
    pathology: "Sub-centimeter Pulmonary Nodule",
    modality: "High-Resolution Chest CT (DeepLesion Benchmark)",
    category: "Thorax CT",
    organ: "Lung (Right Upper Lobe)",
    slice_dim: "512 x 512",
    z: 70,
    description: "Solid non-calcified pulmonary nodule with lobulated margins in the right upper lobe lung parenchyma (DeepLesion Universal Benchmark).",
    box: [290, 160, 370, 240],
    audit_passed: true,
    condition: "Thoracic Universal Lesion Benchmark",
    metrics: { dice_clean: 0.8520, dice_after_decoder_rand: 0.069, spearman_decoder_full: 0.030, ssim_decoder_full: 0.515 }
  },
  case_14: {
    id: "case_14",
    title: "Case 14",
    pathology: "Patient Head Motion Ghosting (SNR -12dB)",
    modality: "Phase-Encoding Ghosting (Synthetic Stress Corpus)",
    category: "Stress Corpus",
    organ: "Brain MRI",
    slice_dim: "512 x 512",
    z: 63,
    description: "Severe patient movement artifact causing periodic phase-direction ghosting. Model parameter randomization test collapses (Audit FAIL).",
    box: [200, 180, 370, 350],
    audit_passed: false,
    condition: "Severe Patient Motion Corrupted",
    metrics: { dice_clean: 0.5420, dice_after_decoder_rand: 0.495, spearman_decoder_full: 0.582, ssim_decoder_full: 0.642 }
  },
  case_15: {
    id: "case_15",
    title: "Case 15",
    pathology: "RF Gradient Spike & Streak Artifact",
    modality: "High-Frequency RF Interference (Synthetic Stress Corpus)",
    category: "Stress Corpus",
    organ: "Brain MRI",
    slice_dim: "512 x 512",
    z: 63,
    description: "Radiofrequency coil spike artifact creating linear geometric streaks across the field of view. Saliency map latches onto noise (Audit FAIL).",
    box: [250, 150, 380, 300],
    audit_passed: false,
    condition: "Hardware RF Interference Corrupted",
    metrics: { dice_clean: 0.6120, dice_after_decoder_rand: 0.420, spearman_decoder_full: 0.468, ssim_decoder_full: 0.618 }
  },
  case_16: {
    id: "case_16",
    title: "Case 16",
    pathology: "Severe Delayed Contrast Extravasation",
    modality: "Low SNR Parenchymal Washout (Synthetic Stress Corpus)",
    category: "Stress Corpus",
    organ: "Brain MRI",
    slice_dim: "512 x 512",
    z: 63,
    description: "Sub-optimal gadolinium timing resulting in diminished lesion contrast-to-noise ratio. Model parameters remain coupled (Audit PASS).",
    box: [230, 190, 350, 310],
    audit_passed: true,
    condition: "Sub-Optimal Contrast Extravasation",
    metrics: { dice_clean: 0.7240, dice_after_decoder_rand: 0.082, spearman_decoder_full: 0.045, ssim_decoder_full: 0.510 }
  }
};

// Smooth Image Loader with Cache-Busting
function setImgSrc(target, subpath) {
  const el = typeof target === "string" ? document.getElementById(target) : target;
  if (!el) return;

  const clean = subpath.replace(/^\/+/, "").split("?")[0];
  const candidates = [
    `data/${clean}?rev=${ASSET_REV}`,
    `public/data/${clean}?rev=${ASSET_REV}`,
    `/${clean}?rev=${ASSET_REV}`,
    `${clean}?rev=${ASSET_REV}`
  ];

  const preloader = new Image();
  let attempt = 0;

  preloader.onload = () => {
    el.classList.add("fade-out");
    setTimeout(() => {
      el.src = preloader.src;
      el.classList.remove("fade-out");
    }, 100);
  };

  preloader.onerror = () => {
    attempt++;
    if (attempt < candidates.length) {
      preloader.src = candidates[attempt];
    }
  };

  preloader.src = candidates[0];
}

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
  setupNavigationTabs();
  setupToolbarControls();
  setupCaseSelector();
  setupAxialSliceExplorer();
  populateVisualCaseGrid();
  populateCohortTable();
  renderActiveCase();
});

// Setup Navigation Tabs
function setupNavigationTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetId = `tab-${tab.getAttribute("data-tab")}`;
      document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.style.display = pane.id === targetId ? "block" : "none";
      });
    });
  });
}

// Setup Toolbar Controls (Modality Filter & Colormaps)
function setupToolbarControls() {
  // Filter chips
  const chips = document.querySelectorAll(".filter-chip");
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      chips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      currentFilter = chip.getAttribute("data-filter");
      populateVisualCaseGrid(currentFilter);
    });
  });

  // Colormap buttons
  const designBtns = document.querySelectorAll(".design-btn");
  designBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      designBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentDesign = btn.getAttribute("data-design");
      updateHeatmapViews();
    });
  });

  // Opacity Slider
  const opSlider = document.getElementById("opacitySlider");
  const opVal = document.getElementById("opacityValue");
  if (opSlider && opVal) {
    opSlider.addEventListener("input", (e) => {
      currentOpacity = e.target.value / 100;
      opVal.textContent = `${e.target.value}%`;
      const p3Img = document.getElementById("p3-image");
      if (p3Img) p3Img.style.opacity = currentOpacity;
    });
  }
}

// Setup Case Selector Dropdown
function setupCaseSelector() {
  const sel = document.getElementById("caseSelector");
  if (sel) {
    sel.addEventListener("change", (e) => {
      selectCaseAndOpen(e.target.value);
    });
  }
}

// Setup Axial Slice Slider & Video Animation
function setupAxialSliceExplorer() {
  const slider = document.getElementById("axialSliceSlider");
  const label = document.getElementById("axialSliceNum");
  const btnPlay = document.getElementById("btnPlaySlice");

  if (slider && label) {
    slider.addEventListener("input", (e) => {
      label.textContent = `z = ${e.target.value} / 155`;
    });
  }

  if (btnPlay) {
    btnPlay.addEventListener("click", () => {
      isPlayingSlices = !isPlayingSlices;
      if (isPlayingSlices) {
        btnPlay.textContent = "Pause Slices";
        btnPlay.style.background = "#fee2e2";
        btnPlay.style.color = "#b91c1c";
        slicePlayInterval = setInterval(() => {
          let cur = parseInt(slider.value, 10);
          cur = (cur + 1) % 155;
          slider.value = cur;
          label.textContent = `z = ${cur} / 155`;
        }, 60);
      } else {
        btnPlay.textContent = "Play Slices";
        btnPlay.style.background = "#ffffff";
        btnPlay.style.color = "var(--accent-blue)";
        clearInterval(slicePlayInterval);
      }
    });
  }
}

// Render Active Case in PACS View
function renderActiveCase() {
  const c = CASE_CATALOG[currentCaseId];
  if (!c) return;

  // Dropdown sync
  const sel = document.getElementById("caseSelector");
  if (sel && sel.value !== currentCaseId) sel.value = currentCaseId;

  // Panel 1: Raw scan with ROI box
  setImgSrc("p1-image", `${c.id}/raw_box.png`);
  const p1Coords = document.getElementById("p1-coords");
  if (p1Coords) p1Coords.textContent = `Prompt Box: [${c.box.join(", ")}] | ${c.organ}`;

  // Panel 2: 3D Segmentation Agreement
  setImgSrc("p2-image", `${c.id}/seg_overlay.png`);
  const diceBadge = document.getElementById("p2-dice-badge");
  if (diceBadge) {
    diceBadge.textContent = `3D Dice: ${c.metrics.dice_clean.toFixed(4)}`;
    diceBadge.className = c.metrics.dice_clean >= 0.70 ? "badge-tag badge-green" : "badge-tag badge-red";
  }

  const case05Alert = document.getElementById("p2-case05-alert");
  if (case05Alert) {
    case05Alert.style.display = currentCaseId === "case_05" ? "block" : "none";
  }

  // Panel 3: TMME Saliency Heatmap
  updateHeatmapViews();

  // Metric Cards
  const mPathology = document.getElementById("metric-pathology");
  const mOrgan = document.getElementById("metric-organ");
  const mModality = document.getElementById("metric-modality");
  const mCondition = document.getElementById("metric-condition");
  const mDice = document.getElementById("metric-dice");
  const mSpearman = document.getElementById("metric-spearman");

  if (mPathology) mPathology.textContent = c.pathology;
  if (mOrgan) mOrgan.textContent = `${c.organ} (${c.category})`;
  if (mModality) mModality.textContent = c.modality;
  if (mCondition) mCondition.textContent = c.condition;
  if (mDice) {
    mDice.textContent = c.metrics.dice_clean.toFixed(4);
    mDice.style.color = c.metrics.dice_clean >= 0.70 ? "#059669" : "#dc2626";
  }
  if (mSpearman) {
    mSpearman.textContent = c.metrics.spearman_decoder_full.toFixed(3);
    mSpearman.style.color = c.audit_passed ? "#059669" : "#dc2626";
  }

  // Audit Policy
  applyAuditPresentationPolicy();

  // Highlight active in Visual Case Grid
  document.querySelectorAll(".case-card-visual").forEach(el => {
    el.classList.toggle("active", el.getAttribute("data-id") === currentCaseId);
  });
}

// Update Panel 3 Heatmap View
function updateHeatmapViews() {
  const c = CASE_CATALOG[currentCaseId];
  if (!c) return;

  const fname = currentDesign === "residual" ? "residual_turbo.png" : `heatmap_${currentDesign}.png`;
  setImgSrc("p3-image", `${c.id}/${fname}`);
}

// Apply Audit Gate Policy
function applyAuditPresentationPolicy() {
  const c = CASE_CATALOG[currentCaseId];
  if (!c) return;

  const passed = c.audit_passed || overrideAuditGate;
  const banner = document.getElementById("auditGateBanner");
  const bannerTitle = document.getElementById("auditGateTitle");
  const bannerDesc = document.getElementById("auditGateDesc");
  const bannerIcon = document.getElementById("auditGateIcon");
  const p3Tag = document.getElementById("p3-audit-status-tag");
  const curtain = document.getElementById("heatmapWithheldCurtain");

  if (passed) {
    if (banner) {
      banner.className = "audit-gate-banner passed";
      bannerTitle.textContent = "Safety Audit Gate: PASSED &mdash; Heatmap Certified for Clinical Review";
      bannerDesc.textContent = `Model Parameter Randomization Test (MPRT) verified: Saliency map demonstrates strong parametric dependence on learned MedSAM-2 weights (Spearman rho collapsed to ${c.metrics.spearman_decoder_full.toFixed(3)} <= 0.30 at full decoder stage). Heatmap certified.`;
      bannerIcon.innerHTML = "&check;";
    }
    if (p3Tag) {
      p3Tag.textContent = "Certified";
      p3Tag.className = "badge-tag badge-green";
    }
    if (curtain) curtain.style.display = "none";
  } else {
    if (banner) {
      banner.className = "audit-gate-banner failed";
      bannerTitle.textContent = "Safety Audit Gate: BLOCKED &mdash; Heatmap Strictly Withheld";
      bannerDesc.textContent = `CRITICAL AUDIT NOTICE: Parameter Randomization Failure (Spearman rho = ${c.metrics.spearman_decoder_full.toFixed(3)} > 0.30). Heatmap withheld from radiological review under safety protocol.`;
      bannerIcon.innerHTML = "&times;";
    }
    if (p3Tag) {
      p3Tag.textContent = "Withheld";
      p3Tag.className = "badge-tag badge-red";
    }
    if (curtain) curtain.style.display = "flex";
  }
}

// Populate Visual Case Grid with Authentic Scans & Cache-Busters
function populateVisualCaseGrid(filter = "all") {
  const grid = document.getElementById("visualCaseGrid");
  if (!grid) return;

  const cases = Object.values(CASE_CATALOG).filter(c => {
    if (filter === "all") return true;
    if (filter === "brain") return c.category.includes("Brain");
    if (filter === "abdomen") return c.category.includes("Abdomen");
    if (filter === "thorax") return c.category.includes("Thorax");
    if (filter === "stress") return c.category.includes("Stress");
    return true;
  });

  grid.innerHTML = cases.map(c => `
    <div class="case-card-visual ${c.id === currentCaseId ? "active" : ""}" data-id="${c.id}" onclick="selectCaseAndOpen('${c.id}')">
      <div class="case-card-visual-media">
        <img src="data/${c.id}/raw.png?rev=${ASSET_REV}" onerror="this.src='public/data/${c.id}/raw.png?rev=${ASSET_REV}'" alt="${c.title}">
      </div>
      <div class="case-card-visual-title">${c.title}</div>
      <div class="case-card-visual-sub">${c.organ}</div>
      <div class="case-card-visual-meta">
        <span>Dice: <strong>${c.metrics.dice_clean.toFixed(3)}</strong></span>
        <span class="badge-tag ${c.audit_passed ? "badge-green" : "badge-red"}" style="padding: 1px 6px; font-size: 0.66rem;">${c.audit_passed ? "PASS" : "FAIL"}</span>
      </div>
    </div>
  `).join("");
}

// Select Case and Smoothly Open
window.selectCaseAndOpen = function(caseId) {
  currentCaseId = caseId;
  renderActiveCase();
  // Smooth scroll to top of PACS view
  const toolbar = document.querySelector(".workstation-toolbar-card");
  if (toolbar) toolbar.scrollIntoView({ behavior: "smooth", block: "start" });
};

// Interactive HITL Step Inspector
window.selectHitlStep = function(stepNum) {
  document.querySelectorAll(".hitl-node").forEach(n => n.classList.remove("active"));
  const node = document.getElementById(`node-step${stepNum}`);
  if (node) node.classList.add("active");

  const titleEl = document.getElementById("hitlDetailTitle");
  const descEl = document.getElementById("hitlDetailDesc");
  const imgEl = document.getElementById("hitlDetailImg");

  const steps = [
    null,
    {
      title: "Step 1: Middle Slice Prompt Conditioning",
      desc: "Radiologists define an initial bounding box around the lesion core on the axial slice with maximal tumor extent (+12px anatomical margin). This prompt conditions the prompt encoder to extract sparse positional embeddings.",
      img: `data/case_01/raw_box.png?rev=${ASSET_REV}`
    },
    {
      title: "Step 2: MedSAM-2 2D Zero-Shot Inference",
      desc: "The ViT image encoder and lightweight two-way cross-attention mask decoder produce an initial 2D segmentation hypothesis on the prompted slice in <15 milliseconds.",
      img: `data/case_01/segmentation.png?rev=${ASSET_REV}`
    },
    {
      title: "Step 3: Clinician 2D Mask Interactive Refinement",
      desc: "The clinician inspects the 2D contour. Optional positive (foreground) or negative (background) clicks are added at difficult boundaries to instantly rectify under- or over-segmentation.",
      img: `data/case_01/seg_overlay.png?rev=${ASSET_REV}`
    },
    {
      title: "Step 4: 3D Volumetric Memory Attention Propagation",
      desc: "MedSAM-2's streaming memory encoder stores the refined slice features in a FIFO memory bank. Bidirectional memory attention then propagates the mask forward and backward across all 155 volume slices.",
      img: `data/case_01/montage.png?rev=${ASSET_REV}`
    },
    {
      title: "Step 5: Final Refined 3D Mask Review & Clinical Export",
      desc: "The complete 3D volumetric segmentation is audited against clinical safety tolerances, verified via TMME layer-wise saliency rollout, and exported directly as DICOM RT-STRUCT.",
      img: `data/case_01/figure.png?rev=${ASSET_REV}`
    }
  ];

  const s = steps[stepNum];
  if (s) {
    if (titleEl) titleEl.textContent = s.title;
    if (descEl) descEl.textContent = s.desc;
    if (imgEl) imgEl.src = s.img;
  }
};

// Populate Cohort Summary Table
function populateCohortTable() {
  const tbody = document.getElementById("cohortTableBody");
  if (!tbody) return;

  tbody.innerHTML = Object.values(CASE_CATALOG).map(c => `
    <tr>
      <td><strong>${c.title}</strong></td>
      <td>${c.pathology} <br><span style="font-size: 0.72rem; color: var(--text-muted);">${c.organ}</span></td>
      <td><span class="badge-tag badge-cyan">${c.category}</span></td>
      <td><strong style="color: ${c.metrics.dice_clean >= 0.7 ? "#059669" : "#dc2626"};">${c.metrics.dice_clean.toFixed(4)}</strong></td>
      <td>${c.metrics.spearman_decoder_full.toFixed(3)}</td>
      <td>${c.metrics.ssim_decoder_full.toFixed(3)}</td>
      <td><span class="badge-tag ${c.audit_passed ? "badge-green" : "badge-red"}">${c.audit_passed ? "PASS" : "FAIL"}</span></td>
      <td><button class="filter-chip" onclick="selectCaseAndOpen('${c.id}')">Load Case</button></td>
    </tr>
  `).join("");
}
