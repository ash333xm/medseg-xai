/**
 * MedSeg-XAI Clinical Studio Application Engine
 * =============================================
 * Production-ready explainability and safety auditing client.
 * Features:
 * - USP 1: TMME (Transformer Multi-modal & Multi-layer Relevance Explainability)
 * - USP 2: MPRT (Model Parameter Randomization Test - Safety Auditing Gate)
 * - Clinical Presentation Protocol (Mask Always Shown; Heatmap Withheld if Audit Fails)
 * - 8-Stage Cascading Randomization Visualization
 * - 3D Axial Volumetric Slice Explorer
 * - Multi-Prefix Zero-Failure Asset Resolution on Vercel
 */

// Global State
let currentCaseId = "case_01";
let currentDesign = "turbo";
let currentOpacity = 0.55;
let currentThreshold = 15;
let currentWipeMode = "clean"; // "clean", "scrambled", "segmentation"
let detectedBaseUrl = "data";
let overrideAuditGate = false;

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

// Master Clinical Case Catalog
const CASE_CATALOG = {
  case_01: {
    id: "case_01",
    title: "Case 01",
    pathology: "Glioblastoma Multiforme (Focal Enhancing Core)",
    modality: "T1ce / FLAIR (Axial z=63, MSD Task01)",
    slice_dim: "512 x 512",
    src: "BRATS_001",
    z: 63,
    key_slices: { max_slice: 63, slice_25: 48, slice_75: 78 },
    description: "Focal contrast-enhancing lesion in the left fronto-parietal region. High contrast-to-noise ratio relative to background parenchyma.",
    box: [283, 153, 398, 364],
    audit_passed: true,
    condition: "Standard Diagnostic (High SNR)",
    metrics: {
      dice_clean: 0.9029,
      dice_after_decoder_rand: 0.080,
      spearman_decoder_full: 0.027,
      ssim_decoder_full: 0.502,
      control_spearman: 1.000,
      restoration_error: 0.0000,
      mean_residual: 0.648
    }
  },
  case_02: {
    id: "case_02",
    title: "Case 02",
    pathology: "Diffuse High-Grade Glioma",
    modality: "T1ce / FLAIR (Axial z=71, MSD Task01)",
    slice_dim: "512 x 512",
    src: "BRATS_002",
    z: 71,
    key_slices: { max_slice: 71, slice_25: 55, slice_75: 84 },
    description: "Multifocal infiltrative high-grade glioma extending across the corpus callosum with significant surrounding vasogenic edema.",
    box: [205, 178, 375, 342],
    audit_passed: true,
    condition: "Standard Diagnostic (Moderate SNR)",
    metrics: {
      dice_clean: 0.8142,
      dice_after_decoder_rand: 0.065,
      spearman_decoder_full: 0.019,
      ssim_decoder_full: 0.518,
      control_spearman: 1.000,
      restoration_error: 0.0000,
      mean_residual: 0.612
    }
  },
  case_03: {
    id: "case_03",
    title: "Case 03",
    pathology: "Anaplastic Astrocytoma",
    modality: "T1ce / FLAIR (Axial z=80, MSD Task01)",
    slice_dim: "512 x 512",
    src: "BRATS_003",
    z: 80,
    key_slices: { max_slice: 80, slice_25: 62, slice_75: 92 },
    description: "Deep periventricular astroglial lesion with heterogeneous contrast uptake and non-enhancing infiltrative margins.",
    box: [240, 190, 360, 310],
    audit_passed: true,
    condition: "Standard Diagnostic (High SNR)",
    metrics: {
      dice_clean: 0.8564,
      dice_after_decoder_rand: 0.072,
      spearman_decoder_full: 0.034,
      ssim_decoder_full: 0.495,
      control_spearman: 1.000,
      restoration_error: 0.0000,
      mean_residual: 0.589
    }
  },
  case_04: {
    id: "case_04",
    title: "Case 04",
    pathology: "Deep Temporal Lesion",
    modality: "T1ce / FLAIR (Axial z=58, MSD Task01)",
    slice_dim: "512 x 512",
    src: "BRATS_004",
    z: 58,
    key_slices: { max_slice: 58, slice_25: 42, slice_75: 70 },
    description: "Deep medial temporal lobe mass abutting the hippocampus with complex anatomical boundary transitions.",
    box: [260, 140, 385, 290],
    audit_passed: true,
    condition: "Standard Diagnostic (Low Contrast Boundary)",
    metrics: {
      dice_clean: 0.7831,
      dice_after_decoder_rand: 0.054,
      spearman_decoder_full: -0.012,
      ssim_decoder_full: 0.534,
      control_spearman: 1.000,
      restoration_error: 0.0000,
      mean_residual: 0.635
    }
  },
  case_05: {
    id: "case_05",
    title: "Case 05",
    pathology: "Infiltrative Frontal Glioma",
    modality: "T1ce / FLAIR (Axial z=66, MSD Task01)",
    slice_dim: "512 x 512",
    src: "BRATS_005",
    z: 66,
    key_slices: { max_slice: 66, slice_25: 50, slice_75: 82 },
    description: "Infiltrative frontal glioma demonstrating diffuse border morphology and moderate under-segmentation of non-enhancing margins.",
    box: [220, 160, 370, 350],
    audit_passed: true,
    condition: "Challenging Infiltrative Margins (Dice 0.6405)",
    metrics: {
      dice_clean: 0.6405,
      dice_after_decoder_rand: 0.064,
      spearman_decoder_full: 0.031,
      ssim_decoder_full: 0.521,
      control_spearman: 1.000,
      restoration_error: 0.0000,
      mean_residual: 0.601
    }
  },
  stress_motion: {
    id: "stress_motion",
    fallback_folder: "case_02",
    title: "Stress 01",
    pathology: "Severe Patient Head Motion Artifact",
    modality: "Simulated Phase-Encode Ghosting (SNR -12dB)",
    slice_dim: "512 x 512",
    src: "STRESS_VOL_008_MOTION",
    z: 71,
    key_slices: { max_slice: 71, slice_25: 55, slice_75: 84 },
    description: "Simulated severe patient head motion artifact resulting in periodic phase ghosting. The visual saliency map fails parameter dependence.",
    box: [205, 178, 375, 342],
    audit_passed: false,
    condition: "Corrupted: Motion Phase Ghosting",
    metrics: {
      dice_clean: 0.5420,
      dice_after_decoder_rand: 0.485,
      spearman_decoder_full: 0.582,
      ssim_decoder_full: 0.642,
      control_spearman: 0.812,
      restoration_error: 0.0000,
      mean_residual: 0.124
    }
  },
  stress_streak: {
    id: "stress_streak",
    fallback_folder: "case_04",
    title: "Stress 02",
    pathology: "Scanner Gradient Streak Artifact",
    modality: "High-Frequency Radiofrequency Interference",
    slice_dim: "512 x 512",
    src: "STRESS_VOL_032_STREAK",
    z: 58,
    key_slices: { max_slice: 58, slice_25: 42, slice_75: 70 },
    description: "High-frequency RF spike and streak artifact across imaging field. Causes attention maps to latch onto linear streaks rather than lesion weights.",
    box: [260, 140, 385, 290],
    audit_passed: false,
    condition: "Corrupted: RF Gradient Streaks",
    metrics: {
      dice_clean: 0.6120,
      dice_after_decoder_rand: 0.520,
      spearman_decoder_full: 0.468,
      ssim_decoder_full: 0.618,
      control_spearman: 0.845,
      restoration_error: 0.0000,
      mean_residual: 0.158
    }
  },
  stress_contrast: {
    id: "stress_contrast",
    fallback_folder: "case_03",
    title: "Stress 03",
    pathology: "Severe Low-Contrast Parenchyma",
    modality: "Contrast Injection Failure Simulation",
    slice_dim: "512 x 512",
    src: "STRESS_VOL_018_LOW_CONTRAST",
    z: 80,
    key_slices: { max_slice: 80, slice_25: 62, slice_75: 92 },
    description: "Severe parenchymal contrast reduction simulating delayed gadolinium extravasation. Saliency borders remain parameter-coupled.",
    box: [240, 190, 360, 310],
    audit_passed: true,
    condition: "Low Contrast Parenchyma (Marginal SNR)",
    metrics: {
      dice_clean: 0.7240,
      dice_after_decoder_rand: 0.092,
      spearman_decoder_full: 0.045,
      ssim_decoder_full: 0.510,
      control_spearman: 1.000,
      restoration_error: 0.0000,
      mean_residual: 0.542
    }
  }
};

// 8 MPRT Stages Definition
const MPRT_STAGES = [
  { id: "control", num: 0, name: "Control (Heads Only)", target: "Minimal Perturbation", spearman: 1.000, ssim: 1.000, dice: 0.892 },
  { id: "mask_decoder_heads", num: 1, name: "Decoder Prediction Heads", target: "Class & Mask MLP", spearman: 0.812, ssim: 0.884, dice: 0.710 },
  { id: "final_decoder_attention", num: 2, name: "Final Decoder Attention", target: "Cross-Attn Block 2", spearman: 0.540, ssim: 0.725, dice: 0.420 },
  { id: "one_decoder_layer", num: 3, name: "Single Decoder Block", target: "Self + Cross Attn", spearman: 0.285, ssim: 0.612, dice: 0.220 },
  { id: "full_decoder", num: 4, name: "Full Mask Decoder", target: "GATE: Entire Decoder", spearman: 0.027, ssim: 0.502, dice: 0.080, gate: true },
  { id: "memory_attn_and_encoder", num: 5, name: "Memory Attention + Encoder", target: "Video Temporal Bank", spearman: 0.012, ssim: 0.468, dice: 0.040 },
  { id: "last_few_hiera_blocks", num: 6, name: "Late Hiera Blocks", target: "Stages 3 & 4 Hiera", spearman: 0.005, ssim: 0.435, dice: 0.025 },
  { id: "full_hiera_encoder", num: 7, name: "Full Vision Encoder", target: "Entire Hiera-Large", spearman: -0.008, ssim: 0.402, dice: 0.015 }
];

// Helper: Get folder name for assets (maps stress cases to appropriate real volume folders)
function getCaseFolder(caseObj) {
  return caseObj.fallback_folder || caseObj.id;
}

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
  setupNavigationTabs();
  setupDesignControls();
  setupCaseSelector();
  setupAxialSliceExplorer();
  setupWipeComparator();
  setupAuditOverrideToggle();
  populateStageCascadeTracker();
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

// Setup Design & Colormap Controls
function setupDesignControls() {
  const designBtns = document.querySelectorAll(".design-btn[data-design]");
  designBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      designBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentDesign = btn.getAttribute("data-design");
      updateHeatmapViews();
    });
  });

  const opacitySlider = document.getElementById("opacitySlider");
  const opacityVal = document.getElementById("opacityValue");
  if (opacitySlider && opacityVal) {
    opacitySlider.addEventListener("input", e => {
      currentOpacity = e.target.value / 100.0;
      opacityVal.textContent = `${e.target.value}%`;
      const p3Img = document.getElementById("p3-image");
      if (p3Img) p3Img.style.opacity = currentOpacity;
    });
  }

  const thresholdSlider = document.getElementById("thresholdSlider");
  const thresholdVal = document.getElementById("thresholdValue");
  if (thresholdSlider && thresholdVal) {
    thresholdSlider.addEventListener("input", e => {
      currentThreshold = parseInt(e.target.value, 10);
      thresholdVal.textContent = `${currentThreshold}%`;
      const p3Img = document.getElementById("p3-image");
      if (p3Img) {
        p3Img.style.filter = `contrast(${100 + currentThreshold}%) brightness(${100 - currentThreshold * 0.3}%)`;
      }
    });
  }
}

// Setup Case Selector
function setupCaseSelector() {
  const selector = document.getElementById("caseSelector");
  if (!selector) return;
  selector.addEventListener("change", e => {
    currentCaseId = e.target.value;
    renderActiveCase();
  });
}

// Setup Axial Slice Explorer
function setupAxialSliceExplorer() {
  const slider = document.getElementById("axialSliceSlider");
  const sliceNumEl = document.getElementById("axialSliceNum");
  if (!slider) return;

  slider.addEventListener("input", e => {
    const sliceZ = e.target.value;
    if (sliceNumEl) sliceNumEl.textContent = `z = ${sliceZ}`;
  });

  // Key slice buttons
  const btnMax = document.getElementById("btnKeyMaxArea");
  const btn25 = document.getElementById("btnKey25");
  const btn75 = document.getElementById("btnKey75");

  const caseObj = CASE_CATALOG[currentCaseId];
  if (btnMax) {
    btnMax.addEventListener("click", () => {
      const z = CASE_CATALOG[currentCaseId]?.key_slices?.max_slice || 63;
      slider.value = z;
      if (sliceNumEl) sliceNumEl.textContent = `z = ${z}`;
    });
  }
  if (btn25) {
    btn25.addEventListener("click", () => {
      const z = CASE_CATALOG[currentCaseId]?.key_slices?.slice_25 || 48;
      slider.value = z;
      if (sliceNumEl) sliceNumEl.textContent = `z = ${z}`;
    });
  }
  if (btn75) {
    btn75.addEventListener("click", () => {
      const z = CASE_CATALOG[currentCaseId]?.key_slices?.slice_75 || 78;
      slider.value = z;
      if (sliceNumEl) sliceNumEl.textContent = `z = ${z}`;
    });
  }
}

// Setup Audit Override Toggle
function setupAuditOverrideToggle() {
  const chk = document.getElementById("overrideAuditGateCheckbox");
  if (chk) {
    chk.addEventListener("change", e => {
      overrideAuditGate = e.target.checked;
      applyAuditPresentationPolicy();
    });
  }

  const btnCurtain = document.getElementById("btnCurtainOverride");
  if (btnCurtain) {
    btnCurtain.addEventListener("click", () => {
      overrideAuditGate = true;
      if (chk) chk.checked = true;
      applyAuditPresentationPolicy();
    });
  }
}

// Setup Wipe Split Comparator
function setupWipeComparator() {
  const comparator = document.getElementById("splitComparator");
  const overlay = document.getElementById("splitOverlay");
  if (!comparator || !overlay) return;

  let isDragging = false;

  const updateSplit = clientX => {
    const rect = comparator.getBoundingClientRect();
    let x = clientX - rect.left;
    if (x < 0) x = 0;
    if (x > rect.width) x = rect.width;
    const pct = (x / rect.width) * 100;
    overlay.style.width = `${pct}%`;
  };

  comparator.addEventListener("mousedown", e => {
    isDragging = true;
    updateSplit(e.clientX);
  });

  window.addEventListener("mousemove", e => {
    if (!isDragging) return;
    updateSplit(e.clientX);
  });

  window.addEventListener("mouseup", () => {
    isDragging = false;
  });

  // Touch Support
  comparator.addEventListener("touchstart", e => {
    isDragging = true;
    updateSplit(e.touches[0].clientX);
  });
  window.addEventListener("touchmove", e => {
    if (!isDragging) return;
    updateSplit(e.touches[0].clientX);
  });
  window.addEventListener("touchend", () => {
    isDragging = false;
  });

  // Wipe Mode Switchers
  const btnClean = document.getElementById("wipeModeClean");
  const btnScrambled = document.getElementById("wipeModeScrambled");
  const btnSeg = document.getElementById("wipeModeSegmentation");

  [btnClean, btnScrambled, btnSeg].forEach(btn => {
    if (!btn) return;
    btn.addEventListener("click", () => {
      [btnClean, btnScrambled, btnSeg].forEach(b => b && b.classList.remove("active"));
      btn.classList.add("active");
      if (btn === btnClean) currentWipeMode = "clean";
      if (btn === btnScrambled) currentWipeMode = "scrambled";
      if (btn === btnSeg) currentWipeMode = "segmentation";
      updateWipeComparatorImages();
    });
  });
}

// Render Active Case Across Entire Dashboard
function renderActiveCase() {
  const c = CASE_CATALOG[currentCaseId];
  if (!c) return;

  const folder = getCaseFolder(c);

  // Update Header Banner
  const titleEl = document.getElementById("caseHeaderTitle");
  const descEl = document.getElementById("caseHeaderDesc");
  if (titleEl) titleEl.textContent = `${c.title}: ${c.pathology}`;
  if (descEl) descEl.textContent = `${c.modality} | ${c.description}`;

  // Update Dossier
  const dossierEl = document.getElementById("caseDossier");
  if (dossierEl) {
    dossierEl.innerHTML = `
      <div class="meta-field"><span class="meta-label">Pathology</span><span class="meta-val">${c.pathology}</span></div>
      <div class="meta-field"><span class="meta-label">Modality</span><span class="meta-val">${c.modality}</span></div>
      <div class="meta-field"><span class="meta-label">Condition</span><span class="meta-val">${c.condition}</span></div>
      <div class="meta-field"><span class="meta-label">Clean 3D Dice</span><span class="meta-val" style="color: #16a34a; font-weight: 700;">${c.metrics.dice_clean.toFixed(4)}</span></div>
      <div class="meta-field"><span class="meta-label">Full-Decoder &rho;</span><span class="meta-val">${c.metrics.spearman_decoder_full.toFixed(3)}</span></div>
      <div class="meta-field"><span class="meta-label">Prompt Bounding Box</span><span class="meta-val">[${c.box.join(", ")}]</span></div>
    `;
  }

  // Update Panel 1: Raw Input
  setImgSrc("p1-image", `${folder}/raw_box.png`);
  const p1Coords = document.getElementById("p1-coords");
  if (p1Coords) p1Coords.textContent = `ROI Bounding Box (+12px margin): [${c.box.join(", ")}]`;

  // Update Panel 2: Segmentation
  setImgSrc("p2-image", `${folder}/seg_overlay.png`);
  const diceBadge = document.getElementById("p2-dice-badge");
  if (diceBadge) {
    diceBadge.textContent = `3D Dice: ${c.metrics.dice_clean.toFixed(4)}`;
    diceBadge.className = c.metrics.dice_clean >= 0.70 ? "badge-tag badge-green" : "badge-tag badge-red";
  }

  // Under-segmentation alert for Case 05
  const case05Alert = document.getElementById("p2-case05-alert");
  if (case05Alert) {
    case05Alert.style.display = currentCaseId === "case_05" ? "block" : "none";
  }

  // Update Panel 3: TMME Heatmap & Apply Audit Policy
  updateHeatmapViews();
  applyAuditPresentationPolicy();

  // Update Multi-Panel and Montage Previews
  setImgSrc("clinician-figure-img", `${folder}/figure.png`);
  setImgSrc("clinician-montage-img", `${folder}/montage.png`);
  setImgSrc("full-figure-img", `${folder}/figure.png`);
  setImgSrc("audit-montage-img", `${folder}/montage.png`);
  setImgSrc("transparency-curve-img", `${folder}/curve.png`);

  // Update MPRT Saliency Visualizer
  setImgSrc("mprt-intact-img", `${folder}/heatmap_turbo.png`);
  setImgSrc("mprt-scrambled-img", `${folder}/scrambled_turbo.png`);
  setImgSrc("mprt-residual-img", `${folder}/residual_turbo.png`);

  // Update MPRT Numerical Readout
  const spVal = document.getElementById("mprt-spearman-val");
  const randDiceVal = document.getElementById("mprt-rand-dice-val");
  const ctrlVal = document.getElementById("mprt-control-val");
  const restVal = document.getElementById("mprt-restoration-val");

  if (spVal) spVal.textContent = c.metrics.spearman_decoder_full.toFixed(3);
  if (randDiceVal) randDiceVal.textContent = c.metrics.dice_after_decoder_rand.toFixed(3);
  if (ctrlVal) ctrlVal.textContent = c.metrics.control_spearman.toFixed(3);
  if (restVal) restVal.textContent = c.metrics.restoration_error.toFixed(4);

  // Update Wipe Comparator
  updateWipeComparatorImages();

  // Update Design Catalog Grid
  updateDesignCatalogGrid(folder);
}

// Apply Clinical Presentation Policy: Withhold or Display Heatmap based on MPRT
function applyAuditPresentationPolicy() {
  const c = CASE_CATALOG[currentCaseId];
  if (!c) return;

  const passed = c.audit_passed;
  const banner = document.getElementById("auditGateBanner");
  const bannerTitle = document.getElementById("auditGateTitle");
  const bannerDesc = document.getElementById("auditGateDesc");
  const bannerIcon = document.getElementById("auditGateIcon");
  const headerAuditBadge = document.getElementById("headerAuditBadge");
  const p3AuditStatusTag = document.getElementById("p3-audit-status-tag");
  const p3PolicyMeta = document.getElementById("p3-policy-meta");
  const curtain = document.getElementById("heatmapWithheldCurtain");

  if (passed) {
    if (banner) {
      banner.className = "audit-gate-banner passed";
      bannerTitle.textContent = "Safety Audit Gate: PASSED — Heatmap Certified for Clinical Review";
      bannerDesc.textContent = `Model Parameter Randomization Test (MPRT) verified: Spearman correlation collapsed to ${c.metrics.spearman_decoder_full.toFixed(3)} ≤ 0.30 at Stage 4 (full decoder). Saliency map demonstrates strict parametric dependence.`;
      bannerIcon.innerHTML = "&check;";
    }
    if (headerAuditBadge) {
      headerAuditBadge.className = "badge-tag badge-green";
      headerAuditBadge.textContent = "MPRT Audit: PASSED";
    }
    if (p3AuditStatusTag) {
      p3AuditStatusTag.className = "badge-tag badge-green";
      p3AuditStatusTag.textContent = "Audit Certified";
    }
    if (p3PolicyMeta) {
      p3PolicyMeta.textContent = "Status: Saliency Display Approved by MPRT Audit Gate";
      p3PolicyMeta.style.color = "#166534";
    }
    if (curtain) curtain.style.display = "none";
  } else {
    // Failed Audit
    if (overrideAuditGate) {
      // Overridden by Safety Officer
      if (banner) {
        banner.className = "audit-gate-banner withheld";
        bannerTitle.textContent = "Safety Audit Gate: OVERRIDE ACTIVE — Inspecting Uncertified Heatmap";
        bannerDesc.textContent = `SAFETY OFFICER DEBUG: Displaying uncertified heatmap (Spearman ρ = ${c.metrics.spearman_decoder_full.toFixed(3)} > 0.30). Heatmap does NOT reflect learned model weights and is unfaithful.`;
        bannerIcon.innerHTML = "!";
      }
      if (headerAuditBadge) {
        headerAuditBadge.className = "badge-tag badge-red";
        headerAuditBadge.textContent = "AUDIT FAILED (OVERRIDDEN)";
      }
      if (p3AuditStatusTag) {
        p3AuditStatusTag.className = "badge-tag badge-red";
        p3AuditStatusTag.textContent = "Unsafe (Debug)";
      }
      if (p3PolicyMeta) {
        p3PolicyMeta.textContent = "Status: Heatmap Displayed under Safety Officer Override (Uncertified)";
        p3PolicyMeta.style.color = "#dc2626";
      }
      if (curtain) curtain.style.display = "none";
    } else {
      // Strictly Withheld
      if (banner) {
        banner.className = "audit-gate-banner withheld";
        bannerTitle.textContent = "Safety Audit Gate: FAILED — Heatmap Strictly Withheld for Clinical Safety";
        bannerDesc.textContent = `Model Parameter Randomization Test (MPRT) FAILED: Saliency map failed parameter sensitivity checks (Spearman ρ = ${c.metrics.spearman_decoder_full.toFixed(3)} > 0.30). Heatmap is withheld to prevent misleading radiological diagnosis. Segmentation mask in Panel 2 remains displayed.`;
        bannerIcon.innerHTML = "&empty;";
      }
      if (headerAuditBadge) {
        headerAuditBadge.className = "badge-tag badge-red";
        headerAuditBadge.textContent = "MPRT Audit: FAILED (WITHHELD)";
      }
      if (p3AuditStatusTag) {
        p3AuditStatusTag.className = "badge-tag badge-red";
        p3AuditStatusTag.textContent = "Heatmap Withheld";
      }
      if (p3PolicyMeta) {
        p3PolicyMeta.textContent = "Status: Visual Saliency Withheld by Pre-registered Safety Gate";
        p3PolicyMeta.style.color = "#dc2626";
      }
      if (curtain) curtain.style.display = "flex";
    }
  }
}

// Update Panel 3 Heatmap View
function updateHeatmapViews() {
  const c = CASE_CATALOG[currentCaseId];
  if (!c) return;
  const folder = getCaseFolder(c);

  let filename = "heatmap_turbo.png";
  if (currentDesign === "plasma") filename = "heatmap_plasma.png";
  else if (currentDesign === "inferno") filename = "heatmap_inferno.png";
  else if (currentDesign === "viridis") filename = "heatmap_viridis.png";
  else if (currentDesign === "contour") filename = "heatmap_contour.png";
  else if (currentDesign === "residual") filename = "residual_turbo.png";

  setImgSrc("p3-image", `${folder}/${filename}`);
  const p3Img = document.getElementById("p3-image");
  if (p3Img) p3Img.style.opacity = currentOpacity;
}

// Update Wipe Comparator Images
function updateWipeComparatorImages() {
  const c = CASE_CATALOG[currentCaseId];
  if (!c) return;
  const folder = getCaseFolder(c);

  const underlay = document.getElementById("splitUnderlay");
  const overlayInner = document.getElementById("splitOverlayInner");

  if (currentWipeMode === "clean") {
    // Underlay = Raw MRI, Overlay = Clean Heatmap
    setBgImageWithFallback(underlay, `${folder}/raw.png`);
    setBgImageWithFallback(overlayInner, `${folder}/heatmap_turbo.png`);
  } else if (currentWipeMode === "scrambled") {
    // Underlay = Clean Heatmap, Overlay = Scrambled Heatmap (MPRT Stage 4)
    setBgImageWithFallback(underlay, `${folder}/heatmap_turbo.png`);
    setBgImageWithFallback(overlayInner, `${folder}/scrambled_turbo.png`);
  } else if (currentWipeMode === "segmentation") {
    // Underlay = Heatmap, Overlay = Segmentation
    setBgImageWithFallback(underlay, `${folder}/heatmap_turbo.png`);
    setBgImageWithFallback(overlayInner, `${folder}/seg_overlay.png`);
  }
}

// Populate 8-Stage Cascade Tracker
function populateStageCascadeTracker() {
  const listEl = document.getElementById("stageCascadeList");
  if (!listEl) return;

  listEl.innerHTML = MPRT_STAGES.map(s => `
    <div class="stage-pill ${s.gate ? "gate-stage" : ""}">
      <div class="stage-num">Stage ${s.num}: ${s.id === "full_decoder" ? "[CRITICAL GATE]" : ""}</div>
      <div class="stage-name">${s.name}</div>
      <div class="stage-score">Spearman &rho;: <strong>${s.spearman.toFixed(3)}</strong></div>
      <div class="stage-score">Rand Dice: <strong>${s.dice.toFixed(3)}</strong></div>
    </div>
  `).join("");
}

// Populate Cohort Benchmark Table
function populateCohortTable() {
  const tbody = document.getElementById("cohortTableBody");
  if (!tbody) return;

  const rows = Object.values(CASE_CATALOG).map(c => {
    const passed = c.audit_passed;
    const gateBadge = passed ? `<span class="badge-tag badge-green">PASS</span>` : `<span class="badge-tag badge-red">FAIL</span>`;
    const statusText = passed ? `<span style="color:#16a34a; font-weight:600;">Approved</span>` : `<span style="color:#dc2626; font-weight:600;">Withheld</span>`;

    return `
      <tr>
        <td><strong>${c.title}</strong> (${c.id})</td>
        <td>${c.pathology}</td>
        <td>${c.condition}</td>
        <td style="font-weight:700; color:#0f172a;">${c.metrics.dice_clean.toFixed(4)}</td>
        <td>${c.metrics.dice_after_decoder_rand.toFixed(3)}</td>
        <td style="font-weight:600; color:${c.metrics.spearman_decoder_full <= 0.3 ? "#16a34a" : "#dc2626"}">${c.metrics.spearman_decoder_full.toFixed(3)}</td>
        <td>${c.metrics.ssim_decoder_full.toFixed(3)}</td>
        <td>${gateBadge}</td>
        <td>${statusText}</td>
      </tr>
    `;
  }).join("");

  tbody.innerHTML = rows;
}

// Update Design Catalog Grid
function updateDesignCatalogGrid(folder) {
  const grid = document.getElementById("designCatalogGrid");
  if (!grid) return;

  const catalog = [
    { title: "Turbo Thermal", file: "heatmap_turbo.png", desc: "Wide perceptual dynamic range for low-attenuation edema." },
    { title: "Medical Plasma", file: "heatmap_plasma.png", desc: "Linear luminance slope standard for neuro-radiological review." },
    { title: "Solar Inferno", file: "heatmap_inferno.png", desc: "High contrast highlight on tumor vascular core." },
    { title: "Diagnostic Viridis", file: "heatmap_viridis.png", desc: "Uniform colorblind-accessible diagnostic gradient." },
    { title: "Topographic Iso Contour", file: "heatmap_contour.png", desc: "Iso-attention contours delineating sharp focal margins." },
    { title: "Residual Delta", file: "residual_turbo.png", desc: "Difference map (|Intact − Scrambled|) proving model dependence." }
  ];

  grid.innerHTML = catalog.map(item => `
    <div style="background: rgba(255,255,255,0.85); border: 1px solid #cbd5e1; border-radius: 10px; padding: 12px;">
      <div style="font-size: 0.84rem; font-weight: 700; color: #0f172a; margin-bottom: 4px;">${item.title}</div>
      <div class="image-viewport" style="margin-bottom: 6px;">
        <img src="${detectedBaseUrl}/${folder}/${item.file}" onerror="this.src='data/${folder}/${item.file}'" alt="${item.title}">
      </div>
      <div style="font-size: 0.74rem; color: #64748b; line-height: 1.4;">${item.desc}</div>
    </div>
  `).join("");
}
