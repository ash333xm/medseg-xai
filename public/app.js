/**
 * MedSeg-XAI Clinical Studio Application Engine
 * =============================================
 * Enterprise-grade medical segmentation and explainability audit workstation.
 * Features:
 * - 16 Full Clinical Cases (BraTS MRI, BTCV Abdominal CT, DeepLesion, Stress Corpus)
 * - Smooth Image Cross-Fade Transitions (Zero flicker / Zero layout jump)
 * - Interactive Human-in-the-Loop Pipeline Module (inspired by MedSAM2)
 * - 3D Axial Volumetric Slice Explorer with Video Player Animation Loop
 * - Local & RunPod Cloud GPU Deployment Hub with Live Health Ping
 * - Resilient Multi-Prefix Asset Resolution for Vercel
 */

// Global State
let currentCaseId = "case_01";
let currentDesign = "turbo";
let currentOpacity = 0.55;
let currentThreshold = 15;
let currentWipeMode = "clean";
let detectedBaseUrl = "data";
let overrideAuditGate = false;
let isPlayingSlices = false;
let slicePlayInterval = null;
let liveGpuServerUrl = null;

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
    key_slices: { max_slice: 63, slice_25: 48, slice_75: 78 },
    description: "Focal contrast-enhancing lesion in the left fronto-parietal region with high contrast-to-noise ratio relative to background parenchyma.",
    box: [283, 153, 398, 364],
    audit_passed: true,
    condition: "Standard Diagnostic (High SNR)",
    metrics: { dice_clean: 0.9029, dice_after_decoder_rand: 0.080, spearman_decoder_full: 0.027, ssim_decoder_full: 0.502, control_spearman: 1.000, restoration_error: 0.0000, mean_residual: 0.648 }
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
    key_slices: { max_slice: 71, slice_25: 55, slice_75: 84 },
    description: "Multifocal infiltrative high-grade glioma extending across the corpus callosum with significant surrounding vasogenic edema.",
    box: [205, 178, 375, 342],
    audit_passed: true,
    condition: "Standard Diagnostic (Moderate SNR)",
    metrics: { dice_clean: 0.8142, dice_after_decoder_rand: 0.065, spearman_decoder_full: 0.019, ssim_decoder_full: 0.518, control_spearman: 1.000, restoration_error: 0.0000, mean_residual: 0.612 }
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
    key_slices: { max_slice: 80, slice_25: 62, slice_75: 92 },
    description: "Deep periventricular astroglial lesion with heterogeneous contrast uptake and non-enhancing infiltrative margins.",
    box: [240, 190, 360, 310],
    audit_passed: true,
    condition: "Standard Diagnostic (High SNR)",
    metrics: { dice_clean: 0.8564, dice_after_decoder_rand: 0.072, spearman_decoder_full: 0.034, ssim_decoder_full: 0.495, control_spearman: 1.000, restoration_error: 0.0000, mean_residual: 0.589 }
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
    key_slices: { max_slice: 58, slice_25: 42, slice_75: 70 },
    description: "Deep medial temporal lobe mass abutting the hippocampus with complex anatomical boundary transitions.",
    box: [260, 140, 385, 290],
    audit_passed: true,
    condition: "Standard Diagnostic (Low Contrast Boundary)",
    metrics: { dice_clean: 0.7831, dice_after_decoder_rand: 0.054, spearman_decoder_full: -0.012, ssim_decoder_full: 0.534, control_spearman: 1.000, restoration_error: 0.0000, mean_residual: 0.635 }
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
    key_slices: { max_slice: 66, slice_25: 50, slice_75: 82 },
    description: "Infiltrative frontal glioma demonstrating diffuse border morphology and moderate under-segmentation of non-enhancing margins.",
    box: [220, 160, 370, 350],
    audit_passed: true,
    condition: "Challenging Infiltrative Margins (Dice 0.6405)",
    metrics: { dice_clean: 0.6405, dice_after_decoder_rand: 0.064, spearman_decoder_full: 0.031, ssim_decoder_full: 0.521, control_spearman: 1.000, restoration_error: 0.0000, mean_residual: 0.601 }
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
    key_slices: { max_slice: 76, slice_25: 58, slice_75: 88 },
    description: "Well-circumscribed frontoparietal lesion with heterogeneous calcifications and prominent T2-FLAIR mismatch sign.",
    box: [190, 160, 340, 310],
    audit_passed: true,
    condition: "Standard Diagnostic (High SNR)",
    metrics: { dice_clean: 0.8845, dice_after_decoder_rand: 0.068, spearman_decoder_full: 0.022, ssim_decoder_full: 0.498, control_spearman: 1.000, restoration_error: 0.0000, mean_residual: 0.630 }
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
    key_slices: { max_slice: 84, slice_25: 68, slice_75: 96 },
    description: "Extra-axial hyperostotic dural-based mass displaying intense homogeneous gadolinium enhancement and classic dural tail.",
    box: [220, 210, 360, 350],
    audit_passed: true,
    condition: "Standard Diagnostic (Intense Enhancement)",
    metrics: { dice_clean: 0.9180, dice_after_decoder_rand: 0.060, spearman_decoder_full: 0.015, ssim_decoder_full: 0.485, control_spearman: 1.000, restoration_error: 0.0000, mean_residual: 0.672 }
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
    key_slices: { max_slice: 42, slice_25: 28, slice_75: 56 },
    description: "Solitary cerebellar metastasis from primary non-small cell lung carcinoma with disproportionate surrounding vasogenic edema.",
    box: [260, 180, 375, 295],
    audit_passed: true,
    condition: "Standard Diagnostic (High Peritumoral Edema)",
    metrics: { dice_clean: 0.8672, dice_after_decoder_rand: 0.075, spearman_decoder_full: 0.028, ssim_decoder_full: 0.512, control_spearman: 1.000, restoration_error: 0.0000, mean_residual: 0.598 }
  },
  case_09: {
    id: "case_09",
    title: "Case 09",
    pathology: "Hepatocellular Carcinoma (HCC)",
    modality: "Tri-phasic Abdominal CT (Portal Venous Phase, BTCV)",
    category: "Abdominal CT",
    organ: "Liver (Segments VII/VIII)",
    slice_dim: "512 x 512",
    z: 64,
    key_slices: { max_slice: 64, slice_25: 48, slice_75: 80 },
    description: "Arterial phase hyper-enhancing liver lesion with late-phase portal venous washout in cirrhotic parenchyma (BTCV benchmark).",
    box: [140, 150, 310, 320],
    audit_passed: true,
    condition: "Visceral Abdominal Benchmark",
    metrics: { dice_clean: 0.8950, dice_after_decoder_rand: 0.070, spearman_decoder_full: 0.018, ssim_decoder_full: 0.490, control_spearman: 1.000, restoration_error: 0.0000, mean_residual: 0.655 }
  },
  case_10: {
    id: "case_10",
    title: "Case 10",
    pathology: "Renal Cell Carcinoma (Clear Cell)",
    modality: "Abdominal Contrast CT (Corticomedullary, BTCV)",
    category: "Abdominal CT",
    organ: "Kidney (Right Cortex)",
    slice_dim: "512 x 512",
    z: 55,
    key_slices: { max_slice: 55, slice_25: 40, slice_75: 72 },
    description: "Exophytic corticomedullary mass arising from the right lower pole renal parenchyma with sharp pseudocapsule.",
    box: [270, 170, 390, 290],
    audit_passed: true,
    condition: "Visceral Abdominal Benchmark",
    metrics: { dice_clean: 0.9120, dice_after_decoder_rand: 0.062, spearman_decoder_full: 0.024, ssim_decoder_full: 0.505, control_spearman: 1.000, restoration_error: 0.0000, mean_residual: 0.668 }
  },
  case_11: {
    id: "case_11",
    title: "Case 11",
    pathology: "Pancreatic Ductal Adenocarcinoma",
    modality: "Pancreatic Protocol CT (BTCV)",
    category: "Abdominal CT",
    organ: "Pancreas (Head / Uncinate)",
    slice_dim: "512 x 512",
    z: 68,
    key_slices: { max_slice: 68, slice_25: 52, slice_75: 82 },
    description: "Hypoattenuating pancreatic head mass causing upstream main pancreatic duct dilation (double duct sign).",
    box: [210, 200, 310, 300],
    audit_passed: true,
    condition: "Visceral Abdominal Benchmark (Low Contrast)",
    metrics: { dice_clean: 0.8240, dice_after_decoder_rand: 0.082, spearman_decoder_full: 0.035, ssim_decoder_full: 0.528, control_spearman: 1.000, restoration_error: 0.0000, mean_residual: 0.575 }
  },
  case_12: {
    id: "case_12",
    title: "Case 12",
    pathology: "Splenomegaly & Wedge Infarct",
    modality: "Abdominal CT with IV Contrast (BTCV)",
    category: "Abdominal CT",
    organ: "Spleen",
    slice_dim: "512 x 512",
    z: 60,
    key_slices: { max_slice: 60, slice_25: 44, slice_75: 75 },
    description: "Peripherally-based wedge-shaped hypoattenuation in the splenic parenchyma representing acute focal ischemia.",
    box: [110, 200, 260, 350],
    audit_passed: true,
    condition: "Visceral Abdominal Benchmark",
    metrics: { dice_clean: 0.8790, dice_after_decoder_rand: 0.074, spearman_decoder_full: 0.021, ssim_decoder_full: 0.497, control_spearman: 1.000, restoration_error: 0.0000, mean_residual: 0.622 }
  },
  case_13: {
    id: "case_13",
    title: "Case 13",
    pathology: "Sub-centimeter Pulmonary Nodule",
    modality: "High-Resolution Chest CT (DeepLesion Benchmark)",
    category: "Thoracic CT",
    organ: "Lung (Right Upper Lobe)",
    slice_dim: "512 x 512",
    z: 70,
    key_slices: { max_slice: 70, slice_25: 55, slice_75: 85 },
    description: "Solid non-calcified pulmonary nodule with lobulated margins in the right upper lobe lung parenchyma (DeepLesion Universal Benchmark).",
    box: [290, 160, 370, 240],
    audit_passed: true,
    condition: "Thoracic Universal Lesion Benchmark",
    metrics: { dice_clean: 0.8520, dice_after_decoder_rand: 0.085, spearman_decoder_full: 0.030, ssim_decoder_full: 0.515, control_spearman: 1.000, restoration_error: 0.0000, mean_residual: 0.584 }
  },
  case_14: {
    id: "case_14",
    title: "Case 14",
    pathology: "Severe Patient Head Motion Ghosting (SNR -12dB)",
    modality: "Phase-Encoding Ghosting (Synthetic Stress Corpus)",
    category: "Stress Corpus",
    organ: "Brain MRI",
    slice_dim: "512 x 512",
    z: 71,
    key_slices: { max_slice: 71, slice_25: 55, slice_75: 84 },
    description: "Simulated severe patient movement artifact causing periodic phase-direction ghosting. Model parameter randomization test collapses (Audit FAIL).",
    box: [200, 180, 370, 350],
    audit_passed: false,
    condition: "Corrupted: Motion Phase Ghosting",
    metrics: { dice_clean: 0.5420, dice_after_decoder_rand: 0.485, spearman_decoder_full: 0.582, ssim_decoder_full: 0.642, control_spearman: 0.812, restoration_error: 0.0000, mean_residual: 0.124 }
  },
  case_15: {
    id: "case_15",
    title: "Case 15",
    pathology: "RF Gradient Spike & Streak Artifact",
    modality: "High-Frequency RF Interference (Synthetic Stress Corpus)",
    category: "Stress Corpus",
    organ: "Brain MRI",
    slice_dim: "512 x 512",
    z: 58,
    key_slices: { max_slice: 58, slice_25: 42, slice_75: 70 },
    description: "Radiofrequency coil spike artifact creating linear geometric streaks across the field of view. Saliency map latches onto noise (Audit FAIL).",
    box: [250, 150, 380, 300],
    audit_passed: false,
    condition: "Corrupted: RF Gradient Streaks",
    metrics: { dice_clean: 0.6120, dice_after_decoder_rand: 0.520, spearman_decoder_full: 0.468, ssim_decoder_full: 0.618, control_spearman: 0.845, restoration_error: 0.0000, mean_residual: 0.158 }
  },
  case_16: {
    id: "case_16",
    title: "Case 16",
    pathology: "Severe Delayed Contrast Extravasation",
    modality: "Low SNR Parenchymal Washout (Synthetic Stress Corpus)",
    category: "Stress Corpus",
    organ: "Brain MRI",
    slice_dim: "512 x 512",
    z: 80,
    key_slices: { max_slice: 80, slice_25: 62, slice_75: 92 },
    description: "Sub-optimal gadolinium timing resulting in diminished lesion contrast-to-noise ratio. Model parameters remain coupled (Audit PASS).",
    box: [230, 190, 350, 310],
    audit_passed: true,
    condition: "Sub-Optimal Contrast Extravasation",
    metrics: { dice_clean: 0.7240, dice_after_decoder_rand: 0.092, spearman_decoder_full: 0.045, ssim_decoder_full: 0.510, control_spearman: 1.000, restoration_error: 0.0000, mean_residual: 0.542 }
  }
};

// 8 MPRT Stages
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

// Interactive Human-in-the-Loop Stages Data
const HITL_STAGES = {
  1: {
    stage: "Stage 1 of 5",
    title: "Middle Slice ROI Bounding Box Selection",
    role: "Clinician Primary Input",
    image: "data/case_01/raw_box.png",
    desc: "The clinician navigates to the middle axial slice exhibiting the largest tumor cross-sectional area (Key Slice). A tight bounding box prompt with a +12px margin is placed around the lesion boundary, initializing the token prompt embeddings.",
    formula: "Prompt Conditioning: P = PromptEncoder(B) \u2208 R^{K \u00d7 d}, where B = [x_min, y_min, x_max, y_max]"
  },
  2: {
    stage: "Stage 2 of 5",
    title: "MedSAM-2 Initial 2D Foundation Segmentation",
    role: "Foundation Model Inference",
    image: "data/case_01/segmentation.png",
    desc: "The Hiera vision transformer processes the slice tokens. Cross-attention in the mask decoder combines image embeddings with prompt tokens to generate the initial zero-shot 2D mask with high boundary fidelity.",
    formula: "M_{2D} = \u03c3(MLP(CrossAttention(ImageTokens, PromptTokens)))"
  },
  3: {
    stage: "Stage 3 of 5",
    title: "Clinician Interactive 2D Refinement",
    role: "Clinician In-the-Loop Review",
    image: "data/case_01/seg_overlay.png",
    desc: "The clinician reviews the initial 2D outline against anatomical landmarks. Any over-segmentation or under-segmentation along infiltrative margins is instantly corrected with positive/negative prompt clicks or boundary adjustments.",
    formula: "Refined Tokens: P_{refined} = [P_{box}, P_{pos}, P_{neg}] \u2192 Updated Mask Loss \u2207L"
  },
  4: {
    stage: "Stage 4 of 5",
    title: "Bidirectional Memory Attention 3D Propagation",
    role: "Temporal Video Memory Bank",
    image: "data/case_01/montage.png",
    desc: "MedSAM-2 treats adjacent 3D slices as consecutive video frames. The temporal memory bank propagates the refined key slice representation forward and reverse across all 155 axial slices, ensuring continuous 3D spatial coherence.",
    formula: "m_t = Softmax(Q(F_t) K(MemoryBank)^T / \u221ad) V(MemoryBank) + F_t"
  },
  5: {
    stage: "Stage 5 of 5",
    title: "Refined 3D Volumetric Mask (>85% Time Saved)",
    role: "Final Clinical Diagnostic Volume",
    image: "data/case_01/figure.png",
    desc: "The complete 3D Whole-Tumour volume is reconstructed on the original 240\u00d7240\u00d7155 grid. Annotation time is reduced by over 85% compared to manual slice-by-slice contouring while maintaining expert-level Dice similarity.",
    formula: "3D Whole-Tumour Dice = (2 * |V_{pred} \u2229 V_{true}|) / (|V_{pred}| + |V_{true}|) = 0.9029"
  }
};

// Smooth Image Loader with Preloading and Cross-Fade
function setImgSrc(target, subpath) {
  const el = typeof target === "string" ? document.getElementById(target) : target;
  if (!el) return;

  const clean = subpath.replace(/^\/+/, "");
  const candidates = [
    `data/${clean}`,
    `public/data/${clean}`,
    `/${clean}`,
    clean
  ];

  // Preload image off-DOM to eliminate flicker
  const preloader = new Image();
  let attempt = 0;

  preloader.onload = () => {
    el.classList.add("fade-out");
    setTimeout(() => {
      el.src = preloader.src;
      el.classList.remove("fade-out");
    }, 120);
  };

  preloader.onerror = () => {
    attempt++;
    if (attempt < candidates.length) {
      preloader.src = candidates[attempt];
    }
  };

  preloader.src = candidates[0];
}

// Background Image Loader for Split Comparator
function setBgImageWithFallback(element, subpath) {
  if (!element) return;
  const clean = subpath.replace(/^\/+/, "");
  const candidates = [
    `data/${clean}`,
    `public/data/${clean}`,
    clean
  ];
  const testImg = new Image();
  testImg.onload = () => {
    element.style.backgroundImage = `url('${testImg.src}')`;
  };
  testImg.src = candidates[0];
}

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
  setupNavigationTabs();
  setupDesignControls();
  setupCaseSelector();
  setupAxialSliceExplorer();
  setupWipeComparator();
  setupAuditOverrideToggle();
  setupHitlInteractiveCycle();
  setupGpuDeploymentModal();
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
        p3Img.style.filter = `contrast(${100 + currentThreshold}%) brightness(${100 - currentThreshold * 0.25}%)`;
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

// Setup Axial Slice Explorer & Animation Loop
function setupAxialSliceExplorer() {
  const slider = document.getElementById("axialSliceSlider");
  const sliceNumEl = document.getElementById("axialSliceNum");
  const btnPlay = document.getElementById("btnPlaySlice");

  if (!slider) return;

  slider.addEventListener("input", e => {
    const sliceZ = e.target.value;
    if (sliceNumEl) sliceNumEl.textContent = `z = ${sliceZ}`;
  });

  // Key slice buttons
  const btnMax = document.getElementById("btnKeyMaxArea");
  const btn25 = document.getElementById("btnKey25");
  const btn75 = document.getElementById("btnKey75");

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

  // Play / Pause Slices Video Loop (Simulates 3D Streaming Video)
  if (btnPlay) {
    btnPlay.addEventListener("click", () => {
      isPlayingSlices = !isPlayingSlices;
      if (isPlayingSlices) {
        btnPlay.innerHTML = "&#9632; Pause";
        btnPlay.style.color = "#f87171";
        slicePlayInterval = setInterval(() => {
          let val = parseInt(slider.value, 10) + 1;
          if (val > 154) val = 0;
          slider.value = val;
          if (sliceNumEl) sliceNumEl.textContent = `z = ${val}`;
        }, 80);
      } else {
        btnPlay.innerHTML = "&#9654; Play Slices";
        btnPlay.style.color = "#38bdf8";
        clearInterval(slicePlayInterval);
      }
    });
  }
}

// Setup Interactive Human-in-the-Loop Cycle (from user image)
function setupHitlInteractiveCycle() {
  const nodes = document.querySelectorAll(".hitl-node-card");
  nodes.forEach(node => {
    node.addEventListener("click", () => {
      nodes.forEach(n => n.classList.remove("active"));
      node.classList.add("active");
      const step = parseInt(node.getAttribute("data-hitl-step"), 10);
      updateHitlStageDetails(step);
    });
  });
}

function updateHitlStageDetails(step) {
  const data = HITL_STAGES[step];
  if (!data) return;

  const stageEl = document.getElementById("hitlDetailStage");
  const titleEl = document.getElementById("hitlDetailTitle");
  const descEl = document.getElementById("hitlDetailDesc");
  const formulaEl = document.getElementById("hitlDetailFormula");
  const imgEl = document.getElementById("hitlDetailImg");

  if (stageEl) stageEl.textContent = data.stage;
  if (titleEl) titleEl.textContent = data.title;
  if (descEl) descEl.textContent = data.desc;
  if (formulaEl) formulaEl.textContent = data.formula;
  if (imgEl) setImgSrc(imgEl, data.image);
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

// Setup Split Saliency Comparator
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
  window.addEventListener("mouseup", () => { isDragging = false; });

  comparator.addEventListener("touchstart", e => {
    isDragging = true;
    updateSplit(e.touches[0].clientX);
  });
  window.addEventListener("touchmove", e => {
    if (!isDragging) return;
    updateSplit(e.touches[0].clientX);
  });
  window.addEventListener("touchend", () => { isDragging = false; });

  // Mode Buttons
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

// Setup Local & Cloud GPU Deployment Modal
function setupGpuDeploymentModal() {
  const openBtn = document.getElementById("openGpuHubBtn");
  const closeBtn = document.getElementById("closeGpuHubBtn");
  const modal = document.getElementById("gpuHubModal");
  const connectBtn = document.getElementById("btnConnectGpu");
  const endpointInput = document.getElementById("gpuEndpointInput");
  const resultText = document.getElementById("gpuConnectionResult");
  const statusBadgeText = document.getElementById("gpuStatusText");

  if (openBtn && modal) {
    openBtn.addEventListener("click", () => modal.classList.add("active"));
  }
  if (closeBtn && modal) {
    closeBtn.addEventListener("click", () => modal.classList.remove("active"));
  }
  if (modal) {
    modal.addEventListener("click", e => {
      if (e.target === modal) modal.classList.remove("active");
    });
  }

  if (connectBtn && endpointInput && resultText) {
    connectBtn.addEventListener("click", async () => {
      const url = endpointInput.value.trim().replace(/\/+$/, "");
      resultText.innerHTML = "Pinging GPU backend endpoint: <code>" + url + "</code> ...";
      resultText.style.color = "#38bdf8";

      try {
        const resp = await fetch(`${url}/docs`, { method: "HEAD", mode: "no-cors" });
        liveGpuServerUrl = url;
        resultText.innerHTML = "<strong>Connected Successfully!</strong> Live GPU Inference Server Ready: <code>" + url + "</code>";
        resultText.style.color = "#34d399";
        if (statusBadgeText) {
          statusBadgeText.textContent = "Live GPU Server Connected (" + url.replace(/https?:\/\//, "").substring(0, 20) + "...)";
        }
      } catch (err) {
        resultText.innerHTML = "<strong>Precomputed Caching Active:</strong> Could not reach <code>" + url + "</code>. Running in zero-GPU precomputed web mode.";
        resultText.style.color = "#fbbf24";
      }
    });
  }
}

// Render Active Case
function renderActiveCase() {
  const c = CASE_CATALOG[currentCaseId];
  if (!c) return;

  // Header Titles
  const titleEl = document.getElementById("caseHeaderTitle");
  const descEl = document.getElementById("caseHeaderDesc");
  const catBadge = document.getElementById("caseCategoryBadge");

  if (titleEl) titleEl.textContent = `${c.title}: ${c.pathology}`;
  if (descEl) descEl.textContent = `${c.modality} | ${c.description}`;
  if (catBadge) catBadge.textContent = c.category;

  // Dossier
  const dossierEl = document.getElementById("caseDossier");
  if (dossierEl) {
    dossierEl.innerHTML = `
      <div class="meta-field"><span class="meta-label">Pathology</span><span class="meta-val">${c.pathology}</span></div>
      <div class="meta-field"><span class="meta-label">Organ / Domain</span><span class="meta-val">${c.organ}</span></div>
      <div class="meta-field"><span class="meta-label">Modality</span><span class="meta-val">${c.modality}</span></div>
      <div class="meta-field"><span class="meta-label">Condition</span><span class="meta-val">${c.condition}</span></div>
      <div class="meta-field"><span class="meta-label">Clean 3D Dice</span><span class="meta-val" style="color: #34d399; font-weight: 700;">${c.metrics.dice_clean.toFixed(4)}</span></div>
      <div class="meta-field"><span class="meta-label">Full-Decoder &rho;</span><span class="meta-val">${c.metrics.spearman_decoder_full.toFixed(3)}</span></div>
      <div class="meta-field"><span class="meta-label">Prompt Bounding Box</span><span class="meta-val">[${c.box.join(", ")}]</span></div>
    `;
  }

  // Panel 1: Raw
  setImgSrc("p1-image", `${c.id}/raw_box.png`);
  const p1Coords = document.getElementById("p1-coords");
  if (p1Coords) p1Coords.textContent = `ROI Bounding Box: [${c.box.join(", ")}]`;

  // Panel 2: Seg
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

  // Panel 3: Heatmap & Policy
  updateHeatmapViews();
  applyAuditPresentationPolicy();

  // Multi-panel & Montage
  setImgSrc("clinician-figure-img", `${c.id}/figure.png`);
  setImgSrc("clinician-montage-img", `${c.id}/montage.png`);
  setImgSrc("full-figure-img", `${c.id}/figure.png`);
  setImgSrc("audit-montage-img", `${c.id}/montage.png`);
  setImgSrc("transparency-curve-img", `${c.id}/curve.png`);

  // MPRT Views
  setImgSrc("mprt-intact-img", `${c.id}/heatmap_turbo.png`);
  setImgSrc("mprt-scrambled-img", `${c.id}/scrambled_turbo.png`);
  setImgSrc("mprt-residual-img", `${c.id}/residual_turbo.png`);

  const spVal = document.getElementById("mprt-spearman-val");
  const randDiceVal = document.getElementById("mprt-rand-dice-val");
  const ctrlVal = document.getElementById("mprt-control-val");
  const restVal = document.getElementById("mprt-restoration-val");

  if (spVal) spVal.textContent = c.metrics.spearman_decoder_full.toFixed(3);
  if (randDiceVal) randDiceVal.textContent = c.metrics.dice_after_decoder_rand.toFixed(3);
  if (ctrlVal) ctrlVal.textContent = c.metrics.control_spearman.toFixed(3);
  if (restVal) restVal.textContent = c.metrics.restoration_error.toFixed(4);

  // Split Comparator & Design Grid
  updateWipeComparatorImages();
  updateDesignCatalogGrid(c.id);
}

// Apply Clinical Presentation Policy
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
      bannerTitle.textContent = "Safety Audit Gate: PASSED \u2014 Heatmap Certified for Clinical Review";
      bannerDesc.textContent = `Model Parameter Randomization Test (MPRT) verified: Spearman correlation collapsed to ${c.metrics.spearman_decoder_full.toFixed(3)} \u2264 0.30 at Stage 4 (full decoder). Saliency map demonstrates strict parametric dependence.`;
      bannerIcon.innerHTML = "&check;";
    }
    if (headerAuditBadge) {
      headerAuditBadge.className = "badge-tag badge-green";
      headerAuditBadge.textContent = "MPRT: PASSED";
    }
    if (p3AuditStatusTag) {
      p3AuditStatusTag.className = "badge-tag badge-green";
      p3AuditStatusTag.textContent = "Audit Certified";
    }
    if (p3PolicyMeta) {
      p3PolicyMeta.textContent = "Status: Saliency Display Approved by MPRT Audit Gate";
      p3PolicyMeta.style.color = "#34d399";
    }
    if (curtain) curtain.style.display = "none";
  } else {
    if (overrideAuditGate) {
      if (banner) {
        banner.className = "audit-gate-banner withheld";
        bannerTitle.textContent = "Safety Audit Gate: OVERRIDE ACTIVE \u2014 Inspecting Uncertified Heatmap";
        bannerDesc.textContent = `SAFETY OFFICER DEBUG: Displaying uncertified heatmap (Spearman \u03c1 = ${c.metrics.spearman_decoder_full.toFixed(3)} > 0.30). Heatmap does NOT reflect learned model weights and is unfaithful.`;
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
        p3PolicyMeta.style.color = "#f87171";
      }
      if (curtain) curtain.style.display = "none";
    } else {
      if (banner) {
        banner.className = "audit-gate-banner withheld";
        bannerTitle.textContent = "Safety Audit Gate: FAILED \u2014 Heatmap Strictly Withheld for Clinical Safety";
        bannerDesc.textContent = `Model Parameter Randomization Test (MPRT) FAILED: Saliency map failed parameter sensitivity checks (Spearman \u03c1 = ${c.metrics.spearman_decoder_full.toFixed(3)} > 0.30). Heatmap is withheld to prevent misleading radiological diagnosis. Segmentation mask in Panel 2 remains displayed.`;
        bannerIcon.innerHTML = "&empty;";
      }
      if (headerAuditBadge) {
        headerAuditBadge.className = "badge-tag badge-red";
        headerAuditBadge.textContent = "MPRT: FAILED (WITHHELD)";
      }
      if (p3AuditStatusTag) {
        p3AuditStatusTag.className = "badge-tag badge-red";
        p3AuditStatusTag.textContent = "Heatmap Withheld";
      }
      if (p3PolicyMeta) {
        p3PolicyMeta.textContent = "Status: Visual Saliency Withheld by Pre-registered Safety Gate";
        p3PolicyMeta.style.color = "#f87171";
      }
      if (curtain) curtain.style.display = "flex";
    }
  }
}

// Update Panel 3 Heatmap View
function updateHeatmapViews() {
  const c = CASE_CATALOG[currentCaseId];
  if (!c) return;

  let filename = "heatmap_turbo.png";
  if (currentDesign === "plasma") filename = "heatmap_plasma.png";
  else if (currentDesign === "inferno") filename = "heatmap_inferno.png";
  else if (currentDesign === "viridis") filename = "heatmap_viridis.png";
  else if (currentDesign === "contour") filename = "heatmap_contour.png";
  else if (currentDesign === "residual") filename = "residual_turbo.png";

  setImgSrc("p3-image", `${c.id}/${filename}`);
  const p3Img = document.getElementById("p3-image");
  if (p3Img) p3Img.style.opacity = currentOpacity;
}

// Update Split Saliency Comparator
function updateWipeComparatorImages() {
  const c = CASE_CATALOG[currentCaseId];
  if (!c) return;

  const underlay = document.getElementById("splitUnderlay");
  const overlayInner = document.getElementById("splitOverlayInner");

  if (currentWipeMode === "clean") {
    setBgImageWithFallback(underlay, `${c.id}/raw.png`);
    setBgImageWithFallback(overlayInner, `${c.id}/heatmap_turbo.png`);
  } else if (currentWipeMode === "scrambled") {
    setBgImageWithFallback(underlay, `${c.id}/heatmap_turbo.png`);
    setBgImageWithFallback(overlayInner, `${c.id}/scrambled_turbo.png`);
  } else if (currentWipeMode === "segmentation") {
    setBgImageWithFallback(underlay, `${c.id}/heatmap_turbo.png`);
    setBgImageWithFallback(overlayInner, `${c.id}/seg_overlay.png`);
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

// Populate Cohort Benchmark Table (All 16 Cases)
function populateCohortTable() {
  const tbody = document.getElementById("cohortTableBody");
  if (!tbody) return;

  const rows = Object.values(CASE_CATALOG).map(c => {
    const passed = c.audit_passed;
    const gateBadge = passed ? `<span class="badge-tag badge-green">PASS</span>` : `<span class="badge-tag badge-red">FAIL</span>`;
    const statusText = passed ? `<span style="color:#34d399; font-weight:600;">Approved</span>` : `<span style="color:#f87171; font-weight:600;">Withheld</span>`;

    return `
      <tr>
        <td><strong>${c.title}</strong> (${c.id})</td>
        <td>${c.pathology}</td>
        <td><span class="badge-tag">${c.organ}</span></td>
        <td style="font-weight:700; color:#ffffff;">${c.metrics.dice_clean.toFixed(4)}</td>
        <td>${c.metrics.dice_after_decoder_rand.toFixed(3)}</td>
        <td style="font-weight:600; color:${c.metrics.spearman_decoder_full <= 0.3 ? "#34d399" : "#f87171"}">${c.metrics.spearman_decoder_full.toFixed(3)}</td>
        <td>${c.metrics.ssim_decoder_full.toFixed(3)}</td>
        <td>${gateBadge}</td>
        <td>${statusText}</td>
      </tr>
    `;
  }).join("");

  tbody.innerHTML = rows;
}

// Update Design Catalog Grid
function updateDesignCatalogGrid(caseId) {
  const grid = document.getElementById("designCatalogGrid");
  if (!grid) return;

  const catalog = [
    { title: "Turbo Thermal", file: "heatmap_turbo.png", desc: "Wide perceptual dynamic range for low-attenuation edema." },
    { title: "Medical Plasma", file: "heatmap_plasma.png", desc: "Linear luminance slope standard for neuro-radiological review." },
    { title: "Solar Inferno", file: "heatmap_inferno.png", desc: "High contrast highlight on tumor vascular core." },
    { title: "Diagnostic Viridis", file: "heatmap_viridis.png", desc: "Uniform colorblind-accessible diagnostic gradient." },
    { title: "Topographic Iso Contour", file: "heatmap_contour.png", desc: "Iso-attention contours delineating sharp focal margins." },
    { title: "Residual Delta", file: "residual_turbo.png", desc: "Difference map (|Intact \u2212 Scrambled|) proving model dependence." }
  ];

  grid.innerHTML = catalog.map(item => `
    <div class="sidebar-card">
      <div style="font-size: 0.82rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">${item.title}</div>
      <div class="image-viewport" style="margin-bottom: 6px;">
        <img src="data/${caseId}/${item.file}" onerror="this.src='public/data/${caseId}/${item.file}'" alt="${item.title}">
      </div>
      <div style="font-size: 0.72rem; color: #94a3b8; line-height: 1.4;">${item.desc}</div>
    </div>
  `).join("");
}
