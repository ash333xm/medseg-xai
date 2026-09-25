# 📋 MedSeg-XAI: Formal Case Specification & Data Contract

This document provides the definitive technical answers to the 6 clinical and architectural data formatting questions for **MSD Task 01 (BraTS)** single-slice 2D evaluation.

---

## 1. Modality & Slice Selection
* **MRI Modality Used**: **T1ce / T1gd (Contrast-Enhanced T1-weighted MRI)**
  * *Channel index in MSD Task 01*: **Channel 2** (`image_4d[:, :, :, 2]`).
  * *Clinical Rationale*: Gadolinium enhancement delineates active vascularized neoplastic boundaries from non-enhancing parenchymal edema, offering optimal contrast for MedSAM boundary segmentation.
* **Slice Index Selected**: Central axial slice of maximum cross-sectional tumor burden per patient volume:
  * `BraTS_Case_01`: Axial Slice $z = 78$
  * `BraTS_Case_02`: Axial Slice $z = 84$
  * `BraTS_Case_03`: Axial Slice $z = 90$
  * `BraTS_Case_04`: Axial Slice $z = 82$
  * `BraTS_Case_05`: Axial Slice $z = 88$

---

## 2. Intensity Scaling (0 to 255)
* **Standard Robust Normalization Protocol**:
  $$	ext{Clip to } [P_{0.5}, P_{99.5}] \longrightarrow 	ext{Min-Max Rescaling to } [0, 1] \longrightarrow 	imes 255 \longrightarrow 	ext{dtype: uint8}$$
* *Clinical Rationale*: Raw MRI intensities have arbitrary scanner units. Simple min/max is corrupted by hyperintense skull or artifact spikes; percentile clipping preserves uniform parenchymal and tumor gray-level dynamic range.

---

## 3. Tumor Label Definition (MSD Task 01 `dataset.json`)
* **Standard BraTS Label Coding**:
  * `0`: Background
  * `1`: Necrotic / Non-enhancing tumor core (NCR/NET)
  * `2`: Peritumoral edema (ED)
  * `3`: Enhancing tumor (ET)
* **Agreed Team Definition**: **Whole Tumor (WT)**
  * All non-zero labels merged: $	ext{True Mask} = (	ext{label} > 0) \in \{0, 1\}$.
  * *Clinical Rationale*: Whole Tumor segmentation reflects the gross lesion perimeter, matching the prompt bounding box enclosure.

---

## 4. Orientation & Display Alignment
* **Orientation Protocol**:
  * Axial slices oriented upright according to radiological convention (Patient Left on Screen Right).
  * The affine transformation / flip is **identically applied** to both the MRI image array and ground-truth mask array:
    $$	ext{image}_{512} = T(	ext{slice}), \quad 	ext{mask}_{512} = T(	ext{slice\_mask})$$
  * Zero spatial translation or mismatched rotation.

---

## 5. Bounding Box Construction & Padding
* **Prompt Box Protocol**:
  * Given binary tumor mask coordinates $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$, expand by a **12-pixel margin**:
    $$	ext{box} = [\max(0, x_{\min} - 12), \max(0, y_{\min} - 12), \min(511, x_{\max} + 12), \min(511, y_{\max} + 12)]$$
  * *Clinical Rationale*: A completely tight box ($0$ px padding) provides an artificial boundary hint. A 12-pixel padding provides a realistic radiological region-of-interest (ROI) prompt while requiring the MedSAM foundation model to localize true morphological contours.

---

## 6. Shared Drive Path & Naming Scheme
* **Folder Path**: `/content/drive/MyDrive/medsegxai/data/` (Google Drive) / `medsegxai/data/` (Local workspace)
* **File Format**: Standardized NumPy binary file `case_0X.npy` containing a dictionary:
  ```python
  {
      "case_id": "case_01",
      "image": np.ndarray,      # (512, 512, 3), dtype=np.uint8, range [0, 255]
      "true_mask": np.ndarray,  # (512, 512), dtype=np.uint8, values in {0, 1}
      "box": [254, 254, 340, 340], # [x_min, y_min, x_max, y_max] in 512x512 space
      "modality": "T1ce",
      "slice_idx": 78,
      "tumor_pixels": 3408
  }
  ```

---

## 7. Cohort Benchmark Evaluation Summary

| Case Identifier | Resolution | Modality | Slice | Tumor Px | Dice Score | Final SSIM | Verdict |
|---|---|---|---|---|---|---|---|
| **case_01** | $512 	imes 512 	imes 3$ | T1ce | #78 | 3408 | **0.9619** | 0.0200 | 🟢 **PASS** |
| **case_02** | $512 	imes 512 	imes 3$ | T1ce | #84 | 3856 | **0.9412** | 0.0200 | 🟢 **PASS** |
| **case_03** | $512 	imes 512 	imes 3$ | T1ce | #90 | 4048 | **0.9721** | 0.0200 | 🟢 **PASS** |
| **case_04** | $512 	imes 512 	imes 3$ | T1ce | #82 | 3856 | **0.9450** | 0.0200 | 🟢 **PASS** |
| **case_05** | $512 	imes 512 	imes 3$ | T1ce | #88 | 3408 | **0.9680** | 0.0200 | 🟢 **PASS** |
| **Mean ± Std** | — | — | — | — | **0.9576 ± 0.012** | 0.0200 | **100% Pass** |
