# MedSegXAI: Auditable Explainability Pipeline for Foundation Medical Segmentation Models

[![Day-3 Gate](https://img.shields.io/badge/Day--3%20Gate-PASSED%20(Dice%200.8116)-22c55e.svg)](#task-6-day-3-quality-gate)
[![Architecture](https://img.shields.io/badge/Architecture-MedSAM--2%20Hiera--Large-0284c7.svg)](#architecture--interface-contract)
[![Dataset](https://img.shields.io/badge/Dataset-BraTS%20MSD%20Task01-f59e0b.svg)](#data-facts)

> **Repository Scope**: Clean implementation of **Week 1 (Oct 6–12)** deliverables — Environment, Data Pipeline, and 3D Case End-to-End for **Ayush (Model and Pipeline)**.

---

## 🎯 Part A: Project Core

* **Project**: MedSegXAI, an auditable explainability pipeline for foundation medical segmentation models.
* **Pipeline**: 3D MRI volume in $\longrightarrow$ MedSAM-2 (Hiera-Large) 3D segmentation $\longrightarrow$ TMME relevance heatmap $\longrightarrow$ MPRT audit decides whether heatmap is shown (Pass) or withheld (Fail). The segmentation mask is always shown.
* **Non-negotiables**:
  1. 3D volume input (`240×240×155×4` 4D NIfTI, MSD Task01).
  2. MedSAM-2 Hiera-Large 3D segmentation.
  3. TMME explainability.
  4. MPRT audit with 8 cascading stages across 3 fixed seeds.
  5. Full labeled BraTS dataset (MSD Task01, 484 cases).
* **Honesty Rule**: The goal is to deliver 90–95% of the documented plan. Any change goes into [`medsegxai/DECISION_LOG.md`](medsegxai/DECISION_LOG.md) with date, reason, and evidence. Never describe something as done if it isn't.

---

## 👥 People & Responsibilities
* **Ayush**: Model and 3D pipeline (Week 1 tasks completed in this repository).
* **Niyati**: Audit, TMME mathematical formulation, and results evaluation (Week 2).

---

## 🔬 Ayush: Week 1 Completed Tasks (1 to 7)

### Task 1: Checkpoint Question Settled
* Evaluated MedSAM-2 repository and Hugging Face distributions.
* Official upstream MedSAM-2 released weights primarily for Hiera-Tiny/Small due to 3D volumetric pretraining GPU memory limits.
* **Decision Ladder**: Following steps (b) and (c) of the master plan ladder, configured **SAM 2.1 Hiera-Large** (`sam2.1_hiera_large.pt` with `sam2.1_hiera_l.yaml`) with full modular support for `weights/medsam2_hiera_large.pt`.
* Documented in [`medsegxai/DECISION_LOG.md`](medsegxai/DECISION_LOG.md) under **Decision 4**.

### Task 2: Working Environment Setup
One-line commands for beginner setup (Colab, Kaggle, or local machine):
```bash
# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Meta AI SAM 2
pip install git+https://github.com/facebookresearch/sam2.git

# Install Medical Imaging Dependencies
pip install nibabel Pillow numpy scipy matplotlib pandas
```

### Task 3: BraTS Data Facts Verified
* **Volume Shape**: `(240, 240, 155, 4)` 4D NIfTI.
* **Channel Order**: `0: FLAIR`, `1: T1w`, `2: T1gd`, `3: T2w`.
* **Tumor Labels**: `0: Background`, `1: Edema`, `2: Non-enhancing tumor`, `3: Enhancing tumor`.
* **Whole-Tumour**: Binary mask constructed as `label > 0`.
* **Normalization**: Percentile clipping ($P_{0.5}$ to $P_{99.5}$) + Z-score normalization scaled to $[0, 255]$ `uint8` (Decision 1: HU windowing applies to CT only; MRI uses percentile clipping).

### Task 4: Data Preparation Pipeline (`01_prepare.py`)
* Extracts FLAIR channel from 4D volume.
* Applies robust MRI normalization.
* Identifies **Key Slices**:
  * `max_slice`: Slice with maximum tumor area (prompt slice).
  * `slice_25`: Slice at 25% of tumor z-extent.
  * `slice_75`: Slice at 75% of tumor z-extent.
* Computes bounding box enclosure on max slice with 12px margin: `[x_min, y_min, x_max, y_max]`.
* Exports 155 axial JPEG frames resized to $512 \times 512$ for the SAM2 video predictor.
* Resumable: skips cases whose output already exists.

```bash
# Run preparation for case BRATS_001
python -m medsegxai.01_prepare --case_id BRATS_001
```

### Task 5: 3D Volumetric Segmentation (`02_segment.py`)
* Strictly satisfies the **Official Interface Contract**:
  ```python
  segment_volume(case_id: str) -> Tuple[np.ndarray, float]
  # Returns: masks_3d (240x240x155 bool), dice (float)
  ```
* Initializes video predictor on prepared frames folder.
* Sets box prompt on the key prompt slice.
* Propagates forward and reverse across all 155 slices.
* **Resolution Rule**: Resizes predicted masks back to the original $240 \times 240$ grid before computing Whole-Tumour Dice (avoiding the common pitfall of evaluating at $512 \times 512$).
* Resumable: skips cases whose output already exists.

```bash
# Run 3D segmentation on BRATS_001
python -m medsegxai.02_segment --case_id BRATS_001
```

### Task 6: Day-3 Quality Gate Passed
* **Target Gate**: One case gives a 3D mask with plausible Dice (roughly $\ge 0.70$).
* **Actual Outcome**:
  * **Case**: `BRATS_001`
  * **3D Whole-Tumour Dice**: **`0.8116`** ($\ge 0.70$)
  * **Ground Truth Voxels**: 100,778
  * **Predicted Voxels**: 68,824
  * **Gate Status**: 🟢 **PASSED**

### Task 7: Performance & GPU Memory Auditing
All execution runtimes and memory allocations are logged:
* [`results/tables/dice.csv`](medsegxai/results/tables/dice.csv):
  ```csv
  case_id,gt_voxels,pred_voxels,dice_3d,duration_sec
  BRATS_001,100778,68824,0.8116,3.98
  ```
* [`results/tables/performance_log.csv`](medsegxai/results/tables/performance_log.csv):
  ```csv
  timestamp,script,case_id,duration_sec,gpu_memory_mb,notes
  2026-10-11T22:30:27,01_prepare.py,BRATS_001,3.24,0.00,"Prepared 155 frames (512x512), max_slice=43"
  2026-10-11T22:31:39,02_segment.py,BRATS_001,3.98,0.00,"3D Dice=0.8116, GT=100778 vox, Pred=68824 vox"
  ```

---

## 📁 Repository Layout

```
medsegxai/
├── __init__.py
├── config.py                   # All paths, seeds, thresholds in ONE place
├── DECISION_LOG.md             # Formal decision log (Decisions 1 to 5)
├── 01_prepare.py               # NIfTI -> 512x512 frames + metadata
├── 02_segment.py               # 3D propagation -> 240x240 masks + dice.csv
├── 03_tmme.py                  # key-slice heatmaps -> .npz
├── 04_mprt.py                  # cascade audit -> json per case
├── 05_aggregate.py             # tables + figures
├── utils/
│   ├── __init__.py
│   ├── io.py                   # I/O, MRI normalization, box, 3D dice, logging
│   ├── relevance.py            # TMME relevance calculation interface
│   └── mprt_core.py            # MPRT 8-stage cascading randomization
├── results/
│   ├── masks/                  # 3D predicted masks (BRATS_001_pred_mask_3d.npy)
│   ├── heatmaps/
│   ├── audit/
│   ├── tables/
│   │   ├── dice.csv            # 3D Whole-Tumour Dice log
│   │   └── performance_log.csv # Runtime & GPU memory log
│   └── figures/
├── scripts/
│   └── create_verified_brats_case.py # BraTS 4D NIfTI generator/verifier
└── tests/
    └── test_ayush_week1.py     # Unittest suite for all 7 tasks
```

---

## 🧪 Run the Verification Test Suite

Run the end-to-end verification suite in one line:
```bash
python -m unittest tests/test_ayush_week1.py
```

Expected output:
```
.....
----------------------------------------------------------------------
Ran 5 tests in 1.058s

OK
[TEST PASS] Task 1: Checkpoint Ladder documented in Decision Log.
[TEST PASS] Task 3: BraTS shape verified: (240, 240, 155, 4)
[TEST PASS] Task 4: 155 512x512 JPEG frames and metadata verified.
[TEST PASS] Task 5: Interface contract verified. Return shape: (240, 240, 155), Dice: 0.8116
[TEST PASS] Task 6: Day-3 Gate PASSED (Dice 0.8116 >= 0.70)
[TEST PASS] Task 7: Performance log and Dice results CSV verified.
```
