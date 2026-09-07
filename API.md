# MedSeg-XAI: Member 1 (M1) Architecture & API Documentation

## 1. System Overview & Role Context

As the **Model Architecture & Inference Lead (M1)** for the MedSeg-XAI platform, this subsystem is responsible for:
- Containerized foundation model deployment on **RunPod Cloud GPUs** (e.g. NVIDIA A100 / RTX 4090).
- Zero-copy, memory-mapped loading (`mmap=True`) of **MedSAM-2** with **Hiera-Large** hierarchical vision transformer backbone.
- 3D-to-2D spatial resampling and dynamic pseudo-video batching formatted strictly as $(B \times T \times C \times H \times W)$.
- Multi-slice 3D memory attention propagation with zero-shot point/box prompt conditioning.
- Non-intrusive isolation and guidance of deep Hiera-Large transformer blocks (Stage 3 & Stage 4) for explainability (TMME / Rollout).
- Deterministic dual-pass VRAM lifecycle management for Live MPRT safety auditing:
  $$\text{Clean Pass} \implies \text{CUDA Cache Clear} \implies \text{Randomized Pass}$$

---

## 2. RunPod Infrastructure & Persistent Volumes

### Volume Layout
On RunPod, model weights and volumetric datasets are mounted to persistent network storage to prevent cold-start re-downloads across container restarts:
```
/runpod-volume/
├── weights/
│   └── medsam2_hiera_large.pt       # Foundation checkpoint (~2.4 GB)
├── data/
│   ├── brats2023/                   # BraTS MRI volumes (.nii.gz)
│   ├── btcv/                        # BTCV Abdominal CT volumes
│   └── deeplesion/                  # DeepLesion CT volumes
└── outputs/                         # Exported masks, logs, DICOM artifacts
```

### Environment Variables
| Variable | Default Value | Description |
|---|---|---|
| `RUNPOD_VOLUME_PATH` | `/runpod-volume` | Root path to network storage volume |
| `MODEL_WEIGHTS_PATH` | `/runpod-volume/weights/medsam2_hiera_large.pt` | Path to MedSAM-2 Hiera-Large weights |
| `DATASET_PATH` | `/runpod-volume/data` | Path to benchmark datasets |
| `PORT` | `8000` | HTTP port exposed by FastAPI |
| `HOST` | `0.0.0.0` | Binding host address |
| `PRECISION` | `float16` | Precision (`float16`, `bfloat16`, or `float32`) |
| `FORCE_CPU` | `0` | Set `1` to force CPU mode in non-GPU environments |

---

## 3. REST API Reference (Port 8000)

### `GET /health`
Returns system status, GPU device information, VRAM telemetry, and checkpoint presence.

#### Response Example
```json
{
  "status": "healthy",
  "cuda_available": true,
  "device": "NVIDIA A100-SXM4-80GB",
  "precision": "float16",
  "weights_loaded_path": "/runpod-volume/weights/medsam2_hiera_large.pt",
  "weights_file_exists": true,
  "vram": {
    "allocated_mb": 2410.5,
    "reserved_mb": 3120.0,
    "peak_mb": 2680.2
  }
}
```

---

### `POST /predict/slice`
Executes single-slice zero-shot prompt inference and extracts deep attention maps.

#### Request Body
```json
{
  "image_shape": [3, 1024, 1024],
  "points": [
    {"coords": [512.0, 512.0], "label": 1},
    {"coords": [400.0, 400.0], "label": 0}
  ],
  "bounding_box": [350.0, 350.0, 650.0, 650.0],
  "extract_attention": true
}
```

#### Response Body
```json
{
  "iou_score": 0.942,
  "mask_shape": [1024, 1024],
  "positive_voxel_count": 48210,
  "deep_attention_layers_extracted": [
    "stage_3.block_34",
    "stage_3.block_35",
    "stage_4.block_2",
    "stage_4.block_3"
  ],
  "inference_time_ms": 34.2
}
```

---

### `POST /predict/volume`
Propagates segmentation across a 3D medical volume formatted as sequential pseudo-video batches $(B \times T \times C \times H \times W)$ using SAM 2 memory cross-attention.

#### Request Body
```json
{
  "num_slices": 64,
  "slice_height": 1024,
  "slice_width": 1024,
  "prompt_slice_idx": 32,
  "bounding_box": [300.0, 300.0, 700.0, 700.0],
  "modality": "ct",
  "plane": "axial"
}
```

#### Response Body
```json
{
  "volume_shape": [64, 1024, 1024],
  "prompt_slice_idx": 32,
  "mean_slice_iou": 0.912,
  "total_segmented_voxels": 1420890,
  "inference_time_ms": 520.4
}
```

---

### `POST /audit/dual_pass`
Executes the Live MPRT (Model Parameter Randomization Test) dual-pass safety verification:
1. **Clean Pass**: Evaluates slice and extracts Clean Heatmap A.
2. **CUDA Cache Clear**: Purges GPU cache and intermediate allocations.
3. **Randomized Pass**: Kaiming Normal resets top-to-bottom layers $\theta_l \sim \mathcal{N}(0, \sqrt{2 / n_l})$ and computes Corrupted Heatmap B.
4. **Safety Gating**:
   $$\text{SSIM}(A, B) < 0.30 \implies \text{PASS (Trustworthy)}$$
   $$\text{SSIM}(A, B) \ge 0.30 \implies \text{REJECT (Superficial Edge Detector)}$$
5. **State Restoration**: Restores pristine weights and returns telemetry.

#### Request Body
```json
{
  "slice_shape": [3, 512, 512],
  "bounding_box": [100.0, 100.0, 400.0, 400.0]
}
```

#### Response Body
```json
{
  "ssim_score": 0.082,
  "gating_status": "PASS",
  "trust_score": 0.9126,
  "clean_vram_allocated_mb": 2410.5,
  "inter_pass_vram_allocated_mb": 420.0,
  "randomized_vram_allocated_mb": 2410.5
}
```

---

## 4. Cross-Functional Team Interfaces

### Interface with Member 2 (M2 - Explainability Lead)
- **Deep Attention Layers**: `model.get_deep_attention_layers()` returns Stage 3 and Stage 4 multi-head attention modules.
- **Hook Integration**: Modules expose `.last_attn_weights` and allow standard forward hook registration without graph disruption (`retain_graph=True`).
- **Data Flow**: Extracted tensors feed directly into M2's `xai_hooks.py` and `tmme_engine.py`.

### Interface with Member 3 (M3 - Safety Auditing Lead)
- **Layer Randomization Hook**: `vram_manager.kaiming_randomize_layers()` exposes clean layer resetting.
- **State Dict Snapshotting**: `vram_manager` captures pre-audit weight backups and guarantees bitwise weight restoration post-audit.
- **SSIM Metric**: Mathematical calculation matching Capstone formulation.

### Interface with Member 4 (M4 - Full-Stack & Data Lead)
- **Data Resampling**: `SpatialResampler3D` ingests NIfTI (`.nii`/`.nii.gz`) and DICOM formats, applies CT HU windowing or MRI Z-score normalization, and formats tensors to $(B \times T \times C \times H \times W)$.
- **Mask Un-resampling**: `inverse_resample_mask()` resamples predicted masks back into original patient coordinates for DICOM/PACS export.
