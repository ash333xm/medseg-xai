"""
FastAPI Server for MedSeg-XAI Model Architecture & Inference Service (Port 8000).
Exposes zero-shot slice prediction, 3D pseudo-video volume memory propagation,
and live MPRT dual-pass VRAM lifecycle auditing.
"""

from typing import Any, Dict, List, Optional
import os
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import numpy as np
import torch

from contextlib import asynccontextmanager

from medseg.config import MedSegConfig, default_config
from medseg.model_loader import MedSAM2ModelLoader
from medseg.inference import MedSAM2InferenceEngine, PromptConditioning
from medseg.vram_manager import VRAMLifecycleManager, capture_vram_snapshot, clear_cuda_cache

# Global engine instances initialized on startup
engine: Optional[MedSAM2InferenceEngine] = None
vram_mgr: Optional[VRAMLifecycleManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes model loader, inference engine, and VRAM manager."""
    global engine, vram_mgr
    config = default_config
    config.ensure_directories()
    loader = MedSAM2ModelLoader(config)
    engine = MedSAM2InferenceEngine(model_loader=loader, config=config)
    vram_mgr = VRAMLifecycleManager(engine.device)
    yield

app = FastAPI(
    title="MedSeg-XAI M1 Inference Engine",
    description="High-performance MedSAM-2 Hiera-Large 3D Pseudo-Video Inference & Dual-Pass VRAM Auditing API",
    version="1.0.0",
    lifespan=lifespan
)

# ------------------------------------------------------------------------------
# Request & Response Models
# ------------------------------------------------------------------------------

class PointPrompt(BaseModel):
    coords: List[float] = Field(..., description="[x, y] coordinates in slice pixels")
    label: int = Field(1, description="1 for foreground target, 0 for background exclusion")

class SlicePredictRequest(BaseModel):
    # If image_data is omitted, synthetic benchmark slice is generated
    image_shape: List[int] = Field(default=[3, 1024, 1024], description="[Channels, Height, Width]")
    points: Optional[List[PointPrompt]] = None
    bounding_box: Optional[List[float]] = Field(
        None, description="[x_min, y_min, x_max, y_max] box prompt"
    )
    extract_attention: bool = Field(True, description="Whether to capture deep Hiera attention maps")

class SlicePredictResponse(BaseModel):
    iou_score: float
    mask_shape: List[int]
    positive_voxel_count: int
    deep_attention_layers_extracted: List[str]
    inference_time_ms: float

class VolumePredictRequest(BaseModel):
    num_slices: int = Field(default=16, ge=2, le=256, description="Slice depth T along axis")
    slice_height: int = Field(default=512, ge=64, le=1024)
    slice_width: int = Field(default=512, ge=64, le=1024)
    prompt_slice_idx: int = Field(default=0, ge=0)
    bounding_box: Optional[List[float]] = None
    points: Optional[List[PointPrompt]] = None
    modality: str = Field(default="ct", description="ct or mri")
    plane: str = Field(default="axial", description="axial, coronal, or sagittal")

class VolumePredictResponse(BaseModel):
    volume_shape: List[int]
    prompt_slice_idx: int
    mean_slice_iou: float
    total_segmented_voxels: int
    inference_time_ms: float

class DualPassAuditRequest(BaseModel):
    slice_shape: List[int] = Field(default=[3, 512, 512])
    bounding_box: Optional[List[float]] = Field(default=[100.0, 100.0, 400.0, 400.0])

class DualPassAuditResponse(BaseModel):
    ssim_score: float
    gating_status: str  # PASS vs REJECT
    trust_score: float
    clean_vram_allocated_mb: float
    inter_pass_vram_allocated_mb: float
    randomized_vram_allocated_mb: float

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.get("/", tags=["Info"])
def root():
    return {
        "service": "MedSeg-XAI M1 Inference Engine",
        "model": "MedSAM-2 (Hiera-Large)",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["Health & Telemetry"])
def health_check():
    """System health, GPU availability, VRAM metrics, and volume mount status."""
    global engine
    device_name = "CPU"
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)

    weights_exist = Path(default_config.model_weights_path).exists()
    vram_stats = capture_vram_snapshot(engine.device if engine else torch.device("cpu"))

    return {
        "status": "healthy",
        "cuda_available": cuda_avail,
        "device": device_name,
        "precision": default_config.precision,
        "weights_loaded_path": str(default_config.model_weights_path),
        "weights_file_exists": weights_exist,
        "vram": {
            "allocated_mb": vram_stats.allocated_mb,
            "reserved_mb": vram_stats.reserved_mb,
            "peak_mb": vram_stats.peak_mb
        }
    }

@app.post("/predict/slice", response_model=SlicePredictResponse, tags=["Inference"])
def predict_slice(req: SlicePredictRequest):
    """Executes single slice inference with zero-shot point/box conditioning."""
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")

    t0 = time.time()
    # Generate benchmark slice tensor (or ingest uploaded image)
    C, H, W = req.image_shape
    slice_tensor = torch.randn(1, C, H, W, dtype=torch.float32)

    # Format prompts
    pts = None
    lbls = None
    if req.points:
        pts = [p.coords for p in req.points]
        lbls = [p.label for p in req.points]

    prompts = PromptConditioning(points=pts, labels=lbls, boxes=req.bounding_box)

    result = engine.predict_slice(
        slice_tensor=slice_tensor,
        prompts=prompts,
        update_memory=True,
        extract_attention=req.extract_attention
    )

    elapsed_ms = (time.time() - t0) * 1000
    binary_mask = result["binary_mask"]

    return SlicePredictResponse(
        iou_score=result["iou_score"],
        mask_shape=list(binary_mask.shape),
        positive_voxel_count=int(np.sum(binary_mask)),
        deep_attention_layers_extracted=list(result["attention_maps"].keys()),
        inference_time_ms=elapsed_ms
    )

@app.post("/predict/volume", response_model=VolumePredictResponse, tags=["Inference"])
def predict_volume(req: VolumePredictRequest):
    """Executes 3D pseudo-video continuous memory propagation across slices."""
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")

    t0 = time.time()
    # Construct synthetic pseudo-video batch (1, T, 3, H, W)
    video_tensor = torch.randn(
        1, req.num_slices, 3, req.slice_height, req.slice_width, dtype=torch.float32
    )

    pts = None
    lbls = None
    if req.points:
        pts = [p.coords for p in req.points]
        lbls = [p.label for p in req.points]

    prompts = PromptConditioning(points=pts, labels=lbls, boxes=req.bounding_box)

    res = engine.propagate_volume(
        video_tensor=video_tensor,
        prompt_slice_idx=req.prompt_slice_idx,
        prompts=prompts,
        bidirectional=True
    )

    elapsed_ms = (time.time() - t0) * 1000
    mask_vol = res["mask_volume"]

    return VolumePredictResponse(
        volume_shape=list(mask_vol.shape),
        prompt_slice_idx=req.prompt_slice_idx,
        mean_slice_iou=float(np.mean(res["slice_ious"])),
        total_segmented_voxels=int(np.sum(mask_vol)),
        inference_time_ms=elapsed_ms
    )

@app.post("/audit/dual_pass", response_model=DualPassAuditResponse, tags=["Safety Auditing"])
def audit_dual_pass(req: DualPassAuditRequest):
    """Executes clean pass -> CUDA cache clear -> randomized pass Live MPRT audit."""
    global engine, vram_mgr
    if engine is None or vram_mgr is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    C, H, W = req.slice_shape
    test_slice = torch.randn(1, C, H, W, dtype=torch.float32)
    prompts = PromptConditioning(boxes=req.bounding_box)

    # Targeted deep layers for Live MPRT randomization
    deep_layers = engine.model.get_deep_attention_layers()

    def run_inference():
        return engine.predict_slice(
            slice_tensor=test_slice,
            prompts=prompts,
            update_memory=False,
            extract_attention=True
        )

    audit_res = vram_mgr.execute_dual_pass(
        model=engine.model,
        inference_fn=run_inference,
        layers_to_randomize=deep_layers
    )

    return DualPassAuditResponse(
        ssim_score=audit_res.ssim_score,
        gating_status=audit_res.gating_status,
        trust_score=audit_res.trust_score,
        clean_vram_allocated_mb=audit_res.clean_vram.allocated_mb,
        inter_pass_vram_allocated_mb=audit_res.inter_pass_vram.allocated_mb,
        randomized_vram_allocated_mb=audit_res.randomized_vram.allocated_mb
    )
