"""
Integration tests for FastAPI inference and dual-pass safety endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.server import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_api_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "MedSeg-XAI" in data["service"]
    assert "MedSAM-2" in data["model"]

def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vram" in data
    assert "cuda_available" in data

def test_api_predict_slice(client):
    payload = {
        "image_shape": [3, 64, 64],
        "points": [{"coords": [32.0, 32.0], "label": 1}],
        "bounding_box": [16.0, 16.0, 48.0, 48.0],
        "extract_attention": True
    }
    response = client.post("/predict/slice", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "iou_score" in data
    assert data["mask_shape"] == [64, 64]
    assert len(data["deep_attention_layers_extracted"]) > 0
    assert data["inference_time_ms"] > 0

def test_api_predict_volume(client):
    payload = {
        "num_slices": 4,
        "slice_height": 64,
        "slice_width": 64,
        "prompt_slice_idx": 1,
        "bounding_box": [10.0, 10.0, 50.0, 50.0],
        "modality": "ct",
        "plane": "axial"
    }
    response = client.post("/predict/volume", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["volume_shape"] == [4, 64, 64]
    assert data["prompt_slice_idx"] == 1
    assert 0.0 <= data["mean_slice_iou"] <= 1.0

def test_api_dual_pass_audit(client):
    payload = {
        "slice_shape": [3, 64, 64],
        "bounding_box": [10.0, 10.0, 50.0, 50.0]
    }
    response = client.post("/audit/dual_pass", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "ssim_score" in data
    assert data["gating_status"] in ("PASS", "REJECT")
    assert "trust_score" in data
    assert "clean_vram_allocated_mb" in data
