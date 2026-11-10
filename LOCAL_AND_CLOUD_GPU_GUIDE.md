# ⚡ MedSeg-XAI: Local & Cloud GPU Hosting Guide

This guide provides end-to-end instructions for deploying the **MedSeg-XAI Clinical Studio** inference engine on a **Local NVIDIA GPU** or a rented **Cloud GPU (RunPod / Vast.ai / Lambda Labs)**, seamlessly interfacing with the production web dashboard.

---

## 🎯 Architecture Overview: Dual-Mode Operation

The MedSeg-XAI dashboard features a **Hybrid Zero-Flicker Architecture**:
1. **Web Precomputed Mode (Default on Vercel)**:
   * URL: `https://medseg-xai-x7lb.vercel.app/`
   * Runs client-side with 16 pre-computed multi-organ cases (BraTS Brain MRI, BTCV Abdominal CT/MRI, DeepLesion Chest CT, Stress Corpus).
   * **Zero GPU costs**: instantaneous loading, zero cold-starts.
2. **Live GPU Server Mode (Local / RunPod)**:
   * Connects to a live FastAPI backend running MedSAM-2 Hiera-Large with live CUDA acceleration.
   * Open the **"GPU Hub"** button in the dashboard header, input your live server URL (e.g., `http://localhost:8000` or `https://xxxx-8000.proxy.runpod.net`), and click **"Connect & Ping"**.
   * The dashboard transitions seamlessly to execute live volumetric prompt propagation and gradient-weighted TMME relevance extraction on new DICOM/NIfTI scans.

---

## ☁️ 1. Renting and Hosting on RunPod (Recommended Cloud GPU)

RunPod provides affordable, on-demand GPU instances (e.g., NVIDIA RTX 4090 24GB for ~\$0.34/hr or A40 48GB for ~\$0.45/hr).

### Step 1: Deploy a Pod on RunPod
1. Go to [RunPod.io](https://www.runpod.io/) and create an account.
2. Click **Deploy** &rarr; **GPU Cloud**.
3. Select an instance with **at least 24 GB VRAM**:
   * **Recommended**: **NVIDIA RTX 4090 (24 GB VRAM)** or **NVIDIA RTX A5000 / A40**.
4. Choose the official template: **RunPod PyTorch 2.2.0 (CUDA 12.1)**.
5. In **Container Disk**, set at least **30 GB** (to store model weights and volume cases).
6. Under **Expose HTTP Ports**, ensure port **`8000`** is added.
7. Click **Deploy**.

### Step 2: Launch the Inference Server inside RunPod
1. When your pod is running, click **Connect** &rarr; **Connect to Web Terminal** (or SSH).
2. Clone this repository and run the startup script:
   ```bash
   git clone https://github.com/ash333xm/medseg-xai.git
   cd medseg-xai
   
   # Run automated setup and server startup
   bash scripts/start_runpod.sh
   ```
3. The server will launch on `http://0.0.0.0:8000` with FastAPI endpoints:
   * `/docs`: Swagger interactive documentation
   * `/api/health`: Health & CUDA device status
   * `/api/segment`: Live 3D volumetric prompt propagation
   * `/api/explain`: Gradient-weighted TMME attention rollout
   * `/api/audit`: 8-Stage cascading MPRT safety check

### Step 3: Connect Dashboard to RunPod
1. In your RunPod dashboard, find your running pod and click **Connect**.
2. Click **Connect to HTTP Service [Port 8000]**. This gives you an HTTPS URL like:
   `https://[pod-id]-8000.proxy.runpod.net`
3. Open your production dashboard: [https://medseg-xai-x7lb.vercel.app/](https://medseg-xai-x7lb.vercel.app/)
4. Click the top-right button: **⚡ GPU Hub**.
5. Paste your RunPod proxy URL and click **Connect & Ping**.
6. The status badge will turn green: **`⚡ GPU Server: Connected (NVIDIA CUDA)`**!

---

## 🖥️ 2. Hosting on a Local NVIDIA GPU (RTX 3060 / 3080 / 3090 / 4090)

### Prerequisites:
* Windows 10/11 or Ubuntu Linux
* NVIDIA GPU with CUDA 11.8 or 12.1+ installed (minimum 12 GB VRAM for inference, 24 GB for Hiera-Large full video memory)
* Python 3.10 or 3.11

### Step 1: Clone and Set Up Virtual Environment
```bash
git clone https://github.com/ash333xm/medseg-xai.git
cd medseg-xai

# Create Python environment
python -m venv .venv

# Activate environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies with PyTorch CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 2: Start the Local FastAPI Backend Server
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```
Test the connection in your browser at `http://localhost:8000/docs`.

### Step 3: Connect Production Dashboard to Local GPU
1. Open the live dashboard at [https://medseg-xai-x7lb.vercel.app/](https://medseg-xai-x7lb.vercel.app/).
2. Click **⚡ GPU Hub** in the header.
3. Keep the default endpoint `http://localhost:8000` and click **Connect & Ping**.
4. The web dashboard will now route segmentation and TMME requests directly through your local GPU!

---

## 🐳 3. Running with Docker Container

For containerized deployment on any Linux workstation or cloud server:

```bash
# Build Docker image
docker build -t medseg-xai-engine:latest .

# Run with all GPUs passed through
docker run --gpus all -d -p 8000:8000 --name medseg_engine medseg-xai-engine:latest

# Check logs
docker logs -f medseg_engine
```

---

## 📊 Summary of API Endpoints for Custom Integration

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Hardware telemetry, GPU VRAM usage, and device name |
| `POST` | `/api/segment` | 3D volumetric pseudo-video propagation across slices |
| `POST` | `/api/explain` | TMME gradient-weighted multi-layer relevance rollout |
| `POST` | `/api/audit` | 8-stage cascading Model Parameter Randomization Test |
| `GET` | `/docs` | Interactive Swagger API documentation & manual tester |
