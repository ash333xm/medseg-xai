"""
Export Vercel Static Web Assets
================================
Generates high-resolution PNGs and JSON metadata for all 5 cases,
including multiple clinical heatmap designs (Turbo, Plasma, Inferno, Viridis, Contour Isolines, Residual Delta),
raw inputs, segmentation overlays, and MPRT scrambled heatmaps.
"""

import io
import json
import os
import shutil

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageDraw

OUTPUT_WEB_DIR = os.path.join("public", "data")
os.makedirs(OUTPUT_WEB_DIR, exist_ok=True)

# Case pathology metadata catalog
CASE_CATALOG = {
    "case_01": {
        "id": "case_01",
        "title": "Case 01",
        "pathology": "Glioblastoma Multiforme (Focal Enhancing Core)",
        "modality": "T1ce (Contrast-Enhanced T1-weighted MRI)",
        "slice_dim": "256 x 256",
        "description": "Circumscribed contrast-enhancing lesion in the left fronto-parietal region. High contrast-to-noise ratio relative to background white matter.",
    },
    "case_02": {
        "id": "case_02",
        "title": "Case 02",
        "pathology": "Diffuse High-Grade Glioma",
        "modality": "T1ce (Contrast-Enhanced T1-weighted MRI)",
        "slice_dim": "512 x 512 x 3",
        "description": "Right temporal mass with prominent central necrotic cavity and thick peripheral rim enhancement.",
    },
    "case_03": {
        "id": "case_03",
        "title": "Case 03",
        "pathology": "Anaplastic Astrocytoma",
        "modality": "T1ce (Contrast-Enhanced T1-weighted MRI)",
        "slice_dim": "512 x 512 x 3",
        "description": "Left temporal lobe intra-axial mass with heterogeneous contrast enhancement and surrounding vasogenic edema.",
    },
    "case_04": {
        "id": "case_04",
        "title": "Case 04",
        "pathology": "Deep Temporal Lesion",
        "modality": "T1ce (Contrast-Enhanced T1-weighted MRI)",
        "slice_dim": "512 x 512 x 3",
        "description": "Right deep temporal lesion adjacent to the lateral ventricle. Distinct hypointense necrotic core with sharp peripheral borders.",
    },
    "case_05": {
        "id": "case_05",
        "title": "Case 05",
        "pathology": "Infiltrative Frontal Glioma (Diffuse Margin)",
        "modality": "T1ce (Contrast-Enhanced T1-weighted MRI)",
        "slice_dim": "512 x 512 x 3",
        "description": "Left frontal lobe infiltrative mass presenting with faint peripheral contrast enhancement and diffuse non-enhancing margins. Real-world challenging boundary.",
    },
}

with open(os.path.join("outputs", "metrics.json"), "r", encoding="utf-8") as f:
    metrics_manifest = json.load(f)


def get_base_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        norm = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        return np.stack([norm, norm, norm], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 3:
        if image.dtype != np.uint8:
            norm = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
            return norm
        return image
    else:
        c0 = image[..., 0]
        norm = ((c0 - c0.min()) / (c0.max() - c0.min() + 1e-8) * 255).astype(np.uint8)
        return np.stack([norm, norm, norm], axis=-1)


def save_fig(fig, filepath):
    fig.savefig(filepath, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)


for case_id in ["case_01", "case_02", "case_03", "case_04", "case_05"]:
    case_dir = os.path.join(OUTPUT_WEB_DIR, case_id)
    os.makedirs(case_dir, exist_ok=True)

    npz_data = np.load(os.path.join("outputs", "cases", f"{case_id}.npz"))
    heat_b = np.load(os.path.join("outputs", "heatmap_B", f"{case_id}_stage4_decoder_full.npy"))

    base_rgb = get_base_rgb(npz_data["image"])
    box = npz_data["box"].tolist()
    true_mask = npz_data["true_mask"]
    pred_mask = npz_data["pred_mask"]
    heat_a = npz_data["heatmap_A"]

    norm_heat_a = (heat_a - heat_a.min()) / (heat_a.max() - heat_a.min() + 1e-8)
    norm_heat_b = (heat_b - heat_b.min()) / (heat_b.max() - heat_b.min() + 1e-8)
    residual_heat = np.abs(norm_heat_a - norm_heat_b)
    residual_heat = (residual_heat - residual_heat.min()) / (residual_heat.max() - residual_heat.min() + 1e-8)

    # 1. Raw image
    pil_raw = Image.fromarray(base_rgb)
    pil_raw.save(os.path.join(case_dir, "raw.png"))

    # 2. Raw with Bounding Box
    pil_box = Image.fromarray(base_rgb.copy())
    draw = ImageDraw.Draw(pil_box)
    draw.rectangle([box[0], box[1], box[2], box[3]], outline="#eab308", width=3)
    pil_box.save(os.path.join(case_dir, "raw_box.png"))

    # 3. Segmentation vector contours
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.imshow(base_rgb)
    if np.any(true_mask > 0):
        ax.contour(true_mask > 0, levels=[0.5], colors=["#16a34a"], linewidths=2.2)
    if np.any(pred_mask > 0):
        ax.contour(pred_mask > 0, levels=[0.5], colors=["#dc2626"], linewidths=2.2)
    ax.axis("off")
    fig.tight_layout(pad=0)
    save_fig(fig, os.path.join(case_dir, "segmentation.png"))

    # 4. Binary masks
    Image.fromarray((true_mask * 255).astype(np.uint8)).save(os.path.join(case_dir, "mask_true.png"))
    Image.fromarray((pred_mask * 255).astype(np.uint8)).save(os.path.join(case_dir, "mask_pred.png"))

    # 5. Heatmap A Designs
    # Design 1: Turbo
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.imshow(base_rgb)
    ax.imshow(norm_heat_a, cmap="turbo", alpha=0.55)
    ax.axis("off")
    fig.tight_layout(pad=0)
    save_fig(fig, os.path.join(case_dir, "heatmap_turbo.png"))

    # Design 2: Plasma
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.imshow(base_rgb)
    ax.imshow(norm_heat_a, cmap="plasma", alpha=0.58)
    ax.axis("off")
    fig.tight_layout(pad=0)
    save_fig(fig, os.path.join(case_dir, "heatmap_plasma.png"))

    # Design 3: Inferno
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.imshow(base_rgb)
    ax.imshow(norm_heat_a, cmap="inferno", alpha=0.58)
    ax.axis("off")
    fig.tight_layout(pad=0)
    save_fig(fig, os.path.join(case_dir, "heatmap_inferno.png"))

    # Design 4: Viridis
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.imshow(base_rgb)
    ax.imshow(norm_heat_a, cmap="viridis", alpha=0.55)
    ax.axis("off")
    fig.tight_layout(pad=0)
    save_fig(fig, os.path.join(case_dir, "heatmap_viridis.png"))

    # Design 5: Topographic Iso-Contour
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.imshow(base_rgb)
    ax.imshow(norm_heat_a, cmap="turbo", alpha=0.45)
    ax.contour(norm_heat_a, levels=np.linspace(0.2, 0.9, 6), cmap="cool", linewidths=1.5)
    ax.axis("off")
    fig.tight_layout(pad=0)
    save_fig(fig, os.path.join(case_dir, "heatmap_contour.png"))

    # 6. Scrambled Heatmap B Designs
    # Scrambled Turbo
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.imshow(base_rgb)
    ax.imshow(norm_heat_b, cmap="turbo", alpha=0.55)
    ax.axis("off")
    fig.tight_layout(pad=0)
    save_fig(fig, os.path.join(case_dir, "scrambled_turbo.png"))

    # Scrambled Plasma
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.imshow(base_rgb)
    ax.imshow(norm_heat_b, cmap="plasma", alpha=0.58)
    ax.axis("off")
    fig.tight_layout(pad=0)
    save_fig(fig, os.path.join(case_dir, "scrambled_plasma.png"))

    # Scrambled Inferno
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.imshow(base_rgb)
    ax.imshow(norm_heat_b, cmap="inferno", alpha=0.58)
    ax.axis("off")
    fig.tight_layout(pad=0)
    save_fig(fig, os.path.join(case_dir, "scrambled_inferno.png"))

    # 7. Residual / Delta Map (Heatmap A - Heatmap B)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.imshow(base_rgb)
    ax.imshow(residual_heat, cmap="magma", alpha=0.62)
    ax.axis("off")
    fig.tight_layout(pad=0)
    save_fig(fig, os.path.join(case_dir, "residual_turbo.png"))

    # 8. Copy curve
    curve_src = os.path.join("outputs", "curves", f"{case_id}_curve.png")
    if os.path.exists(curve_src):
        shutil.copyfile(curve_src, os.path.join(case_dir, "curve.png"))

    # 9. Case metadata
    case_meta = {
        **CASE_CATALOG[case_id],
        "box": box,
        "metrics": metrics_manifest.get(case_id, {}),
    }
    with open(os.path.join(case_dir, "meta.json"), "w", encoding="utf-8") as f_meta:
        json.dump(case_meta, f_meta, indent=2)

    print(f"Exported all web assets for {case_id}")

# Export global manifest
manifest = {
    "cases": list(CASE_CATALOG.values()),
    "metrics": metrics_manifest,
    "heatmap_designs": [
        {"id": "turbo", "name": "Thermal Turbo", "description": "Standard clinical high-dynamic-range radiology spectrum"},
        {"id": "plasma", "name": "Medical Plasma", "description": "High-contrast perceptually uniform saliency gradient"},
        {"id": "inferno", "name": "Solar Inferno", "description": "Deep black-purple-red-gold margin emphasis"},
        {"id": "viridis", "name": "Diagnostic Viridis", "description": "Colorblind-friendly green-yellow medical baseline"},
        {"id": "contour", "name": "Topographic Iso-Contour", "description": "Heatmap with vector attention field isolines"},
        {"id": "residual", "name": "Residual Attention Delta", "description": "Absolute difference highlighting destroyed attention during MPRT audit"},
    ],
    "version": "1.0.4",
    "domain": "pacs.medseg-xai.internal",
}

with open(os.path.join(OUTPUT_WEB_DIR, "manifest.json"), "w", encoding="utf-8") as f_man:
    json.dump(manifest, f_man, indent=2)

print("Export completed successfully. Manifest written to public/data/manifest.json")
