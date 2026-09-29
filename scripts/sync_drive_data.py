"""
sync_drive_data.py
Synchronizes and standardizes all real clinical datasets, outputs, heatmaps,
segmentations, and audit files from the Google Drive download folders into:
- D:/ayush_medseg_workflow/outputs/
- D:/ayush_medseg_workflow/public/
- D:/ayush_medseg_workflow/public/data/
- D:/ayush_medseg_workflow/data/
"""

import os
import shutil
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage as ndi

ROOT_DIR = "D:/ayush_medseg_workflow"
DRIVE_DIR = os.path.join(ROOT_DIR, "drive_download")
DRIVE_CASES = os.path.join(DRIVE_DIR, "cases")
DRIVE_OUTPUTS = os.path.join(DRIVE_DIR, "outputs")

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")
PUBLIC_DATA_DIR = os.path.join(PUBLIC_DIR, "data")
LOCAL_DATA_DIR = os.path.join(ROOT_DIR, "data")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PUBLIC_DIR, exist_ok=True)
os.makedirs(PUBLIC_DATA_DIR, exist_ok=True)
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# 1. Load results.json and dice_table.csv
with open(os.path.join(DRIVE_OUTPUTS, "results", "results.json"), "r") as f:
    results_meta = json.load(f)

# 2. Synchronize all subfolders to root outputs and public
subfolders = ["audit", "cases", "heatmap_A", "heatmap_B", "results", "segmentation"]

for sub in subfolders:
    src_sub = os.path.join(DRIVE_OUTPUTS, sub)
    if not os.path.exists(src_sub):
        continue
    
    # Destination in outputs/
    dest_out = os.path.join(OUTPUTS_DIR, sub)
    shutil.copytree(src_sub, dest_out, dirs_exist_ok=True)
    
    # Case-insensitive / alias copies: heatmap_a, heatmap_b, segmentations
    if sub == "heatmap_A":
        shutil.copytree(src_sub, os.path.join(OUTPUTS_DIR, "heatmap_a"), dirs_exist_ok=True)
        shutil.copytree(src_sub, os.path.join(PUBLIC_DIR, "heatmap_a"), dirs_exist_ok=True)
        shutil.copytree(src_sub, os.path.join(PUBLIC_DIR, "heatmap_A"), dirs_exist_ok=True)
    elif sub == "heatmap_B":
        shutil.copytree(src_sub, os.path.join(OUTPUTS_DIR, "heatmap_b"), dirs_exist_ok=True)
        shutil.copytree(src_sub, os.path.join(PUBLIC_DIR, "heatmap_b"), dirs_exist_ok=True)
        shutil.copytree(src_sub, os.path.join(PUBLIC_DIR, "heatmap_B"), dirs_exist_ok=True)
    elif sub == "segmentation":
        shutil.copytree(src_sub, os.path.join(OUTPUTS_DIR, "segmentations"), dirs_exist_ok=True)
        shutil.copytree(src_sub, os.path.join(PUBLIC_DIR, "segmentation"), dirs_exist_ok=True)
        shutil.copytree(src_sub, os.path.join(PUBLIC_DIR, "segmentations"), dirs_exist_ok=True)
    else:
        dest_pub = os.path.join(PUBLIC_DIR, sub)
        shutil.copytree(src_sub, dest_pub, dirs_exist_ok=True)

# Also ensure drive_download/cases is fully copied to cases/
shutil.copytree(DRIVE_CASES, os.path.join(OUTPUTS_DIR, "cases"), dirs_exist_ok=True)
shutil.copytree(DRIVE_CASES, os.path.join(PUBLIC_DIR, "cases"), dirs_exist_ok=True)

print("Synchronized raw folders to outputs/ and public/")

# Clinical case descriptions
CASE_METAS = {
    "case_01": {
        "title": "Case 01",
        "pathology": "Glioblastoma Multiforme (Focal Enhancing Core)",
        "modality": "T1ce / FLAIR (Axial z=63, MSD Task01)",
        "description": "Focal contrast-enhancing lesion in the left fronto-parietal region. High contrast-to-noise ratio relative to background parenchyma."
    },
    "case_02": {
        "title": "Case 02",
        "pathology": "Diffuse High-Grade Glioma",
        "modality": "T1ce / FLAIR (Axial z=104, MSD Task01)",
        "description": "Right temporal mass with prominent central necrosis and peripheral rim enhancement."
    },
    "case_03": {
        "title": "Case 03",
        "pathology": "Anaplastic Astrocytoma",
        "modality": "T1ce / FLAIR (Axial z=85, MSD Task01)",
        "description": "Left temporal lobe intra-axial mass with heterogeneous contrast enhancement and surrounding vasogenic edema."
    },
    "case_04": {
        "title": "Case 04",
        "pathology": "Deep Temporal Lesion",
        "modality": "T1ce / FLAIR (Axial z=95, MSD Task01)",
        "description": "Right deep temporal lesion adjacent to lateral ventricle. Sharp margins and distinct contrast enhancement."
    },
    "case_05": {
        "title": "Case 05",
        "pathology": "Infiltrative Frontal Glioma (Diffuse Margin)",
        "modality": "T1ce / FLAIR (Axial z=100, MSD Task01)",
        "description": "Left frontal infiltrative glioma with faint peripheral enhancement. Model isolates hyperintense core while under-segmenting infiltrative border (Dice: 0.641)."
    }
}

manifest = {
    "meta": results_meta.get("meta", {}),
    "cases": {}
}

def render_contours_overlay(raw_rgb, gt_mask, pred_mask):
    """
    Renders an axial slice with Expert Ground Truth (Green contour)
    and MedSAM Prediction (Red contour).
    """
    h, w, c = raw_rgb.shape
    base_img = Image.fromarray(raw_rgb).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    
    # Ground truth: semi-transparent green fill + solid green border
    gt_dilated = ndi.binary_dilation(gt_mask, iterations=2)
    gt_border = (gt_dilated & (~gt_mask.astype(bool)))
    
    # Pred: semi-transparent red fill + solid red border
    pred_dilated = ndi.binary_dilation(pred_mask, iterations=2)
    pred_border = (pred_dilated & (~pred_mask.astype(bool)))
    
    overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Fills: Green 25% for GT, Red 25% for Pred
    overlay_arr[gt_mask > 0] = [34, 197, 94, 60]
    overlay_arr[pred_mask > 0] = [239, 68, 68, 60]
    # Overlap fill: Yellow 40%
    overlap = (gt_mask > 0) & (pred_mask > 0)
    overlay_arr[overlap] = [234, 179, 8, 70]
    
    # Borders: Green solid for GT, Red solid for Pred
    overlay_arr[gt_border] = [34, 197, 94, 255]
    overlay_arr[pred_border] = [239, 68, 68, 255]
    
    overlay_img = Image.fromarray(overlay_arr, mode="RGBA")
    combined = Image.alpha_composite(base_img, overlay_img)
    return combined.convert("RGB")

def render_box_overlay(raw_rgb, box):
    """
    Renders an axial slice with prompt bounding box [x_min, y_min, x_max, y_max]
    drawn with medical amber rectangle and corner crosshairs.
    """
    im = Image.fromarray(raw_rgb).convert("RGBA")
    draw = ImageDraw.Draw(im)
    x0, y0, x1, y1 = box
    
    # Main rectangle
    outline_color = (245, 158, 11, 255) # Amber-500
    for offset in range(2):
        draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=outline_color)
    
    # Corner brackets for PACS aesthetic
    bracket_len = min(16, (x1 - x0) // 4, (y1 - y0) // 4)
    # Top-Left
    draw.line([(x0 - 4, y0), (x0 + bracket_len, y0)], fill=(239, 68, 68, 255), width=3)
    draw.line([(x0, y0 - 4), (x0, y0 + bracket_len)], fill=(239, 68, 68, 255), width=3)
    # Top-Right
    draw.line([(x1 - bracket_len, y0), (x1 + 4, y0)], fill=(239, 68, 68, 255), width=3)
    draw.line([(x1, y0 - 4), (x1, y0 + bracket_len)], fill=(239, 68, 68, 255), width=3)
    # Bottom-Left
    draw.line([(x0 - 4, y1), (x0 + bracket_len, y1)], fill=(239, 68, 68, 255), width=3)
    draw.line([(x0, y1 - bracket_len), (x0, y1 + 4)], fill=(239, 68, 68, 255), width=3)
    # Bottom-Right
    draw.line([(x1 - bracket_len, y1), (x1 + 4, y1)], fill=(239, 68, 68, 255), width=3)
    draw.line([(x1, y1 - bracket_len), (x1, y1 + 4)], fill=(239, 68, 68, 255), width=3)
    
    return im.convert("RGB")

def render_heatmap_blend(raw_gray, heatmap_norm, colormap_name):
    """
    Blends normalized heatmap [0, 1] with grayscale MRI using specified colormap.
    """
    cmap = plt.get_cmap(colormap_name)
    colored_heat = cmap(heatmap_norm)[..., :3] # (512, 512, 3) float in [0, 1]
    colored_heat_uint8 = (colored_heat * 255).astype(np.uint8)
    
    # Alpha blend: where heatmap is high, show heatmap; background remains anatomy
    alpha = np.clip(heatmap_norm * 1.2, 0.0, 0.85)[..., np.newaxis]
    mri_float = raw_gray.astype(np.float32) / 255.0
    if len(mri_float.shape) == 2:
        mri_float = np.repeat(mri_float[..., np.newaxis], 3, axis=2)
    
    blended = (alpha * colored_heat + (1.0 - alpha) * mri_float)
    blended = (np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(blended)

def render_contour_heatmap(raw_gray, heatmap_norm):
    """
    Renders isolines of attention map over grayscale MRI.
    """
    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
    fig.patch.set_facecolor('black')
    ax.imshow(raw_gray, cmap='gray', origin='upper')
    
    # Draw contours
    levels = np.linspace(0.15, 0.95, 8)
    cs = ax.contour(heatmap_norm, levels=levels, cmap='autumn', linewidths=1.8)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba).convert("RGB")

# Iterate over all 5 cases
for i in range(1, 6):
    cid = f"case_0{i}"
    print(f"Processing {cid}...")
    
    # Paths in drive_download
    case_npz_path = os.path.join(DRIVE_CASES, f"{cid}.npz")
    pred_mask_path = os.path.join(DRIVE_OUTPUTS, "segmentation", f"{cid}_pred_mask.npy")
    heatmap_a_path = os.path.join(DRIVE_OUTPUTS, "heatmap_A", f"{cid}_heatmap_A.npy")
    heatmap_b_path = os.path.join(DRIVE_OUTPUTS, "heatmap_B", f"{cid}_stage4_decoder_full.npy")
    
    case_data = np.load(case_npz_path)
    raw_img = case_data["image"] # (512, 512, 3) uint8
    gt_mask = case_data["gt_mask"] # (512, 512) uint8
    box = [int(v) for v in case_data["box"]] # [x_min, y_min, x_max, y_max]
    src = str(case_data["src"])
    z_idx = int(case_data["z"])
    
    pred_mask = np.load(pred_mask_path) # (512, 512) uint8
    heatmap_A = np.load(heatmap_a_path) # (512, 512) float32
    heatmap_B = np.load(heatmap_b_path) # (512, 512) float32
    
    # Normalization
    ha_min, ha_max = float(heatmap_A.min()), float(heatmap_A.max())
    ha_norm = (heatmap_A - ha_min) / (ha_max - ha_min + 1e-8)
    
    hb_min, hb_max = float(heatmap_B.min()), float(heatmap_B.max())
    hb_norm = (heatmap_B - hb_min) / (hb_max - hb_min + 1e-8)
    
    residual = np.abs(heatmap_A - heatmap_B)
    res_min, res_max = float(residual.min()), float(residual.max())
    res_norm = (residual - res_min) / (res_max - res_min + 1e-8)
    
    raw_gray = raw_img[..., 0] if raw_img.ndim == 3 else raw_img
    
    # Results metadata
    case_res = results_meta.get("cases", {}).get(cid, {})
    dice_clean = float(case_res.get("dice_clean", 0.0))
    ssim_full = float(case_res.get("ssim_decoder_full", 0.0))
    spearman_full = float(case_res.get("spearman_decoder_full", 0.0))
    dice_rand = float(case_res.get("dice_after_decoder_rand", 0.0))
    pass_spearman = bool(case_res.get("pass_v2_spearman<=0.3", True))
    pass_ssim = bool(case_res.get("pass_v1_ssim<=0.5", False))
    
    # Destination folders
    dest_pub_case = os.path.join(PUBLIC_DATA_DIR, cid)
    dest_loc_case = os.path.join(LOCAL_DATA_DIR, cid)
    os.makedirs(dest_pub_case, exist_ok=True)
    os.makedirs(dest_loc_case, exist_ok=True)
    
    # 1. Raw image
    raw_im = Image.fromarray(raw_img)
    raw_im.save(os.path.join(dest_pub_case, "raw.png"))
    raw_im.save(os.path.join(dest_loc_case, "raw.png"))
    
    # 2. Raw with box
    box_im = render_box_overlay(raw_img, box)
    box_im.save(os.path.join(dest_pub_case, "raw_box.png"))
    box_im.save(os.path.join(dest_loc_case, "raw_box.png"))
    
    # 3. Segmentation overlay (Green GT, Red Pred)
    seg_im = render_contours_overlay(raw_img, gt_mask, pred_mask)
    seg_im.save(os.path.join(dest_pub_case, "segmentation.png"))
    seg_im.save(os.path.join(dest_loc_case, "segmentation.png"))
    
    # 4. Binary masks
    Image.fromarray((gt_mask * 255).astype(np.uint8)).save(os.path.join(dest_pub_case, "mask_true.png"))
    Image.fromarray((gt_mask * 255).astype(np.uint8)).save(os.path.join(dest_loc_case, "mask_true.png"))
    Image.fromarray((pred_mask * 255).astype(np.uint8)).save(os.path.join(dest_pub_case, "mask_pred.png"))
    Image.fromarray((pred_mask * 255).astype(np.uint8)).save(os.path.join(dest_loc_case, "mask_pred.png"))
    
    # 5. Colormap designs
    colormaps = {
        "turbo": "turbo",
        "plasma": "plasma",
        "inferno": "inferno",
        "viridis": "viridis"
    }
    for d_id, cmap_name in colormaps.items():
        h_im = render_heatmap_blend(raw_gray, ha_norm, cmap_name)
        h_im.save(os.path.join(dest_pub_case, f"heatmap_{d_id}.png"))
        h_im.save(os.path.join(dest_loc_case, f"heatmap_{d_id}.png"))
    
    # 6. Topographic contour
    contour_im = render_contour_heatmap(raw_gray, ha_norm)
    contour_im.save(os.path.join(dest_pub_case, "heatmap_contour.png"))
    contour_im.save(os.path.join(dest_loc_case, "heatmap_contour.png"))
    
    # 7. Residual delta
    res_im = render_heatmap_blend(raw_gray, res_norm, "turbo")
    res_im.save(os.path.join(dest_pub_case, "residual_turbo.png"))
    res_im.save(os.path.join(dest_loc_case, "residual_turbo.png"))
    res_im.save(os.path.join(dest_pub_case, "heatmap_residual.png"))
    res_im.save(os.path.join(dest_loc_case, "heatmap_residual.png"))
    
    # 8. Scrambled heatmap B
    scram_im = render_heatmap_blend(raw_gray, hb_norm, "turbo")
    scram_im.save(os.path.join(dest_pub_case, "scrambled_turbo.png"))
    scram_im.save(os.path.join(dest_loc_case, "scrambled_turbo.png"))
    
    # 9. Audit curve from outputs/audit
    curve_src = os.path.join(DRIVE_OUTPUTS, "audit", f"{cid}_curve.png")
    if os.path.exists(curve_src):
        shutil.copy2(curve_src, os.path.join(dest_pub_case, "curve.png"))
        shutil.copy2(curve_src, os.path.join(dest_loc_case, "curve.png"))
        
    # 10. Audit montage from outputs/audit
    montage_src = os.path.join(DRIVE_OUTPUTS, "audit", f"{cid}_montage.png")
    if os.path.exists(montage_src):
        shutil.copy2(montage_src, os.path.join(dest_pub_case, "montage.png"))
        shutil.copy2(montage_src, os.path.join(dest_loc_case, "montage.png"))
        
    # 11. Result figure from outputs/results
    fig_src = os.path.join(DRIVE_OUTPUTS, "results", f"{cid}_figure.png")
    if os.path.exists(fig_src):
        shutil.copy2(fig_src, os.path.join(dest_pub_case, "figure.png"))
        shutil.copy2(fig_src, os.path.join(dest_loc_case, "figure.png"))
        
    # 12. Preview from cases/
    prev_src = os.path.join(DRIVE_CASES, f"{cid}_preview.png")
    if os.path.exists(prev_src):
        shutil.copy2(prev_src, os.path.join(dest_pub_case, "preview.png"))
        shutil.copy2(prev_src, os.path.join(dest_loc_case, "preview.png"))
        
    # 13. Drive overlays
    ha_over_src = os.path.join(DRIVE_OUTPUTS, "heatmap_A", f"{cid}_heatmap_A_overlay.png")
    if os.path.exists(ha_over_src):
        shutil.copy2(ha_over_src, os.path.join(dest_pub_case, "heatmap_A_overlay.png"))
        shutil.copy2(ha_over_src, os.path.join(dest_loc_case, "heatmap_A_overlay.png"))
        
    seg_over_src = os.path.join(DRIVE_OUTPUTS, "segmentation", f"{cid}_overlay.png")
    if os.path.exists(seg_over_src):
        shutil.copy2(seg_over_src, os.path.join(dest_pub_case, "seg_overlay.png"))
        shutil.copy2(seg_over_src, os.path.join(dest_loc_case, "seg_overlay.png"))

    # Update manifest entry
    meta_info = CASE_METAS.get(cid, {})
    manifest["cases"][cid] = {
        "id": cid,
        "title": meta_info.get("title", cid),
        "pathology": meta_info.get("pathology", "Brain Lesion"),
        "modality": meta_info.get("modality", "T1ce MRI"),
        "description": meta_info.get("description", ""),
        "slice_dim": "512 x 512",
        "src": src,
        "z": z_idx,
        "box": box,
        "metrics": {
            "dice_clean": dice_clean,
            "dice_after_decoder_rand": dice_rand,
            "spearman_decoder_full": spearman_full,
            "ssim_decoder_full": ssim_full,
            "pass_v1_ssim": pass_ssim,
            "pass_v2_spearman": pass_spearman
        },
        "assets": {
            "raw": f"{cid}/raw.png",
            "raw_box": f"{cid}/raw_box.png",
            "segmentation": f"{cid}/segmentation.png",
            "mask_true": f"{cid}/mask_true.png",
            "mask_pred": f"{cid}/mask_pred.png",
            "heatmap_turbo": f"{cid}/heatmap_turbo.png",
            "heatmap_plasma": f"{cid}/heatmap_plasma.png",
            "heatmap_inferno": f"{cid}/heatmap_inferno.png",
            "heatmap_viridis": f"{cid}/heatmap_viridis.png",
            "heatmap_contour": f"{cid}/heatmap_contour.png",
            "residual_turbo": f"{cid}/residual_turbo.png",
            "scrambled_turbo": f"{cid}/scrambled_turbo.png",
            "curve": f"{cid}/curve.png",
            "montage": f"{cid}/montage.png",
            "figure": f"{cid}/figure.png",
            "preview": f"{cid}/preview.png"
        }
    }

# Save manifest.json
with open(os.path.join(PUBLIC_DATA_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

with open(os.path.join(LOCAL_DATA_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("SUCCESS: manifest.json and all 5 cases fully synchronized and generated!")
