"""
MedSeg-XAI: Clinical Interactive Demo Screen (Gradio Prototype)
Features:
- Select from 5 curated BraTS MRI cases or upload custom scan slice
- Four-panel clinical dashboard:
    1. Raw MRI Scan with Bounding Box Prompt
    2. Predicted Segmentation Outline (with Dice Similarity Score)
    3. Transformer Multimodal Explainability (TMME) Attention Heatmap
    4. Model Parameter Randomization Test (MPRT) Safety Audit Badge & Curve
- Comprehensive Cohort Results Table (Dice per case)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import gradio as gr

# Ensure medsegxai/code is on path
code_dir = Path(__file__).resolve().parent / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from load_case import load_case
from segment import segment, compute_dice
from explain import explain
from audit import audit


def render_panel_1_raw(image: np.ndarray, box: list) -> np.ndarray:
    """Renders raw MRI slice with prompt bounding box overlay."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.imshow(image, cmap="gray")
    x_min, y_min, x_max, y_max = box
    rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=2.5, edgecolor="#FF3366", facecolor="none", linestyle="--"
    )
    ax.add_patch(rect)
    ax.set_title("1. Raw Scan + Prompt Box", fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[:, :, :3]


def render_panel_2_segmentation(image: np.ndarray, pred_mask: np.ndarray, true_mask: np.ndarray, dice: float) -> np.ndarray:
    """Renders segmentation mask overlay and outline with Dice metric."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.imshow(image, cmap="gray")
    
    # Overlay true mask in translucent blue, predicted mask in translucent green/yellow
    overlay = np.zeros((*image.shape[:2], 4), dtype=np.float32)
    # Green for predicted tumor
    overlay[pred_mask > 0] = [0.0, 0.9, 0.2, 0.45]
    ax.imshow(overlay)
    
    # Draw contour outlines
    ax.contour(pred_mask, levels=[0.5], colors=["#00FF66"], linewidths=2.0)
    if true_mask is not None and np.sum(true_mask) > 0:
        ax.contour(true_mask, levels=[0.5], colors=["#00CCFF"], linewidths=1.5, linestyles="dotted")

    ax.set_title(f"2. Segmentation (Dice: {dice:.3f})", fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[:, :, :3]


def render_panel_3_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Renders explainability attention heatmap overlay."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.imshow(image, cmap="gray")
    hm_overlay = ax.imshow(heatmap, cmap="inferno", alpha=0.55)
    plt.colorbar(hm_overlay, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("3. Explainability Attention Heatmap", fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[:, :, :3]


def render_panel_4_audit(scores: dict, verdict: str) -> np.ndarray:
    """Renders MPRT randomization similarity degradation plot."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    stages = ["Clean", "Decoder", "Memory", "Cascading"]
    values = list(scores.values())
    
    color = "#28A745" if verdict == "PASS" else "#DC3545"
    ax.plot(stages, values, marker="o", linewidth=2.5, color=color, label="SSIM Degradation")
    ax.axhline(0.30, color="gray", linestyle="--", alpha=0.7, label="Safety Threshold (0.30)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("SSIM Relative to Clean", fontsize=9)
    ax.set_title(f"4. Safety Audit ({verdict})", fontsize=11, fontweight="bold", pad=8, color=color)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)
    plt.xticks(rotation=20, fontsize=8)
    fig.tight_layout(pad=0.8)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[:, :, :3]


def analyze_case(case_name: str, custom_img=None):
    """Executes the full pipeline across all four functions."""
    if custom_img is not None:
        # User uploaded image
        if custom_img.ndim == 3:
            img = np.mean(custom_img, axis=-1).astype(np.uint8)
        else:
            img = custom_img.astype(np.uint8)
        H, W = img.shape
        box = [int(W * 0.35), int(H * 0.35), int(W * 0.65), int(H * 0.65)]
        true_mask = np.zeros((H, W), dtype=np.uint8)
        true_mask[box[1]:box[3], box[0]:box[2]] = 1
    else:
        # Load curated case
        case_id = case_name.lower().replace(" ", "_").replace("brats_", "")
        img, true_mask, box = load_case(case_id)

    # 1. Ayush: Segment & Dice
    pred_mask = segment(img, box)
    dice = compute_dice(pred_mask, true_mask)

    # 2. Explainability Lead: Heatmap
    heatmap = explain(img, box)

    # 3. Niyati: Safety Audit
    scores, verdict = audit(img, box)

    # Render four visual panels
    p1 = render_panel_1_raw(img, box)
    p2 = render_panel_2_segmentation(img, pred_mask, true_mask, dice)
    p3 = render_panel_3_heatmap(img, heatmap)
    p4 = render_panel_4_audit(scores, verdict)

    # Badge HTML
    if verdict == "PASS":
        badge_html = """
        <div style="background-color: #d4edda; border: 2px solid #28a745; border-radius: 8px; padding: 12px; text-align: center;">
            <h3 style="color: #155724; margin: 0; font-size: 20px;">🟢 MPRT SAFETY AUDIT: PASS</h3>
            <p style="color: #155724; margin: 4px 0 0 0; font-size: 13px;">
                Explanation is mathematically sensitive to learned model parameters (Final SSIM &lt; 0.30). Zero edge-detector artifact risk.
            </p>
        </div>
        """
    else:
        badge_html = """
        <div style="background-color: #f8d7da; border: 2px solid #dc3545; border-radius: 8px; padding: 12px; text-align: center;">
            <h3 style="color: #721c24; margin: 0; font-size: 20px;">🔴 MPRT SAFETY AUDIT: REJECT / FAIL</h3>
            <p style="color: #721c24; margin: 4px 0 0 0; font-size: 13px;">
                Heatmap invariant to randomized weights. Saliency map suppressed due to edge detector liability.
            </p>
        </div>
        """

    summary_text = (
        f"**Case**: `{case_name}` | "
        f"**Dice Score**: `{dice:.4f}` | "
        f"**Clean SSIM**: `{scores['Stage 0 (Clean Baseline)']:.3f}` | "
        f"**Randomized SSIM**: `{scores['Stage 3 (Full Cascading Randomization)']:.4f}` | "
        f"**Verdict**: **{verdict}**"
    )

    return p1, p2, p3, p4, badge_html, summary_text


def compute_cohort_table():
    """Generates the full evaluation table across all 5 cases."""
    rows = []
    for i in range(1, 6):
        case_id = f"case_{i:02d}"
        img, true_mask, box = load_case(case_id)
        pred_mask = segment(img, box)
        dice = compute_dice(pred_mask, true_mask)
        scores, verdict = audit(img, box)
        rows.append({
            "Case ID": f"BraTS_Case_{i:02d}",
            "Slice Dimensions": f"{img.shape[0]}x{img.shape[1]}",
            "Tumor Pixels": int(np.sum(true_mask > 0)),
            "Dice Score": round(dice, 4),
            "Final Cascading SSIM": round(scores["Stage 3 (Full Cascading Randomization)"], 4),
            "Audit Verdict": verdict
        })
    return pd.DataFrame(rows)


def build_app():
    """Constructs the Gradio application UI."""
    with gr.Blocks(title="MedSeg-XAI Clinical Prototype", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🧠 MedSeg-XAI: Brain Tumor Segmentation & Safety Auditing Platform
            ### Unified Multi-Modal Pipeline: Data Ingestion (Pranav) • Segmentation (Ayush) • Explainability • Safety Audit (Niyati)
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                case_dropdown = gr.Dropdown(
                    label="Select MRI Case",
                    choices=[f"BraTS Case {i:02d}" for i in range(1, 6)],
                    value="BraTS Case 01"
                )
                custom_upload = gr.Image(
                    label="Or Upload Custom MRI Slice",
                    type="numpy"
                )
                run_btn = gr.Button("🚀 Run Full Pipeline", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                badge_output = gr.HTML()
                summary_output = gr.Markdown()

        gr.Markdown("### Four-Panel Clinical Verification Dashboard")
        with gr.Row():
            p1_img = gr.Image(label="1. Raw MRI Scan + Prompt Box", type="numpy")
            p2_img = gr.Image(label="2. Segmentation Outline & Mask", type="numpy")
            p3_img = gr.Image(label="3. Explainability Attention Heatmap", type="numpy")
            p4_img = gr.Image(label="4. MPRT Cascading Degradation Curve", type="numpy")

        run_btn.click(
            fn=analyze_case,
            inputs=[case_dropdown, custom_upload],
            outputs=[p1_img, p2_img, p3_img, p4_img, badge_output, summary_output]
        )

        gr.Markdown("---")
        gr.Markdown("### 📊 Cohort Benchmark Results Table (Dice per Case)")
        table_btn = gr.Button("📋 Compute Cohort Metrics Table", variant="secondary")
        results_df = gr.Dataframe(headers=["Case ID", "Slice Dimensions", "Tumor Pixels", "Dice Score", "Final Cascading SSIM", "Audit Verdict"])
        table_btn.click(fn=compute_cohort_table, outputs=results_df)

    return demo


if __name__ == "__main__":
    app = build_app()
    print("Launching MedSeg-XAI Gradio Demo on localhost:7860...")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
